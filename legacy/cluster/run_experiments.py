"""
STRAT 聚类全量实验驱动（双轨评测）

对每个景区：
  1. 粒球+规则亲和+网格δ(验证子集)+eigen-gap k 景点聚类 → region_id
  2. 内部指标：Silhouette / DBI / CH / Composite
  3. 外部验证：POI 缓冲区区域做真值，计算 ARI/NMI/FMI（只验证，不参与调参）
  4. 区域对应：簇纯度 + GT 覆盖率 + 最终 k vs POI 区域数对照

用法：
    python cluster/run_experiments.py --all --n_workers 8
    python cluster/run_experiments.py --scenery 峨眉山 龙泉
"""

import os
import sys
import json
import time
import glob
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'poi')))

from config import CLEANED_DIR, CLUSTER_OUTPUT_DIR, SEED, adaptive_k_bounds
from scenery_route_clustering import cluster_scenic_spots
import ground_truth as gt

POI_BUFFER_M = 50
PROJECT_RADIUS_M = 300
RADIUS_DEG = POI_BUFFER_M / 111000.0


def load_poi_regions():
    """加载 POI → 投影到路径 → 缓冲区合并区域（与 ground_truth.py 同口径）。"""
    with open(gt.POI_PATH, encoding='utf-8') as f:
        raw = json.load(f)
    filtered = gt.filter_pois(raw)
    poi_df = pd.DataFrame(filtered).drop_duplicates(subset=['scenery', 'poi_id'])
    poi_df = gt.project_pois_to_paths(poi_df, projection_radius_m=PROJECT_RADIUS_M)
    poi_df = gt.create_poi_regions(poi_df, buffer_radius_m=POI_BUFFER_M)
    poi_df = gt.merge_small_regions(poi_df, min_pois=2)
    return poi_df


def region_correspondence(labels_pred, labels_true):
    """
    区域对应指标：
    - 簇纯度 (cluster purity)：每个预测簇内多数 GT 区域标签占比的加权平均
    - GT 覆盖率 (GT coverage)：被至少一个预测簇当作"多数区域"的 POI 区域占比
    均只在有 GT 标签（gt_region_id != -1）的停留点上计算。
    """
    m = labels_pred[labels_true != -1].astype(int)
    t = labels_true[labels_true != -1].astype(int)
    n_labeled = len(m)
    n_gt = len(np.unique(t))
    n_pred = len(np.unique(m))
    if n_labeled < 10 or n_gt < 2 or n_pred < 2:
        return {'n_labeled': n_labeled, 'n_gt_regions': n_gt, 'n_pred': n_pred,
                'purity': None, 'gt_coverage': None}
    agree = 0
    covered = set()
    for c in np.unique(m):
        mask = m == c
        cnt = np.bincount(t[mask])
        majority_gt = int(np.argmax(cnt))
        agree += int(cnt[majority_gt])
        covered.add(majority_gt)
    purity = agree / n_labeled
    coverage = len(covered) / n_gt if n_gt > 0 else 0.0
    return {'n_labeled': n_labeled, 'n_gt_regions': n_gt, 'n_pred': n_pred,
            'purity': float(purity), 'gt_coverage': float(coverage)}


def resolve_scenes(min_tracks=30):
    files = sorted(glob.glob(os.path.join(CLEANED_DIR, '*_cleaned.csv')))
    out = []
    for f in files:
        name = os.path.basename(f).replace('_cleaned.csv', '')
        try:
            df_tmp = pd.read_csv(f, usecols=['trackId'])
            n = df_tmp['trackId'].nunique()
        except Exception:
            continue
        if n >= min_tracks:
            out.append((name, f, n))
    return out


def run_scene(name, csv_path, n_tracks, poi_df, n_workers, seed, with_routes):
    print(f"\n{'#'*64}")
    print(f"# {name}  ({n_tracks} tracks)")
    print(f"{'#'*64}")
    t0 = time.time()

    scene_pois = poi_df[poi_df['scenery'] == name]
    n_poi_regions = int(scene_pois['poi_region_id'].nunique()) if len(scene_pois) > 0 else 0
    print(f"  POI: {len(scene_pois)} 个 → {n_poi_regions} 个 POI 区域")

    # 1. 景点聚类（方法本体）
    df = pd.read_csv(csv_path)
    df, scenic_m, ball_data_list, cluster_features, sampled_labels = cluster_scenic_spots(
        df, n_workers=n_workers, random_state=seed,
        k_lower=None, k_upper=None, n_poi_regions=n_poi_regions or None)

    # 保存带 region_id 的结果（供外部验证复用同一文件保证对齐）
    os.makedirs(CLUSTER_OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(CLUSTER_OUTPUT_DIR, f'{name}_clustered.csv')
    cols = ['经度', '纬度', '海拔', '速度(km/h)', 'is_stop', 'trackId', '时间_秒', 'region_id']
    available = [c for c in cols if c in df.columns]
    df[available].to_csv(out_csv, index=False, encoding='utf-8-sig')

    # 2. 内部指标（来自 scenic_m）
    internal = {
        'n_regions': scenic_m['n_regions'],
        'n_balls': scenic_m['n_balls'],
        'final_k': scenic_m['final_k'],
        'k_lower': scenic_m['k_lower'],
        'k_upper': scenic_m['k_upper'],
        'best_delta': scenic_m['best_delta'],
        'silhouette': scenic_m['silhouette'],
        'davies_bouldin': scenic_m['davies_bouldin'],
        'calinski_harabasz': scenic_m['calinski_harabasz'],
        'composite': scenic_m['composite'],
        'composite_sub': scenic_m.get('composite_sub', {}),
    }

    # 3+4. 外部验证 + 区域对应（只匹配一次，POI 区域真值；仅停留点、开投影、关传播）
    external = {'ari': None, 'nmi': None, 'fmi': None, 'match_rate': 0.0, 'n_poi_regions': n_poi_regions}
    corr = {'purity': None, 'gt_coverage': None, 'n_labeled': 0, 'n_gt_regions': n_poi_regions, 'n_pred': 0}
    if len(scene_pois) > 0 and n_poi_regions >= 2:
        traj = gt.match_trajectory_to_regions(out_csv, scene_pois, RADIUS_DEG, only_stay_points=True)
        true = traj['gt_region_id'].values.astype(np.int32)
        pred = traj['region_id'].values.astype(np.int32)
        ext_metrics = gt.compute_metrics(true, pred)
        n_labeled = ext_metrics.get('n_labeled', 0)
        n_stay = int((traj['is_stop'] == 1).sum()) if 'is_stop' in traj.columns else len(traj)
        external = {
            'ari': ext_metrics.get('ari'), 'nmi': ext_metrics.get('nmi'),
            'fmi': ext_metrics.get('fmi'),
            'match_rate': (n_labeled / n_stay * 100) if n_stay > 0 else 0.0,
            'n_poi_regions': n_poi_regions, 'n_labeled': n_labeled,
        }
        corr = region_correspondence(pred, true)

    elapsed = time.time() - t0
    print(f"  regions={internal['n_regions']} (k∈[{internal['k_lower']},{internal['k_upper']}], "
          f"final k={internal['final_k']})  sil={internal['silhouette']:.4f}  DBI={internal['davies_bouldin']:.4f}")
    ari_s = f"{external['ari']:.4f}" if external.get('ari') is not None else "N/A"
    print(f"  POI 外部验证: ARI={ari_s}, 簇纯度={corr.get('purity')}, GT覆盖={corr.get('gt_coverage')}")
    print(f"  k vs POI区域数: {internal['final_k']} vs {n_poi_regions}, 耗时 {elapsed:.1f}s")

    return {
        'n_tracks': n_tracks, 'n_poi_regions': n_poi_regions,
        'internal': internal, 'external': external, 'correspondence': corr,
        'time_s': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='STRAT 聚类全量实验（双轨评测）')
    parser.add_argument('--scenery', nargs='*', default=None, help='指定景区')
    parser.add_argument('--all', action='store_true', help='跑所有 min_tracks>=30 的景区')
    parser.add_argument('--min_tracks', type=int, default=30)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--with_routes', action='store_true', help='(保留) 仅景点聚类评测')
    parser.add_argument('--out', type=str, default='cluster/output/experiments.json')
    args = parser.parse_args()

    poi_df = load_poi_regions()

    if args.scenery:
        scenes = []
        for name in args.scenery:
            csv_path = os.path.join(CLEANED_DIR, f'{name}_cleaned.csv')
            if os.path.exists(csv_path):
                df_tmp = pd.read_csv(csv_path, usecols=['trackId'])
                scenes.append((name, csv_path, df_tmp['trackId'].nunique()))
    elif args.all:
        scenes = resolve_scenes(args.min_tracks)
    else:
        parser.error('请指定 --scenery 或 --all')

    print("=" * 70)
    print(f"STRAT 聚类全量实验（{len(scenes)} 个景区）")
    print("=" * 70)

    all_results = {}
    for name, csv_path, n_tracks in scenes:
        try:
            all_results[name] = run_scene(name, csv_path, n_tracks, poi_df,
                                          args.n_workers, args.seed, args.with_routes)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[name] = {'status': 'error', 'error': str(e)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {args.out}")

    # 汇总表
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"{'景区':<12}{'k':>4}{'POI区':>6}{'Sil':>8}{'DBI':>8}{'ARI':>8}{'纯度':>8}{'覆盖':>7}")
    for name in sorted(all_results):
        r = all_results[name]
        if 'status' in r and r.get('status') == 'error':
            print(f"{name:<12}  ERROR {r.get('error','')[:40]}")
            continue
        it, ex, co = r['internal'], r['external'], r['correspondence']
        ari = f"{ex['ari']:.4f}" if ex.get('ari') is not None else 'N/A'
        pur = f"{co['purity']:.3f}" if co.get('purity') is not None else 'N/A'
        cov = f"{co['gt_coverage']:.3f}" if co.get('gt_coverage') is not None else 'N/A'
        print(f"{name:<12}{it['final_k']:>4}{r['n_poi_regions']:>6}"
              f"{it['silhouette']:>8.4f}{it['davies_bouldin']:>8.4f}"
              f"{ari:>8}{pur:>8}{cov:>7}")


if __name__ == '__main__':
    main()
