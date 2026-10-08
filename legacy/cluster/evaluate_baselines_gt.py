"""
全基线 POI-GT 领域评测（证据②）

对每个景区，把 STRAT 与各基线方法（KMeans/DBSCAN/HDBSCAN/Agglomerative/
裸点谱聚类）的聚类标签，统一对齐到同一批采样停留点，喂给 POI 缓冲区区域
真值（复用 poi/ground_truth.py），计算 ARI/NMI/FMI + 簇纯度/GT 覆盖。

缓冲区：50m / 100m / 150m 三档（区域以 50m 合并，匹配半径三档），
报告匹配率。噪声点（pred==-1）与未匹配点（gt==-1）一律剔除后再算指标，
避免 DBSCAN 噪声拖低可比性。

跨 8 场景做 Wilcoxon 符号秩检验（STRAT vs 各基线，主缓冲区 100m）。
"""

import os
import sys
import json
import time
import glob
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'poi')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ablation')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'baselines')))

from config import SEED, GB_SAMPLE_SIZE, MIN_POINTS_FOR_SPLIT, SELECTION_SPLIT, adaptive_k_bounds
from scenery_route_clustering import (
    generate_granular_balls, create_ball_dict, assign_labels_by_containment,
)
from run_clustering_ablation import _best_grid_balls, _validation_subset, _spectral_rbf
from clustering_baselines import _tune_k, _tune_dbscan, _tune_hdbscan
import ground_truth as gt
from run_experiments import load_poi_regions, region_correspondence

RADII_M = [50, 100, 150]
PRIMARY_RADIUS_M = 100


# ---------------------------------------------------------------------------
# 预处理（与 load_and_preprocess 完全一致，额外返回索引用于 GT 对齐）
# ---------------------------------------------------------------------------

def _preprocess(csv_path, sample_size=GB_SAMPLE_SIZE, random_state=SEED):
    rng = np.random.RandomState(random_state)
    df = pd.read_csv(csv_path)
    stay_mask = df['is_stop'].values == 1
    stay_pos = np.where(stay_mask)[0]
    features_raw = df.loc[stay_pos, ['经度', '纬度', '海拔']].values.astype(np.float64)
    lat_mean_rad = np.radians(features_raw[:, 1].mean())
    features_raw[:, 0] *= np.cos(lat_mean_rad) * 111320.0
    features_raw[:, 1] *= 111320.0
    n_stay = len(features_raw)
    sample_size = min(sample_size, n_stay)
    if n_stay > sample_size:
        sel = rng.choice(n_stay, sample_size, replace=False)
        sel = np.sort(sel)
        features_sampled = features_raw[sel]
    else:
        sel = np.arange(n_stay)
        features_sampled = features_raw
    scaler = MinMaxScaler(feature_range=(0, 1))
    cluster_features = scaler.fit_transform(features_sampled)
    return cluster_features, sel, stay_pos


def _gt_on_sampled(csv_path, scene_pois, radius_deg, sel, stay_pos):
    """返回采样停留点的 POI GT 标签（未匹配为 -1）。"""
    traj = gt.match_trajectory_to_regions(csv_path, scene_pois, radius_deg, only_stay_points=True)
    gt_all = traj['gt_region_id'].values.astype(np.int32)
    gt_stay = gt_all[stay_pos]
    return gt_stay[sel]


def _ari_nmi_fmi(true, pred):
    valid = (true != -1) & (pred != -1)
    t, p = true[valid], pred[valid]
    if len(t) < 10 or len(np.unique(t)) < 2 or len(np.unique(p)) < 2:
        return None, None, None, len(t)
    return (float(adjusted_rand_score(t, p)),
            float(normalized_mutual_info_score(t, p, average_method='arithmetic')),
            float(fowlkes_mallows_score(t, p)), len(t))


def _spectral_gaussian_with_idx(features, sel_idx, k_lower, k_upper):
    """裸点 RBF 谱聚类（≤2000 子集），返回 (labels, sub_idx)。"""
    n_raw = min(2000, len(features))
    rng = np.random.RandomState(SEED)
    idx = np.sort(rng.choice(len(features), n_raw, replace=False))
    sub = features[idx]
    sub_sel = np.where(np.isin(idx, sel_idx))[0]
    best_sil, best_gamma, best_labels = -1.0, None, None
    for gamma in [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        lab = _spectral_rbf(sub, gamma, k_lower, min(k_upper, n_raw - 1))
        if lab is None or len(np.unique(lab)) < 2:
            continue
        sil = silhouette_score_unsafe(sub, sub_sel, lab)
        if sil > best_sil:
            best_sil, best_gamma, best_labels = sil, gamma, lab
    if best_labels is None:
        return np.full(n_raw, -1, dtype=int), idx
    return best_labels, idx


def silhouette_score_unsafe(features, sel_idx, labels):
    from sklearn.metrics import silhouette_score
    if len(sel_idx) >= 2 and len(np.unique(labels[sel_idx])) >= 2:
        return silhouette_score(features[sel_idx], labels[sel_idx])
    if len(np.unique(labels)) >= 2:
        return silhouette_score(features, labels)
    return -1.0


def run_scene(name, csv_path, poi_df, n_workers, seed):
    print(f"\n{'#'*64}\n# POI-GT 基线评测: {name}\n{'#'*64}")
    scene_pois = poi_df[poi_df['scenery'] == name]
    n_poi_regions = int(scene_pois['poi_region_id'].nunique()) if len(scene_pois) else 0
    if n_poi_regions < 2:
        print(f"  [跳过] POI 区域不足 ({n_poi_regions})")
        return None

    cluster_features, sel, stay_pos = _preprocess(csv_path, GB_SAMPLE_SIZE, seed)
    sel_idx = _validation_subset(cluster_features, seed)
    print(f"  {len(cluster_features)} 采样停留点, POI 区域数={n_poi_regions}")

    # 粒球（STRAT 与 k 范围共用）
    balls = generate_granular_balls(cluster_features, min_points_for_split=MIN_POINTS_FOR_SPLIT)
    bd = create_ball_dict(balls)
    keys = list(bd.keys())
    centers = np.array([bd[k].center for k in keys])
    radii = np.array([bd[k].radius for k in keys])
    n_balls = len(bd)
    k_lower, k_upper = adaptive_k_bounds(n_poi_regions, n_balls)
    k_upper = min(k_upper, n_balls)
    k_lower = max(2, min(k_lower, k_upper - 1))
    min_cb = max(8, n_balls // 16)
    print(f"  Balls={n_balls}, k∈[{k_lower},{k_upper}]")

    # 各方法标签（统一在 cluster_features 上，或标注子集索引）
    gm_labels, gm_delta, gm_k, _ = _best_grid_balls(centers, radii, cluster_features,
                                                    k_lower, k_upper, min_cb, sel_idx, n_workers)
    from sklearn.cluster import KMeans, AgglomerativeClustering
    km_labels, _ = _tune_k(cluster_features, sel_idx,
                           lambda k: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(cluster_features),
                           k_lower, k_upper)
    ag_labels, _ = _tune_k(cluster_features, sel_idx,
                           lambda k: AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(cluster_features),
                           k_lower, k_upper)
    db_labels, db_meta = _tune_dbscan(cluster_features, sel_idx)
    hdb_labels, hdb_meta = _tune_hdbscan(cluster_features, sel_idx)
    sp_labels, sp_idx = _spectral_gaussian_with_idx(cluster_features, sel_idx, k_lower, k_upper)

    methods = {
        'STRAT': (gm_labels, None),
        'K-Means': (km_labels, None),
        'Agglomerative': (ag_labels, None),
        'DBSCAN': (db_labels, None),
        'HDBSCAN': (hdb_labels, None),
        'Spectral-Gauss': (sp_labels, sp_idx),
    }

    out = {'n_poi_regions': n_poi_regions, 'n_balls': n_balls,
           'gm_delta': gm_delta, 'gm_k': gm_k, 'radii': {}}
    for r_m in RADII_M:
        gt_sampled = _gt_on_sampled(csv_path, scene_pois, r_m / 111000.0, sel, stay_pos)
        n_stay = len(sel)
        n_labeled = int((gt_sampled != -1).sum())
        out['radii'][r_m] = {
            'match_rate': n_labeled / n_stay * 100 if n_stay else 0.0,
            'n_labeled': n_labeled,
            'methods': {},
        }
        for mname, (labels, sub_idx) in methods.items():
            if labels is None:
                continue
            if sub_idx is not None:
                t = gt_sampled[sub_idx]
                p = labels
            else:
                t = gt_sampled
                p = labels
            ari, nmi, fmi, nn = _ari_nmi_fmi(t, p)
            valid = (t != -1) & (p != -1)
            corr = region_correspondence(p[valid], t[valid])
            out['radii'][r_m]['methods'][mname] = {
                'ari': ari, 'nmi': nmi, 'fmi': fmi, 'n_used': nn,
                'purity': corr.get('purity'), 'gt_coverage': corr.get('gt_coverage'),
            }
            ari_s = f"{ari:.4f}" if ari is not None else 'N/A'
            print(f"  [{r_m}m] {mname:<16} ARI={ari_s} 纯度={corr.get('purity')} 覆盖={corr.get('gt_coverage')}")

    return out


def wilcoxon_table(results):
    """主缓冲区 100m，STRAT vs 各基线的 ARI 差异 Wilcoxon。"""
    methods = ['K-Means', 'DBSCAN', 'HDBSCAN', 'Agglomerative', 'Spectral-Gauss']
    scenes = [s for s, r in results.items() if r]
    tab = {}
    for m in methods:
        pairs = []
        for s in scenes:
            gm = results[s]['radii'][PRIMARY_RADIUS_M]['methods'].get('STRAT', {}).get('ari')
            bm = results[s]['radii'][PRIMARY_RADIUS_M]['methods'].get(m, {}).get('ari')
            if gm is None or bm is None:
                continue
            pairs.append((gm, bm))
        if len(pairs) >= 2:
            try:
                stat, p = stats.wilcoxon([a - b for a, b in pairs])
            except ValueError:
                stat, p = None, None
            tab[m] = {'n_pairs': len(pairs), 'gm_wins': sum(1 for a, b in pairs if a > b),
                      'wins_ties': sum(1 for a, b in pairs if a >= b),
                      'stat': float(stat) if stat is not None else None,
                      'p': float(p) if p is not None else None}
    return tab


def main():
    parser = argparse.ArgumentParser(description='全基线 POI-GT 领域评测（证据②）')
    parser.add_argument('--scenery', nargs='*', default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--min_tracks', type=int, default=30)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--out', type=str, default='cluster/output/baselines_poi_gt.json')
    args = parser.parse_args()

    poi_df = load_poi_regions()

    if args.scenery:
        scenes = []
        for name in args.scenery:
            p = os.path.join('data-project', 'cleaned_labeled_data', f'{name}_cleaned.csv')
            if os.path.exists(p):
                scenes.append((name, p))
    elif args.all:
        files = sorted(glob.glob(os.path.join('data-project', 'cleaned_labeled_data', '*_cleaned.csv')))
        scenes = []
        for f in files:
            name = os.path.basename(f).replace('_cleaned.csv', '')
            try:
                n = pd.read_csv(f, usecols=['trackId'])['trackId'].nunique()
            except Exception:
                continue
            if n >= args.min_tracks:
                scenes.append((name, f))
    else:
        parser.error('请指定 --scenery 或 --all')

    results = {}
    for name, p in scenes:
        try:
            r = run_scene(name, p, poi_df, args.n_workers, args.seed)
            if r:
                results[name] = r
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = {'error': str(e)}

    wil = wilcoxon_table(results)
    out = {'results': results, 'wilcoxon': wil, 'primary_radius_m': PRIMARY_RADIUS_M}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== 汇总（100m 缓冲区） ===")
    print(f"{'景区':<12}{'STRAT-ARI':>9}{'KMeans-ARI':>11}{'DBSCAN-ARI':>11}{'Agglo-ARI':>10}{'Spec-ARI':>9}{'匹配率':>8}")
    for s in sorted(results):
        r = results[s]
        if 'error' in r:
            continue
        m = r['radii'][PRIMARY_RADIUS_M]['methods']
        a = lambda x: f"{m[x]['ari']:.3f}" if m.get(x, {}).get('ari') is not None else 'N/A'
        print(f"{s:<12}{a('STRAT'):>9}{a('K-Means'):>11}{a('DBSCAN'):>11}{a('Agglomerative'):>10}{a('Spectral-Gauss'):>9}"
              f"{r['radii'][PRIMARY_RADIUS_M]['match_rate']:>7.1f}%")
    print("\n=== Wilcoxon (STRAT vs 基线, ARI@100m) ===")
    for m, w in wil.items():
        print(f"  {m:<16} n={w['n_pairs']} STRAT胜/平={w['wins_ties']} p={w['p']}")
    print(f"\n结果已保存: {args.out}")


if __name__ == '__main__':
    main()
