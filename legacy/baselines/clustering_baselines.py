"""
聚类基线对比：K-Means / DBSCAN / HDBSCAN / Agglomerative / 标准谱聚类
与 STRAT 粒球谱聚类使用完全相同的输入和评估口径，保证公平对比。

公平性约定（与主方法一致）：
- 所有方法使用同一份采样停留点特征（load_and_preprocess，3D MinMax）。
- 超参数（k / eps / gamma / min_samples）在留出验证子集上按 Silhouette 选参，
  最终指标在全部采样点上计算（裸点谱聚类因 O(N³) 限制取 ≤2000 子集，见注释）。
- k 类方法的 k 搜索范围与 STRAT 相同（按 POI 区域数自适应或启发式）。
"""

import os
import sys
import argparse
import json
import time
import glob

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ablation'))
from run_clustering_ablation import (
    load_and_preprocess,
    evaluate_clustering,
    _validation_subset,
    _best_grid_balls,
    _spectral_rbf,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cluster'))
from scenery_route_clustering import generate_granular_balls, create_ball_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import SEED, GB_SAMPLE_SIZE, MIN_POINTS_FOR_SPLIT, adaptive_k_bounds

CLEANED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data-project', 'cleaned_labeled_data')
PAPER_SCENES = {"青城山", "峨眉山", "武侯祠博物馆"}


def _resolve_scenes(scene_name=None, scene_all=False, scene_paper=False, min_tracks=30):
    """解析要处理的景区列表，返回 [(name, csv_path), ...]"""
    if scene_name:
        csv_path = os.path.join(CLEANED_DIR, f"{scene_name}_cleaned.csv")
        if not os.path.exists(csv_path):
            print(f"错误: {csv_path} 不存在")
            return []
        return [(scene_name, csv_path)]
    csv_files = sorted(glob.glob(os.path.join(CLEANED_DIR, "*_cleaned.csv")))
    candidates = []
    for f in csv_files:
        name = os.path.basename(f).replace("_cleaned.csv", "")
        if scene_paper and name not in PAPER_SCENES:
            continue
        df_tmp = pd.read_csv(f, usecols=["trackId"])
        n = df_tmp["trackId"].nunique()
        del df_tmp
        if n >= min_tracks:
            candidates.append((name, f))
        else:
            print(f"  [跳过] {name}: 仅 {n} 条轨迹 (<{min_tracks})")
    return candidates


def _select_best(features, sel_idx, results):
    """在验证子集上选最优 (labels, kwargs)。results: [(sil, labels, meta), ...]"""
    valid = [(s, lab, meta) for s, lab, meta in results if lab is not None and s > -1.0]
    if not valid:
        return None, {}
    sil, labels, meta = max(valid, key=lambda x: x[0])
    return labels, meta


def _tune_k(features, sel_idx, fit_fn, k_lower, k_upper):
    """k 类方法：在 [k_lower, k_upper] 上按验证子集 silhouette 选 k。"""
    best_sil, best_labels, best_k = -1.0, None, k_lower
    for k in range(k_lower, k_upper + 1):
        labels = fit_fn(k)
        if labels is None or len(np.unique(labels)) < 2:
            continue
        sil = silhouette_score(features[sel_idx], labels[sel_idx])
        if sil > best_sil:
            best_sil, best_labels, best_k = sil, labels, k
    return best_labels, {'k': best_k, 'sel_sil': best_sil}


def _tune_dbscan(features, sel_idx):
    best_sil, best_labels, best_eps, best_ms = -1.0, None, 0.08, 5
    for eps in [0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.25]:
        for ms in [5, 10]:
            db = DBSCAN(eps=eps, min_samples=ms, metric='euclidean', n_jobs=-1)
            labels = db.fit_predict(features)
            valid = labels != -1
            if np.sum(valid) < 2 or len(np.unique(labels[valid])) < 2:
                continue
            sil = silhouette_score(features[sel_idx][labels[sel_idx] != -1], labels[sel_idx][labels[sel_idx] != -1])
            if sil > best_sil:
                best_sil, best_labels, best_eps, best_ms = sil, labels, eps, ms
    return best_labels, {'eps': best_eps, 'min_samples': best_ms, 'sel_sil': best_sil}


def _tune_hdbscan(features, sel_idx):
    try:
        import hdbscan
    except ImportError:
        return None, {}
    best_sil, best_labels, best_mcs = -1.0, None, 5
    for mcs in [5, 10, 15]:
        cl = hdbscan.HDBSCAN(min_cluster_size=mcs, metric='euclidean', cluster_selection_method='eom')
        labels = cl.fit_predict(features)
        valid = labels != -1
        if np.sum(valid) < 2 or len(np.unique(labels[valid])) < 2:
            continue
        sil = silhouette_score(features[sel_idx][labels[sel_idx] != -1], labels[sel_idx][labels[sel_idx] != -1])
        if sil > best_sil:
            best_sil, best_labels, best_mcs = sil, labels, mcs
    return best_labels, {'min_cluster_size': best_mcs, 'sel_sil': best_sil}


def _tune_spectral_gaussian(features, sel_idx, k_lower, k_upper):
    """裸点 RBF 谱聚类：γ 调参 + eigen-gap k（O(N³) 限制，取 ≤2000 子集）。
    返回 (labels, sub, meta)，labels 与 sub 严格对应。"""
    n_raw = min(2000, len(features))
    rng = np.random.RandomState(SEED)
    idx = np.sort(rng.choice(len(features), n_raw, replace=False))
    sub = features[idx]
    sub_sel = np.where(np.isin(idx, sel_idx))[0]
    best_sil, best_gamma, best_labels = -1.0, None, None
    for gamma in [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        labels = _spectral_rbf(sub, gamma, k_lower, min(k_upper, n_raw - 1))
        if labels is None or len(np.unique(labels)) < 2:
            continue
        if len(sub_sel) < 2:
            sil = silhouette_score(sub, labels)
        else:
            sil = silhouette_score(sub[sub_sel], labels[sub_sel])
        if sil > best_sil:
            best_sil, best_gamma, best_labels = sil, gamma, labels
    if best_labels is None:
        return None, sub, {}
    return best_labels, sub, {'gamma': best_gamma, 'sel_sil': best_sil, 'raw_n': n_raw}


def evaluate_baseline(name, features_scaled, labels, elapsed, extra=None):
    noise_mask = labels == -1
    noise_rate = float(np.mean(noise_mask)) if len(labels) > 0 else 0.0
    metrics = evaluate_clustering(features_scaled, labels)
    metrics['method'] = name
    metrics['noise_rate'] = noise_rate
    metrics['time_s'] = elapsed
    if extra:
        metrics.update(extra)
    return metrics


def run_one_scene(csv_path, output_dir, scene_name, sample_size=GB_SAMPLE_SIZE,
                  seed=SEED, n_poi_regions=None, n_workers=None):
    """对单个景区运行全部聚类基线。"""
    print(f"\n{'#'*60}")
    print(f"# [聚类基线] {scene_name}")
    print(f"{'#'*60}")

    cluster_features = load_and_preprocess(csv_path, sample_size, seed)
    n_points = len(cluster_features)
    if n_points < 100:
        print(f"  [跳过] 停留点不足 ({n_points})")
        return None
    sel_idx = _validation_subset(cluster_features, seed)
    print(f"  {n_points} stay points, 3D (lon/lat scaled + elev), δ/k 选参子集 {len(sel_idx)}")

    # STRAT（方法本体）：粒球 + 规则亲和 + 网格δ(验证子集) + eigen-gap k
    print("  Generating granular balls for STRAT...")
    ball_data_list = generate_granular_balls(cluster_features, min_points_for_split=MIN_POINTS_FOR_SPLIT)
    ball_dict = create_ball_dict(ball_data_list)
    ball_keys = list(ball_dict.keys())
    ball_centers = np.array([ball_dict[k].center for k in ball_keys])
    ball_radii = np.array([ball_dict[k].radius for k in ball_keys])
    n_balls = len(ball_dict)

    if n_poi_regions:
        k_lower, k_upper = adaptive_k_bounds(n_poi_regions, n_balls)
    else:
        k_lower, k_upper = 2, min(max(12, n_balls // 4), n_balls)
    k_upper = min(k_upper, n_balls)
    k_lower = max(2, min(k_lower, k_upper - 1))
    min_cb = max(8, n_balls // 16)
    print(f"  Balls: {n_balls}, k range: [{k_lower}, {k_upper}]")

    t0 = time.time()
    gm_labels, gm_delta, gm_k, gm_sel_sil = _best_grid_balls(
        ball_centers, ball_radii, cluster_features, k_lower, k_upper, min_cb, sel_idx, n_workers)
    gm_time = time.time() - t0
    gm_metrics = evaluate_baseline('STRAT (Ours)', cluster_features, gm_labels, gm_time)
    gm_metrics['best_delta'] = gm_delta
    gm_metrics['best_k'] = gm_k
    gm_metrics['n_balls'] = n_balls
    gm_metrics['k_lower'] = k_lower
    gm_metrics['k_upper'] = k_upper
    gm_metrics['sel_sil'] = gm_sel_sil
    print(f"  STRAT: Sil={gm_metrics['silhouette']:.4f}, DB={gm_metrics['davies_bouldin']:.4f}, "
          f"CH={gm_metrics['calinski_harabasz']:.2f}, k={gm_k}, δ={gm_delta:.1f}, time={gm_time:.1f}s")

    scene_results = {'strat': gm_metrics}

    # K-Means / Agglomerative：k 网格（同范围）选参
    for name, fn in [('K-Means', lambda k: KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(cluster_features)),
                     ('Agglomerative', lambda k: AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(cluster_features))]:
        t0 = time.time()
        labels, meta = _tune_k(cluster_features, sel_idx, fn, k_lower, k_upper)
        m = evaluate_baseline(name, cluster_features, labels if labels is not None else np.full(n_points, -1, dtype=int),
                              time.time() - t0, meta)
        scene_results[name.lower().replace(' ', '_')] = m
        print(f"  {name}: Sil={m['silhouette']:.4f}, DB={m['davies_bouldin']:.4f}, "
              f"k={m['n_clusters']}, time={m['time_s']:.1f}s")

    # DBSCAN / HDBSCAN：eps、min_samples / min_cluster_size 调参
    for name, fn in [('DBSCAN', _tune_dbscan), ('HDBSCAN', _tune_hdbscan)]:
        t0 = time.time()
        labels, meta = fn(cluster_features, sel_idx)
        m = evaluate_baseline(name, cluster_features, labels if labels is not None else np.full(n_points, -1, dtype=int),
                              time.time() - t0, meta)
        scene_results[name.lower()] = m
        print(f"  {name}: Sil={m['silhouette']:.4f}, DB={m['davies_bouldin']:.4f}, "
              f"k={m['n_clusters']}, noise={m['noise_rate']:.1%}, time={m['time_s']:.1f}s")

    # 裸点 RBF 谱聚类（≤2000 子集，γ 调参）
    t0 = time.time()
    labels, sub, meta = _tune_spectral_gaussian(cluster_features, sel_idx, k_lower, k_upper)
    raw_n = meta.get('raw_n', len(cluster_features))
    if labels is not None:
        m = evaluate_baseline('Spectral (Gaussian)', sub, labels, time.time() - t0, meta)
    else:
        m = evaluate_baseline('Spectral (Gaussian)', cluster_features, np.full(n_points, -1, dtype=int),
                              time.time() - t0, meta)
    scene_results['spectral_gaussian'] = m
    print(f"  Spectral (Gaussian): Sil={m['silhouette']:.4f}, DB={m['davies_bouldin']:.4f}, "
          f"k={m['n_clusters']}, γ={meta.get('gamma')}, raw_n={raw_n}, time={m['time_s']:.1f}s")

    return scene_results


def main():
    parser = argparse.ArgumentParser(description='Clustering Baselines Comparison')
    parser.add_argument('--scene', type=str, default=None, help='景区名称')
    parser.add_argument('--scene_all', action='store_true', help='对所有景区运行')
    parser.add_argument('--scene_paper', action='store_true',
                        help='只跑论文3个景区 (青城山/峨眉山/武侯祠)')
    parser.add_argument('--min_tracks', type=int, default=30,
                        help='跳过轨迹数少于该值的景区 (默认: 30)')
    parser.add_argument('--csv', type=str, default=None, help='(兼容旧用法) 直接指定 CSV')
    parser.add_argument('--output_dir', type=str, default='baselines/output')
    parser.add_argument('--sample_size', type=int, default=GB_SAMPLE_SIZE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--poi_regions', type=str, default='',
                        help='逗号分隔 "景区:POI区域数"，用于自适应 k 范围，如 峨眉山:35,龙泉:60')
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    poi_regions = {}
    if args.poi_regions:
        for kv in args.poi_regions.split(','):
            if ':' in kv:
                k, v = kv.split(':', 1)
                poi_regions[k.strip()] = int(v.strip())

    if args.csv:
        scenes = [("景区", args.csv)]
    else:
        scenes = _resolve_scenes(args.scene, args.scene_all, args.scene_paper, args.min_tracks)

    if not scenes:
        print("没有符合条件的景区。")
        return

    print("=" * 70)
    print("聚类基线对比实验（逐场景调参，验证子集选参）")
    print("=" * 70)
    print(f"将处理 {len(scenes)} 个景区")

    all_results = {}
    for scene_name, csv_path in scenes:
        r = run_one_scene(csv_path, args.output_dir, scene_name,
                          args.sample_size, args.seed,
                          n_poi_regions=poi_regions.get(scene_name),
                          n_workers=args.n_workers)
        if r is not None:
            all_results[scene_name] = r

    out_path = os.path.join(args.output_dir, 'clustering_baselines.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n所有结果已保存到 {out_path}")

    print(f"\n{'='*70}")
    print("LaTeX Table:")
    print(f"{'='*70}\n")
    order = ['strat', 'k-means', 'dbscan', 'hdbscan', 'agglomerative', 'spectral_gaussian']
    names = {
        'strat': 'STRAT (Ours)',
        'k-means': 'K-Means',
        'dbscan': 'DBSCAN',
        'hdbscan': 'HDBSCAN',
        'agglomerative': 'Agglomerative',
        'spectral_gaussian': 'Spectral (Gaussian)',
    }
    for key in order:
        m = all_results.get(key)
        if m is None:
            continue
        print(f"{names.get(key, key)}: Sil={m['silhouette']:.4f} DB={m['davies_bouldin']:.4f} "
              f"CH={m['calinski_harabasz']:.2f} k={m['n_clusters']} noise={m['noise_rate']:.1%} t={m['time_s']:.1f}s")
    for scene_name, scene_results in all_results.items():
        latex = (f"\n--- {scene_name} ---\n"
                 + "\\begin{tabular}{lcccccc}\n\\toprule\n"
                 + "\\textbf{Method} & Sil $\\uparrow$ & DB $\\downarrow$ & CH $\\uparrow$ & \\#k & Noise & t(s) \\\\\n\\midrule\n")
        for key in order:
            m = scene_results.get(key)
            if m is None:
                continue
            latex += (f"{names.get(key, key)} & {m['silhouette']:.4f} & {m['davies_bouldin']:.4f} & "
                      f"{m['calinski_harabasz']:.2f} & {m['n_clusters']} & {m['noise_rate']:.1%} & "
                      f"{m['time_s']:.1f} \\\\\n")
        latex += "\\bottomrule\n\\end{tabular}\n"
        print(latex)


if __name__ == '__main__':
    main()
