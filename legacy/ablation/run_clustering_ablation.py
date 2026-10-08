"""
消融实验：聚类侧消融

1. 粒球 vs 裸点：验证粒球抽象层的价值
   裸点谱聚类受 O(N³) 谱分解限制，取 ≤2000 点子集（这正是粒球抽象
   的动机——谱聚类作用在 M 个粒球而非 N 个原始点上）。
2. δ 分辨率敏感性：网格 0.1 步长是否足够（替代旧的"搜索策略"对比，
   因为 δ 拍离散后 BKOA≈网格，原对比无意义）。

选参与评估口径与主方法一致：δ/σ 在留出验证子集上选参，最终指标在
全部采样点上计算。
"""

import os
import sys
import argparse
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cluster'))
from scenery_route_clustering import (
    generate_granular_balls,
    create_ball_dict,
    perform_clustering_eigengap,
    compute_composite_score,
    _eval_delta,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import SEED, GB_SAMPLE_SIZE, MIN_POINTS_FOR_SPLIT, SELECTION_SPLIT


def load_and_preprocess(csv_path, sample_size=GB_SAMPLE_SIZE, random_state=SEED):
    """加载数据并进行预处理（与主方法一致：经纬度转米 + MinMax 3D）。"""
    rng = np.random.RandomState(random_state)
    df = pd.read_csv(csv_path)
    stay_mask = df['is_stop'] == 1
    stay_df = df[stay_mask].copy()

    features_raw = stay_df[['经度', '纬度', '海拔']].values.astype(np.float64)

    lat_mean_rad = np.radians(features_raw[:, 1].mean())
    lon_scale = np.cos(lat_mean_rad) * 111320.0
    lat_scale = 111320.0
    features_raw[:, 0] *= lon_scale
    features_raw[:, 1] *= lat_scale

    n_stay = len(features_raw)
    sample_size = min(sample_size, n_stay)
    if n_stay > sample_size:
        indices = rng.choice(n_stay, sample_size, replace=False)
        features_sampled = features_raw[indices]
    else:
        features_sampled = features_raw

    scaler = MinMaxScaler(feature_range=(0, 1))
    cluster_features = scaler.fit_transform(features_sampled)

    return cluster_features


def evaluate_clustering(features, labels):
    """计算无监督聚类指标。"""
    valid = labels != -1
    if np.sum(valid) < 2 or len(np.unique(labels[valid])) < 2:
        return {
            'silhouette': 0.0, 'davies_bouldin': 0.0,
            'calinski_harabasz': 0.0, 'composite': 0.0,
            'n_clusters': 0,
        }
    sil = silhouette_score(features[valid], labels[valid])
    db = davies_bouldin_score(features[valid], labels[valid])
    ch = calinski_harabasz_score(features[valid], labels[valid])
    comp, _ = compute_composite_score(features[valid], labels[valid])
    return {
        'silhouette': float(sil),
        'davies_bouldin': float(db),
        'calinski_harabasz': float(ch),
        'composite': float(comp),
        'n_clusters': int(len(np.unique(labels[valid]))),
    }


def _validation_subset(features, random_state):
    n = len(features)
    n_sel = max(int(n * SELECTION_SPLIT), min(50, n))
    rng = np.random.RandomState(random_state)
    return np.sort(rng.choice(n, n_sel, replace=False))


def _best_grid_balls(ball_centers, ball_radii, features, k_lower, k_upper, min_cb, sel_idx, n_workers=None):
    """网格搜索 δ（验证子集选参），返回 (best_labels, best_delta, best_k, best_sel_sil)。"""
    if n_workers is None:
        n_workers = 1
    best_sil, best_delta, best_labels, best_k = -1.0, 0.5, None, k_lower
    tasks = [(ball_centers, ball_radii, features, d, k_lower, k_upper, min_cb, sel_idx)
             for d in np.arange(0.1, 1.0 + 1e-9, 0.1)]
    if len(tasks) <= 4 or n_workers <= 1:
        for t in tasks:
            sil, k, d = _eval_delta(t)
            if sil > best_sil:
                best_sil, best_k, best_delta = sil, k, float(d)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_eval_delta, t): t for t in tasks}
            for fu in as_completed(futs):
                try:
                    sil, k, d = fu.result()
                except Exception:
                    continue
                if sil > best_sil:
                    best_sil, best_k, best_delta = sil, k, float(d)
    labels, k_final, _ = perform_clustering_eigengap(
        ball_centers, ball_radii, features, best_delta, k_lower, k_upper, min_cluster_balls=min_cb)
    return labels, best_delta, k_final, best_sil


def _spectral_rbf(features_sub, gamma, k_lower, k_upper):
    """RBF 亲和 + 谱聚类 + eigen-gap 定 k（裸点基线）。"""
    from sklearn.cluster import SpectralClustering
    from scipy.linalg import eigh
    aff = np.exp(-gamma * pairwise_distances(features_sub, features_sub) ** 2)
    aff = np.maximum(aff, 0)
    aff = 0.5 * (aff + aff.T)
    np.fill_diagonal(aff, 1.0)
    n = aff.shape[0]
    degree = np.sum(aff, axis=1)
    d_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
    L = np.eye(n) - d_sqrt_inv @ aff @ d_sqrt_inv
    kmax = min(k_upper, n - 1)
    if n < 3 or k_lower >= kmax:
        return None
    evals = eigh(L, eigvals_only=True, subset_by_index=[0, kmax])
    gaps = np.diff(evals)
    lo, hi = max(0, k_lower - 1), min(len(gaps), kmax - 1)
    if lo >= hi:
        return None
    k = int(np.argmax(gaps[lo:hi]) + lo + 1)
    k = max(k_lower, min(k, kmax))
    try:
        spec = SpectralClustering(n_clusters=k, affinity='precomputed',
                                  assign_labels='discretize', random_state=42, n_init=10)
        return spec.fit_predict(aff)
    except Exception:
        return None


def run_ablation_1_balls_vs_raw(cluster_features, random_state=SEED, n_workers=None):
    """消融实验 1：粒球 + 规则亲和 vs 裸点 RBF 谱聚类（同一输入口径）。"""
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    print("\n" + "=" * 60)
    print("消融实验 1: 粒球 vs 裸点")
    print("=" * 60)
    results = {}
    sel_idx = _validation_subset(cluster_features, random_state)

    # --- A: 粒球 + 规则亲和力 ---
    print("\n  [A] 粒球 + 规则亲和力...")
    t0 = time.time()
    ball_data_list = generate_granular_balls(cluster_features, min_points_for_split=MIN_POINTS_FOR_SPLIT)
    ball_dict = create_ball_dict(ball_data_list)
    ball_keys = list(ball_dict.keys())
    ball_centers = np.array([ball_dict[k].center for k in ball_keys])
    ball_radii = np.array([ball_dict[k].radius for k in ball_keys])
    n_balls = len(ball_dict)
    k_lower, k_upper = 2, min(max(12, n_balls // 4), n_balls)
    min_cb = max(8, n_balls // 16)
    labels, best_delta, best_k, best_sel_sil = _best_grid_balls(
        ball_centers, ball_radii, cluster_features, k_lower, k_upper, min_cb, sel_idx, n_workers)
    m = evaluate_clustering(cluster_features, labels)
    m['n_balls'] = n_balls
    m['best_delta'] = best_delta
    m['best_k'] = best_k
    m['best_sel_sil'] = best_sel_sil
    m['time_s'] = time.time() - t0
    results['with_balls'] = m
    print(f"    Balls: {n_balls}, Sil: {m['silhouette']:.4f}, DB: {m['davies_bouldin']:.4f}, "
          f"k: {m['n_clusters']}, time: {m['time_s']:.1f}s")

    # --- B: 裸点 RBF 谱聚类（σ 调参 + eigen-gap k）---
    # 谱分解 O(N³) 限制：取 ≤2000 点子集，这正是粒球抽象要解决的瓶颈。
    n_raw = min(2000, len(cluster_features))
    rng = np.random.RandomState(random_state)
    idx = np.sort(rng.choice(len(cluster_features), n_raw, replace=False))
    sub = cluster_features[idx]
    print(f"\n  [B] 裸点 RBF 谱聚类（N={n_raw}，O(N³) 限制，σ 在验证子集调参）...")
    t0 = time.time()
    sel_sub = _validation_subset(sub, random_state)
    best_sil, best_gamma, best_labels = -1.0, None, None
    for gamma in [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        lab = _spectral_rbf(sub, gamma, 2, min(12, n_raw - 1))
        if lab is None:
            continue
        if len(np.unique(lab)) < 2:
            continue
        sil = silhouette_score(sub[sel_sub], lab[sel_sub])
        if sil > best_sil:
            best_sil, best_gamma, best_labels = sil, gamma, lab
    if best_labels is None:
        best_labels = np.full(len(sub), -1, dtype=int)
    m = evaluate_clustering(sub, best_labels)
    m['n_balls'] = 0
    m['best_delta'] = float(best_gamma if best_gamma else 0.0)
    m['best_k'] = int(m['n_clusters'])
    m['time_s'] = time.time() - t0
    results['without_balls'] = m
    print(f"    Sil: {m['silhouette']:.4f}, DB: {m['davies_bouldin']:.4f}, "
          f"k: {m['n_clusters']}, γ={best_gamma}, time: {m['time_s']:.1f}s")

    return results


def run_ablation_2_delta_sensitivity(cluster_features, random_state=SEED, n_workers=None):
    """消融实验 2：δ 分辨率敏感性（验证网格 0.1 步长足够）。"""
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    print("\n" + "=" * 60)
    print("消融实验 2: δ 分辨率敏感性（0.05 vs 0.1）")
    print("=" * 60)

    ball_data_list = generate_granular_balls(cluster_features, min_points_for_split=MIN_POINTS_FOR_SPLIT)
    ball_dict = create_ball_dict(ball_data_list)
    ball_keys = list(ball_dict.keys())
    ball_centers = np.array([ball_dict[k].center for k in ball_keys])
    ball_radii = np.array([ball_dict[k].radius for k in ball_keys])
    n_balls = len(ball_dict)
    k_lower, k_upper = 2, min(max(12, n_balls // 4), n_balls)
    min_cb = max(8, n_balls // 16)
    sel_idx = _validation_subset(cluster_features, random_state)
    print(f"  Balls: {n_balls}, k ∈ [{k_lower}, {k_upper}]")

    fine_grid = np.arange(0.05, 1.0 + 1e-9, 0.05)
    tasks = [(ball_centers, ball_radii, cluster_features, d, k_lower, k_upper, min_cb, sel_idx)
             for d in fine_grid]
    sils = {}
    if n_workers <= 1:
        for t in tasks:
            sil, k, d = _eval_delta(t)
            sils[d] = (sil, k)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_eval_delta, t): t for t in tasks}
            for fu in as_completed(futs):
                try:
                    sil, k, d = fu.result()
                except Exception:
                    continue
                sils[d] = (sil, k)

    curve = sorted(sils.items())
    fine_best = max(curve, key=lambda x: x[1][0])
    coarse_vals = {d: sils[d] for d in np.arange(0.1, 1.0 + 1e-9, 0.1) if d in sils}
    coarse_best = max(coarse_vals.items(), key=lambda x: x[1][0])

    print(f"  Fine grid (0.05) best: δ={fine_best[0]:.2f}, sel-sil={fine_best[1][0]:.4f}, k={fine_best[1][1]}")
    print(f"  Coarse grid (0.10) best: δ={coarse_best[0]:.1f}, sel-sil={coarse_best[1][0]:.4f}, k={coarse_best[1][1]}")
    print(f"  差值: {fine_best[1][0] - coarse_best[1][0]:+.5f}")

    return {
        'n_balls': n_balls,
        'k_lower': k_lower,
        'k_upper': k_upper,
        'fine_best': {'delta': float(fine_best[0]), 'sel_sil': fine_best[1][0], 'k': fine_best[1][1]},
        'coarse_best': {'delta': float(coarse_best[0]), 'sel_sil': coarse_best[1][0], 'k': coarse_best[1][1]},
        'diff': float(fine_best[1][0] - coarse_best[1][0]),
        'curve': [{'delta': float(d), 'sel_sil': v[0], 'k': v[1]} for d, v in curve],
    }


def main():
    parser = argparse.ArgumentParser(description='Clustering Ablation Study')
    parser.add_argument('--csv', type=str, required=True,
                        help='清洗后的 CSV 路径')
    parser.add_argument('--output_dir', type=str, default='ablation/output')
    parser.add_argument('--sample_size', type=int, default=GB_SAMPLE_SIZE)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--ablation', type=str, nargs='*',
                        default=['balls_vs_raw', 'delta_sensitivity'])
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("STRAT 聚类侧消融实验")
    print("=" * 70)

    if not os.path.exists(args.csv):
        print(f"\n[ERROR] 数据文件不存在: {args.csv}")
        return

    cluster_features = load_and_preprocess(args.csv, args.sample_size, args.seed)
    print(f"\n[Data] {len(cluster_features)} stay points, {cluster_features.shape[1]}D features")

    all_results = {}
    if 'balls_vs_raw' in args.ablation:
        all_results['balls_vs_raw'] = run_ablation_1_balls_vs_raw(cluster_features, args.seed, args.n_workers)
    if 'delta_sensitivity' in args.ablation:
        all_results['delta_sensitivity'] = run_ablation_2_delta_sensitivity(cluster_features, args.seed, args.n_workers)

    print("\n" + "=" * 70)
    print("聚类消融实验结果汇总")
    print("=" * 70)
    if 'balls_vs_raw' in all_results:
        r = all_results['balls_vs_raw']
        for key, label in [('with_balls', '粒球+规则亲和'), ('without_balls', '裸点RBF谱聚类')]:
            m = r[key]
            print(f"  {label}: Sil={m['silhouette']:.4f}, DB={m['davies_bouldin']:.4f}, "
                  f"CH={m['calinski_harabasz']:.2f}, k={m['n_clusters']}, time={m['time_s']:.1f}s")
    if 'delta_sensitivity' in all_results:
        r = all_results['delta_sensitivity']
        print(f"  δ: fine={r['fine_best']['delta']:.2f}@{r['fine_best']['sel_sil']:.4f} "
              f"vs coarse={r['coarse_best']['delta']:.1f}@{r['coarse_best']['sel_sil']:.4f}, "
              f"diff={r['diff']:+.5f}")

    out_path = os.path.join(args.output_dir, os.path.basename(args.csv).replace('.csv', '_ablation.json'))
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
