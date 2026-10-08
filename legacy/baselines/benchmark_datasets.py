"""
标准基准数据集测试（证据①）：证明粒球谱聚类的通用有效性。

数据集（sklearn 内置，离线可跑）：
- Iris / Wine / Digits（真实标签）
- make_blobs 若干组：不同 K、重叠、簇大小不均衡

方法：
- STRAT：粒球 + 规则亲和 + 网格δ(验证子集) + eigen-gap 自动 k
- 基线：KMeans / Agglomerative（k=真实类别数，对基线有利）、
  DBSCAN（eps 调参）、裸点 RBF 谱聚类（γ 调参，k=真实类别数）

指标：ARI / NMI / FMI / silhouette，5 个种子 mean±std；
跨数据集 Wilcoxon 符号秩检验（STRAT vs 各基线）。

口径：StandardScaler；超过 1200 点的数据集按种子子采样到 1200（所有方法
用同一份数据，保证公平）。
"""

import os
import sys
import json
import time
import argparse

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_wine, load_digits, make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             fowlkes_mallows_score, silhouette_score)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cluster')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ablation')))
from scenery_route_clustering import generate_granular_balls, create_ball_dict
from run_clustering_ablation import _best_grid_balls, _validation_subset

SEEDS = [42, 100, 2024, 7, 123]


def load_datasets():
    iris = load_iris()
    wine = load_wine()
    digits = load_digits()
    ds = {
        'iris': (iris.data, iris.target),
        'wine': (wine.data, wine.target),
        'digits': (digits.data, digits.target),
    }
    rng = np.random.RandomState(42)
    blob_specs = {
        'blobs_k4': dict(n_samples=1200, centers=4, cluster_std=0.6),
        'blobs_k6': dict(n_samples=1200, centers=6, cluster_std=0.8),
        'blobs_k8': dict(n_samples=1200, centers=8, cluster_std=0.5),
        'blobs_overlap': dict(n_samples=1200, centers=4, cluster_std=1.3),
    }
    for name, spec in blob_specs.items():
        X, y = make_blobs(random_state=rng, **spec)
        ds[name] = (X, y)
    # 簇大小不均衡：分别生成再拼接
    xs, ys = [], []
    sizes = [600, 250, 200, 100, 50]
    centers = rng.uniform(-6, 6, (len(sizes), 2))
    for i, (n, c) in enumerate(zip(sizes, centers)):
        Xc, yc = make_blobs(n_samples=n, centers=[c], cluster_std=0.6, random_state=rng)
        xs.append(Xc)
        ys.append(np.full(n, i))
    ds['blobs_imb'] = (np.vstack(xs), np.concatenate(ys))
    return ds


def _cluster_strat(X):
    balls = generate_granular_balls(X, min_points_for_split=20)
    bd = create_ball_dict(balls)
    keys = list(bd.keys())
    centers = np.array([bd[k].center for k in keys])
    radii = np.array([bd[k].radius for k in keys])
    n_balls = len(bd)
    k_lower, k_upper = 2, min(max(12, n_balls // 4), n_balls)
    min_cb = max(8, n_balls // 16)
    sel_idx = _validation_subset(X, 42)
    labels, delta, k, sil = _best_grid_balls(centers, radii, X, k_lower, k_upper, min_cb, sel_idx, 1)
    return labels, {'delta': delta, 'k': k, 'n_balls': n_balls}


def _cluster_kmeans(X, k, seed):
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X)


def _cluster_agglo(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(X)


def _tune_dbscan_bench(X, sel_idx):
    best_sil, best_lab, best_eps = -1.0, None, None
    for eps in np.arange(0.1, 1.6, 0.15):
        lab = DBSCAN(eps=eps, min_samples=5, metric='euclidean').fit_predict(X)
        valid = lab != -1
        if np.sum(valid) < 2 or len(np.unique(lab[valid])) < 2:
            continue
        sil = silhouette_score(X[sel_idx][lab[sel_idx] != -1], lab[sel_idx][lab[sel_idx] != -1])
        if sil > best_sil:
            best_sil, best_lab, best_eps = sil, lab, eps
    if best_lab is None:
        return np.full(len(X), -1, dtype=int), None
    return best_lab, float(best_eps)


def _tune_spectral_bench(X, k, sel_idx):
    best_sil, best_lab, best_g = -1.0, None, None
    for gamma in [3.0, 10.0, 30.0, 100.0]:
        try:
            lab = SpectralClustering(n_clusters=k, affinity='rbf', gamma=gamma,
                                     assign_labels='discretize', random_state=42, n_init=10).fit_predict(X)
        except Exception:
            continue
        if len(np.unique(lab)) < 2:
            continue
        sil = silhouette_score(X[sel_idx], lab[sel_idx])
        if sil > best_sil:
            best_sil, best_lab, best_g = sil, lab, gamma
    if best_lab is None:
        return np.full(len(X), -1, dtype=int), None
    return best_lab, float(best_g)


def _metrics(X, y_true, labels):
    valid = labels != -1
    t, p = y_true[valid], labels[valid]
    if len(t) < 10 or len(np.unique(t)) < 2 or len(np.unique(p)) < 2:
        return None
    return {
        'ari': float(adjusted_rand_score(t, p)),
        'nmi': float(normalized_mutual_info_score(t, p, average_method='arithmetic')),
        'fmi': float(fowlkes_mallows_score(t, p)),
        'sil': float(silhouette_score(X, labels)) if len(np.unique(labels)) >= 2 else 0.0,
    }


def run_dataset(name, X_full, y_full):
    """对单个数据集跑全部方法，5 种子 mean±std。"""
    n_full, true_k = len(X_full), len(np.unique(y_full))
    sc = StandardScaler()
    X_full = sc.fit_transform(X_full)

    results = {'true_k': true_k, 'n_full': n_full}
    method_scores = {m: [] for m in ['STRAT', 'K-Means', 'DBSCAN', 'Agglomerative', 'Spectral']}

    # γ/eps 调参用固定种子数据（减少重复计算）
    sel_tune = _validation_subset(X_full, 42)

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        if n_full > 1200:
            idx = np.sort(rng.choice(n_full, 1200, replace=False))
            X, y = X_full[idx], y_full[idx]
        else:
            X, y = X_full, y_full
        sel_idx = _validation_subset(X, seed)

        strat_lab, strat_meta = _cluster_strat(X)
        km_lab = _cluster_kmeans(X, true_k, seed)
        ag_lab = _cluster_agglo(X, true_k)
        db_lab, db_eps = _tune_dbscan_bench(X, sel_idx)
        sp_lab, sp_g = _tune_spectral_bench(X, true_k, sel_idx)

        for mname, lab in [('STRAT', strat_lab), ('K-Means', km_lab),
                           ('DBSCAN', db_lab), ('Agglomerative', ag_lab),
                           ('Spectral', sp_lab)]:
            m = _metrics(X, y, lab)
            if m is not None:
                method_scores[mname].append(m)
            else:
                method_scores[mname].append(None)

    def agg(lst, key):
        vals = [v[key] for v in lst if v is not None]
        if not vals:
            return None
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'n': len(vals)}

    out = {'true_k': true_k, 'n': min(n_full, 1200)}
    for mname, lst in method_scores.items():
        out[mname] = {k: agg(lst, k) for k in ['ari', 'nmi', 'fmi', 'sil']}
    out['strat_k'] = strat_meta['k']
    out['strat_delta'] = strat_meta['delta']
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='baselines/output/benchmark_datasets.json')
    args = parser.parse_args()

    datasets = load_datasets()
    results = {}
    for name, (X, y) in datasets.items():
        t0 = time.time()
        print(f"\n=== {name} (n={len(X)}, k={len(np.unique(y))}) ===")
        results[name] = run_dataset(name, X, y)
        for m in ['STRAT', 'K-Means', 'DBSCAN', 'Agglomerative', 'Spectral']:
            r = results[name][m]
            ari = f"{r['ari']['mean']:.4f}±{r['ari']['std']:.4f}" if r['ari'] else 'N/A'
            print(f"  {m:<14} ARI={ari}")
        print(f"  (STRAT auto-k={results[name]['strat_k']}, δ={results[name]['strat_delta']})  {time.time()-t0:.1f}s")

    # Wilcoxon: STRAT vs 各基线 ARI
    methods = ['K-Means', 'DBSCAN', 'Agglomerative', 'Spectral']
    wil = {}
    for m in methods:
        pairs = []
        for name, r in results.items():
            g = r['STRAT']['ari']; b = r[m]['ari']
            if g and b:
                pairs.append((g['mean'], b['mean']))
        if len(pairs) >= 2:
            try:
                stat, p = stats.wilcoxon([a - b for a, b in pairs])
            except ValueError:
                stat, p = None, None
            wil[m] = {'n': len(pairs),
                      'strat_win': sum(1 for a, b in pairs if a > b),
                      'stat': float(stat) if stat is not None else None,
                      'p': float(p) if p is not None else None}
    results['_wilcoxon_ari'] = wil

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== Wilcoxon (STRAT vs 基线, ARI) ===")
    for m, w in wil.items():
        print(f"  {m:<14} n={w['n']} STRAT胜={w['strat_win']} p={w['p']}")
    print(f"\n结果已保存: {args.out}")


if __name__ == '__main__':
    main()
