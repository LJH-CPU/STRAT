"""
全基线对比模块：聚类基线 + 预测基线

聚类基线（无监督，内部指标）：
  - KMeans, DBSCAN, HDBSCAN, Agglomerative, 标准谱聚类(高斯核)

预测基线（有监督，MAE + Window Acc）：
  - Region-Median, MLP, BiLSTM, Transformer
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline import build_knn_baseline, evaluate_knn, predict_knn
from data_utils import (
    extract_first_occurrence_sequences,
    build_region_metadata,
    split_by_route,
    GPS_SEG_DIM,
)


# ═══════════════════════════════════════════════════════════════
#  公共工具函数
# ═══════════════════════════════════════════════════════════════

def load_and_preprocess_cluster_data(csv_path, sample_size=10000, seed=42):
    np.random.seed(seed)
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
    sample_size = min(sample_size, max(5000, int(np.sqrt(n_stay))))
    if n_stay > sample_size:
        indices = np.random.choice(n_stay, sample_size, replace=False)
        features_sampled = features_raw[indices]
    else:
        features_sampled = features_raw
    scaler = MinMaxScaler(feature_range=(0, 1))
    cluster_features = scaler.fit_transform(features_sampled)
    return cluster_features, features_sampled


def eval_clustering(labels, features):
    valid = labels != -1
    if np.sum(valid) < 2 or len(np.unique(labels[valid])) < 2:
        return {'silhouette': 0.0, 'davies_bouldin': 0.0,
                'calinski_harabasz': 0.0, 'n_clusters': 0, 'noise_ratio': 0.0}
    sil = float(silhouette_score(features[valid], labels[valid]))
    db = float(davies_bouldin_score(features[valid], labels[valid]))
    ch = float(calinski_harabasz_score(features[valid], labels[valid]))
    n_clusters = int(len(np.unique(labels[valid])))
    noise_ratio = float(np.sum(labels == -1) / len(labels))
    return {'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch,
            'n_clusters': n_clusters, 'noise_ratio': noise_ratio}


def segments_to_cumulative(segments):
    return [0.0] + list(np.cumsum(segments))


# ═══════════════════════════════════════════════════════════════
#  聚类基线
# ═══════════════════════════════════════════════════════════════

def baseline_kmeans(cluster_features, n_clusters=7, seed=42):
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(cluster_features)
    elapsed = time.time() - t0
    m = eval_clustering(labels, cluster_features)
    m['time_s'] = elapsed
    m['method'] = 'KMeans'
    m['n_clusters'] = n_clusters
    return m


def baseline_dbscan(cluster_features, eps=0.08, min_samples=15):
    t0 = time.time()
    dbs = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbs.fit_predict(cluster_features)
    elapsed = time.time() - t0
    m = eval_clustering(labels, cluster_features)
    m['time_s'] = elapsed
    m['method'] = 'DBSCAN'
    m['eps'] = eps
    m['min_samples'] = min_samples
    return m


def baseline_hdbscan(cluster_features, min_cluster_size=15, min_samples=5):
    if not HAS_HDBSCAN:
        return {'method': 'HDBSCAN', 'error': 'hdbscan not installed'}
    t0 = time.time()
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           gen_min_span_tree=False)
    labels = hdb.fit_predict(cluster_features)
    elapsed = time.time() - t0
    m = eval_clustering(labels, cluster_features)
    m['time_s'] = elapsed
    m['method'] = 'HDBSCAN'
    return m


def baseline_agglomerative(cluster_features, n_clusters=7):
    t0 = time.time()
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels = agg.fit_predict(cluster_features)
    elapsed = time.time() - t0
    m = eval_clustering(labels, cluster_features)
    m['time_s'] = elapsed
    m['method'] = 'Agglomerative'
    m['n_clusters'] = n_clusters
    return m


def baseline_spectral_gaussian(cluster_features, n_clusters=7, gamma=3.0, seed=42):
    t0 = time.time()
    n = len(cluster_features)
    from sklearn.metrics.pairwise import rbf_kernel
    affinity = rbf_kernel(cluster_features, gamma=gamma)
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                            assign_labels='discretize', random_state=seed, n_init=10, n_jobs=1)
    labels = sc.fit_predict(affinity)
    elapsed = time.time() - t0
    m = eval_clustering(labels, cluster_features)
    m['time_s'] = elapsed
    m['method'] = 'Spectral (Gaussian)'
    m['n_clusters'] = n_clusters
    m['gamma'] = gamma
    return m


def run_all_clustering_baselines(csv_path, n_clusters=7, sample_size=10000, seed=42):
    print("=" * 60)
    print("Clustering Baselines")
    print("=" * 60)
    cluster_features, features_sampled = load_and_preprocess_cluster_data(
        csv_path, sample_size=sample_size, seed=seed)
    print(f"  Sampled {len(cluster_features)} stay points")

    results = {}
    methods = [
        ('KMeans',              lambda: baseline_kmeans(cluster_features, n_clusters, seed)),
        ('DBSCAN',              lambda: baseline_dbscan(cluster_features)),
        ('HDBSCAN',             lambda: baseline_hdbscan(cluster_features)),
        ('Agglomerative',       lambda: baseline_agglomerative(cluster_features, n_clusters)),
        ('Spectral(Gaussian)', lambda: baseline_spectral_gaussian(cluster_features, n_clusters)),
    ]

    for name, fn in methods:
        print(f"\n  [{name}] running...")
        m = fn()
        if 'error' in m:
            print(f"    SKIP: {m['error']}")
        else:
            print(f"    Sil={m['silhouette']:.4f}  DB={m['davies_bouldin']:.4f}  "
                  f"CH={m['calinski_harabasz']:.2f}  k={m['n_clusters']}  "
                  f"noise={m.get('noise_ratio', 0):.1%}  time={m['time_s']:.1f}s")
        results[name] = m

    print(f"\n{'Method':<22} {'Sil ↑':>8} {'DB ↓':>8} {'CH ↑':>10} {'k':>5} {'Noise%':>8} {'Time':>8}")
    print("-" * 72)
    for name in methods:
        m = results.get(name[0], {})
        if 'error' in m:
            print(f"{name[0]:<22} {'SKIP':>8}")
        else:
            print(f"{name[0]:<22} {m['silhouette']:8.4f} {m['davies_bouldin']:8.4f} "
                  f"{m['calinski_harabasz']:10.2f} {m['n_clusters']:5d} "
                  f"{m.get('noise_ratio', 0):7.1%} {m['time_s']:7.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════
#  预测基线
# ═══════════════════════════════════════════════════════════════

def baseline_region_median(train_records, test_records):
    region_dur = defaultdict(list)
    for r in train_records:
        for i, dur in enumerate(r['segment_durations']):
            rid = r['seq'][i] if i < len(r['seq']) else -1
            region_dur[rid].append(dur)
    region_median = {rid: float(np.median(vals)) if vals else 0.0
                     for rid, vals in region_dur.items()}
    global_median = float(np.median([d for vals in region_dur.values() for d in vals])) \
        if region_dur else 0.0

    results = []
    for r in test_records:
        pred_segs = []
        for i, dur in enumerate(r['segment_durations']):
            rid = r['seq'][i] if i < len(r['seq']) else -1
            pred_segs.append(region_median.get(rid, global_median))
        results.append({
            'track_id': r['track_id'], 'route_id': r['route_id'],
            'pred_segments': pred_segs, 'true_segments': r['segment_durations'],
            'n_segs': len(pred_segs),
        })
    return results


def evaluate_prediction_results(pred_results):
    seg_errs = []
    for r in pred_results:
        for i in range(r['n_segs']):
            err = abs(r['pred_segments'][i] - r['true_segments'][i])
            seg_errs.append(err)
    seg_mae = float(np.mean(seg_errs)) if seg_errs else 0.0

    all_cum = []
    dur_errs_min = []
    for r in pred_results:
        true_cum = segments_to_cumulative(r['true_segments'])
        pred_cum = segments_to_cumulative(r['pred_segments'])
        dur_errs_min.append(abs(pred_cum[-1] - true_cum[-1]) / 60.0)
        for i in range(1, len(true_cum)):
            all_cum.append(abs(pred_cum[i] - true_cum[i]))
    cum_mae = float(np.mean(all_cum)) if all_cum else 0.0
    dur_mae_s = float(np.mean(dur_errs_min) * 60) if dur_errs_min else 0.0

    window_accs = {}
    for wm in [15, 30, 60]:
        ws = wm * 60
        correct = 0
        total = 0
        for r in pred_results:
            true_cum = segments_to_cumulative(r['true_segments'])
            pred_cum = segments_to_cumulative(r['pred_segments'])
            for i in range(1, len(true_cum)):
                if abs(pred_cum[i] - true_cum[i]) < ws:
                    correct += 1
                total += 1
        window_accs[f'win_{wm}min'] = 100.0 * correct / max(total, 1) if total > 0 else 0.0

    return {'seg_mae': seg_mae, 'cum_mae': cum_mae, 'dur_mae_s': dur_mae_s,
            'window_acc': window_accs}


# ═══════════════════════════════════════════════════════════════
#  深度预测基线 (