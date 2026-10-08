"""
用 4 种聚类方法分别给原数据打 region_id 标签，产出 clustered CSV。
一次只跑一种聚类方法，避免内存爆炸。

用法：
  # 对指定景区跑一种方法
  python baselines/generate_clustered_csvs.py --method strat --scene 峨眉山
  # 对所有景区跑所有方法
  python baselines/generate_clustered_csvs.py --method all --scene_all
  # 只跑论文3个景区
  python baselines/generate_clustered_csvs.py --method strat --scene_paper
"""

import os
import sys
import argparse
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cluster'))
from scenery_route_clustering import _compute_region_centroids

# 论文核心景区 + 自动检测路径
CLEANED_DIR = Path(__file__).parent.parent / "data-project" / "cleaned_labeled_data"
PAPER_SCENES = {"青城山", "峨眉山", "武侯祠博物馆"}


def resolve_scenes(scene_name=None, scene_all=False, scene_paper=False, min_tracks=30):
    """解析要处理的景区列表，返回 [(name, csv_path), ...]"""
    if scene_name:
        csv_path = CLEANED_DIR / f"{scene_name}_cleaned.csv"
        if not csv_path.exists():
            print(f"错误: {csv_path} 不存在")
            sys.exit(1)
        return [(scene_name, csv_path)]

    csv_files = sorted(CLEANED_DIR.glob("*_cleaned.csv"))
    if scene_paper:
        csv_files = [f for f in csv_files if f.stem.replace("_cleaned", "") in PAPER_SCENES]

    candidates = []
    for f in csv_files:
        name = f.stem.replace("_cleaned", "")
        df_tmp = pd.read_csv(f, usecols=["trackId"])
        n = df_tmp["trackId"].nunique()
        del df_tmp
        if n >= min_tracks:
            candidates.append((name, f))
        else:
            print(f"  [跳过] {name}: 仅 {n} 条轨迹 (<{min_tracks})")
    return candidates


def _remap_contiguous(df):
    df = df[df['region_id'] != -1].copy()
    unique_regions = sorted(df['region_id'].unique())
    region_map = {old: new for new, old in enumerate(unique_regions)}
    df['region_id'] = df['region_id'].map(region_map)
    df['region_id'] = df['region_id'].astype(int)
    return df


def _assign_moving_points(df, stay_mask, centroids):
    moving_mask = ~stay_mask
    moving_indices = np.where(moving_mask)[0]
    if len(moving_indices) == 0 or len(centroids) == 0:
        return df
    moving_pts = df.loc[df.index[moving_indices], ['经度', '纬度']].values.astype(np.float64)
    c_arr = np.array([c for c in centroids.values()], dtype=np.float64)
    rids = np.array(list(centroids.keys()))
    dists_sq = np.sum((moving_pts[:, np.newaxis, :] - c_arr[np.newaxis, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists_sq, axis=1)
    df.loc[df.index[moving_indices], 'region_id'] = rids[nearest]
    return df


def _load_and_prepare(df):
    stay_mask = df['is_stop'] == 1
    features_raw = df.loc[stay_mask, ['经度', '纬度', '海拔']].values.astype(np.float64)
    lat_mean_rad = np.radians(features_raw[:, 1].mean())
    features_raw[:, 0] *= np.cos(lat_mean_rad) * 111320.0
    features_raw[:, 1] *= 111320.0
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features_raw)
    return df.copy(), stay_mask, features_scaled, features_raw


def cluster_strat(df, output_dir, scene_name, seed=42):
    print("  STRAT: 粒球 + 多因子亲和度 + eigen-gap 谱聚类（直接调用原始 pipeline）")
    from scenery_route_clustering import cluster_scenic_spots

    df_out, metrics_dict, ball_data_list, _, _ = cluster_scenic_spots(df)
    del ball_data_list
    gc.collect()

    n_regions = metrics_dict['n_regions']
    sil = metrics_dict['silhouette']
    k = metrics_dict['final_k']

    out_path = os.path.join(output_dir, f'{scene_name}_strat.csv')
    df_out.to_csv(out_path, index=False)
    print(f"    结果: k={k}, Sil={sil:.4f}, regions={n_regions}")
    print(f"    保存: {out_path}")
    return k, sil, n_regions


def cluster_kmeans(df, output_dir, scene_name, n_clusters=7, seed=42):
    print(f"  K-Means: k={n_clusters}")
    df, stay_mask, features_scaled, _ = _load_and_prepare(df)

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(features_scaled)
    del features_scaled
    gc.collect()

    stay_indices = np.where(stay_mask)[0]
    df.loc[df.index[stay_indices], 'region_id'] = labels
    centroids = _compute_region_centroids(df, stay_mask)
    df = _assign_moving_points(df, stay_mask, centroids)
    df = _remap_contiguous(df)

    out_path = os.path.join(output_dir, f'{scene_name}_kmeans.csv')
    df.to_csv(out_path, index=False)
    valid = labels != -1
    sil = silhouette_score(
        df.loc[df['is_stop'] == 1, ['经度', '纬度', '海拔']].values[:len(labels)][valid],
        labels[valid]) if np.sum(valid) >= 2 else 0.0
    n_regions = df['region_id'].nunique()
    print(f"    结果: k={n_clusters}, Sil={sil:.4f}, regions={n_regions}")
    print(f"    保存: {out_path}")
    del labels, df
    gc.collect()
    return n_clusters, sil, n_regions


def cluster_dbscan(df, output_dir, scene_name, seed=42):
    print("  DBSCAN: eps=0.001, min_samples=30（采样聚类+KDTree分配）")
    df, stay_mask, features_scaled, _ = _load_and_prepare(df)
    n_stay = len(features_scaled)

    sample_n = min(50000, max(10000, int(np.sqrt(n_stay) * 4)))
    idx = np.random.choice(n_stay, sample_n, replace=False)
    sampled_features = features_scaled[idx]
    print(f"    停留点: {n_stay:,}, 采样: {sample_n:,}")

    db = DBSCAN(eps=0.001, min_samples=30, metric='euclidean', n_jobs=-1)
    sampled_labels = db.fit_predict(sampled_features)
    n_clusters = len(np.unique(sampled_labels[sampled_labels != -1]))
    noise_rate = (sampled_labels == -1).mean()
    print(f"    采样聚类结果: {n_clusters} 个簇, 噪声率: {noise_rate:.1%}")
    del db, sampled_features
    gc.collect()

    # KDTree 分配全部停留点到最近采样点标签
    from scipy.spatial import KDTree
    tree = KDTree(features_scaled[idx])
    _, nearest = tree.query(features_scaled, k=1)
    labels = sampled_labels[nearest]
    labels[np.min(features_scaled - features_scaled[idx[nearest]], axis=1) > 0.05] = -1
    del tree, nearest, sampled_labels, idx
    gc.collect()

    stay_indices = np.where(stay_mask)[0]
    df.loc[df.index[stay_indices], 'region_id'] = labels
    centroids = _compute_region_centroids(df, stay_mask)
    df = _assign_moving_points(df, stay_mask, centroids)
    df = _remap_contiguous(df)
    del labels, centroids
    gc.collect()

    out_path = os.path.join(output_dir, f'{scene_name}_dbscan.csv')
    df.to_csv(out_path, index=False)
    n_regions = df['region_id'].nunique()
    print(f"    结果: k={n_regions}, regions={n_regions}")
    print(f"    保存: {out_path}")
    return n_regions, 0.0, n_regions


def cluster_spectral(df, output_dir, scene_name, n_clusters=7, seed=42):
    print(f"  Spectral (Gaussian): n_clusters={n_clusters}, 采样聚类+分配")
    df, stay_mask, features_scaled, features_raw = _load_and_prepare(df)
    n_stay = len(features_scaled)

    sample_n = min(10000, max(5000, int(np.sqrt(n_stay))))
    idx = np.random.choice(n_stay, sample_n, replace=False)
    sampled_features = features_scaled[idx]

    print(f"    停留点: {n_stay:,}, 采样: {sample_n:,}")
    spec = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=10.0,
                               assign_labels='discretize', random_state=seed, n_init=10,
                               n_jobs=1)
    sampled_labels = spec.fit_predict(sampled_features)
    del sampled_features, spec
    gc.collect()

    # 计算各簇质心（用采样点的质心）
    centroid_coords = {}
    for c in range(n_clusters):
        mask = sampled_labels == c
        if mask.sum() > 0:
            centroid_coords[c] = features_scaled[idx[mask]].mean(axis=0)
        else:
            centroid_coords[c] = features_scaled[idx[0]]

    # 向量化分配所有停留点到最近簇质心
    c_arr = np.array([centroid_coords[c] for c in range(n_clusters)], dtype=np.float64)
    dists_sq = np.sum((features_scaled[:, np.newaxis, :] - c_arr[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(dists_sq, axis=1).astype(np.int32)

    del features_scaled, features_raw, centroid_coords, idx, sampled_labels
    gc.collect()

    stay_indices = np.where(stay_mask)[0]
    df.loc[df.index[stay_indices], 'region_id'] = labels
    centroids = _compute_region_centroids(df, stay_mask)
    df = _assign_moving_points(df, stay_mask, centroids)
    df = _remap_contiguous(df)

    out_path = os.path.join(output_dir, f'{scene_name}_spectral.csv')
    df.to_csv(out_path, index=False)
    valid = labels != -1
    sil = silhouette_score(
        df.loc[df['is_stop'] == 1, ['经度', '纬度', '海拔']].values[:len(labels)][valid],
        labels[valid]) if np.sum(valid) >= 2 else 0.0
    n_regions = df['region_id'].nunique()
    print(f"    结果: k={n_clusters}, Sil={sil:.4f}, regions={n_regions}")
    print(f"    保存: {out_path}")
    del labels, df
    gc.collect()
    return n_clusters, sil, n_regions


METHODS = {
    'strat':   ('STRAT (Ours)',    cluster_strat),
    'kmeans':   ('K-Means',           cluster_kmeans),
    'dbscan':   ('DBSCAN',            cluster_dbscan),
    'spectral': ('Spectral (Gaussian)', cluster_spectral),
}


def main():
    parser = argparse.ArgumentParser(description='Generate clustered CSV (one method at a time)')
    parser.add_argument('--method', type=str, required=True,
                        choices=['strat', 'kmeans', 'dbscan', 'spectral', 'all'],
                        help='Clustering method to run, or "all"')
    parser.add_argument('--scene', type=str, default=None,
                        help='景区名称，留空则自动检测所有')
    parser.add_argument('--scene_all', action='store_true',
                        help='对所有有数据的景区运行')
    parser.add_argument('--scene_paper', action='store_true',
                        help='只跑论文3个景区 (青城山/峨眉山/武侯祠)')
    parser.add_argument('--min_tracks', type=int, default=30,
                        help='跳过轨迹数少于该值的景区 (默认: 30)')
    parser.add_argument('--output_dir', type=str, default='baselines/output')
    parser.add_argument('--k', type=int, default=7)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    scenes = resolve_scenes(args.scene, args.scene_all, args.scene_paper, args.min_tracks)
    if not scenes:
        print("没有符合条件的景区。")
        return

    print("=" * 70)
    print("生成聚类 CSV（一次一个方法，避免内存爆炸）")
    print("=" * 70)
    print(f"将处理 {len(scenes)} 个景区: {', '.join(s[0] for s in scenes)}")

    to_run = list(METHODS.keys()) if args.method == 'all' else [args.method]

    for scene_name, csv_path in scenes:
        print(f"\n{'='*70}")
        print(f"[景区] {scene_name}")
        print(f"{'='*70}")

        print(f"\n[加载] {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  行数: {len(df):,}, 轨迹数: {df['trackId'].nunique():,}")

        for method in to_run:
            name, func = METHODS[method]
            print(f"\n  [{method}] {name}")
            t0 = time.time()
            df_copy = df.copy()
            if method in ('kmeans', 'spectral'):
                result = func(df_copy, args.output_dir, scene_name, n_clusters=args.k, seed=args.seed)
            else:
                result = func(df_copy, args.output_dir, scene_name, seed=args.seed)
            del df_copy
            gc.collect()
            elapsed = time.time() - t0
            print(f"  耗时: {elapsed:.1f}s")
        del df
        gc.collect()

    print(f"\n完成。")



if __name__ == '__main__':
    main()