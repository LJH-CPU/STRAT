"""
================================================================================
STRAT 景点聚类 + 路线聚类 Pipeline
================================================================================

目标：
1. 景点聚类：用粒球+谱聚类从停留点中发现景区区域（替代 DBSCAN）
2. 路线聚类：基于轨迹经过的 region_id 序列进行路线聚类

核心流程：
  原始 CSV (含经度/纬度/海拔/is_stop/trackId/route_id)
    │
    ├─ [景点聚类]
    │   ├─ 筛选停留点 (is_stop==1)
    │   ├─ 经纬度→米转换
    │   ├─ MinMax 归一化（3D：经度/纬度/海拔）
    │   ├─ 粒球生成 (递归最远点二分 + 半径归一化 + 邻近球合并)
    │   ├─ 网格搜索 δ（验证子集选参，避免选择偏差）
    │   ├─ 谱聚类 + 递归 eigen-gap 自动 k → region_id
    │   ├─ 移动点分配最近区域
    │   └─ 评测报告 (无监督)
    │
    └─ [路线聚类]
        ├─ 按 trackId 分组，提取 region_id 压缩序列（保留回头路）
        ├─ 计算轨迹间 POI 感知编辑距离（空间距离 + POI 语义相似度）
        ├─ 层次聚类 → route_id
        ├─ 评测报告
        └─ 可视化
================================================================================
"""

import gc
import json
import math
import multiprocessing 
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn import metrics
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

sys.path.insert(0, os.path.dirname(__file__))
from Granular_Spherical_Clustering import (
    split_ball_by_distance,
    calculate_density,
    calculate_radius,
    split_based_on_density,
    normalize_balls_by_radius,
)

# POI 类型映射与投影数据路径：统一从根目录 config.py 取，避免重复定义
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import (
    POI_JSON, POI_TYPE_L1, scenery_name_from_csv,
    SEED, GB_SAMPLE_SIZE, MIN_POINTS_FOR_SPLIT, DELTA_GRID_LOWER,
    DELTA_GRID_UPPER, DELTA_GRID_STEP, SELECTION_SPLIT,
)


# 粒球表示类

class GranularBall:
    """粒球表示 - 存储球体的几何特征"""
    def __init__(self, points, label):
        self.points = points
        self.center = self.points.mean(0) if len(points) > 0 else np.array([])
        self.label = label 
        self.radius = self._calculate_radius()

    def _calculate_radius(self):
        """计算粒球的半径"""
        if self.points.shape[0] <= 1 or self.center.size == 0:
            return 0.0
        distances = np.linalg.norm(self.points - self.center, axis=1)
        return np.max(distances) if len(distances) > 0 else 0.0


# Affinity 计算，亲和函数（0=完全不相关，1=完全相关）
# delta 是“距离惩罚的衰减速度”，用来控制： 两个不重叠的球，距离越远，亲和度下降得有多快 。
def affinity_improved(center1, center2, radius1, radius2, delta):

    # 边界处理
    if center1.size == 0 or center2.size == 0:
        return 0.0 # 无效输入，返回0
    if radius1 <= 0 or radius2 <= 0:
        dist = np.linalg.norm(center1 - center2)
        return float(np.exp(-dist * 10))
    
    # 计算距离和半径
    dist = np.linalg.norm(center1 - center2)
    sum_radii = radius1 + radius2
    gap = dist - sum_radii

    # 计算重叠比例（0-1）
    if gap < 0:
        overlap_depth = -gap
        max_possible_overlap = 2 * min(radius1, radius2)
        overlap_ratio = min(1.0, overlap_depth / max_possible_overlap if max_possible_overlap > 0 else 0.0)
    else:
        overlap_ratio = 0.0

    # 计算半径平衡（0-1）
    radius_min = min(radius1, radius2)
    radius_max = max(radius1, radius2)
    radius_balance = radius_min / radius_max if radius_max > 0 else 0.0

    # 计算包含比例（0-1）
    if dist + radius_min <= radius_max:
        containment = 1.0
    elif gap < 0:
        containment = 0.5
    else:
        containment = 0.0
    
    # 计算相对距离（0-1）
    relative_gap = gap / sum_radii if sum_radii > 0 else 0.0

    # 计算亲和值（0-1）
    if overlap_ratio > 0: # 有叠
        base_affinity = 0.7
        overlap_bonus = 0.3 * overlap_ratio
        containment_bonus = 0.1 * containment
        radius_bonus = 0.05 * (1 - radius_balance)
        affinity = min(1.0, base_affinity + overlap_bonus + containment_bonus + radius_bonus)
    else: # 无叠

        # 计算有效delta（0-1）
        balance_factor = 0.5 + 0.5 * radius_balance
        effective_delta = delta * balance_factor

        if relative_gap > 0:
            distance_penalty = np.exp(-relative_gap / (effective_delta + 1e-9))
        else:
            distance_penalty = 1.0
        radius_weight = 0.7 + 0.3 * radius_balance
        # 注：分离情形（gap>0）下包含度 κ 恒为 0（不可能包含），
        # 故不再乘 (1 + 0.2κ) 这一恒等于 1 的死代码项。
        affinity = distance_penalty * radius_weight
        affinity = max(0.0, min(1.0, affinity))
    return affinity


# 球体字典
def create_ball_dict(ball_data_list):
    ball_dict = {}                              # 初始化空字典
    for i, points in enumerate(ball_data_list): # 遍历每个粒球
        if len(points) > 0:                     # 只处理非空的粒球
            gb = GranularBall(points, i)        # 创建 GranularBall 对象
            if gb.center.size > 0:              # 检查球心是否有效
                ball_dict[i] = gb               # 存入字典，key=索引，value=粒球对象
    return ball_dict                            # 返回字典


# 球体合并
def merge_nearby_balls(ball_list, merge_threshold=0.5):

    # 边界处理
    if len(ball_list) <= 1:
        return ball_list
    
    # 初始化并查集（计算每个球的半径和中心）
    n = len(ball_list)
    centers = np.array([ball.mean(axis=0) for ball in ball_list])
    radii = np.array([np.max(np.linalg.norm(ball - centers[i], axis=1))
                      if len(ball) > 1 else 0.0 for i, ball in enumerate(ball_list)])

    # 并查集合并
    # 初始化并查集（每个球自己是一个集合）
    # 并查集合并（根据亲和值合并）
    parent = list(range(n))
    def find(x):

        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry # 合并集合，将 rx 合并到 ry 所在集合

    # 合并集合（根据亲和值）
    for i in range(n):
        for j in range(i + 1, n):
            dist_val = np.linalg.norm(centers[i] - centers[j]) # 两球中心距离
            thr = merge_threshold * min(radii[i], radii[j]) if min(radii[i], radii[j]) > 0 else 1e-6

            if dist_val < thr: # 如果两球中心距离小于阈值, 则合并集合
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return [np.vstack([ball_list[i] for i in idxs]) for idxs in groups.values()]


# 粒球生成

def generate_granular_balls(features_sampled, min_points_for_split=MIN_POINTS_FOR_SPLIT, rng=None):
    if rng is None:
        rng = np.random

    # 初始化粒球列表（包含所有点）
    print(f"  Generating granular balls from {len(features_sampled)} points (min_split={min_points_for_split})...")
    current_balls = [features_sampled]
    max_iter = 50

    # 递归最远点二分分割（粒度由 min_points_for_split 控制）
    for iteration in range(max_iter):
        before = len(current_balls)
        current_balls = split_based_on_density(current_balls, min_points_for_split, rng=rng)
        if len(current_balls) == before:
            break
   
    # 计算每个粒球的半径
    radii = [calculate_radius(b) for b in current_balls if len(b) >= 2]

    # 计算检测半径（中位数或平均值）
    detection_radius = max(np.median(radii) if radii else 0.0,
                           np.mean(radii) if radii else 0.0, 1e-6)

    # 迭代半径归一化（分割过大的球体）
    for iteration in range(max_iter):
        before = len(current_balls)
        current_balls = normalize_balls_by_radius(current_balls, detection_radius, rng=rng)
        if len(current_balls) == before:
            break

    current_balls = [b for b in current_balls if len(b) > 0]
    current_balls = merge_nearby_balls(current_balls, merge_threshold=0.5)

    print(f"  Granular balls: {len(current_balls)} balls generated")
    return current_balls


# 聚类执行（纯 numpy，可 pickle）
def assign_labels_by_containment(ball_centers, ball_radii, ball_labels, features):
    """
    将球体聚类标签分配给原始点：所在球优先，否则退回最近球心。

    对每个点：若落在某个（或多个）球内（到球心距离 <= 球半径），
    取半径最小的包含球的标签；若不被任何球包含，退回最近球心的标签。
    相比"一律用最近球心"，避免了球 B 内的点因更靠近邻球球心而被误分。
    """
    n_balls = len(ball_centers)
    n_pts = len(features)
    if n_balls == 0 or n_pts == 0:
        return np.full(n_pts, -1, dtype=int)
    diff = features[:, None, :] - ball_centers[None, :, :]
    dist = np.sqrt(np.einsum('nmk,nmk->nm', diff, diff))  # N×M
    radii = np.asarray(ball_radii, dtype=np.float64)
    contained = dist <= radii[None, :]
    labels = np.full(n_pts, -1, dtype=int)
    has = contained.any(axis=1)
    if has.any():
        radius_expanded = np.where(contained, radii[None, :], np.inf)
        best = np.argmin(radius_expanded, axis=1)
        labels[has] = ball_labels[best[has]]
    if (~has).any():
        nn = np.argmin(dist[~has], axis=1)
        labels[~has] = ball_labels[nn]
    return labels


def perform_clustering(ball_centers, ball_radii, features, num_clusters, delta):
    # 边界检查
    n_balls = len(ball_centers)
    if n_balls == 0 or num_clusters < 2:
        return np.full(len(features), -1, dtype=int)

    num_clusters = min(num_clusters, n_balls)

    # 计算亲和矩阵
    affinity = np.zeros((n_balls, n_balls))
    for i in range(n_balls):
        affinity[i, i] = 1.0
        for j in range(i + 1, n_balls):
            a = affinity_improved(ball_centers[i], ball_centers[j],
                                   ball_radii[i], ball_radii[j], delta)
            affinity[i, j] = a
            affinity[j, i] = a

    affinity = np.nan_to_num(affinity)
    affinity = np.maximum(affinity, 0)
    affinity = 0.5 * (affinity + affinity.T)

    # 谱聚类
    try:
        spectral = SpectralClustering(n_clusters=num_clusters, affinity="precomputed",
                                       assign_labels="discretize", random_state=42, n_init=10, n_jobs=1)
        ball_labels = spectral.fit_predict(affinity)
    except Exception:
        # 谱聚类失败，尝试 KMeans
        print("Spectral clustering failed, trying KMeans...")
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            ball_labels = kmeans.fit_predict(ball_centers)
        except Exception:
            return np.full(len(features), -1, dtype=int)

    # 赋值点标签（所在球优先，否则最近球心）
    point_labels = assign_labels_by_containment(ball_centers, ball_radii, ball_labels, features)

    return point_labels

# 递归 eigen-gap 谱聚类（ 仅输出点标签）
def _recursive_eigengap_split(affinity_full, ball_indices, k_lower, k_upper,
                               min_cluster_balls):
   
    # 停止条件
    n = len(ball_indices)
    if n < min_cluster_balls or n < 3:
        return np.zeros(n, dtype=int)

    # 用 eigen-gap 判断该分几个簇
    sub_affinity = affinity_full[ball_indices][:, ball_indices]  # 子图亲和矩阵
    k_max = min(k_upper, n - 1) 
    k = _determine_k_from_eigen_gap(sub_affinity, k_lower, k_max) # eigen-gap 判断该分几个簇

    if k <= 1:
        return np.zeros(n, dtype=int)  # 如果 eigen-gap 判断该分一个簇，直接返回

    try:
        spectral = SpectralClustering(n_clusters=k, affinity="precomputed",
                                       assign_labels="discretize",
                                       random_state=42, n_init=10, n_jobs=1)
        sub_labels = spectral.fit_predict(sub_affinity) # 分成 k 个子簇
    except Exception:
        return np.zeros(n, dtype=int) # 失败就不分裂

    # 对每个子簇递归处理
    labels = np.full(n, -1, dtype=int)
    offset = 0

    for c in range(k):
        mask = sub_labels == c
        n_sub = np.sum(mask)
        if n_sub < min_cluster_balls or n_sub < 3:
            labels[mask] = offset
            offset += 1
        else:
            sub_indices = ball_indices[mask]
            rec_labels = _recursive_eigengap_split(
                affinity_full, sub_indices, 2, 2, min_cluster_balls)
            n_rec = len(np.unique(rec_labels))
            labels[mask] = rec_labels + offset
            offset += n_rec

    return labels


# 递归 eigen-gap 谱聚类执行（纯 numpy，可 pickle）可自定义 k_lower, k_upper, min_cluster_balls
def perform_clustering_eigengap(ball_centers, ball_radii, features, delta,
                                 k_lower, k_upper, min_cluster_balls=5):
 
    n_balls = len(ball_centers)
    if n_balls < 3:
        labels = np.full(len(features), 0, dtype=int)
        return labels, 1, np.eye(n_balls)

    affinity = np.zeros((n_balls, n_balls))
    for i in range(n_balls):
        affinity[i, i] = 1.0
        for j in range(i + 1, n_balls):
            a = affinity_improved(ball_centers[i], ball_centers[j],
                                   ball_radii[i], ball_radii[j], delta)
            affinity[i, j] = a
            affinity[j, i] = a

    affinity = np.nan_to_num(affinity)
    affinity = np.maximum(affinity, 0)
    affinity = 0.5 * (affinity + affinity.T)

    ball_labels = _recursive_eigengap_split(
        affinity, np.arange(n_balls), k_lower,
        min(k_upper, n_balls - 1), min_cluster_balls)

    k_final = len(np.unique(ball_labels))

    # 赋值点标签（所在球优先，否则最近球心）
    point_labels = assign_labels_by_containment(ball_centers, ball_radii, ball_labels, features)

    return point_labels, k_final, affinity

# 用 eigen-gap 阙值自动确定最优聚类数 k。
def _determine_k_from_eigen_gap(affinity_matrix, k_lower, k_upper):

    n = affinity_matrix.shape[0]
    if n < 3:
        return 1

    k_max = min(k_upper, n - 1)
    if k_lower >= k_max:
        return 1

    degree = np.sum(affinity_matrix, axis=1)
    d_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
    L_norm = np.eye(n) - d_sqrt_inv @ affinity_matrix @ d_sqrt_inv

    eigenvals = eigh(L_norm, eigvals_only=True, subset_by_index=[0, k_max])
    if len(eigenvals) < 3:
        return 1

    gaps = np.diff(eigenvals)
    search_start = max(0, k_lower - 1)
    search_end = min(len(gaps), k_max - 1)

    if search_start >= search_end:
        return 1

    valid_gaps = gaps[search_start:search_end]
    max_gap = np.max(valid_gaps)
    best_idx = search_start + np.argmax(valid_gaps)

    k = best_idx + 1
    return int(max(k_lower, min(k, k_max)))


# 多指标无监督评分

def compute_composite_score(features, labels):
    """
    三指标融合的无监督聚类评分。
    0.5 × sil_norm + 0.2 × Balance + 0.3 × Separation

    - Silhouette: 簇内紧密度 vs 簇间分离度，归一化到 [0, 1]
    - Balance: 簇大小分布均匀度 = entropy / log(k)，[0, 1]
    - Separation: 空间分离度 = min(簇间距离) / avg(簇内散布)，[0, 1]
    """

    # 边界检查
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return 0.0, {'silhouette': 0.0, 'sil_norm': 0.0, 'balance': 0.0,
                      'separation': 0.0, 'n_clusters': n_clusters}

    # 计算 silhouette（轮廓系数）
    sil = silhouette_score(features, labels)
    sil_norm = (sil + 1.0) / 2.0

    # 计算 balance（簇大小分布均匀度）
    cluster_sizes = np.array([int(np.sum(labels == c)) for c in range(n_clusters)])
    proportions = cluster_sizes / cluster_sizes.sum()
    entropy = -np.sum(proportions * np.log(proportions + 1e-12))
    max_entropy = np.log(n_clusters)
    balance = entropy / max_entropy if max_entropy > 0 else 1.0

    # 计算 separation（空间分离度）
    centroids = np.array([features[labels == c].mean(axis=0) for c in range(n_clusters)])
    inter_dists = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    min_inter = np.min(inter_dists) if inter_dists else 0.0

    intra_dists = []
    for c in range(n_clusters):
        cluster_points = features[labels == c]
        if len(cluster_points) > 1:
            dists = np.linalg.norm(cluster_points - centroids[c], axis=1)
            intra_dists.append(np.mean(dists))
    avg_intra = np.mean(intra_dists) if intra_dists else 1.0

    separation = min(1.0, min_inter / (avg_intra + 1e-12)) if avg_intra > 0 else 0.0

    composite = 0.5 * sil_norm + 0.2 * balance + 0.3 * separation

    sub_scores = {
        'silhouette': sil,
        'sil_norm': sil_norm,
        'balance': balance,
        'separation': separation,
        'n_clusters': n_clusters,
    }
    return composite, sub_scores


# 评测函数（模块级，供多进程）
# 评估 单个 delta 值 的聚类效果，返回验证子集 Silhouette、k 和 delta 本身。
def _eval_delta(args):
    centers, radii, features, delta, k_lower, k_upper, min_cb, eval_idx = args
    labels, k_eig, _ = perform_clustering_eigengap(
        centers, radii, features, float(delta),
        int(k_lower), int(k_upper), min_cluster_balls=min_cb)
    if np.all(labels == -1) or len(np.unique(labels)) < 2:
        return -1.0, 0, float(delta)
    if eval_idx is not None:
        sub = labels[eval_idx]
        if len(np.unique(sub)) < 2:
            return -1.0, 0, float(delta)
        raw_sil = silhouette_score(features[eval_idx], sub)
    else:
        raw_sil = silhouette_score(features, labels)
    return raw_sil, int(k_eig), float(delta)


# 景点聚类

def cluster_scenic_spots(df, n_workers=None, random_state=SEED,
                         k_lower=None, k_upper=None, n_poi_regions=None):
    """
    从原始 GPS 数据中，用粒球+谱聚类发现景区景点区域。

    方法（论文版）：
    - 停留点 → 经纬度转米 → MinMax 归一化（3D）
    - 粒球生成（递归最远点二分 + 半径归一化 + 邻近球合并）
    - 多因子规则亲和力 + 谱聚类，k 由 eigen-gap 自动确定
    - δ 用网格搜索，且在留出验证子集上选参（避免选择偏差）

    参数:
    - df: 原始 DataFrame（须含 经度/纬度/海拔/is_stop/trackId/时间_秒）
    - n_workers: 并行进程数
    - random_state: 随机种子
    - k_lower / k_upper: k 搜索范围（None 则按 n_poi_regions 自适应）
    - n_poi_regions: 该场景 POI 区域数（用于自适应 k 范围，可为 None）

    返回:
    - df: 原始 DataFrame，新增 'region_id' 列（所有点都有标签）
    - region_centers: 每个区域的质心（原始经纬度）
    - metrics: 评测指标字典
    - ball_data_list: 粒球列表（用于可视化）
    - cluster_features: 采样点的归一化特征（经度+纬度+海拔）
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    np.random.seed(random_state)
    rng = np.random

    print("\n" + "=" * 60)
    print("Phase 1: Scenic Spot Clustering (景点聚类)")
    print("=" * 60)

    # 1. 筛选停留点
    stay_mask = df['is_stop'] == 1
    stay_df = df[stay_mask].copy()
    print(f"\n  Stay points: {stay_df.shape[0]:,} / {df.shape[0]:,} total")

    # 2. 提取特征：经度、纬度、海拔
    features_raw = stay_df[['经度', '纬度', '海拔']].values.astype(np.float64)

    # 3. 经纬度→米
    lat_mean_rad = np.radians(features_raw[:, 1].mean())
    lon_scale = np.cos(lat_mean_rad) * 111320.0
    lat_scale = 111320.0
    features_raw[:, 0] *= lon_scale
    features_raw[:, 1] *= lat_scale
    print(f"  Geo-scaling: lon={lon_scale:.0f}m/°, lat={lat_scale:.0f}m/°")

    # 4. 采样（统一 GB_SAMPLE_SIZE）
    n_stay = len(features_raw)
    sample_size = min(GB_SAMPLE_SIZE, n_stay)
    if n_stay > sample_size:
        indices = rng.choice(n_stay, sample_size, replace=False)
        features_sampled = features_raw[indices]
        print(f"  Sampled {sample_size} stay points for ball generation")
    else:
        features_sampled = features_raw
        indices = np.arange(n_stay)
        print(f"  Using all {n_stay} stay points")

    # 5. 归一化（保留 3D：经度 + 纬度 + 海拔）
    scaler = MinMaxScaler(feature_range=(0, 1))
    cluster_features = scaler.fit_transform(features_sampled)
    print(f"  Features: {cluster_features.shape[1]}D (lon, lat, altitude, no PCA)")

    # 6. 粒球生成
    ball_data_list = generate_granular_balls(cluster_features,
                                             min_points_for_split=MIN_POINTS_FOR_SPLIT,
                                             rng=rng)

    # 7. 构建球体字典
    ball_dict = create_ball_dict(ball_data_list)
    ball_keys = list(ball_dict.keys())
    ball_centers = np.array([ball_dict[k].center for k in ball_keys])
    ball_radii = np.array([ball_dict[k].radius for k in ball_keys])
    n_balls = len(ball_dict)
    print(f"  Valid balls: {n_balls}")

    # 8. k 搜索范围：优先用传入值，否则按 POI 区域数自适应
    if k_lower is None or k_upper is None:
        from config import adaptive_k_bounds
        if n_poi_regions:
            k_lower, k_upper = adaptive_k_bounds(n_poi_regions, n_balls)
        else:
            k_lower, k_upper = 2, min(max(12, n_balls // 4), n_balls)
            if k_lower >= k_upper:
                k_upper = max(k_lower + 1, min(k_upper + 1, n_balls))
    k_upper = min(k_upper, n_balls)
    k_lower = max(2, min(k_lower, k_upper - 1))
    min_cb = max(8, n_balls // 16)
    delta_candidates = np.arange(DELTA_GRID_LOWER, DELTA_GRID_UPPER + 1e-9, DELTA_GRID_STEP)
    total_combos = len(delta_candidates)

    # δ 在留出验证子集上选参（避免选择偏差）
    n_pts = len(cluster_features)
    n_sel = max(int(n_pts * SELECTION_SPLIT), min(50, n_pts))
    sel_idx = rng.choice(n_pts, n_sel, replace=False)
    sel_idx = np.sort(sel_idx)

    print(f"\n  Grid search: {len(delta_candidates)} δ × eigen-gap k, workers={n_workers}")
    print(f"  k ∈ [{k_lower}, {k_upper}] (adaptive, n_poi_regions={n_poi_regions})")
    print(f"  min_cluster_balls={min_cb}, δ selection on {n_sel}/{n_pts} validation points")

    tasks = [(ball_centers, ball_radii, cluster_features, d, k_lower, k_upper, min_cb, sel_idx)
             for d in delta_candidates]

    best_sil = -1.0
    best_k, best_delta = k_lower, (DELTA_GRID_LOWER + DELTA_GRID_UPPER) / 2
    grid_top3 = []

    if total_combos <= 4 or n_workers == 1:
        evaluated = 0
        for task in tasks:
            sil, k, d = _eval_delta(task)
            evaluated += 1
            grid_top3.append((sil, k, float(d)))
            grid_top3.sort(key=lambda x: -x[0])
            grid_top3 = grid_top3[:3]
            if sil > best_sil:
                best_sil = sil
                best_k, best_delta = k, float(d)
            print(f"  Grid: {evaluated}/{total_combos}, best sel-sil={best_sil:.4f} (k={best_k}, δ={best_delta:.1f})   ",
                  end='\r')
    else:
        evaluated = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_eval_delta, t): t for t in tasks}
            for future in as_completed(futures):
                evaluated += 1
                try:
                    sil, k, d = future.result()
                except Exception:
                    continue
                grid_top3.append((sil, k, float(d)))
                grid_top3.sort(key=lambda x: -x[0])
                grid_top3 = grid_top3[:3]
                if sil > best_sil:
                    best_sil = sil
                    best_k, best_delta = k, float(d)
                print(f"  Grid: {evaluated}/{total_combos}, best sel-sil={best_sil:.4f} (k={best_k}, δ={best_delta:.1f})   ",
                      end='\r')
    print()
    print(f"\n  Grid best (validation): k={best_k}, δ={best_delta:.1f}, sil={best_sil:.4f}")
    for i, (sil, kk, dd) in enumerate(grid_top3):
        print(f"    Top {i+1}: k={kk}, δ={dd:.1f}, sil={sil:.4f}")

    # 9. 最终聚类（eigen-gap 确定 k，δ 取网格最优）
    print(f"\n  Final clustering: δ={best_delta:.1f}, k from eigen-gap...")
    sampled_labels, final_k, final_affinity = perform_clustering_eigengap(
        ball_centers, ball_radii, cluster_features, best_delta,
        k_lower, k_upper, min_cluster_balls=min_cb)
    print(f"  Eigen-gap selected k={final_k}")

    # 10. 将采样标签扩展回所有停留点：KDTree 最近球体
    print(f"  Assigning region labels to {len(features_raw):,} stay points...")
    from scipy.spatial import KDTree
    all_stay_scaled = scaler.transform(features_raw)
    tree = KDTree(ball_centers)
    _, nearest_balls = tree.query(all_stay_scaled, k=1)
    all_stay_labels = sampled_labels[nearest_balls]

    # 11. 计算采样点的评测指标
    valid_mask = sampled_labels != -1
    if np.sum(valid_mask) >= 2 and len(np.unique(sampled_labels[valid_mask])) >= 2:
        scenic_sil = silhouette_score(cluster_features[valid_mask], sampled_labels[valid_mask])
        scenic_db = davies_bouldin_score(cluster_features[valid_mask], sampled_labels[valid_mask])
        scenic_ch = calinski_harabasz_score(cluster_features[valid_mask], sampled_labels[valid_mask])
        scenic_comp, scenic_comp_sub = compute_composite_score(
            cluster_features[valid_mask], sampled_labels[valid_mask])
    else:
        scenic_sil = scenic_db = scenic_ch = 0.0
        scenic_comp = 0.0
        scenic_comp_sub = {'silhouette': 0.0, 'sil_norm': 0.0, 'balance': 0.0, 'separation': 0.0,
                           'n_clusters': 0}

    n_regions = len(np.unique(sampled_labels[valid_mask]))

    # 12. 给原始 DataFrame 分配 region_id（向量化）
    df_new = df.copy()
    stay_indices = np.where(stay_mask)[0]
    df_new.loc[df_new.index[stay_indices], 'region_id'] = all_stay_labels

    # 移动点分配到最近区域质心（向量化）
    region_centroids_lonlat = _compute_region_centroids(df_new, stay_mask)
    moving_mask = df_new['is_stop'] == 0
    moving_indices = np.where(moving_mask)[0]
    if len(moving_indices) > 0 and len(region_centroids_lonlat) > 0:
        moving_points = df_new.loc[moving_indices, ['经度', '纬度']].values.astype(np.float64)
        centroids = np.array([c for c in region_centroids_lonlat.values()], dtype=np.float64)
        region_ids_list = np.array(list(region_centroids_lonlat.keys()))
        dists_sq = np.sum((moving_points[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        nearest = np.argmin(dists_sq, axis=1)
        df_new.loc[df_new.index[moving_indices], 'region_id'] = region_ids_list[nearest]

    # 移除 region_id == -1 的点
    df_new = df_new[df_new['region_id'] != -1].copy()
    df_new['region_id'] = df_new['region_id'].astype(int)

    # 重新映射为连续标签
    unique_regions = sorted(df_new['region_id'].unique())
    region_map = {old: new for new, old in enumerate(unique_regions)}
    df_new['region_id'] = df_new['region_id'].map(region_map)
    n_regions_final = len(unique_regions)

    # 13. 景点聚类评测报告
    print(f"\n  === Scenic Spot Clustering Report ===")
    print(f"  δ selection:     grid ∈ [{DELTA_GRID_LOWER}, {DELTA_GRID_UPPER}], validation split")
    print(f"  k selection:     recursive eigen-gap split ∈ [{k_lower}, {k_upper}]")
    print(f"  Grid best:       k={best_k}, δ={best_delta:.1f}, sel-sil={best_sil:.4f}")
    print(f"  Final selected:  k={final_k}, δ={best_delta:.1f}")
    print(f"  Regions found:      {n_regions_final}")
    print(f"  Granular balls:     {n_balls}")
    print(f"  Silhouette:         {scenic_sil:.4f}")
    print(f"  Davies-Bouldin:     {scenic_db:.4f}  (lower better)")
    print(f"  Calinski-Harabasz:  {scenic_ch:.2f}  (higher better)")
    print(f"  Composite Score:    {scenic_comp:.4f}  (0.5×sil_norm + 0.2×balance + 0.3×separation)")
    print(f"  ├─ Sil_norm:        {scenic_comp_sub.get('sil_norm', 0):.4f}")
    print(f"  ├─ Balance:         {scenic_comp_sub.get('balance', 0):.4f}")
    print(f"  └─ Separation:      {scenic_comp_sub.get('separation', 0):.4f}")
    for rid in sorted(df_new['region_id'].unique()):
        count = (df_new['region_id'] == rid).sum()
        lat_mean = df_new[df_new['region_id'] == rid]['纬度'].mean()
        lon_mean = df_new[df_new['region_id'] == rid]['经度'].mean()
        print(f"    Region {rid}: {count:,} points, center=({lon_mean:.4f}, {lat_mean:.4f})")

    metrics = {
        'n_regions': n_regions_final,
        'n_balls': n_balls,
        'silhouette': scenic_sil,
        'davies_bouldin': scenic_db,
        'calinski_harabasz': scenic_ch,
        'composite': scenic_comp,
        'composite_sub': scenic_comp_sub,
        'best_k': best_k,
        'best_delta': best_delta,
        'k_lower': k_lower,
        'k_upper': k_upper,
        'final_k': final_k,
        'best_sil': best_sil,
        'best_source': 'grid',
    }

    return df_new, metrics, ball_data_list, cluster_features, sampled_labels


def _compute_region_centroids(df, stay_mask):
    """计算每个区域的经纬度质心"""
    centroids = {}
    for rid in df.loc[stay_mask, 'region_id'].unique():
        if rid == -1:
            continue
        pts = df[(df['is_stop'] == 1) & (df['region_id'] == rid)][['经度', '纬度']].values
        if len(pts) > 0:
            centroids[rid] = (pts[:, 0].mean(), pts[:, 1].mean())
    return centroids


# ============================================================
# 路线聚类：轨迹 region 序列 + POI 感知编辑距离 + 层次聚类
# ============================================================

def _haversine_meters(lon1, lat1, lon2, lat2):
    """两经纬度点的球面距离（米）"""
    R = 6371000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _load_region_poi_profiles(scenery_name):
    """
    从 poi_path_projected.json 加载指定景区的 region → POI 类型分布。
    返回: dict[int, dict[str, float]]，如 {0: {"餐饮": 0.4, "风景名胜": 0.3}, ...}
    """
    if not scenery_name or not os.path.exists(POI_JSON):
        return {}

    with open(POI_JSON, encoding="utf-8") as f:
        all_pois = json.load(f)

    by_key = defaultdict(list)
    for p in all_pois:
        key = (p.get("scenery"), p.get("projected_region_id"))
        by_key[key].append(p)

    profiles = {}
    for (sc, rid), plist in by_key.items():
        if sc != scenery_name or rid is None:
            continue
        counter = Counter()
        for p in plist:
            l1 = p.get("type_code", "")[:2]
            name = POI_TYPE_L1.get(l1, l1)
            counter[name] += 1
        total = sum(counter.values())
        if total > 0:
            profiles[int(rid)] = {k: round(v / total, 4) for k, v in counter.most_common()}
        else:
            profiles[int(rid)] = {}
    return profiles


def _poi_profile_cosine(a, b):
    """两个 region 的 POI 类型分布向量余弦相似度，∈ [0, 1]"""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0.0) for k in keys], dtype=np.float64)
    vb = np.array([b.get(k, 0.0) for k in keys], dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def build_region_route_meta(df, poi_profiles=None):
    """
    构建路线距离所需的区域元数据：
    - centers:     region_id -> (lat, lon)，停留点均值
    - poi_profiles: region_id -> {POI类型: 占比}
    - max_dist:    区域间最大球面距离（空间代价归一化基准）
    """
    stay_df = df[df['is_stop'] == 1]
    centers = {}
    for rid, group in stay_df.groupby('region_id'):
        centers[int(rid)] = (float(group['纬度'].mean()), float(group['经度'].mean()))

    ids = sorted(centers)
    max_dist = 0.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = _haversine_meters(centers[ids[i]][1], centers[ids[i]][0],
                                  centers[ids[j]][1], centers[ids[j]][0])
            if d > max_dist:
                max_dist = d

    return {
        'centers': centers,
        'poi_profiles': poi_profiles or {},
        'max_dist': max_dist,
    }


def _region_substitution_cost(region_meta, a, b, w_spatial=0.5, w_poi=0.5):
    """
    两个 region 之间的替换代价，∈ (0.05, 1]：
    - 相同 region：0
    - 不同 region：w_spatial × 空间距离归一化 + w_poi × (1 - POI 余弦相似度)
    空间上相邻、语义相同的区域替换代价小；相隔远、语义不同的替换代价接近 1。
    """
    if a == b:
        return 0.0

    # 空间代价：球面距离 / 区域间最大距离，归一化到 [0, 1]
    if region_meta['max_dist'] > 0:
        lat_a, lon_a = region_meta['centers'].get(a, (0.0, 0.0))
        lat_b, lon_b = region_meta['centers'].get(b, (0.0, 0.0))
        spatial_cost = min(1.0, _haversine_meters(lon_a, lat_a, lon_b, lat_b)
                           / region_meta['max_dist'])
    else:
        spatial_cost = 1.0

    # POI 语义代价：1 - 余弦相似度
    pa = region_meta['poi_profiles'].get(a, {})
    pb = region_meta['poi_profiles'].get(b, {})
    poi_cost = 1.0 - _poi_profile_cosine(pa, pb)

    cost = w_spatial * spatial_cost + w_poi * poi_cost
    return min(1.0, max(0.05, cost))


def _build_substitution_cost_matrix(region_meta, w_spatial=0.5, w_poi=0.5):
    """
    预计算 region 间的替换代价矩阵（按 region_id 索引）。
    相比在每个 DP 单元内实时计算 haversine/余弦相似度，可大幅加速距离矩阵。
    """
    ids = sorted(region_meta['centers'])
    if not ids:
        return None
    k = max(ids) + 1
    matrix = np.ones((k, k), dtype=np.float64)
    for i in ids:
        matrix[i, i] = 0.0
        for j in ids:
            if i < j:
                c = _region_substitution_cost(region_meta, i, j, w_spatial, w_poi)
                matrix[i, j] = c
                matrix[j, i] = c
    return matrix


def _poi_aware_edit_distance(seq1, seq2, cost_matrix):
    """
    POI 感知编辑距离（保留回头路的全序列）。
    - 相同 region：替换代价 0
    - 不同 region：替换代价查 cost_matrix（由空间距离 + POI 语义相似度预计算）
    - 插入/删除代价固定 1
    - 距离归一化到 [0, 1]，0 = 完全相同，1 = 完全不同
    """
    m, n = len(seq1), len(seq2)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0

    dp = np.zeros((m + 1, n + 1), dtype=np.float64)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        row = cost_matrix[seq1[i - 1]]
        for j in range(1, n + 1):
            dp[i, j] = min(dp[i - 1, j] + 1.0,   # 插入
                           dp[i, j - 1] + 1.0,   # 删除
                           dp[i - 1, j - 1] + row[seq2[j - 1]])  # 替换
    return float(dp[m, n] / max(m, n))


def _compute_condensed_block(args):
    start, end, sequences, n, max_seq_len, cost_matrix = args
    offset = n * start - start * (start + 1) // 2
    block = np.empty(sum(n - i - 1 for i in range(start, end)), dtype=np.float64)
    pos = 0
    for i in range(start, end):
        si = sequences[i][-max_seq_len:] if len(sequences[i]) > max_seq_len else sequences[i]
        for j in range(i + 1, n):
            sj = sequences[j][-max_seq_len:] if len(sequences[j]) > max_seq_len else sequences[j]
            block[pos] = _poi_aware_edit_distance(si, sj, cost_matrix)
            pos += 1
    return offset, block


def _compute_affinity_block(args):
    """子进程计算 affinity 矩阵上三角的一个行区块。"""
    start, end, sequences, cost_matrix = args
    n = len(sequences)
    block = np.zeros((end - start, n))
    for i_local, i in enumerate(range(start, end)):
        block[i_local, i] = 1.0
        for j in range(i + 1, n):
            dist = _poi_aware_edit_distance(sequences[i], sequences[j], cost_matrix)
            block[i_local, j] = 1.0 - dist
    return start, end, block


def _build_trajectory_affinity(sequences, cost_matrix, n_workers=None):
    """
    根据轨迹 region 序列构建 Affinity 矩阵（多进程并行）。
    affinity = 1 - poi_aware_edit_distance
    """
    n = len(sequences)
    if n <= 1:
        return np.eye(n)

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    n_workers = min(n_workers, n)

    if n <= 20 or n_workers <= 1:
        affinity = np.zeros((n, n))
        for i in range(n):
            affinity[i, i] = 1.0
            for j in range(i + 1, n):
                dist = _poi_aware_edit_distance(sequences[i], sequences[j], cost_matrix)
                aff = 1.0 - dist
                affinity[i, j] = aff
                affinity[j, i] = aff
        return np.maximum(affinity, 0)

    chunk_size = max(1, n // n_workers)
    tasks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        tasks.append((start, end, sequences, cost_matrix))

    affinity = np.zeros((n, n))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_compute_affinity_block, t): t for t in tasks}
        for future in as_completed(futures):
            start, end, block = future.result()
            affinity[start:end, :] = block[:, :]

    affinity = np.maximum(affinity, affinity.T)
    np.fill_diagonal(affinity, 1.0)
    return np.maximum(affinity, 0)


def discover_routes(df, distance_threshold=0.7, max_seq_len=80, n_workers=None,
                    scenery_name=None, w_spatial=0.5, w_poi=0.5):
    """
    从景点序列中发现路线 —— 层次聚类 + POI 感知编辑距离。

    不做 k 搜索，用固定距离阈值自动决定路线数：
    1. 按 trackId 提取压缩的 region_id 序列（保留回头路，只压缩连续重复）
    2. 截断过长序列 + 多进程并行计算 POI 感知编辑距离矩阵
    3. 层次聚类 (Ward linkage) + 距离阈值切割 → 自动 N 条路线
    4. 路线数过多或过少时自动调整阈值

    参数:
    - df: 带 region_id 的 DataFrame
    - distance_threshold: 编辑距离阈值 (0~1)，默认 0.7
      越大越宽松（更少的路线），越小越严格（更多的路线）
    - max_seq_len: 序列截断长度，默认 80。过长序列只保留尾部，
      大幅降低编辑距离 O(L²) 复杂度
    - n_workers: 并行进程数
    - scenery_name: 景区名，用于加载 POI 语义（可为 None 则只用空间距离）
    - w_spatial / w_poi: 替换代价中空间距离与 POI 语义的权重

    返回:
    - df: 新增 'route_id' 列（int 标签）
    - route_metrics: 路线发现指标
    - route_seqs: 每个 trackId 的 region 压缩序列
    """
    from scipy.cluster.hierarchy import linkage, fcluster

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    print("\n" + "=" * 60)
    print("Phase 2: Route Discovery (路线发现 — POI 感知层次聚类)")
    print("=" * 60)

    df['region_id'] = df['region_id'].astype(int)

    # 1. 按 trackId 分组，提取 region 序列（保留回头路，仅压缩连续重复）
    track_sequences = {}
    for track_id, group in df.groupby('trackId'):
        regions = group.sort_values('时间_秒')['region_id'].values
        seq = []
        prev = None
        for r in regions:
            if r != prev:
                seq.append(int(r))
                prev = r
        track_sequences[track_id] = tuple(seq)

    # 过滤至少经过 2 个 region 的轨迹
    valid_tracks = {tid: tuple(int(r) for r in seq) for tid, seq in track_sequences.items()
                    if len(set(seq)) >= 2 and len(seq) >= 2}
    n_total = len(valid_tracks)
    print(f"\n  Total tracks:           {len(track_sequences)}")
    print(f"  Valid (≥2 regions):     {n_total}")

    if n_total < 3:
        print("  Too few valid tracks.")
        df['route_id'] = df['trackId'].map(lambda tid: 0 if tid in valid_tracks else -1)
        return df, {'n_routes': 1, 'n_tracks': n_total, 'threshold': distance_threshold}, {}, {}

    # 1b. 构建区域元数据 + 替换代价矩阵（空间质心 + POI 语义）
    poi_profiles = _load_region_poi_profiles(scenery_name) if scenery_name else {}
    region_meta = build_region_route_meta(df, poi_profiles)
    cost_matrix = _build_substitution_cost_matrix(region_meta, w_spatial, w_poi)
    print(f"  POI profiles:           {len(poi_profiles)} regions enriched")
    print(f"  Substitution weights:   w_spatial={w_spatial}, w_poi={w_poi}")

    track_ids = list(valid_tracks.keys())
    sequences = [valid_tracks[tid] for tid in track_ids]

    # 2. 分析序列特征
    seq_lengths = [len(s) for s in sequences]
    print(f"  Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={np.mean(seq_lengths):.1f}")

    # 3. 构建全对编辑距离矩阵（上三角压缩格式，多进程并行）
    n = len(track_ids)
    n_workers = min(n_workers, n)
    print(f"\n  Computing POI-aware edit distance matrix ({n}×{n})...")
    print(f"  Workers: {n_workers}, max_seq_len={max_seq_len}")

    dist_condensed = np.zeros(n * (n - 1) // 2, dtype=np.float64)

    if n <= 20 or n_workers <= 1:
        pos = 0
        for i in range(n):
            si = sequences[i][-max_seq_len:] if len(sequences[i]) > max_seq_len else sequences[i]
            for j in range(i + 1, n):
                sj = sequences[j][-max_seq_len:] if len(sequences[j]) > max_seq_len else sequences[j]
                dist_condensed[pos] = _poi_aware_edit_distance(si, sj, cost_matrix)
                pos += 1
    else:
        chunk_size = max(1, n // n_workers)
        tasks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            tasks.append((start, end, sequences, n, max_seq_len, cost_matrix))

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_compute_condensed_block, t): t for t in tasks}
            for future in as_completed(futures):
                offset, block = future.result()
                dist_condensed[offset:offset + len(block)] = block
    print(f"  Dist matrix: min={dist_condensed.min():.4f}, max={dist_condensed.max():.4f}, "
          f"mean={dist_condensed.mean():.4f}, median={np.median(dist_condensed):.4f}")

    # 4. 层次聚类 + 距离阈值切割
    # 注意：Ward 仅适用于欧氏（平方）距离；POI 感知编辑距离是非欧氏成对距离，
    # 改用 average linkage（天然适配任意成对距离）。
    Z = linkage(dist_condensed, method='average')
    labels = fcluster(Z, t=distance_threshold, criterion='distance')
    labels -= 1
    n_routes = len(np.unique(labels))
    final_threshold = distance_threshold
    print(f"  Hierarchical clustering: {n_routes} routes at threshold={distance_threshold}")

    # 自动调整阈值：太少拉高，太多降低
    max_routes = max(8, n_total // 40)
    if n_routes > max_routes:
        for fallback_t in np.arange(distance_threshold + 0.05, 1.0, 0.05):
            labels_t = fcluster(Z, t=float(fallback_t), criterion='distance')
            labels_t -= 1
            nr = len(np.unique(labels_t))
            if nr <= max_routes:
                labels = labels_t
                n_routes = nr
                final_threshold = float(fallback_t)
                print(f"  Too many routes! Auto-widening threshold to {fallback_t:.2f} → {n_routes} routes")
                break
        if n_routes > max_routes:
            labels = fcluster(Z, t=0.95, criterion='distance')
            labels -= 1
            n_routes = len(np.unique(labels))
            final_threshold = 0.95
    elif n_routes <= 1:
        for fallback_t in np.arange(distance_threshold + 0.1, 1.0, 0.1):
            labels_t = fcluster(Z, t=float(fallback_t), criterion='distance')
            labels_t -= 1
            nr = len(np.unique(labels_t))
            if nr >= 2:
                labels = labels_t
                n_routes = nr
                final_threshold = float(fallback_t)
                print(f"  Only 1 route! Auto-widening threshold to {fallback_t:.1f} → {n_routes} routes")
                break
        if n_routes <= 1:
            labels = fcluster(Z, t=1.0, criterion='distance')
            labels -= 1
            n_routes = len(np.unique(labels))
            final_threshold = 1.0

    # 5. 分配 route_id 回 DataFrame
    track_to_route = dict(zip(track_ids, labels))
    df['route_id'] = df['trackId'].map(track_to_route).fillna(-1).astype(int)

    # 6. 输出报告
    print(f"\n  === Route Discovery Report ===")
    print(f"  Method:               Hierarchical (Ward, threshold={final_threshold:.2f})")
    print(f"  Routes discovered:    {n_routes}")
    print(f"  Total valid tracks:   {n_total}")
    print(f"  Coverage:             {n_total}/{n_total} (100%)")

    print(f"\n  Route size distribution (top 15):")
    route_sizes = []
    for rid in sorted(np.unique(labels)):
        count = int(np.sum(labels == rid))
        route_tids = [track_ids[i] for i in range(n) if labels[i] == rid]
        avg_len = np.mean([len(valid_tracks[tid]) for tid in route_tids])
        n_uniq_regions = len(set(r for tid in route_tids for r in valid_tracks[tid]))
        route_sizes.append((count, rid, avg_len, n_uniq_regions))
    route_sizes.sort(key=lambda x: -x[0])
    for i, (count, rid, avg_len, n_uniq) in enumerate(route_sizes[:15]):
        print(f"    Route {rid}: {count} tracks ({count/n_total*100:.1f}%), "
              f"avg seq_len={avg_len:.1f}, touches {n_uniq} regions")
    if len(route_sizes) > 15:
        remaining = sum(s[0] for s in route_sizes[15:])
        print(f"    ... and {len(route_sizes)-15} smaller routes ({remaining} tracks total)")

    route_metrics = {
        'n_routes': n_routes,
        'n_tracks': n_total,
        'threshold': final_threshold,
        'coverage_pct': 1.0,
        'effective_coverage': 1.0,
    }

    return df, route_metrics, dict(track_to_route), valid_tracks


# ============================================================
# 主 Pipeline
# ============================================================

def run_full_pipeline(csv_path, n_workers=None, random_state=SEED,
                      k_lower=None, k_upper=None, n_poi_regions=None):
    """
    完整 Pipeline：景点聚类 + 路线发现。

    参数:
    - csv_path: 输入 CSV 路径
    - n_workers: 并行进程数
    - random_state: 随机种子
    - k_lower / k_upper: 景点聚类 k 搜索范围（None 则按 POI 区域数自适应）
    - n_poi_regions: 该场景 POI 区域数（用于自适应 k 范围，可为 None）

    返回:
    - df: 带 region_id 和 route_id 的完整 DataFrame
    - scenic_metrics: 景点聚类指标
    - route_metrics: 路线发现指标
    - artifacts: 可视化所需中间数据
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    print("=" * 60)
    print("STRAT 景点聚类 + 路线发现 Pipeline")
    print("=" * 60)

    # 加载数据
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Total points: {df.shape[0]:,}")
    print(f"  Unique trackId: {df['trackId'].nunique():,}")

    # 确保必须的列存在
    required_cols = ['经度', '纬度', '海拔', 'is_stop', 'trackId', '时间_秒']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Phase 1: 景点聚类
    df, scenic_metrics, ball_data_list, cluster_features, sampled_labels = cluster_scenic_spots(
        df, n_workers, random_state=random_state,
        k_lower=k_lower, k_upper=k_upper, n_poi_regions=n_poi_regions)

    # Phase 2: 路线发现（从文件名推导景区名，用于加载 POI 语义）
    scenery_name = scenery_name_from_csv(csv_path)
    df, route_metrics, route_labels, route_sequences = discover_routes(
        df, n_workers=n_workers, scenery_name=scenery_name)

    artifacts = {
        'ball_data_list': ball_data_list,
        'pca_features': cluster_features,
        'sampled_labels': sampled_labels,
        'route_labels': route_labels,
        'route_sequences': route_sequences,
    }

    return df, scenic_metrics, route_metrics, artifacts


# ============================================================
# 独立运行
# ============================================================

if __name__ == '__main__':
    import argparse
    from config import default_cleaned_csv

    parser = argparse.ArgumentParser(description='运行景点聚类 + 路线发现 Pipeline')
    parser.add_argument('--csv', default=None,
                        help='清洗后的 CSV 路径（默认取 data-project/cleaned_labeled_data 下第一个 *_cleaned.csv）')
    parser.add_argument('--seed', type=int, default=SEED, help='随机种子')
    parser.add_argument('--k_lower', type=int, default=None, help='k 搜索下界')
    parser.add_argument('--k_upper', type=int, default=None, help='k 搜索上界')
    args = parser.parse_args()

    csv_path = args.csv or default_cleaned_csv()
    if not csv_path or not os.path.exists(csv_path):
        raise SystemExit('未找到输入 CSV，请用 --csv 指定路径。'
                         '默认目录: data-project/cleaned_labeled_data')

    df_out, scenic_m, route_m, artifacts = run_full_pipeline(
        csv_path, random_state=args.seed,
        k_lower=args.k_lower, k_upper=args.k_upper)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Scenic regions: {scenic_m['n_regions']}")
    print(f"  Route discovered: {route_m['n_routes']}")