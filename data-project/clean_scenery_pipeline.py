#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
景区数据清洗流水线（新版）

五步流程：
  1. 读取CSV + 时间转换
  2. 多边形边界过滤
  3. 飞点过滤
  4. 过短轨迹删除
  5. 区域标签生成（停留检测 + DBSCAN + 移动点继承 + 标签规整）

输出：cleaned_labeled_data/{景区名}_cleaned.csv
可选：GeoJSON 导出供 QGIS 人工审查
可视化：自动生成数据质量报告图表

使用方法：
    python clean_scenery_pipeline.py --name 青城山 --csv data/青城山.csv
    python clean_scenery_pipeline.py --name 黄龙溪 --csv data/黄龙溪.csv --boundary "103.96,30.31;..."
    python clean_scenery_pipeline.py --name 青城山 --csv data/青城山.csv --export_geojson
    python clean_scenery_pipeline.py --name 青城山 --csv data/青城山.csv --visualize
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，避免GUI依赖
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PolyCollection
    from matplotlib.font_manager import FontManager
    
    # 自动检测可用的中文字体（按优先级）
    font_priority = [
        'WenQuanYi Micro Hei',  # Linux 通用 (fonts-wqy-microhei)
        'Noto Sans CJK SC',     # Noto CJK 简体中文
        'Noto Sans CJK JP',     # Noto CJK 日文 (在某些发行版上注册为此名)
        'Noto Sans CJK',        # Noto CJK 通用名
        'AR PL UKai CN',        # Linux 备用中文字体
        'AR PL UMing CN',       # Linux 备用中文字体
        'WenQuanYi Zen Hei',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
    ]
    
    available_fonts = set(f.name for f in FontManager().ttflist)
    
    selected_font = None
    for font_name in font_priority:
        if font_name in available_fonts:
            selected_font = font_name
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font]
        print(f"  [可视化] 使用字体: {selected_font}")
    else:
        print("  [可视化] 未找到中文字体，将使用默认字体（可能显示乱码）")
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    print("Warning: matplotlib not available, visualization disabled")

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    pass

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(SCRIPT_DIR))
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "cleaned_labeled_data"
QGIS_DIR = SCRIPT_DIR / "qgis_data"

from scenery_config import SCENERY_CONFIGS


def parse_time_to_seconds(time_str):
    """
    将 此刻时间 字段转为连续数值秒（当天内的秒数）。

    支持格式：
      - "2014-03-15 05:26.16" (HH:MM.SS → 时:分.秒)
      - "00:07.5"              (分:秒.十分秒)
    """
    if pd.isna(time_str):
        return np.nan
    if not isinstance(time_str, str):
        try:
            return float(time_str)
        except (ValueError, TypeError):
            return np.nan
    try:
        t_str = time_str.strip()
        if " " in t_str:
            t_str = t_str.split()[-1]

        if ":" in t_str and "." in t_str:
            hh_mm, ss = t_str.split(".", 1)
            hh_str, mm_str = hh_mm.split(":", 1)
            hours = float(hh_str)
            minutes = float(mm_str)
            seconds = float(ss)
            return hours * 3600 + minutes * 60 + seconds

        if ":" in t_str:
            parts = t_str.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])

        return float(t_str)
    except (ValueError, TypeError):
        return np.nan


def ray_cast(x, y, polygon):
    """
    射线投射法：判断点 (x, y) 是否在多边形内部。
    从 process_scenery_pipeline.py 复用。
    """
    n = len(polygon)
    inside = False
    x1, y1 = polygon[0]

    for i in range(n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    if x1 == x2 or x <= xinters:
                        inside = not inside
        x1, y1 = x2, y2

    return inside


def is_in_scenery(lon, lat, boundary):
    """
    判断点是否在景区多边形内，支持单个点和数组。
    从 process_scenery_pipeline.py 复用。
    """
    if isinstance(lon, (int, float)) and isinstance(lat, (int, float)):
        return ray_cast(lon, lat, boundary)

    result = np.zeros(len(lon), dtype=bool)
    for i in range(len(lon)):
        result[i] = ray_cast(lon[i], lat[i], boundary)
    return result


def haversine_meters(lon1, lat1, lon2, lat2):
    """
    Haversine 距离计算（米）。
    从 process_scenery_pipeline.py 复用。
    """
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_csv(csv_path):
    """
    加载CSV，尝试多种编码，自动识别经纬度列。
    从 process_scenery_pipeline.py 复用。
    支持跳过格式错误的行（字段数不匹配）。
    """
    print(f"[Step 1] 加载CSV: {csv_path}")

    encodings = ["utf-8", "gb18030", "gbk", "gb2312", "utf-8-sig", "latin1"]
    df = None

    for enc in encodings:
        try:
            df = pd.read_csv(
                csv_path,
                encoding=enc,
                on_bad_lines='skip',
                low_memory=False,
            )
            df.columns = df.columns.str.strip()
            if "经度" in df.columns and "纬度" in df.columns:
                break
            df = None
        except Exception as e:
            print(f"    尝试编码 {enc} 失败: {e}")
            continue

    if df is None:
        raise RuntimeError(f"无法读取CSV文件: {csv_path}")

    numeric_cols = ["经度", "纬度", "海拔", "速度(km/h)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["经度", "纬度"])

    required_cols = ["经度", "纬度", "海拔", "此刻时间", "速度(km/h)", "trackId"]
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f"CSV 缺少必要列: {col}")

    print(f"    总点数: {len(df):,}")
    print(f"    轨迹数: {df['trackId'].nunique():,}")

    return df


def add_time_seconds(df):
    """将 此刻时间 转为 时间_秒 列，删除无效行。"""
    print(f"    解析时间字段...")
    df = df.copy()
    df["时间_秒"] = df["此刻时间"].apply(parse_time_to_seconds)
    before = len(df)
    df = df.dropna(subset=["时间_秒"])
    removed = before - len(df)
    if removed > 0:
        print(f"    删除时间无效行: {removed:,}")
    return df


# ============================================================
# 清洗阶段
# ============================================================


def filter_by_boundary(df, boundary):
    """
    Step 2: 多边形边界过滤，只保留景区内的点。
    """
    print(f"[Step 2] 多边形边界过滤")

    lon = df["经度"].values
    lat = df["纬度"].values

    inside = is_in_scenery(lon, lat, boundary)
    df = df.copy()
    df["in_scenery"] = inside

    before = len(df)
    if before == 0:
        print(f"    过滤前: 0 点，跳过")
        return df

    df = df[df["in_scenery"]].copy()
    df = df.drop("in_scenery", axis=1)
    removed = before - len(df)

    print(f"    过滤前: {before:,} 点")
    print(f"    过滤后: {len(df):,} 点")
    print(f"    删除: {removed:,} 点 ({100*removed/before:.1f}%)")
    print(f"    剩余轨迹数: {df['trackId'].nunique():,}")

    return df


def filter_flying_points(df, max_jump_meters=100, max_speed_kmh=None):
    """
    Step 3: 飞点过滤
    - 按 trackId 分组，计算相邻点
    - 只要轨迹中存在任何超过阈值的跳点，整条轨迹删除
    """
    if max_jump_meters is not None:
        print(f"[Step 3] 飞点过滤 (位移阈值={max_jump_meters}m，含跳点轨迹删除)")
    elif max_speed_kmh is not None:
        print(f"[Step 3] 飞点过滤 (速度阈值={max_speed_kmh}km/h，含跳点轨迹删除)")

    df = df.copy()
    df = df.sort_values(["trackId", "时间_秒"]).reset_index(drop=True)

    tracks_with_fly = set()

    for track_id, group in df.groupby("trackId", sort=False):
        group = group.sort_values("时间_秒")
        if len(group) < 2:
            continue

        lons = group["经度"].values
        lats = group["纬度"].values
        times = group["时间_秒"].values

        for i in range(len(group) - 1):
            time_diff = times[i + 1] - times[i]
            if time_diff <= 0:
                continue

            dist_m = haversine_meters(lons[i], lats[i], lons[i + 1], lats[i + 1])

            is_fly = False
            if max_jump_meters is not None:
                if dist_m > max_jump_meters:
                    is_fly = True
            elif max_speed_kmh is not None:
                speed = dist_m / time_diff * 3.6
                if speed > max_speed_kmh:
                    is_fly = True

            if is_fly:
                tracks_with_fly.add(track_id)
                break

    before_tracks = df["trackId"].nunique()
    df_filtered = df[~df["trackId"].isin(tracks_with_fly)].copy()
    removed_tracks = before_tracks - df_filtered["trackId"].nunique()
    removed_points = len(df) - len(df_filtered)
    after_tracks = df_filtered["trackId"].nunique()

    print(f"    检测到含飞点轨迹: {len(tracks_with_fly):,} 条")
    print(f"    删除轨迹: {removed_tracks:,} 条")
    print(f"    删除点: {removed_points:,}")
    print(f"    剩余: {len(df_filtered):,} 点, {after_tracks:,} 条轨迹")

    return df_filtered


def filter_short_tracks(df, min_length=300):
    """
    Step 4: 删除过短轨迹（点数 < min_length），重映射 trackId 为 0~N-1。
    """
    print(f"[Step 4] 删除过短轨迹 (最小长度={min_length})")

    track_lens = df.groupby("trackId").size()
    before_tracks = len(track_lens)
    valid_tracks = track_lens[track_lens >= min_length].index

    df_filtered = df[df["trackId"].isin(valid_tracks)].copy()
    removed_tracks = before_tracks - len(valid_tracks)
    removed_points = len(df) - len(df_filtered)

    unique_tracks = sorted(df_filtered["trackId"].unique())
    track_mapping = {tid: i for i, tid in enumerate(unique_tracks)}
    df_filtered["trackId"] = df_filtered["trackId"].map(track_mapping)

    print(f"    过滤前: {before_tracks:,} 条轨迹, {len(df):,} 点")
    print(f"    删除短轨迹: {removed_tracks:,} 条")
    print(f"    删除点: {removed_points:,}")
    print(f"    过滤后: {df_filtered['trackId'].nunique():,} 条轨迹, {len(df_filtered):,} 点")

    return df_filtered


# ============================================================
# 区域标签生成
# ============================================================


def detect_stays(df, speed_thresh=1.0, duration_thresh=30):
    """
    Step 5a: 停留检测。
    在同一 trackId 内，按 时间_秒 排序，寻找连续速度 <= speed_thresh(km/h)
    且持续时间 >= duration_thresh(秒) 的片段，标记 is_stop=1。
    """
    print(f"[Step 5a] 停留检测 (速度<={speed_thresh}km/h, 持续>={duration_thresh}s)")

    df = df.copy()
    df["is_stop"] = 0
    df = df.sort_values(["trackId", "时间_秒"]).reset_index(drop=True)

    stop_segments = 0
    total_stop_points = 0

    for track_id, group in df.groupby("trackId", sort=False):
        speeds = group["速度(km/h)"].values
        times = group["时间_秒"].values
        indices = group.index.values

        n = len(group)
        i = 0
        while i < n:
            if speeds[i] <= speed_thresh:
                j = i + 1
                while j < n and speeds[j] <= speed_thresh:
                    j += 1

                duration = times[j - 1] - times[i]

                if duration >= duration_thresh:
                    df.loc[indices[i:j], "is_stop"] = 1
                    stop_segments += 1
                    total_stop_points += j - i

                i = j
            else:
                i += 1

    if len(df) == 0:
        print(f"    无数据，跳过")
        return df

    print(f"    停留片段数: {stop_segments:,}")
    print(f"    停留点数:    {total_stop_points:,} ({100*total_stop_points/len(df):.1f}%)")
    print(f"    移动点数:    {len(df)-total_stop_points:,}")

    return df


def cluster_stay_points(df, eps=0.001, min_samples=30, min_region_size=100):
    """
    Step 5b: DBSCAN 对停留点进行空间聚类 + 小簇合并。
    取出 is_stop==1 的点的经纬度，用 DBSCAN 聚类。
    为每个停留点赋予 cluster_id（-1 为噪声）。
    之后将点数 < min_region_size 的小簇合并到最近的足够大的簇。
    """
    print(f"[Step 5b] DBSCAN 聚类停留点 (eps={eps}, min_samples={min_samples})")

    from sklearn.cluster import DBSCAN

    df = df.copy()
    df["cluster_id"] = -1

    stay_mask = df["is_stop"] == 1
    n_stay = stay_mask.sum()

    if n_stay < min_samples:
        print(f"    停留点不足 ({n_stay} < {min_samples})，跳过聚类")
        return df

    stay_coords = df.loc[stay_mask, ["经度", "纬度"]].values

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(stay_coords)

    df.loc[stay_mask, "cluster_id"] = labels

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    print(f"    DBSCAN 有效簇数: {n_clusters}")
    print(f"    噪声点数: {n_noise} ({100*n_noise/n_stay:.1f}%)")

    valid_clusters = sorted(set(labels) - {-1})

    if not valid_clusters:
        return df

    cluster_counts = {c: int(np.sum(labels == c)) for c in valid_clusters}
    cluster_centroids = {}
    for c in valid_clusters:
        cmask = labels == c
        cluster_centroids[c] = (
            stay_coords[cmask, 0].mean(),
            stay_coords[cmask, 1].mean(),
        )

    small_clusters = [c for c in valid_clusters if cluster_counts[c] < min_region_size]
    large_clusters = [c for c in valid_clusters if cluster_counts[c] >= min_region_size]

    if small_clusters and large_clusters:
        n_merged = 0
        for sc in small_clusters:
            best_lc = min(large_clusters, key=lambda lc: (
                (cluster_centroids[sc][0] - cluster_centroids[lc][0])**2 +
                (cluster_centroids[sc][1] - cluster_centroids[lc][1])**2
            ))
            stay_indices = df[stay_mask].index
            sc_indices = stay_indices[labels == sc]
            df.loc[sc_indices, "cluster_id"] = best_lc
            n_merged += 1
        if n_merged > 0:
            print(f"    小簇合并: {n_merged} 个 (<{min_region_size}点) → 最近大簇")

    final_clusters = sorted(set(df.loc[stay_mask, "cluster_id"].unique()) - {-1})
    print(f"    最终区域簇数: {len(final_clusters)}")
    for c in final_clusters:
        count = int((df["cluster_id"] == c).sum())
        print(f"      区域簇{int(c)}: {count:,} 点")

    return df


def assign_regions_for_moving_points(df):
    """
    Step 5c: 移动点继承最近区域标签（KD-Tree 加速版）。
    计算每个有效簇（cluster_id != -1）的质心，
    对 is_stop==0 的每个移动点，分配给最近质心的 cluster_id。
    使用 KD-Tree 大幅加速最近邻搜索。
    """
    print(f"[Step 5c] 移动点分配最近区域 (KD-Tree 加速)")

    df = df.copy()

    valid_clusters = sorted(set(df["cluster_id"].unique()) - {-1})

    if not valid_clusters:
        print(f"    无有效簇，无法分配")
        return df

    centroids = {}
    for cid in valid_clusters:
        mask = df["cluster_id"] == cid
        centroids[cid] = (
            df.loc[mask, "经度"].mean(),
            df.loc[mask, "纬度"].mean(),
        )

    moving_mask = df["is_stop"] == 0
    n_moving = moving_mask.sum()

    if n_moving == 0:
        print(f"    无移动点需要分配")
        return df

    # 准备质心数据用于 KD-Tree
    centroid_coords = np.array([[centroids[c][0], centroids[c][1]] for c in valid_clusters])
    
    # 准备移动点数据
    moving_indices = df[moving_mask].index
    moving_coords = df.loc[moving_indices, ["经度", "纬度"]].values

    try:
        # 尝试使用 scipy 的 KD-Tree（最快）
        from scipy.spatial import KDTree
        
        kdtree = KDTree(centroid_coords)
        _, nearest_indices = kdtree.query(moving_coords, k=1)
        
        # 将最近邻索引映射回 cluster_id
        nearest_cluster_ids = [valid_clusters[idx] for idx in nearest_indices]
        df.loc[moving_indices, "cluster_id"] = nearest_cluster_ids
        
    except ImportError:
        # 如果没有 scipy，使用 numpy 向量化（比原来快）
        print(f"    [提示] 未安装 scipy，使用 numpy 向量化计算")
        
        centroid_lons = np.array([centroids[c][0] for c in valid_clusters])
        centroid_lats = np.array([centroids[c][1] for c in valid_clusters])
        moving_lons = moving_coords[:, 0]
        moving_lats = moving_coords[:, 1]
        
        # 向量化计算所有距离
        dlon = moving_lons[:, np.newaxis] - centroid_lons[np.newaxis, :]
        dlat = moving_lats[:, np.newaxis] - centroid_lats[np.newaxis, :]
        dists = np.sqrt(dlon**2 + dlat**2)
        
        # 找到每个移动点的最近质心
        nearest_indices = np.argmin(dists, axis=1)
        nearest_cluster_ids = [valid_clusters[idx] for idx in nearest_indices]
        df.loc[moving_indices, "cluster_id"] = nearest_cluster_ids

    print(f"    移动点已分配: {n_moving:,}")

    return df


def normalize_region_ids(df):
    """
    Step 5d: 标签规整。
    移除 cluster_id == -1 的点，
    将 cluster_id 映射为 0~N-1 连续整数的 region_id，
    添加 reviewed 列（初始全 False）。
    """
    print(f"[Step 5d] 标签规整")

    before = len(df)

    noise_mask = df["cluster_id"] == -1
    n_noise = noise_mask.sum()

    df = df[~noise_mask].copy()

    unique_ids = sorted(df["cluster_id"].unique())
    mapping = {old: new for new, old in enumerate(unique_ids)}
    df["region_id"] = df["cluster_id"].map(mapping)

    df["reviewed"] = False

    removed = before - len(df)

    print(f"    删除 cluster_id=-1 点: {n_noise:,}")
    print(f"    总删除: {removed:,}")
    print(f"    最终区域数: {len(unique_ids)}")
    print(f"    最终点数: {len(df):,}")
    print(f"    最终轨迹数: {df['trackId'].nunique():,}")

    region_dist = df.groupby("region_id").size()
    for rid in sorted(region_dist.index):
        count = region_dist[rid]
        print(f"      区域{int(rid)}: {count:,} 点 ({100*count/len(df):.1f}%)")

    return df


def cluster_routes(df, route_eps=1.5, route_min_samples=2, route_sample_points=10):
    """
    Step 6: 路线聚类。
    每条轨迹提取固定数量采样点的经纬度作为特征向量，
    用 DBSCAN 对轨迹级特征聚类，生成 route_id。
    路线本质是点组成的线，通过等距采样实现密度聚类。
    """
    print(f"[Step 6] 路线聚类 (track采样{route_sample_points}点, eps={route_eps}, min_samples={route_min_samples})")
    from sklearn.cluster import DBSCAN

    df = df.copy()
    df["route_id"] = -1
    df["route_name"] = ""

    track_ids = sorted(df["trackId"].unique())
    n_tracks = len(track_ids)

    features = np.zeros((n_tracks, route_sample_points * 2))
    valid_tracks = []

    for i, tid in enumerate(track_ids):
        track = df[df["trackId"] == tid].sort_values("时间_秒")
        n = len(track)
        if n < 2:
            continue
        indices = np.linspace(0, n - 1, route_sample_points, dtype=int)
        lons = track.iloc[indices]["经度"].values.astype(float)
        lats = track.iloc[indices]["纬度"].values.astype(float)
        features[i, 0::2] = lons
        features[i, 1::2] = lats
        valid_tracks.append((i, tid))

    if len(valid_tracks) < route_min_samples:
        print(f"    有效轨迹不足 ({len(valid_tracks)} < {route_min_samples})，全部归为路线0")
        df["route_id"] = 0
        df["route_name"] = "路线0"
        return df

    valid_indices = [vt[0] for vt in valid_tracks]
    valid_features = features[valid_indices]

    eps_scaled = route_eps * 0.001
    dbscan = DBSCAN(eps=eps_scaled, min_samples=route_min_samples)
    labels = dbscan.fit_predict(valid_features)

    n_routes = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    print(f"    路线数: {n_routes}")
    print(f"    噪声轨迹数: {n_noise} ({100*n_noise/len(valid_tracks):.1f}%)")

    centroids = {}
    for cid in sorted(set(labels) - {-1}):
        cmask = labels == cid
        centroids[cid] = valid_features[cmask].mean(axis=0)

    for k, (i, tid) in enumerate(valid_tracks):
        route_id = int(labels[k])
        if route_id == -1:
            if centroids:
                best_cid = min(centroids.keys(), key=lambda c: np.sum((valid_features[k] - centroids[c])**2))
                route_id = best_cid
            else:
                route_id = 0
        track_mask = df["trackId"] == tid
        track = df[track_mask].sort_values("时间_秒")
        region_compress = _compress_sequence(track["region_id"].values)

        if len(region_compress) <= 2:
            route_type = "短路径"
        elif region_compress[0] == region_compress[-1]:
            route_type = "环线"
        else:
            start_lat = track.iloc[0]["纬度"]
            end_lat = track.iloc[-1]["纬度"]
            if end_lat > start_lat:
                route_type = "向北路径"
            else:
                route_type = "向南路径"

        main_regions = region_compress[:min(len(region_compress), 10)]
        region_str = ",".join(str(int(r)) for r in main_regions)
        name = f"路线{route_id}_{route_type}"
        if len(region_str) > 0 and len(region_str) < 80:
            name += f"(主{region_str})"

        df.loc[track_mask, "route_id"] = route_id
        df.loc[track_mask, "route_name"] = name

    route_dist = df.groupby("route_id")["trackId"].nunique()
    for rid in sorted(route_dist.index):
        count = route_dist[rid]
        sample_name = df[df["route_id"] == rid]["route_name"].iloc[0][:60]
        print(f"      路线{int(rid)}: {count} 条轨迹, 示例: {sample_name}")

    return df


def _compress_sequence(seq):
    if len(seq) == 0:
        return []
    compressed = [seq[0]]
    for v in seq[1:]:
        if v != compressed[-1]:
            compressed.append(v)
    return compressed


# ============================================================
# 输出
# ============================================================

OUTPUT_COLS = [
    "经度",
    "纬度",
    "海拔",
    "此刻时间",
    "时间_秒",
    "速度(km/h)",
    "trackId",
    "is_stop",
    "region_id",
    "reviewed",
    "route_id",
    "route_name",
]


def convert_to_geojson(df, output_path, boundary, scenery_name):
    """
    将带 region_id 的清洗数据导出为 GeoJSON，供 QGIS 审查。
    每条轨迹一个 LineString feature，按 region_id 设色。
    从 process_scenery_pipeline.py 复用并修改。
    """
    print(f"[Export] 生成 GeoJSON: {output_path}")

    features = []
    for track_id in sorted(df["trackId"].unique()):
        track_df = df[df["trackId"] == track_id].sort_values("时间_秒")
        coords = track_df[["经度", "纬度"]].values.tolist()

        if len(coords) < 2:
            continue

        region_ids = track_df["region_id"].unique()
        main_region = int(region_ids[0]) if len(region_ids) > 0 else -1

        route_id = int(track_df["route_id"].iloc[0]) if "route_id" in track_df.columns else -1
        route_name = str(track_df["route_name"].iloc[0]) if "route_name" in track_df.columns else ""

        props = {
            "trackId": int(track_id),
            "n_points": len(coords),
            "region_id": main_region,
            "route_id": route_id,
            "route_name": route_name,
            "mean_alt": float(track_df["海拔"].mean()),
            "mean_speed": float(track_df["速度(km/h)"].mean()),
            "is_stop_ratio": float(track_df["is_stop"].mean()),
        }

        features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": props})

    boundary_coords = [[p[0], p[1]] for p in boundary] + [[boundary[0][0], boundary[0][1]]]
    features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [boundary_coords]}, "properties": {"type": "boundary", "name": scenery_name}})

    geojson = {"type": "FeatureCollection", "features": features}

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"    Features: {len(features)}")
    print(f"    已保存: {output_path}")


def generate_report(scenery_name, boundary, stages, df_final, csv_output, geojson_output):
    """生成清洗报告。"""
    n_regions = df_final["region_id"].nunique()
    n_tracks = df_final["trackId"].nunique()
    n_points = len(df_final)

    lines = []
    lines.append("=" * 60)
    lines.append(f"{scenery_name} 数据清洗报告")
    lines.append("=" * 60)
    lines.append("")

    lines.append("【景区边界】")
    lon_min = min(p[0] for p in boundary)
    lon_max = max(p[0] for p in boundary)
    lat_min = min(p[1] for p in boundary)
    lat_max = max(p[1] for p in boundary)
    lines.append(f"  顶点数: {len(boundary)}")
    lines.append(f"  经度范围: [{lon_min:.6f}, {lon_max:.6f}]")
    lines.append(f"  纬度范围: [{lat_min:.6f}, {lat_max:.6f}]")
    lines.append("")

    lines.append("【各阶段过滤统计】")
    for name, before, after in stages:
        kept_pct = 100 * after / before if before > 0 else 0
        lines.append(f"  {name}: {before:,} → {after:,} (保留 {kept_pct:.1f}%)")
    lines.append("")

    lines.append("【最终数据】")
    lines.append(f"  总点数: {n_points:,}")
    lines.append(f"  轨迹数: {n_tracks:,}")
    lines.append(f"  区域数: {n_regions}")
    lines.append(f"  路线数: {df_final['route_id'].nunique()}")
    lines.append("")

    lines.append("【路线分布】")
    route_dist = df_final.groupby("route_id")["trackId"].nunique()
    for rid in sorted(route_dist.index):
        count = route_dist[rid]
        sample_name = df_final[df_final["route_id"] == rid]["route_name"].iloc[0][:60]
        lines.append(f"  路线{int(rid)}: {count:,} 条轨迹, {sample_name}")
    lines.append("")

    lines.append("【区域分布】")
    for rid in sorted(df_final["region_id"].unique()):
        count = (df_final["region_id"] == rid).sum()
        stop_ratio = df_final[df_final["region_id"] == rid]["is_stop"].mean()
        lines.append(f"  区域{int(rid)}: {count:,} 点 ({100*count/n_points:.1f}%), " f"停留比={stop_ratio:.2f}")
    lines.append("")

    lines.append("【轨迹长度分布】")
    track_lens = df_final.groupby("trackId").size()
    lines.append(f"  最短: {track_lens.min()} 点")
    lines.append(f"  最长: {track_lens.max():,} 点")
    lines.append(f"  平均: {track_lens.mean():.1f} 点")
    lines.append(f"  中位数: {track_lens.median():.1f} 点")
    lines.append("")

    lines.append("【输出文件】")
    lines.append(f"  CSV:     {csv_output}")
    if geojson_output:
        lines.append(f"  GeoJSON: {geojson_output}")
    lines.append("")

    lines.append("【QGIS 审查指引】")
    lines.append(f"  1. 用 QGIS 打开 GeoJSON 文件")
    lines.append(f"  2. 按 region_id 字段分类设色")
    lines.append(f"  3. 检查各区域地理上是否合理")
    lines.append(f"  4. 修正错分点、合并/拆分区域")
    lines.append(f"  5. 导出修正后的 CSV，将 reviewed 标记为 True")
    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    report_path = SCRIPT_DIR / f"{scenery_name}_清洗报告.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存: {report_path}")


# ============================================================
# 主流水线
# ============================================================


def process_scenery(
    scenery_name,
    csv_path,
    custom_boundary=None,
    max_jump_meters=100,
    min_track_length=300,
    stay_speed_thresh=1.0,
    stay_duration_thresh=30,
    dbscan_eps=0.001,
    dbscan_min_samples=30,
    min_region_size=100,
    route_eps=1.5,
    route_min_samples=2,
    export_geojson=False,
    visualize=False,
):
    """
    执行完整的清洗流水线。
    """
    print("=" * 60)
    print(f"景区数据清洗: {scenery_name}")
    print("=" * 60)

    if custom_boundary:
        boundary = custom_boundary
        print(f"  使用自定义边界: {len(boundary)} 个顶点")
    elif scenery_name in SCENERY_CONFIGS:
        boundary = SCENERY_CONFIGS[scenery_name]["boundary"]
        print(f"  使用预配置边界: {SCENERY_CONFIGS[scenery_name].get('description', scenery_name)}")
    else:
        raise ValueError(f"未知景区 '{scenery_name}'，请在 scenery_config.py 中添加配置，" f"或使用 --boundary 指定自定义边界")

    stages = []

    df = load_csv(csv_path)
    n0 = len(df)
    stages.append(("原始数据", n0, n0))

    df = add_time_seconds(df)

    df = filter_by_boundary(df, boundary)
    stages.append(("边界过滤", n0, len(df)))

    n_before_fly = len(df)
    df = filter_flying_points(df, max_jump_meters)
    stages.append(("飞点过滤", n_before_fly, len(df)))

    n_before_short = len(df)
    df = filter_short_tracks(df, min_track_length)
    stages.append(("短轨迹删除", n_before_short, len(df)))

    if len(df) == 0:
        print(f"\n警告: 过滤后无数据，跳过后续处理")
        csv_output = OUTPUT_DIR / f"{scenery_name}_cleaned.csv"
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df_empty = pd.DataFrame(columns=["经度", "纬度", "海拔", "此刻时间", "时间_秒", "速度(km/h)", "trackId", "is_stop", "region_id", "reviewed", "route_id", "route_name"])
        df_empty.to_csv(csv_output, index=False, encoding="utf-8")
        print(f"已保存空文件: {csv_output}")
        return df_empty

    print(f"\n{'='*60}")
    print(f"区域标签生成")
    print(f"{'='*60}")

    df = detect_stays(df, stay_speed_thresh, stay_duration_thresh)
    df = cluster_stay_points(df, dbscan_eps, dbscan_min_samples, min_region_size)
    df = assign_regions_for_moving_points(df)
    df = normalize_region_ids(df)

    print(f"\n{'='*60}")
    print(f"路线聚类")
    print(f"{'='*60}")
    df = cluster_routes(df, route_eps, route_min_samples)

    n_final = len(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_output = OUTPUT_DIR / f"{scenery_name}_cleaned.csv"
    df_out = df[OUTPUT_COLS].copy()
    df_out.to_csv(csv_output, index=False, encoding="utf-8")
    print(f"\n[保存] CSV: {csv_output}")

    geojson_output = None
    if export_geojson:
        os.makedirs(QGIS_DIR, exist_ok=True)
        geojson_output = QGIS_DIR / f"{scenery_name}_labeled.geojson"
        convert_to_geojson(df, str(geojson_output), boundary, scenery_name)

    if visualize:
        print(f"\n{'='*60}")
        print(f"生成可视化报告")
        print(f"{'='*60}")
        visualize_dir = OUTPUT_DIR / f"{scenery_name}_可视化"
        os.makedirs(visualize_dir, exist_ok=True)
        
        generate_data_quality_report(df, scenery_name, visualize_dir)
        generate_spatial_distribution_map(df, scenery_name, boundary, visualize_dir)
        generate_detailed_region_analysis(df, scenery_name, visualize_dir)
        generate_review_checklist(df, scenery_name, visualize_dir)

    print(f"\n{'='*60}")
    print(f"清洗完成！")
    print(f"{'='*60}")
    print(f"  输入:   {n0:,} 点")
    print(f"  输出:   {n_final:,} 点 ({(100*n_final/n0):.1f}%)")
    print(f"  轨迹:   {df['trackId'].nunique():,} 条")
    print(f"  区域:   {df['region_id'].nunique()} 个")
    print(f"  路线:   {df['route_id'].nunique()} 个")

    generate_report(scenery_name, boundary, stages, df, str(csv_output), str(geojson_output) if geojson_output else None)

    return df


# ============================================================
# CLI
# ============================================================


def parse_boundary(boundary_str):
    """解析命令行传入的边界字符串:
    "lon1,lat1;lon2,lat2;lon3,lat3" → [(lon1,lat1), (lon2,lat2), ...]
    """
    points = boundary_str.split(";")
    boundary = []
    for p in points:
        p = p.strip()
        if not p:
            continue
        parts = p.split(",")
        if len(parts) != 2:
            raise ValueError(f"无效的边界点格式: '{p}'，需为 'lon,lat'")
        boundary.append((float(parts[0].strip()), float(parts[1].strip())))
    return boundary


def main():
    parser = argparse.ArgumentParser(
        description="景区数据清洗流水线（新版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python clean_scenery_pipeline.py --name 青城山 --csv data/青城山.csv
    python clean_scenery_pipeline.py --name 黄龙溪 --csv data/黄龙溪.csv --boundary "103.96,30.31;103.98,30.33;..."
    python clean_scenery_pipeline.py --name 青城山 --csv data/青城山.csv --export_geojson
        """,
    )
    parser.add_argument("--name", type=str, required=True, help="景区名称（需在 scenery_config.py 中配置，或配合 --boundary 使用）")
    parser.add_argument("--csv", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--boundary", type=str, default=None, help='自定义多边形边界: "lon1,lat1;lon2,lat2;..."')
    parser.add_argument("--max_jump", type=float, default=500, help="飞点位移阈值 米 (默认: 500)")
    parser.add_argument("--min_track_length", type=int, default=300, help="轨迹最小点数 (默认: 300)")
    parser.add_argument("--stay_speed", type=float, default=1.0, help="停留速度阈值 km/h (默认: 1.0)")
    parser.add_argument("--stay_duration", type=float, default=30, help="停留最短持续时间 秒 (默认: 30)")
    parser.add_argument("--dbscan_eps", type=float, default=0.0005, help="DBSCAN eps 约50m (默认: 0.0005)")
    parser.add_argument("--dbscan_min", type=int, default=10, help="DBSCAN min_samples (默认: 10)")
    parser.add_argument("--export_geojson", action="store_true", default=False, help="导出 GeoJSON 供 QGIS 审查")
    parser.add_argument("--visualize", action="store_true", default=False, help="生成可视化报告（图表和审核清单）")

    args = parser.parse_args()

    custom_boundary = None
    if args.boundary:
        try:
            custom_boundary = parse_boundary(args.boundary)
            print(f"解析自定义边界: {len(custom_boundary)} 个顶点")
        except ValueError as e:
            print(f"边界格式错误: {e}")
            sys.exit(1)

    process_scenery(
        scenery_name=args.name,
        csv_path=args.csv,
        custom_boundary=custom_boundary,
        max_jump_meters=args.max_jump,
        min_track_length=args.min_track_length,
        stay_speed_thresh=args.stay_speed,
        stay_duration_thresh=args.stay_duration,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min,
        export_geojson=args.export_geojson,
        visualize=args.visualize if 'visualize' in args else False,
    )


# ============================================================
# 可视化模块
# ============================================================

def generate_data_quality_report(df, scenery_name, output_dir):
    """
    生成数据质量报告图表，用于人工审核
    
    输出内容：
    1. 数据清洗流程概览（饼图）
    2. 轨迹长度分布（直方图）
    3. 速度分布（直方图）
    4. 海拔分布（直方图）
    5. 停留vs移动点分布（饼图）
    6. 路线分布（条形图）
    7. 区域分布（条形图）
    """
    if not PLOT_AVAILABLE:
        print("  [可视化] matplotlib未安装，跳过图表生成")
        return
    
    print("  [可视化] 生成数据质量报告...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'{scenery_name} 数据质量报告', fontsize=20, fontweight='bold')
    
    n_regions = df['region_id'].nunique()
    n_tracks = df['trackId'].nunique()
    n_routes = df['route_id'].nunique()
    n_points = len(df)
    n_stops = df['is_stop'].sum()
    
    ax1 = fig.add_subplot(3, 3, 1)
    sizes = [n_stops, n_points - n_stops]
    labels = [f'停留点\n{n_stops:,}\n({100*n_stops/n_points:.1f}%)', 
              f'移动点\n{n_points-n_stops:,}\n({100*(n_points-n_stops)/n_points:.1f}%)']
    colors = ['#ff6b6b', '#4ecdc4']
    if sum(sizes) > 0:
        ax1.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
        ax1.set_title('停留点 vs 移动点分布', fontsize=12, fontweight='bold')
    
    ax2 = fig.add_subplot(3, 3, 2)
    track_lens = df.groupby('trackId').size()
    ax2.hist(track_lens, bins=50, color='#45b7d1', edgecolor='white', alpha=0.7)
    ax2.axvline(track_lens.median(), color='red', linestyle='--', linewidth=2, label=f'中位数: {track_lens.median():.0f}')
    ax2.axvline(track_lens.mean(), color='orange', linestyle='--', linewidth=2, label=f'平均值: {track_lens.mean():.0f}')
    ax2.set_xlabel('轨迹长度 (点数)')
    ax2.set_ylabel('轨迹数量')
    ax2.set_title('轨迹长度分布', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(3, 3, 3)
    speeds = df['速度(km/h)'].dropna()
    speeds_clipped = speeds[speeds <= 20]
    ax3.hist(speeds_clipped, bins=50, color='#96ceb4', edgecolor='white', alpha=0.7)
    ax3.axvline(speeds_clipped.median(), color='red', linestyle='--', linewidth=2, label=f'中位数: {speeds_clipped.median():.1f}')
    ax3.axvline(speeds_clipped.mean(), color='orange', linestyle='--', linewidth=2, label=f'平均值: {speeds_clipped.mean():.1f}')
    ax3.set_xlabel('速度 (km/h)')
    ax3.set_ylabel('点数')
    ax3.set_title('速度分布 (<=20 km/h)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(3, 3, 4)
    altitudes = df['海拔'].dropna()
    ax4.hist(altitudes, bins=50, color='#dda0dd', edgecolor='white', alpha=0.7)
    ax4.axvline(altitudes.median(), color='red', linestyle='--', linewidth=2, label=f'中位数: {altitudes.median():.0f}m')
    ax4.axvline(altitudes.mean(), color='orange', linestyle='--', linewidth=2, label=f'平均值: {altitudes.mean():.0f}m')
    ax4.set_xlabel('海拔 (m)')
    ax4.set_ylabel('点数')
    ax4.set_title('海拔分布', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(3, 3, 5)
    route_dist = df.groupby('route_id')['trackId'].nunique().sort_index()
    bars = ax5.bar(range(len(route_dist)), route_dist.values, color=plt.cm.tab20(np.linspace(0, 1, len(route_dist))))
    ax5.set_xlabel('路线ID')
    ax5.set_ylabel('轨迹数量')
    ax5.set_title(f'路线分布 (共{n_routes}条路线)', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(len(route_dist)))
    ax5.set_xticklabels([f'R{r}' for r in route_dist.index], rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    ax6 = fig.add_subplot(3, 3, 6)
    region_dist = df.groupby('region_id').size().sort_index()
    bars = ax6.bar(range(len(region_dist)), region_dist.values, color=plt.cm.Set3(np.linspace(0, 1, len(region_dist))))
    ax6.set_xlabel('区域ID')
    ax6.set_ylabel('点数')
    ax6.set_title(f'区域分布 (共{n_regions}个区域)', fontsize=12, fontweight='bold')
    ax6.set_xticks(range(len(region_dist)))
    ax6.set_xticklabels([f'{r}' for r in region_dist.index], rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}',
                ha='center', va='bottom', fontsize=8)
    
    ax7 = fig.add_subplot(3, 3, 7)
    stats_text = f"""
    【数据统计摘要】
    
    总点数: {n_points:,}
    总轨迹: {n_tracks:,} 条
    总路线: {n_routes} 条
    总区域: {n_regions} 个
    
    停留点: {n_stops:,} ({100*n_stops/n_points:.1f}%)
    移动点: {n_points-n_stops:,} ({100*(n_points-n_stops)/n_points:.1f}%)
    
    轨迹长度:
      最短: {track_lens.min():,} 点
      最长: {track_lens.max():,} 点
      平均: {track_lens.mean():,.0f} 点
    
    速度统计:
      中位数: {speeds.median():.1f} km/h
      平均值: {speeds.mean():.1f} km/h
    
    海拔统计:
      最低: {altitudes.min():.0f} m
      最高: {altitudes.max():.0f} m
    """
    ax7.text(0.1, 0.5, stats_text, transform=ax7.transAxes, fontsize=11,
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax7.axis('off')
    ax7.set_title('统计摘要', fontsize=12, fontweight='bold')
    
    ax8 = fig.add_subplot(3, 3, 8)
    route_point_dist = df.groupby('route_id').size().sort_index()
    bars = ax8.bar(range(len(route_point_dist)), route_point_dist.values, color=plt.cm.Paired(np.linspace(0, 1, len(route_point_dist))))
    ax8.set_xlabel('路线ID')
    ax8.set_ylabel('点数')
    ax8.set_title('各路线数据点分布', fontsize=12, fontweight='bold')
    ax8.set_xticks(range(len(route_point_dist)))
    ax8.set_xticklabels([f'R{r}' for r in route_point_dist.index], rotation=45)
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(output_dir, f'{scenery_name}_数据质量报告.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    数据质量报告已保存: {output_path}")


def generate_spatial_distribution_map(df, scenery_name, boundary, output_dir):
    """
    生成空间分布图，用于人工审核
    """
    if not PLOT_AVAILABLE:
        print("  [可视化] matplotlib未安装，跳过空间分布图生成")
        return
    
    print("  [可视化] 生成空间分布图...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{scenery_name} 空间分布图', fontsize=20, fontweight='bold')
    
    ax1 = fig.add_subplot(2, 2, 1)
    sample_tracks = np.random.choice(df['trackId'].unique(), min(50, df['trackId'].nunique()), replace=False)
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, track_id in enumerate(sample_tracks):
        track = df[df['trackId'] == track_id].sort_values('时间_秒')
        ax1.plot(track['经度'], track['纬度'], color=colors[i % 20], alpha=0.6, linewidth=1)
    boundary_arr = np.array(boundary + [boundary[0]])
    ax1.plot(boundary_arr[:, 0], boundary_arr[:, 1], 'r--', linewidth=2, label='景区边界')
    ax1.set_xlabel('经度')
    ax1.set_ylabel('纬度')
    ax1.set_title(f'采样轨迹分布 (显示{sample_tracks.shape[0]}条)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    ax2 = fig.add_subplot(2, 2, 2)
    sample_size = min(5000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    scatter = ax2.scatter(sample_df['经度'], sample_df['纬度'], 
                         c=sample_df['is_stop'], cmap='coolwarm',
                         s=1, alpha=0.5)
    plt.colorbar(scatter, ax=ax2, label='停留(1) / 移动(0)')
    boundary_arr = np.array(boundary + [boundary[0]])
    ax2.plot(boundary_arr[:, 0], boundary_arr[:, 1], 'k--', linewidth=2, label='景区边界')
    ax2.set_xlabel('经度')
    ax2.set_ylabel('纬度')
    ax2.set_title(f'停留点 vs 移动点分布 (采样{sample_size}点)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    ax3 = fig.add_subplot(2, 2, 3)
    n_routes = df['route_id'].nunique()
    route_colors = plt.cm.tab20(np.linspace(0, 1, max(n_routes, 1)))
    for route_id in df['route_id'].unique()[:20]:
        route_tracks = df[df['route_id'] == route_id]['trackId'].unique()[:3]
        for track_id in route_tracks:
            track = df[(df['trackId'] == track_id) & (df['route_id'] == route_id)].sort_values('时间_秒')
            if len(track) > 0:
                ax3.plot(track['经度'], track['纬度'], color=route_colors[int(route_id) % 20], 
                        alpha=0.7, linewidth=2, label=f'路线{route_id}')
    boundary_arr = np.array(boundary + [boundary[0]])
    ax3.plot(boundary_arr[:, 0], boundary_arr[:, 1], 'k--', linewidth=2, label='景区边界')
    ax3.set_xlabel('经度')
    ax3.set_ylabel('纬度')
    ax3.set_title(f'路线分布 (显示前20条)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal', adjustable='box')
    
    ax4 = fig.add_subplot(2, 2, 4)
    n_regions = df['region_id'].nunique()
    region_colors = plt.cm.Set3(np.linspace(0, 1, max(n_regions, 1)))
    for region_id in df['region_id'].unique():
        region_points = df[df['region_id'] == region_id]
        if len(region_points) > 0:
            center_lon = region_points['经度'].mean()
            center_lat = region_points['纬度'].mean()
            ax4.scatter(center_lon, center_lat, s=100, c=[region_colors[int(region_id) % 12]], 
                       edgecolors='black', linewidth=1, alpha=0.7)
            ax4.annotate(f'R{region_id}', (center_lon, center_lat), 
                        fontsize=8, ha='center', va='bottom')
    boundary_arr = np.array(boundary + [boundary[0]])
    ax4.plot(boundary_arr[:, 0], boundary_arr[:, 1], 'k--', linewidth=2, label='景区边界')
    ax4.set_xlabel('经度')
    ax4.set_ylabel('纬度')
    ax4.set_title(f'区域质心分布 (共{n_regions}个区域)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal', adjustable='box')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(output_dir, f'{scenery_name}_空间分布图.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    空间分布图已保存: {output_path}")


def generate_detailed_region_analysis(df, scenery_name, output_dir):
    """
    生成各区域详细分析图
    """
    if not PLOT_AVAILABLE:
        return
    
    print("  [可视化] 生成区域详细分析...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    n_regions = df['region_id'].nunique()
    if n_regions == 0:
        return
    
    n_cols = min(4, n_regions)
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    fig.suptitle(f'{scenery_name} 各区域详细分析', fontsize=16, fontweight='bold')
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, region_id in enumerate(sorted(df['region_id'].unique())):
        if idx >= len(axes):
            break
        
        region_df = df[df['region_id'] == region_id]
        ax = axes[idx]
        
        sample_tracks = region_df['trackId'].unique()[:5]
        for track_id in sample_tracks:
            track = region_df[region_df['trackId'] == track_id].sort_values('时间_秒')
            ax.plot(track['经度'], track['纬度'], alpha=0.7, linewidth=1)
        
        ax.set_title(f'区域 {region_id}\n{len(region_df):,} 点, {len(sample_tracks)} 轨迹', fontsize=10)
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    for idx in range(len(df['region_id'].unique()), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(output_dir, f'{scenery_name}_区域详细分析.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    区域详细分析已保存: {output_path}")


def generate_review_checklist(df, scenery_name, output_dir):
    """
    生成人工审核清单，帮助快速检查数据质量
    """
    print("  [可视化] 生成人工审核清单...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    lines = []
    lines.append("=" * 70)
    lines.append(f"{scenery_name} 人工审核清单")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append("【请逐一检查以下内容】")
    lines.append("")
    
    lines.append("1. 空间分布检查:")
    lines.append(f"   - 打开 '空间分布图.png' 检查轨迹是否在景区内")
    lines.append(f"   - 检查是否有明显的异常轨迹（偏离主区域）")
    lines.append(f"   - 检查区域质心是否在合理位置")
    lines.append("")
    
    lines.append("2. 停留点检查:")
    n_stops = df['is_stop'].sum()
    stop_ratio = n_stops / len(df) * 100
    lines.append(f"   - 停留点总数: {n_stops:,} ({stop_ratio:.1f}%)")
    lines.append(f"   - 如果停留点过少，检查速度阈值设置")
    lines.append(f"   - 如果停留点过多，检查是否有误判")
    lines.append("")
    
    lines.append("3. 路线检查:")
    n_routes = df['route_id'].nunique()
    lines.append(f"   - 总路线数: {n_routes}")
    lines.append(f"   - 打开图表检查路线分布")
    lines.append(f"   - 检查是否有路线被错误合并或拆分")
    lines.append(f"   - 记录需要调整的路线ID")
    lines.append("")
    
    lines.append("4. 区域检查:")
    n_regions = df['region_id'].nunique()
    lines.append(f"   - 总区域数: {n_regions}")
    for region_id in sorted(df['region_id'].unique()):
        region_df = df[df['region_id'] == region_id]
        n_points = len(region_df)
        n_tracks = region_df['trackId'].nunique()
        stop_ratio = region_df['is_stop'].mean() * 100
        lines.append(f"   - 区域 {region_id}: {n_points:,} 点, {n_tracks} 轨迹, 停留比 {stop_ratio:.1f}%")
    lines.append("")
    
    lines.append("5. 轨迹长度检查:")
    track_lens = df.groupby('trackId').size()
    lines.append(f"   - 最短轨迹: {track_lens.min()} 点")
    lines.append(f"   - 最长轨迹: {track_lens.max():,} 点")
    lines.append(f"   - 平均长度: {track_lens.mean():,.0f} 点")
    lines.append(f"   - 如果有异常短的轨迹，检查是否应该合并")
    lines.append("")
    
    lines.append("【需要修正的问题】")
    lines.append("请在下方记录发现的问题和修正建议:")
    lines.append("")
    lines.append("_" * 70)
    lines.append("问题1: ")
    lines.append("  位置: ")
    lines.append("  描述: ")
    lines.append("  建议修正: ")
    lines.append("")
    lines.append("_" * 70)
    lines.append("问题2: ")
    lines.append("  位置: ")
    lines.append("  描述: ")
    lines.append("  建议修正: ")
    lines.append("")
    lines.append("_" * 70)
    lines.append("问题3: ")
    lines.append("  位置: ")
    lines.append("  描述: ")
    lines.append("  建议修正: ")
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("审核完成签名: ___________________")
    lines.append("审核日期: ___________________")
    lines.append("=" * 70)
    
    content = "\n".join(lines)
    
    output_path = os.path.join(output_dir, f'{scenery_name}_人工审核清单.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"    审核清单已保存: {output_path}")
    print("\n" + content)


if __name__ == "__main__":
    main()
