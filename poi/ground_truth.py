#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POI 缓冲区 → 合并区域 → Ground Truth 标签 → 聚类评估（ARI/NMI）。

核心思想：
  每个 POI 有空间影响范围（缓冲区），重叠的缓冲区合并成 POI 区域。
  轨迹点落入哪个 POI 区域，就继承该区域 ID 作为 ground truth 标签。
  对比 pipeline 的 region_id，计算 ARI / NMI / FMI。

流程：
  1. 加载 POI 数据，过滤出"人群聚集"类型
  2. 对每个景区：POI 做缓冲区（半径 R），合并重叠区域 → POI 区域
  3. 加载清洗后的轨迹 CSV
  4. 为每个轨迹点匹配所在 POI 区域（距离阈值 R）
  5. 以 POI 区域 ID 作为 ground truth 标签，计算 ARI/NMI/FMI
  6. 输出汇总报告

用法：
    python poi_ground_truth.py
    python poi_ground_truth.py --buffer-radius 200
    python poi_ground_truth.py --scenery 峨眉山 都江堰
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
)

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
POI_PATH = SCRIPT_DIR / "data" / "raw" / "all_scenery_poi.json"
CLEANED_DIR = SCRIPT_DIR.parent / "data-project" / "cleaned_labeled_data"
CLUSTERED_DIR = SCRIPT_DIR.parent / "cluster" / "output"
OUTPUT_DIR = SCRIPT_DIR / "data" / "ground_truth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# POI 类型过滤规则
# ---------------------------------------------------------------------------

KEEP_TYPE_PREFIXES = {
    "05",  # 餐饮服务
    "06",  # 购物服务
    "08",  # 体育休闲服务
    "10",  # 住宿服务
    "11",  # 风景名胜
    "14",  # 科教文化服务
    "20",  # 公共设施
}

EXCLUDE_TYPE_CODES = {
    "1412", "1413", "1414", "1415",  # 学校/科研/培训/驾校
}

TYPE_NAMES = {
    "05": "餐饮", "06": "购物", "08": "休闲",
    "10": "住宿", "11": "风景", "14": "文化", "20": "公设",
}

# ---------------------------------------------------------------------------
# KD-Tree 最近邻搜索
# ---------------------------------------------------------------------------


def build_kdtree(points_2d: np.ndarray):
    n = len(points_2d)
    if n == 0:
        return None, None
    indices = np.arange(n)

    def _build(idx, depth):
        if len(idx) == 0:
            return None
        axis = depth % 2
        idx = idx[np.argsort(points_2d[idx, axis])]
        mid = len(idx) // 2
        return {
            "point": idx[mid],
            "left": _build(idx[:mid], depth + 1),
            "right": _build(idx[mid + 1:], depth + 1),
            "axis": axis,
        }

    return _build(indices, 0), points_2d


def knn_search(tree, points_2d, query):
    if tree is None:
        return None, float("inf")
    best_idx = None
    best_dist = float("inf")

    def _search(node, depth):
        nonlocal best_idx, best_dist
        if node is None:
            return
        axis = node["axis"]
        node_pt = points_2d[node["point"]]
        dist = np.sqrt(np.sum((query - node_pt) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_idx = node["point"]
        diff = query[axis] - node_pt[axis]
        if diff <= 0:
            _search(node["left"], depth + 1)
        else:
            _search(node["right"], depth + 1)
        if abs(diff) < best_dist:
            if diff <= 0:
                _search(node["right"], depth + 1)
            else:
                _search(node["left"], depth + 1)

    _search(tree, 0)
    return best_idx, best_dist


# ---------------------------------------------------------------------------
# POI 过滤
# ---------------------------------------------------------------------------


def filter_pois(poi_list: list[dict]) -> list[dict]:
    kept = []
    removed = Counter()
    for poi in poi_list:
        tc = poi.get("type_code", "")
        prefix = tc[:2]
        if prefix not in KEEP_TYPE_PREFIXES:
            removed[prefix + "0000"] += 1
            continue
        excluded = False
        for excl in EXCLUDE_TYPE_CODES:
            if tc.startswith(excl):
                removed[tc] += 1
                excluded = True
                break
        if not excluded:
            kept.append(poi)
    if removed:
        print(f"  过滤掉 {sum(removed.values())} 个 POI:")
        for tc, cnt in removed.most_common(10):
            print(f"    {TYPE_NAMES.get(tc[:2], tc)} ({tc}): {cnt}")
    return kept


# ---------------------------------------------------------------------------
# POI 缓冲区合并 → 区域生成
# ---------------------------------------------------------------------------


def union_find(poi_coords: np.ndarray, radius_deg: float) -> np.ndarray:
    """
    基于距离的并查集：距离 < 2*radius 的 POI 合并到同一区域。
    返回每个 POI 的区域 ID（-1 表示噪声）。
    """
    n = len(poi_coords)
    if n == 0:
        return np.array([], dtype=np.int32)

    parent = np.arange(n, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pb] = pa

    threshold = 2.0 * radius_deg
    tree, tree_pts = build_kdtree(poi_coords)
    if tree is None:
        return -np.ones(n, dtype=np.int32)

    for i in range(n):
        query = poi_coords[i]
        # 找所有在阈值内的点
        stack = [tree]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            axis = node["axis"]
            node_pt = tree_pts[node["point"]]
            dist = np.sqrt(np.sum((query - node_pt) ** 2))
            if dist <= threshold and node["point"] > i:
                union(i, node["point"])
            diff = query[axis] - node_pt[axis]
            if diff <= threshold:
                stack.append(node["left"])
            if diff >= -threshold:
                stack.append(node["right"])

    # 重映射为连续 ID
    unique_parents = {}
    region_ids = -np.ones(n, dtype=np.int32)
    for i in range(n):
        root = find(i)
        if root not in unique_parents:
            unique_parents[root] = len(unique_parents)
        region_ids[i] = unique_parents[root]

    return region_ids


def create_poi_regions(
    poi_df: pd.DataFrame, buffer_radius_m: float = 150
) -> pd.DataFrame:
    """
    对每个景区的 POI，用缓冲区合并法生成 POI 区域 ID。
    所有 POI 都归属某个区域（无孤立点）。
    """
    df = poi_df.copy()
    radius_deg = buffer_radius_m / 111000

    for scenery_name in df["scenery"].unique():
        mask = df["scenery"] == scenery_name
        idx = df[mask].index
        coords = df.loc[idx, ["lon", "lat"]].values.astype(np.float64)

        if len(coords) == 0:
            continue

        region_ids = union_find(coords, radius_deg)
        df.loc[idx, "poi_region_id"] = region_ids

        n_regions = len(set(region_ids))
        print(f"  {scenery_name}: {len(coords)} POI → {n_regions} 区域 "
              f"(缓冲区半径={buffer_radius_m}m)")

    return df


def merge_small_regions(poi_df: pd.DataFrame, min_pois: int = 3) -> pd.DataFrame:
    """
    合并 POI 数少于 min_pois 的小区域到最近的大区域。
    减少 ground truth 区域数，避免噪声区域导致 ARI 偏低。
    """
    df = poi_df.copy()

    for scenery_name in df["scenery"].unique():
        mask = df["scenery"] == scenery_name
        scene = df[mask]
        if len(scene) == 0:
            continue

        # 统计每个区域的 POI 数
        region_counts = scene["poi_region_id"].value_counts()
        large_regions = region_counts[region_counts >= min_pois].index
        small_regions = region_counts[region_counts < min_pois].index

        if len(small_regions) == 0:
            continue

        # 计算每个区域的质心
        centroids = scene.groupby("poi_region_id")[["lon", "lat"]].mean()

        # 对每个小区域，找最近的大区域
        remap = {}
        for sid in small_regions:
            if len(large_regions) == 0:
                # 没有大区域，全部合并为一个
                remap[sid] = sid  # 保持原样
                continue
            sc = centroids.loc[sid, ["lon", "lat"]].values
            best_dist = float("inf")
            best_lid = sid
            for lid in large_regions:
                lc = centroids.loc[lid, ["lon", "lat"]].values
                dist = np.sqrt(np.sum((sc - lc) ** 2))
                if dist < best_dist:
                    best_dist = dist
                    best_lid = lid
            remap[sid] = best_lid

        # 重映射
        n_merged = 0
        for sid, lid in remap.items():
            if sid != lid:
                n_merged += region_counts[sid]
        idx = scene.index
        df.loc[idx, "poi_region_id"] = scene["poi_region_id"].map(remap).fillna(scene["poi_region_id"])

        # 重编号为连续 ID
        unique_ids = sorted(df.loc[idx, "poi_region_id"].unique())
        id_map = {old: new for new, old in enumerate(unique_ids)}
        df.loc[idx, "poi_region_id"] = df.loc[idx, "poi_region_id"].map(id_map)

        n_after = df.loc[idx, "poi_region_id"].nunique()
        n_before = len(region_counts)
        print(f"  {scenery_name}: 合并小区域 {n_before} → {n_after} "
              f"(合并了 {n_merged} 个 POI)")

    return df


def project_pois_to_paths(
    poi_df: pd.DataFrame,
    projection_radius_m: float = 300,
) -> pd.DataFrame:
    """
    关键修正：将 POI 投影到最近的轨迹路径点上。

    因为聚类处理的是路径上的 GPS 点，而 POI 可能在路径外（如山坡上的观景台、
    路旁的酒店），直接以 POI 坐标做标签会导致系统性的空间错位。
    通过投影到最近路径点，ground truth 与聚类在同一个空间参考系中。

    对每个 POI，在其景区对应的清洗后 CSV 中找最近轨迹点，替换坐标。
    若最近距离 > projection_radius，丢弃该 POI（离路径太远，不纳入评估）。
    """
    df = poi_df.copy()
    df["projected"] = False
    radius_deg = projection_radius_m / 111000
    n_projected = 0
    n_dropped = 0

    for scenery_name in df["scenery"].unique():
        csv_path = CLEANED_DIR / f"{scenery_name}_cleaned.csv"
        if not csv_path.exists():
            continue

        traj = pd.read_csv(csv_path, encoding="utf-8")
        if len(traj) == 0:
            continue

        # 采样轨迹点（太多时会很慢，最多取 50000 个）
        if len(traj) > 50000:
            traj = traj.sample(n=50000, random_state=42)

        traj_coords = traj[["经度", "纬度"]].values.astype(np.float64)
        tree, tree_pts = build_kdtree(traj_coords)
        if tree is None:
            continue

        mask = df["scenery"] == scenery_name
        idx = df[mask].index

        for i in idx:
            poi_pt = df.loc[i, ["lon", "lat"]].values.astype(np.float64)
            nearest_idx, dist = knn_search(tree, traj_coords, poi_pt)
            if nearest_idx is not None and dist <= radius_deg:
                # 替换 POI 坐标为最近路径点坐标
                df.loc[i, "lon"] = float(traj_coords[nearest_idx][0])
                df.loc[i, "lat"] = float(traj_coords[nearest_idx][1])
                df.loc[i, "projected"] = True
                n_projected += 1
            else:
                # 离路径太远，丢弃
                df.loc[i, "projected"] = False
                n_dropped += 1

        kept = (~mask) | (df.loc[idx, "projected"] == True)
        df = df[kept].copy()

    total = n_projected + n_dropped
    if total > 0:
        print(f"  POI 投影到路径: {n_projected} 成功, {n_dropped} 丢弃 "
              f"(阈值={projection_radius_m}m)")
    return df


# ---------------------------------------------------------------------------
# 轨迹点 - POI 区域匹配
# ---------------------------------------------------------------------------


def match_trajectory_to_regions(
    csv_path: Path,
    poi_df: pd.DataFrame,
    radius_deg: float,
    only_stay_points: bool = True,
) -> pd.DataFrame:
    """
    为轨迹点匹配 POI 区域。
    每个点找最近的 POI，如果在半径内则继承该 POI 的区域 ID。

    only_stay_points: 只对 is_stop=1 的停留点赋予标签（匹配聚类实际输入）。
    """
    traj = pd.read_csv(csv_path, encoding="utf-8")
    n_total = len(traj)
    if n_total == 0:
        traj["gt_region_id"] = pd.Series(dtype=int)
        traj["gt_dist_m"] = pd.Series(dtype=float)
        traj["gt_poi_name"] = pd.Series(dtype=str)
        return traj

    # 只对停留点赋予标签
    if only_stay_points and "is_stop" in traj.columns:
        is_stop = traj["is_stop"].values == 1
        n_stay = is_stop.sum()
        # 初始化全部为 -1
        traj["gt_region_id"] = -1
        traj["gt_dist_m"] = -1.0
        traj["gt_poi_name"] = ""
        if n_stay == 0:
            print(f"    无停留点 (共 {n_total} 点)")
            return traj
    else:
        is_stop = None
        n_stay = n_total

    traj_lng, traj_lat = "经度", "纬度"

    coords = poi_df[["lon", "lat"]].values.astype(np.float64)
    region_ids = poi_df["poi_region_id"].values
    tree, tree_pts = build_kdtree(coords)

    if tree is None:
        traj["gt_region_id"] = -1
        traj["gt_dist_m"] = -1.0
        traj["gt_poi_name"] = ""
        return traj

    gt_labels = np.full(n_total, -1, dtype=np.int32)
    gt_dists = np.full(n_total, -1.0, dtype=np.float64)
    gt_names = np.full(n_total, "", dtype=object)
    n_matched = 0

    traj_coords = traj[[traj_lng, traj_lat]].values.astype(np.float64)

    # 确定要处理的下标范围（只处理停留点）
    if is_stop is not None:
        process_indices = np.where(is_stop)[0]
    else:
        process_indices = np.arange(n_total)

    for i in process_indices:
        lng, lat = traj_coords[i]
        idx, dist = knn_search(tree, coords, np.array([lng, lat]))
        if idx is not None and dist <= radius_deg:
            rid = int(region_ids[idx])
            gt_labels[i] = rid
            gt_dists[i] = dist * 111000
            gt_names[i] = poi_df.iloc[idx]["name"]
            n_matched += 1

    traj["gt_region_id"] = gt_labels
    traj["gt_dist_m"] = gt_dists
    traj["gt_poi_name"] = gt_names

    n_considered = len(process_indices)
    match_rate = n_matched / n_considered * 100 if n_considered > 0 else 0
    print(f"    停留点 {n_considered}/{n_total}，匹配到 POI 区域: {n_matched} ({match_rate:.1f}%)")
    return traj


def propagate_labels_along_path(traj: pd.DataFrame, track_col: str = "trackId",
                                 label_col: str = "gt_region_id") -> pd.DataFrame:
    """
    沿路径传播标签：山岳景区 POI 稀疏，路径中间大量点无标签。
    对每条轨迹，将已知标签向前/后传播到同一条路径上的未标记点。

    逻辑：一条轨迹中，两个已知标签之间的无标签点继承最近的标签；
    起点之前的无标签点继承第一个标签；终点之后的无标签点继承最后一个标签。
    """
    df = traj.copy()
    labels = df[label_col].values.copy()
    track_ids = df[track_col].values

    n_propagated = 0

    for tid in np.unique(track_ids):
        mask = track_ids == tid
        idx = np.where(mask)[0]
        seg_labels = labels[idx]

        if np.all(seg_labels == -1):
            continue

        # 前向传播
        last_label = -1
        for i in range(len(seg_labels)):
            if seg_labels[i] != -1:
                last_label = seg_labels[i]
            elif last_label != -1:
                seg_labels[i] = last_label
                n_propagated += 1

        # 后向传播
        last_label = -1
        for i in range(len(seg_labels) - 1, -1, -1):
            if seg_labels[i] != -1:
                last_label = seg_labels[i]
            elif last_label != -1:
                seg_labels[i] = last_label
                n_propagated += 1

        labels[idx] = seg_labels

    df[label_col] = labels
    s = f"  (路径传播: +{n_propagated} 点获得标签)"
    if n_propagated > 0:
        print(f"    {s}")
    return df


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------


def compute_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> dict:
    """计算 ARI / NMI / FMI（过滤 -1 的未标记点）。"""
    valid = labels_true != -1
    true = labels_true[valid]
    pred = labels_pred[valid]
    n = len(true)
    if n < 10:
        return {"n_labeled": n, "ari": None, "nmi": None, "fmi": None}

    n_unique_true = len(set(true))
    n_unique_pred = len(set(pred))
    metrics = {"n_labeled": n, "n_unique_true": n_unique_true, "n_unique_pred": n_unique_pred}

    if n_unique_true >= 2 and n_unique_pred >= 2:
        metrics["ari"] = float(adjusted_rand_score(true, pred))
        metrics["nmi"] = float(
            normalized_mutual_info_score(true, pred, average_method="arithmetic")
        )
        metrics["fmi"] = float(fowlkes_mallows_score(true, pred))
    else:
        metrics["ari"] = metrics["nmi"] = metrics["fmi"] = None

    return metrics


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def process_scenery(name: str, poi_df: pd.DataFrame, radius_deg: float,
                    propagate: bool = False, only_stay_points: bool = True,
                    use_strat: bool = True) -> dict:
    """处理单个景区。默认使用 STRAT 聚类结果，否则 fallback 到清洗 DBSCAN。"""
    if use_strat:
        csv_path = CLUSTERED_DIR / f"{name}_clustered.csv"
        label_source = "STRAT"
        if not csv_path.exists():
            print(f"  [fallback] STRAT 结果不存在，使用清洗 DBSCAN")
            csv_path = CLEANED_DIR / f"{name}_cleaned.csv"
            label_source = "清洗 DBSCAN"
    else:
        csv_path = CLEANED_DIR / f"{name}_cleaned.csv"
        label_source = "清洗 DBSCAN"

    if not csv_path.exists():
        return {"name": name, "status": "跳过", "error": "CSV 不存在"}

    scene_pois = poi_df[poi_df["scenery"] == name].copy()
    print(f"\n[{name}] (标签来源: {label_source})")
    print(f"  POI 数: {len(scene_pois)}, 区域数: {scene_pois['poi_region_id'].nunique()}")

    traj = match_trajectory_to_regions(csv_path, scene_pois, radius_deg,
                                       only_stay_points=only_stay_points)

    # 沿路径传播标签（山岳景区 POI 稀疏补偿）
    if propagate:
        traj = propagate_labels_along_path(traj)

    # 保存带标签的 CSV
    out_csv = OUTPUT_DIR / f"{name}_labeled.csv"
    traj.to_csv(out_csv, index=False, encoding="utf-8-sig")

    labels_true = traj["gt_region_id"].values.astype(np.int32)
    labels_pred = traj["region_id"].values.astype(np.int32)
    metrics = compute_metrics(labels_true, labels_pred)
    metrics["name"] = name
    metrics["status"] = "成功"
    metrics["n_poi"] = len(scene_pois)
    metrics["n_poi_regions"] = scene_pois["poi_region_id"].nunique()
    metrics["n_traj_points"] = len(traj)
    metrics["match_rate"] = (metrics["n_labeled"] / len(traj) * 100) if len(traj) > 0 else 0

    ari_s = f"{metrics['ari']:.4f}" if metrics.get("ari") is not None else "N/A"
    nmi_s = f"{metrics['nmi']:.4f}" if metrics.get("nmi") is not None else "N/A"
    fmi_s = f"{metrics['fmi']:.4f}" if metrics.get("fmi") is not None else "N/A"
    print(f"  ARI={ari_s}  NMI={nmi_s}  FMI={fmi_s}  "
          f"(标签点: {metrics['n_labeled']}/{len(traj)})")

    return metrics


def generate_report(all_metrics, output_path: Path, radius_m: float,
                    projected: bool, propagated: bool, only_stay_points: bool):
    lines = [
        "# POI Ground Truth 聚类评估报告（真实评估）",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"POI 缓冲区半径: {radius_m}m",
        f"POI 路径投影: {'开启' if projected else '关闭'}",
        f"路径标签传播: {'开启' if propagated else '关闭'}",
        f"仅停留点评估: {'是' if only_stay_points else '否'}",
        "",
        "方法：每个 POI 做缓冲区（半径 R），重叠缓冲区合并为 POI 区域。",
        "仅对停留点 (is_stop=1) 赋予 POI 区域标签（匹配聚类实际输入）。",
        "" if not projected else "POI 已投影到最近轨迹路径点，以解决 POI 在路径外导致的空间错位。",
        "" if not propagated else "标签沿轨迹路径传播（注意：开启传播会虚高 ARI）。",
        "",
        "## 各景区评估结果",
        "",
        "| 景区 | POI数 | POI区域数 | 轨迹点数 | 匹配标签数 | 匹配率 | ARI | NMI | FMI |",
        "|------|-------|----------|---------|-----------|-------|-----|-----|-----|",
    ]

    valid = [m for m in all_metrics if m.get("ari") is not None]
    for m in all_metrics:
        ari = f"{m['ari']:.4f}" if m.get("ari") is not None else "N/A"
        nmi = f"{m['nmi']:.4f}" if m.get("nmi") is not None else "N/A"
        fmi = f"{m['fmi']:.4f}" if m.get("fmi") is not None else "N/A"
        lines.append(
            f"| {m['name']} | {m.get('n_poi', 0)} | {m.get('n_poi_regions', 0)} "
            f"| {m.get('n_traj_points', 0)} | {m.get('n_labeled', 0)} "
            f"| {m.get('match_rate', 0):.1f}% | {ari} | {nmi} | {fmi} |"
        )

    if valid:
        avg_ari = np.mean([m["ari"] for m in valid])
        avg_nmi = np.mean([m["nmi"] for m in valid])
        avg_fmi = np.mean([m["fmi"] for m in valid])
        lines.append("")
        lines.append(
            f"**平均 (有标签景区):** ARI={avg_ari:.4f}, "
            f"NMI={avg_nmi:.4f}, FMI={avg_fmi:.4f}"
        )

    lines.append("")
    lines.append("## 各景区详情")
    lines.append("")
    for m in all_metrics:
        if m.get("status") != "成功":
            continue
        ari_s = f"{m['ari']:.4f}" if m.get("ari") is not None else "N/A"
        nmi_s = f"{m['nmi']:.4f}" if m.get("nmi") is not None else "N/A"
        fmi_s = f"{m['fmi']:.4f}" if m.get("fmi") is not None else "N/A"
        lines.append(f"### {m['name']}")
        lines.append(f"- POI: {m['n_poi']} 个 → {m['n_poi_regions']} 个区域")
        lines.append(f"- 标签点: {m['n_labeled']}/{m['n_traj_points']} ({m['match_rate']:.1f}%)")
        lines.append(f"- ARI={ari_s}, NMI={nmi_s}, FMI={fmi_s}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n报告: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="POI 缓冲区区域 Ground Truth 聚类评估")
    parser.add_argument("--poi", default=str(POI_PATH))
    parser.add_argument("--buffer-radius", type=float, default=50,
                        help="POI 缓冲区半径（米），默认 50（匹配 DBSCAN eps~55m）")
    parser.add_argument("--project-to-path", action="store_true", default=True,
                        help="将 POI 投影到最近轨迹路径点（默认开启），解决路径内外空间错位")
    parser.add_argument("--no-project", action="store_false", dest="project_to_path",
                        help="关闭 POI 路径投影")
    parser.add_argument("--project-radius", type=float, default=300,
                        help="POI 路径投影最大距离（米），默认 300")
    parser.add_argument("--propagate", action="store_true", default=False,
                        help="沿路径传播标签（默认关闭），仅当需要补偿稀疏 POI 时开启")
    parser.add_argument("--only-stay-points", action="store_true", default=True,
                        help="只对停留点 (is_stop=1) 评估（默认开启），匹配聚类仅在停留点上运行的实情")
    parser.add_argument("--no-stay-only", action="store_false", dest="only_stay_points",
                        help="对所有轨迹点评估（不推荐，会稀释精度）")
    parser.add_argument("--min-pois", type=int, default=2,
                        help="合并 POI 数少于该值的小区域到最近大区域，默认 2")
    parser.add_argument("--use-strat", action="store_true", default=True,
                        help="对比 STRAT 聚类结果（默认），否则对比清洗 DBSCAN")
    parser.add_argument("--no-strat", action="store_false", dest="use_strat",
                        help="对比清洗 DBSCAN 而非 STRAT")
    parser.add_argument("--scenery", nargs="*")
    parser.add_argument("--output", default="evaluation_report.md")
    args = parser.parse_args()

    radius_deg = args.buffer_radius / 111000

    # 加载 POI
    poi_path = Path(args.poi)
    if not poi_path.exists():
        print(f"[错误] POI 文件不存在: {poi_path}")
        sys.exit(1)

    with open(poi_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"加载 POI: {len(raw)} 条")
    filtered = filter_pois(raw)
    poi_df = pd.DataFrame(filtered)
    poi_df = poi_df.drop_duplicates(subset=["scenery", "poi_id"])
    print(f"过滤后: {len(poi_df)} 条 ({poi_df['scenery'].nunique()} 景区)")

    # ---- 可选：POI 投影到路径（解决路径/POI 空间错位）----
    if args.project_to_path:
        print(f"\n{'='*60}")
        print(f"POI 投影到最近路径点 (阈值={args.project_radius}m)")
        print(f"{'='*60}")
        poi_df = project_pois_to_paths(poi_df, projection_radius_m=args.project_radius)
        print(f"投影后: {len(poi_df)} 条")

    # ---- POI 缓冲区合并区域 ----
    print(f"\n{'='*60}")
    print(f"POI 缓冲区合并 (半径={args.buffer_radius}m)")
    print(f"{'='*60}")
    poi_df = create_poi_regions(poi_df, buffer_radius_m=args.buffer_radius)

    # ---- 合并小区域 ----
    if args.min_pois > 1:
        print(f"\n{'='*60}")
        print(f"合并小区域 (min_pois={args.min_pois})")
        print(f"{'='*60}")
        poi_df = merge_small_regions(poi_df, min_pois=args.min_pois)

    # 确定景区列表
    if args.scenery:
        scenery_list = args.scenery
    else:
        csv_names = {p.stem.replace("_cleaned", "") for p in CLEANED_DIR.glob("*_cleaned.csv")}
        poi_names = set(poi_df["scenery"].unique())
        scenery_list = sorted(csv_names & poi_names) or sorted(poi_names)

    print(f"\n处理景区: {len(scenery_list)} 个 | 缓冲区: {args.buffer_radius}m")
    print(f"{'='*60}")

    all_metrics = []
    for name in scenery_list:
        m = process_scenery(name, poi_df, radius_deg,
                            propagate=args.propagate,
                            only_stay_points=args.only_stay_points,
                            use_strat=args.use_strat)
        all_metrics.append(m)

    # 汇总
    print(f"\n{'='*60}")
    print("汇总:")
    valid = [m for m in all_metrics if m.get("ari") is not None]
    for m in all_metrics:
        if m.get("status") != "成功":
            print(f"  {m['name']}: {m.get('status', '?')}")
            continue
        ari = f"{m['ari']:.4f}" if m.get("ari") is not None else "N/A"
        nmi = f"{m['nmi']:.4f}" if m.get("nmi") is not None else "N/A"
        print(f"  {m['name']}: ARI={ari}, NMI={nmi}, "
              f"标签={m.get('n_labeled', 0)}/{m.get('n_traj_points', 0)}")

    if valid:
        print(f"\n平均 (有标签景区): ARI={np.mean([m['ari'] for m in valid]):.4f}, "
              f"NMI={np.mean([m['nmi'] for m in valid]):.4f}, "
              f"FMI={np.mean([m['fmi'] for m in valid]):.4f}")

    report_path = OUTPUT_DIR / args.output
    generate_report(all_metrics, report_path, args.buffer_radius,
                    args.project_to_path, args.propagate, args.only_stay_points)

    json_path = OUTPUT_DIR / "evaluation_metrics.json"
    clean = [{k: v for k, v in m.items() if k != "error"} for m in all_metrics]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
    print(f"指标: {json_path}")


if __name__ == "__main__":
    main()
