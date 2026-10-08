#!/usr/bin/env python3
"""
POI 路径投影：将 POI 点映射到最近的 GPS 轨迹点（路径）上。

目的：
  - 游客的移动只在路径上，POI 原始坐标可能在路径之外
  - 投影后：每个 POI 获得一个"路径锚点"(path_lon, path_lat)
  - 区域间路径距离可在预测阶段按轨迹实际路径计算

输出:
  - poi/data/projected/poi_path_projected.json: 所有 POI 投影结果（含 region_id）
  - poi/data/projected/poi_path_projected.csv:   同上，CSV 格式
  - poi/data/projected/poi_path_各景区.csv:      按景区拆分
  - poi/data/projected/region_anchors.csv:       每个区域在路径上的锚点
"""

import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


# ============================================================
# 常量
# ============================================================
CLUSTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cluster', 'output'))
POI_JSON = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'raw', 'scenery_poi_scenic_only.json'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'projected'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENERY_CSV_MAP = {
    "峨眉山": "峨眉山", "都江堰": "都江堰", "青城山": "青城山",
    "龙泉": "龙泉", "武侯祠博物馆": "武侯祠博物馆",
    "熊猫基地": "熊猫基地", "锦江": "锦江",
}

# ============================================================
# 工具函数
# ============================================================
def haversine_meters(lon1, lat1, lon2, lat2):
    """两点间球面距离（米）"""
    R = 6371000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def project_poi(poi_lon, poi_lat, path_lons, path_lats, path_tree, path_df, time_col):
    """
    将 POI 投影到最近的 GPS 路径点。
    返回: 投影坐标、到路径距离、所在 region_id
    """
    raw_point = np.array([[poi_lon, poi_lat]])
    _, idx_candidates = path_tree.query(raw_point, k=5)
    if idx_candidates.ndim > 1:
        idx_candidates = idx_candidates[0]

    best_idx = idx_candidates[0]
    best_dist = float('inf')
    for idx in idx_candidates:
        d = haversine_meters(poi_lon, poi_lat, path_lons[idx], path_lats[idx])
        if d < best_dist:
            best_dist = d
            best_idx = idx

    nearest_row = path_df.iloc[best_idx]
    return {
        'projected_lon': float(path_lons[best_idx]),
        'projected_lat': float(path_lats[best_idx]),
        'dist_to_path_m': round(best_dist, 1),
        'projected_region_id': int(nearest_row['region_id']),
    }


# ============================================================
# 主流程
# ============================================================
def main():
    # 1. 加载 POI
    with open(POI_JSON, 'r', encoding='utf-8') as f:
        all_pois = json.load(f)
    print(f"Loaded {len(all_pois):,} POIs total")

    pois_by_scenery = defaultdict(list)
    for poi in all_pois:
        pois_by_scenery[poi['scenery']].append(poi)

    all_projected = []
    summary_rows = []

    for scenery_name, csv_name in SCENERY_CSV_MAP.items():
        csv_path = os.path.join(CLUSTER_DIR, f"{csv_name}_clustered.csv")
        if not os.path.exists(csv_path):
            continue

        pois_scene = pois_by_scenery.get(scenery_name, [])
        if not pois_scene:
            continue

        print(f"\n{'='*60}")
        print(f"[{scenery_name}]")

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        time_col = '时间_秒'
        if time_col not in df.columns:
            alt = [c for c in df.columns if '时间' in c or 'time' in c.lower()]
            time_col = alt[0] if alt else df.columns[0]

        df = df.sort_values(time_col).reset_index(drop=True)

        n_total = len(df)
        n_stay = int((df['is_stop'] == 1).sum())
        n_regions = df['region_id'].nunique()
        print(f"  总点数: {n_total:,}, 停留点: {n_stay:,}, 区域数: {n_regions}")

        path_lons = df['经度'].values.astype(np.float64)
        path_lats = df['纬度'].values.astype(np.float64)
        path_points = np.column_stack([path_lons, path_lats])
        path_tree = KDTree(path_points)

        # 投影每个 POI
        projected_list = []
        for poi in pois_scene:
            p = project_poi(poi['lon'], poi['lat'],
                           path_lons, path_lats, path_tree, df, time_col)
            p['scenery'] = scenery_name
            p['poi_id'] = poi['poi_id']
            p['poi_name'] = poi['name']
            p['original_lon'] = poi['lon']
            p['original_lat'] = poi['lat']
            p['type_code'] = poi.get('type_code', '')
            p['type_name'] = poi.get('type_name', '')
            projected_list.append(p)

        n_poi = len(projected_list)
        dists = [p['dist_to_path_m'] for p in projected_list]
        print(f"  POIs: {n_poi}")
        print(f"  距路径: 中位数={np.median(dists):.1f}m, P90={np.percentile(dists, 90):.1f}m")

        # 按 region 汇总
        by_region = defaultdict(list)
        for p in projected_list:
            by_region[p['projected_region_id']].append(p)

        for rid in sorted(by_region.keys()):
            ps = by_region[rid]
            near = sum(1 for p in ps if p['dist_to_path_m'] <= 50)
            top5 = [p['poi_name'] for p in ps[:5]]
            print(f"    R{rid}: {len(ps)} POIs, ≤50m={near}/{len(ps)}, eg. {', '.join(top5)}")

        all_projected.extend(projected_list)
        for p in projected_list:
            summary_rows.append(p)

        # 保存每景区的 CSV
        pd.DataFrame(projected_list).to_csv(
            os.path.join(OUTPUT_DIR, f"poi_path_{scenery_name}.csv"),
            index=False, encoding='utf-8-sig')
        print(f"  已保存: poi_path_{scenery_name}.csv")

    # === 全部投影结果 ===
    json_path = os.path.join(OUTPUT_DIR, 'poi_path_projected.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_projected, f, ensure_ascii=False, indent=2)
    print(f"\n全部投影: {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, 'poi_path_projected.csv')
    df_all = pd.DataFrame(summary_rows)
    df_all.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"全部投影: {csv_path}")

    # === 区域锚点 ===
    print(f"\n{'='*60}")
    print("区域锚点 (region anchors)")
    anchor_rows = []
    for scenery_name in SCENERY_CSV_MAP:
        g = df_all[df_all['scenery'] == scenery_name]
        if len(g) == 0:
            continue
        for rid, grp in g.groupby('projected_region_id'):
            near_mask = grp['dist_to_path_m'] <= 100
            if near_mask.sum() == 0:
                # 无 POI 在 100m 内，用所有 POI 投影的中位数
                anchor_lon = grp['projected_lon'].median()
                anchor_lat = grp['projected_lat'].median()
            else:
                # 用路径附近 POI 的中位数
                near = grp[near_mask]
                anchor_lon = near['projected_lon'].median()
                anchor_lat = near['projected_lat'].median()
            n_poi = len(grp)
            n_near = int(near_mask.sum())
            anchor_rows.append({
                'scenery': scenery_name,
                'region_id': int(rid),
                'anchor_lon': round(anchor_lon, 6),
                'anchor_lat': round(anchor_lat, 6),
                'n_poi_total': n_poi,
                'n_poi_near_path': n_near,
            })
            print(f"  {scenery_name} R{rid}: anchor=({anchor_lon:.4f}, {anchor_lat:.4f}), "
                  f"{n_near}/{n_poi} POIs ≤100m")

    df_anchor = pd.DataFrame(anchor_rows)
    anchor_path = os.path.join(OUTPUT_DIR, 'region_anchors.csv')
    df_anchor.to_csv(anchor_path, index=False, encoding='utf-8-sig')
    print(f"\n区域锚点: {anchor_path}")

    # === 总体统计 ===
    print(f"\n{'='*60}")
    print("总体统计")
    print(f"  POI 总数: {len(df_all)}")
    print(f"  覆盖景区: {df_all['scenery'].nunique()}")
    for q in [10, 25, 50, 75, 90]:
        v = np.percentile(df_all['dist_to_path_m'], q)
        print(f"  距路径 P{q}: {v:.1f}m")
    print(f"  ≤30m: {(df_all['dist_to_path_m']<=30).sum()}/{len(df_all)} ({(df_all['dist_to_path_m']<=30).mean()*100:.0f}%)")
    print(f"  ≤50m: {(df_all['dist_to_path_m']<=50).sum()}/{len(df_all)} ({(df_all['dist_to_path_m']<=50).mean()*100:.0f}%)")
    print(f"  ≤100m: {(df_all['dist_to_path_m']<=100).sum()}/{len(df_all)} ({(df_all['dist_to_path_m']<=100).mean()*100:.0f}%)")

    for scene, grp in df_all.groupby('scenery'):
        print(f"  {scene}: {len(grp)} POIs, "
              f"中位数={grp['dist_to_path_m'].median():.1f}m, "
              f"≤50m={grp['dist_to_path_m'].le(50).mean()*100:.0f}%")


if __name__ == '__main__':
    main()
