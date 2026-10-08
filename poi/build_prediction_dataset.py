#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建预测数据集。

从聚类输出 + POI 投影数据中提取路段序列，
每条记录 = 一次区域转移，包含 POI 语义特征、路径距离、时间特征。

输出:
  prediction_dataset/各景区.csv   — 每个景区独立的训练/评估集
  prediction_dataset/all.csv       — 全部景区合并
"""

import json
import os
import sys
import argparse
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin

# ── 路径 ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = SCRIPT_DIR.parent
CLUSTER_DIR = PROJECT_DIR / "cluster" / "output"
POI_PROJECTED_JSON = SCRIPT_DIR / "data" / "projected" / "poi_path_projected.json"
OUTPUT_DIR = SCRIPT_DIR / "data" / "prediction_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 工具函数 ──────────────────────────────────────────────────
def haversine(lon1, lat1, lon2, lat2):
    """米"""
    R = 6371000
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))

TYPE_L1_NAMES = {
    "05": "餐饮", "08": "休闲", "10": "住宿",
    "11": "风景名胜", "14": "科教文化", "20": "公共设施",
}

# ── 核心 ──────────────────────────────────────────────────────
def load_clustered(scenery_name: str) -> pd.DataFrame:
    """加载聚类输出（含非停留点，为了准确算路径距离）"""
    csv_path = CLUSTER_DIR / f"{scenery_name}_clustered.csv"
    df = pd.read_csv(csv_path)
    df.columns = ["lon", "lat", "elev", "speed", "is_stop", "trackId", "time_sec", "region_id", "route_id"]
    return df


def load_poi_projection() -> dict:
    """加载 POI 投影数据，按 scenery+region_id 组织"""
    with open(POI_PROJECTED_JSON, encoding="utf-8") as f:
        pois = json.load(f)

    # 按 (scenery, region_id) 分组
    by_key = defaultdict(list)
    for p in pois:
        key = (p["scenery"], p["projected_region_id"])
        by_key[key].append(p)
    return dict(by_key)


def compute_region_poi_profile(poi_list: list) -> dict:
    """计算区域 POI 类型分布"""
    if not poi_list:
        return {}
    counter = Counter()
    for p in poi_list:
        tc = p.get("type_code", "")
        l1 = tc[:2]
        name = TYPE_L1_NAMES.get(l1, l1)
        counter[name] += 1
    total = sum(counter.values())
    return {k: round(v / total, 4) for k, v in counter.most_common()}


def extract_sequences(df: pd.DataFrame, poi_by_key: dict) -> pd.DataFrame:
    """提取路段序列，每条 = 一次区域转移"""
    records = []
    for tid, group in df.groupby("trackId", sort=False):
        group = group.sort_values("time_sec")
        regions = group["region_id"].values
        times = group["time_sec"].values
        lons = group["lon"].values
        lats = group["lat"].values
        is_stop = group["is_stop"].values

        # 提取首次出现序列（基于停留点）
        seq_r, start_idx = [], []
        prev = -1
        for i, r in enumerate(regions):
            if is_stop[i] and r != prev:
                seq_r.append(r)
                start_idx.append(i)
                prev = r

        if len(seq_r) < 2:
            continue

        # 取景區名稱
        scenery_name = df.attrs.get("scenery", group.attrs.get("scenery", ""))

        for k in range(len(seq_r) - 1):
            r_from = int(seq_r[k])
            r_to = int(seq_r[k + 1])
            i_from = start_idx[k]
            i_to = start_idx[k + 1]

            # 离开时间 = 进入 r_from 的时刻
            t_start = times[i_from]
            # 到达时间 = 首次进入 r_to 的时刻
            t_end = times[i_to]

            # 区域 POI 特征
            key_from = (scenery_name, r_from)
            key_to = (scenery_name, r_to)
            poi_from = poi_by_key.get(key_from, [])
            poi_to = poi_by_key.get(key_to, [])

            profile_from = compute_region_poi_profile(poi_from)
            profile_to = compute_region_poi_profile(poi_to)

            # 路径距离：沿轨迹相邻点累加
            path_dist = 0.0
            for j in range(i_from, i_to):
                path_dist += haversine(lons[j], lats[j], lons[j+1], lats[j+1])

            # 在 region_from 的停留时长
            stay_sec = times[i_from] - (times[start_idx[k - 1]] if k > 0 else times[0])

            # 在 r_from 区间内的停留点数
            stop_count = int(is_stop[i_from:i_to].sum())

            records.append({
                "scenery": scenery_name,
                "trackId": tid,
                "region_from": r_from,
                "region_to": r_to,
                "t_start_sec": t_start,
                "t_end_sec": t_end,
                "duration_sec": t_end - t_start,
                "dist_m": round(path_dist, 1),
                "stay_sec_in_from": stay_sec,
                "stop_count_in_from": stop_count,
                "lon_from": lons[i_from],
                "lat_from": lats[i_from],
                "lon_to": lons[i_to],
                "lat_to": lats[i_to],
                "profile_from": poi_profile_to_str(profile_from),
                "profile_to": poi_profile_to_str(profile_to),
            })
    return pd.DataFrame(records)


def poi_profile_to_str(profile: dict) -> str:
    """类型分布 → JSON 字符串"""
    return json.dumps(profile, ensure_ascii=False)


def poi_profile_to_vec(profile: dict, type_order: list) -> list:
    """类型分布 → 定长向量"""
    return [profile.get(t, 0.0) for t in type_order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenery", nargs="*", default=None,
                        help="景区列表（默认全部）")
    args = parser.parse_args()

    # 加载 POI 投影
    print("加载 POI 投影...")
    poi_by_key = load_poi_projection()

    # 发现可用景区
    all_files = sorted(CLUSTER_DIR.glob("*_clustered.csv"))
    available = [f.stem.replace("_clustered", "") for f in all_files]

    if args.scenery:
        targets = [s for s in args.scenery if s in available]
    else:
        targets = available

    if not targets:
        print(f"无可用景区。可用: {available}")
        sys.exit(1)

    print(f"目标景区: {targets}")

    # 收集所有类型，统一向量顺序
    all_types = set()
    for plist in poi_by_key.values():
        for p in plist:
            tc = p.get("type_code", "")
            l1 = tc[:2]
            name = TYPE_L1_NAMES.get(l1, l1)
            all_types.add(name)
    type_order = sorted(all_types)
    print(f"POI 类型顺序: {type_order}")

    all_records = []
    for sname in targets:
        print(f"\n{'-'*50}")
        print(f"[{sname}]")
        df = load_clustered(sname)
        df.attrs["scenery"] = sname

        poi_count = sum(1 for k in poi_by_key if k[0] == sname)
        print(f"  聚类点: {len(df)}, 涉及区域数: {df['region_id'].nunique()}, 区域有POI: {poi_count}")

        seq_df = extract_sequences(df, poi_by_key)

        if len(seq_df) == 0:
            print(f"  [SKIP] 无有效路段")
            continue

        # 展开 POI 向量
        for t in type_order:
            seq_df[f"from_{t}"] = seq_df["profile_from"].apply(
                lambda x: json.loads(x).get(t, 0.0))
            seq_df[f"to_{t}"] = seq_df["profile_to"].apply(
                lambda x: json.loads(x).get(t, 0.0))

        seq_df.drop(columns=["profile_from", "profile_to"], inplace=True)

        # 保存
        out = OUTPUT_DIR / f"{sname}.csv"
        seq_df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  路段数: {len(seq_df)}, 已保存: {out.name}")

        all_records.append(seq_df)

    # 合并全部
    if all_records:
        all_df = pd.concat(all_records, ignore_index=True)
        all_out = OUTPUT_DIR / "all.csv"
        all_df.to_csv(all_out, index=False, encoding="utf-8-sig")
        print(f"\n{'='*50}")
        print(f"总路段数: {len(all_df)}")
        print(f"全部合并: {all_out}")
        print(f"列: {list(all_df.columns)}")

        # 统计摘要
        print(f"\n各景区路段数:")
        print(all_df["scenery"].value_counts().to_string())
    else:
        print("无任何有效路段。")


if __name__ == "__main__":
    main()
