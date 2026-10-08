#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
2bulu GPX → 我们管线 cleaned CSV 的稳定转换器。

- 输入：data/gpx/*.gpx（含 <time> 的；JSON createTime 全 0 不可用）
- 输出：与 data-project/cleaned_labeled_data/*_cleaned.csv 同 schema：
  经度,纬度,海拔,此刻时间,时间_秒,速度(km/h),trackId,is_stop,region_id,reviewed,route_id,route_name
- 过滤：bbox / 最短点数 / 最短里程 / 剔 >N 天长线 / 近重复去重
- 幂等：trackId=GPX 文件名（稳定），新文件加入后重跑即可追加覆盖。

用法：
  python prediction/convert_2bulu.py --gpx-dir data/gpx --meta data/tracks_metadata.csv \
      --out data-project/cleaned_labeled_data/峨眉山_2bulu_cleaned.csv \
      --bbox "29.3,29.7,103.2,103.5" --min-pts 50 --min-km 5 --max-days 3
"""
import argparse
import datetime
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

GPX_NS = {"gpx": "http://www.topografix.com/GPX/1/1"}


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_gpx_points(path):
    """返回 [(lat, lon, ele, epoch_ms), ...]，无 <time> 的点跳过。"""
    tree = ET.parse(path)
    root = tree.getroot()
    pts = []
    for tp in root.findall(".//gpx:trkpt", GPX_NS):
        lat = float(tp.get("lat"))
        lon = float(tp.get("lon"))
        ele_el = tp.find("gpx:ele", GPX_NS)
        ele = float(ele_el.text) if (ele_el is not None and ele_el.text) else 0.0
        t_el = tp.find("gpx:time", GPX_NS)
        if t_el is None or not (t_el.text or "").strip():
            continue
        t = datetime.datetime.fromisoformat(t_el.text.strip().replace("Z", "+00:00"))
        ms = int(t.timestamp() * 1000)
        pts.append((lat, lon, ele, ms))
    return pts


def speed_kmh(prev, cur):
    d = haversine_m(prev[0], prev[1], cur[0], cur[1])
    dt = (cur[3] - prev[3]) / 1000.0
    if dt <= 0:
        return 0.0
    return d / dt * 3.6


def detect_stays(speeds, times_ms, speed_thresh=1.0, duration_thresh_s=30):
    """与 clean_scenery_pipeline.detect_stays 同口径；times_ms 毫秒 → duration 按秒。"""
    n = len(speeds)
    is_stop = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        if speeds[i] <= speed_thresh:
            j = i + 1
            while j < n and speeds[j] <= speed_thresh:
                j += 1
            duration = (times_ms[j - 1] - times_ms[i]) / 1000.0
            if duration >= duration_thresh_s and (j - i) >= 2:
                is_stop[i:j] = 1
            i = j
        else:
            i += 1
    return is_stop


def load_meta(path):
    meta = {}
    if path and os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"track_no": str})
        for _, r in df.iterrows():
            meta[str(r["track_no"])] = dict(r)
    return meta


def process_track(path, bbox, min_pts, min_km, max_days, meta):
    stem = os.path.splitext(os.path.basename(path))[0]
    pts = parse_gpx_points(path)
    if len(pts) < min_pts:
        return None, f"<{min_pts}点"
    # bbox 过滤（点级），再查点数
    lat_min, lat_max, lon_min, lon_max = bbox
    pts = [p for p in pts if lat_min <= p[0] <= lat_max and lon_min <= p[1] <= lon_max]
    if len(pts) < min_pts:
        return None, "bbox后点不足"
    pts.sort(key=lambda p: p[3])
    # 单段间隔 > max_days 剔除（时间戳损坏；不依赖总时长/标签）
    gaps_h = np.diff([p[3] for p in pts]) / 1000.0 / 3600.0
    if max_days and gaps_h.size and gaps_h.max() > max_days * 24:
        return None, f"单段间隔>{max_days}天"
    # 里程（轨迹总长）
    length = sum(haversine_m(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                 for i in range(len(pts) - 1))
    if min_km and length / 1000.0 < min_km:
        return None, f"<{min_km}km"
    speeds = [0.0] + [speed_kmh(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    times = np.array([p[3] for p in pts], dtype=np.float64)
    is_stop = detect_stays(np.array(speeds, dtype=np.float64), times)
    m = meta.get(stem, {})
    rows = pd.DataFrame({
        "经度": [p[1] for p in pts],
        "纬度": [p[0] for p in pts],
        "海拔": [p[2] for p in pts],
        "此刻时间": [int(p[3]) for p in pts],
        "时间_秒": times,
        "速度(km/h)": speeds,
        "trackId": stem,
        "is_stop": is_stop,
        "region_id": -1,
        "reviewed": True,
        "route_id": 0,
        "route_name": "",
    })
    rows["_author"] = m.get("author", "")
    rows["_length_km"] = length / 1000.0
    return rows, None


def main():
    p = argparse.ArgumentParser(description="2bulu GPX → cleaned CSV")
    p.add_argument("--gpx-dir", default="data/gpx")
    p.add_argument("--meta", default="data/tracks_metadata.csv")
    p.add_argument("--out", default="data-project/cleaned_labeled_data/峨眉山_2bulu_cleaned.csv")
    p.add_argument("--bbox", default="29.3,29.7,103.2,103.5", help="latmin,latmax,lonmin,lonmax")
    p.add_argument("--min-pts", type=int, default=50)
    p.add_argument("--min-km", type=float, default=5.0)
    p.add_argument("--max-days", type=float, default=3.0)
    p.add_argument("--dedup", action="store_true", default=True, help="按 起点+里程 近重复去重")
    p.add_argument("--no-dedup", action="store_true")
    args = p.parse_args()
    if args.no_dedup:
        args.dedup = False
    bbox = tuple(float(x) for x in args.bbox.split(","))
    meta = load_meta(args.meta)
    files = sorted(glob.glob(os.path.join(args.gpx_dir, "*.gpx")))
    print(f"GPX 文件数: {len(files)}")
    parts = []
    used = rejected = 0
    reasons = {}
    for f in files:
        rows, why = process_track(f, bbox, args.min_pts, args.min_km, args.max_days, meta)
        if rows is None:
            rejected += 1
            reasons[why] = reasons.get(why, 0) + 1
            continue
        used += 1
        parts.append(rows)
    if not parts:
        print("无可用轨迹，退出。")
        return
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["trackId", "时间_秒"]).reset_index(drop=True)
    print(f"可用轨迹: {used}  拒绝: {rejected}")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  拒绝[{k}]: {v}")
    # 去重：起点(0.001°≈100m)+里程(0.1km) 签名，每组保留里程最长
    if args.dedup:
        sigs = {}
        for tid, g in df.groupby("trackId"):
            sig = (round(g["纬度"].iloc[0], 3), round(g["经度"].iloc[0], 3),
                   round(g["_length_km"].iloc[0], 1))
            sigs.setdefault(sig, []).append((g["_length_km"].iloc[0], tid))
        keep = set()
        for sig, lst in sigs.items():
            lst.sort(reverse=True)
            keep.add(lst[0][1])
        before = df["trackId"].nunique()
        df = df[df["trackId"].isin(keep)].copy()
        print(f"去重后轨迹: {df['trackId'].nunique()} (去重前 {before})")
    df = df.drop(columns=["_author", "_length_km"])
    # 保持与现有 cleaned CSV 完全一致的列顺序
    df = df[["经度", "纬度", "海拔", "此刻时间", "时间_秒", "速度(km/h)",
             "trackId", "is_stop", "region_id", "reviewed", "route_id", "route_name"]]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8")
    n_tracks = df["trackId"].nunique()
    n_pts = len(df)
    print(f"写入: {args.out}")
    print(f"轨迹数: {n_tracks}  点数: {n_pts}  停留点: {int((df['is_stop']==1).sum())}")
    # 输出作者统计（个性化用）
    auth = meta
    # 简单统计长度分布
    print(f"每轨点数: min={df.groupby('trackId').size().min()} "
          f"med={int(df.groupby('trackId').size().median())} max={df.groupby('trackId').size().max()}")


if __name__ == "__main__":
    main()
