#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
段级地形特征集构建（跨场景 ETA 用，场景无关，严格 forecast 无泄漏）。

每景区 cleaned CSV → 段 = 按路径距离累积到 500m；
特征（场景无关，8 个）：距离、上下坡增益/损失、平均/最大坡度、平均海拔、起点海拔、时段；
标签：走完该段的时间(秒)。

无泄漏口径（P0-1）：
- 地形量（up/down/grade/elev_mean/start_ele）一律沿段内经纬度折线查 **SRTM DEM**，
  不使用本次轨迹的 GPX 海拔，也不使用段内未来时间戳；
- 删除 move_time_s、n_pts；
- 路线几何（段内经纬度折线）视为已知地图路线（论文需声明）。
另输出 Naismith 理论时长作为物理基线特征。

用法：
  python data-project/build_segment_features.py \
      --scenes 峨眉山 华山 黄山 泰山 \
      --indir data-project/cleaned_labeled_data \
      --outdir data-project/segment_features
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dem_lookup import DEM  # noqa: E402
from feature_groups import FEATURES  # noqa: E402

# Naismith：t_h = dist_km/5 + climb_m/600


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _fill_nan_1d(x):
    """沿索引线性插值填补 nan（前后边界用最近有效值）。全 nan 返回 None。"""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return None
    if ok.all():
        return x
    idx = np.arange(len(x))
    x = x.copy()
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def build_segments(df, dem, seg_len_m=500.0, veh_speed_kmh=15.0):
    """df: 单景区 cleaned CSV。固定距离分段（每 seg_len_m 一段）：
    标签=走完该段的时间(秒)；特征=DEM 地形（场景无关）。
    车载/缆车轨迹剔除：轨迹逐点速度中位 > veh_speed_kmh 的整条剔除（>15km/h 必非徒步）。"""
    segs = []
    n_veh = 0
    n_void = 0
    for tid, g in df.groupby("trackId"):
        g = g.sort_values("时间_秒")
        lat = g["纬度"].values.astype(np.float64)
        lon = g["经度"].values.astype(np.float64)
        t = g["时间_秒"].values.astype(np.float64)
        spd = g["速度(km/h)"].values.astype(np.float64)
        if len(lat) < 10:
            continue
        if veh_speed_kmh and float(np.median(spd[1:])) > veh_speed_kmh:
            n_veh += 1
            continue
        if t.max() > 1e10:
            t = t / 1000.0
        # DEM 高程（沿段内经纬度折线；非本次轨迹 GPX 海拔）
        ele = dem.elev(lat, lon)
        ele = _fill_nan_1d(ele)
        if ele is None:
            n_void += 1
            continue
        # 累积距离
        cum = np.zeros(len(lat))
        for i in range(1, len(lat)):
            cum[i] = cum[i - 1] + haversine_m(lat[i - 1], lon[i - 1], lat[i], lon[i])
        start = 0
        k = 0
        i = 1
        while i < len(cum):
            if cum[i] - cum[start] >= seg_len_m:
                sl = slice(start, i + 1)
                if i > start:
                    dist = cum[i] - cum[start]
                    de = np.diff(ele[sl])
                    up = float(de[de > 0].sum())
                    down = float(-de[de < 0].sum())
                    dt = np.diff(t[sl])
                    if dt.size and (dt.max() > 3600 or dt.min() <= 0):
                        # 段内出现 >1h 跳变或非正时间戳 = 时间戳损坏（与标签 dur 无关）
                        start = i
                        i += 1
                        continue
                    dur = float(t[i] - t[start])
                    if dur <= 0:
                        start = i
                        i += 1
                        continue
                    seg_lens = np.array([haversine_m(lat[j - 1], lon[j - 1], lat[j], lon[j])
                                         for j in range(start + 1, i + 1)])
                    grade = de / (np.maximum(seg_lens, 1e-6))
                    naismith_h = dist / 1000.0 / 5.0 + up / 600.0
                    segs.append({
                        "track": tid,
                        "seg_idx": k,
                        "dist_m": dist,
                        "up_m": up,
                        "down_m": down,
                        "mean_grade": float(np.mean(grade)),
                        "max_grade": float(np.max(np.abs(grade))),
                        "elev_mean": float(ele[sl].mean()),
                        "start_ele": float(ele[start]),
                        "tod_hour": float(t[start] % 86400) / 3600.0,
                        "dur_s": dur,
                        "naismith_s": naismith_h * 3600.0,
                    })
                    k += 1
                start = i
            i += 1
    out = pd.DataFrame(segs)
    out.attrs["n_veh"] = n_veh
    out.attrs["n_void"] = n_void
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=["峨眉山", "华山", "黄山", "泰山"])
    p.add_argument("--indir", default="data-project/cleaned_labeled_data")
    p.add_argument("--outdir", default="data-project/segment_features")
    p.add_argument("--suffix", default="_2bulu_cleaned.csv", help="cleaned CSV 后缀")
    p.add_argument("--dem-cache", default=None)
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dem = DEM(cache_dir=args.dem_cache)
    summary = {}
    for s in args.scenes:
        csvp = os.path.join(args.indir, f"{s}{args.suffix}")
        if not os.path.exists(csvp):
            print(f"[跳过] {s} 无 cleaned CSV: {csvp}")
            continue
        df = pd.read_csv(csvp)
        segs = build_segments(df, dem)
        if len(segs) == 0:
            print(f"[{s}] 无有效段")
            continue
        segs["scene"] = s
        # 列顺序：特征 + 标签/元数据
        cols = ["track", "seg_idx"] + FEATURES + ["dur_s", "naismith_s", "scene"]
        segs = segs[cols]
        outp = os.path.join(args.outdir, f"{s}_segments.csv")
        segs.to_csv(outp, index=False)
        summary[s] = {"tracks": int(segs["track"].nunique()), "segments": len(segs),
                      "median_dur_min": float(segs["dur_s"].median()) / 60.0}
        print(f"[{s}] 段数={len(segs)} 轨迹={segs['track'].nunique()} "
              f"剔除车载轨迹={segs.attrs.get('n_veh', 0)} DEM空洞轨迹={segs.attrs.get('n_void', 0)} "
              f"中位段时长={segs['dur_s'].median()/60:.1f}min "
              f"中位距离={segs['dist_m'].median()/1000:.2f}km "
              f"中位爬升={segs['up_m'].median():.0f}m", flush=True)
    print("汇总:", summary)


if __name__ == "__main__":
    main()
