# -*- coding: utf-8 -*-
"""
从 gpx 首点推导每景区 bbox（convert_2bulu 的区域过滤兜底）。

背景：爬取按关键词搜索，景区目录内混入大量同名异地轨迹（如华山目录中
仅 ~24% 是真华山）。纯分位数法会被离群点拉宽，纯峰值法可能指到污染峰
（华山第二峰是黄山）。因此用「人工锚点（景区主峰/中心公开坐标）+ 扩窗」：

  1. 采样每景区 gpx 的首点（轨迹起点，最能代表轨迹所属景区）
  2. 以锚点为中心，w 从 0.20° 逐步扩大至覆盖率 >= --cover 或 w = --cap
  3. bbox = [锚-w, 锚+w]（纬/经同宽），输出 scenes_20_bbox.json + 覆盖率报告

用法：python data-project/derive_bbox_20.py [--sample 300]
"""
import argparse
import glob
import json
import os
import random
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenes_20 import scenes, BBOX_FILE

GPX_NS = "{http://www.topografix.com/GPX/1/1}"

# 景区主峰/中心参考坐标（公开常识，粗粒度即可，扩窗算法自适应）
ANCHOR = {
    "长白山": (42.03, 128.06),
    "丹霞山": (25.03, 113.74),
    "峨眉山": (29.52, 103.33),
    "梵净山": (27.92, 108.69),
    "恒山":   (39.67, 113.73),
    "衡山":   (27.25, 112.69),
    "华山":   (34.48, 110.08),
    "黄山":   (30.13, 118.17),
    "九华山": (30.48, 117.80),
    "庐山":   (29.56, 115.98),
    "青城山": (30.90, 103.57),
    "三清山": (28.90, 118.06),
    "嵩山":   (34.49, 113.02),
    "太白山": (33.95, 107.78),
    "泰山":   (36.25, 117.10),
    "五台山": (39.02, 113.57),
    "武当山": (32.40, 111.00),
    "武夷山": (27.72, 117.99),
    "雁荡山": (28.37, 121.06),
    "张家界": (29.32, 110.44),
}


def first_points(path, max_pts=3):
    pts = []
    try:
        for _, el in ET.iterparse(path, events=("end",)):
            if el.tag == GPX_NS + "trkpt":
                pts.append((float(el.get("lat")), float(el.get("lon"))))
                el.clear()
                if len(pts) >= max_pts:
                    break
    except ET.ParseError:
        pass
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300, help="每景区抽样 gpx 数")
    ap.add_argument("--cover", type=float, default=0.85, help="目标覆盖率")
    ap.add_argument("--cap", type=float, default=0.35, help="最大窗口半径（度）")
    args = ap.parse_args()

    report = {}
    out = {}
    for s in scenes():
        name = s["name"]
        if name not in ANCHOR:
            print(f"[{name}] 无锚点，跳过")
            continue
        files = sorted(glob.glob(os.path.join(s["gpx_dir"], "*.gpx")))
        if not files:
            print(f"[{name}] 无 gpx，跳过")
            continue
        rng = random.Random(42)
        picks = rng.sample(files, min(args.sample, len(files)))
        pts = []
        for f in picks:
            pts.extend(first_points(f))
        if not pts:
            print(f"[{name}] 无可解析点，跳过")
            continue
        ala, alo = ANCHOR[name]
        w, cov = args.cap, 0.0
        for w_try in (0.20, 0.25, 0.30, 0.35):
            cov = sum(1 for la, lo in pts
                      if abs(la - ala) <= w_try and abs(lo - alo) <= w_try) / len(pts)
            w = w_try
            if cov >= args.cover:
                break
        bbox = [round(ala - w, 4), round(ala + w, 4), round(alo - w, 4), round(alo + w, 4)]
        out[name] = bbox
        report[name] = {"n_gpx": len(files), "sampled": len(picks), "pts": len(pts),
                        "anchor": [ala, alo], "half_width_deg": w, "coverage": round(cov, 3),
                        "bbox": bbox}
        flag = "OK " if cov >= args.cover else "LOW"
        print(f"[{flag}] {name}: cov={cov:.0%} w={w} bbox={bbox}", flush=True)

    with open(BBOX_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(BBOX_FILE.replace(".json", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("saved:", BBOX_FILE)


if __name__ == "__main__":
    main()
