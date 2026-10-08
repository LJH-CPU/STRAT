# -*- coding: utf-8 -*-
"""
统计补强（P0-2 修正版）：scene 级主分析（n=20），seed 仅作随机重复。

- 不再把 20 景区 × 5 seed 当 100 独立样本；
- 每景区先跨 seed 求均值，再做配对 Wilcoxon（n=20）；
- 同时报每景区 seed 的稳健性。

用法：python data-project/stat_analysis.py
输出：stdout（可选 --out 写 JSON）
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stat_scene_level import analyze  # noqa: E402

SEEDS = [42, 100, 2024, 7, 17]
TREES_20LO = "data-project/cross_scene_trees_20lo_s{seed}.json"


def load_20lo(pattern, seeds=SEEDS):
    tidy = []
    for s in seeds:
        p = pattern.format(seed=s)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        rows = d["rows"] if isinstance(d, dict) and "rows" in d else (
            d if isinstance(d, list) else [r for v in d.values() if isinstance(v, list) for r in v])
        for r in rows:
            if "mae_min" in r and "setting" in r:
                tidy.append({"scene": r["target"], "seed": r.get("seed", s),
                             "condition": r["setting"], "value": float(r["mae_min"])})
    return tidy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=TREES_20LO)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tidy = load_20lo(args.pattern)
    if not tidy:
        print(f"无数据：{args.pattern}")
        return
    out = {}
    print("=== 段级特征消融（scene 级 n=20，seed 平均）===")
    for a, b, label in [("zero-no-slope", "zero-shot", "去坡度 vs 零样本"),
                        ("zero-no-elev", "zero-shot", "去海拔 vs 零样本")]:
        res = analyze(tidy, a, b)
        out[a] = res
        if "wilcoxon_p" not in res:
            print(f"  {label}: 数据不足")
            continue
        print(f"  {label}: Δ={res['scene_mean_delta_min']:+.3f} min "
              f"95%CI={res['scene_ci95_min']} "
              f"{res['k_scenes_positive']}/{res['n_scenes']} 景区同向 "
              f"(seed 正号 {res['seed_positive_frac']:.2f}) "
              f"Wilcoxon p={res['wilcoxon_p']:.4g} rank_biserial={res['rank_biserial']}")
    print("\n注：主检验 n=20（景区），seed 仅稳健性；不再报告 100 对 p 值。")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
