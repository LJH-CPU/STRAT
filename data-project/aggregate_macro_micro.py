#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Macro vs weighted 聚合（#9）。

- macro   : 每景区等权（先跨 seed 求均值，再对景区求均值）
- weighted: 按 n_seg 或 n_track 加权

输入支持 segment_benchmark.py 的 {rows:[...]} 或 legacy {seedN:[...]}。
用法：
  python data-project/aggregate_macro_micro.py --input data-project/bench_segment.json \
      --model xgb --setting zero-shot --condition-col group --weight-col n_seg
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict) and "rows" in d:
        return d["rows"]
    if isinstance(d, list):
        return d
    rows = []
    for v in d.values():
        if isinstance(v, list):
            rows.extend(v)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--setting", default=None)
    ap.add_argument("--condition-col", default="group")
    ap.add_argument("--value-col", default="mae_min")
    ap.add_argument("--weight-col", default="n_seg", choices=["n_seg", "n_track"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(args.input)
    # {cond: {scene: {seed: (value, weight)}}}
    per = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if args.model and r.get("model") != args.model:
            continue
        if args.setting and r.get("setting") != args.setting:
            continue
        if args.value_col not in r or args.condition_col not in r:
            continue
        w = r.get(args.weight_col, 1)
        per[r[args.condition_col]][r["target"]][r["seed"]] = (float(r[args.value_col]), float(w))

    print(f"macro vs weighted  model={args.model} setting={args.setting} weight={args.weight_col}")
    print(f"{'condition':16s} {'macro':>8s} {'weighted':>9s} {'n_scenes':>8s}")
    out = {}
    for cond in sorted(per):
        scene_means, scene_w = [], []
        for s, seeds in per[cond].items():
            vals = [v for v, _ in seeds.values()]
            ws = [w for _, w in seeds.values()]
            scene_means.append(float(np.mean(vals)))
            scene_w.append(float(np.mean(ws)))
        macro = float(np.mean(scene_means))
        weighted = float(np.sum(np.array(scene_means) * np.array(scene_w)) / (np.sum(scene_w) + 1e-12))
        out[cond] = {"macro": round(macro, 4), "weighted": round(weighted, 4),
                     "n_scenes": len(scene_means)}
        print(f"{cond:16s} {macro:8.3f} {weighted:9.3f} {len(scene_means):8d}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
