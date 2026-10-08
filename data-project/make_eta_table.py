#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P2：从 result_segment.json 自动生成 few-shot / matched-size 表与图（Figure 4）。

口径（全部来自同一 JSON，禁止手填）：
  zero    = XGB zero-shot full
  fs{r}   = few-shot-agg ratio r
  matched-A / matched-B
  gain = MAE_zero - MAE_condition

用法：
  python data-project/make_eta_table.py \
      --input final_results/result_segment.json \
      --out-csv final_results/eta_table.csv \
      --out-fig final_results/figure4_fewshot_matched.png
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


def load_rows(path):
    d = json.load(open(path, encoding="utf-8"))
    return d["rows"] if isinstance(d, dict) and "rows" in d else d


def scene_mean(rows, model, setting, group="full", ratio=None, value="mae_min"):
    per = defaultdict(list)
    for r in rows:
        if r.get("model") != model or r.get("setting") != setting:
            continue
        if group is not None and r.get("group") != group:
            continue
        if ratio is not None and abs(float(r.get("ratio", -1)) - ratio) > 1e-9:
            continue
        if value in r:
            per[r["target"]].append(r[value])
    if not per:
        return None
    return float(np.mean([np.mean(v) for v in per.values()])), len(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="final_results/result_segment.json")
    ap.add_argument("--model", default="xgb")
    ap.add_argument("--out-csv", default="final_results/eta_table.csv")
    ap.add_argument("--out-fig", default="final_results/figure4_fewshot_matched.png")
    args = ap.parse_args()
    rows = load_rows(args.input)

    zero, n = scene_mean(rows, args.model, "zero-shot")
    conds = [("zero-shot", "zero-shot", None)]
    for r in [0.05, 0.10, 0.25]:
        conds.append((f"few-shot {int(r*100)}%", "few-shot-agg", r))
    conds.append(("matched-A (extra source)", "matched-A", None))
    conds.append(("matched-B (size-matched)", "matched-B", None))

    table = []
    for label, setting, ratio in conds:
        val = zero if setting == "zero-shot" else scene_mean(
            rows, args.model, setting, ratio=ratio, value="mae_mean" if setting == "few-shot-agg" else "mae_min")[0]
        table.append({"condition": label, "mae_min": round(val, 3),
                      "gain_vs_zero_min": round(zero - val, 3)})

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "mae_min", "gain_vs_zero_min"])
        w.writeheader()
        w.writerows(table)
    print(f"zero-shot MAE = {zero:.3f} (n_scenes={n})")
    for row in table:
        print(f"  {row['condition']:28s} MAE={row['mae_min']:.3f}  gain={row['gain_vs_zero_min']:+.3f}")
    print("saved", args.out_csv)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [t["condition"] for t in table]
        vals = [t["mae_min"] for t in table]
        colors = ["#888888", "#2E86AB", "#2E86AB", "#2E86AB", "#D62828", "#2E86AB"]
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(range(len(vals)), vals, color=colors[:len(vals)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Zero-shot MAE (min / 500 m)")
        ax.set_title("Few-shot vs matched-size controls", loc="left")
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_fig, dpi=200)
        print("saved", args.out_fig)
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
