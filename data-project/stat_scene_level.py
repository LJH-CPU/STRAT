#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scene 级统计（P0-2）：以景区为独立单位，seed 仅作随机重复。

主分析：每景区先跨 seed 求均值 Δ̄_i，再对 n=20 做配对 Wilcoxon。
稳健性：报告每景区 seed 的 mean±sd、k/20 景区均值同向、正号 scene-seed 比例（描述性）。

输入：tidy 结果，支持
  --adapter bench   : segment_benchmark.py 的 {rows:[{target,seed,model,setting,group,mae_min,...}]}
  --adapter tidy    : JSON/CSV，列 {scene,seed,condition,value}

用法：
  python data-project/stat_scene_level.py --input data-project/bench_segment.json \
      --adapter bench --model xgb --setting zero-shot --condition-col group \
      --a no_slope --b full --out data-project/stat_scene_level.json
"""
import argparse
import json
import os

import numpy as np
from scipy import stats


def load_bench(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict) and "rows" in d:
        return d["rows"]
    if isinstance(d, list):
        return d
    # legacy: {"seed42": [rows], ...}
    rows = []
    for v in d.values():
        if isinstance(v, list):
            rows.extend(v)
    return rows


def to_tidy_bench(rows, model=None, setting=None, condition_col="group", value_col="mae_min"):
    tidy = []
    for r in rows:
        if model and r.get("model") != model:
            continue
        if setting and r.get("setting") != setting:
            continue
        if value_col not in r:
            continue
        tidy.append({"scene": r["target"], "seed": r["seed"],
                     "condition": r.get(condition_col), "value": float(r[value_col])})
    return tidy


def load_tidy(path):
    if path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(path)
        return df.to_dict("records")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def rank_biserial(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def ci95(vals):
    v = np.asarray(vals, float)
    se = v.std(ddof=1) / np.sqrt(len(v))
    t = stats.t.ppf(0.975, len(v) - 1)
    return float(v.mean() - t * se), float(v.mean() + t * se)


def analyze(tidy, a, b):
    # {scene: {seed: {cond: value}}}
    per = {}
    for r in tidy:
        per.setdefault(r["scene"], {}).setdefault(r["seed"], {})[r["condition"]] = r["value"]
    scenes = sorted(per)
    scene_delta, seed_delta, used = [], [], []
    for s in scenes:
        d_seeds = [v[a] - v[b] for v in per[s].values() if a in v and b in v]
        if not d_seeds:
            continue
        used.append(s)
        scene_delta.append(float(np.mean(d_seeds)))
        seed_delta.extend(d_seeds)
    scene_delta = np.array(scene_delta)
    seed_delta = np.array(seed_delta)
    out = {"comparison": f"{a} - {b}", "n_scenes": len(used), "scenes": used,
           "scene_mean_delta_min": round(float(scene_delta.mean()), 4),
           "scene_ci95_min": [round(x, 4) for x in ci95(scene_delta)],
           "k_scenes_positive": int((scene_delta > 0).sum()),
           "seed_positive_frac": round(float((seed_delta > 0).mean()), 4)}
    if len(scene_delta) >= 3 and not np.allclose(scene_delta, 0):
        w = stats.wilcoxon(scene_delta)
        out.update({"wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
                    "rank_biserial": round(rank_biserial(scene_delta), 4),
                    "cohen_dz": round(float(scene_delta.mean() / (scene_delta.std(ddof=1) + 1e-12)), 4)})
    # 每景区 seed 稳健性
    out["per_scene_seed_sd"] = {}
    for s in used:
        ds = [v[a] - v[b] for v in per[s].values() if a in v and b in v]
        out["per_scene_seed_sd"][s] = round(float(np.std(ds, ddof=1)), 4) if len(ds) > 1 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--adapter", default="bench", choices=["bench", "tidy"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--setting", default=None)
    ap.add_argument("--condition-col", default="group")
    ap.add_argument("--value-col", default="mae_min")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.adapter == "bench":
        tidy = to_tidy_bench(load_bench(args.input), args.model, args.setting,
                             args.condition_col, args.value_col)
    else:
        tidy = load_tidy(args.input)
    res = analyze(tidy, args.a, args.b)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
