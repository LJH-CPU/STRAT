#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drift 量化（#4）：per-target SMD + MMD，及与 few-shot gain 的相关（Spearman + bootstrap CI + 偏相关）。

严格无 transductive（H2）：
  source-train → fit scaler → transform(source-train), transform(target-test) → SMD / MMD
target-test 统计量绝不进入 fit。

用法：
  python data-project/drift_analysis.py --scenes <20> --seed 42 \
      --seg-dir data-project/segment_features \
      --bench data-project/bench_segment.json --out data-project/drift.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, cdist, squareform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_groups import FEATURES  # noqa: E402
from scenes_20 import NAMES_20  # noqa: E402


def split_scene(df, seed, train_ratio=0.8):
    tracks = df["track"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(tracks)
    n_tr = int(len(tracks) * train_ratio)
    tr, te = set(tracks[:n_tr]), set(tracks[n_tr:])
    return df[df["track"].isin(tr)], df[df["track"].isin(te)]


def mmd_rbf(X, Y, seed, n=2000):
    """RBF-kernel MMD^2，median-heuristic 带宽，等量无放回子采样。"""
    rng = np.random.RandomState(seed)
    nx = min(len(X), n)
    ny = min(len(Y), n)
    Xs = X[rng.choice(len(X), nx, replace=False)]
    Ys = Y[rng.choice(len(Y), ny, replace=False)]
    # 带宽只用 source-train 估计（H2：target-test 不得进入任何 fit）
    d2 = pdist(Xs.astype(np.float64), "sqeuclidean")
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    gamma = 1.0 / max(med, 1e-12)

    def K(A, B):
        return np.exp(-gamma * cdist(A, B, "sqeuclidean"))

    return float(K(Xs, Xs).mean() + K(Ys, Ys).mean() - 2 * K(Xs, Ys).mean())


def spearman_ci(x, y, seed=42, B=2000):
    rho, p = stats.spearmanr(x, y)
    rng = np.random.RandomState(seed)
    bs = []
    n = len(x)
    for _ in range(B):
        idx = rng.randint(0, n, n)
        if len(set(idx)) < 3:
            continue
        r = stats.spearmanr(np.asarray(x)[idx], np.asarray(y)[idx]).correlation
        if np.isfinite(r):
            bs.append(r)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    return float(rho), float(p), float(lo), float(hi)


def partial_spearman(x, y, z):
    """控制 z 后 x,y 的偏 Spearman（秩残差法）。"""
    rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
    A = np.column_stack([np.ones_like(rz), rz])
    bx = np.linalg.lstsq(A, rx, rcond=None)[0]
    by = np.linalg.lstsq(A, ry, rcond=None)[0]
    ex = rx - A @ bx
    ey = ry - A @ by
    return float(stats.pearsonr(ex, ey)[0])


def load_bench_gain(path, model="xgb"):
    """G_i = MAE_0% - MAE_25%（scene 级，跨 seed 均值）。"""
    d = json.load(open(path, encoding="utf-8"))
    rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
    zs, fs = defaultdict(list), defaultdict(list)
    for r in rows:
        if r.get("model") != model or r.get("group") != "full":
            continue
        if r.get("setting") == "zero-shot":
            zs[r["target"]].append(r["mae_min"])
        if r.get("setting") == "few-shot-agg" and abs(float(r.get("ratio", -1)) - 0.25) < 1e-9:
            fs[r["target"]].append(r["mae_mean"])
    gain, mae0 = {}, {}
    for s in zs:
        if s in fs:
            gain[s] = float(np.mean(zs[s]) - np.mean(fs[s]))
            mae0[s] = float(np.mean(zs[s]))
    return gain, mae0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=NAMES_20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seg-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "segment_features"))
    ap.add_argument("--mmd-n", type=int, default=2000)
    ap.add_argument("--bench", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = {}
    for s in args.scenes:
        p = os.path.join(args.seg_dir, f"{s}_segments.csv")
        if os.path.exists(p):
            data[s] = pd.read_csv(p)
    res = {}
    for target in args.scenes:
        if target not in data:
            continue
        te_tr, te_te = split_scene(data[target], args.seed)
        src = pd.concat([split_scene(data[s], args.seed)[0] for s in data if s != target],
                        ignore_index=True)
        mu = src[FEATURES].mean(0).values
        sd = src[FEATURES].std(0).values + 1e-9
        Xs = ((src[FEATURES].values - mu) / sd).astype(np.float64)
        Xt = ((te_te[FEATURES].values - mu) / sd).astype(np.float64)
        smd = float(np.mean(np.abs(Xs.mean(0) - Xt.mean(0))))  # source std=1 after scaling
        mmd = mmd_rbf(Xs, Xt, args.seed, n=args.mmd_n)
        res[target] = {"smd": round(smd, 4), "mmd": round(mmd, 6),
                       "n_src": int(len(Xs)), "n_tgt": int(len(Xt))}
        print(f"[{target}] SMD={smd:.3f} MMD={mmd:.5f}", flush=True)

    out = {"seed": args.seed, "per_target": res}
    if args.bench and os.path.exists(args.bench):
        gain, mae0 = load_bench_gain(args.bench)
        common = [s for s in res if s in gain]
        if len(common) >= 3:
            smd = [res[s]["smd"] for s in common]
            mmd = [res[s]["mmd"] for s in common]
            g = [gain[s] for s in common]
            m0 = [mae0[s] for s in common]
            r_smd = spearman_ci(smd, g, args.seed)
            r_mmd = spearman_ci(mmd, g, args.seed)
            out["corr"] = {
                "n": len(common),
                "SMD_vs_gain": {"rho": r_smd[0], "p": r_smd[1], "ci95": [r_smd[2], r_smd[3]]},
                "MMD_vs_gain": {"rho": r_mmd[0], "p": r_mmd[1], "ci95": [r_mmd[2], r_mmd[3]]},
                "SMD_vs_gain_partial_MAE0": round(partial_spearman(smd, g, m0), 4),
                "MMD_vs_gain_partial_MAE0": round(partial_spearman(mmd, g, m0), 4),
            }
            print("corr:", json.dumps(out["corr"], ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
