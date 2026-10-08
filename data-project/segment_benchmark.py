#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一段级 benchmark（Phase 8 H5）：一次加载段 CSV，输出 #3/#5/#6/#9/#10/#11 全部指标。

- 特征集唯一真源：data-project/feature_groups.py（H1）
- LOSO：每 target 用其余场景池化作源；track 级 80/20
- 模型阶梯（#11）：Naismith / Ridge / RF / ExtraTrees / kNN / XGB / LightGBM
- 特征组消融（#3）：核心 8 组 × 5 seed，补充 4 组 × 1 seed
- few-shot（#5）：R=5 严格 track-level 独立子抽样（model seed 与 subset seed 解耦，H3）
- matched-size 对照（#6）：A（等量额外源）与 B（总量匹配），分开报告（H4）
- macro vs weighted（#9）：保存每 target 的 mae 与 n_seg/n_track，供后处理
- TOD（#10）：XGB full 零样本的分时段 MAE

用法：
  python data-project/segment_benchmark.py --scenes <20> --seeds 42 100 2024 7 17 \
      --settings ladder ablation fewshot matched --out data-project/bench_segment.json
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_groups import (  # noqa: E402
    FEATURES, FEATURE_GROUPS, CORE_GROUPS, SUPPLEMENT_GROUPS, CORE_SEEDS, SUPPLEMENT_SEEDS,
)
from scenes_20 import NAMES_20  # noqa: E402

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    _HAS_SK = True
except Exception:
    _HAS_SK = False

SEG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segment_features")
TOD_BINS = [0, 6, 9, 12, 15, 18, 24]  # night/morning/midday/afternoon/evening/night
N_JOBS = 8  # 由 --n-jobs 覆盖（多进程分片时调小避免超订）
DEVICE = "cuda"  # 由 --device 覆盖（"cuda" 或 "cpu"）


# ── 数据 ────────────────────────────────────────────────────
def load_scenes(scenes, seg_dir=SEG_DIR):
    out = {}
    for s in scenes:
        p = os.path.join(seg_dir, f"{s}_segments.csv")
        if os.path.exists(p):
            out[s] = pd.read_csv(p)
        else:
            print(f"[warn] missing {p}", flush=True)
    return out


def split_scene(df, seed, train_ratio=0.8):
    tracks = df["track"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(tracks)
    n_tr = int(len(tracks) * train_ratio)
    tr, te = set(tracks[:n_tr]), set(tracks[n_tr:])
    return df[df["track"].isin(tr)], df[df["track"].isin(te)]


def _xy(df, cols):
    return np.asarray(df[cols].values, dtype=np.float32), np.asarray(df["dur_s"].values, dtype=np.float64)


def _gtrack(df):
    """全局 track 键（场景前缀，避免跨场景 trackId 碰撞）。"""
    return (df["scene"].astype(str) + "|" + df["track"].astype(str)).values


def mae_min(pred, true):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(true, float)))) / 60.0


def tod_bin(h):
    return int(np.digitize(h, TOD_BINS[1:-1]))


# ── 模型 ────────────────────────────────────────────────────
def _val_idx(n, seed, tracks=None, ratio=0.15):
    """验证索引：tracks 提供时按 track 级留出，否则行级随机。
    track 键先 factorize 成整数（字符串 np.isin 在 ~70 万行上慢 ~1800x）。"""
    if tracks is not None and len(tracks) == n:
        codes = pd.factorize(np.asarray(tracks))[0]
        uniq = np.unique(codes)
        rng = np.random.RandomState(seed); rng.shuffle(uniq)
        nv = max(int(len(uniq) * ratio), 1)
        va = uniq[:nv]
        return np.where(np.isin(codes, va))[0]
    n_va = max(int(n * ratio), 20)
    return np.random.RandomState(seed).choice(n, n_va, replace=False)


def train_xgb(X, y, seed, tracks=None):
    X = np.asarray(X, np.float32); y = np.asarray(y, np.float64)
    vi = _val_idx(len(X), seed, tracks)
    m = np.ones(len(X), bool); m[vi] = False
    model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             objective="reg:absoluteerror", tree_method="hist",
                             device=DEVICE,
                             random_state=seed, n_jobs=N_JOBS, early_stopping_rounds=30)
    model.fit(X[m], y[m], eval_set=[(X[vi], y[vi])], verbose=False)
    return model


def train_lgb(X, y, seed, tracks=None):
    X = np.asarray(X, np.float32); y = np.asarray(y, np.float64)
    vi = _val_idx(len(X), seed, tracks)
    m = np.ones(len(X), bool); m[vi] = False
    dtr = lgb.Dataset(X[m], y[m]); dva = lgb.Dataset(X[vi], y[vi], reference=dtr)
    model = lgb.train({"objective": "mae", "learning_rate": 0.05, "n_estimators": 400,
                       "max_depth": 6, "num_leaves": 63, "feature_fraction": 0.8,
                       "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                       "seed": seed, "num_threads": N_JOBS, "verbosity": -1},
                      dtr, num_boost_round=400, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def train_sklearn(kind, X, y, seed):
    X = np.asarray(X, np.float32); y = np.asarray(y, np.float64)
    if kind == "ridge":
        m = Ridge(alpha=1.0)
    elif kind == "rf":
        m = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=N_JOBS, random_state=seed)
    elif kind == "et":
        m = ExtraTreesRegressor(n_estimators=200, max_depth=None, n_jobs=N_JOBS, random_state=seed)
    elif kind == "knn":
        m = KNeighborsRegressor(n_neighbors=5, n_jobs=N_JOBS)
    else:
        raise ValueError(kind)
    m.fit(X, y)
    return m


def train_model(kind, X, y, seed, tracks=None):
    if kind == "xgb":
        return train_xgb(X, y, seed, tracks)
    if kind == "lgb":
        return train_lgb(X, y, seed, tracks)
    return train_sklearn(kind, X, y, seed)


def predict(model, X):
    return np.maximum(np.asarray(model.predict(np.asarray(X, np.float32)), float), 0)


# ── few-shot / matched-size 子抽样 ──────────────────────────
def subsample_target_tracks(tr_df, ratio, subset_seed):
    """严格 track-level：同 track 全部段整体进入/排除。"""
    tracks = np.array(tr_df["track"].unique())
    rng = np.random.RandomState(subset_seed)
    rng.shuffle(tracks)
    n = max(int(len(tracks) * ratio), 1)
    keep = set(tracks[:n])
    return tr_df[tr_df["track"].isin(keep)]


def sample_source_segments(src_df, n, subset_seed):
    """从源池等量抽样 n 个段（H4 matched-size 用）。"""
    if n >= len(src_df):
        return src_df
    return src_df.sample(n=n, random_state=subset_seed)


# ── 单 target×seed 评估 ─────────────────────────────────────
def eval_target(target, seed, scenes, args, rows):
    te_tr, te_te = split_scene(scenes[target], seed)
    y_te = te_te["dur_s"].values
    n_track_te = te_te["track"].nunique()
    src = pd.concat([split_scene(scenes[s], seed)[0] for s in scenes if s != target],
                    ignore_index=True)

    def rec(model_name, setting, group, pred, y=y_te, extra=None):
        r = {"target": target, "seed": seed, "model": model_name, "setting": setting,
             "group": group, "mae_min": round(mae_min(pred, y), 3),
             "n_seg": int(len(y)), "n_track": int(n_track_te)}
        if extra:
            r.update(extra)
        rows.append(r)

    # ── #11 baseline ladder（full 特征）──
    if "ladder" in args.settings:
        _t = time.time()
        rec("Naismith", "physics", "full", te_te["naismith_s"].values)
        Xtr_self, ytr_self = _xy(te_tr, FEATURES)
        Xte, _ = _xy(te_te, FEATURES)
        Xsrc, ysrc = _xy(src, FEATURES)
        for kind in ["ridge", "rf", "et", "knn", "xgb", "lgb"]:
            if kind == "lgb" and not _HAS_LGB:
                continue
            if kind in ("ridge", "rf", "et", "knn") and not _HAS_SK:
                continue
            if kind in ("ridge", "knn"):
                mu, sd = Xsrc.mean(0), Xsrc.std(0) + 1e-6
                m = train_model(kind, (Xsrc - mu) / sd, ysrc, seed)
                pz = predict(m, (Xte - mu) / sd)
                mu2, sd2 = Xtr_self.mean(0), Xtr_self.std(0) + 1e-6
                m2 = train_model(kind, (Xtr_self - mu2) / sd2, ytr_self, seed)
                ps = predict(m2, (Xte - mu2) / sd2)
            else:
                m = train_model(kind, Xsrc, ysrc, seed, _gtrack(src))
                pz = predict(m, Xte)
                m2 = train_model(kind, Xtr_self, ytr_self, seed, _gtrack(te_tr))
                ps = predict(m2, Xte)
            rec(kind, "zero-shot", "full", pz)
            rec(kind, "self-train", "full", ps)
            if kind == "xgb":
                tb = defaultdict(list)
                for e, h in zip(np.abs(pz - y_te), te_te["tod_hour"].values):
                    tb[tod_bin(h)].append(e)
                rows[-2]["tod_mae_min"] = {str(k): round(float(np.mean(v)) / 60.0, 3)
                                           for k, v in sorted(tb.items())}
        print(f"  [{target}|s{seed}] ladder {time.time()-_t:.0f}s", flush=True)

    # ── #3 feature-group ablation（XGB；核心组 + LGB；补充组 XGB）──
    if "ablation" in args.settings:
        _t = time.time()
        groups = list(CORE_GROUPS) + list(SUPPLEMENT_GROUPS)
        for g in groups:
            is_core = g in CORE_GROUPS
            if not is_core and seed != SUPPLEMENT_SEEDS[0]:
                continue
            cols = FEATURE_GROUPS[g]
            Xs, ys = _xy(src, cols)
            Xt, _ = _xy(te_te, cols)
            mx = train_xgb(Xs, ys, seed, _gtrack(src))
            rec("xgb", "zero-shot", g, predict(mx, Xt))
            if is_core and _HAS_LGB:
                ml = train_lgb(Xs, ys, seed, _gtrack(src))
                rec("lgb", "zero-shot", g, predict(ml, Xt))
        print(f"  [{target}|s{seed}] ablation {time.time()-_t:.0f}s", flush=True)

    # ── #5 few-shot（R=5 track-level 子抽样）+ #6 matched-size ──
    if "fewshot" in args.settings or "matched" in args.settings:
        _t = time.time()
        Xsrc, ysrc = _xy(src, FEATURES)
        Xte, _ = _xy(te_te, FEATURES)
        for ratio in args.fs_ratios:
            gains = []
            for rep in range(args.reps):
                subset_seed = 1000 + rep  # H3: 与 model seed 解耦
                fs = subsample_target_tracks(te_tr, ratio, subset_seed)
                Xfs, yfs = _xy(fs, FEATURES)
                Xmix = np.vstack([Xsrc, Xfs]); ymix = np.concatenate([ysrc, yfs])
                tkmix = np.concatenate([_gtrack(src), _gtrack(fs)])
                m = train_xgb(Xmix, ymix, seed, tkmix)
                p = predict(m, Xte)
                mae = mae_min(p, y_te)
                gains.append(mae)
                rows.append({"target": target, "seed": seed, "model": "xgb", "setting": "few-shot",
                             "group": "full", "ratio": ratio, "rep": rep, "subset_seed": subset_seed,
                             "mae_min": round(mae, 3), "n_seg": int(len(y_te)), "n_track": int(n_track_te)})
                # matched A: 等量额外源
                if "matched" in args.settings:
                    extra = sample_source_segments(src, len(fs), subset_seed)
                    Xe, ye = _xy(extra, FEATURES)
                    XmixA = np.vstack([Xsrc, Xe]); ymixA = np.concatenate([ysrc, ye])
                    mA = train_xgb(XmixA, ymixA, seed, np.concatenate([_gtrack(src), _gtrack(extra)]))
                    rows.append({"target": target, "seed": seed, "model": "xgb",
                                 "setting": "matched-A", "group": "full", "ratio": ratio, "rep": rep,
                                 "subset_seed": subset_seed, "mae_min": round(mae_min(predict(mA, Xte), y_te), 3),
                                 "n_seg": int(len(y_te)), "n_track": int(n_track_te)})
                    # matched B: 源子采样至 N_src - N_target_r + target_r = N_src 总量
                    keep = max(len(src) - len(fs), 1)
                    src_b = sample_source_segments(src, keep, subset_seed)
                    Xb, yb = _xy(src_b, FEATURES)
                    XmixB = np.vstack([Xb, Xfs]); ymixB = np.concatenate([yb, yfs])
                    mB = train_xgb(XmixB, ymixB, seed, np.concatenate([_gtrack(src_b), _gtrack(fs)]))
                    rows.append({"target": target, "seed": seed, "model": "xgb",
                                 "setting": "matched-B", "group": "full", "ratio": ratio, "rep": rep,
                                 "subset_seed": subset_seed, "mae_min": round(mae_min(predict(mB, Xte), y_te), 3),
                                 "n_seg": int(len(y_te)), "n_track": int(n_track_te)})
            if "fewshot" in args.settings:
                rows.append({"target": target, "seed": seed, "model": "xgb", "setting": "few-shot-agg",
                             "group": "full", "ratio": ratio, "mae_mean": round(float(np.mean(gains)), 3),
                             "mae_sd": round(float(np.std(gains, ddof=1)), 3), "reps": args.reps,
                             "n_seg": int(len(y_te)), "n_track": int(n_track_te)})
        print(f"  [{target}|s{seed}] fewshot+matched {time.time()-_t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=NAMES_20)
    ap.add_argument("--targets", nargs="*", default=None, help="只跑这些 target（源=其余 scenes）")
    ap.add_argument("--seeds", type=int, nargs="*", default=CORE_SEEDS)
    ap.add_argument("--settings", nargs="*", default=["ladder", "ablation", "fewshot", "matched"])
    ap.add_argument("--fs-ratios", type=float, nargs="*", default=[0.05, 0.10, 0.25])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=8, help="每进程线程数（分片时调小避免超订）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="XGBoost 设备")
    ap.add_argument("--seg-dir", default=SEG_DIR)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "bench_segment.json"))
    args = ap.parse_args()
    global N_JOBS, DEVICE
    N_JOBS = args.n_jobs
    DEVICE = args.device
    scenes = load_scenes(args.scenes, args.seg_dir)
    targets = args.targets or args.scenes
    print(f"scenes={len(scenes)} targets={len(targets)} seeds={args.seeds} settings={args.settings}", flush=True)
    rows = []
    t0 = time.time()
    for seed in args.seeds:
        for target in targets:
            if target not in scenes:
                continue
            tt = time.time()
            eval_target(target, seed, scenes, args, rows)
            print(f"[{target}|s{seed}] {time.time()-tt:.0f}s  rows={len(rows)}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"scenes": args.scenes, "seeds": args.seeds,
                            "settings": args.settings, "reps": args.reps},
                   "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"saved {args.out}  total {time.time()-t0:.0f}s  rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
