#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨场景 ETA 基线阶梯（Naismith / 每场景 XGBoost / 共享 XGBoost 零·少样本）。

数据：data-project/segment_features/<scene>_segments.csv（每段 500m，dur_s 标签）。
问题：共享地形模型在"未见景区"上的零/少样本 ETA 是否接近"该景区自训"？
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_groups import FEATURES, sample_tracks  # noqa: F401  (H1 单一真源)


def load_scenes(scenes, indir):
    out = {}
    for s in scenes:
        p = os.path.join(indir, f"{s}_segments.csv")
        if os.path.exists(p):
            out[s] = pd.read_csv(p)
    return out


def mae_min(pred, true):
    return float(np.mean(np.abs(pred - true))) / 60.0


def train_xgb(X, y, seed=42):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    n_va = max(int(len(X) * 0.15), 20)
    rng = np.random.RandomState(seed)
    vi = rng.choice(len(X), n_va, replace=False)
    m = np.ones(len(X), dtype=bool)
    m[vi] = False
    model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             objective="reg:absoluteerror", tree_method="hist", device="cuda", random_state=seed,
                             n_jobs=8, early_stopping_rounds=30)
    model.fit(X[m], y[m], eval_set=[(X[vi], y[vi])], verbose=False)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=["峨眉山", "华山", "黄山", "泰山"])
    p.add_argument("--train-scenes", nargs="*", default=None, help="源池（默认=scenes；大→小协议传 12 大景区）")
    p.add_argument("--target-scenes", nargs="*", default=None, help="目标（默认=scenes；大→小协议传 8 小景区）")
    p.add_argument("--indir", default="data-project/segment_features")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    scenes = load_scenes(args.scenes, args.indir)
    print(f"场景: {list(scenes.keys())}")

    # 每场景划分（track 级，80/20）
    splits = {}
    for s, df in scenes.items():
        tracks = df["track"].unique()
        rng = np.random.RandomState(args.seed)
        rng.shuffle(tracks)
        n_tr = int(len(tracks) * 0.8)
        tr, te = set(tracks[:n_tr]), set(tracks[n_tr:])
        splits[s] = (df[df["track"].isin(tr)].copy(), df[df["track"].isin(te)].copy())

    TRAIN = args.train_scenes or list(scenes.keys())
    TARGETS = args.target_scenes or list(scenes.keys())
    rows = []
    for ts in TARGETS:
        te_tr, te_te = splits[ts]
        y_true = te_te["dur_s"].values
        # 1) Naismith
        nais = mae_min(te_te["naismith_s"].values, y_true)
        rows.append({"target": ts, "model": "Naismith", "setting": "physics", "mae_min": round(nais, 2)})
        # 2) 每场景 XGBoost（目标自训）
        m = train_xgb(te_tr[FEATURES], te_tr["dur_s"], args.seed)
        per = mae_min(m.predict(te_te[FEATURES]), y_true)
        rows.append({"target": ts, "model": "Per-scene XGB", "setting": "self-train", "mae_min": round(per, 2)})
        # 3) 共享 XGBoost（其它景区训练，零样本）
        others = pd.concat([splits[s][0] for s in TRAIN if s != ts], ignore_index=True)
        ms = train_xgb(others[FEATURES], others["dur_s"], args.seed)
        zero = mae_min(ms.predict(te_te[FEATURES]), y_true)
        rows.append({"target": ts, "model": "Shared XGB", "setting": "zero-shot", "mae_min": round(zero, 2)})
        # 4) 共享 + 少样本（10% 目标数据）
        fs = sample_tracks(te_tr, 0.10, 1000 + args.seed)
        mfs = train_xgb(pd.concat([others, fs], ignore_index=True)[FEATURES],
                        pd.concat([others, fs], ignore_index=True)["dur_s"], args.seed)
        fs10 = mae_min(mfs.predict(te_te[FEATURES]), y_true)
        rows.append({"target": ts, "model": "Shared XGB", "setting": "few-shot10%", "mae_min": round(fs10, 2)})
        print(f"[{ts}] Naismith={nais:.1f}min  PerXGB={per:.1f}  Zero={zero:.1f}  FS10%={fs10:.1f}")

    res = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(args.indir), f"cross_scene_xgb_s{args.seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n=== 汇总（按模型×设置）===")
    piv = res.pivot_table(index="model", columns="setting", values="mae_min", aggfunc="mean")
    print(piv.round(1))
    print("保存:", out)


if __name__ == "__main__":
    main()
