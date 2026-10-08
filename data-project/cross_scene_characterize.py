#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨场景地形迁移"条件与边界"刻画（4 景区，seed 可多跑）。

Exp1 跨场景校准：共享分位 XGBoost，源景区验证校准 → 目标零样本 PICP/CRPS/可靠性
Exp2 少样本曲线：0/5/10/25% 目标数据
Exp3 特征消融：full / no-slope / no-elevation 的零样本 MAE
Exp4 场景身份探针：从特征判场景的准确率（海拔是否编码场景）
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_groups import FEATURES, SLOPE, ELEV, sample_tracks  # noqa: F401  (H1 单一真源)

QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def load_scenes(scenes, indir):
    return {s: pd.read_csv(os.path.join(indir, f"{s}_segments.csv")) for s in scenes}


def mae_min(pred, true):
    return float(np.mean(np.abs(pred - true))) / 60.0


def split_scene(df, seed):
    tracks = df["track"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(tracks)
    n_tr = int(len(tracks) * 0.8)
    tr, te = set(tracks[:n_tr]), set(tracks[n_tr:])
    return df[df["track"].isin(tr)], df[df["track"].isin(te)]


def fit_quantile(X, y, seed, qs=QUANTILES):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    n_va = max(int(len(X) * 0.15), 20)
    rng = np.random.RandomState(seed)
    vi = rng.choice(len(X), n_va, replace=False)
    m = np.ones(len(X), dtype=bool)
    m[vi] = False
    models = []
    for q in qs:
        md = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              objective="reg:quantileerror", quantile_alpha=float(q),
                              tree_method="hist", device="cuda",
                              random_state=seed, n_jobs=8, early_stopping_rounds=30)
        md.fit(X[m], y[m], eval_set=[(X[vi], y[vi])], verbose=False)
        models.append(md)
    return models, vi


def quantile_metrics(models, X, y):
    Q = np.vstack([np.maximum(md.predict(np.asarray(X, dtype=np.float32)), 0) for md in models])
    med = Q[list(QUANTILES).index(0.5)]
    errs = np.abs(med - y)
    qlo, qhi = Q[0], Q[-1]
    picp90 = float(np.mean((y >= qlo) & (y <= qhi)))
    width = float(np.mean(qhi - qlo)) / 60.0
    # 离散分位 CRPS
    K = len(models)
    term1 = (1.0 / K) * np.mean(np.abs(Q.T - y[:, None]).sum(1))
    diff = np.abs(Q[:, None, :] - Q[None, :, :]).sum(axis=(0, 1))
    term2 = (1.0 / (K * K)) * np.mean(diff)
    crps = float(term1 - 0.5 * term2) / 60.0
    return {"mae_min": mae_min(med, y), "picp90": round(picp90, 3),
            "width_min": round(width, 1), "crps_min": round(crps, 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=["峨眉山", "华山", "黄山", "泰山"])
    p.add_argument("--train-scenes", nargs="*", default=None, help="源池（默认=scenes；大→小协议传 12 大景区）")
    p.add_argument("--target-scenes", nargs="*", default=None, help="目标（默认=scenes；大→小协议传 8 小景区）")
    p.add_argument("--indir", default="data-project/segment_features")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="输出 json 路径（默认 data-project/cross_scene_char_s{seed}.json）")
    args = p.parse_args()
    scenes = load_scenes(args.scenes, args.indir)
    TRAIN = args.train_scenes or list(scenes.keys())
    TARGETS = args.target_scenes or list(scenes.keys())
    splits = {s: split_scene(df, args.seed) for s, df in scenes.items()}

    out = {"seed": args.seed}
    op = args.out or os.path.join(os.path.dirname(args.indir), f"cross_scene_char_s{args.seed}.json")
    if os.path.exists(op):
        try:
            prev = json.load(open(op, encoding="utf-8"))
            for k, v in prev.items():
                if k.startswith("target_"):
                    out[k] = v
            print(f"续跑: 已有 {sum(1 for k in out if k.startswith('target_'))} 个 target", flush=True)
        except Exception:
            pass
    for ts in TARGETS:
        te_tr, te_te = splits[ts]
        y_true = te_te["dur_s"].values
        others = pd.concat([splits[s][0] for s in TRAIN if s != ts], ignore_index=True)
        row = {"target": ts}

        # Exp1: 共享分位 → 目标零样本校准
        qmodels, _ = fit_quantile(others[FEATURES], others["dur_s"], args.seed)
        row["zero_quantile"] = quantile_metrics(qmodels, te_te[FEATURES], y_true)

        # Exp2: 少样本曲线（只用中位数模型，无需全分位）
        fs_mae = {}
        for ratio in [0.0, 0.05, 0.10, 0.25]:
            if ratio == 0.0:
                df_train = others
            else:
                fs = sample_tracks(te_tr, ratio, 1000 + args.seed)
                df_train = pd.concat([others, fs], ignore_index=True)
            m = fit_quantile(df_train[FEATURES], df_train["dur_s"], args.seed, qs=(0.5,))[0][0]
            fs_mae[ratio] = round(mae_min(m.predict(np.asarray(te_te[FEATURES], dtype=np.float32)), y_true), 2)
        row["few_shot_mae"] = fs_mae

        # Exp3: 特征消融（零样本 MAE，只用中位数模型）
        abl = {}
        for name, cols in [("full", FEATURES), ("no_slope", [c for c in FEATURES if c not in SLOPE]),
                           ("no_elev", [c for c in FEATURES if c not in ELEV])]:
            m = fit_quantile(others[cols], others["dur_s"], args.seed, qs=(0.5,))[0][0]
            abl[name] = round(mae_min(m.predict(np.asarray(te_te[cols], dtype=np.float32)), y_true), 2)
        row["ablation_mae"] = abl

        # Exp4: 场景身份探针（train 拟合 / test 打分，避免 in-sample）
        tr_feats = pd.concat([splits[s][0][FEATURES] for s in scenes], ignore_index=True)
        tr_labels = np.concatenate([np.full(len(splits[s][0]), i) for i, s in enumerate(scenes)])
        te_feats = pd.concat([splits[s][1][FEATURES] for s in scenes], ignore_index=True)
        te_labels = np.concatenate([np.full(len(splits[s][1]), i) for i, s in enumerate(scenes)])
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                random_state=args.seed, n_jobs=8,
                                tree_method="hist", device="cuda")
        clf.fit(np.asarray(tr_feats, dtype=np.float32), tr_labels)
        acc = float((clf.predict(np.asarray(te_feats, dtype=np.float32)) == te_labels).mean())
        row["scene_probe_acc"] = round(acc, 3)
        out[f"target_{ts}"] = row
        # 每 target 增量写盘（支持断点续跑）
        with open(op, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[{ts}] zero_quantile={row['zero_quantile']} fs={row['few_shot_mae']} "
              f"abl={row['ablation_mae']} probe={row['scene_probe_acc']}", flush=True)

    print("保存:", op)


if __name__ == "__main__":
    main()
