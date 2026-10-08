#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双树模型跨场景 ETA 基线（XGBoost + LightGBM，严格同协议，5 种子）：
- Naismith 物理锚
- 每场景自训 / 共享零样本 / 共享少样本 10%（两模型）
- 共享零样本特征消融 full / no-slope / no-elevation（两模型，验证坡度普适非 XGBoost 偶然）

数据：data-project/segment_features/<scene>_segments.csv（500m 段，dur_s 标签）
划分：track 级 80/20，种子 {42,100,2024,7,17}，与 cross_scene_eta/characterize 同协议。
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenes_20 import NAMES_20
from feature_groups import FEATURES, SLOPE, ELEV, sample_tracks  # noqa: F401  (H1 单一真源)

SEEDS = [42, 100, 2024, 7, 17]
INDIR = "data-project/segment_features"
OUT = "data-project/cross_scene_trees.json"


def load_scenes(scenes, indir=INDIR):
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


def train_xgb(X, y, seed):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    n_va = max(int(len(X) * 0.15), 20)
    rng = np.random.RandomState(seed)
    vi = rng.choice(len(X), n_va, replace=False)
    m = np.ones(len(X), dtype=bool); m[vi] = False
    model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             objective="reg:absoluteerror", tree_method="hist", device="cuda",
                             random_state=seed, n_jobs=8, early_stopping_rounds=30)
    model.fit(X[m], y[m], eval_set=[(X[vi], y[vi])], verbose=False)
    return model


def train_lgb(X, y, seed):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float64)
    n_va = max(int(len(X) * 0.15), 20)
    rng = np.random.RandomState(seed)
    vi = rng.choice(len(X), n_va, replace=False)
    m = np.ones(len(X), dtype=bool); m[vi] = False
    dtr = lgb.Dataset(X[m], y[m])
    dva = lgb.Dataset(X[vi], y[vi], reference=dtr)
    model = lgb.train(
        {"objective": "mae", "learning_rate": 0.05, "n_estimators": 400,
         "max_depth": 6, "num_leaves": 63, "feature_fraction": 0.8,
         "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
         "seed": seed, "num_threads": 8, "verbosity": -1},
        dtr, num_boost_round=400, valid_sets=[dva],
        callbacks=[lgb.early_stopping(30, verbose=False)])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=NAMES_20)
    ap.add_argument("--train-scenes", nargs="*", default=None, help="源池（默认=全部场景；大→小协议传 12 大景区）")
    ap.add_argument("--target-scenes", nargs="*", default=None, help="目标景区（默认=全部场景；大→小协议传 8 小景区）")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--indir", default=INDIR)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    scenes = load_scenes(args.scenes, args.indir)
    seeds = args.seeds
    TRAIN = args.train_scenes or list(scenes.keys())
    TARGETS = args.target_scenes or list(scenes.keys())
    results = {}
    importance = {}
    for seed in seeds:
        splits = {s: split_scene(df, seed) for s, df in scenes.items()}
        rows = []
        for ts in TARGETS:
            te_tr, te_te = splits[ts]
            y_true = te_te["dur_s"].values
            X_te = te_te[FEATURES].values
            rows.append({"target": ts, "seed": seed, "model": "Naismith", "setting": "physics",
                         "mae_min": round(mae_min(te_te["naismith_s"].values, y_true), 2)})
            others = pd.concat([splits[s][0] for s in TRAIN if s != ts], ignore_index=True)
            X_other, y_other = others[FEATURES].values, others["dur_s"].values
            # per-scene self-train
            for mdl, tr in [("XGB", train_xgb(te_tr[FEATURES], te_tr["dur_s"], seed)),
                            ("LightGBM", train_lgb(te_tr[FEATURES], te_tr["dur_s"], seed))]:
                rows.append({"target": ts, "seed": seed, "model": mdl, "setting": "self-train",
                             "mae_min": round(mae_min(tr.predict(X_te), y_true), 2)})
            # shared zero-shot + few-shot 10% + ablations
            for mdl, tr in [("XGB", train_xgb(X_other, y_other, seed)),
                            ("LightGBM", train_lgb(X_other, y_other, seed))]:
                rows.append({"target": ts, "seed": seed, "model": mdl, "setting": "zero-shot",
                             "mae_min": round(mae_min(tr.predict(X_te), y_true), 2)})
                if mdl == "XGB":
                    # Fig.3b 数据：共享零样本 XGB 的 gain 特征重要度（每 target×seed）
                    gi = tr.get_booster().get_score(importance_type="gain")
                    tot = sum(gi.values()) or 1.0
                    importance[f"{ts}|{seed}"] = {k: round(v / tot, 5) for k, v in gi.items()}
                for tag, drop in [("no-slope", SLOPE), ("no-elev", ELEV)]:
                    keep = [f for f in FEATURES if f not in drop]
                    m_a = train_xgb(others[keep].values, y_other, seed) if mdl == "XGB" else \
                          train_lgb(others[keep].values, y_other, seed)
                    rows.append({"target": ts, "seed": seed, "model": mdl, "setting": f"zero-{tag}",
                                 "mae_min": round(mae_min(m_a.predict(te_te[keep].values), y_true), 2)})
                fs = sample_tracks(te_tr, 0.10, 1000 + seed)
                X_fs = pd.concat([others, fs], ignore_index=True)
                m_f = train_xgb(X_fs[FEATURES].values, X_fs["dur_s"].values, seed) if mdl == "XGB" else \
                      train_lgb(X_fs[FEATURES].values, X_fs["dur_s"].values, seed)
                rows.append({"target": ts, "seed": seed, "model": mdl, "setting": "few-shot10%",
                             "mae_min": round(mae_min(m_f.predict(X_te), y_true), 2)})
            print(f"[{ts}|s{seed}] done", flush=True)
        results[f"seed{seed}"] = rows

    outp = args.out
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("saved:", outp)
    imp_path = outp.replace(".json", "_importance.json")
    with open(imp_path, "w", encoding="utf-8") as f:
        json.dump(importance, f, ensure_ascii=False, indent=1)
    print("saved:", imp_path)

    # aggregate ladder
    import collections
    agg = collections.defaultdict(list)
    for rows in results.values():
        for r in rows:
            agg[(r["model"], r["setting"])].append(r["mae_min"])
    print("\n=== 阶梯（5 种子 mean±std） ===")
    for (mdl, setting), vals in sorted(agg.items()):
        print(f"  {mdl:<10}{setting:<16}{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}")
    # ablation mean per model
    print(f"\n=== 消融（零样本，mean over {len(TARGETS)} targets × {len(TRAIN)} 池化源 × {len(seeds)} seeds） ===")
    for mdl in ["XGB", "LightGBM"]:
        for tag in ["zero-shot", "zero-no-slope", "zero-no-elev"]:
            v = [r["mae_min"] for rows in results.values() for r in rows
                 if r["model"] == mdl and r["setting"] == tag]
            print(f"  {mdl:<10}{tag:<16}{np.mean(v):.2f}")


if __name__ == "__main__":
    main()
