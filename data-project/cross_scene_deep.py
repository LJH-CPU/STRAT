# -*- coding: utf-8 -*-
"""
段级深度家族跨场景迁移（P4：architecture vs historical-context）。

- 逐段滑窗：target = 当前 segment 的 dur_s；context = 前 L-1 段 + 当前段。
- 公平性硬约束：所有 L 使用**同一批 target segments**（i >= MIN_HISTORY = 7），
  只改变可见历史长度；训练与评测同集合。
- L=1 是合法的 no-history 基线（仅当前段特征）。
- 与 XGB/LGB 单步表格基线对照。

用法：
  python data-project/cross_scene_deep.py --model transformer --lens 1 2 4 8 \
      --targets 太白山 长白山 庐山 华山 恒山 --seeds 42 100 2024 7 17 \
      --out data-project/cross_scene_deep_seq.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scenes_20 import NAMES_20
from feature_groups import FEATURES  # noqa: F401  (H1 单一真源)

SEEDS = [42, 100, 2024, 7, 17]
MIN_HISTORY = 7          # = max(lens)-1，保证所有 L 共用同一 target 集合
BATCH = 256
EPOCHS = 30
PATIENCE = 8
LR = 1e-3


def split_scene(df, seed, train_ratio=0.8):
    tracks = df["track"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(tracks)
    n_tr = int(len(tracks) * train_ratio)
    tr, te = set(tracks[:n_tr]), set(tracks[n_tr:])
    return df[df["track"].isin(tr)], df[df["track"].isin(te)]


def make_windows(df, L, min_history=MIN_HISTORY):
    """逐段滑窗。返回 (seqs[N,L,F], durs[N], tracks[N])；target = 段 i（i>=min_history）。"""
    F = len(FEATURES)
    seqs, durs, tids = [], [], []
    for tid, g in df.groupby("track"):
        g = g.sort_values("seg_idx")
        xs = np.nan_to_num(g[FEATURES].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        ds = g["dur_s"].values.astype(np.float32)
        ds = np.clip(np.nan_to_num(ds, nan=0.0, posinf=7200.0, neginf=0.0), 0.0, 7200.0)
        n = len(xs)
        for i in range(min_history, n):
            ctx = xs[i - L + 1:i + 1]
            if len(ctx) < L:
                continue
            seqs.append(ctx); durs.append(ds[i]); tids.append(tid)
    if not seqs:
        return np.zeros((0, L, F), np.float32), np.zeros(0, np.float32)
    return np.asarray(seqs, np.float32), np.asarray(durs, np.float32)


class LSTMModel(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        h, _ = self.lstm(x)
        return self.head(h).squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(self, d_in, max_len=8, d_model=64, layers=2, nhead=4):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer("_tril", torch.tril(torch.ones(max_len, max_len)).bool())

    def forward(self, x, mask):
        B, L, _ = x.shape
        h = self.proj(x) + self.pos(torch.arange(L, device=x.device))
        causal = ~self._tril[:L, :L]
        h = self.enc(h, mask=causal)
        return self.head(h).squeeze(-1)


def train_eval(model, tr_seqs, tr_dur, te_seqs, te_dur, seed, dev, epochs=EPOCHS):
    """target 为滑窗最后一段时长；只在该位置算损失/指标。"""
    torch.manual_seed(seed); np.random.seed(seed)
    F = tr_seqs.shape[-1]
    mu = tr_seqs.reshape(-1, F).mean(0)
    sd = tr_seqs.reshape(-1, F).std(0) + 1e-6
    tr_seqs = (tr_seqs - mu) / sd
    te_seqs = (te_seqs - mu) / sd
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    Xt = torch.tensor(tr_seqs, device=dev); Yt = torch.tensor(tr_dur, device=dev)
    Xe = torch.tensor(te_seqs, device=dev); Ye = torch.tensor(te_dur, device=dev)
    n = Xt.shape[0]
    perm0 = np.random.RandomState(seed).permutation(n)
    n_va = max(int(n * 0.15), 1)
    va_idx = torch.tensor(perm0[:n_va], device=dev)
    fit_idx = perm0[n_va:]
    best, best_sd, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        ep_perm = np.random.RandomState(seed + ep).permutation(len(fit_idx))
        for i in range(0, len(fit_idx), BATCH):
            idx = torch.tensor(fit_idx[ep_perm[i:i + BATCH]], device=dev)
            out = model(Xt[idx], None)[:, -1]
            loss = (out - Yt[idx]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            err = (model(Xt[va_idx], None)[:, -1] - Yt[va_idx]).abs().mean().item()
        if np.isnan(err) or np.isinf(err):
            bad += 1
            if bad >= PATIENCE:
                break
            continue
        if err < best:
            best = err; best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_sd)
    model.eval()
    with torch.no_grad():
        mae = (model(Xe, None)[:, -1] - Ye).abs().mean().item() / 60.0
    return mae


def train_eval_tabular(kind, tr_X, tr_y, te_X, te_y, seed, dev=None):
    """单步表格基线（XGB/LGB），与序列模型使用完全相同的 target segments。

    只用窗口最后一段（当前段）特征；target = 该段时长。超参与 segment_benchmark 一致。
    """
    Xtr = np.asarray(tr_X, np.float32); ytr = np.asarray(tr_y, np.float64)
    Xte = np.asarray(te_X, np.float32); yte = np.asarray(te_y, np.float64)
    n_va = max(int(len(Xtr) * 0.15), 20)
    vi = np.random.RandomState(seed).choice(len(Xtr), n_va, replace=False)
    m = np.ones(len(Xtr), bool); m[vi] = False
    if kind == "xgb":
        model = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                 objective="reg:absoluteerror", tree_method="hist",
                                 random_state=seed, n_jobs=8, early_stopping_rounds=30)
        model.fit(Xtr[m], ytr[m], eval_set=[(Xtr[vi], ytr[vi])], verbose=False)
    elif kind == "lgb":
        dtr = lgb.Dataset(Xtr[m], ytr[m]); dva = lgb.Dataset(Xtr[vi], ytr[vi], reference=dtr)
        model = lgb.train({"objective": "mae", "learning_rate": 0.05, "n_estimators": 400,
                           "max_depth": 6, "num_leaves": 63, "feature_fraction": 0.8,
                           "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                           "seed": seed, "num_threads": 8, "verbosity": -1},
                          dtr, num_boost_round=400, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        raise ValueError(kind)
    pred = np.maximum(np.asarray(model.predict(Xte), float), 0)
    return float(np.mean(np.abs(pred - yte))) / 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=NAMES_20)
    ap.add_argument("--train-scenes", nargs="*", default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--lens", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--model", default="transformer", choices=["lstm", "transformer", "xgb", "lgb"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--min-history", type=int, default=MIN_HISTORY)
    ap.add_argument("--indir", default="data-project/segment_features")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train_names = args.train_scenes or list(args.scenes)
    targets = args.targets or list(args.scenes)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model={args.model} device={dev} lens={args.lens} min_history={args.min_history}", flush=True)

    data = {s: pd.read_csv(os.path.join(args.indir, f"{s}_segments.csv")) for s in args.scenes}
    results = {}
    for seed in args.seeds:
        rows = []
        for ts in targets:
            te_tr, te_te = split_scene(data[ts], seed)
            others = pd.concat([split_scene(data[s], seed)[0] for s in train_names if s != ts],
                               ignore_index=True)
            for L in args.lens:
                tr_seq, tr_dur = make_windows(others, L, args.min_history)
                te_seq, te_dur = make_windows(te_te, L, args.min_history)
                if len(tr_seq) < 50 or len(te_seq) < 5:
                    continue
                if args.model in ("xgb", "lgb"):
                    if args.model == "lgb" and not _HAS_LGB:
                        continue
                    # 单步表格：只用窗口最后一段（当前段）特征，与序列模型同 target 集
                    t0 = time.time()
                    mae = train_eval_tabular(args.model, tr_seq[:, -1, :], tr_dur,
                                             te_seq[:, -1, :], te_dur, seed, dev)
                    rows.append({"target": ts, "seed": seed, "model": args.model, "L": L,
                                 "setting": "zero-shot", "mae_min": round(mae, 3),
                                 "n_train": int(len(tr_seq)), "n_test": int(len(te_seq)),
                                 "secs": round(time.time() - t0, 1)})
                    print(f"[{ts}|s{seed}|L{L}] {args.model} MAE={mae:.2f} "
                          f"(ntr={len(tr_seq)} nte={len(te_seq)} {time.time()-t0:.0f}s)", flush=True)
                    continue
                model = LSTMModel(len(FEATURES)) if args.model == "lstm" else \
                    TransformerModel(len(FEATURES), max_len=L)
                t0 = time.time()
                mae = train_eval(model, tr_seq, tr_dur, te_seq, te_dur, seed, dev, args.epochs)
                rows.append({"target": ts, "seed": seed, "model": args.model, "L": L,
                             "setting": "zero-shot", "mae_min": round(mae, 3),
                             "n_train": int(len(tr_seq)), "n_test": int(len(te_seq)),
                             "secs": round(time.time() - t0, 1)})
                print(f"[{ts}|s{seed}|L{L}] {args.model} MAE={mae:.2f} "
                      f"(ntr={len(tr_seq)} nte={len(te_seq)} {time.time()-t0:.0f}s)", flush=True)
        results[f"seed{seed}"] = rows

    out = args.out or f"data-project/cross_scene_deep_{args.model}_seq.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("saved:", out)


if __name__ == "__main__":
    main()
