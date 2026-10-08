#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨场景 ETA 深度模型对比（共享 MLP / MLP+物理残差，零·少样本）+ 物理特征消融。

数据：data-project/segment_features/<scene>_segments.csv。
对比 XGBoost（cross_scene_eta.py 输出）看深度在零样本迁移上是否更强。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_groups import FEATURES, sample_tracks  # noqa: F401  (H1 单一真源)


def load_scenes(scenes, indir):
    return {s: pd.read_csv(os.path.join(indir, f"{s}_segments.csv")) for s in scenes}


def mae_min(pred, true):
    return float(np.mean(np.abs(pred - true))) / 60.0


def normalize(X, mu=None, sd=None):
    X = np.asarray(X, dtype=np.float32)
    if mu is None:
        mu = X.mean(0)
        sd = X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


class MLP(nn.Module):
    def __init__(self, d_in, hidden=64, out=1, phys=False):
        super().__init__()
        self.phys = phys
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out))

    def forward(self, x, naismith=None):
        out = self.net(x)
        if self.phys:
            if naismith.dim() == 1:
                naismith = naismith.unsqueeze(-1)
            out = out + torch.log1p(naismith)
        return out


def train_mlp(X, y, nais, phys, seed=42, epochs=40, bs=256, lr=2e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xn, mu, sd = normalize(X)
    y = np.asarray(y, dtype=np.float32)
    nais = np.asarray(nais, dtype=np.float32) if phys else None
    n = len(Xn)
    vi = np.random.RandomState(seed).choice(n, max(int(n * 0.15), 20), replace=False)
    m = np.ones(n, dtype=bool)
    m[vi] = False
    Xt = torch.tensor(Xn[m], device=dev); yt = torch.tensor(y[m], device=dev)
    Xv = torch.tensor(Xn[vi], device=dev); yv = torch.tensor(y[vi], device=dev)
    nis_t = torch.tensor(nais[m], device=dev) if phys else None
    nis_v = torch.tensor(nais[vi], device=dev) if phys else None
    model = MLP(Xt.shape[1], phys=phys).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float("inf")
    best_sd = None
    bad = 0
    gen = torch.Generator().manual_seed(seed)
    n_tr = Xt.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, generator=gen).to(dev)
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            if phys:
                nb = nis_t[idx]
                out = model(xb, nb)
            else:
                out = model(xb)
            loss = (out - yb).abs().mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv) if not phys else model(Xv, nis_v)
            err = (pv.squeeze(-1) - yv).abs().mean().item()
        if err < best:
            best = err
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best_sd)
    model.eval()
    return model, mu, sd


def predict(model, X, nais, phys, mu, sd):
    Xn = (np.asarray(X, dtype=np.float32) - mu) / sd
    dev = next(model.parameters()).device
    with torch.no_grad():
        tX = torch.tensor(Xn, device=dev)
        if phys:
            tn = torch.tensor(np.asarray(nais, dtype=np.float32).reshape(-1, 1), device=dev)
            out = model(tX, tn)
        else:
            out = model(tX)
    return out.cpu().numpy().ravel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", default=["峨眉山", "华山", "黄山", "泰山"])
    p.add_argument("--train-scenes", nargs="*", default=None, help="源池（默认=scenes；大→小协议传 12 大景区）")
    p.add_argument("--target-scenes", nargs="*", default=None, help="目标（默认=scenes；大→小协议传 8 小景区）")
    p.add_argument("--indir", default="data-project/segment_features")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    scenes = load_scenes(args.scenes, args.indir)
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
        others = pd.concat([splits[s][0] for s in TRAIN if s != ts], ignore_index=True)
        # 零样本
        for phys in [False, True]:
            m, mu, sd = train_mlp(others[FEATURES], others["dur_s"], others["naismith_s"], phys, args.seed)
            pr = predict(m, te_te[FEATURES], te_te["naismith_s"], phys, mu, sd)
            r = mae_min(pr, y_true)
            rows.append({"target": ts, "model": "Shared MLP" + ("+phys" if phys else ""),
                         "setting": "zero-shot", "mae_min": round(r, 2)})
        # 少样本 10%
        fs = sample_tracks(te_tr, 0.10, 1000 + args.seed)
        for phys in [False, True]:
            m, mu, sd = train_mlp(pd.concat([others, fs], ignore_index=True)[FEATURES],
                                  pd.concat([others, fs], ignore_index=True)["dur_s"],
                                  pd.concat([others, fs], ignore_index=True)["naismith_s"],
                                  phys, args.seed)
            pr = predict(m, te_te[FEATURES], te_te["naismith_s"], phys, mu, sd)
            r = mae_min(pr, y_true)
            rows.append({"target": ts, "model": "Shared MLP" + ("+phys" if phys else ""),
                         "setting": "few-shot10%", "mae_min": round(r, 2)})
        print(f"[{ts}] " + " ".join(
            f"{r['model']}/{r['setting']}={r['mae_min']}" for r in rows[-4:]))

    res = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(args.indir), f"cross_scene_mlp_s{args.seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    piv = res.pivot_table(index="model", columns="setting", values="mae_min", aggfunc="mean")
    print("\n=== MLP 汇总（均值）===")
    print(piv.round(1))
    print("保存:", out)


if __name__ == "__main__":
    main()
