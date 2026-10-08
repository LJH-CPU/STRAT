"""
预测基线：
- 下一区域：单向 LSTM（torch 自写）、一阶马尔可夫（复用 ar_baselines）
- 到达时间：kNN 区域对历史均值（复用 ar_baselines）、线性/岭回归（同特征）
- 路线：多数类、k-NN 前缀（复用 ar_baselines）
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from ar_model import ARDataset, GPS_SEG_DIM, POI_DIM
import ar_baselines as ab


class LSTMNextRegion(nn.Module):
    """单向 LSTM 下一区域分类（非 Transformer 深度基线）。"""

    def __init__(self, num_regions, geo_dim, region_embed_dim=32, hidden=64, layers=1):
        super().__init__()
        self.region_embed = nn.Embedding(num_regions, region_embed_dim, padding_idx=-1)
        self.role_embed = nn.Embedding(16, 8)
        in_dim = region_embed_dim + geo_dim + GPS_SEG_DIM + 2 + 8
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_regions))

    def forward(self, region_ids, geo, gps, time_feat, role_ids, route_id):
        B, L, _ = geo.shape
        re = self.region_embed(region_ids.clamp(min=0))
        ro = self.role_embed(role_ids.clamp(min=0))
        x = torch.cat([re, geo, gps, time_feat, ro], dim=-1)
        lens = (region_ids != -1).sum(1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)
        return self.head(out)


def train_lstm(model, loader, device, epochs=60, lr=1e-3, patience=20, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float('inf')
    best_sd = None
    wait = 0
    for _ in range(epochs):
        model.train()
        tot = 0.0
        nb = 0
        for b in loader:
            dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            logits = model(dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                           dev['role_ids'], torch.tensor(dev['route_id'], device=device))
            mask = dev['mask'] & (dev['tgt_region'] != -1)
            loss = nn.functional.cross_entropy(logits[mask], dev['tgt_region'][mask])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        vl = tot / max(nb, 1)
        if vl < best:
            best = vl
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_sd)
    return model


@torch.no_grad()
def eval_lstm_next_region(model, loader, device):
    model.eval()
    per_pos = {}
    for b in loader:
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        logits = model(dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                       dev['role_ids'], torch.tensor(dev['route_id'], device=device))
        mask = dev['mask'] & (dev['tgt_region'] != -1)
        pred = logits[mask].argmax(-1)
        true = dev['tgt_region'][mask]
        pos = mask.nonzero(as_tuple=False)[:, 0].cpu().numpy() + 1
        for p, pr, tr in zip(pos, pred.cpu().numpy(), true.cpu().numpy()):
            per_pos.setdefault(int(p), [0, 0])
            per_pos[int(p)][0] += int(pr == tr)
            per_pos[int(p)][1] += 1
    acc = sum(c for c, _ in per_pos.values()) / max(sum(t for _, t in per_pos.values()), 1)
    return acc, {str(p): c / t for p, (c, t) in sorted(per_pos.items())}


# ── 线性时长回归（同特征对照）────────────────────────────
def build_flat_time(records, pair_mean, global_mean):
    X, y = [], []
    for r in records:
        seq = r['seq']
        gf = r['gps_seg_features']
        for i, d in enumerate(r['segment_durations']):
            a = int(seq[i])
            b = a                      # 无真值下一区域（无泄漏回退）
            prior = global_mean
            f = list(gf[i]) + [a, b, prior]
            X.append(f)
            y.append(d)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def linear_time_mae(tr_records, te_records, pair_mean, global_mean):
    Xtr, ytr = build_flat_time(tr_records, pair_mean, global_mean)
    Xte, yte = build_flat_time(te_records, pair_mean, global_mean)
    if len(Xtr) < 10 or len(Xte) < 2:
        return None, 0
    clf = Ridge(alpha=1.0)
    clf.fit(Xtr, ytr)
    pred = np.maximum(clf.predict(Xte), 0)
    return float(np.mean(np.abs(pred - yte))), len(yte)


# 复用 ar_baselines 的统计基线
build_transition_model = ab.build_transition_model
next_region_accuracy = ab.next_region_accuracy
majority_route = ab.majority_route
knn_prefix_route = ab.knn_prefix_route
route_accuracy = ab.route_accuracy
build_duration_model = ab.build_duration_model
knn_duration_mae = ab.knn_duration_mae
