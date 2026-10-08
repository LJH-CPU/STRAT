"""DeepMove 式注意力 RNN 基线（下一区域）。

忠于 DeepMove (Feng et al., KDD'18) 的双注意力思想：
  - 时序注意力：对过去各步隐状态，按时间间隔加权的注意力上下文；
  - 历史注意力：对过去"同时段"步的注意力上下文；
两者拼接后预测下一区域。因果（只注意 ≤i 的步），teacher-forced 训练，
与 STRAT/LSTM 同一无泄漏数据流（run.py 缓存 bundle）。
"""
import os
import json
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ar_pipeline import split_tracks, cluster_regions_train, map_test_regions
from ar_pipeline import discover_train_routes, _route_representatives, assign_test_route_pseudolabels
from ar_pipeline import build_poi_profiles, build_region_meta_train, build_records

np.random.seed(0)
torch.manual_seed(0)


class DeepMoveBaseline(nn.Module):
    def __init__(self, num_regions, d_model=64, hidden=128, nhead=1, dropout=0.1):
        super().__init__()
        self.V = num_regions
        self.d_model = d_model
        self.emb = nn.Embedding(num_regions, d_model, padding_idx=-1)
        self.geo_proj = nn.Linear(3, d_model)          # lat/lon/elev
        self.time_proj = nn.Linear(6, d_model)         # hour sin/cos, dow sin/cos, start, idx
        self.gru = nn.GRU(d_model, hidden, batch_first=True, bidirectional=False)
        self.Wh = nn.Linear(hidden, hidden)
        self.Wt = nn.Linear(hidden, hidden)            # 时序注意力 query
        self.Wtime = nn.Linear(1, 1)                   # 时间间隔标量偏置
        self.Wha = nn.Linear(hidden, 1)                # 历史注意力打分
        self.fuse = nn.Linear(hidden * 3, hidden)
        self.head = nn.Linear(hidden, num_regions)
        self.drop = nn.Dropout(dropout)

    def _time_feats(self, start_time_of_day, positions):
        B, L = positions.shape
        hour = (start_time_of_day.unsqueeze(1) * 24 + positions) % 24
        dw = (positions + 1) % 7
        t = torch.cat([
            torch.sin(2 * np.pi * hour / 24).unsqueeze(-1),
            torch.cos(2 * np.pi * hour / 24).unsqueeze(-1),
            torch.sin(2 * np.pi * dw / 7).unsqueeze(-1),
            torch.cos(2 * np.pi * dw / 7).unsqueeze(-1),
            hour.unsqueeze(-1) / 24.0,
            positions.unsqueeze(-1).float() / 24.0,
        ], dim=-1)
        return t

    def forward(self, region_ids, geo, start_time_of_day, positions, mask):
        B, L = region_ids.shape
        h = self.emb(region_ids.clamp(min=0)) + self.geo_proj(geo) + \
            self.time_proj(self._time_feats(start_time_of_day, positions))
        h = self.drop(h)
        out, _ = self.gru(h)                            # (B,L,hidden)
        logits = torch.full((B, L, self.V), -1e9, device=h.device)
        for i in range(L):
            if i == 0:
                q = out[:, 0]                           # (B,hidden)
                logits[:, 0] = self.head(self.fuse(torch.cat([q, q, q], -1)))
                continue
            hist = out[:, :i]                           # (B,i,hidden)
            cur = out[:, i]                             # (B,hidden)
            # 时序注意力：query=cur，key=hist，并加时间间隔偏置
            scores = torch.bmm(self.Wt(cur).unsqueeze(1), hist.transpose(1, 2)).squeeze(1)  # (B,i)
            gap = (positions[:, i:i + 1] - positions[:, :i]).float().unsqueeze(-1)          # (B,i,1)
            gap_feat = torch.tanh(self.Wtime(gap)).squeeze(-1)                              # (B,i)
            scores = scores + gap_feat
            scores = scores.masked_fill(~mask[:, :i], -1e9)
            a = F.softmax(scores, -1)
            ctx_t = (a.unsqueeze(-1) * hist).sum(1)     # (B,hidden)
            # 历史注意力：与当前步同时段（小时相近）的过去步
            cur_hour = (start_time_of_day * 24 + positions[:, i].float()).unsqueeze(1)   # (B,1)
            hist_hour = (start_time_of_day.unsqueeze(1) * 24 + positions[:, :i].float())  # (B,i)
            hour_sim = torch.exp(-((hist_hour - cur_hour).abs() / 2.0) ** 2)   # (B,i)
            a2 = F.softmax(self.Wha(hist).squeeze(-1) + hour_sim.log().clamp(min=-20), -1)
            a2 = a2.masked_fill(~mask[:, :i], 0)
            ctx_h = (a2.unsqueeze(-1) * hist).sum(1)
            ctx = torch.cat([cur, ctx_t, ctx_h], -1)
            logits[:, i] = self.head(F.relu(self.fuse(ctx)))
        return logits


def collate(records, max_len, V, geo_mean=None, geo_std=None):
    geo_mean = geo_mean if geo_mean is not None else [0.0, 0.0, 0.0]
    geo_std = geo_std if geo_std is not None else [1.0, 1.0, 1.0]
    seq = []
    geo = []
    stod = []
    positions = []
    masks = []
    tgts = []
    for r in records:
        s = r['seq'][:max_len]
        L = len(s)
        seq.append([-1] * max_len)
        seq[-1][:L] = s
        gl = []
        for k in range(L):
            gl.append([(r['region_lats'][k] - geo_mean[0]) / geo_std[0],
                       (r['region_lons'][k] - geo_mean[1]) / geo_std[1],
                       (r['region_elevs'][k] - geo_mean[2]) / geo_std[2]])
        gp = gl + [[0.0, 0.0, 0.0]] * (max_len - L)
        geo.append(gp)
        stod.append(float(r['start_time_of_day']))
        positions.append(list(range(L)) + [0] * (max_len - L))
        masks.append([1] * L + [0] * (max_len - L))
        t = s[1:] + [-1] * (max_len - (L - 1))
        tgts.append(t)
    return {
        'region_ids': torch.tensor(seq, dtype=torch.long),
        'geo': torch.tensor(geo, dtype=torch.float),
        'stod': torch.tensor(stod, dtype=torch.float),
        'positions': torch.tensor(positions, dtype=torch.long),
        'mask': torch.tensor(masks, dtype=torch.bool),
        'tgt': torch.tensor(tgts, dtype=torch.long),
    }


def train_eval(model, train_dl, val_dl, test_dl, device, epochs=60, lr=1e-3, seed=42, patience=15, V=0,
               geo_mean=None, geo_std=None):
    """早停/选模型用 val_dl（train 内 track 级留出）；最终指标用 test_dl。"""
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = -1.0
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        tl = 0.0; nb = 0
        for b in train_dl:
            dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            lg = model(dev['region_ids'], dev['geo'], dev['stod'], dev['positions'], dev['mask'])
            m = dev['mask'] & (dev['tgt'] != -1)
            loss = F.cross_entropy(lg[m], dev['tgt'][m])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        model.eval()
        acc, acc3 = 0.0, 0.0
        cnt = 0
        with torch.no_grad():
            for b in val_dl:
                dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
                lg = model(dev['region_ids'], dev['geo'], dev['stod'], dev['positions'], dev['mask'])
                m = dev['mask'] & (dev['tgt'] != -1)
                pr = lg[m].argmax(-1); t = dev['tgt'][m]
                acc += (pr == t).sum().item()
                acc3 += (t.unsqueeze(1) == lg[m].topk(3, -1).indices).any(-1).sum().item()
                cnt += m.sum().item()
        if acc / cnt > best:
            best = acc / cnt; best_state = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    acc = acc3 = 0.0; cnt = 0
    with torch.no_grad():
        for b in test_dl:
            dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            lg = model(dev['region_ids'], dev['geo'], dev['stod'], dev['positions'], dev['mask'])
            m = dev['mask'] & (dev['tgt'] != -1)
            pr = lg[m].argmax(-1); t = dev['tgt'][m]
            acc += (pr == t).sum().item()
            acc3 += (t.unsqueeze(1) == lg[m].topk(3, -1).indices).any(-1).sum().item()
            cnt += m.sum().item()
    return {'acc': acc / cnt, 'acc3': acc3 / cnt}


def build_bundle(name, seed, max_len=8, n_workers=8):
    cp = os.path.join(os.path.dirname(__file__), 'output', 'cache', f'{name}_{seed}_auto.pkl')
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            b = pickle.load(f)
        if 'tr_records' in b:
            lat = np.array([r['region_lats'][0] for r in b['tr_records'] if r['region_lats']])
            lon = np.array([r['region_lons'][0] for r in b['tr_records'] if r['region_lons']])
            elev = np.array([r['region_elevs'][0] for r in b['tr_records'] if r['region_elevs']])
            gmean = [float(lat.mean()), float(lon.mean()), float(elev.mean())]
            gstd = [float(lat.std()) + 1e-6, float(lon.std()) + 1e-6, float(elev.std()) + 1e-6]
            return b['tr_records'], b['te_records'], b['V'], gmean, gstd
    from pathlib import Path
    import pandas as pd
    CLEANED_DIR = Path(__file__).resolve().parent.parent / 'data-project' / 'cleaned_labeled_data'
    csv = str(CLEANED_DIR / f'{name}_cleaned.csv')
    df = pd.read_csv(csv)
    dtr, dte = split_tracks(df, 0.7, seed)
    npr = __import__('ar_pipeline').n_poi_regions_static(name)
    model_reg, dtr = cluster_regions_train(dtr, n_poi_regions=npr, seed=seed, n_workers=n_workers)
    dte, _ = map_test_regions(dte, model_reg)
    dtr, t2r, _ = discover_train_routes(dtr, n_workers=n_workers)
    reps = _route_representatives(dtr, t2r)
    dte, _ = assign_test_route_pseudolabels(dte, reps)
    profiles = build_poi_profiles(model_reg, name)
    region_meta = build_region_meta_train(dtr, model_reg, profiles)
    tr_records = build_records(dtr, region_meta, name, model_reg)
    te_records = build_records(dte, region_meta, name, model_reg,
                               track_route_map=dict(zip(dte['trackId'], dte['route_id'])))
    lat = np.array([r['region_lats'][0] for r in tr_records if r['region_lats']])
    lon = np.array([r['region_lons'][0] for r in tr_records if r['region_lons']])
    elev = np.array([r['region_elevs'][0] for r in tr_records if r['region_elevs']])
    gmean = [float(lat.mean()), float(lon.mean()), float(elev.mean())]
    gstd = [float(lat.std()) + 1e-6, float(lon.std()) + 1e-6, float(elev.std()) + 1e-6]
    return tr_records, te_records, model_reg['k'], gmean, gstd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '峨眉山', '熊猫基地', '龙泉'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--max_len', type=int, default=8)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--out', default='prediction/output/llm_deepmove.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for name in args.scenes:
        for seed in args.seeds:
            try:
                tr, te, V, gmean, gstd = build_bundle(name, seed, args.max_len, args.n_workers)
                if len(tr) < 30:
                    print(f'[{name}|{seed}] 样本不足'); continue
                _tracks = sorted({int(r['track_id']) for r in tr})
                _rng = np.random.RandomState(seed); _rng.shuffle(_tracks)
                _nva = max(int(len(_tracks) * 0.15), 1)
                _va = set(_tracks[:_nva])
                tr2 = [r for r in tr if int(r['track_id']) not in _va]
                va = [r for r in tr if int(r['track_id']) in _va]
                tr_dl = DataLoader(tr2, batch_size=args.batch, shuffle=True,
                                   collate_fn=lambda r: collate(r, args.max_len, V, gmean, gstd))
                va_dl = DataLoader(va, batch_size=args.batch, shuffle=False,
                                   collate_fn=lambda r: collate(r, args.max_len, V, gmean, gstd))
                te_dl = DataLoader(te, batch_size=args.batch, shuffle=False,
                                   collate_fn=lambda r: collate(r, args.max_len, V, gmean, gstd))
                model = DeepMoveBaseline(V).to(device)
                r = train_eval(model, tr_dl, va_dl, te_dl, device, args.epochs, seed=seed, V=V,
                               geo_mean=gmean, geo_std=gstd)
                print(f'[{name}|{seed}] DeepMove acc={r["acc"]:.4f} acc3={r["acc3"]:.4f} n_tr={len(tr)} n_te={len(te)}')
                results[f'{name}|seed{seed}'] = r
            except Exception as e:
                import traceback; traceback.print_exc()
                results[f'{name}|seed{seed}'] = {'error': str(e)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
