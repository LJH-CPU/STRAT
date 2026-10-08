"""② 多场景联合训练：共享特征表示 + 共享 Transformer 背板 + per-scene 分类头。

用 8 倍数据学跨场景的转移结构（role/POI/geo/time/gps 特征），
区域 ID 不作输入（避免跨场景词汇错配），仅 per-scene 头输出本场景区域。
试点：青城山/熊猫基地/峨眉山，teacher-forced acc vs 每场景 LSTM。
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import CLEANED_DIR
from ar_model import POI_DIM, GPS_SEG_DIM
from transformer_model import TransformerBackbone
import run_transformer_experiment as rte


def load_bundle(name, seed):
    cp = os.path.join(os.path.dirname(__file__), 'output', 'cache', f'{name}_{seed}_auto.pkl')
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            return pickle.load(f)
    import argparse as _ap
    a = _ap.Namespace(seed=seed, k_override=None, n_poi_regions=None, n_workers=8, max_len=8)
    b = rte.build_bundle(name, str(CLEANED_DIR / f'{name}_cleaned.csv'), seed, a)
    with open(cp, 'wb') as f:
        pickle.dump(b, f)
    return b


class JointDataset(Dataset):
    """跨场景样本：每步用本场景 norm 构建共享特征（不含区域 ID）。"""

    def __init__(self, scene_meta, max_len=8):
        # scene_meta: [(scene_id, records, region_meta, norm), ...]
        self.items = []
        for sid, recs, region_meta, norm in scene_meta:
            for r in recs:
                self.items.append((sid, r, region_meta, norm))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, r, region_meta, norm = self.items[idx]
        seq = r['seq'][:8]
        L = len(seq)
        role = np.zeros(8, dtype=np.int64)
        poi = np.zeros((8, POI_DIM), dtype=np.float32)
        geo = np.zeros((8, 3), dtype=np.float32)
        tm = np.zeros((8, 2), dtype=np.float32)
        gps = np.zeros((8, GPS_SEG_DIM), dtype=np.float32)
        hour = r['start_time_of_day']
        tm[:, 0] = np.sin(2 * np.pi * hour / 24.0)
        tm[:, 1] = np.cos(2 * np.pi * hour / 24.0)
        poi_names = ['餐饮', '休闲', '住宿', '风景名胜', '科教文化', '公共设施']
        for i, rid in enumerate(seq):
            m = region_meta.get(rid, {})
            role[i] = m.get('role_id', 0)
            geo[i, 0] = (m.get('lat', 0) - norm['lat_mean']) / norm['lat_std']
            geo[i, 1] = (m.get('lon', 0) - norm['lon_mean']) / norm['lon_std']
            geo[i, 2] = (m.get('elev_mean', 0) - norm['elev_mean']) / norm['elev_std']
            pv = np.zeros(POI_DIM)
            for k, v in m.get('poi_profile', {}).items():
                if k in poi_names:
                    pv[poi_names.index(k)] = v
            poi[i] = (pv - norm['poi_mean']) / (norm['poi_std'] + 1e-9)
        gf = r['gps_seg_features']
        for i in range(min(L - 1, 8)):
            arr = np.array(gf[i], dtype=np.float32)
            gps[i] = (arr - norm['gps_mean']) / (norm['gps_std'] + 1e-9)
        tgt = np.full(8, -1, dtype=np.int64)
        for i in range(L - 1):
            tgt[i] = seq[i + 1]
        mask = np.zeros(8, dtype=bool)
        mask[:L] = True
        return sid, role, poi, geo, tm, gps, tgt, mask, L


def collate(batch, Vs):
    sid = torch.tensor([b[0] for b in batch], dtype=torch.long)
    role = torch.tensor([b[1] for b in batch])
    poi = torch.tensor([b[2] for b in batch])
    geo = torch.tensor([b[3] for b in batch])
    tm = torch.tensor([b[4] for b in batch])
    gps = torch.tensor([b[5] for b in batch])
    tgt = torch.tensor([b[6] for b in batch])
    mask = torch.tensor([b[7] for b in batch])
    return {'sid': sid, 'role': role, 'poi': poi, 'geo': geo, 'tm': tm, 'gps': gps,
            'tgt': tgt, 'mask': mask}


class JointModel(nn.Module):
    def __init__(self, Vs, n_scenes, d_model=64, hidden=32):
        super().__init__()
        self.Vs = Vs
        self.role_emb = nn.Embedding(16, 8)
        self.poi_proj = nn.Linear(POI_DIM, 16)
        self.geo_proj = nn.Linear(3, 16)
        self.tm_proj = nn.Linear(2, 16)
        self.gps_proj = nn.Linear(GPS_SEG_DIM, 16)
        self.scene_emb = nn.Embedding(n_scenes, 16)
        in_dim = 8 + 16 + 16 + 16 + 16 + 16
        self.proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.backbone = TransformerBackbone(d_model)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(),
                                                  nn.Linear(hidden, v)) for v in Vs])

    def forward(self, sid, role, poi, geo, tm, gps):
        x = torch.cat([self.role_emb(role).reshape(role.shape[0], 8, -1), self.poi_proj(poi),
                       self.geo_proj(geo), self.tm_proj(tm), self.gps_proj(gps),
                       self.scene_emb(sid).unsqueeze(1).expand(-1, 8, -1)], dim=-1)
        h = self.proj(x)
        z = self.backbone(h)
        B, L, _ = z.shape
        out = torch.zeros(B, L, max(self.Vs), device=z.device)
        for i in range(B):
            v = self.Vs[sid[i].item()]
            out[i, :, :v] = self.heads[sid[i].item()](z[i])
        return out


def train_and_eval(scene_meta, Vs, seeds_train, device, epochs=60, bs=64):
    ds = JointDataset(scene_meta)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=lambda b: collate(b, Vs))
    model = JointModel(Vs, len(scene_meta)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        for b in dl:
            dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            lg = model(dev['sid'], dev['role'], dev['poi'], dev['geo'], dev['tm'], dev['gps'])
            loss = 0.0
            for i in range(dev['sid'].shape[0]):
                v = Vs[dev['sid'][i].item()]
                m = dev['mask'][i] & (dev['tgt'][i] != -1)
                loss = loss + F.cross_entropy(lg[i][m][:, :v], dev['tgt'][i][m])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    per_scene = {}
    with torch.no_grad():
        for b in dl:
            dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            lg = model(dev['sid'], dev['role'], dev['poi'], dev['geo'], dev['tm'], dev['gps'])
            for i in range(dev['sid'].shape[0]):
                sid = dev['sid'][i].item()
                m = dev['mask'][i] & (dev['tgt'][i] != -1)
                v = Vs[sid]
                pr = lg[i][m][:, :v].argmax(-1)
                t = dev['tgt'][i][m]
                per_scene.setdefault(sid, [0, 0])
                per_scene[sid][0] += (pr == t).sum().item()
                per_scene[sid][1] += m.sum().item()
    return model, {sid: c / t for sid, (c, t) in per_scene.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '熊猫基地', '峨眉山'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--out', default='prediction/output/llm_multiscene.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 训练用各场景训练集；评测用各场景测试集（同一联合模型）
    tr_meta, te_meta, Vs = [], [], []
    names = args.scenes
    for i, name in enumerate(names):
        b = load_bundle(name, args.seed)
        Vs.append(b['V'])
        tr_meta.append((i, b['tr_records'], b['region_meta'], b['norm']))
        te_meta.append((i, b['te_records'], b['region_meta'], b['norm']))
    model, tr_acc = train_and_eval(tr_meta, Vs, None, device, args.epochs)
    te_ds = JointDataset(te_meta)
    te_dl = DataLoader(te_ds, batch_size=64, shuffle=False, collate_fn=lambda b: collate(b, Vs))
    model.eval()
    te_acc = {}
    with torch.no_grad():
        for b in te_dl:
            dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            lg = model(dev['sid'], dev['role'], dev['poi'], dev['geo'], dev['tm'], dev['gps'])
            for i in range(dev['sid'].shape[0]):
                sid = dev['sid'][i].item()
                m = dev['mask'][i] & (dev['tgt'][i] != -1)
                v = Vs[sid]
                pr = lg[i][m][:, :v].argmax(-1)
                t = dev['tgt'][i][m]
                te_acc.setdefault(sid, [0, 0])
                te_acc[sid][0] += (pr == t).sum().item()
                te_acc[sid][1] += m.sum().item()
    # 每场景 LSTM（从主结果 JSON）
    main = json.load(open(os.path.join(os.path.dirname(__file__), 'output', 'llm_main_transformer.json'), encoding='utf-8'))
    results = {}
    print(f'{"Scene":<10}{"joint_tf":>9}{"LSTM_tf":>8}')
    for i, name in enumerate(names):
        jt = te_acc[i][0] / max(te_acc[i][1], 1)
        lstm = None
        for k, v in main.items():
            if v.get('status') == 'ok' and k.startswith(f'{name}|') and f'seed{args.seed}|' in k:
                lstm = v['baselines'].get('lstm_acc')
        print(f'{name:<10}{jt:>9.3f}{lstm if lstm is not None else 0:>8.3f}')
        results[name] = {'joint_tf': jt, 'lstm_tf': lstm}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
