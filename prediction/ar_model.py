"""
自回归预测模型：GRU 编码器 + 三头输出（下一区域 / 段时长 / 路线）。

训练：teacher forcing（输入真实历史区域序列）。
推理：自回归 rollout（从已观察前缀开始，把预测的区域喂回继续预测）。

特征（无泄漏）：
- region embedding（train 词汇表）
- geo：lat/lon/elev + POI 类型分布（train 归一化）
- gps_seg：静态路径距离/地形/时刻/历史均时长速度（轨迹自身历史）
- time-of-day、role（train KMeans）、position
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

GPS_SEG_DIM = 2  # [时刻sin, 时刻cos]（严格无泄漏：仅预测原点可观测）
POI_NAMES = ['餐饮', '休闲', '住宿', '风景名胜', '科教文化', '公共设施']
POI_DIM = len(POI_NAMES)


class ARDataset(Dataset):
    def __init__(self, records, region_meta, norm, max_len=8, poi_dim=POI_DIM):
        self.max_len = max_len
        self.poi_dim = poi_dim
        self.region_meta = region_meta
        self.norm = norm
        self.samples = []
        for r in records:
            if len(r['seq']) < 2:
                continue
            s = self._build(r)
            if s is not None:
                self.samples.append(s)

    def _poi_vec(self, prof):
        v = np.zeros(self.poi_dim, dtype=np.float32)
        for k in POI_NAMES:
            if k in prof:
                v[POI_NAMES.index(k)] = prof[k]
        return v

    def _build(self, r):
        L = self.max_len
        seq_full = r['seq']
        seq = seq_full[:L]  # 截断到 max_len
        Lr = len(seq)
        n = self.norm

        region_ids = torch.full((L,), -1, dtype=torch.long)
        role_ids = torch.zeros(L, dtype=torch.long)
        geo = torch.zeros(L, 3 + self.poi_dim)
        gps = torch.zeros(L, GPS_SEG_DIM)
        time_feat = torch.zeros(L, 2)

        hour = r['start_time_of_day']
        time_feat[:, 0] = np.sin(2 * np.pi * hour / 24.0)
        time_feat[:, 1] = np.cos(2 * np.pi * hour / 24.0)

        for i, rid in enumerate(seq[:L]):
            m = self.region_meta.get(rid, {})
            region_ids[i] = rid
            role_ids[i] = m.get('role_id', 0)
            geo[i, 0] = (m.get('lat', 0) - n['lat_mean']) / n['lat_std']
            geo[i, 1] = (m.get('lon', 0) - n['lon_mean']) / n['lon_std']
            geo[i, 2] = (m.get('elev_mean', 0) - n['elev_mean']) / n['elev_std']
            pv = self._poi_vec(m.get('poi_profile', {}))
            geo[i, 3:] = torch.from_numpy((pv - n['poi_mean']) / n['poi_std'])

        gf = r['gps_seg_features']  # len Lr-1
        for i in range(min(Lr - 1, L)):
            arr = np.array(gf[i], dtype=np.float32)
            gps[i] = torch.from_numpy((arr - n['gps_mean']) / n['gps_std'])

        # 行程上下文：段特征已改为仅时刻（无静态路径距离），置零避免误用
        cum_dist = torch.zeros(L)

        # 目标：位置 i 预测 下一区域 r_{i+1} 与段时长 d_i（i in 0..Lr-2）
        tgt_region = torch.full((L,), -1, dtype=torch.long)
        tgt_dur = torch.zeros(L)
        for i in range(Lr - 1):
            tgt_region[i] = seq[i + 1]
            tgt_dur[i] = r['segment_durations'][i] / (n['seg_p95'] + 1e-6)

        mask = torch.zeros(L, dtype=torch.bool)
        mask[:Lr] = True

        positions = torch.arange(L, dtype=torch.long)

        return {
            'region_ids': region_ids, 'geo': geo, 'gps': gps, 'time_feat': time_feat,
            'role_ids': role_ids, 'positions': positions,
            'cum_dist': cum_dist,
            'tgt_region': tgt_region, 'tgt_dur': tgt_dur,
            'mask': mask, 'length': Lr, 'route_id': int(r['route_id']),
            'arrival_offsets': np.array(r['arrival_offsets'], dtype=np.float64),
            'true_seqs': seq_full,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    def st(k):
        v = [b[k] for b in batch]
        if isinstance(v[0], torch.Tensor):
            return torch.stack(v)
        return v
    return {
        'region_ids': st('region_ids'), 'geo': st('geo'), 'gps': st('gps'),
        'time_feat': st('time_feat'), 'role_ids': st('role_ids'),
        'positions': st('positions'),
        'cum_dist': st('cum_dist'),
        'tgt_region': st('tgt_region'), 'tgt_dur': st('tgt_dur'),
        'mask': st('mask'), 'route_id': st('route_id'),
        'arrival_offsets': st('arrival_offsets'), 'true_seqs': st('true_seqs'),
        'length': st('length'),
    }


class ARPredictor(nn.Module):
    def __init__(self, num_regions, num_routes, region_embed_dim=32, geo_dim=9,
                 gps_dim=GPS_SEG_DIM, time_dim=2, role_embed_dim=8,
                 hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.region_embed = nn.Embedding(num_regions, region_embed_dim, padding_idx=-1)
        self.role_embed = nn.Embedding(16, role_embed_dim)
        self.pos_embed = nn.Embedding(16, 8)
        in_dim = region_embed_dim + geo_dim + gps_dim + time_dim + role_embed_dim + 8
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        h = hidden_dim
        self.region_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, num_regions))
        self.dur_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.route_head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, num_routes))

    def forward(self, region_ids, geo, gps, time_feat, role_ids, positions):
        B, L, _ = geo.shape
        emb = self.region_embed(region_ids.clamp(min=0))
        role_emb = self.role_embed(role_ids.clamp(min=0))
        pos_emb = self.pos_embed(positions)
        x = torch.cat([emb, geo, gps, time_feat, role_emb, pos_emb], dim=-1)
        lengths = (region_ids != -1).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=L)
        return self.region_head(out), self.dur_head(out).squeeze(-1), self.route_head(out)


def build_normalizers(records, region_meta, poi_dim=POI_DIM):
    """归一化参数只在 train records 上算。"""
    lats = [m['lat'] for m in region_meta.values()]
    lons = [m['lon'] for m in region_meta.values()]
    elevs = [m['elev_mean'] for m in region_meta.values()]
    pois = []
    for m in region_meta.values():
        v = np.zeros(poi_dim, dtype=np.float32)
        for k in POI_NAMES:
            if k in m.get('poi_profile', {}):
                v[POI_NAMES.index(k)] = m['poi_profile'][k]
        pois.append(v)
    pois = np.array(pois)
    gps = np.array([f for r in records for f in r['gps_seg_features']], dtype=np.float32)
    segs = np.array([d for r in records for d in r['segment_durations']], dtype=np.float64)
    return {
        'lat_mean': float(np.mean(lats)), 'lat_std': float(np.std(lats)) + 1e-6,
        'lon_mean': float(np.mean(lons)), 'lon_std': float(np.std(lons)) + 1e-6,
        'elev_mean': float(np.mean(elevs)), 'elev_std': float(np.std(elevs)) + 1e-6,
        'poi_mean': pois.mean(axis=0) if len(pois) else np.zeros(poi_dim),
        'poi_std': pois.std(axis=0) + 1e-6 if len(pois) else np.ones(poi_dim),
        'gps_mean': gps.mean(axis=0) if len(gps) else np.zeros(GPS_SEG_DIM),
        'gps_std': gps.std(axis=0) + 1e-6 if len(gps) else np.ones(GPS_SEG_DIM),
        'seg_p95': float(np.percentile(segs, 95)) if len(segs) else 3600.0,
    }


def create_dataloader(records, region_meta, norm, batch_size=16, shuffle=True,
                      max_len=8, poi_dim=POI_DIM, num_workers=0):
    ds = ARDataset(records, region_meta, norm, max_len, poi_dim)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                    num_workers=num_workers,
                    persistent_workers=(num_workers > 0))
    return dl, ds


def rel_l1(pred, target, mask):
    p = pred[mask]
    t = target[mask]
    return (p - t).abs().div(t.abs() + 1e-3).mean()
