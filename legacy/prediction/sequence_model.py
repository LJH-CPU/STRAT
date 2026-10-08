"""
序列预测模型：Learnable Region Embedding + 双向 GRU 编码器。
V5: GPS 段特征降为 8 维（去泄露），新增 POI 语义特征。
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import GPS_SEG_DIM


class SequencePredictor(nn.Module):
    """
    序列预测模型。
    V5: 支持 POI 特征，geo_dim 包含 (lat, lon, elev, poi1, poi2, ...)
    """

    def __init__(self, num_regions, region_embed_dim, geo_dim, gps_seg_dim,
                 pos_dim, time_dim, num_roles, role_embed_dim,
                 hidden_dim=128, num_layers=2, dropout=0.2,
                 use_gps=True, use_role=True, use_time=True, use_pos=True,
                 bidirectional=True):
        super().__init__()

        self.region_embed = nn.Embedding(num_regions, region_embed_dim, padding_idx=-1)
        self.role_embed = nn.Embedding(num_roles, role_embed_dim)
        self.hidden_dim = hidden_dim

        self.use_gps = use_gps
        self.use_role = use_role
        self.use_time = use_time
        self.use_pos = use_pos
        self.bidirectional = bidirectional

        input_dim = (region_embed_dim + geo_dim + gps_seg_dim + pos_dim
                     + time_dim + role_embed_dim)

        self.pos_embed = nn.Embedding(10, pos_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.time_head = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, region_ids, geo_feats, gps_feats, time_feats,
                role_ids, lengths):
        B, L, _ = geo_feats.shape
        device = geo_feats.device

        gnn_embeds = self.region_embed(region_ids.clamp(min=0))
        role_embeds = self.role_embed(role_ids.clamp(min=0))

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        pos_embeds = self.pos_embed(positions)

        if not self.use_gps:
            gps_feats = torch.zeros_like(gps_feats)
        if not self.use_time:
            time_feats = torch.zeros_like(time_feats)
        if not self.use_role:
            role_embeds = torch.zeros_like(role_embeds)
        if not self.use_pos:
            pos_embeds = torch.zeros_like(pos_embeds)

        x = torch.cat([gnn_embeds, geo_feats, gps_feats,
                       time_feats, role_embeds, pos_embeds], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        gru_out, _ = self.gru(packed)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True, total_length=L
        )

        seg_preds = self.time_head(gru_out)
        return seg_preds


class TrajectoryDataset(Dataset):
    def __init__(self, records, region_meta, num_regions, max_len=8, poi_dim=0):
        self.max_len = max_len
        self.num_regions = num_regions
        self.region_meta = region_meta
        self.poi_dim = poi_dim
        self.samples = []

        # geo归一化参数（lat, lon, elev）
        all_lats = [m['lat'] for m in region_meta.values()]
        all_lons = [m['lon'] for m in region_meta.values()]
        all_elevs = [m['elev_mean'] for m in region_meta.values()]
        self.lat_mean = np.mean(all_lats)
        self.lat_std = np.std(all_lats) + 1e-6
        self.lon_mean = np.mean(all_lons)
        self.lon_std = np.std(all_lons) + 1e-6
        self.elev_mean = np.mean(all_elevs)
        self.elev_std = np.std(all_elevs) + 1e-6

        # POI归一化参数
        if poi_dim > 0:
            all_poi = []
            for r in records:
                for prof in r.get('poi_profiles', []):
                    vec = self._poi_dict_to_vec(prof)
                    all_poi.append(vec)
            all_poi = np.array(all_poi, dtype=np.float32)
            self.poi_mean = all_poi.mean(axis=0) if len(all_poi) > 0 else np.zeros(poi_dim)
            self.poi_std = all_poi.std(axis=0) + 1e-6 if len(all_poi) > 0 else np.ones(poi_dim)
        else:
            self.poi_mean = np.zeros(poi_dim)
            self.poi_std = np.ones(poi_dim)

        # GPS段特征归一化
        all_gps = []
        for r in records:
            for feats in r.get('gps_seg_features', []):
                all_gps.append(feats)
        all_gps = np.array(all_gps, dtype=np.float32)
        self.gps_mean = all_gps.mean(axis=0) if len(all_gps) > 0 else np.zeros(GPS_SEG_DIM)
        self.gps_std = all_gps.std(axis=0) + 1e-6 if len(all_gps) > 0 else np.ones(GPS_SEG_DIM)

        # 段时长归一化参数
        all_segs = []
        for r in records:
            all_segs.extend(r['segment_durations'])
        self.seg_p95 = float(np.percentile(all_segs, 95)) if all_segs else 3600.0

        for r in records:
            if len(r['seq']) > max_len:
                continue
            sample = self._build_sample(r)
            self.samples.append(sample)

    def _poi_dict_to_vec(self, prof_dict):
        """POI类型分布dict → 定长向量"""
        if not prof_dict or self.poi_dim == 0:
            return np.zeros(self.poi_dim, dtype=np.float32)
        # 按类型名排序以保证顺序一致
        order = sorted(prof_dict.keys())
        vec = np.array([prof_dict.get(k, 0.0) for k in order], dtype=np.float32)
        if len(vec) < self.poi_dim:
            vec = np.pad(vec, (0, self.poi_dim - len(vec)))
        return vec[:self.poi_dim]

    def _build_sample(self, r):
        seq_len = len(r['seq'])
        L = self.max_len

        region_ids = torch.full((L,), -1, dtype=torch.long)
        role_ids = torch.full((L,), 0, dtype=torch.long)
        for i, rid in enumerate(r['seq']):
            region_ids[i] = rid
            role_ids[i] = self.region_meta.get(rid, {}).get('role_id', 0)

        # geo特征: (lat, lon, elev, poi1, poi2, ...)
        geo_dim_base = 3 + self.poi_dim
        geo = torch.zeros(L, geo_dim_base)
        for i, rid in enumerate(r['seq']):
            m = self.region_meta.get(rid, {})
            geo[i, 0] = (m.get('lat', 0) - self.lat_mean) / self.lat_std
            geo[i, 1] = (m.get('lon', 0) - self.lon_mean) / self.lon_std
            geo[i, 2] = (m.get('elev_mean', 0) - self.elev_mean) / self.elev_std
            # POI特征
            if self.poi_dim > 0:
                poi_dict = m.get('poi_profile', {})
                poi_vec = self._poi_dict_to_vec(poi_dict)
                poi_vec = (poi_vec - self.poi_mean) / self.poi_std
                geo[i, 3:] = torch.from_numpy(poi_vec)

        # GPS段特征
        gps_seg = torch.zeros(L, GPS_SEG_DIM)
        gps_feats_list = r.get('gps_seg_features', [])
        for i, feats in enumerate(gps_feats_list):
            arr = np.array(feats, dtype=np.float32)
            gps_seg[i + 1] = torch.from_numpy(
                (arr - self.gps_mean) / self.gps_std
            )

        # 时间特征
        hour = r.get('start_time_of_day', 12.0)
        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
        time_feat = torch.zeros(L, 2)
        time_feat[:, 0] = hour_sin
        time_feat[:, 1] = hour_cos

        # 目标
        segs = r['segment_durations']
        targets_norm = np.zeros(L, dtype=np.float32)
        for i in range(len(segs)):
            targets_norm[i + 1] = segs[i] / (self.seg_p95 + 1e-6)

        target_tensor = torch.from_numpy(targets_norm).unsqueeze(-1)
        mask = torch.zeros(L, dtype=torch.bool)
        mask[:seq_len] = True

        return {
            'region_ids': region_ids,
            'geo': geo,
            'gps_seg': gps_seg,
            'time_feat': time_feat,
            'role_ids': role_ids,
            'targets': target_tensor,
            'mask': mask,
            'length': seq_len,
            'track_id': r['track_id'],
            'route_id': r['route_id'],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    region_ids = torch.stack([b['region_ids'] for b in batch])
    geo = torch.stack([b['geo'] for b in batch])
    gps_seg = torch.stack([b['gps_seg'] for b in batch])
    time_feat = torch.stack([b['time_feat'] for b in batch])
    role_ids = torch.stack([b['role_ids'] for b in batch])
    targets = torch.stack([b['targets'] for b in batch])
    mask = torch.stack([b['mask'] for b in batch])
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    track_ids = [b['track_id'] for b in batch]
    route_ids = [b['route_id'] for b in batch]
    return (region_ids, geo, gps_seg, time_feat, role_ids,
            targets, mask, lengths, track_ids, route_ids)


def create_dataloader(records, region_meta, num_regions, batch_size=16,
                      shuffle=True, max_len=8, poi_dim=0):
    dataset = TrajectoryDataset(records, region_meta, num_regions, max_len, poi_dim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=collate_fn)
    return loader, dataset
