"""
数据提取模块：从聚类后的 CSV 提取首次出现序列 + 段时长 + 特征。
V5: 去除特征泄露（段内速度/停靠比/点数），
     添加 POI 语义特征、历史平均时长/速度。
"""

import math
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


MAX_SEGMENT_SECONDS = 4 * 3600
GPS_SEG_DIM = 8  # [静态路径距离, 海拔增益, 损失, 极差, 时刻sin, 时刻cos, 历史均时长, 历史均速度]

# POI 路径与类型映射：统一从根目录 config.py 取，避免重复定义
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import POI_JSON, POI_TYPE_L1


# ── 工具 ──────────────────────────────────────────────────
def _haversine_meters(lon1, lat1, lon2, lat2):
    R = 6371000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _load_poi_profiles(scenery_name):
    """
    从 poi_path_projected.json 加载指定景区的 POI 投影，
    返回 dict: region_id -> {type_name: proportion}
    """
    if not os.path.exists(POI_JSON):
        return {}

    with open(POI_JSON, encoding="utf-8") as f:
        all_pois = json.load(f)

    # 按 (scenery, region_id) 分组
    by_key = defaultdict(list)
    for p in all_pois:
        key = (p["scenery"], p["projected_region_id"])
        by_key[key].append(p)

    profiles = {}
    for (sc, rid), plist in by_key.items():
        if sc != scenery_name:
            continue
        counter = Counter()
        for p in plist:
            l1 = p.get("type_code", "")[:2]
            name = POI_TYPE_L1.get(l1, l1)
            counter[name] += 1
        total = sum(counter.values())
        if total > 0:
            profiles[int(rid)] = {k: round(v / total, 4) for k, v in counter.most_common()}
        else:
            profiles[int(rid)] = {}
    return profiles


# ── 主函数 ────────────────────────────────────────────────
def extract_first_occurrence_sequences(df, scenery_name=None):
    """
    按 trackId 提取区域首次出现序列 + 段时长 + 特征（无泄露）。

    参数:
        df: 聚类后的 DataFrame（列: 经度,纬度,海拔,速度(km/h),is_stop,trackId,时间_秒,region_id）
        scenery_name: 景区名（用于加载 POI 特征），可选

    返回:
        records: list of dict, 每条轨迹包含:
            - track_id, route_id, seq, segment_durations
            - arrival_offsets, start_time_of_day
            - gps_seg_features: list[list[float]], 每段 8 维
            - poi_profiles: list[dict], 每个区域的 POI 类型分布
    """
    poi_profiles = _load_poi_profiles(scenery_name) if scenery_name else {}

    records = []
    skipped_toxic = 0
    for track_id, group in df.groupby('trackId'):
        group = group.sort_values('时间_秒')
        regions = group['region_id'].values
        times = group['时间_秒'].values
        if times.max() > 1e10:
            times = times / 1000.0
        lons = group['经度'].values
        lats = group['纬度'].values
        elevs = group['海拔'].values
        route_id = group['route_id'].iloc[0]

        # 首次出现序列（去重）
        seen = set()
        seq = []
        seg_start_indices = []
        region_lons_list = []
        region_lats_list = []
        region_elevs_list = []
        arrival_times = []

        for i, r in enumerate(regions):
            if r not in seen:
                seen.add(r)
                seq.append(int(r))
                seg_start_indices.append(i)
                region_lons_list.append(lons[i])
                region_lats_list.append(lats[i])
                region_elevs_list.append(elevs[i])
                arrival_times.append(times[i])

        if len(seq) < 2:
            continue

        t0 = arrival_times[0]
        arrival_offsets = [t - t0 for t in arrival_times]
        total_duration = arrival_offsets[-1]

        segment_durations = [
            arrival_offsets[i] - arrival_offsets[i - 1]
            for i in range(1, len(arrival_offsets))
        ]

        max_seg = max(segment_durations)
        if max_seg > MAX_SEGMENT_SECONDS or total_duration <= 0:
            skipped_toxic += 1
            continue

        start_hour = (t0 % 86400) / 3600.0

        # --- 构建段特征（无泄露） ---
        total_segs = len(seq) - 1
        gps_seg_feats = []
        hist_durations = []   # 历史段时长（用于计算历史均值）
        hist_distances = []   # 历史段静态距离

        for j in range(total_segs):
            r_from = seq[j]
            r_to = seq[j + 1]
            s_idx = seg_start_indices[j]
            e_idx = seg_start_indices[j + 1]

            # 段内 GPS 点（只用于海拔、路径距离）
            seg_lons = lons[s_idx:e_idx + 1]
            seg_lats = lats[s_idx:e_idx + 1]
            seg_elevs = elevs[s_idx:e_idx + 1]

            # 1) 静态路径距离：区域锚点间直线距离（不是 GPS 累积路径）
            path_dist_static = _haversine_meters(
                region_lons_list[j], region_lats_list[j],
                region_lons_list[j + 1], region_lats_list[j + 1]
            )

            # 2-4) 海拔特征（静态地理，不含游客因素）
            elev_diffs = np.diff(seg_elevs)
            elev_gain = float(elev_diffs[elev_diffs > 0].sum())
            elev_loss = float(abs(elev_diffs[elev_diffs < 0].sum()))
            elev_range = float(seg_elevs.max() - seg_elevs.min())

            # 5-6) 时间特征
            seg_hour = (times[s_idx] % 86400) / 3600.0
            seg_sin = math.sin(2.0 * math.pi * seg_hour / 24.0)
            seg_cos = math.cos(2.0 * math.pi * seg_hour / 24.0)

            # 7) 历史平均时长（前面段的均值）
            hist_avg_dur = float(np.mean(hist_durations)) if hist_durations else 0.0

            # 8) 历史平均速度（前面段的总距离 / 总时长）
            if hist_durations and sum(hist_durations) > 0:
                hist_avg_spd = sum(hist_distances) / sum(hist_durations)
            else:
                hist_avg_spd = 0.0

            gps_seg_feats.append([
                round(path_dist_static, 1),
                elev_gain, elev_loss, elev_range,
                seg_sin, seg_cos,
                round(hist_avg_dur, 1),
                round(hist_avg_spd, 5),
            ])

            # 填入历史数据（供下段使用）
            hist_durations.append(segment_durations[j])
            hist_distances.append(path_dist_static)

        # --- 构建 POI 语义特征 ---
        seq_poi_profiles = []
        for rid in seq:
            if rid in poi_profiles:
                seq_poi_profiles.append(poi_profiles[rid])
            else:
                seq_poi_profiles.append({})

        records.append({
            'track_id': int(track_id),
            'route_id': int(route_id),
            'seq': seq,
            'segment_durations': segment_durations,
            'arrival_offsets': [0.0] + list(np.cumsum(segment_durations)),
            'start_time_of_day': start_hour,
            'region_lats': region_lats_list,
            'region_lons': region_lons_list,
            'region_elevs': region_elevs_list,
            'gps_seg_features': gps_seg_feats,
            'poi_profiles': seq_poi_profiles,
            'total_duration': total_duration,
        })

    if skipped_toxic > 0:
        print(f"  [data] Filtered {skipped_toxic} toxic trajectories (segment > {MAX_SEGMENT_SECONDS//3600}h)")
    return records


# ── 区域元数据 ────────────────────────────────────────────
def build_region_metadata(df, poi_profiles=None):
    """
    构建区域静态特征 + KMeans 角色标签。
    新增：若有 POI 数据，将 POI 类型分布作为区域特征。
    """
    region_meta = {}
    for rid in sorted(df['region_id'].unique()):
        mask = df['region_id'] == rid
        meta = {
            'lat': float(df.loc[mask, '纬度'].mean()),
            'lon': float(df.loc[mask, '经度'].mean()),
            'elev_mean': float(df.loc[mask, '海拔'].mean()),
            'elev_std': float(df.loc[mask, '海拔'].std()),
            'visit_count': int(df.loc[mask & (df['is_stop'] == 1), 'trackId'].nunique()),
        }
        # 添加 POI 类型分布
        if poi_profiles and int(rid) in poi_profiles:
            meta['poi_profile'] = poi_profiles[int(rid)]
        else:
            meta['poi_profile'] = {}
        region_meta[int(rid)] = meta

    for rid in region_meta:
        mask = df['region_id'] == rid
        stop_mask = mask & (df['is_stop'] == 1)
        region_meta[rid]['avg_stop_dur'] = 0.0
        if stop_mask.sum() > 1:
            stop_times = df.loc[stop_mask, '时间_秒'].values
            if stop_times.max() > 1e10:
                stop_times = stop_times / 1000.0
            time_diffs = np.diff(stop_times)
            region_meta[rid]['avg_stop_dur'] = float(np.median(time_diffs[time_diffs < 3600]))

    # KMeans 角色标签（基于海拔 + 平均停靠时长）
    rids = sorted(region_meta.keys())
    if len(rids) >= 3:
        feats = np.array([
            [region_meta[rid]['elev_mean'], region_meta[rid]['avg_stop_dur']]
            for rid in rids
        ])
        feats_norm = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
        km = KMeans(n_clusters=min(3, len(rids)), random_state=42, n_init=10)
        labels = km.fit_predict(feats_norm)
        for i, rid in enumerate(rids):
            region_meta[rid]['role_id'] = int(labels[i])
    else:
        for i, rid in enumerate(rids):
            region_meta[rid]['role_id'] = i

    print(f"  [region] KMeans role labels: "
          + ", ".join(f"R{rid}={region_meta[rid]['role_id']}" for rid in rids))
    return region_meta


# ── 数据集划分 ────────────────────────────────────────────
def split_by_route(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.RandomState(seed)
    route_groups = {}
    for r in records:
        route_groups.setdefault(r['route_id'], []).append(r)

    train, val, test = [], [], []
    for rid, group in route_groups.items():
        n = len(group)
        indices = list(range(n))
        rng.shuffle(indices)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        for i in indices[:n_train]:
            train.append(group[i])
        for i in indices[n_train:n_train + n_val]:
            val.append(group[i])
        for i in indices[n_train + n_val:]:
            test.append(group[i])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test
