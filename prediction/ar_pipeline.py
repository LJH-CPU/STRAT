"""
严格无泄漏数据流（自回归预测用）

核心协议：先划分 → 一切无监督/统计只在 train 内做 → test 只映射与推断。

步骤：
  0. 按 trackId 种子随机 70/30 划分（轨迹不跨集）
  1. 在 train 轨迹上跑粒球+谱聚类 → 区域定义（区域质心）
  2. test 点 KDTree 映射到最近 train 区域
  3. 在 train 轨迹上做路线发现 → train route_id
  4. test 轨迹匹配最近 train 路线 → route 伪标签
  5. region_meta（visit_count/avg_stop_dur/role/geo 统计）只统计 train
  6. POI 归属 = 最近 train 区域质心（静态外部数据）
  7. records 特征/目标在各自集合内构造（历史均时长/速度=轨迹自身历史）

用法：本模块只负责数据，不负责训练。
"""

import os
import sys
import math
import json
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cluster')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'poi')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data-project')))
from config import POI_JSON, POI_TYPE_L1
from dem_lookup import DEM
from scenery_route_clustering import (
    create_ball_dict,
    calculate_radius,
    perform_clustering,
    perform_clustering_eigengap,
    discover_routes,
)
from cluster_search import _best_grid_balls, _validation_subset

MAX_SEGMENT_SECONDS = 4 * 3600
# [时刻sin, 时刻cos]（严格无泄漏：仅预测原点可观测的当前区域进入时刻）
# 段内未来路径地形（gain/loss/range）与真值下一区域距离都不进入输入；
# 区域对时长先验由模型用“预测的”下一区域计算。
GPS_SEG_DIM = 2

_DEM = None


def _dem():
    global _DEM
    if _DEM is None:
        _DEM = DEM()
    return _DEM


def _dem_fill(lats, lons):
    """DEM 高程，nan 沿索引线性插值填补。"""
    z = _dem().elev(lats, lons)
    ok = np.isfinite(z)
    if not ok.all():
        if ok.any():
            z = np.interp(np.arange(len(z)), np.where(ok)[0], z[ok])
        else:
            z = np.zeros_like(z)
    return z


# ---------------------------------------------------------------------------
# 0) 划分
# ---------------------------------------------------------------------------

def split_tracks(df, train_ratio=0.7, seed=42):
    """按 trackId 随机划分（轨迹不跨集）。返回 df_train, df_test。"""
    rng = np.random.RandomState(seed)
    tracks = df['trackId'].unique()
    rng.shuffle(tracks)
    n_train = int(len(tracks) * train_ratio)
    train_tracks = set(tracks[:n_train])
    test_tracks = set(tracks[n_train:])
    return df[df['trackId'].isin(train_tracks)].copy(), df[df['trackId'].isin(test_tracks)].copy()


# ---------------------------------------------------------------------------
# 地理/缩放工具
# ---------------------------------------------------------------------------

def geo_scale_params(lons, lats):
    lat_mean_rad = np.radians(lats.mean())
    lon_scale = np.cos(lat_mean_rad) * 111320.0
    lat_scale = 111320.0
    return lon_scale, lat_scale


def to_meters(df, lon_scale, lat_scale):
    lon = df['经度'].values.astype(np.float64) * lon_scale
    lat = df['纬度'].values.astype(np.float64) * lat_scale
    elev = df['海拔'].values.astype(np.float64)
    return np.column_stack([lon, lat, elev])


# ---------------------------------------------------------------------------
# 1) train 内聚类 → 区域模型
# ---------------------------------------------------------------------------

def _balanced_split(points):
    """主轴中位数二分（均衡），避免最远点对分配在稠密+离群分布下的不平衡。"""
    mu = points.mean(axis=0)
    Xc = points - mu
    cov = Xc.T @ Xc / len(points)
    w, v = np.linalg.eigh(cov)
    dir_vec = v[:, -1]
    proj = Xc @ dir_vec
    med = np.median(proj)
    a = proj <= med
    return points[a], points[~a]


def generate_balls_balanced(features, min_points=20, rng=None):
    """
    预测用粒球生成（平衡二分，不做邻近合并）。
    递归按主轴中位数切分到粒度 < min_points；再按半径阈值切分超大球。
    """
    rng = rng or np.random
    balls = [features]
    min_after = max(2, min_points // 2)

    def _split_all(ball_list):
        out = []
        for b in ball_list:
            if len(b) < min_points:
                out.append(b)
                continue
            c1, c2 = _balanced_split(b)
            if len(c1) >= min_after and len(c2) >= min_after:
                out.extend([c1, c2])
            else:
                out.append(b)
        return out

    for _ in range(60):
        before = len(balls)
        balls = _split_all(balls)
        if len(balls) == before:
            break

    radii = [calculate_radius(b) for b in balls if len(b) >= 2]
    det = max(np.median(radii) if radii else 0.0, np.mean(radii) if radii else 0.0, 1e-6)
    for _ in range(60):
        new = []
        for b in balls:
            if len(b) < 2 or calculate_radius(b) <= 2.0 * det:
                new.append(b)
                continue
            c1, c2 = _balanced_split(b)
            if len(c1) > 0:
                new.append(c1)
            if len(c2) > 0:
                new.append(c2)
        if len(new) == len(balls):
            break
        balls = new

    return [b for b in balls if len(b) > 0]


def cluster_regions_train(df_train, n_poi_regions=None, seed=42, min_points=20,
                          sample_size=5000, n_workers=1, force_k=None):
    """
    只在 train 停留点上做粒球+谱聚类，返回区域模型。
    区域模型 = {scaler, centroids_scaled, lon_scale, lat_scale, ball_centers, ball_radii, k}
    force_k: 若给定，强制 k（用于聚类粒度消融）。
    """
    rng = np.random.RandomState(seed)
    stay = df_train[df_train['is_stop'] == 1].copy()
    lon_scale, lat_scale = geo_scale_params(stay['经度'].values, stay['纬度'].values)
    feats = to_meters(stay, lon_scale, lat_scale)

    n_stay = len(feats)
    sample = min(sample_size, n_stay)
    idx = np.arange(n_stay)
    if n_stay > sample:
        idx = np.sort(rng.choice(n_stay, sample, replace=False))
        feats = feats[idx]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(feats)

    balls = generate_balls_balanced(X, min_points=min_points)
    bd = create_ball_dict(balls)
    keys = list(bd.keys())
    centers = np.array([bd[k].center for k in keys])
    radii = np.array([bd[k].radius for k in keys])
    n_balls = len(bd)

    k_lower, k_upper = _pred_k_bounds(n_poi_regions, n_balls)
    if force_k is not None:
        k_lower = k_upper = max(2, min(int(force_k), n_balls))
    min_cb = max(8, n_balls // 16)

    sel_idx = _validation_subset(X, seed)
    if force_k is not None:
        # 固定 k：单层谱聚类 + 网格 δ（验证子集选参）
        from sklearn.metrics import silhouette_score
        best_sil, best_delta = -1.0, 0.5
        for d in np.arange(0.1, 1.0 + 1e-9, 0.1):
            lab = perform_clustering(centers, radii, X, k_lower, float(d))
            if np.all(lab == -1) or len(np.unique(lab)) < 2:
                continue
            s = silhouette_score(X[sel_idx], lab[sel_idx])
            if s > best_sil:
                best_sil, best_delta = s, float(d)
        labels = perform_clustering(centers, radii, X, k_lower, best_delta)
        delta, k = best_delta, k_lower
    else:
        labels, delta, k, _ = _best_grid_balls(centers, radii, X, k_lower, k_upper, min_cb, sel_idx, n_workers)

    # 区域质心（scaled 空间，只取有标签的点）
    centroids = []
    for c in range(k):
        m = labels == c
        if m.sum() > 0:
            centroids.append(X[m].mean(axis=0))
        else:
            centroids.append(centers[c])
    centroids = np.array(centroids)

    model = {
        'scaler': scaler,
        'centroids_scaled': centroids,
        'lon_scale': lon_scale,
        'lat_scale': lat_scale,
        'n_balls': n_balls,
        'delta': delta,
        'k': k,
        'train_stay_sampled_idx': idx,
        'min_points': min_points,
    }
    # 把 train 停留点标签写回（未采样点用最近质心映射）
    all_X = scaler.transform(to_meters(stay, lon_scale, lat_scale))
    all_labels = _map_points(all_X, centroids)
    stay = stay.copy()
    stay['region_id'] = all_labels
    df_train = df_train.copy()
    stay_idx = df_train['is_stop'] == 1
    df_train.loc[stay_idx, 'region_id'] = all_labels
    # 移动点 → 最近区域质心
    df_train = _assign_moving(df_train, scaler, lon_scale, lat_scale, centroids)
    df_train['region_id'] = df_train['region_id'].astype(int)
    return model, df_train


def _pred_k_bounds(n_poi_regions, n_balls):
    """预测用 k 范围。有 POI 时以其为引导；无 POI 时数据驱动（停留点粒球数定范围）：
    无 POI 景区（本数据 18/20）k 退化为 4-12 会导致区域过粗、records 大量丢失，
    故改用 n_balls 推导 k ∈ [15, 40]，与有 POI 景区（峨眉山 k≈29）对齐。"""
    if n_poi_regions and n_poi_regions >= 3:
        lower = max(3, int(n_poi_regions) - 2)
        upper = min(n_balls, max(12, int(n_poi_regions) + 4))
    else:
        lower = max(12, n_balls // 20)
        upper = min(n_balls, max(40, n_balls // 8))
    upper = max(upper, lower + 1)
    upper = min(upper, n_balls)
    lower = max(3, min(lower, upper - 1))
    return int(lower), int(upper)


def n_poi_regions_static(scenery_name, buffer_m=120):
    """静态 POI 缓冲区合并区域数（只用 POI 坐标，不涉及轨迹，无泄漏）。"""
    pois = load_pois(scenery_name)
    if len(pois) < 3:
        return None
    coords = np.array([[p.get('original_lon', p.get('projected_lon')),
                        p.get('original_lat', p.get('projected_lat'))] for p in pois], dtype=np.float64)
    from ground_truth import union_find
    ids = union_find(coords, buffer_m / 111000.0)
    n = len(set(int(i) for i in ids)) if len(ids) else 0
    return max(1, n)


def _map_points(X_scaled, centroids):
    tree = cKDTree(centroids)
    _, idx = tree.query(X_scaled, k=1)
    return idx


def _assign_moving(df, scaler, lon_scale, lat_scale, centroids):
    mov = df['is_stop'] == 0
    if mov.sum() == 0:
        return df
    Xm = scaler.transform(to_meters(df.loc[mov], lon_scale, lat_scale))
    labels = _map_points(Xm, centroids)
    df = df.copy()
    df.loc[mov, 'region_id'] = labels
    return df


# ---------------------------------------------------------------------------
# 2) test 点映射
# ---------------------------------------------------------------------------

def map_test_regions(df_test, model):
    """把 test 点映射到最近 train 区域。返回带 region_id 的 df_test + 距离统计。"""
    df = df_test.copy()
    X = scaler_transform_pts(df, model)
    tree = cKDTree(model['centroids_scaled'])
    dist, idx = tree.query(X, k=1)
    df['region_id'] = idx
    df['region_dist'] = dist
    df['region_id'] = df['region_id'].astype(int)
    return df, {'p50': float(np.median(dist)), 'p90': float(np.percentile(dist, 90)),
                'max': float(dist.max())}


def scaler_transform_pts(df, model):
    lon = df['经度'].values.astype(np.float64) * model['lon_scale']
    lat = df['纬度'].values.astype(np.float64) * model['lat_scale']
    elev = df['海拔'].values.astype(np.float64)
    return model['scaler'].transform(np.column_stack([lon, lat, elev]))


# ---------------------------------------------------------------------------
# 3/4) 路线发现（train 内）+ test 伪标签
# ---------------------------------------------------------------------------

def discover_train_routes(df_train, n_workers=1):
    df = df_train.copy()
    df, route_metrics, track_to_route, valid_tracks = discover_routes(
        df, n_workers=n_workers, scenery_name=None)
    df['route_id'] = df['trackId'].map(track_to_route).fillna(-1).astype(int)
    return df, track_to_route, route_metrics


def _seq_of_track(df):
    g = df.sort_values('时间_秒')
    seq = []
    prev = None
    for r in g['region_id'].values:
        if r != prev:
            seq.append(int(r))
            prev = r
    return tuple(seq)


def _route_representatives(df_train, track_to_route):
    """每条 train 路线 → 代表性 region 序列（出现最多的序列）。"""
    reps = {}
    for tid, route in track_to_route.items():
        tdf = df_train[df_train['trackId'] == tid]
        if len(tdf) == 0:
            continue
        seq = _seq_of_track(tdf)
        reps.setdefault(route, []).append(seq)
    final = {}
    for route, seqs in reps.items():
        cnt = Counter(seqs)
        final[route] = cnt.most_common(1)[0][0]
    return final


def assign_test_route_pseudolabels(df_test, route_reps):
    """test 轨迹 → 最近 train 路线（编辑距离到代表序列）。"""
    df = df_test.copy()
    track_seqs = {}
    for tid, g in df.groupby('trackId'):
        track_seqs[tid] = _seq_of_track(g)
    route_ids = list(route_reps.keys())
    rep_seqs = [route_reps[r] for r in route_ids]

    def _ed(a, b):
        m, n = len(a), len(b)
        if m == 0 or n == 0:
            return max(m, n)
        dp = np.zeros((m + 1, n + 1))
        dp[:, 0] = np.arange(m + 1)
        dp[0, :] = np.arange(n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
        return dp[m, n] / max(m, n)

    pseudo = {}
    best_d = {}
    for tid, seq in track_seqs.items():
        dists = [_ed(seq, rs) for rs in rep_seqs]
        bi = int(np.argmin(dists))
        pseudo[tid] = route_ids[bi]
        best_d[tid] = float(dists[bi])
    df['route_id'] = df['trackId'].map(pseudo).fillna(-1).astype(int)
    df['route_match_dist'] = df['trackId'].map(best_d).fillna(1.0)
    matched = float(np.mean([d < 0.5 for d in best_d.values()])) if best_d else 0.0
    return df, {'n_tracks': len(pseudo), 'mean_best_dist': float(np.mean(list(best_d.values()))) if best_d else None,
                'coverage_lt0.5': matched}


# ---------------------------------------------------------------------------
# 5/6) region_meta + POI 归属（train 统计）
# ---------------------------------------------------------------------------

def load_pois(scenery_name):
    """加载该景区 POI（静态外部数据），返回 list[dict]。"""
    if not os.path.exists(POI_JSON):
        return []
    with open(POI_JSON, encoding='utf-8') as f:
        all_pois = json.load(f)
    return [p for p in all_pois if p.get('scenery') == scenery_name]


def map_pois_to_regions(pois, model):
    """POI(lon,lat) → 最近 train 区域。返回 (poi_region_ids, poi_type_codes)。"""
    if not pois:
        return np.array([], dtype=int), []
    # 用静态原始 POI 坐标（AMap），不用投影到轨迹的坐标（后者为 transductive）
    lons = np.array([p.get('original_lon', p.get('projected_lon')) for p in pois], dtype=np.float64)
    lats = np.array([p.get('original_lat', p.get('projected_lat')) for p in pois], dtype=np.float64)
    elev = np.zeros(len(pois))
    lon = lons * model['lon_scale']
    lat = lats * model['lat_scale']
    X = model['scaler'].transform(np.column_stack([lon, lat, elev]))
    tree = cKDTree(model['centroids_scaled'])
    _, idx = tree.query(X, k=1)
    types = [p.get('type_code', '')[:2] for p in pois]
    return idx, types


def build_poi_profiles(model, scenery_name):
    """每区域 POI 类型分布（用我们的区域归属）。返回 {region_id: {type: proportion}}。"""
    pois = load_pois(scenery_name)
    if not pois:
        return {}
    rid, types = map_pois_to_regions(pois, model)
    by_region = defaultdict(list)
    for r, t in zip(rid, types):
        by_region[int(r)].append(POI_TYPE_L1.get(t, t))
    profiles = {}
    for r, tl in by_region.items():
        cnt = Counter(tl)
        total = sum(cnt.values())
        profiles[r] = {k: round(v / total, 4) for k, v in cnt.items()}
    return profiles


def build_region_meta_train(df_train, model, poi_profiles=None, n_roles=3):
    """区域统计只在 train 上算。"""
    region_meta = {}
    for rid in sorted(df_train['region_id'].unique()):
        mask = df_train['region_id'] == rid
        meta = {
            'lat': float(df_train.loc[mask, '纬度'].mean()),
            'lon': float(df_train.loc[mask, '经度'].mean()),
            'elev_mean': float(df_train.loc[mask, '海拔'].mean()),
            'elev_std': float(df_train.loc[mask, '海拔'].std()),
            'visit_count': int(df_train.loc[mask & (df_train['is_stop'] == 1), 'trackId'].nunique()),
        }
        stop_mask = mask & (df_train['is_stop'] == 1)
        meta['avg_stop_dur'] = 0.0
        if stop_mask.sum() > 1:
            st = df_train.loc[stop_mask, '时间_秒'].values.astype(np.float64)
            if st.max() > 1e10:
                st = st / 1000.0
            diffs = np.diff(np.sort(st))
            diffs = diffs[diffs < 3600]
            if len(diffs) > 0:
                meta['avg_stop_dur'] = float(np.median(diffs))
        meta['poi_profile'] = poi_profiles.get(int(rid), {}) if poi_profiles else {}
        region_meta[int(rid)] = meta

    rids = sorted(region_meta.keys())
    if len(rids) >= 3:
        feats = np.array([[region_meta[r]['elev_mean'], region_meta[r]['avg_stop_dur']] for r in rids])
        fn = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
        km = KMeans(n_clusters=min(n_roles, len(rids)), random_state=42, n_init=10)
        lab = km.fit_predict(fn)
        for i, r in enumerate(rids):
            region_meta[r]['role_id'] = int(lab[i])
    else:
        for i, r in enumerate(rids):
            region_meta[r]['role_id'] = i
    return region_meta


# ---------------------------------------------------------------------------
# 7) records（各自集合内构造，无泄漏）
# ---------------------------------------------------------------------------

def _haversine(lon1, lat1, lon2, lat2):
    R = 6371000.0
    dlon = math.radians(lon2 - lon1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def build_records(df, region_meta, scenery_name, model, track_route_map=None):
    """
    从带 region_id 的 df 构建轨迹 records（无泄漏）。
    track_route_map: trackId→route_id（test 传伪标签映射）。
    """
    if track_route_map is None:
        track_route_map = dict(zip(df['trackId'], df['route_id']))
    poi_profiles_by_region = build_poi_profiles(model, scenery_name)
    records = []
    skipped = 0
    for track_id, group in df.groupby('trackId'):
        group = group.sort_values('时间_秒')
        regions = group['region_id'].values
        times = group['时间_秒'].values.astype(np.float64)
        if times.max() > 1e10:
            times = times / 1000.0
        route_id = int(track_route_map.get(track_id, -1))

        seen = set()
        seq = []
        seg_start = []
        r_lons, r_lats, arrival = [], [], []
        for i, r in enumerate(regions):
            if r not in seen:
                seen.add(r)
                seq.append(int(r))
                seg_start.append(i)
                r_lons.append(group['经度'].values[i])
                r_lats.append(group['纬度'].values[i])
                arrival.append(times[i])
        if len(seq) < 2:
            continue
        r_elevs = [float(z) for z in _dem_fill(np.asarray(r_lats), np.asarray(r_lons))]
        t0 = arrival[0]
        offsets = [t - t0 for t in arrival]
        total_dur = offsets[-1]
        seg_durs = [offsets[i] - offsets[i - 1] for i in range(1, len(offsets))]
        if max(seg_durs) > MAX_SEGMENT_SECONDS or total_dur <= 0:
            skipped += 1
            continue

        # 段特征（严格无泄漏）：只用预测原点可观测信息（当前区域进入时刻）。
        # 不使用段内未来路径地形，也不使用真值下一区域。
        total_segs = len(seq) - 1
        feats = []
        for j in range(total_segs):
            s = seg_start[j]
            hour = (times[s] % 86400) / 3600.0
            seg_sin = math.sin(2 * math.pi * hour / 24.0)
            seg_cos = math.cos(2 * math.pi * hour / 24.0)
            feats.append([seg_sin, seg_cos])

        records.append({
            'track_id': int(track_id),
            'route_id': route_id,
            'seq': seq,
            'segment_durations': seg_durs,
            'arrival_offsets': [0.0] + list(np.cumsum(seg_durs)),
            'start_time_of_day': (t0 % 86400) / 3600.0,
            'region_lats': r_lats,
            'region_lons': r_lons,
            'region_elevs': r_elevs,
            'gps_seg_features': feats,
            'poi_profiles': [poi_profiles_by_region.get(r, {}) for r in seq],
            'total_duration': total_dur,
        })
    return records


def build_normalizers(records):
    """归一化参数只在 train records 上算。"""
    gps = np.array([f for r in records for f in r['gps_seg_features']], dtype=np.float32)
    segs = np.array([d for r in records for d in r['segment_durations']], dtype=np.float64)
    norm = {
        'gps_mean': gps.mean(axis=0) if len(gps) else np.zeros(GPS_SEG_DIM),
        'gps_std': gps.std(axis=0) + 1e-6 if len(gps) else np.ones(GPS_SEG_DIM),
        'seg_p95': float(np.percentile(segs, 95)) if len(segs) else 3600.0,
    }
    lats = [m['lat'] for m in []]  # placeholder
    return norm


if __name__ == '__main__':
    print("ar_pipeline: strict data flow module")
