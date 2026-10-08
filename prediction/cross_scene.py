"""
跨场景到达时间迁移（E3b）：
- 深度模型在 7 个景区上训练（区域原型共享词表），零样本测第 8 个景区（不用它的标签）
- 对比：XGBoost 每场景重训（用了目标景区标签）、kNN 每场景
- 卖点：深度模型"跨场景可迁移、省标签"

设计：
1. 对每个景区跑严格数据流（ar_pipeline），得到 region_meta（含 geo/POI）与 records
2. 用 7 个训练景区的区域原型（经纬度+海拔+POI 分布）建共享 KDTree
3. 目标景区区域 → 最近原型 id（场景无关的连续特征映射）
4. 深度模型在 7 景区 records 上训练（映射到共享词表），目标景区零样本预测时长
5. 对比 XGBoost(目标景区 train 重训) 与 kNN(目标景区 train)
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cluster')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'poi')))
from config import CLEANED_DIR
import ar_pipeline as ap
from ar_model import build_normalizers, create_dataloader, rel_l1, POI_DIM
from transformer_model import TrajectoryPredictor, build_pair_mean
from run_transformer_experiment import _time_loss, split_subtrain_val
from transformer_eval import teacher_forced_eval
import ar_baselines as ab
import xgboost_baseline as xb

SCENES = ['峨眉山', '武侯祠博物馆', '熊猫基地', '都江堰', '锦江', '青城山', '黄龙溪', '龙泉']


CACHE_DIR = os.path.join(os.path.dirname(__file__), 'output', 'cache')
CACHE_VERSION = 'v2'


def scene_data(name, seed=42, n_workers=8):
    """返回 region_meta + train records + test records（严格无泄漏，带缓存）。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f'cs_{name}_{seed}_{CACHE_VERSION}.pkl')
    if os.path.exists(cp):
        import pickle
        with open(cp, 'rb') as f:
            return pickle.load(f)
    csv = str(CLEANED_DIR / f'{name}_cleaned.csv')
    df = pd.read_csv(csv)
    dtr, dte = ap.split_tracks(df, 0.7, seed)
    npr = ap.n_poi_regions_static(name)
    model_reg, dtr = ap.cluster_regions_train(dtr, n_poi_regions=npr, seed=seed, n_workers=n_workers)
    dte, _ = ap.map_test_regions(dte, model_reg)
    dtr, t2r, _ = ap.discover_train_routes(dtr, n_workers=n_workers)
    reps = ap._route_representatives(dtr, t2r)
    dte, _ = ap.assign_test_route_pseudolabels(dte, reps)
    profiles = ap.build_poi_profiles(model_reg, name)
    region_meta = ap.build_region_meta_train(dtr, model_reg, profiles)
    tr_records = ap.build_records(dtr, region_meta, name, model_reg)
    te_records = ap.build_records(dte, region_meta, name, model_reg,
                                  track_route_map=dict(zip(dte['trackId'], dte['route_id'])))
    out = (region_meta, tr_records, te_records)
    import pickle
    with open(cp, 'wb') as f:
        pickle.dump(out, f)
    return out


def region_proto(region_meta):
    """区域原型：lat/lon/elev + POI 分布（场景无关连续特征）。"""
    X = []
    ids = []
    for rid in sorted(region_meta.keys()):
        m = region_meta[rid]
        v = [m.get('lat', 0), m.get('lon', 0), m.get('elev_mean', 0)]
        for k in ['餐饮', '休闲', '住宿', '风景名胜', '科教文化', '公共设施']:
            v.append(m.get('poi_profile', {}).get(k, 0.0))
        X.append(v)
        ids.append(rid)
    return np.array(X, dtype=np.float32), ids


def map_to_prototypes(region_meta, proto_X):
    """目标区域 → 最近共享原型 id。"""
    tree = cKDTree(proto_X)
    X, _ = region_proto(region_meta)
    _, idx = tree.query(X, k=1)
    return dict(zip(sorted(region_meta.keys()), idx.tolist()))


def remap_records(records, rid_map):
    """把 records 的区域 token 映射到共享原型词表；route_id 归零（路线不跨场景）。"""
    out = []
    for r in records:
        rr = dict(r)
        rr['seq'] = [rid_map.get(int(x), 0) for x in r['seq']]
        rr['route_id'] = 0
        out.append(rr)
    return out


def run_loo(target, seed=42, n_workers=8, epochs=60):
    sources = [s for s in SCENES if s != target]
    from sklearn.cluster import KMeans
    # 1) 训练景区数据 + 原型空间（KMeans 归并到 P 个共享原型）
    src_data = {}
    proto_Xs = []
    for s in sources:
        meta, tr, te = scene_data(s, seed, n_workers)
        src_data[s] = (meta, tr)
        X, _ = region_proto(meta)
        proto_Xs.append(X)
    proto_X = np.vstack(proto_Xs)
    P = min(64, len(proto_X))
    km = KMeans(n_clusters=P, random_state=seed, n_init=10)
    km.fit(proto_X)
    shared_centers = km.cluster_centers_
    shared_meta = {p: {'lat': float(c[0]), 'lon': float(c[1]), 'elev_mean': float(c[2]),
                       'poi_profile': {}, 'role_id': 0} for p, c in enumerate(shared_centers)}

    # 2) 每个景区的区域 → 最近共享原型
    def _map_meta(meta):
        X, _ = region_proto(meta)
        _, idx = cKDTree(shared_centers).query(X, k=1)
        return dict(zip(sorted(meta.keys()), idx.tolist()))

    pooled_records = []
    for s in sources:
        meta, tr = src_data[s]
        pooled_records.extend(remap_records(tr, _map_meta(meta)))

    # 3) 目标景区数据 + 映射到共享原型
    t_meta, t_tr, t_te = scene_data(target, seed, n_workers)
    t_map = _map_meta(t_meta)
    te_records = remap_records(t_te, t_map)

    # 4) 深度模型在 pooled 训练，目标景区零样本测试
    norm = build_normalizers(pooled_records, shared_meta)
    V = P
    R = 2
    pair_mean, gmean = build_pair_mean(pooled_records, V)
    geo_dim = 3 + POI_DIM
    _tracks = sorted({int(r['track_id']) for r in pooled_records})
    _rng = np.random.RandomState(seed); _rng.shuffle(_tracks)
    _nva = max(int(len(_tracks) * 0.15), 1)
    _va = set(_tracks[:_nva])
    tr_recs = [r for r in pooled_records if int(r['track_id']) not in _va]
    va_recs = [r for r in pooled_records if int(r['track_id']) in _va]
    train_dl, _ = create_dataloader(tr_recs, shared_meta, norm, batch_size=32, shuffle=True, max_len=8)
    val_dl, _ = create_dataloader(va_recs, shared_meta, norm, batch_size=32, shuffle=False, max_len=8)
    test_dl, _ = create_dataloader(te_records, shared_meta, norm, batch_size=32, shuffle=False, max_len=8)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrajectoryPredictor(num_regions=V, num_routes=R, geo_dim=geo_dim, pair_mean=pair_mean,
                                global_mean=gmean, seg_p95=norm['seg_p95'], backbone='transformer',
                                d_model=64, use_struct=True, use_prior=True, time_feats='full').to(dev)
    from run_transformer_experiment import train_model
    model = train_model(model, train_dl, val_dl, dev, epochs=epochs, seed=seed, loss='mae')
    tf = teacher_forced_eval(model, test_dl, dev, norm['seg_p95'])

    # 5) XGBoost 每场景重训（用目标景区标签）+ kNN 每场景
    V_t = max([max(r['seq']) for r in t_tr] + [0]) + 1
    xgb_r = xb.run_xgb(t_tr, t_te, V_t, seed=seed)
    km_mae, _ = ab.knn_duration_mae(*ab.build_duration_model(t_tr), t_te)

    res = {
        'target': target, 'n_train_pooled': len(pooled_records), 'n_test': len(te_records),
        'deep_transfer_seg_mae_min': round(tf['dur_mae_min'], 2),
        'deep_transfer_cum_mae_min': round(tf.get('cum_mae_min', 0.0), 2),
        'deep_transfer_w30': round(tf.get('cum_window_acc', {}).get('30', 0), 4),
        'xgb_per_scene_seg_mae_min': round(xgb_r['mae_min'], 2) if xgb_r else None,
        'xgb_per_scene_cum_mae_min': round(xgb_r['cum_mae_min'], 2) if xgb_r and xgb_r.get('cum_mae_min') else None,
        'knn_per_scene_seg_mae_min': round(km_mae / 60, 2),
    }
    print(f"[{target}] deep-transfer seg={res['deep_transfer_seg_mae_min']}min cum={res['deep_transfer_cum_mae_min']}min "
          f"w30={res['deep_transfer_w30']} | XGB-per-scene seg={res['xgb_per_scene_seg_mae_min']}min "
          f"cum={res['xgb_per_scene_cum_mae_min']}min | kNN={res['knn_per_scene_seg_mae_min']}min")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', nargs='*', default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--seeds', type=int, nargs='*', default=[42])
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--out', default='prediction/output/llm_cross_scene.json')
    args = p.parse_args()

    targets = args.target or (SCENES if args.all else [])
    results = {}
    for t in targets:
        for seed in args.seeds:
            try:
                results[f'{t}|{seed}'] = run_loo(t, seed, args.n_workers, args.epochs)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[f'{t}|{seed}'] = {'error': str(e)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果已保存: {args.out}')


if __name__ == '__main__':
    main()
