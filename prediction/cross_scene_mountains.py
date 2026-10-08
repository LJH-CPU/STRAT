#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨场景诊断·区域级 STRAT（合并论文 §4.2）：
在 4 个真实山区（2bulu 版）上，用 STRAT 框架（当前修正版：causal_edges, prob+校准）做
留一景区迁移诊断：
  - 同场景 STRAT（per-scene 自训）
  - 零样本 STRAT（其余 3 山 → 共享地理原型词表 → 目标山零样本）
  - 少样本 STRAT（目标山 5/10/25% 轨迹加入 pooled 训练）
  - 区域级物理特征 XGBoost（同场景 / 零样本 / 少样本，D1'/D4' 同层对比）
指标：next-region tf_acc、区域段时长 MAE(min)、路径级 PICP90 / path-CRPS、迁移效率。
严格无泄漏：聚类/路线/统计/归一化仅在训练集（池化或目标自身）内。
"""
import os
import sys
import json
import time
import pickle
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CLEANED_DIR
import ar_pipeline as ap
from ar_model import build_normalizers, create_dataloader, POI_DIM
from transformer_model import TrajectoryPredictor, build_pair_mean
from run_transformer_experiment import (
    train_model, build_bundle, split_subtrain_val, _calibrate_sigma, _calibrate_path_sigma, run_bundle)
from transformer_eval import teacher_forced_eval, rollout_eval

MOUNTAINS = None  # 延迟到 main() 由 --scenes / scenes_20 决定
SEEDS = [42, 100, 2024, 7, 17]
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'output', 'cache')
SEG_CSV = {}
OUT = os.path.join(os.path.dirname(__file__), 'output', 'cross_scene_mountains.json')


def _setup_scenes(scenes):
    global MOUNTAINS, SEG_CSV
    MOUNTAINS = list(scenes)
    SEG_CSV = {m: str(CLEANED_DIR / f'{m}_2bulu_cleaned.csv') for m in MOUNTAINS}


class Args:
    """与 run_transformer_experiment 默认一致的实验参数（当前修正版模型）。"""
    n_poi_regions = None
    n_workers = 8
    k_override = None
    max_len = 8
    interval = 0
    interval_bins = 8
    backbone = 'transformer'
    gpt2_ckpt = ''
    d_model = 64
    use_struct = 1
    use_prior = 1
    time_feats = 'full'
    prob = 1
    loss = 'nll'
    aggregator = 'rgcn'
    edge_mask = (True, True, True, True)
    causal_edges = 1
    epochs = 80
    verbose = False
    no_cal = False
    fs_ratios = [0.05, 0.10, 0.25]


def _cfg_key(args):
    """缓存键纳入影响 bundle 的配置，避免静默复用旧聚类。"""
    import hashlib
    s = (f"{args.n_poi_regions}|{args.k_override}|{args.max_len}|{args.time_feats}|"
         f"{args.causal_edges}|{tuple(args.edge_mask)}|{args.aggregator}|{args.use_struct}|"
         f"{args.use_prior}|{args.interval}|{args.interval_bins}|{args.backbone}")
    return hashlib.md5(s.encode()).hexdigest()[:8]


def cache_bundle(name, seed, args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = os.path.join(CACHE_DIR, f'csb_{name}_{seed}_{_cfg_key(args)}.pkl')
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            return pickle.load(f)
    b = build_bundle(name, SEG_CSV[name], seed, args)
    with open(cp, 'wb') as f:
        pickle.dump(b, f)
    return b


def region_proto(region_meta):
    X, ids = [], []
    for rid in sorted(region_meta.keys()):
        m = region_meta[rid]
        X.append([m.get('lat', 0), m.get('lon', 0), m.get('elev_mean', 0)])
        ids.append(rid)
    return np.array(X, dtype=np.float32), ids


def remap_records(records, rid_map):
    out = []
    for r in records:
        rr = dict(r)
        rr['seq'] = [rid_map.get(int(x), 0) for x in r['seq']]
        rr['route_id'] = 0
        out.append(rr)
    return out


def pooled_shared_space(bundles, k_proto=None):
    proto_X = np.vstack([region_proto(b['region_meta'])[0] for b in bundles])
    P = min(int(k_proto), len(proto_X)) if k_proto else min(64, len(proto_X))
    km = KMeans(n_clusters=P, random_state=42, n_init=10).fit(proto_X)
    centers = km.cluster_centers_
    shared_meta = {p: {'lat': float(c[0]), 'lon': float(c[1]), 'elev_mean': float(c[2]),
                       'poi_profile': {}, 'role_id': 0} for p, c in enumerate(centers)}
    tree = cKDTree(centers)

    def _map(meta):
        X, _ = region_proto(meta)
        _, idx = tree.query(X, k=1)
        return dict(zip(sorted(meta.keys()), idx.tolist()))

    return shared_meta, _map


def model_factory(records, shared_meta, norm, seed, args, dev):
    V = len(shared_meta)
    R = 2
    pair_mean, gmean = build_pair_mean(records, V)
    geo_dim = 3 + POI_DIM
    model = TrajectoryPredictor(
        num_regions=V, num_routes=R, geo_dim=geo_dim, pair_mean=pair_mean, global_mean=gmean,
        seg_p95=norm['seg_p95'], backbone=args.backbone, d_model=args.d_model,
        use_struct=(args.use_struct == 1), use_prior=(args.use_prior == 1),
        time_feats=args.time_feats, prob=True, aggregator=args.aggregator,
        edge_mask=args.edge_mask, causal_edges=True).to(dev)
    return model


def train_eval_shared(records, shared_meta, norm, target_records, seed, args, dev):
    _tracks = sorted({int(r['track_id']) for r in records})
    _rng = np.random.RandomState(seed); _rng.shuffle(_tracks)
    _nva = max(int(len(_tracks) * 0.15), 1)
    _va = set(_tracks[:_nva])
    tr_recs = [r for r in records if int(r['track_id']) not in _va]
    va_recs = [r for r in records if int(r['track_id']) in _va]
    train_dl, _ = create_dataloader(tr_recs, shared_meta, norm, batch_size=32, shuffle=True,
                                    max_len=args.max_len, num_workers=6)
    val_dl, _ = create_dataloader(va_recs, shared_meta, norm, batch_size=32, shuffle=False,
                                  max_len=args.max_len, num_workers=6)
    test_dl, _ = create_dataloader(target_records, shared_meta, norm, batch_size=32, shuffle=False,
                                   max_len=args.max_len, num_workers=6)
    model = model_factory(records, shared_meta, norm, seed, args, dev)
    model = train_model(model, train_dl, val_dl, dev, epochs=args.epochs, seed=seed, loss='nll')
    sc = _calibrate_sigma(model, val_dl, dev, norm['seg_p95'])
    ps = _calibrate_path_sigma(model, val_dl, dev, norm['seg_p95'])
    tf = teacher_forced_eval(model, test_dl, dev, norm['seg_p95'], path_sigma_cal=ps)
    return tf, sc, ps


def region_xgb(records_list, target_records, seed=42):
    """区域级物理特征 XGBoost：用 records 自带的 gps_seg_features（连续、无区域 id）。
    返回 (model, X_te, y_te)。"""
    import xgboost as xgb
    X_all, y_all = [], []
    for recs in records_list:
        for r in recs:
            for j, f in enumerate(r['gps_seg_features']):
                X_all.append(f)
                y_all.append(r['segment_durations'][j])
    X_tr = np.array(X_all, dtype=np.float32)
    y_tr = np.array(y_all, dtype=np.float32)
    X_te, y_te = [], []
    for r in target_records:
        for j, f in enumerate(r['gps_seg_features']):
            X_te.append(f)
            y_te.append(r['segment_durations'][j])
    X_te = np.array(X_te, dtype=np.float32)
    y_te = np.array(y_te, dtype=np.float32)
    m = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, objective='reg:absoluteerror',
                         n_jobs=8, early_stopping_rounds=30, random_state=seed)
    n_va = max(int(len(X_tr) * 0.15), 20)
    m.fit(X_tr[:-n_va], y_tr[:-n_va], eval_set=[(X_tr[-n_va:], y_tr[-n_va:])], verbose=False)
    return m, X_te, y_te


def sample_fewshot(records, ratio, seed):
    """按轨迹抽样目标山 train 的 ratio 比例。"""
    tracks = sorted({r['track_id'] for r in records})
    rng = np.random.RandomState(seed)
    rng.shuffle(tracks)
    n = max(int(len(tracks) * ratio), 1)
    keep = set(tracks[:n])
    return [r for r in records if r['track_id'] in keep]


def run_target(target, seed, args, dev):
    out = {'target': target, 'seed': seed, 'status': 'ok'}
    t_all = time.time()
    bundles = {m: cache_bundle(m, seed, args) for m in set(MOUNTAINS) | {target}}
    tb = bundles[target]
    srcs = [bundles[m] for m in MOUNTAINS if m != target]

    # ---- 1) 同场景 STRAT ----
    res_same = run_bundle(target, tb, seed, dev, args)
    out['same_scene'] = {'tf_acc': res_same['model']['tf_acc'], 'dur_mae_min': res_same['model']['tf_dur_mae_min'],
                         'path_picp90': res_same['model']['path_picp90'], 'path_crps_min': res_same['model']['path_crps_min'],
                         'roll_acc': res_same['model']['roll_acc']}

    # ---- 2) 零样本 STRAT（其余 3 山池化 → 共享原型） ----
    shared_meta, map_fn = pooled_shared_space(srcs)
    pooled = []
    for b in srcs:
        pooled.extend(remap_records(b['tr_records'], map_fn(b['region_meta'])))
    norm = build_normalizers(pooled, shared_meta)
    te_records = remap_records(tb['te_records'], map_fn(tb['region_meta']))
    tf0, sc0, ps0 = train_eval_shared(pooled, shared_meta, norm, te_records, seed, args, dev)
    out['zero_shot'] = {'tf_acc': round(tf0['acc'], 4), 'dur_mae_min': round(tf0['dur_mae_min'], 1),
                        'path_picp90': round(tf0.get('path_picp90', 0), 4), 'path_crps_min': round(tf0.get('path_crps_min', 0), 2),
                        'sigma_cal': round(sc0, 3) if sc0 else None, 'path_sigma_cal': round(ps0, 3) if ps0 else None}

    # ---- 3) 少样本 STRAT（按轨迹） ----
    out['few_shot'] = {}
    for ratio in args.fs_ratios:
        fs = sample_fewshot(tb['tr_records'], ratio, seed)
        fs_recs = remap_records(fs, map_fn(tb['region_meta']))
        pooled_fs = list(pooled) + fs_recs
        norm_fs = build_normalizers(pooled_fs, shared_meta)
        tf_fs, _, _ = train_eval_shared(pooled_fs, shared_meta, norm_fs, te_records, seed, args, dev)
        out['few_shot'][str(ratio)] = {'dur_mae_min': round(tf_fs['dur_mae_min'], 1),
                                       'path_picp90': round(tf_fs.get('path_picp90', 0), 4),
                                       'tf_acc': round(tf_fs['acc'], 4)}

    # ---- 4) 区域级 XGBoost（物理特征，D1'/D4' 同层对比） ----
    m_s, X_te_s, y_te_s = region_xgb([tb['tr_records']], tb['te_records'], seed)
    out['xgb_region_same'] = {'dur_mae_min': round(float(np.mean(np.abs(m_s.predict(X_te_s) - y_te_s))) / 60, 1)}
    m_z, X_te_z, y_te_z = region_xgb([b['tr_records'] for b in srcs], tb['te_records'], seed)
    out['xgb_region_zero'] = {'dur_mae_min': round(float(np.mean(np.abs(m_z.predict(X_te_z) - y_te_z))) / 60, 1)}
    out['xgb_region_few'] = {}
    for ratio in args.fs_ratios:
        fs = sample_fewshot(tb['tr_records'], ratio, seed)
        m_f, _, _ = region_xgb([b['tr_records'] for b in srcs] + [fs], tb['te_records'], seed)
        out['xgb_region_few'][str(ratio)] = {'dur_mae_min': round(float(np.mean(np.abs(m_f.predict(X_te_z) - y_te_z))) / 60, 1)}

    out['n_regions'] = tb['regions']
    out['n_routes'] = tb['n_routes']
    out['n_records'] = tb['n_records']
    out['time_s'] = round(time.time() - t_all, 1)
    print(f"[{target}|s{seed}] same={out['same_scene']['dur_mae_min']} zero={out['zero_shot']['dur_mae_min']} "
          f"fs5={out['few_shot']['0.05']['dur_mae_min']} | pathPICP same={out['same_scene']['path_picp90']} "
          f"zero={out['zero_shot']['path_picp90']} | xgbZ={out['xgb_region_zero']['dur_mae_min']} | {out['time_s']}s")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets', nargs='*', default=None)
    p.add_argument('--scenes', nargs='*', default=None,
                   help='全部景区列表（默认 scenes_20.NAMES_20）；targets 必须是其子集')
    p.add_argument('--train-scenes', nargs='*', default=None,
                   help='源池（默认=scenes；大→小协议传 12 大景区，--targets 传 8 小景区）')
    p.add_argument('--seeds', type=int, nargs='*', default=None)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--fs', type=float, nargs='*', default=[0.05, 0.10, 0.25],
                   help='少样本 STRAT 比例（默认 5/10/25%）')
    p.add_argument('--out', default=OUT)
    args = p.parse_args()
    Args.epochs = args.epochs
    Args.fs_ratios = args.fs
    if args.train_scenes:
        # 大→小协议：MOUNTAINS = 源池（12 大景区），--targets = 8 小目标
        _setup_scenes(args.train_scenes)
        if not args.targets:
            print('--train-scenes 模式下必须显式传 --targets')
            return 1
    elif args.scenes:
        _setup_scenes(args.scenes)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data-project'))
        from scenes_20 import NAMES_20
        _setup_scenes(NAMES_20)
    targets = args.targets or MOUNTAINS
    seeds = args.seeds or SEEDS
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', dev)
    print('scenes:', MOUNTAINS)
    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out, encoding='utf-8'))
    for t in targets:
        for seed in seeds:
            key = f'{t}|{seed}'
            if key in results:
                print(f'skip {key}')
                continue
            try:
                results[key] = run_target(t, seed, Args(), dev)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key] = {'status': 'error', 'error': str(e)}
            json.dump(results, open(args.out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print('saved:', args.out)


if __name__ == '__main__':
    main()
