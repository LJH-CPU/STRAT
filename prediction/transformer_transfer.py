"""
跨场景到达时间迁移（严格无泄漏版）：
- 用"场景无关特征"（段地形/时刻，不含区域 id、不含真值下一区域）训练 Ridge
- 在未见景区**真正留出的 test split** 上零样本测试
- 对照：目标场景同场景线性、同场景 kNN（均 train 拟合 / test 打分）
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CLEANED_DIR
import ar_pipeline as ap
import transformer_baselines as lb


def _scene_records(name, seed=42, n_workers=8):
    """返回 (tr_records, te_records, pair_mean, global_mean, k)：train 内聚类/路线，test 只映射。"""
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
    pm, gm = __import__('transformer_model', fromlist=['build_pair_mean']).build_pair_mean(
        tr_records, model_reg['k'])
    return tr_records, te_records, pm, gm, model_reg['k']


def _flat_scene_agnostic(records, global_mean):
    """场景无关特征：段地形 + 当前区域 + 全局均值 prior（不含真值下一区域）。"""
    X, y = [], []
    for r in records:
        seq = r['seq']
        gf = r['gps_seg_features']
        for i, d in enumerate(r['segment_durations']):
            a = int(seq[i])
            X.append(list(gf[i]) + [a, global_mean])
            y.append(d)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _knn_mae(train_records, test_records, global_mean):
    """先验基线：train 上按当前区域均值，test 打分（无真值下一区域）。"""
    sums, cnts = {}, {}
    for r in train_records:
        seq = r['seq']
        for i, d in enumerate(r['segment_durations']):
            a = int(seq[i])
            sums[a] = sums.get(a, 0.0) + d
            cnts[a] = cnts.get(a, 0) + 1
    errs = []
    for r in test_records:
        seq = r['seq']
        for i, d in enumerate(r['segment_durations']):
            a = int(seq[i])
            pred = sums[a] / cnts[a] if a in sums else global_mean
            errs.append(abs(pred - d))
    return float(np.mean(errs)) if errs else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', nargs='*', default=['峨眉山', '青城山'])
    p.add_argument('--target', nargs='*', default=['都江堰', '熊猫基地'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--out', default='prediction/output/llm_transfer.json')
    args = p.parse_args()

    # 训练集：所有 source 场景的 train split
    Xtr_all, ytr_all = [], []
    src_meta = {}
    for name in args.source:
        tr, te, pm, gm, k = _scene_records(name, args.seed, args.n_workers)
        X, y = _flat_scene_agnostic(tr, gm)
        Xtr_all.append(X); ytr_all.append(y)
        src_meta[name] = {'n_records': len(tr), 'n_samples': len(X), 'k': k}
    Xtr = np.concatenate(Xtr_all); ytr = np.concatenate(ytr_all)
    clf = Ridge(alpha=1.0).fit(Xtr, ytr)

    results = {'source': args.source, 'src_meta': src_meta, 'targets': {}}
    for name in args.target:
        tr, te, pm, gm, k = _scene_records(name, args.seed, args.n_workers)
        Xte, yte = _flat_scene_agnostic(te, gm)
        pred = np.maximum(clf.predict(Xte), 0)
        mae_transfer = float(np.mean(np.abs(pred - yte))) if len(yte) else None
        # 目标场景同场景：train 拟合 / test 打分（无 in-sample）
        lin_mae = None
        if len(tr) > 10:
            Xtr_t, ytr_t = _flat_scene_agnostic(tr, gm)
            clf2 = Ridge(alpha=1.0).fit(Xtr_t, ytr_t)
            lin_mae = float(np.mean(np.abs(clf2.predict(Xte) - yte)))
        knn_mae = _knn_mae(tr, te, gm)
        results['targets'][name] = {
            'n_records_train': len(tr), 'n_records_test': len(te), 'n_samples': len(yte), 'k': k,
            'transfer_mae_min': round(mae_transfer / 60, 1) if mae_transfer else None,
            'in_scene_linear_mae_min': round(lin_mae / 60, 1) if lin_mae else None,
            'in_scene_knn_mae_min': round(knn_mae / 60, 1),
        }
        print(f"[{name}] transfer={mae_transfer/60 if mae_transfer else None:.1f}min | "
              f"in-scene linear={lin_mae/60 if lin_mae else None:.1f}min kNN={knn_mae/60:.1f}min")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'结果已保存: {args.out}')


if __name__ == '__main__':
    main()
