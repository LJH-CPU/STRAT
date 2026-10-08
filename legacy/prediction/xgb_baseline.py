"""
XGBoost 段时长回归：用 15 维 GPS 段特征 + 位置上下文做回归。
对比 k-NN baseline 和 BiGRU V4。
"""

import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import default_clustered_csv, PREDICTION_OUTPUT_DIR

from data_utils import (
    extract_first_occurrence_sequences,
    build_region_metadata,
    split_by_route,
    GPS_SEG_DIM,
)
from baseline import build_knn_baseline, evaluate_knn

try:
    import xgboost as xgb
except ImportError:
    print("请先安装 xgboost: pip install xgboost")
    sys.exit(1)

plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
except Exception:
    pass


def build_flat_dataset(records, seg_p95, gps_mean, gps_std):
    """
    把每条轨迹的每个段展开为独立样本。
    特征: 15 GPS特征(归一化) + 位置 + 总段数 + from_region + to_region + 前累计距离 + 前累计时间
    """
    X_rows = []
    y_rows = []
    for r in records:
        n_segs = len(r['segment_durations'])
        gps_feats = r.get('gps_seg_features', [])
        cum_dist = 0.0
        cum_time = 0.0
        for i in range(n_segs):
            gps_f = np.array(gps_feats[i], dtype=np.float32)
            gps_f = (gps_f - gps_mean) / gps_std

            feat_vec = list(gps_f) + [
                i,
                n_segs,
                r['seq'][i],
                r['seq'][i + 1],
                cum_dist / 1000.0,
                cum_time / 3600.0,
            ]
            X_rows.append(feat_vec)
            y_rows.append(r['segment_durations'][i])

            seg_dist = gps_feats[i][0]
            cum_dist += seg_dist
            cum_time += r['segment_durations'][i]

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def predict_xgb_trajectories(model, test_records, gps_mean, gps_std):
    """
    用 XGBoost 逐段预测，累积得到轨迹预测。
    """
    results = []
    for r in test_records:
        n_segs = len(r['segment_durations'])
        gps_feats = r.get('gps_seg_features', [])
        pred_segs = []
        cum_dist = 0.0
        cum_time = 0.0
        for i in range(n_segs):
            gps_f = np.array(gps_feats[i], dtype=np.float32)
            gps_f = (gps_f - gps_mean) / gps_std
            feat_vec = np.array(list(gps_f) + [
                i, n_segs,
                r['seq'][i], r['seq'][i + 1],
                cum_dist / 1000.0, cum_time / 3600.0,
            ], dtype=np.float32).reshape(1, -1)

            pred = max(model.predict(feat_vec)[0], 0.0)
            pred_segs.append(pred)

            seg_dist = gps_feats[i][0]
            cum_dist += seg_dist
            cum_time += pred

        results.append({
            'track_id': r['track_id'],
            'route_id': r['route_id'],
            'pred_segments': pred_segs,
            'true_segments': r['segment_durations'],
            'n_segs': n_segs,
        })
    return results


def evaluate_xgb_results(xgb_results):
    seg_errs = []
    per_pos_seg = {}
    cum_errs = []
    per_pos_cum = {}
    dur_errs = []

    for r in xgb_results:
        n = r['n_segs']
        for i in range(n):
            err = abs(r['pred_segments'][i] - r['true_segments'][i])
            seg_errs.append(err)
            per_pos_seg.setdefault(i + 1, []).append(err)

        pred_cum = np.cumsum(np.concatenate([[0], r['pred_segments']]))
        true_cum = np.cumsum(np.concatenate([[0], r['true_segments']]))
        for i in range(1, n + 1):
            ce = abs(pred_cum[i] - true_cum[i])
            cum_errs.append(ce)
            per_pos_cum.setdefault(i + 1, []).append(ce)
        dur_errs.append(abs(pred_cum[-1] - true_cum[-1]))

    return {
        'seg_mae': np.mean(seg_errs),
        'per_pos_seg': {p: np.mean(e) for p, e in per_pos_seg.items()},
        'cum_mae': np.mean(cum_errs),
        'per_pos_cum': {p: np.mean(e) for p, e in per_pos_cum.items()},
        'dur_mae': np.mean(dur_errs),
    }


def main():
    parser = argparse.ArgumentParser(description='XGBoost Segment Duration Regression')
    parser.add_argument('--csv', type=str, default=default_clustered_csv(),
                        help='聚类后的 CSV 路径（默认取 cluster/output 下第一个 *_clustered.csv）')
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=" * 60)
    print("XGBoost: Segment Duration Regression on GPS Features")
    print("=" * 60)

    df = pd.read_csv(args.csv)
    records = extract_first_occurrence_sequences(df)
    region_meta = build_region_metadata(df)
    num_regions = len(region_meta)

    train_s, val_s, test_s = split_by_route(records, seed=args.seed)
    print(f"  Samples: {len(train_s)} train / {len(val_s)} val / {len(test_s)} test")

    all_gps = []
    for r in records:
        for feats in r.get('gps_seg_features', []):
            all_gps.append(feats)
    all_gps = np.array(all_gps, dtype=np.float32)
    gps_mean = all_gps.mean(axis=0)
    gps_std = all_gps.std(axis=0) + 1e-6

    all_segs = []
    for r in records:
        all_segs.extend(r['segment_durations'])
    seg_p95 = float(np.percentile(all_segs, 95))

    X_train, y_train = build_flat_dataset(train_s, seg_p95, gps_mean, gps_std)
    X_val, y_val = build_flat_dataset(val_s, seg_p95, gps_mean, gps_std)

    print(f"  Flat samples: {len(X_train)} train / {len(X_val)} val")
    print(f"  Features: {X_train.shape[1]} (15 GPS + 5 context)")
    print(f"  Target range: [{y_train.min():.0f}, {y_train.max():.0f}]s")
    print()

    print("=" * 60)
    print("Training XGBoost")
    print("=" * 60)

    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        objective='reg:absoluteerror',
        eval_metric='mae',
        random_state=args.seed,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=20,
    )

    print()

    print("=" * 60)
    print("Evaluating on Test Set")
    print("=" * 60)

    route_stats = build_knn_baseline(train_s)
    perc_pos_knn, knn_seg_mae, _ = evaluate_knn(route_stats, test_s)

    xgb_results = predict_xgb_trajectories(model, test_s, gps_mean, gps_std)
    xgb_metrics = evaluate_xgb_results(xgb_results)

    print(f"\n  === Segment-Level MAE ===")
    print(f"  k-NN:   {knn_seg_mae:.0f}s ({knn_seg_mae/60:.1f}min)")
    print(f"  XGBoost:{xgb_metrics['seg_mae']:.0f}s ({xgb_metrics['seg_mae']/60:.1f}min)")
    if knn_seg_mae > 0:
        print(f"  vs k-NN: {(knn_seg_mae - xgb_metrics['seg_mae'])/knn_seg_mae*100:+.1f}%")

    print(f"\n  === Cumulative MAE ===")
    print(f"  k-NN:   {xgb_metrics['seg_mae']:.0f}s (same as above)")
    print(f"  XGBoost:{xgb_metrics['cum_mae']:.0f}s ({xgb_metrics['cum_mae']/60:.1f}min)")

    print(f"\n  === Total Duration MAE ===")
    print(f"  XGBoost:{xgb_metrics['dur_mae']:.0f}s ({xgb_metrics['dur_mae']/60:.1f}min)")

    print(f"\n  {'Seg':>5} {'XGB(s)':>10} {'XGB(min)':>10} {'k-NN(s)':>10} {'k-NN(min)':>10}")
    for pos in sorted(set(list(xgb_metrics['per_pos_seg'].keys()) + list(perc_pos_knn.keys()))):
        xm = xgb_metrics['per_pos_seg'].get(pos, 0)
        km = perc_pos_knn.get(pos, 0)
        print(f"  {pos:5d} {xm:10.0f} {xm/60:10.1f} {km:10.0f} {km/60:10.1f}")

    print(f"\n  Feature importance (top 10):")
    feat_names = [f'GPS_{i}' for i in range(GPS_SEG_DIM)] + [
        'position', 'n_segs', 'from_region', 'to_region', 'cum_dist_km', 'cum_time_h'
    ]
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    for idx in top_idx:
        print(f"    {feat_names[idx]:<18s} {importances[idx]:.4f}")

    os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)
    model_path = os.path.join(str(PREDICTION_OUTPUT_DIR), 'xgboost_model.json')
    model.save_model(model_path)
    print(f"\n  Saved to {model_path}")


if __name__ == '__main__':
    main()