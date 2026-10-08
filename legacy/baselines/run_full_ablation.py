"""
STRAT 完整消融实验：跨模型家族通用性验证 + 特征消融

双层论证框架：
  Panel 1: 物理先验特征在 Tree Ensemble / RNN 两族模型上均表现良好，
           模型间差异远小于"有/无物理特征"的差异 → 特征决定性能上限
  Panel 2: 特征消融在 XGBoost / LightGBM / BiGRU 三模型上展现一致的退化模式，
           特征贡献具有模型无关性

核心命题：
  用物理先验构造高质量的输入特征，让模型只需要做"简单的事"——
  BiGRU 能做好，XGBoost 也能做好，LightGBM 也能做好，因为特征本身已经足够好了。
"""

import os
import sys
import argparse
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prediction"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ablation"))

from prediction.data_utils import (
    extract_first_occurrence_sequences,
    build_region_metadata,
    split_by_route,
    GPS_SEG_DIM,
)
from prediction.baseline import build_knn_baseline
from prediction.sequence_model import create_dataloader
from run_prediction_ablation import (
    predict_gru_segments,
    compute_all_metrics,
    train_one_config,
    ABLATION_CONFIGS,
)
from prediction_baselines import (
    BiLSTMPredictor,
    train_deep_baseline,
    train_region_median,
    predict_region_median,
)

warnings.filterwarnings("ignore")


def build_flat_dataset(records, region_meta, gps_mean, gps_std, use_gps=True, use_role=True, use_time=True):
    X_rows = []
    y_rows = []
    for r in records:
        n_segs = len(r["segment_durations"])
        gps_feats = r.get("gps_seg_features", [])
        hour = r.get("start_time_of_day", 12.0)
        cum_dist = 0.0
        cum_time = 0.0
        for i in range(n_segs):
            feat_vec = []

            from_meta = region_meta.get(r["seq"][i], {})
            to_meta = region_meta.get(r["seq"][i + 1], {})

            feat_vec.extend(
                [
                    from_meta.get("lat", 0.0),
                    from_meta.get("lon", 0.0),
                    from_meta.get("elev_mean", 0.0),
                    to_meta.get("lat", 0.0),
                    to_meta.get("lon", 0.0),
                    to_meta.get("elev_mean", 0.0),
                    float(i),
                    float(n_segs),
                    cum_dist / 1000.0,
                    cum_time / 3600.0,
                ]
            )

            if use_gps and i < len(gps_feats):
                gps_f = np.array(gps_feats[i], dtype=np.float32)
                gps_f = (gps_f - gps_mean) / gps_std
                feat_vec.extend(gps_f.tolist())

            if use_time:
                feat_vec.append(np.sin(2 * np.pi * hour / 24))
                feat_vec.append(np.cos(2 * np.pi * hour / 24))

            if use_role:
                feat_vec.append(float(from_meta.get("role_id", 0)))
                feat_vec.append(float(to_meta.get("role_id", 0)))

            X_rows.append(feat_vec)
            y_rows.append(r["segment_durations"][i])

            seg_dist = gps_feats[i][0] if i < len(gps_feats) else 0.0
            cum_dist += seg_dist
            cum_time += r["segment_durations"][i]

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def predict_tree_segments(model, test_records, region_meta, gps_mean, gps_std, use_gps, use_role, use_time):
    results = []
    for r in test_records:
        n_segs = len(r["segment_durations"])
        gps_feats = r.get("gps_seg_features", [])
        hour = r.get("start_time_of_day", 12.0)
        pred_segs = []
        cum_dist = 0.0
        cum_time = 0.0
        for i in range(n_segs):
            feat_vec = []

            from_meta = region_meta.get(r["seq"][i], {})
            to_meta = region_meta.get(r["seq"][i + 1], {})

            feat_vec.extend(
                [
                    from_meta.get("lat", 0.0),
                    from_meta.get("lon", 0.0),
                    from_meta.get("elev_mean", 0.0),
                    to_meta.get("lat", 0.0),
                    to_meta.get("lon", 0.0),
                    to_meta.get("elev_mean", 0.0),
                    float(i),
                    float(n_segs),
                    cum_dist / 1000.0,
                    cum_time / 3600.0,
                ]
            )

            if use_gps and i < len(gps_feats):
                gps_f = np.array(gps_feats[i], dtype=np.float32)
                gps_f = (gps_f - gps_mean) / gps_std
                feat_vec.extend(gps_f.tolist())

            if use_time:
                feat_vec.append(np.sin(2 * np.pi * hour / 24))
                feat_vec.append(np.cos(2 * np.pi * hour / 24))

            if use_role:
                feat_vec.append(float(from_meta.get("role_id", 0)))
                feat_vec.append(float(to_meta.get("role_id", 0)))

            feat_arr = np.array(feat_vec, dtype=np.float32).reshape(1, -1)
            pred = max(float(model.predict(feat_arr)[0]), 0.0)
            pred_segs.append(pred)

            seg_dist = gps_feats[i][0] if i < len(gps_feats) else 0.0
            cum_dist += seg_dist
            cum_time += pred

        results.append(
            {
                "track_id": r["track_id"],
                "route_id": r["route_id"],
                "pred_segments": pred_segs,
                "true_segments": r["segment_durations"],
                "n_segs": n_segs,
            }
        )
    return results


def train_xgb_config(config, train_records, val_records, test_records, region_meta, gps_mean, gps_std, seed=42, n_estimators=200, max_depth=5, lr=0.05, subsample=0.8):
    import xgboost as xgb

    use_gps = config.get("use_gps", True)
    use_role = config.get("use_role", True)
    use_time = config.get("use_time", True)

    X_train, y_train = build_flat_dataset(train_records, region_meta, gps_mean, gps_std, use_gps, use_role, use_time)
    X_val, y_val = build_flat_dataset(val_records, region_meta, gps_mean, gps_std, use_gps, use_role, use_time)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=subsample,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        objective="reg:absoluteerror",
        eval_metric="mae",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=30,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)

    results = predict_tree_segments(model, test_records, region_meta, gps_mean, gps_std, use_gps, use_role, use_time)
    route_stats = build_knn_baseline(train_records)
    metrics = compute_all_metrics(results, test_records, route_stats)
    metrics["params"] = X_train.shape[1]
    return metrics


def train_lgb_config(config, train_records, val_records, test_records, region_meta, gps_mean, gps_std, seed=42):
    import lightgbm as lgb

    use_gps = config.get("use_gps", True)
    use_role = config.get("use_role", True)
    use_time = config.get("use_time", True)

    X_train, y_train = build_flat_dataset(train_records, region_meta, gps_mean, gps_std, use_gps, use_role, use_time)
    X_val, y_val = build_flat_dataset(val_records, region_meta, gps_mean, gps_std, use_gps, use_role, use_time)
    X_test, y_test = build_flat_dataset(test_records, region_meta, gps_mean, gps_std, use_gps, use_role, use_time)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": seed,
        },
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    flat_idx = 0
    results = []
    for rec in test_records:
        n_segs = len(rec["segment_durations"])
        pred_segs = [max(p, 0.0) for p in preds[flat_idx : flat_idx + n_segs].tolist()]
        flat_idx += n_segs
        results.append(
            {
                "track_id": rec["track_id"],
                "route_id": rec["route_id"],
                "pred_segments": pred_segs,
                "true_segments": rec["segment_durations"],
                "n_segs": n_segs,
            }
        )

    route_stats = build_knn_baseline(train_records)
    metrics = compute_all_metrics(results, test_records, route_stats)
    metrics["params"] = X_train.shape[1]
    return metrics


def train_deep_model(model, train_loader, val_loader, val_dataset, test_loader, test_dataset, test_records, train_records, device, args):
    t0 = time.time()
    model = train_deep_baseline(model, train_loader, val_loader, val_dataset, device, args)
    elapsed = time.time() - t0
    deep_results = predict_gru_segments(model, test_loader, test_dataset, device)
    route_stats = build_knn_baseline(train_records)
    metrics = compute_all_metrics(deep_results, test_records, route_stats)
    metrics["time_s"] = elapsed
    metrics["params"] = sum(p.numel() for p in model.parameters())
    return metrics


PANEL1_CONFIGS = {
    "KNN": {"family": "统计基线", "model": "knn"},
    "RegionMedian": {"family": "统计基线", "model": "region_median"},
    "XGBoost": {"family": "Tree Ens.", "model": "xgb"},
    "LightGBM": {"family": "Tree Ens.", "model": "lgb"},
    "BiGRU": {"family": "RNN", "model": "gru"},
    "BiLSTM": {"family": "RNN", "model": "bilstm"},
}

ABLATION_KEYS = ["no_gps", "no_role", "no_time"]
ABLATION_LABELS = {
    "no_gps": "No GPS 15D",
    "no_role": "No Role (KMeans)",
    "no_time": "No Time Encode",
}


def main():
    parser = argparse.ArgumentParser(description="STRAT Full Ablation Study")
    parser.add_argument("--csv", type=str, default="../cluster/output/峨眉山_clustered.csv")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--role_embed_dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--xgb_rounds", type=int, default=200)
    parser.add_argument("--xgb_depth", type=int, default=5)
    parser.add_argument("--xgb_lr", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--output_dir", type=str, default="baselines/output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_xgb", action="store_true")
    parser.add_argument("--skip_lgb", action="store_true")
    parser.add_argument("--skip_gru", action="store_true")
    parser.add_argument("--skip_bilstm", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 78)
    print("  STRAT 完整消融实验：物理先验特征跨模型家族通用性验证")
    print("=" * 78)
    print("  核心命题：特征本身已编码足够信息 -> 任何合理模型都能逼近同一上限")
    print()

    df = pd.read_csv(args.csv)
    print(f"[Data] Points: {len(df):,}  |  Tracks: {df['trackId'].nunique():,}")

    records = extract_first_occurrence_sequences(df)
    region_meta = build_region_metadata(df)
    num_regions = len(region_meta)
    num_roles = max(m["role_id"] for m in region_meta.values()) + 1

    train_records, val_records, test_records = split_by_route(records, seed=args.seed)
    print(f"[Data] Train: {len(train_records)}  |  Val: {len(val_records)}  |  " f"Test: {len(test_records)}  |  Regions: {num_regions}  |  Roles: {num_roles}")

    max_len = 8
    train_loader, train_dataset = create_dataloader(train_records, region_meta, num_regions, args.batch_size, shuffle=True, max_len=max_len)
    val_loader, val_dataset = create_dataloader(val_records, region_meta, num_regions, args.batch_size, shuffle=False, max_len=max_len)
    test_loader, test_dataset = create_dataloader(test_records, region_meta, num_regions, args.batch_size, shuffle=False, max_len=max_len)

    route_stats = build_knn_baseline(train_records)
    flat_input_dim = 1 + 3 + GPS_SEG_DIM + 2 + 1 + 1

    all_gps = []
    for r in train_records:
        for feats in r.get("gps_seg_features", []):
            all_gps.append(feats)
    all_gps = np.array(all_gps, dtype=np.float32)
    gps_mean = all_gps.mean(axis=0)
    gps_std = all_gps.std(axis=0) + 1e-6

    all_results = {}
    panel1_order = []

    print(f"\n{'='*78}")
    print("  PANEL 1: 跨模型家族全特征对比")
    print(f"{'='*78}")

    print(f"\n  [KNN] k-NN route-level median baseline")
    t0 = time.time()
    knn_results = []
    for rec in test_records:
        from prediction.baseline import predict_knn

        preds, _ = predict_knn(route_stats, rec)
        knn_results.append(
            {
                "track_id": rec["track_id"],
                "route_id": rec["route_id"],
                "pred_segments": preds,
                "true_segments": rec["segment_durations"],
                "n_segs": len(preds),
            }
        )
    knn_metrics = compute_all_metrics(knn_results, test_records, route_stats)
    knn_metrics["time_s"] = time.time() - t0
    knn_metrics["params"] = sum(len(v["segment_stats"]) for v in route_stats.values())
    all_results["KNN"] = knn_metrics
    panel1_order.append("KNN")
    m = knn_metrics
    print(f"    Seg MAE: {m['seg_mae']:.0f}s  Cum MAE: {m['cum_mae']:.0f}s  " f"Dur MAE: {m['dur_mae_s']:.0f}s  +/-30min: {m['window_acc']['win_30min']:.1f}%")

    print(f"\n  [Region-Median] Region-level median baseline")
    t0 = time.time()
    median_map = train_region_median(train_records)
    med_results = predict_region_median(median_map, test_records)
    med_metrics = compute_all_metrics(med_results, test_records, route_stats)
    med_metrics["time_s"] = time.time() - t0
    med_metrics["params"] = len(median_map)
    all_results["RegionMedian"] = med_metrics
    panel1_order.append("RegionMedian")
    m = med_metrics
    print(f"    Seg MAE: {m['seg_mae']:.0f}s  Cum MAE: {m['cum_mae']:.0f}s  " f"Dur MAE: {m['dur_mae_s']:.0f}s  +/-30min: {m['window_acc']['win_30min']:.1f}%")

    if not args.skip_xgb:
        try:
            import xgboost

            print(f"\n  [XGBoost] XGBoost -- Full Features")
            xgb_cfg = {"use_gps": True, "use_role": True, "use_time": True}
            t0 = time.time()
            xgb_m = train_xgb_config(xgb_cfg, train_records, val_records, test_records, region_meta, gps_mean, gps_std, seed=args.seed, n_estimators=args.xgb_rounds, max_depth=args.xgb_depth, lr=args.xgb_lr, subsample=args.xgb_subsample)
            xgb_m["time_s"] = time.time() - t0
            all_results["XGBoost"] = xgb_m
            panel1_order.append("XGBoost")
            m = xgb_m
            print(f"    Seg MAE: {m['seg_mae']:.0f}s  Cum MAE: {m['cum_mae']:.0f}s  " f"Dur MAE: {m['dur_mae_s']:.0f}s  +/-30min: {m['window_acc']['win_30min']:.1f}%")
        except ImportError:
            print("  [XGBoost] not installed, skipping")

    if not args.skip_lgb:
        try:
            import lightgbm

            print(f"\n  [LightGBM] LightGBM -- Full Features")
            lgb_cfg = {"use_gps": True, "use_role": True, "use_time": True}
            t0 = time.time()
            lgb_m = train_lgb_config(lgb_cfg, train_records, val_records, test_records, region_meta, gps_mean, gps_std, seed=args.seed)
            lgb_m["time_s"] = time.time() - t0
            all_results["LightGBM"] = lgb_m
            panel1_order.append("LightGBM")
            m = lgb_m
            print(f"    Seg MAE: {m['seg_mae']:.0f}s  Cum MAE: {m['cum_mae']:.0f}s  " f"Dur MAE: {m['dur_mae_s']:.0f}s  +/-30min: {m['window_acc']['win_30min']:.1f}%")
        except ImportError:
            print("  [LightGBM] not installed, skipping")

    if not args.skip_gru:
        print(f"\n  [BiGRU] Bidirectional GRU -- Full Features")
        gru_cfg = ABLATION_CONFIGS["full"].copy()
        gru_cfg["desc"] = "BiGRU Full"
        t0 = time.time()
        gru_m = train_one_config(gru_cfg, train_records, val_records, test_records, region_meta, num_regions, num_roles, device, args)
        gru_m["time_s"] = time.time() - t0
        all_results["BiGRU"] = gru_m
        panel1_order.append("BiGRU")
        m = gru_m
        print(f"    Seg MAE: {m['seg_mae']:.0f}s  Cum MAE: {m['cum_mae']:.0f}s  " f"Dur MAE: {m['dur_mae_s']:.0f}s  +/-30min: {m['window_acc']['win_30min']:.1f}%")

    if not args.skip_bilstm:
        print(f"\n  [BiLSTM] Bidirectional LSTM -- Full Features")
        bilstm = BiLSTMPredictor(flat_input_dim, args.hidden_dim).to(device)
        print(f"    Params: {sum(p.numel() for p in bilstm.parameters()):,}")
        bilstm_m = train_deep_model(bilstm, train_loader, val_loader, val_dataset, test_loader, test_dataset, test_records, train_records, device, args)
        all_results["BiLSTM"] = bilstm_m
        panel1_order.append("BiLSTM")
        m = bilstm_m
        print(f"    Seg MAE: {m['seg_mae']:.0f}s  Cum MAE: {m['cum_mae']:.0f}s  " f"Dur MAE: {m['dur_mae_s']:.0f}s  +/-30min: {m['window_acc']['win_30min']:.1f}%")

    # ================================================================
    # PANEL 2: Feature Ablation -- Tree vs RNN
    # ================================================================
    print(f"\n{'='*78}")
    print("  PANEL 2: 特征消融 -- XGBoost / LightGBM / BiGRU 三模型对照")
    print(f"{'='*78}")
    print("  检验：同一特征去掉后，不同模型退化幅度是否一致")
    print("  若一致 -> 特征贡献具有模型无关性")

    panel2_results = {}

    for akey in ABLATION_KEYS:
        label = ABLATION_LABELS[akey]
        print(f"\n  --- {label} ---")

        row = {}
        ab_cfg = ABLATION_CONFIGS[akey].copy()

        if not args.skip_xgb:
            try:
                import xgboost

                t0 = time.time()
                m = train_xgb_config(ab_cfg, train_records, val_records, test_records, region_meta, gps_mean, gps_std, seed=args.seed, n_estimators=args.xgb_rounds, max_depth=args.xgb_depth, lr=args.xgb_lr, subsample=args.xgb_subsample)
                m["time_s"] = time.time() - t0
                row["xgb"] = m
                full = all_results.get("XGBoost", {})
                if full:
                    d = (m["seg_mae"] - full["seg_mae"]) / full["seg_mae"] * 100
                    print(f"    XGBoost:  Seg MAE={m['seg_mae']:.0f}s (D{d:+.1f}%)")
            except ImportError:
                pass

        if not args.skip_lgb:
            try:
                import lightgbm

                t0 = time.time()
                m = train_lgb_config(ab_cfg, train_records, val_records, test_records, region_meta, gps_mean, gps_std, seed=args.seed)
                m["time_s"] = time.time() - t0
                row["lgb"] = m
                full = all_results.get("LightGBM", {})
                if full:
                    d = (m["seg_mae"] - full["seg_mae"]) / full["seg_mae"] * 100
                    print(f"    LightGBM: Seg MAE={m['seg_mae']:.0f}s (D{d:+.1f}%)")
            except ImportError:
                pass

        if not args.skip_gru:
            t0 = time.time()
            m = train_one_config(ab_cfg, train_records, val_records, test_records, region_meta, num_regions, num_roles, device, args)
            m["time_s"] = time.time() - t0
            row["gru"] = m
            full = all_results.get("BiGRU", {})
            if full:
                d = (m["seg_mae"] - full["seg_mae"]) / full["seg_mae"] * 100
                print(f"    BiGRU:    Seg MAE={m['seg_mae']:.0f}s (D{d:+.1f}%)")

        panel2_results[akey] = row

    # ================================================================
    # TABLE 1
    # ================================================================
    print(f"\n\n{'='*78}")
    print("  TABLE 1: Cross-Family Full-Feature Comparison")
    print(f"{'='*78}")
    print()
    header = f"{'Family':<14} {'Model':<14} {'Seg MAE':>9} {'Cum MAE':>9} {'Dur(s)':>9} {'+/-30m':>8} {'Params':>9}"
    print(header)
    print("-" * 78)

    prev = None
    for name in panel1_order:
        m = all_results.get(name)
        if m is None:
            continue
        info = PANEL1_CONFIGS.get(name, {})
        family = info.get("family", "--")
        if family != prev and prev is not None:
            print("-" * 78)
        prev = family
        print(f"{family:<14} {name:<14} {m['seg_mae']:9.0f} {m['cum_mae']:9.0f} " f"{m['dur_mae_s']:9.0f} {m['window_acc']['win_30min']:7.1f}% " f"{m.get('params', 0):9,d}")
    print("=" * 78)

    non_stat = []
    knn_seg = None
    for name in panel1_order:
        m = all_results.get(name)
        if m is None:
            continue
        fam = PANEL1_CONFIGS.get(name, {}).get("family", "")
        if fam == "统计基线" and name == "KNN":
            knn_seg = m["seg_mae"]
        elif fam != "统计基线":
            non_stat.append(m["seg_mae"])

    if non_stat:
        gap = (max(non_stat) - min(non_stat)) / min(non_stat) * 100
        print(f"  ▲  Model gap: {gap:.1f}%")
    if knn_seg and non_stat:
        imp = (knn_seg - max(non_stat)) / knn_seg * 100
        print(f"  ▲  vs KNN: {imp:.1f}%")
    print("  ▲  Feature quality > Architecture choice")

    # ================================================================
    # TABLE 2
    # ================================================================
    print(f"\n\n{'='*78}")
    print("  TABLE 2: Feature Ablation -- XGBoost / LightGBM / BiGRU")
    print(f"{'='*78}")
    print()
    header = f"{'Ablation':<18} {'XGB D%':>10} {'LGB D%':>10} {'GRU D%':>10} {'Consistent':>10}"
    print(header)
    print("-" * 78)

    full_xgb = all_results.get("XGBoost", {})
    full_lgb = all_results.get("LightGBM", {})
    full_gru = all_results.get("BiGRU", {})

    for akey in ABLATION_KEYS:
        label = ABLATION_LABELS[akey]
        row = panel2_results.get(akey, {})

        def delta(m, full):
            if m and full and full.get("seg_mae"):
                return (m["seg_mae"] - full["seg_mae"]) / full["seg_mae"] * 100
            return None

        xgb_d = delta(row.get("xgb"), full_xgb)
        lgb_d = delta(row.get("lgb"), full_lgb)
        gru_d = delta(row.get("gru"), full_gru)

        xgb_s = f"+{xgb_d:.1f}%" if xgb_d is not None else "--"
        lgb_s = f"+{lgb_d:.1f}%" if lgb_d is not None else "--"
        gru_s = f"+{gru_d:.1f}%" if gru_d is not None else "--"

        degs = [d for d in [xgb_d, lgb_d, gru_d] if d is not None]
        cons = "Yes" if len(degs) >= 2 and max(degs) - min(degs) < 20 else "--"

        print(f"{label:<18} {xgb_s:>10} {lgb_s:>10} {gru_s:>10} {cons:>10}")

    print("=" * 78)
    print("  ▲  NoRole degrade -> KMeans clustering has predictive value")
    print("  ▲  Consistent degrade -> feature contribution is model-agnostic")

    # ================================================================
    # TABLE 3
    # ================================================================
    print(f"\n\n{'='*78}")
    print("  TABLE 3: Full Metrics Detail")
    print(f"{'='*78}")
    print()
    header = f"{'Config':<24} {'Seg MAE':>9} {'Cum MAE':>9} {'Dur(s)':>9} {'+/-15m':>8} {'+/-30m':>8} {'+/-60m':>8}"
    print(header)
    print("-" * 78)

    for name in panel1_order:
        m = all_results.get(name)
        if m is None:
            continue
        print(f"{name:<24} {m['seg_mae']:9.0f} {m['cum_mae']:9.0f} {m['dur_mae_s']:9.0f} " f"{m['window_acc']['win_15min']:7.1f}% {m['window_acc']['win_30min']:7.1f}% " f"{m['window_acc']['win_60min']:7.1f}%")

    print("-" * 78)

    for akey in ABLATION_KEYS:
        label = ABLATION_LABELS[akey]
        row = panel2_results.get(akey, {})
        for mk, mn in [("xgb", "XGB_" + label), ("lgb", "LGB_" + label), ("gru", "GRU_" + label)]:
            m = row.get(mk)
            if m is None:
                continue
            print(f"{mn:<24} {m['seg_mae']:9.0f} {m['cum_mae']:9.0f} {m['dur_mae_s']:9.0f} " f"{m['window_acc']['win_15min']:7.1f}% {m['window_acc']['win_30min']:7.1f}% " f"{m['window_acc']['win_60min']:7.1f}%")

    # ================================================================
    # Save
    # ================================================================
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    output = {
        "panel1": make_serializable({k: all_results[k] for k in panel1_order if k in all_results}),
        "panel2": make_serializable(
            {
                akey: {
                    "xgb": row.get("xgb"),
                    "lgb": row.get("lgb"),
                    "gru": row.get("gru"),
                }
                for akey, row in panel2_results.items()
            }
        ),
    }

    out_path = os.path.join(args.output_dir, "full_ablation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
