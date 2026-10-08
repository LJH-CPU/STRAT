"""
跨管线对比实验：4 种聚类 × 3 种预测器。

一次只跑一个管道，避免内存爆炸。
每跑完一个就把结果追加到 JSON，如果中断了可以从中断点继续。

用法：
  python baselines/pipeline_comparison.py --cluster strat  --predictor median
  python baselines/pipeline_comparison.py --cluster kmeans  --predictor lightgbm
  python baselines/pipeline_comparison.py --cluster dbscan  --predictor bigru
  ...
"""

import os
import sys
import argparse
import json
import time
import gc

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prediction"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ablation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from prediction.data_utils import (
    extract_first_occurrence_sequences,
    build_region_metadata,
    split_by_route,
    GPS_SEG_DIM,
)
from prediction.sequence_model import SequencePredictor, create_dataloader
from run_prediction_ablation import (
    train_epoch,
    evaluate_model,
    predict_gru_segments,
    compute_all_metrics,
    relative_l1_loss,
)
from prediction.baseline import build_knn_baseline
from prediction_baselines import (
    train_region_median,
    predict_region_median,
    flatten_records,
    run_lightgbm,
    train_deep_baseline,
    BiLSTMPredictor,
)


def load_and_split(csv_path, seed):
    df = pd.read_csv(csv_path)
    records = extract_first_occurrence_sequences(df)
    region_meta = build_region_metadata(df)
    num_regions = len(region_meta)
    num_roles = max(m["role_id"] for m in region_meta.values()) + 1
    train_records, val_records, test_records = split_by_route(records, seed=seed)
    del df
    gc.collect()
    return train_records, val_records, test_records, region_meta, num_regions, num_roles


def make_dataloaders(train_records, val_records, test_records, region_meta, num_regions, batch_size):
    max_len = 8
    train_loader, train_dataset = create_dataloader(train_records, region_meta, num_regions, batch_size, shuffle=True, max_len=max_len)
    val_loader, val_dataset = create_dataloader(val_records, region_meta, num_regions, batch_size, shuffle=False, max_len=max_len)
    test_loader, test_dataset = create_dataloader(test_records, region_meta, num_regions, batch_size, shuffle=False, max_len=max_len)
    return train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset


def run_one_pipeline(csv_path, cluster_method, predictor_type, train_records, val_records, test_records, region_meta, num_regions, num_roles, device, args):
    print(f"\n  [{predictor_type}] 训练中...")

    loaders = make_dataloaders(train_records, val_records, test_records, region_meta, num_regions, args.batch_size)
    train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset = loaders
    route_stats = build_knn_baseline(train_records)
    flat_input_dim = 1 + 3 + GPS_SEG_DIM + 2 + 1 + 1

    if predictor_type == "median":
        t0 = time.time()
        median_map = train_region_median(train_records)
        results = predict_region_median(median_map, test_records)
        metrics = compute_all_metrics(results, test_records, route_stats)
        metrics["time_s"] = time.time() - t0
        metrics["params"] = len(median_map)

    elif predictor_type == "lightgbm":
        t0 = time.time()
        X_train, y_train = flatten_records(train_records, region_meta, num_regions)
        X_val, y_val = flatten_records(val_records, region_meta, num_regions)
        X_test, y_test = flatten_records(test_records, region_meta, num_regions)
        result = run_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
        del X_train, X_val
        gc.collect()
        if result is None:
            print("    LightGBM 失败")
            return None
        lgb_preds, lgb_time = result
        flat_idx = 0
        lgb_results = []
        for rec in test_records:
            n_segs = len(rec["segment_durations"])
            pred_segs = lgb_preds[flat_idx : flat_idx + n_segs].tolist()
            flat_idx += n_segs
            lgb_results.append(
                {
                    "track_id": rec["track_id"],
                    "route_id": rec["route_id"],
                    "pred_segments": pred_segs,
                    "true_segments": rec["segment_durations"],
                    "n_segs": n_segs,
                }
            )
        metrics = compute_all_metrics(lgb_results, test_records, route_stats)
        metrics["time_s"] = lgb_time
        metrics["params"] = len(lgb_preds)
        del lgb_results
        gc.collect()

    elif predictor_type == "bigru":
        model = SequencePredictor(
            num_regions=num_regions + 1,
            region_embed_dim=args.embed_dim,
            geo_dim=3,
            gps_seg_dim=GPS_SEG_DIM,
            pos_dim=8,
            time_dim=2,
            num_roles=num_roles,
            role_embed_dim=args.role_embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=1,
            dropout=0.4,
            use_gps=True,
            use_role=True,
            use_time=True,
            use_pos=True,
            bidirectional=True,
        ).to(device)
        t0 = time.time()
        model = train_deep_baseline(model, train_loader, val_loader, val_dataset, device, args)
        elapsed = time.time() - t0
        results = predict_gru_segments(model, test_loader, test_dataset, device)
        metrics = compute_all_metrics(results, test_records, route_stats)
        metrics["time_s"] = elapsed
        metrics["params"] = sum(p.numel() for p in model.parameters())
        del model
        gc.collect()

    else:
        print(f"    Unknown predictor: {predictor_type}")
        return None

    del train_loader, train_dataset, val_loader, val_dataset, test_loader, test_dataset
    gc.collect()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Cross-Pipeline Comparison (one at a time)")
    parser.add_argument("--cluster", type=str, required=True, choices=["strat", "kmeans", "dbscan", "spectral"])
    parser.add_argument("--predictor", type=str, required=True, choices=["median", "lightgbm", "bigru"])
    parser.add_argument("--scene", type=str, default="峨眉山",
                        help='景区名称 (默认: 峨眉山)')
    parser.add_argument("--output_dir", type=str, default="baselines/output")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--role_embed_dim", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cpu")

    print("=" * 70)
    print(f"跨管线对比: {args.cluster} × {args.predictor}")
    print("=" * 70)

    csv_path = os.path.join(args.output_dir, f"{args.scene}_{args.cluster}.csv")
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] {csv_path} 不存在。请先运行:")
        print(f"  python baselines/generate_clustered_csvs.py --method {args.cluster} --scene {args.scene}")
        return

    print(f"\n[数据] {csv_path}")
    train_records, val_records, test_records, region_meta, num_regions, num_roles = load_and_split(csv_path, args.seed)
    print(f"  Scene: {args.scene}, Train: {len(train_records)}, Val: {len(val_records)}, " f"Test: {len(test_records)}, Regions: {num_regions}")

    t0 = time.time()
    metrics = run_one_pipeline(csv_path, args.cluster, args.predictor, train_records, val_records, test_records, region_meta, num_regions, num_roles, device, args)
    elapsed = time.time() - t0

    if metrics is None:
        print("\n[FAILED]")
        return

    seg = metrics.get("seg_mae", 0)
    cum = metrics.get("cum_mae", 0)
    dur = metrics.get("dur_mae_s", 0)
    win30 = metrics.get("window_acc", {}).get("win_30min", 0)

    print(f"\n{'='*70}")
    print(f"结果: {args.cluster} + {args.predictor}")
    print(f"  Seg MAE:  {seg:.0f}s ({seg/60:.1f}min)")
    print(f"  Cum MAE:  {cum:.0f}s ({cum/60:.1f}min)")
    print(f"  Dur MAE:  {dur:.0f}s ({dur/60:.1f}min)")
    print(f"  ±30min:   {win30:.1f}%")
    print(f"  耗时:     {elapsed:.0f}s")
    print(f"  参数:     {metrics.get('params', 0):,}")

    # 追加到汇总 JSON
    json_path = os.path.join(args.output_dir, "pipeline_comparison.json")
    all_results = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)

    all_results.setdefault(args.cluster, {})[args.predictor] = {
        "seg_mae": seg,
        "cum_mae": cum,
        "dur_mae_s": dur,
        "win_30min": win30,
        "params": metrics.get("params", 0),
        "time_s": metrics.get("time_s", elapsed),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印当前所有已完成的结果
    cluster_names = {"strat": "STRAT", "kmeans": "K-Means", "dbscan": "DBSCAN", "spectral": "Spectral(G)"}
    pred_names = {"median": "Median", "lightgbm": "LightGBM", "bigru": "BiGRU"}

    print(f"\n{'='*70}")
    print("已完成管道汇总:")
    print(f"{'='*70}")
    header = f"{'Clustering':<14} {'Predictor':<10} {'Seg MAE':>10} {'Cum MAE':>10} {'±30min':>8}"
    print(header)
    print("-" * 54)
    for cm in ["strat", "kmeans", "dbscan", "spectral"]:
        for pr in ["median", "lightgbm", "bigru"]:
            entry = all_results.get(cm, {}).get(pr)
            if entry:
                print(f"{cluster_names[cm]:<14} {pred_names[pr]:<10} " f"{entry['seg_mae']:10.0f} {entry['cum_mae']:10.0f} " f"{entry['win_30min']:7.1f}%")

    print(f"\n结果已追加到: {json_path}")


if __name__ == "__main__":
    main()
