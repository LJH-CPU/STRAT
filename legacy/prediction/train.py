"""
训练入口：BiGRU 段时长预测（V5：去除特征泄露，加入 POI 语义 + 历史特征）。
"""

import os
import sys
import argparse
import pickle
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import default_clustered_csv, PREDICTION_OUTPUT_DIR, scenery_name_from_csv

from data_utils import (
    extract_first_occurrence_sequences,
    build_region_metadata,
    split_by_route,
    GPS_SEG_DIM,
    POI_JSON,
    POI_TYPE_L1,
)
from baseline import build_knn_baseline, evaluate_knn
from sequence_model import (
    SequencePredictor,
    create_dataloader,
)


def _get_poi_dim(scenery_name):
    """读取 POI 投影数据，返回该景区的 POI 类型维度"""
    if not os.path.exists(POI_JSON):
        return 0
    with open(POI_JSON, encoding="utf-8") as f:
        all_pois = json.load(f)
    seen_types = set()
    for p in all_pois:
        if p.get("scenery") != scenery_name:
            continue
        tc = p.get("type_code", "")[:2]
        name = POI_TYPE_L1.get(tc, tc)
        seen_types.add(name)
    return len(seen_types)


def relative_l1_loss(preds, targets, mask, eps=1e-3):
    p = preds[mask]
    t = targets[mask]
    denom = t.abs() + eps
    return (p - t).abs().div(denom).mean()


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        (region_ids, geo, gps_seg, time_feat, role_ids,
         targets, mask, lengths, _, _) = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
        ]

        optimizer.zero_grad()
        preds = model(region_ids, geo, gps_seg, time_feat, role_ids, lengths)
        seg_mask = mask.clone()
        loss = relative_l1_loss(preds, targets, seg_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * seg_mask.sum().item()
        total_samples += seg_mask.sum().item()

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_model(model, loader, dataset, device):
    model.eval()
    total_rl1 = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    for batch in loader:
        (region_ids, geo, gps_seg, time_feat, role_ids,
         targets, mask, lengths, _, _) = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
        ]

        preds = model(region_ids, geo, gps_seg, time_feat, role_ids, lengths)
        seg_mask = mask.clone()
        loss = relative_l1_loss(preds, targets, seg_mask)

        total_rl1 += loss.item() * seg_mask.sum().item()
        total_samples += seg_mask.sum().item()

        preds_denorm = preds[seg_mask].cpu().numpy() * (dataset.seg_p95 + 1e-6)
        targets_denorm = targets[seg_mask].cpu().numpy() * (dataset.seg_p95 + 1e-6)
        all_preds.extend(preds_denorm.flatten().tolist())
        all_targets.extend(targets_denorm.flatten().tolist())

    rl1 = total_rl1 / max(total_samples, 1)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))

    return rl1, mae


def main():
    parser = argparse.ArgumentParser(description='Train BiGRU segment-duration predictor (V5 POI)')
    parser.add_argument('--csv', type=str, default=default_clustered_csv(),
                        help='聚类后的 CSV 路径（默认取 cluster/output 下第一个 *_clustered.csv）')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--role_embed_dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default=str(PREDICTION_OUTPUT_DIR),
                        help='模型输出目录')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu')

    # ── 提取景区名（从 CSV 文件名） ──
    scenery_name = scenery_name_from_csv(args.csv)
    print("=" * 60)
    print(f"Data Preparation  ({scenery_name})")
    print("=" * 60)

    df = pd.read_csv(args.csv)
    print(f"  Points: {len(df):,}, Tracks: {df['trackId'].nunique():,}")

    # POI 维度（用于 geo 特征扩展）
    poi_dim = _get_poi_dim(scenery_name)
    print(f"  POI profile dimension: {poi_dim}")

    records = extract_first_occurrence_sequences(df, scenery_name=scenery_name)
    print(f"  Valid trajectories: {len(records)}")

    if poi_dim > 0:
        from data_utils import _load_poi_profiles as _load_p
        poi_profiles = _load_p(scenery_name)
    else:
        poi_profiles = None

    region_meta = build_region_metadata(df, poi_profiles=poi_profiles)
    num_regions = len(region_meta)
    num_roles = max(m['role_id'] for m in region_meta.values()) + 1
    print(f"  Regions: {num_regions}, Roles: {num_roles}")

    seq_lens = [len(r['seq']) for r in records]
    print(f"  Seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={np.mean(seq_lens):.1f}")

    train_records, val_records, test_records = split_by_route(records, seed=args.seed)
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    # ── k-NN Baseline ──
    print("\n" + "=" * 60)
    print("k-NN Baseline")
    print("=" * 60)
    route_stats = build_knn_baseline(train_records)
    perc_pos_mae, knn_mae, per_route_knn = evaluate_knn(route_stats, test_records)
    print(f"  k-NN Test MAE (segment): {knn_mae:.0f}s ({knn_mae/60:.1f}min)")
    for pos, mae in sorted(perc_pos_mae.items()):
        print(f"    Seg {pos}: MAE={mae:.0f}s ({mae/60:.1f}min)")

    # ── BiGRU Model ──
    print("\n" + "=" * 60)
    print("BiGRU Model Training (Segment Duration Prediction)")
    print("=" * 60)

    max_len = 8
    geo_dim = 3 + poi_dim  # (lat, lon, elev) + POI types
    print(f"  GPS_SEG_DIM={GPS_SEG_DIM}, geo_dim={geo_dim}")

    train_loader, train_dataset = create_dataloader(
        train_records, region_meta, num_regions, args.batch_size,
        shuffle=True, max_len=max_len, poi_dim=poi_dim,
    )
    val_loader, val_dataset = create_dataloader(
        val_records, region_meta, num_regions, args.batch_size,
        shuffle=False, max_len=max_len, poi_dim=poi_dim,
    )
    test_loader, test_dataset = create_dataloader(
        test_records, region_meta, num_regions, args.batch_size,
        shuffle=False, max_len=max_len, poi_dim=poi_dim,
    )

    print(f"  Segment P95: {train_dataset.seg_p95:.0f}s ({train_dataset.seg_p95/60:.1f}min)")

    model = SequencePredictor(
        num_regions=num_regions + 1,
        region_embed_dim=args.embed_dim,
        geo_dim=geo_dim,
        gps_seg_dim=GPS_SEG_DIM,
        pos_dim=8,
        time_dim=2,
        num_roles=num_roles,
        role_embed_dim=args.role_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20,
    )

    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_rl1, val_mae = evaluate_model(model, val_loader, val_dataset, device)

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch < 10:
            print(f"  Epoch {epoch:3d}: train_rl1={train_loss:.4f}, "
                  f"val_rl1={val_rl1:.4f}, val_mae={val_mae:.0f}s, lr={current_lr:.6f}")

        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    test_rl1, test_mae = evaluate_model(model, test_loader, test_dataset, device)

    print(f"\n  === Segment-Level Results ===")
    print(f"  k-NN Test MAE: {knn_mae:.0f}s ({knn_mae/60:.1f}min)")
    print(f"  GRU  Test MAE: {test_mae:.0f}s ({test_mae/60:.1f}min)")
    if knn_mae > 0:
        improvement = (knn_mae - test_mae) / knn_mae * 100
        print(f"  Improvement over k-NN: {improvement:+.1f}%")

    # Save
    torch.save(best_state, os.path.join(args.output_dir, 'sequence_model.pt'))
    metadata = {
        'region_meta': region_meta,
        'num_regions': num_regions,
        'seg_p95': train_dataset.seg_p95,
        'poi_dim': poi_dim,
        'geo_dim': geo_dim,
    }
    with open(os.path.join(args.output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    with open(os.path.join(args.output_dir, 'route_stats.pkl'), 'wb') as f:
        pickle.dump(route_stats, f)

    print(f"\n  Artifacts saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
