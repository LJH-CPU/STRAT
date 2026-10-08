"""
评估与可视化：GRU 段时长预测 + k-NN baseline 对比 + 三项验证。
V2: 段时长 → 累积时间转换，同时展示段级和累积级指标。
"""

import os
import sys
import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import default_clustered_csv, PREDICTION_OUTPUT_DIR

from data_utils import (
    extract_first_occurrence_sequences,
    build_region_metadata,
    split_by_route,
    GPS_SEG_DIM,
)
from baseline import build_knn_baseline, evaluate_knn, predict_knn
from sequence_model import (
    SequencePredictor,
    create_dataloader,
)

plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                        'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
except Exception:
    pass

# 这是一个 辅助函数 ，用于将段时长列表转换为累积到达时间列表。
def segments_to_cumulative(segments):
    """段时长列表 → 累积到达时间（起点=0）"""
    return [0.0] + list(np.cumsum(segments))

# 这是一个 评估函数 ，用于评估模型在测试集上的性能。
@torch.no_grad()
def predict_gru_segments(model, loader, dataset, device):
    """返回每条轨迹的段时长预测结果"""
    model.eval()
    results = []
    sp = dataset.seg_p95
    for batch in loader:
        (region_ids, geo, gps_seg, time_feat, role_ids,
         targets, mask, lengths, track_ids, route_ids) = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch
        ]
        preds = model(region_ids, geo, gps_seg, time_feat, role_ids, lengths)
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()

        for b in range(len(track_ids)):
            seq_len = lengths[b].item()
            n_segs = seq_len - 1
            if n_segs < 1:
                continue
            pred_segs = [preds_np[b, i + 1, 0] * sp for i in range(n_segs)]
            true_segs = [targets_np[b, i + 1, 0] * sp for i in range(n_segs)]
            results.append({
                'track_id': track_ids[b],
                'route_id': route_ids[b],
                'pred_segments': pred_segs,
                'true_segments': true_segs,
                'n_segs': n_segs,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate GRU + k-NN with 3 validations')
    parser.add_argument('--csv', type=str, default=default_clustered_csv(),
                        help='聚类后的 CSV 路径（默认取 cluster/output 下第一个 *_clustered.csv）')
    parser.add_argument('--compare_csv', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=str(PREDICTION_OUTPUT_DIR))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu')

    print("=" * 60)
    print("STRAT Prediction Evaluation")
    print("=" * 60)

    df = pd.read_csv(args.csv)
    records = extract_first_occurrence_sequences(df)
    region_meta = build_region_metadata(df)
    num_regions = len(region_meta)

    state_dict = torch.load(os.path.join(args.model_dir, 'sequence_model.pt'),
                            map_location=device, weights_only=True)
    with open(os.path.join(args.model_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    with open(os.path.join(args.model_dir, 'route_stats.pkl'), 'rb') as f:
        route_stats = pickle.load(f)

    train_records, val_records, test_records = split_by_route(records, seed=args.seed)
    print(f"\n  Dataset: {len(records)} valid, {num_regions} regions")
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")

    test_loader, test_dataset = create_dataloader(
        test_records, region_meta, num_regions, args.batch_size, shuffle=False, max_len=8,
    )
    seg_p95 = test_dataset.seg_p95
    print(f"  Segment P95: {seg_p95:.0f}s ({seg_p95/60:.1f}min)")

    embed_dim = state_dict['region_embed.weight'].shape[1]
    role_embed_dim = state_dict.get('role_embed.weight', state_dict['region_embed.weight']).shape[1]
    num_roles = state_dict['role_embed.weight'].shape[0]
    model = SequencePredictor(
        num_regions=num_regions + 1,
        region_embed_dim=embed_dim,
        geo_dim=3,
        gps_seg_dim=GPS_SEG_DIM,
        pos_dim=8,
        time_dim=2,
        num_roles=num_roles,
        role_embed_dim=role_embed_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(state_dict)

    # === GRU segment predictions ===
    gru_results = predict_gru_segments(model, test_loader, test_dataset, device)

    # === k-NN segment baseline ===
    perc_pos_knn, knn_seg_mae, _ = evaluate_knn(route_stats, test_records)

    # === Segment-level GRU MAE ===
    gru_seg_errs = {}
    for r in gru_results:
        for i in range(r['n_segs']):
            err = abs(r['pred_segments'][i] - r['true_segments'][i])
            gru_seg_errs.setdefault(i + 1, []).append(err)
    all_gru_seg_errs = np.concatenate(list(gru_seg_errs.values()))
    gru_seg_mae = float(np.mean(all_gru_seg_errs))

    print("\n" + "=" * 60)
    print("Segment-Level MAE (per segment duration)")
    print("=" * 60)
    print(f"  k-NN: {knn_seg_mae:.0f}s ({knn_seg_mae/60:.1f}min)")
    print(f"  GRU:  {gru_seg_mae:.0f}s ({gru_seg_mae/60:.1f}min)")
    if knn_seg_mae > 0:
        print(f"  GRU vs k-NN: {(knn_seg_mae - gru_seg_mae)/knn_seg_mae*100:+.1f}%")

    print(f"\n  {'Seg':>5} {'GRU(s)':>10} {'GRU(min)':>10} {'k-NN(s)':>10} {'k-NN(min)':>10}")
    for pos in sorted(set(list(gru_seg_errs.keys()) + list(perc_pos_knn.keys()))):
        gm = np.mean(gru_seg_errs.get(pos, [0]))
        km = perc_pos_knn.get(pos, 0)
        print(f"  {pos:5d} {gm:10.0f} {gm/60:10.1f} {km:10.0f} {km/60:10.1f}")

    # === Cumulative-level metrics ===
    print("\n" + "=" * 60)
    print("Cumulative-Level MAE (arrival time, cumsum of segments)")
    print("=" * 60)

    pos_gru_cum = {}
    pos_knn_cum = {}
    all_gru_cum = []
    all_knn_cum = []
    all_gru_dur = []
    all_knn_dur = []

    for r_rec, r_gru in zip(test_records, gru_results):
        true_segs = r_rec['segment_durations']
        true_cum = segments_to_cumulative(true_segs)
        gru_cum = segments_to_cumulative(r_gru['pred_segments'])
        knn_segs, _ = predict_knn(route_stats, r_rec)
        knn_cum = segments_to_cumulative(knn_segs)

        # Total duration error
        all_gru_dur.append(abs(gru_cum[-1] - true_cum[-1]) / 60.0)
        all_knn_dur.append(abs(knn_cum[-1] - true_cum[-1]) / 60.0)

        for i in range(1, len(true_cum)):
            ge = abs(gru_cum[i] - true_cum[i])
            ke = abs(knn_cum[i] - true_cum[i])
            all_gru_cum.append(ge)
            all_knn_cum.append(ke)
            pos_gru_cum.setdefault(i + 1, []).append(ge)
            pos_knn_cum.setdefault(i + 1, []).append(ke)

    gru_cum_mae = np.mean(all_gru_cum)
    knn_cum_mae = np.mean(all_knn_cum)

    print(f"  k-NN Cumulative MAE: {knn_cum_mae:.0f}s ({knn_cum_mae/60:.1f}min)")
    print(f"  GRU  Cumulative MAE: {gru_cum_mae:.0f}s ({gru_cum_mae/60:.1f}min)")
    if knn_cum_mae > 0:
        print(f"  GRU vs k-NN: {(knn_cum_mae - gru_cum_mae)/knn_cum_mae*100:+.1f}%")

    print(f"\n  {'Pos':>5} {'GRU(s)':>10} {'GRU(min)':>10} {'k-NN(s)':>10} {'k-NN(min)':>10}")
    for pos in sorted(set(list(pos_gru_cum.keys()) + list(pos_knn_cum.keys()))):
        gm = np.mean(pos_gru_cum.get(pos, [0]))
        km = np.mean(pos_knn_cum.get(pos, [0]))
        print(f"  {pos:5d} {gm:10.0f} {gm/60:10.1f} {km:10.0f} {km/60:10.1f}")

    # === Validation 1: Total Duration ===
    print("\n" + "=" * 60)
    print("Validation 1: Total Duration Prediction")
    print("=" * 60)
    all_gru_dur = np.array(all_gru_dur)
    all_knn_dur = np.array(all_knn_dur)
    print(f"  GRU  MAE:  {all_gru_dur.mean():.1f} min")
    print(f"  k-NN MAE:  {all_knn_dur.mean():.1f} min")
    print(f"  GRU  Median: {np.median(all_gru_dur):.1f} min")
    print(f"  GRU  P90:    {np.percentile(all_gru_dur, 90):.1f} min")

    # === Validation 2: Time Window Accuracy (cumulative) ===
    print("\n" + "=" * 60)
    print("Validation 2: Time Window Accuracy (cumulative arrival)")
    print("=" * 60)

    for wm in [15, 30, 60]:
        ws = wm * 60
        gru_correct = 0
        gru_total = 0
        knn_correct = 0
        knn_total = 0
        pos_gru = {}
        pos_knn = {}

        for r_rec, r_gru in zip(test_records, gru_results):
            true_cum = segments_to_cumulative(r_rec['segment_durations'])
            gru_cum = segments_to_cumulative(r_gru['pred_segments'])
            knn_segs, _ = predict_knn(route_stats, r_rec)
            knn_cum = segments_to_cumulative(knn_segs)

            for i in range(1, len(true_cum)):
                ge = abs(gru_cum[i] - true_cum[i]) < ws
                ke = abs(knn_cum[i] - true_cum[i]) < ws
                gru_total += 1
                knn_total += 1
                gru_correct += int(ge)
                knn_correct += int(ke)
                pos = i + 1
                pos_gru[pos] = pos_gru.get(pos, [0, 0])
                pos_gru[pos][0] += int(ge)
                pos_gru[pos][1] += 1
                pos_knn[pos] = pos_knn.get(pos, [0, 0])
                pos_knn[pos][0] += int(ke)
                pos_knn[pos][1] += 1

        print(f"\n  Window: \u00b1{wm}min")
        print(f"  GRU  Overall: {100*gru_correct/max(gru_total,1):.1f}% ({gru_correct}/{gru_total})")
        print(f"  k-NN Overall: {100*knn_correct/max(knn_total,1):.1f}% ({knn_correct}/{knn_total})")
        if wm == 30:
            print(f"  {'Pos':>5} {'GRU':>8} {'k-NN':>8}")
            for pos in sorted(set(list(pos_gru.keys()) + list(pos_knn.keys()))):
                ga = 100 * pos_gru.get(pos, [0, 1])[0] / pos_gru.get(pos, [0, 1])[1]
                ka = 100 * pos_knn.get(pos, [0, 1])[0] / pos_knn.get(pos, [0, 1])[1]
                print(f"  {pos:5d} {ga:7.1f}% {ka:7.1f}%")

    # === Validation 3: Clustering Quality Ablation ===
    print("\n" + "=" * 60)
    print("Validation 3: Clustering Quality Ablation")
    print("=" * 60)

    if not args.compare_csv or not os.path.exists(args.compare_csv):
        print("  (skipped: --compare_csv not provided)")
    else:
        print(f"  NEW: {os.path.basename(args.csv)}")
        print(f"  OLD: {os.path.basename(args.compare_csv)}")
        for label, path in [("NEW (first-occ)", args.csv), ("OLD (compare)", args.compare_csv)]:
            dfo = pd.read_csv(path)
            recs = extract_first_occurrence_sequences(dfo)
            tr, _, te = split_by_route(recs, seed=args.seed)
            rstats = build_knn_baseline(tr)
            _, seg_mae, _ = evaluate_knn(rstats, te)
            n_routes = dfo['route_id'].nunique()
            n_regions = dfo['region_id'].nunique()
            avg_seq = np.mean([len(r['seq']) for r in recs])
            print(f"\n  [{label}]")
            print(f"    Regions: {n_regions}, Routes: {n_routes}, Tracks: {len(recs)}")
            print(f"    Avg seq len: {avg_seq:.1f}")
            print(f"    k-NN Seg MAE: {seg_mae:.0f}s ({seg_mae/60:.1f}min)")

    # === Plot ===
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # 1: Segment-level MAE by position
    ax = axes[0, 0]
    positions = sorted(gru_seg_errs.keys())
    x = np.arange(len(positions))
    w = 0.35
    ax.bar(x - w/2, [np.mean(gru_seg_errs.get(p, [0]))/60 for p in positions], w, label='GRU')
    ax.bar(x + w/2, [perc_pos_knn.get(p, 0)/60 for p in positions], w, label='k-NN', color='orange')
    ax.set_title('Segment-Level MAE (per-segment duration error)')
    ax.set_xlabel('Segment (1=first transition, 2=second, ...)')
    ax.set_ylabel('MAE (minutes)')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()

    # 2: Cumulative MAE by position
    ax = axes[0, 1]
    positions = sorted(pos_gru_cum.keys())
    x = np.arange(len(positions))
    ax.bar(x - w/2, [np.mean(pos_gru_cum.get(p, [0]))/60 for p in positions], w, label='GRU')
    ax.bar(x + w/2, [np.mean(pos_knn_cum.get(p, [0]))/60 for p in positions], w, label='k-NN', color='orange')
    ax.set_title('Cumulative MAE (arrival time error, cumsum of segments)')
    ax.set_xlabel('Position (1=start, 2=first arrival, ...)')
    ax.set_ylabel('MAE (minutes)')
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.legend()

    # 3: Total Duration Error Distribution
    ax = axes[0, 2]
    m = np.percentile(np.concatenate([all_gru_dur, all_knn_dur]), 95)
    ax.hist(np.clip(all_gru_dur, 0, m), bins=25, alpha=0.6,
            label=f'GRU (mean={all_gru_dur.mean():.1f}min)', edgecolor='black')
    ax.hist(np.clip(all_knn_dur, 0, m), bins=25, alpha=0.6,
            label=f'k-NN (mean={all_knn_dur.mean():.1f}min)', edgecolor='black', color='orange')
    ax.axvline(all_gru_dur.mean(), color='#1f77b4', linestyle='--')
    ax.axvline(all_knn_dur.mean(), color='orange', linestyle='--')
    ax.set_title('Validation 1: Total Duration Error')
    ax.set_xlabel('Error (minutes)')
    ax.legend()

    # 4: Time window accuracy (cumulative)
    ax = axes[1, 0]
    for wm, color in [(15, '#2196F3'), (30, '#4CAF50'), (60, '#FF9800')]:
        ws = wm * 60
        accs = []
        for pos in range(2, 8):
            correct = 0
            total = 0
            for r_rec, r_gru in zip(test_records, gru_results):
                if pos > len(r_rec['segment_durations']) + 1:
                    continue
                true_cum = segments_to_cumulative(r_rec['segment_durations'])
                gru_cum = segments_to_cumulative(r_gru['pred_segments'])
                if pos <= len(gru_cum):
                    total += 1
                    if abs(gru_cum[pos - 1] - true_cum[pos - 1]) < ws:
                        correct += 1
            accs.append(100 * correct / max(total, 1) if total > 0 else 0)
        ax.plot(range(2, 8), accs, 'o-', color=color, label=f'\u00b1{wm}min')

    ax.set_title('Validation 2: GRU Window Accuracy (cumulative)')
    ax.set_xlabel('Position')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(range(2, 8))
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    # 5: Predicted vs True scatter (cumulative)
    ax = axes[1, 1]
    all_t = np.array(all_gru_cum) / 3600.0
    all_p = []
    for r_rec, r_gru in zip(test_records, gru_results):
        true_cum = segments_to_cumulative(r_rec['segment_durations'])
        gru_cum = segments_to_cumulative(r_gru['pred_segments'])
        for i in range(1, len(true_cum)):
            all_p.append(gru_cum[i] / 3600.0)
    all_p = np.array(all_p)
    ax.scatter(all_t, all_p, alpha=0.3, s=8)
    mv = max(all_t.max(), all_p.max()) * 1.1
    ax.plot([0, mv], [0, mv], 'r--')
    ax.fill_between([0, mv], [0, mv], [0.5, mv + 0.5], alpha=0.1, color='green', label='\u00b130min')
    ax.set_xlabel('True (hours)')
    ax.set_ylabel('Predicted (hours)')
    ax.set_title('GRU: Cumulative Predicted vs True')
    ax.legend()

    # 6: Example trajectories
    ax = axes[1, 2]
    n_show = min(5, len(test_records))
    for i in range(n_show):
        r = test_records[i]
        gr = gru_results[i]
        n = len(r['segment_durations'])
        true_cum = np.array(segments_to_cumulative(r['segment_durations'])) / 60.0
        gru_cum = np.array(segments_to_cumulative(gr['pred_segments'])) / 60.0
        ax.plot(range(n + 1), true_cum, 'o-', alpha=0.6, markersize=4)
        ax.plot(range(n + 1), gru_cum, 's--', alpha=0.6, markersize=4)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], color='gray', marker='o', linestyle='-'),
               Line2D([0], [0], color='gray', marker='s', linestyle='--')],
              ['True', 'GRU'], loc='upper left')
    ax.set_xlabel('Position')
    ax.set_ylabel('Cumulative Time (min)')
    ax.set_title('Example: Cumulative Arrival Predictions')

    plt.tight_layout()
    fig.savefig(os.path.join(args.model_dir, 'prediction_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Charts saved to {args.model_dir}/prediction_analysis.png")


if __name__ == '__main__':
    main()