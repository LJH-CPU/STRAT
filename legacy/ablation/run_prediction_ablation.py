"""
消融实验：预测侧特征/架构消融

遍历所有特征组合，训练+评估，汇总结果。
每次只改变一个变量，其他条件完全一致。
"""

import os
import sys
import argparse
import json
import pickle
import time
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'prediction'))

from prediction.data_utils import (
    extract_first_occurrence_sequences,
    build_region_metadata,
    split_by_route,
    GPS_SEG_DIM,
)
from prediction.baseline import build_knn_baseline, evaluate_knn, predict_knn
from prediction.sequence_model import (
    SequencePredictor,
    create_dataloader,
)


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


def segments_to_cumulative(segments):
    return [0.0] + list(np.cumsum(segments))


@torch.no_grad()
def predict_gru_segments(model, loader, dataset, device):
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


def compute_all_metrics(gru_results, test_records, route_stats):
    """计算全部评估指标：Seg MAE, Cum MAE, Dur MAE, Window Acc + 逐轨迹误差"""
    # 用 track_id 对齐预测结果和测试记录（防止 TrajectoryDataset 过滤导致长度不匹配）
    pred_map = {r['track_id']: r for r in gru_results}

    gru_seg_errs = []
    per_track_seg_mae = []
    for r in gru_results:
        track_errs = []
        for i in range(r['n_segs']):
            err = abs(r['pred_segments'][i] - r['true_segments'][i])
            gru_seg_errs.append(err)
            track_errs.append(err)
        per_track_seg_mae.append(float(np.mean(track_errs)) if track_errs else 0.0)
    seg_mae = float(np.mean(gru_seg_errs)) if gru_seg_errs else 0.0

    all_gru_cum = []
    all_gru_dur = []
    per_track_cum_mae = []
    for r_rec in test_records:
        r_gru = pred_map.get(r_rec['track_id'])
        if r_gru is None:
            continue
        true_cum = segments_to_cumulative(r_rec['segment_durations'])
        gru_cum = segments_to_cumulative(r_gru['pred_segments'])
        all_gru_dur.append(abs(gru_cum[-1] - true_cum[-1]) / 60.0)
        track_cum_errs = []
        for i in range(1, len(true_cum)):
            ge = abs(gru_cum[i] - true_cum[i])
            all_gru_cum.append(ge)
            track_cum_errs.append(ge)
        per_track_cum_mae.append(float(np.mean(track_cum_errs)) if track_cum_errs else 0.0)
    cum_mae = float(np.mean(all_gru_cum)) if all_gru_cum else 0.0
    dur_mae = float(np.mean(all_gru_dur) * 60) if all_gru_dur else 0.0

    window_accs = {}
    for wm in [15, 30, 60]:
        ws = wm * 60
        correct = 0
        total = 0
        for r_rec in test_records:
            r_gru = pred_map.get(r_rec['track_id'])
            if r_gru is None:
                continue
            true_cum = segments_to_cumulative(r_rec['segment_durations'])
            gru_cum = segments_to_cumulative(r_gru['pred_segments'])
            for i in range(1, len(true_cum)):
                if abs(gru_cum[i] - true_cum[i]) < ws:
                    correct += 1
                total += 1
        window_accs[f'win_{wm}min'] = 100.0 * correct / max(total, 1) if total > 0 else 0.0

    knn_seg_errs = []
    for r_rec in test_records:
        knn_preds, _ = predict_knn(route_stats, r_rec)
        for p, t in zip(knn_preds, r_rec['segment_durations']):
            knn_seg_errs.append(abs(p - t))
    knn_seg_mae = float(np.mean(knn_seg_errs)) if knn_seg_errs else 0.0

    return {
        'seg_mae': seg_mae,
        'cum_mae': cum_mae,
        'dur_mae_s': dur_mae,
        'window_acc': window_accs,
        'knn_seg_mae': knn_seg_mae,
        'per_track_seg_mae': per_track_seg_mae,
        'per_track_cum_mae': per_track_cum_mae,
    }


def train_one_config(config, train_records, val_records, test_records,
                     region_meta, num_regions, num_roles, device, args):
    """训练一个配置的模型并返回评估指标"""
    max_len = 8
    train_loader, train_dataset = create_dataloader(
        train_records, region_meta, num_regions, args.batch_size, shuffle=True, max_len=max_len)
    val_loader, val_dataset = create_dataloader(
        val_records, region_meta, num_regions, args.batch_size, shuffle=False, max_len=max_len)
    test_loader, test_dataset = create_dataloader(
        test_records, region_meta, num_regions, args.batch_size, shuffle=False, max_len=max_len)

    seg_p95 = train_dataset.seg_p95

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
        num_layers=2,
        dropout=args.dropout,
        use_gps=config.get('use_gps', True),
        use_role=config.get('use_role', True),
        use_time=config.get('use_time', True),
        use_pos=config.get('use_pos', True),
        bidirectional=config.get('bidirectional', True),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20)

    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_rl1, val_mae = evaluate_model(model, val_loader, val_dataset, device)
        scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    test_rl1, test_mae = evaluate_model(model, test_loader, test_dataset, device)

    gru_results = predict_gru_segments(model, test_loader, test_dataset, device)
    route_stats = build_knn_baseline(train_records)
    metrics = compute_all_metrics(gru_results, test_records, route_stats)

    metrics['test_rl1'] = float(test_rl1)
    metrics['test_mae'] = float(test_mae)
    metrics['seg_p95'] = float(seg_p95)
    metrics['best_val_mae'] = float(best_val_mae)
    metrics['params'] = total_params

    return metrics


ABLATION_CONFIGS = {
    "full": {
        "desc": "完整模型（所有特征 + BiGRU）",
        "use_gps": True, "use_role": True, "use_time": True,
        "use_pos": True, "bidirectional": True,
    },
    "no_gps": {
        "desc": "去掉 GPS 15维段内特征",
        "use_gps": False, "use_role": True, "use_time": True,
        "use_pos": True, "bidirectional": True,
    },
    "no_role": {
        "desc": "去掉 KMeans 景区角色 Embedding",
        "use_gps": True, "use_role": False, "use_time": True,
        "use_pos": True, "bidirectional": True,
    },
    "no_time": {
        "desc": "去掉时段特征 (sin/cos hour)",
        "use_gps": True, "use_role": True, "use_time": False,
        "use_pos": True, "bidirectional": True,
    },
    "no_pos": {
        "desc": "去掉位置编码 (Pos Embedding)",
        "use_gps": True, "use_role": True, "use_time": True,
        "use_pos": False, "bidirectional": True,
    },
    "uni_gru": {
        "desc": "单向 GRU（替代双向 GRU）",
        "use_gps": True, "use_role": True, "use_time": True,
        "use_pos": True, "bidirectional": False,
    },
}


def pvalue_to_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def compute_wilcoxon(all_results, configs):
    """对 full vs 每个变体做 Wilcoxon signed-rank test"""
    full = all_results.get('full', {})
    full_seg_per_track = full.get('per_track_seg_mae')
    full_cum_per_track = full.get('per_track_cum_mae')

    if full_seg_per_track is None:
        return {}

    pvalues = {}
    for name in configs:
        if name == 'full':
            continue
        m = all_results.get(name, {})
        seg_per_track = m.get('per_track_seg_mae')
        cum_per_track = m.get('per_track_cum_mae')

        if seg_per_track is None or len(seg_per_track) != len(full_seg_per_track):
            pvalues[name] = {'seg_p': None, 'seg_stars': 'N/A',
                             'cum_p': None, 'cum_stars': 'N/A'}
            continue

        try:
            _, seg_p = wilcoxon(full_seg_per_track, seg_per_track,
                                alternative='two-sided', zero_method='zsplit')
            seg_p = float(seg_p)
        except Exception:
            seg_p = None

        try:
            _, cum_p = wilcoxon(full_cum_per_track, cum_per_track,
                                alternative='two-sided', zero_method='zsplit')
            cum_p = float(cum_p)
        except Exception:
            cum_p = None

        pvalues[name] = {
            'seg_p': seg_p,
            'seg_stars': pvalue_to_stars(seg_p) if seg_p is not None else 'N/A',
            'cum_p': cum_p,
            'cum_stars': pvalue_to_stars(cum_p) if cum_p is not None else 'N/A',
        }
        stars = pvalue_to_stars(seg_p) if seg_p is not None else 'N/A'
        print(f"    Wilcoxon vs full: Seg p={seg_p:.4f} {stars}" if seg_p is not None
              else f"    Wilcoxon vs full: N/A")

    return pvalues


def main():
    parser = argparse.ArgumentParser(description='Prediction Ablation Study')
    parser.add_argument('--csv', type=str, default='cluster/output/峨眉山_clustered.csv')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--role_embed_dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='ablation/output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--configs', type=str, nargs='*',
                        default=['full', 'no_gps', 'no_role', 'no_time', 'no_pos', 'uni_gru'])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cpu')

    print("=" * 70)
    print("STRAT 预测侧消融实验")
    print("=" * 70)

    df = pd.read_csv(args.csv)
    print(f"\n[Data] Points: {len(df):,}, Tracks: {df['trackId'].nunique():,}")

    records = extract_first_occurrence_sequences(df)
    print(f"[Data] Valid trajectories: {len(records)}")

    region_meta = build_region_metadata(df)
    num_regions = len(region_meta)
    num_roles = max(m['role_id'] for m in region_meta.values()) + 1

    train_records, val_records, test_records = split_by_route(records, seed=args.seed)
    print(f"[Data] Train: {len(train_records)}, Val: {len(val_records)}, "
          f"Test: {len(test_records)}, Regions: {num_regions}")

    all_results = {}
    full_metrics = None

    for name in args.configs:
        config = ABLATION_CONFIGS[name]
        print(f"\n{'='*70}")
        print(f"  [{name}] {config['desc']}")
        print(f"{'='*70}")

        t_start = time.time()
        metrics = train_one_config(
            config, train_records, val_records, test_records,
            region_meta, num_regions, num_roles, device, args)
        elapsed = time.time() - t_start

        metrics['elapsed_s'] = elapsed
        metrics['desc'] = config['desc']
        all_results[name] = metrics

        if name == 'full':
            full_metrics = metrics

        print(f"    Seg MAE:  {metrics['seg_mae']:.0f}s ({metrics['seg_mae']/60:.1f}min)")
        print(f"    Cum MAE:  {metrics['cum_mae']:.0f}s ({metrics['cum_mae']/60:.1f}min)")
        print(f"    Dur MAE:  {metrics['dur_mae_s']:.0f}s ({metrics['dur_mae_s']/60:.1f}min)")
        print(f"    ±30min Acc: {metrics['window_acc'].get('win_30min', 0):.1f}%")
        print(f"    Time: {elapsed:.0f}s")

    # 统计检验：Wilcoxon signed-rank test
    print(f"\n{'='*70}")
    print("统计检验：Wilcoxon Signed-Rank Test (Full vs each variant)")
    print(f"{'='*70}")
    pvalues = compute_wilcoxon(all_results, args.configs)
    all_results['_wilcoxon_pvalues'] = pvalues

    print(f"\n{'='*70}")
    print("消融实验结果汇总")
    print(f"{'='*70}")

    header = f"{'Config':<14} {'Seg MAE':>10} {'Cum MAE':>10} {'Dur MAE':>10} {'±30min':>8} {'p(Seg)':>8} {'Δ Seg':>8}"
    print(header)
    print("-" * 78)

    for name in args.configs:
        m = all_results[name]
        seg_s = m['seg_mae']
        delta = ""
        if full_metrics and name != 'full':
            pct = (seg_s - full_metrics['seg_mae']) / full_metrics['seg_mae'] * 100
            delta = f"+{pct:.1f}%"
        elif name == 'full':
            delta = "baseline"
        p_stars = pvalues.get(name, {}).get('seg_stars', '—') if name != 'full' else '—'
        print(f"{name:<14} {seg_s:10.0f} {m['cum_mae']:10.0f} "
              f"{m['dur_mae_s']:10.0f} {m['window_acc']['win_30min']:7.1f}% {p_stars:>8} {delta:>8}")

    out_path = os.path.join(args.output_dir, 'prediction_ablation_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()