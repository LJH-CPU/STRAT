"""
Baseline 模块：k-NN 基线预测 —— 同路线历史轨迹的段时长中位数。
V2: 预测段时长（非累积时间）。
"""

import numpy as np
from collections import defaultdict


def build_knn_baseline(train_records):
    """
    对训练集中每条路线，统计每个段位（segment position）的时长分布。
    
    返回:
        route_stats: dict, route_id -> {
            seq_pattern: tuple,
            segment_stats: list[dict{median, p25, p75}],  # 每段的中位数等
        }
    """
    route_groups = defaultdict(list)
    for r in train_records:
        route_groups[r['route_id']].append(r)

    route_stats = {}
    for rid, recs in route_groups.items():
        seq_counter = defaultdict(int)
        for r in recs:
            seq_counter[tuple(r['seq'])] += 1
        main_seq = max(seq_counter, key=seq_counter.get)

        max_segs = max(len(r['segment_durations']) for r in recs)
        segment_stats = []
        for pos in range(max_segs):
            values = []
            for r in recs:
                if pos < len(r['segment_durations']):
                    values.append(r['segment_durations'][pos])
            if values:
                arr = np.array(values)
                segment_stats.append({
                    'median': float(np.median(arr)),
                    'p25': float(np.percentile(arr, 25)),
                    'p75': float(np.percentile(arr, 75)),
                    'mean': float(np.mean(arr)),
                    'count': len(arr),
                })
            else:
                segment_stats.append(None)

        route_stats[rid] = {
            'seq_pattern': main_seq,
            'segment_stats': segment_stats,
            'n_tracks': len(recs),
        }

    return route_stats


def predict_knn(route_stats, test_record):
    """
    用 k-NN baseline 预测一条测试轨迹的段时长。

    返回:
        pred_segments: list[float], 每段时长的中位数（秒）
        source_route: int
    """
    rid = test_record['route_id']
    n_segs = len(test_record['segment_durations'])

    if rid in route_stats:
        stats = route_stats[rid]
        preds = []
        for pos in range(n_segs):
            if pos < len(stats['segment_stats']) and stats['segment_stats'][pos] is not None:
                preds.append(stats['segment_stats'][pos]['median'])
            else:
                preds.append(0.0)
        return preds, rid

    test_seq = test_record['seq']
    best_rid = -1
    best_overlap = -1
    for rid2, info in route_stats.items():
        overlap = len(set(test_seq) & set(info['seq_pattern']))
        if overlap > best_overlap:
            best_overlap = overlap
            best_rid = rid2

    if best_rid > -1:
        stats = route_stats[best_rid]
        preds = []
        for pos in range(n_segs):
            if pos < len(stats['segment_stats']) and stats['segment_stats'][pos] is not None:
                preds.append(stats['segment_stats'][pos]['median'])
            else:
                preds.append(0.0)
        return preds, best_rid

    return [0.0] * n_segs, -1


def evaluate_knn(route_stats, test_records):
    """
    评估 k-NN baseline，返回段级别的误差。
    
    返回:
        per_position_mae: dict, seg_pos -> MAE (秒)  (1 = 第一段, 2 = 第二段, ...)
        overall_mae: float
    """
    per_position_errs = defaultdict(list)
    per_route_errs = defaultdict(list)

    for r in test_records:
        preds, _ = predict_knn(route_stats, r)
        for i, (p, t) in enumerate(zip(preds, r['segment_durations'])):
            err = abs(p - t)
            per_position_errs[i + 1].append(err)
            per_route_errs[r['route_id']].append(err)

    per_position_mae = {pos: np.mean(errs) for pos, errs in per_position_errs.items()}
    per_route_mae = {rid: np.mean(errs) for rid, errs in per_route_errs.items()}
    all_errs = np.concatenate(list(per_route_errs.values()))
    overall_mae = float(np.mean(all_errs))

    return per_position_mae, overall_mae, per_route_mae