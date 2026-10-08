"""
自回归预测基线（全部只在 train 内统计）：
- 下一区域：一阶马尔可夫 / 全局最频繁转移
- 路线：多数类 / k-NN 前缀匹配
- 段时长：按 (from,to) 区域对的历史均值
"""

from collections import Counter, defaultdict

import numpy as np


# ── 下一区域 ──────────────────────────────────────────────
def build_transition_model(train_records):
    cnt = Counter()
    global_next = Counter()
    for r in train_records:
        seq = r['seq']
        for a, b in zip(seq, seq[1:]):
            cnt[(a, b)] += 1
            global_next[b] += 1
    model = {}
    for (a, b), c in cnt.items():
        model.setdefault(a, {})[b] = c
    global_best = global_next.most_common(1)[0][0] if global_next else -1
    return model, global_best


def predict_next_region_markov(model, global_best, seq):
    preds = []
    for a, b in zip(seq, seq[1:]):
        tbl = model.get(a, {})
        if tbl:
            preds.append(max(tbl, key=tbl.get))
        else:
            preds.append(global_best)
    return preds  # 长度 = len(seq)-1，对应真实下一区域 seq[1:]


def next_region_accuracy(model, global_best, test_records):
    correct = 0
    total = 0
    for r in test_records:
        seq = r['seq']
        if len(seq) < 2:
            continue
        preds = predict_next_region_markov(model, global_best, seq)
        for pred, true in zip(preds, seq[1:]):
            correct += int(pred == true)
            total += 1
    return correct / max(total, 1), total


def markov_rollout_regions(model, global_best, records, prefix_len=2):
    """一阶马尔可夫自回归 rollout，返回每记录预测的下一区域序列 {rec_idx: {seg_idx: pred_b}}（与 STRAT 同对齐）。"""
    out = {}
    for ri, r in enumerate(records):
        seq = r['seq']
        Lr = len(seq)
        if Lr < prefix_len + 1:
            continue
        obs = list(seq[:prefix_len])
        aligned = {}
        for step_i in range(prefix_len, Lr):
            tbl = model.get(obs[-1], {})
            pred = max(tbl, key=tbl.get) if tbl else global_best
            aligned[step_i - 1] = pred
            if pred == -1:
                break
            obs.append(pred)
        out[ri] = aligned
    return out


def next_region_global_freq(global_best, test_records):
    correct = 0
    total = 0
    for r in test_records:
        seq = r['seq']
        for true in seq[1:]:
            correct += int(global_best == true)
            total += 1
    return correct / max(total, 1), total


# ── 路线 ──────────────────────────────────────────────────
def majority_route(train_records):
    cnt = Counter(r['route_id'] for r in train_records)
    return cnt.most_common(1)[0][0] if cnt else -1


def knn_prefix_route(train_records, test_records, k=5):
    """最近 train 轨迹（最长公共前缀长度/编辑距离）的 route_id 投票。"""
    def _lcp_len(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    preds = []
    for r in test_records:
        seq = r['seq']
        sims = []
        for tr in train_records:
            if len(tr['seq']) < 2:
                continue
            lcp = _lcp_len(seq, tr['seq'])
            sims.append((lcp, tr['route_id']))
        sims.sort(key=lambda x: -x[0])
        top = sims[:k]
        votes = Counter(r for _, r in top if r >= 0)
        preds.append(votes.most_common(1)[0][0] if votes else -1)
    return preds


def route_accuracy(preds, test_records):
    correct = sum(1 for p, r in zip(preds, test_records) if p == r['route_id'])
    return correct / max(len(test_records), 1), len(test_records)


# ── 段时长 ────────────────────────────────────────────────
def build_duration_model(train_records):
    """返回 (按当前区域的时长均值, 全局均值)；不使用真值下一区域（无泄漏）。"""
    sums = defaultdict(float)
    counts = defaultdict(int)
    all_durs = []
    for r in train_records:
        seq = r['seq']
        for i, d in enumerate(r['segment_durations']):
            a = int(seq[i])
            sums[a] += d
            counts[a] += 1
            all_durs.append(d)
    mean = {k: sums[k] / counts[k] for k in sums}
    global_mean = float(np.mean(all_durs)) if all_durs else 3600.0
    return mean, global_mean


def knn_duration_mae(dur_model, global_mean, test_records):
    errs = []
    for r in test_records:
        seq = r['seq']
        for i, d in enumerate(r['segment_durations']):
            pred = dur_model.get(int(seq[i]), global_mean)
            errs.append(abs(pred - d))
    return float(np.mean(errs)) if errs else 0.0, len(errs)
