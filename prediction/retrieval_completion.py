"""A: 检索式全程补全（实例化/非自回归范式）。

对测试前缀，从训练集检索前缀最相似的真实轨迹，取多数后缀一次性补全剩余行程；
到达时间用检索后缀的段时长均值。对比 STRAT 自回归 rollout。
无泄漏：检索仅训练集；测试前缀为观测值。
"""
import os
import sys
import json
import pickle
import argparse
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import CLEANED_DIR
import run_transformer_experiment as rte


def load_bundle(name, seed):
    cp = os.path.join(os.path.dirname(__file__), 'output', 'cache', f'{name}_{seed}_auto.pkl')
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            return pickle.load(f)
    import argparse as _ap
    a = _ap.Namespace(seed=seed, k_override=None, n_poi_regions=None, n_workers=8, max_len=8)
    b = rte.build_bundle(name, str(CLEANED_DIR / f'{name}_cleaned.csv'), seed, a)
    with open(cp, 'wb') as f:
        pickle.dump(b, f)
    return b


def complete(seqs, durs, prefix, min_matches=3, max_suffix=5):
    """一次性多数后缀补全。返回 (suffix_regions, suffix_durations)。"""
    idxs = []
    m = len(prefix)
    while m >= 1 and len(idxs) < min_matches:
        pm = prefix[:m]
        idxs = [i for i, s in enumerate(seqs) if len(s) > m and s[:m] == pm]
        m -= 1
    if not idxs:
        return [], []
    suffix = []
    dur = []
    for j in range(1, max_suffix + 1):
        nxt = Counter()
        ds = []
        for i in idxs:
            s = seqs[i]
            if len(s) >= len(prefix) + j:
                nxt[s[len(prefix) + j - 1]] += 1
                if len(durs[i]) > len(prefix) + j - 1:
                    ds.append(durs[i][len(prefix) + j - 1])
        if not nxt:
            break
        suffix.append(nxt.most_common(1)[0][0])
        dur.append(float(np.mean(ds)) if ds else 0.0)
    return suffix, dur


def evaluate(te_records, seqs, durs, prefix_len, min_matches=3, max_suffix=5):
    nxt_correct = nxt_total = 0
    per_pos = {}
    route_exact = route_total = 0
    edit_tot = edit_n = 0
    cum_errs, cum_trues = [], []
    for r in te_records:
        seq = r['seq']
        L = len(seq)
        if L <= prefix_len:
            continue
        prefix = tuple(seq[:prefix_len])
        suffix, dur = complete(seqs, durs, prefix, min_matches, max_suffix)
        true_rem = seq[prefix_len:]
        if not suffix:
            continue
        # 下一区域（A2）
        nxt_correct += int(suffix[0] == true_rem[0])
        nxt_total += 1
        # 逐位置 + 路线精确匹配 + 编辑距离（剩余行程）
        k = min(len(suffix), len(true_rem))
        for j in range(k):
            per_pos.setdefault(j + 1, [0, 0])
            per_pos[j + 1][0] += int(suffix[j] == true_rem[j])
            per_pos[j + 1][1] += 1
        route_exact += int(k == len(true_rem) and all(suffix[j] == true_rem[j] for j in range(k)))
        route_total += 1
        # 编辑距离（Levenshtein on compressed? 简化：|suffix|-|true| 长度差 + 不匹配数）
        ed = abs(len(suffix) - len(true_rem)) + sum(1 for j in range(k) if suffix[j] != true_rem[j])
        edit_tot += ed
        edit_n += 1
        # 累计到达 MAE：预测累计时长 vs 真实到达偏移（从 prefix 之后）
        cp = np.cumsum(dur)
        ct = np.array(r['arrival_offsets'][prefix_len:prefix_len + len(dur)]) - r['arrival_offsets'][prefix_len - 1]
        n = min(len(cp), len(ct))
        if n > 0:
            cum_errs.extend(np.abs(cp[:n] - ct[:n]))
            cum_trues.extend(ct[:n])
    return {
        'next_acc': nxt_correct / max(nxt_total, 1),
        'per_pos': {str(k): c / t for k, (c, t) in sorted(per_pos.items())},
        'route_exact': route_exact / max(route_total, 1),
        'edit_dist': edit_tot / max(edit_n, 1),
        'cum_mae_min': float(np.mean(cum_errs)) / 60.0 if cum_errs else None,
        'n': nxt_total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '峨眉山'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--prefix', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--out', default='prediction/output/llm_retrieval.json')
    args = p.parse_args()

    sttt = json.load(open(os.path.join(os.path.dirname(__file__), 'output', 'llm_main_transformer.json'), encoding='utf-8'))
    def sttt_key(name, seed):
        for k, v in sttt.items():
            if v.get('status') == 'ok' and k.startswith(f'{name}|') and f'seed{seed}|' in k:
                return v['model']
        return {}

    results = {}
    for name in args.scenes:
        for seed in args.seeds:
            b = load_bundle(name, seed)
            seqs = [tuple(r['seq']) for r in b['tr_records']]
            durs = [list(r['segment_durations']) for r in b['tr_records']]
            sm = sttt_key(name, seed)
            line = f"[{name}|{seed}] "
            for pl in args.prefix:
                e = evaluate(b['te_records'], seqs, durs, pl)
                cum = f"{e['cum_mae_min']:.1f}min" if e['cum_mae_min'] is not None else "NA"
                line += f"p{pl}: next={e['next_acc']:.3f} routeExact={e['route_exact']:.3f} cum={cum} | "
                results[f'{name}|{seed}|p{pl}'] = e
            line += f"STRAT roll={sm.get('roll_acc',0):.3f} rollCum={sm.get('roll_arr_mae_min')}min"
            print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
