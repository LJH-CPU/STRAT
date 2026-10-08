"""① 路线模板补全（非参数）：利用路线这一最强结构先验。

训练集：按 route_id 分组 → 每路线的位置向众数模板 M_r + 每位置段时长均值。
测试（无泄漏）：仅用观测前缀 → 按 LCP(prefix, M_r) 匹配路线 → 用 M_r 剩余段补全行程；
时长用该路线各位置均值。对比 STRAT rollout 与区域级检索补全。
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
from collections import Counter
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


def build_route_templates(tr_records, max_len=8):
    """{route_id: {'seq': [位置向众数区域], 'durs': [每位置均值时长]}}"""
    groups = {}
    for r in tr_records:
        rid = int(r['route_id'])
        groups.setdefault(rid, []).append(r)
    temps = {}
    for rid, recs in groups.items():
        L = max_len
        seq = []
        durs = []
        for j in range(L):
            regs = Counter()
            ds = []
            for r in recs:
                s = r['seq']
                if len(s) > j:
                    regs[s[j]] += 1
                    if len(r['segment_durations']) > j:
                        ds.append(r['segment_durations'][j])
            if regs:
                seq.append(regs.most_common(1)[0][0])
                durs.append(float(np.mean(ds)) if ds else 0.0)
            else:
                break
        temps[rid] = {'seq': seq, 'durs': durs, 'n': len(recs)}
    return temps


def assign_route(temps, prefix):
    best_rid, best_lcp = None, -1
    for rid, t in temps.items():
        m = t['seq']
        lcp = 0
        for j in range(min(len(prefix), len(m))):
            if prefix[j] == m[j]:
                lcp += 1
            else:
                break
        if lcp > best_lcp or (lcp == best_lcp and best_rid is not None and temps[rid]['n'] > temps[best_rid]['n']):
            best_lcp = lcp
            best_rid = rid
    return best_rid, best_lcp


def complete(temps, prefix, max_suffix=5):
    rid, _ = assign_route(temps, prefix)
    t = temps[rid]
    m = t['seq']
    p = len(prefix)
    suf = m[p:p + max_suffix]
    dur = t['durs'][p:p + max_suffix]
    return rid, suf, dur


def evaluate(te_records, temps, prefix_len, max_suffix=5, max_len=8):
    nxt_c = nxt_t = 0
    per_pos = {}
    route_exact_c = route_t = 0
    cum_errs = []
    for r in te_records:
        seq = r['seq']
        L = len(seq)
        if L <= prefix_len:
            continue
        prefix = tuple(seq[:prefix_len])
        rid, suf, dur = complete(temps, prefix, max_suffix)
        true_rem = seq[prefix_len:]
        if not suf:
            continue
        nxt_c += int(suf[0] == true_rem[0])
        nxt_t += 1
        k = min(len(suf), len(true_rem))
        for j in range(k):
            per_pos.setdefault(j + 1, [0, 0])
            per_pos[j + 1][0] += int(suf[j] == true_rem[j])
            per_pos[j + 1][1] += 1
        route_exact_c += int(k == len(true_rem) and all(suf[j] == true_rem[j] for j in range(k)))
        route_t += 1
        base = r['arrival_offsets'][prefix_len - 1]
        cp = 0.0
        for j, d in enumerate(dur):
            seg = prefix_len - 1 + j
            if seg >= len(r['arrival_offsets']) - 1 or seg >= max_len - 1:
                break
            cp += d
            cum_errs.append(abs(cp - (r['arrival_offsets'][seg + 1] - base)))
    return {
        'next_acc': nxt_c / max(nxt_t, 1),
        'route_exact': route_exact_c / max(route_t, 1),
        'per_pos': {str(k): c / t for k, (c, t) in sorted(per_pos.items())},
        'cum_mae_min': float(np.mean(cum_errs)) / 60.0 if cum_errs else None,
        'n': nxt_t,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '峨眉山'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--prefix', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--out', default='prediction/output/llm_route_template.json')
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
            temps = build_route_templates(b['tr_records'])
            sm = sttt_key(name, seed)
            line = f"[{name}|{seed}] n_routes={len(temps)} "
            for pl in args.prefix:
                e = evaluate(b['te_records'], temps, pl)
                cum = f"{e['cum_mae_min']:.1f}" if e['cum_mae_min'] is not None else "NA"
                line += f"p{pl}: next={e['next_acc']:.3f} rExact={e['route_exact']:.3f} cum={cum} | "
                results[f'{name}|{seed}|p{pl}'] = e
            line += f"STRAT roll={sm.get('roll_acc',0):.3f} rollCum={sm.get('roll_arr_mae_min')}"
            print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
