"""A2: 隔离分解 —— 检索式补全 vs 自回归深度模型 的赢因在哪。

系统 = (区域源, 时长估计器)。给定 prefix_len=2，评估剩余行程。
区域源产生剩余段 i (i>=prefix_len-1) 的预测下一区域 b；时长估计器据 (真 a, 预测 b) 给段时长。
累计到达 MAE：cp[j]=Σ 前 j 段预测时长，真值 = off[prefix_len-1+j]-off[prefix_len-1]。

区域源：retr(检索多数后缀) / strat(STRAT rollout) / lstm(LSTM rollout) / true(真值)
时长估计：retrmean(检索后缀均值) / prior(区域对先验) / xgb
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import CLEANED_DIR
from ar_model import create_dataloader
from transformer_model import TrajectoryPredictor
from transformer_eval import rollout_predicted_regions
import transformer_baselines as lb
from xgboost_baseline import build_prior_table, train_xgb_duration
from retrieval_completion import complete
from diag_lstm_rollout import lstm_rollout_regions
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


def cum_mae(pred_dur, te, prefix_len=2, max_len=8):
    """pred_dur: {ri: {seg_idx: 秒}}；seg_idx 从 prefix_len-1 起（剩余段）。"""
    errs = []
    for ri, r in enumerate(te):
        pd = pred_dur.get(ri, {})
        if not pd:
            continue
        off = r['arrival_offsets']
        base = off[prefix_len - 1]
        order = sorted(pd.keys())
        cp = 0.0
        for seg in order:
            if seg >= max_len - 1 or seg >= len(off) - 1:
                break
            cp += pd[seg]
            true = off[seg + 1] - base
            errs.append(abs(cp - true))
    return float(np.mean(errs)) / 60.0 if errs else None


def build_source_durations(source, te, V, prior, gm, estimator, xgb_model=None, max_len=8, prefix_len=2):
    """source: {ri: {seg_idx: b}}；estimator ∈ {prior, xgb}。返回 {ri: {seg_idx: dur}}。"""
    out = {}
    for ri, r in enumerate(te):
        seq = r['seq']
        L = len(seq)
        pd = {}
        for i in range(prefix_len - 1, min(L - 1, max_len - 1)):
            a = int(seq[i])
            b = int(source.get(ri, {}).get(i, a))   # 缺预测时回退当前区域，不用真值下一区域
            pr = prior[a, b] if (a < V and b < V and prior[a, b] > 0) else gm
            if estimator == 'prior':
                d = pr
            else:
                gf = r['gps_seg_features'][i]
                row = np.array(list(gf) + [a, b, pr], dtype=np.float32).reshape(1, -1)
                d = float(np.maximum(xgb_model.predict(row), 0)[0])
            pd[i] = d
        if pd:
            out[ri] = pd
    return out


def run_scene(name, seed, device, epochs=80, max_len=8, prefix_len=2, min_matches=3):
    b = load_bundle(name, seed)
    region_meta, norm = b['region_meta'], b['norm']
    V, R, geo_dim = b['V'], b['R'], b['geo_dim']
    tr, va, te = b['tr_records'], b['va_records'], b['te_records']
    seqs = [tuple(r['seq']) for r in tr]
    durs = [list(r['segment_durations']) for r in tr]
    prior, gm = build_prior_table(tr, V)
    xgb_model, _, _ = train_xgb_duration(tr, V, seed)

    # 区域源
    retr_src, retr_dur = {}, {}
    for ri, r in enumerate(te):
        prefix = tuple(r['seq'][:prefix_len])
        suf, dur = complete(seqs, durs, prefix, min_matches, max_suffix=max_len - prefix_len)
        if suf:
            src = {i: suf[i - prefix_len] for i in range(prefix_len - 1, prefix_len - 1 + len(suf))}
            retr_src[ri] = src
            retr_dur[ri] = {i: dur[i - prefix_len] for i in range(prefix_len - 1, prefix_len - 1 + len(dur))}
    strat_roll = None
    # 训练 STRAT / LSTM
    train_dl, _ = create_dataloader(tr, region_meta, norm, batch_size=16, shuffle=True, max_len=max_len)
    val_dl, _ = create_dataloader(va, region_meta, norm, batch_size=16, shuffle=False, max_len=max_len)
    strat = TrajectoryPredictor(num_regions=V, num_routes=R, geo_dim=geo_dim, pair_mean=b['pair_mean'],
                                global_mean=b['gmean'], seg_p95=norm['seg_p95'], backbone='transformer',
                                d_model=64, use_struct=True, use_prior=True, time_feats='full').to(device)
    strat = rte.train_model(strat, train_dl, val_dl, device, epochs=epochs, seed=seed, loss='mae')
    strat_roll = rollout_predicted_regions(strat, te, region_meta, norm, device, prefix_len, max_len)
    lstm = lb.LSTMNextRegion(V, geo_dim).to(device)
    lstm = lb.train_lstm(lstm, train_dl, device, epochs=epochs, seed=seed)
    lstm_roll = lstm_rollout_regions(lstm, te, region_meta, norm, device, prefix_len, max_len)

    # 真值源
    true_src = {}
    for ri, r in enumerate(te):
        true_src[ri] = {i: int(r['seq'][i + 1]) for i in range(prefix_len - 1, min(len(r['seq']) - 1, max_len - 1))}

    def est(source):
        if source is None:
            return {}
        return build_source_durations(source, te, V, prior, gm, 'prior', max_len=max_len, prefix_len=prefix_len)

    def est_xgb(source):
        if source is None:
            return {}
        return build_source_durations(source, te, V, prior, gm, 'xgb', xgb_model, max_len, prefix_len)

    res = {
        'retr_full': cum_mae(retr_dur, te),
        'retr_prior': cum_mae(est(retr_src), te),
        'strat_prior': cum_mae(est(strat_roll), te),
        'lstm_prior': cum_mae(est(lstm_roll), te),
        'true_prior': cum_mae(est(true_src), te),
        'retr_xgb': cum_mae(est_xgb(retr_src), te),
        'strat_xgb': cum_mae(est_xgb(strat_roll), te),
        'lstm_xgb': cum_mae(est_xgb(lstm_roll), te),
        'true_xgb': cum_mae(est_xgb(true_src), te),
    }
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '峨眉山'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--out', default='prediction/output/llm_isolation.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    keys = ['retr_full', 'retr_prior', 'strat_prior', 'lstm_prior', 'true_prior',
            'retr_xgb', 'strat_xgb', 'lstm_xgb', 'true_xgb']
    for name in args.scenes:
        for seed in args.seeds:
            try:
                r = run_scene(name, seed, device, args.epochs)
                results[f'{name}|{seed}'] = r
                print(f"[{name}|{seed}] " + " ".join(f"{k}={r.get(k):.1f}" if r.get(k) is not None else f"{k}=NA" for k in keys))
            except Exception as e:
                import traceback; traceback.print_exc()
                results[f'{name}|{seed}'] = {'error': str(e)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
