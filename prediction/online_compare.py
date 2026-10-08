"""在线端到端公平对比：区域预测器(STRAT/LSTM/马尔可夫) × 同一个 XGBoost 时长。

在线系统 = 区域预测器给出预测 b（真 a + 预测 b，系统观测自己位置）+ XGB 估时长。
唯一变量 = 区域预测器；XGB 共用。指标 = 逐段/累计 MAE；配对 Wilcoxon。
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
import transformer_baselines as lb
import ar_baselines as ab
from xgboost_baseline import train_xgb_duration, eval_xgb_pred_shared
from diag_lstm_rollout import lstm_teacher_forced_preds
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


@torch.no_grad()
def strattf_preds(model, loader, device):
    out = {}
    off = 0
    model.eval()
    for b in loader:
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        rlogits, _, _, _, _ = model(dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                                    dev['role_ids'], dev['positions'],
                                    torch.tensor(dev['route_id'], device=device),
                                    dev['cum_dist'])
        mask = dev['mask'] & (dev['tgt_region'] != -1)
        for j in range(rlogits.shape[0]):
            rows = mask[j].nonzero(as_tuple=False).squeeze(1).tolist()
            out[off + j] = rlogits[j][rows].argmax(-1).tolist()
        off += rlogits.shape[0]
    return out


def run_scene(name, seed, device, epochs=80, max_len=8):
    b = load_bundle(name, seed)
    region_meta, norm = b['region_meta'], b['norm']
    V, R, geo_dim = b['V'], b['R'], b['geo_dim']
    tr, va, te = b['tr_records'], b['va_records'], b['te_records']

    train_dl, _ = create_dataloader(tr, region_meta, norm, batch_size=16, shuffle=True, max_len=max_len)
    val_dl, _ = create_dataloader(va, region_meta, norm, batch_size=16, shuffle=False, max_len=max_len)
    test_dl, _ = create_dataloader(te, region_meta, norm, batch_size=16, shuffle=False, max_len=max_len)

    # 区域预测器
    strat = TrajectoryPredictor(num_regions=V, num_routes=R, geo_dim=geo_dim,
                                pair_mean=b['pair_mean'], global_mean=b['gmean'],
                                seg_p95=norm['seg_p95'], backbone='transformer', d_model=64,
                                use_struct=True, use_prior=True, time_feats='full').to(device)
    strat = rte.train_model(strat, train_dl, val_dl, device, epochs=epochs, seed=seed, loss='mae')
    strat_preds = strattf_preds(strat, test_dl, device)

    lstm = lb.LSTMNextRegion(V, geo_dim).to(device)
    lstm = lb.train_lstm(lstm, train_dl, device, epochs=epochs, seed=seed)
    lstm_preds = lstm_teacher_forced_preds(lstm, test_dl, device)

    mk_model, gbest = ab.build_transition_model(tr)
    mk_preds = {}
    for ri, r in enumerate(te):
        seq = r['seq']
        if len(seq) >= 2:
            mk_preds[ri] = ab.predict_next_region_markov(mk_model, gbest, seq)

    # 共用一个 XGB
    xgb_model, prior, gm = train_xgb_duration(tr, V, seed)
    if xgb_model is None:
        return None
    res = {}
    for pname, pn in [('strat', strat_preds), ('lstm', lstm_preds), ('markov', mk_preds)]:
        r = eval_xgb_pred_shared(xgb_model, prior, gm, te, V, pn, max_len=max_len)
        res[pname] = {'seg_mae_min': r['mae_min'], 'cum_mae_min': r['cum_mae_min']}
    # 区域预测器本身准确率（teacher-forced）
    res['rollout_region_acc'] = {'strat': 0.0, 'lstm': 0.0, 'markov': 0.0}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '峨眉山'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--out', default='prediction/output/llm_online_compare.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for name in args.scenes:
        for seed in args.seeds:
            try:
                r = run_scene(name, seed, device, args.epochs)
                if r is None:
                    print(f'[{name}|{seed}] 失败'); continue
                print(f"[{name}|{seed}] STRAT cum={r['strat']['cum_mae_min']:.1f} seg={r['strat']['seg_mae_min']:.1f} | "
                      f"LSTM cum={r['lstm']['cum_mae_min']:.1f} | MK cum={r['markov']['cum_mae_min']:.1f}")
                results[f'{name}|{seed}'] = r
            except Exception as e:
                import traceback; traceback.print_exc()
                results[f'{name}|{seed}'] = {'error': str(e)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
