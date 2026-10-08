"""④D: 训练集可达约束的 rollout 解码 —— 是否改善 STRAT 的部署（rollout）表现。"""
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
from transformer_eval import rollout_eval
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


def build_reach(tr_records, V):
    reach = np.zeros((V, V), dtype=bool)
    for r in tr_records:
        seq = r['seq']
        for a, b in zip(seq, seq[1:]):
            if a < V and b < V:
                reach[a, b] = True
    return reach


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '峨眉山', '熊猫基地', '龙泉'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--out', default='prediction/output/llm_constrained_rollout.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for name in args.scenes:
        for seed in args.seeds:
            try:
                b = load_bundle(name, seed)
                region_meta, norm = b['region_meta'], b['norm']
                tr, va, te = b['tr_records'], b['va_records'], b['te_records']
                V, R, geo_dim = b['V'], b['R'], b['geo_dim']
                train_dl, _ = create_dataloader(tr, region_meta, norm, batch_size=16, shuffle=True, max_len=8)
                val_dl, _ = create_dataloader(va, region_meta, norm, batch_size=16, shuffle=False, max_len=8)
                model = TrajectoryPredictor(num_regions=V, num_routes=R, geo_dim=geo_dim,
                                            pair_mean=b['pair_mean'], global_mean=b['gmean'],
                                            seg_p95=norm['seg_p95'], backbone='transformer', d_model=64,
                                            use_struct=True, use_prior=True, time_feats='full').to(device)
                model = rte.train_model(model, train_dl, val_dl, device, epochs=args.epochs, seed=seed, loss='mae')
                reach = build_reach(tr, V)
                r0 = rollout_eval(model, te, region_meta, norm, device)
                r1 = rollout_eval(model, te, region_meta, norm, device, reach=reach)
                a0 = r0['acc'][0] / max(r0['acc'][1], 1)
                a1 = r1['acc'][0] / max(r1['acc'][1], 1)
                c0 = float(np.mean(r0['arr_errs'])) / 60.0
                c1 = float(np.mean(r1['arr_errs'])) / 60.0
                print(f"[{name}|{seed}] roll: {a0:.3f}->{a1:.3f} | cumMAE: {c0:.1f}->{c1:.1f} min")
                results[f'{name}|{seed}'] = {'roll_plain': a0, 'roll_constrained': a1,
                                             'cum_plain': c0, 'cum_constrained': c1,
                                             'reach_density': float(reach.sum()) / max(V * V, 1)}
            except Exception as e:
                import traceback; traceback.print_exc()
                results[f'{name}|{seed}'] = {'error': str(e)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
