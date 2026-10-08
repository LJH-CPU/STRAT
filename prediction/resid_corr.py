"""A2: 段间残差相关检验（独立假设验证）。

收集概率头逐位置归一化残差 e_{i,k}=(true−pred)/σ，按位置算 Pearson 相关矩阵，
报平均 |r|（相邻位置对与总体）。若平均 |r|<0.1 则认为独立假设近似成立。
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import CLEANED_DIR
from ar_model import create_dataloader
from transformer_model import TrajectoryPredictor
from transformer_eval import teacher_forced_eval
import run_transformer_experiment as rte
import pickle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '峨眉山'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--out', default='prediction/output/llm_resid_corr.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for name in args.scenes:
        for seed in args.seeds:
            cp = os.path.join(os.path.dirname(__file__), 'output', 'cache', f'{name}_{seed}_auto.pkl')
            with open(cp, 'rb') as f:
                bundle = pickle.load(f)
            region_meta, norm = bundle['region_meta'], bundle['norm']
            tr, va, te = bundle['tr_records'], bundle['va_records'], bundle['te_records']
            train_dl, _ = create_dataloader(tr, region_meta, norm, batch_size=16, shuffle=True, max_len=8)
            val_dl, _ = create_dataloader(va, region_meta, norm, batch_size=16, shuffle=False, max_len=8)
            test_dl, _ = create_dataloader(te, region_meta, norm, batch_size=16, shuffle=False, max_len=8)
            model = TrajectoryPredictor(
                num_regions=bundle['V'], num_routes=bundle['R'], geo_dim=bundle['geo_dim'],
                pair_mean=bundle['pair_mean'], global_mean=bundle['gmean'], seg_p95=norm['seg_p95'],
                backbone='transformer', d_model=64, use_struct=True, use_prior=True, time_feats='full',
                prob=True,
            ).to(device)
            model = rte.train_model(model, train_dl, val_dl, device, epochs=args.epochs, seed=seed, loss='nll')
            sc = rte._calibrate_sigma(model, val_dl, device, norm['seg_p95'])
            ps = rte._calibrate_path_sigma(model, val_dl, device, norm['seg_p95'])
            tf = teacher_forced_eval(model, test_dl, device, norm['seg_p95'],
                                     path_sigma_cal=ps, return_path_z=True)
            rb = tf.get('_seg_resid_by_pos', {})
            positions = sorted(int(k) for k in rb.keys())
            if len(positions) < 2:
                print(f'[{name}|{seed}] 位置不足'); continue
            arr = {k: rb[str(k)] for k in positions}
            n = min(len(v) for v in arr.values())
            M = np.stack([arr[k][:n] for k in positions])  # (npos, n)
            corr = np.corrcoef(M)
            np.fill_diagonal(corr, np.nan)
            mean_abs = float(np.nanmean(np.abs(corr)))
            adj = [abs(corr[k, k + 1]) for k in range(len(positions) - 1)]
            mean_adj = float(np.mean(adj))
            print(f"[{name}|{seed}] positions={positions} mean|r|={mean_abs:.3f} adj|r|={mean_adj:.3f} "
                  f"sigma_cal={sc:.2f} path_cal={ps:.2f}")
            results[f'{name}|{seed}'] = {'mean_abs_r': mean_abs, 'mean_adj_r': mean_adj,
                                         'n_pos': len(positions), 'sigma_cal': sc, 'path_sigma_cal': ps}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
