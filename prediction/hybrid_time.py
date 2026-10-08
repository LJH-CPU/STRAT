"""
H2 混合时长线：STRAT 预测下一区域（where）→ XGBoost 估段时长（when）。

流程（严格无泄漏）：
  bundle（run.py 缓存）→ 训练 STRAT（标准配置）→ teacher-forced 收集测试集预测区域
  → XGBoost 在训练集真值 b 上训练 → 用预测 b 评测 → 输出三线对比
    [STRAT 联合] [STRAT+XGB(预测区域)=混合] [XGB(真值区域)=基准]

用法：
  python prediction/hybrid_time.py --scenes 峨眉山 青城山 熊猫基地 龙泉 --seeds 42 100 2024
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
from transformer_eval import teacher_forced_eval
from xgboost_baseline import run_xgb_pred
import run_transformer_experiment as rte


def load_bundle(name, seed, max_len=8, n_workers=8):
    cp = os.path.join(os.path.dirname(__file__), 'output', 'cache', f'{name}_{seed}_auto.pkl')
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            return pickle.load(f)
    import argparse as _ap
    a = _ap.Namespace(seed=seed, k_override=None, n_poi_regions=None, n_workers=n_workers, max_len=max_len)
    csv = str(CLEANED_DIR / f'{name}_cleaned.csv')
    b = rte.build_bundle(name, csv, seed, a)
    with open(cp, 'wb') as f:
        pickle.dump(b, f)
    return b


@torch.no_grad()
def collect_predicted_regions(model, test_dl, device):
    """teacher-forced 逐位置 argmax 预测的下一区域（与 STRAT 内部 where→when 同口径）。"""
    pred_next = {}
    off = 0
    model.eval()
    for b in test_dl:
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        rlogits, _, _, _, _ = model(
            dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
            dev['role_ids'], dev['positions'], torch.tensor(dev['route_id'], device=device),
            dev['cum_dist'])
        mask = dev['mask'] & (dev['tgt_region'] != -1)
        for j in range(rlogits.shape[0]):
            rows = mask[j].nonzero(as_tuple=False).squeeze(1).tolist()
            pred_next[off + j] = rlogits[j][rows].argmax(-1).tolist()
        off += rlogits.shape[0]
    return pred_next


def run_scene(name, seed, device, epochs=80, max_len=8):
    bundle = load_bundle(name, seed, max_len)
    region_meta, norm = bundle['region_meta'], bundle['norm']
    V, R, geo_dim = bundle['V'], bundle['R'], bundle['geo_dim']
    tr, va, te = bundle['tr_records'], bundle['va_records'], bundle['te_records']

    train_dl, _ = create_dataloader(tr, region_meta, norm, batch_size=16, shuffle=True, max_len=max_len)
    val_dl, _ = create_dataloader(va, region_meta, norm, batch_size=16, shuffle=False, max_len=max_len)
    test_dl, _ = create_dataloader(te, region_meta, norm, batch_size=16, shuffle=False, max_len=max_len)

    model = TrajectoryPredictor(
        num_regions=V, num_routes=R, geo_dim=geo_dim, pair_mean=bundle['pair_mean'],
        global_mean=bundle['gmean'], seg_p95=norm['seg_p95'], backbone='transformer',
        d_model=64, use_struct=True, use_prior=True, time_feats='full',
    ).to(device)
    model = rte.train_model(model, train_dl, val_dl, device, epochs=epochs, seed=seed, loss='mae')

    pred_next = collect_predicted_regions(model, test_dl, device)
    tf = teacher_forced_eval(model, test_dl, device, norm['seg_p95'])
    hx = run_xgb_pred(tr, te, V, pred_next, seed=seed, max_len=max_len)

    return {
        'sttt_mae_min': tf['dur_mae_min'],
        'sttt_cum_mae_min': tf['cum_mae_min'],
        'hybrid_mae_min': hx['mae_min'],
        'hybrid_cum_mae_min': hx['cum_mae_min'],
        'hybrid_n': hx['n'],
        'n_te': len(te),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['峨眉山', '青城山', '熊猫基地', '龙泉'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--max_len', type=int, default=8)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--out', default='prediction/output/llm_hybrid_xgb.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    for name in args.scenes:
        for seed in args.seeds:
            try:
                r = run_scene(name, seed, device, args.epochs, args.max_len)
                print(f"[{name}|{seed}] STRAT mae={r['sttt_mae_min']:.1f} cum={r['sttt_cum_mae_min']:.1f} | "
                      f"HYBRID mae={r['hybrid_mae_min']:.1f} cum={r['hybrid_cum_mae_min']:.1f} | n_te={r['n_te']}")
                results[f'{name}|seed{seed}'] = r
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[f'{name}|seed{seed}'] = {'error': str(e)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
