"""① 检索增强轨迹 Transformer —— 轻量验证：训练集检索先验(记忆) 注入深度模型 logits。

假设：5000 条轨迹的全局统计是 LSTM/纯 Transformer 用不到的信息；
把"相似上下文的下一区域分布"作为先验叠加到模型 logits，应能帮下一区域。
实现：训练集步级 h0(投影输出) 为 key、下一区域为 value；
测试每步 h0 为 query → top-K 余弦 → P_mem → logits += λ·log(P_mem)。
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from config import CLEANED_DIR
from ar_model import create_dataloader
from transformer_model import TrajectoryPredictor
import transformer_baselines as lb
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
def build_memory(model, train_dl, device):
    """训练集步级记忆：h0(投影输出) 为 key，下一区域为 value。"""
    keys = []
    vals = []
    for b in train_dl:
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        h0 = model.projector(dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                             dev['role_ids'], dev['positions'])
        mask = dev['mask'] & (dev['tgt_region'] != -1)
        for j in range(h0.shape[0]):
            rows = mask[j].nonzero(as_tuple=False).squeeze(1).tolist()
            if rows:
                keys.append(h0[j][rows].cpu())
                vals.append(dev['tgt_region'][j][rows].cpu())
    keys = torch.cat(keys, 0)          # (K, d)
    vals = torch.cat(vals, 0)          # (K,)
    return keys, vals


@torch.no_grad()
def eval_with_prior(model, test_dl, device, keys, vals, V, lam, top_k=50, exclude_self=True):
    """teacher-forced：logits = 模型 logits + λ·log(P_mem)。"""
    model.eval()
    keys = keys.to(device)
    vals = vals.to(device)
    n = keys.shape[0]
    per_pos = {}
    for b in test_dl:
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        rlogits, dur_pred, _, _, _ = model(dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                                           dev['role_ids'], dev['positions'],
                                           torch.tensor(dev['route_id'], device=device),
                                           dev['cum_dist'])
        h0 = model.projector(dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                             dev['role_ids'], dev['positions'])
        mask = dev['mask'] & (dev['tgt_region'] != -1)
        # 逐样本逐位置检索
        for j in range(h0.shape[0]):
            rows = mask[j].nonzero(as_tuple=False).squeeze(1).tolist()
            if not rows:
                continue
            q = h0[j][rows]                       # (nrows, d)
            sim = F.normalize(q, dim=-1) @ F.normalize(keys, dim=-1).t()  # (nrows, K)
            sim, idx = sim.topk(top_k, dim=-1)     # (nrows, top_k)
            sv = torch.softmax(sim / 0.1, -1)      # (nrows, top_k)
            # 排除同轨迹（本测试记录不在训练 key 里；训练时需防，测试天然无泄漏）
            nb = vals[idx]                          # (nrows, top_k)
            # 分布
            P = torch.zeros(len(rows), V, device=device)
            P.scatter_add_(1, nb, sv)
            P = P / (P.sum(1, keepdim=True) + 1e-9)
            lg = rlogits[j][rows]
            new_logits = lg + lam * torch.log(P + 1e-9)
            pr = new_logits.argmax(-1)
            t = dev['tgt_region'][j][rows]
            pos = rows
            for kk, p in enumerate(pos):
                per_pos.setdefault(int(p + 1), [0, 0])
                per_pos[int(p + 1)][0] += int(pr[kk].item() == t[kk].item())
                per_pos[int(p + 1)][1] += 1
    acc = sum(c for c, _ in per_pos.values()) / max(sum(t for _, t in per_pos.values()), 1)
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['青城山', '熊猫基地'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lam', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument('--out', default='prediction/output/llm_memory_prior.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_json = json.load(open(os.path.join(os.path.dirname(__file__), 'output', 'llm_main_transformer.json'), encoding='utf-8'))
    def lstm_acc(name, seed):
        for k, v in main_json.items():
            if v.get('status') == 'ok' and k.startswith(f'{name}|') and f'seed{seed}|' in k:
                return v['baselines'].get('lstm_acc')
        return None

    results = {}
    for name in args.scenes:
        for seed in args.seeds:
            b = load_bundle(name, seed)
            region_meta, norm = b['region_meta'], b['norm']
            tr, va, te = b['tr_records'], b['va_records'], b['te_records']
            V, R, geo_dim = b['V'], b['R'], b['geo_dim']
            train_dl, _ = create_dataloader(tr, region_meta, norm, batch_size=16, shuffle=True, max_len=8)
            val_dl, _ = create_dataloader(va, region_meta, norm, batch_size=16, shuffle=False, max_len=8)
            test_dl, _ = create_dataloader(te, region_meta, norm, batch_size=16, shuffle=False, max_len=8)
            model = TrajectoryPredictor(num_regions=V, num_routes=R, geo_dim=geo_dim, pair_mean=b['pair_mean'],
                                        global_mean=b['gmean'], seg_p95=norm['seg_p95'], backbone='transformer',
                                        d_model=64, use_struct=True, use_prior=True, time_feats='full').to(device)
            model = rte.train_model(model, train_dl, val_dl, device, epochs=args.epochs, seed=seed, loss='mae')
            keys, vals = build_memory(model, train_dl, device)
            line = f"[{name}|{seed}] memK={keys.shape[0]} "
            for lam in args.lam:
                acc = eval_with_prior(model, test_dl, device, keys, vals, V, lam)
                line += f"λ={lam}:{acc:.3f} "
                results[f'{name}|{seed}|lam{lam}'] = acc
            line += f"| LSTM={lstm_acc(name, seed)}"
            print(line)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('已保存:', args.out)


if __name__ == '__main__':
    main()
