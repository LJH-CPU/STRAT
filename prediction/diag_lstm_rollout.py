"""D1 诊断：诚实配置下 LSTM 的 teacher-forced 与 rollout 下一区域准确率，对比 STTT。

自回归 rollout 协议与 STTT 的 rollout_eval 一致（prefix_len=2，逐位置预测+扩展）。
仅诊断用。
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
from ar_model import create_dataloader, GPS_SEG_DIM
import transformer_baselines as lb
from transformer_eval import rollout_eval
import run_transformer_experiment as rte


def load_bundle(name, seed, max_len=8):
    cp = os.path.join(os.path.dirname(__file__), 'output', 'cache', f'{name}_{seed}_auto.pkl')
    if os.path.exists(cp):
        with open(cp, 'rb') as f:
            return pickle.load(f)
    import argparse as _ap
    a = _ap.Namespace(seed=seed, k_override=None, n_poi_regions=None, n_workers=8, max_len=max_len)
    csv = str(CLEANED_DIR / f'{name}_cleaned.csv')
    b = rte.build_bundle(name, csv, seed, a)
    with open(cp, 'wb') as f:
        pickle.dump(b, f)
    return b


@torch.no_grad()
def lstm_teacher_forced_preds(model, loader, device):
    """LSTM teacher-forced 逐位置 argmax 预测的下一区域，{ri: [b0,b1,...]}（段索引对齐）。"""
    model.eval()
    out = {}
    off = 0
    for b in loader:
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        logits = model(dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                       dev['role_ids'], torch.tensor(dev['route_id'], device=device))
        mask = dev['mask'] & (dev['tgt_region'] != -1)
        for j in range(logits.shape[0]):
            rows = mask[j].nonzero(as_tuple=False).squeeze(1).tolist()
            out[off + j] = logits[j][rows].argmax(-1).tolist()
        off += logits.shape[0]
    return out


@torch.no_grad()
def lstm_rollout_regions(model, records, region_meta, norm, device, prefix_len=2, max_len=8):
    """LSTM 自回归 rollout，返回每记录预测的下一区域序列 {rec_idx: {seg_idx: pred_b}}（与 STRAT 同对齐）。"""
    model.eval()
    from ar_model import POI_DIM
    out = {}
    for ri, r in enumerate(records):
        seq = r['seq']
        Lr = len(seq)
        if Lr < prefix_len + 1:
            continue
        obs = seq[:prefix_len]
        aligned = {}
        for step_i in range(prefix_len, Lr):
            obs_use = obs[:max_len]
            L = len(obs_use)
            ids = torch.full((1, max_len), -1, dtype=torch.long)
            role = torch.zeros(1, max_len, dtype=torch.long)
            geo = torch.zeros(1, max_len, 3 + POI_DIM)
            gps = torch.zeros(1, max_len, GPS_SEG_DIM)
            tm = torch.zeros(1, max_len, 2)
            hour = r['start_time_of_day']
            tm[0, :, 0] = np.sin(2 * np.pi * hour / 24.0)
            tm[0, :, 1] = np.cos(2 * np.pi * hour / 24.0)
            for i, rid in enumerate(obs_use):
                m = region_meta.get(rid, {})
                ids[0, i] = rid
                role[0, i] = m.get('role_id', 0)
                geo[0, i, 0] = (m.get('lat', 0) - norm['lat_mean']) / norm['lat_std']
                geo[0, i, 1] = (m.get('lon', 0) - norm['lon_mean']) / norm['lon_std']
                geo[0, i, 2] = (m.get('elev_mean', 0) - norm['elev_mean']) / norm['elev_std']
                pv = np.zeros(POI_DIM)
                for k, v in m.get('poi_profile', {}).items():
                    pv[0] = v
                geo[0, i, 3:] = torch.from_numpy((pv - norm['poi_mean']) / norm['poi_std'])
            gf = r['gps_seg_features']
            for i in range(min(L - 1, max_len)):
                arr = np.array(gf[i], dtype=np.float32)
                gps[0, i] = torch.from_numpy((arr - norm['gps_mean']) / norm['gps_std'])
            dev = dict(region_ids=ids.to(device), geo=geo.to(device), gps=gps.to(device),
                       time_feat=tm.to(device), role_ids=role.to(device),
                       route_id=torch.tensor([r['route_id']], device=device))
            logits = model(dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                           dev['role_ids'], dev['route_id'])
            pred = int(logits[0, L - 1].argmax().item())
            aligned[step_i - 1] = pred
            if pred == -1:
                break
            obs = obs + [pred]
        out[ri] = aligned
    return out


@torch.no_grad()
def lstm_rollout_acc(model, records, region_meta, norm, device, prefix_len=2, max_len=8):
    """与 rollout_eval 同协议：LSTM 逐位置自回归下一区域准确率。"""
    model.eval()
    from ar_model import POI_DIM
    correct = total = 0
    per_pos = {}
    for r in records:
        seq = r['seq']
        Lr = len(seq)
        if Lr < prefix_len + 1:
            continue
        obs = seq[:prefix_len]
        for step_i in range(prefix_len, Lr):
            L = len(obs)
            obs = obs[:max_len]
            L = len(obs)
            ids = torch.full((1, max_len), -1, dtype=torch.long)
            role = torch.zeros(1, max_len, dtype=torch.long)
            geo = torch.zeros(1, max_len, 3 + POI_DIM)
            gps = torch.zeros(1, max_len, GPS_SEG_DIM)
            tm = torch.zeros(1, max_len, 2)
            hour = r['start_time_of_day']
            tm[0, :, 0] = np.sin(2 * np.pi * hour / 24.0)
            tm[0, :, 1] = np.cos(2 * np.pi * hour / 24.0)
            for i, rid in enumerate(obs):
                m = region_meta.get(rid, {})
                ids[0, i] = rid
                role[0, i] = m.get('role_id', 0)
                geo[0, i, 0] = (m.get('lat', 0) - norm['lat_mean']) / norm['lat_std']
                geo[0, i, 1] = (m.get('lon', 0) - norm['lon_mean']) / norm['lon_std']
                geo[0, i, 2] = (m.get('elev_mean', 0) - norm['elev_mean']) / norm['elev_std']
                pv = np.zeros(POI_DIM)
                for k, v in m.get('poi_profile', {}).items():
                    pv[0] = v  # 简化：诊断用
                geo[0, i, 3:] = torch.from_numpy((pv - norm['poi_mean']) / norm['poi_std'])
            gf = r['gps_seg_features']
            for i in range(min(L - 1, max_len)):
                arr = np.array(gf[i], dtype=np.float32)
                gps[0, i] = torch.from_numpy((arr - norm['gps_mean']) / norm['gps_std'])
            dev = dict(region_ids=ids.to(device), geo=geo.to(device), gps=gps.to(device),
                       time_feat=tm.to(device), role_ids=role.to(device),
                       route_id=torch.tensor([r['route_id']], device=device))
            logits = model(dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                           dev['role_ids'], dev['route_id'])
            pred = int(logits[0, L - 1].argmax().item())
            correct += int(pred == seq[step_i])
            total += 1
            per_pos.setdefault(step_i, [0, 0])
            per_pos[step_i][0] += int(pred == seq[step_i])
            per_pos[step_i][1] += 1
            obs = obs + [pred]
    return correct / max(total, 1), per_pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes', nargs='+', default=['峨眉山', '青城山', '熊猫基地', '龙泉'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 100, 2024])
    p.add_argument('--epochs', type=int, default=80)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sttt = json.load(open(os.path.join(os.path.dirname(__file__), 'output', 'llm_main_transformer.json'), encoding='utf-8'))
    def sttt_key(name, seed):
        for k, v in sttt.items():
            if v.get('status') == 'ok' and k.startswith(f'{name}|') and f'seed{seed}|' in k:
                return v['model']
        return {}

    out = {}
    print(f'{"Scene":<10}{"seed":>5}{"LSTM tf":>8}{"LSTM roll":>9}{"STTT tf":>8}{"STTT roll":>9}')
    for name in args.scenes:
        for seed in args.seeds:
            b = load_bundle(name, seed)
            region_meta, norm = b['region_meta'], b['norm']
            test_dl, _ = create_dataloader(b['te_records'], region_meta, norm, batch_size=16, shuffle=False, max_len=8)
            model = lb.LSTMNextRegion(b['V'], b['geo_dim']).to(device)
            train_dl, _ = create_dataloader(b['tr_records'], region_meta, norm, batch_size=16, shuffle=True, max_len=8)
            model = lb.train_lstm(model, train_dl, device, epochs=args.epochs, seed=seed)
            tf, _ = lb.eval_lstm_next_region(model, test_dl, device)
            roll, _ = lstm_rollout_acc(model, b['te_records'], region_meta, norm, device)
            sm = sttt_key(name, seed)
            print(f'{name:<10}{seed:>5}{tf:>8.3f}{roll:>9.3f}{sm.get("tf_acc", 0):>8.3f}{sm.get("roll_acc", 0):>9.3f}')
            out[f'{name}|seed{seed}'] = {'lstm_tf': tf, 'lstm_roll': roll,
                                         'sttt_tf': sm.get('tf_acc'), 'sttt_roll': sm.get('roll_acc')}
    with open(os.path.join(os.path.dirname(__file__), 'output', 'llm_diag_lstm_rollout.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('saved llm_diag_lstm_rollout.json')


if __name__ == '__main__':
    main()
