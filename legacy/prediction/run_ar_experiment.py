"""
单景区自回归预测实验（严格无泄漏协议）。

流程：划分 → train 内聚类 → test 映射 → train 路线 → test 伪标签 →
      region_meta(train) → records → 归一化(train) → 训练/评测 + 基线。
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CLEANED_DIR

import ar_pipeline as ap
from ar_model import (
    ARPredictor, create_dataloader, build_normalizers, rel_l1, POI_DIM,
)
import ar_baselines as ab


def split_subtrain_val(df_train, ratio=0.85, seed=42):
    """train 再分 train/val（按 track，早停用）。"""
    rng = np.random.RandomState(seed)
    tracks = df_train['trackId'].unique()
    rng.shuffle(tracks)
    n = int(len(tracks) * ratio)
    tr = df_train[df_train['trackId'].isin(tracks[:n])].copy()
    va = df_train[df_train['trackId'].isin(tracks[n:])].copy()
    return tr, va


def train_ar(model, train_dl, val_dl, device, epochs=100, lr=1e-3, patience=25, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    best_loss = float('inf')
    best_state = None
    wait = 0

    def step(loader, train=True):
        model.train(train)
        tot = 0.0
        nb = 0
        for b in loader:
            dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            ids, geo, gps, tm, role, pos = (dev['region_ids'], dev['geo'], dev['gps'],
                                            dev['time_feat'], dev['role_ids'], dev['positions'])
            rlogits, dur, rlogits2 = model(ids, geo, gps, tm, role, pos)
            mask = dev['mask'] & (dev['tgt_region'] != -1)
            loss_r = nn.functional.cross_entropy(rlogits[mask], dev['tgt_region'][mask])
            loss_d = rel_l1(dur, dev['tgt_dur'], mask)
            route_t = torch.tensor(dev['route_id'], dtype=torch.long, device=device)
            # 用每条轨迹最后一个有效位置的 route 输出
            lens = torch.tensor(dev['length'], device=device) - 1
            rp = rlogits2[torch.arange(len(lens), device=device), lens]
            loss_c = nn.functional.cross_entropy(rp, route_t)
            loss = loss_r + loss_d + 0.5 * loss_c
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item()
            nb += 1
        return tot / max(nb, 1)

    for ep in range(epochs):
        tl = step(train_dl, True)
        vl = step(val_dl, False)
        sched.step(vl)
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def teacher_forced_eval(model, loader, device, seg_p95):
    """喂真实前缀，测 top-1 下一区域准确率 + 段时长 MAE（按位置）。"""
    model.eval()
    per_pos = {}
    dur_errs = []
    for b in loader:
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        ids, geo, gps, tm, role, pos = (dev['region_ids'], dev['geo'], dev['gps'],
                                        dev['time_feat'], dev['role_ids'], dev['positions'])
        rlogits, dur, _ = model(ids, geo, gps, tm, role, pos)
        mask = dev['mask'] & (dev['tgt_region'] != -1)
        pred = rlogits[mask].argmax(dim=1)
        true = dev['tgt_region'][mask]
        pos_idx = mask.nonzero(as_tuple=False)[:, 0].cpu().numpy()
        for pi, p, t in zip(pos_idx, pred.cpu().numpy(), true.cpu().numpy()):
            pp = pi + 1  # 第 pp 次转移
            per_pos.setdefault(pp, [0, 0])
            per_pos[pp][0] += int(p == t)
            per_pos[pp][1] += 1
        dur_pred = dur[mask].cpu().numpy() * seg_p95
        dur_true = dev['tgt_dur'][mask].cpu().numpy() * seg_p95
        dur_errs.extend(np.abs(dur_pred - dur_true))
    return per_pos, float(np.mean(dur_errs)) if dur_errs else 0.0


@torch.no_grad()
def rollout_eval(model, records, region_meta, norm, device, prefix_len=2, max_len=8):
    """从长度为 prefix_len 的真实前缀开始自回归 rollout，测下一区域 acc + 到达时间 MAE。"""
    model.eval()
    from ar_model import ARDataset
    stats = {'acc': [0, 0], 'arr_mae': [], 'per_pos': {}}
    for r in records:
        seq = r['seq']
        Lr = len(seq)
        if Lr < prefix_len + 1:
            continue
        obs = seq[:prefix_len]
        cur_arrival = r['arrival_offsets'][prefix_len - 1]
        # 构造输入（用 ARDataset 特征构造函数思路，但允许喂预测 token）
        for step_i in range(prefix_len, Lr):
            true_next = seq[step_i]
            pred_next, pred_dur = _predict_next(model, obs, r, region_meta, norm, device, max_len)
            stats['acc'][1] += 1
            stats['acc'][0] += int(pred_next == true_next)
            stats['per_pos'].setdefault(step_i, [0, 0])
            stats['per_pos'][step_i][0] += int(pred_next == true_next)
            stats['per_pos'][step_i][1] += 1
            cur_arrival += pred_dur
            true_arrival = r['arrival_offsets'][step_i]
            stats['arr_mae'].append(abs(cur_arrival - true_arrival))
            if pred_next == -1:
                break
            obs = obs + [pred_next]
    return stats


@torch.no_grad()
def _predict_next(model, obs_seq, record, region_meta, norm, device, max_len):
    """给定已观察区域序列，预测下一区域与段时长。"""
    from ar_model import ARDataset
    L = max_len
    obs = obs_seq[:L]
    Lr = len(obs)
    ids = torch.full((L,), -1, dtype=torch.long)
    role = torch.zeros(L, dtype=torch.long)
    geo = torch.zeros(L, 3 + POI_DIM)
    gps = torch.zeros(L, 8)
    tm = torch.zeros(L, 2)
    hour = record['start_time_of_day']
    tm[:, 0] = np.sin(2 * np.pi * hour / 24.0)
    tm[:, 1] = np.cos(2 * np.pi * hour / 24.0)
    for i, rid in enumerate(obs):
        m = region_meta.get(rid, {})
        ids[i] = rid
        role[i] = m.get('role_id', 0)
        geo[i, 0] = (m.get('lat', 0) - norm['lat_mean']) / norm['lat_std']
        geo[i, 1] = (m.get('lon', 0) - norm['lon_mean']) / norm['lon_std']
        geo[i, 2] = (m.get('elev_mean', 0) - norm['elev_mean']) / norm['elev_std']
        pv = np.zeros(POI_DIM)
        for k, v in m.get('poi_profile', {}).items():
            if k in ['餐饮', '休闲', '住宿', '风景名胜', '科教文化', '公共设施']:
                pv[['餐饮', '休闲', '住宿', '风景名胜', '科教文化', '公共设施'].index(k)] = v
        geo[i, 3:] = torch.from_numpy((pv - norm['poi_mean']) / norm['poi_std'])
    gf = record['gps_seg_features']
    # 已观察前缀对应的段特征（0..Lr-1）
    for i in range(min(Lr - 1, L)):
        arr = np.array(gf[i], dtype=np.float32)
        gps[i] = torch.from_numpy((arr - norm['gps_mean']) / norm['gps_std'])
    pos = torch.arange(L, dtype=torch.long)
    ids = ids.unsqueeze(0).to(device)
    geo = geo.unsqueeze(0).to(device)
    gps = gps.unsqueeze(0).to(device)
    tm = tm.unsqueeze(0).to(device)
    role = role.unsqueeze(0).to(device)
    pos = pos.unsqueeze(0).to(device)
    rlogits, dur, _ = model(ids, geo, gps, tm, role, pos)
    pred_next = int(rlogits[0, Lr - 1].argmax().item())
    pred_dur = float(dur[0, Lr - 1].item()) * norm['seg_p95']
    return pred_next, pred_dur


def run_scene(name, csv_path, seed, device, n_workers=4, epochs=80, max_len=8,
              n_poi_regions=None, force_k=None):
    t0 = time.time()
    out = {'seed': seed, 'status': 'ok'}
    df = pd.read_csv(csv_path)
    dtr, dte = ap.split_tracks(df, 0.7, seed)
    out['n_tracks'] = {'train': int(dtr['trackId'].nunique()), 'test': int(dte['trackId'].nunique())}

    # 1) train 内聚类
    if n_poi_regions is None and force_k is None:
        npr = ap.n_poi_regions_static(name)
    else:
        npr = n_poi_regions
    model_reg, dtr = ap.cluster_regions_train(dtr, n_poi_regions=npr,
                                              seed=seed, n_workers=n_workers, force_k=force_k)
    out['regions'] = model_reg['k']
    out['delta'] = model_reg['delta']
    out['n_poi_regions_hint'] = npr

    # 2) test 映射
    dte, mstats = ap.map_test_regions(dte, model_reg)
    out['test_map_dist'] = {k: round(v, 4) for k, v in mstats.items()}

    # 3/4) 路线
    dtr, t2r, rm = ap.discover_train_routes(dtr, n_workers=n_workers)
    out['n_routes'] = rm.get('n_routes')
    reps = ap._route_representatives(dtr, t2r)
    dte, pseudo = ap.assign_test_route_pseudolabels(dte, reps)
    out['test_route_match'] = {k: (round(v, 3) if isinstance(v, float) else v)
                               for k, v in pseudo.items()}

    # 5) region_meta + POI（train 统计）
    profiles = ap.build_poi_profiles(model_reg, name)
    region_meta = ap.build_region_meta_train(dtr, model_reg, profiles)

    # 6) records
    sub_tr, sub_va = split_subtrain_val(dtr, 0.85, seed)
    tr_records = ap.build_records(sub_tr, region_meta, name, model_reg)
    va_records = ap.build_records(sub_va, region_meta, name, model_reg)
    te_records = ap.build_records(dte, region_meta, name, model_reg,
                                  track_route_map=dict(zip(dte['trackId'], dte['route_id'])))
    out['n_records'] = {'train': len(tr_records), 'val': len(va_records), 'test': len(te_records)}

    norm = build_normalizers(tr_records, region_meta)

    V = model_reg['k']  # 区域 id = 质心下标 0..k-1
    R = max([r['route_id'] for r in tr_records] + [0]) + 1
    train_dl, _ = create_dataloader(tr_records, region_meta, norm, batch_size=16, shuffle=True, max_len=max_len)
    val_dl, _ = create_dataloader(va_records, region_meta, norm, batch_size=16, shuffle=False, max_len=max_len)
    test_dl, _ = create_dataloader(te_records, region_meta, norm, batch_size=16, shuffle=False, max_len=max_len)

    # 7) AR 模型
    model = ARPredictor(num_regions=V, num_routes=R, hidden_dim=64, num_layers=1).to(device)
    model = train_ar(model, train_dl, val_dl, device, epochs=epochs, seed=seed)

    per_pos, dur_mae = teacher_forced_eval(model, test_dl, device, norm['seg_p95'])
    tf_acc = {str(p): round(c / t, 4) for p, (c, t) in sorted(per_pos.items())}
    tf_total = sum(c for c, _ in per_pos.values())
    tf_correct = sum(c for c, _ in per_pos.values())
    out['teacher_forced'] = {
        'next_region_acc': round(sum(c for c, _ in per_pos.values()) / max(sum(t for _, t in per_pos.values()), 1), 4),
        'per_pos': tf_acc,
        'dur_mae_s': round(dur_mae, 1),
        'dur_mae_min': round(dur_mae / 60, 1),
    }

    roll = rollout_eval(model, te_records, region_meta, norm, device, prefix_len=2, max_len=max_len)
    out['rollout'] = {
        'next_region_acc': round(roll['acc'][0] / max(roll['acc'][1], 1), 4),
        'n_steps': roll['acc'][1],
        'per_pos': {str(p): round(c / t, 4) for p, (c, t) in sorted(roll['per_pos'].items())},
        'arrival_mae_s': round(float(np.mean(roll['arr_mae'])) if roll['arr_mae'] else 0.0, 1),
        'arrival_mae_min': round(float(np.mean(roll['arr_mae'])) / 60 if roll['arr_mae'] else 0.0, 1),
    }

    # 8) 基线
    tr_model, gb = ab.build_transition_model(tr_records)
    mk_acc, mk_n = ab.next_region_accuracy(tr_model, gb, te_records)
    gb_acc, gb_n = ab.next_region_global_freq(gb, te_records)
    maj = ab.majority_route(tr_records)
    maj_preds = [maj for _ in te_records]
    maj_acc, _ = ab.route_accuracy(maj_preds, te_records)
    knn_preds = ab.knn_prefix_route(tr_records, te_records)
    knn_acc, _ = ab.route_accuracy(knn_preds, te_records)
    dm, gmean = ab.build_duration_model(tr_records)
    knn_dur, knn_dur_n = ab.knn_duration_mae(dm, gmean, te_records)

    out['baselines'] = {
        'markov1_next_region_acc': round(mk_acc, 4),
        'global_freq_next_region_acc': round(gb_acc, 4),
        'majority_route_acc': round(maj_acc, 4),
        'knn_prefix_route_acc': round(knn_acc, 4),
        'knn_duration_mae_min': round(knn_dur / 60, 1),
    }
    out['time_s'] = round(time.time() - t0, 1)
    print(f"[{name}] k={out['regions']} routes={out.get('n_routes')} | "
          f"TF-acc={out['teacher_forced']['next_region_acc']} rollout-acc={out['rollout']['next_region_acc']} "
          f"rollout-arrMAE={out['rollout']['arrival_mae_min']}min | "
          f"MK={mk_acc:.3f} R-maj={maj_acc:.3f} R-knn={knn_acc:.3f} kNNdur={knn_dur/60:.1f}min")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenery', nargs='*', default=None)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='*', default=[42])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--k_override', type=int, default=None,
                        help='强制区域数 k（聚类粒度消融）')
    parser.add_argument('--out', type=str, default='prediction/output/ar_results.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    if args.scenery:
        scenes = [(n, str(CLEANED_DIR / f'{n}_cleaned.csv')) for n in args.scenery]
    else:
        import glob
        scenes = []
        for f in sorted(glob.glob(str(CLEANED_DIR / '*_cleaned.csv'))):
            name = os.path.basename(f).replace('_cleaned.csv', '')
            n = pd.read_csv(f, usecols=['trackId'])['trackId'].nunique()
            if n >= 30:
                scenes.append((name, f))

    results = {}
    for name, csv_path in scenes:
        for seed in args.seeds:
            key = f'{name}|seed{seed}'
            try:
                results[key] = run_scene(name, csv_path, seed, device,
                                         n_workers=args.n_workers, epochs=args.epochs,
                                         force_k=args.k_override)
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[key] = {'status': 'error', 'error': str(e)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {args.out}")


if __name__ == '__main__':
    main()
