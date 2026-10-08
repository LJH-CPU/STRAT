"""
单景区 LLM/Transformer 轨迹预测实验（严格无泄漏协议）。

用法：
  python prediction/run_transformer_experiment.py --scenery 峨眉山 --backbone gpt2 --gpt2_ckpt /tmp/opencode/ms_cache/models/AI-ModelScope--gpt2/snapshots/master
  python prediction/run_transformer_experiment.py --all --backbone transformer --seeds 42 100 2024
  python prediction/run_transformer_experiment.py --scenery 青城山 --backbone lstm --use_struct 0 --use_prior 0   # 消融
"""

import os
import sys
import json
import time
import glob
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CLEANED_DIR

import ar_pipeline as ap
from ar_model import build_normalizers, create_dataloader, rel_l1
from transformer_model import TrajectoryPredictor, build_pair_mean
from transformer_eval import teacher_forced_eval, rollout_eval, dm_test, paired_wilcoxon
import transformer_baselines as lb
import ar_baselines as ab


def split_subtrain_val(df_train, ratio=0.85, seed=42):
    rng = np.random.RandomState(seed)
    tracks = df_train['trackId'].unique()
    rng.shuffle(tracks)
    n = int(len(tracks) * ratio)
    return (df_train[df_train['trackId'].isin(tracks[:n])].copy(),
            df_train[df_train['trackId'].isin(tracks[n:])].copy())


def _time_loss(pred, target, mask, loss):
    if loss == 'rel1':
        return rel_l1(pred, target, mask)
    p = pred[mask]
    t = target[mask]
    if loss == 'logmae':
        return (p.abs() + 1e-3).log().sub((t.abs() + 1e-3).log()).abs().mean()
    return (p - t).abs().mean()  # mae


def _prob_loss(pred, sigma, target, mask):
    """Gaussian NLL（概率化时长头）。"""
    p = pred[mask]
    s = sigma[mask]
    t = target[mask]
    return (s.log() + 0.5 * ((t - p) / s) ** 2).mean()


@torch.no_grad()
def _calibrate_sigma(model, loader, device, seg_p95, target=0.9):
    """验证集上校准 σ：缩放系数使目标 PICP(90%) 命中，缓解过度自信。"""
    model.eval()
    zs = []
    for b in loader:
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        _, dur_pred, _, _, sigma = model(
            dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
            dev['role_ids'], dev['positions'], torch.tensor(dev['route_id'], device=device),
            dev['cum_dist'])
        mask = dev['mask'] & (dev['tgt_region'] != -1)
        if float(sigma.abs().max().item()) <= 0:
            return None
        pred = dur_pred[mask].cpu().numpy() * seg_p95
        tru = dev['tgt_dur'][mask].cpu().numpy() * seg_p95
        sig = sigma[mask].cpu().numpy() * seg_p95
        zs.extend(np.abs((tru - pred) / (sig + 1e-9)))
    if len(zs) < 10:
        return None
    zs = np.array(zs)
    q = np.percentile(zs, 100.0 * target)
    s = float(q / 1.6449)
    model.set_sigma_cal(s)
    return s


def _calibrate_path_sigma(model, loader, device, seg_p95, target=0.9):
    """验证集上校准路径级 σ：乘性因子使路径级 PICP(90%) 命中。
    段级 σ 校准只保证逐段覆盖；路径级（累计）覆盖因误差相关/独立假设通常偏低，需单独校准。"""
    pz = teacher_forced_eval(model, loader, device, seg_p95, return_path_z=True).get('_path_z')
    if pz is None or len(pz) < 10:
        return None
    s = float(np.percentile(np.abs(pz), 100.0 * target) / 1.6449)
    return s


def train_model(model, train_dl, val_dl, device, epochs=60, lr=1e-3, patience=20, seed=42, verbose=False,
                loss='mae', route_loss_w=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8)
    best = float('inf')
    best_sd = None
    wait = 0

    def step(loader, train=True):
        model.train(train)
        tot = 0.0
        nb = 0
        for b in loader:
            dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            rlogits, dur_pred, route_logits, lens, sigma = model(
                dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
                dev['role_ids'], dev['positions'], torch.tensor(dev['route_id'], device=device),
                dev['cum_dist'])
            mask = dev['mask'] & (dev['tgt_region'] != -1)
            loss_r = nn.functional.cross_entropy(rlogits[mask], dev['tgt_region'][mask])
            if loss == 'nll' and sigma is not None:
                loss_t = _prob_loss(dur_pred, sigma, dev['tgt_dur'], mask)
            else:
                loss_t = _time_loss(dur_pred, dev['tgt_dur'], mask, loss)
            rt = torch.tensor(dev['route_id'], dtype=torch.long, device=device)
            valid_route = rt >= 0
            if valid_route.any():
                loss_c = nn.functional.cross_entropy(route_logits[valid_route], rt[valid_route])
            else:
                loss_c = torch.zeros((), device=device)
            loss_total = loss_r + loss_t + route_loss_w * loss_c
            if train:
                opt.zero_grad(); loss_total.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
            tot += loss_total.item(); nb += 1
        return tot / max(nb, 1)

    for ep in range(epochs):
        tl = step(train_dl, True)
        vl = step(val_dl, False)
        sched.step(vl)
        if verbose:
            print(f'    epoch {ep:3d}: train={tl:.4f} val={vl:.4f}')
        if vl < best:
            best = vl
            best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_sd is not None:
        model.load_state_dict(best_sd)
    return model


def build_bundle(name, csv_path, seed, args):
    """严格无泄漏数据流 → 可缓存的数据包（聚类/路线/统计/records）。"""
    df = pd.read_csv(csv_path)
    dtr, dte = ap.split_tracks(df, 0.7, seed)
    bundle = {'name': name, 'seed': seed,
              'n_tracks': {'train': int(dtr['trackId'].nunique()), 'test': int(dte['trackId'].nunique())}}

    npr = ap.n_poi_regions_static(name) if args.n_poi_regions is None else args.n_poi_regions
    model_reg, dtr = ap.cluster_regions_train(dtr, n_poi_regions=npr, seed=seed,
                                              n_workers=args.n_workers, force_k=args.k_override)
    bundle['regions'] = model_reg['k']
    dte, mstats = ap.map_test_regions(dte, model_reg)
    bundle['test_map_dist'] = {k: round(v, 4) for k, v in mstats.items()}

    dtr, t2r, rm = ap.discover_train_routes(dtr, n_workers=args.n_workers)
    bundle['n_routes'] = rm.get('n_routes')
    reps = ap._route_representatives(dtr, t2r)
    dte, pseudo = ap.assign_test_route_pseudolabels(dte, reps)
    bundle['test_route_match'] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pseudo.items()}

    profiles = ap.build_poi_profiles(model_reg, name)
    region_meta = ap.build_region_meta_train(dtr, model_reg, profiles)

    sub_tr, sub_va = split_subtrain_val(dtr, 0.85, seed)
    tr_records = ap.build_records(sub_tr, region_meta, name, model_reg)
    va_records = ap.build_records(sub_va, region_meta, name, model_reg)
    te_records = ap.build_records(dte, region_meta, name, model_reg,
                                  track_route_map=dict(zip(dte['trackId'], dte['route_id'])))
    bundle['n_records'] = {'train': len(tr_records), 'val': len(va_records), 'test': len(te_records)}

    norm = build_normalizers(tr_records, region_meta)
    V = model_reg['k']
    R = max([r['route_id'] for r in tr_records + va_records] + [0]) + 1
    pair_mean, gmean = build_pair_mean(tr_records, V)
    from ar_model import POI_DIM
    geo_dim = 3 + POI_DIM

    bundle.update({
        'tr_records': tr_records, 'va_records': va_records, 'te_records': te_records,
        'region_meta': region_meta, 'norm': norm, 'pair_mean': pair_mean, 'gmean': gmean,
        'V': V, 'R': R, 'geo_dim': geo_dim,
    })
    return bundle


def run_bundle(name, bundle, seed, device, args):
    """加载数据包 → 训练模型 → 评测 + 基线 + DM，返回结果 dict。"""
    t0 = time.time()
    out = {'seed': seed, 'status': 'ok'}
    for k in ['n_tracks', 'regions', 'n_routes', 'test_map_dist', 'test_route_match', 'n_records']:
        if k in bundle:
            out[k] = bundle[k]

    tr_records, va_records, te_records = bundle['tr_records'], bundle['va_records'], bundle['te_records']
    region_meta, norm = bundle['region_meta'], bundle['norm']
    V, geo_dim = bundle['V'], bundle['geo_dim']
    R = max([r['route_id'] for r in tr_records + va_records] + [0]) + 1  # 覆盖 val 中可能更大的 route_id
    pair_mean, gmean = bundle['pair_mean'], bundle['gmean']

    train_dl, _ = create_dataloader(tr_records, region_meta, norm, batch_size=16, shuffle=True, max_len=args.max_len, num_workers=4)
    val_dl, _ = create_dataloader(va_records, region_meta, norm, batch_size=16, shuffle=False, max_len=args.max_len, num_workers=4)
    test_dl, _ = create_dataloader(te_records, region_meta, norm, batch_size=16, shuffle=False, max_len=args.max_len, num_workers=4)

    # 时间间隔嵌入：训练集归一化间隔的分位阈值（无泄漏：仅训练集统计）
    interval_bins = None
    if getattr(args, 'interval', 0) == 1:
        nb = getattr(args, 'interval_bins', 8)
        segs = np.array([d for r in tr_records for d in r['segment_durations']],
                        dtype=np.float64) / max(norm['seg_p95'], 1e-6)
        interval_bins = np.quantile(segs, np.linspace(1.0 / nb, 1.0 - 1.0 / nb, nb - 1)).tolist()

    model = TrajectoryPredictor(
        num_regions=V, num_routes=R, geo_dim=geo_dim, pair_mean=pair_mean, global_mean=gmean,
        seg_p95=norm['seg_p95'], backbone=args.backbone, gpt2_ckpt=args.gpt2_ckpt,
        d_model=args.d_model, use_struct=(args.use_struct == 1), use_prior=(args.use_prior == 1),
        time_feats=getattr(args, 'time_feats', 'full'),
        prob=(getattr(args, 'prob', 0) == 1),
        aggregator=getattr(args, 'aggregator', 'rgcn'),
        edge_mask=getattr(args, 'edge_mask', (True, True, True, True)),
        interval_bins=interval_bins,
        causal_edges=(getattr(args, 'causal_edges', 1) == 1),
    ).to(device)
    model = train_model(model, train_dl, val_dl, device, epochs=args.epochs, seed=seed,
                        verbose=getattr(args, 'verbose', False),
                        loss=getattr(args, 'loss', 'mae'))

    if getattr(args, 'prob', 0) == 1 and not getattr(args, 'no_cal', False):
        sc = _calibrate_sigma(model, val_dl, device, norm['seg_p95'])
        out['sigma_cal'] = round(sc, 3) if sc else None

    # 路径级 σ 校准（仅概率化 + 校准模式）
    path_sigma_cal = None
    if getattr(args, 'prob', 0) == 1 and not getattr(args, 'no_cal', False):
        ps = _calibrate_path_sigma(model, val_dl, device, norm['seg_p95'])
        path_sigma_cal = ps
        out['path_sigma_cal'] = round(ps, 3) if ps else None

    # 门控 α：结构注入的残差权重（用于报告 mean±std）
    gate_alpha = None
    if getattr(args, 'use_struct', 1) == 1 and model.rgcn is not None and hasattr(model.rgcn, 'gate'):
        gate_alpha = float(torch.sigmoid(model.rgcn.gate).item())
    out['gate_alpha'] = gate_alpha

    tf = teacher_forced_eval(model, test_dl, device, norm['seg_p95'], path_sigma_cal=path_sigma_cal)
    roll = rollout_eval(model, te_records, region_meta, norm, device, prefix_len=2, max_len=args.max_len)
    out['model'] = {
        'backbone': args.backbone,
        'tf_acc': round(tf['acc'], 4),
        'tf_acc3': round(tf.get('acc3', 0.0), 4),
        'tf_acc5': round(tf.get('acc5', 0.0), 4),
        'tf_macro_f1': round(tf.get('macro_f1'), 4) if tf.get('macro_f1') is not None else None,
        'tf_per_pos': tf['per_pos'],
        'tf_per_pos3': tf.get('per_pos3', {}),
        'tf_dur_mae_min': round(tf['dur_mae_min'], 1),
        'tf_mape': round(tf['mape'], 4),
        'tf_cum_mae_min': round(tf.get('cum_mae_min', 0.0), 1),
        'tf_cum_window_acc': {k: round(v, 4) for k, v in tf.get('cum_window_acc', {}).items()},
        'prob': getattr(args, 'prob', 0) == 1,
        'crps': round(tf.get('crps', 0.0), 4) if tf.get('crps') is not None else None,
        'picp90': round(tf.get('picp90', 0.0), 4) if tf.get('picp90') is not None else None,
        'width90_min': round(tf.get('width90_min', 0.0), 1) if tf.get('width90_min') is not None else None,
        'reliability': tf.get('reliability', None),
        'path_crps_min': round(tf.get('path_crps_min', 0.0), 2) if tf.get('path_crps_min') is not None else None,
        'path_picp90': round(tf.get('path_picp90', 0.0), 4) if tf.get('path_picp90') is not None else None,
        'path_width90_min': round(tf.get('path_width90_min', 0.0), 1) if tf.get('path_width90_min') is not None else None,
        'path_reliability': tf.get('path_reliability', None),
        'roll_acc': round(roll['acc'][0] / max(roll['acc'][1], 1), 4),
        'roll_n': roll['acc'][1],
        'roll_per_pos': {str(k): v for k, v in sorted(roll['per_pos'].items())},
        'roll_arr_mae_min': round(float(np.mean(roll['arr_errs'])) / 60 if roll['arr_errs'] else 0.0, 1),
        'roll_arr_mape': round(float(np.mean(np.abs(np.array(roll['arr_errs'])) / (np.array(roll['arr_trues']) + 1e-6))) if roll['arr_trues'] else 0.0, 4),
    }

    # 基线
    tr_model, gb = ab.build_transition_model(tr_records)
    mk_acc, mk_n = ab.next_region_accuracy(tr_model, gb, te_records)
    km_dur, km_n = ab.knn_duration_mae(*ab.build_duration_model(tr_records), te_records)
    lin_dur, lin_n = lb.linear_time_mae(tr_records, te_records, pair_mean, gmean)
    maj = ab.majority_route(tr_records)
    maj_acc, _ = ab.route_accuracy([maj for _ in te_records], te_records)
    knn_route = ab.knn_prefix_route(tr_records, te_records)
    knn_acc, _ = ab.route_accuracy(knn_route, te_records)
    out['baselines'] = {
        'markov_acc': round(mk_acc, 4),
        'knn_dur_mae_min': round(km_dur / 60, 1),
        'linear_dur_mae_min': round(lin_dur / 60, 1) if lin_dur else None,
        'majority_route_acc': round(maj_acc, 4),
        'knn_prefix_route_acc': round(knn_acc, 4),
    }

    # LSTM 深度基线（下一区域）
    lstm_dl, _ = create_dataloader(tr_records, region_meta, norm, batch_size=16, shuffle=True, max_len=args.max_len)
    lstm_test_dl, _ = create_dataloader(te_records, region_meta, norm, batch_size=16, shuffle=False, max_len=args.max_len)
    lstm = lb.LSTMNextRegion(V, geo_dim).to(device)
    lstm = lb.train_lstm(lstm, lstm_dl, device, epochs=args.epochs, seed=seed)
    lstm_acc, _ = lb.eval_lstm_next_region(lstm, lstm_test_dl, device)
    out['baselines']['lstm_acc'] = round(lstm_acc, 4)

    # 显著性：DM（模型 teacher-forced 逐段时长误差 vs kNN 逐段误差，按 max_len 截断对齐）
    dur_model, gmean2 = ab.build_duration_model(tr_records)
    knn_seg_errs = []
    for r in te_records:
        seq = r['seq']
        if len(seq) < 2:
            continue
        n_seg = min(len(seq), args.max_len) - 1
        for i in range(n_seg):
            key = int(seq[i])
            pred = dur_model.get(key, gmean2)
            knn_seg_errs.append(abs(pred - r['segment_durations'][i]))
    knn_seg_errs = np.array(knn_seg_errs, dtype=np.float64)
    if len(tf['seg_errs']) == len(knn_seg_errs) and len(tf['seg_errs']) > 3:
        dm_stat, dm_p = dm_test(tf['seg_errs'], knn_seg_errs)
    else:
        dm_stat, dm_p = None, None
    out['dm_time_vs_knn'] = {'stat': dm_stat, 'p': dm_p,
                             'n': int(len(tf['seg_errs'])),
                             'n_knn': int(len(knn_seg_errs))}

    out['time_s'] = round(time.time() - t0, 1)
    print(f"[{name}|{args.backbone}] k={out['regions']} | tf_acc={tf['acc']:.3f} roll={out['model']['roll_acc']:.3f} "
          f"tf_durMAE={tf['dur_mae_min']:.1f}min | MK={mk_acc:.3f} LSTM={lstm_acc:.3f} "
          f"kNNdur={km_dur/60:.1f} lin_dur={lin_dur/60 if lin_dur else float('nan'):.1f}min")
    return out


def run_scene(name, csv_path, seed, device, args):
    """数据构建 + 训练评测（无缓存时用）。"""
    bundle = build_bundle(name, csv_path, seed, args)
    return run_bundle(name, bundle, seed, device, args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenery', nargs='*', default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--seeds', type=int, nargs='*', default=[42])
    p.add_argument('--backbone', default='transformer', choices=['gpt2', 'transformer', 'lstm'])
    p.add_argument('--gpt2_ckpt', default='/tmp/opencode/ms_cache/models/AI-ModelScope--gpt2/snapshots/master')
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--use_struct', type=int, default=1)
    p.add_argument('--use_prior', type=int, default=1)
    p.add_argument('--loss', default='mae', choices=['rel1', 'mae', 'logmae', 'nll'],
                   help='时长头损失：nll(概率化)/mae(推荐)/logmae/rel1(旧)')
    p.add_argument('--prob', type=int, default=0, help='概率化时长头(μ,σ)+Gaussian NLL')
    p.add_argument('--no_cal', action='store_true', help='跳过 σ 校准（未校准对照）')
    p.add_argument('--aggregator', default='rgcn', choices=['rgcn', 'gat', 'mlp'],
                   help='结构注入聚合器：rgcn(默认)/gat/mlp(无图)')
    p.add_argument('--edge_mask', type=int, nargs=4, default=[1, 1, 1, 1],
                   help='逐边开关 [时序,角色,POI,时段]，用于消融')
    p.add_argument('--interval', type=int, default=0, help='时间间隔嵌入（DSMR/GeoChronos 思路）')
    p.add_argument('--interval_bins', type=int, default=8)
    p.add_argument('--causal_edges', type=int, default=1, help='因果图（仅前向边）；0=旧双向（泄漏对照）')
    p.add_argument('--time_feats', default='full', choices=['basic', 'full'],
                   help='时长头特征：full 加 route/位置/累计(推荐), basic 为旧')
    p.add_argument('--k_override', type=int, default=None)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--max_len', type=int, default=8)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--n_poi_regions', type=int, default=None)
    p.add_argument('--out', default='prediction/output/llm_results.json')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    if args.scenery:
        scenes = [(n, str(CLEANED_DIR / f'{n}_cleaned.csv')) for n in args.scenery]
    else:
        scenes = []
        for f in sorted(glob.glob(str(CLEANED_DIR / '*_cleaned.csv'))):
            name = os.path.basename(f).replace('_cleaned.csv', '')
            n = pd.read_csv(f, usecols=['trackId'])['trackId'].nunique()
            if n >= 30:
                scenes.append((name, f))

    results = {}
    for name, csv_path in scenes:
        for seed in args.seeds:
            key = f'{name}|{args.backbone}|seed{seed}|s{args.use_struct}|p{args.use_prior}'
            try:
                results[key] = run_scene(name, csv_path, seed, device, args)
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key] = {'status': 'error', 'error': str(e)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {args.out}")


if __name__ == '__main__':
    main()
