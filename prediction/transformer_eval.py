"""
评测工具：teacher-forced / rollout 评估 + DM 检验 + Wilcoxon。
"""

import numpy as np
import torch
from scipy import stats
from ar_model import GPS_SEG_DIM


@torch.no_grad()
def teacher_forced_eval(model, loader, device, seg_p95, path_sigma_cal=None, return_path_z=False):
    """喂真实前缀，测下一区域 top-1/top-3 准确率（按位置）+ 时长 MAE（归一化域）。"""
    model.eval()
    per_pos = {}
    per_pos3 = {}
    per_pos5 = {}
    all_true = []
    all_pred = []
    dur_errs = []
    dur_trues = []
    cum_errs = []      # 累计到达误差（秒）
    cum_trues = []
    sigmas = []
    zvals = []
    path_segs = {}     # 记录级逐段 (pred, sigma, true)，用于路径级概率指标
    _off = 0
    for b in loader:
        dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        rlogits, dur_pred, _, lens, sigma = model(
            dev['region_ids'], dev['geo'], dev['gps'], dev['time_feat'],
            dev['role_ids'], dev['positions'], torch.tensor(dev['route_id'], device=device),
            dev['cum_dist'])
        mask = dev['mask'] & (dev['tgt_region'] != -1)
        top1 = rlogits[mask].argmax(-1)
        top3 = rlogits[mask].topk(3, dim=-1).indices
        top5 = rlogits[mask].topk(min(5, rlogits.shape[-1]), dim=-1).indices
        true = dev['tgt_region'][mask]
        pos = mask.nonzero(as_tuple=False)[:, 0].cpu().numpy() + 1
        all_true.extend(true.cpu().numpy().tolist())
        all_pred.extend(top1.cpu().numpy().tolist())
        for p, pr, t, t3, t5 in zip(pos, top1.cpu().numpy(), true.cpu().numpy(),
                                    top3.cpu().numpy(), top5.cpu().numpy()):
            per_pos.setdefault(int(p), [0, 0])
            per_pos[int(p)][0] += int(pr == t)
            per_pos[int(p)][1] += 1
            per_pos3.setdefault(int(p), [0, 0])
            per_pos3[int(p)][0] += int(t in t3)
            per_pos3[int(p)][1] += 1
            per_pos5.setdefault(int(p), [0, 0])
            per_pos5[int(p)][0] += int(t in t5)
            per_pos5[int(p)][1] += 1
        dp = dur_pred[mask].cpu().numpy() * seg_p95
        dt = dev['tgt_dur'][mask].cpu().numpy() * seg_p95
        dur_errs.extend(np.abs(dp - dt))
        dur_trues.extend(dt)
        # 概率化指标：收集 sigma（若概率头）
        if sigma is not None and float(sigma.abs().max().item()) > 0:
            ds = sigma[mask].cpu().numpy() * seg_p95
            z = (dt - dp) / (ds + 1e-9)
            sigmas.extend(ds); zvals.extend(z)
        # 逐样本累计到达误差
        B, L = dur_pred.shape
        for bi in range(B):
            vp = (mask[bi]).nonzero(as_tuple=False)[:, 0].cpu().numpy()
            if len(vp) == 0:
                _off += 1
                continue
            pred_seg = dur_pred[bi, vp].cpu().numpy() * seg_p95
            true_seg = dev['tgt_dur'][bi, vp].cpu().numpy() * seg_p95
            cum_pred = np.cumsum(pred_seg)
            cum_true = np.cumsum(true_seg)
            cum_errs.extend(np.abs(cum_pred - cum_true))
            if sigma is not None and float(sigma.abs().max().item()) > 0:
                sigma_seg = sigma[bi, vp].cpu().numpy() * seg_p95
                path_segs[_off] = {'p': pred_seg, 's': sigma_seg, 't': true_seg}
            _off += 1
            cum_trues.extend(cum_true)
    total = sum(t for _, t in per_pos.values())
    total3 = sum(t for _, t in per_pos3.values())
    total5 = sum(t for _, t in per_pos5.values())
    acc = sum(c for c, _ in per_pos.values()) / max(total, 1)
    acc3 = sum(c for c, _ in per_pos3.values()) / max(total3, 1)
    acc5 = sum(c for c, _ in per_pos5.values()) / max(total5, 1)
    # 分类完整性指标（#7）：macro-F1 / per-class recall
    try:
        from sklearn.metrics import f1_score, recall_score
        _yt = np.array(all_true); _yp = np.array(all_pred)
        macro_f1 = float(f1_score(_yt, _yp, average="macro", zero_division=0))
        per_class_recall = recall_score(_yt, _yp, average=None, zero_division=0).tolist()
    except Exception:
        macro_f1, per_class_recall = None, None
    mae = float(np.mean(dur_errs)) if dur_errs else 0.0
    mape = float(np.mean(np.abs(np.array(dur_errs)) / (np.array(dur_trues) + 1e-6))) if dur_trues else 0.0
    cum_errs = np.array(cum_errs, dtype=np.float64)
    cum_mae = float(np.mean(cum_errs)) if len(cum_errs) else 0.0
    windows = {}
    for w in [15, 30, 60]:
        windows[str(w)] = float(np.mean(cum_errs < w * 60)) if len(cum_errs) else 0.0
    # 概率化指标（若 sigma 可用）
    prob_out = {}
    if len(zvals) > 3:
        z = np.array(zvals, dtype=np.float64)
        sig = np.array(sigmas, dtype=np.float64)
        from scipy.stats import norm as _norm
        Phi = _norm.cdf(z); phi = _norm.pdf(z)
        crps = np.mean(sig * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi)))
        picp90 = float(np.mean(np.abs(z) <= 1.6449))
        width90 = float(np.mean(2 * 1.6449 * sig))
        qs = np.arange(0.05, 0.96, 0.05)
        reli = [(float(q), float(np.mean(Phi <= q))) for q in qs]
        prob_out = {'crps': float(crps), 'picp90': picp90,
                    'width90_min': width90 / 60.0, 'reliability': reli}
    # 路径级概率指标：沿轨迹传播 (μ,σ) → 整条到达时间分布 N(Σμ,Σσ²)
    if path_segs:
        z_paths, sig_paths, w_paths = [], [], []
        for segs in path_segs.values():
            cp = np.cumsum(segs['p'])
            cs = np.sqrt(np.cumsum(segs['s'] ** 2) + 1e-9)
            if path_sigma_cal is not None:
                cs = cs * path_sigma_cal
            ct = np.cumsum(segs['t'])
            z_paths.extend((ct - cp) / cs)
            sig_paths.extend(cs)
        z_path = np.array(z_paths, dtype=np.float64)
        sig_path = np.array(sig_paths, dtype=np.float64)
        from scipy.stats import norm as _norm
        Phi_p = _norm.cdf(z_path); phi_p = _norm.pdf(z_path)
        path_crps = np.mean(sig_path * (z_path * (2 * Phi_p - 1) + 2 * phi_p - 1.0 / np.sqrt(np.pi)))
        path_picp90 = float(np.mean(np.abs(z_path) <= 1.6449))
        path_width90 = float(np.mean(2 * 1.6449 * sig_path))
        reli_p = [(float(q), float(np.mean(Phi_p <= q))) for q in qs]
        prob_out['path_crps'] = float(path_crps)
        prob_out['path_crps_min'] = float(path_crps) / 60.0
        prob_out['path_picp90'] = path_picp90
        prob_out['path_width90_min'] = path_width90 / 60.0
        prob_out['path_reliability'] = reli_p
        if return_path_z:
            prob_out['_path_z'] = z_path
        # 逐位置段残差（用于独立假设检验）
        resid_by_pos = {}
        for segs in path_segs.values():
            e = (segs['t'] - segs['p']) / (segs['s'] + 1e-9)
            for k, v in enumerate(e):
                resid_by_pos.setdefault(k + 1, []).append(float(v))
        prob_out['_seg_resid_by_pos'] = {str(k): np.array(v) for k, v in resid_by_pos.items()}
    return {
        'acc': acc,
        'acc3': acc3,
        'acc5': acc5,
        'macro_f1': macro_f1,
        'per_class_recall': per_class_recall,
        'per_pos': {str(p): c / t for p, (c, t) in sorted(per_pos.items())},
        'per_pos3': {str(p): c / t for p, (c, t) in sorted(per_pos3.items())},
        'dur_mae_s': mae, 'dur_mae_min': mae / 60.0, 'mape': mape,
        'seg_errs': np.array(dur_errs, dtype=np.float64),
        'seg_trues': np.array(dur_trues, dtype=np.float64),
        'cum_mae_s': cum_mae, 'cum_mae_min': cum_mae / 60.0,
        'cum_window_acc': windows,
        'cum_errs': cum_errs,
        'cum_trues': np.array(cum_trues, dtype=np.float64),
        **prob_out,
    }


@torch.no_grad()
def rollout_predicted_regions(model, records, region_meta, norm, device, prefix_len=2, max_len=8):
    """自回归 rollout，返回每记录预测的下一区域序列 {rec_idx: [pred_b0, pred_b1, ...]}。
    与 rollout_eval 同协议（预测区域回灌继续）。用于在线端到端时长对比。"""
    model.eval()
    out = {}
    for ri, r in enumerate(records):
        seq = r['seq']
        Lr = len(seq)
        if Lr < prefix_len + 1:
            continue
        obs = seq[:prefix_len]
        preds = []
        for step_i in range(prefix_len, Lr):
            pred_next, _ = _predict_step(model, r, obs, region_meta, norm, device, max_len)
            preds.append(pred_next)
            if pred_next == -1:
                break
            obs = obs + [pred_next]
        # 对齐：preds[k] 在 step_i=prefix_len+k 预测 r_{step_i}，即段 (step_i-1) 的 b。
        aligned = {}
        for k, pn in enumerate(preds):
            seg_idx = prefix_len + k - 1
            aligned[seg_idx] = pn
        out[ri] = aligned
    return out


@torch.no_grad()
def rollout_eval(model, records, region_meta, norm, device, prefix_len=2, max_len=8, seg_p95=None, reach=None):
    """从长度为 prefix_len 的真实前缀自回归 rollout：下一区域 acc + 累计到达 MAE。"""
    model.eval()
    if seg_p95 is None:
        seg_p95 = norm['seg_p95']
    stats_out = {'acc': [0, 0], 'per_pos': {}, 'arr_errs': [], 'arr_trues': []}
    for r in records:
        seq = r['seq']
        Lr = len(seq)
        if Lr < prefix_len + 1:
            continue
        obs = seq[:prefix_len]
        cur_arr = r['arrival_offsets'][prefix_len - 1]
        for step_i in range(prefix_len, Lr):
            pred_next, pred_dur = _predict_step(model, r, obs, region_meta, norm, device, max_len, reach)
            stats_out['acc'][1] += 1
            stats_out['acc'][0] += int(pred_next == seq[step_i])
            stats_out['per_pos'].setdefault(step_i, [0, 0])
            stats_out['per_pos'][step_i][0] += int(pred_next == seq[step_i])
            stats_out['per_pos'][step_i][1] += 1
            cur_arr += pred_dur
            true_arr = r['arrival_offsets'][step_i]
            stats_out['arr_errs'].append(abs(cur_arr - true_arr))
            stats_out['arr_trues'].append(true_arr)
            if pred_next == -1:
                break
            obs = obs + [pred_next]
    return stats_out


@torch.no_grad()
def _predict_step(model, record, obs_seq, region_meta, norm, device, max_len, reach=None):
    from ar_model import POI_DIM
    L = max_len
    obs = obs_seq[:L]
    Lr = len(obs)
    ids = torch.full((L,), -1, dtype=torch.long)
    role = torch.zeros(L, dtype=torch.long)
    geo = torch.zeros(L, 3 + POI_DIM)
    gps = torch.zeros(L, GPS_SEG_DIM)
    tm = torch.zeros(L, 2)
    hour = record['start_time_of_day']
    tm[:, 0] = np.sin(2 * np.pi * hour / 24.0)
    tm[:, 1] = np.cos(2 * np.pi * hour / 24.0)
    poi_names = ['餐饮', '休闲', '住宿', '风景名胜', '科教文化', '公共设施']
    for i, rid in enumerate(obs):
        m = region_meta.get(rid, {})
        ids[i] = rid
        role[i] = m.get('role_id', 0)
        geo[i, 0] = (m.get('lat', 0) - norm['lat_mean']) / norm['lat_std']
        geo[i, 1] = (m.get('lon', 0) - norm['lon_mean']) / norm['lon_std']
        geo[i, 2] = (m.get('elev_mean', 0) - norm['elev_mean']) / norm['elev_std']
        pv = np.zeros(POI_DIM)
        for k, v in m.get('poi_profile', {}).items():
            if k in poi_names:
                pv[poi_names.index(k)] = v
        geo[i, 3:] = torch.from_numpy((pv - norm['poi_mean']) / norm['poi_std'])
    gf = record['gps_seg_features']
    for i in range(min(Lr - 1, L)):
        arr = np.array(gf[i], dtype=np.float32)
        gps[i] = torch.from_numpy((arr - norm['gps_mean']) / norm['gps_std'])
    # 累计行程上下文：段特征已改为仅时刻，置零
    cum_dist = torch.zeros(L)
    pos = torch.arange(L, dtype=torch.long)
    ids = ids.unsqueeze(0).to(device); geo = geo.unsqueeze(0).to(device)
    gps = gps.unsqueeze(0).to(device); tm = tm.unsqueeze(0).to(device)
    role = role.unsqueeze(0).to(device); pos = pos.unsqueeze(0).to(device)
    cd = cum_dist.unsqueeze(0).to(device)
    route_id = torch.tensor([record['route_id']], device=device)
    rlogits, dur_pred, _, _, _ = model(ids, geo, gps, tm, role, pos, route_id, cd)
    if reach is not None:
        cur = int(obs_seq[-1])
        mask = torch.ones_like(rlogits[0, Lr - 1]) * -1e9
        mask[reach[cur]] = 0.0
        rlogits[0, Lr - 1] = rlogits[0, Lr - 1] + mask
    pred_next = int(rlogits[0, Lr - 1].argmax().item())
    pred_dur = float(dur_pred[0, Lr - 1].item()) * norm['seg_p95']
    return pred_next, pred_dur


def dm_test(err1, err2):
    """Diebold–Mariano 检验（1-step，loss differential）。err1/err2 等长成对误差序列。"""
    e1 = np.asarray(err1, dtype=np.float64)
    e2 = np.asarray(err2, dtype=np.float64)
    n = min(len(e1), len(e2))
    if n < 3:
        return None, None
    d = e1 - e2
    mean = d.mean()
    sd = d.std(ddof=1)
    if sd < 1e-12:
        return None, None
    dm = mean / (sd / np.sqrt(n))
    from scipy import stats as st
    p = 2 * (1 - st.norm.cdf(abs(dm)))
    return float(dm), float(p)


def paired_wilcoxon(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if len(a) < 2:
        return None, None
    try:
        stat, p = stats.wilcoxon(a - b)
        return float(stat), float(p)
    except ValueError:
        return None, None
