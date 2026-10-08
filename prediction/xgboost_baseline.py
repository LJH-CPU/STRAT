"""
XGBoost 段时长回归基线（同特征、严格无泄漏、同 test 口径）。

特征 = 深度时长头同款输入特征：
  段特征8维[路径距离/海拔增益/损失/极差/时刻sin/cos/历史均时长/历史均速度]
  + from_region + to_region + 区域对先验（train-only）
流程：70/30 按 track → train 内聚类/路线/先验 → XGB 在 train 轨迹上训练
       → 在 test 轨迹上评估 MAE/MAPE，与深度模型对比。
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CLEANED_DIR
import ar_pipeline as ap


def build_scene_data(name, seed=42, n_workers=8):
    # 优先复用 run.py 的数据缓存（聚类/路线/records 已构建）
    cp = os.path.join(os.path.dirname(__file__), 'output', 'cache', f'{name}_{seed}_auto.pkl')
    if os.path.exists(cp):
        import pickle
        with open(cp, 'rb') as f:
            b = pickle.load(f)
        if 'tr_records' in b and 'te_records' in b:
            return b['tr_records'], b['te_records'], b['V']
    csv = str(CLEANED_DIR / f'{name}_cleaned.csv')
    df = pd.read_csv(csv)
    dtr, dte = ap.split_tracks(df, 0.7, seed)
    npr = ap.n_poi_regions_static(name)
    model_reg, dtr = ap.cluster_regions_train(dtr, n_poi_regions=npr, seed=seed, n_workers=n_workers)
    dte, _ = ap.map_test_regions(dte, model_reg)
    dtr, t2r, _ = ap.discover_train_routes(dtr, n_workers=n_workers)
    reps = ap._route_representatives(dtr, t2r)
    dte, _ = ap.assign_test_route_pseudolabels(dte, reps)
    profiles = ap.build_poi_profiles(model_reg, name)
    region_meta = ap.build_region_meta_train(dtr, model_reg, profiles)
    tr_records = ap.build_records(dtr, region_meta, name, model_reg)
    te_records = ap.build_records(dte, region_meta, name, model_reg,
                                  track_route_map=dict(zip(dte['trackId'], dte['route_id'])))
    return tr_records, te_records, model_reg['k']


def build_prior_table(records, V):
    """训练集构建区域对历史均时长表（无泄漏：只从传入的训练集 records 算一次）。"""
    sums, cnts, all_d = {}, {}, []
    for r in records:
        seq = r['seq']
        for i, d in enumerate(r['segment_durations']):
            k = (int(seq[i]), int(seq[i + 1]))
            sums[k] = sums.get(k, 0.0) + d
            cnts[k] = cnts.get(k, 0) + 1
            all_d.append(d)
    prior = np.zeros((V, V), dtype=np.float32)
    for (a, b), s in sums.items():
        if a < V and b < V:
            prior[a, b] = s / cnts[(a, b)]
    gm = float(np.mean(all_d)) if all_d else 3600.0
    return prior, gm


def flat_features(records, prior, gm, V, pred_next=None, true_next=False):
    """构造 (X, y)。
    pred_next: {record_index: [预测的下一区域 b0,b1,...]}（测试用预测 b）。
    true_next: True 时用真值 seq[i+1]（仅训练/教师强制）；否则无预测器时 b=当前区域、prior=全局均值
    （无泄漏回退，绝不把真值下一区域喂给测试）。"""
    X, y = [], []
    for ri, r in enumerate(records):
        seq = r['seq']
        gf = r['gps_seg_features']
        pseq = pred_next.get(ri) if pred_next else None
        for i, d in enumerate(r['segment_durations']):
            a = int(seq[i])
            if pseq is not None and i < len(pseq):
                b = int(pseq[i])
            elif true_next:
                b = int(seq[i + 1])
            else:
                b = a
            pr = prior[a, b] if (a < V and b < V and prior[a, b] > 0) else gm
            X.append(list(gf[i]) + [a, b, pr])
            y.append(d)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_xgb_quantile(tr_records, te_records, V, seed=42, max_len=8,
                     quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    """XGBoost 分位数回归（reg:quantileerror）：点估计(中位)、CRPS(离散集成)、PICP、区间宽度、可靠性。"""
    prior, gm = build_prior_table(tr_records, V)
    Xtr, ytr = flat_features(tr_records, prior, gm, V)
    Xte, yte = flat_features(te_records, prior, gm, V)
    if len(Xtr) < 20 or len(Xte) < 2:
        return None
    n_va = max(int(len(Xtr) * 0.15), 10)
    rng = np.random.RandomState(seed)
    vi = rng.choice(len(Xtr), n_va, replace=False)
    mask = np.ones(len(Xtr), dtype=bool); mask[vi] = False
    models = []
    for q in quantiles:
        m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             objective='reg:quantileerror', quantile_alpha=float(q),
                             random_state=seed, n_jobs=-1, early_stopping_rounds=20)
        m.fit(Xtr[mask], ytr[mask], eval_set=[(Xtr[vi], ytr[vi])], verbose=False)
        models.append(m)
    q_arr = np.array(quantiles)
    Q = np.vstack([np.maximum(m.predict(Xte), 0) for m in models])   # (K,N)
    med_idx = int(np.argmin(np.abs(q_arr - 0.5)))
    pred = Q[med_idx]
    errs = np.abs(pred - yte)
    qlo, qhi = Q[0], Q[-1]
    picp90 = float(np.mean((yte >= qlo) & (yte <= qhi)))
    width90 = float(np.mean(qhi - qlo))
    reli = [(float(q), float(np.mean(yte <= Q[k]))) for k, q in enumerate(q_arr)]
    K = len(models); N = len(yte)
    term1 = (1.0 / K) * np.mean(np.abs(Q.T - yte[:, None]).sum(axis=1))
    diff = np.abs(Q[:, None, :] - Q[None, :, :]).sum(axis=(0, 1))
    term2 = (1.0 / (K * K)) * np.mean(diff)
    crps = float(term1 - 0.5 * term2)
    # 路径级：逐记录传播分位（中位=均值，σ=(qhi-qlo)/(2*z0.95)，独立假设同深度模型）
    path_z, path_s = [], []
    for r in te_records:
        seq = r['seq']
        if len(seq) < 2:
            continue
        Xr, yr = flat_features([r], prior, gm, V)
        if len(yr) == 0:
            continue
        Qr = np.vstack([np.maximum(m.predict(Xr), 0) for m in models])  # (K,nseg)
        med = Qr[med_idx]
        sig = ((Qr[-1] - Qr[0]) / (2 * 1.6449)).clip(min=1e-6)
        cp = np.cumsum(med)
        cs = np.sqrt(np.cumsum(sig ** 2) + 1e-9)
        ct = np.cumsum(yr)
        path_z.extend((ct - cp) / cs)
        path_s.extend(cs)
    path_metrics = {}
    if len(path_z) > 3:
        pz = np.array(path_z, dtype=np.float64)
        ps = np.array(path_s, dtype=np.float64)
        from scipy.stats import norm as _norm
        Ph = _norm.cdf(pz); ph = _norm.pdf(pz)
        pcrps = np.mean(ps * (pz * (2 * Ph - 1) + 2 * ph - 1.0 / np.sqrt(np.pi)))
        path_metrics = {
            'path_picp90': float(np.mean(np.abs(pz) <= 1.6449)),
            'path_crps_min': float(pcrps) / 60.0,
            'path_width90_min': float(np.mean(2 * 1.6449 * ps)) / 60.0,
            'path_reliability': [(float(q), float(np.mean(Ph <= q))) for q in np.arange(0.05, 0.96, 0.05)],
        }
    return {
        'mae_min': float(np.mean(errs)) / 60.0,
        'crps': crps,
        'crps_min': crps / 60.0,
        'picp90': picp90,
        'width90_min': width90 / 60.0,
        'reliability': reli,
        'n': int(N),
        **path_metrics,
    }


def _split_records_by_track(records, val_ratio=0.15, seed=42):
    """按 track 划分（与深度模型 val 协议一致）。"""
    rng = np.random.RandomState(seed)
    tracks = sorted({int(r['track_id']) for r in records})
    rng.shuffle(tracks)
    n_va = max(int(len(tracks) * val_ratio), 1)
    va_tracks = set(tracks[:n_va])
    va = [r for r in records if int(r['track_id']) in va_tracks]
    tr = [r for r in records if int(r['track_id']) not in va_tracks]
    return tr, va


def _xgb_quantile_models(tr_records, va_records, V, seed, quantiles):
    prior, gm = build_prior_table(tr_records, V)
    Xtr, ytr = flat_features(tr_records, prior, gm, V)
    n_va_f = max(int(len(Xtr) * 0.15), 10)
    rng = np.random.RandomState(seed)
    vi = rng.choice(len(Xtr), n_va_f, replace=False)
    mask = np.ones(len(Xtr), dtype=bool)
    mask[vi] = False
    models = []
    for q in quantiles:
        m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             objective='reg:quantileerror', quantile_alpha=float(q),
                             random_state=seed, n_jobs=-1, early_stopping_rounds=20)
        m.fit(Xtr[mask], ytr[mask], eval_set=[(Xtr[vi], ytr[vi])], verbose=False)
        models.append(m)
    return models, prior, gm


def _xgb_path_prop(models, records, prior, gm, V, med_idx):
    """逐记录传播分位 → (path z, path σ, 每步计数)。独立假设同 STRAT。"""
    zs, ss = [], []
    for r in records:
        seq = r['seq']
        if len(seq) < 2:
            continue
        Xr, yr = flat_features([r], prior, gm, V)
        if len(yr) == 0:
            continue
        Qr = np.vstack([np.maximum(m.predict(Xr), 0) for m in models])
        med = Qr[med_idx]
        sig = ((Qr[-1] - Qr[0]) / (2 * 1.6449)).clip(min=1e-6)
        cp = np.cumsum(med)
        cs = np.sqrt(np.cumsum(sig ** 2) + 1e-9)
        ct = np.cumsum(yr)
        zs.extend((ct - cp) / cs)
        ss.extend(cs)
    return np.array(zs, dtype=np.float64), np.array(ss, dtype=np.float64)


def run_xgb_quantile_cal(tr_records, te_records, V, seed=42, max_len=8, target=0.9,
                         quantiles=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    """XGBoost 分位 + 路径级公平校准：track 级 val 调路径 σ 因子使路径 PICP≈target。
    与 STRAT 的路径级校准协议一致（验证集仅用于调因子）。"""
    tr, va = _split_records_by_track(tr_records, 0.15, seed)
    if len(tr) < 20:
        return None
    models, prior, gm = _xgb_quantile_models(tr, va, V, seed, quantiles)
    med_idx = int(np.argmin(np.abs(np.array(quantiles) - 0.5)))
    vz, vs = _xgb_path_prop(models, va, prior, gm, V, med_idx)
    if len(vz) < 10:
        return None
    s = float(np.percentile(np.abs(vz), 100.0 * target) / 1.6449)
    tz, ts = _xgb_path_prop(models, te_records, prior, gm, V, med_idx)
    if len(tz) < 3:
        return None
    from scipy.stats import norm as _norm
    def _metrics(z, sscale):
        zc = z / sscale
        Ph = _norm.cdf(zc); ph = _norm.pdf(zc)
        crps = np.mean(ts * sscale * (zc * (2 * Ph - 1) + 2 * ph - 1.0 / np.sqrt(np.pi)))
        return {
            'picp90': float(np.mean(np.abs(zc) <= 1.6449)),
            'crps_min': float(crps) / 60.0,
            'width90_min': float(np.mean(2 * 1.6449 * ts * sscale)) / 60.0,
        }
    raw = _metrics(tz, 1.0)
    cal = _metrics(tz, s)
    return {
        'raw_path_picp90': raw['picp90'],
        'cal_path_picp90': cal['picp90'],
        'raw_path_crps_min': raw['crps_min'],
        'cal_path_crps_min': cal['crps_min'],
        'raw_path_width90_min': raw['width90_min'],
        'cal_path_width90_min': cal['width90_min'],
        'path_sigma_cal': s,
        'n': len(tz),
    }


def run_xgb(tr_records, te_records, V, seed=42, max_len=8):
    prior, gm = build_prior_table(tr_records, V)          # 只从训练集算 prior
    Xtr, ytr = flat_features(tr_records, prior, gm, V)    # 训练特征：训练集 prior
    Xte, yte = flat_features(te_records, prior, gm, V)    # 测试特征：同一张训练集 prior（无泄漏）
    if len(Xtr) < 20 or len(Xte) < 2:
        return None
    n_va = max(int(len(Xtr) * 0.15), 10)
    rng = np.random.RandomState(seed)
    vi = rng.choice(len(Xtr), n_va, replace=False)
    mask = np.ones(len(Xtr), dtype=bool)
    mask[vi] = False
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='reg:absoluteerror', eval_metric='mae',
        random_state=seed, n_jobs=-1, early_stopping_rounds=30,
    )
    model.fit(Xtr[mask], ytr[mask], eval_set=[(Xtr[vi], ytr[vi])], verbose=False)
    pred = np.maximum(model.predict(Xte), 0)
    errs = np.abs(pred - yte)
    # 累计到达指标（按记录分组，与深度模型 teacher-forced 同口径：截断到 max_len）
    cum_errs, cum_trues = [], []
    for r in te_records:
        seq = r['seq']
        if len(seq) < 2:
            continue
        n_seg = min(len(seq), max_len) - 1
        Xr, yr = flat_features([r], prior, gm, V)
        pr = np.maximum(model.predict(Xr[:n_seg]), 0)
        cp = np.cumsum(pr)
        ct = np.cumsum(yr[:n_seg])
        cum_errs.extend(np.abs(cp - ct))
        cum_trues.extend(ct)
    cum_errs = np.array(cum_errs, dtype=np.float64)
    return {
        'mae_min': float(np.mean(errs)) / 60.0,
        'mape': float(np.mean(errs / (yte + 1e-6))),
        'n': int(len(yte)), 'n_train': int(mask.sum()),
        'cum_mae_min': float(np.mean(cum_errs)) / 60.0 if len(cum_errs) else None,
        'cum_window_acc': {str(w): float(np.mean(cum_errs < w * 60)) for w in [15, 30, 60]} if len(cum_errs) else {},
    }


def run_xgb_pred(tr_records, te_records, V, pred_next, seed=42, max_len=8):
    """混合时长线：XGBoost 在训练集真值 b 上训练，测试用 STRAT 预测区域 b（where→when）。"""
    prior, gm = build_prior_table(tr_records, V)          # 只从训练集算 prior
    Xtr, ytr = flat_features(tr_records, prior, gm, V, true_next=True)   # 训练：真值 b（教师强制）
    Xte, yte = flat_features(te_records, prior, gm, V, pred_next)        # 测试：预测 b（无真值泄漏）
    if len(Xtr) < 20 or len(Xte) < 2:
        return None
    n_va = max(int(len(Xtr) * 0.15), 10)
    rng = np.random.RandomState(seed)
    vi = rng.choice(len(Xtr), n_va, replace=False)
    mask = np.ones(len(Xtr), dtype=bool)
    mask[vi] = False
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='reg:absoluteerror', eval_metric='mae',
        random_state=seed, n_jobs=-1, early_stopping_rounds=30,
    )
    model.fit(Xtr[mask], ytr[mask], eval_set=[(Xtr[vi], ytr[vi])], verbose=False)
    pred = np.maximum(model.predict(Xte), 0)
    errs = np.abs(pred - yte)
    cum_errs, cum_trues = [], []
    for ri, r in enumerate(te_records):
        seq = r['seq']
        if len(seq) < 2:
            continue
        n_seg = min(len(seq), max_len) - 1
        Xr, yr = flat_features([r], prior, gm, V, {0: pred_next[ri]} if ri in pred_next else None)
        pr = np.maximum(model.predict(Xr[:n_seg]), 0)
        cp = np.cumsum(pr)
        ct = np.cumsum(yr[:n_seg])
        cum_errs.extend(np.abs(cp - ct))
        cum_trues.extend(ct)
    cum_errs = np.array(cum_errs, dtype=np.float64)
    return {
        'mae_min': float(np.mean(errs)) / 60.0,
        'mape': float(np.mean(errs / (yte + 1e-6))),
        'n': int(len(yte)), 'n_train': int(mask.sum()),
        'cum_mae_min': float(np.mean(cum_errs)) / 60.0 if len(cum_errs) else None,
        'cum_window_acc': {str(w): float(np.mean(cum_errs < w * 60)) for w in [15, 30, 60]} if len(cum_errs) else {},
    }


def train_xgb_duration(tr_records, V, seed=42):
    """训练一个 XGBoost 时长模型（真值区域特征，无泄漏），返回 (model, prior, gm)。"""
    prior, gm = build_prior_table(tr_records, V)
    Xtr, ytr = flat_features(tr_records, prior, gm, V, true_next=True)
    if len(Xtr) < 20:
        return None, None, None
    n_va = max(int(len(Xtr) * 0.15), 10)
    rng = np.random.RandomState(seed)
    vi = rng.choice(len(Xtr), n_va, replace=False)
    mask = np.ones(len(Xtr), dtype=bool)
    mask[vi] = False
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective='reg:absoluteerror', eval_metric='mae',
        random_state=seed, n_jobs=-1, early_stopping_rounds=30,
    )
    model.fit(Xtr[mask], ytr[mask], eval_set=[(Xtr[vi], ytr[vi])], verbose=False)
    return model, prior, gm


def eval_xgb_pred_shared(model, prior, gm, te_records, V, pred_next, max_len=8):
    """用预测 b 评测共享 XGB 时长模型（pred_next: {ri: [b0,b1,...]}，段索引对齐）。"""
    Xte, yte = flat_features(te_records, prior, gm, V, pred_next)
    pred = np.maximum(model.predict(Xte), 0)
    errs = np.abs(pred - yte)
    cum_errs, cum_trues = [], []
    for ri, r in enumerate(te_records):
        seq = r['seq']
        if len(seq) < 2:
            continue
        n_seg = min(len(seq), max_len) - 1
        Xr, yr = flat_features([r], prior, gm, V, {0: pred_next[ri]} if ri in pred_next else None)
        pr = np.maximum(model.predict(Xr[:n_seg]), 0)
        cp = np.cumsum(pr)
        ct = np.cumsum(yr[:n_seg])
        cum_errs.extend(np.abs(cp - ct))
        cum_trues.extend(ct)
    cum_errs = np.array(cum_errs, dtype=np.float64)
    return {
        'mae_min': float(np.mean(errs)) / 60.0,
        'cum_mae_min': float(np.mean(cum_errs)) / 60.0 if len(cum_errs) else None,
        'n': int(len(yte)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenery', nargs='*', default=None)
    p.add_argument('--all', action='store_true')
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 100, 2024])
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--max_len', type=int, default=8)
    p.add_argument('--quantile', action='store_true', help='同时跑 XGBoost 分位数回归（概率对比）')
    p.add_argument('--out', default='prediction/output/llm_xgb.json')
    args = p.parse_args()

    scenes = args.scenery or []
    if not scenes:
        for f in sorted(glob.glob(str(CLEANED_DIR / '*_cleaned.csv'))):
            name = os.path.basename(f).replace('_cleaned.csv', '')
            n = pd.read_csv(f, usecols=['trackId'])['trackId'].nunique()
            if n >= 30:
                scenes.append(name)

    deep = {}
    try:
        md = json.load(open('prediction/output/llm_main_transformer.json', encoding='utf-8'))
        for k, v in md.items():
            if isinstance(v, dict) and v.get('status') == 'ok':
                name = k.split('|')[0]
                seed = int(k.split('|')[2].replace('seed', ''))
                deep.setdefault(name, {})[seed] = {
                    'mae_min': v['model']['tf_dur_mae_min'],
                    'mape': v['model']['tf_mape'],
                }
    except FileNotFoundError:
        print('未找到深度模型结果')

    results = {}
    for name in scenes:
        for seed in args.seeds:
            try:
                tr, te, V = build_scene_data(name, seed, args.n_workers)
                r = run_xgb(tr, te, V, seed, max_len=args.max_len)
                d = deep.get(name, {}).get(seed, {})
                if r is None:
                    print(f'[{name}|{seed}] 样本不足'); continue
                if args.quantile:
                    rq = run_xgb_quantile(tr, te, V, seed, max_len=args.max_len)
                    r['quantile'] = rq
                    print(f"[{name}|{seed}] XGB点MAE={r['mae_min']:.1f}min | XGB分位 CRPS={rq['crps_min']:.1f}min PICP90={rq['picp90']:.3f} | "
                          f"Deep: {d.get('mae_min')}min")
                else:
                    print(f"[{name}|{seed}] XGB: {r['mae_min']:.1f}min MAPE={r['mape']:.3f} | "
                          f"Deep: {d.get('mae_min')}min MAPE={d.get('mape')} | n={r['n']}")
                results[f'{name}|{seed}'] = {**r, 'deep_mae_min': d.get('mae_min'),
                                             'deep_mape': d.get('mape')}
            except Exception as e:
                import traceback; traceback.print_exc()
                results[f'{name}|{seed}'] = {'error': str(e)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'\n结果已保存: {args.out}')


if __name__ == '__main__':
    main()
