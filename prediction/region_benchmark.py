#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
区域级统一 benchmark（Phase 8 #7/#8/#12）。

- #7 分类完整性：top-1/3/5、macro-F1、per-class recall、majority、1/K（20 target × 5 seed）
- #8 原型词表 K 敏感性：K/2, K, 2K（5 representative target × 3 seed）
- #12 模块消融：Transformer / −R-GCN / −route loss / −region(identity) / Full + 序列长度 L∈{1,2,4,8}
      （5 representative target × 5 seed）

严格无泄漏：聚类/路线/原型/归一化/先验仅训练集（源池）内；target 只映射与推断。

用法：
  python prediction/region_benchmark.py --mode main --scenes <20> --seeds 42 100 2024 7 17 \
      --out data-project/bench_region_main.json
  python prediction/region_benchmark.py --mode kvocab --targets 峨眉山 长白山 华山 黄山 泰山 \
      --seeds 42 100 2024 --out data-project/bench_region_kvocab.json
  python prediction/region_benchmark.py --mode module --targets <5> --seeds 42 100 2024 7 17 \
      --out data-project/bench_region_module.json
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data-project')))
from config import CLEANED_DIR  # noqa: E402
from ar_model import build_normalizers, create_dataloader, POI_DIM  # noqa: E402
from transformer_model import TrajectoryPredictor, build_pair_mean  # noqa: E402
from run_transformer_experiment import train_model, _calibrate_path_sigma  # noqa: E402
from transformer_eval import teacher_forced_eval  # noqa: E402
import cross_scene_mountains as csm  # noqa: E402
from scenes_20 import NAMES_20  # noqa: E402

DEV = None


def _split(pooled, seed, ratio=0.15):
    tracks = sorted({int(r['track_id']) for r in pooled})
    rng = np.random.RandomState(seed); rng.shuffle(tracks)
    nva = max(int(len(tracks) * ratio), 1)
    va = set(tracks[:nva])
    tr = [r for r in pooled if int(r['track_id']) not in va]
    vr = [r for r in pooled if int(r['track_id']) in va]
    return tr, vr


def _majority_acc(pooled, te_records):
    cnt = Counter()
    for r in pooled:
        seq = r['seq']
        for j in range(len(r['segment_durations'])):
            cnt[int(seq[j + 1])] += 1
    maj = cnt.most_common(1)[0][0] if cnt else 0
    n = c = 0
    for r in te_records:
        seq = r['seq']
        for j in range(len(r['segment_durations'])):
            n += 1; c += int(int(seq[j + 1]) == maj)
    return c / max(n, 1)


def eval_config(target, seed, args, k_proto=None, use_struct=True, use_prior=True,
                route_loss_w=0.5, use_region_id=True, max_len=8,
                randomize_region_id=False, rand_seed=0):
    """零样本：源池训练 → 目标 test 评测。返回 #7 指标 + 时长/校准。"""
    bundles = {m: csm.cache_bundle(m, seed, csm.Args()) for m in set(csm.MOUNTAINS) | {target}}
    tb = bundles[target]
    srcs = [bundles[m] for m in csm.MOUNTAINS if m != target]
    shared_meta, map_fn = csm.pooled_shared_space(srcs, k_proto=k_proto)
    pooled = []
    for b in srcs:
        pooled.extend(csm.remap_records(b['tr_records'], map_fn(b['region_meta'])))
    norm = build_normalizers(pooled, shared_meta)
    te_records = csm.remap_records(tb['te_records'], map_fn(tb['region_meta']))
    tr, vr = _split(pooled, seed)
    tr_dl, _ = create_dataloader(tr, shared_meta, norm, batch_size=128, shuffle=True, max_len=max_len, num_workers=8)
    va_dl, _ = create_dataloader(vr, shared_meta, norm, batch_size=128, shuffle=False, max_len=max_len, num_workers=8)
    te_dl, _ = create_dataloader(te_records, shared_meta, norm, batch_size=128, shuffle=False, max_len=max_len, num_workers=8)
    V = len(shared_meta)
    pair_mean, gmean = build_pair_mean(pooled, V)
    model = TrajectoryPredictor(
        num_regions=V, num_routes=2, geo_dim=3 + POI_DIM, pair_mean=pair_mean, global_mean=gmean,
        seg_p95=norm['seg_p95'], backbone='transformer', d_model=64,
        use_struct=use_struct, use_prior=use_prior, time_feats='full', prob=True,
        aggregator='rgcn', edge_mask=(True, True, True, True), causal_edges=True,
        use_region_id=use_region_id,
        randomize_region_id=False, rand_seed=rand_seed).to(DEV)
    model = train_model(model, tr_dl, va_dl, DEV, epochs=args.epochs, seed=seed,
                        loss='nll', route_loss_w=route_loss_w, patience=8)
    ps = _calibrate_path_sigma(model, va_dl, DEV, norm['seg_p95'])
    if randomize_region_id:
        # Target-only identity negative control: the model is trained on source scenes with
        # the true prototype ids; at target evaluation the learned region-identity embedding
        # is looked up through a fixed random permutation, breaking the source->target
        # identity correspondence while leaving all physical/temporal inputs unchanged.
        g = torch.Generator().manual_seed(int(rand_seed))
        model.projector.region_perm = torch.randperm(V, generator=g).to(DEV)
        model.projector.randomize_region_id = True
    tf = teacher_forced_eval(model, te_dl, DEV, norm['seg_p95'], path_sigma_cal=ps)
    return {
        'target': target, 'seed': seed, 'V': V, 'k_proto': k_proto,
        'acc': round(tf['acc'], 4), 'acc3': round(tf['acc3'], 4), 'acc5': round(tf['acc5'], 4),
        'macro_f1': round(tf['macro_f1'], 4) if tf.get('macro_f1') is not None else None,
        'dur_mae_min': round(tf['dur_mae_min'], 2),
        'path_picp90': round(tf.get('path_picp90', 0), 4),
        'path_crps_min': round(tf.get('path_crps_min', 0), 2),
        'majority_acc': round(_majority_acc(pooled, te_records), 4),
        'random_acc': round(1.0 / max(V, 1), 4),
        'n_regions_target': tb['regions'],
    }


def run_main(args):
    rows = []
    for seed in args.seeds:
        for target in (args.targets or args.scenes):
            t0 = time.time()
            r = eval_config(target, seed, args)
            r['time_s'] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"[main|{target}|s{seed}] acc={r['acc']} acc3={r['acc3']} acc5={r['acc5']} "
                  f"f1={r['macro_f1']} maj={r['majority_acc']} rnd={r['random_acc']} "
                  f"dur={r['dur_mae_min']} | {r['time_s']}s", flush=True)
    return rows


def run_kvocab(args):
    rows = []
    for seed in args.seeds:
        for target in (args.targets or args.scenes[:5]):
            # 基准 K = 源池区域质心总数（pooled_shared_space 默认 cap 64）
            bundles = {m: csm.cache_bundle(m, seed, csm.Args()) for m in set(csm.MOUNTAINS) | {target}}
            srcs = [bundles[m] for m in csm.MOUNTAINS if m != target]
            n_proto = sum(len(b['region_meta']) for b in srcs)
            K = min(64, n_proto)
            for tag, kk in [('K/2', max(K // 2, 2)), ('K', K), ('2K', min(2 * K, n_proto))]:
                t0 = time.time()
                r = eval_config(target, seed, args, k_proto=kk)
                r['k_tag'] = tag
                r['time_s'] = round(time.time() - t0, 1)
                rows.append(r)
                print(f"[kvocab|{target}|s{seed}|{tag}={kk}] acc={r['acc']} dur={r['dur_mae_min']} "
                      f"| {r['time_s']}s", flush=True)
    return rows


MODULE_CONFIGS = {
    'transformer':      dict(use_struct=False, use_prior=False, route_loss_w=0.0),
    'minus_rgcn':       dict(use_struct=False, use_prior=True,  route_loss_w=0.5),
    'minus_route':      dict(use_struct=True,  use_prior=True,  route_loss_w=0.0),
    'minus_region':     dict(use_struct=True,  use_prior=True,  route_loss_w=0.5, use_region_id=False),
    'full':             dict(use_struct=True,  use_prior=True,  route_loss_w=0.5),
}


def run_module(args):
    rows = []
    for seed in args.seeds:
        for target in (args.targets or args.scenes[:5]):
            for name, cfg in MODULE_CONFIGS.items():
                t0 = time.time()
                r = eval_config(target, seed, args, **cfg)
                r['config'] = name
                r['time_s'] = round(time.time() - t0, 1)
                rows.append(r)
                print(f"[module|{target}|s{seed}|{name}] acc={r['acc']} f1={r['macro_f1']} "
                      f"dur={r['dur_mae_min']} | {r['time_s']}s", flush=True)
            # 序列长度（full 配置）；L=1 无 next-region 目标，排除（P1）
            for L in [2, 4, 8]:
                t0 = time.time()
                r = eval_config(target, seed, args, max_len=L, **MODULE_CONFIGS['full'])
                r['config'] = f'seqlen{L}'
                r['time_s'] = round(time.time() - t0, 1)
                rows.append(r)
                print(f"[module|{target}|s{seed}|seqlen{L}] acc={r['acc']} dur={r['dur_mae_min']} "
                      f"| {r['time_s']}s", flush=True)
    return rows


def run_same(args):
    """同场景 STRAT（目标自训/自测）：collapse 的 in-domain 上界。"""
    rows = []
    for seed in args.seeds:
        for target in (args.targets or args.scenes):
            t0 = time.time()
            b = csm.cache_bundle(target, seed, csm.Args())
            res = csm.run_bundle(target, b, seed, DEV, csm.Args())
            m = res.get('model', {})
            rows.append({
                'target': target, 'seed': seed, 'V': b.get('regions'),
                'acc': m.get('tf_acc'), 'acc3': m.get('tf_acc3'), 'acc5': m.get('tf_acc5'),
                'macro_f1': m.get('tf_macro_f1'), 'dur_mae_min': m.get('tf_dur_mae_min'),
                'path_picp90': m.get('path_picp90'), 'path_crps_min': m.get('path_crps_min'),
                'time_s': round(time.time() - t0, 1),
            })
            print(f"[same|{target}|s{seed}] acc={rows[-1]['acc']} acc3={rows[-1]['acc3']} "
                  f"f1={rows[-1]['macro_f1']} dur={rows[-1]['dur_mae_min']} | {rows[-1]['time_s']}s", flush=True)
    return rows


RAND_SEEDS = [0, 1, 2]  # 预注册随机置换种子（P3 负控）


def run_randid(args):
    """P3 负控：target-only 固定随机置换 region 身份嵌入（只动目标侧身份对应，不动物理/geo/prior）。"""
    rows = []
    for seed in args.seeds:
        for target in (args.targets or args.scenes[:5]):
            for rs in RAND_SEEDS:
                t0 = time.time()
                r = eval_config(target, seed, args, randomize_region_id=True, rand_seed=rs)
                r['config'] = 'randomized_identity'; r['rand_seed'] = rs
                r['time_s'] = round(time.time() - t0, 1)
                rows.append(r)
                print(f"[randid|{target}|s{seed}|rs{rs}] acc={r['acc']} f1={r['macro_f1']} "
                      f"dur={r['dur_mae_min']} | {r['time_s']}s", flush=True)
    return rows


def main():
    global DEV
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='main', choices=['main', 'kvocab', 'module', 'same', 'randid'])
    p.add_argument('--scenes', nargs='*', default=NAMES_20)
    p.add_argument('--targets', nargs='*', default=None)
    p.add_argument('--seeds', type=int, nargs='*', default=[42, 100, 2024, 7, 17])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    csm._setup_scenes(args.scenes)
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', DEV, 'scenes:', len(csm.MOUNTAINS), 'mode:', args.mode, flush=True)
    t0 = time.time()
    if args.mode == 'main':
        rows = run_main(args)
    elif args.mode == 'kvocab':
        rows = run_kvocab(args)
    elif args.mode == 'same':
        rows = run_same(args)
    elif args.mode == 'randid':
        rows = run_randid(args)
    else:
        rows = run_module(args)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'output', f'bench_region_{args.mode}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'mode': args.mode, 'rows': rows}, f, ensure_ascii=False, indent=1)
    print(f'saved {out}  total {time.time()-t0:.0f}s  rows={len(rows)}', flush=True)


if __name__ == '__main__':
    main()
