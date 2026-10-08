"""
STRAT 总入口：聚类→数据(缓存)→模型训练评测→基线→迁移→报告。

用法：
  python run.py --all                        # 全流程 8 景区 × 3 种子
  python run.py --scene 峨眉山 青城山        # 指定景区
  python run.py --paper                      # 论文三景区
  python run.py --stage model --scene 青城山 # 只重训模型（用缓存数据）
  python run.py --scene 青城山 --debug       # 调试：1 景区 1 种子 10 epoch + 逐轮 loss
  python run.py --scene 黄龙溪 --quick       # 快速冒烟
  python run.py --report-only                # 只从现有结果 JSON 重出报告
"""

import os
import sys
import json
import glob
import time
import argparse
import subprocess

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'prediction')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'cluster')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'poi')))

from config import CLEANED_DIR, PAPER_SCENERIES, SEED
import ar_pipeline as ap
import run_transformer_experiment as rte

OUT_DIR = os.path.join(os.path.dirname(__file__), 'prediction', 'output')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')
CACHE_VERSION = 'v2'  # 特征 schema/协议版本；改动特征维度或口径时递增，避免复用旧缓存
DEFAULT_BACKBONE = 'transformer'


def resolve_scenes(scenes, all_, paper, min_tracks=30):
    if scenes:
        return [(n, str(CLEANED_DIR / f'{n}_cleaned.csv')) for n in scenes]
    files = sorted(glob.glob(str(CLEANED_DIR / '*_cleaned.csv')))
    out = []
    for f in files:
        name = os.path.basename(f).replace('_cleaned.csv', '')
        if paper and name not in PAPER_SCENERIES:
            continue
        n = pd.read_csv(f, usecols=['trackId'])['trackId'].nunique()
        if n >= min_tracks:
            out.append((name, f))
    return out


def make_args(args, seed):
    """构造实验参数对象（含 debug/quick 覆盖）。"""
    a = argparse.Namespace(**vars(args))
    a.seeds = [seed]
    a.seed = seed
    if a.debug:
        a.epochs = min(a.epochs, 10)
        a.verbose = True
    if a.quick:
        a.epochs = min(a.epochs, 5)
    return a


def cache_path(name, seed, k_override, n_poi_regions=None):
    k = f'k{k_override}' if k_override else 'auto'
    npr = f'npr{n_poi_regions}' if n_poi_regions else 'nprAuto'
    return os.path.join(CACHE_DIR, f'{name}_{seed}_{k}_{npr}_{CACHE_VERSION}.pkl')


def build_data(name, csv_path, args):
    """构建并缓存数据包。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = cache_path(name, args.seed, args.k_override, getattr(args, 'n_poi_regions', None))
    if os.path.exists(cp) and not args.no_cache:
        import pickle
        with open(cp, 'rb') as f:
            return pickle.load(f)
    t0 = time.time()
    print(f'[data] 构建 {name} (seed={args.seed})...')
    bundle = rte.build_bundle(name, csv_path, args.seed, args)
    if not args.no_cache:
        import pickle
        with open(cp, 'wb') as f:
            pickle.dump(bundle, f)
        print(f'[data] 缓存: {cp} ({time.time()-t0:.1f}s)')
    return bundle


def run_model_stage(scenes, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[model] device={device}, backbone={args.backbone}')
    results = {}
    for name, csv_path in scenes:
        for seed in args.seeds:
            key = f'{name}|{args.backbone}|seed{seed}|s{args.use_struct}|p{args.use_prior}|a{args.aggregator}|e{"".join(map(str,args.edge_mask))}|i{args.interval}|c{args.causal_edges}|nc{int(args.no_cal)}'
            try:
                bundle = build_data(name, csv_path, make_args(args, seed))
                results[key] = rte.run_bundle(name, bundle, seed, device, make_args(args, seed))
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key] = {'status': 'error', 'error': str(e)}
    out = os.path.join(args.out_dir, args.results_file)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f'[model] 结果已保存: {out}')


def run_data_stage(scenes, args):
    for name, csv_path in scenes:
        for seed in args.seeds:
            build_data(name, csv_path, make_args(args, seed))


def run_baseline_stage(scenes, args):
    """XGBoost 时长基线（子进程，重建数据）。"""
    names = ' '.join(n for n, _ in scenes)
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'prediction', 'xgboost_baseline.py'),
           '--scenery'] + [n for n, _ in scenes] + ['--n_workers', str(args.n_workers)]
    print('[baseline] 运行 XGBoost 时长基线...')
    subprocess.run(cmd, check=False)
    print('[baseline] 完成')


def run_transfer_stage(args):
    print('[transfer] 运行跨场景到达时间迁移...')
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'prediction', 'transformer_transfer.py'),
           '--n_workers', str(args.n_workers)]
    subprocess.run(cmd, check=False)
    print('[transfer] 完成')


def run_report_stage():
    print('[report] 生成报告...')
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), 'prediction', 'transformer_report.py')]
    subprocess.run(cmd, check=False)
    print('[report] 完成')


def main():
    p = argparse.ArgumentParser(description='STRAT 总入口')
    p.add_argument('--scenes', nargs='*', dest='scenes', default=None, help='景区名')
    p.add_argument('--scene', nargs='*', dest='scenes', default=None)
    p.add_argument('--all', action='store_true', help='全部 min_tracks>=30 的景区')
    p.add_argument('--paper', action='store_true', help='论文三景区(青城山/峨眉山/武侯祠)')
    p.add_argument('--stage', nargs='*', default=None,
                   help='阶段: data,model,baseline,transfer,report（默认 data+model+baseline+transfer+report）')
    p.add_argument('--backbone', default=DEFAULT_BACKBONE, choices=['transformer', 'gpt2', 'lstm'])
    p.add_argument('--gpt2_ckpt', default='/tmp/opencode/ms_cache/models/AI-ModelScope--gpt2/snapshots/master')
    p.add_argument('--seeds', type=int, nargs='*', default=[SEED, 100, 2024])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--use_struct', type=int, default=1)
    p.add_argument('--use_prior', type=int, default=1)
    p.add_argument('--aggregator', default='rgcn', choices=['rgcn', 'gat', 'mlp'])
    p.add_argument('--edge_mask', type=int, nargs=4, default=[1, 1, 1, 1])
    p.add_argument('--interval', type=int, default=0)
    p.add_argument('--interval_bins', type=int, default=8)
    p.add_argument('--causal_edges', type=int, default=1)
    p.add_argument('--loss', default='mae', choices=['rel1', 'mae', 'logmae', 'nll'])
    p.add_argument('--prob', type=int, default=0)
    p.add_argument('--no_cal', action='store_true')
    p.add_argument('--time_feats', default='full', choices=['basic', 'full'])
    p.add_argument('--k_override', type=int, default=None)
    p.add_argument('--n_poi_regions', type=int, default=None)
    p.add_argument('--max_len', type=int, default=8)
    p.add_argument('--n_workers', type=int, default=8)
    p.add_argument('--no_cache', action='store_true', help='强制重建数据缓存')
    p.add_argument('--debug', action='store_true', help='调试:1景区1种子10epoch+逐轮loss')
    p.add_argument('--quick', action='store_true', help='冒烟:小采样短训练')
    p.add_argument('--verbose', action='store_true', help='打印逐轮训练loss')
    p.add_argument('--log', type=str, default=None, help='日志输出文件')
    p.add_argument('--out_dir', default=OUT_DIR, help='结果目录')
    p.add_argument('--results_file', default='llm_main_transformer.json', help='结果 JSON 文件名')
    p.add_argument('--report-only', action='store_true', help='只重出报告')
    p.add_argument('--dry-run', action='store_true', help='只打印将运行的阶段/场景，不执行')
    args = p.parse_args()

    # 日志 tee
    if args.log:
        import sys as _sys
        _logf = open(args.log, 'w', encoding='utf-8')
        class Tee:
            def write(self, s): _sys.stdout.write(s); _logf.write(s); return len(s)
            def flush(self): _sys.stdout.flush(); _logf.flush()
        _sys.stdout = Tee()

    if args.report_only:
        run_report_stage()
        return

    # debug/quick 覆盖
    if args.debug:
        args.scenes = args.scenes or ['青城山']
        args.seeds = args.seeds[:1]
    if args.quick:
        args.scenes = args.scenes or ['黄龙溪']
        args.seeds = args.seeds[:1]
        args.epochs = 5

    scenes = resolve_scenes(args.scenes, args.all, args.paper)
    if not scenes:
        print('没有符合条件的景区')
        return

    stages = args.stage or ['data', 'model', 'baseline', 'transfer', 'report']
    if args.dry_run:
        print(f'将运行场景: {[n for n,_ in scenes]}')
        print(f'阶段: {stages}  seeds: {args.seeds}  backbone: {args.backbone}  epochs: {args.epochs}')
        return

    for stage in stages:
        if stage == 'data':
            run_data_stage(scenes, args)
        elif stage == 'model':
            run_model_stage(scenes, args)
        elif stage == 'baseline':
            run_baseline_stage(scenes, args)
        elif stage == 'transfer':
            run_transfer_stage(args)
        elif stage == 'report':
            run_report_stage()


if __name__ == '__main__':
    main()
