# -*- coding: utf-8 -*-
"""
STRAT 区域级（cross_scene_mountains）的数据 bundle 预热：
20 景区 × 5 seeds = 100 个 csb_{name}_{seed}.pkl（聚类/路线发现，CPU 任务）。
与 mountains.cache_bundle 同路径同格式；GPU 空出后 mountains.py 直接命中缓存训练。

用法：python prediction/warm_csb_20.py [--workers 4] [--seeds 42 100 2024 7 17]
"""
import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# fork worker 前限制 BLAS/OpenMP 线程（否则每 worker 唤醒 32 线程打爆 load）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'prediction'))
sys.path.insert(0, os.path.join(ROOT, 'cluster'))
sys.path.insert(0, os.path.join(ROOT, 'poi'))

import cross_scene_mountains as csm
from run_transformer_experiment import build_bundle


def build_one(job):
    name, seed = job
    import torch
    torch.set_num_threads(4)
    cp = os.path.join(csm.CACHE_DIR, f'csb_{name}_{seed}_{csm._cfg_key(csm.Args())}.pkl')
    if os.path.exists(cp):
        return name, seed, 'cached', 0.0
    t0 = time.time()
    b = build_bundle(name, csm.SEG_CSV[name], seed, csm.Args())
    with open(cp, 'wb') as f:
        pickle.dump(b, f)
    return name, seed, 'built', round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seeds', type=int, nargs='*', default=csm.SEEDS)
    ap.add_argument('--scenes', nargs='*', default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, 'data-project'))
    from scenes_20 import NAMES_20
    csm._setup_scenes(args.scenes or NAMES_20)

    jobs = [(m, s) for s in args.seeds for m in csm.MOUNTAINS]
    print(f"预热 {len(jobs)} 个 bundle, workers={args.workers}", flush=True)
    n_ok = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(build_one, j): j for j in jobs}
        for fut in as_completed(futs):
            name, seed, st, secs = fut.result()
            if st != 'error':
                n_ok += 1
            print(f"[{name}|s{seed}] {st} {secs}s", flush=True)
    print(f"完成 {n_ok}/{len(jobs)}")


if __name__ == '__main__':
    main()
