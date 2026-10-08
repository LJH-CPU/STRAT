"""
聚类网格选参工具（从 ablation 抽出，供预测数据流使用）。

- _validation_subset: 留出验证子集（δ 选参，避免选择偏差）
- _best_grid_balls: 网格搜索 δ（验证子集 silhouette 选参）+ eigen-gap 自动 k
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cluster')))
from config import SELECTION_SPLIT
from scenery_route_clustering import _eval_delta, perform_clustering_eigengap


def _validation_subset(features, random_state):
    n = len(features)
    n_sel = max(int(n * SELECTION_SPLIT), min(50, n))
    rng = np.random.RandomState(random_state)
    return np.sort(rng.choice(n, n_sel, replace=False))


def _best_grid_balls(ball_centers, ball_radii, features, k_lower, k_upper, min_cb, sel_idx, n_workers=None):
    """网格搜索 δ（验证子集选参），返回 (best_labels, best_delta, best_k, best_sel_sil)。"""
    if n_workers is None:
        n_workers = 1
    best_sil, best_delta, best_labels, best_k = -1.0, 0.5, None, k_lower
    tasks = [(ball_centers, ball_radii, features, d, k_lower, k_upper, min_cb, sel_idx)
             for d in np.arange(0.1, 1.0 + 1e-9, 0.1)]
    if len(tasks) <= 4 or n_workers <= 1:
        for t in tasks:
            sil, k, d = _eval_delta(t)
            if sil > best_sil:
                best_sil, best_k, best_delta = sil, k, float(d)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_eval_delta, t): t for t in tasks}
            for fu in as_completed(futs):
                try:
                    sil, k, d = fu.result()
                except Exception:
                    continue
                if sil > best_sil:
                    best_sil, best_k, best_delta = sil, k, float(d)
    labels, k_final, _ = perform_clustering_eigengap(
        ball_centers, ball_radii, features, best_delta, k_lower, k_upper, min_cluster_balls=min_cb)
    return labels, best_delta, k_final, best_sil
