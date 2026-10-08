# -*- coding: utf-8 -*-
"""
单一特征组配置源（H1，唯一真源）。

所有 benchmark、JSON key、图表脚本、论文表格必须从此处 import；
禁止在其它模块手写 feature list。

去泄漏后（P0-1）段级特征锁定为 8 个：
  dist_m, up_m, down_m, mean_grade, max_grade, elev_mean, start_ele, tod_hour
（move_time_s / n_pts 已删除；地形量来自 DEM，非本次轨迹海拔）

用法：
  from feature_groups import FEATURES, FEATURE_GROUPS, CORE_GROUPS, SUPPLEMENT_GROUPS, subset
"""
from __future__ import annotations

import numpy as np

# ── 段级特征全集（顺序即模型输入顺序）──────────────────────
FEATURES = [
    "dist_m", "up_m", "down_m", "mean_grade", "max_grade",
    "elev_mean", "start_ele", "tod_hour",
]

# 明确禁止进入任何特征集（标签/时间泄漏）——护栏
FORBIDDEN = {"move_time_s", "n_pts", "dur_s", "naismith_s"}

# ── 原子物理信息组 ─────────────────────────────────────────
UP = ["up_m"]
DOWN = ["down_m"]
GRADE = ["mean_grade", "max_grade"]
ELEV = ["elev_mean", "start_ele"]
DIST = ["dist_m"]
TOD = ["tod_hour"]
SLOPE = UP + DOWN + GRADE          # 坡度相关（含海拔变化）
TERRAIN = SLOPE + ELEV             # 全部地形


def _minus(*drop_groups):
    drop = {f for g in drop_groups for f in g}
    return [f for f in FEATURES if f not in drop]


# ── Phase 8 #3 固定 12 组（核心 8 + 补充 4）─────────────────
FEATURE_GROUPS = {
    # 核心 8 组（20 target × 5 seed，正式统计）
    "full": list(FEATURES),
    "no_slope": _minus(SLOPE),
    "no_elev": _minus(ELEV),
    "no_updown": _minus(UP + DOWN),
    "slope_only": list(SLOPE),
    "elev_only": list(ELEV),
    "terrain_only": list(TERRAIN),
    "temporal_only": list(TOD),
    # 补充 4 组（20 target × 1 seed，机制/敏感性，不进主 p 值）
    "no_grade": _minus(GRADE),
    "no_dist": _minus(DIST),
    "dist_terrain": list(DIST) + list(TERRAIN),
    "updown_only": list(UP + DOWN),
}

CORE_GROUPS = [
    "full", "no_slope", "no_elev", "no_updown",
    "slope_only", "elev_only", "terrain_only", "temporal_only",
]
SUPPLEMENT_GROUPS = ["no_grade", "no_dist", "dist_terrain", "updown_only"]

# 各组 seed 预算（Phase 8 冻结）
CORE_SEEDS = [42, 100, 2024, 7, 17]
SUPPLEMENT_SEEDS = [42]

# 物理信息层次（论文/图表用）
ATOMIC_GROUPS = {
    "slope_related": GRADE,
    "elevation_related": ELEV,
    "elevation_change": UP + DOWN,
    "temporal": TOD,
}


def subset(name: str) -> list:
    """按组名返回特征子集（拷贝）。"""
    if name not in FEATURE_GROUPS:
        raise KeyError(f"unknown feature group: {name!r}; known={list(FEATURE_GROUPS)}")
    return list(FEATURE_GROUPS[name])


def sample_tracks(df, ratio, subset_seed):
    """严格 track-level 子抽样（H3）：同 track 全部段整体进入/排除。"""
    tracks = np.array(df["track"].unique())
    rng = np.random.RandomState(subset_seed)
    rng.shuffle(tracks)
    n = max(int(len(tracks) * ratio), 1)
    keep = set(tracks[:n].tolist())
    return df[df["track"].isin(keep)]


def _validate():
    for gname, cols in FEATURE_GROUPS.items():
        assert len(cols) == len(set(cols)), f"duplicate feature in group {gname}: {cols}"
        bad = set(cols) & FORBIDDEN
        assert not bad, f"forbidden feature(s) {bad} in group {gname}"
        unknown = set(cols) - set(FEATURES)
        assert not unknown, f"unknown feature(s) {unknown} in group {gname}"
    assert set(CORE_GROUPS) | set(SUPPLEMENT_GROUPS) == set(FEATURE_GROUPS)
    assert not (set(CORE_GROUPS) & set(SUPPLEMENT_GROUPS))


_validate()

if __name__ == "__main__":
    for g in CORE_GROUPS + SUPPLEMENT_GROUPS:
        tag = "core" if g in CORE_GROUPS else "supp"
        print(f"[{tag}] {g:14s} n={len(FEATURE_GROUPS[g])} {FEATURE_GROUPS[g]}")
