#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""区域级 STRAT 结果汇总：读 cross_scene_mountains.json，输出 §4.2 表格 md + 统计。
用法：python prediction/summarize_region.py"""
import json
import os
import sys
import numpy as np

SCENES = ["峨眉山", "华山", "黄山", "泰山"]
SEEDS = [42, 100, 2024, 7, 17]
SEEDS_EN = {"峨眉山": "Emeishan", "华山": "Huashan", "黄山": "Huangshan", "泰山": "Taishan"}

HERE = os.path.dirname(os.path.abspath(__file__))
CSM = os.path.join(HERE, "output", "cross_scene_mountains.json")


def load():
    d = json.load(open(CSM, encoding="utf-8"))
    recs = {k: v for k, v in d.items() if v.get("status") == "ok"}
    out = {s: {seed: None for seed in SEEDS} for s in SCENES}
    for k, v in recs.items():
        s, seed = k.split("|")
        out[s][int(seed)] = v
    return out


def stats(col):
    a = np.array([x for x in col if x is not None], dtype=float)
    if len(a) == 0:
        return None
    return a.mean(), a.std(ddof=1)


def eff(same, zero):
    """迁移效率 = 保留比例 same/zero（时长越小越好；1=完全迁移）。"""
    m_s = np.mean(same)
    m_z = np.mean(zero)
    return m_s / m_z


def fmt(st):
    if st is None:
        return "—"
    return f"{st[0]:.2f} ± {st[1]:.2f}"


def main():
    data = load()
    n_done = sum(1 for s in SCENES for seed in SEEDS if data[s][seed] is not None)
    print(f"# 完成 {n_done}/20\n")

    print("## 4.2.1 同场景 vs 零样本 STRAT（区域级，5 种子 mean±std）\n")
    print("| 目标 | k 区域 | 同场景 acc | 零样本 acc | 同场景 时长MAE | 零样本 时长MAE | 迁移效率 | 同场景 路径PICP | 零样本 路径PICP | 同场景 path-CRPS | 零样本 path-CRPS |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in SCENES:
        recs = [data[s][seed] for seed in SEEDS if data[s][seed]]
        if not recs:
            print(f"| {SEEDS_EN[s]} | — | | | | | | | | | |")
            continue
        k = recs[0].get("n_regions")
        acc_s = [r["same_scene"]["tf_acc"] for r in recs]
        acc_z = [r["zero_shot"]["tf_acc"] for r in recs]
        dur_s = [r["same_scene"]["dur_mae_min"] for r in recs]
        dur_z = [r["zero_shot"]["dur_mae_min"] for r in recs]
        picp_s = [r["same_scene"]["path_picp90"] for r in recs]
        picp_z = [r["zero_shot"]["path_picp90"] for r in recs]
        crps_s = [r["same_scene"]["path_crps_min"] for r in recs]
        crps_z = [r["zero_shot"]["path_crps_min"] for r in recs]
        eff_t = eff(dur_s, dur_z)
        print(f"| {SEEDS_EN[s]} | {k} | {fmt(stats(acc_s))} | {fmt(stats(acc_z))} | "
              f"{fmt(stats(dur_s))} | {fmt(stats(dur_z))} | {eff_t*100:+.0f}% | "
              f"{fmt(stats(picp_s))} | {fmt(stats(picp_z))} | "
              f"{fmt(stats(crps_s))} | {fmt(stats(crps_z))} |")

    print("\n## 4.2.2 区域级物理特征 XGBoost（同层对比）\n")
    print("| 目标 | 同场景 MAE | 零样本 MAE | 迁移效率 | 少样本5% MAE |")
    print("|---|---:|---:|---:|---:|")
    for s in SCENES:
        recs = [data[s][seed] for seed in SEEDS if data[s][seed]]
        if not recs:
            print(f"| {SEEDS_EN[s]} | | | | |")
            continue
        sm = [r["xgb_region_same"]["dur_mae_min"] for r in recs]
        zr = [r["xgb_region_zero"]["dur_mae_min"] for r in recs]
        fs = [r["xgb_region_few"]["0.05"]["dur_mae_min"] for r in recs]
        print(f"| {SEEDS_EN[s]} | {fmt(stats(sm))} | {fmt(stats(zr))} | {eff(sm, zr)*100:+.0f}% | {fmt(stats(fs))} |")

    print("\n## 4.2.3 少样本 STRAT 5%（区域级）\n")
    print("| 目标 | 零样本 MAE | 5% MAE | Δ | 零样本 acc | 5% acc |")
    print("|---|---:|---:|---:|---:|---:|")
    for s in SCENES:
        recs = [data[s][seed] for seed in SEEDS if data[s][seed]]
        if not recs:
            continue
        z = [r["zero_shot"]["dur_mae_min"] for r in recs]
        f = [r["few_shot"]["0.05"]["dur_mae_min"] for r in recs]
        az = [r["zero_shot"]["tf_acc"] for r in recs]
        af = [r["few_shot"]["0.05"]["tf_acc"] for r in recs]
        mz, sz = np.mean(z), np.std(z, ddof=1)
        mf, sf = np.mean(f), np.std(f, ddof=1)
        print(f"| {SEEDS_EN[s]} | {mz:.2f} ± {sz:.2f} | {mf:.2f} ± {sf:.2f} | {mf-mz:+.2f} | "
              f"{np.mean(az):.3f} | {np.mean(af):.3f} |")


if __name__ == "__main__":
    main()
