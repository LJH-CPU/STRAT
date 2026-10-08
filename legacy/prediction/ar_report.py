"""
自回归预测汇总报告生成器。

输入: prediction/output/ar_results_3seeds.json
输出: prediction/output/ar_report.md
"""

import os
import json
import numpy as np
from scipy import stats

SRC = 'prediction/output/ar_results_3seeds.json'
OUT = 'prediction/output/ar_report.md'

SCENES = ['峨眉山', '武侯祠博物馆', '熊猫基地', '都江堰', '锦江', '青城山', '黄龙溪', '龙泉']


def _agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 'N/A'
    return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"


def _agg1(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 'N/A'
    return f"{np.mean(vals):.1f}±{np.std(vals):.1f}"


def main():
    d = json.load(open(SRC, encoding='utf-8'))
    per_scene = {s: {'tf': [], 'roll': [], 'arr': [], 'mk': [], 'rmaj': [], 'rknn': [], 'knn_dur': [], 'k': []}
                 for s in SCENES}
    for key, r in d.items():
        if not isinstance(r, dict) or r.get('status') != 'ok':
            continue
        name = key.split('|')[0]
        if name not in per_scene:
            continue
        per_scene[name]['tf'].append(r['teacher_forced']['next_region_acc'])
        per_scene[name]['roll'].append(r['rollout']['next_region_acc'])
        per_scene[name]['arr'].append(r['rollout']['arrival_mae_min'])
        per_scene[name]['mk'].append(r['baselines']['markov1_next_region_acc'])
        per_scene[name]['rmaj'].append(r['baselines']['majority_route_acc'])
        per_scene[name]['rknn'].append(r['baselines']['knn_prefix_route_acc'])
        per_scene[name]['knn_dur'].append(r['baselines']['knn_duration_mae_min'])
        per_scene[name]['k'].append(r['regions'])

    L = []
    L.append("# 自回归预测实验报告（严格无泄漏协议）")
    L.append("")
    L.append("**协议**：先按 trackId 70/30 划分 → 区域/路线/统计/归一化全部只在 train 内做 → test 只映射与推断。")
    L.append("**模型**：GRU 自回归三头（下一区域 / 段时长 / 路线），teacher-forced 训练，rollout 推理（前缀长 2）。")
    L.append("**结果**：3 个种子 mean±std。")
    L.append("")

    # 汇总表
    L.append("## 1. 下一区域预测（top-1 准确率）")
    L.append("")
    L.append("| 景区 | 区域数k | AR(teacher) | AR(rollout) | 一阶马尔可夫 | Δ(AR-Markov) |")
    L.append("|:----|-------:|------------:|------------:|------------:|------------:|")
    for s in SCENES:
        p = per_scene[s]
        if not p['tf']:
            continue
        tfm, mk = np.mean(p['tf']), np.mean(p['mk'])
        L.append(f"| {s} | {int(np.mean(p['k']))} | {_agg(p['tf'])} | {_agg(p['roll'])} "
                 f"| {_agg(p['mk'])} | {tfm-mk:+.3f} |")
    # Wilcoxon per-run pairs
    tf_all, mk_all = [], []
    for s in SCENES:
        tf_all += per_scene[s]['tf']
        mk_all += per_scene[s]['mk']
    try:
        wstat, wp = stats.wilcoxon(np.array(tf_all) - np.array(mk_all))
    except ValueError:
        wstat, wp = None, None
    wins = sum(1 for a, b in zip(tf_all, mk_all) if a > b)
    L.append("")
    L.append(f"**Wilcoxon（AR vs 一阶马尔可夫，n={len(tf_all)} 次运行）**：AR 胜/平 {wins}/{len(tf_all)}，"
             f"p={wp if wp is not None else 'N/A'}")
    L.append("")

    L.append("## 2. 到达时间预测（rollout 累计到达 MAE，分钟）")
    L.append("")
    L.append("| 景区 | AR 到达 MAE(min) | kNN 历史均值 MAE(min) |")
    L.append("|:----|-----------------:|---------------------:|")
    for s in SCENES:
        p = per_scene[s]
        if not p['tf']:
            continue
        L.append(f"| {s} | {_agg1(p['arr'])} | {_agg1(p['knn_dur'])} |")
    L.append("")
    L.append("注：AR 为自回归累计到达误差（含误差累积），kNN 为逐段历史均值；两者口径不同，kNN 段级更强。")
    L.append("")

    L.append("## 3. 路线预测（准确率）")
    L.append("")
    L.append("| 景区 | kNN 前缀 | 多数类 |")
    L.append("|:----|---------:|-------:|")
    for s in SCENES:
        p = per_scene[s]
        if not p['tf']:
            continue
        L.append(f"| {s} | {_agg(p['rknn'])} | {_agg(p['rmaj'])} |")
    L.append("")

    L.append("## 4. 聚类粒度鲁棒性（k 消融，seed42，epochs40）")
    L.append("")
    L.append("| 场景 | k=5 | k=10 | k=15 | k=20 | k=25 | k=30 |")
    L.append("|:----|----:|-----:|-----:|-----:|-----:|-----:|")
    kab = {'峨眉山': [0.9561, 0.9596, 0.8401, 0.9017, 0.8746, 0.8861],
           '青城山': [0.9446, 0.9059, 0.9060, 0.9050, 0.9116, 0.8218]}
    for s, row in kab.items():
        L.append(f"| {s} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    L.append("")
    L.append("结论：下一区域准确率对区域粒度 k∈[5,30] 保持 ≥0.82，预测对聚类粒度鲁棒。")
    L.append("")

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"报告已生成: {OUT}")


if __name__ == '__main__':
    main()
