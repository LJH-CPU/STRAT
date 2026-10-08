#!/usr/bin/env python3
"""
全景区全基线对比实验入口。
按顺序执行：
  1. 聚类基线（K-Means / DBSCAN / HDBSCAN / Agglomerative / Spectral / STRAT）
  2. 预测基线（Region-Median / LightGBM / BiLSTM）
  3. 生成汇总报告

用法：
    uv run baselines/run_all_baselines.py                       # 全部景区
    uv run baselines/run_all_baselines.py --scene_paper          # 论文3个景区
    uv run baselines/run_all_baselines.py --skip_clustering      # 跳过聚类阶段
    uv run baselines/run_all_baselines.py --skip_prediction      # 跳过预测阶段
"""

import os
import sys
import argparse
import time
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
BASELINES_DIR = ROOT / "baselines"
OUTPUT_DIR = BASELINES_DIR / "output"


def run_script(name, args_list):
    """运行一个 Python 脚本并实时输出。"""
    cmd = [sys.executable, str(BASELINES_DIR / name)] + args_list
    print(f"\n{'='*60}")
    print(f"$ {' '.join(str(a) for a in cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def generate_report(all_results):
    """生成 Markdown 汇总报告，包含聚类和预测结果。"""
    report_path = ROOT / "baseline_comparison_report.md"
    lines = []
    lines.append("# STRAT 基线对比实验报告")
    lines.append("")
    now = __import__('datetime').datetime.now()
    lines.append(f"**生成时间**: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # 聚类汇总
    cluster_data = all_results.get("clustering", {})
    if cluster_data:
        lines.append("## 1. 聚类基线对比 (轮廓系数 ↑)")
        lines.append("")
        scenes = sorted(cluster_data.keys())
        methods = ['strat', 'k-means', 'dbscan', 'agglomerative', 'spectral_gaussian']
        method_names = {
            'strat': 'STRAT(ours)', 'k-means': 'K-Means', 'dbscan': 'DBSCAN',
            'agglomerative': 'Agglomerative', 'spectral_gaussian': 'Spectral(G)',
        }
        # Header
        header = f"| {'场景':<10} | {'轨迹':<6}"
        for m in methods:
            header += f" | {method_names[m]:<16}"
        header += " |"
        lines.append(header)
        sep = "|" + "-" * 12 + "|" + "-" * 8
        for _ in methods:
            sep += "|" + "-" * 18
        sep += "|"
        lines.append(sep)

        for scene in scenes:
            sdata = cluster_data.get(scene, {})
            gm = sdata.get('strat', {})
            n_tr = gm.get('n_clusters', '')
            row = f"| {scene:<10} | {str(n_tr):<6}"
            for m in methods:
                entry = sdata.get(m, {})
                sil = entry.get('silhouette', '—')
                if isinstance(sil, float):
                    row += f" | Sil={sil:.4f}"
                else:
                    row += f" | {'—':>16}"
            row += " |"
            lines.append(row)
        lines.append("")

    # 预测汇总
    pred_data = all_results.get("prediction", {})
    if pred_data:
        lines.append("## 2. 预测基线对比 (Seg MAE ↓, 秒)")
        lines.append("")
        scenes = sorted(pred_data.keys())
        methods = ['region_median', 'lightgbm', 'bilstm']
        method_names = {'region_median': 'Median', 'lightgbm': 'LightGBM', 'bilstm': 'BiLSTM'}

        header = f"| {'场景':<12}"
        for m in methods:
            header += f" | {method_names[m]:<14}"
        header += " |"
        lines.append(header)
        sep = "|" + "-" * 14
        for _ in methods:
            sep += "|" + "-" * 16
        sep += "|"
        lines.append(sep)

        for scene in scenes:
            sdata = pred_data.get(scene, {})
            row = f"| {scene:<12}"
            for m in methods:
                entry = sdata.get(m, {})
                mae = entry.get('seg_mae', '—')
                if isinstance(mae, (int, float)):
                    row += f" | {mae:>8.0f}s"
                else:
                    row += f" | {'—':>14}"
            row += " |"
            lines.append(row)
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="全景区全基线对比实验")
    parser.add_argument("--scene_paper", action="store_true",
                        help="只跑论文3个景区")
    parser.add_argument("--min_tracks", type=int, default=30)
    parser.add_argument("--skip_clustering", action="store_true")
    parser.add_argument("--skip_prediction", action="store_true")
    parser.add_argument("--epochs", type=int, default=100,
                        help="预测训练轮数")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scene_flag = []
    if args.scene_paper:
        scene_flag = ["--scene_paper"]
    else:
        scene_flag = ["--scene_all"]

    all_results = {}

    # ── Phase 1: 聚类基线 ──
    if not args.skip_clustering:
        print("\n" + "#" * 70)
        print("# Phase 1: 聚类基线对比")
        print("#" * 70)
        ok = run_script("clustering_baselines.py", scene_flag + [
            "--min_tracks", str(args.min_tracks),
        ])
        # 读取结果
        cluster_json = OUTPUT_DIR / "clustering_baselines.json"
        if cluster_json.exists():
            with open(cluster_json, "r", encoding="utf-8") as f:
                all_results["clustering"] = json.load(f)
        if not ok:
            print("  [警告] 聚类阶段部分失败")
    else:
        print("\n[跳过] 聚类阶段")

    # ── Phase 2: 预测基线 ──
    if not args.skip_prediction:
        print("\n" + "#" * 70)
        print("# Phase 2: 预测基线对比")
        print("#" * 70)
        ok = run_script("prediction_baselines.py", scene_flag + [
            "--min_tracks", str(args.min_tracks),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--models", "median", "lightgbm", "bilstm",
        ])
        pred_json = OUTPUT_DIR / "prediction_baselines.json"
        if pred_json.exists():
            with open(pred_json, "r", encoding="utf-8") as f:
                all_results["prediction"] = json.load(f)
        if not ok:
            print("  [警告] 预测阶段部分失败")
    else:
        print("\n[跳过] 预测阶段")

    # ── Phase 3: 生成报告 ──
    report_path = generate_report(all_results)
    print(f"\n{'='*60}")
    print("全部基线实验完成！")
    print(f"{'='*60}")
    print(f"  报告: {report_path}")
    print(f"  聚类结果: {OUTPUT_DIR / 'clustering_baselines.json'}")
    print(f"  预测结果: {OUTPUT_DIR / 'prediction_baselines.json'}")


if __name__ == "__main__":
    main()
