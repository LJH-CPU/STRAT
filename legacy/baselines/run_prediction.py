"""
批量运行预测 Pipeline（BiGRU 分段时长预测）。
支持对所有景区或指定景区批量训练。

用法：
  # 对论文3个景区跑 BiGRU 预测（使用 cluster/output 下的聚类结果）
  uv run baselines/run_prediction.py --scene_paper

  # 对所有景区跑
  uv run baselines/run_prediction.py --scene_all

  # 指定景区和聚类方法
  uv run baselines/run_prediction.py --scene 青城山 --cluster_method strat

  # 自定义训练参数
  uv run baselines/run_prediction.py --scene_paper --epochs 200 --batch_size 32
"""

import os
import sys
import argparse
import json
import time
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── 路径配置 ─────────────────────────────────────────────
CLUSTER_OUTPUT_DIR = Path(__file__).parent.parent / "cluster" / "output"
CLEANED_DIR = Path(__file__).parent.parent / "data-project" / "cleaned_labeled_data"
PREDICTION_OUTPUT_DIR = Path(__file__).parent.parent / "prediction" / "output"
SUMMARY_FILE = Path(__file__).parent.parent / "prediction_results.json"

PAPER_SCENES = {"青城山", "峨眉山", "武侯祠博物馆"}


def resolve_scenes(scene_name=None, scene_all=False, scene_paper=False, min_tracks=30):
    """解析要处理的景区列表，返回 [(name, csv_path), ...]"""
    if scene_name:
        csv_path = CLUSTER_OUTPUT_DIR / f"{scene_name}_strat.csv"
        if not csv_path.exists():
            csv_path = CLEANED_DIR / f"{scene_name}_cleaned.csv"
        if not csv_path.exists():
            print(f"错误: 找不到 {scene_name} 的聚类结果")
            sys.exit(1)
        return [(scene_name, csv_path)]

    # 自动检测 cluster/output 下的聚类结果
    if scene_all or scene_paper:
        cluster_files = sorted(CLUSTER_OUTPUT_DIR.glob("*_strat.csv"))
        if not cluster_files:
            print(f"警告: {CLUSTER_OUTPUT_DIR} 无聚类结果，回退到清洗后 CSV")
            cluster_files = sorted(CLEANED_DIR.glob("*_cleaned.csv"))
        csv_files = []
        for f in cluster_files:
            name = f.stem.replace("_strat", "").replace("_clustered", "")
            if scene_paper and name not in PAPER_SCENES:
                continue
            csv_files.append((name, f))
        return csv_files

    return []


def run_one_prediction(scene_name, csv_path, args):
    """对单个景区跑预测训练。"""
    output_dir = PREDICTION_OUTPUT_DIR / scene_name
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[预测] {scene_name}")
    print(f"  CSV:       {csv_path}")
    print(f"  输出:      {output_dir}")
    print(f"  模型:      BiGRU | epochs={args.epochs} | batch={args.batch_size}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "prediction" / "train.py"),
        "--csv", str(csv_path),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--hidden_dim", str(args.hidden_dim),
        "--embed_dim", str(args.embed_dim),
        "--lr", str(args.lr),
        "--dropout", str(args.dropout),
        "--output_dir", str(output_dir),
        "--seed", str(args.seed),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    # 解析结果文件
    metrics_path = output_dir / "metrics.json"
    metrics = {"status": "成功" if result.returncode == 0 else "失败", "time_min": round(elapsed / 60, 1)}
    if (output_dir / "route_stats.pkl").exists():
        metrics["model_saved"] = True

    return metrics


def main():
    parser = argparse.ArgumentParser(description="批量运行 BiGRU 预测")
    parser.add_argument("--scene", type=str, default=None, help="景区名称")
    parser.add_argument("--scene_all", action="store_true", help="对所有景区运行")
    parser.add_argument("--scene_paper", action="store_true",
                        help="只跑论文3个景区 (青城山/峨眉山/武侯祠)")
    parser.add_argument("--cluster_method", type=str, default="strat",
                        help="使用的聚类方法 (默认: strat)")
    parser.add_argument("--min_tracks", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    scenes = resolve_scenes(args.scene, args.scene_all, args.scene_paper, args.min_tracks)
    if not scenes:
        print("没有符合条件的景区。")
        return

    print("=" * 60)
    print("批量预测 Pipeline (BiGRU 分段时长)")
    print("=" * 60)
    print(f"将处理 {len(scenes)} 个景区: {', '.join(s[0] for s in scenes)}")

    results = {}
    for scene_name, csv_path in scenes:
        m = run_one_prediction(scene_name, csv_path, args)
        results[scene_name] = m

    # ── 汇总 ──
    print(f"\n\n{'='*60}")
    print("汇总")
    print(f"{'='*60}")
    print(f"{'景区':<12} {'状态':<8} {'耗时(min)':<10}")
    print("-" * 36)
    for name, r in sorted(results.items()):
        print(f"{name:<12} {r.get('status',''):<8} {r.get('time_min','—'):<10}")
    print()

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
