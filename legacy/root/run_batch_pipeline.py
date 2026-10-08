#!/usr/bin/env python3
"""
批量运行 STRAT 聚类 + 预测 Pipeline。

流程：
  1. 遍历 cleaned_labeled_data 下所有 *_cleaned.csv
  2. 跳过轨迹数不足的景区（默认 < 50）
  3. 对每个景区：
     a. 粒球谱聚类 → 带 region_id 的 CSV
     b. 路线发现 → 带 route_id 的 CSV
     c. BiGRU 分段时长预测 → 模型 + 指标
  4. 汇总输出表格

用量：
    uv run run_batch_pipeline.py
    uv run run_batch_pipeline.py --min_tracks 30 --epochs 200
"""

import os
import sys
import argparse
import json
import time
import subprocess
from pathlib import Path

from config import (
    PROJECT_DIR,
    CLEANED_DIR,
    CLUSTER_OUTPUT_DIR,
    PREDICTION_OUTPUT_DIR,
    BATCH_RESULTS_JSON,
    PAPER_SCENERIES,
)

os.makedirs(CLUSTER_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTION_OUTPUT_DIR, exist_ok=True)


def count_tracks(csv_path):
    """快速读取 CSV 的 trackId 数量（只读第一列节省内存）。"""
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["trackId"])
    return df["trackId"].nunique()


def run_clustering(csv_path, output_path):
    """运行粒球谱聚类 + 路线发现，保存带标签的 CSV。"""
    print(f"\n{'='*60}")
    print(f"[聚类] {csv_path.name}")
    print(f"{'='*60}")

    sys.path.insert(0, str(PROJECT_DIR / "cluster"))
    from scenery_route_clustering import run_full_pipeline

    df_out, scenic_m, route_m, artifacts = run_full_pipeline(str(csv_path))

    # 保存输出
    cols_to_save = [
        "经度", "纬度", "海拔", "速度(km/h)", "is_stop",
        "trackId", "时间_秒", "region_id", "route_id"
    ]
    available = [c for c in cols_to_save if c in df_out.columns]
    df_out[available].to_csv(output_path, index=False)

    print(f"\n  ✓ 聚类完成: region_id 已分配")
    print(f"  ✓ 路线发现: {route_m['n_routes']} 条路线")
    print(f"  ✓ 输出: {output_path}")
    print(f"  📊 聚类指标: sil={scenic_m.get('silhouette', 'N/A'):.4f}, "
          f"DBI={scenic_m.get('davies_bouldin', 'N/A'):.4f}, "
          f"regions={scenic_m.get('n_regions', 'N/A')}")

    return scenic_m, route_m


def run_prediction(csv_path, args):
    """训练 BiGRU 分段时长预测模型。"""
    name = csv_path.stem.replace("_clustered", "")

    # 每个景区一个独立的输出子目录
    output_dir = PREDICTION_OUTPUT_DIR / name
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[预测] {name}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, str(PROJECT_DIR / "prediction" / "train.py"),
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

    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="STRAT 批量聚类+预测")
    parser.add_argument("--min_tracks", type=int, default=50,
                        help="跳过轨迹数少于该值的景区 (默认: 50)")
    parser.add_argument("--skip_prediction", action="store_true",
                        help="跳过预测阶段，只做聚类")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only_paper", action="store_true",
                        help="只跑论文使用的三个景区 (青城山/峨眉山/武侯祠)")
    args = parser.parse_args()

    # ── 查找清洗后的 CSV ──────────────────────────────
    csv_files = sorted(CLEANED_DIR.glob("*_cleaned.csv"))
    if not csv_files:
        print(f"错误: 在 {CLEANED_DIR} 中未找到 *_cleaned.csv")
        print("请先运行 data-project/clean_all.py")
        sys.exit(1)

    print("=" * 60)
    print("STRAT 批量 Pipeline")
    print("=" * 60)
    print(f"清洗数据目录: {CLEANED_DIR}")
    print(f"聚类输出目录: {CLUSTER_OUTPUT_DIR}")
    print(f"预测输出目录: {PREDICTION_OUTPUT_DIR}")
    print()

    # ── 统计可用景区 ──────────────────────────────
    candidates = []
    for f in csv_files:
        name = f.stem.replace("_cleaned", "")
        if args.only_paper and name not in PAPER_SCENERIES:
            continue
        n = count_tracks(f)
        if n >= args.min_tracks:
            candidates.append((name, f, n))
            print(f"  ✓ {name}: {n} 条轨迹")
        else:
            print(f"  ✗ {name}: {n} 条轨迹 (跳过，<{args.min_tracks})")

    if not candidates:
        print(f"\n没有满足条件的景区 (min_tracks={args.min_tracks})")
        sys.exit(0)

    print(f"\n将处理 {len(candidates)} 个景区")

    # ── 执行 Pipeline ──────────────────────────────
    results = {}
    for name, cleaned_csv, n_tracks in candidates:
        clustered_csv = CLUSTER_OUTPUT_DIR / f"{name}_clustered.csv"
        start_t = time.time()

        print(f"\n{'#'*60}")
        print(f"# 处理: {name} ({n_tracks} 条轨迹)")
        print(f"{'#'*60}")

        # 阶段 1: 聚类
        try:
            scenic_m, route_m = run_clustering(cleaned_csv, clustered_csv)
        except Exception as e:
            print(f"  [错误] 聚类失败: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"status": "聚类失败", "error": str(e)}
            continue

        # 阶段 2: 预测
        pred_success = True
        if not args.skip_prediction:
            try:
                pred_success = run_prediction(clustered_csv, args)
            except Exception as e:
                print(f"  [错误] 预测失败: {e}")
                pred_success = False

        elapsed = time.time() - start_t
        results[name] = {
            "status": "成功" if pred_success else "聚类成功+预测失败",
            "tracks": n_tracks,
            "regions": scenic_m.get("n_regions", "N/A"),
            "routes": route_m.get("n_routes", "N/A"),
            "silhouette": round(scenic_m.get("silhouette", 0), 4),
            "davies_bouldin": round(scenic_m.get("davies_bouldin", 0), 4),
            "time_min": round(elapsed / 60, 1),
        }

    # ── 汇总 ──────────────────────────────
    print(f"\n\n{'='*60}")
    print("Pipeline 汇总")
    print(f"{'='*60}")
    print(f"{'景区':<12} {'轨迹':<6} {'区域':<6} {'路线':<6} {'轮廓系数':<10} {'DBI':<8} {'耗时(min)':<10} {'状态'}")
    print("-" * 70)
    total_tracks = 0
    for name, r in sorted(results.items()):
        total_tracks += r.get("tracks", 0)
        print(f"{name:<12} {r.get('tracks','—'):<6} {r.get('regions','—'):<6} "
              f"{r.get('routes','—'):<6} {r.get('silhouette','—'):<10} "
              f"{r.get('davies_bouldin','—'):<8} {r.get('time_min','—'):<10} {r.get('status','')}")
    print("-" * 70)
    print(f"{'总计':<12} {total_tracks:<6}")
    print()

    # 保存结果
    with open(BATCH_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {BATCH_RESULTS_JSON}")


if __name__ == "__main__":
    main()
