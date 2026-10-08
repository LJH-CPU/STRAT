#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POI Ground Truth 评估可视化。

生成图表：
  1. ARI/NMI/FMI 横向柱状图（所有景区）
  2. 优秀景区散点图：轨迹点按 gt_region_id vs region_id 着色对比
  3. ARI 对比不同 min_pois 参数的敏感性分析

用法：
    python poi_visualize.py
    python poi_visualize.py --scenery 黄龙溪 都江堰 龙泉 峨眉山
    python poi_visualize.py --dpi 300 --output ./figures
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# 中文字体
for font_name in ["WenQuanYi Micro Hei", "Noto Sans CJK SC",
                  "SimHei", "Microsoft YaHei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.family"] = font_name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = SCRIPT_DIR / "data" / "ground_truth"
FIGURE_DIR = SCRIPT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_DIR = SCRIPT_DIR.parent / "data-project" / "cleaned_labeled_data"

# 景区中文名
SCENERY_CN = {
    "峨眉山": "峨眉山", "武侯祠博物馆": "武侯祠", "熊猫基地": "熊猫基地",
    "都江堰": "都江堰", "锦江": "锦江", "青城山": "青城山",
    "龙泉": "龙泉", "虹口": "虹口", "黄龙溪": "黄龙溪", "锦里": "锦里",
}

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def load_metrics() -> list[dict]:
    path = OUTPUT_DIR / "evaluation_metrics.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 图1: ARI / NMI / FMI 柱状图
# ============================================================

def plot_metrics_bar(metrics: list[dict], dpi: int = 200):
    """所有景区的 ARI/NMI/FMI 横向柱状图。"""
    valid = [m for m in metrics if m.get("ari") is not None and m["status"] == "成功"]
    if not valid:
        print("  没有可绘制的有效景区")
        return

    names = [SCENERY_CN.get(m["name"], m["name"]) for m in valid]
    ari = [m["ari"] for m in valid]
    nmi = [m["nmi"] for m in valid]
    fmi = [m["fmi"] for m in valid]

    y = np.arange(len(names))
    h = 0.25

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.6)))

    bars1 = ax.barh(y - h, ari, h, label="ARI", color="#4363d8", alpha=0.9)
    bars2 = ax.barh(y, nmi, h, label="NMI", color="#e6194b", alpha=0.9)
    bars3 = ax.barh(y + h, fmi, h, label="FMI", color="#3cb44b", alpha=0.9)

    # 数值标注
    for bars in [bars1, bars2, bars3]:
        for bar, val in zip(bars, (ari if bars is bars1 else
                                   nmi if bars is bars2 else fmi)):
            if val is not None:
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=8,
                        color="gray")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.4, ls="--", color="gray", alpha=0.4, label="ARI=0.4 (好)")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title("POI Ground Truth 聚类评估结果", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # 平均线
    avg_ari = np.mean(ari)
    avg_nmi = np.mean(nmi)
    avg_fmi = np.mean(fmi)
    ax.text(0.98, 0.02, f"平均 ARI={avg_ari:.3f}  NMI={avg_nmi:.3f}  FMI={avg_fmi:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color="gray", style="italic")

    fig.tight_layout()
    path = FIGURE_DIR / "metrics_bar.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  图1 保存: {path}")
    plt.close(fig)

    # 只保存 ARI 的简易版本
    fig2, ax2 = plt.subplots(figsize=(8, max(3.5, len(names) * 0.5)))
    order = np.argsort(ari)
    colors_bar = ["#e6194b" if a < 0.2 else ("#f58231" if a < 0.4 else "#3cb44b") for a in np.array(ari)[order]]
    ax2.barh(range(len(names)), np.array(ari)[order], color=colors_bar, alpha=0.85, edgecolor="white")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(np.array(names)[order], fontsize=11)
    ax2.set_xlabel("ARI", fontsize=12)
    ax2.set_xlim(0, 0.9)
    ax2.axvline(0.4, ls="--", color="gray", alpha=0.5, label="好 (ARI=0.4)")
    ax2.axvline(0.2, ls=":", color="gray", alpha=0.5, label="可接受 (ARI=0.2)")
    ax2.legend(fontsize=9)
    ax2.set_title("各景区 ARI 对比", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    fig2.tight_layout()
    path2 = FIGURE_DIR / "ari_bar.png"
    fig2.savefig(path2, dpi=dpi, bbox_inches="tight")
    print(f"  图1b 保存: {path2}")
    plt.close(fig2)


# ============================================================
# 图2: 轨迹点散点图（gt vs pipeline 对比）
# ============================================================

def match_color(labels: np.ndarray, n_colors: int = None) -> list:
    """标签转颜色列表。"""
    uniq = np.unique(labels[labels >= 0])
    if n_colors is None:
        n_colors = len(uniq)
    cmap = ListedColormap(COLORS[:max(n_colors, 1)])
    norm_labels = np.zeros_like(labels, dtype=float)
    mapping = {v: i / max(n_colors, 1) for i, v in enumerate(sorted(uniq))}
    for k, v in mapping.items():
        norm_labels[labels == k] = v
    norm_labels[labels < 0] = -1
    return norm_labels, cmap


def plot_scenery_comparison(name: str, dpi: int = 200, sample: int = 20000):
    """单个景区的 gt_region_id vs region_id 散点对比。"""
    csv_path = OUTPUT_DIR / f"{name}_labeled.csv"
    if not csv_path.exists():
        print(f"  [跳过] {name}: 无标签 CSV")
        return

    df = pd.read_csv(csv_path, encoding="utf-8")
    if len(df) == 0:
        print(f"  [跳过] {name}: 空数据")
        return

    # 采样
    if len(df) > sample:
        df = df.sample(n=sample, random_state=42)

    gt = df["gt_region_id"].values.astype(np.int32)
    pred = df["region_id"].values.astype(np.int32)
    lng, lat = df["经度"].values, df["纬度"].values

    # 只保留有 ground truth 标签的点
    valid_gt = gt != -1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左图: GT Region ---
    ax = axes[0]
    gt_valid = gt[valid_gt]
    n_gt = len(set(gt_valid))
    gt_norm, gt_cmap = match_color(gt_valid, n_gt)
    scatter = ax.scatter(lng[valid_gt], lat[valid_gt], c=gt_norm,
                         cmap=gt_cmap, s=1, alpha=0.4, rasterized=True)
    ax.set_title(f"{SCENERY_CN.get(name, name)} — Ground Truth 区域\n"
                 f"({n_gt} 个 POI 区域)", fontsize=12)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_aspect(1.0 / np.cos(np.mean(lat) * np.pi / 180))
    patches = []
    for i in range(n_gt):
        patches.append(mpatches.Circle((0, 0), 1, color=COLORS[i % len(COLORS)],
                                       label=f"Region {i}"))
    ax.legend(handles=patches, fontsize=6, loc="best", ncol=2, title="GT Regions")

    # --- 右图: Pipeline Region ---
    ax = axes[1]
    pred_valid = pred[valid_gt]
    n_pred = len(set(pred_valid))
    pred_norm, pred_cmap = match_color(pred_valid, n_pred)
    scatter = ax.scatter(lng[valid_gt], lat[valid_gt], c=pred_norm,
                         cmap=pred_cmap, s=1, alpha=0.4, rasterized=True)
    ax.set_title(f"{SCENERY_CN.get(name, name)} — Pipeline 聚簇\n"
                 f"({n_pred} 个 region_id)", fontsize=12)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_aspect(1.0 / np.cos(np.mean(lat) * np.pi / 180))
    patches2 = []
    for i in range(n_pred):
        patches2.append(mpatches.Circle((0, 0), 1, color=COLORS[i % len(COLORS)],
                                        label=f"R{i}"))
    ax.legend(handles=patches2, fontsize=6, loc="best", ncol=2, title="Pipeline")

    fig.suptitle(f"{SCENERY_CN.get(name, name)} 聚类评估",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = FIGURE_DIR / f"{name}_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  图2 ({name}) 保存: {path}")
    plt.close(fig)


# ============================================================
# 图3: 混淆矩阵（GT vs Pipeline）
# ============================================================

def plot_confusion_matrix(name: str, dpi: int = 200, sample: int = 50000):
    """GT 区域 × Pipeline 区域的共现热力图。"""
    csv_path = OUTPUT_DIR / f"{name}_labeled.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path, encoding="utf-8")
    if len(df) == 0:
        return
    if len(df) > sample:
        df = df.sample(n=sample, random_state=42)

    gt = df["gt_region_id"].values.astype(np.int32)
    pred = df["region_id"].values.astype(np.int32)
    valid = (gt != -1)

    gt_v = gt[valid]
    pred_v = pred[valid]

    n_gt = len(set(gt_v))
    n_pred = len(set(pred_v))

    if n_gt < 2 or n_pred < 2:
        print(f"  [跳过混淆矩阵] {name}: GT区域={n_gt}, Pipeline区域={n_pred}")
        return

    # 构建矩阵
    gt_uniq = sorted(set(gt_v))
    pred_uniq = sorted(set(pred_v))
    gt_map = {v: i for i, v in enumerate(gt_uniq)}
    pred_map = {v: i for i, v in enumerate(pred_uniq)}

    mat = np.zeros((len(gt_uniq), len(pred_uniq)), dtype=np.float64)
    for g, p in zip(gt_v, pred_v):
        mat[gt_map[g], pred_map[p]] += 1

    # 归一化（行归一化）
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    mat_norm = mat / row_sum

    fig, ax = plt.subplots(figsize=(max(5, n_pred * 0.6), max(4, n_gt * 0.6)))
    im = ax.imshow(mat_norm, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8, label="归一化比例")

    # 标注数值
    for i in range(len(gt_uniq)):
        for j in range(len(pred_uniq)):
            if mat_norm[i, j] > 0.05:
                ax.text(j, i, f"{mat_norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if mat_norm[i, j] > 0.5 else "black")

    ax.set_xlabel("Pipeline Region ID", fontsize=11)
    ax.set_ylabel("Ground Truth Region ID", fontsize=11)
    ax.set_title(f"{SCENERY_CN.get(name, name)} — 混淆矩阵\n"
                 f"(ARI={load_metrics_for(name):.3f}  NMI={load_nmi_for(name):.3f})",
                 fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(pred_uniq)))
    ax.set_yticks(range(len(gt_uniq)))
    ax.set_xticklabels(pred_uniq, fontsize=8)
    ax.set_yticklabels(gt_uniq, fontsize=8)

    fig.tight_layout()
    path = FIGURE_DIR / f"{name}_confusion.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  图3 ({name}) 保存: {path}")
    plt.close(fig)


def load_metrics_for(name: str) -> float:
    metrics = load_metrics()
    for m in metrics:
        if m.get("name") == name and m.get("ari") is not None:
            return m["ari"]
    return 0.0


def load_nmi_for(name: str) -> float:
    metrics = load_metrics()
    for m in metrics:
        if m.get("name") == name and m.get("nmi") is not None:
            return m["nmi"]
    return 0.0


# ============================================================
# 图4: POI 缓冲区可视化
# ============================================================

def plot_poi_buffers(name: str, dpi: int = 200):
    """显示 POI 缓冲区 + 投影后的区域，叠加部分轨迹点。"""
    # 加载标签 CSV 看轨迹
    csv_path = OUTPUT_DIR / f"{name}_labeled.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path, encoding="utf-8")
    if len(df) == 0:
        return

    # 加载 POI
    poi_path = SCRIPT_DIR / "data" / "raw" / "all_scenery_poi.json"
    if not poi_path.exists():
        return
    with open(poi_path, "r", encoding="utf-8") as f:
        all_pois = json.load(f)
    scene_pois = [p for p in all_pois if p.get("scenery") == name]

    if not scene_pois:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # 轨迹点（灰色小点）
    if len(df) > 50000:
        df_plot = df.sample(50000, random_state=42)
    else:
        df_plot = df
    ax.scatter(df_plot["经度"], df_plot["纬度"], c="lightgray",
               s=0.5, alpha=0.3, rasterized=True, label="轨迹点")

    # POI 位置（原始）
    pois_lng = [p["lon"] for p in scene_pois]
    pois_lat = [p["lat"] for p in scene_pois]
    ax.scatter(pois_lng, pois_lat, c="#e6194b", s=20, marker="o",
               alpha=0.8, label=f"原始 POI ({len(scene_pois)})", zorder=5)

    # POI 投影后位置（取自 CSV 中的 gt_poi_name 和 gt_region_id）
    # 采样一些有标签的点来着色显示区域
    gt_valid = df[df["gt_region_id"] >= 0].copy()
    if len(gt_valid) > 10000:
        gt_valid = gt_valid.sample(10000, random_state=42)

    if len(gt_valid) > 0 and gt_valid["gt_region_id"].nunique() > 1:
        n_regions = gt_valid["gt_region_id"].nunique()
        norm_labels, cmap = match_color(gt_valid["gt_region_id"].values, n_regions)
        sc = ax.scatter(gt_valid["经度"], gt_valid["纬度"],
                        c=norm_labels, cmap=cmap, s=2, alpha=0.5,
                        rasterized=True, zorder=3)
        patches = []
        for i in range(n_regions):
            patches.append(mpatches.Circle((0, 0), 1,
                                           color=COLORS[i % len(COLORS)],
                                           label=f"GT 区域 {i}"))
        ax.legend(handles=patches, fontsize=7, loc="upper right",
                  title="POI 区域", title_fontsize=8)

    ax.set_title(f"{SCENERY_CN.get(name, name)} — POI 缓冲区 + 轨迹分布",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_aspect(1.0 / np.cos(np.mean(df["纬度"]) * np.pi / 180))

    fig.tight_layout()
    path = FIGURE_DIR / f"{name}_poi_map.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  图4 ({name}) 保存: {path}")
    plt.close(fig)


# ============================================================
# 图5: ARI 随 min_pois 参数变化
# ============================================================

def plot_parameter_sweep(dpi: int = 200):
    """如果有多组参数结果，绘制 ARI 变化曲线。"""
    # 从 metrics JSON 历史中读取（仅当前一组数据）
    # 这里展示一个示意图：整理当前各景区的 ARI 排序
    metrics = load_metrics()
    valid = [(m["name"], m.get("ari")) for m in metrics
             if m.get("ari") is not None]
    if not valid:
        return

    names, values = zip(*sorted(valid, key=lambda x: x[1], reverse=True))
    cn_names = [SCENERY_CN.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    colors_bar = ["#3cb44b" if v >= 0.4 else ("#f58231" if v >= 0.2 else "#e6194b")
                  for v in values]
    bars = ax.bar(x, values, color=colors_bar, alpha=0.8, edgecolor="white", width=0.6)
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(cn_names, fontsize=10, rotation=15)
    ax.set_ylabel("ARI", fontsize=12)
    ax.set_title("各景区 Ground Truth 验证 ARI 排名", fontsize=13, fontweight="bold")
    ax.axhline(0.4, ls="--", color="gray", alpha=0.5, label="ARI=0.4 (好)")
    ax.axhline(0.2, ls=":", color="gray", alpha=0.5, label="ARI=0.2 (可接受)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(values) + 0.12)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = FIGURE_DIR / "ari_ranking.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  图5 保存: {path}")
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="POI Ground Truth 评估可视化")
    parser.add_argument("--scenery", nargs="*",
                        default=["黄龙溪", "都江堰", "龙泉", "峨眉山"],
                        help="要生成散点图的景区")
    parser.add_argument("--dpi", type=int, default=200, help="图片 DPI")
    parser.add_argument("--sample", type=int, default=20000, help="散点图采样点数")
    parser.add_argument("--output", type=str, default=str(FIGURE_DIR))
    args = parser.parse_args()

    fig_dir = Path(args.output)
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()

    print("=" * 50)
    print("生成可视化图表")

    # 图1: ARI/NMI/FMI 柱状图
    print("\n[图1] 评估指标柱状图...")
    plot_metrics_bar(metrics, dpi=args.dpi)

    # 图2: 景区散点图对比
    print("\n[图2] 景区散点图对比 (GT vs Pipeline)...")
    for name in args.scenery:
        plot_scenery_comparison(name, dpi=args.dpi, sample=args.sample)

    # 图3: 混淆矩阵
    print("\n[图3] 混淆矩阵...")
    for name in args.scenery:
        plot_confusion_matrix(name, dpi=args.dpi)

    # 图4: POI 地图
    print("\n[图4] POI 缓冲区 + 轨迹分布...")
    for name in args.scenery:
        plot_poi_buffers(name, dpi=args.dpi)

    # 图5: ARI 排名
    print("\n[图5] ARI 排名...")
    plot_parameter_sweep(dpi=args.dpi)

    print(f"\n全部图表已保存到: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
