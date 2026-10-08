#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POI Enrichment — 对 STRAT 每个聚簇做 POI 语义标注。

核心逻辑：
  把每个 POI 匹配到最近的 STRAT 区域，统计各区域的 POI 类型组成。
  回答："这个聚簇是什么功能区域？"

输出：
  1. 每个景区的 POI 类型 × 区域 矩阵（CSV）
  2. 堆叠柱状图（各区域 POI 类型构成）
  3. 标注地图（聚簇按主导 POI 类型着色）
  4. 语义标签表（给每个区域起个名字）

用法：
    python poi_enrich.py
    python poi_enrich.py --scenery 峨眉山 青城山 黄龙溪
    python poi_enrich.py --buffer-radius 100 --dpi 200
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# 中文字体
for fn in ["WenQuanYi Micro Hei", "Noto Sans CJK SC",
           "SimHei", "Microsoft YaHei", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(fn, fallback_to_default=False)
        plt.rcParams["font.family"] = fn
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
POI_PATH = SCRIPT_DIR / "data" / "raw" / "all_scenery_poi.json"
CLUSTERED_DIR = SCRIPT_DIR.parent / "cluster" / "output"
OUTPUT_DIR = SCRIPT_DIR / "data" / "enrichment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# POI 类型映射（2 位码 → 中文名）
TYPE_NAMES = {
    "05": "餐饮", "06": "购物", "07": "生活服务",
    "08": "休闲", "10": "住宿", "11": "风景名胜",
    "14": "科教文化", "17": "公司企业", "20": "公共设施",
    "19": "地名地址", "12": "商务住宅", "13": "政府机构",
    "15": "交通设施", "16": "金融保险",
}
# 配色
TYPE_COLORS = {
    "餐饮": "#e74c3c", "购物": "#f39c12", "生活服务": "#95a5a6",
    "休闲": "#9b59b6", "住宿": "#3498db", "风景名胜": "#2ecc71",
    "科教文化": "#1abc9c", "公司企业": "#7f8c8d", "公共设施": "#e67e22",
    "地名地址": "#bdc3c7", "商务住宅": "#34495e", "政府机构": "#c0392b",
    "交通设施": "#16a085", "金融保险": "#f1c40f",
}

# ---------------------------------------------------------------------------
# KD-Tree
# ---------------------------------------------------------------------------

def build_kdtree(points_2d):
    n = len(points_2d)
    if n == 0:
        return None, None
    indices = np.arange(n)

    def _build(idx, depth):
        if len(idx) == 0:
            return None
        axis = depth % 2
        idx = idx[np.argsort(points_2d[idx, axis])]
        mid = len(idx) // 2
        return {"point": idx[mid], "left": _build(idx[:mid], depth + 1),
                "right": _build(idx[mid + 1:], depth + 1), "axis": axis}

    return _build(indices, 0), points_2d


def knn_search(tree, points_2d, query):
    if tree is None:
        return None, float("inf")
    best_idx, best_dist = None, float("inf")

    def _search(node, depth):
        nonlocal best_idx, best_dist
        if node is None:
            return
        node_pt = points_2d[node["point"]]
        dist = np.sqrt(np.sum((query - node_pt) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_idx = node["point"]
        axis = node["axis"]
        diff = query[axis] - node_pt[axis]
        if diff <= 0:
            _search(node["left"], depth + 1)
        else:
            _search(node["right"], depth + 1)
        if abs(diff) < best_dist:
            if diff <= 0:
                _search(node["right"], depth + 1)
            else:
                _search(node["left"], depth + 1)

    _search(tree, 0)
    return best_idx, best_dist


# ---------------------------------------------------------------------------
# POI → 区域匹配
# ---------------------------------------------------------------------------

def assign_pois_to_regions(pois: list[dict], clustered_csv: Path,
                           radius_m: float = 200) -> pd.DataFrame:
    """
    将 POI 匹配到最近的 STRAT 区域。
    对每个 POI，在 STRAT 聚类的停留点中找最近点，若在半径内则继承其 region_id。
    """
    traj = pd.read_csv(clustered_csv, encoding="utf-8")
    if len(traj) == 0:
        return pd.DataFrame()

    # 只使用停留点（聚类的依据）
    if "is_stop" in traj.columns:
        stay = traj[traj["is_stop"] == 1].copy()
    else:
        stay = traj.copy()

    if len(stay) == 0:
        return pd.DataFrame()

    # 采样避免太慢
    if len(stay) > 100000:
        stay = stay.sample(n=100000, random_state=42)

    radius_deg = radius_m / 111000
    coords = stay[["经度", "纬度"]].values.astype(np.float64)
    region_ids = stay["region_id"].values
    tree, pts = build_kdtree(coords)

    rows = []
    for poi in pois:
        q = np.array([poi["lon"], poi["lat"]], dtype=np.float64)
        idx, dist = knn_search(tree, coords, q)
        if idx is not None and dist <= radius_deg:
            rid = int(region_ids[idx])
        else:
            rid = -1
        rows.append({
            "scenery": poi.get("scenery", ""),
            "poi_name": poi.get("name", ""),
            "type_code": poi.get("type_code", ""),
            "type_name": poi.get("type_name", ""),
            "type_prefix": poi.get("type_code", "")[:2],
            "region_id": rid,
            "dist_m": dist * 111000 if idx is not None else -1,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 统计 + 图表
# ---------------------------------------------------------------------------

def enrich_scenery(name: str, poi_df_all: pd.DataFrame,
                   radius_m: float, dpi: int):
    """对单个景区做 POI Enrichment。"""
    csv_path = CLUSTERED_DIR / f"{name}_clustered.csv"
    if not csv_path.exists():
        print(f"  [跳过] {name}: 无聚类结果")
        return None

    scene_pois = poi_df_all[poi_df_all["scenery"] == name].to_dict("records")
    if not scene_pois:
        print(f"  [跳过] {name}: 无 POI 数据")
        return None

    print(f"\n[{name}]")

    # POI → 区域匹配
    df = assign_pois_to_regions(scene_pois, csv_path, radius_m=radius_m)
    if len(df) == 0:
        print(f"  无 POI 匹配到任何区域")
        return None

    # 过滤未匹配的
    df_valid = df[df["region_id"] >= 0].copy()
    n_matched = len(df_valid)
    n_total = len(df)
    print(f"  POI 匹配: {n_matched}/{n_total} ({n_matched/n_total*100:.1f}%)")

    if n_matched < 5:
        print(f"  匹配 POI 太少，跳过")
        return None

    # 每个区域的 POI 类型分布
    ct = pd.crosstab(df_valid["region_id"], df_valid["type_prefix"])
    # 重命名列
    ct.columns = [TYPE_NAMES.get(c, c) for c in ct.columns]

    # 归一化为百分比
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # 主导类型
    dominant = ct_pct.idxmax(axis=1)
    dominant_pct = ct_pct.max(axis=1)

    # 语义标签
    def make_label(dom, pct, n_poi):
        if pct > 60:
            return f"核心{dom}区"
        elif pct > 40:
            return f"{dom}为主"
        elif pct > 25:
            return f"混合{dom}区"
        else:
            return f"综合服务区"

    labels = {}
    for rid in ct.index:
        labels[rid] = make_label(dominant[rid], dominant_pct[rid],
                                 ct.loc[rid].sum())

    # ---- 输出表格 ----
    tbl = ct.copy()
    tbl["POI总数"] = ct.sum(axis=1).astype(int)
    tbl["主导类型"] = [f"{dominant[r]} ({dominant_pct[r]:.0f}%)" for r in ct.index]
    tbl["语义标签"] = [labels[r] for r in ct.index]
    tbl = tbl.sort_index()

    print(f"\n  区域语义标注:")
    for rid in tbl.index:
        print(f"    Region {rid}: {tbl.loc[rid, '语义标签']} "
              f"(主导: {tbl.loc[rid, '主导类型']}, "
              f"{int(tbl.loc[rid, 'POI总数'])} POI)")

    out_csv = OUTPUT_DIR / f"{name}_enrich.csv"
    tbl.to_csv(out_csv, encoding="utf-8-sig")
    print(f"  表格: {out_csv}")

    # ---- 图1: 堆叠柱状图 ----
    fig, ax = plt.subplots(figsize=(max(6, len(ct) * 1.2), 5))
    bottom = np.zeros(len(ct))
    cols_sorted = sorted(ct.columns, key=lambda c: ct[c].sum(), reverse=True)
    x = np.arange(len(ct.index))

    for col in cols_sorted:
        vals = ct[col].values
        ax.bar(x, vals, bottom=bottom, label=col,
               color=TYPE_COLORS.get(col, "#95a5a6"), alpha=0.85,
               edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"R{int(r)}\n({labels[r]})" for r in ct.index],
                       fontsize=9)
    ax.set_xlabel("STRAT 区域", fontsize=11)
    ax.set_ylabel("POI 数量", fontsize=11)
    ax.set_title(f"{name} — 各区域 POI 类型构成", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path1 = OUTPUT_DIR / f"{name}_enrich_bar.png"
    fig.savefig(path1, dpi=dpi, bbox_inches="tight")
    print(f"  图1 (堆叠柱状图): {path1}")
    plt.close(fig)

    # ---- 图2: 百分比堆叠图 ----
    fig2, ax2 = plt.subplots(figsize=(max(6, len(ct) * 1.2), 5))
    bottom2 = np.zeros(len(ct_pct))
    for col in cols_sorted:
        if col in ct_pct.columns:
            vals = ct_pct[col].values
            ax2.bar(x, vals, bottom=bottom2, label=col,
                    color=TYPE_COLORS.get(col, "#95a5a6"), alpha=0.85,
                    edgecolor="white", linewidth=0.5)
            bottom2 += vals

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"R{int(r)}\n({labels[r]})" for r in ct.index],
                        fontsize=9)
    ax2.set_xlabel("STRAT 区域", fontsize=11)
    ax2.set_ylabel("POI 类型占比 (%)", fontsize=11)
    ax2.set_title(f"{name} — 各区域 POI 类型占比", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=7, ncol=3, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 105)

    fig2.tight_layout()
    path2 = OUTPUT_DIR / f"{name}_enrich_pct.png"
    fig2.savefig(path2, dpi=dpi, bbox_inches="tight")
    print(f"  图2 (百分比堆叠): {path2}")
    plt.close(fig2)

    # ---- 图3: 区域主导类型地图 ----
    fig3, ax3 = plt.subplots(figsize=(9, 7))
    traj = pd.read_csv(csv_path, encoding="utf-8")
    if len(traj) > 30000:
        traj_plot = traj.sample(30000, random_state=42)
    else:
        traj_plot = traj

    # 背景轨迹点灰色
    ax3.scatter(traj_plot["经度"], traj_plot["纬度"], c="lightgray",
                s=0.5, alpha=0.3, rasterized=True, label="轨迹")

    # 加载原始 POI 坐标用于绘图
    with open(POI_PATH, "r", encoding="utf-8") as f:
        all_pois = json.load(f)
    scene_pois_orig = {p["name"]: p for p in all_pois if p.get("scenery") == name}

    # POI 按主导类型着色
    for dom_type in set(dominant.values):
        color = TYPE_COLORS.get(dom_type, "#95a5a6")
        rid_sub = dominant[dominant == dom_type].index
        sub = df_valid[df_valid["region_id"].isin(rid_sub)]
        lngs, lats = [], []
        for _, row in sub.iterrows():
            orig = scene_pois_orig.get(row["poi_name"])
            if orig:
                lngs.append(orig["lon"])
                lats.append(orig["lat"])
        if lngs:
            ax3.scatter(lngs, lats, c=color, s=25, alpha=0.8,
                        edgecolors="white", linewidth=0.3, label=dom_type,
                        zorder=5)

    ax3.set_xlabel("经度", fontsize=10)
    ax3.set_ylabel("纬度", fontsize=10)
    ax3.set_title(f"{name} — POI 语义标注地图", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=8, loc="best", title="POI 类型")
    ax3.set_aspect(1.0 / np.cos(np.mean(traj["纬度"]) * np.pi / 180))

    fig3.tight_layout()
    path3 = OUTPUT_DIR / f"{name}_enrich_map.png"
    fig3.savefig(path3, dpi=dpi, bbox_inches="tight")
    print(f"  图3 (标注地图): {path3}")
    plt.close(fig3)

    return {
        "name": name,
        "n_poi_matched": n_matched,
        "n_poi_total": n_total,
        "n_regions": len(ct),
        "table": tbl,
        "labels": labels,
        "dominant": dominant.to_dict(),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def generate_summary_report(results: list[dict], output_path: Path):
    """生成汇总报告 Markdown。"""
    lines = [
        "# POI Enrichment — STRAT 聚簇语义标注报告",
        "",
        "对 STRAT 每个聚簇，统计其内部 POI 类型构成，标注功能语义。",
        "",
        "## 各景区结果",
        "",
    ]

    for r in results:
        if r is None:
            continue
        lines.append(f"### {r['name']}")
        lines.append(f"- 匹配 POI: {r['n_poi_matched']}/{r['n_poi_total']}")
        lines.append(f"- 区域数: {r['n_regions']}")
        lines.append("")
        lines.append(f"| 区域 | 语义标签 | POI 数 | 主导类型 |")
        lines.append(f"|------|---------|-------|---------|")
        tbl = r["table"]
        for rid in tbl.index:
            lines.append(
                f"| R{int(rid)} | {tbl.loc[rid, '语义标签']} "
                f"| {int(tbl.loc[rid, 'POI总数'])} "
                f"| {tbl.loc[rid, '主导类型']} |"
            )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n汇总报告: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="POI Enrichment 语义标注")
    parser.add_argument("--scenery", nargs="*",
                        help="景区名，默认处理所有有 STRAT 结果的景区")
    parser.add_argument("--buffer-radius", type=float, default=200,
                        help="POI 匹配聚类区域的距离阈值（米），默认 200")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--output", default="enrich_summary.md")
    args = parser.parse_args()

    # 加载 POI
    with open(POI_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    poi_df = pd.DataFrame(raw)

    # 确定景区
    if args.scenery:
        scenery_list = args.scenery
    else:
        csv_names = {p.stem.replace("_clustered", "")
                     for p in CLUSTERED_DIR.glob("*_clustered.csv")}
        poi_names = set(poi_df["scenery"].unique())
        scenery_list = sorted(csv_names & poi_names)

    print("=" * 50)
    print(f"POI Enrichment 语义标注")
    print(f"匹配半径: {args.buffer_radius}m")
    print(f"景区: {len(scenery_list)} 个")
    print("=" * 50)

    results = []
    for name in scenery_list:
        r = enrich_scenery(name, poi_df, args.buffer_radius, args.dpi)
        if r is not None:
            results.append(r)

    # 汇总
    report_path = OUTPUT_DIR / args.output
    generate_summary_report(results, report_path)

    print(f"\n所有结果已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
