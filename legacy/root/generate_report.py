#!/usr/bin/env python3
"""
STRAT 批量实验报告生成器。
读取聚类 + 预测结果，生成 Markdown 格式的综合报告。
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
BATCH_FILE = PROJECT_DIR / "batch_results.json"
PRED_DIR = PROJECT_DIR / "prediction" / "output"
REPORT_FILE = PROJECT_DIR / "实验报告.md"

PAPER_SCENES = {"青城山", "峨眉山", "武侯祠博物馆"}

CATEGORY = {
    "青城山": "山地",
    "峨眉山": "山地",
    "武侯祠博物馆": "园林",
    "熊猫基地": "园区",
    "都江堰": "水利景区",
    "锦江": "滨水",
    "龙泉": "城市/山地",
    "黄龙溪": "古镇",
}


def load_batch_results():
    if not BATCH_FILE.exists():
        return {}
    with open(BATCH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def generate():
    cluster_results = load_batch_results()

    print("正在生成实验报告...")

    # 统计
    total_tracks = sum(r.get("tracks", 0) for r in cluster_results.values())
    total_regions = sum(r.get("regions", 0) for r in cluster_results.values())
    total_routes = sum(r.get("routes", 0) for r in cluster_results.values())
    n_scenes = len(cluster_results)

    avg_sil = (
        sum(r.get("silhouette", 0) for r in cluster_results.values()) / n_scenes
        if n_scenes else 0
    )
    avg_dbi = (
        sum(r.get("davies_bouldin", 0) for r in cluster_results.values()) / n_scenes
        if n_scenes else 0
    )

    lines = []
    lines.append("# STRAT 批量实验结果报告")
    lines.append("")
    lines.append(f"**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**实验景区数**: {n_scenes} 个")
    lines.append("")

    # ── 概览 ──
    lines.append("## 1. 总体概览")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 景区总数 | {n_scenes} |")
    lines.append(f"| 总轨迹数 | {total_tracks:,} |")
    lines.append(f"| 总区域数 | {total_regions} |")
    lines.append(f"| 总路线数 | {total_routes} |")
    lines.append(f"| 平均轮廓系数 | {avg_sil:.4f} |")
    lines.append(f"| 平均 DBI | {avg_dbi:.4f} |")
    lines.append("")

    # ── 聚类结果 ──
    lines.append("## 2. 景区聚类结果")
    lines.append("")
    lines.append(
        "| 景区 | 类型 | 轨迹数 | 区域数 | 路线数 | 轮廓系数 | DBI |"
    )
    lines.append("|------|------|:------:|:------:|:------:|:--------:|:----:|")
    for name in sorted(cluster_results.keys()):
        r = cluster_results[name]
        cat = CATEGORY.get(name, "其他")
        mark = "⭐" if name in PAPER_SCENES else ""
        lines.append(
            f"| {mark}{name} | {cat} | {r.get('tracks','—'):,} | "
            f"{r.get('regions','—')} | {r.get('routes','—')} | "
            f"{r.get('silhouette','—'):.4f} | {r.get('davies_bouldin','—'):.4f} |"
        )
    lines.append("")
    lines.append("> ⭐ 标注为论文核心实验景区")
    lines.append("")

    # ── 论文景区 vs 论文报告对比 ──
    lines.append("## 3. 与论文结果对比")
    lines.append("")
    lines.append(
        "| 景区 | 指标 | 论文值 | 本次实验 | 变化 |"
    )
    lines.append("|------|------|:------:|:--------:|:----:|")
    paper_ref = {
        "青城山": {"tracks": 358, "silhouette": 0.417, "regions": 17},
        "峨眉山": {"tracks": 499, "silhouette": None, "regions": None},
        "武侯祠博物馆": {"tracks": 43, "silhouette": None, "regions": 8},
    }
    for name in sorted(PAPER_SCENES):
        r = cluster_results.get(name, {})
        ref = paper_ref.get(name, {})
        if not r:
            continue

        # 轨迹数对比
        t_old = ref.get("tracks", "—")
        t_new = r.get("tracks", "—")
        if isinstance(t_old, int) and isinstance(t_new, int):
            delta = f"+{t_new - t_old} ({100*(t_new-t_old)/t_old:+.0f}%)"
        else:
            delta = "—"
        lines.append(
            f"| {name} | 轨迹数 | {t_old} | {t_new:,} | {delta} |"
        )

        # 轮廓系数对比
        s_old = ref.get("silhouette")
        s_new = r.get("silhouette")
        if s_old is not None and s_new is not None:
            lines.append(
                f"| | 轮廓系数 | {s_old:.4f} | {s_new:.4f} | "
                f"{s_new-s_old:+.4f} |"
            )

        # 区域数对比
        k_old = ref.get("regions")
        k_new = r.get("regions")
        if k_old is not None and k_new is not None:
            lines.append(
                f"| | 区域数 | {k_old} | {k_new} | "
                f"{k_new-k_old:+d} |"
            )
    lines.append("")

    # ── 预测结果 ──
    lines.append("## 4. 预测结果（BiGRU 段时长）")
    lines.append("")

    # 从 train.py 输出中提取指标
    pred_metrics = {}
    for d in sorted(PRED_DIR.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        # train.py 只输出到 stdout，没有单独 metrics 文件。
        # 检查模型是否存在作为完成标志
        pkl = d / "route_stats.pkl"
        if pkl.exists():
            pred_metrics[name] = True
        else:
            pred_metrics[name] = False

    if pred_metrics:
        lines.append(
            "| 景区 | 模型状态 | 输出路径 |"
        )
        lines.append("|------|:--------:|------|")
        for name in sorted(pred_metrics.keys()):
            status = "✅ 已完成" if pred_metrics[name] else "❌ 失败"
            path = f"`prediction/output/{name}/`"
            lines.append(f"| {name} | {status} | {path} |")
        lines.append("")

    # ── 数据链路概览 ──
    lines.append("## 5. 数据管道")
    lines.append("")
    lines.append("```")
    lines.append("原始 CSV (data/)")
    lines.append("  │  clean_all.py (边界过滤 → 飞点过滤 → 短轨迹删除 → DBSCAN)")
    lines.append("  ▼")
    lines.append("清洗后 CSV (cleaned_labeled_data/*_cleaned.csv)")
    lines.append("  │  generate_clustered_csvs.py (粒球生成 → BKOA 谱聚类 → 路线发现)")
    lines.append("  ▼")
    lines.append("聚类后 CSV (cluster/output/*_strat.csv)")
    lines.append("  │  run_prediction.py → train.py (BiGRU 段时长预测)")
    lines.append("  ▼")
    lines.append("模型文件 (prediction/output/{景区}/)")
    lines.append("```")
    lines.append("")

    # ── 参数 ──
    lines.append("## 6. 实验参数")
    lines.append("")
    lines.append("| 阶段 | 参数 | 值 |")
    lines.append("|------|------|-----|")
    lines.append("| 数据清洗 | 飞点阈值 | 500m |")
    lines.append("| | 最小轨迹长度 | 50 点 |")
    lines.append("| | DBSCAN eps | 0.001 |")
    lines.append("| | DBSCAN min_samples | 10 |")
    lines.append("| | 停留速度阈值 | 2.0 km/h |")
    lines.append("| 聚类 | 粒球采样 | 5000 点 |")
    lines.append("| | 特征维度 | 3D (lon, lat, elev) |")
    lines.append("| | BKOA 种群 | 10 |")
    lines.append("| | BKOA 迭代 | 20 |")
    lines.append("| | k 选择 | eigen-gap 启发 |")
    lines.append("| 预测 | 模型 | BiGRU |")
    lines.append("| | hidden_dim | 64 |")
    lines.append("| | embed_dim | 32 |")
    lines.append("| | 段特征维度 | 15 (GPS) |")
    lines.append("| | 训练轮数 | 100 |")
    lines.append("| | 优化器 | AdamW |")
    lines.append("")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"报告已生成: {REPORT_FILE}")
    print(f"  {n_scenes} 个景区, {total_tracks:,} 条轨迹")
    print(f"  平均轮廓系数: {avg_sil:.4f}")

    # 也输出到终端
    print("\n" + "=" * 60)
    print("\n".join(lines[:80]))
    print("\n...(完整报告见文件)")


if __name__ == "__main__":
    generate()
