#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键清洗所有景区的脚本。

自动从 datajson/ 目录读取各景区的边界配置，
然后依次对 data/ 中对应的 CSV 执行完整清洗流水线。

用法：
    python clean_all.py
    python clean_all.py --dbscan_eps 0.0003
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8") # type: ignore

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_JSON_DIR = SCRIPT_DIR / "data" / "datajson"
DATA_CSV_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "cleaned_labeled_data"
QGIS_DIR = SCRIPT_DIR / "qgis_data"

sys.path.insert(0, str(SCRIPT_DIR))
from clean_scenery_pipeline import process_scenery


def load_boundary_from_json(json_path):
    """
    从 GeoJSON 文件中读取景区边界多边形。
    支持 Polygon 和 LineString（将 LineString 首尾相连视为闭合多边形）。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return None, None
        data = json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        print(f"    JSON 解析失败: {e}")
        return None, None

    features = data.get("features", [])
    if not features:
        return None, None

    feat = features[0]
    geom = feat.get("geometry", {})
    geom_type = geom.get("type", "")
    coords = geom.get("coordinates", [])

    if geom_type == "Polygon":
        ring = coords[0] if coords else []
        boundary = [(c[0], c[1]) for c in ring]
    elif geom_type == "LineString":
        ring = coords if coords else []
        boundary = [(c[0], c[1]) for c in ring]
        if len(boundary) > 1 and boundary[0] != boundary[-1]:
            boundary.append(boundary[0])
    else:
        return None, None

    props = feat.get("properties", {})
    name = props.get("name", "").strip()
    if not name:
        name = json_path.stem

    return boundary, name


def find_matching_csv(json_name, csv_dir):
    """
    根据 JSON 文件名查找对应的 CSV 文件。
    精确匹配优先，其次模糊匹配（去掉"博物馆"等后缀）。
    """
    json_base = json_name.replace("博物馆", "").strip()

    exact = csv_dir / f"{json_name}.csv"
    if exact.exists():
        return exact

    base = csv_dir / f"{json_base}.csv"
    if base.exists():
        return base

    for csv_file in csv_dir.glob("*.csv"):
        csv_base = csv_file.stem.replace("博物馆", "").strip()
        if csv_base == json_base:
            return csv_file

    return None


def clean_all(
    dbscan_eps=0.001,
    dbscan_min=30,
    min_region_size=100,
    route_eps=1.5,
    route_min_samples=2,
    max_jump_meters=100,
    min_track_length=300,
    stay_speed=1.0,
    stay_duration=30,
    export_geojson=False,
    visualize=True,
):
    """
    清洗 datajson/ 下所有景区。
    """
    print("=" * 60)
    print("一键清洗所有景区")
    print("=" * 60)
    print(f"JSON 目录: {DATA_JSON_DIR}")
    print(f"CSV 目录:  {DATA_CSV_DIR}")
    print(f"输出目录:  {OUTPUT_DIR}")
    print(f"可视化:  {visualize}")
    print()

    json_files = sorted(DATA_JSON_DIR.glob("*.json"))
    if not json_files:
        print(f"未找到 JSON 文件: {DATA_JSON_DIR}")
        return

    print(f"找到 {len(json_files)} 个 JSON 文件:")
    for jf in json_files:
        print(f"  - {jf.name}")
    print()

    results = {}

    for json_path in json_files:
        json_name = json_path.stem
        boundary, scenery_name = load_boundary_from_json(json_path)

        if not boundary or len(boundary) < 3:
            print(f"[跳过] {json_name}: 无法解析边界")
            results[json_name] = "跳过(无边界)"
            continue

        csv_path = find_matching_csv(json_name, DATA_CSV_DIR)
        if not csv_path:
            print(f"[跳过] {json_name}: 未找到对应 CSV")
            results[json_name] = "跳过(无CSV)"
            continue

        print(f"\n{'='*60}")
        print(f"处理: {scenery_name} ({json_name})")
        print(f"  CSV:  {csv_path.name}")
        print(f"  边界: {len(boundary)} 个顶点")
        print(f"{'='*60}")

        try:
            process_scenery(
                scenery_name=scenery_name,
                csv_path=str(csv_path),
                custom_boundary=boundary,
                max_jump_meters=max_jump_meters,
                min_track_length=min_track_length,
                stay_speed_thresh=stay_speed,
                stay_duration_thresh=stay_duration,
                dbscan_eps=dbscan_eps,
                dbscan_min_samples=dbscan_min,
                min_region_size=min_region_size,
                route_eps=route_eps,
                route_min_samples=route_min_samples,
                export_geojson=export_geojson,
                visualize=visualize,
            )
            results[json_name] = "成功"
        except Exception as e:
            print(f"[错误] {json_name}: {e}")
            import traceback

            traceback.print_exc()
            results[json_name] = f"失败: {e}"

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    success = sum(1 for v in results.values() if v == "成功")
    total = len(results)
    for name, status in sorted(results.items()):
        icon = "✓" if status == "成功" else "✗"
        print(f"  {icon} {name}: {status}")
    print(f"\n成功: {success}/{total}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键清洗所有景区")
    parser.add_argument("--dbscan_eps", type=float, default=0.001, help="DBSCAN eps (默认: 0.001)")
    parser.add_argument("--dbscan_min", type=int, default=30, help="DBSCAN min_samples (默认: 30)")
    parser.add_argument("--min_region_size", type=int, default=100, help="最小区域点数，小于此值合并到最近大簇 (默认: 100)")
    parser.add_argument("--route_eps", type=float, default=1.5, help="路线DBSCAN eps (相对比例, 默认: 1.5)")
    parser.add_argument("--route_min", type=int, default=2, help="路线DBSCAN min_samples (默认: 2)")
    parser.add_argument("--max_jump", type=float, default=100, help="飞点位移阈值 米 (默认: 100)")
    parser.add_argument("--min_track_length", type=int, default=300, help="轨迹最小点数 (默认: 300)")
    parser.add_argument("--stay_speed", type=float, default=1.0, help="停留速度阈值 km/h (默认: 1.0)")
    parser.add_argument("--stay_duration", type=float, default=30, help="停留最短持续时间 秒 (默认: 30)")
    parser.add_argument("--export_geojson", action="store_true", default=False, help="导出 GeoJSON 供 QGIS")
    parser.add_argument("--visualize", action="store_true", default=True, help="生成可视化报告图表 (默认: 开启)")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false", help="关闭可视化报告图表")
    parser.add_argument("--disable-visualize", action="store_true", help="关闭可视化报告图表")

    args = parser.parse_args()
    
    # 处理 --disable-visualize 参数
    if args.disable_visualize:
        visualize = False
    else:
        visualize = args.visualize

    clean_all(
        dbscan_eps=args.dbscan_eps,
        dbscan_min=args.dbscan_min,
        min_region_size=args.min_region_size,
        route_eps=args.route_eps,
        route_min_samples=args.route_min,
        max_jump_meters=args.max_jump,
        min_track_length=args.min_track_length,
        stay_speed=args.stay_speed,
        stay_duration=args.stay_duration,
        export_geojson=args.export_geojson,
        visualize=visualize,
    )
