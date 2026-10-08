#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量获取景区内部 POI 坐标（基于高德地图 API）。

通过高德 POI 搜索接口，对指定景区列表查询其内部兴趣点（景点、餐饮、
卫生间、出入口等），导出为 CSV/JSON，供后续分析使用。

用法：
    # 设置环境变量
    export AMAP_KEY="你的高德Key"

    # 运行（使用默认景区列表）
    python fetch_scenery_poi.py

    # 指定城市和输出格式
    python fetch_scenery_poi.py --city 成都 --output poi_result.json

    # 只查特定类型
    python fetch_scenery_poi.py --types 110000,050000
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Optional

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = SCRIPT_DIR / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 默认景区列表（与项目中已有景区数据对应）
# ---------------------------------------------------------------------------
DEFAULT_SCENERIES = [
    {"name": "峨眉山", "city": "乐山"},
    {"name": "武侯祠博物馆", "city": "成都"},
    {"name": "熊猫基地", "city": "成都"},
    {"name": "都江堰", "city": "成都"},
    {"name": "锦江", "city": "成都"},
    {"name": "青城山", "city": "成都"},
    {"name": "龙泉", "city": "成都"},
    {"name": "虹口", "city": "成都"},
    {"name": "黄龙溪", "city": "成都"},
    {"name": "锦里", "city": "成都"},
]

# 高德 POI 类型码（typecode）
# 完整列表：https://lbs.amap.com/api/webservice/download
DEFAULT_TYPES = "110000|050000|060000|080000|100000|140000|170000|200000"
TYPE_EXPLANATIONS = {
    "110000": "风景名胜",
    "050000": "餐饮服务",
    "060000": "购物服务",
    "080000": "住宿服务",
    "100000": "汽车服务",
    "140000": "公共设施（卫生间等）",
    "170000": "出入口",
    "200000": "事件活动",
}

# ---------------------------------------------------------------------------
# 高德 API 封装
# ---------------------------------------------------------------------------


def amap_text_search(
    keywords: str,
    key: str,
    city: str = "",
    types: str = "",
    citylimit: bool = False,
    offset: int = 25,
    page: int = 1,
    extensions: str = "all",
) -> dict:
    """
    调用高德 POI 关键字搜索接口。
    https://lbs.amap.com/api/webservice/guide/api/search
    """
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "keywords": keywords,
        "types": types,
        "city": city,
        "citylimit": "true" if citylimit else "false",
        "offset": offset,
        "page": page,
        "extensions": extensions,
        "key": key,
        "output": "json",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def amap_around_search(
    location: str,
    key: str,
    keywords: str = "",
    types: str = "",
    radius: int = 3000,
    offset: int = 25,
    page: int = 1,
    extensions: str = "all",
) -> dict:
    """
    调用高德 POI 周边搜索接口。
    location 格式："经度,纬度"
    """
    url = "https://restapi.amap.com/v3/place/around"
    params = {
        "location": location,
        "keywords": keywords,
        "types": types,
        "radius": radius,
        "offset": offset,
        "page": page,
        "extensions": extensions,
        "key": key,
        "output": "json",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def amap_polygon_search(
    polygon: str,
    key: str,
    keywords: str = "",
    types: str = "",
    offset: int = 25,
    page: int = 1,
    extensions: str = "all",
) -> dict:
    """
    调用高德 POI 多边形搜索接口。
    polygon 格式："经度1,纬度1|经度2,纬度2|..." （多边形顶点，顺时针/逆时针）
    """
    url = "https://restapi.amap.com/v3/place/polygon"
    params = {
        "polygon": polygon,
        "keywords": keywords,
        "types": types,
        "offset": offset,
        "page": page,
        "extensions": extensions,
        "key": key,
        "output": "json",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_all_pages(
    search_fn,
    max_pages: int = 20,
    delay: float = 0.3,
    **search_kwargs,
) -> list[dict]:
    """
    通用分页获取：循环翻页直到无更多结果，合并所有 POI。
    search_fn 为 amap_text_search / amap_around_search / amap_polygon_search。
    """
    all_pois: list[dict] = []
    for page in range(1, max_pages + 1):
        result = search_fn(page=page, **search_kwargs)
        if result.get("status") != "1":
            print(f"    API 返回错误 (page={page}): {result.get('info', 'unknown')}")
            break
        pois = result.get("pois", [])
        if not pois:
            break
        all_pois.extend(pois)
        # 判断是否还有下一页
        total = int(result.get("count", "0"))
        if page * search_kwargs.get("offset", 25) >= total:
            break
        time.sleep(delay)
    return all_pois


# ---------------------------------------------------------------------------
# 数据提取 & 去重
# ---------------------------------------------------------------------------


def extract_poi_fields(poi: dict, scenery_name: str) -> dict:
    """从高德返回的单个 POI 对象中提取关键字段。"""
    location = poi.get("location", "0,0")
    lon_str, lat_str = location.split(",")
    return {
        "scenery": scenery_name,
        "poi_id": poi.get("id", ""),
        "name": poi.get("name", ""),
        "type_code": poi.get("typecode", ""),
        "type_name": poi.get("type", ""),
        "lon": float(lon_str),
        "lat": float(lat_str),
        "address": poi.get("address", ""),
        "city": poi.get("cityname", "") or poi.get("pname", ""),
        "adcode": poi.get("adcode", ""),
        "pname": poi.get("pname", ""),
        "biz_ext": json.dumps(poi.get("biz_ext", {}), ensure_ascii=False)
        if poi.get("biz_ext")
        else "",
        "photos": json.dumps(
            [p.get("url", "") for p in poi.get("photos", [])], ensure_ascii=False
        )
        if poi.get("photos")
        else "",
    }


def dedup_and_enrich(poi_list: list[dict]) -> list[dict]:
    """按 poi_id 去重，同时统计每个景区的 POI 数量。"""
    seen = set()
    unique = []
    for p in poi_list:
        pid = p["poi_id"]
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)
        elif not pid:
            # 无 id 的也保留（非标准 POI）
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def simplify_polygon(
    coords: list[list[float]], max_vertices: int = 30
) -> list[list[float]]:
    """
    使用 Ramer-Douglas-Peucker 算法简化多边形顶点数，
    确保不超过 max_vertices 个点（高德 polygon 搜索限制）。
    """
    if len(coords) <= max_vertices:
        return coords

    # 使用 Douglas-Peucker 简化
    def perpendicular_distance(point, line_start, line_end):
        """点到线段的垂直距离"""
        if line_start == line_end:
            return ((point[0] - line_start[0]) ** 2 + (point[1] - line_start[1]) ** 2) ** 0.5
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        numerator = abs(
            dy * point[0] - dx * point[1] + line_end[0] * line_start[1] - line_end[1] * line_start[0]
        )
        denominator = (dx * dx + dy * dy) ** 0.5
        return numerator / denominator

    def douglas_peucker(points, tolerance):
        """递归简化"""
        if len(points) <= 2:
            return points
        dmax = 0.0
        index = 0
        for i in range(1, len(points) - 1):
            d = perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                dmax = d
                index = i
        if dmax > tolerance:
            left = douglas_peucker(points[: index + 1], tolerance)
            right = douglas_peucker(points[index:], tolerance)
            return left[:-1] + right
        return [points[0], points[-1]]

    # 二分搜索合适的 tolerance
    lo, hi = 0.0, 0.1
    simplified = coords
    while True:
        mid = (lo + hi) / 2
        simplified = douglas_peucker(coords, mid)
        if len(simplified) <= max_vertices:
            hi = mid
            break
        lo = mid
        if hi - lo < 1e-8:
            break
    # 用找到的 tolerance 再算一次
    simplified = douglas_peucker(coords, hi)
    # 如果还是太多，均匀采样
    if len(simplified) > max_vertices:
        indices = set()
        step = len(coords) / max_vertices
        for i in range(max_vertices):
            indices.add(int(i * step) % len(coords))
        # 保证首尾点
        indices.add(0)
        simplified = [coords[i] for i in sorted(indices)]

    print(f"    多边形简化: {len(coords)} 顶点 → {len(simplified)} 顶点")
    return simplified


def load_boundary_for_scenery(scenery_name: str) -> Optional[str]:
    """
    从 data/datajson/ 目录读取景区 GeoJSON 边界，
    简化后返回适合高德 polygon API 的字符串格式。
    """
    json_path = SCRIPT_DIR / "data" / "datajson" / f"{scenery_name}.json"
    if not json_path.exists():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    features = data.get("features", [])
    if not features:
        return None

    geom = features[0].get("geometry", {})
    coords = None
    if geom.get("type") == "Polygon":
        coords = geom["coordinates"][0]  # 外环
    elif geom.get("type") == "MultiPolygon":
        coords = geom["coordinates"][0][0]

    if not coords:
        return None

    # 简化顶点数（高德 polygon URL 长度限制约 4KB）
    coords = simplify_polygon(coords, max_vertices=30)

    # 格式：lon1,lat1|lon2,lat2|...
    polygon_str = "|".join(f"{p[0]},{p[1]}" for p in coords)
    return polygon_str


def fetch_scenery_pois(
    scenery_cfg: dict,
    api_key: str,
    types: str,
    use_polygon: bool = True,
    delay: float = 0.3,
) -> list[dict]:
    """
    获取单个景区内部 POI。
    优先使用多边形搜索（需要有 GeoJSON 边界），
    其次使用关键字搜索。
    """
    name = scenery_cfg["name"]
    city = scenery_cfg.get("city", "")
    pois = []

    # 方式1：多边形搜索
    if use_polygon:
        polygon = load_boundary_for_scenery(name)
        if polygon:
            print(f"  [多边形搜索] {name}")
            try:
                pois = fetch_all_pages(
                    amap_polygon_search,
                    polygon=polygon,
                    key=api_key,
                    keywords=name,
                    types=types,
                    delay=delay,
                )
            except requests.exceptions.RequestException as e:
                print(f"    多边形搜索失败: {e}，将降级为关键字搜索")
                pois = []
            if pois:
                print(f"    获取到 {len(pois)} 个 POI")
                return [extract_poi_fields(p, name) for p in pois]

    # 方式2：关键字搜索（带 citylimit 约束城市范围）
    print(f"  [关键字搜索] {name}（城市: {city}）")
    try:
        pois = fetch_all_pages(
            amap_text_search,
            keywords=name,
            key=api_key,
            city=city,
            types=types,
            citylimit=True,
            delay=delay,
        )
    except requests.exceptions.RequestException as e:
        print(f"    关键字搜索失败: {e}")
        return []
    print(f"    获取到 {len(pois)} 个 POI")
    return [extract_poi_fields(p, name) for p in pois]


def main():
    parser = argparse.ArgumentParser(
        description="批量获取景区内部 POI 坐标（高德地图 API）"
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("AMAP_KEY", "633d709003854212255bf02a46617b34"),
        help="高德 Web 服务 API Key（也可通过环境变量 AMAP_KEY 设置）",
    )
    parser.add_argument(
        "--city",
        default="",
        help="默认城市（当景区配置未指定 city 时使用）",
    )
    parser.add_argument(
        "--types",
        default=os.environ.get("AMAP_TYPES", DEFAULT_TYPES),
        help="POI 类型码，竖线分隔。默认: 风景名胜|餐饮|购物|住宿|汽车|公共设施|出入口|活动",
    )
    parser.add_argument(
        "--output",
        default="poi_result.json",
        help="输出文件名（.csv 或 .json），默认 poi_result.json",
    )
    parser.add_argument(
        "--use-polygon",
        action="store_true",
        default=True,
        help="优先使用多边形边界搜索（需 data/datajson/*.json）",
    )
    parser.add_argument(
        "--no-polygon",
        action="store_false",
        dest="use_polygon",
        help="禁用多边形搜索，仅使用关键字搜索",
    )
    parser.add_argument(
        "--scenery",
        nargs="*",
        help="指定景区名称（多个用空格分隔），留空则使用默认列表",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="API 请求间隔（秒），默认 0.3",
    )
    args = parser.parse_args()

    # ---- 验证 API Key ----
    api_key = args.key.strip()
    if not api_key:
        print("[错误] 请设置高德 API Key:")
        print("  export AMAP_KEY='你的Key'")
        print("  或通过 --key 参数传入")
        sys.exit(1)

    # ---- 构建景区列表 ----
    if args.scenery:
        sceneries = [
            {"name": s, "city": args.city or "成都"} for s in args.scenery
        ]
    else:
        sceneries = DEFAULT_SCENERIES
        if args.city:
            for s in sceneries:
                s["city"] = args.city

    print(f"=" * 56)
    print(f"景区 POI 批量获取")
    print(f"景区数量: {len(sceneries)}")
    print(f"高德 Key: {api_key[:8]}...{api_key[-4:]}")
    default_city = args.city or "（各景区独立配置）"
    print(f"默认城市: {default_city}")
    print(f"搜索类型: {args.types}")
    print(f"搜索方式: {'多边形优先' if args.use_polygon else '仅关键字'}")
    print(f"=" * 56)

    # ---- 逐景区获取 ----
    all_pois: list[dict] = []
    stats: dict[str, int] = {}

    for i, cfg in enumerate(sceneries, 1):
        name = cfg["name"]
        print(f"\n[{i}/{len(sceneries)}] {name}")
        try:
            poi_records = fetch_scenery_pois(
                cfg,
                api_key,
                types=args.types,
                use_polygon=args.use_polygon,
                delay=args.delay,
            )
        except Exception as e:
            print(f"    ⚠ 获取失败: {e}")
            poi_records = []

        if poi_records:
            all_pois.extend(poi_records)
            # 按类型统计
            type_counts: dict[str, int] = {}
            for p in poi_records:
                tc = p["type_code"][:2] + "0000"
                type_counts[tc] = type_counts.get(tc, 0) + 1
            stats[name] = len(poi_records)
            summary = ", ".join(
                f"{TYPE_EXPLANATIONS.get(k, k)}: {v}" for k, v in type_counts.items()
            )
            print(f"    ✓ 共 {len(poi_records)} 个 POI ({summary})")
        else:
            print(f"    ✗ 未获取到 POI")

        # 请求间隔（避免 QPS 限制）
        if i < len(sceneries):
            time.sleep(args.delay * 2)

    # ---- 去重 ----
    all_pois = dedup_and_enrich(all_pois)

    # ---- 输出 ----
    output_path = OUTPUT_DIR / args.output
    if not all_pois:
        print(f"\n[结果] 未获取到任何 POI，请检查 API Key 或网络。")
        sys.exit(1)

    if output_path.suffix == ".csv":
        df = pd.DataFrame(all_pois)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_pois, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 56}")
    print(f"完成！共获取 {len(all_pois)} 个去重 POI")
    print(f"输出文件: {output_path}")
    print(f"\n各景区 POI 数量:")
    for name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    main()
