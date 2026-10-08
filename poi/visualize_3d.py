#!/usr/bin/env python3
"""
3D 地图：路径（GPS轨迹）+ POI 点 + 海拔。

生成交互式 3D 散点图，用鼠标可旋转查看。
路径按 region_id 着色，POI 点用不同形状/颜色标记。

用法:
    python poi_3d_map.py --scenery 峨眉山
    python poi_3d_map.py --scenery 都江堰 --max_path_points 5000
    python poi_3d_map.py --all              # 全部景区
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ── 中文字体设置 ──
_CN_FONT = None
for _fname in ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
               'Noto Serif CJK SC', 'AR PL UKai CN', 'SimHei', 'DejaVu Sans']:
    try:
        _prop = fm.FontProperties(family=_fname)
        _test_fname = _prop.get_name()
        if _test_fname != 'sans-serif' or _fname == 'DejaVu Sans':
            _CN_FONT = _fname
            break
    except Exception:
        continue

if _CN_FONT:
    plt.rcParams['font.family'] = _CN_FONT
    plt.rcParams['axes.unicode_minus'] = False
    # print(f"字体: {_CN_FONT}")
else:
    print("[WARN] 未找到中文字体，中文可能显示为方框")

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3d
except ImportError:
    print("需要 mpl_toolkits.mplot3d，请安装 matplotlib")
    sys.exit(1)


# ── 路径配置 ───────────────────────────────────────────────
CLUSTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cluster', 'output'))
POI_PROJECTED_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'projected', 'poi_path_projected.csv'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'figures'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENERY_CSV_MAP = {
    "峨眉山": "峨眉山", "都江堰": "都江堰", "青城山": "青城山",
    "龙泉": "龙泉", "武侯祠博物馆": "武侯祠博物馆",
    "熊猫基地": "熊猫基地", "黄龙溪": "黄龙溪",
    "锦里": "锦里", "虹口": "虹口",
}


def load_path_data(scenery_name, max_points=20000):
    """加载聚类后的 GPS 路径数据（采样）。"""
    csv_name = SCENERY_CSV_MAP.get(scenery_name, scenery_name)
    csv_path = os.path.join(CLUSTER_DIR, f"{csv_name}_clustered.csv")

    if not os.path.exists(csv_path):
        csv_path = os.path.join(CLUSTER_DIR, f"{scenery_name}_clustered.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] 未找到: {csv_path}")
        return None

    print(f"  加载: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # 采样
    n_total = len(df)
    if n_total > max_points:
        idx = np.random.choice(n_total, max_points, replace=False)
        df = df.iloc[idx].copy()
        print(f"  采样: {max_points}/{n_total:,}")

    # 检查必要列
    alt_col = '海拔'
    if alt_col not in df.columns:
        print(f"  [WARN] 无海拔列，用 0 填充")
        df[alt_col] = 0.0

    print(f"  区域数: {df['region_id'].nunique()}")
    return df


def load_poi_data(scenery_name):
    """加载 POI 路径投影数据。"""
    if not os.path.exists(POI_PROJECTED_CSV):
        print(f"  [SKIP] 无 POI 投影数据: {POI_PROJECTED_CSV}")
        return None

    df = pd.read_csv(POI_PROJECTED_CSV)
    df_poi = df[df['scenery'] == scenery_name].copy()

    if len(df_poi) == 0:
        print(f"  [SKIP] 无该景区 POI")
        return None

    # 用投影坐标画图
    df_poi['plot_lon'] = df_poi['projected_lon']
    df_poi['plot_lat'] = df_poi['projected_lat']

    # 为 POI 添加海拔：从路径数据中找对应海拔
    print(f"  POI 投影点: {len(df_poi)}")
    return df_poi


def enrich_poi_elevation(df_poi, path_df):
    """从路径数据中为 POI 点补充海拔（用最近路径点的海拔）。"""
    from scipy.spatial import KDTree

    path_lons = path_df['经度'].values.astype(np.float64)
    path_lats = path_df['纬度'].values.astype(np.float64)
    path_alt = path_df['海拔'].values.astype(np.float64)

    path_points = np.column_stack([path_lons, path_lats])
    tree = KDTree(path_points)

    elevs = []
    for _, row in df_poi.iterrows():
        _, idx = tree.query([[row['plot_lon'], row['plot_lat']]], k=1)
        elevs.append(float(path_alt[idx[0]]))
    df_poi['plot_alt'] = elevs
    return df_poi


def plot_3d(scenery_name, path_df, poi_df, max_tracks=50):
    """生成 3D 地图。"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # ── 1. 路径点 ──
    n_regions = path_df['region_id'].nunique()
    cmap = plt.colormaps['tab10' if n_regions <= 10 else 'tab20']
    region_ids = sorted(path_df['region_id'].unique())
    region_color_map = {rid: cmap(i / max(n_regions, 1)) for i, rid in enumerate(region_ids)}

    # 采样轨迹显示
    track_ids = path_df['trackId'].unique()
    if len(track_ids) > max_tracks:
        track_ids = np.random.choice(track_ids, max_tracks, replace=False)
        print(f"  显示轨迹: {max_tracks}/{len(path_df['trackId'].unique())}")

    track_df = path_df[path_df['trackId'].isin(track_ids)].copy()
    track_region_colors = track_df['region_id'].map(region_color_map).tolist()

    # 用散点图显示路径点（采样，避免点太多）
    path_sample_n = min(50000, len(track_df))
    if len(track_df) > path_sample_n:
        idx = np.random.choice(len(track_df), path_sample_n, replace=False)
        track_show = track_df.iloc[idx]
    else:
        track_show = track_df

    ax.scatter(
        track_show['经度'], track_show['纬度'], track_show['海拔'],
        c=track_show['region_id'].map(region_color_map),
        s=1.0, alpha=0.3, label='Path (GPS)'
    )

    # ── 2. 画几条完整轨迹线 ──
    sample_tracks = np.random.choice(track_ids, min(10, len(track_ids)), replace=False)
    for tid in sample_tracks:
        one = track_df[track_df['trackId'] == tid].sort_values('时间_秒')
        ax.plot(
            one['经度'], one['纬度'], one['海拔'],
            color='gray', linewidth=0.3, alpha=0.3
        )

    # ── 3. POI 点 ──
    if poi_df is not None and len(poi_df) > 0:
        # 按距路径距离分色：≤30m 绿色，≤100m 橙色，>100m 红色
        poi_near = poi_df[poi_df['dist_to_path_m'] <= 30]
        poi_mid = poi_df[(poi_df['dist_to_path_m'] > 30) & (poi_df['dist_to_path_m'] <= 100)]
        poi_far = poi_df[poi_df['dist_to_path_m'] > 100]

        for grp, color, marker, label in [
            (poi_near, 'lime', 'o', 'POI ≤30m'),
            (poi_mid, 'orange', '^', 'POI ≤100m'),
            (poi_far, 'red', 'x', 'POI >100m'),
        ]:
            if len(grp) > 0:
                ax.scatter(
                    grp['plot_lon'], grp['plot_lat'], grp['plot_alt'],
                    c=color, s=30, marker=marker, alpha=0.8,
                    label=label, edgecolors='k', linewidths=0.3
                )

        # 标注前 10 个最近 POI 的名称
        top_poi = poi_df.nsmallest(10, 'dist_to_path_m')
        for _, row in top_poi.iterrows():
            ax.text(
                row['plot_lon'], row['plot_lat'], row['plot_alt'],
                row['poi_name'], fontsize=5, alpha=0.7,
                ha='left', va='bottom'
            )

    # ── 美化 ──
    alt_min, alt_max = path_df['海拔'].min(), path_df['海拔'].max()
    ax.set_xlabel('经度')
    ax.set_ylabel('纬度')
    ax.set_zlabel('海拔 (m)')
    ax.set_title(f'{scenery_name} - 3D 路径 + POI 分布\n'
                 f'{n_regions} regions, {len(poi_df) if poi_df is not None else 0} POIs, '
                 f'海拔 {alt_min:.0f}~{alt_max:.0f}m',
                 fontsize=14)
    ax.legend(loc='upper left', fontsize=8)

    # 设置视角
    ax.view_init(elev=25, azim=-60)

    # 保存
    out_path = os.path.join(OUTPUT_DIR, f'{scenery_name}_3d_map.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"  已保存: {out_path}")
    plt.close(fig)

    # 生成无 POI 的纯路径版本
    fig2 = plt.figure(figsize=(16, 12))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(
        track_show['经度'], track_show['纬度'], track_show['海拔'],
        c=track_show['region_id'].map(region_color_map),
        s=0.8, alpha=0.3
    )
    for tid in sample_tracks:
        one = track_df[track_df['trackId'] == tid].sort_values('时间_秒')
        ax2.plot(one['经度'], one['纬度'], one['海拔'],
                 color='gray', linewidth=0.3, alpha=0.2)
    ax2.set_xlabel('经度'); ax2.set_ylabel('纬度'); ax2.set_zlabel('海拔 (m)')
    ax2.set_title(f'{scenery_name} - 路径海拔分布')
    ax2.view_init(elev=25, azim=-60)
    out_path2 = os.path.join(OUTPUT_DIR, f'{scenery_name}_3d_path.png')
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"  已保存: {out_path2}")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description='3D 地图：路径 + POI')
    parser.add_argument('--scenery', type=str, default='峨眉山',
                        help='景区名')
    parser.add_argument('--all', action='store_true',
                        help='输出全部景区')
    parser.add_argument('--max_path_points', type=int, default=20000,
                        help='路径最大采样点数')
    parser.add_argument('--max_tracks', type=int, default=50,
                        help='显示的最大轨迹数')
    args = parser.parse_args()

    # 决定要处理的景区
    if args.all:
        scenery_list = list(SCENERY_CSV_MAP.keys())
    else:
        scenery_list = [args.scenery]

    for scenery_name in scenery_list:
        print(f"\n{'='*60}")
        print(f"[{scenery_name}]")

        path_df = load_path_data(scenery_name, args.max_path_points)
        if path_df is None:
            continue

        poi_df = load_poi_data(scenery_name)
        if poi_df is not None:
            poi_df = enrich_poi_elevation(poi_df, path_df)

        plot_3d(scenery_name, path_df, poi_df, args.max_tracks)

    print(f"\n完成! 图片在 {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
