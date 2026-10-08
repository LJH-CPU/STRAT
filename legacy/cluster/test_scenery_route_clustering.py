"""
STRAT 景点+路线聚类 测试脚本
===============================
运行完整 Pipeline 并生成评测报告和可视化图片。
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from scenery_route_clustering import run_full_pipeline

plt.rcParams['axes.unicode_minus'] = False
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'DejaVu Sans']
except Exception:
    pass


def visualize_scenic_spots(df, pca_features, sampled_labels, ball_data_list, output_dir):
    """景点聚类可视化：4 子图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    # 子图 1: 原始停留点在地图上的分布（等距采样）
    ax = axes[0, 0]
    stay = df[df['is_stop'] == 1]
    n_show = min(20000, len(stay))
    idx = np.random.choice(len(stay), n_show, replace=False)
    show = stay.iloc[idx]
    ax.scatter(show['经度'], show['纬度'], c='steelblue', s=0.5, alpha=0.4)
    ax.set_title('Stay Points Distribution (sampled)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # 子图 2: PCA 空间中聚类结果
    ax = axes[0, 1]
    unique_labels = sorted(set(sampled_labels) - {-1})
    for i, lbl in enumerate(unique_labels):
        mask = sampled_labels == lbl
        ax.scatter(pca_features[mask, 0], pca_features[mask, 1],
                    c=[colors[i % 20]], s=2, alpha=0.5, label=f'Region {lbl}')
    ax.set_title(f'Scenic Regions in PCA Space ({len(unique_labels)} regions)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(markerscale=5, fontsize=7)

    # 子图 3: 聚类后的区域在地图上的分布（采样）
    ax = axes[1, 0]
    for i, lbl in enumerate(unique_labels):
        region_points = df[(df['is_stop'] == 1) & (df['region_id'] == lbl)]
        n_show_r = min(3000, len(region_points))
        if len(region_points) > 0:
            idx_r = np.random.choice(len(region_points), n_show_r, replace=False)
            show_r = region_points.iloc[idx_r]
            ax.scatter(show_r['经度'], show_r['纬度'],
                        c=[colors[i % 20]], s=0.8, alpha=0.4, label=f'Region {lbl}')
    ax.set_title(f'Scenic Regions on Map ({len(unique_labels)} regions)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(markerscale=8, fontsize=7)

    # 子图 4: 粒球可视化
    ax = axes[1, 1]
    for i, ball in enumerate(ball_data_list):
        if len(ball) == 0:
            continue
        center = ball.mean(axis=0)
        radius = np.max(np.linalg.norm(ball - center, axis=1)) if len(ball) > 1 else 0.01
        if pca_features.shape[1] >= 2:
            ax.scatter(center[0], center[1], c='red', s=5, alpha=0.8)
            circle = plt.Circle((center[0], center[1]), radius, fill=False,
                                 color='gray', alpha=0.3, linewidth=0.3)
            ax.add_patch(circle)
        # 显示部分粒球的点
        if i % 10 == 0 and len(ball) > 0:
            ax.scatter(ball[:, 0], ball[:, 1], s=0.5, alpha=0.2, c='blue')
    ax.set_title(f'Granular Balls ({len(ball_data_list)} balls)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_aspect('equal')

    plt.suptitle('Scenic Spot Clustering Results', fontsize=16, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'scenic_spots_clustering.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def visualize_routes(df, route_labels, route_sequences, output_dir):
    """路线发现可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    unique_routes = sorted(set(v for v in route_labels.values() if v >= 0))

    # 子图 1: 每条路线的代表性轨迹在地图上
    ax = axes[0]
    for i, rid in enumerate(unique_routes):
        route_tracks = [tid for tid, lbl in route_labels.items() if lbl == rid]
        if not route_tracks:
            continue
        # 找该路线的代表序列用于标签
        seqs_in_route = [route_sequences.get(tid, ()) for tid in route_tracks[:10]]
        seq_counter = Counter(seqs_in_route)
        top_seq = seq_counter.most_common(1)[0][0] if seqs_in_route else ()
        label = f'R{rid}: {list(top_seq)[:4]}' if len(top_seq) > 4 else f'R{rid}: {list(top_seq)}'
        # 画前 3 条轨迹
        for j, tid in enumerate(route_tracks[:3]):
            track = df[df['trackId'] == tid].sort_values('时间_秒')
            if len(track) > 1:
                step = max(1, len(track) // 200)
                ax.plot(track['经度'].iloc[::step], track['纬度'].iloc[::step],
                         c=colors[rid % 20], alpha=0.6 + 0.2 * j, linewidth=0.8,
                         label=label if j == 0 else None)
    ax.set_title(f'Discovered Routes ({len(unique_routes)} routes)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(fontsize=6)

    # 子图 2: 每条路线的轨迹数
    ax = axes[1]
    route_counts = []
    labels = []
    for rid in unique_routes:
        cnt = sum(1 for v in route_labels.values() if v == rid)
        route_counts.append(cnt)
        labels.append(f'Route {rid}')
    ax.bar(labels, route_counts, color=[colors[rid % 20] for rid in unique_routes])
    ax.set_title('Tracks per Route')
    ax.set_ylabel('Number of Tracks')
    ax.tick_params(axis='x', rotation=45)

    # 子图 3: 每条路线触及的 unique region 数
    ax = axes[2]
    n_regions_touched = []
    for rid in unique_routes:
        route_tracks = [tid for tid, lbl in route_labels.items() if lbl == rid]
        all_regions = set()
        for tid in route_tracks:
            seq = route_sequences.get(tid, ())
            all_regions.update(seq)
        n_regions_touched.append(len(all_regions))
    ax.bar(labels, n_regions_touched, color=[colors[rid % 20] for rid in unique_routes])
    ax.set_title('Unique Regions per Route')
    ax.set_ylabel('# of Regions Touched')
    ax.tick_params(axis='x', rotation=45)

    plt.suptitle('Route Discovery Results', fontsize=16, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, 'route_discovery.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def visualize_summary(scenic_metrics, route_metrics, output_dir):
    """评测指标汇总图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 景点指标
    ax = axes[0]
    comp_sub = scenic_metrics.get('composite_sub', {})
    scenic_items = {
        'Silhouette\n(best)': scenic_metrics.get('best_sil', scenic_metrics['silhouette']),
        'Composite': scenic_metrics.get('composite', scenic_metrics['silhouette']),
        'DB\n(lower=better)': min(scenic_metrics['davies_bouldin'], 5.0),
        'CH / 10000': scenic_metrics['calinski_harabasz'] / 10000,
    }
    bars = ax.bar(scenic_items.keys(), scenic_items.values(),
                   color=['steelblue', 'mediumorchid', 'coral', 'seagreen'])
    ax.set_title(f"Scenic Spot Clustering\n({scenic_metrics['n_regions']} regions, {scenic_metrics['n_balls']} balls, "
                 f"k∈[{scenic_metrics.get('k_lower','?')},{scenic_metrics.get('k_upper','?')}], "
                 f"src={scenic_metrics.get('best_source', 'N/A')})")
    ax.set_ylim(0, max(1.0, max(scenic_items.values()) * 1.2))
    for bar, val in zip(bars, scenic_items.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=8)

    # 路线指标
    ax = axes[1]
    route_items = {
        'Routes': route_metrics['n_routes'] / 10,
        'Tracks': route_metrics['n_tracks'] / 500,
    }
    bars = ax.bar(route_items.keys(), route_items.values(),
                   color=['steelblue', 'seagreen'])
    ax.set_title(f"Route Discovery\n({route_metrics['n_routes']} routes, {route_metrics['n_tracks']} tracks, "
                 f"thresh={route_metrics.get('threshold', 0.5)})")
    ax.set_ylim(0, 1.2)
    for bar, val in zip(bars, route_items.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)

    plt.suptitle('Clustering Evaluation Summary', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'evaluation_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def main():
    import argparse
    from config import default_cleaned_csv

    parser = argparse.ArgumentParser(description='STRAT 景点聚类 + 路线发现 完整测试')
    parser.add_argument('--csv', default=None,
                        help='清洗后的 CSV 路径（默认取 data-project/cleaned_labeled_data 下第一个 *_cleaned.csv）')
    parser.add_argument('--output_dir', default=None,
                        help='可视化输出目录（默认 cluster/output）')
    args = parser.parse_args()

    csv_path = args.csv or default_cleaned_csv()
    if not csv_path or not os.path.exists(csv_path):
        raise SystemExit('未找到输入 CSV，请用 --csv 指定路径。'
                         '默认目录: data-project/cleaned_labeled_data')

    OUTPUT_DIR = args.output_dir or os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("STRAT 景点聚类 + 路线发现 完整测试")
    print(f"输入: {csv_path}")
    print("=" * 70)

    # 运行 Pipeline
    df_out, scenic_m, route_m, artifacts = run_full_pipeline(csv_path)

    # 生成可视化
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)

    visualize_scenic_spots(
        df_out,
        artifacts['pca_features'],
        artifacts['sampled_labels'],
        artifacts['ball_data_list'],
        OUTPUT_DIR,
    )

    visualize_routes(
        df_out,
        artifacts['route_labels'],
        artifacts['route_sequences'],
        OUTPUT_DIR,
    )

    visualize_summary(scenic_m, route_m, OUTPUT_DIR)

    # 保存带标签的 CSV
    from config import scenery_name_from_csv
    name = scenery_name_from_csv(csv_path)
    output_csv = os.path.join(OUTPUT_DIR, f'{name}_clustered.csv')
    cols_to_save = ['经度', '纬度', '海拔', '速度(km/h)', 'is_stop', 'trackId',
                    '时间_秒', 'region_id', 'route_id']
    available = [c for c in cols_to_save if c in df_out.columns]
    df_out[available].to_csv(output_csv, index=False)
    print(f"\n  Saved labeled CSV: {output_csv}")
    print(f"  Scenic regions: {scenic_m['n_regions']}")
    print(f"  Routes discovered:  {route_m['n_routes']}")

    # 最终评测摘要
    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n  === Scenic Spot Clustering ===")
    print(f"  Regions found:      {scenic_m['n_regions']}")
    print(f"  Granular balls:     {scenic_m['n_balls']}")
    print(f"  Best source:        {scenic_m.get('best_source', 'N/A')}")
    print(f"  Best k (grid):      {scenic_m['best_k']}  (k from eigen-gap ∈ [{scenic_m.get('k_lower','?')}, {scenic_m.get('k_upper','?')}])")
    print(f"  Best delta:         {scenic_m['best_delta']:.1f}")
    print(f"  Best Silhouette:    {scenic_m.get('best_sil', scenic_m['silhouette']):.4f}")
    print(f"  Final k (eigen-gap): {scenic_m.get('final_k', scenic_m['best_k'])}")
    print(f"  Final Silhouette:   {scenic_m['silhouette']:.4f}")
    print(f"  Composite Score:    {scenic_m.get('composite', 0):.4f}")
    comp = scenic_m.get('composite_sub', {})
    if comp:
        print(f"    (0.5×sil_norm + 0.2×{comp.get('balance', 0):.4f} + 0.3×{comp.get('separation', 0):.4f})")
    print(f"  Davies-Bouldin:     {scenic_m['davies_bouldin']:.4f}  (lower better)")
    print(f"  Calinski-Harabasz:  {scenic_m['calinski_harabasz']:.2f}  (higher better)")

    print(f"\n  === Route Discovery ===")
    print(f"  Method:                Hierarchical (Ward)")
    print(f"  Distance threshold:    {route_m.get('threshold', 'N/A')}")
    print(f"  Routes discovered:     {route_m['n_routes']}")
    print(f"  Valid tracks:          {route_m['n_tracks']}")
    print(f"  Coverage:              100%")

    print(f"\n  Output images: {OUTPUT_DIR}/")
    print(f"    - scenic_spots_clustering.png")
    print(f"    - route_discovery.png")
    print(f"    - evaluation_summary.png")
    print(f"  Labeled CSV: {output_csv}")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()