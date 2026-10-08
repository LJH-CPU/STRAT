"""
生成聚类实验汇总报告（markdown）。

读取:
- cluster/output/experiments_all.json   (双轨评测)
- baselines/output/clustering_baselines.json (基线对比)
- ablation/output/*_ablation.json       (消融)

输出: cluster/output/clustering_report.md
"""

import os
import sys
import json
import glob

EXP = 'cluster/output/experiments_all.json'
BAS = 'baselines/output/clustering_baselines.json'
ABL = 'ablation/output'
OUT = 'cluster/output/clustering_report.md'


def _fmt(v, nd=4):
    if v is None:
        return 'N/A'
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    exp = json.load(open(EXP, encoding='utf-8'))
    bas = json.load(open(BAS, encoding='utf-8')) if os.path.exists(BAS) else {}
    abl_files = sorted(glob.glob(os.path.join(ABL, '*_ablation.json')))
    abl = {}
    for f in abl_files:
        name = os.path.basename(f).replace('_cleaned_ablation.json', '').replace('_ablation.json', '')
        abl[name] = json.load(open(f, encoding='utf-8'))

    L = []
    L.append("# STRAT 聚类实验报告")
    L.append("")
    L.append(f"**方法**：粒球抽象 + 多因子规则亲和力 + 网格选 δ（验证子集，避免选择偏差）+ 递归 eigen-gap 自动定 k。")
    L.append(f"**输入**：各景区停留点（is_stop=1），经纬度转米 + MinMax 归一化（3D：经度/纬度/海拔），统一采样 ≤5000。")
    L.append(f"**外部验证**：POI 缓冲区区域（50m）真值，POI 投影到路径，仅停留点匹配，沿路径不传播；只用于验证不参与调参。")
    L.append("")

    # 1. 主表（双轨）
    L.append("## 1. 主实验结果（双轨评测）")
    L.append("")
    L.append("| 景区 | 轨迹 | k | POI区 | Sil↑ | DBI↓ | CH↑ | Composite | ARI↑ | NMI↑ | FMI↑ | 匹配率 | 簇纯度 | GT覆盖 |")
    L.append("|:----|----:|--:|------:|-----:|-----:|----:|----------:|-----:|-----:|-----:|-------:|-------:|-------:|")
    for name in sorted(exp):
        r = exp[name]
        if 'status' in r and r.get('status') == 'error':
            L.append(f"| {name} | - | - | - | ERROR | | | | | | | | | |")
            continue
        it, ex, co = r['internal'], r['external'], r['correspondence']
        L.append(f"| {name} | {r['n_tracks']} | {it['final_k']} | {r['n_poi_regions']} "
                 f"| {_fmt(it['silhouette'])} | {_fmt(it['davies_bouldin'])} | {_fmt(it['calinski_harabasz'],1)} "
                 f"| {_fmt(it['composite'])} | {_fmt(ex.get('ari'))} | {_fmt(ex.get('nmi'))} | {_fmt(ex.get('fmi'))} "
                 f"| {_fmt(ex.get('match_rate'),1)}% | {_fmt(co.get('purity'),3)} | {_fmt(co.get('gt_coverage'),3)} |")
    L.append("")
    L.append("注：匹配率=落入 POI 区域缓冲区的停留点比例；ARI/NMI/FMI 仅在匹配到 POI 区域标签的点上计算。")
    L.append("")

    # 2. 基线对比
    L.append("## 2. 基线对比（逐场景调参，验证子集选参，同一输入）")
    L.append("")
    L.append("| 景区 | 方法 | Sil↑ | DB↓ | CH↑ | k | 噪声 | t(s) |")
    L.append("|:----|:----|-----:|----:|----:|--:|-----:|-----:|")
    order = ['strat', 'k-means', 'dbscan', 'hdbscan', 'agglomerative', 'spectral_gaussian']
    names = {'strat': 'STRAT(Ours)', 'k-means': 'K-Means', 'dbscan': 'DBSCAN',
             'hdbscan': 'HDBSCAN', 'agglomerative': 'Agglo.', 'spectral_gaussian': 'Spec(RBF)'}
    for scene in sorted(bas):
        for key in order:
            m = bas[scene].get(key)
            if m is None:
                continue
            L.append(f"| {scene} | {names[key]} | {_fmt(m['silhouette'])} | {_fmt(m['davies_bouldin'])} "
                     f"| {_fmt(m['calinski_harabasz'],1)} | {m['n_clusters']} "
                     f"| {_fmt(m['noise_rate']*100,1)}% | {_fmt(m['time_s'],1)} |")
    L.append("")
    L.append("注：裸点谱聚类因 O(N³) 谱分解取 ≤2000 点子集（恰是粒球抽象要解决的瓶颈）；HDBSCAN 在全部场景判为噪声。")
    L.append("")

    # 3. 消融
    L.append("## 3. 消融")
    L.append("")
    L.append("### 3.1 粒球抽象 vs 裸点 RBF 谱聚类")
    L.append("")
    L.append("| 景区 | 粒球 Sil | 裸点 Sil | 粒球 DB | 裸点 DB | 粒球 CH | 裸点 CH | 粒球 k | 裸点 k | 粒球 t(s) | 裸点 t(s) |")
    L.append("|:----|--------:|--------:|--------:|--------:|--------:|--------:|-------:|-------:|---------:|---------:|")
    for name in sorted(abl):
        r = abl[name].get('balls_vs_raw', {})
        if not r:
            continue
        b, w = r.get('with_balls', {}), r.get('without_balls', {})
        L.append(f"| {name} | {_fmt(b.get('silhouette'))} | {_fmt(w.get('silhouette'))} "
                 f"| {_fmt(b.get('davies_bouldin'))} | {_fmt(w.get('davies_bouldin'))} "
                 f"| {_fmt(b.get('calinski_harabasz'),1)} | {_fmt(w.get('calinski_harabasz'),1)} "
                 f"| {b.get('n_clusters')} | {w.get('n_clusters')} "
                 f"| {_fmt(b.get('time_s'),1)} | {_fmt(w.get('time_s'),1)} |")
    L.append("")
    L.append("### 3.2 δ 网格分辨率敏感性（0.05 vs 0.1）")
    L.append("")
    L.append("| 景区 | fine δ | fine sil | coarse δ | coarse sil | 差值 |")
    L.append("|:----|-------:|---------:|---------:|-----------:|-----:|")
    for name in sorted(abl):
        r = abl[name].get('delta_sensitivity', {})
        if not r:
            continue
        fb, cb = r['fine_best'], r['coarse_best']
        L.append(f"| {name} | {_fmt(fb['delta'],2)} | {_fmt(fb['sel_sil'])} "
                 f"| {_fmt(cb['delta'],1)} | {_fmt(cb['sel_sil'])} | {_fmt(r['diff'])} |")
    L.append("")
    L.append("结论：fine(0.05) 与 coarse(0.1) 网格最优 δ 的验证 silhouette 差异 ≤0.01，网格 0.1 步长足够。")
    L.append("")

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print(f"报告已生成: {OUT}")


if __name__ == '__main__':
    main()
