"""
结果汇总 & LaTeX 表格生成

从 ablation/output/*.json 读取消融实验结果，生成论文可用的 LaTeX 表格和汇总文本。
"""

import os
import sys
import json
import argparse


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def pvalue_to_str(p):
    if p is None:
        return '---'
    s = f'{p:.4f}'
    if p < 0.001:
        return f'{s}$^{{***}}$'
    elif p < 0.01:
        return f'{s}$^{{**}}$'
    elif p < 0.05:
        return f'{s}$^{{*}}$'
    else:
        return f'{s}'


def generate_prediction_table(results, full_config='full'):
    """生成预测侧消融实验结果表格"""
    full = results.get(full_config, {})
    full_seg = full.get('seg_mae', 0)
    pvalues = results.get('_wilcoxon_pvalues', {})

    config_names = {
        'full': 'Full Model',
        'no_gps': '$-$ GPS(15)',
        'no_role': '$-$ Role Embed',
        'no_time': '$-$ Time Feature',
        'no_pos': '$-$ Pos Embed',
        'uni_gru': '$-$ BiGRU',
    }

    print("\n" + "=" * 80)
    print("Table: Ablation Study on Prediction Components (预测侧消融实验)")
    print("=" * 80)

    print(f"\n{'Variant':<22} {'Seg MAE':>10} {'Cum MAE':>10} {'Dur MAE':>10} "
          f"{'±30min':>8} {'p(Seg)':>10} {'Δ Seg':>10}")
    print("-" * 82)

    for key in ['full', 'no_gps', 'no_role', 'no_time', 'no_pos', 'uni_gru']:
        m = results.get(key, {})
        seg = m.get('seg_mae', 0)
        cum = m.get('cum_mae', 0)
        dur = m.get('dur_mae_s', 0)
        win30 = m.get('window_acc', {}).get('win_30min', 0)

        if key == full_config:
            delta_str = ""
            p_str = "---"
        else:
            pct = (seg - full_seg) / full_seg * 100 if full_seg > 0 else 0
            delta_str = f"+{pct:.1f}\\%"
            pv = pvalues.get(key, {})
            p_str = pv.get('seg_stars', 'N/A')
        name = config_names.get(key, key)
        print(f"{name:<22} {seg:10.0f} {cum:10.0f} {dur:10.0f} "
              f"{win30:7.1f}\\% {p_str:>10} {delta_str:>10}")

    print("\n--- LaTeX Table ---\n")
    latex = r"""
\\begin{table*}[t]
\\centering
\\caption{Ablation study on prediction components. Each row removes one component
  while keeping all others identical. $\\Delta$ reports relative Seg MAE increase
  compared to Full Model. $p$-values from two-sided Wilcoxon signed-rank test
  between Full Model and each variant ($^{*}: p<0.05$, $^{**}: p<0.01$, $^{***}: p<0.001$).}
\\label{tab:ablation_prediction}
\\begin{tabular}{lccccc}
\\toprule
\\textbf{Variant} & \\textbf{Seg MAE (s)} & \\textbf{Cum MAE (s)} & \\textbf{Dur MAE (s)} & \\textbf{$\\pm$30min (\\%)} & \\textbf{$p$ (Seg)} \\\\
\\midrule
"""
    for key in ['full', 'no_gps', 'no_role', 'no_time', 'no_pos', 'uni_gru']:
        m = results.get(key, {})
        seg = m.get('seg_mae', 0)
        cum = m.get('cum_mae', 0)
        dur = m.get('dur_mae_s', 0)
        win30 = m.get('window_acc', {}).get('win_30min', 0)
        name = config_names.get(key, key)

        if key == 'full':
            p_latex = '---'
        else:
            pv = pvalues.get(key, {})
            p_val = pv.get('seg_p')
            p_latex = pvalue_to_str(p_val)

        latex += f"{name} & {seg:.0f} & {cum:.0f} & {dur:.0f} & {win30:.1f} & {p_latex} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    print(latex)


def generate_clustering_table_1(results):
    """粒球 vs 裸点消融表格"""
    r = results.get('balls_vs_raw', {})
    wb = r.get('with_balls', {})
    nb = r.get('without_balls', {})

    print("\n" + "=" * 80)
    print("Table: Ablation on Granular Ball Representation (粒球 vs 裸点)")
    print("=" * 80)

    print(f"\n{'Method':<28} {'Silhouette':>12} {'DB':>10} {'CH':>10} {'#Clusters':>10} {'Time(s)':>10}")
    print("-" * 82)
    print(f"{'Granular Balls + Spectral':<28} {wb.get('silhouette', 0):12.4f} "
          f"{wb.get('davies_bouldin', 0):10.4f} {wb.get('calinski_harabasz', 0):10.2f} "
          f"{wb.get('n_clusters', 0):10d} {wb.get('time_s', 0):10.1f}")
    print(f"{'Raw Points + Spectral':<28} {nb.get('silhouette', 0):12.4f} "
          f"{nb.get('davies_bouldin', 0):10.4f} {nb.get('calinski_harabasz', 0):10.2f} "
          f"{nb.get('n_clusters', 0):10d} {nb.get('time_s', 0):10.1f}")

    wb_sil = wb.get('silhouette', 0)
    nb_sil = nb.get('silhouette', 0)
    if nb_sil > 0:
        print(f"\n  Improvement: Sil +{(wb_sil - nb_sil) / nb_sil * 100:.1f}%, "
              f"DB {(nb.get('davies_bouldin', 0) - wb.get('davies_bouldin', 0)) / nb.get('davies_bouldin', 1) * 100:+.1f}%")

    print("\n--- LaTeX Table ---\n")
    latex = r"""
\begin{table*}[t]
\centering
\caption{Ablation study on Granular Ball representation.}
\label{tab:ablation_balls}
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{Silhouette $\uparrow$} & \textbf{DB $\downarrow$} & \textbf{CH $\uparrow$} & \textbf{\# Clusters} \\
\midrule
"""
    latex += f"Granular Balls + Spectral & {wb.get('silhouette', 0):.4f} & {wb.get('davies_bouldin', 0):.4f} & {wb.get('calinski_harabasz', 0):.2f} & {wb.get('n_clusters', 0)} \\\\\n"
    latex += f"Raw Points + Spectral & {nb.get('silhouette', 0):.4f} & {nb.get('davies_bouldin', 0):.4f} & {nb.get('calinski_harabasz', 0):.2f} & {nb.get('n_clusters', 0)} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    print(latex)


def generate_clustering_table_2(results):
    """BKOA vs Grid vs Random 搜索消融表格"""
    r = results.get('search_strategies', {})

    print("\n" + "=" * 80)
    print("Table: Comparison of Parameter Search Strategies (BKOA vs Grid vs Random)")
    print("=" * 80)

    print(f"\n{'Strategy':<20} {'Best Sil':>10} {'Mean Sil':>10} {'Std Sil':>10} "
          f"{'Evals':>8} {'Time(s)':>10}")
    print("-" * 72)

    for key in ['grid', 'random', 'bkoa']:
        m = r.get(key, {})
        mean_sil = m.get('mean_sil', m.get('best_sil', 0))
        std_sil = m.get('std_sil', 0)
        best_sil = m.get('best_sil', 0)
        evals = m.get('n_evals', m.get('mean_n_evals', 0))
        t = m.get('time_s', m.get('mean_time_s', 0))
        name = {'grid': 'Grid Search', 'random': 'Random Search', 'bkoa': 'BKOA'}.get(key, key)
        print(f"{name:<20} {best_sil:10.4f} {mean_sil:10.4f} {std_sil:10.4f} "
              f"{evals:8.0f} {t:10.1f}")

    print("\n--- LaTeX Table ---\n")
    latex = r"""
\begin{table*}[t]
\centering
\caption{Comparison of parameter search strategies. BKOA achieves higher best
  and mean Silhouette with lower variance compared to Random Search, and improves
  over deterministic Grid Search.}
\label{tab:ablation_search}
\begin{tabular}{lcccc}
\toprule
\textbf{Strategy} & \textbf{Best Sil $\uparrow$} & \textbf{Mean Sil} & \textbf{Std Sil} & \textbf{\# Evals} \\
\midrule
"""
    for key in ['grid', 'random', 'bkoa']:
        m = r.get(key, {})
        mean_sil = m.get('mean_sil', m.get('best_sil', 0))
        std_sil = m.get('std_sil', 0.0)
        best_sil = m.get('best_sil', 0)
        evals = m.get('n_evals', m.get('mean_n_evals', 0))
        name = {'grid': 'Grid Search', 'random': 'Random Search', 'bkoa': 'BKOA'}.get(key, key)
        if 'std_sil' in m or 'mean_sil' in m:
            latex += f"{name} & {best_sil:.4f} & {mean_sil:.4f} & {std_sil:.4f} & {int(evals)} \\\\\n"
        else:
            latex += f"{name} & {best_sil:.4f} & --- & --- & {int(evals)} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    print(latex)


def main():
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from ablation results')
    parser.add_argument('--output_dir', type=str, default='ablation/output')
    parser.add_argument('--prediction_json', type=str,
                        default='ablation/output/prediction_ablation_results.json')
    parser.add_argument('--clustering_json', type=str,
                        default='ablation/output/clustering_ablation_results.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prediction_exists = os.path.exists(args.prediction_json)
    clustering_exists = os.path.exists(args.clustering_json)

    if not prediction_exists and not clustering_exists:
        print("No ablation results found. Run run_prediction_ablation.py and "
              "run_clustering_ablation.py first.")
        return

    if prediction_exists:
        print("\n" + "=" * 80)
        print("PREDICTION ABLATION RESULTS")
        print("=" * 80)
        pred_results = load_json(args.prediction_json)
        generate_prediction_table(pred_results)

    if clustering_exists:
        print("\n" + "=" * 80)
        print("CLUSTERING ABLATION RESULTS")
        print("=" * 80)
        clust_results = load_json(args.clustering_json)

        if 'balls_vs_raw' in clust_results:
            generate_clustering_table_1(clust_results)

        if 'search_strategies' in clust_results:
            generate_clustering_table_2(clust_results)

    latex_output = os.path.join(args.output_dir, 'ablation_tables.tex')
    print(f"\nAll LaTeX tables printed above. Redirect to save: "
          f"python ablation/generate_ablation_tables.py > {latex_output}")


if __name__ == '__main__':
    main()