import os
import pandas as pd
import numpy as np

BASE_DIR = r'D:\pythonProjectSiChuan - 副本 (2)\statistical_test_results_v2'

FRIEDMAN_CSV = os.path.join(BASE_DIR, 'friedman_all_v2.csv')
WILCOXON_CSV = os.path.join(BASE_DIR, 'wilcoxon_all_v2.csv')

OUTPUT_TEX = os.path.join(BASE_DIR, 'maintext_afs_oaq_stats_table.tex')

METRIC_MAP = {
    'avg_fitness_score':      r'AFS $\uparrow$',
    'avg_allocation_quality': r'OAQ $\uparrow$',
}

EXP_ORDER     = ['Capacity', 'Weight']
DATASET_ORDER = ['Beijing', 'Chengdu']
BASELINE_ORDER = ['PX', 'QITA', 'DTAA', 'FTSA']

df_f = pd.read_csv(FRIEDMAN_CSV)
df_w = pd.read_csv(WILCOXON_CSV)

df_f = df_f[df_f['metric'].isin(METRIC_MAP.keys())].copy()
df_w = df_w[df_w['metric'].isin(METRIC_MAP.keys())].copy()

df_w['baseline'] = df_w['comparison'].str.replace('BDTA vs ', '', regex=False)

def fmt_p_value(p, bold_if_sig=False, alpha=0.05):
    if pd.isna(p):
        return r'---'
    s = f'{p:.4f}'
    if bold_if_sig and p < alpha:
        return r'\textbf{' + s + r'}'
    return s

def fmt_pair_cell(row):
    """
    每个 baseline 单元格显示为: p / r
    其中:
      - p 为 Wilcoxon 原始双侧 p 值
      - r 为 matched-pairs rank-biserial correlation
    仅当 BDTA 在该比较中方向更优时，对 r 加粗（不是显著性，只是方向性强调）
    """
    if row is None or row.empty:
        return r'---'

    p = row.iloc[0]['p_value']
    r = row.iloc[0]['effect_size_r']
    better = bool(row.iloc[0].get('bdta_better', False))

    p_str = f'{p:.4f}' if not pd.isna(p) else '---'
    r_str = f'{r:+.2f}' if not pd.isna(r) else '---'

    if better and r_str != '---':
        r_str = r'\textbf{' + r_str + r'}'

    return f'{p_str} / {r_str}'

def make_maintext_stats_table(df_f, df_w):
    lines = []
    lines.append(r'\begin{table*}[htbp]')
    lines.append(r'\caption{Statistical significance analysis on the two primary evaluation metrics: average fitness score (AFS) and overall allocation quality (OAQ). '
                 r'For each experiment--dataset pair, the Friedman test reports the omnibus significance across all five methods. '
                 r'Each pairwise comparison cell reports the raw two-sided Wilcoxon signed-rank $p$-value and the matched-pairs rank-biserial correlation $r$ in the form ``$p/r$''. '
                 r'Positive $r$ indicates that BDTA outperforms the corresponding baseline. '
                 r'Due to the small number of parameter levels ($n=6$ for capacity and $n=5$ for weight), '
                 r'the minimum achievable Bonferroni-corrected $p$-value exceeds $0.05$ by design; therefore, emphasis is placed on effect size rather than per-cell significance stars.}')
    lines.append(r'\label{tab:maintext_afs_oaq_stats}')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\renewcommand{\arraystretch}{1.15}')
    lines.append(r'\resizebox{\textwidth}{!}{%')
    lines.append(r'\begin{tabular}{lllccccc}')
    lines.append(r'\hline')
    lines.append(r'Experiment & Dataset & Metric & Friedman $p$ & PX ($p/r$) & QITA ($p/r$) & DTAA ($p/r$) & FTSA ($p/r$) \\')
    lines.append(r'\hline')

    for exp in EXP_ORDER:
        df_f_exp = df_f[df_f['exp'] == exp].copy()
        df_w_exp = df_w[df_w['exp'] == exp].copy()

        first_exp_row = True

        n_exp_rows = len(DATASET_ORDER) * len(METRIC_MAP)

        for dataset in DATASET_ORDER:
            first_dataset_row = True
            n_dataset_rows = len(METRIC_MAP)

            for metric, metric_label in METRIC_MAP.items():
                row_f = df_f_exp[
                    (df_f_exp['dataset'] == dataset) &
                    (df_f_exp['metric']  == metric)
                ]

                friedman_p = r'---'
                if not row_f.empty:
                    friedman_p = fmt_p_value(
                        row_f.iloc[0]['p_value'],
                        bold_if_sig=True,
                        alpha=0.05
                    )

                pair_cells = []
                for base in BASELINE_ORDER:
                    row_w = df_w_exp[
                        (df_w_exp['dataset']  == dataset) &
                        (df_w_exp['metric']   == metric) &
                        (df_w_exp['baseline'] == base)
                    ]
                    pair_cells.append(fmt_pair_cell(row_w))

                exp_prefix = ''
                data_prefix = ''

                if first_exp_row:
                    exp_prefix = rf'\multirow{{{n_exp_rows}}}{{*}}{{{exp}}}'
                    first_exp_row = False
                else:
                    exp_prefix = ''

                if first_dataset_row:
                    data_prefix = rf'\multirow{{{n_dataset_rows}}}{{*}}{{{dataset}}}'
                    first_dataset_row = False
                else:
                    data_prefix = ''

                if exp_prefix and data_prefix:
                    line = (
                        rf'{exp_prefix} & {data_prefix} & {metric_label} & {friedman_p} & '
                        rf'{pair_cells[0]} & {pair_cells[1]} & {pair_cells[2]} & {pair_cells[3]} \\'
                    )
                elif exp_prefix and not data_prefix:
                    line = (
                        rf'{exp_prefix} &  & {metric_label} & {friedman_p} & '
                        rf'{pair_cells[0]} & {pair_cells[1]} & {pair_cells[2]} & {pair_cells[3]} \\'
                    )
                elif (not exp_prefix) and data_prefix:
                    line = (
                        rf' & {data_prefix} & {metric_label} & {friedman_p} & '
                        rf'{pair_cells[0]} & {pair_cells[1]} & {pair_cells[2]} & {pair_cells[3]} \\'
                    )
                else:
                    line = (
                        rf' &  & {metric_label} & {friedman_p} & '
                        rf'{pair_cells[0]} & {pair_cells[1]} & {pair_cells[2]} & {pair_cells[3]} \\'
                    )

                lines.append(line)

            lines.append(r'\hline')

    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\end{table*}')

    return '\n'.join(lines)

latex_str = make_maintext_stats_table(df_f, df_w)
print()

with open(OUTPUT_TEX, 'w', encoding='utf-8') as f:
    f.write(latex_str)

print()