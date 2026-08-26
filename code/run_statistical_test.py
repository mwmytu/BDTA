import pandas as pd
import numpy as np
from scipy import stats
import os

BASE = r'D:\pythonProjectSiChuan - 副本 (2)'

FILES = {
    'beijing_capacity': os.path.join(
        BASE, 'beijing_capacity_combined_results',
        'beijing_capacity_combined.csv'
    ),
    'chengdu_capacity': os.path.join(
        BASE, 'chengdu_capacity_combined_results',
        'chengdu_capacity_combined.csv'
    ),
    'beijing_weight': os.path.join(
        BASE, 'beijing_weight_combined_results',
        'beijing_weight_combined.csv'
    ),
    'chengdu_weight': os.path.join(
        BASE, 'chengdu_weight_combined_results',
        'chengdu_weight_combined.csv'
    ),
}

OUTPUT_DIR = os.path.join(BASE, 'statistical_test_results_v3')
os.makedirs(OUTPUT_DIR, exist_ok=True)

METHOD_ORDER = ['BDTA', 'PX', 'QITA', 'DTAA', 'FTSA']
BASELINES    = ['PX', 'QITA', 'DTAA', 'FTSA']
ALPHA        = 0.05

CAP_METRICS = {
    'demand_satisfaction_rate':         'DSR',
    'on_time_demand_satisfaction_rate': 'OT-DSR',
    'travel_km_per_served_demand':      'TSD',
    'worker_participation_rate':        'WPR',
    'avg_fitness_score':                'AFS',
    'avg_allocation_quality':           'OAQ',
}

WEIGHT_METRICS = {
    'demand_satisfaction_rate':         'DSR',
    'on_time_demand_satisfaction_rate': 'OT-DSR',
    'travel_km_per_served_demand':      'TSD',
    'worker_participation_rate':        'WPR',
    'avg_fitness_score':                'AFS',
    'avg_allocation_quality':           'OAQ',
}

MAIN_METRICS = {
    'avg_fitness_score':                'AFS',
    'avg_allocation_quality':           'OAQ',
    'demand_satisfaction_rate':         'DSR',
    'travel_km_per_served_demand':      'TSD',
}

METRIC_SHORT = {**CAP_METRICS, **WEIGHT_METRICS}

METRIC_DIRECTION = {
    'demand_satisfaction_rate':         True,
    'on_time_demand_satisfaction_rate': True,
    'travel_km_per_served_demand':      False,
    'worker_participation_rate':        True,
    'avg_fitness_score':                True,
    'avg_allocation_quality':           True,
}

def rank_biserial_correct(a: np.ndarray, b: np.ndarray) -> float:
    """
    标准 matched-pairs rank-biserial correlation。
    正值表示 a > b 的趋势，负值表示 a < b 的趋势。
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d_nonzero = d[d != 0]
    n_eff     = len(d_nonzero)
    if n_eff == 0:
        return 0.0

    abs_ranks    = stats.rankdata(np.abs(d_nonzero))
    pos_rank_sum = np.sum(abs_ranks[d_nonzero > 0])
    neg_rank_sum = np.sum(abs_ranks[d_nonzero < 0])
    total_rank   = n_eff * (n_eff + 1) / 2.0

    r = (pos_rank_sum - neg_rank_sum) / total_rank
    return float(np.clip(r, -1.0, 1.0))

def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return pd.read_csv(path)

df_cap = pd.concat([
    load(FILES['beijing_capacity']),
    load(FILES['chengdu_capacity']),
], ignore_index=True)

df_wt = pd.concat([
    load(FILES['beijing_weight']),
    load(FILES['chengdu_weight']),
], ignore_index=True)

if 'scenario_label' not in df_cap.columns and 'scenario' in df_cap.columns:
    df_cap['scenario_label'] = df_cap['scenario']
if 'scenario_label' not in df_wt.columns and 'scenario' in df_wt.columns:
    df_wt['scenario_label'] = df_wt['scenario']

print()
print()

def friedman_test(df, metric, x_col, dataset):
    """
    Friedman 检验：在同一 dataset 下，对五种方法在不同参数水平上的结果
    做整体秩差异检验。
    """
    groups = []
    for m in METHOD_ORDER:
        sub = df[(df['scenario_label'] == m) &
                 (df['dataset'] == dataset)].sort_values(x_col)
        if sub.empty or metric not in sub.columns:
            continue
        groups.append(sub[metric].values)

    if len(groups) < 3:
        return None

    n = min(len(g) for g in groups)
    if n < 2:
        return None

    groups = [g[:n] for g in groups]
    stat, p = stats.friedmanchisquare(*groups)

    return {
        'dataset':     dataset,
        'metric':      metric,
        'statistic':   round(stat, 4),
        'p_value':     round(p,    6),
        'significant': p < ALPHA,
        'n_methods':   len(groups),
        'n_blocks':    n,
    }

def wilcoxon_pairwise(df, metric, x_col, dataset,
                      n_comparisons=4):
    """
    对每个 dataset/metric，比较 BDTA 与 4 个 baseline。
    返回：
      - 原始 p 值
      - Bonferroni 校正后的 p 值
      - rank-biserial effect size
    """
    results  = []
    bdta_sub = df[(df['scenario_label'] == 'BDTA') &
                  (df['dataset'] == dataset)].sort_values(x_col)

    if bdta_sub.empty or metric not in bdta_sub.columns:
        return results

    a = bdta_sub[metric].values

    for baseline in BASELINES:
        base_sub = df[(df['scenario_label'] == baseline) &
                      (df['dataset'] == dataset)].sort_values(x_col)
        if base_sub.empty or metric not in base_sub.columns:
            continue

        b = base_sub[metric].values
        n = min(len(a), len(b))
        if n < 3:
            continue

        av, bv = a[:n], b[:n]

        r_rb          = rank_biserial_correct(av, bv)
        higher_better = METRIC_DIRECTION.get(metric, True)
        bdta_better   = (r_rb > 0) if higher_better else (r_rb < 0)

        entry = {
            'dataset':        dataset,
            'metric':         metric,
            'comparison':     f'BDTA vs {baseline}',
            'n_pairs':        n,
            'bdta_mean':      round(float(np.mean(av)), 4),
            'base_mean':      round(float(np.mean(bv)), 4),
            'diff_mean':      round(float(np.mean(av - bv)), 4),
            'effect_size_r':  round(r_rb, 4),
            'bdta_better':    bdta_better,
            'higher_better':  higher_better,
        }

        if np.allclose(av, bv):
            entry.update(
                statistic    = np.nan,
                p_value_raw  = 1.0,
                p_value_bonf = min(1.0, 1.0 * n_comparisons),
                sig_bonf     = False,
                note         = 'identical'
            )
            results.append(entry)
            continue

        try:
            stat, p_raw = stats.wilcoxon(av, bv, alternative='two-sided')
            p_bonf      = min(1.0, p_raw * n_comparisons)
            entry.update(
                statistic    = round(stat, 4),
                p_value_raw  = round(p_raw, 6),
                p_value_bonf = round(p_bonf, 6),
                sig_bonf     = p_bonf < ALPHA,
                note         = ''
            )
        except ValueError as e:
            entry.update(
                statistic    = np.nan,
                p_value_raw  = np.nan,
                p_value_bonf = np.nan,
                sig_bonf     = False,
                note         = str(e)
            )

        results.append(entry)

    return results

def run_comparative_summary(df_cap, metrics_dict, default_c=6):
    """
    对主实验（固定 c_value=6）做简洁摘要：
    仅比较 BDTA 在每个 dataset / metric 上是否优于 baseline。
    注：由于主实验文件是跨日均值汇总，这里做描述性比较，不再做 Friedman/Wilcoxon。
    """
    all_rows = []
    datasets = sorted(df_cap['dataset'].dropna().unique().tolist())

    print()
    print()
    print()

    df_main = df_cap[df_cap['c_value'] == default_c].copy()

    for dataset in datasets:
        print()
        sub = df_main[df_main['dataset'] == dataset]

        for metric, short in metrics_dict.items():
            if metric not in sub.columns:
                continue

            bdta_row = sub[sub['scenario_label'] == 'BDTA']
            if bdta_row.empty:
                continue
            bdta_v = float(bdta_row[metric].values[0])

            print()
            for bl in BASELINES:
                bl_row = sub[sub['scenario_label'] == bl]
                if bl_row.empty:
                    continue
                bl_v   = float(bl_row[metric].values[0])
                higher = METRIC_DIRECTION.get(metric, True)
                better = (bdta_v > bl_v) if higher else (bdta_v < bl_v)
                symbol = '✅' if better else '❌'
                print()

                all_rows.append({
                    'dataset':       dataset,
                    'metric':        metric,
                    'metric_short':  short,
                    'comparison':    f'BDTA vs {bl}',
                    'bdta_value':    round(bdta_v, 4),
                    'base_value':    round(bl_v, 4),
                    'bdta_better':   better,
                    'higher_better': higher,
                    'exp':           'Comparative',
                    'c_value':       default_c,
                })

    return pd.DataFrame(all_rows)

def run_all_tests(df, metrics_dict, x_col, exp_label):
    all_fr, all_wt = [], []
    datasets = sorted(df['dataset'].dropna().unique().tolist())

    print()
    print()
    print()

    for dataset in datasets:
        print()
        for metric, short in metrics_dict.items():
            if metric not in df.columns:
                continue

            sub = df[df['dataset'] == dataset]
            if sub[metric].dropna().empty:
                continue

            print()

            fr = friedman_test(df, metric, x_col, dataset)
            if fr:
                fr['exp'] = exp_label
                all_fr.append(fr)
                sig = "✅" if fr['significant'] else "❌"
                print()

            wrs = wilcoxon_pairwise(df, metric, x_col, dataset)
            for r in wrs:
                r['exp'] = exp_label
                all_wt.append(r)
                better = "↑BDTA" if r['bdta_better'] else "↓BDTA"
                p_raw  = r.get('p_value_raw', float('nan'))
                p_bonf = r.get('p_value_bonf', float('nan'))
                print()

    return pd.DataFrame(all_fr), pd.DataFrame(all_wt)

df_fr_cap, df_wt_cap = run_all_tests(df_cap, CAP_METRICS, 'c_value', 'Capacity')
df_fr_wt,  df_wt_wt  = run_all_tests(df_wt,  WEIGHT_METRICS, 'w_task', 'Weight')

df_comparative = run_comparative_summary(df_cap, MAIN_METRICS, default_c=6)

df_friedman_all = pd.concat([df_fr_cap, df_fr_wt], ignore_index=True)
df_wilcoxon_all = pd.concat([df_wt_cap, df_wt_wt], ignore_index=True)

csv_outputs = [
    (os.path.join(OUTPUT_DIR, 'friedman_all_v3.csv'),      df_friedman_all),
    (os.path.join(OUTPUT_DIR, 'wilcoxon_all_v3.csv'),      df_wilcoxon_all),
    (os.path.join(OUTPUT_DIR, 'comparative_summary.csv'),  df_comparative),
]

for path, df_out in csv_outputs:
    df_out.to_csv(path, index=False)
    print()

def make_friedman_latex(df_f, exp_label):
    sub = df_f[df_f['exp'] == exp_label].copy()
    if sub.empty:
        return f"% 无 {exp_label} Friedman 结果"

    tag   = exp_label.lower()
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(
        rf'\caption{{Friedman test results for the '
        rf'{exp_label.lower()} sensitivity experiment across five '
        rf'methods ($n_{{\mathrm{{methods}}}}=5$, $\alpha=0.05$). '
        rf'A significant result ($p<0.05$, shown in \textbf{{bold}}) '
        rf'indicates that at least one method differs systematically '
        rf'from the others across the tested parameter levels.}}'
    )
    lines.append(rf'\label{{tab:friedman_{tag}}}')
    lines.append(r'\centering')
    lines.append(r'\begin{tabularx}{\linewidth}{llXXX}')
    lines.append(r'\hline')
    lines.append(r'Dataset & Metric & $\chi^2$ & $p$-value & Sig. \\')
    lines.append(r'\hline')

    for dataset in ['Beijing', 'Chengdu']:
        dsub = sub[sub['dataset'] == dataset]
        if dsub.empty:
            continue
        first = True
        n     = len(dsub)
        for _, row in dsub.iterrows():
            m   = METRIC_SHORT.get(row['metric'], row['metric'])
            chi = f"{row['statistic']:.4f}"
            p   = f"{row['p_value']:.6f}"
            sig = r'\checkmark' if row['significant'] else '---'
            if row['significant']:
                p = r'\textbf{' + p + r'}'
            if first:
                lines.append(
                    rf'\multirow{{{n}}}{{*}}{{{dataset}}} & '
                    rf'{m} & {chi} & {p} & {sig} \\'
                )
                first = False
            else:
                lines.append(
                    rf' & {m} & {chi} & {p} & {sig} \\'
                )
        lines.append(r'\hline')

    lines.append(r'\end{tabularx}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

def make_wilcoxon_latex(df_w, exp_label, n_levels):
    sub = df_w[df_w['exp'] == exp_label].copy()
    if sub.empty:
        return f"% 无 {exp_label} Wilcoxon 结果"

    tag      = exp_label.lower()
    min_p    = round(2 ** (-(n_levels - 1)), 4)
    min_bonf = round(min_p * 4, 4)

    caption = (
        rf'Pairwise Wilcoxon signed-rank tests comparing BDTA with '
        rf'each baseline in the {exp_label.lower()} sensitivity '
        rf'experiment ($n={n_levels}$ parameter levels). '
        rf'With $n={n_levels}$ paired observations, the minimum '
        rf'achievable two-sided $p$-value is ${min_p}$; after '
        rf'Bonferroni correction ($k=4$ comparisons per '
        rf'metric--dataset family), the minimum corrected '
        rf'$p$-value is ${min_bonf}$, which exceeds '
        rf'$\alpha=0.05$ by design. We therefore report the raw '
        rf'$p$-value, the Bonferroni-corrected $p$-value '
        rf'($p_{{\mathrm{{bonf}}}}$), and the matched-pairs '
        rf'rank-biserial correlation $r$ as the primary effect-size '
        rf'measure~\cite{{kerby2014simple}}. Positive $r$ indicates '
        rf'BDTA outperforms the baseline on higher-is-better metrics; '
        rf'negative $r$ indicates BDTA outperforms on lower-is-better '
        rf'metrics (\textit{{TSD}}). \textbf{{Bold}} $r$ values denote '
        rf'that the direction favours BDTA.'
    )

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(rf'\caption{{{caption}}}')
    lines.append(rf'\label{{tab:wilcoxon_{tag}}}')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\begin{tabularx}{\linewidth}{lllXXXXX}')
    lines.append(r'\hline')
    lines.append(
        r'Dataset & Metric & Comparison & BDTA & Baseline & '
        r'$p_{\mathrm{raw}}$ & $p_{\mathrm{bonf}}$ & $r$ \\'
    )
    lines.append(r'\hline')

    for dataset in ['Beijing', 'Chengdu']:
        dsub = sub[sub['dataset'] == dataset].copy()
        if dsub.empty:
            continue

        metric_order = list(METRIC_SHORT.keys())
        dsub['_mo']  = dsub['metric'].apply(
            lambda x: metric_order.index(x)
            if x in metric_order else 999
        )
        dsub = dsub.sort_values(['_mo', 'comparison']).drop(columns='_mo')

        first = True
        n     = len(dsub)

        for _, row in dsub.iterrows():
            m_str    = METRIC_SHORT.get(row['metric'], row['metric'])
            comp_str = row['comparison'].replace('BDTA vs ', '')
            b_str    = f"{row['bdta_mean']:.4f}"
            s_str    = f"{row['base_mean']:.4f}"
            p_raw    = row.get('p_value_raw',  float('nan'))
            p_bonf   = row.get('p_value_bonf', float('nan'))
            r_val    = row['effect_size_r']

            p_raw_str  = f"{p_raw:.4f}" if not np.isnan(p_raw) else '---'
            p_bonf_str = f"{p_bonf:.4f}" if not np.isnan(p_bonf) else '---'

            if not np.isnan(p_bonf) and p_bonf < ALPHA:
                p_bonf_str = r'\textbf{' + p_bonf_str + r'}'

            r_str = f"{r_val:+.4f}"
            if row.get('bdta_better', False):
                r_str = r'\textbf{' + r_str + r'}'

            if first:
                lines.append(
                    rf'\multirow{{{n}}}{{*}}{{{dataset}}} & '
                    rf'{m_str} & {comp_str} & {b_str} & {s_str} & '
                    rf'{p_raw_str} & {p_bonf_str} & {r_str} \\'
                )
                first = False
            else:
                lines.append(
                    rf' & {m_str} & {comp_str} & {b_str} & {s_str} & '
                    rf'{p_raw_str} & {p_bonf_str} & {r_str} \\'
                )

        lines.append(r'\hline')

    lines.append(r'\end{tabularx}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

def make_comparative_latex(df_comp):
    if df_comp.empty:
        return '% 无主实验统计摘要'

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(
        r'\caption{Summary of pairwise comparisons between BDTA and the '
        r'baseline methods in the main comparative experiment '
        r'($c(w)=6$). A \checkmark\ indicates BDTA outperforms the '
        r'corresponding baseline on the given metric; $\times$ indicates '
        r'the opposite. AFS, OAQ, and DSR are higher-is-better, whereas '
        r'TSD is lower-is-better.}'
    )
    lines.append(r'\label{tab:comparative_summary}')
    lines.append(r'\centering')
    lines.append(r'\begin{tabularx}{\linewidth}{llXXXX}')
    lines.append(r'\hline')
    lines.append(r'Dataset & Metric & PX & QITA & DTAA & FTSA \\')
    lines.append(r'\hline')

    for dataset in ['Beijing', 'Chengdu']:
        dsub = df_comp[df_comp['dataset'] == dataset]
        if dsub.empty:
            continue

        metrics_in = dsub['metric'].unique().tolist()
        first = True
        n     = len(metrics_in)

        for metric in metrics_in:
            m_str = METRIC_SHORT.get(metric, metric)
            cells = []
            for bl in BASELINES:
                row = dsub[
                    (dsub['metric'] == metric) &
                    (dsub['comparison'] == f'BDTA vs {bl}')
                ]
                if row.empty:
                    cells.append('---')
                else:
                    better = bool(row['bdta_better'].values[0])
                    cells.append(r'\checkmark' if better else r'$\times$')

            cell_str = ' & '.join(cells)
            if first:
                lines.append(
                    rf'\multirow{{{n}}}{{*}}{{{dataset}}} & '
                    rf'{m_str} & {cell_str} \\'
                )
                first = False
            else:
                lines.append(rf' & {m_str} & {cell_str} \\')

        lines.append(r'\hline')

    lines.append(r'\end{tabularx}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

sep = '\n' + '=' * 65 + '\n'

latex_blocks = {}

for exp in ['Capacity', 'Weight']:
    n_lv = 6 if exp == 'Capacity' else 5
    latex_blocks[f'friedman_{exp}'] = make_friedman_latex(
        df_friedman_all, exp)
    latex_blocks[f'wilcoxon_{exp}'] = make_wilcoxon_latex(
        df_wilcoxon_all, exp, n_lv)

latex_blocks['comparative_summary'] = make_comparative_latex(df_comparative)

for title, latex in latex_blocks.items():
    print()
    print()

latex_out = os.path.join(OUTPUT_DIR, 'statistical_tables_v3.tex')
with open(latex_out, 'w', encoding='utf-8') as f:
    for title, latex in latex_blocks.items():
        f.write(f'% ── {title} ──\n')
        f.write(latex + '\n\n')

print()
print()

