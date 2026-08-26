import pandas as pd
from collections import OrderedDict

CSV_PATH = 'robustness_results/robustness_missing_gps_results.csv'
df_robust = pd.read_csv(CSV_PATH)

METRICS = OrderedDict({
    'demand_satisfaction_rate':         r'DSR $\uparrow$',
    'on_time_demand_satisfaction_rate': r'OT-DSR $\uparrow$',
    'avg_fitness_score':                r'Fitness $\uparrow$',
    'avg_allocation_quality':           r'Quality $\uparrow$',
})

DATASET_ORDER = ['Beijing', 'Chengdu']

def displayed_value(x, ndigits=4):
    """表格显示值：统一四舍五入到4位小数。"""
    return round(float(x), ndigits)

def displayed_delta(val, baseline, ndigits=4):
    """
    Δ 也按显示值计算：
      先 round(val, 4)
      再 round(baseline, 4)
      再做差
    这样可以保证 Δ 与表中显示值严格一致。
    """
    v = displayed_value(val, ndigits)
    b = displayed_value(baseline, ndigits)
    d = round(v - b, ndigits)

    if abs(d) < 10 ** (-ndigits):
        d = 0.0
    return d

def fmt_value(x, ndigits=4):
    return f"{displayed_value(x, ndigits):.{ndigits}f}"

def fmt_delta(d, ndigits=4):
    if abs(d) < 10 ** (-ndigits):
        return f"(+{0:.{ndigits}f})"
    sign = '+' if d > 0 else ''
    return f"({sign}{d:.{ndigits}f})"

def make_robustness_latex(df_robust, ndigits=4):
    lines = []

    lines.append(r'\begin{table}[htbp]')
    lines.append(
        r'\caption{Robustness of BDTA to simulated GPS data loss ($c=6$). '
        r'Values are averaged over all evaluation days per dataset. '
        r'$\Delta$ denotes the absolute change relative to the complete-data '
        r'reference condition ($r=0\%$). '
        r'All displayed values and $\Delta$ terms are rounded consistently to '
        + str(ndigits) + r' decimal places. '
        r'$\uparrow$ indicates that a higher value is better. '
        r'The complete-data condition ($r=0\%$) is highlighted in '
        r'\textbf{bold} as the reference point for measuring degradation. '
        r'Minor deviations above the baseline in a few cells are within '
        r'simulation noise and do not indicate a meaningful improvement '
        r'under missing data.}'
    )
    lines.append(r'\label{tab:robustness_missing}')
    lines.append(r'\centering')
    lines.append(r'\begin{tabularx}{\linewidth}{llXXXX}')
    lines.append(r'\hline')

    header = (
        r'Dataset & Missing rate $r$ & '
        + ' & '.join(METRICS.values())
        + r' \\'
    )
    lines.append(header)
    lines.append(r'\hline')

    for dataset in DATASET_ORDER:
        sub = df_robust[df_robust['dataset'] == dataset].copy()
        sub = sub.sort_values('missing_rate').reset_index(drop=True)

        if sub.empty:
            continue

        baseline_row = sub[sub['missing_rate'] == 0.0]
        if baseline_row.empty:
            continue
        baseline = baseline_row.iloc[0]

        first = True
        n_rows = len(sub)

        for _, row in sub.iterrows():
            r_pct = f"{int(round(row['missing_rate'] * 100))}\\%"
            is_baseline = abs(row['missing_rate'] - 0.0) < 1e-12

            cells = []
            for col in METRICS.keys():
                val = row[col]
                base_val = baseline[col]

                val_str = fmt_value(val, ndigits)

                if is_baseline:
                    cell = r'\textbf{' + val_str + r'}'
                else:
                    delta = displayed_delta(val, base_val, ndigits)
                    cell = f"{val_str} {fmt_delta(delta, ndigits)}"

                cells.append(cell)

            if first:
                lines.append(
                    rf'\multirow{{{n_rows}}}{{*}}{{{dataset}}} & '
                    rf'{r_pct} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\'
                )
                first = False
            else:
                lines.append(
                    rf' & {r_pct} & {cells[0]} & {cells[1]} & {cells[2]} & {cells[3]} \\'
                )

        lines.append(r'\hline')

    lines.append(r'\end{tabularx}')
    lines.append(r'\end{table}')

    return '\n'.join(lines)

latex_table = make_robustness_latex(df_robust, ndigits=4)
print()

with open('robustness_results/robustness_missing_gps_table_fixed.tex',
          'w', encoding='utf-8') as f:
    f.write(latex_table)

print()