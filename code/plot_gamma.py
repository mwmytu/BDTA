import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib import rcParams
import os
import numpy as np

sns.set_theme(style="white", font_scale=1.4)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.unicode_minus'] = False

BJ_CSV = r'D:\pythonProjectSiChuan - 副本 (2)\bdta_sensitivity_gamma_results\beijing_sensitivity_gamma_BDTA.csv'
SC_CSV = r'D:\pythonProjectSiChuan - 副本 (2)\bdta_sensitivity_gamma_results\chengdu_sensitivity_gamma_BDTA.csv'
OUTPUT_DIR = r'D:\pythonProjectSiChuan - 副本 (2)\bdta_sensitivity_gamma_results\gamma_graphs'

os.makedirs(OUTPUT_DIR, exist_ok=True)

df_bj = pd.read_csv(BJ_CSV)
df_sc = pd.read_csv(SC_CSV)

COLOR_BJ = '#0072BD'
COLOR_SC = '#D95319'
GAMMA_OPT = 1.2

def apply_spine_style(ax):
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_visible(True)

def plot_gamma_sensitivity(metric_key, y_label, filename,
                           y_fmt='%.1f',
                           legend_pos=(0.04, 0.96)):
    fig, ax = plt.subplots(figsize=(6.8, 6))

    ax.plot(
        df_bj['gamma'], df_bj[metric_key],
        color=COLOR_BJ, marker='o', linewidth=2.0,
        markersize=7, label='Beijing', zorder=4
    )
    ax.plot(
        df_sc['gamma'], df_sc[metric_key],
        color=COLOR_SC, marker='s', linewidth=2.0,
        markersize=7, label='Chengdu', zorder=4
    )

    ax.set_xlabel(
        'γ',
        fontsize=18, fontfamily='Arial', labelpad=10
    )
    ax.set_ylabel(
        y_label,
        fontsize=18, fontfamily='Arial', color='black', labelpad=12
    )

    ax.set_xticks(df_bj['gamma'].tolist())
    ax.set_xticklabels(
        [str(g) for g in df_bj['gamma'].tolist()],
        fontsize=17, fontfamily='Arial'
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(y_fmt))
    plt.yticks(fontsize=17, fontfamily='Arial')

    ax.tick_params(
        axis='x', which='major', direction='in',
        length=6, width=1.5, colors='black',
        bottom=True, top=False, pad=6
    )
    ax.tick_params(
        axis='y', which='major', direction='in',
        length=6, width=1.5, colors='black',
        left=True, right=False, pad=6
    )
    ax.minorticks_off()

    apply_spine_style(ax)

    ax.grid(
        True, which='major', axis='y',
        linestyle='-', color='#E0E0E0', linewidth=1, zorder=0
    )
    ax.set_axisbelow(True)

    all_vals = df_bj[metric_key].tolist() + df_sc[metric_key].tolist()
    y_min, y_max = min(all_vals), max(all_vals)
    margin = max((y_max - y_min) * 0.15, 0.001)
    ax.set_ylim(y_min - margin, y_max + margin)

    leg = ax.legend(
        loc='upper left',
        bbox_to_anchor=legend_pos,
        fontsize=17,
        frameon=True,
        edgecolor='black',
        fancybox=False
    )
    leg.set_zorder(10)
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_linewidth(1.5)

    fig.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.12)

    pdf_path = os.path.join(OUTPUT_DIR, filename + '.pdf')
    png_path = os.path.join(OUTPUT_DIR, filename + '.png')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print()
    plt.close()

print()

plot_gamma_sensitivity(
    metric_key='avg_allocation_quality',
    y_label='Allocation Quality ↑',
    filename='fig5_gamma_oaq',
    y_fmt='%.3f',
    legend_pos=(0.03, 0.98)
)

plot_gamma_sensitivity(
    metric_key='avg_fitness_score',
    y_label='Fitness Score ↑',
    filename='fig5_gamma_fitness',
    y_fmt='%.2f',
    legend_pos=(0.03, 0.99)
)

plot_gamma_sensitivity(
    metric_key='task_coverage',
    y_label='Task Coverage ↑',
    filename='fig5_gamma_task_coverage',
    y_fmt='%.1f',
    legend_pos=(0.04, 0.96)
)

print()
print()

print()
print()
print()
print()
print()

for _, bj_row in df_bj.iterrows():
    g = bj_row['gamma']
    sc_row = df_sc[df_sc['gamma'] == g]
    if sc_row.empty:
        continue
    sc_row = sc_row.iloc[0]
    marker = " ← selected" if np.isclose(g, GAMMA_OPT) else ""
    print()

print()
print()
print()