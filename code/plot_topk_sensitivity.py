

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

BEIJING_CSV = 'beijing_topk_time_results/beijing_topk_time_comparison.csv'
CHENGDU_CSV = 'chengdu_topk_time_results/chengdu_topk_time_comparison.csv'

OUTPUT_DIR = 'topk_time_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_NAME = 'topk_time_tasks_only'

DATASET_ORDER = ['Beijing', 'Chengdu']
TOPK_ORDER = [0.8, 0.9]

LINE_CONFIG = {
    ('Beijing', 0.8): {
        'color':  '#0072BD',
        'marker': 'o',
        'label':  r'Beijing, $\rho=0.8$',
    },
    ('Beijing', 0.9): {
        'color':  '#77AC30',
        'marker': 'D',
        'label':  r'Beijing, $\rho=0.9$',
    },
    ('Chengdu', 0.8): {
        'color':  '#D95319',
        'marker': 's',
        'label':  r'Chengdu, $\rho=0.8$',
    },
    ('Chengdu', 0.9): {
        'color':  '#7E2F8E',
        'marker': '^',
        'label':  r'Chengdu, $\rho=0.9$',
    },
}

sns.set_theme(style="whitegrid", font_scale=1.4)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 20
LEGEND_SIZE = 18
GRID_GRAY = '#E0E0E0'

def compute_ylim(vals, padding_ratio=0.08):
    vmin = float(vals.min())
    vmax = float(vals.max())
    vrng = vmax - vmin
    if vrng < 1.0:
        vrng = 1.0
    pad = vrng * padding_ratio
    y0 = max(0.0, vmin - pad)
    y1 = vmax + pad
    return y0, y1

def int_formatter(x, pos):
    return f'{int(x)}'

def load_and_merge():
    dfs = []

    if os.path.exists(BEIJING_CSV):
        df_bj = pd.read_csv(BEIJING_CSV)
        df_bj['dataset'] = 'Beijing'
        dfs.append(df_bj)
    else:
        print()

    if os.path.exists(CHENGDU_CSV):
        df_sc = pd.read_csv(CHENGDU_CSV)
        df_sc['dataset'] = 'Chengdu'
        dfs.append(df_sc)
    else:
        print()

    if not dfs:
        raise FileNotFoundError('未找到任何 topk 时间对比结果文件。')

    return pd.concat(dfs, ignore_index=True)

def plot_task_time_only(df):
    df = df[df['experiment_type'] == 'Tasks'].copy()

    fig, ax = plt.subplots(figsize=(7.5, 7))

    vals = df['avg_decision_time_ms'].dropna()
    y_lo, y_hi = compute_ylim(vals)

    for dataset in DATASET_ORDER:
        for rho in TOPK_ORDER:
            key = (dataset, rho)
            cfg = LINE_CONFIG[key]

            sub = df[
                (df['dataset'] == dataset) &
                (df['topk_ratio'].round(1) == rho)
            ].sort_values('num_tasks')

            if sub.empty:
                continue

            ax.plot(
                sub['num_tasks'],
                sub['avg_decision_time_ms'],
                color=cfg['color'],
                marker=cfg['marker'],
                linestyle='-',
                markersize=13,
                linewidth=2.5,
                label=cfg['label'],
            )

    ax.set_ylabel(r'Decision time (ms) $\downarrow$',
                  fontsize=AXIS_LABEL_SIZE,
                  color='black',
                  labelpad=12)

    ax.set_xlabel('Number of tasks',
                  fontsize=AXIS_LABEL_SIZE,
                  color='black',
                  labelpad=12)

    ax.set_xticks([400, 800, 1200, 1600, 2000])
    ax.xaxis.set_major_formatter(FuncFormatter(int_formatter))

    ax.set_ylim(y_lo, y_hi)

    ax.minorticks_off()

    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=6,
        width=1.5,
        colors='black',
        top=False,
        bottom=True,
        left=True,
        right=False,
        labelsize=TICK_LABEL_SIZE
    )

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_visible(True)

    ax.grid(True, which='major', linestyle='-',
            color=GRID_GRAY, linewidth=1)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles=handles,
        labels=labels,
        title=None,
        fontsize=LEGEND_SIZE,
        loc='upper left',
        frameon=True,
        edgecolor=GRID_GRAY,
        framealpha=0.6,
        fancybox=False,
        ncol=1
    )
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor('black')

    png_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_NAME}.png')
    pdf_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_NAME}.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print()

def print_task_summary(df):
    df = df[df['experiment_type'] == 'Tasks'].copy()

    print()
    print()
    print()

    for dataset in DATASET_ORDER:
        sub_ds = df[df['dataset'] == dataset]
        if sub_ds.empty:
            continue

        print()
        print()

        for scale in sorted(sub_ds['num_tasks'].unique()):
            r08 = sub_ds[
                (sub_ds['num_tasks'] == scale) &
                (sub_ds['topk_ratio'].round(1) == 0.8)
            ]
            r09 = sub_ds[
                (sub_ds['num_tasks'] == scale) &
                (sub_ds['topk_ratio'].round(1) == 0.9)
            ]

            t08 = (r08['avg_decision_time_ms'].values[0]
                   if not r08.empty else float('nan'))
            t09 = (r09['avg_decision_time_ms'].values[0]
                   if not r09.empty else float('nan'))
            ratio = t09 / t08 if t08 > 0 else float('nan')

            print()

if __name__ == '__main__':
    df_all = load_and_merge()
    plot_task_time_only(df_all)
    print_task_summary(df_all)
    print()