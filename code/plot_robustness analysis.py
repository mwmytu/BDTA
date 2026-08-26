import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

CSV_PATH = 'robustness_results/robustness_missing_gps_results.csv'
OUTPUT_DIR = 'robustness_results/robustness_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS_TO_PLOT = {
    'demand_satisfaction_rate': {
        'label': r'DSR $\uparrow$',
        'filename': 'robustness_dsr',
    },
    'avg_fitness_score': {
        'label': r'AFS $\uparrow$',
        'filename': 'robustness_fitness',
    },
    'avg_allocation_quality': {
        'label': r'OAQ $\uparrow$',
        'filename': 'robustness_quality',
    },
}

DATASET_ORDER = ['Beijing', 'Chengdu']

COLOR_MAP = {
    'Beijing': '#0072BD',
    'Chengdu': '#D95319',
}

MARKER_MAP = {
    'Beijing': 'o',
    'Chengdu': 's',
}

sns.set_theme(style="whitegrid", font_scale=1.4)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

AXIS_LABEL_SIZE = 27
TICK_LABEL_SIZE = 26
LEGEND_SIZE = 20
GRID_GRAY = '#E0E0E0'

def compute_ylim(vals, padding_ratio=0.08):
    vmin = float(vals.min())
    vmax = float(vals.max())
    vrng = vmax - vmin
    if vrng < 0.02:
        vrng = 0.02
    pad = vrng * padding_ratio
    y0 = max(0.0, vmin - pad)
    y1 = vmax + pad
    return y0, y1

def pct_formatter(x, pos):
    return f'{int(x)}%'

def plot_single_robustness(df, metric_col, y_label, save_name):
    df = df.copy()
    df['missing_rate_pct'] = df['missing_rate'] * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    vals = df[metric_col].dropna()
    y_lo, y_hi = compute_ylim(vals)

    for dataset in DATASET_ORDER:
        sub = df[df['dataset'] == dataset].sort_values('missing_rate_pct')
        if sub.empty:
            continue

        ax.plot(
            sub['missing_rate_pct'],
            sub[metric_col],
            color=COLOR_MAP[dataset],
            marker=MARKER_MAP[dataset],
            markersize=13,
            linewidth=2.5,
            label=dataset
        )

    ax.set_ylabel(y_label,
                  fontsize=AXIS_LABEL_SIZE,
                  color='black',
                  labelpad=12)

    ax.set_xlabel('GPS missing rate',
                  fontsize=AXIS_LABEL_SIZE,
                  color='black',
                  labelpad=12)

    ax.set_xticks([0, 20, 40, 60, 80])
    ax.xaxis.set_major_formatter(FuncFormatter(pct_formatter))

    ax.set_ylim(y_lo, y_hi)

    ax.minorticks_off()

    ax.tick_params(
        axis='both',
        which='major',
        direction='in',
        length=6,
        width=1.5,
        colors='black',
        top=True,
        bottom=True,
        left=True,
        right=True,
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
        loc='center',
        bbox_to_anchor=(0.80, 0.90),
        frameon=True,
        edgecolor=GRID_GRAY,
        framealpha=0.6,
        fancybox=False,
        ncol=1
    )
    leg.get_frame().set_linewidth(0.8)
    leg.get_frame().set_edgecolor('black')

    png_path = os.path.join(OUTPUT_DIR, f'{save_name}.png')
    pdf_path = os.path.join(OUTPUT_DIR, f'{save_name}.pdf')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print()

if __name__ == '__main__':
    df = pd.read_csv(CSV_PATH)

    for metric_col, cfg in METRICS_TO_PLOT.items():
        plot_single_robustness(
            df,
            metric_col=metric_col,
            y_label=cfg['label'],
            save_name=cfg['filename'],
        )

    print()