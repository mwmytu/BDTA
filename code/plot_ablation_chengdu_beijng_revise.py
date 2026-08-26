import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from matplotlib import rcParams

FONT_SIZE = 20
FIG_SIZE = (8.7, 7)

sns.set_theme(style="white")
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.unicode_minus'] = False

rcParams['font.size'] = FONT_SIZE
rcParams['axes.labelsize'] = FONT_SIZE
rcParams['xtick.labelsize'] = FONT_SIZE
rcParams['ytick.labelsize'] = FONT_SIZE
rcParams['legend.fontsize'] = FONT_SIZE

COLOR_PALETTE_HEX = {
    'nature_red':    '#E64B35',
    'nature_blue':   '#0072BD',
    'nature_green':  '#00A087',
    'nature_purple': '#76608A',
    'nature_orange': '#D55E00',
    'nature_yellow': '#CCBB44',
    'nature_magenta':'#CC79A7',
}

SCENARIO_LABELS = {
    "BDTA_FULL":          "BDTA",
    "w/o_AdaptiveW":      "w/o AW",
    "w/o_DynPriority":    "w/o DP",
    "w/o_RepBonus":       "w/o RB",
    "w/o_TwoStage":       "w/o TS",
    "w/o_RepInComposite": "w/o RC",
    "BASELINE":           "w/o_Rep",
}

SCENARIO_ORDER_MAIN = [
    "BDTA_FULL",
    "w/o_AdaptiveW",
    "w/o_DynPriority",
    "w/o_RepBonus",
    "w/o_TwoStage",
    "w/o_RepInComposite",
    "BASELINE",
]

SCENARIO_COLORS = {
    "BDTA":      COLOR_PALETTE_HEX['nature_blue'],
    "w/o AW":    COLOR_PALETTE_HEX['nature_red'],
    "w/o DP":    COLOR_PALETTE_HEX['nature_green'],
    "w/o RB":    COLOR_PALETTE_HEX['nature_purple'],
    "w/o TS":    COLOR_PALETTE_HEX['nature_orange'],
    "w/o RC":    COLOR_PALETTE_HEX['nature_yellow'],
    "w/o_Rep":   COLOR_PALETTE_HEX['nature_magenta'],
}

DATASET_CONFIGS = [
    {
        "name": "Beijing",
        "csv_path": r'D:\pythonProjectSiChuan - 副本 (2)\beijing_ablation_v10\beijing_ablation_v10.csv',
        "output_dir": r'D:\pythonProjectSiChuan - 副本 (2)\beijing_ablation_v10\ablation_graphs'
    },
    {
        "name": "Chengdu",
        "csv_path": r'D:\pythonProjectSiChuan - 副本 (2)\chengdu_ablation_v10\chengdu_ablation_v10.csv',
        "output_dir": r'D:\pythonProjectSiChuan - 副本 (2)\chengdu_ablation_v10\ablation_graphs'
    }
]

C_VALUE_TO_PLOT = 6

MAIN_METRICS = [
    {
        'key':       'avg_allocation_quality',
        'label':     'Allocation Quality',
        'direction': 'higher',
        'suffix':    'quality',
    },
    {
        'key':       'avg_fitness_score',
        'label':     'Fitness Score',
        'direction': 'higher',
        'suffix':    'fitness',
    },
    {
        'key':       'demand_satisfaction_rate',
        'label':     'Demand Satisfaction Rate',
        'direction': 'higher',
        'suffix':    'dsr',
    },
    {
        'key':       'travel_km_per_served_demand',
        'label':     'km / Demand',
        'direction': 'lower',
        'suffix':    'tsd',
    },
]

def _save_fig(fig, output_filename):
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    png_path = output_filename.replace('.pdf', '.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print()
    print()
    plt.close(fig)

def _prepare_data(df, scenario_order):
    d = df[df['scenario'].isin(scenario_order)].copy()
    d['_order'] = d['scenario'].apply(
        lambda x: scenario_order.index(x)
        if x in scenario_order else 999
    )
    return d.sort_values('_order')

def _make_ylabel(label, direction):
    if direction == 'higher':
        return f'{label} ↑'
    else:
        return f'{label} ↓'

def plot_ablation_bars(data, scenario_order,
                       y_axis_metric, y_axis_label,
                       direction,
                       output_filename,
                       figsize=FIG_SIZE):

    df_sorted = _prepare_data(data, scenario_order)
    if df_sorted.empty:
        print()
        return

    scenarios = df_sorted['scenario'].tolist()
    x_labels = [SCENARIO_LABELS.get(s, s) for s in scenarios]
    y_values = df_sorted[y_axis_metric].tolist()
    palette = [SCENARIO_COLORS.get(lbl, '#333333') for lbl in x_labels]

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(x_labels))

    bars = ax.bar(
        x_pos, y_values,
        color=palette,
        edgecolor='black',
        linewidth=1.5,
        width=0.62,
        zorder=3
    )

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            (bar.get_x() + bar.get_width() / 2., height),
            ha='center', va='bottom',
            xytext=(0, 4), textcoords='offset points',
            fontsize=FONT_SIZE, fontfamily='Arial', color='black'
        )

    ylabel = _make_ylabel(y_axis_label, direction)

    ax.set_xlabel('Ablation Variants',
                  fontsize=FONT_SIZE, fontfamily='Arial',
                  labelpad=10, color='black')
    ax.set_ylabel(ylabel,
                  fontsize=FONT_SIZE, fontfamily='Arial',
                  labelpad=12, color='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        x_labels,
        rotation=0,
        ha='center',
        fontsize=FONT_SIZE,
        fontfamily='Arial'
    )

    for label in ax.get_yticklabels():
        label.set_fontsize(FONT_SIZE)
        label.set_fontfamily('Arial')

    ax.set_xlim(-0.5, len(x_labels) - 0.5)

    ax.tick_params(
        axis='both', which='major', direction='in',
        length=6, width=1.4, colors='black',
        top=False, bottom=True, left=True, right=False, pad=6
    )
    ax.minorticks_off()

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.4)
        spine.set_visible(True)

    ax.grid(True, which='major', axis='y',
            linestyle='-', color='#E0E0E0', linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    min_val = min(y_values)
    max_val = max(y_values)

    if y_axis_metric == 'travel_km_per_served_demand':
        y_bottom = min_val * 0.88 if min_val > 0 else 0
        y_top = max_val * 1.12
    else:
        y_bottom = min_val * 0.70 if min_val > 0 else 0
        y_top = max_val * 1.18

    ax.set_ylim(bottom=y_bottom, top=y_top)

    plt.tight_layout()
    _save_fig(fig, output_filename)

def generate_ablation_plots_for_dataset(dataset_name,
                                        csv_path,
                                        output_dir):
    print()
    print()
    print()

    if not os.path.exists(csv_path):
        print()
        return

    try:
        df_full = pd.read_csv(csv_path)
        df_plot = df_full[df_full['c_value'] == C_VALUE_TO_PLOT].copy()
        if df_plot.empty:
            print()
            return
        print()
        print()
    except Exception as e:
        print()
        return

    os.makedirs(output_dir, exist_ok=True)
    dn = dataset_name.lower()

    print()
    for m in MAIN_METRICS:
        plot_ablation_bars(
            data=df_plot,
            scenario_order=SCENARIO_ORDER_MAIN,
            y_axis_metric=m['key'],
            y_axis_label=m['label'],
            direction=m['direction'],
            output_filename=os.path.join(
                output_dir,
                f"Ablation_{m['suffix']}_{dn}.pdf"
            )
        )

    print()

if __name__ == "__main__":
    for config in DATASET_CONFIGS:
        generate_ablation_plots_for_dataset(
            dataset_name=config["name"],
            csv_path=config["csv_path"],
            output_dir=config["output_dir"]
        )

    print()
    print()
    print()