

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

DATASET = 'Chengdu'

DATA_CONFIG = {
    'Beijing': {
        'subdir': 'beijing_weight_combined_results',
        'filename': 'beijing_weight_combined.csv',
        'output': 'beijing_weight_graphs',
    },
    'Chengdu': {
        'subdir': 'chengdu_weight_combined_results',
        'filename': 'chengdu_weight_combined.csv',
        'output': 'chengdu_weight_graphs',
    },
}

SCENARIO_DISPLAY = {
    'BDTA': 'BDTA',
    'PX': 'PX',
    'QITA': 'QITA',
    'DTAA': 'DTAA',
    'FTSA': 'FTSA',
}

COLOR_MAP = {
    'BDTA': '#D95319',
    'PX': '#0072BD',
    'QITA': '#7E2F8E',
    'DTAA': '#EDB120',
    'FTSA': '#77AC30',
}
MARKER_MAP = {
    'BDTA': 'o',
    'PX': 's',
    'QITA': 'X',
    'DTAA': '^',
    'FTSA': 'v',
}

METRICS_CONFIG = [
    ('avg_fitness_score',
     'Fitness Score ↑',
     'lower right'),
    ('avg_allocation_quality',
     'Allocation Quality ↑',
     'lower right'),
    ('demand_satisfaction_rate',
     'DSR ↑',
     'upper left'),
    ('travel_km_per_served_demand',
     'km / Demand ↓',
     'upper left'),
    ('served_demand_per_km',
     'Demand / km ↑',
     'lower right'),
]

AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 20
LEGEND_SIZE = 16

OLD_TO_NEW = {
    'OUR_ENHANCED_MODEL_GRID': 'BDTA',
    'PAPER_X_SIMULATION_GRID': 'PX',
    'QITA_QUALITY_MODEL_GRID': 'QITA',
    'MULTI_OBJECTIVE_SIM_GRID': 'DTAA',
    'FTSA_OTAGA_GRID': 'FTSA',
}

def preprocess(df: pd.DataFrame, dataset: str) -> pd.DataFrame:

    if 'scenario_label' not in df.columns:
        if 'scenario' in df.columns:
            df['scenario_label'] = (
                df['scenario']
                .map(OLD_TO_NEW)
                .fillna(df['scenario']))
        else:
            raise ValueError("CSV中既没有 'scenario_label' 也没有 'scenario' 列")
    else:
        df['scenario_label'] = df['scenario_label'].map(
            lambda x: OLD_TO_NEW.get(x, x))

    valid = list(SCENARIO_DISPLAY.values())
    df = df[df['scenario_label'].isin(valid)].copy()

    if 'dataset' in df.columns:
        df = df[df['dataset'] == dataset].copy()

    if 'w_task' not in df.columns:
        for alt in ['weight', 'w', 'w_value']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'w_task'})
                break

    df['w_task'] = pd.to_numeric(df['w_task'], errors='coerce')
    df = df.dropna(subset=['w_task'])
    df = df.sort_values('w_task')
    return df

def plot_single_metric(df, metric_col, y_label, legend_loc, ax,
                       show_legend=True):
    """在给定 ax 上绘制单个指标，图顶不显示任何标题文字。"""

    for scenario_key in SCENARIO_DISPLAY:
        label = SCENARIO_DISPLAY[scenario_key]
        sub   = df[df['scenario_label'] == label].sort_values('w_task')
        if sub.empty:
            continue
        ax.plot(
            sub['w_task'],
            sub[metric_col],
            color      = COLOR_MAP[label],
            marker     = MARKER_MAP[label],
            markersize = 10,
            linewidth  = 2.5,
            label      = label,
        )

    ax.set_xlabel('Task Priority Weight',
                  fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, labelpad=8)

    x_vals = sorted(df['w_task'].unique())
    ax.set_xticks(x_vals)
    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: f"{x:.1f}"))
    ax.minorticks_off()

    ax.tick_params(
        axis='both', which='major',
        direction='in', length=5, width=1.2,
        colors='black',
        bottom=True, top=False,
        left=True, right=False,
        labelsize=TICK_LABEL_SIZE
    )
    ax.tick_params(axis='y', which='both', right=False, labelright=False)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
        spine.set_visible(True)

    ax.grid(True, which='major', linestyle='-',
            color='#E0E0E0', linewidth=0.8)
    ax.set_axisbelow(True)

    if show_legend:
        leg = ax.legend(
            fontsize   = LEGEND_SIZE,
            loc        = legend_loc,
            frameon    = True,
            edgecolor  = 'black',
            framealpha = 0.8,
            fancybox   = False,
        )
        leg.get_frame().set_linewidth(0.8)

def plot_individual(df, output_dir, dataset: str):

    prefix = 'beijing_' if dataset == 'Beijing' else 'chengdu_'
    print()

    for metric_col, y_label, legend_loc in METRICS_CONFIG:
        if metric_col not in df.columns:
            print()
            continue

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_single_metric(df, metric_col, y_label, legend_loc, ax)
        plt.tight_layout()

        base = f"{prefix}weight_{metric_col}"
        for ext in ['png', 'pdf']:
            path = os.path.join(output_dir, f"{base}.{ext}")
            plt.savefig(path, dpi=300, bbox_inches='tight')
        print()
        plt.close()

if __name__ == '__main__':
    sns.set_theme(style='whitegrid', font_scale=1.0)
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    cfg = DATA_CONFIG.get(DATASET)
    if cfg is None:
        raise ValueError(
            f"DATASET 必须为 'Beijing' 或 'Chengdu'，当前值: {DATASET}")

    GRAPH_OUTPUT_DIR = cfg['output']

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    candidates = [
        os.path.join(script_dir, cfg['subdir'], cfg['filename']),
        os.path.join(script_dir, cfg['filename']),
        cfg['filename'],
    ]

    csv_path = None
    for c in candidates:
        if os.path.exists(c):
            csv_path = c
            break

    if csv_path is None:
        print()
        for c in candidates:
            print()
        raise FileNotFoundError(cfg['filename'])

    print()
    df_raw = pd.read_csv(csv_path)
    print()

    df = preprocess(df_raw, DATASET)
    print()
    print()
    print()
    print()

    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

    plot_individual(df, GRAPH_OUTPUT_DIR, DATASET)

    print()