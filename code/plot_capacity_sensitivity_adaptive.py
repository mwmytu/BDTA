

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter

DATASET = 'Chengdu'

DATA_CONFIG = {
    'Beijing': {
        'subdir':   'beijing_capacity_combined_results',
        'filename': 'beijing_capacity_combined.csv',
        'output':   'beijing_capacity_graphs',
    },
    'Chengdu': {
        'subdir':   'chengdu_capacity_combined_results',
        'filename': 'chengdu_capacity_combined.csv',
        'output':   'chengdu_capacity_graphs',
    },
}

SCENARIO_DISPLAY = {
    'BDTA': 'BDTA',
    'PX':   'PX',
    'QITA': 'QITA',
    'DTAA': 'DTAA',
    'FTSA': 'FTSA',
}

COLOR_MAP = {
    'BDTA': '#D95319',
    'PX':   '#0072BD',
    'QITA': '#7E2F8E',
    'DTAA': '#EDB120',
    'FTSA': '#77AC30',
}
MARKER_MAP = {
    'BDTA': 'o',
    'PX':   's',
    'QITA': 'X',
    'DTAA': '^',
    'FTSA': 'v',
}

METRICS_CONFIG = [
    ('avg_fitness_score',
     'Fitness Score ↑',
     'up'),
    ('avg_allocation_quality',
     'Allocation Quality ↑',
     'up'),
    ('demand_satisfaction_rate',
     'Demand Satisfaction Rate ↑',
     'up'),
    ('on_time_demand_satisfaction_rate',
     'OT-DSR ↑',
     'up'),
    ('travel_km_per_served_demand',
     'km / Demand ↓',
     'down'),
    ('served_demand_per_km',
     'Demand / km ↑',
     'up'),
]

AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 20
LEGEND_SIZE     = 16

LEGEND_LOC = {
    'avg_fitness_score':                'upper right',
    'avg_allocation_quality':           'upper right',
    'demand_satisfaction_rate':         'upper left',
    'on_time_demand_satisfaction_rate': 'upper left',
    'travel_km_per_served_demand':      'upper left',
    'served_demand_per_km':             'upper right',
}

def plot_single_metric(df, metric_col, y_label, ax, show_legend=True):
    for scenario_key in SCENARIO_DISPLAY:
        label = SCENARIO_DISPLAY[scenario_key]
        sub   = df[df['scenario_label'] == label].sort_values('c_value')
        if sub.empty:
            continue
        ax.plot(
            sub['c_value'],
            sub[metric_col],
            color      = COLOR_MAP[label],
            marker     = MARKER_MAP[label],
            markersize = 10,
            linewidth  = 2.5,
            label      = label,
        )

    ax.set_xlabel('Maximum Tasks per Taxi',
                  fontsize=AXIS_LABEL_SIZE, labelpad=8)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE, labelpad=8)

    x_vals = sorted(df['c_value'].unique())
    ax.set_xticks(x_vals)
    ax.get_xaxis().set_major_formatter(
        FuncFormatter(lambda x, p: str(int(x))))
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
        loc = LEGEND_LOC.get(metric_col, 'best')
        leg = ax.legend(
            fontsize   = LEGEND_SIZE,
            loc        = loc,
            frameon    = True,
            edgecolor  = 'black',
            framealpha = 0.8,
            fancybox   = False,
        )
        leg.get_frame().set_linewidth(0.8)

def plot_individual(df, output_dir, dataset: str):
    prefix = 'beijing_' if dataset == 'Beijing' else 'chengdu_'
    print()

    for metric_col, y_label, _ in METRICS_CONFIG:
        if metric_col not in df.columns:
            print()
            continue

        fig, ax = plt.subplots(figsize=(7, 6))
        plot_single_metric(df, metric_col, y_label, ax)
        plt.tight_layout()

        base = f"{prefix}capacity_{metric_col}"
        for ext in ['png', 'pdf']:
            path = os.path.join(output_dir, f"{base}.{ext}")
            plt.savefig(path, dpi=300, bbox_inches='tight')
        print()
        plt.close()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    OLD_TO_NEW = {
        'OUR_ENHANCED_MODEL_GRID':  'BDTA',
        'PAPER_X_SIMULATION_GRID':  'PX',
        'QITA_QUALITY_MODEL_GRID':  'QITA',
        'MULTI_OBJECTIVE_SIM_GRID': 'DTAA',
        'FTSA_OTAGA_GRID':          'FTSA',
    }

    if 'scenario_label' not in df.columns:
        if 'scenario' in df.columns:
            df['scenario_label'] = (
                df['scenario']
                .map(OLD_TO_NEW)
                .fillna(df['scenario']))
        else:
            raise ValueError("CSV中既没有 'scenario_label' 也没有 'scenario' 列")
    else:
        df['scenario_label'] = (
            df['scenario_label']
            .map(lambda x: OLD_TO_NEW.get(x, x)))

    valid = list(SCENARIO_DISPLAY.values())
    df    = df[df['scenario_label'].isin(valid)].copy()

    if 'c_value' not in df.columns:
        for alt in ['capacity', 'max_tasks', 'c']:
            if alt in df.columns:
                df = df.rename(columns={alt: 'c_value'})
                break

    df = df.sort_values('c_value')
    return df

if __name__ == '__main__':
    sns.set_theme(style='whitegrid', font_scale=1.0)
    rcParams['font.family']     = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial']

    _cfg        = DATA_CONFIG[DATASET]
    _script_dir = os.path.dirname(os.path.abspath(__file__))

    RESULTS_CSV      = os.path.join(_script_dir, _cfg['subdir'], _cfg['filename'])
    GRAPH_OUTPUT_DIR = os.path.join(_script_dir, _cfg['output'])

    if not os.path.exists(RESULTS_CSV):
        fallback = os.path.join(_cfg['subdir'], _cfg['filename'])
        if os.path.exists(fallback):
            RESULTS_CSV = fallback
        else:
            print()
            print()
            print()
            raise FileNotFoundError(
                f"请确认 CSV 文件存在: {RESULTS_CSV}"
            )

    print()
    df_raw = pd.read_csv(RESULTS_CSV)
    print()

    df = preprocess(df_raw)
    print()
    print()
    print()

    os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

    plot_individual(df, GRAPH_OUTPUT_DIR, DATASET)

    print()