import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.lines
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter, FixedLocator, NullLocator

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

CSV_FILE_PATH = r'chengdu_scalability_results\chengdu_scalability_results2.csv'
OUTPUT_FILENAME_WORKERS = 'scalability_vs_workers_chengdu_final_v4.pdf'
OUTPUT_FILENAME_TASKS   = 'scalability_vs_tasks_chengdu_final_v4.pdf'

if not os.path.exists(CSV_FILE_PATH):
    print()
    exit()
try:
    df_full    = pd.read_csv(CSV_FILE_PATH)
    df_workers = df_full[df_full['experiment_type'] == 'Workers'].copy()
    df_tasks   = df_full[df_full['experiment_type'] == 'Tasks'].copy()
    print()
    print()
    print()
    print()
except Exception as e:
    print()
    exit()

scenario_labels = {
    "OUR_ENHANCED_MODEL_GRID": "BDTA",
    "BDTA-noPrune":            "BDTA w/o Pruning",
    "PAPER_X_SIMULATION_GRID": "PAPER-X",
    "QITA_QUALITY_MODEL_GRID": "QITA",
    "MULTI_OBJECTIVE_SIM_GRID":"DTAA",
    "FTSA_OTAGA_GRID":         "FTSA",
}

plot_order_full = ["BDTA", "BDTA w/o Pruning", "PAPER-X", "QITA", "DTAA", "FTSA"]

deep_palette = sns.color_palette("deep", n_colors=len(plot_order_full))
color_map = {
    "BDTA":             deep_palette[3],
    "DTAA":             deep_palette[2],
    "BDTA w/o Pruning": deep_palette[4],
    "PAPER-X":          deep_palette[1],
    "QITA":             deep_palette[0],
    "FTSA":             deep_palette[5],
}
markers = {
    "BDTA":             "o",
    "BDTA w/o Pruning": "X",
    "PAPER-X":          "s",
    "QITA":             "^",
    "DTAA":             "D",
    "FTSA":             "v",
}

def plot_scalability_chart(data, x_axis_col, x_axis_label,
                           y_axis_col, y_axis_label,
                           output_filename, title=None):
    print()

    data = data.copy()
    data['scenario_label'] = data['scenario'].map(scenario_labels)

    present_labels = data['scenario_label'].dropna().unique().tolist()
    plot_order     = [s for s in plot_order_full if s in present_labels]
    print()

    data = data[data['scenario_label'].isin(plot_order)]

    sns.set_theme(style="white", font_scale=1.3)
    plt.figure(figsize=(10, 7))
    ax = sns.lineplot(
        data=data,
        x=x_axis_col, y=y_axis_col,
        hue='scenario_label', hue_order=plot_order,
        style='scenario_label', style_order=plot_order,
        markers={k: v for k, v in markers.items() if k in plot_order},
        markersize=10, linewidth=3.0,
        palette={k: v for k, v in color_map.items() if k in plot_order},
        legend='full',
    )
    for line in ax.lines:
        line.set_linestyle("-")

    ax.set_yscale('log')

    ax.set_xticks(sorted(data[x_axis_col].unique()))

    ymin, ymax   = ax.get_ylim()
    num_y_ticks  = 7
    tick_values  = np.logspace(np.log10(ymin), np.log10(ymax), num=num_y_ticks)
    ax.yaxis.set_major_locator(FixedLocator(tick_values))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))

    ax.yaxis.set_minor_locator(NullLocator())

    ax.grid(True, which='major', linestyle='-', linewidth='0.8', color='lightgray')

    if title:
        ax.set_title(title, fontsize=18, weight='bold', pad=20)
    ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel(y_axis_label, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=13)

    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        if isinstance(handle, matplotlib.lines.Line2D):
            handle.set_linestyle("-")
    ax.legend(handles=handles, labels=labels,
               fontsize=12, frameon=True, facecolor='white', framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print()
    plt.show()
    plt.close()

plot_scalability_chart(
    data=df_workers,
    x_axis_col='num_workers',
    x_axis_label='Number of Workers',
    y_axis_col='avg_decision_time_ms',
    y_axis_label='Average Decision Time (ms, log scale)',
    output_filename=OUTPUT_FILENAME_WORKERS,
)
plot_scalability_chart(
    data=df_tasks,
    x_axis_col='num_tasks',
    x_axis_label='Number of Tasks',
    y_axis_col='avg_decision_time_ms',
    y_axis_label='Average Decision Time (ms, log scale)',
    output_filename=OUTPUT_FILENAME_TASKS,
)
print()