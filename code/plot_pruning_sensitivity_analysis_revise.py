import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = 'bdta_sensitivity_pruning_results'
os.makedirs(save_dir, exist_ok=True)

bj_path = os.path.join(save_dir, 'beijing_sensitivity_pruning.csv')
sc_path = os.path.join(save_dir, 'chengdu_sensitivity_pruning.csv')

bj = pd.read_csv(bj_path)
sc = pd.read_csv(sc_path)

bj['decision_time_sec'] = bj['avg_decision_time_ms'] / 1000
sc['decision_time_sec'] = sc['avg_decision_time_ms'] / 1000

color_blue   = '#0072BD'
color_orange = '#D95319'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.dpi'] = 300

AXIS_LABEL_SIZE = 28
TICK_LABEL_SIZE = 27
LEGEND_SIZE     = 23
LABEL_PAD       = 15
NUM_TICKS       = 6

configs = {
    'beijing': {
        'df': bj,
        'y1_min': 0,
        'y1_max': 25,
        'y2_min': 500,
        'y2_max': 3000,
        'legend_loc': 'lower right',
        'output_name': 'fig6_sensitivity_pruning_beijing'
    },
    'chengdu': {
        'df': sc,
        'y1_min': 0,
        'y1_max': 60,
        'y2_min': 1500,
        'y2_max': 3500,
        'legend_loc': 'lower right',
        'output_name': 'fig6_sensitivity_pruning_chengdu'
    }
}

def plot_single_dataset(cfg, save_dir):
    df = cfg['df']

    y1_min, y1_max = cfg['y1_min'], cfg['y1_max']
    y2_min, y2_max = cfg['y2_min'], cfg['y2_max']
    y1_ticks = np.linspace(y1_min, y1_max, NUM_TICKS)
    y2_ticks = np.linspace(y2_min, y2_max, NUM_TICKS)

    fig, ax1 = plt.subplots(figsize=(9, 6.5))

    line1 = ax1.plot(df['pruning_ratio'], df['decision_time_sec'],
                     color=color_blue,
                     linestyle='-',
                     marker='s',
                     markersize=9,
                     linewidth=2.5,
                     label='Avg. Decision Time (s)')

    ax1.set_xlabel('Pruning Ratio', fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax1.set_ylabel('Avg. Decision Time (s)',
                   color=color_blue,
                   fontsize=AXIS_LABEL_SIZE,
                   labelpad=LABEL_PAD)

    ax1.set_ylim(y1_min, y1_max)
    ax1.set_yticks(y1_ticks)
    ax1.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])

    ax1.tick_params(axis='x',
                    labelsize=TICK_LABEL_SIZE,
                    direction='in',
                    length=6)
    ax1.tick_params(axis='y',
                    labelcolor=color_blue,
                    labelsize=TICK_LABEL_SIZE,
                    width=2,
                    colors=color_blue,
                    direction='in',
                    length=6)

    ax1.spines['left'].set_color(color_blue)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.spines['top'].set_linewidth(1.5)

    ax1.grid(True, which='major', linestyle='-', alpha=0.3)

    ax2 = ax1.twinx()

    line2 = ax2.plot(df['pruning_ratio'], df['total_allocations'],
                     color=color_orange,
                     linestyle='-',
                     marker='o',
                     markersize=9,
                     linewidth=2.5,
                     label='Total Allocations')

    ax2.set_ylabel('Total Allocations',
                   color=color_orange,
                   fontsize=AXIS_LABEL_SIZE,
                   labelpad=LABEL_PAD)

    ax2.set_ylim(y2_min, y2_max)
    ax2.set_yticks(y2_ticks)

    ax2.tick_params(axis='y',
                    labelcolor=color_orange,
                    labelsize=TICK_LABEL_SIZE,
                    width=2,
                    colors=color_orange,
                    direction='in',
                    length=6)

    ax2.spines['right'].set_color(color_orange)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_visible(False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels,
               loc=cfg['legend_loc'],
               fontsize=LEGEND_SIZE,
               frameon=True,
               edgecolor='black',
               framealpha=0.9)

    plt.tight_layout()

    output_pdf = os.path.join(save_dir, cfg['output_name'] + '.pdf')
    output_png = os.path.join(save_dir, cfg['output_name'] + '.png')

    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(output_png, format='png', bbox_inches='tight', dpi=300)

    print()

    plt.show()
    plt.close(fig)

plot_single_dataset(configs['beijing'], save_dir)
plot_single_dataset(configs['chengdu'], save_dir)