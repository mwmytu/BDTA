import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = 'bdta_sensitivity_results'
file_path = os.path.join(save_dir, 'chengdu_sensitivity_alpha_BDTA.csv')

os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(file_path):
    print()
    data_content = """total_allocations,avg_fitness_score,avg_allocation_quality,task_coverage,taxi_utilization,alpha_value,fixed_capacity
2753.5,0.8322,1.1684,0.3106,0.9178,1.0,6
2753.5,0.8721,1.2245,0.3106,0.9178,1.1,6
2753.5,0.9087,1.2758,0.3106,0.9178,1.2,6
2753.0,0.9416,1.3219,0.3106,0.9176,1.3,6
2752.5,0.9671,1.3571,0.3105,0.9175,1.4,6
2753.5,0.9780,1.3709,0.3106,0.9178,1.5,6
2753.5,0.9861,1.3808,0.3106,0.9178,1.6,6
2753.5,0.9902,1.3856,0.3106,0.9178,1.7,6
2753.5,0.9925,1.3882,0.3106,0.9178,1.8,6
2753.5,0.9941,1.3899,0.3106,0.9178,1.9,6
2753.5,0.9952,1.3910,0.3106,0.9178,2.0,6"""
    with open(file_path, "w") as f:
        f.write(data_content)

df = pd.read_csv(file_path)

color_blue = '#0072BD'
color_orange = '#D95319'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.dpi'] = 300

AXIS_LABEL_SIZE = 28
TICK_LABEL_SIZE = 27
LEGEND_SIZE = 23
LABEL_PAD = 15

fig, ax1 = plt.subplots(figsize=(9, 6.5))

NUM_TICKS = 6

y1_min, y1_max = 1.15, 1.40
y1_ticks = np.linspace(y1_min, y1_max, NUM_TICKS)

y2_min, y2_max = 2740, 2765
y2_ticks = np.linspace(y2_min, y2_max, NUM_TICKS)

line1 = ax1.plot(df['alpha_value'], df['avg_allocation_quality'],
                 color=color_blue,
                 linestyle='-',
                 marker='s',
                 markersize=9,
                 linewidth=2.5,
                 label='Avg. Allocation Quality')

ax1.set_xlabel(r'Amplification Exponent ($\gamma$)', fontsize=AXIS_LABEL_SIZE, labelpad=10)
ax1.set_ylabel('Avg. Allocation Quality', color=color_blue, fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)

ax1.set_ylim(y1_min, y1_max)
ax1.set_yticks(y1_ticks)

ax1.tick_params(axis='x', labelsize=TICK_LABEL_SIZE, direction='in', length=6)
ax1.tick_params(axis='y', labelcolor=color_blue, labelsize=TICK_LABEL_SIZE, width=2, colors=color_blue, direction='in', length=6)

ax1.spines['left'].set_color(color_blue)
ax1.spines['left'].set_linewidth(2)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['top'].set_linewidth(1.5)

ax1.grid(True, which='major', linestyle='-', alpha=0.3)

ax2 = ax1.twinx()

line2 = ax2.plot(df['alpha_value'], df['total_allocations'],
                 color=color_orange,
                 linestyle='-',
                 marker='o',
                 markersize=9,
                 linewidth=2.5,
                 label='Total Allocations')

ax2.set_ylabel('Total Allocations', color=color_orange, fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)

ax2.set_ylim(y2_min, y2_max)
ax2.set_yticks(y2_ticks)

ax2.tick_params(axis='y', labelcolor=color_orange, labelsize=TICK_LABEL_SIZE, width=2, colors=color_orange, direction='in', length=6)

ax2.spines['right'].set_color(color_orange)
ax2.spines['right'].set_linewidth(2)
ax2.spines['left'].set_visible(False)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower right', fontsize=LEGEND_SIZE, frameon=True, edgecolor='black')

plt.tight_layout()

output_pdf = os.path.join(save_dir, 'sensitivity_gamma_aligned_ticks.pdf')
output_png = os.path.join(save_dir, 'sensitivity_gamma_aligned_ticks.png')

plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
plt.savefig(output_png, format='png', bbox_inches='tight', dpi=300)

print()
plt.show()