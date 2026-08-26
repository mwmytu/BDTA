import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

save_dir = 'bdta_sensitivity_results'
file_path = os.path.join(save_dir, 'beijing_sensitivity_alpha_BDTA.csv')
os.makedirs(save_dir, exist_ok=True)

if not os.path.exists(file_path):
    data_content = """total_allocations,avg_fitness_score,avg_allocation_quality,task_coverage,taxi_utilization,alpha_value,fixed_capacity
2355.4285714285716,0.42824431303131333,0.28083032411318387,0.8924743495050625,0.7854285714285714,1.0,6
2355.4285714285716,0.43240249007252934,0.28739033668761665,0.8924743495050625,0.7854285714285714,1.1,6
2355.4285714285716,0.43687607489846547,0.29400514513461695,0.8924743495050625,0.7854285714285714,1.2,6
2355.4285714285716,0.44160504892209335,0.3006321338947529,0.8924743495050625,0.7854285714285714,1.3,6
2355.4285714285716,0.4465291284019806,0.3072860172108051,0.8924743495050625,0.7854285714285714,1.4,6
2355.4285714285716,0.4515886096185076,0.31387910099580646,0.8924743495050625,0.7854285714285714,1.5,6
2355.4285714285716,0.45672112241237606,0.320346735164953,0.8924743495050625,0.7854285714285714,1.6,6
2355.4285714285716,0.46186741780162793,0.3265589483049277,0.8924743495050625,0.7854285714285714,1.7,6
2355.4285714285716,0.4669809668661441,0.3326298195373301,0.8924743495050625,0.7854285714285714,1.8,6
2355.4285714285716,0.47202637107402257,0.33851487638373884,0.8924743495050625,0.7854285714285714,1.9,6
2355.4285714285716,0.4769691385551253,0.34419205433577676,0.8924743495050625,0.7854285714285714,2.0,6"""
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
LEGEND_SIZE = 22
LABEL_PAD = 15

TICK_WIDTH = 2.5
TICK_LENGTH = 8
SPINE_WIDTH = 2.5

fig, ax1 = plt.subplots(figsize=(9, 6.5))

NUM_TICKS = 6

y1_min, y1_max = 0.28, 0.355
y1_ticks = np.linspace(y1_min, y1_max, NUM_TICKS)

y2_val = 2355.428
y2_min, y2_max = 2345, 2365
y2_ticks = np.linspace(y2_min, y2_max, NUM_TICKS)

line1 = ax1.plot(df['alpha_value'], df['avg_allocation_quality'],
                 color=color_blue, linestyle='-', marker='s', markersize=9, linewidth=2.5,
                 label='Avg. Allocation Quality')

ax1.set_xlabel(r'Amplification Exponent ($\gamma$)', fontsize=AXIS_LABEL_SIZE, labelpad=10)
ax1.set_ylabel('Avg. Allocation Quality', color=color_blue, fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
ax1.set_ylim(y1_min, y1_max)
ax1.set_yticks(y1_ticks)

ax1.tick_params(axis='x', direction='in',
                width=TICK_WIDTH, length=TICK_LENGTH,
                colors='black', labelsize=TICK_LABEL_SIZE)

ax1.tick_params(axis='y', direction='in',
                width=TICK_WIDTH, length=TICK_LENGTH,
                colors=color_blue, labelcolor=color_blue, labelsize=TICK_LABEL_SIZE)

ax1.spines['left'].set_color(color_blue)
ax1.spines['left'].set_linewidth(SPINE_WIDTH)
ax1.spines['bottom'].set_linewidth(SPINE_WIDTH)
ax1.spines['top'].set_linewidth(SPINE_WIDTH)
ax1.spines['right'].set_linewidth(SPINE_WIDTH)

ax1.grid(True, which='major', linestyle='-', alpha=0.3)

ax2 = ax1.twinx()
line2 = ax2.plot(df['alpha_value'], df['total_allocations'],
                 color=color_orange, linestyle='-', marker='o', markersize=9, linewidth=2.5,
                 label='Total Allocations')

ax2.set_ylabel('Total Allocations', color=color_orange, fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
ax2.set_ylim(y2_min, y2_max)
ax2.set_yticks(y2_ticks)

ax2.tick_params(axis='y', direction='in',
                width=TICK_WIDTH, length=TICK_LENGTH,
                colors=color_orange, labelcolor=color_orange, labelsize=TICK_LABEL_SIZE)

ax2.spines['right'].set_color(color_orange)
ax2.spines['right'].set_linewidth(SPINE_WIDTH)

lines = line1 + line2
labels = [l.get_label() for l in lines]

ax1.legend(lines, labels, loc='upper left', fontsize=LEGEND_SIZE, frameon=True, edgecolor='black')

plt.tight_layout()
output_file = os.path.join(save_dir, 'beijing_sensitivity_gamma_final.pdf')
plt.savefig(output_file, format='pdf', bbox_inches='tight')
plt.show()
print()