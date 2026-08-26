import os
import pandas as pd
import matplotlib.pyplot as plt

BJ_CSV_PATH = 'C_norm_sensitivity_beijing/raw_data.csv'
SC_CSV_PATH = 'C_norm_sensitivity_chengdu/raw_data.csv'
save_dir    = 'cnorm_sensitivity_combined_results'

os.makedirs(save_dir, exist_ok=True)

for path in [BJ_CSV_PATH, SC_CSV_PATH]:
    if not os.path.exists(path):
        print()
        print()
        exit()

df_bj = pd.read_csv(BJ_CSV_PATH)
df_bj.columns = df_bj.columns.str.strip()
df_sc = pd.read_csv(SC_CSV_PATH)
df_sc.columns = df_sc.columns.str.strip()

print()
print()
print()
print()

c_col_bj = [c for c in df_bj.columns if c.lower() in ['c_norm', 'cnorm']][0]
c_col_sc = [c for c in df_sc.columns if c.lower() in ['c_norm', 'cnorm']][0]

df_bj = df_bj.sort_values(c_col_bj).reset_index(drop=True)
df_sc = df_sc.sort_values(c_col_sc).reset_index(drop=True)

COLOR_BJ = '#0072BD'
COLOR_SC = '#D95319'

plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['font.sans-serif']  = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi']      = 300
plt.rcParams['pdf.fonttype']    = 42
plt.rcParams['ps.fonttype']     = 42

AXIS_LABEL_SIZE = 28
TICK_LABEL_SIZE = 27
LEGEND_SIZE     = 22
LABEL_PAD       = 15
TICK_WIDTH      = 2
TICK_LENGTH     = 8
SPINE_WIDTH     = 2
LINE_WIDTH      = 3.0
MARKER_SIZE     = 10

def plot_combined_metric(
    bj_col,
    sc_col,
    y_label_base,
    output_filename,
    legend_loc='upper right'
):
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        df_bj[c_col_bj], df_bj[bj_col],
        color=COLOR_BJ, linestyle='-',
        marker='o', markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH, label='Beijing', zorder=3
    )

    ax.plot(
        df_sc[c_col_sc], df_sc[sc_col],
        color=COLOR_SC, linestyle='-',
        marker='s', markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH, label='Chengdu', zorder=3
    )

    ax.set_xlabel('$C_{\\mathrm{norm}}$', fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylabel(f'{y_label_base} $\\uparrow$', fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)

    x_ticks = df_bj[c_col_bj].unique()
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='x', direction='in', width=TICK_WIDTH,
                    length=TICK_LENGTH, colors='black', labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis='y', direction='in', width=TICK_WIDTH,
                    length=TICK_LENGTH, colors='black', labelsize=TICK_LABEL_SIZE)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color('black')

    ax.grid(True, which='major', linestyle='-', alpha=0.3)

    leg = ax.legend(loc=legend_loc,
                    fontsize=LEGEND_SIZE,
                    frameon=True,
                    edgecolor='black',
                    fancybox=False)
    leg.set_zorder(10)
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    out_pdf = os.path.join(save_dir, output_filename + '.pdf')
    out_png = os.path.join(save_dir, output_filename + '.png')
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print()

def find_col(df_target, candidates):
    for c in candidates:
        if c in df_target.columns:
            return c
    return None

if __name__ == '__main__':

    bj_oaq = find_col(df_bj, ['avg_allocation_quality', 'OAQ', 'oaq'])
    sc_oaq = find_col(df_sc, ['avg_allocation_quality', 'OAQ', 'oaq'])

    bj_afs = find_col(df_bj, ['avg_fitness_score', 'AFS', 'afs'])
    sc_afs = find_col(df_sc, ['avg_fitness_score', 'AFS', 'afs'])

    if bj_oaq and sc_oaq:
        print()
        plot_combined_metric(
            bj_col=bj_oaq,
            sc_col=sc_oaq,
            y_label_base='Allocation Quality',
            output_filename='cnorm_sensitivity_oaq',
            legend_loc='upper right'
        )

    if bj_afs and sc_afs:
        print()
        plot_combined_metric(
            bj_col=bj_afs,
            sc_col=sc_afs,
            y_label_base='Fitness Score',
            output_filename='cnorm_sensitivity_afs',
            legend_loc='upper right'
        )

    print()