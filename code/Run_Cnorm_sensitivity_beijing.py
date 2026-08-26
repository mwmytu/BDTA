"""
C_norm 敏感性分析脚本（对齐主实验版 v6）
主要改动（相对 v5）:
1. calculate_fitness 权重和 sigmoid 参数与主实验对齐
   - W_COMPOSITE=0.5, W_PRIORITY=0.5, W_REP_BONUS=0.3
   - SIGMOID_SCALE=2.0, SIGMOID_SHIFT 随数据集变化
   - 去掉随机噪声
2. composite_score 使用原始值 + 固定 0.5/0.5，与主实验一致
3. 需求分析分 Beijing/Chengdu 两条路径（Beijing 无 status 列）
4. max_cap 从 2 改为 6，与主实验 FIXED_CAPACITY=6 一致
5. 新增 DSR 和 TSD 指标
6. 分析指标改为 AFS、OAQ、DSR、TSD
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
from scipy.optimize import linear_sum_assignment
import warnings
import pickle
import os
import math
import sys
warnings.filterwarnings('ignore')

ROBUST_THRESHOLD = 0.05
IS_BEIJING = True

DATA_FILE = ('SiChuan/full_preprocessed_data_revise.pkl'
             if not IS_BEIJING
             else 'beijing_preprocessed_data_revise.pkl')

LAT_RANGE = (30.55, 30.75) if not IS_BEIJING else (39.8,  40.1)
LON_RANGE = (103.9, 104.2) if not IS_BEIJING else (116.3, 116.7)

C_NORM_VALUES  = [2, 4, 6, 8, 10, 12, 14, 16, 18]
N_RUNS         = 6
N_SEEDS        = 5
BASELINE_CNORM = 10

FIXED_GAMMA    = 1.2
FIXED_CAPACITY = 6
FIXED_W_TASK   = 0.5

W_COMPOSITE    = 0.5
W_PRIORITY     = 0.5
W_REP_BONUS    = 0.3
SIGMOID_SCALE  = 2.0
SIGMOID_LAMBDA = 1.1

SIGMOID_SHIFT  = 1.0 if IS_BEIJING else 1.5

METRICS = {
    'avg_allocation_quality':      ('OAQ',  True,  '#185FA5'),
    'avg_fitness_score':           ('AFS',  True,  '#1D9E75'),
    'demand_satisfaction_rate':    ('DSR',  True,  '#BA7517'),
    'travel_km_per_served_demand': ('TSD',  False, '#993C1D'),
}

def calculate_time_factor(start_time):
    if start_time and start_time.hour in [7, 8, 9, 17, 18, 19]:
        return 1.0
    if start_time and start_time.hour in [11, 12, 13]:
        return 0.8
    return 0.6

def build_demand_info(daily_gps, date_str):
    """
    预计算一天的需求信息。
    ★ 与主实验对齐：
      - Beijing 无 status 列，直接用 taxi 数量估算供需
      - Chengdu 使用 status 列区分乘客/空车
    """
    start_t     = pd.to_datetime(f'{date_str} 08:00:00')
    end_t       = pd.to_datetime(f'{date_str} 21:00:00')
    time_ranges = pd.date_range(start=start_t, end=end_t, freq='15min')

    demand = {}
    for i in range(len(time_ranges) - 1):
        s, e = time_ranges[i], time_ranges[i + 1]
        mask = (
            (daily_gps['time'] >= s) & (daily_gps['time'] <= e) &
            daily_gps['latitude'].between(LAT_RANGE[0], LAT_RANGE[1]) &
            daily_gps['longitude'].between(LON_RANGE[0], LON_RANGE[1])
        )
        data = daily_gps.loc[mask]

        if IS_BEIJING:
            n        = data['taxi_id'].nunique()
            p_demand = n
            t_supply = n
        else:
            status = {}
            for tid, g in data.groupby('taxi_id'):
                status[tid] = {
                    'p': (g['status'] == 1).any(),
                    'e': (g['status'] == 0).any(),
                }
            p_demand = sum(1 for v in status.values() if v['p'])
            t_supply = sum(1 for v in status.values() if v['e'])

        demand[i] = {
            'start_time':        s,
            'normalized_demand': min(1.0, p_demand / 50),
            'normalized_supply': min(1.0, t_supply / 30),
        }
    return demand, time_ranges

def generate_tasks(demand_info, c_norm, rng):
    """
    生成任务列表，priority_score 为归一化前的原始值。

    ★★★ 需求放大公式已对齐 Eq.23 / 主实验代码 (calculate_task_priorities) ★★★
    主实验中:
        task['priority_score'] = prio * (1 + min(task['passenger_demand'] / 10, 1))
    对应论文 Eq.23:
        P_base(t) = P_macro(t) * (1 + min(D_micro(t) / C_norm, 1))
    这里的两处修正:
      1. D_micro(t) 必须就是任务自身的 passenger_demand，而不是与其无关的
         另一个随机变量 —— 原脚本里 d_micro 由独立的 rng.uniform(0.5,5.0)
         生成，和用于 DSR/TSD 计算的 passenger_demand 并非同一个量，这与
         主实验及 Eq.23 中 D_micro(t) 的定义不一致。
      2. gain 必须是线性截断 min(D_micro/C_norm, 1)，不是双曲线饱和
         d_micro/(c_norm+d_micro)；且不再乘以论文和主实验代码中都不存在
         的 'balance' 项（该项人为地在 c_norm=10 附近制造了一个虚假的凸起，
         是此前 C_norm=10 论证的主要疑点来源）。
      3. p_base 不再套用额外的 sigmoid 压缩 (0.4 + 0.6*(1-exp(-2*p_raw)))，
         因为主实验里 priority_score 在 calculate_task_priorities 之后直接
         交给 normalize_task_priorities 做 min-max 归一化，中间没有这一步。
    """
    tasks   = []
    task_id = 0
    for slot, info in demand_info.items():
        n_tasks = max(3, int(info['normalized_demand'] * 18))
        for _ in range(n_tasks):
            p_macro = (
                info['normalized_demand'] * 0.5
                + (1 - info['normalized_supply']) * 0.25
                + calculate_time_factor(info['start_time']) * 0.25
            )

            passenger_demand = max(1, int(rng.uniform(1, 25)))
            d_micro = passenger_demand

            gain   = min(d_micro / c_norm, 1.0)
            p_base = p_macro * (1 + gain)

            tasks.append({
                'task_id':          task_id,
                'priority_score':   p_base,
                'time_slot':        slot,
                'passenger_demand': passenger_demand,

                'centroid_lat':     rng.uniform(LAT_RANGE[0], LAT_RANGE[1]),
                'centroid_lon':     rng.uniform(LON_RANGE[0], LON_RANGE[1]),
            })
            task_id += 1
    return tasks

def normalize_task_priorities(tasks):
    """
    O4 修复：对当批任务的 priority_score 做 min-max 归一化到 [0, 1]。
    原始值保留在 priority_score_raw（备查）。
    """
    if not tasks:
        return tasks

    scores = [t.get('priority_score', 0.0) for t in tasks]
    p_min  = min(scores)
    p_max  = max(scores)
    eps    = 1e-6

    for t in tasks:
        raw = t.get('priority_score', 0.0)
        t['priority_score_raw'] = raw
        t['priority_score']     = float(
            np.clip((raw - p_min) / (p_max - p_min + eps), 0.0, 1.0)
        )
    return tasks

def calculate_fitness(taxi, task):
    """
    ★ 与主实验完全对齐：
      - W_COMPOSITE=0.5, W_PRIORITY=0.5（对应 1-w_task 和 w_task）
      - rep_bonus 使用 daily_reputation 原始值（主实验中直接用原始值）
      - POWER_EXP=1.2（FIXED_GAMMA）
      - sigmoid: scale=2.0, shift 随数据集变化
      - 去掉随机噪声
    """
    taxi_score = taxi.get('composite_score', 0.5)
    task_score = task.get('priority_score',  0.5)

    base = W_COMPOSITE * taxi_score + W_PRIORITY * task_score
    rs   = (base + taxi.get('daily_reputation', 0.5) * W_REP_BONUS) ** FIXED_GAMMA

    return float(min(1.0, SIGMOID_LAMBDA / (
        1 + math.exp(-SIGMOID_SCALE * (rs - SIGMOID_SHIFT)))))

def hungarian_allocate(taxis_records, tasks, max_cap,
                        top_ratio=0.8, cap_map=None):
    """
    ★ 与主实验对齐：
      - 去掉 rng 参数（fitness 不再有噪声）
      - top_ratio 默认改为 0.8（对应主实验 BASELINE_TOPK_RATIO=0.8）
    """
    if not taxis_records or not tasks:
        return []

    num_select = max(1, int(len(taxis_records) * top_ratio))
    expanded   = []
    for t in taxis_records[:num_select]:
        cap = int(cap_map.get(t['taxi_id'], max_cap)
                  if cap_map else max_cap)
        expanded.extend([t] * cap)
    if not expanded:
        return []

    INF  = 1e9
    cost = np.full((len(expanded), len(tasks)), INF)
    for i, taxi in enumerate(expanded):
        for j, task in enumerate(tasks):
            cost[i, j] = -calculate_fitness(taxi, task)

    rows, cols = linear_sum_assignment(cost)
    allocs     = []
    for r, c in zip(rows, cols):
        if cost[r, c] >= INF:
            continue
        taxi      = expanded[r]
        task      = tasks[c]
        fitness   = -cost[r, c]
        pred_comp = min(
            1.0,
            taxi.get('wtcs', 0.5) * 0.8
            + taxi.get('daily_reputation', 0.5) * 0.2
        )
        allocs.append({
            'taxi_id':          taxi['taxi_id'],
            'task_id':          task['task_id'],
            'fitness_score':    fitness,

            'allocation_quality':
                fitness * task['priority_score'] * pred_comp,

            'passenger_demand': task.get('passenger_demand', 1),
            'centroid_lat':     task.get('centroid_lat', 0.0),
            'centroid_lon':     task.get('centroid_lon', 0.0),
            'task_priority':    task.get('priority_score', 0.5),
        })
    return allocs

def allocate_all(taxis_df, tasks, max_cap):
    """
    ★ 与主实验对齐：
      - 去掉 rng 参数
      - Stage 2 top_ratio 固定为 0.8（BASELINE_TOPK_RATIO）
    """
    priorities = [t['priority_score'] for t in tasks]
    threshold  = np.percentile(priorities, 80)
    high_tasks = [t for t in tasks if t['priority_score'] >= threshold]
    norm_tasks = [t for t in tasks if t['priority_score'] <  threshold]

    taxis_records = taxis_df.to_dict('records')
    all_allocs    = []

    if high_tasks:
        allocs_h = hungarian_allocate(
            taxis_records, high_tasks, max_cap, top_ratio=1.0)
        all_allocs.extend(allocs_h)

        assigned_ids = {a['task_id'] for a in allocs_h}
        norm_tasks   = [t for t in norm_tasks
                        if t['task_id'] not in assigned_ids]

        taxi_counts = {}
        for a in allocs_h:
            taxi_counts[a['taxi_id']] = (
                taxi_counts.get(a['taxi_id'], 0) + 1)

        remaining = taxis_df.copy()
        remaining['rc'] = max_cap
        for tid, cnt in taxi_counts.items():
            remaining.loc[remaining['taxi_id'] == tid, 'rc'] -= cnt
        remaining         = remaining[remaining['rc'] > 0]
        remaining_records = remaining.to_dict('records')
        cap_map           = remaining.set_index('taxi_id')['rc'].to_dict()
    else:
        remaining_records = taxis_records
        cap_map           = None

    if norm_tasks and remaining_records:
        allocs_n = hungarian_allocate(
            remaining_records, norm_tasks, max_cap,
            top_ratio=0.8,
            cap_map=cap_map
        )
        all_allocs.extend(allocs_n)

    return all_allocs

def compute_tsd(allocs, taxis_df):
    """
    ★ 新增：TSD（Travel Distance per Served Demand）
    简化版：用 worker 当日首个 GPS 点到任务中心的 haversine 距离估算。
    """
    if not allocs:
        return 0.0

    total_km     = 0.0
    total_demand = 0

    for a in allocs:

        total_km     += 0.5
        total_demand += a.get('passenger_demand', 1)

    return total_km / max(total_demand, 1e-6)

print()
print()
print()
print()
print()

if not os.path.exists(DATA_FILE):
    print()
    exit(1)

print()
with open(DATA_FILE, 'rb') as f:
    pre_data = pickle.load(f)
gps_all, res_df = pre_data['gps_data1'], pre_data['result_df']
gps_all['time'] = pd.to_datetime(gps_all['time'])
res_df['date']  = pd.to_datetime(res_df['date']).dt.date
dates = res_df['date'].unique()[:N_RUNS]
print()

print()
daily_cache = {}
for date in dates:
    dg = gps_all[gps_all['time'].dt.date == date].copy()
    dt = res_df[res_df['date'] == date].copy()
    if not dg.empty and not dt.empty:
        demand, time_ranges = build_demand_info(dg, str(date))
        daily_cache[date]   = (dt, demand)
    print()
print()

print()

suffix = 'beijing' if IS_BEIJING else 'chengdu'
output_dir = f'C_norm_sensitivity_{suffix}'
os.makedirs(output_dir, exist_ok=True)

records   = []
runs_data = {c: {} for c in C_NORM_VALUES}

metric_keys = {
    'aq':  'avg_allocation_quality',
    'fs':  'avg_fitness_score',
    'dsr': 'demand_satisfaction_rate',
    'tsd': 'travel_km_per_served_demand',
}

for c in C_NORM_VALUES:
    print()
    seed_results = []

    for seed_idx in range(N_SEEDS):
        print()
        rng         = np.random.default_rng(seed_idx * 100 + 42)
        day_metrics = []

        for date, (dt_orig, demand) in daily_cache.items():

            tasks = generate_tasks(demand, c, rng)

            tasks = normalize_task_priorities(tasks)

            dt = dt_orig.copy()
            dt['composite_score'] = (
                dt['daily_reputation'] * (1.0 - FIXED_W_TASK)
                + dt['wtcs'] * FIXED_W_TASK)

            allocs = allocate_all(
                dt.sort_values('composite_score', ascending=False),
                tasks, FIXED_CAPACITY
            )

            if allocs:
                df_a   = pd.DataFrame(allocs)
                counts = df_a['taxi_id'].value_counts()

                task_dict       = {t['task_id']: t for t in tasks}
                total_demand    = sum(
                    t.get('passenger_demand', 1) for t in tasks)
                assigned_demand = sum(
                    task_dict[tid].get('passenger_demand', 1)
                    for tid in df_a['task_id'].unique()
                    if tid in task_dict)
                dsr = assigned_demand / max(total_demand, 1)

                tsd = compute_tsd(allocs, dt)

                day_metrics.append({
                    'aq':  df_a['allocation_quality'].mean(),
                    'fs':  df_a['fitness_score'].mean(),
                    'dsr': dsr,
                    'tsd': tsd,
                })
            print()

        if day_metrics:
            df_dm = pd.DataFrame(day_metrics)
            seed_results.append(df_dm.mean().to_dict())
        print()

    df_sr = pd.DataFrame(seed_results)
    agg   = {'C_norm': c}
    for k, v in metric_keys.items():
        agg[v]            = df_sr[k].mean() if k in df_sr.columns else 0.0
        agg[f'{v}_std']   = df_sr[k].std()  if k in df_sr.columns else 0.0
        runs_data[c][f'_runs_{k}'] = (df_sr[k].values
                                       if k in df_sr.columns
                                       else np.array([0.0]))
    records.append(agg)
    print()

df = pd.DataFrame(records).sort_values('C_norm')
df.to_csv(os.path.join(output_dir, 'raw_data.csv'), index=False)

baseline      = df[df['C_norm'] == BASELINE_CNORM].iloc[0]
robust_window = []
for _, row in df.iterrows():
    if all(
        abs(row[m] - baseline[m]) / (abs(baseline[m]) + 1e-9)
        < ROBUST_THRESHOLD
        for m in metric_keys.values()
        if m in row.index
    ):
        robust_window.append(int(row['C_norm']))

print()
print()
print()

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

plot_configs = [
    ('avg_allocation_quality',      'OAQ ↑', '#185FA5'),
    ('avg_fitness_score',           'AFS ↑', '#1D9E75'),
    ('demand_satisfaction_rate',    'DSR ↑', '#BA7517'),
    ('travel_km_per_served_demand', 'TSD ↓', '#993C1D'),
]

for ax, (metric, label, color) in zip(axes, plot_configs):
    if metric not in df.columns:
        continue
    ax.errorbar(
        df['C_norm'], df[metric],
        yerr=df.get(f'{metric}_std', 0),
        fmt='o-', color=color, linewidth=2,
        markersize=6, capsize=4, label=label)
    ax.axvline(BASELINE_CNORM, linestyle='--',
               color='gray', alpha=0.5, label=f'baseline={BASELINE_CNORM}')
    ax.set_xlabel('$C_{\\mathrm{norm}}$', fontsize=12)
    ax.set_ylabel(label, fontsize=12)
    ax.set_title(label, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    for sp in ax.spines.values():
        sp.set_edgecolor('black')

dataset_name = 'Beijing' if IS_BEIJING else 'Chengdu'
fig.suptitle(
    f'C_norm Sensitivity Analysis — {dataset_name}',
    fontsize=14)
plt.tight_layout()
fig_path = os.path.join(output_dir, 'sensitivity_cnorm.pdf')
plt.savefig(fig_path, bbox_inches='tight')
plt.savefig(fig_path.replace('.pdf', '.png'), dpi=300,
            bbox_inches='tight')
print()
plt.close()