import os
import math
import time
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

PRUNING_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

FIXED_CAPACITY = 6
FIXED_ALPHA    = 1.2
FIXED_W_REP    = 0.5

IS_BEIJING_DATASET = True

if IS_BEIJING_DATASET:
    print()
    PREPROCESSED_DATA_FILE = 'beijing_preprocessed_data.pkl'
    RESULTS_FILENAME        = 'beijing_sensitivity_pruning.csv'
    data_dir                = ''
    lat_min, lat_max        = 39.8, 40.1
    lon_min, lon_max        = 116.3, 116.7
    lat_step_val            = 0.04
    lon_step_val            = 0.04
    excluded_dates          = []
else:
    print()
    PREPROCESSED_DATA_FILE = 'full_preprocessed_data.pkl'
    RESULTS_FILENAME        = 'chengdu_sensitivity_pruning.csv'
    data_dir                = 'SiChuan'
    lat_min, lat_max        = 30.55, 30.75
    lon_min, lon_max        = 103.9, 104.2
    lat_step_val            = 0.015
    lon_step_val            = 0.015
    excluded_dates          = [
        pd.Timestamp('2014-08-07').date(),
        pd.Timestamp('2014-08-13').date()
    ]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sampling_trigger_log = {ratio: [] for ratio in PRUNING_LEVELS}

data_file_to_load = (PREPROCESSED_DATA_FILE if IS_BEIJING_DATASET
                     else os.path.join(data_dir, PREPROCESSED_DATA_FILE))

print()
try:
    with open(data_file_to_load, 'rb') as f:
        preprocessed_data = pickle.load(f)
    gps_data1  = preprocessed_data['gps_data1']
    result_df  = preprocessed_data['result_df']
    print()
except FileNotFoundError:
    print()
    exit()

date_range          = pd.to_datetime(result_df['date'].unique())
daily_start_time    = pd.to_datetime('08:00:00').time()
daily_end_time      = pd.to_datetime('21:00:00').time()
filtered_date_range = [
    date for date in date_range
    if date.date() not in excluded_dates
]

def normalize_task_priorities(tasks):
    if not tasks:
        return tasks
    scores = [t.get('priority_score', 0.0) for t in tasks]
    p_min  = min(scores)
    p_max  = max(scores)
    eps    = 1e-6
    for t in tasks:
        raw = t.get('priority_score', 0.0)
        t['priority_score_raw'] = raw
        t['priority_score']     = (raw - p_min) / (p_max - p_min + eps)
    return tasks

def generate_grid_tasks(time_ranges, gps_data,
                         lat_range, lon_range, lat_step, lon_step):
    tasks, task_id_counter = [], 0
    lat_bins = np.arange(lat_range[0], lat_range[1], lat_step)
    lon_bins = np.arange(lon_range[0], lon_range[1], lon_step)

    for slot_idx, (start, end) in enumerate(
            zip(time_ranges[:-1], time_ranges[1:])):
        slot_data = gps_data[
            (gps_data['time'] >= start) & (gps_data['time'] < end)
        ]
        if slot_data.empty:
            continue
        for i, lat0 in enumerate(lat_bins):
            for j, lon0 in enumerate(lon_bins):
                demand_df = slot_data[
                    (slot_data['latitude'].between(
                        lat0, lat0 + lat_step, inclusive='left')) &
                    (slot_data['longitude'].between(
                        lon0, lon0 + lon_step, inclusive='left'))
                ]
                p_demand = demand_df['taxi_id'].nunique()
                if p_demand > 0:
                    tasks.append({
                        "task_id":          task_id_counter,
                        "time_slot":        slot_idx,
                        "priority_score":   0,
                        "passenger_demand": p_demand,
                        "start_time":       start
                    })
                    task_id_counter += 1
    return tasks

class TaskPreferenceSorter:
    def __init__(self, weights=None):
        self.weights = weights or {
            'passenger_demand': 0.5,
            'taxi_supply':      0.25,
            'time_factor':      0.25
        }

    def analyze_passenger_demand_enhanced(self, gps_data, time_ranges,
                                           lat_range, lon_range):
        demand = {}
        for i in range(len(time_ranges) - 1):
            s, e = time_ranges[i], time_ranges[i + 1]
            data = gps_data[
                (gps_data['time'] >= s) & (gps_data['time'] <= e) &
                (gps_data['latitude'].between(lat_range[0], lat_range[1])) &
                (gps_data['longitude'].between(lon_range[0], lon_range[1]))
            ]
            if IS_BEIJING_DATASET:
                cnt      = data['taxi_id'].nunique()
                p_demand = t_supply = cnt
            else:
                status   = {
                    tid: {'p': any(g['status'] == 1),
                          'e': any(g['status'] == 0)}
                    for tid, g in data.groupby('taxi_id')
                }
                p_demand = sum(1 for s_ in status.values() if s_['p'])
                t_supply = sum(1 for s_ in status.values() if s_['e'])
            demand[i] = {
                'start_time':        s,
                'normalized_demand': min(1.0, p_demand / 50),
                'normalized_supply': min(1.0, t_supply / 30)
            }
        return demand

    def calculate_time_factor(self, start_time):
        if start_time and start_time.hour in [7, 8, 9, 17, 18, 19]: return 1.0
        if start_time and start_time.hour in [11, 12, 13]:           return 0.8
        return 0.6

    def sort_taxis(self, taxi_data, w_rep, w_comp):
        taxi_data = taxi_data.copy()
        taxi_data['composite_score'] = (
            taxi_data.get('daily_reputation', 0.5) * w_rep +
            taxi_data.get('wtcs', 0.5) * w_comp
        )
        return taxi_data.sort_values('composite_score', ascending=False)

MATRIX_SIZE_THRESHOLD = 6000 * 6000
matrix_size_log = {ratio: [] for ratio in PRUNING_LEVELS}

class TaskAllocator:
    def __init__(self, max_tasks_per_taxi, amplification_factor,
                 current_ratio=None, current_date=None):
        self.max_tasks_per_taxi   = max_tasks_per_taxi
        self.amplification_factor = amplification_factor
        self.current_ratio        = current_ratio
        self.current_date         = current_date

    def _calculate_fitness(self, taxi, task):
        base  = (0.7 * taxi.get('composite_score', 0.5)
                 + 0.5 * task.get('priority_score', 0.5))
        bonus = taxi.get('daily_reputation', 0.5) * 0.3
        raw   = (base + bonus) ** self.amplification_factor
        k, x0, scaling = 2.5, 1.2, 1.1
        fitness = 1 / (1 + math.exp(-k * (raw - x0)))
        return min(1.0, fitness * scaling)

    def allocate_tasks(self, taxis, tasks, pruning_ratio):
        if not tasks:
            return []

        threshold    = np.percentile([t['priority_score'] for t in tasks], 80)
        high_tasks   = [t for t in tasks if t['priority_score'] >= threshold]
        normal_tasks = [t for t in tasks if t['priority_score'] <  threshold]
        all_allocs   = []

        if high_tasks:
            allocs_high = self._hungarian_kernel(
                taxis, high_tasks, top_p_ratio=1.0,
                phase='high')
            if allocs_high:
                all_allocs.extend(allocs_high)
                assigned_counts   = pd.Series(
                    [a['taxi_id'] for a in allocs_high]
                ).value_counts().to_dict()
                assigned_task_ids = {a['task_id'] for a in allocs_high}
                normal_tasks      = [t for t in normal_tasks
                                     if t['task_id'] not in assigned_task_ids]
                remaining_taxis   = taxis.copy()
                remaining_taxis['remaining_capacity'] = self.max_tasks_per_taxi
                for tid, cnt in assigned_counts.items():
                    remaining_taxis.loc[
                        remaining_taxis['taxi_id'] == tid,
                        'remaining_capacity'
                    ] -= cnt
                remaining_taxis = remaining_taxis[
                    remaining_taxis['remaining_capacity'] > 0]
            else:
                remaining_taxis = taxis.copy()
                remaining_taxis['remaining_capacity'] = self.max_tasks_per_taxi
        else:
            remaining_taxis = taxis.copy()
            remaining_taxis['remaining_capacity'] = self.max_tasks_per_taxi

        if normal_tasks and not remaining_taxis.empty:
            allocs_normal = self._hungarian_kernel(
                remaining_taxis, normal_tasks,
                top_p_ratio=pruning_ratio,
                phase='normal',
                capacity_map=remaining_taxis.set_index(
                    'taxi_id')['remaining_capacity'].to_dict()
            )
            if allocs_normal:
                all_allocs.extend(allocs_normal)

        return all_allocs

    def _hungarian_kernel(self, taxis, tasks,
                           top_p_ratio=1.0, capacity_map=None,
                           phase='normal'):
        num_select   = max(1, int(len(taxis) * top_p_ratio))
        pruned_taxis = taxis.head(num_select)

        if pruned_taxis.empty or not tasks:
            return []

        expanded = []
        for t in pruned_taxis.to_dict('records'):
            cap = int(
                capacity_map.get(t['taxi_id'], self.max_tasks_per_taxi)
                if capacity_map else self.max_tasks_per_taxi
            )
            expanded.extend([t] * cap)

        if not expanded:
            return []

        matrix_size = len(expanded) * len(tasks)

        if self.current_ratio is not None:
            matrix_size_log[self.current_ratio].append(matrix_size)

        if matrix_size > MATRIX_SIZE_THRESHOLD:
            import random
            print()

            if self.current_ratio is not None:
                sampling_trigger_log[self.current_ratio].append({
                    'date':        str(self.current_date),
                    'phase':       phase,
                    'expanded':    len(expanded),
                    'tasks':       len(tasks),
                    'matrix_size': matrix_size,
                })
            expanded = random.sample(expanded, 2000)
        else:

            if self.current_ratio == 1.0 and phase == 'normal':
                print()

        cost_matrix = np.zeros((len(expanded), len(tasks)))
        for i, taxi in enumerate(expanded):
            for j, task in enumerate(tasks):
                cost_matrix[i, j] = -self._calculate_fitness(taxi, task)

        rows, cols = linear_sum_assignment(cost_matrix)

        res = []
        for r, c in zip(rows, cols):
            taxi    = expanded[r]
            task    = tasks[c]
            fitness = -cost_matrix[r, c]
            comp    = min(
                1.0,
                taxi.get('wtcs', 0.5) * 0.8
                + taxi.get('daily_reputation', 0.5) * 0.2
            )
            res.append({
                'taxi_id':            taxi['taxi_id'],
                'task_id':            task['task_id'],
                'fitness_score':      fitness,
                'allocation_quality': fitness * task.get('priority_score', 0) * comp
            })
        return res

if __name__ == '__main__':
    all_results = []
    output_dir  = 'bdta_sensitivity_results'
    os.makedirs(output_dir, exist_ok=True)

    print()
    print()
    print()

    for ratio in tqdm(PRUNING_LEVELS, desc="Varying Pruning Ratio"):
        daily_metrics = []

        for date in tqdm(filtered_date_range,
                          desc=f"Ratio={ratio:.1f}", leave=False):
            daily_data = gps_data1[
                gps_data1['time'].dt.date == date.date()
            ].copy()
            if daily_data.empty:
                continue

            start_ts    = pd.to_datetime(f'{date.date()} {daily_start_time}')
            end_ts      = pd.to_datetime(f'{date.date()} {daily_end_time}')
            time_ranges = pd.date_range(
                start=start_ts, end=end_ts, freq='15min')

            grid_tasks = generate_grid_tasks(
                time_ranges, daily_data,
                (lat_min, lat_max), (lon_min, lon_max),
                lat_step_val, lon_step_val
            )
            if not grid_tasks:
                continue

            task_sorter = TaskPreferenceSorter()
            demand_info = task_sorter.analyze_passenger_demand_enhanced(
                daily_data, time_ranges,
                (lat_min, lat_max), (lon_min, lon_max)
            )
            supply_per_ts = {
                idx: max(1, daily_data[
                    (daily_data['time'] >= s) &
                    (daily_data['time'] < e)
                ]['taxi_id'].nunique())
                for idx, (s, e) in enumerate(
                    zip(time_ranges[:-1], time_ranges[1:]))
            }

            for task in grid_tasks:
                si = demand_info.get(task['time_slot'])
                if si:
                    p  = (si['normalized_demand'] * 0.5
                          + (1 - si['normalized_supply']) * 0.25
                          + task_sorter.calculate_time_factor(
                              si['start_time']) * 0.25)
                    bp = p * (1 + min(task['passenger_demand'] / 10, 1))
                    if task['time_slot'] in supply_per_ts:
                        bp *= min(
                            4.0,
                            1.0 + task['passenger_demand']
                                  / supply_per_ts[task['time_slot']]
                        )
                    task['priority_score'] = bp
                else:
                    task['priority_score'] = 0.1

            grid_tasks  = normalize_task_priorities(grid_tasks)
            daily_tasks = sorted(
                grid_tasks,
                key=lambda x: x['priority_score'],
                reverse=True
            )

            taxi_data_for_day = result_df[
                result_df['date'] == date.date()
            ].copy()
            if taxi_data_for_day.empty:
                continue

            final_taxis = task_sorter.sort_taxis(
                taxi_data_for_day,
                w_rep=FIXED_W_REP,
                w_comp=(1 - FIXED_W_REP)
            )

            allocator = TaskAllocator(
                max_tasks_per_taxi=FIXED_CAPACITY,
                amplification_factor=FIXED_ALPHA,
                current_ratio=ratio,
                current_date=date.date()
            )

            timer_start  = time.time()
            allocations  = allocator.allocate_tasks(
                final_taxis, daily_tasks, pruning_ratio=ratio)
            timer_end    = time.time()

            decision_time_ms = (timer_end - timer_start) * 1000

            if allocations:
                df_alloc = pd.DataFrame(allocations)
                daily_metrics.append({
                    'avg_allocation_quality': df_alloc['allocation_quality'].mean(),
                    'avg_decision_time_ms':   decision_time_ms,
                    'total_allocations':      len(df_alloc)
                })

        if daily_metrics:
            avg                  = pd.DataFrame(daily_metrics).mean().to_dict()
            avg['pruning_ratio'] = ratio
            all_results.append(avg)

    print()
    print()
    print()
    any_triggered = False
    for ratio in PRUNING_LEVELS:
        triggers = sampling_trigger_log[ratio]
        sizes    = matrix_size_log[ratio]
        max_size = max(sizes) if sizes else 0
        if triggers:
            any_triggered = True
            print()
            for rec in triggers:
                print()
        else:
            print()

    if not any_triggered:
        print()
        print()
        print()
    else:
        print()
        print()
        print()
        print()
        print()
    print()

    if all_results:
        final_df  = pd.DataFrame(all_results)
        save_path = os.path.join(output_dir, RESULTS_FILENAME)
        final_df.to_csv(save_path, index=False)
        print()
        print()
    else:
        print()