import os
import math
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

GAMMA_LEVELS   = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                  1.6, 1.7, 1.8, 1.9, 2.0]
FIXED_CAPACITY = 6
IS_BEIJING_DATASET = True

W_COMPOSITE    = 0.5
W_PRIORITY     = 0.5
W_REP_BONUS    = 0.3
SIGMOID_SCALE  = 2.0
SIGMOID_LAMBDA = 1.1
SIGMOID_SHIFT  = 1.0 if IS_BEIJING_DATASET else 1.5

SLOT_MINUTES     = 15
AVG_SPEED_KMH    = 40.0
DEADLINE_MINUTES = 15

if IS_BEIJING_DATASET:
    print()
    PREPROCESSED_DATA_FILE = 'beijing_preprocessed_data_revise.pkl'
    RESULTS_FILENAME       = 'beijing_sensitivity_gamma_BDTA.csv'
    data_dir               = ''
    lat_min, lat_max       = 39.8, 40.1
    lon_min, lon_max       = 116.3, 116.7
    lat_step_val           = 0.015
    lon_step_val           = 0.015
    excluded_dates         = []
else:
    print()
    PREPROCESSED_DATA_FILE = 'full_preprocessed_data_revise.pkl'
    RESULTS_FILENAME       = 'chengdu_sensitivity_gamma_BDTA.csv'
    data_dir               = 'SiChuan'
    lat_min, lat_max       = 30.55, 30.75
    lon_min, lon_max       = 103.9, 104.2
    lat_step_val           = 0.015
    lon_step_val           = 0.015
    excluded_dates         = [
        pd.Timestamp('2014-08-07').date(),
        pd.Timestamp('2014-08-13').date()
    ]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

data_file_to_load = (PREPROCESSED_DATA_FILE if IS_BEIJING_DATASET
                     else os.path.join(data_dir, PREPROCESSED_DATA_FILE))

print()
try:
    with open(data_file_to_load, 'rb') as f:
        preprocessed_data = pickle.load(f)
    gps_data1 = preprocessed_data['gps_data1']
    result_df = preprocessed_data['result_df']
    print()
except FileNotFoundError:
    print()
    exit()

gps_data1['time'] = pd.to_datetime(gps_data1['time'])
result_df['date'] = pd.to_datetime(result_df['date']).dt.date

date_range          = pd.to_datetime(result_df['date'].unique())
daily_start_time    = pd.to_datetime('08:00:00').time()
daily_end_time      = pd.to_datetime('21:00:00').time()
filtered_date_range = [d for d in date_range
                       if d.date() not in excluded_dates]

def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R    = 6371.0
    phi1 = radians(lat1); phi2 = radians(lat2)
    dp   = radians(lat2 - lat1); dl = radians(lon2 - lon1)
    a    = sin(dp/2)**2 + cos(phi1)*cos(phi2)*sin(dl/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def build_daily_position_index(daily_gps):
    index = {}
    for tid, group in daily_gps.groupby('taxi_id'):
        g = group[['time', 'latitude', 'longitude']].sort_values('time')
        index[tid] = {
            'times': g['time'].to_numpy(dtype='datetime64[ns]'),
            'lats':  g['latitude'].to_numpy(float),
            'lons':  g['longitude'].to_numpy(float),
        }
    return index

def get_position_at(pos_index, taxi_id, query_time):
    entry = pos_index.get(taxi_id)
    if entry is None or len(entry['times']) == 0:
        return None
    i = np.searchsorted(
        entry['times'], np.datetime64(query_time), side='right') - 1
    if i < 0:
        return None
    return float(entry['lats'][i]), float(entry['lons'][i])

def _time_factor(start_time):
    h = start_time.hour
    if h in [7, 8, 9, 17, 18, 19]: return 1.0
    if h in [11, 12, 13]:           return 0.8
    return 0.6

def _analyze_demand(gps_data, time_ranges):
    demand = {}
    for i, (s, e) in enumerate(zip(time_ranges[:-1], time_ranges[1:])):
        data = gps_data[(gps_data['time'] >= s) & (gps_data['time'] < e)]
        if IS_BEIJING_DATASET:
            n = data['taxi_id'].nunique()
            p, sup = n, n
        else:
            st  = {tid: (g['status'] == 1).any()
                   for tid, g in data.groupby('taxi_id')}
            p   = sum(st.values())
            sup = len(st) - p
        demand[i] = {
            'start_time':        s,
            'normalized_demand': min(1.0, p   / 50),
            'normalized_supply': min(1.0, sup / 30)
        }
    return demand

def _minmax_series(s):
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-6:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)

def compute_full_model_priority_bounds(tasks_template, daily_data,
                                       time_ranges):
    tasks_copy = [dict(t) for t in tasks_template]
    supply_per_slot = {
        idx: max(1, daily_data[
            (daily_data['time'] >= s) & (daily_data['time'] < e)
        ]['taxi_id'].nunique())
        for idx, (s, e) in enumerate(
            zip(time_ranges[:-1], time_ranges[1:]))
    }
    demand_info = _analyze_demand(daily_data, time_ranges)
    for t in tasks_copy:
        si = demand_info.get(t['time_slot'])
        if si:
            base  = (si['normalized_demand'] * 0.5
                     + (1 - si['normalized_supply']) * 0.25
                     + _time_factor(si['start_time']) * 0.25)
            prio  = base * (1 + min(t['passenger_demand'] / 10, 1))
            prio *= min(4.0, 1.0 + t['passenger_demand']
                        / supply_per_slot.get(t['time_slot'], 1))
            t['priority_score'] = prio
        else:
            t['priority_score'] = t['passenger_demand'] / 10.0
    scores = [t['priority_score'] for t in tasks_copy]
    if not scores:
        return 0.0, 1.0
    return min(scores), max(scores)

def normalize_with_ref(tasks, p_min, p_max):
    denom = p_max - p_min + 1e-6
    for t in tasks:
        raw = t.get('priority_score', 0.0)
        t['priority_score_raw'] = raw
        t['priority_score'] = float(
            np.clip((raw - p_min) / denom, 0.0, 1.0))
    return tasks

def generate_grid_tasks(time_ranges, gps_data,
                        lat_range, lon_range,
                        lat_step, lon_step):
    tasks, tid = [], 0
    lat_bins = np.arange(lat_range[0], lat_range[1], lat_step)
    lon_bins = np.arange(lon_range[0], lon_range[1], lon_step)
    for slot_idx, (s, e) in enumerate(
            zip(time_ranges[:-1], time_ranges[1:])):
        slot_data = gps_data[
            (gps_data['time'] >= s) & (gps_data['time'] < e)]
        if slot_data.empty:
            continue
        for i, lat0 in enumerate(lat_bins):
            for j, lon0 in enumerate(lon_bins):
                cell = slot_data[
                    slot_data['latitude'].between(
                        lat0, lat0 + lat_step, inclusive='left') &
                    slot_data['longitude'].between(
                        lon0, lon0 + lon_step, inclusive='left')]
                p = cell['taxi_id'].nunique()
                if p > 0:
                    tasks.append({
                        "task_id":          tid,
                        "start_time":       s,
                        "time_slot":        slot_idx,
                        "grid_idx":         (i, j),
                        "centroid_lat":     lat0 + lat_step / 2,
                        "centroid_lon":     lon0 + lon_step / 2,
                        "passenger_demand": p,
                    })
                    tid += 1
    return tasks

class TaskAllocator:
    def __init__(self, max_tasks_per_taxi, gamma=1.2):
        self.max_tasks = max_tasks_per_taxi
        self.gamma     = gamma

    def _get_fitness(self, taxi, task):
        taxi_score = taxi.get('composite_score', 0.5)
        task_score = task.get('priority_score',  0.5)
        rep_bonus  = taxi.get('daily_reputation_norm', 0.5)

        base = W_COMPOSITE * taxi_score + W_PRIORITY * task_score
        raw  = (base + W_REP_BONUS * rep_bonus) ** self.gamma

        return float(min(1.0, SIGMOID_LAMBDA / (
            1 + math.exp(-SIGMOID_SCALE * (raw - SIGMOID_SHIFT)))))

    def allocate(self, taxis, tasks):
        if not tasks:
            return []

        thr    = np.percentile(
            [t['priority_score'] for t in tasks], 80)
        high   = [t for t in tasks if t['priority_score'] >= thr]
        normal = [t for t in tasks if t['priority_score'] <  thr]

        allocs = []
        a1 = self._hungarian(taxis, high, ratio=1.0)
        allocs.extend(a1)

        used = pd.Series(
            [x['taxi_id'] for x in a1]).value_counts().to_dict()
        rem  = taxis.copy()
        rem['rem_cap'] = self.max_tasks
        for tid, cnt in used.items():
            idx = rem[rem['taxi_id'] == tid].index
            if not idx.empty:
                rem.loc[idx, 'rem_cap'] -= cnt
        rem = rem[rem['rem_cap'] > 0]

        if normal and not rem.empty:
            cap_map = rem.set_index('taxi_id')['rem_cap'].to_dict()
            a2 = self._hungarian(
                rem, normal, ratio=0.8, cap_map=cap_map)
            allocs.extend(a2)

        return allocs

    def _hungarian(self, taxis, tasks, ratio=1.0, cap_map=None):
        n = (max(1, int(len(taxis) * ratio))
             if not taxis.empty else 0)
        pruned = taxis.head(n)
        if pruned.empty or not tasks:
            return []

        expanded = []
        for r in pruned.to_dict('records'):
            cap = int(cap_map.get(r['taxi_id'], self.max_tasks)
                      if cap_map else self.max_tasks)
            expanded.extend([r] * cap)
        if not expanded:
            return []

        cost = np.array([
            [-self._get_fitness(t, tk) for tk in tasks]
            for t in expanded
        ])
        ri, ci = linear_sum_assignment(cost)

        results = []
        for r, c in zip(ri, ci):
            taxi = expanded[r]
            task = tasks[c]
            fit  = -cost[r, c]
            pc   = min(1.0,
                       taxi.get('wtcs', 0.5) * 0.8
                       + taxi.get('daily_reputation', 0.5) * 0.2)
            results.append({
                'taxi_id':              taxi['taxi_id'],
                'task_id':              task['task_id'],
                'task_start_time':      task['start_time'],
                'task_centroid_lat':    task['centroid_lat'],
                'task_centroid_lon':    task['centroid_lon'],
                'task_demand':          task.get('passenger_demand', 1),
                'fitness_score':        fit,
                'predicted_completion': pc,
                'allocation_quality':
                    fit * task.get('priority_score', 0.5) * pc,
            })
        return results

def compute_metrics(allocations, tasks, taxis_df,
                    daily_gps, n_workers):
    if not allocations:
        return None

    df       = pd.DataFrame(allocations)
    asgn     = df['taxi_id'].value_counts()
    task_map = {t['task_id']: t for t in tasks}

    total_demand = max(
        sum(t.get('passenger_demand', 1) for t in tasks), 1)
    assigned_demand = sum(
        task_map[tid].get('passenger_demand', 1)
        for tid in df['task_id'].unique()
        if tid in task_map)
    dsr = assigned_demand / total_demand

    pos_index       = build_daily_position_index(daily_gps)
    total_travel_km = 0.0

    for taxi_id, group in df.groupby('taxi_id'):
        seq = sorted(
            group.to_dict('records'),
            key=lambda r: (
                pd.to_datetime(
                    r.get('task_start_time', pd.Timestamp.min)),
                r.get('task_id', -1)))
        prev_loc     = None
        prev_arrival = None

        for row in seq:
            task_start = pd.to_datetime(row.get('task_start_time'))
            if task_start is None:
                continue
            task_loc = (row['task_centroid_lat'],
                        row['task_centroid_lon'])

            if prev_loc is None:
                origin = get_position_at(
                    pos_index, taxi_id, task_start)
                depart = task_start
            else:
                gap_min = (
                    (task_start - prev_arrival).total_seconds() / 60.0
                    if prev_arrival is not None else SLOT_MINUTES)
                if gap_min >= SLOT_MINUTES:
                    origin = get_position_at(
                        pos_index, taxi_id, task_start)
                    if origin is None:
                        origin = prev_loc
                    depart = task_start
                else:
                    origin = prev_loc
                    depart = max(prev_arrival, task_start)

            if origin is None:
                continue

            dist_km  = haversine_km(
                origin[0], origin[1],
                task_loc[0], task_loc[1])
            travel_m = (dist_km / AVG_SPEED_KMH) * 60.0
            arrival  = depart + pd.Timedelta(minutes=travel_m)
            total_travel_km += dist_km

            prev_loc     = task_loc
            prev_arrival = arrival

    tsd = (total_travel_km / assigned_demand
           if assigned_demand > 0 else 0.0)

    return {
        'total_allocations':        len(df),
        'avg_fitness_score':        float(df['fitness_score'].mean()),
        'avg_predicted_completion': float(
            df['predicted_completion'].mean()),
        'avg_allocation_quality':   float(
            df['allocation_quality'].mean()),
        'task_coverage':
            df['task_id'].nunique() / max(len(tasks), 1),
        'taxi_utilization':
            len(asgn) / max(n_workers, 1),
        'load_balance_index':
            (float(asgn.std() / asgn.mean())
             if len(asgn) > 1 else 0.0),
        'demand_satisfaction_rate':    round(float(dsr), 4),
        'travel_km_per_served_demand': round(float(tsd), 4),
        'total_travel_km':             round(float(total_travel_km), 4),
        'total_demand':                float(total_demand),
        'assigned_demand':             float(assigned_demand),
    }

def compute_assignment_overlap(allocs_base, allocs_gamma):
    """
    以 γ=1.0 的分配结果为基准，计算当前 γ 的匹配重合情况。

    返回：
      pair_overlap_ratio : |A_gamma ∩ A_base| / |A_base|
      jaccard             : |A_gamma ∩ A_base| / |A_gamma ∪ A_base|
      worker_overlap      : 被选中 worker 集合的 Jaccard
      task_overlap        : 被服务 task 集合的 Jaccard
    """

    pairs_base  = set(
        (r['taxi_id'], r['task_id']) for r in allocs_base)
    pairs_gamma = set(
        (r['taxi_id'], r['task_id']) for r in allocs_gamma)

    inter = pairs_base & pairs_gamma
    union = pairs_base | pairs_gamma

    pair_overlap_ratio = (len(inter) / len(pairs_base)
                          if pairs_base else 0.0)
    jaccard = (len(inter) / len(union)
               if union else 0.0)

    workers_base  = set(r['taxi_id'] for r in allocs_base)
    workers_gamma = set(r['taxi_id'] for r in allocs_gamma)
    w_inter = workers_base & workers_gamma
    w_union = workers_base | workers_gamma
    worker_overlap = (len(w_inter) / len(w_union)
                      if w_union else 0.0)

    tasks_base  = set(r['task_id'] for r in allocs_base)
    tasks_gamma = set(r['task_id'] for r in allocs_gamma)
    t_inter = tasks_base & tasks_gamma
    t_union = tasks_base | tasks_gamma
    task_overlap = (len(t_inter) / len(t_union)
                    if t_union else 0.0)

    return {
        'pair_overlap_ratio': round(pair_overlap_ratio, 4),
        'jaccard':            round(jaccard, 4),
        'worker_overlap':     round(worker_overlap, 4),
        'task_overlap':       round(task_overlap, 4),
    }

def run_single_gamma(gamma_val, filtered_date_range,
                     gps_data1, result_df,
                     lat_min, lat_max, lon_min, lon_max,
                     lat_step_val, lon_step_val,
                     daily_start_time, daily_end_time):
    """
    对单个 gamma_val 跑完所有日期，返回：
      - daily_metrics : list of dict（每日指标）
      - daily_allocs  : dict {date -> list of allocation records}
    """
    daily_metrics = []
    daily_allocs  = {}

    for date in tqdm(filtered_date_range,
                     desc=f"γ={gamma_val:.1f}", leave=False):

        daily_data = gps_data1[
            gps_data1['time'].dt.date == date.date()].copy()
        if daily_data.empty:
            continue

        s_ts = pd.to_datetime(f'{date.date()} {daily_start_time}')
        e_ts = pd.to_datetime(f'{date.date()} {daily_end_time}')
        tr   = pd.date_range(start=s_ts, end=e_ts, freq='15min')

        tasks_template = generate_grid_tasks(
            tr, daily_data,
            (lat_min, lat_max), (lon_min, lon_max),
            lat_step_val, lon_step_val)
        if not tasks_template:
            continue

        p_min, p_max = compute_full_model_priority_bounds(
            tasks_template, daily_data, tr)

        demand_info     = _analyze_demand(daily_data, tr)
        supply_per_slot = {
            idx: max(1, daily_data[
                (daily_data['time'] >= s) & (daily_data['time'] < e)
            ]['taxi_id'].nunique())
            for idx, (s, e) in enumerate(zip(tr[:-1], tr[1:]))
        }

        tasks = [dict(t) for t in tasks_template]
        for t in tasks:
            si = demand_info.get(t['time_slot'])
            if si:
                base  = (si['normalized_demand'] * 0.5
                         + (1 - si['normalized_supply']) * 0.25
                         + _time_factor(si['start_time']) * 0.25)
                prio  = base * (1 + min(t['passenger_demand'] / 10, 1))
                prio *= min(4.0, 1.0 + t['passenger_demand']
                            / supply_per_slot.get(t['time_slot'], 1))
                t['priority_score'] = prio
            else:
                t['priority_score'] = t['passenger_demand'] / 10.0

        tasks = normalize_with_ref(tasks, p_min, p_max)
        tasks = sorted(tasks,
                       key=lambda x: x.get('priority_score', 0),
                       reverse=True)

        taxi_df = result_df[result_df['date'] == date.date()].copy()
        if taxi_df.empty:
            continue

        taxi_df['composite_score'] = (
            0.5 * taxi_df['daily_reputation']
            + 0.5 * taxi_df['wtcs'])
        taxi_df['daily_reputation_norm'] = _minmax_series(
            taxi_df['daily_reputation'])

        taxis     = taxi_df.sort_values('composite_score',
                                        ascending=False)
        n_workers = len(taxis)

        allocator = TaskAllocator(
            max_tasks_per_taxi=FIXED_CAPACITY,
            gamma=gamma_val)
        allocs = allocator.allocate(taxis, tasks)

        if allocs:
            metrics = compute_metrics(
                allocs, tasks, taxis, daily_data, n_workers)
            if metrics:
                daily_metrics.append(metrics)

            daily_allocs[date.date()] = allocs

    return daily_metrics, daily_allocs

if __name__ == '__main__':
    output_dir = 'bdta_sensitivity_gamma_results'
    os.makedirs(output_dir, exist_ok=True)

    print()
    print()
    print()
    print()

    all_results = []

    print()
    base_metrics, base_daily_allocs = run_single_gamma(
        gamma_val        = 1.0,
        filtered_date_range = filtered_date_range,
        gps_data1        = gps_data1,
        result_df        = result_df,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        lat_step_val     = lat_step_val,
        lon_step_val     = lon_step_val,
        daily_start_time = daily_start_time,
        daily_end_time   = daily_end_time,
    )

    if base_metrics:
        avg_base = pd.DataFrame(base_metrics).mean(
            numeric_only=True).to_dict()
        avg_base['gamma']          = 1.0
        avg_base['fixed_capacity'] = FIXED_CAPACITY

        avg_base['pair_overlap_ratio'] = 1.0
        avg_base['jaccard']            = 1.0
        avg_base['worker_overlap']     = 1.0
        avg_base['task_overlap']       = 1.0
        all_results.append(avg_base)
        print()

    print()
    for gamma_val in tqdm(GAMMA_LEVELS[1:], desc="Varying γ"):

        daily_metrics, daily_allocs = run_single_gamma(
            gamma_val        = gamma_val,
            filtered_date_range = filtered_date_range,
            gps_data1        = gps_data1,
            result_df        = result_df,
            lat_min=lat_min, lat_max=lat_max,
            lon_min=lon_min, lon_max=lon_max,
            lat_step_val     = lat_step_val,
            lon_step_val     = lon_step_val,
            daily_start_time = daily_start_time,
            daily_end_time   = daily_end_time,
        )

        if not daily_metrics:
            continue

        overlap_records = []
        for date_key, allocs_gamma in daily_allocs.items():
            allocs_base = base_daily_allocs.get(date_key)
            if allocs_base is None:
                continue
            ov = compute_assignment_overlap(allocs_base, allocs_gamma)
            overlap_records.append(ov)

        if overlap_records:
            avg_overlap = pd.DataFrame(overlap_records).mean(
                numeric_only=True).to_dict()
        else:
            avg_overlap = {
                'pair_overlap_ratio': np.nan,
                'jaccard':            np.nan,
                'worker_overlap':     np.nan,
                'task_overlap':       np.nan,
            }

        avg = pd.DataFrame(daily_metrics).mean(
            numeric_only=True).to_dict()
        avg['gamma']          = gamma_val
        avg['fixed_capacity'] = FIXED_CAPACITY
        avg.update(avg_overlap)
        all_results.append(avg)

        print()

    if all_results:
        final_df  = pd.DataFrame(all_results)
        save_path = os.path.join(output_dir, RESULTS_FILENAME)
        final_df.to_csv(save_path, index=False)
        print()

        print()
        cols = ['gamma',
                'avg_allocation_quality',
                'avg_fitness_score',
                'demand_satisfaction_rate',
                'travel_km_per_served_demand',
                'task_coverage',
                'pair_overlap_ratio',
                'jaccard',
                'worker_overlap',
                'task_overlap']
        print()

        print()
        overlap_cols = ['gamma',
                        'pair_overlap_ratio',
                        'jaccard',
                        'worker_overlap',
                        'task_overlap']
        print()

        print()
        min_pair = final_df['pair_overlap_ratio'].min()
        min_jacc = final_df['jaccard'].min()
        min_work = final_df['worker_overlap'].min()
        min_task = final_df['task_overlap'].min()

        print()
        print()
        print()
        print()

        if min_pair >= 0.95:
            print()
        elif min_pair >= 0.85:
            print()
        else:
            print()
    else:
        print()