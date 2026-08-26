

import os
import random
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import pickle
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

RANDOM_SEED   = 42
MISSING_RATES = [0.0, 0.2, 0.4, 0.6, 0.8]
TARGET_C      = 6
SLOT_MINUTES  = 15
AVG_SPEED_KMH = 40.0
DEADLINE_MINUTES = 15

ADAPTIVE_WINDOW_DAYS = 7
ADAPTIVE_MIN_DAYS    = 1
WEIGHT_SMOOTH_ALPHA  = 0.3

W_COMPOSITE   = 0.5
W_PRIORITY    = 0.5
W_REP_BONUS   = 0.3
POWER_EXP     = 1.2
SIGMOID_SCALE = 2.0
SIGMOID_LAMBDA = 1.1

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

DATASET_CONFIGS = {
    'Beijing': {
        'pkl':      'beijing_preprocessed_data_revise.pkl',
        'lat_min':  39.8,  'lat_max': 40.1,
        'lon_min': 116.3,  'lon_max': 116.7,
        'lat_step': 0.015, 'lon_step': 0.015,
        'excluded': [],
        'is_beijing': True,
    },
    'Chengdu': {
        'pkl':      os.path.join('SiChuan', 'full_preprocessed_data_revise.pkl'),
        'lat_min':  30.55, 'lat_max': 30.75,
        'lon_min': 103.9,  'lon_max': 104.2,
        'lat_step': 0.015, 'lon_step': 0.015,
        'excluded': [
            pd.Timestamp('2014-08-07').date(),
            pd.Timestamp('2014-08-13').date(),
        ],
        'is_beijing': False,
    },
}

DATASETS_TO_RUN = ['Beijing', 'Chengdu']
OUTPUT_CSV  = 'robustness_missing_gps_results.csv'
OUTPUT_DIR  = 'robustness_results'

def haversine_km(lat1, lon1, lat2, lon2):
    R    = 6371.0
    phi1 = radians(lat1); phi2 = radians(lat2)
    dp   = radians(lat2 - lat1); dl = radians(lon2 - lon1)
    a    = sin(dp/2)**2 + cos(phi1)*cos(phi2)*sin(dl/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

class AdaptiveWeightLearner:
    DEFAULT_WEIGHTS = {
        'passenger_demand': 0.5,
        'taxi_supply':      0.25,
        'time_factor':      0.25,
    }
    DEFAULT_TIME_FACTORS = {
        h: (1.0 if h in [7, 8, 9, 17, 18, 19]
            else 0.8 if h in [11, 12, 13]
            else 0.6)
        for h in range(24)
    }

    def __init__(self, is_beijing=False):
        self.is_beijing        = is_beijing
        self.weights           = dict(self.DEFAULT_WEIGHTS)
        self.time_factors      = dict(self.DEFAULT_TIME_FACTORS)
        self._history          = {}
        self._adaptive_enabled = False

    def update(self, cur_date, gps_data, tr):
        window_dates = sorted(self._history.keys())
        if len(window_dates) < ADAPTIVE_MIN_DAYS:
            return
        self._adaptive_enabled = True
        use_dates = window_dates[-ADAPTIVE_WINDOW_DAYS:]

        hds = defaultdict(float); hdc = defaultdict(int)
        for d in use_dates:
            for h, demand in self._history[d].get('hour_demand', {}).items():
                hds[h] += demand; hdc[h] += 1
        hdm = {h: hds[h]/hdc[h] for h in hds if hdc[h] > 0}
        if hdm:
            md  = max(hdm.values()) + 1e-8
            ntf = {h: float(np.clip(v/md, 0.3, 1.0))
                   for h, v in hdm.items()}
            for h in range(24):
                old = self.time_factors.get(h, self.DEFAULT_TIME_FACTORS[h])
                new = ntf.get(h, old)
                self.time_factors[h] = (WEIGHT_SMOOTH_ALPHA * new
                                        + (1 - WEIGHT_SMOOTH_ALPHA) * old)

        ds = []; ss = []
        for d in use_dates:
            ds.extend(self._history[d].get('slot_demand', []))
            ss.extend(self._history[d].get('slot_supply', []))
        if len(ds) >= 2 and len(ss) >= 2:
            vd = float(np.var(ds)); vs = float(np.var(ss))
            vt = float(np.var(list(self.time_factors.values())))
            tt = vd + vs + vt + 1e-8
            nd = float(np.clip(vd/tt, 0.2, 0.7))
            ns = float(np.clip(vs/tt, 0.1, 0.5))
            nt = float(np.clip(vt/tt, 0.1, 0.4))
            nm = nd + ns + nt
            nd /= nm; ns /= nm; nt /= nm
            a = WEIGHT_SMOOTH_ALPHA
            self.weights['passenger_demand'] = (
                a*nd + (1-a)*self.weights['passenger_demand'])
            self.weights['taxi_supply'] = (
                a*ns + (1-a)*self.weights['taxi_supply'])
            self.weights['time_factor'] = (
                a*nt + (1-a)*self.weights['time_factor'])

    def record_day(self, cur_date, gps_data, tr):
        slot_demand = []; slot_supply = []; hour_demand = {}
        for s, e in zip(tr[:-1], tr[1:]):
            data = gps_data[
                (gps_data['time'] >= s) & (gps_data['time'] < e)]
            if data.empty: continue
            h = s.hour
            if self.is_beijing:
                n   = data['taxi_id'].nunique()
                p   = min(1.0, n/50); sup = min(1.0, n/30)
            else:
                st  = {tid: (g['status'] == 1).any()
                       for tid, g in data.groupby('taxi_id')}
                p   = min(1.0, sum(st.values())/50)
                sup = min(1.0, (len(st) - sum(st.values()))/30)
            slot_demand.append(p); slot_supply.append(sup)
            hour_demand[h] = hour_demand.get(h, 0.0) + p
        self._history[cur_date] = {
            'slot_demand': slot_demand,
            'slot_supply': slot_supply,
            'hour_demand': hour_demand,
        }

    def get_weights(self):       return dict(self.weights)
    def get_time_factor(self, h): return self.time_factors.get(h, 0.6)
    def is_adaptive(self):       return self._adaptive_enabled

def drop_gps_randomly(daily_gps, missing_rate, seed=42):
    if missing_rate == 0.0:
        return daily_gps.copy()
    rng = np.random.RandomState(seed)
    keep_indices = []
    for tid, group in daily_gps.groupby('taxi_id'):
        n      = len(group)
        n_keep = max(1, int(round(n * (1.0 - missing_rate))))
        chosen = rng.choice(group.index, size=n_keep, replace=False)
        keep_indices.extend(chosen.tolist())
    return daily_gps.loc[sorted(keep_indices)].copy()

def recompute_worker_metrics(daily_gps, taxi_ids, time_ranges,
                              lat_range, lon_range, lat_step, lon_step):
    FALLBACK_ALPHA = 0.70; FALLBACK_BETA = 0.80; FALLBACK_DELTA = 0.20
    MAX_POINTS_PER_SLOT = 30
    n_slots = len(time_ranges) - 1
    records = []

    for tid in taxi_ids:
        worker_gps = daily_gps[daily_gps['taxi_id'] == tid]
        if worker_gps.empty:
            wtcs = (0.5*FALLBACK_ALPHA + 0.3*FALLBACK_BETA
                    + 0.2*(1.0 - FALLBACK_DELTA))
            records.append({
                'taxi_id': tid,
                'alpha_w': FALLBACK_ALPHA,
                'beta_w': FALLBACK_BETA,
                'delta_w': FALLBACK_DELTA,
                'wtcs': float(np.clip(wtcs, 0, 1)),
                'daily_reputation': float(np.clip(wtcs, 0, 1)),
            })
            continue

        slot_arrived = []; slot_point_count = []
        for slot_idx in range(n_slots):
            s = time_ranges[slot_idx]; e = time_ranges[slot_idx + 1]
            slot_gps = worker_gps[
                (worker_gps['time'] >= s) & (worker_gps['time'] < e) &
                worker_gps['latitude'].between(lat_range[0], lat_range[1]) &
                worker_gps['longitude'].between(lon_range[0], lon_range[1])]
            slot_arrived.append(int(len(slot_gps) > 0))
            slot_point_count.append(len(slot_gps))

        alpha_w = (float(np.mean(slot_arrived))
                   if slot_arrived else FALLBACK_ALPHA)
        nonzero = [c for c in slot_point_count if c > 0]
        beta_w  = (float(np.clip(np.mean(nonzero)/MAX_POINTS_PER_SLOT, 0, 1))
                   if nonzero else FALLBACK_BETA)
        delta_w = float(np.clip(1.0 - beta_w, 0, 1))
        wtcs    = 0.5*alpha_w + 0.3*beta_w + 0.2*(1.0 - delta_w)
        records.append({
            'taxi_id': tid,
            'alpha_w': alpha_w,
            'beta_w': beta_w,
            'delta_w': delta_w,
            'wtcs': float(np.clip(wtcs, 0, 1)),
            'daily_reputation': float(np.clip(wtcs, 0, 1)),
        })

    df_metrics = pd.DataFrame(records)

    df_metrics['composite_score'] = (
        df_metrics['daily_reputation'] * 0.5
        + df_metrics['wtcs'] * 0.5)

    lo = df_metrics['daily_reputation'].min()
    hi = df_metrics['daily_reputation'].max()
    if hi - lo < 1e-6:
        df_metrics['daily_reputation_norm'] = 0.5
    else:
        df_metrics['daily_reputation_norm'] = (
            (df_metrics['daily_reputation'] - lo) / (hi - lo))

    return df_metrics

def generate_grid_tasks(time_ranges, gps_data,
                         lat_range, lon_range,
                         lat_step, lon_step):
    tasks = []; task_id_counter = 0
    lat_bins = np.arange(lat_range[0], lat_range[1], lat_step)
    lon_bins = np.arange(lon_range[0], lon_range[1], lon_step)

    for slot_idx, (start, end) in enumerate(
            zip(time_ranges[:-1], time_ranges[1:])):
        slot_data = gps_data[
            (gps_data['time'] >= start) & (gps_data['time'] < end)]
        if slot_data.empty: continue
        for i, lat0 in enumerate(lat_bins):
            for j, lon0 in enumerate(lon_bins):
                cell = slot_data[
                    slot_data['latitude'].between(
                        lat0, lat0+lat_step, inclusive='left') &
                    slot_data['longitude'].between(
                        lon0, lon0+lon_step, inclusive='left')]
                demand = cell['taxi_id'].nunique()
                if demand > 0:
                    c_lat = lat0 + lat_step/2
                    c_lon = lon0 + lon_step/2
                    tasks.append({
                        'task_id':          task_id_counter,
                        'time_slot':        slot_idx,
                        'start_time':       start,
                        'passenger_demand': demand,
                        'priority_score':   float(demand),
                        'centroid_lat':     c_lat,
                        'centroid_lon':     c_lon,
                        'grid_idx':         (i, j),
                        'lat_range':        (lat0, lat_step),
                        'lon_range':        (lon0, lon_step),
                        'centroid':         (c_lat, c_lon),
                    })
                    task_id_counter += 1
    return tasks

def apply_three_layer_priority(tasks, gps_data, time_ranges,
                                is_beijing, learner=None):
    if not tasks:
        return tasks

    use_adaptive = (learner is not None and learner.is_adaptive())
    FIXED_PW     = {
        'passenger_demand': 0.5,
        'taxi_supply':      0.25,
        'time_factor':      0.25,
    }

    demand_info = {}; supply_rt = {}
    for i, (s, e) in enumerate(zip(time_ranges[:-1], time_ranges[1:])):
        data = gps_data[(gps_data['time'] >= s) & (gps_data['time'] < e)]
        if is_beijing:
            n   = data['taxi_id'].nunique(); pd_ = ts_ = n
        else:
            st_ = {tid: (g['status'] == 1).any()
                   for tid, g in data.groupby('taxi_id')}
            pd_ = sum(st_.values()); ts_ = len(st_) - pd_
        demand_info[i] = {
            'start_time':        s,
            'normalized_demand': min(1.0, pd_/50),
            'normalized_supply': min(1.0, ts_/30),
        }
        supply_rt[i] = max(1, ts_)

    weights = learner.get_weights() if use_adaptive else FIXED_PW

    for task in tasks:
        si = demand_info.get(task['time_slot'])
        if si is None:
            task['priority_score'] = task['passenger_demand'] / 10.0
            continue
        h  = si['start_time'].hour
        tf = (learner.get_time_factor(h) if use_adaptive
              else (1.0 if h in [7,8,9,17,18,19]
                    else 0.8 if h in [11,12,13] else 0.6))
        p_macro = (si['normalized_demand'] * weights['passenger_demand']
                   + (1 - si['normalized_supply']) * weights['taxi_supply']
                   + tf * weights['time_factor'])
        d_micro = task['passenger_demand']
        p_base  = p_macro * (1 + min(d_micro / 10.0, 1))
        s_rt    = supply_rt.get(task['time_slot'], 1)
        adj     = min(4.0, 1.0 + d_micro / max(1, s_rt))
        task['priority_score'] = p_base * adj

    return tasks

def normalize_task_priorities(tasks):
    if not tasks: return tasks
    scores       = [t['priority_score'] for t in tasks]
    p_min, p_max = min(scores), max(scores)
    eps          = 1e-6
    for t in tasks:
        raw = t['priority_score']
        t['priority_score_raw'] = raw
        t['priority_score']     = (raw - p_min) / (p_max - p_min + eps)
    return tasks

def build_reachability_matrix(worker_locs, task_locs):
    if len(worker_locs) == 0 or len(task_locs) == 0:
        return np.zeros((len(worker_locs), len(task_locs)), dtype=bool)
    w_lat = np.radians(worker_locs[:, 0:1])
    w_lon = np.radians(worker_locs[:, 1:2])
    t_lat = np.radians(task_locs[:, 0])
    t_lon = np.radians(task_locs[:, 1])
    dlat  = t_lat - w_lat; dlon = t_lon - w_lon
    a     = (np.sin(dlat/2)**2
             + np.cos(w_lat)*np.cos(t_lat)*np.sin(dlon/2)**2)
    dist_km  = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    travel_m = (dist_km / AVG_SPEED_KMH) * 60.0
    return travel_m <= SLOT_MINUTES

def precompute_reachability(daily_gps, daily_tasks, taxi_ids, time_ranges):
    pos_index = {}
    for tid, grp in daily_gps.groupby('taxi_id'):
        pos_index[tid] = (grp[['time','latitude','longitude']]
                          .sort_values('time').reset_index(drop=True))
    tasks_by_slot = {}
    for task in daily_tasks:
        tasks_by_slot.setdefault(task['time_slot'], []).append(task)
    slot_reachability = {}
    for slot_idx, slot_tasks in tasks_by_slot.items():
        if slot_idx >= len(time_ranges) - 1: continue
        slot_start = time_ranges[slot_idx]
        w_lats = []; w_lons = []; valid_ids = []
        for tid in taxi_ids:
            rec = pos_index.get(tid)
            if rec is None or rec.empty: continue
            before = rec[rec['time'] <= slot_start]
            if before.empty: continue
            last = before.iloc[-1]
            valid_ids.append(tid)
            w_lats.append(last['latitude'])
            w_lons.append(last['longitude'])
        task_ids = [t['task_id'] for t in slot_tasks]
        if not valid_ids:
            slot_reachability[slot_idx] = {
                'taxi_ids': [], 'task_ids': task_ids,
                'matrix': np.zeros((0, len(slot_tasks)), dtype=bool),
                'taxi_id_to_idx': {},
                'task_id_to_idx': {
                    tid: i for i, tid in enumerate(task_ids)}}
            continue
        w_arr  = np.array(list(zip(w_lats, w_lons)))
        t_arr  = np.array([[t['centroid_lat'], t['centroid_lon']]
                           for t in slot_tasks])
        matrix = build_reachability_matrix(w_arr, t_arr)
        slot_reachability[slot_idx] = {
            'taxi_ids': valid_ids, 'task_ids': task_ids,
            'matrix': matrix,
            'taxi_id_to_idx': {
                tid: i for i, tid in enumerate(valid_ids)},
            'task_id_to_idx': {
                tid: i for i, tid in enumerate(task_ids)}}
    return slot_reachability

def check_reachable(slot_reachability, taxi_id, task_id, slot_idx):
    sr = slot_reachability.get(slot_idx)
    if sr is None: return True
    w_idx = sr['taxi_id_to_idx'].get(taxi_id)
    t_idx = sr['task_id_to_idx'].get(task_id)
    if w_idx is None or t_idx is None: return True
    return bool(sr['matrix'][w_idx, t_idx])

def compute_fitness(taxi, task, is_beijing=True):
    """
    ★ 与主实验 TaskAllocator._get_fitness() 完全一致：
      - W_COMPOSITE=0.5, W_PRIORITY=0.5
      - rep_bonus 使用 daily_reputation_norm（min-max 归一化）
      - SIGMOID_SCALE=2.0
      - sigmoid_shift 随数据集变化（Beijing=1.0，Chengdu=1.5）
    """
    taxi_score = taxi.get('composite_score', 0.5)
    task_score = task.get('priority_score',  0.5)
    base       = W_COMPOSITE * taxi_score + W_PRIORITY * task_score

    rep_bonus  = taxi.get('daily_reputation_norm', 0.5)
    raw        = (base + W_REP_BONUS * rep_bonus) ** POWER_EXP

    shift = 1.0 if is_beijing else 1.5

    return float(min(1.0, SIGMOID_LAMBDA / (
        1 + math.exp(-SIGMOID_SCALE * (raw - shift)))))

def bdta_allocate(worker_metrics_df, daily_tasks,
                  slot_reachability, c_value=6, is_beijing=True):
    if worker_metrics_df.empty or not daily_tasks:
        return []
    taxis = worker_metrics_df.sort_values(
        'composite_score', ascending=False)
    tasks = sorted(daily_tasks,
                   key=lambda t: t.get('priority_score', 0),
                   reverse=True)

    threshold    = np.percentile(
        [t['priority_score'] for t in tasks], 80)
    high_tasks   = [t for t in tasks if t['priority_score'] >= threshold]
    normal_tasks = [t for t in tasks if t['priority_score'] <  threshold]

    def hungarian_with_expansion(taxi_df, task_list, cap_map=None):
        if taxi_df.empty or not task_list: return []
        expanded = []
        for tx in taxi_df.to_dict('records'):
            cap = int(cap_map.get(tx['taxi_id'], c_value)
                      if cap_map else c_value)
            expanded.extend([tx] * cap)
        if not expanded: return []
        INF  = 1e9
        cost = np.full((len(expanded), len(task_list)), INF)
        for i, tx in enumerate(expanded):
            for j, tk in enumerate(task_list):
                if check_reachable(slot_reachability,
                                   tx['taxi_id'], tk['task_id'],
                                   tk.get('time_slot', -1)):

                    cost[i, j] = -compute_fitness(tx, tk, is_beijing)
        rows, cols = linear_sum_assignment(cost)
        allocs = []
        for r, c in zip(rows, cols):
            if cost[r, c] >= INF: continue
            tx = expanded[r]; tk = task_list[c]
            fitness = -cost[r, c]
            pred_c  = min(1.0,
                          tx.get('wtcs', 0.5) * 0.8
                          + tx.get('daily_reputation', 0.5) * 0.2)
            allocs.append({
                'taxi_id':              tx['taxi_id'],
                'task_id':              tk['task_id'],
                'fitness_score':        fitness,
                'predicted_completion': pred_c,
                'allocation_quality':
                    fitness * tk.get('priority_score', 0.5) * pred_c,
                'task_start_time':  tk['start_time'],
                'task_centroid_lat':tk['centroid_lat'],
                'task_centroid_lon':tk['centroid_lon'],
                'task_slot':        tk.get('time_slot', -1),
                'task_demand':      tk.get('passenger_demand', 1),
            })
        return allocs

    allocs_high = hungarian_with_expansion(taxis, high_tasks)
    all_allocs  = list(allocs_high)

    assigned_counts   = pd.Series(
        [a['taxi_id'] for a in allocs_high]).value_counts().to_dict()
    assigned_task_ids = {a['task_id'] for a in allocs_high}
    normal_tasks      = [t for t in normal_tasks
                         if t['task_id'] not in assigned_task_ids]

    remain = taxis.copy()
    remain['_rem_cap'] = c_value
    for tid, cnt in assigned_counts.items():
        remain.loc[remain['taxi_id'] == tid, '_rem_cap'] -= cnt
    remain  = remain[remain['_rem_cap'] > 0].copy()
    cap_map = remain.set_index('taxi_id')['_rem_cap'].to_dict()

    if normal_tasks and not remain.empty:
        n_select   = max(1, int(len(remain) * 0.8))
        remain_top = remain.head(n_select)
        allocs_norm = hungarian_with_expansion(
            remain_top, normal_tasks, cap_map)
        all_allocs.extend(allocs_norm)

    return all_allocs

def build_pos_index(daily_gps):
    pos_index = {}
    for tid, grp in daily_gps.groupby('taxi_id'):
        g = grp[['time', 'latitude', 'longitude']].sort_values('time')
        pos_index[tid] = {
            'times': g['time'].to_numpy(dtype='datetime64[ns]'),
            'lats':  g['latitude'].to_numpy(float),
            'lons':  g['longitude'].to_numpy(float),
        }
    return pos_index

def get_position_at(pos_index, taxi_id, query_time):
    entry = pos_index.get(taxi_id)
    if entry is None or len(entry['times']) == 0:
        return None
    i = np.searchsorted(
        entry['times'], np.datetime64(query_time), side='right') - 1
    if i < 0:
        return None
    return float(entry['lats'][i]), float(entry['lons'][i])

def compute_metrics(allocations, tasks, taxis_df, daily_gps, time_ranges):
    task_dict    = {t['task_id']: t for t in tasks}
    total_demand = sum(t.get('passenger_demand', 1) for t in tasks)
    n_tasks      = len(tasks)
    n_workers    = len(taxis_df)

    empty = {
        'total_allocations': 0, 'avg_fitness_score': 0.0,
        'avg_allocation_quality': 0.0, 'taxi_coverage': 0.0,
        'task_coverage': 0.0, 'load_balance_index': 0.0,
        'avg_tasks_per_taxi': 0.0, 'total_demand': float(total_demand),
        'assigned_demand': 0.0, 'on_time_demand': 0.0,
        'total_travel_km': 0.0, 'demand_satisfaction_rate': 0.0,
        'on_time_demand_satisfaction_rate': 0.0,
        'travel_km_per_served_demand': 0.0,
        'served_demand_per_km': 0.0,
        'worker_participation_rate': 0.0,
        'n_active_workers': 0.0, 'n_total_workers': float(n_workers),
        'n_allocated': 0.0, 'n_tasks': float(n_tasks),
    }
    if not allocations:
        return empty

    df          = pd.DataFrame(allocations)
    assignments = df['taxi_id'].value_counts()
    avg_fitness = float(df['fitness_score'].mean())
    avg_quality = float(df['allocation_quality'].mean())
    taxi_cov    = float(len(assignments) / max(n_workers, 1))
    task_cov    = float(df['task_id'].nunique() / max(n_tasks, 1))
    lbi         = float(assignments.std() / assignments.mean()
                        if len(assignments) > 1 else 0.0)
    avg_per_taxi = float(assignments.mean())

    pos_index       = build_pos_index(daily_gps)
    total_travel_km = 0.0
    on_time_demand  = 0.0
    assigned_demand = 0.0

    for taxi_id, group in df.groupby('taxi_id'):
        seq = sorted(
            group.to_dict('records'),
            key=lambda r: (
                pd.to_datetime(r.get('task_start_time', pd.Timestamp.min)),
                r.get('task_id', -1)))
        prev_loc     = None
        prev_arrival = None

        for row in seq:
            task_start = pd.to_datetime(row.get('task_start_time'))
            if task_start is None: continue
            task_loc   = (row['task_centroid_lat'],
                          row['task_centroid_lon'])
            p_demand_i = row.get('task_demand', 1)

            if prev_loc is None:
                origin = get_position_at(pos_index, taxi_id, task_start)
                depart = task_start
            else:
                gap_min = ((task_start - prev_arrival).total_seconds() / 60.0
                           if prev_arrival is not None else SLOT_MINUTES)
                if gap_min >= SLOT_MINUTES:
                    origin = get_position_at(pos_index, taxi_id, task_start)
                    if origin is None: origin = prev_loc
                    depart = task_start
                else:
                    origin = prev_loc
                    depart = max(prev_arrival, task_start)

            if origin is None: continue

            dist_km  = haversine_km(origin[0], origin[1],
                                    task_loc[0], task_loc[1])
            travel_m = (dist_km / AVG_SPEED_KMH) * 60.0
            arrival  = depart + pd.Timedelta(minutes=travel_m)
            deadline = task_start + pd.Timedelta(minutes=DEADLINE_MINUTES)

            total_travel_km += dist_km
            assigned_demand += p_demand_i
            if arrival <= deadline:
                on_time_demand += p_demand_i

            prev_loc     = task_loc
            prev_arrival = arrival

    dsr    = float(min(1.0, assigned_demand / max(total_demand, 1)))
    ot_dsr = float(min(1.0, on_time_demand  / max(total_demand, 1)))
    tk_per = float(total_travel_km / max(assigned_demand, 1e-6))
    sd_per = float(assigned_demand  / max(total_travel_km, 1e-6))
    wpr    = float(len(assignments)  / max(n_workers, 1))

    metrics = {
        'total_allocations':                len(df),
        'avg_fitness_score':                avg_fitness,
        'avg_allocation_quality':           avg_quality,
        'taxi_coverage':                    taxi_cov,
        'task_coverage':                    task_cov,
        'load_balance_index':               lbi,
        'avg_tasks_per_taxi':               avg_per_taxi,
        'total_demand':                     float(total_demand),
        'assigned_demand':                  float(assigned_demand),
        'on_time_demand':                   float(on_time_demand),
        'total_travel_km':                  float(total_travel_km),
        'demand_satisfaction_rate':         dsr,
        'on_time_demand_satisfaction_rate': ot_dsr,
        'travel_km_per_served_demand':      tk_per,
        'served_demand_per_km':             sd_per,
        'worker_participation_rate':        wpr,
        'n_active_workers':                 float(len(assignments)),
        'n_total_workers':                  float(n_workers),
        'n_allocated':                      float(len(assignments)),
        'n_tasks':                          float(n_tasks),
    }
    return {k: (0.0 if pd.isna(v) else v) for k, v in metrics.items()}

def run_robustness_for_dataset(dataset_name, cfg):
    print()
    print()
    print()
    print()
    print()

    try:
        with open(cfg['pkl'], 'rb') as f:
            data = pickle.load(f)
        gps_data  = data['gps_data1']
        result_df = data['result_df']
        gps_data['time'] = pd.to_datetime(gps_data['time'])
        result_df['date'] = pd.to_datetime(result_df['date']).dt.date
        print()
    except FileNotFoundError:
        print()
        return []

    date_range = pd.to_datetime(result_df['date'].unique())
    dates      = sorted([d for d in date_range
                         if d.date() not in cfg['excluded']])

    lat_range = (cfg['lat_min'], cfg['lat_max'])
    lon_range = (cfg['lon_min'], cfg['lon_max'])
    lat_step  = cfg['lat_step']
    lon_step  = cfg['lon_step']
    is_bj     = cfg['is_beijing']

    all_results = []

    for missing_rate in MISSING_RATES:
        print()
        print()
        print()

        learner = AdaptiveWeightLearner(is_beijing=is_bj)
        day_metrics_list = []

        for date in tqdm(dates, desc=f"  r={missing_rate:.0%}"):
            daily_gps = gps_data[
                gps_data['time'].dt.date == date.date()].copy()
            if daily_gps.empty: continue

            start_ts    = pd.to_datetime(f'{date.date()} 08:00:00')
            end_ts      = pd.to_datetime(f'{date.date()} 21:00:00')
            time_ranges = pd.date_range(
                start=start_ts, end=end_ts, freq='15min')

            learner.update(date.date(), daily_gps, time_ranges)

            daily_tasks = generate_grid_tasks(
                time_ranges, daily_gps,
                lat_range, lon_range, lat_step, lon_step)
            if not daily_tasks:
                learner.record_day(date.date(), daily_gps, time_ranges)
                continue

            daily_tasks = apply_three_layer_priority(
                daily_tasks, daily_gps, time_ranges, is_bj,
                learner=learner)
            daily_tasks = normalize_task_priorities(daily_tasks)

            seed_for_day = (RANDOM_SEED
                            + abs(hash(str(date.date()))) % 10000)
            gps_sparse   = drop_gps_randomly(
                daily_gps, missing_rate, seed=seed_for_day)

            taxi_ids = result_df[
                result_df['date'] == date.date()]['taxi_id'].tolist()
            if not taxi_ids:
                learner.record_day(date.date(), daily_gps, time_ranges)
                continue

            worker_metrics = recompute_worker_metrics(
                gps_sparse, taxi_ids, time_ranges,
                lat_range, lon_range, lat_step, lon_step)
            if worker_metrics.empty:
                learner.record_day(date.date(), daily_gps, time_ranges)
                continue

            slot_reach = precompute_reachability(
                daily_gps, daily_tasks, taxi_ids, time_ranges)

            allocations = bdta_allocate(
                worker_metrics, daily_tasks, slot_reach,
                TARGET_C, is_beijing=is_bj)

            day_metrics = compute_metrics(
                allocations, daily_tasks, worker_metrics,
                daily_gps, time_ranges)
            day_metrics_list.append(day_metrics)

            learner.record_day(date.date(), daily_gps, time_ranges)

        if not day_metrics_list:
            print()
            continue

        avg = pd.DataFrame(day_metrics_list).mean().to_dict()
        avg['missing_rate'] = missing_rate
        avg['dataset']      = dataset_name
        all_results.append(avg)

        print()
        for k in ['demand_satisfaction_rate',
                  'on_time_demand_satisfaction_rate',
                  'travel_km_per_served_demand',
                  'avg_fitness_score',
                  'avg_allocation_quality',
                  'worker_participation_rate',
                  'task_coverage']:
            if k in avg:
                print()

    return all_results

def make_latex_table(df_robust):
    key_metrics = {
        'demand_satisfaction_rate':         r'DSR $\uparrow$',
        'on_time_demand_satisfaction_rate': r'OT-DSR $\uparrow$',
        'travel_km_per_served_demand':      r'TSD $\downarrow$',
        'worker_participation_rate':        r'WPR $\uparrow$',
        'avg_fitness_score':                r'AFS $\uparrow$',
        'avg_allocation_quality':           r'OAQ $\uparrow$',
    }
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(
        r'\caption{Robustness of BDTA under simulated GPS data loss '
        r'($c=' + str(TARGET_C) + r'$). '
        r'Values are averaged over all evaluation days. '
        r'$r$ denotes the proportion of missing GPS records. '
        r'\textbf{Bold} values indicate the complete-data baseline '
        r'($r=0\%$). $\Delta$ denotes absolute change relative to '
        r'the baseline. '
        r'$\uparrow$ higher is better; $\downarrow$ lower is better.}')
    lines.append(r'\label{tab:robustness_missing}')
    lines.append(r'\centering')
    n_m      = len(key_metrics)
    col_spec = 'll' + 'X' * n_m
    lines.append(r'\begin{tabularx}{\linewidth}{' + col_spec + r'}')
    lines.append(r'\hline')
    lines.append(r'Dataset & $r$ & '
                 + ' & '.join(key_metrics.values()) + r' \\')
    lines.append(r'\hline')

    for dataset in DATASETS_TO_RUN:
        sub = df_robust[
            df_robust['dataset'] == dataset
        ].sort_values('missing_rate').reset_index(drop=True)
        if sub.empty: continue
        base_row = sub[sub['missing_rate'] == 0.0]
        if base_row.empty: continue
        base  = base_row.iloc[0]
        first = True; n = len(sub)

        for _, row in sub.iterrows():
            r_pct   = f"{int(row['missing_rate']*100)}\\%"
            is_base = (row['missing_rate'] == 0.0)
            cells   = []
            for col in key_metrics:
                val      = row.get(col, np.nan)
                base_val = base.get(col, np.nan)
                val_str  = f"{val:.4f}" if not np.isnan(val) else '---'
                if is_base:
                    cell = r'\textbf{' + val_str + r'}'
                else:
                    delta = val - base_val
                    sign  = '+' if delta >= 0 else ''
                    cell  = f"{val_str} ({sign}{delta:.4f})"
                cells.append(cell)
            row_str = ' & '.join(cells)
            if first:
                lines.append(
                    rf'\multirow{{{n}}}{{*}}{{{dataset}}} & '
                    rf'{r_pct} & {row_str} \\')
                first = False
            else:
                lines.append(rf' & {r_pct} & {row_str} \\')
        lines.append(r'\hline')

    lines.append(r'\end{tabularx}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = []

    for dataset_name in DATASETS_TO_RUN:
        cfg     = DATASET_CONFIGS[dataset_name]
        results = run_robustness_for_dataset(dataset_name, cfg)
        all_results.extend(results)

    if not all_results:
        print()
        exit()

    df_robust = pd.DataFrame(all_results)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
    df_robust.to_csv(save_path, index=False)
    print()

    print()
    show_cols = [c for c in [
        'dataset', 'missing_rate',
        'demand_satisfaction_rate',
        'on_time_demand_satisfaction_rate',
        'travel_km_per_served_demand',
        'worker_participation_rate',
        'avg_fitness_score',
        'avg_allocation_quality',
    ] if c in df_robust.columns]
    print()

    print()
    print()