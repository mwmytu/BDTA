

import os
import copy
import random
import math
import pandas as pd
import numpy as np
import time
import pickle
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

IS_BEIJING_DATASET = False

SLOT_MINUTES    = 15
AVG_SPEED_KMH   = 40.0
FIXED_W_TASK    = 0.5
FIXED_CAPACITY  = 6
NUM_REPETITIONS = 5

BASELINE_TOPK_RATIO = 0.8
QITA_REDUNDANCY_R   = 1
QITA_QUALITY_THETA  = 0.3
GA_N_POP            = 20
GA_MAX_ITER         = 50

ADAPTIVE_WINDOW_DAYS = 7
ADAPTIVE_MIN_DAYS    = 1
WEIGHT_SMOOTH_ALPHA  = 0.3

if IS_BEIJING_DATASET:
    print()
    PREPROCESSED_DATA_FILE = 'beijing_preprocessed_data_revise.pkl'
    RESULTS_FILENAME       = 'beijing_scalability_results.csv'
    data_dir               = '.'
    lat_min, lat_max       = 39.8,  40.1
    lon_min, lon_max       = 116.3, 116.7

    lat_step_val           = 0.015
    lon_step_val           = 0.015
    excluded_dates         = []
else:
    print()
    PREPROCESSED_DATA_FILE = 'full_preprocessed_data_revise.pkl'
    RESULTS_FILENAME       = 'chengdu_scalability_results.csv'
    data_dir               = 'SiChuan'
    lat_min, lat_max       = 30.55, 30.75
    lon_min, lon_max       = 103.9, 104.2
    lat_step_val           = 0.015
    lon_step_val           = 0.015
    excluded_dates         = [pd.Timestamp('2014-08-07').date(),
                              pd.Timestamp('2014-08-13').date()]

DAILY_START_TIME = pd.to_datetime('08:00:00').time()
DAILY_END_TIME   = pd.to_datetime('21:00:00').time()

WORKER_SCALES        = [100, 200, 300, 400, 500]
FIXED_TASK_COUNT_W   = 1000
TASK_SCALES          = [400, 800, 1200, 1600, 2000]
FIXED_WORKER_COUNT_T = 300

SCENARIOS_TO_RUN = ["BDTA", "PX", "QITA", "DTAA", "FTSA"]

SCENARIO_DISPLAY = {
    "BDTA": "BDTA",
    "PX":   "PX",
    "QITA": "QITA",
    "DTAA": "DTAA",
    "FTSA": "FTSA",
}

print()
print()

print()

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
        entry['times'],
        np.datetime64(query_time),
        side='right') - 1
    if i < 0:
        return None
    return float(entry['lats'][i]), float(entry['lons'][i])

def build_reachability_matrix(worker_locs, task_locs):
    if len(worker_locs) == 0 or len(task_locs) == 0:
        return np.zeros((len(worker_locs), len(task_locs)), dtype=bool)
    w_lat = np.radians(worker_locs[:, 0:1])
    w_lon = np.radians(worker_locs[:, 1:2])
    t_lat = np.radians(task_locs[:, 0])
    t_lon = np.radians(task_locs[:, 1])
    dlat  = t_lat - w_lat
    dlon  = t_lon - w_lon
    a     = (np.sin(dlat / 2) ** 2
             + np.cos(w_lat) * np.cos(t_lat) * np.sin(dlon / 2) ** 2)
    dist_km  = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    travel_m = (dist_km / AVG_SPEED_KMH) * 60.0
    return travel_m <= SLOT_MINUTES

def precompute_daily_reachability(daily_gps, daily_tasks,
                                   taxi_ids, time_ranges):
    pos_index     = build_daily_position_index(daily_gps)
    tasks_by_slot = {}
    for task in daily_tasks:
        tasks_by_slot.setdefault(task['time_slot'], []).append(task)

    slot_reachability = {}
    for slot_idx, slot_tasks in tasks_by_slot.items():
        if slot_idx >= len(time_ranges) - 1:
            continue
        slot_start  = time_ranges[slot_idx]
        worker_lats = []; worker_lons = []; valid_ids = []
        for tid in taxi_ids:
            loc = get_position_at(pos_index, tid, slot_start)
            if loc is not None:
                valid_ids.append(tid)
                worker_lats.append(loc[0])
                worker_lons.append(loc[1])

        task_ids = [t['task_id'] for t in slot_tasks]
        if not valid_ids:
            slot_reachability[slot_idx] = {
                'taxi_ids':       [],
                'task_ids':       task_ids,
                'matrix':         np.zeros((0, len(slot_tasks)), dtype=bool),
                'taxi_id_to_idx': {},
                'task_id_to_idx': {tid: i
                                   for i, tid in enumerate(task_ids)},
            }
            continue
        w_arr  = np.array(list(zip(worker_lats, worker_lons)))
        t_arr  = np.array([[t['centroid_lat'], t['centroid_lon']]
                           for t in slot_tasks])
        matrix = build_reachability_matrix(w_arr, t_arr)
        slot_reachability[slot_idx] = {
            'taxi_ids':       valid_ids,
            'task_ids':       task_ids,
            'matrix':         matrix,
            'taxi_id_to_idx': {tid: i for i, tid in enumerate(valid_ids)},
            'task_id_to_idx': {tid: i for i, tid in enumerate(task_ids)},
        }
    return slot_reachability

def check_reachable(slot_reachability, taxi_id, task_id, slot_idx):
    sr = slot_reachability.get(slot_idx)
    if sr is None:
        return True
    w_idx = sr['taxi_id_to_idx'].get(taxi_id)
    t_idx = sr['task_id_to_idx'].get(task_id)
    if w_idx is None or t_idx is None:
        return True
    return bool(sr['matrix'][w_idx, t_idx])

class FuzzyTimeSeries:
    def __init__(self, n=7):
        self.n_intervals = n
        self.q_min = self.q_max = self.interval_length = None
        self.centroids = self.rm = None

    def _fuz(self, v):
        if v <= self.q_min: return 0
        if v >= self.q_max: return self.n_intervals - 1
        return int(np.argmin([abs(v - c) for c in self.centroids]))

    def fit(self, h):
        if len(h) < 2:
            self.rm  = None
            self._fb = float(np.mean(h)) if h else 0.0
            return self
        h = np.array(h, dtype=float)
        self.q_min           = max(0.0, float(np.floor(h.min())) - 1)
        self.q_max           = float(np.ceil(h.max())) + 1
        self.interval_length = (self.q_max - self.q_min) / self.n_intervals
        self.centroids = [
            self.q_min + (i + 0.5) * self.interval_length
            for i in range(self.n_intervals)]
        fs = [self._fuz(v) for v in h]
        rm = np.zeros((self.n_intervals, self.n_intervals), dtype=float)
        for t in range(len(fs) - 1):
            rm[fs[t], fs[t + 1]] += 1.0
        self.rm  = rm
        self._lf = fs[-1]
        return self

    def predict(self):
        if self.rm is None:
            return getattr(self, '_fb', 0.0)
        r  = self.rm[self._lf]
        sm = r.sum()
        return (self.centroids[self._lf] if sm == 0
                else float(np.dot(r / sm, self.centroids)))

    def fit_predict(self, h):
        self.fit(h)
        return self.predict()

def build_ftsa_supply_prediction(dd, tr, lmin, lmax,
                                  omin, omax, ls, os_):
    lb = np.arange(lmin, lmax, ls)
    ob = np.arange(omin, omax, os_)
    ns = len(tr) - 1
    ac = {}
    for si, (s, e) in enumerate(zip(tr[:-1], tr[1:])):
        sd = dd[(dd['time'] >= s) & (dd['time'] < e)]
        for i, l0 in enumerate(lb):
            for j, o0 in enumerate(ob):
                c = sd[
                    sd['latitude'].between(
                        l0, l0 + ls, inclusive='left') &
                    sd['longitude'].between(
                        o0, o0 + os_, inclusive='left')]
                ac[(i, j, si)] = c['taxi_id'].nunique()
    ps = {}
    f  = FuzzyTimeSeries(n=7)
    for i, _ in enumerate(lb):
        for j, _ in enumerate(ob):
            h = []
            for si in range(ns):
                sv = ac.get((i, j, si), 0)
                ps[(i, j, si)] = (f.fit_predict(h)
                                  if len(h) >= 2 else float(sv))
                h.append(sv)
    return ps, ac

def calculate_dynamic_task_config(tasks, ps, nt, el=0.1, eb=0.5):
    tc = {}
    ms = max(nt, 1)
    for t in tasks:
        tid    = t['task_id']
        gi, gj = t.get('grid_idx', (0, 0))
        si     = t.get('time_slot', 0)
        pn     = max(0.0, ps.get((gi, gj, si), nt / 2))
        rm     = max(1.0, float(t.get('passenger_demand', 1)))
        ei     = ((eb - el) * (pn / rm) + el if pn <= rm
                  else (1.0 - eb) * (pn - rm) / pn + eb)
        ei     = float(np.clip(ei, el, 1.0))
        sub    = int(np.clip(pn / ms * 7, 0, 6)) + 1
        bp     = t.get('priority_score', 0.5)
        ap     = (bp * (1.0 + (4 - sub) * 0.15) if sub <= 4
                  else bp * (1.0 - (sub - 4) * 0.05))
        tc[tid] = {
            'eps_i':             float(ei),
            'adjusted_priority': float(max(0.01, ap)),
        }
    return tc

def generate_grid_tasks(time_ranges, gps_data,
                         lat_range, lon_range,
                         lat_step, lon_step):
    tasks = []; tid = 0
    lat_bins = np.arange(lat_range[0], lat_range[1], lat_step)
    lon_bins = np.arange(lon_range[0], lon_range[1], lon_step)
    for slot_idx, (start, end) in enumerate(
            zip(time_ranges[:-1], time_ranges[1:])):
        slot_data = gps_data[
            (gps_data['time'] >= start) & (gps_data['time'] < end)]
        if slot_data.empty:
            continue
        for i, lat0 in enumerate(lat_bins):
            for j, lon0 in enumerate(lon_bins):
                demand = slot_data[
                    slot_data['latitude'].between(
                        lat0, lat0 + lat_step, inclusive='left') &
                    slot_data['longitude'].between(
                        lon0, lon0 + lon_step, inclusive='left')]
                p_demand = demand['taxi_id'].nunique()
                if p_demand > 0:
                    tasks.append({
                        "task_id":          tid,
                        "start_time":       start,
                        "time_slot":        slot_idx,
                        "lat_range":        (lat0, lat0 + lat_step),
                        "lon_range":        (lon0, lon0 + lon_step),
                        "grid_idx":         (i, j),
                        "centroid_lat":     lat0 + lat_step / 2,
                        "centroid_lon":     lon0 + lon_step / 2,
                        "passenger_demand": p_demand,
                    })
                    tid += 1
    return tasks

def normalize_task_priorities(tasks):
    if not tasks:
        return tasks
    sc     = [t.get('priority_score', 0.0) for t in tasks]
    pm, px = min(sc), max(sc)
    eps    = 1e-6
    for t in tasks:
        r = t.get('priority_score', 0.0)
        t['priority_score_raw'] = r
        t['priority_score']     = (r - pm) / (px - pm + eps)
    return tasks

def calculate_spatial_accuracy(taxi_id, daily_data, grid_map,
                                lat_step, lon_step,
                                lat_min_, lat_max_,
                                lon_min_, lon_max_):
    taxi_data = daily_data[daily_data['taxi_id'] == taxi_id]
    if taxi_data.empty:
        return 0.5
    total_dist_sq = 0.0
    count         = 0
    for _, row in taxi_data.iterrows():
        la, lo = row['latitude'], row['longitude']
        if not (lat_min_ <= la < lat_max_ and lon_min_ <= lo < lon_max_):
            continue
        gi = int((la - lat_min_) / lat_step)
        gj = int((lo - lon_min_) / lon_step)
        if (gi, gj) in grid_map:
            task = grid_map[(gi, gj)]
            cl   = task['centroid_lat']
            co   = task['centroid_lon']
            total_dist_sq += (la - cl) ** 2 + (lo - co) ** 2
            count += 1
    if count == 0:
        return 0.5
    return 1 / (1 + (total_dist_sq / count) * 10000)

class TaskPreferenceSorter:
    def __init__(self, weights=None):
        self.weights = weights or {
            'passenger_demand': 0.5,
            'taxi_supply':      0.25,
            'time_factor':      0.25,
        }

    def analyze_passenger_demand_enhanced(self, gps_data, time_ranges,
                                           lat_range, lon_range,
                                           is_bj=False):
        demand = {}
        for i in range(len(time_ranges) - 1):
            s, e = time_ranges[i], time_ranges[i + 1]
            data = gps_data[
                (gps_data['time'] >= s) & (gps_data['time'] <= e) &
                gps_data['latitude'].between(lat_range[0], lat_range[1]) &
                gps_data['longitude'].between(lon_range[0], lon_range[1])]
            if is_bj:
                n        = data['taxi_id'].nunique()
                p_demand = t_supply = n
            else:
                status = {
                    tid: {'p': any(g['status'] == 1),
                          'e': any(g['status'] == 0)}
                    for tid, g in data.groupby('taxi_id')}
                p_demand = sum(1 for v in status.values() if v['p'])
                t_supply = sum(1 for v in status.values() if v['e'])
            demand[i] = {
                'start_time':        s,
                'passenger_demand':  p_demand,
                'taxi_supply':       t_supply,
                'normalized_demand': min(1.0, p_demand / 50),
                'normalized_supply': min(1.0, t_supply / 30),
            }
        return demand

    def calculate_time_factor(self, start_time):
        if start_time and start_time.hour in [7, 8, 9, 17, 18, 19]:
            return 1.0
        if start_time and start_time.hour in [11, 12, 13]:
            return 0.8
        return 0.6

    def sort_taxis(self, taxi_data, weights=None):
        weights   = weights or {'reputation': 0.5, 'completion_rate': 0.5}
        taxi_data = taxi_data.copy()
        if 'composite_score' not in taxi_data.columns:
            taxi_data['composite_score'] = (
                taxi_data.get('daily_reputation', 0.5)
                * weights['reputation']
                + taxi_data.get('wtcs', 0.5)
                * weights['completion_rate'])
        return taxi_data.sort_values('composite_score', ascending=False)

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
        hds = defaultdict(float)
        hdc = defaultdict(int)
        for d in use_dates:
            for h, demand in self._history[d].get(
                    'hour_demand', {}).items():
                hds[h] += demand
                hdc[h] += 1
        hdm = {h: hds[h] / hdc[h] for h in hds if hdc[h] > 0}
        if hdm:
            md  = max(hdm.values()) + 1e-8
            ntf = {h: float(np.clip(v / md, 0.3, 1.0))
                   for h, v in hdm.items()}
            for h in range(24):
                old = self.time_factors.get(
                    h, self.DEFAULT_TIME_FACTORS[h])
                new = ntf.get(h, old)
                self.time_factors[h] = (
                    WEIGHT_SMOOTH_ALPHA * new
                    + (1 - WEIGHT_SMOOTH_ALPHA) * old)
        ds = []; ss = []
        for d in use_dates:
            ds.extend(self._history[d].get('slot_demand', []))
            ss.extend(self._history[d].get('slot_supply', []))
        if len(ds) >= 2 and len(ss) >= 2:
            vd = float(np.var(ds))
            vs = float(np.var(ss))
            vt = float(np.var(list(self.time_factors.values())))
            tt = vd + vs + vt + 1e-8
            nd = float(np.clip(vd / tt, 0.2, 0.7))
            ns = float(np.clip(vs / tt, 0.1, 0.5))
            nt = float(np.clip(vt / tt, 0.1, 0.4))
            nm = nd + ns + nt
            nd /= nm; ns /= nm; nt /= nm
            a = WEIGHT_SMOOTH_ALPHA
            self.weights['passenger_demand'] = (
                a * nd + (1 - a) * self.weights['passenger_demand'])
            self.weights['taxi_supply'] = (
                a * ns + (1 - a) * self.weights['taxi_supply'])
            self.weights['time_factor'] = (
                a * nt + (1 - a) * self.weights['time_factor'])

    def record_day(self, cur_date, gps_data, tr):
        slot_demand = []; slot_supply = []; hour_demand = {}
        for s, e in zip(tr[:-1], tr[1:]):
            data = gps_data[
                (gps_data['time'] >= s) & (gps_data['time'] < e)]
            if data.empty:
                continue
            h = s.hour
            if self.is_beijing:
                n   = data['taxi_id'].nunique()
                p   = min(1.0, n / 50)
                sup = min(1.0, n / 30)
            else:
                st  = {tid: (g['status'] == 1).any()
                       for tid, g in data.groupby('taxi_id')}
                p   = min(1.0, sum(st.values()) / 50)
                sup = min(1.0,
                          (len(st) - sum(st.values())) / 30)
            slot_demand.append(p)
            slot_supply.append(sup)
            hour_demand[h] = hour_demand.get(h, 0.0) + p
        self._history[cur_date] = {
            'slot_demand': slot_demand,
            'slot_supply': slot_supply,
            'hour_demand': hour_demand,
        }

    def get_weights(self):
        return dict(self.weights)

    def get_time_factor(self, h):
        return self.time_factors.get(h, 0.6)

    def is_adaptive(self):
        return self._adaptive_enabled

class TaskAllocator:
    def __init__(self, scenario_name, is_bj=False, mtp=6):
        self.scenario_name      = scenario_name
        self.is_beijing         = is_bj
        self.max_tasks_per_taxi = mtp

    @staticmethod
    def _check_reachable(sr, tid, tkid, si):
        return check_reachable(sr, tid, tkid, si)

    def _calculate_base_fitness(self, taxi, task, w_task, apply_rep):
        ts = taxi.get('composite_score', 0.5)
        ps = task.get('priority_score', 0.5)
        bs = (1.0 - w_task) * ts + w_task * ps
        rs = ((bs + taxi.get('daily_reputation', 0.5) * 0.3) ** 1.2
              if apply_rep else bs ** 1.2)
        x0 = 1.0 if self.is_beijing else 1.5

        return float(min(1.0, 1.1 / (1 + math.exp(-2.0 * (rs - x0)))))

    def _get_fitness(self, taxi, task, w_task):
        ar = (self.scenario_name == "BDTA")
        bf = self._calculate_base_fitness(taxi, task, w_task, ar)
        if self.scenario_name == "DTAA":
            c = 1.0 - taxi.get('composite_score', 0.5)
            return max(0.0, 0.8 * bf - 0.2 * c)
        return bf

    def allocate_all_tasks(self, taxis, tasks, **kw):
        if self.scenario_name == 'PX':
            return self._paper_x_allocation(taxis, tasks, **kw)
        elif self.scenario_name == 'FTSA':
            return self._ftsa_allocation(taxis, tasks, **kw)
        elif self.scenario_name == 'QITA':
            return self._qita_allocation(taxis, tasks, **kw)
        elif self.scenario_name == 'BDTA':
            return self._our_two_stage(taxis, tasks, **kw)
        else:
            return self._hungarian_allocation(taxis, tasks, **kw)

    def _our_two_stage(self, taxis, tasks, **kw):
        if not tasks:
            return []
        sr = kw.get('slot_reachability', {})
        wt = kw.get('w_task', FIXED_W_TASK)
        hp = np.percentile([t['priority_score'] for t in tasks], 80)
        ht = [t for t in tasks if t['priority_score'] >= hp]
        nt = [t for t in tasks if t['priority_score'] <  hp]
        aa = []
        if ht:
            ah = self._run_stage(taxis, ht, 1.0, wt, None, sr)
            if ah:
                aa.extend(ah)
                ac = pd.Series(
                    [a['taxi_id'] for a in ah]
                ).value_counts().to_dict()
                at = {a['task_id'] for a in ah}
                nt = [t for t in nt if t['task_id'] not in at]
                rt = taxis.copy()
                rt['remaining_capacity'] = self.max_tasks_per_taxi
                for t, c in ac.items():
                    rt.loc[rt['taxi_id'] == t,
                           'remaining_capacity'] -= c
                rt = rt[rt['remaining_capacity'] > 0]
            else:
                rt = taxis.copy()
                rt['remaining_capacity'] = self.max_tasks_per_taxi
        else:
            rt = taxis.copy()
            rt['remaining_capacity'] = self.max_tasks_per_taxi
        if nt and not rt.empty:
            cm_ = rt.set_index('taxi_id')[
                'remaining_capacity'].to_dict()
            an  = self._run_stage(
                rt, nt, BASELINE_TOPK_RATIO, wt, cm_, sr)
            if an:
                aa.extend(an)
        return aa

    def _run_stage(self, taxis, tasks, ratio, w_task,
                   capacity_map, slot_reachability):
        ns = max(1, int(len(taxis) * ratio))
        pt = taxis.head(ns)
        if pt.empty or not tasks:
            return []
        el = []
        for t in pt.to_dict('records'):
            cap = int(capacity_map.get(
                t['taxi_id'], self.max_tasks_per_taxi)
                if capacity_map else self.max_tasks_per_taxi)
            el.extend([t] * cap)
        if not el:
            return []
        INF = 1e9
        cm  = np.full((len(el), len(tasks)), INF)
        for i, tx in enumerate(el):
            for j, tk in enumerate(tasks):
                if self._check_reachable(
                        slot_reachability, tx['taxi_id'],
                        tk['task_id'], tk.get('time_slot', -1)):
                    cm[i, j] = -self._get_fitness(tx, tk, w_task)
        rs, cs = linear_sum_assignment(cm)
        return [
            {'taxi_id': el[r]['taxi_id'],
             'task_id': tasks[c]['task_id']}
            for r, c in zip(rs, cs) if cm[r, c] < INF
        ]

    def _hungarian_allocation(self, taxis, tasks, **kw):
        sr = kw.get('slot_reachability', {})
        wt = kw.get('w_task', FIXED_W_TASK)
        ns = max(1, int(len(taxis) * BASELINE_TOPK_RATIO))
        pt = taxis.head(ns)
        if pt.empty or not tasks:
            return []
        el = [t for t in pt.to_dict('records')
              for _ in range(self.max_tasks_per_taxi)]
        if not el:
            return []
        INF = 1e9
        cm  = np.full((len(el), len(tasks)), INF)
        for i, tx in enumerate(el):
            for j, tk in enumerate(tasks):
                if self._check_reachable(
                        sr, tx['taxi_id'],
                        tk['task_id'], tk.get('time_slot', -1)):
                    cm[i, j] = -self._get_fitness(tx, tk, wt)
        rs, cs = linear_sum_assignment(cm)
        return [
            {'taxi_id': el[r]['taxi_id'],
             'task_id': tasks[c]['task_id']}
            for r, c in zip(rs, cs) if cm[r, c] < INF
        ]

    def _paper_x_allocation(self, taxis, tasks, **kw):
        sr = kw.get('slot_reachability', {})
        wt = kw.get('w_task', FIXED_W_TASK)
        ns = max(1, int(len(taxis) * BASELINE_TOPK_RATIO))
        pt = taxis.head(ns)
        tl = pt.to_dict('records')
        tp = {t['taxi_id']: 0 for t in tl}
        ia = []; ml = 0
        for tk in sorted(tasks,
                         key=lambda t: t.get('priority_score', 0),
                         reverse=True):
            si = tk.get('time_slot', -1)
            cd = [t for t in tl
                  if (tp[t['taxi_id']] < self.max_tasks_per_taxi
                      and self._check_reachable(
                          sr, t['taxi_id'],
                          tk['task_id'], si))]
            if not cd:
                continue
            sc = []
            for tx in cd:
                qs  = self._get_fitness(tx, tk, wt)
                cs_ = 1.0 - tx.get('composite_score', 0.5)
                lf  = tp[tx['taxi_id']] / (ml + 1)
                fs  = 1.0 - lf
                w   = 1 / 3
                sc.append({'taxi':  tx,
                           'score': w * qs - (1 - w) * cs_ + w * fs,
                           'fitness': qs})
            wn = max(sc, key=lambda x: x['score'])
            ia.append({
                'taxi_id': wn['taxi']['taxi_id'],
                'task_id': tk['task_id'],
            })
            tp[wn['taxi']['taxi_id']] += 1
            if tp[wn['taxi']['taxi_id']] > ml:
                ml = tp[wn['taxi']['taxi_id']]
        return ia

    def _qita_allocation(self, taxis, tasks, **kw):
        sr        = kw.get('slot_reachability', {})
        wt        = kw.get('w_task', FIXED_W_TASK)
        qualified = taxis[
            taxis['composite_score'] >= QITA_QUALITY_THETA].copy()
        if qualified.empty:
            qualified = taxis.copy()
        ns = max(1, int(len(qualified) * BASELINE_TOPK_RATIO))
        pt = qualified.head(ns)
        if pt.empty or not tasks:
            return []
        worker_list = pt.to_dict('records')
        R   = QITA_REDUNDANCY_R
        MTP = self.max_tasks_per_taxi
        n_w = len(worker_list) * MTP
        n_t = len(tasks) * R
        INF = 1e9
        cm  = np.full((n_w, n_t), INF)
        for wi, tx in enumerate(worker_list):
            for ti, tk in enumerate(tasks):
                if not self._check_reachable(
                        sr, tx['taxi_id'],
                        tk['task_id'], tk.get('time_slot', -1)):
                    continue
                fitness = self._get_fitness(tx, tk, wt)
                for ri in range(MTP):
                    for rj in range(R):
                        cm[wi * MTP + ri, ti * R + rj] = -fitness
        row_idx, col_idx = linear_sum_assignment(cm)
        seen   = set()
        result = []
        for r, c in zip(row_idx, col_idx):
            if cm[r, c] >= INF:
                continue
            wi_orig = r // MTP
            ti_orig = c // R
            tx  = worker_list[wi_orig]
            tk  = tasks[ti_orig]
            key = (tx['taxi_id'], tk['task_id'])
            if key in seen:
                continue
            seen.add(key)
            result.append({
                'taxi_id': tx['taxi_id'],
                'task_id': tk['task_id'],
            })
        return result

    def _ftsa_allocation(self, taxis, tasks, **kw):
        if not tasks or taxis.empty:
            return []
        NP  = GA_N_POP; MI = GA_MAX_ITER
        PM, PX_ = 0.1, 0.9; MR = 0.15
        wt  = kw.get('w_task', FIXED_W_TASK)
        sr  = kw.get('slot_reachability', {})
        tc  = kw.get('task_configs', {})
        tbi = {t['task_id']: t for t in tasks}
        ns  = max(1, int(len(taxis) * BASELINE_TOPK_RATIO))
        tl  = taxis.head(ns).to_dict('records')
        tcs = {}
        for tx in tl:
            tid = tx['taxi_id']
            ts_ = tx.get('composite_score', 0.5)
            tcs[tid] = [
                t['task_id'] for t in tasks
                if (ts_ >= tc.get(t['task_id'], {}).get('eps_i', 0.0)
                    and self._check_reachable(
                        sr, tid,
                        t['task_id'], t.get('time_slot', -1)))
            ]

        def cu(tx, tl_):
            ts_ = tx.get('composite_score', 0.5)
            return sum(
                ts_ * ts_ * tc.get(t, {}).get(
                    'adjusted_priority',
                    tbi.get(t, {}).get('priority_score', 0.5))
                for t in tl_ if t in tbi)

        def cc(tx, tl_):
            if not tl_: return 0.0
            return (len(tl_)
                    * (1.0 - tx.get('composite_score', 0.5))
                    * (1.0 - tx.get('daily_reputation', 0.5)))

        def fl(tx, tl_):
            return (cu(tx, tl_), 1.0 / (cc(tx, tl_) + 1e-9))

        aa = []; ati = set()
        for tx in tl:
            tid = tx['taxi_id']
            cd  = [t for t in tcs.get(tid, []) if t not in ati]
            if not cd:
                continue
            mc  = self.max_tasks_per_taxi
            pop = [
                random.sample(cd, random.randint(1, min(mc, len(cd))))
                for _ in range(NP)
            ]
            for _ in range(MI):
                pop.sort(key=lambda c: fl(tx, c), reverse=True)
                n  = len(pop)
                pr = list(reversed(
                    [PM + (PX_ - PM) * i / max(n - 1, 1)
                     for i in range(n)]))
                tt = sum(pr)
                pr = [p / tt for p in pr]
                si_ = np.random.choice(
                    n, size=min(n, max(2, n // 2)),
                    replace=False, p=pr)
                sv  = [pop[i] for i in si_]
                np_ = list(sv)
                while len(np_) < NP and len(sv) >= 2:
                    p1, p2 = random.sample(sv, 2)
                    u1 = list(set(p1) - set(p2))
                    u2 = list(set(p2) - set(p1))
                    if u1 and u2:
                        g1 = random.choice(u1)
                        g2 = random.choice(u2)
                        np_.extend([
                            [g2 if x == g1 else x for x in p1][:mc],
                            [g1 if x == g2 else x for x in p2][:mc],
                        ])
                    elif len(p1) > 1:
                        ch     = p1.copy()
                        i1, i2 = sorted(
                            random.sample(range(len(ch)), 2))
                        ch[i1:i2 + 1] = ch[i1:i2 + 1][::-1]
                        np_.append(ch)
                    else:
                        np_.append(p1.copy())
                for idx in range(len(np_)):
                    if random.random() < MR:
                        ch = np_[idx].copy()
                        am = [t for t in cd
                              if t not in ati and t not in ch]
                        if len(ch) < mc and am:
                            ch.append(random.choice(am))
                        elif len(ch) > 1:
                            ch.pop(random.randint(0, len(ch) - 1))
                        np_[idx] = ch
                pop = np_[:NP]
            if not pop:
                continue
            pop.sort(key=lambda c: fl(tx, c), reverse=True)
            bc = pop[0]
            for tkid in bc:
                if tkid in ati:
                    continue
                tk = tbi.get(tkid)
                if tk is None:
                    continue
                aa.append({
                    'taxi_id': tid,
                    'task_id': tkid,
                })
                ati.add(tkid)
        return aa

def run_scalability_experiment(experiment_type, scales,
                               all_taxis,
                               all_tasks_base,
                               all_tasks_bdta,
                               time_ranges,
                               daily_gps_data,
                               slot_reachability):
    results = []

    bdta_by_id = {t['task_id']: t for t in all_tasks_bdta}
    base_by_id = {t['task_id']: t for t in all_tasks_base}

    for scale in tqdm(scales, desc=f"{experiment_type}"):
        if experiment_type == "Workers":
            num_workers = scale
            num_tasks   = min(FIXED_TASK_COUNT_W, len(all_tasks_base))
        else:
            num_workers = min(FIXED_WORKER_COUNT_T, len(all_taxis))
            num_tasks   = scale

        current_taxis = (
            all_taxis.sample(n=num_workers, random_state=RANDOM_SEED)
            if len(all_taxis) >= num_workers else all_taxis)

        task_rng = random.Random(RANDOM_SEED + scale)
        if len(all_tasks_base) >= num_tasks:
            sampled_ids = [t['task_id']
                           for t in task_rng.sample(all_tasks_base, num_tasks)]
        else:
            sampled_ids = [t['task_id'] for t in all_tasks_base]

        current_tasks_base = [base_by_id[i]
                              for i in sampled_ids if i in base_by_id]
        current_tasks_bdta = [bdta_by_id[i]
                              for i in sampled_ids if i in bdta_by_id]

        for sn in SCENARIOS_TO_RUN:
            if sn == 'BDTA':
                current_tasks = current_tasks_bdta
            else:
                current_tasks = current_tasks_base

            allocator = TaskAllocator(
                sn,
                is_bj=IS_BEIJING_DATASET,
                mtp=FIXED_CAPACITY)

            kw = {
                'w_task':            FIXED_W_TASK,
                'slot_reachability': slot_reachability,
            }

            if sn == 'FTSA':
                psu, _ = build_ftsa_supply_prediction(
                    daily_gps_data, time_ranges,
                    lat_min, lat_max,
                    lon_min, lon_max,
                    lat_step_val, lon_step_val)
                kw['task_configs'] = calculate_dynamic_task_config(
                    current_tasks, psu, len(current_taxis))

            if sn == 'QITA':
                gm = {tk['grid_idx']: tk for tk in current_tasks}
                ss = current_taxis['taxi_id'].apply(
                    lambda t: calculate_spatial_accuracy(
                        t, daily_gps_data, gm,
                        lat_step_val, lon_step_val,
                        lat_min, lat_max,
                        lon_min, lon_max))
                ct = current_taxis.copy()
                ct['composite_score'] = np.clip(ss.values, 0, 1)
                ct = ct.sort_values('composite_score', ascending=False)
            else:
                ct = current_taxis

            allocator.allocate_all_tasks(ct.copy(), current_tasks, **kw)

            total_ms = 0.0
            for _ in range(NUM_REPETITIONS):
                t0 = time.perf_counter()
                allocator.allocate_all_tasks(
                    ct.copy(), current_tasks, **kw)
                total_ms += (time.perf_counter() - t0) * 1000

            avg_ms = total_ms / NUM_REPETITIONS
            results.append({
                'experiment_type':      experiment_type,
                'num_workers':          num_workers,
                'num_tasks':            num_tasks,
                'scenario':             sn,
                'scenario_label':       SCENARIO_DISPLAY.get(sn, sn),
                'avg_decision_time_ms': avg_ms,
            })
            print()

    return results

if __name__ == '__main__':
    data_file = (PREPROCESSED_DATA_FILE if IS_BEIJING_DATASET
                 else os.path.join(data_dir, PREPROCESSED_DATA_FILE))
    try:
        with open(data_file, 'rb') as f:
            preprocessed_data = pickle.load(f)
        gps_data1 = preprocessed_data['gps_data1']
        result_df = preprocessed_data['result_df']
        gps_data1['time'] = pd.to_datetime(gps_data1['time'])
        result_df['date'] = pd.to_datetime(result_df['date']).dt.date
    except FileNotFoundError:
        print()
        exit()

    date_range = pd.to_datetime(result_df['date'].unique())
    filtered   = [d for d in date_range
                  if d.date() not in excluded_dates]
    if not filtered:
        print()
        exit()

    test_date = filtered[0]
    print()

    daily_data  = gps_data1[
        gps_data1['time'].dt.date == test_date.date()].copy()
    start_dt    = pd.to_datetime(
        f'{test_date.date()} {DAILY_START_TIME}')
    end_dt      = pd.to_datetime(
        f'{test_date.date()} {DAILY_END_TIME}')
    time_ranges = pd.date_range(
        start=start_dt, end=end_dt, freq='15min')

    sorter        = TaskPreferenceSorter()
    all_tasks_raw = generate_grid_tasks(
        time_ranges, daily_data,
        (lat_min, lat_max), (lon_min, lon_max),
        lat_step_val, lon_step_val)

    demand_info = sorter.analyze_passenger_demand_enhanced(
        daily_data, time_ranges,
        (lat_min, lat_max), (lon_min, lon_max),
        is_bj=IS_BEIJING_DATASET)

    for task in all_tasks_raw:
        si = demand_info.get(task.get('time_slot'))
        if si:
            prio = (
                si['normalized_demand']
                * sorter.weights['passenger_demand']
                + (1 - si['normalized_supply'])
                * sorter.weights['taxi_supply']
                + sorter.calculate_time_factor(si['start_time'])
                * sorter.weights['time_factor'])
            task['priority_score'] = prio * (
                1 + min(task.get('passenger_demand', 0) / 10, 1))
        else:
            task['priority_score'] = (
                task.get('passenger_demand', 0) / 10.0)

    all_tasks_base = normalize_task_priorities(
        copy.deepcopy(all_tasks_raw))
    all_tasks_base = sorted(
        all_tasks_base,
        key=lambda x: x.get('priority_score', 0),
        reverse=True)

    all_tasks_bdta = copy.deepcopy(all_tasks_raw)
    spt = {
        idx: max(1, daily_data[
            (daily_data['time'] >= s) & (daily_data['time'] < e)
        ]['taxi_id'].nunique())
        for idx, (s, e) in enumerate(
            zip(time_ranges[:-1], time_ranges[1:]))
    }
    for tk in all_tasks_bdta:
        ps_ = spt.get(tk['time_slot'], 1)
        tk['priority_score'] *= min(
            4.0, 1.0 + tk['passenger_demand'] / ps_)
    all_tasks_bdta = normalize_task_priorities(all_tasks_bdta)
    all_tasks_bdta = sorted(
        all_tasks_bdta,
        key=lambda x: x.get('priority_score', 0),
        reverse=True)

    all_taxis = result_df[
        result_df['date'] == test_date.date()].copy()

    all_taxis['composite_score'] = (
        all_taxis['daily_reputation'] * (1.0 - FIXED_W_TASK)
        + all_taxis['wtcs'] * FIXED_W_TASK)
    all_taxis = all_taxis.sort_values(
        'composite_score', ascending=False)

    print()

    print()
    taxi_ids_all      = all_taxis['taxi_id'].tolist()
    slot_reachability = precompute_daily_reachability(
        daily_data, all_tasks_base, taxi_ids_all, time_ranges)
    print()

    worker_results = run_scalability_experiment(
        "Workers",
        WORKER_SCALES,
        all_taxis,
        all_tasks_base,
        all_tasks_bdta,
        time_ranges,
        daily_data,
        slot_reachability=slot_reachability)

    task_results = run_scalability_experiment(
        "Tasks",
        TASK_SCALES,
        all_taxis,
        all_tasks_base,
        all_tasks_bdta,
        time_ranges,
        daily_data,
        slot_reachability=slot_reachability)

    final_df   = pd.DataFrame(worker_results + task_results)
    output_dir = ('beijing_scalability_results'
                  if IS_BEIJING_DATASET
                  else 'chengdu_scalability_results')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, RESULTS_FILENAME)
    final_df.to_csv(out_path, index=False)
    print()
    print()
    print()