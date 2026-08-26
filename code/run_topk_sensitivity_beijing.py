

import os, random, math, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

TOPK_LEVELS = [0.5,0.6, 0.7, 0.8, 0.9]

SLOT_MINUTES       = 15
AVG_SPEED_KMH      = 40.0
DEADLINE_MINUTES   = 15
FIXED_W_TASK       = 0.5
MAX_TASKS_PER_TAXI = 6
GA_N_POP           = 20
GA_MAX_ITER        = 50
QITA_QUALITY_THETA = 0.3
QITA_REDUNDANCY_R  = 1
BDTA_STAGE1_RATIO  = 1.0

ADAPTIVE_WINDOW_DAYS = 7
ADAPTIVE_MIN_DAYS    = 1
WEIGHT_SMOOTH_ALPHA  = 0.3

SCENARIOS_TO_RUN = ["BDTA", "PX", "QITA", "DTAA", "FTSA"]

RUN_BEIJING  = True
RUN_CHENGDU  = False

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = radians(lat1); phi2 = radians(lat2)
    dphi = radians(lat2 - lat1); dlam = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def build_daily_position_index(daily_gps):
    index = {}
    for tid, group in daily_gps.groupby('taxi_id'):
        index[tid] = (group[['time', 'latitude', 'longitude']]
                      .sort_values('time').reset_index(drop=True))
    return index

def get_position_at(pos_index, taxi_id, query_time):
    records = pos_index.get(taxi_id)
    if records is None or records.empty: return None
    before = records[records['time'] <= query_time]
    if before.empty: return None
    return (before.iloc[-1]['latitude'], before.iloc[-1]['longitude'])

def build_reachability_matrix(w_arr, t_arr):
    if len(w_arr) == 0 or len(t_arr) == 0:
        return np.zeros((len(w_arr), len(t_arr)), dtype=bool)
    wla = np.radians(w_arr[:, 0:1]); wlo = np.radians(w_arr[:, 1:2])
    tla = np.radians(t_arr[:, 0]);   tlo = np.radians(t_arr[:, 1])
    dla = tla - wla; dlo = tlo - wlo
    a = np.sin(dla/2)**2 + np.cos(wla)*np.cos(tla)*np.sin(dlo/2)**2
    d = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (d / AVG_SPEED_KMH) * 60.0 <= SLOT_MINUTES

def precompute_daily_reachability(daily_gps, daily_tasks, taxi_ids, time_ranges):
    pos_idx = build_daily_position_index(daily_gps)
    tbs = {}
    for t in daily_tasks:
        tbs.setdefault(t['time_slot'], []).append(t)
    sr = {}
    for si, st in tbs.items():
        if si >= len(time_ranges) - 1: continue
        ss = time_ranges[si]; wl = []; wo = []; vi = []
        for tid in taxi_ids:
            loc = get_position_at(pos_idx, tid, ss)
            if loc: vi.append(tid); wl.append(loc[0]); wo.append(loc[1])
        tids = [t['task_id'] for t in st]
        if not vi:
            sr[si] = {'taxi_ids': [], 'task_ids': tids,
                      'matrix': np.zeros((0, len(st)), dtype=bool),
                      'taxi_id_to_idx': {},
                      'task_id_to_idx': {t: i for i, t in enumerate(tids)}}
            continue
        wa = np.array(list(zip(wl, wo)))
        ta = np.array([[t['centroid_lat'], t['centroid_lon']] for t in st])
        m  = build_reachability_matrix(wa, ta)
        sr[si] = {'taxi_ids': vi, 'task_ids': tids, 'matrix': m,
                  'taxi_id_to_idx': {t: i for i, t in enumerate(vi)},
                  'task_id_to_idx': {t: i for i, t in enumerate(tids)}}
    return sr

def check_reachable(sr, tid, tkid, si):
    r = sr.get(si)
    if r is None: return True
    wi = r['taxi_id_to_idx'].get(tid); ti = r['task_id_to_idx'].get(tkid)
    if wi is None or ti is None: return True
    return bool(r['matrix'][wi, ti])

def generate_grid_tasks(tr, gd, lr, orng, ls, os_):
    tasks = []; tc = 0
    lb = np.arange(lr[0], lr[1], ls); ob = np.arange(orng[0], orng[1], os_)
    for si, (s, e) in enumerate(zip(tr[:-1], tr[1:])):
        sd = gd[(gd['time'] >= s) & (gd['time'] < e)]
        if sd.empty: continue
        for i, l0 in enumerate(lb):
            for j, o0 in enumerate(ob):
                df = sd[sd['latitude'].between(l0, l0 + ls, inclusive='left') &
                        sd['longitude'].between(o0, o0 + os_, inclusive='left')]
                p = df['taxi_id'].nunique()
                if p > 0:
                    cl = l0 + ls/2; co = o0 + os_/2
                    tasks.append({'task_id': tc, 'start_time': s, 'time_slot': si,
                                  'grid_idx': (i, j), 'centroid': (cl, co),
                                  'centroid_lat': cl, 'centroid_lon': co,
                                  'passenger_demand': p})
                    tc += 1
    return tasks

def normalize_task_priorities(tasks):
    if not tasks: return tasks
    sc = [t.get('priority_score', 0.0) for t in tasks]
    pm, px = min(sc), max(sc)
    for t in tasks:
        r = t.get('priority_score', 0.0)
        t['priority_score_raw'] = r
        t['priority_score'] = (r - pm) / (px - pm + 1e-6)
    return tasks

def compute_metrics(allocations, tasks, pos_index, all_taxi_ids):
    if not allocations: return {}
    df_alloc   = pd.DataFrame(allocations)
    task_by_id = {t['task_id']: t for t in tasks}
    total_demand    = max(sum(t.get('passenger_demand', 1) for t in tasks), 1)
    assigned_demand = sum(task_by_id[tid].get('passenger_demand', 1)
                          for tid in df_alloc['task_id'].unique()
                          if tid in task_by_id)
    dsr = assigned_demand / total_demand
    on_time_demand = 0.0; total_travel_km = 0.0
    for taxi_id, group in df_alloc.groupby('taxi_id'):
        seq = sorted([task_by_id[tid] for tid in group['task_id']
                      if tid in task_by_id],
                     key=lambda t: (t.get('start_time', pd.Timestamp.min),
                                    t.get('task_id', -1)))
        prev_loc = None; prev_arrival = None
        for task in seq:
            ts = task.get('start_time')
            if ts is None: continue
            tloc = (task['centroid_lat'], task['centroid_lon'])
            p_d  = task.get('passenger_demand', 1)
            if prev_loc is None:
                origin = get_position_at(pos_index, taxi_id, ts); depart = ts
            else:
                gap = ((ts - prev_arrival).total_seconds() / 60.0
                       if prev_arrival else SLOT_MINUTES)
                if gap >= SLOT_MINUTES:
                    origin = get_position_at(pos_index, taxi_id, ts) or prev_loc
                    depart = ts
                else:
                    origin = prev_loc; depart = max(prev_arrival, ts)
            if origin is None: continue
            dist_km  = haversine_km(origin[0], origin[1], tloc[0], tloc[1])
            travel_m = (dist_km / AVG_SPEED_KMH) * 60.0
            arrival  = depart + pd.Timedelta(minutes=travel_m)
            total_travel_km += dist_km
            if arrival <= ts + pd.Timedelta(minutes=DEADLINE_MINUTES):
                on_time_demand += p_d
            prev_loc = tloc; prev_arrival = arrival
    ot_dsr    = on_time_demand / total_demand
    km_per    = total_travel_km / assigned_demand if assigned_demand > 0 else 0.0
    d_per_km  = assigned_demand / max(total_travel_km, 1e-6) if assigned_demand > 0 else 0.0
    n_total   = len(set(all_taxi_ids)); n_active = int(df_alloc['taxi_id'].nunique())
    return {
        'avg_fitness_score':                round(float(df_alloc['fitness_score'].mean()), 4),
        'avg_allocation_quality':           round(float(df_alloc['allocation_quality'].mean()), 4),
        'demand_satisfaction_rate':         round(float(dsr), 4),
        'on_time_demand_satisfaction_rate': round(float(ot_dsr), 4),
        'travel_km_per_served_demand':      round(float(km_per), 4),
        'served_demand_per_km':             round(float(d_per_km), 4),
        'worker_participation_rate':        round(n_active / max(n_total, 1), 4),
        'n_allocations':                    float(len(allocations)),
    }

class AdaptiveWeightLearner:
    DEFAULT_WEIGHTS = {'passenger_demand': 0.5, 'taxi_supply': 0.25, 'time_factor': 0.25}
    DEFAULT_TIME_FACTORS = {h: (1.0 if h in [7, 8, 9, 17, 18, 19]
                                else 0.8 if h in [11, 12, 13] else 0.6)
                            for h in range(24)}

    def __init__(self, is_beijing=False):
        self.is_beijing    = is_beijing
        self.weights       = dict(self.DEFAULT_WEIGHTS)
        self.time_factors  = dict(self.DEFAULT_TIME_FACTORS)
        self._history      = {}
        self._adaptive_enabled = False

    def update(self, cur_date, gps_data, tr):
        window_dates = sorted(self._history.keys())
        if len(window_dates) < ADAPTIVE_MIN_DAYS: return
        self._adaptive_enabled = True
        use_dates = window_dates[-ADAPTIVE_WINDOW_DAYS:]
        hds = defaultdict(float); hdc = defaultdict(int)
        for d in use_dates:
            for h, demand in self._history[d].get('hour_demand', {}).items():
                hds[h] += demand; hdc[h] += 1
        hdm = {h: hds[h]/hdc[h] for h in hds if hdc[h] > 0}
        if hdm:
            md  = max(hdm.values()) + 1e-8
            ntf = {h: float(np.clip(v/md, 0.3, 1.0)) for h, v in hdm.items()}
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
            nd = float(np.clip(vd/tt, 0.2, 0.7)); ns = float(np.clip(vs/tt, 0.1, 0.5))
            nt = float(np.clip(vt/tt, 0.1, 0.4)); nm = nd + ns + nt
            nd /= nm; ns /= nm; nt /= nm
            a = WEIGHT_SMOOTH_ALPHA
            self.weights['passenger_demand'] = a*nd + (1-a)*self.weights['passenger_demand']
            self.weights['taxi_supply']      = a*ns + (1-a)*self.weights['taxi_supply']
            self.weights['time_factor']      = a*nt + (1-a)*self.weights['time_factor']

    def record_day(self, cur_date, gps_data, tr):
        slot_demand = []; slot_supply = []; hour_demand = {}
        for s, e in zip(tr[:-1], tr[1:]):
            data = gps_data[(gps_data['time'] >= s) & (gps_data['time'] < e)]
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
        self._history[cur_date] = {'slot_demand': slot_demand,
                                   'slot_supply': slot_supply,
                                   'hour_demand': hour_demand}

    def get_weights(self):       return dict(self.weights)
    def get_time_factor(self, h): return self.time_factors.get(h, 0.6)
    def is_adaptive(self):       return self._adaptive_enabled

def get_task_priorities(gt, dd, tr, is_bj: bool, learner=None):
    use_adaptive = (learner is not None and learner.is_adaptive())
    FIXED_PW     = {'passenger_demand': 0.5, 'taxi_supply': 0.25, 'time_factor': 0.25}

    demand_info = {}
    for i, (s, e) in enumerate(zip(tr[:-1], tr[1:])):
        data = dd[(dd['time'] >= s) & (dd['time'] < e)]

        if is_bj:
            n   = data['taxi_id'].nunique()
            pd_ = ts_ = n
        else:
            st_ = {tid: (g['status'] == 1).any()
                   for tid, g in data.groupby('taxi_id')}
            pd_ = sum(st_.values())
            ts_ = len(st_) - pd_
        demand_info[i] = {
            'start_time':        s,
            'normalized_demand': min(1.0, pd_/50),
            'normalized_supply': min(1.0, ts_/30),
        }

    weights = learner.get_weights() if use_adaptive else FIXED_PW
    for task in gt:
        si = demand_info.get(task['time_slot'])
        if si:
            h  = si['start_time'].hour
            if use_adaptive:
                tf = learner.get_time_factor(h)
            else:
                tf = (1.0 if h in [7, 8, 9, 17, 18, 19]
                      else 0.8 if h in [11, 12, 13] else 0.6)
            prio = (si['normalized_demand'] * weights['passenger_demand']
                    + (1 - si['normalized_supply']) * weights['taxi_supply']
                    + tf * weights['time_factor'])
            task['priority_score'] = prio * (1 + min(task['passenger_demand']/10, 1))
        else:
            task['priority_score'] = task['passenger_demand'] / 10.0
    return gt

def base_fitness(taxi, task, apply_rep=False, is_bj=False):
    ts = taxi.get('composite_score', 0.5)
    ps = task.get('priority_score', 0.5)
    bs = (1.0 - FIXED_W_TASK) * ts + FIXED_W_TASK * ps
    rs = (bs + taxi.get('daily_reputation', 0.5) * 0.3) ** 1.2         if apply_rep else bs ** 1.2
    x0 = 1.0 if is_bj else 1.5
    return 1 / (1 + math.exp(-2.0 * (rs - x0)))

def run_hungarian(taxis, tasks, slot_reachability, topk_ratio,
                  is_bj=False, capacity_map=None, apply_rep=False):
    ns = max(1, int(len(taxis) * topk_ratio))
    pt = taxis.head(ns)
    if pt.empty or not tasks: return []
    el = []
    for r in pt.to_dict('records'):
        cap = int(capacity_map.get(r['taxi_id'], MAX_TASKS_PER_TAXI)
                  if capacity_map else MAX_TASKS_PER_TAXI)
        el.extend([r] * max(1, cap))
    if not el: return []
    INF = 1e9
    cm  = np.full((len(el), len(tasks)), INF)
    for i, tx in enumerate(el):
        for j, tk in enumerate(tasks):
            if check_reachable(slot_reachability, tx['taxi_id'],
                               tk['task_id'], tk.get('time_slot', -1)):
                cm[i, j] = -base_fitness(tx, tk, apply_rep, is_bj)
    rs, cs = linear_sum_assignment(cm)
    al = []
    for r, c in zip(rs, cs):
        if cm[r, c] >= INF: continue
        tx = el[r]; tk = tasks[c]; f = -cm[r, c]
        pc = min(1.0, tx.get('wtcs', 0.5)*0.8 + tx.get('daily_reputation', 0.5)*0.2)
        al.append({'taxi_id': tx['taxi_id'], 'task_id': tk['task_id'],
                   'fitness_score': f, 'predicted_completion': pc,
                   'allocation_quality': f * tk.get('priority_score', 0.5) * pc})
    return al

def allocate_bdta(taxis, tasks, slot_reachability, topk_ratio, is_bj=False):
    if not tasks: return []
    hp = np.percentile([t['priority_score'] for t in tasks], 80)
    ht = [t for t in tasks if t['priority_score'] >= hp]
    nt = [t for t in tasks if t['priority_score'] <  hp]
    aa = []
    if ht:
        ah = run_hungarian(taxis, ht, slot_reachability,
                           BDTA_STAGE1_RATIO,
                           is_bj=is_bj, apply_rep=True)
        aa.extend(ah)
        ac = pd.Series([a['taxi_id'] for a in ah]).value_counts().to_dict()
        at = {a['task_id'] for a in ah}
        nt = [t for t in nt if t['task_id'] not in at]
        rt = taxis.copy()
        rt['remaining_capacity'] = MAX_TASKS_PER_TAXI
        for t, c in ac.items():
            rt.loc[rt['taxi_id'] == t, 'remaining_capacity'] -= c
        rt = rt[rt['remaining_capacity'] > 0]
    else:
        rt = taxis.copy(); rt['remaining_capacity'] = MAX_TASKS_PER_TAXI
    if nt and not rt.empty:
        cm_ = rt.set_index('taxi_id')['remaining_capacity'].to_dict()
        an  = run_hungarian(rt, nt, slot_reachability,
                            topk_ratio,
                            is_bj=is_bj, capacity_map=cm_, apply_rep=True)
        aa.extend(an)
    return aa

def allocate_px(taxis, tasks, slot_reachability, topk_ratio, is_bj=False):
    ns = max(1, int(len(taxis) * topk_ratio))
    pt = taxis.head(ns); tl = pt.to_dict('records')
    tp = {t['taxi_id']: 0 for t in tl}; ia = []; ml = 0
    for tk in sorted(tasks, key=lambda t: t.get('priority_score', 0), reverse=True):
        si = tk.get('time_slot', -1)
        cd = [t for t in tl
              if tp[t['taxi_id']] < MAX_TASKS_PER_TAXI
              and check_reachable(slot_reachability, t['taxi_id'], tk['task_id'], si)]
        if not cd: continue
        sc = []
        for tx in cd:
            qs  = base_fitness(tx, tk, False, is_bj)
            cs_ = 1.0 - tx.get('composite_score', 0.5)
            lf  = tp[tx['taxi_id']] / (ml + 1); fs = 1.0 - lf; w = 1/3
            sc.append({'taxi': tx, 'score': w*qs - (1-w)*cs_ + w*fs, 'fitness': qs})
        wn = max(sc, key=lambda x: x['score']); ch = wn['taxi']; ci = ch['taxi_id']
        pc = min(1.0, ch.get('wtcs', 0.5)*0.8 + ch.get('daily_reputation', 0.5)*0.2)
        ia.append({'taxi_id': ci, 'task_id': tk['task_id'],
                   'fitness_score': wn['fitness'], 'predicted_completion': pc,
                   'allocation_quality': wn['fitness']*tk.get('priority_score', 0.5)*pc})
        tp[ci] += 1
        if tp[ci] > ml: ml = tp[ci]
    return ia

def allocate_dtaa(taxis, tasks, slot_reachability, topk_ratio, is_bj=False):
    ns = max(1, int(len(taxis) * topk_ratio))
    pt = taxis.head(ns)
    if pt.empty or not tasks: return []
    el = [t for t in pt.to_dict('records') for _ in range(MAX_TASKS_PER_TAXI)]
    INF = 1e9; cm = np.full((len(el), len(tasks)), INF)
    for i, tx in enumerate(el):
        for j, tk in enumerate(tasks):
            if check_reachable(slot_reachability, tx['taxi_id'],
                               tk['task_id'], tk.get('time_slot', -1)):
                bf  = base_fitness(tx, tk, False, is_bj)
                pen = 1.0 - tx.get('composite_score', 0.5)
                cm[i, j] = -(max(0.0, 0.8*bf - 0.2*pen))
    rs, cs = linear_sum_assignment(cm)
    al = []
    for r, c in zip(rs, cs):
        if cm[r, c] >= INF: continue
        tx = el[r]; tk = tasks[c]; f = -cm[r, c]
        pc = min(1.0, tx.get('wtcs', 0.5)*0.8 + tx.get('daily_reputation', 0.5)*0.2)
        al.append({'taxi_id': tx['taxi_id'], 'task_id': tk['task_id'],
                   'fitness_score': f, 'predicted_completion': pc,
                   'allocation_quality': f*tk.get('priority_score', 0.5)*pc})
    return al

def allocate_qita(taxis, tasks, slot_reachability, topk_ratio, is_bj=False):
    qualified = taxis[taxis['composite_score'] >= QITA_QUALITY_THETA].copy()
    if qualified.empty: qualified = taxis.copy()
    ns = max(1, int(len(qualified) * topk_ratio))
    pt = qualified.head(ns)
    if pt.empty or not tasks: return []
    worker_list = pt.to_dict('records')
    R = QITA_REDUNDANCY_R; MTP = MAX_TASKS_PER_TAXI
    INF = 1e9; cm = np.full((len(worker_list)*MTP, len(tasks)*R), INF)
    for wi, tx in enumerate(worker_list):
        for ti, tk in enumerate(tasks):
            if not check_reachable(slot_reachability, tx['taxi_id'],
                                   tk['task_id'], tk.get('time_slot', -1)): continue
            f = base_fitness(tx, tk, False, is_bj); cost = -f
            for ri in range(MTP):
                for rj in range(R):
                    cm[wi*MTP+ri, ti*R+rj] = cost
    row_idx, col_idx = linear_sum_assignment(cm)
    seen = set(); result = []
    for r, c in zip(row_idx, col_idx):
        if cm[r, c] >= INF: continue
        wi_orig = r // MTP; ti_orig = c // R
        tx = worker_list[wi_orig]; tk = tasks[ti_orig]
        key = (tx['taxi_id'], tk['task_id'])
        if key in seen: continue
        seen.add(key); f = -cm[r, c]
        pc = min(1.0, tx.get('wtcs', 0.5)*0.8 + tx.get('daily_reputation', 0.5)*0.2)
        result.append({'taxi_id': tx['taxi_id'], 'task_id': tk['task_id'],
                       'fitness_score': f, 'predicted_completion': pc,
                       'allocation_quality': f*tk.get('priority_score', 0.5)*pc})
    return result

class FuzzyTimeSeries:
    def __init__(self, n=7):
        self.n_intervals = n; self.q_min = self.q_max = None
        self.centroids = self.rm = None

    def _fuz(self, v):
        if v <= self.q_min: return 0
        if v >= self.q_max: return self.n_intervals - 1
        return int(np.argmin([abs(v - c) for c in self.centroids]))

    def fit(self, h):
        if len(h) < 2:
            self.rm = None; self._fb = float(np.mean(h)) if h else 0.0; return self
        h = np.array(h, dtype=float)
        self.q_min = max(0.0, float(np.floor(h.min())) - 1)
        self.q_max = float(np.ceil(h.max())) + 1
        il = (self.q_max - self.q_min) / self.n_intervals
        self.centroids = [self.q_min + (i + 0.5)*il for i in range(self.n_intervals)]
        fs = [self._fuz(v) for v in h]
        rm = np.zeros((self.n_intervals, self.n_intervals), dtype=float)
        for t in range(len(fs) - 1): rm[fs[t], fs[t+1]] += 1.0
        self.rm = rm; self._lf = fs[-1]; return self

    def predict(self):
        if self.rm is None: return getattr(self, '_fb', 0.0)
        r = self.rm[self._lf]; sm = r.sum()
        return self.centroids[self._lf] if sm == 0 else float(np.dot(r/sm, self.centroids))

    def fit_predict(self, h): self.fit(h); return self.predict()

def build_ftsa_supply_prediction(dd, tr, lmin, lmax, omin, omax, ls, os_):
    lb = np.arange(lmin, lmax, ls); ob = np.arange(omin, omax, os_); ns = len(tr) - 1
    ac = {}
    for si, (s, e) in enumerate(zip(tr[:-1], tr[1:])):
        sd = dd[(dd['time'] >= s) & (dd['time'] < e)]
        for i, l0 in enumerate(lb):
            for j, o0 in enumerate(ob):
                c = sd[sd['latitude'].between(l0, l0+ls, inclusive='left') &
                       sd['longitude'].between(o0, o0+os_, inclusive='left')]
                ac[(i, j, si)] = c['taxi_id'].nunique()
    ps = {}; f = FuzzyTimeSeries(n=7)
    for i, _ in enumerate(lb):
        for j, _ in enumerate(ob):
            h = []
            for si in range(ns):
                sv = ac.get((i, j, si), 0)
                ps[(i, j, si)] = f.fit_predict(h) if len(h) >= 2 else float(sv)
                h.append(sv)
    return ps, ac

def calculate_dynamic_task_config(tasks, ps, nt, el=0.1, eb=0.5):
    tc = {}; ms = max(nt, 1)
    for t in tasks:
        tid = t['task_id']; gi, gj = t.get('grid_idx', (0, 0)); si = t.get('time_slot', 0)
        pn  = max(0.0, ps.get((gi, gj, si), nt/2))
        rm  = max(1.0, float(t.get('passenger_demand', 1)))
        ei  = ((eb - el)*(pn/rm) + el if pn <= rm else (1.0 - eb)*(pn - rm)/pn + eb)
        ei  = float(np.clip(ei, el, 1.0))
        sub = int(np.clip(pn/ms*7, 0, 6)) + 1
        bp  = t.get('priority_score_raw', t.get('priority_score', 0.5))
        ap  = (bp*(1.0 + (4 - sub)*0.15) if sub <= 4 else bp*(1.0 - (sub - 4)*0.05))
        tc[tid] = {'eps_i': float(ei), 'adjusted_priority': float(max(0.01, ap)),
                   'predicted_n': pn, 'required_m': rm}
    return tc

def allocate_ftsa(taxis, tasks, slot_reachability, topk_ratio,
                  task_configs, is_bj=False):
    if not tasks or taxis.empty: return []
    NP = GA_N_POP; MI = GA_MAX_ITER; PM, PX_ = 0.1, 0.9; MR = 0.15
    tc  = task_configs; tbi = {t['task_id']: t for t in tasks}
    ns  = max(1, int(len(taxis) * topk_ratio))
    tl  = taxis.head(ns).to_dict('records')
    tcs = {}
    for tx in tl:
        tid = tx['taxi_id']; ts_ = tx.get('composite_score', 0.5)
        tcs[tid] = [t['task_id'] for t in tasks
                    if (ts_ >= tc.get(t['task_id'], {}).get('eps_i', 0.0)
                        and check_reachable(slot_reachability, tid,
                                            t['task_id'], t.get('time_slot', -1)))]

    def cu(tx, tl_):
        ts_ = tx.get('composite_score', 0.5)
        return sum(ts_*ts_*tc.get(t, {}).get('adjusted_priority',
               tbi.get(t, {}).get('priority_score', 0.5)) for t in tl_ if t in tbi)

    def cc(tx, tl_):
        if not tl_: return 0.0
        return len(tl_)*(1.0 - tx.get('composite_score', 0.5))*(1.0 - tx.get('daily_reputation', 0.5))

    def fl(tx, tl_): return (cu(tx, tl_), 1.0 / (cc(tx, tl_) + 1e-9))

    aa = []; ati = set()
    for tx in tl:
        tid = tx['taxi_id']
        cd  = [t for t in tcs.get(tid, []) if t not in ati]
        if not cd: continue
        mc  = MAX_TASKS_PER_TAXI
        pop = [random.sample(cd, random.randint(1, min(mc, len(cd)))) for _ in range(NP)]
        for _ in range(MI):
            pop.sort(key=lambda c: fl(tx, c), reverse=True)
            n  = len(pop)
            pr = list(reversed([PM + (PX_ - PM)*i/max(n-1, 1) for i in range(n)]))
            tt = sum(pr); pr = [p/tt for p in pr]
            si_ = np.random.choice(n, size=min(n, max(2, n//2)), replace=False, p=pr)
            sv  = [pop[i] for i in si_]; np_ = list(sv)
            while len(np_) < NP and len(sv) >= 2:
                p1, p2 = random.sample(sv, 2)
                u1 = list(set(p1) - set(p2)); u2 = list(set(p2) - set(p1))
                if u1 and u2:
                    g1 = random.choice(u1); g2 = random.choice(u2)
                    np_.extend([[g2 if x == g1 else x for x in p1][:mc],
                                 [g1 if x == g2 else x for x in p2][:mc]])
                elif len(p1) > 1:
                    ch = p1.copy(); i1, i2 = sorted(random.sample(range(len(ch)), 2))
                    ch[i1:i2+1] = ch[i1:i2+1][::-1]; np_.append(ch)
                else:
                    np_.append(p1.copy())
            for idx in range(len(np_)):
                if random.random() < MR:
                    ch = np_[idx].copy()
                    am = [t for t in cd if t not in ati and t not in ch]
                    if len(ch) < mc and am: ch.append(random.choice(am))
                    elif len(ch) > 1: ch.pop(random.randint(0, len(ch)-1))
                    np_[idx] = ch
            pop = np_[:NP]
        if not pop: continue
        pop.sort(key=lambda c: fl(tx, c), reverse=True); bc = pop[0]
        pc = min(1.0, tx.get('wtcs', 0.5)*0.8 + tx.get('daily_reputation', 0.5)*0.2)
        for tkid in bc:
            if tkid in ati: continue
            tk = tbi.get(tkid)
            if tk is None: continue
            fs = base_fitness(tx, tk, False, is_bj)
            aa.append({'taxi_id': tid, 'task_id': tkid, 'fitness_score': fs,
                       'predicted_completion': pc,
                       'allocation_quality': fs*tk.get('priority_score', 0.5)*pc})
            ati.add(tkid)
    return aa

def run_topk_sensitivity(is_beijing: bool):
    is_bj = is_beijing

    if is_bj:
        dl = "Beijing"; lmin, lmax = 39.8, 40.1; omin, omax = 116.3, 116.7
        ls = os_ = 0.015; ed = []
        pkl_name   = 'beijing_preprocessed_data_revise.pkl'
        output_dir = 'topk_sensitivity_beijing'
        cwd_change = None
    else:
        dl = "Chengdu"; lmin, lmax = 30.55, 30.75; omin, omax = 103.9, 104.2
        ls = os_ = 0.015
        ed = [pd.Timestamp('2014-08-07').date(),
              pd.Timestamp('2014-08-13').date()]
        pkl_name   = 'full_preprocessed_data_revise.pkl'
        output_dir = 'topk_sensitivity_chengdu'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd_change = (script_dir, os.path.join(script_dir, 'SiChuan'))

    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()

    if cwd_change: os.chdir(cwd_change[1])
    with open(pkl_name, 'rb') as f: pp = pickle.load(f)
    gd = pp['gps_data1']; rd = pp['result_df']
    if cwd_change: os.chdir(cwd_change[0])

    rd['date'] = pd.to_datetime(rd['date']).dt.date
    gd['time'] = pd.to_datetime(gd['time'])
    dr = pd.to_datetime(rd['date'].unique())
    fd = sorted([d for d in dr if d.date() not in ed])
    ds_t = pd.to_datetime('08:00:00').time()
    de_t = pd.to_datetime('21:00:00').time()
    os.makedirs(output_dir, exist_ok=True)

    drc = {}
    for date in tqdm(fd, desc="可达性预计算"):
        dd = gd[gd['time'].dt.date == date.date()].copy()
        if dd.empty: continue
        st = pd.to_datetime(f'{date.date()} {ds_t}')
        et = pd.to_datetime(f'{date.date()} {de_t}')
        tr = pd.date_range(start=st, end=et, freq='15min')
        tt = generate_grid_tasks(tr, dd, (lmin, lmax), (omin, omax), ls, os_)
        if not tt: continue
        ti = rd[rd['date'] == date.date()]['taxi_id'].tolist()
        drc[date.date()] = precompute_daily_reachability(dd, tt, ti, tr)

    all_results = []

    for topk in tqdm(TOPK_LEVELS, desc="topk 扫描"):
        print()

        learner = AdaptiveWeightLearner(is_beijing=is_bj)

        for sn in SCENARIOS_TO_RUN:
            daily_list   = []
            use_adaptive = (sn == 'BDTA')

            for date in tqdm(fd, desc=f"topk={topk}|{sn}", leave=False):
                dd = gd[gd['time'].dt.date == date.date()].copy()
                if dd.empty: continue
                st = pd.to_datetime(f'{date.date()} {ds_t}')
                et = pd.to_datetime(f'{date.date()} {de_t}')
                tr = pd.date_range(start=st, end=et, freq='15min')

                if use_adaptive: learner.update(date.date(), dd, tr)

                gt = generate_grid_tasks(tr, dd, (lmin, lmax), (omin, omax), ls, os_)
                if not gt:
                    if use_adaptive: learner.record_day(date.date(), dd, tr)
                    continue

                gt = get_task_priorities(
                    gt, dd, tr,
                    is_bj   = is_bj,
                    learner = learner if use_adaptive else None)

                if use_adaptive:
                    spt = {idx: max(1, dd[(dd['time'] >= s) & (dd['time'] < e)]
                                   ['taxi_id'].nunique())
                           for idx, (s, e) in enumerate(zip(tr[:-1], tr[1:]))}
                    for tk in gt:
                        ps_ = spt.get(tk['time_slot'], 1)
                        tk['priority_score'] *= min(4.0, 1.0 + tk['passenger_demand']/ps_)

                gt = normalize_task_priorities(gt)
                dt = sorted(gt, key=lambda x: x.get('priority_score', 0), reverse=True)

                td = rd[rd['date'] == date.date()].copy()
                if td.empty:
                    if use_adaptive: learner.record_day(date.date(), dd, tr)
                    continue

                td['composite_score'] = (0.5 * td['daily_reputation']
                                         + 0.5 * td['wtcs'])
                ft  = td.sort_values('composite_score', ascending=False)
                slr = drc.get(date.date(), {})

                if sn == 'BDTA':
                    al_ = allocate_bdta(ft, dt, slr, topk, is_bj=is_bj)
                elif sn == 'PX':
                    al_ = allocate_px(ft, dt, slr, topk, is_bj=is_bj)
                elif sn == 'DTAA':
                    al_ = allocate_dtaa(ft, dt, slr, topk, is_bj=is_bj)
                elif sn == 'QITA':
                    al_ = allocate_qita(ft, dt, slr, topk, is_bj=is_bj)
                elif sn == 'FTSA':
                    psu, _ = build_ftsa_supply_prediction(
                        dd, tr, lmin, lmax, omin, omax, ls, os_)
                    tcf = calculate_dynamic_task_config(dt, psu, len(ft))
                    al_ = allocate_ftsa(ft, dt, slr, topk, tcf, is_bj=is_bj)
                else:
                    al_ = []

                if al_:
                    pi = build_daily_position_index(dd)
                    m  = compute_metrics(al_, dt, pi, ft['taxi_id'].tolist())
                    if m: daily_list.append(m)

                if use_adaptive: learner.record_day(date.date(), dd, tr)

            if daily_list:
                avg = pd.DataFrame(daily_list).mean(numeric_only=True).to_dict()
                avg['scenario']    = sn
                avg['topk_ratio']  = topk
                avg['dataset']     = dl
                all_results.append(avg)
                print()

        if all_results:
            pd.DataFrame(all_results).to_csv(
                os.path.join(output_dir, f'topk_sensitivity_{dl.lower()}.csv'),
                index=False)

    if all_results:
        df   = pd.DataFrame(all_results)
        path = os.path.join(output_dir, f'topk_sensitivity_{dl.lower()}.csv')
        df.to_csv(path, index=False)

        print()
        print()
        print()
        pivot = df.pivot_table(index='topk_ratio', columns='scenario',
                               values='demand_satisfaction_rate')
        print()
        print()

        print()
        for sn in SCENARIOS_TO_RUN:
            sub = df[df['scenario'] == sn]
            if sub.empty: continue
            best = sub.loc[sub['demand_satisfaction_rate'].idxmax()]
            print()

        print()

    return pd.DataFrame(all_results) if all_results else pd.DataFrame()

if __name__ == '__main__':
    if RUN_BEIJING:
        run_topk_sensitivity(is_beijing=True)
    if RUN_CHENGDU:
        run_topk_sensitivity(is_beijing=False)