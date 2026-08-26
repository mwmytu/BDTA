

import os
import random
import math
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

IS_BEIJING_DATASET = True
CAPACITY_LEVELS    = [6]

SLOT_MINUTES             = 15
AVG_SPEED_KMH            = 40.0
DEADLINE_MINUTES         = 15
HIGH_PRIORITY_PERCENTILE = 80
TOP_WORKER_RATIO         = 0.2

STAGE1_RATIO = 1.0
STAGE2_RATIO = 0.8

DAILY_START_TIME = pd.to_datetime('08:00:00').time()
DAILY_END_TIME   = pd.to_datetime('21:00:00').time()

ADAPTIVE_WINDOW_DAYS = 7
ADAPTIVE_MIN_DAYS    = 1
WEIGHT_SMOOTH_ALPHA  = 0.3

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class DatasetConfig:
    name:              str
    preprocessed_file: str
    results_filename:  str
    data_dir:          Path
    geo_bounds:        Tuple[float, float, float, float]
    grid_steps:        Tuple[float, float]
    excluded_dates:    List[date]
    avg_speed_kmh:     float = AVG_SPEED_KMH

if IS_BEIJING_DATASET:
    DATA_CONFIG = DatasetConfig(
        name               = "Beijing",
        preprocessed_file  = 'beijing_preprocessed_data_revise.pkl',
        results_filename   = 'beijing_ablation_v10.csv',
        data_dir           = SCRIPT_DIR,
        geo_bounds         = (39.8, 40.1, 116.3, 116.7),
        grid_steps         = (0.015, 0.015),
        excluded_dates     = [],
    )
else:
    DATA_CONFIG = DatasetConfig(
        name               = "Chengdu",
        preprocessed_file  = 'full_preprocessed_data_revise.pkl',
        results_filename   = 'chengdu_ablation_v10.csv',
        data_dir           = SCRIPT_DIR / 'SiChuan',
        geo_bounds         = (30.55, 30.75, 103.9, 104.2),
        grid_steps         = (0.015, 0.015),
        excluded_dates     = [
            pd.Timestamp('2014-08-07').date(),
            pd.Timestamp('2014-08-13').date()
        ],
    )

@dataclass
class AblationConfig:
    name:                        str
    use_bdta_priority:           bool = True
    use_bdta_fitness:            bool = True
    use_two_stage:               bool = True
    use_reputation_in_composite: bool = True
    use_adaptive_weights:        bool = True

ABLATION_SCENARIOS: Dict[str, AblationConfig] = {

    "BDTA_FULL": AblationConfig(
        name                        = "BDTA_FULL",
        use_adaptive_weights        = True,
        use_bdta_priority           = True,
        use_bdta_fitness            = True,
        use_two_stage               = True,
        use_reputation_in_composite = True,
    ),
    "w/o_AdaptiveW": AblationConfig(
        name                        = "w/o_AdaptiveW",
        use_adaptive_weights        = False,
        use_bdta_priority           = True,
        use_bdta_fitness            = True,
        use_two_stage               = True,
        use_reputation_in_composite = True,
    ),
    "w/o_DynPriority": AblationConfig(
        name                        = "w/o_DynPriority",
        use_adaptive_weights        = True,
        use_bdta_priority           = False,
        use_bdta_fitness            = True,
        use_two_stage               = True,
        use_reputation_in_composite = True,
    ),
    "w/o_RepBonus": AblationConfig(
        name                        = "w/o_RepBonus",
        use_adaptive_weights        = True,
        use_bdta_priority           = True,
        use_bdta_fitness            = False,
        use_two_stage               = True,
        use_reputation_in_composite = True,
    ),
    "w/o_TwoStage": AblationConfig(
        name                        = "w/o_TwoStage",
        use_adaptive_weights        = True,
        use_bdta_priority           = True,
        use_bdta_fitness            = True,
        use_two_stage               = False,
        use_reputation_in_composite = True,
    ),
    "w/o_RepInComposite": AblationConfig(
        name                        = "w/o_RepInComposite",
        use_adaptive_weights        = True,
        use_bdta_priority           = True,
        use_bdta_fitness            = True,
        use_two_stage               = True,
        use_reputation_in_composite = False,
    ),
    "BASELINE": AblationConfig(
        name                        = "BASELINE",
        use_adaptive_weights        = True,
        use_bdta_priority           = True,
        use_bdta_fitness            = False,
        use_two_stage               = True,
        use_reputation_in_composite = False,
    ),
}

FACTORIAL_METRICS = [
    'avg_allocation_quality',
    'avg_fitness_score',
    'demand_satisfaction_rate',
    'on_time_demand_satisfaction_rate',
    'task_coverage',
    'high_priority_coverage',
    'high_priority_avg_quality',
    'c3_satisfaction_rate',
    'hp_rank_advantage',
]

def load_preprocessed_data(
        config: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = config.data_dir / config.preprocessed_file
    print()
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: '{path}'")
    with open(path, 'rb') as f:
        cache = pickle.load(f)
    gps = cache['gps_data1'].copy()
    rdf = cache['result_df'].copy()
    gps['time'] = pd.to_datetime(gps['time'])
    rdf['date'] = pd.to_datetime(rdf['date']).dt.date
    print()
    return gps, rdf

def generate_grid_tasks(
        time_ranges: pd.DatetimeIndex,
        gps_data:    pd.DataFrame,
        lat_range:   Tuple[float, float],
        lon_range:   Tuple[float, float],
        lat_step:    float,
        lon_step:    float) -> List[Dict[str, Any]]:

    tasks    = []
    tid      = 0
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
                    c_lat = lat0 + lat_step / 2
                    c_lon = lon0 + lon_step / 2
                    tasks.append({
                        "task_id":          tid,
                        "start_time":       s,
                        "time_slot":        slot_idx,
                        "grid_idx":         (i, j),
                        "centroid":         (c_lat, c_lon),
                        "centroid_lat":     c_lat,
                        "centroid_lon":     c_lon,
                        "passenger_demand": p,
                    })
                    tid += 1
    return tasks

def haversine_km(lat1, lon1, lat2, lon2):
    R    = 6371.0
    phi1 = radians(lat1); phi2 = radians(lat2)
    dphi = radians(lat2 - lat1); dlam = radians(lon2 - lon1)
    a    = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def build_daily_position_index(daily_gps: pd.DataFrame) -> Dict:
    index = {}
    for tid, group in daily_gps.groupby('taxi_id'):
        g = group[['time', 'latitude', 'longitude']].sort_values('time')
        index[tid] = {
            'times': g['time'].to_numpy(dtype='datetime64[ns]'),
            'lats':  g['latitude'].to_numpy(float),
            'lons':  g['longitude'].to_numpy(float),
        }
    return index

def get_position_at(
        pos_index: Dict, taxi_id,
        query_time: pd.Timestamp) -> Optional[Tuple[float, float]]:
    entry = pos_index.get(taxi_id)
    if entry is None or len(entry['times']) == 0:
        return None
    i = np.searchsorted(
        entry['times'], np.datetime64(query_time), side='right') - 1
    if i < 0:
        return None
    return float(entry['lats'][i]), float(entry['lons'][i])

def build_reachability_matrix(worker_locs: np.ndarray,
                               task_locs:   np.ndarray) -> np.ndarray:
    if len(worker_locs) == 0 or len(task_locs) == 0:
        return np.zeros((len(worker_locs), len(task_locs)), dtype=bool)
    w_lat = np.radians(worker_locs[:, 0:1])
    w_lon = np.radians(worker_locs[:, 1:2])
    t_lat = np.radians(task_locs[:, 0])
    t_lon = np.radians(task_locs[:, 1])
    dlat  = t_lat - w_lat
    dlon  = t_lon - w_lon
    a     = (np.sin(dlat/2)**2
             + np.cos(w_lat)*np.cos(t_lat)*np.sin(dlon/2)**2)
    dist_km  = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    travel_m = (dist_km / AVG_SPEED_KMH) * 60.0
    return travel_m <= SLOT_MINUTES

def precompute_daily_reachability(
        daily_gps:   pd.DataFrame,
        daily_tasks: List[Dict],
        taxi_ids:    List,
        time_ranges: pd.DatetimeIndex) -> Dict:

    pos_index      = build_daily_position_index(daily_gps)
    tasks_by_slot: Dict[int, List] = {}
    for task in daily_tasks:
        tasks_by_slot.setdefault(task['time_slot'], []).append(task)

    slot_reachability = {}
    for slot_idx, slot_tasks in tasks_by_slot.items():
        if slot_idx >= len(time_ranges) - 1:
            continue
        slot_start = time_ranges[slot_idx]

        worker_lats, worker_lons, valid_ids = [], [], []
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
                'matrix':         np.zeros(
                    (0, len(slot_tasks)), dtype=bool),
                'taxi_id_to_idx': {},
                'task_id_to_idx': {
                    tid: i for i, tid in enumerate(task_ids)},
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
            'taxi_id_to_idx': {
                tid: i for i, tid in enumerate(valid_ids)},
            'task_id_to_idx': {
                tid: i for i, tid in enumerate(task_ids)},
        }
    return slot_reachability

def check_reachable(slot_reachability: Dict,
                    taxi_id, task_id, slot_idx: int) -> bool:
    sr = slot_reachability.get(slot_idx)
    if sr is None:
        return True
    w_idx = sr['taxi_id_to_idx'].get(taxi_id)
    t_idx = sr['task_id_to_idx'].get(task_id)
    if w_idx is None or t_idx is None:
        return True
    return bool(sr['matrix'][w_idx, t_idx])

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

    def __init__(self, window_days: int  = ADAPTIVE_WINDOW_DAYS,
                 smooth_alpha:     float = WEIGHT_SMOOTH_ALPHA):
        self.window_days       = window_days
        self.smooth_alpha      = smooth_alpha
        self.weights           = dict(self.DEFAULT_WEIGHTS)
        self.time_factors      = dict(self.DEFAULT_TIME_FACTORS)
        self._history: Dict[date, Dict] = {}
        self._adaptive_enabled = False

    def update(self, cur_date: date,
               gps_data: pd.DataFrame,
               tr: pd.DatetimeIndex) -> None:
        window_dates = sorted(self._history.keys())
        if len(window_dates) < ADAPTIVE_MIN_DAYS:
            return
        self._adaptive_enabled = True
        use_dates = window_dates[-self.window_days:]

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
                self.time_factors[h] = (self.smooth_alpha * new
                                        + (1 - self.smooth_alpha) * old)

        ds, ss = [], []
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
            a  = self.smooth_alpha
            self.weights['passenger_demand'] = (
                a * nd + (1 - a) * self.weights['passenger_demand'])
            self.weights['taxi_supply'] = (
                a * ns + (1 - a) * self.weights['taxi_supply'])
            self.weights['time_factor'] = (
                a * nt + (1 - a) * self.weights['time_factor'])

    def record_day(self, cur_date: date,
                   gps_data: pd.DataFrame,
                   tr: pd.DatetimeIndex) -> None:
        slot_demand: List[float]      = []
        slot_supply: List[float]      = []
        hour_demand: Dict[int, float] = {}
        for s, e in zip(tr[:-1], tr[1:]):
            data = gps_data[
                (gps_data['time'] >= s) & (gps_data['time'] < e)]
            if data.empty:
                continue
            h = s.hour
            if IS_BEIJING_DATASET:
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

    def get_weights(self) -> Dict[str, float]:
        return dict(self.weights)

    def get_time_factor(self, h: int) -> float:
        return self.time_factors.get(h, 0.6)

    def is_adaptive(self) -> bool:
        return self._adaptive_enabled

    def summary(self) -> str:
        return (
            f"adaptive={'ON' if self._adaptive_enabled else 'OFF'}  "
            f"weights=demand:{self.weights['passenger_demand']:.3f} "
            f"supply:{self.weights['taxi_supply']:.3f} "
            f"time:{self.weights['time_factor']:.3f}")

class TaskPrioritizer:
    FIXED_WEIGHTS = {
        'passenger_demand': 0.5,
        'taxi_supply':      0.25,
        'time_factor':      0.25,
    }

    def __init__(self,
                 learner:      Optional[AdaptiveWeightLearner] = None,
                 use_adaptive: bool = True):
        self.learner      = learner
        self.use_adaptive = use_adaptive and (learner is not None)

    def _get_weights(self) -> Dict[str, float]:
        if self.use_adaptive and self.learner.is_adaptive():
            return self.learner.get_weights()
        return dict(self.FIXED_WEIGHTS)

    def _time_factor(self, h: int) -> float:
        if self.use_adaptive and self.learner.is_adaptive():
            return self.learner.get_time_factor(h)
        if h in [7, 8, 9, 17, 18, 19]: return 1.0
        if h in [11, 12, 13]:           return 0.8
        return 0.6

    def _analyze_demand(self, gps: pd.DataFrame,
                        tr: pd.DatetimeIndex) -> Dict[int, Dict]:
        demand: Dict[int, Dict] = {}
        for i, (s, e) in enumerate(zip(tr[:-1], tr[1:])):
            data = gps[(gps['time'] >= s) & (gps['time'] < e)]
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
                'normalized_supply': min(1.0, sup / 30),
            }
        return demand

    def calculate_base_priorities(self, tasks: List[Dict],
                                   gps: pd.DataFrame,
                                   tr: pd.DatetimeIndex) -> List[Dict]:
        info    = self._analyze_demand(gps, tr)
        weights = self._get_weights()
        for t in tasks:
            si = info.get(t['time_slot'])
            if si:
                tf   = self._time_factor(si['start_time'].hour)
                base = (
                    si['normalized_demand'] * weights['passenger_demand']
                    + (1 - si['normalized_supply']) * weights['taxi_supply']
                    + tf * weights['time_factor'])
                t['priority_score'] = base * (
                    1 + min(t['passenger_demand'] / 10, 1))
            else:
                t['priority_score'] = t['passenger_demand'] / 10.0
        return tasks

    def apply_bdta_dynamic_adjustment(
            self, tasks: List[Dict],
            gps: pd.DataFrame,
            tr: pd.DatetimeIndex) -> List[Dict]:
        supply = {
            idx: max(1, gps[
                (gps['time'] >= s) & (gps['time'] < e)
            ]['taxi_id'].nunique())
            for idx, (s, e) in enumerate(zip(tr[:-1], tr[1:]))
        }
        for t in tasks:
            f = min(4.0, 1.0 + t['passenger_demand']
                    / supply.get(t['time_slot'], 1))
            t['priority_score'] *= f
        return tasks

def compute_full_model_priority_bounds(
        tasks_template: List[Dict],
        gps:            pd.DataFrame,
        tr:             pd.DatetimeIndex,
        learner:        Optional[AdaptiveWeightLearner] = None
) -> Tuple[float, float]:
    p          = TaskPrioritizer(learner=learner, use_adaptive=True)
    tasks_copy = [dict(t) for t in tasks_template]
    tasks_copy = p.calculate_base_priorities(tasks_copy, gps, tr)
    tasks_copy = p.apply_bdta_dynamic_adjustment(tasks_copy, gps, tr)
    scores     = [t['priority_score'] for t in tasks_copy]
    if not scores:
        return 0.0, 1.0
    return min(scores), max(scores)

def normalize_with_ref(tasks: List[Dict],
                       p_min: float,
                       p_max: float) -> List[Dict]:
    denom = p_max - p_min + 1e-6
    for t in tasks:
        raw = t.get('priority_score', 0.0)
        t['priority_score_raw'] = raw
        t['priority_score']     = float(
            np.clip((raw - p_min) / denom, 0.0, 1.0))
    return tasks

class TaxiScorer:

    @staticmethod
    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-6:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    def get_scores(self,
                   taxi_df: pd.DataFrame,
                   use_reputation_in_composite: bool = True
                   ) -> pd.DataFrame:
        taxi_df = taxi_df.copy()

        taxi_df['daily_reputation_norm'] = self._minmax(
            taxi_df['daily_reputation'])

        if use_reputation_in_composite:

            taxi_df['composite_score'] = (
                0.5 * taxi_df['daily_reputation']
                + 0.5 * taxi_df['wtcs'])
            taxi_df['_rep_weight_used'] = 0.5
        else:

            taxi_df['composite_score']  = taxi_df['wtcs']
            taxi_df['_rep_weight_used'] = 0.0

        return taxi_df

class TaskAllocator:
    W_COMPOSITE   = 0.5
    W_PRIORITY    = 0.5
    W_REP_BONUS   = 0.3
    POWER_EXP     = 1.2
    SIGMOID_SCALE = 2.0

    def __init__(self, cfg: AblationConfig, max_tasks: int):
        self.cfg           = cfg
        self.max_tasks     = max_tasks

        self.sigmoid_shift = 1.0 if IS_BEIJING_DATASET else 1.5

    def _get_fitness(self, taxi: Dict, task: Dict) -> float:
        taxi_score = taxi.get('composite_score', 0.5)
        task_score = task.get('priority_score',  0.5)
        base       = (self.W_COMPOSITE * taxi_score
                      + self.W_PRIORITY * task_score)
        if self.cfg.use_bdta_fitness:
            rep_bonus = taxi.get('daily_reputation_norm', 0.5)
            raw = (base
                   + self.W_REP_BONUS * rep_bonus) ** self.POWER_EXP
        else:
            raw = base ** self.POWER_EXP

        return min(1.0, 1.1 / (
            1 + math.exp(
                -self.SIGMOID_SCALE * (raw - self.sigmoid_shift))))

    def allocate(self, taxis: pd.DataFrame,
                 tasks: List[Dict],
                 slot_reachability: Dict) -> List[Dict[str, Any]]:
        taxis_s = taxis.sort_values('composite_score', ascending=False)
        tasks_s = sorted(tasks,
                         key=lambda t: t.get('priority_score', 0),
                         reverse=True)
        if self.cfg.use_two_stage:
            return self._two_stage(taxis_s, tasks_s, slot_reachability)
        else:
            return self._run_stage(
                taxis_s, tasks_s,
                ratio        = 1.0,
                capacity_map = None,
                slot_reach   = slot_reachability)

    def _two_stage(self, taxis: pd.DataFrame,
                   tasks: List[Dict],
                   slot_reachability: Dict) -> List[Dict]:
        if not tasks:
            return []

        thr    = np.percentile(
            [t['priority_score'] for t in tasks],
            HIGH_PRIORITY_PERCENTILE)
        high   = [t for t in tasks if t['priority_score'] >= thr]
        normal = [t for t in tasks if t['priority_score'] <  thr]
        allocs = []

        if high:
            a1 = self._run_stage(
                taxis, high,
                ratio        = STAGE1_RATIO,
                capacity_map = None,
                slot_reach   = slot_reachability)
            allocs.extend(a1)

            used_count = pd.Series(
                [x['taxi_id'] for x in a1]
            ).value_counts().to_dict()
            assigned_task_ids = {x['task_id'] for x in a1}
            normal = [t for t in normal
                      if t['task_id'] not in assigned_task_ids]

            remaining = taxis.copy()
            remaining['remaining_capacity'] = self.max_tasks
            for tid, cnt in used_count.items():
                remaining.loc[
                    remaining['taxi_id'] == tid,
                    'remaining_capacity'] -= cnt
            remaining = remaining[remaining['remaining_capacity'] > 0]
        else:
            remaining = taxis.copy()
            remaining['remaining_capacity'] = self.max_tasks

        if normal and not remaining.empty:
            cap_map = (remaining
                       .set_index('taxi_id')['remaining_capacity']
                       .to_dict())
            a2 = self._run_stage(
                remaining, normal,
                ratio        = STAGE2_RATIO,
                capacity_map = cap_map,
                slot_reach   = slot_reachability)
            allocs.extend(a2)

        return allocs

    def _run_stage(self,
                   taxis:        pd.DataFrame,
                   tasks:        List[Dict],
                   ratio:        float,
                   capacity_map: Optional[Dict],
                   slot_reach:   Dict) -> List[Dict]:

        n_workers = max(1, int(len(taxis) * ratio))
        pruned    = taxis.head(n_workers)
        if pruned.empty or not tasks:
            return []

        expanded = []
        for r in pruned.to_dict('records'):
            cap = int(
                capacity_map.get(r['taxi_id'], self.max_tasks)
                if capacity_map is not None
                else self.max_tasks)
            cap = max(1, cap)
            expanded.extend([r] * cap)

        if not expanded:
            return []

        INF         = 1e9
        cost_matrix = np.full((len(expanded), len(tasks)), INF)
        for i, taxi in enumerate(expanded):
            for j, task in enumerate(tasks):
                if check_reachable(slot_reach,
                                   taxi['taxi_id'],
                                   task['task_id'],
                                   task.get('time_slot', -1)):
                    cost_matrix[i, j] = -self._get_fitness(taxi, task)

        rows, cols  = linear_sum_assignment(cost_matrix)
        allocations = []
        for r, c in zip(rows, cols):
            if cost_matrix[r, c] >= INF:
                continue
            allocations.append(
                self._record(expanded[r], tasks[c],
                             -cost_matrix[r, c]))
        return allocations

    def _record(self, taxi: Dict, task: Dict, fit: float) -> Dict:
        pc = min(1.0,
                 taxi.get('wtcs', 0.5) * 0.8
                 + taxi.get('daily_reputation', 0.5) * 0.2)
        return {
            'taxi_id':             taxi['taxi_id'],
            'task_id':             task['task_id'],
            'task_start_time':     task['start_time'],
            'task_grid_idx':       task['grid_idx'],
            'task_centroid_lat':   task['centroid_lat'],
            'task_centroid_lon':   task['centroid_lon'],
            'task_priority_score': task.get('priority_score', 0.0),
            'time_slot':           task.get('time_slot', -1),
            'fitness_score':       fit,
            'predicted_completion':pc,
            'allocation_quality':  (fit
                                    * task.get('priority_score', 0.5)
                                    * pc),
            'rep_in_composite':    taxi.get('composite_score', 0.5),
            'rep_bonus_value':     taxi.get('daily_reputation_norm',
                                            0.5),
        }

def analyze_results(allocs:            List[Dict],
                    taxis:             pd.DataFrame,
                    tasks:             List[Dict],
                    daily_data:        pd.DataFrame,
                    slot_reachability: Dict) -> Dict:
    if not allocs:
        return {}

    df             = pd.DataFrame(allocs)
    asgn           = df['taxi_id'].value_counts()
    all_grids      = {t['grid_idx'] for t in tasks}
    assigned_grids = (
        set(tuple(g) if isinstance(g, list) else g
            for g in df['task_grid_idx'].tolist())
        if 'task_grid_idx' in df.columns else set())

    pos_idx         = build_daily_position_index(daily_data)
    task_demand_map = {t['task_id']: t.get('passenger_demand', 1)
                       for t in tasks}

    dists           = []
    times_min_list  = []
    viol            = 0
    evaled          = 0
    on_time_demand  = 0.0
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
            task_loc   = (row['task_centroid_lat'],
                          row['task_centroid_lon'])
            p_demand_i = task_demand_map.get(row['task_id'], 1)

            if prev_loc is None:
                origin = get_position_at(pos_idx, taxi_id, task_start)
                depart = task_start
            else:
                gap_min = (
                    (task_start - prev_arrival).total_seconds() / 60.0
                    if prev_arrival is not None else SLOT_MINUTES)
                if gap_min >= SLOT_MINUTES:
                    origin = get_position_at(
                        pos_idx, taxi_id, task_start)
                    if origin is None:
                        origin = prev_loc
                    depart = task_start
                else:
                    origin = prev_loc
                    depart = max(prev_arrival, task_start)

            if origin is None:
                continue

            dist_km  = haversine_km(origin[0], origin[1],
                                    task_loc[0], task_loc[1])
            travel_m = (dist_km / DATA_CONFIG.avg_speed_kmh) * 60.0
            arrival  = depart + pd.Timedelta(minutes=travel_m)
            deadline = task_start + pd.Timedelta(
                minutes=DEADLINE_MINUTES)

            total_travel_km += dist_km
            dists.append(dist_km)
            times_min_list.append(travel_m)
            evaled += 1
            if travel_m > SLOT_MINUTES:
                viol += 1
            if arrival <= deadline:
                on_time_demand += p_demand_i

            prev_loc     = task_loc
            prev_arrival = arrival

    total_demand    = max(sum(task_demand_map.values()), 1)
    assigned_demand = sum(
        task_demand_map[tid]
        for tid in df['task_id'].unique()
        if tid in task_demand_map)
    dsr    = assigned_demand / total_demand
    ot_dsr = on_time_demand  / total_demand

    km_per_demand = (total_travel_km / assigned_demand
                     if assigned_demand > 0 else 0.0)
    demand_per_km = (assigned_demand / max(total_travel_km, 1e-6)
                     if assigned_demand > 0 else 0.0)

    n_total_workers  = max(len(taxis), 1)
    n_active_workers = int(df['taxi_id'].nunique())
    part_rate        = n_active_workers / n_total_workers

    c3_satisfied = sum(
        1 for _, row in df.iterrows()
        if check_reachable(slot_reachability,
                           row['taxi_id'], row['task_id'],
                           int(row.get('time_slot', -1))))

    all_priorities = [t.get('priority_score', 0.0) for t in tasks]
    high_thr       = (np.percentile(
        all_priorities, HIGH_PRIORITY_PERCENTILE)
                      if all_priorities else 0.5)
    high_task_ids  = {t['task_id'] for t in tasks
                      if t.get('priority_score', 0.0) >= high_thr}
    normal_task_ids= {t['task_id'] for t in tasks
                      if t.get('priority_score', 0.0) < high_thr}

    df_high   = df[df['task_id'].isin(high_task_ids)].copy()
    df_normal = df[df['task_id'].isin(normal_task_ids)].copy()

    total_alloc_quality    = float(df['allocation_quality'].sum())
    total_quality_per_taxi = (total_alloc_quality
                              / max(df['taxi_id'].nunique(), 1))
    high_priority_coverage = (df_high['task_id'].nunique()
                               / max(len(high_task_ids), 1))
    high_priority_avg_quality = float(
        df_high['allocation_quality'].mean()
        if not df_high.empty else 0.0)
    normal_task_coverage = (df_normal['task_id'].nunique()
                             / max(len(normal_task_ids), 1))

    taxi_score_map = (taxis.set_index('taxi_id')['composite_score']
                      .to_dict())
    df['taxi_composite'] = (df['taxi_id']
                            .map(taxi_score_map).fillna(0.0))
    all_scores    = sorted(taxi_score_map.values(), reverse=True)
    n_top         = max(1, int(len(all_scores) * TOP_WORKER_RATIO))
    top_threshold = all_scores[n_top - 1] if all_scores else 0.5

    if not df_high.empty:
        df_high['taxi_composite'] = (
            df_high['taxi_id'].map(taxi_score_map).fillna(0.0))
        hp_top_worker_rate = float(
            (df_high['taxi_composite'] >= top_threshold).sum()
            / max(len(df_high), 1))
        hp_avg_fitness = float(
            df_high['fitness_score'].mean()
            if 'fitness_score' in df_high.columns else 0.0)
    else:
        hp_top_worker_rate = 0.0
        hp_avg_fitness     = 0.0

    if not df_normal.empty:
        df_normal['taxi_composite'] = (
            df_normal['taxi_id'].map(taxi_score_map).fillna(0.0))
        normal_top_worker_rate = float(
            (df_normal['taxi_composite'] >= top_threshold).sum()
            / max(len(df_normal), 1))
        normal_avg_fitness = float(
            df_normal['fitness_score'].mean()
            if 'fitness_score' in df_normal.columns else 0.0)
    else:
        normal_top_worker_rate = 0.0
        normal_avg_fitness     = 0.0

    result = {
        'avg_fitness_score':
            round(float(df['fitness_score'].mean()), 4),
        'avg_predicted_completion':
            round(float(df['predicted_completion'].mean()), 4),
        'avg_allocation_quality':
            round(float(df['allocation_quality'].mean()), 4),
        'demand_satisfaction_rate':
            round(float(dsr), 4),
        'on_time_demand_satisfaction_rate':
            round(float(ot_dsr), 4),
        'travel_km_per_served_demand':
            round(float(km_per_demand), 4),
        'served_demand_per_km':
            round(float(demand_per_km), 4),
        'worker_participation_rate':
            round(float(part_rate), 4),
        'total_demand':       float(total_demand),
        'assigned_demand':    float(assigned_demand),
        'on_time_demand':     float(on_time_demand),
        'total_travel_km':    round(float(total_travel_km), 4),
        'n_active_workers':   float(n_active_workers),
        'n_total_workers':    float(n_total_workers),
        'n_allocations':      float(len(df)),
        'total_allocations':        len(df),
        'taxi_coverage':
            n_active_workers / n_total_workers,
        'task_coverage':
            df['task_id'].nunique() / max(len(tasks), 1),
        'spatial_coverage_ratio':
            len(assigned_grids) / max(len(all_grids), 1),
        'load_balance_index':
            (float(asgn.std() / asgn.mean())
             if len(asgn) > 1 else 0.0),
        'avg_tasks_per_taxi':
            float(asgn.mean()) if not asgn.empty else 0.0,
        'avg_travel_distance_km':
            float(np.mean(dists)) if dists else 0.0,
        'avg_travel_time_min':
            float(np.mean(times_min_list)) if times_min_list else 0.0,
        'deadline_violation_rate':
            viol / evaled if evaled > 0 else 0.0,
        'c3_satisfaction_rate':
            c3_satisfied / max(len(df), 1),
        'total_allocation_quality': total_alloc_quality,
        'total_quality_per_taxi':   total_quality_per_taxi,
        'high_priority_coverage':   high_priority_coverage,
        'high_priority_avg_quality':high_priority_avg_quality,
        'normal_task_coverage':     normal_task_coverage,
        'n_high_priority_tasks':    float(len(high_task_ids)),
        'n_normal_tasks':           float(len(normal_task_ids)),
        'hp_top_worker_rate':       hp_top_worker_rate,
        'normal_top_worker_rate':   normal_top_worker_rate,
        'hp_rank_advantage':
            hp_top_worker_rate - normal_top_worker_rate,
        'hp_avg_worker_composite':
            (float(df_high['taxi_composite'].mean())
             if not df_high.empty else 0.0),
        'normal_avg_worker_composite':
            (float(df_normal['taxi_composite'].mean())
             if not df_normal.empty else 0.0),
        'hp_fitness_advantage':
            hp_avg_fitness - normal_avg_fitness,
        'avg_rep_in_composite':
            float(df['rep_in_composite'].mean()
                  if 'rep_in_composite' in df.columns else 0.0),
        'avg_rep_bonus_value':
            float(df['rep_bonus_value'].mean()
                  if 'rep_bonus_value' in df.columns else 0.0),
        'rep_weight_in_composite':
            float(taxis['_rep_weight_used'].mean()
                  if '_rep_weight_used' in taxis.columns else 0.0),
    }
    return {k: (0.0 if pd.isna(v) else v) for k, v in result.items()}

def print_summary(all_results: List[Dict], c_value: int) -> None:
    c_df = pd.DataFrame(
        [r for r in all_results if r['c_value'] == c_value])
    if c_df.empty:
        return

    print()
    print()
    print()
    print()

    for title, cols in [
        ("核心指标", [
            'scenario', 'avg_fitness_score',
            'avg_allocation_quality',
            'demand_satisfaction_rate',
            'on_time_demand_satisfaction_rate',
            'worker_participation_rate',
        ]),
        ("两阶段效应", [
            'scenario', 'hp_top_worker_rate',
            'normal_top_worker_rate', 'hp_rank_advantage',
            'hp_fitness_advantage', 'high_priority_coverage',
            'high_priority_avg_quality',
        ]),
        ("声誉路径", [
            'scenario', 'avg_rep_in_composite',
            'avg_rep_bonus_value', 'rep_weight_in_composite',
        ]),
    ]:
        valid = [c for c in cols if c in c_df.columns]
        print()
        print()

    base = c_df[c_df['scenario'] == 'BDTA_FULL']
    if not base.empty:
        print()
        key_metrics = [m for m in [
            'avg_fitness_score', 'avg_allocation_quality',
            'hp_rank_advantage', 'high_priority_avg_quality',
            'on_time_demand_satisfaction_rate',
            'demand_satisfaction_rate',
        ] if m in c_df.columns]
        for _, row in c_df[
                c_df['scenario'] != 'BDTA_FULL'].iterrows():
            diffs = []
            for m in key_metrics:
                bv = base[m].values[0]
                rv = row[m]
                if abs(bv) > 1e-9:
                    diffs.append(
                        f"{m}={((rv-bv)/abs(bv)*100):+.1f}%")
                else:
                    diffs.append(f"{m}={rv:+.4f}(abs)")
            print()
    print()

def compute_factorial_effects(
        all_results: List[Dict], c_value: int) -> pd.DataFrame:
    c_df = pd.DataFrame(
        [r for r in all_results if r['c_value'] == c_value])
    if c_df.empty:
        return pd.DataFrame()

    def get(sc, metric):
        row = c_df[c_df['scenario'] == sc]
        if row.empty or metric not in row.columns:
            return float('nan')
        return float(row[metric].values[0])

    rows = []
    for metric in FACTORIAL_METRICS:
        f11 = get("BDTA_FULL",          metric)
        f10 = get("w/o_RepBonus",       metric)
        f01 = get("w/o_RepInComposite", metric)
        f00 = get("BASELINE",           metric)
        ea  = 0.5 * ((f11 - f01) + (f10 - f00))
        eb  = 0.5 * ((f11 - f10) + (f01 - f00))
        eab = f11 - f10 - f01 + f00
        rows.append({
            'metric':                metric,
            'BDTA_FULL':             f11,
            'w/o_RepBonus':          f10,
            'w/o_RepInComposite':    f01,
            'BASELINE':              f00,
            'MainEffect_A_C(w)':     round(ea,  4),
            'MainEffect_B_RepBonus': round(eb,  4),
            'Interaction_AxB':       round(eab, 4),
        })
    return pd.DataFrame(rows)

def print_factorial_analysis(
        all_results: List[Dict], c_value: int) -> None:
    effects_df = compute_factorial_effects(all_results, c_value)
    if effects_df.empty:
        return
    print()
    print()
    print()
    print()
    print()
    for _, r in effects_df.iterrows():
        print()
    print()

def run_one_day_all_scenarios(
        date_ts:   pd.Timestamp,
        gps:       pd.DataFrame,
        result_df: pd.DataFrame,
        c_value:   int,
        learner:   AdaptiveWeightLearner) -> Dict[str, Dict]:

    cur        = date_ts.date()
    daily_data = gps[gps['time'].dt.date == cur].copy()
    if daily_data.empty:
        return {}

    s_ts = pd.to_datetime(f'{cur} {DAILY_START_TIME}')
    e_ts = pd.to_datetime(f'{cur} {DAILY_END_TIME}')
    tr   = pd.date_range(start=s_ts, end=e_ts, freq='15min')

    learner.update(cur, daily_data, tr)

    tasks_template = generate_grid_tasks(
        tr, daily_data,
        DATA_CONFIG.geo_bounds[:2],
        DATA_CONFIG.geo_bounds[2:],
        *DATA_CONFIG.grid_steps)

    if not tasks_template:
        learner.record_day(cur, daily_data, tr)
        return {}

    taxi_df = result_df[result_df['date'] == cur].copy()
    if taxi_df.empty:
        learner.record_day(cur, daily_data, tr)
        return {}

    taxi_ids          = taxi_df['taxi_id'].tolist()
    slot_reachability = precompute_daily_reachability(
        daily_data, tasks_template, taxi_ids, tr)

    p_min, p_max = compute_full_model_priority_bounds(
        tasks_template, daily_data, tr, learner=learner)

    scorer      = TaxiScorer()
    day_results: Dict[str, Dict] = {}

    for sc_name, cfg in ABLATION_SCENARIOS.items():

        tasks = [dict(t) for t in tasks_template]

        prio = TaskPrioritizer(
            learner      = learner,
            use_adaptive = cfg.use_adaptive_weights)
        prio.calculate_base_priorities(tasks, daily_data, tr)
        if cfg.use_bdta_priority:
            prio.apply_bdta_dynamic_adjustment(tasks, daily_data, tr)

        tasks = normalize_with_ref(tasks, p_min, p_max)

        taxis_scored = scorer.get_scores(
            taxi_df,
            use_reputation_in_composite=(
                cfg.use_reputation_in_composite))

        alloc  = TaskAllocator(cfg, c_value)
        allocs = alloc.allocate(
            taxis_scored, tasks, slot_reachability)

        if allocs:
            task_slot_map = {
                t['task_id']: t['time_slot'] for t in tasks}
            for a in allocs:
                a['time_slot'] = task_slot_map.get(
                    a['task_id'], -1)
            metrics = analyze_results(
                allocs, taxis_scored, tasks,
                daily_data, slot_reachability)
            if metrics:
                day_results[sc_name] = metrics

    learner.record_day(cur, daily_data, tr)
    return day_results

def main():
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()

    gps, result_df = load_preprocessed_data(DATA_CONFIG)

    all_dates = pd.to_datetime(sorted(result_df['date'].unique()))
    filtered_dates = [
        d for d in all_dates
        if d.date() not in DATA_CONFIG.excluded_dates]
    print()

    output_dir = (SCRIPT_DIR
                  / f"{DATA_CONFIG.name.lower()}_ablation_v10")
    output_dir.mkdir(exist_ok=True)
    print()

    all_results: List[Dict] = []

    for c in tqdm(CAPACITY_LEVELS, desc="容量 C"):
        sc_daily: Dict[str, List[Dict]] = {
            n: [] for n in ABLATION_SCENARIOS}

        learner = AdaptiveWeightLearner(
            window_days  = ADAPTIVE_WINDOW_DAYS,
            smooth_alpha = WEIGHT_SMOOTH_ALPHA)

        for d in tqdm(filtered_dates,
                      desc=f"C={c}", leave=False):
            day = run_one_day_all_scenarios(
                d, gps, result_df, c, learner)
            for sc in ABLATION_SCENARIOS:
                if day.get(sc):
                    sc_daily[sc].append(day[sc])

        print()

        for sc, rows in sc_daily.items():
            if rows:
                avg = pd.DataFrame(rows).mean(
                    numeric_only=True).to_dict()
                avg['scenario'] = sc
                avg['c_value']  = c
                all_results.append(avg)

        print_summary(all_results, c)
        print_factorial_analysis(all_results, c)

        pd.DataFrame(all_results).to_csv(
            output_dir / DATA_CONFIG.results_filename,
            index=False)

    final = pd.DataFrame(all_results)
    path  = output_dir / DATA_CONFIG.results_filename
    final.to_csv(path, index=False)

    for c in CAPACITY_LEVELS:
        effects_df = compute_factorial_effects(all_results, c)
        if not effects_df.empty:
            ep = output_dir / f"factorial_effects_C{c}.csv"
            effects_df.to_csv(ep, index=False)
            print()

    print()
    print()

    key_cols = [col for col in [
        'scenario', 'c_value',
        'avg_fitness_score', 'avg_allocation_quality',
        'demand_satisfaction_rate',
        'on_time_demand_satisfaction_rate',
        'hp_rank_advantage', 'high_priority_avg_quality',
        'task_coverage', 'c3_satisfaction_rate',
    ] if col in final.columns]
    print()

if __name__ == '__main__':
    main()