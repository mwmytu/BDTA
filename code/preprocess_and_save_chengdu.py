import os
import random
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pickle

OUTPUT_FILENAME = 'full_preprocessed_data_revise.pkl'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

data_dir = 'SiChuan'
lat_min, lat_max, lon_min, lon_max = 30.55, 30.75, 103.9, 104.2
task_interval = 20

if __name__ == '__main__':

    if os.path.basename(os.getcwd()) != data_dir:
        try:
            os.chdir(data_dir)
        except FileNotFoundError:
            print(); exit()

    files = [f for f in os.listdir() if f.endswith('_train_filtered.txt')];
    files.sort()

    def read_single_file(file):
        return pd.read_csv(file, names=['taxi_id', 'latitude', 'longitude', 'status', 'time'], header=None)

    print()
    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:
        gps_data_list = list(tqdm(executor.map(read_single_file, files), total=len(files), desc="读取文件"))
    gps_data = pd.concat(gps_data_list, ignore_index=True)

    print()
    gps_data1 = gps_data.sort_values(by=['taxi_id', 'time'], ignore_index=True)
    gps_data1['time'] = pd.to_datetime(gps_data1['time'], format='%Y/%m/%d %H:%M:%S', errors='coerce')

    print()

    date_range = pd.date_range(start='2014-08-03', end='2014-08-16', freq='D')
    daily_start_time = pd.to_datetime('08:00:00').time()
    daily_end_time = pd.to_datetime('21:00:00').time()
    taxi_ids = list(range(1, 501))

    def calculate_stay_duration(taxi_data, lat_range, lon_range):
        if taxi_data.empty: return pd.Timedelta(0)
        taxi_data = taxi_data.sort_values(by='time');
        taxi_data['in_region'] = (taxi_data['latitude'].between(lat_range[0], lat_range[1])) & (
            taxi_data['longitude'].between(lon_range[0], lon_range[1]))
        taxi_data['region_group'] = (taxi_data['in_region'] != taxi_data['in_region'].shift()).cumsum()
        stay_seconds = sum(
            [(d['time'].iloc[-1] - d['time'].iloc[0]).total_seconds() for _, d in taxi_data.groupby('region_group') if
             d['in_region'].all() and len(d) > 1])
        return pd.Timedelta(seconds=stay_seconds)

    _arrival_status, _stay_duration_rate, _deviation_time_hours, _task_deviation_counts = ({}, {}, {}, {})
    for taxi_id in taxi_ids:
        _arrival_status[taxi_id] = {d: 0 for d in date_range}
        _stay_duration_rate[taxi_id] = {d: [] for d in date_range}
        _deviation_time_hours[taxi_id] = {d: 0 for d in date_range}
        _task_deviation_counts[taxi_id] = {d: 0 for d in date_range}

    for date in tqdm(date_range, desc="预处理每日数据"):
        daily_data = gps_data1[gps_data1['time'].dt.date == date.date()].copy()
        if daily_data.empty: continue
        start_time_day, end_time_day = pd.to_datetime(f"{date.date()} {daily_start_time}"), pd.to_datetime(
            f"{date.date()} {daily_end_time}")
        time_ranges_day = pd.date_range(start=start_time_day, end=end_time_day, freq=f"{task_interval}min")
        for i, task_start in enumerate(time_ranges_day[:-1]):
            task_end, arrival_window_end = time_ranges_day[i + 1], task_start + pd.Timedelta(minutes=15)
            arrival_data = daily_data[
                daily_data['time'].between(task_start, arrival_window_end) & daily_data['latitude'].between(lat_min,
                                                                                                            lat_max) &
                daily_data['longitude'].between(lon_min, lon_max)]
            task_data = daily_data[daily_data['time'].between(arrival_window_end, task_end)]
            for taxi_id in taxi_ids:
                if taxi_id in arrival_data['taxi_id'].unique(): _arrival_status[taxi_id][date] += 1
                taxi_task_data = task_data[task_data['taxi_id'] == taxi_id]
                stay_duration = calculate_stay_duration(taxi_task_data, (lat_min, lat_max), (lon_min, lon_max))
                task_duration_seconds = (task_end - arrival_window_end).total_seconds()
                if task_duration_seconds > 0:
                    stay_seconds = stay_duration.total_seconds()
                    _stay_duration_rate[taxi_id][date].append(stay_seconds / task_duration_seconds)
                    deviation_time = (task_duration_seconds - stay_seconds) / 3600
                    _deviation_time_hours[taxi_id][date] += deviation_time
                    if deviation_time > 0.25: _task_deviation_counts[taxi_id][date] += 1

    print()

    print()

    def calculate_simple_reputation(on_time_rate, avg_stay_rate, deviation_count, total_tasks, historical_reputations):
        daily_score = (on_time_rate * 0.6 + avg_stay_rate * 0.3 + (1 - deviation_count / max(1, total_tasks)) * 0.1)
        if not historical_reputations: return daily_score
        return (daily_score * 0.5 + np.mean(historical_reputations) * 0.5)

    previous_simple_reputations = {taxi_id: [] for taxi_id in taxi_ids}
    daily_scores_df_list = []
    for date in tqdm(sorted(date_range), desc="计算每日动态评分"):
        total_tasks_in_day = (daily_end_time.hour - daily_start_time.hour) * 60 / task_interval
        effective_work_hours = total_tasks_in_day * (task_interval - 15) / 60.0
        for taxi_id in taxi_ids:
            on_time_rate = _arrival_status[taxi_id][date] / total_tasks_in_day if total_tasks_in_day > 0 else 0
            avg_stay_rate = np.mean(_stay_duration_rate[taxi_id][date]) if _stay_duration_rate[taxi_id][date] else 0
            wtcs_val = 0.5 * on_time_rate + 0.3 * avg_stay_rate + 0.2 * (
                    1 - min(_deviation_time_hours[taxi_id][date]
                            / max(effective_work_hours, 1e-6), 1.0))
            deviation_count = _task_deviation_counts[taxi_id][date]
            current_reputation = calculate_simple_reputation(on_time_rate, avg_stay_rate, deviation_count,
                                                             total_tasks_in_day, previous_simple_reputations[taxi_id])
            previous_simple_reputations[taxi_id].append(current_reputation)
            daily_scores_df_list.append(
                {'date': date.date(), 'taxi_id': taxi_id, 'daily_reputation': current_reputation, 'wtcs': wtcs_val})

    result_df = pd.DataFrame(daily_scores_df_list)
    print()

    preprocessed_data = {
        'gps_data1': gps_data1,
        'result_df': result_df,
    }

    with open(OUTPUT_FILENAME, 'wb') as f:
        pickle.dump(preprocessed_data, f)

    print()
    print()