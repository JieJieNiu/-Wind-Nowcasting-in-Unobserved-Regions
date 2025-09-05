#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 10:07:11 2025

@author: jolie
"""


import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import argparse
import gc
from datetime import datetime
from tqdm import tqdm

TARGETS = ['dd', 'ff', 'gff']

def angular_mae(y_true, y_pred):
    diff = np.abs(y_true - y_pred) % 360
    return np.mean(np.minimum(diff, 360 - diff))

def angular_rmse(y_true, y_pred):
    diff = np.abs(y_true - y_pred) % 360
    diff = np.minimum(diff, 360 - diff)
    return np.sqrt(np.mean(diff ** 2))

def angular_mape(y_true, y_pred):
    diff = np.abs(y_true - y_pred) % 360
    diff = np.minimum(diff, 360 - diff)
    denom = np.abs(y_true) + 1e-6
    return np.mean(diff / denom) * 100

def load_ground_truth(station_id, timestamps, args):
    path = os.path.join(args.TEST_DATA_DIR, f"{station_id}.csv")
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()

    gt_seq = []
    valid_indices = []
    skipped_times = []

    for i in range(36 + args.START_IDX, min(36 + args.END_IDX, len(timestamps) - args.HORIZON + 1)):
        horizon_times = timestamps[i : i + args.HORIZON]
        try:
            window = df.loc[horizon_times][TARGETS]
            if window.shape[0] == args.HORIZON and not window.isnull().values.any():
                gt_seq.append(window.values)
                valid_indices.append(i - 36 - args.START_IDX)
            else:
                skipped_times.append(horizon_times[0])
        except KeyError:
            skipped_times.append(horizon_times[0])

    if gt_seq:
        return np.stack(gt_seq), valid_indices, skipped_times
    else:
        return np.empty((0, args.HORIZON, len(TARGETS))), [], skipped_times

def evaluate_station_custom_range(station_id, timestamps, args):
    pred_path = os.path.join(args.SAVE_PATH, f"{args.MODEL_NAME}_station_{station_id}_pred.npy")
    all_pred = np.load(pred_path, mmap_mode='r')[args.START_IDX:args.END_IDX]

    true, valid_indices, skipped_times = load_ground_truth(station_id, timestamps, args)
    pred = all_pred[valid_indices]

    
    if skipped_times:
        skipped_path = os.path.join(args.SAVE_PATH, f"{args.MODEL_NAME}_station_{station_id}_skipped.csv")
        pd.Series(skipped_times).to_csv(skipped_path, index=False)
        print(f" skip {len(skipped_times)} timesteps（{station_id}），save to {skipped_path}")

    min_len = min(len(pred), len(true))
    pred = pred[:min_len]
    true = true[:min_len]

    metrics = {"station_id": station_id}
    for j, var in enumerate(TARGETS):
        pred_flat = pred[:, :, j].reshape(-1)
        true_flat = true[:, :, j].reshape(-1)

        if var == 'dd':
            mae = angular_mae(true_flat, pred_flat)
            rmse = angular_rmse(true_flat, pred_flat)
            mape = angular_mape(true_flat, pred_flat)
        else:
            mae = mean_absolute_error(true_flat, pred_flat)
            rmse = np.sqrt(np.mean((true_flat - pred_flat)**2))
            mape = np.mean(np.abs((true_flat - pred_flat) / (true_flat + 1e-6))) * 100

        metrics[f"{var}_MAE"] = mae
        metrics[f"{var}_RMSE"] = rmse
        metrics[f"{var}_MAPE"] = mape

        for step in range(args.HORIZON):
            step_pred = pred[:, step, j]
            step_true = true[:, step, j]
            if var == 'dd':
                step_mae = angular_mae(step_true, step_pred)
            else:
                step_mae = mean_absolute_error(step_true, step_pred)
            metrics[f"{var}_MAE_t+{step+1}"] = step_mae

    del pred, true, pred_flat, true_flat, step_pred, step_true
    gc.collect()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SAVE_PATH", required=True)
    parser.add_argument("--MODEL_NAME", required=True)
    parser.add_argument("--TEST_DATA_DIR", required=True)
    parser.add_argument("--START_TIME", required=True)
    parser.add_argument("--END_TIME", required=True)
    parser.add_argument("--HORIZON", type=int, default=6)
    parser.add_argument("--LOG_DIR", default="./logs")
    parser.add_argument("--OUTPUT_CSV", default="eval_results.csv")
    parser.add_argument("--START_IDX", type=int, default=0)
    parser.add_argument("--END_IDX", type=int, default=20000)
    args = parser.parse_args()

    os.makedirs(args.LOG_DIR, exist_ok=True)
    log_path = os.path.join(args.LOG_DIR, f"{args.MODEL_NAME}_eval_log.txt")

    timestamps = pd.date_range(start=args.START_TIME, end=args.END_TIME, freq="10T")
    mapping_df = pd.read_csv(os.path.join(args.SAVE_PATH, "virtual_to_station_mapping.csv"), dtype={'station_id': str})
    station_ids = mapping_df['station_id'].astype(str).tolist()

    results = []
    for sid in tqdm(station_ids, desc="Evaluating stations"):
        try:
            result = evaluate_station_custom_range(sid, timestamps, args)
            results.append(result)
            with open(log_path, 'a') as f:
                f.write(f"[{datetime.now()}]  {sid} done\n")
        except Exception as e:
            with open(log_path, 'a') as f:
                f.write(f"[{datetime.now()}]  {sid} failed: {e}\n")
            continue
        gc.collect()

    df_result = pd.DataFrame(results)
    output_path = os.path.join(args.SAVE_PATH, args.OUTPUT_CSV)
    df_result.to_csv(output_path, index=False)



if __name__ == "__main__":
    import sys
    sys.argv = [
        'evaluate_fixed.py',
        '--SAVE_PATH', './model_save',
        '--MODEL_NAME', 'Augmented_moco',
        '--TEST_DATA_DIR', './test_data',
        '--START_TIME', "2022-12-10T23:30:00",
        '--END_TIME', "2025-04-08T12:40:00",
        '--START_IDX', '0',
        '--END_IDX', '122294'
    ]
    
    main()
