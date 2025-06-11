#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:44:28 2025

@author: jolie
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from geopy.distance import geodesic
import args

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

def load_ground_truth(station_id, timestamps):
    path = os.path.join(args.DATA_DIR, f"{station_id}.csv")
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    df = df.reindex(timestamps)
    df = df.interpolate(method='time', limit_direction='both')
    df = df.fillna(method='ffill').fillna(method='bfill')

    target_seq = []
    #for i in range(args.WINDOW_SIZE, len(df) - args.HORIZON + 1):
    for i in range(args.WINDOW_SIZE, 6400 - args.HORIZON + 1):
        window = df[TARGETS].iloc[i : i + args.HORIZON].values
        if window.shape[0] == args.HORIZON:
            target_seq.append(window)

    return np.stack(target_seq)  # shape: [T, 6, 3]

def main():
    mapping_df = pd.read_csv(os.path.join(args.SAVE_PATH, "virtual_to_station_mapping.csv"),dtype={'station_id': str})
    preds = torch.load(os.path.join(args.SAVE_PATH, "multi_step_preds_virtual8_h6.pt"))  # [T, 8, 6, 3]
    preds = preds.numpy()

    timestamps = pd.date_range(start=args.START_TIME, end=args.END_TIME, freq="10T")
    results = []

    for i, row in mapping_df.iterrows():
        sid = str(row['station_id'])
        pred = preds[:, i, :, :]  # shape: [T, 6, 3]
        true = load_ground_truth(sid, timestamps)  # [T, 6, 3]

        min_len = min(len(pred), len(true))
        pred = pred[:min_len]
        true = true[:min_len]

        row_result = {"station_id": sid}

        for j, var in enumerate(TARGETS):
            p = pred[:, :, j].reshape(-1)
            t = true[:, :, j].reshape(-1)

            if var == 'dd':
                mae = angular_mae(t, p)
                rmse = angular_rmse(t, p)
                mape = angular_mape(t, p)
            else:
                mae = mean_absolute_error(t, p)
                rmse = sqrt(mean_squared_error(t, p))
                mape = np.mean(np.abs((t - p) / (t + 1e-6))) * 100

            row_result[f"{var}_MAE"] = mae
            row_result[f"{var}_RMSE"] = rmse
            row_result[f"{var}_MAPE"] = mape

            # Step-wise MAE
            for step in range(args.HORIZON):
                step_t = true[:, step, j]
                step_p = pred[:, step, j]
                if var == 'dd':
                    step_mae = angular_mae(step_t, step_p)
                else:
                    step_mae = mean_absolute_error(step_t, step_p)
                row_result[f"{var}_MAE_t+{step+1}"] = step_mae

        results.append(row_result)

    df_result = pd.DataFrame(results)
    csv_name = f"{args.MODEL_NAME}_evaluation_metrics_virtual8.csv"
    output_path = os.path.join(args.SAVE_PATH, csv_name)
    df_result.to_csv(output_path, index=False)
    print(f"✅ 评估完成，结果保存在: {output_path}")

if __name__ == "__main__":
    main()
