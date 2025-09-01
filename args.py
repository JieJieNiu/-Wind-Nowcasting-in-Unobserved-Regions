#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:18:35 2025

@author: Jie
"""

import argparse
import os
import pandas as pd

DEFAULT_DIR=os.getcwd()

"""Raw data directory and model directory"""

DATA_DIR = DEFAULT_DIR + os.path.sep + 'station_data' + os.path.sep  ##where the orignal data is saved
TEST_DATA_DIR= DEFAULT_DIR + os.path.sep + 'test_data' + os.path.sep
INFO_PATH=DEFAULT_DIR + os.path.sep + 'info'       ##where the station information csv saved
SAVE_PATH=DEFAULT_DIR + os.path.sep +'model_save' ##where trained model saved
DATA_SAVE_PATH=DEFAULT_DIR + os.path.sep +'data_save' ##where the processed data is saved
RESUME_PATH=DEFAULT_DIR +os.path.sep + 'model_save' +os.path.sep + 'model_epoch200.pt' ##if need resume


"""Stations Info"""

df=pd.read_csv(os.path.join(INFO_PATH, 'station_info.csv'))

K_EDGES=3  #how many edges a node connect with neighbours
REAL_STATIONS = df[df['test'] == 0][['station_id', 'station_name', 'latitude', 'longitude']].reset_index(drop=True) ##real stations for training
TEST_STATIONS = df[df['test'] == 1][['station_id', 'station_name', 'latitude', 'longitude']].reset_index(drop=True) ##test stations for testing


START_TIME=pd.to_datetime("2022-12-10 23:30:00") 
END_TIME=pd.to_datetime("2025-04-08 12:40:00")
START_IDX=0
END_IDX=None

"""MAP Grid"""
GRID_ROW=9 
GRID_COL=9

"""input and output step"""
WINDOW_SIZE=36 ##6 hours input
HORIZON=6 ###1 hour prediction


"""GDC"""
DIFFUSION_METHOD="ppr" #ppr or heat or raw, raw for no diffusion
ALPHA=0.05
t=5.0
k=8
esp=0.1
weight_config={
    0: 1.0,   # real to real edge weight
    1: 3.0,   # real to virtual edge weight enhance the diffsion
    2: 0.3    # Suppressing virtual-virtual diffusion
}

"""GNN"""
BATCH_SIZE=32
EPOCHS=200
LR=1e-3
PATIENCE=10 #Early stop
WARMUP_EPOCH=20 #Contrastive loss warmup epochs

"""MSE Loss"""
CL_ALPHA = 10
CL_MAE_THRESHOLD = 2

"""Contrastive learning"""
CL_WEIGHT = 2
MULTI_STEP_WEIGHT=1
AuG_WEIGHT=1
MULTI_DIFFUSION_WEIGHT=1
MOCO_WEIGHT=1
MOCO_DELTA=3
TEMPERATURE=1
MASK_RATIO=0.3
enable_contrastive=True##if don't use contrastive, choose false
enable_multi_step=False
enable_multi_step_moco=False
enable_augmented=False
enable_augmented_moco=True
QUEUE_SIZE = 512
MODEL_NAME="Augmented_moco"

TAG="expertient_results"
