#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 16:07:22 2025

@author: jolie
"""
from geopy.distance import geodesic
from shapely.geometry import Polygon
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import geopandas as gpd
import requests
from io import BytesIO
import torch
from torch_geometric.data import Data
import args
from math import sin, cos, radians
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn


def load_netherlands_geometry():
    """Download Netherlands boundary from Natural Earth"""
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    response = requests.get(url)
    response.raise_for_status()
    
    gdf = gpd.read_file(BytesIO(response.content))
    print("Available columns:", gdf.columns)

    # Choose Netherlands
    if 'ADMIN' in gdf.columns:
        netherlands = gdf[gdf['ADMIN'] == 'Netherlands']
    elif 'admin' in gdf.columns:
        netherlands = gdf[gdf['admin'] == 'Netherlands']
    elif 'name' in gdf.columns:
        netherlands = gdf[gdf['name'] == 'Netherlands']
    else:
        raise ValueError("Can't find country column (e.g., ADMIN, admin, name)")

    if netherlands.empty:
        raise ValueError("Cannot find Netherlands boundary")

    return netherlands.geometry.unary_union


def generate_virtual_nodes(row, col, real_stations, test_stations):
    min_lon, max_lon = 3.3635, 7.2275
    min_lat, max_lat = 50.7504, 53.6316

    netherlands_geometry = load_netherlands_geometry()

    lat_step = (max_lat - min_lat) / row
    lon_step = (max_lon - min_lon) / col

    nodes = []
    matrix = np.zeros((row, col), dtype=int)

    for i in range(row):
        for j in range(col):
            cell_max_lat = max_lat - i * lat_step
            cell_min_lat = cell_max_lat - lat_step
            cell_min_lon = min_lon + j * lon_step
            cell_max_lon = cell_min_lon + lon_step

            grid_cell_polygon = Polygon([
                (cell_min_lon, cell_min_lat),
                (cell_max_lon, cell_min_lat),
                (cell_max_lon, cell_max_lat),
                (cell_min_lon, cell_max_lat)
            ])

            if not netherlands_geometry.intersects(grid_cell_polygon):
                matrix[i, j] = 0
                continue

            in_real = real_stations[
                (real_stations['latitude'] >= cell_min_lat) & (real_stations['latitude'] < cell_max_lat) &
                (real_stations['longitude'] >= cell_min_lon) & (real_stations['longitude'] < cell_max_lon)
            ]

            if not in_real.empty:
                for _, station in in_real.iterrows():
                    nodes.append({
                        "station_id": station["station_id"],
                        "station_name": station["station_name"],
                        "latitude": station["latitude"],
                        "longitude": station["longitude"]
                    })
                    matrix[i, j] = 1
            else:
                # check if any test station falls in the cell
                in_test = test_stations[
                    (test_stations['latitude'] >= cell_min_lat) & (test_stations['latitude'] < cell_max_lat) &
                    (test_stations['longitude'] >= cell_min_lon) & (test_stations['longitude'] < cell_max_lon)
                ]

                if not in_test.empty:
                    test_station = in_test.iloc[0]  # if multiple test nodes, take first
                    nodes.append({
                        "station_id": f"{i}_{j}",
                        "station_name": f"virtual_node_{i}_{j}",
                        "latitude": test_station['latitude'],
                        "longitude": test_station['longitude']
                    })
                    matrix[i, j] = 2
                else:
                    center_lat = (cell_min_lat + cell_max_lat) / 2
                    center_lon = (cell_min_lon + cell_max_lon) / 2
                    nodes.append({
                        "station_id": f"{i}_{j}",
                        "station_name": f"virtual_node_{i}_{j}",
                        "latitude": center_lat,
                        "longitude": center_lon
                    })
                    matrix[i, j] = 2

    nodes_df = pd.DataFrame(nodes)

    # ===  Add calculation of the nearest real node distance ===
    real_df = nodes_df[~nodes_df['station_name'].str.contains('virtual_node')].copy()
    distance_list = []
    for _, row in nodes_df.iterrows():
        if 'virtual_node' in row['station_name']:
            virtual_coord = (row['latitude'], row['longitude'])
            distances = real_df.apply(
                lambda r: geodesic(virtual_coord, (r['latitude'], r['longitude'])).kilometers,
                axis=1
            )
            distance_list.append(distances.min())
        else:
            distance_list.append(0.0)

    nodes_df["DFNRD"] = distance_list
    return nodes_df, matrix



def generate_edges(nodes_df, k=args.K_EDGES):
    """
    - every virtual node connect to nearest real node
    - every node connect with k neibours
    - eedge type：
        0 = real ↔ real
        1 = real ↔ virtual
        2 = virtual ↔ virtual
    return[(src, tgt, distance, edge_type), ...]
    """
    node_coords = nodes_df[['latitude', 'longitude']].values
    station_ids = nodes_df['station_id'].astype(str)
    is_real = station_ids.str.fullmatch(r'\d{4}').fillna(False).values
    is_virtual = ~is_real

    edges = []

    # virtual → nearest real
    for i, isv in enumerate(is_virtual):
        if not isv:
            continue
        coord = node_coords[i]
        real_indices = np.where(is_real)[0]
        distances = [geodesic(coord, node_coords[j]).kilometers for j in real_indices]
        nearest_idx = real_indices[np.argmin(distances)]
        dist = min(distances)
        edges.append((i, nearest_idx, dist, 1))  # real-virtual

    # k nearest neighbors for all
    for i, coord in enumerate(node_coords):
        distances = []
        for j, other_coord in enumerate(node_coords):
            if i == j:
                continue
            dist = geodesic(coord, other_coord).kilometers
            distances.append((j, dist))
        distances.sort(key=lambda x: x[1])
        for j, d in distances[:k]:
            # classify edge type
            if is_real[i] and is_real[j]:
                edge_type = 0
            elif is_virtual[i] and is_virtual[j]:
                edge_type = 2
            else:
                edge_type = 1
            edges.append((i, j, d, edge_type))

    return edges

def plot_graph(nodes_df, edges, figsize=(12, 10)):
    # create NetworkX graph
    G = nx.Graph()

    # Add location and category information to each node
    pos = {}
    color_map = []
    for idx, row in nodes_df.iterrows():
        pos[idx] = (row['longitude'], row['latitude'])
        if 'virtual_node' in row['station_name']:
            color_map.append('orange')  # virtual node
        else:
            color_map.append('skyblue')  # real node
        G.add_node(idx)

    # add edge
    for src, tgt, dist, _ in edges:
        G.add_edge(src, tgt, weight=dist)

    # plot
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=40, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=0.5)

    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Real Node'),
        Patch(facecolor='orange', label='Virtual Node')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title("Virtual + Real Node Graph")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def generate_real_node_features(data_dir, station_info_path, time_start, time_end, freq="10T"):

    station_info = pd.read_csv(os.path.join(station_info_path, 'station_info.csv'))
    real_station_ids = station_info[station_info['test'] == 0]['station_id'].astype(str).tolist()
    all_timestamps = pd.date_range(start=time_start, end=time_end, freq=freq)

    node_features = []
    selected_ids = []

    print("📥 Processing real station files...")
    for station_id in tqdm(real_station_ids):
        file_path = os.path.join(data_dir, f"{station_id}.csv")
        if not os.path.exists(file_path):
            print(f" Missing file: {station_id}.csv")
            continue

        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time').sort_index()

        feature_cols = [col for col in df.columns if col != 'station_id']
        df = df[feature_cols]
        df = df.reindex(all_timestamps)

        # Interpolation to fill missing values
        df = df.interpolate(method='time', limit_direction='both')
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.fillna(0)  # Fill NaN values ​​with 0

        node_features.append(df.values)
        selected_ids.append(station_id)

    features = np.stack(node_features, axis=1)  # shape: [T, N_real, F]
    return features, selected_ids, all_timestamps

def generate_all_timestamps(start_time, end_time, step_minutes=10):
    """
    Generate a complete timestamp sequence based on the start and end time and time granularity.
    """
    return pd.date_range(start=start_time, end=end_time, freq=f"{step_minutes}T")

def generate_virtual_node_features(nodes_df, timestamps):
    virtual_nodes = nodes_df[~nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}').fillna(False)].copy()
    virtual_coords = virtual_nodes[['latitude', 'longitude']].values
    N_virtual = len(virtual_coords)

    if N_virtual == 0:
        raise ValueError(" No virtual nodes found. Please check your nodes_df.")

    geo_features = []
    for lat, lon in virtual_coords:
        geo = [
            sin(radians(lat)), cos(radians(lat)),
            sin(radians(lon)), cos(radians(lon))
        ]
        geo_features.append(geo)
    geo_features = np.array(geo_features)  # [N_virtual, 4]

    time_encoding = []
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60
        time_encoding.append([hour / 24.0])
    time_encoding = np.array(time_encoding)  # [T, 1]

    virtual_geo_time = np.concatenate([
        np.repeat(time_encoding[:, None, :], N_virtual, axis=1),
        np.repeat(geo_features[None, :, :], len(timestamps), axis=0)
    ], axis=-1)  # [T, N_virtual, 5]

    return virtual_geo_time


def augment_real_features_with_geo_time(real_features, nodes_df, timestamps):
    real_nodes = nodes_df[nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}').fillna(False)].copy()
    T, N_real, _ = real_features.shape
    coords = real_nodes[['latitude', 'longitude']].values

    geo_features = []
    for lat, lon in coords:
        geo = [
            sin(radians(lat)), cos(radians(lat)),
            sin(radians(lon)), cos(radians(lon))
        ]
        geo_features.append(geo)
    geo_features = np.array(geo_features)  # [N_real, 4]

    time_encoding = []
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60
        time_encoding.append([hour / 24.0])
    time_encoding = np.array(time_encoding)  # [T, 1]

    geo_time = np.concatenate([
        np.repeat(time_encoding[:, None, :], N_real, axis=1),
        np.repeat(geo_features[None, :, :], T, axis=0)
    ], axis=-1)  # [T, N_real, 5]

    return geo_time


def build_pyg_data_object(
    real_features,
    virtual_geo_time,
    edge_index,
    edge_attr,
    nodes_df,
    timestamps,
    embedding_module: nn.Module,
    all_real_features):
    T, N_real, F_real = real_features.shape
    T, N_virtual, _ = virtual_geo_time.shape

    if N_virtual == 0:
        raise ValueError(" No virtual nodes found. Please check your nodes_df.")

    nodes_df = nodes_df.copy()
    nodes_df['station_id'] = nodes_df['station_id'].astype(str)
    is_real = nodes_df['station_id'].str.fullmatch(r'\d{4}').fillna(False).values.astype(bool)

    real_idx_map = np.where(is_real)[0]
    virt_idx_map = np.where(~is_real)[0]

    node_type_scalar = torch.zeros((len(nodes_df), 1), dtype=torch.float)
    node_type_scalar[~is_real, 0] = 1

    # === real: geo+time + node_type + real_features（5+1+29） ===
    real_geo_time = augment_real_features_with_geo_time(real_features, nodes_df, timestamps)  # [T, N_real, 5]
    real_full = np.concatenate([
        real_geo_time,
        np.repeat(node_type_scalar[real_idx_map].numpy()[None, :, :], T, axis=0),
        real_features  # [T, N_real, 29]
    ], axis=-1)  # [T, N_real, 35]

    # === virtual: geo+time + node_type + weighted avg of 26 + learned embedding for 3 ===
    all_real_features_np = np.nan_to_num(all_real_features.copy())  # [T, N_real, 29] + nan fix
    weights_matrix = np.zeros((N_virtual, N_real))

    # Get coordinates
    coords = nodes_df[['latitude', 'longitude']].values
    for i, isv in enumerate(~is_real):
        if not isv:
            continue
        v_idx = i
        v_coord = coords[v_idx]
        distances = [geodesic(v_coord, coords[j]).kilometers if is_real[j] else np.inf for j in range(len(coords))]
        sorted_real = np.argsort(distances)[:3]
        dists = np.array([distances[j] for j in sorted_real])
        weights = 1 / (dists + 1e-6)
        weights /= weights.sum()
        for j, w in zip(sorted_real, weights):
            real_pos = np.where(real_idx_map == j)[0][0]  # real_features location
            weights_matrix[v_idx - N_real, real_pos] = w

    #Aggregate 26-d features
    aggregated = []
    for t in range(T):
        #real_subset = np.nan_to_num(all_real_features_np[t, :, :26])
        real_subset = all_real_features_np[t, :, :26]
        weighted_avg = weights_matrix @ real_subset  # [N_virtual, 26]
        aggregated.append(weighted_avg)
    aggregated = np.stack(aggregated, axis=0)  # [T, N_virtual, 26]

    # embedding (3D，for dd, ff, gff)
    virtual_node_indices = torch.arange(N_virtual, device=next(embedding_module.parameters()).device)
    learned_ddffgff = embedding_module(virtual_node_indices)  # [N_virtual, 3]
    learned_ddffgff = learned_ddffgff.unsqueeze(0).expand(T, -1, -1).cpu().detach().numpy()

    # concat virtual features：5 + 1 + 26 + 3 = 35
    virtual_full = np.concatenate([
        virtual_geo_time,
        np.repeat(node_type_scalar[virt_idx_map].numpy()[None, :, :], T, axis=0),
        aggregated,
        learned_ddffgff
    ], axis=-1)  # [T, N_virtual, 35]

    # Merge All Features
    F_final = virtual_full.shape[2]
    features_final = np.zeros((T, len(nodes_df), F_final), dtype=np.float32)
    features_final[:, real_idx_map, :] = real_full
    features_final[:, virt_idx_map, :] = virtual_full



    # Build the graph (each step is independent, predicting y at t+1)
    data_list = []
    for t in range(T - 1):
        x_feat = torch.tensor(features_final[t], dtype=torch.float)
        y_target = features_final[t + 1, :, -3:]  # predict t+1 的 dd/ff/gff
        y_tensor = torch.tensor(y_target, dtype=torch.float)  # [N, 3]

        data = Data(
            x=x_feat,
            edge_index=edge_index.clone(),
            edge_attr=edge_attr.clone(),
            y=y_tensor,
            t_idx=t
        )
        data_list.append(data)

    return data_list

###Learnable embedding for virtual node wind target
class VirtualNodeEmbedding(nn.Module):
    def __init__(self, num_virtual_nodes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_virtual_nodes, embed_dim)

    def forward(self, indices):
        return self.embedding(indices)  # shape: [N_virtual, embed_dim]

    

