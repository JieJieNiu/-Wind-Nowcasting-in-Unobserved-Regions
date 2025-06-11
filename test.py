#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:41:20 2025

@author: jolie
"""

import os
import torch
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from models import TemporalGCN
from create_virtual_nodes import VirtualNodeEmbedding
from GDC_data import TemporalGraphDataset
from torch_geometric.data import DataLoader
import args
import gc


def get_virtual_indices(test_stations, nodes_df):
    virtual_mask = ~nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}').fillna(False)
    virtual_nodes = nodes_df[virtual_mask].copy()
    coords = virtual_nodes[['latitude', 'longitude']].values

    def find_virtual(lat, lon):
        dists = [geodesic((lat, lon), (lat2, lon2)).kilometers for lat2, lon2 in coords]
        return virtual_nodes.index[np.argmin(dists)]

    virtual_indices = {}
    for _, row in test_stations.iterrows():
        sid = str(row['station_id'])
        lat, lon = row['latitude'], row['longitude']
        virtual_indices[sid] = find_virtual(lat, lon)

    return virtual_indices


def main():
    # 📂 加载数据
    nodes_df = pd.read_csv(os.path.join(args.DATA_SAVE_PATH, "nodes_df.csv"))
    station_info = pd.read_csv(os.path.join(args.INFO_PATH, 'station_info.csv'))
    test_stations = station_info[station_info['test'] == 1]
    virtual_indices = get_virtual_indices(test_stations, nodes_df)
    virtual_ids = list(virtual_indices.values())

    # 🧠 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_virtual = (~nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}')).sum()
    embedding_module = VirtualNodeEmbedding(num_virtual_nodes=N_virtual, embed_dim=3)
    model = TemporalGCN(
        in_channels=35,
        hidden_channels=64,
        out_channels=3,
        num_layers=3,
        seq_len=args.WINDOW_SIZE,
        virtual_node_embedder=embedding_module
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(args.SAVE_PATH, "model_epoch200.pt"), map_location=device))
    model.eval()

    # 🔁 准备图数据
    graph_seq = torch.load(os.path.join(args.DATA_SAVE_PATH, f"diffused_{args.DIFFUSION_METHOD}.pt"))
    dataset = TemporalGraphDataset(graph_seq)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []

    # 🔮 模型推理
    for _, (input_seq, _) in enumerate(loader):
        input_seq = [g.to(device) for g in input_seq]
        with torch.no_grad():
            out = model(input_seq)  # shape: [N_nodes, 3]

        # 为每个 virtual 节点构造 6 步预测
        step_preds = []
        for virtual_id in virtual_ids:
            pred = out[virtual_id].unsqueeze(0).repeat(args.HORIZON, 1).cpu()  # [6, 3]
            step_preds.append(pred.unsqueeze(0))  # [1, 6, 3]

        step_preds = torch.cat(step_preds, dim=0)  # [8, 6, 3]
        all_preds.append(step_preds.unsqueeze(0))  # [1, 8, 6, 3]

        del input_seq, out, step_preds
        torch.cuda.empty_cache()
        gc.collect()

    # 🧪 合并并保存预测
    preds_tensor = torch.cat(all_preds, dim=0)  # [T, 8, 6, 3]
    filename = f"{args.MODEL_NAME}_preds_virtual8_h6.pt"
    save_path = os.path.join(args.SAVE_PATH, filename)
    torch.save(preds_tensor, save_path)
    print(f"✅ 所有预测已保存至 {save_path}")

    # 🔁 保存 virtual ↔ station 映射表
    mapping_df = pd.DataFrame([
        {
            "station_id": sid,
            "latitude": test_stations[test_stations['station_id'] == int(sid)]['latitude'].values[0],
            "longitude": test_stations[test_stations['station_id'] == int(sid)]['longitude'].values[0],
            "virtual_node_index": virtual_indices[sid]
        }
        for sid in virtual_indices
    ])
    mapping_path = os.path.join(args.SAVE_PATH, "virtual_to_station_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)
    print(f"📌 虚拟节点 ↔ 测试站点映射已保存至 {mapping_path}")


if __name__ == "__main__":
    main()
