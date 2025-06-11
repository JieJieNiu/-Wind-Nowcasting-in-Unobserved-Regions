#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 10:49:38 2025

@author: jolie
"""


import os
import torch
from GDC_data import apply_diffusion_to_datalist
from create_virtual_nodes import (
    generate_virtual_nodes,
    generate_real_node_features,
    generate_virtual_node_features,
    generate_edges,
    build_pyg_data_object,
    VirtualNodeEmbedding
)
import args

def main():
    print("Start building the original graph sequence...")

    # Step 1: Construct the original graph sequence (not diffused)
    real_features, real_ids, timestamps = generate_real_node_features(
        data_dir=args.DATA_DIR,
        station_info_path=args.INFO_PATH,
        time_start=args.START_TIME,
        time_end=args.END_TIME
    )

    nodes_df, _ = generate_virtual_nodes(args.GRID_ROW, args.GRID_COL, args.REAL_STATIONS, args.TEST_STATIONS)
    edges = generate_edges(nodes_df, k=args.K_EDGES)

    edge_index = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
    edge_attr = torch.tensor([[e[2], e[3]] for e in edges], dtype=torch.float)

    virtual_geo_time = generate_virtual_node_features(nodes_df, timestamps)
    virtual_mask = ~nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}').fillna(False)
    N_virtual = virtual_mask.sum()
    embedding_module = VirtualNodeEmbedding(num_virtual_nodes=N_virtual, embed_dim=3)

    print("Build PyG Graph Sequence...")
    data_list = build_pyg_data_object(
        real_features=real_features,
        virtual_geo_time=virtual_geo_time,
        edge_index=edge_index,
        edge_attr=edge_attr,
        nodes_df=nodes_df,
        timestamps=timestamps,
        embedding_module=embedding_module,
        all_real_features=real_features
    )
    # Optional:save row graph
    torch.save(data_list, os.path.join(args.DATA_SAVE_PATH, "graph_seq.pt"))
    print(f"row graph sequence save in  {args.SAVE_PATH}/raw_graph_seq.pt")
    
    nodes_df.to_csv(os.path.join(args.DATA_SAVE_PATH, "nodes_df.csv"), index=False)

    torch.save(N_virtual, os.path.join(args.DATA_SAVE_PATH, "N_virtual.pt"))

    # Step 2: apply diffusion
    print(f"starting diffusion（method: {args.DIFFUSION_METHOD}）...")

    diffused_list = apply_diffusion_to_datalist(
        data_list,
        method=args.DIFFUSION_METHOD,
        alpha=args.ALPHA,
        t=args.t,
        k=args.k,
        weight_config=args.weight_config
    )

    # Step 3: Save the diffusion graph sequence
    filename = f"diffused_{args.DIFFUSION_METHOD}.pt"
    full_path = os.path.join(args.DATA_SAVE_PATH, filename)
    torch.save(diffused_list, full_path)

    print(f"Diffusion save to {full_path}")

if __name__ == '__main__':
    main()