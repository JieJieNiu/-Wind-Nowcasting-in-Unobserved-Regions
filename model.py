#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 09:45:19 2025

@author: Jie
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import args





class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=3, seq_len=args.WINDOW_SIZE, pred_len=args.HORIZON,
                 virtual_node_embedder=None, num_nodes=None):
        super(TemporalGCN, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes               
        self.virtual_node_embedder = virtual_node_embedder
        self.out_channels = out_channels

        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(hidden_channels, pred_len * out_channels)

    def forward(self, batched_graph_seq):
        node_embeddings = []
        for data in batched_graph_seq:
            x = data.x
            edge_index = data.edge_index
            edge_weight = None
            if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.dim() == 1:
                edge_weight = data.edge_attr
            for gcn in self.gcn_layers:
                x = torch.relu(gcn(x, edge_index, edge_weight=edge_weight))
            node_embeddings.append(x)

        x_seq = torch.stack(node_embeddings, dim=0).permute(1, 2, 0)  
        x_temporal = self.temporal_pool(x_seq).squeeze(-1)           

        BN, F = x_temporal.size()
        N = self.num_nodes                                            
        B = BN // N
        x_temporal = x_temporal.view(B, N, F)                         

        out = self.fc_out(x_temporal)                                 
        out = out.view(B, N, self.pred_len, self.out_channels)        
        return [out[:, :, t, :] for t in range(self.pred_len)]

    def encode_first_layer(self, graph_seq):
        outputs = []
        for data in graph_seq:
            edge_weight = data.edge_attr.squeeze().to(data.x.device) if data.edge_attr is not None else None
            x = torch.relu(self.gcn_layers[0](data.x, data.edge_index, edge_weight=edge_weight))
            outputs.append(x)
        return outputs
