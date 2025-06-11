#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 16:39:46 2025

@author: jolie
"""

import args
import numpy as np
from scipy.linalg import expm
import torch
from torch_geometric.data import Data, InMemoryDataset

def get_component_from_data(data: Data, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes

def get_largest_connected_component_from_data(data: Data) -> np.ndarray:
    remaining_nodes = set(range(data.num_nodes))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component_from_data(data, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))

def apply_lcc_to_datalist(data_list):
    new_list = []
    stats = []

    for t, data in enumerate(data_list):
        lcc = get_largest_connected_component_from_data(data)

        x_new = data.x[lcc]
        y_new = data.y[lcc] if hasattr(data, 'y') else None

        row, col = data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]

        # Remap nodes
        old_to_new = {old: new for new, old in enumerate(lcc)}
        remapped_edges = torch.tensor(
            [[old_to_new[i], old_to_new[j]] for i, j in edges], dtype=torch.long
        ).T

        edge_mask = torch.tensor([i in lcc and j in lcc for i, j in zip(row, col)])

        new_data = Data(
            x=x_new,
            edge_index=remapped_edges,
            edge_attr=data.edge_attr[edge_mask] if data.edge_attr is not None else None,
            y=y_new,
            t_idx=data.t_idx
        )
        new_list.append(new_data)

        
        ratio = len(lcc) / data.num_nodes
        stats.append(ratio)

    print("\n LCC：")
    print(f"  avg remain rate: {np.mean(stats):.4f}")
    print(f"  min remain rate: {np.min(stats):.4f}")
    print(f"  max remain rate: {np.max(stats):.4f}")
    print(f"  total graph     : {len(stats)}")

    return new_list


def get_node_mapper(lcc: np.ndarray) -> dict:
    return {node: idx for idx, node in enumerate(lcc)}

def remap_edges(edges: list, mapper: dict) -> list:
    row = [mapper[e[0]] for e in edges]
    col = [mapper[e[1]] for e in edges]
    return [row, col]

def get_adj_matrix(data: Data) -> np.ndarray:
    num_nodes = data.x.shape[0]
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for i, j in zip(data.edge_index[0], data.edge_index[1]):
        adj_matrix[i, j] = 1.
    return adj_matrix

def get_ppr_matrix(adj_matrix: np.ndarray, alpha: float = args.ALPHA) -> np.ndarray:
    A_tilde = adj_matrix + np.eye(adj_matrix.shape[0])
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return alpha * np.linalg.inv(np.eye(H.shape[0]) - (1 - alpha) * H)

def get_heat_matrix(adj_matrix: np.ndarray, t: float = args.t) -> np.ndarray:
    A_tilde = adj_matrix + np.eye(adj_matrix.shape[0])
    D_tilde = np.diag(1 / np.sqrt(A_tilde.sum(axis=1)))
    H = D_tilde @ A_tilde @ D_tilde
    return expm(-t * (np.eye(H.shape[0]) - H))

def get_top_k_matrix(A: np.ndarray, k: int = args.k) -> np.ndarray:
    num_nodes = A.shape[0]
    row_idx = np.arange(num_nodes)
    A[A.argsort(axis=0)[:num_nodes - k], row_idx] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1
    return A / norm

def get_clipped_matrix(A: np.ndarray, eps: float = args.esp) -> np.ndarray:
    A[A < eps] = 0.
    norm = A.sum(axis=0)
    norm[norm <= 0] = 1
    return A / norm

def apply_diffusion_to_data(data: Data, method: str = args.DIFFUSION_METHOD, alpha: float = args.ALPHA, t: float = args.t, k: int = args.k,
                             weight_config={0: 1.0, 1: 2.0, 2: 0.5}) -> Data:
    adj = get_adj_matrix(data)
    if method == 'ppr':
        A = get_ppr_matrix(adj, alpha=alpha)
    elif method == 'heat':
        A = get_heat_matrix(adj, t=t)
    else:
        raise ValueError(f"Unsupported method: {method}")

    A = get_top_k_matrix(A, k=k)
    edge_index_i, edge_index_j, edge_attr = [], [], []

    # Determine the node type: if the length of station_id is 4, it is a real node, otherwise it is a virtual node
    is_real = torch.tensor([len(str(i)) == 4 for i in range(data.x.shape[0])])

    for i in range(A.shape[0]):
        for j in np.where(A[i] > 0)[0]:
            edge_index_i.append(i)
            edge_index_j.append(j)
            base_weight = A[i, j]
            # Dynamically determine edge type
            if is_real[i] and is_real[j]:
                edge_type = 0  # real-real
            elif not is_real[i] and not is_real[j]:
                edge_type = 2  # virtual-virtual
            else:
                edge_type = 1  # real-virtual
            adjusted_weight = base_weight * weight_config.get(edge_type, 1.0)
            edge_attr.append(adjusted_weight)

    edge_index = torch.tensor([edge_index_i, edge_index_j], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(
        x=data.x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=data.y,
        t_idx=data.t_idx
    )

def apply_diffusion_to_datalist(data_list, method=args.DIFFUSION_METHOD, alpha=args.ALPHA, t=args.t, k=args.k,
                                 weight_config=args.weight_config):
    return [apply_diffusion_to_data(data, method, alpha, t, k, weight_config) for data in data_list]


def generate_multi_diffusion_graphs(graph_seq, alpha=args.ALPHA, t=args.t, k=args.k, weight_config=args.weight_config):
    """
    For the original graph sequence, two diffusion versions are generated: PPR and Heat
    """
    graphs_ppr = apply_diffusion_to_datalist(
        graph_seq,
        method='ppr',
        alpha=alpha,
        k=k,
        weight_config=weight_config
    )

    graphs_heat = apply_diffusion_to_datalist(
        graph_seq,
        method='heat',
        t=t,
        k=k,
        weight_config=weight_config
    )

    return graphs_ppr, graphs_heat



class TemporalGraphDataset(InMemoryDataset):
    def __init__(self, full_graph_list, input_len=args.WINDOW_SIZE, pred_len=args.HORIZON):
        self.graphs = full_graph_list
        self.input_len = input_len
        self.pred_len = pred_len
        self.sequence_len = input_len + pred_len

        # Calculate the number of valid windows
        self.valid_indices = [
            i for i in range(len(self.graphs) - self.sequence_len + 1)
        ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.sequence_len

        sequence = self.graphs[start:end]
        input_graphs = sequence[:self.input_len]     # input
        target_graphs = sequence[self.input_len:]    # prediction

        return input_graphs, target_graphs
