


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 13:40:13 2025

@author: Jie
"""

import torch
import torch.nn.functional as F
import args
from GDC_data import generate_multi_diffusion_graphs
import torch.nn as nn
import random
import os
import pandas as pd
import torch_geometric


def safe_normalize(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)


def save_embeddings_csv(z1, z2, save_path, tag="contrastive", epoch=0, step=0):
    os.makedirs(save_path, exist_ok=True)
    data = []
    for i in range(z1.size(0)):
        data.append({"id": i, "type": "z1", **{f"dim_{j}": z1[i, j].item() for j in range(z1.size(1))}})
    for i in range(z2.size(0)):
        data.append({"id": i, "type": "z2", **{f"dim_{j}": z2[i, j].item() for j in range(z2.size(1))}})
    df = pd.DataFrame(data)
    file_path = os.path.join(save_path, f"{tag}_epoch{epoch}_step{step}_embeddings.csv")
    df.to_csv(file_path, index=False)


# ======== Multi_step_contrstive loss ========

def multi_step_contrastive_loss(z_t_all_list, z_tp1_all_list, virtual_to_real_map, is_virtual_mask, temperature=args.TEMPERATURE):
    device = z_t_all_list[0].device
    virtual_indices = torch.where(is_virtual_mask)[0]
    if len(virtual_indices) == 0:
        return torch.tensor(0.0, device=device)

    total_loss = 0.0
    count = 0
    max_t = min(len(z_t_all_list), len(z_tp1_all_list)) - 3

    for t in range(max_t):
        z_t_all = z_t_all_list[t]
        z_tp6_all = z_tp1_all_list[t + 3]

        pos_z1 = z_t_all[virtual_indices]
        real_indices = torch.tensor([virtual_to_real_map[v.item()] for v in virtual_indices], device=device)
        pos_z2 = z_tp6_all[real_indices]

        z1 = safe_normalize(pos_z1)
        z2 = safe_normalize(pos_z2)

        representations = torch.cat([z1, z2], dim=0)
        similarity = torch.matmul(representations, representations.T)
        similarity = similarity.clamp(min=-10.0, max=10.0)

        B = z1.size(0)
        labels = torch.arange(B, device=device)
        labels = torch.cat([labels, labels], dim=0)

        mask = torch.eye(2 * B, device=device).bool()
        similarity[mask] = 0.0
        similarity = torch.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)

        logits = similarity / (temperature + 1e-6)
        
        #debug_contrastive(z1, z2, similarity=similarity, logits=logits, prefix="T6")

        #if args.CURRENT_EPOCH % 20 == 0: 
            #save_embeddings_csv(z1, z2, save_path=args.SAVE_PATH, tag='multi', epoch=args.CURRENT_EPOCH, step=0)
        loss = F.cross_entropy(logits, labels)
        if not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss
            count += 1

    return total_loss / count if count > 0 else torch.tensor(0.0, device=device)


# ======== Augmented_contrstive loss ========
def augmentation_contrastive_loss(model, original_seq, augmented_seq, temperature=0.1, step_idx=0):
    """
    高效版本：只对每个 batch 中第一个图做对比学习，降低计算量。
    """
    
    z1 = torch.relu(model.gcn_layers[0](original_seq.x, original_seq.edge_index, edge_weight=original_seq.edge_attr.squeeze()))
    z2 = torch.relu(model.gcn_layers[0](augmented_seq.x, augmented_seq.edge_index, edge_weight=augmented_seq.edge_attr.squeeze()))

    # 归一化
    z1 = safe_normalize(z1)
    z2 = safe_normalize(z2)

    reps = torch.cat([z1, z2], dim=0)  # [2N, D]
    sim = torch.matmul(reps, reps.T).clamp(min=-10.0, max=10.0)  # [2N, 2N]

    N = z1.size(0)
    labels = torch.arange(N, device=z1.device)
    labels = torch.cat([labels, labels], dim=0)

    # 去掉自身对比
    mask = torch.eye(2 * N, device=z1.device).bool()
    sim[mask] = 0.0
    sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)

    logits = sim / (temperature + 1e-6)

    #if args.CURRENT_EPOCH % 20 == 0:
         #save_embeddings_csv(z1, z2, save_path=args.SAVE_PATH, tag='augmentation', epoch=args.CURRENT_EPOCH, step=step_idx)

    loss = F.cross_entropy(logits, labels)
    return loss if not torch.isnan(loss) and not torch.isinf(loss) else torch.tensor(0.0, device=reps.device)




def mask_node_features(data, mask_ratio=args.MASK_RATIO):
    masked_data = data.clone()
    x = masked_data.x.clone()

    num_nodes, num_features = x.shape
    num_mask = int(mask_ratio * num_features)

    for i in range(num_nodes):
        if num_mask >= num_features:
            continue
        mask_idx = torch.randperm(num_features)[:num_mask]
        x[i, mask_idx] = 0

    masked_data.x = x
    return masked_data



def generate_augmented_graphs(graph, mask_ratio=args.MASK_RATIO):
    return mask_node_features(graph, mask_ratio)


# ======== multi_step——MOCOv1 ========

class ContrastiveQueue:
    def __init__(self, embedding_dim, queue_size=args.QUEUE_SIZE, device='cpu'):
        self.queue_size = queue_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.queue = torch.randn(queue_size, embedding_dim, device=device)
        self.ptr = 0

    @torch.no_grad()
    def enqueue(self, embeddings):
        B = embeddings.size(0)
        if B >= self.queue_size:
            self.queue = embeddings[-self.queue_size:].detach()
            self.ptr = 0
        else:
            space = self.queue_size - self.ptr
            if B <= space:
                self.queue[self.ptr:self.ptr+B] = embeddings.detach()
                self.ptr = (self.ptr + B) % self.queue_size
            else:
                self.queue[self.ptr:] = embeddings[:space].detach()
                self.queue[:B-space] = embeddings[space:].detach()
                self.ptr = B - space

    def get_negatives(self):
        return self.queue.clone().detach()

def moco_contrastive_loss(z_q, z_k, queue, temperature=args.TEMPERATURE):
    z_q = safe_normalize(z_q)
    z_k = safe_normalize(z_k)
    negatives = safe_normalize(queue.get_negatives())

    pos_sim = torch.sum(z_q * z_k, dim=1, keepdim=True)
    neg_sim = torch.matmul(z_q, negatives.T)

    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(z_q.size(0), dtype=torch.long, device=z_q.device)

    return F.cross_entropy(logits / temperature, labels)

def compute_moco_loss_dynamic_delta(
    z_list_t, z_list_tp, virtual_indices, virtual_to_real_map, queue,
    delta=6, temperature=args.TEMPERATURE
):
    total_loss = 0.0
    count = 0
    max_t = min(len(z_list_t), len(z_list_tp)) - delta
    device = z_list_t[0].device

    for t in range(max_t):
        z_q = z_list_t[t][virtual_indices]
        real_indices = torch.tensor(
            [virtual_to_real_map[v.item()] for v in virtual_indices],
            device=device
        )
        z_k = z_list_tp[t + delta][real_indices]
        loss = moco_contrastive_loss(z_q, z_k, queue, temperature)
        
        #if args.CURRENT_EPOCH % 20 == 0:
            #save_embeddings_csv(z_q, z_k, save_path=args.SAVE_PATH, tag='multi_step_moco', epoch=args.CURRENT_EPOCH, step=0)
        
        if not torch.isnan(loss) and not torch.isinf(loss):
            total_loss += loss
            count += 1
            queue.enqueue(z_k)

    return total_loss / count if count > 0 else torch.tensor(0.0, device=device)



# ======== augmentation_contrastive_loss——MOCOv1 ========



def augmentation_contrastive_loss_moco(model, original_graph, augmented_graph, moco_queue, temperature=args.TEMPERATURE):

    z_q = torch.relu(model.gcn_layers[0](
        original_graph.x, original_graph.edge_index,
        edge_weight=original_graph.edge_attr.squeeze() if original_graph.edge_attr is not None else None
    ))

    z_k = torch.relu(model.gcn_layers[0](
        augmented_graph.x, augmented_graph.edge_index,
        edge_weight=augmented_graph.edge_attr.squeeze() if augmented_graph.edge_attr is not None else None
    )).detach()

    z_q = safe_normalize(z_q)
    z_k = safe_normalize(z_k)

    if z_q.size(0) == 0 or z_k.size(0) == 0:
        return torch.tensor(0.0, requires_grad=True, device=z_q.device)


    #if args.CURRENT_EPOCH % 20 == 0: 
        #save_embeddings_csv(z_q, z_k, save_path=args.SAVE_PATH, tag="aug_moco", epoch=args.CURRENT_EPOCH, step=0)

    loss = moco_contrastive_loss(z_q, z_k, moco_queue, temperature)
    moco_queue.enqueue(z_k)
    return loss



