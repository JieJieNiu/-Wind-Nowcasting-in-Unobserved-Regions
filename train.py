#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:49:06 2025

@author: jolie
"""
import os
import torch
import math
import time
from datetime import datetime, timedelta
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import args
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import TemporalGCN
from create_virtual_nodes import generate_virtual_nodes, generate_real_node_features, VirtualNodeEmbedding, build_pyg_data_object, generate_virtual_node_features, generate_edges
from GDC_data import apply_diffusion_to_datalist, TemporalGraphDataset
from geopy.distance import geodesic
import numpy as np
from torch_geometric.data import Batch
import pandas as pd
from CL_loss import (
    multi_step_contrastive_loss,
    generate_augmented_graphs,
    augmentation_contrastive_loss,
    augmentation_contrastive_loss_moco,
    generate_multi_diffusion_graphs,
    multi_diffusion_contrastive_loss,
    ContrastiveQueue, 
    compute_moco_loss_dynamic_delta,
)
from torch_geometric.data import Batch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

        
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def get_experiment_save_path(base_path, tag="default"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{tag}_{timestamp}"
    full_path = os.path.join(base_path, exp_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def load_training_data(save_path=args.DATA_SAVE_PATH, diffusion_method=args.DIFFUSION_METHOD):
    """
    Load cached graph data, node information table, number of virtual nodes
    Parameters:
    save_path (str): save path of cache file
    diffusion_method (str): diffusion method name ("ppr" / "heat" / "raw")

    return:
    dataset: PyG TemporalGraphDataset object
    nodes_df: DataFrame, containing all node information
    N_virtual: int, number of virtual nodes
    """
    # 图数据路径
    graph_file = {
        "ppr": "diffused_ppr.pt",
        "heat": "diffused_heat.pt",
        "raw": "graph_seq.pt"
    }.get(diffusion_method.lower())

    if graph_file is None:
        raise ValueError(f"Unknown diffusion method: {diffusion_method}")

    graph_path = os.path.join(save_path, graph_file)
    nodes_path = os.path.join(save_path, "nodes_df.csv")
    nvirt_path = os.path.join(save_path, "N_virtual.pt")

    graph_seq = torch.load(graph_path)
    dataset = TemporalGraphDataset(graph_seq)

    nodes_df = pd.read_csv(nodes_path)

    # Load the number of virtual nodes (automatically calculated if not present)
    if os.path.exists(nvirt_path):
        N_virtual = torch.load(nvirt_path)
    else:
        virtual_mask = ~nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}').fillna(False)
        N_virtual = virtual_mask.sum()

    return dataset, nodes_df, N_virtual



def get_virtual_to_real_map(nodes_df):
    coords = nodes_df[['latitude', 'longitude']].values
    is_real = nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}').fillna(False).values
    real_indices = [i for i, r in enumerate(is_real) if r]

    mapping = {}
    for i, isv in enumerate(is_real):
        if isv:
            continue
        dists = [(j, geodesic(coords[i], coords[j]).kilometers) for j in real_indices]
        nearest_idx = min(dists, key=lambda x: x[1])[0]
        mapping[i] = nearest_idx
    return mapping



def temporal_collate_fn(device):
    def collate(batch):
        batch_input, batch_target = zip(*batch)  # List[List[Data]], List[List[Data]]
        
        input_seq = [
            Batch.from_data_list([seq[t] for seq in batch_input]).to(device)
            for t in range(len(batch_input[0]))
        ]
        target_seq = [
            Batch.from_data_list([seq[t] for seq in batch_target]).to(device)
            for t in range(len(batch_target[0]))
        ]
        return input_seq, target_seq
    return collate

def angle_to_vector(degree_tensor):
    if torch.isnan(degree_tensor).any():
        print("[Debug] NaN in angle input")
    if torch.isinf(degree_tensor).any():
        print("[Debug] Inf in angle input")
    rad = torch.deg2rad(degree_tensor)
    return torch.stack([torch.cos(rad), torch.sin(rad)], dim=-1)



def compute_multistep_loss_with_angle(pred_seq, target_seq, is_real_mask, dd_weight=80.0):
    loss_dd, loss_ff, loss_gff, count = 0.0, 0.0, 0.0, 0

    for t, tgt in enumerate(target_seq):
        pred = pred_seq[is_real_mask]
        true = tgt.y.to(is_real_mask.device)[is_real_mask]
        if torch.isnan(true).any():
            continue

        pred_dd_vec = angle_to_vector(pred[:, 0])
        true_dd_vec = angle_to_vector(true[:, 0])
        loss_dd += F.mse_loss(pred_dd_vec, true_dd_vec)

        loss_ff += F.mse_loss(pred[:, 1], true[:, 1])
        loss_gff += F.mse_loss(pred[:, 2], true[:, 2])
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True).to(pred_seq.device), 0.0, 0.0, 0.0

    loss_dd /= count
    loss_ff /= count
    loss_gff /= count
    total_loss = args.DD_WEIGHT * loss_dd + loss_ff + loss_gff

    return total_loss, loss_dd, loss_ff, loss_gff



def compute_multistep_mae(pred_seq, target_seq, is_real_mask):
    mae_dd, mae_ff, mae_gff, count = 0.0, 0.0, 0.0, 0
    for t, tgt in enumerate(target_seq):
        pred = pred_seq[is_real_mask]
        true = tgt.y.to(is_real_mask.device)[is_real_mask]
        if torch.isnan(true).any():
            continue
        # dd: angular MAE
        diff = torch.abs(true[:, 0] - pred[:, 0]) % 360
        angular_error = torch.min(diff, 360 - diff)
        mae_dd += angular_error.mean()
        # ff/gff: absolute error
        mae_ff += F.l1_loss(pred[:, 1], true[:, 1])
        mae_gff += F.l1_loss(pred[:, 2], true[:, 2])
        count += 1
    if count == 0:
        return torch.tensor(0.0, requires_grad=False).to(pred_seq.device)
    mae_dd /= count
    mae_ff /= count
    mae_gff /= count
    return mae_dd, mae_ff, mae_gff



def print_queue_distribution(queue, tag=""):
    """
    Prints the norm, mean, and variance information of the embedding vector in the current MoCo queue for debugging the queue status.
    """
    embeddings = queue.get_negatives()  # shape: [queue_size, embedding_dim]

    norms = torch.norm(embeddings, dim=1)  # L2 norm of each vector
    mean_vector = torch.mean(embeddings, dim=0)
    std_vector = torch.std(embeddings, dim=0)

    print(f"\n MoCo Queue Distribution [{tag}]")
    print(f" Embedding shape        : {embeddings.shape}")
    print(f" Norms (mean ± std)     : {norms.mean():.4f} ± {norms.std():.4f}")
    print(f" Mean vector L2 norm    : {mean_vector.norm():.4f}")
    print(f" Std  vector L2 norm    : {std_vector.norm():.4f}")

def mask_node_features(graph, mask_ratio=0.1):
    """Efficient in-place masking of node features."""
    graph = graph.clone()
    mask = torch.rand(graph.x.shape[0], device=graph.x.device) > mask_ratio
    graph.x = graph.x * mask.unsqueeze(1)
    return graph

def train_loop(dataset, nodes_df, N_virtual,
               device='cuda' if torch.cuda.is_available() else 'cpu',
               epochs=None,
               lr=None,
               batch_size=None,
               input_len=None,
               pred_len=None,
               patience=None,
               save_path=None,
               resume_path=None):  

    
    args.SAVE_PATH = get_experiment_save_path(base_path=args.SAVE_PATH, tag=args.TAG)
    model_name=args. MODEL_NAME
    model_dir = os.path.join(save_path, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_module = VirtualNodeEmbedding(num_virtual_nodes=N_virtual, embed_dim=3)

    
    model = TemporalGCN(
        in_channels=dataset.graphs[0].x.size(-1),
        hidden_channels=64,
        out_channels=3,
        num_layers=3,
        seq_len=input_len,
        virtual_node_embedder=embedding_module
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    os.makedirs(save_path, exist_ok=True)
    best_loss = float('inf')
    best_dd = float('inf') 
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard'))
    prev_mae = 100.0

    if resume_path is not None and os.path.isfile(resume_path):
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f" Resumed training from: {resume_path}")

    is_real_mask = torch.tensor([len(str(sid)) == 4 for sid in nodes_df['station_id']]).to(device)
    is_virtual_mask = ~is_real_mask
    virtual_to_real_map = get_virtual_to_real_map(nodes_df)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=temporal_collate_fn(device),
        num_workers=4,
        pin_memory=True
    )
    moco_queue = ContrastiveQueue(embedding_dim=64, queue_size=args.QUEUE_SIZE, device=device)


    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        alpha = args.CL_ALPHA
        args.CURRENT_EPOCH = epoch
        mae_threshold = args.CL_MAE_THRESHOLD
        warmup_epochs =args.WARMUP_EPOCH
        #  warm-up 
        warmup_factor = min(1.0, epoch / warmup_epochs)
        if epoch == 0:
            mae_factor = 0.0
        else:
            mae_factor = 1.0 / (1 + math.exp(-alpha * (prev_mae - mae_threshold)))

        lambda_t = args.CL_WEIGHT * warmup_factor * mae_factor
        
        epoch_loss = 0
        epoch_mae = 0
        count = 0

        for batch_idx, (input_seq, target_seq) in enumerate(loader):
       

            optimizer.zero_grad()

            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                out = model(input_seq)

                multi_step_loss, loss_dd, loss_ff, loss_gff = compute_multistep_loss_with_angle(
                    out, target_seq, is_real_mask.repeat(batch_size)[:out.shape[0]],
                    dd_weight=args.DD_WEIGHT
                )
                mae_dd, mae_ff, mae_gff = compute_multistep_mae(out, target_seq, is_real_mask.repeat(batch_size)[:out.shape[0]])
                multi_step_mae = (mae_dd + mae_ff + mae_gff) / 3

                

                contrastive_loss = 0.0
                if args.enable_contrastive:
                    if args.enable_multi_step or args.enable_multi_step_moco:
                        z_t_all_list = model.encode_first_layer(input_seq)
                        z_tp1_all_list = model.encode_first_layer(target_seq)
                    
                    if args.enable_multi_step:
                        contrastive_loss += args.MULTI_STEP_WEIGHT * multi_step_contrastive_loss(
                            z_t_all_list, z_tp1_all_list, virtual_to_real_map, is_virtual_mask,
                            temperature=args.TEMPERATURE
                        )
                    if args.enable_augmented:
                        augmented_seq = generate_augmented_graphs(input_seq[0], mask_ratio=args.MASK_RATIO)
                        augmented_seq = augmented_seq.to(device)
                        orignal_graph=input_seq[0].to(device)

                        contrastive_loss += args.AuG_WEIGHT * augmentation_contrastive_loss(
                                        model, orignal_graph, augmented_seq, temperature=args.TEMPERATURE, step_idx=0
                        )
                    if args.enable_diffusion:
                        graphs_ppr, graphs_heat = generate_multi_diffusion_graphs(input_seq)
                        contrastive_loss += args.MULTI_DIFFUSION_WEIGHT * multi_diffusion_contrastive_loss(
                            model, graphs_ppr, graphs_heat, temperature=args.TEMPERATURE
                        )
                    if args.enable_multi_step_moco:
                        virtual_indices = torch.where(is_virtual_mask)[0]
                        contrastive_loss += args.MOCO_WEIGHT * compute_moco_loss_dynamic_delta(
                            z_t_all_list, z_tp1_all_list,
                            virtual_indices, virtual_to_real_map,
                            moco_queue, delta=args.MOCO_DELTA,
                            temperature=args.TEMPERATURE
                        )
                        
                    if args.enable_augmented_moco:
                        augmented_seq = generate_augmented_graphs(input_seq[0], mask_ratio=args.MASK_RATIO)
                        augmented_seq = augmented_seq.to(device)
                        orignal_graph=input_seq[0].to(device)
                            
                        contrastive_loss += args.AuG_WEIGHT * augmentation_contrastive_loss_moco(
                            model=model,
                            original_graph=orignal_graph,
                            augmented_graph=augmented_seq,
                            moco_queue=moco_queue,
                            temperature=args.TEMPERATURE
                        )

                
                #print_queue_distribution(moco_queue, tag=f"step {0}")

                total_loss = multi_step_loss + lambda_t * contrastive_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            epoch_mae += multi_step_mae.item()
            count += 1
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                avg_per_batch = elapsed / (batch_idx + 1)
                remaining = avg_per_batch * (len(loader) - batch_idx - 1)
                print(f" Epoch {epoch+1}/{args.EPOCHS} | Batch {batch_idx+1}/{len(loader)} | "
                      f"Elapsed: {format_time(elapsed)} | ETA: {format_time(remaining)}")
        
        epoch_duration = time.time() - start_time
        print(f" Epoch {epoch+1} completed in {format_time(epoch_duration)}\n")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        if count == 0:
            continue

        avg_loss = epoch_loss / count
        avg_mae = epoch_mae / count

        writer.add_scalar('Loss/total', avg_loss, epoch)
        writer.add_scalar('Loss/mse', multi_step_loss.item(), epoch)
        writer.add_scalar('MAE/train', avg_mae, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('Lambda/contrastive_weight', lambda_t, epoch)


        writer.add_scalar('Loss/dd_weighted', args.DD_WEIGHT * loss_dd, epoch)
        writer.add_scalar('Loss/ff', loss_ff, epoch)
        writer.add_scalar('Loss/gff', loss_gff, epoch)
        writer.add_scalar('MAE/dd', mae_dd, epoch)
        writer.add_scalar('MAE/ff', mae_ff, epoch)
        writer.add_scalar('MAE/gff', mae_gff, epoch)

        if args.enable_contrastive:
            writer.add_scalar('Loss/contrastive', contrastive_loss.item(), epoch)
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | MAE: {avg_mae:.4f} | LR: {current_lr:.6f}")
            print(f"  Contrastive: {contrastive_loss.item():.4f}|  Lambda: {lambda_t:.4f}")
            print(f"  dd_loss: {args.DD_WEIGHT:.1f} * {loss_dd:.4f} = {(args.DD_WEIGHT * loss_dd):.4f}")
            print(f"  ff_loss: {loss_ff:.4f} | gff_loss: {loss_gff:.4f}")
            print(f"  MAE_dd: {mae_dd:.2f} | MAE_ff: {mae_ff:.2f} | MAE_gff: {mae_gff:.2f}")

        else:
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f} | MAE: {avg_mae:.4f} | LR: {current_lr:.6f}")
            print(f"  dd_loss: {args.DD_WEIGHT:.1f} * {loss_dd:.4f} = {(args.DD_WEIGHT * loss_dd):.4f}")
            print(f"  ff_loss: {loss_ff:.4f} | gff_loss: {loss_gff:.4f}")
            print(f"  MAE_dd: {mae_dd:.2f} | MAE_ff: {mae_ff:.2f} | MAE_gff: {mae_gff:.2f}")    
    
        model_filename = f"model_epoch{epoch+1}.pt"
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
             'optimizer_state_dict':optimizer.state_dict(),
              }, os.path.join(model_dir, model_filename))
        # save MAE_dd min model
        if mae_dd < best_dd:
            best_dd = mae_dd
            patience_counter = 0
            model_filename = f"best_model_epoch{epoch+1}.pt"
            torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
             'optimizer_state_dict':optimizer.state_dict(),
              }, os.path.join(model_dir, model_filename))
            print(f" Best model saved as {model_filename}.")
        # else:
        #   patience_counter += 1
        #   if patience_counter >= patience:
        #       print(f" Early stopping at epoch {epoch+1}")
        #       break
        

        
        # if avg_loss < best_loss:
        #    best_loss = avg_loss
        #    patience_counter = 0
        #    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
        #    print(" Best model saved.")
        # else:
        #   patience_counter += 1
        #   if patience_counter >= patience:
        #       print(f" Early stopping at epoch {epoch+1}")
        #       break
        
    writer.close()
    prev_mae = avg_mae
    return model


if __name__ == "__main__":
    dataset, nodes_df, N_virtual = load_training_data(
        save_path=args.DATA_SAVE_PATH,
        diffusion_method=args.DIFFUSION_METHOD
    )

    model = train_loop(
        dataset=dataset,
        nodes_df=nodes_df,
        N_virtual=N_virtual,
        device='cuda',
        epochs=args.EPOCHS,
        lr=args.LR,
        batch_size=args.BATCH_SIZE,
        input_len=args.WINDOW_SIZE,
        pred_len=args.HORIZON,
        patience=args.PATIENCE,
        save_path=args.SAVE_PATH,
        resume_path=None  
    )

