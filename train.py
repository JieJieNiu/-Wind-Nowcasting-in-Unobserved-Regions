
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:49:06 2025

@author: jie
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
from model import TemporalGCN
from create_virtual_nodes import generate_virtual_nodes, generate_real_node_features, VirtualNodeEmbedding, build_pyg_data_object, generate_virtual_node_features, generate_edges
from GDC_data import apply_diffusion_to_datalist, TemporalGraphDataset
from geopy.distance import geodesic
import numpy as np
from torch_geometric.data import Batch
import pandas as pd
import random
from CL_loss import (
    multi_step_contrastive_loss,
    generate_augmented_graphs,
    augmentation_contrastive_loss,
    augmentation_contrastive_loss_moco,
    generate_multi_diffusion_graphs,
    ContrastiveQueue, 
    compute_moco_loss_dynamic_delta,
)
from torch.cuda.amp import autocast, GradScaler
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
    Load cached graph data, node information table, and number of virtual nodes
        save_path (str): Path where cache files are saved
        diffusion_method (str): diffusion approach ("ppr" / "heat" / "raw")

    return:
        dataset: PyG TemporalGraphDataset 
        nodes_df: DataFrame，node information
        N_virtual: int，virtual node number
    """
    # graph path
    graph_file = {
        "ppr": "diffused_ppr.pt",
        "heat": "diffused_heat.pt",
        "raw": "graph_seq.pt"
    }.get(diffusion_method.lower())

    if graph_file is None:
        raise ValueError(f"unknow diffusion approach: {diffusion_method}")

    graph_path = os.path.join(save_path, graph_file)
    nodes_path = os.path.join(save_path, "nodes_df.csv")
    nvirt_path = os.path.join(save_path, "N_virtual.pt")

   
    graph_seq = torch.load(graph_path)
    dataset = TemporalGraphDataset(graph_seq)


    nodes_df = pd.read_csv(nodes_path)

   
    if os.path.exists(nvirt_path):
        N_virtual = torch.load(nvirt_path)
    else:
        virtual_mask = ~nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}').fillna(False)
        N_virtual = virtual_mask.sum()

    return dataset, nodes_df, N_virtual

def get_real_node_mask(batch):
    """
    Given a Batched Data object, return a bool mask indicating which nodes are real nodes.
    Assume that the node_type scalar exists in x[:, 5], where real is 0 and virtual is 1.
    """
    return batch.x[:, 5] == 0  # shape: [B*N]

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
        batch_input, batch_target = zip(*batch)  
        # combine time step to Batch
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
    rad = torch.deg2rad(degree_tensor)
    return torch.stack([torch.cos(rad), torch.sin(rad)], dim=-1)


def compute_multistep_loss_with_angle(pred_seq, target_seq, dd_weight=50.0):
    """
    pred_seq: List of [B, N, 3] tensors
    target_seq: List of Batched Data 
    """
    loss_dd, loss_ff, loss_gff, count = 0.0, 0.0, 0.0, 0

    for t, tgt in enumerate(target_seq):

        is_real_mask = get_real_node_mask(tgt)          
        pred_flat = pred_seq[t].reshape(-1, 3)           
        pred = pred_flat[is_real_mask]                     
        true = tgt.y[is_real_mask]                         

        if torch.isnan(true).any():
            continue

        pred_dd_vec = angle_to_vector(pred[:, 0])
        true_dd_vec = angle_to_vector(true[:, 0])
        loss_dd += F.mse_loss(pred_dd_vec, true_dd_vec)
        loss_ff += F.mse_loss(pred[:, 1], true[:, 1])
        loss_gff += F.mse_loss(pred[:, 2], true[:, 2])
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True).to(pred_seq[0].device), 0.0, 0.0, 0.0

    loss_dd /= count
    loss_ff /= count
    loss_gff /= count
    total_loss = dd_weight * loss_dd + loss_ff + loss_gff

    return total_loss, loss_dd, loss_ff, loss_gff


def compute_multistep_mae(pred_seq, target_seq):
    mae_dd, mae_ff, mae_gff, count = 0.0, 0.0, 0.0, 0
    for t, tgt in enumerate(target_seq):
        is_real_mask = get_real_node_mask(tgt)        
        pred = pred_seq[t].reshape(-1, 3)[is_real_mask]  
        true = tgt.y[is_real_mask]                      

        if torch.isnan(true).any():
            continue

        diff = torch.abs(true[:, 0] - pred[:, 0]) % 360
        angular_error = torch.min(diff, 360 - diff)
        mae_dd += angular_error.mean()

        mae_ff += F.l1_loss(pred[:, 1], true[:, 1])
        mae_gff += F.l1_loss(pred[:, 2], true[:, 2])
        count += 1

    if count == 0:
        return torch.tensor(0.0, requires_grad=True).to(pred_seq.device), 0.0, 0.0, 0.0

    mae_dd /= count
    mae_ff /= count
    mae_gff /= count
    return mae_dd, mae_ff, mae_gff




def print_queue_distribution(queue, tag=""):

    embeddings = queue.get_negatives()  # shape: [queue_size, embedding_dim]

    norms = torch.norm(embeddings, dim=1)  
    mean_vector = torch.mean(embeddings, dim=0)
    std_vector = torch.std(embeddings, dim=0)

    print(f"\ MoCo Queue Distribution [{tag}]")
    print(f" Embedding shape        : {embeddings.shape}")
    print(f" Norms (mean ± std)     : {norms.mean():.4f} ± {norms.std():.4f}")
    print(f" Mean vector L2 norm    : {mean_vector.norm():.4f}")
    print(f" Std  vector L2 norm    : {std_vector.norm():.4f}")
    


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

    N_nodes = nodes_df.shape[0]
    model = TemporalGCN(
        in_channels=dataset.graphs[0].x.size(-1),
        hidden_channels=64,
        out_channels=3,
        num_layers=3,
        seq_len=input_len,
        pred_len=pred_len,
        virtual_node_embedder=embedding_module,
        num_nodes=N_nodes
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()
    os.makedirs(save_path, exist_ok=True)
    best_loss = float('inf')
    best_dd = float('inf') 
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard'))
    prev_mae = 100.0

    if resume_path is not None and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1  
        print(f" Resumed training from: {resume_path}, starting at epoch {start_epoch}")
    else:
        start_epoch = 0

    # if resume_path is not None and os.path.isfile(resume_path):
    #     model.load_state_dict(torch.load(resume_path, map_location=device))
    #     print(f" Resumed training from: {resume_path}")
    #     start_epoch = 201
    #     print(f"✅ Resumed training from: {resume_path}, starting at epoch {start_epoch}")
    # else:
    #      start_epoch = 0



    is_real_mask = torch.tensor([len(str(sid)) == 4 for sid in nodes_df['station_id']]).to(device)
    is_virtual_mask = ~is_real_mask
    virtual_to_real_map = get_virtual_to_real_map(nodes_df)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=temporal_collate_fn(device),
        num_workers=1,
        pin_memory=True
    )
    #total_batches = len(loader)
    moco_queue = ContrastiveQueue(embedding_dim=64, queue_size=args.QUEUE_SIZE, device=device)


    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        model.train()
        alpha = args.CL_ALPHA
        args.CURRENT_EPOCH = epoch
        mae_threshold = args.CL_MAE_THRESHOLD
        warmup_epochs =args.WARMUP_EPOCH
        # 1： warm-up 
        warmup_factor = min(1.0, epoch / warmup_epochs)
        if epoch == 0:
            mae_factor = 0.0
        else:
            mae_factor = 1.0 / (1 + math.exp(-alpha * (prev_mae - mae_threshold)))

        lambda_t = args.CL_WEIGHT * warmup_factor * mae_factor
        
        epoch_loss = 0
        epoch_mae = 0
        count = 0
        #contrastive_batch_indices = set(random.sample(range(total_batches), 10))
        for batch_idx, (input_seq, target_seq) in enumerate(loader):
            if any(torch.isnan(g.x).any() for g in input_seq) or any(torch.isnan(g.y).any() for g in target_seq):
                continue

            optimizer.zero_grad()

            with autocast():
            #with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                out = model(input_seq)
                if batch_idx == 0 and epoch % 10 == 0:
                    for t in range(len(out)):
                        print(f"t+{t+1} prediction: mean={out[t].mean():.4f}, std={out[t].std():.4f}")
                        print(f"[debug] step t+{t+1} output min/max: {out[t].min():.2f}, {out[t].max():.2f}")
                multi_step_loss, loss_dd, loss_ff, loss_gff =compute_multistep_loss_with_angle(out, target_seq, dd_weight=50)

                mae_dd, mae_ff, mae_gff = compute_multistep_mae(out, target_seq)
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

                total_loss = multi_step_loss + lambda_t * contrastive_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()
            epoch_mae += multi_step_mae.item()
            count += 1
            
            if batch_idx % 100 == 0 and batch_idx > 0:
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
        prev_mae = avg_mae
        
        

        writer.add_scalar('Loss/total', avg_loss, epoch)
        writer.add_scalar('Loss/mse', multi_step_loss.item(), epoch)
        writer.add_scalar('MAE/train', avg_mae, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('Lambda/contrastive_weight', lambda_t, epoch)


        writer.add_scalar('Loss/dd_weighted', 50 * loss_dd, epoch)
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
        
        
        if mae_dd < best_dd:
            best_dd = mae_dd
            #patience_counter = 0
            model_filename = f"best_model_epoch{epoch+1}.pt"
            torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            }, os.path.join(model_dir, model_filename))
            print(f"✅ Best model saved as {model_filename}.")

        

        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                   'epoch':epoch,
                   'model_state_dict':model.state_dict(),
                   'optimizer_state_dict':optimizer.state_dict(),
                   }, os.path.join(model_dir, model_filename))
            print("✅ model saved as {model_filename}.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
    writer.close()
    
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
        device='cuda' if torch.cuda.is_available() else 'cpu',
        epochs=args.EPOCHS,
        lr=args.LR,
        batch_size=args.BATCH_SIZE,
        input_len=args.WINDOW_SIZE,
        pred_len=args.HORIZON,
        patience=args.PATIENCE,
        save_path=args.SAVE_PATH,
        resume_path=None # 
    )

