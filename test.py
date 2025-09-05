import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from model import TemporalGCN
from create_virtual_nodes import VirtualNodeEmbedding
from GDC_data import TemporalGraphDataset, custom_collate_fn
import args
import sys
from torch_geometric.data import Batch

def get_virtual_indices(test_stations, nodes_df):
    virtual_mask = ~nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}').fillna(False)
    virtual_nodes = nodes_df[virtual_mask].copy()
    coords = virtual_nodes[['latitude', 'longitude']].values

    def find_virtual(lat, lon):
        dists = [((lat - lat2)**2 + (lon - lon2)**2)**0.5 for lat2, lon2 in coords]
        return virtual_nodes.index[np.argmin(dists)]

    virtual_indices = {}
    for _, row in test_stations.iterrows():
        sid = str(row['station_id'])
        lat, lon = row['latitude'], row['longitude']
        virtual_indices[sid] = find_virtual(lat, lon)

    return virtual_indices


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nodes_df = pd.read_csv(os.path.join(args.DATA_SAVE_PATH, "nodes_df.csv"))
    station_info = pd.read_csv(os.path.join(args.INFO_PATH, 'station_info.csv'))
    test_stations = station_info[station_info['test'] == 1]
    virtual_indices = get_virtual_indices(test_stations, nodes_df)
    virtual_ids = list(virtual_indices.values())
    station_ids = list(virtual_indices.keys())

    virtual_mask = ~nodes_df['station_id'].astype(str).str.fullmatch(r'\d{4}').fillna(False)
    num_virtual_nodes = virtual_mask.sum()
    embedding_module = VirtualNodeEmbedding(num_virtual_nodes=num_virtual_nodes, embed_dim=3)

    model = TemporalGCN(
        in_channels=35,
        hidden_channels=64,
        out_channels=3,
        num_layers=3,
        seq_len=args.WINDOW_SIZE,
        pred_len=args.HORIZON,
        virtual_node_embedder=embedding_module,
        num_nodes=len(nodes_df)  

    checkpoint=torch.load(os.path.join(args.SAVE_PATH, "model_epoch121.pt"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    if args.DIFFUSION_METHOD!="raw":
        graph_seq = torch.load(os.path.join(args.DATA_SAVE_PATH, f"diffused_{args.DIFFUSION_METHOD}.pt"),weights_only=False)
        print("use diffusion")
    else:
        graph_seq = torch.load(os.path.join(args.DATA_SAVE_PATH, "graph_seq.pt"),weights_only=False)
        print("use row")
    print(f" graph_seq lenth: {len(graph_seq)}")

    dataset = TemporalGraphDataset(graph_seq)
    print(f" dataset samples: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=args.BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    all_preds = []

    for input_seqs, _ in tqdm(loader):
        # input_seqs: List[List[Data]] with shape [B, T] → transpose to [T, B]
        input_seq = [Batch.from_data_list(t).to(device) for t in zip(*input_seqs)]

        with torch.no_grad(), autocast():
            out_seq = model(input_seq)  # List of [B, N, 3], length = HORIZON

        
        step_outputs = []
        for step_out in out_seq: 
            step_outputs.append(step_out[:, virtual_ids, :])  # [B, V, 3]
        out_virtual = torch.stack(step_outputs, dim=2)  # [B, V, H, 3]
        
        # for t in range(out_virtual.shape[2]):
        #     step = out_virtual[:, :, t, :]  # [B, V, 3]
        #     flat = step.reshape(-1, 3)      # [B*V, 3]
        #     mean = flat.mean(dim=0)
        #     std = flat.std(dim=0)
        #     min_ = flat.min(dim=0).values
        #     max_ = flat.max(dim=0).values
        #     print(f"[t+{t+1}] mean={mean.tolist()}, std={std.tolist()}, min={min_.tolist()}, max={max_.tolist()}")


        all_preds.append(out_virtual.cpu())

        del input_seq, out_seq, out_virtual
        torch.cuda.empty_cache()
        gc.collect()

    preds_tensor = torch.cat(all_preds, dim=0)  # [T, V, H, 3]
    print(f" final shape: {preds_tensor.shape}")

    pt_path = os.path.join(args.SAVE_PATH, f"{args.MODEL_NAME}_preds_virtual{len(virtual_ids)}_h{args.HORIZON}.pt")
    torch.save(preds_tensor, pt_path)
    print(f" prediction results save to: {pt_path}")

    for idx, sid in enumerate(station_ids):
        npy = preds_tensor[:, idx, :, :].numpy()
        npy_path = os.path.join(args.SAVE_PATH, f"{args.MODEL_NAME}_station_{sid}_pred.npy")
        np.save(npy_path, npy)
        print(f" {sid} prediction results save to: {npy_path}")

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
    print(f" Virtual ↔ testing station map: {mapping_path}")


if __name__ == "__main__":
    main()
