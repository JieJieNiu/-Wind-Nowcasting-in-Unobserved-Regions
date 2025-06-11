__author__ = "Stefan WeiÃŸenberger and Johannes Gasteiger"
__license__ = "MIT"


import torch
import torch.nn as nn
import args
from torch_geometric.nn import GCNConv, BatchNorm


import torch
import torch.nn as nn
import args
from torch_geometric.nn import GCNConv, BatchNorm



class TemporalGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, seq_len=args.WINDOW_SIZE,virtual_node_embedder=None):
        super(TemporalGCN, self).__init__()
        self.seq_len = seq_len
        self.virtual_node_embedder = virtual_node_embedder
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)  

        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, batched_graph_seq):
        device = next(self.parameters()).device
        if self.virtual_node_embedder is not None:
            self.virtual_node_embedder = self.virtual_node_embedder.to(device)
        node_embeddings = []

        for data in batched_graph_seq:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)

            edge_weight = None
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_weight = data.edge_attr.squeeze().to(device) 

            for gcn in self.gcn_layers:
                x = torch.relu(gcn(x, edge_index, edge_weight=edge_weight))
            node_embeddings.append(x)

        x_seq = torch.stack(node_embeddings, dim=0).permute(1, 2, 0)
        x_temporal = self.temporal_pool(x_seq).squeeze(-1)
        out = self.fc_out(x_temporal)
        return out

    def encode_first_layer(self, graph_seq):
        """
        Return the node representation of each graph after the first layer of GCN for comparative learning
        """
        outputs = []
        device = next(self.parameters()).device
        for data in graph_seq:
            data = data.to(device)
            edge_index = data.edge_index
            edge_weight = data.edge_attr.squeeze() if data.edge_attr is not None else None
            x = torch.relu(self.gcn_layers[0](data.x, edge_index, edge_weight=edge_weight))
            outputs.append(x)
        return outputs