from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ============================================================
# Center-aware Subgraph Encoder
# ============================================================

class CenterAwareSubgraphEncoder(nn.Module):
    """
    Center-aware subgraph encoder.

    This encoder converts each center-aware subgraph into one fixed-size
    vector by:

        1. Applying several GCN layers on the subgraph.
        2. Extracting the representation of the center node.

    Input:
        List[Data]

    Output:
        Tensor of shape [num_subgraphs, output_dim]

    Used by:
        - V1-5 Binary Subgraph ProtoNet
        - V1-7 Multiclass Subgraph ProtoNet
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(input_dim, output_dim))
        else:
            self.convs.append(GCNConv(input_dim, hidden_dim))

            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))

            self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward_single(self, subgraph: Data) -> torch.Tensor:
        """
        Encode a single center-aware subgraph.

        Args:
            subgraph:
                PyG Data object containing:
                    - x
                    - edge_index
                    - center_node_idx

        Returns:
            center_embedding:
                Tensor of shape [output_dim].
        """
        if not hasattr(subgraph, "center_node_idx"):
            raise ValueError("Subgraph must contain center_node_idx.")

        x = subgraph.x
        edge_index = subgraph.edge_index
        center_node_idx = subgraph.center_node_idx

        if center_node_idx.dim() == 0:
            center_node_idx = center_node_idx.view(1)

        h = x

        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)

            if layer_idx != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        center_embedding = h[center_node_idx].squeeze(0)

        return center_embedding

    def forward(self, subgraphs: List[Data]) -> torch.Tensor:
        """
        Encode a list of center-aware subgraphs.

        Args:
            subgraphs:
                List of PyG Data objects.

        Returns:
            embeddings:
                Tensor of shape [num_subgraphs, output_dim].
        """
        if len(subgraphs) == 0:
            raise ValueError("Input subgraphs list is empty.")

        embeddings = []

        for subgraph in subgraphs:
            embedding = self.forward_single(subgraph)
            embeddings.append(embedding)

        embeddings = torch.stack(embeddings, dim=0)

        return embeddings