from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class MLPEncoder(nn.Module):
    """
    Attribute-view encoder.

    This branch only uses node attributes X and does not perform
    graph message passing. It is mainly used to preserve attribute
    anomaly signals that may be weakened by neighborhood aggregation.
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
            raise ValueError("num_layers must be >= 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []

        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))

            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))

            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        for layer_idx, layer in enumerate(self.layers):
            h = layer(h)

            if layer_idx != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h


class GCNBranchEncoder(nn.Module):
    """
    GCN branch for structural or joint view encoding.
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
            raise ValueError("num_layers must be >= 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        convs = []

        if num_layers == 1:
            convs.append(GCNConv(input_dim, output_dim))
        else:
            convs.append(GCNConv(input_dim, hidden_dim))

            for _ in range(num_layers - 2):
                convs.append(GCNConv(hidden_dim, hidden_dim))

            convs.append(GCNConv(hidden_dim, output_dim))

        self.convs = nn.ModuleList(convs)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = x

        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)

            if layer_idx != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h


class MultiViewFusion(nn.Module):
    """
    Fuse structural, attribute and joint representations.

    Supported:
        - concat_mlp
        - mean
    """

    def __init__(
        self,
        view_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        fusion_type: str = "concat_mlp",
        dropout: float = 0.1,
        normalize: bool = False,
        active_views: Optional[Tuple[bool, bool, bool]] = None,
    ):
        super().__init__()

        self.view_dim = view_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.dropout = dropout
        self.normalize = normalize
        self.active_views = active_views or (True, True, True)

        if not any(self.active_views):
            raise ValueError("At least one multi-view branch must be enabled.")

        if fusion_type == "concat_mlp":
            self.fusion_net = nn.Sequential(
                nn.Linear(view_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        elif fusion_type == "mean":
            if output_dim != view_dim:
                raise ValueError(
                    "For mean fusion, output_dim must equal view_dim. "
                    f"Got output_dim={output_dim}, view_dim={view_dim}."
                )

            self.fusion_net = None

        else:
            raise ValueError(
                f"Unsupported fusion_type: {fusion_type}. "
                "Supported: concat_mlp, mean."
            )

    def forward(
        self,
        structural: torch.Tensor,
        attribute: torch.Tensor,
        joint: torch.Tensor,
    ) -> torch.Tensor:
        if self.fusion_type == "concat_mlp":
            fused = torch.cat(
                [structural, attribute, joint],
                dim=-1,
            )
            fused = self.fusion_net(fused)

        elif self.fusion_type == "mean":
            views = [structural, attribute, joint]
            active = [
                view
                for view, enabled in zip(views, self.active_views)
                if enabled
            ]
            fused = sum(active) / float(len(active))

        else:
            raise RuntimeError(f"Invalid fusion_type: {self.fusion_type}")

        if self.normalize:
            fused = F.normalize(fused, p=2, dim=-1)

        return fused


class MultiViewCenterAwareSubgraphEncoder(nn.Module):
    """
    Multi-view center-aware subgraph encoder for V2-8.

    For each center-aware subgraph, this encoder computes:

        structural view:
            h_s = GCN_s(A, X)

        attribute view:
            h_a = MLP_a(X)

        joint view:
            h_j = GCN_j(A, X)

    Then it extracts the center node representation from each view
    and fuses them into one final subgraph representation.

    Output:
        By default:
            fused_embeddings: Tensor[num_subgraphs, output_dim]

        If return_view_info=True:
            fused_embeddings, view_info
    """

    def __init__(
        self,
        input_dim: int,
        structural_hidden_dim: int = 64,
        structural_output_dim: int = 64,
        structural_num_layers: int = 2,
        structural_dropout: float = 0.5,
        attribute_hidden_dim: int = 64,
        attribute_output_dim: int = 64,
        attribute_num_layers: int = 2,
        attribute_dropout: float = 0.5,
        joint_hidden_dim: int = 64,
        joint_output_dim: int = 64,
        joint_num_layers: int = 2,
        joint_dropout: float = 0.5,
        fusion_type: str = "concat_mlp",
        fusion_hidden_dim: int = 128,
        fusion_output_dim: int = 64,
        fusion_dropout: float = 0.1,
        normalize_fused: bool = False,
        pooling: str = "center",
        use_structural_view: bool = True,
        use_attribute_view: bool = True,
        use_joint_view: bool = True,
    ):
        super().__init__()

        if pooling not in ["center", "center_mean"]:
            raise ValueError(
                f"Unsupported pooling: {pooling}. "
                "Supported: center, center_mean."
            )

        if not (
            structural_output_dim
            == attribute_output_dim
            == joint_output_dim
        ):
            raise ValueError(
                "V2-8.1 requires all view output dims to be equal. "
                f"Got structural={structural_output_dim}, "
                f"attribute={attribute_output_dim}, "
                f"joint={joint_output_dim}."
            )

        if not any([use_structural_view, use_attribute_view, use_joint_view]):
            raise ValueError("At least one multi-view branch must be enabled.")

        self.input_dim = input_dim
        self.view_dim = structural_output_dim
        self.output_dim = fusion_output_dim
        self.pooling = pooling
        self.use_structural_view = use_structural_view
        self.use_attribute_view = use_attribute_view
        self.use_joint_view = use_joint_view

        self.structural_encoder = GCNBranchEncoder(
            input_dim=input_dim,
            hidden_dim=structural_hidden_dim,
            output_dim=structural_output_dim,
            num_layers=structural_num_layers,
            dropout=structural_dropout,
        )

        self.attribute_encoder = MLPEncoder(
            input_dim=input_dim,
            hidden_dim=attribute_hidden_dim,
            output_dim=attribute_output_dim,
            num_layers=attribute_num_layers,
            dropout=attribute_dropout,
        )

        self.joint_encoder = GCNBranchEncoder(
            input_dim=input_dim,
            hidden_dim=joint_hidden_dim,
            output_dim=joint_output_dim,
            num_layers=joint_num_layers,
            dropout=joint_dropout,
        )

        self.view_fusion = MultiViewFusion(
            view_dim=self.view_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
            fusion_type=fusion_type,
            dropout=fusion_dropout,
            normalize=normalize_fused,
            active_views=(
                self.use_structural_view,
                self.use_attribute_view,
                self.use_joint_view,
            ),
        )

    def _get_center_idx(self, subgraph: Data) -> int:
        if not hasattr(subgraph, "center_node_idx"):
            raise ValueError("Subgraph must contain center_node_idx.")

        center_node_idx = subgraph.center_node_idx

        if center_node_idx.dim() == 0:
            center_node_idx = center_node_idx.view(1)

        return int(center_node_idx.item())

    def _pool_view(
        self,
        node_embeddings: torch.Tensor,
        center_idx: int,
    ) -> torch.Tensor:
        center_embedding = node_embeddings[center_idx]

        if self.pooling == "center":
            return center_embedding

        if self.pooling == "center_mean":
            mean_embedding = node_embeddings.mean(dim=0)
            return 0.5 * (center_embedding + mean_embedding)

        raise RuntimeError(f"Invalid pooling: {self.pooling}")

    def encode_one_subgraph(
        self,
        subgraph: Data,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not hasattr(subgraph, "x"):
            raise ValueError("Subgraph must contain x.")

        if not hasattr(subgraph, "edge_index"):
            raise ValueError("Subgraph must contain edge_index.")

        x = subgraph.x
        edge_index = subgraph.edge_index
        center_idx = self._get_center_idx(subgraph)

        if self.use_structural_view:
            structural_nodes = self.structural_encoder(x, edge_index)
            structural_center = self._pool_view(structural_nodes, center_idx)
        else:
            structural_center = x.new_zeros(self.view_dim)

        if self.use_attribute_view:
            attribute_nodes = self.attribute_encoder(x)
            attribute_center = self._pool_view(attribute_nodes, center_idx)
        else:
            attribute_center = x.new_zeros(self.view_dim)

        if self.use_joint_view:
            joint_nodes = self.joint_encoder(x, edge_index)
            joint_center = self._pool_view(joint_nodes, center_idx)
        else:
            joint_center = x.new_zeros(self.view_dim)

        fused_center = self.view_fusion(
            structural_center.unsqueeze(0),
            attribute_center.unsqueeze(0),
            joint_center.unsqueeze(0),
        ).squeeze(0)

        view_info = {
            "structural": structural_center,
            "attribute": attribute_center,
            "joint": joint_center,
            "fused": fused_center,
        }

        return fused_center, view_info

    def forward(
        self,
        subgraphs: Union[List[Data], Tuple[Data, ...]],
        return_view_info: bool = False,
    ):
        if not isinstance(subgraphs, (list, tuple)):
            raise TypeError(
                "MultiViewCenterAwareSubgraphEncoder expects a list/tuple "
                f"of Data objects, got {type(subgraphs)}."
            )

        fused_list = []
        structural_list = []
        attribute_list = []
        joint_list = []

        for subgraph in subgraphs:
            fused, view_info = self.encode_one_subgraph(subgraph)

            fused_list.append(fused)
            structural_list.append(view_info["structural"])
            attribute_list.append(view_info["attribute"])
            joint_list.append(view_info["joint"])

        fused_embeddings = torch.stack(fused_list, dim=0)

        if not return_view_info:
            return fused_embeddings

        view_info_batch = {
            "structural": torch.stack(structural_list, dim=0),
            "attribute": torch.stack(attribute_list, dim=0),
            "joint": torch.stack(joint_list, dim=0),
            "fused": fused_embeddings,
        }

        return fused_embeddings, view_info_batch
