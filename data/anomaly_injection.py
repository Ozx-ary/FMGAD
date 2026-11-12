import random
from typing import Dict, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def set_injection_seed(seed: int) -> None:
    """
    Set random seed for reproducible anomaly injection.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def split_anomaly_nodes(
    num_nodes: int,
    anomaly_ratio: float,
    type_ratio: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select anomaly nodes and split them into structural, attribute, and hybrid groups.

    Args:
        num_nodes: Number of nodes in the graph.
        anomaly_ratio: Ratio of anomaly nodes.
        type_ratio: Ratio among different anomaly types.
            Example:
                {
                    "structural": 0.4,
                    "attribute": 0.4,
                    "hybrid": 0.2
                }
        seed: Random seed.

    Returns:
        structural_nodes: Tensor of structural anomaly node indices.
        attribute_nodes: Tensor of attribute anomaly node indices.
        hybrid_nodes: Tensor of hybrid anomaly node indices.
    """
    set_injection_seed(seed)

    if type_ratio is None:
        type_ratio = {
            "structural": 0.4,
            "attribute": 0.4,
            "hybrid": 0.2,
        }

    assert 0.0 <= anomaly_ratio < 1.0, "anomaly_ratio should be in [0, 1)."

    ratio_sum = (
        type_ratio.get("structural", 0.0)
        + type_ratio.get("attribute", 0.0)
        + type_ratio.get("hybrid", 0.0)
    )
    assert abs(ratio_sum - 1.0) < 1e-6, "type_ratio values should sum to 1.0."

    num_anomalies = int(num_nodes * anomaly_ratio)

    perm = torch.randperm(num_nodes)
    anomaly_nodes = perm[:num_anomalies]

    num_structural = int(num_anomalies * type_ratio["structural"])
    num_attribute = int(num_anomalies * type_ratio["attribute"])
    num_hybrid = num_anomalies - num_structural - num_attribute

    structural_nodes = anomaly_nodes[:num_structural]
    attribute_nodes = anomaly_nodes[num_structural:num_structural + num_attribute]
    hybrid_nodes = anomaly_nodes[num_structural + num_attribute:]

    return structural_nodes, attribute_nodes, hybrid_nodes


def inject_structural_anomalies(
    edge_index: torch.Tensor,
    num_nodes: int,
    anomaly_nodes: torch.Tensor,
    num_edges_per_node: int = 10,
    seed: int = 42,
) -> torch.Tensor:
    """
    Inject structural anomalies by adding random edges for selected nodes.

    Args:
        edge_index: Original edge index with shape [2, num_edges].
        num_nodes: Number of nodes.
        anomaly_nodes: Selected structural anomaly nodes.
        num_edges_per_node: Number of random edges added for each anomaly node.
        seed: Random seed.

    Returns:
        new_edge_index: Edge index after structural anomaly injection.
    """
    set_injection_seed(seed)

    if anomaly_nodes.numel() == 0:
        return edge_index

    new_edges = []

    for node in anomaly_nodes.tolist():
        targets = torch.randint(
            low=0,
            high=num_nodes,
            size=(num_edges_per_node,),
            dtype=torch.long,
        )

        sources = torch.full_like(targets, fill_value=node)

        edges = torch.stack([sources, targets], dim=0)
        new_edges.append(edges)

    if len(new_edges) == 0:
        return edge_index

    new_edges = torch.cat(new_edges, dim=1)

    new_edge_index = torch.cat([edge_index, new_edges], dim=1)

    # Convert to undirected graph and remove duplicate direction issues.
    new_edge_index = to_undirected(new_edge_index, num_nodes=num_nodes)

    return new_edge_index


def inject_attribute_anomalies(
    x: torch.Tensor,
    anomaly_nodes: torch.Tensor,
    noise_scale: float = 1.0,
    mode: str = "gaussian",
    seed: int = 42,
) -> torch.Tensor:
    """
    Inject attribute anomalies by perturbing node features.

    Args:
        x: Original node feature matrix with shape [num_nodes, num_features].
        anomaly_nodes: Selected attribute anomaly nodes.
        noise_scale: Scale of noise.
        mode: Attribute injection mode.
            - "gaussian": Replace features with Gaussian noise based on global mean/std.
            - "uniform": Replace features with uniform random values.
        seed: Random seed.

    Returns:
        new_x: Node features after attribute anomaly injection.
    """
    set_injection_seed(seed)

    new_x = x.clone()

    if anomaly_nodes.numel() == 0:
        return new_x

    if mode == "gaussian":
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-6)

        noise = mean + noise_scale * torch.randn_like(new_x[anomaly_nodes]) * std
        new_x[anomaly_nodes] = noise

    elif mode == "uniform":
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values

        random_values = torch.rand_like(new_x[anomaly_nodes])
        new_x[anomaly_nodes] = x_min + random_values * (x_max - x_min)

    else:
        raise ValueError(f"Unknown attribute anomaly mode: {mode}")

    return new_x


def inject_multi_type_anomalies(
    data: Data,
    anomaly_ratio: float = 0.05,
    type_ratio: Optional[Dict[str, float]] = None,
    structural_edges_per_node: int = 10,
    attribute_noise_scale: float = 1.0,
    attribute_mode: str = "gaussian",
    seed: int = 42,
) -> Data:
    """
    Inject multi-type anomalies into a PyTorch Geometric Data object.

    Args:
        data: PyTorch Geometric data object.
        anomaly_ratio: Total anomaly node ratio.
        type_ratio: Ratio among structural, attribute, and hybrid anomalies.
        structural_edges_per_node: Number of random edges added for each structural anomaly node.
        attribute_noise_scale: Noise scale for attribute anomaly injection.
        attribute_mode: Attribute anomaly injection mode.
        seed: Random seed.

    Returns:
        new_data: Data object with injected anomalies.
    """
    set_injection_seed(seed)

    new_data = data.clone()

    num_nodes = new_data.num_nodes

    structural_nodes, attribute_nodes, hybrid_nodes = split_anomaly_nodes(
        num_nodes=num_nodes,
        anomaly_ratio=anomaly_ratio,
        type_ratio=type_ratio,
        seed=seed,
    )

    # 1. Inject structural anomalies.
    structural_related_nodes = torch.cat([structural_nodes, hybrid_nodes], dim=0)
    new_edge_index = inject_structural_anomalies(
        edge_index=new_data.edge_index,
        num_nodes=num_nodes,
        anomaly_nodes=structural_related_nodes,
        num_edges_per_node=structural_edges_per_node,
        seed=seed,
    )

    # 2. Inject attribute anomalies.
    attribute_related_nodes = torch.cat([attribute_nodes, hybrid_nodes], dim=0)
    new_x = inject_attribute_anomalies(
        x=new_data.x,
        anomaly_nodes=attribute_related_nodes,
        noise_scale=attribute_noise_scale,
        mode=attribute_mode,
        seed=seed,
    )

    # 3. Build anomaly labels.
    anomaly_label = torch.zeros(num_nodes, dtype=torch.long)
    anomaly_type = torch.zeros(num_nodes, dtype=torch.long)

    anomaly_label[structural_nodes] = 1
    anomaly_label[attribute_nodes] = 1
    anomaly_label[hybrid_nodes] = 1

    anomaly_type[structural_nodes] = 1
    anomaly_type[attribute_nodes] = 2
    anomaly_type[hybrid_nodes] = 3

    new_data.x = new_x
    new_data.edge_index = new_edge_index
    new_data.anomaly_label = anomaly_label
    new_data.anomaly_type = anomaly_type

    # Save index groups for debugging or analysis.
    new_data.structural_anomaly_nodes = structural_nodes
    new_data.attribute_anomaly_nodes = attribute_nodes
    new_data.hybrid_anomaly_nodes = hybrid_nodes

    return new_data


def summarize_anomalies(data: Data) -> Dict[str, int]:
    """
    Summarize anomaly labels and types.

    Args:
        data: Data object after anomaly injection.

    Returns:
        summary: Dictionary containing anomaly statistics.
    """
    if not hasattr(data, "anomaly_label"):
        return {
            "num_nodes": data.num_nodes,
            "num_anomalies": 0,
            "num_normal": data.num_nodes,
            "num_structural": 0,
            "num_attribute": 0,
            "num_hybrid": 0,
        }

    anomaly_label = data.anomaly_label
    anomaly_type = data.anomaly_type

    summary = {
        "num_nodes": int(data.num_nodes),
        "num_anomalies": int((anomaly_label == 1).sum().item()),
        "num_normal": int((anomaly_label == 0).sum().item()),
        "num_structural": int((anomaly_type == 1).sum().item()),
        "num_attribute": int((anomaly_type == 2).sum().item()),
        "num_hybrid": int((anomaly_type == 3).sum().item()),
    }

    return summary