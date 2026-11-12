from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph


def _debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[SamplerDiagnostics] {message}", flush=True)


def _tensor_stats(value) -> str:
    if not isinstance(value, torch.Tensor):
        return f"type={type(value).__name__}"

    return (
        f"shape={tuple(value.shape)}, dtype={value.dtype}, "
        f"device={value.device}, contiguous={value.is_contiguous()}"
    )


def _num_nodes(data: Data) -> int:
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)

    if hasattr(data, "x") and data.x is not None:
        return int(data.x.size(0))

    raise ValueError("Cannot infer num_nodes from data.")


def _as_cpu_long_index(
    value,
    name: str,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    if value.device.type != "cpu":
        value = value.detach().cpu()

    if value.dtype != torch.long:
        value = value.long()

    if not value.is_contiguous():
        value = value.contiguous()

    if value.dim() != 1:
        raise ValueError(
            f"{name} must be a 1D index tensor, got shape {tuple(value.shape)}."
        )

    if num_nodes is not None and value.numel() > 0:
        min_value = int(value.min().item())
        max_value = int(value.max().item())
        if min_value < 0 or max_value >= num_nodes:
            raise ValueError(
                f"{name} contains out-of-range indices: "
                f"min={min_value}, max={max_value}, valid=[0, {num_nodes - 1}]."
            )

    return value


def _edge_index_cpu_long(data: Data) -> torch.Tensor:
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("data.edge_index is required for subgraph sampling.")

    edge_index = data.edge_index
    if not isinstance(edge_index, torch.Tensor):
        raise ValueError(
            f"data.edge_index must be a tensor, got {type(edge_index).__name__}."
        )

    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            "data.edge_index must have shape [2, num_edges], "
            f"got {tuple(edge_index.shape)}."
        )

    if edge_index.device.type != "cpu":
        edge_index = edge_index.detach().cpu()

    if edge_index.dtype != torch.long:
        edge_index = edge_index.long()

    if not edge_index.is_contiguous():
        edge_index = edge_index.contiguous()

    return edge_index


def _deterministic_choice(
    values: torch.Tensor,
    max_count: Optional[int],
    seed: int,
    salt: int,
) -> torch.Tensor:
    values = _as_cpu_long_index(values, name="deterministic_choice values")

    if max_count is None or max_count <= 0 or values.numel() <= max_count:
        return values

    generator = torch.Generator()
    generator.manual_seed(int((seed + salt) % (2**63 - 1)))
    perm = torch.randperm(values.numel(), generator=generator)
    chosen = values[perm[:max_count]]
    return torch.sort(chosen.long().contiguous())[0].contiguous()


def _cap_subset_keep_center(
    subset: torch.Tensor,
    center_node: int,
    max_nodes: Optional[int],
    seed: int,
    salt: int,
) -> torch.Tensor:
    subset = _as_cpu_long_index(subset, name="expanded subset")

    if max_nodes is None or max_nodes <= 0 or subset.numel() <= max_nodes:
        return subset

    if max_nodes < 1:
        raise ValueError("max_nodes_per_subgraph must be at least 1.")

    center = torch.tensor([center_node], dtype=torch.long)
    if max_nodes == 1:
        return center

    non_center = subset[subset != center_node]
    sampled = _deterministic_choice(
        non_center,
        max_nodes - 1,
        seed=seed,
        salt=salt,
    )
    capped = torch.cat([center, sampled], dim=0)
    return torch.sort(torch.unique(capped).long().contiguous())[0].contiguous()


def _relabel_or_keep_edges(
    subset: torch.Tensor,
    sub_edge_index: torch.Tensor,
    num_nodes: int,
    center_node: int,
    relabel_nodes: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = torch.tensor([center_node], dtype=torch.long)

    if relabel_nodes:
        relabel_map = torch.full((num_nodes,), -1, dtype=torch.long)
        relabel_map[subset] = torch.arange(subset.numel(), dtype=torch.long)
        sub_edge_index = relabel_map[sub_edge_index].long().contiguous()
        mapping = relabel_map[center].long().contiguous()
    else:
        mapping = center.long().contiguous()

    return sub_edge_index, mapping


def _estimate_feature_memory_mb(data: Data, num_subset_nodes: int) -> float:
    if not hasattr(data, "x") or data.x is None:
        return 0.0

    element_size = data.x.element_size()
    feature_dim = int(data.x.size(1)) if data.x.dim() > 1 else 1
    return (num_subset_nodes * feature_dim * element_size) / (1024.0 * 1024.0)


def _safe_cpu_k_hop_subgraph(
    center_node: int,
    num_hops: int,
    edge_index: torch.Tensor,
    num_nodes: int,
    relabel_nodes: bool,
    seed: int = 42,
    max_nodes_per_subgraph: Optional[int] = None,
    max_edges_per_subgraph: Optional[int] = None,
    neighbor_sample_per_hop: Optional[int] = None,
    debug_diagnostics: bool = False,
    context: str = "",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CPU-only k-hop extraction that scans edge_index without building dense NxN
    adjacency or calling native PyG subgraph kernels.
    """
    center = torch.tensor([center_node], dtype=torch.long)
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    frontier = center
    visited[center] = True

    row = edge_index[0]
    col = edge_index[1]

    for hop in range(num_hops):
        visited_before_hop = visited.clone()
        _debug_print(
            debug_diagnostics,
            (
                f"{context} safe_k_hop before hop={hop + 1}: "
                f"frontier={_tensor_stats(frontier)}"
            ),
        )

        frontier_mask = torch.zeros(num_nodes, dtype=torch.bool)
        frontier_mask[frontier] = True
        incident_edge_mask = frontier_mask[row] | frontier_mask[col]

        if incident_edge_mask.any():
            neighbors = torch.cat(
                [row[incident_edge_mask], col[incident_edge_mask]],
                dim=0,
            )
            neighbors = torch.unique(neighbors).long().contiguous()
        else:
            neighbors = torch.empty(0, dtype=torch.long)

        new_frontier = neighbors[~visited[neighbors]]
        original_new_frontier_count = int(new_frontier.numel())
        new_frontier = _deterministic_choice(
            new_frontier,
            neighbor_sample_per_hop,
            seed=seed,
            salt=(center_node + 1) * 1009 + (hop + 1) * 9176,
        )
        if neighbor_sample_per_hop is not None and neighbor_sample_per_hop > 0:
            visited = visited_before_hop
            visited[new_frontier] = True
            visited[center] = True
        else:
            visited[neighbors] = True
        frontier = new_frontier.long().contiguous()

        _debug_print(
            debug_diagnostics,
            (
                f"{context} safe_k_hop after hop={hop + 1}: "
                f"neighbors={_tensor_stats(neighbors)}, "
                f"original_new_frontier_count={original_new_frontier_count}, "
                f"bounded_new_frontier={_tensor_stats(frontier)}"
            ),
        )

        if frontier.numel() == 0:
            break

    expanded_subset = torch.where(visited)[0].long().contiguous()
    original_expanded_node_count = int(expanded_subset.numel())

    subset = _cap_subset_keep_center(
        expanded_subset,
        center_node=center_node,
        max_nodes=max_nodes_per_subgraph,
        seed=seed,
        salt=(center_node + 1) * 1543 + 29,
    )
    edge_mask = visited[row] & visited[col]
    original_edge_count = int(edge_mask.sum().item())

    capped_mask = torch.zeros(num_nodes, dtype=torch.bool)
    capped_mask[subset] = True
    capped_edge_mask = capped_mask[row] & capped_mask[col]
    capped_edge_positions = torch.where(capped_edge_mask)[0].long().contiguous()
    capped_edge_positions = _deterministic_choice(
        capped_edge_positions,
        max_edges_per_subgraph,
        seed=seed,
        salt=(center_node + 1) * 2003 + 71,
    )

    selected_edge_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    selected_edge_mask[capped_edge_positions] = True
    sub_edge_index = edge_index[:, capped_edge_positions].long().contiguous()
    sub_edge_index, mapping = _relabel_or_keep_edges(
        subset=subset,
        sub_edge_index=sub_edge_index,
        num_nodes=num_nodes,
        center_node=center_node,
        relabel_nodes=relabel_nodes,
    )

    _debug_print(
        debug_diagnostics,
        (
            f"{context} bounded safe_k_hop summary: "
            f"original_expanded_node_count={original_expanded_node_count}, "
            f"capped_node_count={subset.numel()}, "
            f"original_edge_count={original_edge_count}, "
            f"capped_edge_count={sub_edge_index.size(1)}"
        ),
    )

    return subset, sub_edge_index, mapping, selected_edge_mask


def sample_center_aware_subgraph(
    data: Data,
    center_node: int,
    num_hops: int = 2,
    relabel_nodes: bool = True,
    debug_diagnostics: bool = False,
    safe_mode: bool = False,
    seed: int = 42,
    max_nodes_per_subgraph: Optional[int] = None,
    max_edges_per_subgraph: Optional[int] = None,
    neighbor_sample_per_hop: Optional[int] = None,
    context: str = "",
) -> Data:
    """
    Sample a center-aware k-hop subgraph around a target center node.

    Args:
        data: Full graph data object.
        center_node: Global node index of the center node.
        num_hops: Number of hops for k-hop subgraph sampling.
        relabel_nodes: Whether to relabel nodes inside the subgraph.

    Returns:
        subgraph_data: A PyG Data object representing the sampled subgraph.
    """
    num_nodes = _num_nodes(data)
    center_node_tensor = _as_cpu_long_index(
        torch.tensor([center_node], dtype=torch.long),
        name="center_node",
        num_nodes=num_nodes,
    )
    edge_index_full = _edge_index_cpu_long(data)

    _debug_print(
        debug_diagnostics,
        (
            f"{context} before edge_index/subgraph extraction: "
            f"center={center_node}, num_hops={num_hops}, "
            f"safe_mode={safe_mode}, edge_index={_tensor_stats(edge_index_full)}"
        ),
    )

    if safe_mode:
        use_indexed = bool(getattr(data, "indexed_subgraph_sampling", False))
        extractor = _safe_cpu_k_hop_subgraph
        if use_indexed:
            from data.indexed_subgraph_sampler import indexed_k_hop_subgraph
            extractor = indexed_k_hop_subgraph
        subset, edge_index, mapping, edge_mask = extractor(
            center_node=center_node,
            num_hops=num_hops,
            edge_index=edge_index_full,
            num_nodes=num_nodes,
            relabel_nodes=relabel_nodes,
            seed=seed,
            max_nodes_per_subgraph=max_nodes_per_subgraph,
            max_edges_per_subgraph=max_edges_per_subgraph,
            neighbor_sample_per_hop=neighbor_sample_per_hop,
            debug_diagnostics=debug_diagnostics,
            context=context,
        )
    else:
        _debug_print(
            debug_diagnostics,
            f"{context} before torch_geometric.k_hop_subgraph native call",
        )
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=center_node_tensor,
            num_hops=num_hops,
            edge_index=edge_index_full,
            relabel_nodes=relabel_nodes,
            num_nodes=num_nodes,
        )
        _debug_print(
            debug_diagnostics,
            f"{context} after torch_geometric.k_hop_subgraph native call",
        )

    subset = _as_cpu_long_index(subset, name="subset", num_nodes=num_nodes)
    edge_index = edge_index.long().contiguous()
    mapping = _as_cpu_long_index(mapping, name="mapping")

    subgraph_data = Data()

    _debug_print(
        debug_diagnostics,
        (
            f"{context} before feature indexing x[idx]: "
            f"idx={_tensor_stats(subset)}, "
            f"estimated_feature_memory_mb="
            f"{_estimate_feature_memory_mb(data, int(subset.numel())):.2f}"
        ),
    )
    subset_for_x = subset.to(data.x.device) if data.x.device.type != "cpu" else subset
    subgraph_data.x = data.x[subset_for_x]
    _debug_print(
        debug_diagnostics,
        f"{context} after feature indexing x[idx]: x={_tensor_stats(subgraph_data.x)}",
    )
    subgraph_data.edge_index = edge_index

    # Original labels if available.
    if hasattr(data, "y") and data.y is not None:
        subset_for_y = subset.to(data.y.device) if data.y.device.type != "cpu" else subset
        subgraph_data.y = data.y[subset_for_y]

    # Anomaly labels if available.
    if hasattr(data, "anomaly_label"):
        subset_for_label = (
            subset.to(data.anomaly_label.device)
            if data.anomaly_label.device.type != "cpu"
            else subset
        )
        subgraph_data.anomaly_label = data.anomaly_label[subset_for_label]
        subgraph_data.center_anomaly_label = data.anomaly_label[center_node]

    if hasattr(data, "anomaly_type"):
        subset_for_type = (
            subset.to(data.anomaly_type.device)
            if data.anomaly_type.device.type != "cpu"
            else subset
        )
        subgraph_data.anomaly_type = data.anomaly_type[subset_for_type]
        subgraph_data.center_anomaly_type = data.anomaly_type[center_node]

    # Global node indices in the original graph.
    subgraph_data.node_global_idx = subset

    # Center node global index.
    subgraph_data.center_node_global_idx = torch.tensor(center_node, dtype=torch.long)

    # Center node local index inside the subgraph.
    if relabel_nodes:
        subgraph_data.center_node_idx = mapping[0]
    else:
        # If relabel_nodes=False, local and global ids are not remapped.
        subgraph_data.center_node_idx = torch.tensor(center_node, dtype=torch.long)

    # Metadata.
    subgraph_data.num_hops = num_hops
    if safe_mode and bool(getattr(data, "indexed_subgraph_sampling", False)):
        subgraph_data.edge_global_idx = edge_mask
    else:
        subgraph_data.edge_mask = edge_mask

    _debug_print(
        debug_diagnostics,
        (
            f"{context} after edge_index/subgraph extraction: "
            f"subset={_tensor_stats(subset)}, "
            f"sub_edge_index={_tensor_stats(edge_index)}"
        ),
    )

    return subgraph_data


def sample_multiple_center_aware_subgraphs(
    data: Data,
    center_nodes: torch.Tensor,
    num_hops: int = 2,
    relabel_nodes: bool = True,
    max_samples: Optional[int] = None,
    debug_diagnostics: bool = False,
    safe_mode: bool = False,
    seed: int = 42,
    max_nodes_per_subgraph: Optional[int] = None,
    max_edges_per_subgraph: Optional[int] = None,
    neighbor_sample_per_hop: Optional[int] = None,
    context: str = "",
) -> List[Data]:
    """
    Sample center-aware subgraphs for multiple center nodes.

    Args:
        data: Full graph data object.
        center_nodes: Tensor of global node indices.
        num_hops: Number of hops for k-hop subgraph sampling.
        relabel_nodes: Whether to relabel nodes inside each subgraph.
        max_samples: Maximum number of sampled subgraphs. If None, sample all.

    Returns:
        subgraphs: List of center-aware subgraph Data objects.
    """
    num_nodes = _num_nodes(data)
    center_nodes = _as_cpu_long_index(
        center_nodes,
        name=f"{context} center_nodes",
        num_nodes=num_nodes,
    )

    _debug_print(
        debug_diagnostics,
        (
            f"{context} before center-aware subgraph batch: "
            f"center_nodes={_tensor_stats(center_nodes)}, "
            f"max_samples={max_samples}, safe_mode={safe_mode}"
        ),
    )

    if max_samples is not None:
        center_nodes = center_nodes[:max_samples].contiguous()

    subgraphs = []

    for position, node in enumerate(center_nodes.tolist()):
        _debug_print(
            debug_diagnostics,
            f"{context} before subgraph[{position}] center_node={node}",
        )
        subgraph = sample_center_aware_subgraph(
            data=data,
            center_node=node,
            num_hops=num_hops,
            relabel_nodes=relabel_nodes,
            debug_diagnostics=debug_diagnostics,
            safe_mode=safe_mode,
            seed=seed,
            max_nodes_per_subgraph=max_nodes_per_subgraph,
            max_edges_per_subgraph=max_edges_per_subgraph,
            neighbor_sample_per_hop=neighbor_sample_per_hop,
            context=f"{context} subgraph[{position}]",
        )
        subgraphs.append(subgraph)
        _debug_print(
            debug_diagnostics,
            (
                f"{context} after subgraph[{position}]: "
                f"num_nodes={subgraph.num_nodes}, "
                f"num_edges={subgraph.edge_index.size(1)}"
            ),
        )

    _debug_print(
        debug_diagnostics,
        f"{context} returning subgraph batch: count={len(subgraphs)}",
    )
    return subgraphs


def summarize_subgraph(subgraph_data: Data) -> Dict[str, int]:
    """
    Summarize a sampled center-aware subgraph.

    Args:
        subgraph_data: A center-aware subgraph Data object.

    Returns:
        summary: Dictionary containing subgraph statistics.
    """
    summary = {
        "num_nodes": int(subgraph_data.num_nodes),
        "num_edges": int(subgraph_data.edge_index.size(1)),
        "center_node_global_idx": int(subgraph_data.center_node_global_idx.item()),
        "center_node_idx": int(subgraph_data.center_node_idx.item()),
    }

    if hasattr(subgraph_data, "center_anomaly_label"):
        summary["center_anomaly_label"] = int(
            subgraph_data.center_anomaly_label.item()
        )

    if hasattr(subgraph_data, "center_anomaly_type"):
        summary["center_anomaly_type"] = int(
            subgraph_data.center_anomaly_type.item()
        )

    return summary
