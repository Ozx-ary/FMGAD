"""CSR acceleration of the existing deterministic bounded CPU sampler.

Stores original edge positions so directed, duplicate, and self edges retain
exactly the same order as the scan implementation. No dense edge masks are
attached to individual subgraphs.
"""
import weakref
import numpy as np
import torch

_CACHE = None


class EdgeIndexCSR:
    def __init__(self, edges, num_nodes):
        self.ptrs = []
        self.orders = []
        for axis in range(2):
            values = edges[axis].numpy()
            order = np.argsort(values, kind="stable")
            ptr = np.zeros(num_nodes + 1, dtype=np.int64)
            np.cumsum(np.bincount(values, minlength=num_nodes), out=ptr[1:])
            self.orders.append(order)
            self.ptrs.append(ptr)

    def positions(self, nodes, axis):
        ptr, order = self.ptrs[axis], self.orders[axis]
        chunks = [order[ptr[n]:ptr[n + 1]] for n in nodes.tolist()]
        return torch.from_numpy(np.concatenate(chunks)) if chunks else torch.empty(0, dtype=torch.long)


def get_index(edges, num_nodes):
    global _CACHE
    if _CACHE is None or _CACHE[0]() is not edges or _CACHE[1] != edges._version or _CACHE[2] != num_nodes:
        _CACHE = (weakref.ref(edges), edges._version, num_nodes, EdgeIndexCSR(edges, num_nodes))
    return _CACHE[3]


def indexed_k_hop_subgraph(center_node, num_hops, edge_index, num_nodes,
        relabel_nodes, seed=42, max_nodes_per_subgraph=None,
        max_edges_per_subgraph=None, neighbor_sample_per_hop=None, **kwargs):
    from data.subgraph_sampler import _deterministic_choice, _cap_subset_keep_center
    index = get_index(edge_index, num_nodes)
    visited = torch.zeros(num_nodes, dtype=torch.bool)
    frontier = torch.tensor([center_node], dtype=torch.long)
    visited[frontier] = True
    for hop in range(num_hops):
        positions = torch.cat([index.positions(frontier, 0), index.positions(frontier, 1)])
        neighbors = torch.unique(edge_index[:, positions].reshape(-1))
        frontier = neighbors[~visited[neighbors]]
        frontier = _deterministic_choice(frontier, neighbor_sample_per_hop,
            seed, (center_node + 1) * 1009 + (hop + 1) * 9176)
        visited[frontier] = True
        if not frontier.numel():
            break
    subset = _cap_subset_keep_center(torch.where(visited)[0], center_node,
        max_nodes_per_subgraph, seed, (center_node + 1) * 1543 + 29)
    positions = index.positions(subset, 0)
    targets = edge_index[1, positions]
    loc = torch.searchsorted(subset, targets)
    inside = (loc < len(subset)) & (subset[loc.clamp(max=len(subset) - 1)] == targets)
    positions = torch.sort(positions[inside])[0]
    positions = _deterministic_choice(positions, max_edges_per_subgraph,
        seed, (center_node + 1) * 2003 + 71)
    local_edges = edge_index[:, positions]
    if relabel_nodes:
        local_edges = torch.searchsorted(subset, local_edges)
        mapping = torch.searchsorted(subset, torch.tensor([center_node]))
    else:
        mapping = torch.tensor([center_node])
    return subset, local_edges.contiguous(), mapping, positions
