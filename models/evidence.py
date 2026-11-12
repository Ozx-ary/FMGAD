from typing import Dict, List

import torch
import torch.nn as nn
from torch_geometric.data import Data


# ============================================================
# Explicit Anomaly Evidence
# ============================================================

class ExplicitAnomalyEvidence(nn.Module):
    """
    Explicit anomaly evidence extractor for center-aware subgraphs.

    V1-8 simple version computes three scalar evidences for each subgraph:

        r_s: structural evidence
            A simple structural abnormality score based on local density.

        r_a: attribute evidence
            Attribute inconsistency between the center node and its local context.

        r_c: combined/contextual evidence
            Interaction between structural and attribute evidence.

    Output:
        evidence = [r_s, r_a, r_c]
        shape = [num_subgraphs, 3]

    Notes:
        This module is intentionally simple in V1-8.
        It only extracts explicit evidence and does not fuse it with
        neural embeddings. Fusion will be implemented in V1-9.
    """

    def __init__(
        self,
        normalize: bool = True,
        eps: float = 1e-8,
    ):
        """
        Args:
            normalize:
                Whether to normalize evidence values into stable ranges.

            eps:
                Small value for numerical stability.
        """
        super().__init__()

        self.normalize = normalize
        self.eps = eps
        self.evidence_dim = 3

    # ========================================================
    # Single-subgraph Evidence
    # ========================================================

    def compute_structural_evidence(self, subgraph: Data) -> torch.Tensor:
        """
        Compute structural evidence r_s.

        Simple V1-8 definition:
            r_s = 1 - local_density

        where:
            local_density = num_edges / max_possible_directed_edges

        Since PyG edge_index usually stores directed edges for undirected graphs,
        we use directed density:

            max_edges = n * (n - 1)

        Intuition:
            Lower local density may indicate structural irregularity.
            Therefore, r_s becomes larger when the local structure is sparse.

        Returns:
            r_s:
                Scalar tensor.
        """
        num_nodes = int(subgraph.num_nodes)
        num_edges = int(subgraph.edge_index.size(1))

        device = subgraph.x.device
        dtype = subgraph.x.dtype

        if num_nodes <= 1:
            return torch.tensor(0.0, device=device, dtype=dtype)

        max_edges = num_nodes * (num_nodes - 1)
        density = float(num_edges) / float(max_edges + self.eps)

        r_s = 1.0 - density

        if self.normalize:
            r_s = max(0.0, min(1.0, r_s))

        return torch.tensor(r_s, device=device, dtype=dtype)

    def compute_attribute_evidence(self, subgraph: Data) -> torch.Tensor:
        """
        Compute attribute evidence r_a.

        Simple V1-8 definition:
            r_a = || x_center - mean(x_context) ||_2

        where x_context is the mean feature vector of all non-center nodes
        in the subgraph. If the subgraph only contains the center node,
        we return 0.

        If normalize=True:
            r_a = r_a / sqrt(feature_dim)

        This makes the value more stable for high-dimensional sparse features.

        Returns:
            r_a:
                Scalar tensor.
        """
        x = subgraph.x
        center_node_idx = subgraph.center_node_idx

        if center_node_idx.dim() == 0:
            center_node_idx = center_node_idx.view(1)

        center_idx = int(center_node_idx.item())
        num_nodes = int(subgraph.num_nodes)

        if num_nodes <= 1:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
        mask[center_idx] = False

        x_center = x[center_idx]
        x_context = x[mask].mean(dim=0)

        r_a = torch.norm(x_center - x_context, p=2)

        if self.normalize:
            feature_dim = x.size(1)
            r_a = r_a / torch.sqrt(
                torch.tensor(float(feature_dim), device=x.device, dtype=x.dtype)
            )

        return r_a

    def compute_combined_evidence(
        self,
        r_s: torch.Tensor,
        r_a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined/contextual evidence r_c.

        Simple V1-8 definition:
            r_c = r_s * r_a

        Intuition:
            r_c becomes large when both structure and attributes show
            abnormality signals.

        Returns:
            r_c:
                Scalar tensor.
        """
        return r_s * r_a

    def forward_single(self, subgraph: Data) -> torch.Tensor:
        """
        Compute evidence vector for one subgraph.

        Args:
            subgraph:
                Center-aware PyG Data object.

        Returns:
            evidence:
                Tensor of shape [3], ordered as:
                    [r_s, r_a, r_c]
        """
        if not hasattr(subgraph, "center_node_idx"):
            raise ValueError("Subgraph must contain center_node_idx.")

        if not hasattr(subgraph, "x"):
            raise ValueError("Subgraph must contain node features x.")

        if not hasattr(subgraph, "edge_index"):
            raise ValueError("Subgraph must contain edge_index.")

        r_s = self.compute_structural_evidence(subgraph)
        r_a = self.compute_attribute_evidence(subgraph)
        r_c = self.compute_combined_evidence(r_s, r_a)

        evidence = torch.stack([r_s, r_a, r_c], dim=0)

        return evidence

    # ========================================================
    # Batch Evidence
    # ========================================================

    def forward(self, subgraphs: List[Data]) -> torch.Tensor:
        """
        Compute evidence vectors for a list of subgraphs.

        Args:
            subgraphs:
                List of center-aware subgraphs.

        Returns:
            evidences:
                Tensor of shape [num_subgraphs, 3].
        """
        if len(subgraphs) == 0:
            raise ValueError("Input subgraphs list is empty.")

        evidences = []

        for subgraph in subgraphs:
            evidence = self.forward_single(subgraph)
            evidences.append(evidence)

        evidences = torch.stack(evidences, dim=0)

        return evidences


# ============================================================
# Evidence Summary
# ============================================================

def summarize_evidence(
    evidences: torch.Tensor,
    max_samples: int = 5,
) -> Dict[str, object]:
    """
    Summarize evidence tensor for dry-run printing.

    Args:
        evidences:
            Tensor of shape [num_subgraphs, 3].

        max_samples:
            Maximum number of sample evidence vectors to print.

    Returns:
        summary:
            Dictionary containing shape, mean values, and sample values.
    """
    if evidences.dim() != 2 or evidences.size(1) != 3:
        raise ValueError(
            "Expected evidences to have shape [num_subgraphs, 3], "
            f"but got {tuple(evidences.shape)}."
        )

    evidences_cpu = evidences.detach().cpu()

    means = evidences_cpu.mean(dim=0)
    stds = evidences_cpu.std(dim=0, unbiased=False)

    num_samples = min(max_samples, evidences_cpu.size(0))

    samples = []

    for i in range(num_samples):
        r_s, r_a, r_c = evidences_cpu[i].tolist()
        samples.append(
            {
                "index": i,
                "r_s": float(r_s),
                "r_a": float(r_a),
                "r_c": float(r_c),
            }
        )

    summary = {
        "shape": tuple(evidences_cpu.shape),
        "evidence_dim": int(evidences_cpu.size(1)),
        "mean": {
            "r_s": float(means[0].item()),
            "r_a": float(means[1].item()),
            "r_c": float(means[2].item()),
        },
        "std": {
            "r_s": float(stds[0].item()),
            "r_a": float(stds[1].item()),
            "r_c": float(stds[2].item()),
        },
        "samples": samples,
    }

    return summary