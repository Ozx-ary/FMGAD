from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


# ============================================================
# Matrix Reconstruction Evidence Module
# ============================================================

class MatrixReconstructionEvidence(nn.Module):
    """
    Matrix reconstruction based anomaly evidence module.

    V2-4 initial version.

    This module computes three reconstruction-style evidence scores
    for each center-aware subgraph:

        r_struct:
            Structural reconstruction error.

        r_attr:
            Attribute reconstruction error.

        r_comb:
            Combined reconstruction evidence.

    Output:
        evidence = [r_struct, r_attr, r_comb]
        shape = [num_subgraphs, 3]

    Notes:
        This initial version is intentionally lightweight and does not
        require changing the existing CenterAwareSubgraphEncoder.

        It reconstructs local structure and attributes using simple
        local statistics from each subgraph.
    """

    def __init__(
        self,
        normalize: bool = True,
        structure_weight: float = 1.0,
        attribute_weight: float = 1.0,
        combined_mode: str = "product",
        eps: float = 1e-8,
    ):
        """
        Args:
            normalize:
                Whether to normalize scores into stable ranges.

            structure_weight:
                Weight applied to structural reconstruction evidence.

            attribute_weight:
                Weight applied to attribute reconstruction evidence.

            combined_mode:
                How to combine structural and attribute evidence.
                Supported:
                    - "product"
                    - "mean"
                    - "sum"

            eps:
                Small value for numerical stability.
        """
        super().__init__()

        self.normalize = normalize
        self.structure_weight = structure_weight
        self.attribute_weight = attribute_weight
        self.combined_mode = combined_mode
        self.eps = eps
        self.evidence_dim = 3

        if combined_mode not in ["product", "mean", "sum"]:
            raise ValueError(
                f"Unsupported combined_mode: {combined_mode}. "
                "Supported modes are: product, mean, sum."
            )

    # ========================================================
    # Utilities
    # ========================================================

    def _get_center_idx(self, subgraph: Data) -> int:
        """
        Get center node index as Python int.
        """
        if not hasattr(subgraph, "center_node_idx"):
            raise ValueError("Subgraph must contain center_node_idx.")

        center_node_idx = subgraph.center_node_idx

        if center_node_idx.dim() == 0:
            center_node_idx = center_node_idx.view(1)

        return int(center_node_idx.item())

    def _safe_scalar(
        self,
        value: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create scalar tensor on target device and dtype.
        """
        return torch.tensor(value, device=device, dtype=dtype)

    # ========================================================
    # Structural Reconstruction Evidence
    # ========================================================

    def compute_structure_reconstruction_error(
        self,
        subgraph: Data,
    ) -> torch.Tensor:
        """
        Compute structure reconstruction evidence.

        Initial V2-4 definition:

            1. Build a dense local adjacency matrix A.
            2. Reconstruct A by a simple degree-based expectation:
                   A_hat[i, j] = deg_i * deg_j / sum_deg
            3. Use normalized MSE as reconstruction error.

        Intuition:
            If the local adjacency matrix is poorly explained by the
            degree-based reconstruction, the local structure may be abnormal.
        """
        if not hasattr(subgraph, "edge_index"):
            raise ValueError("Subgraph must contain edge_index.")

        num_nodes = int(subgraph.num_nodes)
        device = subgraph.x.device
        dtype = subgraph.x.dtype

        if num_nodes <= 1:
            return self._safe_scalar(0.0, device=device, dtype=dtype)

        edge_index = subgraph.edge_index.to(device)

        adj = torch.zeros(
            (num_nodes, num_nodes),
            device=device,
            dtype=dtype,
        )

        row = edge_index[0]
        col = edge_index[1]

        valid_mask = (
            (row >= 0)
            & (row < num_nodes)
            & (col >= 0)
            & (col < num_nodes)
        )

        row = row[valid_mask]
        col = col[valid_mask]

        adj[row, col] = 1.0

        # Remove self-loops from reconstruction target.
        eye = torch.eye(num_nodes, device=device, dtype=dtype)
        adj = adj * (1.0 - eye)

        deg = adj.sum(dim=1)

        sum_deg = deg.sum()

        if float(sum_deg.detach().cpu().item()) <= self.eps:
            return self._safe_scalar(0.0, device=device, dtype=dtype)

        expected_adj = torch.outer(deg, deg) / (sum_deg + self.eps)
        expected_adj = expected_adj * (1.0 - eye)
        expected_adj = torch.clamp(expected_adj, min=0.0, max=1.0)

        mse = F.mse_loss(expected_adj, adj, reduction="mean")

        if self.normalize:
            mse = torch.clamp(mse, min=0.0, max=1.0)

        return mse * self.structure_weight

    # ========================================================
    # Attribute Reconstruction Evidence
    # ========================================================

    def compute_attribute_reconstruction_error(
        self,
        subgraph: Data,
    ) -> torch.Tensor:
        """
        Compute attribute reconstruction evidence.

        Initial V2-4 definition:

            x_center_hat = mean feature of non-center context nodes
            r_attr = ||x_center - x_center_hat||_2 / sqrt(feature_dim)

        This is a reconstruction-style version of center attribute
        inconsistency.
        """
        if not hasattr(subgraph, "x"):
            raise ValueError("Subgraph must contain node features x.")

        x = subgraph.x
        device = x.device
        dtype = x.dtype

        num_nodes = int(subgraph.num_nodes)

        if num_nodes <= 1:
            return self._safe_scalar(0.0, device=device, dtype=dtype)

        center_idx = self._get_center_idx(subgraph)

        mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        mask[center_idx] = False

        x_center = x[center_idx]
        x_context = x[mask]

        if x_context.size(0) == 0:
            return self._safe_scalar(0.0, device=device, dtype=dtype)

        x_reconstructed = x_context.mean(dim=0)

        error = torch.norm(x_center - x_reconstructed, p=2)

        if self.normalize:
            feature_dim = x.size(1)
            error = error / torch.sqrt(
                torch.tensor(
                    float(feature_dim),
                    device=device,
                    dtype=dtype,
                )
            )

        return error * self.attribute_weight

    # ========================================================
    # Combined Evidence
    # ========================================================

    def compute_combined_reconstruction_error(
        self,
        r_struct: torch.Tensor,
        r_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine structure and attribute reconstruction evidence.
        """
        if self.combined_mode == "product":
            return r_struct * r_attr

        if self.combined_mode == "mean":
            return 0.5 * (r_struct + r_attr)

        if self.combined_mode == "sum":
            return r_struct + r_attr

        raise RuntimeError(f"Unexpected combined_mode: {self.combined_mode}")

    # ========================================================
    # Forward
    # ========================================================

    def forward_single(self, subgraph: Data) -> torch.Tensor:
        """
        Compute reconstruction evidence for one subgraph.

        Returns:
            Tensor of shape [3]:
                [r_struct, r_attr, r_comb]
        """
        if not hasattr(subgraph, "x"):
            raise ValueError("Subgraph must contain x.")

        if not hasattr(subgraph, "edge_index"):
            raise ValueError("Subgraph must contain edge_index.")

        if not hasattr(subgraph, "center_node_idx"):
            raise ValueError("Subgraph must contain center_node_idx.")

        r_struct = self.compute_structure_reconstruction_error(subgraph)
        r_attr = self.compute_attribute_reconstruction_error(subgraph)
        r_comb = self.compute_combined_reconstruction_error(
            r_struct=r_struct,
            r_attr=r_attr,
        )

        evidence = torch.stack([r_struct, r_attr, r_comb], dim=0)

        return evidence

    def forward(self, subgraphs: List[Data]) -> torch.Tensor:
        """
        Compute reconstruction evidence for a list of subgraphs.

        Args:
            subgraphs:
                List of center-aware PyG Data objects.

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
# Reconstruction Evidence Summary
# ============================================================

def summarize_reconstruction_evidence(
    evidences: torch.Tensor,
    max_samples: int = 5,
) -> Dict[str, object]:
    """
    Summarize reconstruction evidence tensor.

    Args:
        evidences:
            Tensor of shape [num_subgraphs, 3].

        max_samples:
            Maximum number of samples to include.

    Returns:
        summary dict.
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
        r_struct, r_attr, r_comb = evidences_cpu[i].tolist()
        samples.append(
            {
                "index": i,
                "r_struct": float(r_struct),
                "r_attr": float(r_attr),
                "r_comb": float(r_comb),
            }
        )

    summary = {
        "shape": tuple(evidences_cpu.shape),
        "evidence_dim": int(evidences_cpu.size(1)),
        "mean": {
            "r_struct": float(means[0].item()),
            "r_attr": float(means[1].item()),
            "r_comb": float(means[2].item()),
        },
        "std": {
            "r_struct": float(stds[0].item()),
            "r_attr": float(stds[1].item()),
            "r_comb": float(stds[2].item()),
        },
        "samples": samples,
    }

    return summary