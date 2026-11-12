from typing import Dict, List

import torch
import torch.nn as nn
from torch_geometric.data import Data


# ============================================================
# Anomaly-aware Fusion Module
# ============================================================

class AnomalyAwareFusion(nn.Module):
    """
    Anomaly-aware fusion module.

    It fuses neural subgraph embeddings z with explicit anomaly evidence r.

    Inputs:
        embeddings:
            Tensor of shape [num_subgraphs, embedding_dim]

        evidences:
            Tensor of shape [num_subgraphs, evidence_dim]

    Output:
        fused_embeddings:
            Tensor of shape [num_subgraphs, output_dim]

    Supported fusion types:
        1. concat:
            z_fused = concat([z, r])
            output_dim = embedding_dim + evidence_dim

        2. concat_mlp:
            z_fused = MLP(concat([z, r]))
            output_dim is configurable, usually equal to embedding_dim

        3. gated:
            gate = sigmoid(MLP(r))
            z_fused = z * gate
            output_dim = embedding_dim
    """

    def __init__(
        self,
        embedding_dim: int,
        evidence_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 64,
        fusion_type: str = "concat_mlp",
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.evidence_dim = evidence_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.dropout = dropout

        if fusion_type == "concat":
            self.actual_output_dim = embedding_dim + evidence_dim
            self.fusion_net = None

        elif fusion_type == "concat_mlp":
            self.actual_output_dim = output_dim
            self.fusion_net = nn.Sequential(
                nn.Linear(embedding_dim + evidence_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        elif fusion_type == "gated":
            if output_dim != embedding_dim:
                raise ValueError(
                    "For gated fusion, output_dim must equal embedding_dim. "
                    f"Got output_dim={output_dim}, embedding_dim={embedding_dim}."
                )

            self.actual_output_dim = embedding_dim
            self.gate_net = nn.Sequential(
                nn.Linear(evidence_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
                nn.Sigmoid(),
            )

        else:
            raise ValueError(
                f"Unsupported fusion_type: {fusion_type}. "
                "Supported types are: concat, concat_mlp, gated."
            )

    def forward(
        self,
        embeddings: torch.Tensor,
        evidences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse embeddings with explicit anomaly evidence.

        Args:
            embeddings:
                Tensor of shape [N, embedding_dim].

            evidences:
                Tensor of shape [N, evidence_dim].

        Returns:
            fused_embeddings:
                Tensor of shape [N, actual_output_dim].
        """
        if embeddings.dim() != 2:
            raise ValueError(
                f"Expected embeddings to be 2D, got shape {tuple(embeddings.shape)}."
            )

        if evidences.dim() != 2:
            raise ValueError(
                f"Expected evidences to be 2D, got shape {tuple(evidences.shape)}."
            )

        if embeddings.size(0) != evidences.size(0):
            raise ValueError(
                "Embeddings and evidences must have the same batch size. "
                f"Got embeddings={tuple(embeddings.shape)}, "
                f"evidences={tuple(evidences.shape)}."
            )

        if embeddings.size(1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {embeddings.size(1)}."
            )

        if evidences.size(1) != self.evidence_dim:
            raise ValueError(
                f"Expected evidence_dim={self.evidence_dim}, "
                f"got {evidences.size(1)}."
            )

        evidences = evidences.to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        if self.fusion_type == "concat":
            fused_embeddings = torch.cat([embeddings, evidences], dim=-1)

        elif self.fusion_type == "concat_mlp":
            fusion_input = torch.cat([embeddings, evidences], dim=-1)
            fused_embeddings = self.fusion_net(fusion_input)

        elif self.fusion_type == "gated":
            gate = self.gate_net(evidences)
            fused_embeddings = embeddings * gate

        else:
            raise RuntimeError(f"Unexpected fusion_type: {self.fusion_type}")

        return fused_embeddings


# ============================================================
# Evidence-aware Encoder Wrapper
# ============================================================

class EvidenceAwareSubgraphEncoder(nn.Module):
    """
    Wrapper encoder for V1-9.

    It wraps three modules:

        1. base_encoder:
            subgraphs -> neural embeddings z

        2. evidence_module:
            subgraphs -> explicit evidence r = [r_s, r_a, r_c]

        3. fusion_module:
            z, r -> z_fused

    This wrapper allows SubgraphProtoNet to stay unchanged.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        evidence_module: nn.Module,
        fusion_module: AnomalyAwareFusion,
    ):
        super().__init__()

        self.base_encoder = base_encoder
        self.evidence_module = evidence_module
        self.fusion_module = fusion_module

    def forward(self, subgraphs: List[Data]) -> torch.Tensor:
        """
        Encode subgraphs into evidence-aware fused embeddings.

        Args:
            subgraphs:
                List of center-aware PyG Data objects.

        Returns:
            fused_embeddings:
                Tensor of shape [num_subgraphs, fusion_output_dim].
        """
        embeddings = self.base_encoder(subgraphs)
        evidences = self.evidence_module(subgraphs)

        evidences = evidences.to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        fused_embeddings = self.fusion_module(
            embeddings=embeddings,
            evidences=evidences,
        )

        return fused_embeddings


# ============================================================
# Fusion Summary
# ============================================================

def summarize_fusion_output(
    embeddings: torch.Tensor,
    evidences: torch.Tensor,
    fused_embeddings: torch.Tensor,
) -> Dict[str, object]:
    """
    Summarize fusion output for dry-run printing.
    """
    summary = {
        "embedding_shape": tuple(embeddings.shape),
        "evidence_shape": tuple(evidences.shape),
        "fused_embedding_shape": tuple(fused_embeddings.shape),
    }

    fused_cpu = fused_embeddings.detach().cpu()

    summary["fused_mean"] = float(fused_cpu.mean().item())
    summary["fused_std"] = float(fused_cpu.std(unbiased=False).item())

    return summary