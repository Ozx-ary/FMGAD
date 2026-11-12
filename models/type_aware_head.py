from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TypeAwareEvidenceScoringHead(nn.Module):
    """
    Type-aware evidence scoring head.

    V2-5 module.

    This head follows the designed FMGAD pipeline:

        fused query embedding
        + reconstruction/statistical evidence
        + hierarchical prototype distances
        -> type-aware evidence vector
        -> type-aware scoring function
        -> 4-way type logits

    Class space:
        0 = normal
        1 = structural anomaly
        2 = attribute anomaly
        3 = hybrid anomaly
    """

    def __init__(
        self,
        embedding_dim: int,
        evidence_dim: int = 3,
        prototype_distance_dim: int = 5,
        hidden_dim: int = 64,
        num_classes: int = 4,
        dropout: float = 0.1,
        lambda_type: float = 1.0,
        use_query_embeddings: bool = True,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive, "
                f"got {embedding_dim}."
            )

        if evidence_dim <= 0:
            raise ValueError(
                "evidence_dim must be positive, "
                f"got {evidence_dim}."
            )

        if prototype_distance_dim <= 0:
            raise ValueError(
                "prototype_distance_dim must be positive, "
                f"got {prototype_distance_dim}."
            )

        self.embedding_dim = embedding_dim
        self.evidence_dim = evidence_dim
        self.prototype_distance_dim = prototype_distance_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.lambda_type = lambda_type
        self.use_query_embeddings = use_query_embeddings

        # Ablation: remove query embeddings from the type-aware evidence scoring head.
        self.input_dim = (
            (embedding_dim if use_query_embeddings else 0)
            + evidence_dim
            + prototype_distance_dim
        )

        self.scoring_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def build_type_aware_vector(
        self,
        query_embeddings: Optional[torch.Tensor] = None,
        query_evidence: Optional[torch.Tensor] = None,
        coarse_distances: Optional[torch.Tensor] = None,
        fine_distances_all: Optional[torch.Tensor] = None,
        flat_distances: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build type-aware evidence vector.

        Args:
            query_embeddings:
                Tensor of shape [num_query, embedding_dim].

            query_evidence:
                Tensor of shape [num_query, evidence_dim].
                Usually [r_struct, r_attr, r_comb].

            coarse_distances:
                Tensor of shape [num_query, 2].
                Distances to [q_0, q_ano].

            fine_distances_all:
                Tensor of shape [num_query, 3].
                Distances to [q_S, q_A, q_M].

            flat_distances:
                Optional tensor of shape [num_query, num_classes].
                Used by the w/o hierarchical prototype ablation instead of
                coarse/fine hierarchical distances.

        Returns:
            type_aware_vector:
                Tensor of shape [num_query, input_dim].
        """
        if query_evidence is None:
            raise ValueError("query_evidence is required.")

        if query_evidence.dim() != 2:
            raise ValueError(
                "query_evidence must be 2D, "
                f"got shape {tuple(query_evidence.shape)}."
            )

        if query_evidence.size(1) != self.evidence_dim:
            raise ValueError(
                "query_evidence feature dimension mismatch: "
                f"expected evidence_dim={self.evidence_dim}, "
                f"got {query_evidence.size(1)}."
            )

        if self.use_query_embeddings:
            if query_embeddings is None:
                raise ValueError(
                    "query_embeddings is required when "
                    "use_query_embeddings=True."
                )

            if query_embeddings.dim() != 2:
                raise ValueError(
                    "query_embeddings must be 2D, "
                    f"got shape {tuple(query_embeddings.shape)}."
                )

            if query_embeddings.size(1) != self.embedding_dim:
                raise ValueError(
                    "query_embeddings feature dimension mismatch: "
                    f"expected embedding_dim={self.embedding_dim}, "
                    f"got {query_embeddings.size(1)}."
                )

            num_query = query_embeddings.size(0)
            reference_tensor = query_embeddings
        else:
            # Ablation: remove query embeddings from the type-aware evidence scoring head.
            num_query = query_evidence.size(0)
            reference_tensor = query_evidence

        if query_evidence.size(0) != num_query:
            raise ValueError(
                "query_evidence batch size mismatch: "
                f"{num_query} vs {query_evidence.size(0)}."
            )

        if flat_distances is not None:
            if flat_distances.dim() != 2:
                raise ValueError(
                    "flat_distances must be 2D, "
                    f"got shape {tuple(flat_distances.shape)}."
                )

            if flat_distances.size(0) != num_query:
                raise ValueError(
                    "flat_distances batch size mismatch: "
                    f"{num_query} vs {flat_distances.size(0)}."
                )

            prototype_distances = flat_distances.to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )
            prototype_feature_detail = (
                f"{prototype_distances.size(1)} flat"
            )

        else:
            if coarse_distances is None:
                raise ValueError("coarse_distances is required.")

            if fine_distances_all is None:
                raise ValueError("fine_distances_all is required.")

            if coarse_distances.dim() != 2 or coarse_distances.size(1) != 2:
                raise ValueError(
                    "coarse_distances must have shape [N, 2], "
                    f"got shape {tuple(coarse_distances.shape)}."
                )

            if fine_distances_all.dim() != 2 or fine_distances_all.size(1) != 3:
                raise ValueError(
                    "fine_distances_all must have shape [N, 3], "
                    f"got shape {tuple(fine_distances_all.shape)}."
                )

            if coarse_distances.size(0) != num_query:
                raise ValueError(
                    "coarse_distances batch size mismatch: "
                    f"{num_query} vs {coarse_distances.size(0)}."
                )

            if fine_distances_all.size(0) != num_query:
                raise ValueError(
                    "fine_distances_all batch size mismatch: "
                    f"{num_query} vs {fine_distances_all.size(0)}."
                )

            coarse_distances = coarse_distances.to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )

            fine_distances_all = fine_distances_all.to(
                device=reference_tensor.device,
                dtype=reference_tensor.dtype,
            )

            prototype_distances = torch.cat(
                [
                    coarse_distances,
                    fine_distances_all,
                ],
                dim=-1,
            )
            prototype_feature_detail = (
                f"{coarse_distances.size(1)} coarse + "
                f"{fine_distances_all.size(1)} fine"
            )

        prototype_feature_dim = prototype_distances.size(1)

        if prototype_feature_dim != self.prototype_distance_dim:
            raise ValueError(
                "prototype distance feature dimension mismatch: "
                f"expected prototype_distance_dim="
                f"{self.prototype_distance_dim}, got "
                f"{prototype_feature_dim} "
                f"({prototype_feature_detail})."
            )

        query_evidence = query_evidence.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )

        vector_parts = [
            query_evidence,
            prototype_distances,
        ]

        if self.use_query_embeddings:
            vector_parts.insert(0, query_embeddings)

        type_aware_vector = torch.cat(
            vector_parts,
            dim=-1,
        )

        if type_aware_vector.size(1) != self.input_dim:
            raise ValueError(
                "type_aware_vector feature dimension mismatch: "
                f"expected {self.input_dim}, got {type_aware_vector.size(1)}."
            )

        return type_aware_vector

    def forward(
        self,
        query_embeddings: Optional[torch.Tensor] = None,
        query_evidence: Optional[torch.Tensor] = None,
        coarse_distances: Optional[torch.Tensor] = None,
        fine_distances_all: Optional[torch.Tensor] = None,
        flat_distances: Optional[torch.Tensor] = None,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward type-aware evidence scoring head.
        """
        type_aware_vector = self.build_type_aware_vector(
            query_embeddings=query_embeddings,
            query_evidence=query_evidence,
            coarse_distances=coarse_distances,
            fine_distances_all=fine_distances_all,
            flat_distances=flat_distances,
        )

        type_logits = self.scoring_net(type_aware_vector)
        type_pred = torch.argmax(type_logits, dim=-1)

        output = {
            "type_aware_vector": type_aware_vector,
            "type_logits": type_logits,
            "type_pred": type_pred,
        }

        if query_labels is not None:
            query_labels = query_labels.long().to(type_logits.device)

            type_aware_loss = F.cross_entropy(
                type_logits,
                query_labels,
            )

            type_accuracy = (
                type_pred == query_labels
            ).float().mean()

            output["type_aware_loss"] = type_aware_loss
            output["type_accuracy"] = type_accuracy

        return output
