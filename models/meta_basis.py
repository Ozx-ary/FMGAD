from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Meta Prototype Basis
# ============================================================

class MetaPrototypeBasis(nn.Module):
    """
    Meta-learned prototype basis module for V2-2.

    It refines empirical anomaly type prototypes using:

        1. Shared basis:
            basis vectors shared across all anomaly types.

        2. Type-specific basis:
            basis vectors owned by each anomaly type.

    Input:
        empirical_type_prototypes:
            Tensor of shape [num_types, embedding_dim]

            For current task:
                index 0 -> structural prototype q_S
                index 1 -> attribute prototype q_A
                index 2 -> hybrid prototype q_M

    Output:
        refined_type_prototypes:
            Tensor of shape [num_types, embedding_dim]

    Formula:

        alpha_t = softmax(q_t B_shared^T)
        q_shared_t = alpha_t B_shared

        beta_t = softmax(q_t B_specific_t^T)
        q_specific_t = beta_t B_specific_t

        q_refined_t =
            lambda_empirical * q_empirical_t
            + lambda_shared * q_shared_t
            + lambda_specific * q_specific_t
    """

    def __init__(
        self,
        embedding_dim: int,
        num_types: int = 3,
        num_shared_basis: int = 4,
        num_specific_basis: int = 2,
        lambda_empirical: float = 1.0,
        lambda_shared: float = 0.5,
        lambda_specific: float = 0.5,
        normalize: bool = True,
    ):
        super().__init__()

        if num_types <= 0:
            raise ValueError("num_types must be positive.")

        if num_shared_basis <= 0:
            raise ValueError("num_shared_basis must be positive.")

        if num_specific_basis <= 0:
            raise ValueError("num_specific_basis must be positive.")

        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.num_shared_basis = num_shared_basis
        self.num_specific_basis = num_specific_basis
        self.lambda_empirical = lambda_empirical
        self.lambda_shared = lambda_shared
        self.lambda_specific = lambda_specific
        self.normalize = normalize

        # Shared basis:
        #   [K_shared, d]
        self.shared_basis = nn.Parameter(
            torch.empty(num_shared_basis, embedding_dim)
        )

        # Type-specific basis:
        #   [T, K_specific, d]
        self.type_specific_basis = nn.Parameter(
            torch.empty(num_types, num_specific_basis, embedding_dim)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize basis vectors.
        """
        nn.init.xavier_uniform_(self.shared_basis)

        for type_idx in range(self.num_types):
            nn.init.xavier_uniform_(self.type_specific_basis[type_idx])

    def forward(
        self,
        empirical_type_prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Refine empirical type prototypes.

        Args:
            empirical_type_prototypes:
                Tensor of shape [num_types, embedding_dim].

        Returns:
            refined_type_prototypes:
                Tensor of shape [num_types, embedding_dim].

            basis_info:
                Dictionary containing attention weights and intermediate
                prototype components.
        """
        if empirical_type_prototypes.dim() != 2:
            raise ValueError(
                "Expected empirical_type_prototypes to be 2D, "
                f"got shape {tuple(empirical_type_prototypes.shape)}."
            )

        if empirical_type_prototypes.size(0) != self.num_types:
            raise ValueError(
                f"Expected num_types={self.num_types}, "
                f"got {empirical_type_prototypes.size(0)}."
            )

        if empirical_type_prototypes.size(1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {empirical_type_prototypes.size(1)}."
            )

        device = empirical_type_prototypes.device
        dtype = empirical_type_prototypes.dtype

        shared_basis = self.shared_basis.to(device=device, dtype=dtype)
        type_specific_basis = self.type_specific_basis.to(
            device=device,
            dtype=dtype,
        )

        # ----------------------------------------------------
        # Shared basis attention
        # ----------------------------------------------------
        # empirical_type_prototypes: [T, d]
        # shared_basis:              [K_shared, d]
        # shared_scores:             [T, K_shared]

        shared_scores = torch.matmul(
            empirical_type_prototypes,
            shared_basis.t(),
        )

        shared_attention = F.softmax(shared_scores, dim=-1)

        # shared_components:
        #   [T, K_shared] @ [K_shared, d] -> [T, d]

        shared_components = torch.matmul(
            shared_attention,
            shared_basis,
        )

        # ----------------------------------------------------
        # Type-specific basis attention
        # ----------------------------------------------------

        specific_attention_list = []
        specific_component_list = []

        for type_idx in range(self.num_types):
            q_t = empirical_type_prototypes[type_idx]  # [d]
            basis_t = type_specific_basis[type_idx]   # [K_specific, d]

            scores_t = torch.matmul(
                basis_t,
                q_t,
            )  # [K_specific]

            attention_t = F.softmax(scores_t, dim=-1)  # [K_specific]

            component_t = torch.matmul(
                attention_t,
                basis_t,
            )  # [d]

            specific_attention_list.append(attention_t)
            specific_component_list.append(component_t)

        specific_attention = torch.stack(
            specific_attention_list,
            dim=0,
        )  # [T, K_specific]

        specific_components = torch.stack(
            specific_component_list,
            dim=0,
        )  # [T, d]

        # ----------------------------------------------------
        # Refined prototypes
        # ----------------------------------------------------

        refined_type_prototypes = (
            self.lambda_empirical * empirical_type_prototypes
            + self.lambda_shared * shared_components
            + self.lambda_specific * specific_components
        )

        if self.normalize:
            refined_type_prototypes = F.normalize(
                refined_type_prototypes,
                p=2,
                dim=-1,
            )

        basis_info = {
            "empirical_type_prototypes": empirical_type_prototypes,
            "shared_basis": shared_basis,
            "type_specific_basis": type_specific_basis,
            "shared_scores": shared_scores,
            "shared_attention": shared_attention,
            "shared_components": shared_components,
            "specific_attention": specific_attention,
            "specific_components": specific_components,
            "refined_type_prototypes": refined_type_prototypes,
        }

        return refined_type_prototypes, basis_info

class ParentDeltaMetaPrototypeBasis(nn.Module):
    """
    Parent-delta meta-learned prototype basis.

    V2-6:
        Generates a parent anomaly prototype q_ano from shared abnormal bases,
        and type-specific deltas from type-specific bases.

        q_ano = Attn(q_ano_emp, B_shared) B_shared
        delta_t = Attn(q_t_emp, B_specific_t) B_specific_t
        q_t = Norm(
            lambda_parent * q_ano
            + lambda_delta * delta_t
            + lambda_empirical * q_t_emp
        )

    This is more faithful to the intended hierarchical design:
        q_S = q_ano + Δq_S
        q_A = q_ano + Δq_A
        q_M = q_ano + Δq_M
    """

    def __init__(
        self,
        embedding_dim: int,
        num_types: int = 3,
        num_shared_basis: int = 4,
        num_specific_basis: int = 2,
        lambda_empirical: float = 1.0,
        lambda_parent: float = 1.0,
        lambda_delta: float = 0.5,
        normalize: bool = True,
    ):
        super().__init__()

        if num_types <= 0:
            raise ValueError("num_types must be positive.")

        if num_shared_basis <= 0:
            raise ValueError("num_shared_basis must be positive.")

        if num_specific_basis <= 0:
            raise ValueError("num_specific_basis must be positive.")

        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.num_shared_basis = num_shared_basis
        self.num_specific_basis = num_specific_basis
        self.lambda_empirical = lambda_empirical
        self.lambda_parent = lambda_parent
        self.lambda_delta = lambda_delta
        self.normalize = normalize
        self.mode = "parent_delta"

        self.shared_basis = nn.Parameter(
            torch.empty(num_shared_basis, embedding_dim)
        )

        self.type_specific_basis = nn.Parameter(
            torch.empty(num_types, num_specific_basis, embedding_dim)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.shared_basis)

        for type_idx in range(self.num_types):
            nn.init.xavier_uniform_(self.type_specific_basis[type_idx])

    def forward(
        self,
        empirical_type_prototypes: torch.Tensor,
    ):
        if empirical_type_prototypes.dim() != 2:
            raise ValueError(
                "Expected empirical_type_prototypes to be 2D, "
                f"got shape {tuple(empirical_type_prototypes.shape)}."
            )

        if empirical_type_prototypes.size(0) != self.num_types:
            raise ValueError(
                f"Expected num_types={self.num_types}, "
                f"got {empirical_type_prototypes.size(0)}."
            )

        if empirical_type_prototypes.size(1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {empirical_type_prototypes.size(1)}."
            )

        device = empirical_type_prototypes.device
        dtype = empirical_type_prototypes.dtype

        shared_basis = self.shared_basis.to(device=device, dtype=dtype)
        type_specific_basis = self.type_specific_basis.to(
            device=device,
            dtype=dtype,
        )

        # ----------------------------------------------------
        # Parent anomaly prototype from shared abnormal bases
        # ----------------------------------------------------
        empirical_anomaly_prototype = empirical_type_prototypes.mean(dim=0)

        shared_scores = torch.matmul(
            empirical_anomaly_prototype,
            shared_basis.t(),
        )  # [K_shared]

        shared_attention = F.softmax(shared_scores, dim=-1)  # [K_shared]

        q_ano = torch.matmul(
            shared_attention,
            shared_basis,
        )  # [d]

        # ----------------------------------------------------
        # Type-specific deltas
        # ----------------------------------------------------
        specific_attention_list = []
        delta_list = []

        for type_idx in range(self.num_types):
            q_t_emp = empirical_type_prototypes[type_idx]        # [d]
            basis_t = type_specific_basis[type_idx]             # [K_specific, d]

            scores_t = torch.matmul(
                basis_t,
                q_t_emp,
            )  # [K_specific]

            attention_t = F.softmax(scores_t, dim=-1)           # [K_specific]

            delta_t = torch.matmul(
                attention_t,
                basis_t,
            )  # [d]

            specific_attention_list.append(attention_t)
            delta_list.append(delta_t)

        specific_attention = torch.stack(
            specific_attention_list,
            dim=0,
        )  # [T, K_specific]

        delta_types = torch.stack(
            delta_list,
            dim=0,
        )  # [T, d]

        # ----------------------------------------------------
        # Parent + delta + empirical residual
        # ----------------------------------------------------
        q_ano_expanded = q_ano.unsqueeze(0).expand_as(delta_types)

        type_prototypes = (
            self.lambda_parent * q_ano_expanded
            + self.lambda_delta * delta_types
            + self.lambda_empirical * empirical_type_prototypes
        )

        if self.normalize:
            q_ano = F.normalize(q_ano, p=2, dim=-1)
            delta_types = F.normalize(delta_types, p=2, dim=-1)
            type_prototypes = F.normalize(type_prototypes, p=2, dim=-1)

        basis_info = {
            "meta_basis_mode": "parent_delta",
            "parent_delta_enabled": torch.tensor(
                1,
                device=device,
                dtype=torch.long,
            ),
            "empirical_type_prototypes": empirical_type_prototypes,
            "empirical_anomaly_prototype": empirical_anomaly_prototype,
            "shared_basis": shared_basis,
            "type_specific_basis": type_specific_basis,
            "shared_scores": shared_scores,
            "shared_attention": shared_attention,
            "specific_attention": specific_attention,
            "q_ano": q_ano,
            "delta_types": delta_types,
            "type_prototypes": type_prototypes,
            "q_S": type_prototypes[0],
            "q_A": type_prototypes[1],
            "q_M": type_prototypes[2],
            "shared_component": q_ano,
            "specific_components": delta_types,
        }

        return type_prototypes, basis_info







# ============================================================
# Summary
# ============================================================

def summarize_meta_basis(
    meta_basis: nn.Module,
) -> Dict[str, object]:
    """
    Summarize meta basis module configuration.
    """
    summary = {
        "class_name": meta_basis.__class__.__name__,
        "embedding_dim": meta_basis.embedding_dim,
        "num_types": meta_basis.num_types,
        "num_shared_basis": meta_basis.num_shared_basis,
        "num_specific_basis": meta_basis.num_specific_basis,
        "normalize": meta_basis.normalize,
    }

    if hasattr(meta_basis, "mode"):
        summary["mode"] = meta_basis.mode
    else:
        summary["mode"] = "refine"

    if hasattr(meta_basis, "lambda_empirical"):
        summary["lambda_empirical"] = meta_basis.lambda_empirical

    if hasattr(meta_basis, "lambda_shared"):
        summary["lambda_shared"] = meta_basis.lambda_shared

    if hasattr(meta_basis, "lambda_specific"):
        summary["lambda_specific"] = meta_basis.lambda_specific

    if hasattr(meta_basis, "lambda_parent"):
        summary["lambda_parent"] = meta_basis.lambda_parent

    if hasattr(meta_basis, "lambda_delta"):
        summary["lambda_delta"] = meta_basis.lambda_delta

    return summary