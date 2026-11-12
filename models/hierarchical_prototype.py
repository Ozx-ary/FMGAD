from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Distance Utility
# ============================================================

def pairwise_euclidean_distance(
    query_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute pairwise Euclidean distance.

    Args:
        query_embeddings:
            Tensor of shape [num_query, embedding_dim].

        prototypes:
            Tensor of shape [num_prototypes, embedding_dim].

    Returns:
        distances:
            Tensor of shape [num_query, num_prototypes].
    """
    return torch.cdist(query_embeddings, prototypes, p=2)


# ============================================================
# Hierarchical Prototype Head
# ============================================================

class HierarchicalPrototypeHead(nn.Module):
    """
    Hierarchical prototype head.

    V2-1:
        Builds empirical hierarchical prototypes:

            q_0, q_ano, q_S, q_A, q_M

    V2-2:
        Optionally refines anomaly type prototypes q_S/q_A/q_M
        using a meta basis module:

            shared basis + type-specific basis

    If meta_basis is None:
        This module behaves as the original V2-1 head.

    If meta_basis is not None:
        q_S/q_A/q_M are refined by meta_basis, and q_ano is computed
        as the mean of refined q_S/q_A/q_M.
    """

    def __init__(
        self,
        distance: str = "euclidean",
        coarse_loss_weight: float = 1.0,
        fine_loss_weight: float = 1.0,
        meta_basis: Optional[nn.Module] = None,
        use_meta_basis: bool = False,
    ):
        super().__init__()

        if distance != "euclidean":
            raise ValueError(
                f"Unsupported distance: {distance}. "
                "Currently only 'euclidean' is supported."
            )

        self.distance = distance
        self.coarse_loss_weight = coarse_loss_weight
        self.fine_loss_weight = fine_loss_weight
        self.meta_basis = meta_basis
        self.use_meta_basis = use_meta_basis

        if self.use_meta_basis and self.meta_basis is None:
            raise ValueError(
                "use_meta_basis=True requires meta_basis to be provided."
            )

    # ========================================================
    # Prototype Construction
    # ========================================================

    def _mean_by_mask(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        """
        Compute mean embedding by mask.
        """
        if mask.sum().item() == 0:
            raise ValueError(f"No support samples found for prototype {name}.")

        return embeddings[mask].mean(dim=0)

    def compute_hierarchical_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical prototypes.

        Args:
            support_embeddings:
                Tensor of shape [num_support, embedding_dim].

            support_labels:
                Tensor of shape [num_support].
                Expected labels: 0, 1, 2, 3.

        Returns:
            prototypes:
                Dict containing q_0, q_ano, q_S, q_A, q_M.
                If meta basis is enabled, also contains basis information.
        """
        support_labels = support_labels.long().to(support_embeddings.device)

        normal_mask = support_labels == 0
        structural_mask = support_labels == 1
        attribute_mask = support_labels == 2
        hybrid_mask = support_labels == 3
        anomaly_mask = support_labels != 0

        q_0 = self._mean_by_mask(
            embeddings=support_embeddings,
            mask=normal_mask,
            name="q_0",
        )

        # Empirical anomaly superclass prototype.
        q_ano_empirical = self._mean_by_mask(
            embeddings=support_embeddings,
            mask=anomaly_mask,
            name="q_ano",
        )

        q_S_empirical = self._mean_by_mask(
            embeddings=support_embeddings,
            mask=structural_mask,
            name="q_S",
        )

        q_A_empirical = self._mean_by_mask(
            embeddings=support_embeddings,
            mask=attribute_mask,
            name="q_A",
        )

        q_M_empirical = self._mean_by_mask(
            embeddings=support_embeddings,
            mask=hybrid_mask,
            name="q_M",
        )

        empirical_type_prototypes = torch.stack(
            [q_S_empirical, q_A_empirical, q_M_empirical],
            dim=0,
        )

        meta_basis_enabled = self.use_meta_basis and self.meta_basis is not None

        basis_info = None

        if meta_basis_enabled:
            refined_type_prototypes, basis_info = self.meta_basis(
                empirical_type_prototypes
            )

            meta_basis_mode = basis_info.get("meta_basis_mode", "refine")

            if meta_basis_mode == "parent_delta":
                type_prototypes = basis_info["type_prototypes"]

                q_S = type_prototypes[0]
                q_A = type_prototypes[1]
                q_M = type_prototypes[2]

                q_ano = basis_info["q_ano"]
                q_ano_source = "parent_delta_shared_basis"

            else:
                q_S = refined_type_prototypes[0]
                q_A = refined_type_prototypes[1]
                q_M = refined_type_prototypes[2]

                # V2-2 design:
                # q_ano is aggregated from refined anomaly type prototypes.
                q_ano = refined_type_prototypes.mean(dim=0)
                q_ano_source = "refined_type_mean"

        else:
            q_S = q_S_empirical
            q_A = q_A_empirical
            q_M = q_M_empirical
            q_ano = q_ano_empirical
            q_ano_source = "empirical_anomaly_mean"

        prototypes = {
            "q_0": q_0,
            "q_ano": q_ano,
            "q_S": q_S,
            "q_A": q_A,
            "q_M": q_M,
            "q_ano_empirical": q_ano_empirical,
            "q_S_empirical": q_S_empirical,
            "q_A_empirical": q_A_empirical,
            "q_M_empirical": q_M_empirical,
            "empirical_type_prototypes": empirical_type_prototypes,
            "meta_basis_enabled": torch.tensor(
                int(meta_basis_enabled),
                device=support_embeddings.device,
            ),
            "q_ano_source": q_ano_source,

            "meta_basis_mode": (
                basis_info.get("meta_basis_mode", "none")
                if basis_info is not None
                else "none"
            ),
        }

        if basis_info is not None:
            prototypes["basis_info"] = basis_info

            optional_basis_keys = [
                "refined_type_prototypes",
                "shared_attention",
                "specific_attention",
                "shared_components",
                "specific_components",
                "shared_component",
                "delta_types",
                "type_prototypes",
                "empirical_anomaly_prototype",
                "parent_delta_enabled",
            ]

            for key in optional_basis_keys:
                if key in basis_info:
                    prototypes[key] = basis_info[key]

        return prototypes

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward one episode.
        """
        device = support_embeddings.device

        support_labels = support_labels.long().to(device)

        if query_labels is not None:
            query_labels = query_labels.long().to(device)

        # ----------------------------------------------------
        # Build hierarchical prototypes
        # ----------------------------------------------------

        prototypes = self.compute_hierarchical_prototypes(
            support_embeddings=support_embeddings,
            support_labels=support_labels,
        )

        q_0 = prototypes["q_0"]
        q_ano = prototypes["q_ano"]
        q_S = prototypes["q_S"]
        q_A = prototypes["q_A"]
        q_M = prototypes["q_M"]

        # ----------------------------------------------------
        # Coarse level: normal vs anomaly
        # ----------------------------------------------------

        coarse_prototypes = torch.stack([q_0, q_ano], dim=0)

        coarse_distances = pairwise_euclidean_distance(
            query_embeddings=query_embeddings,
            prototypes=coarse_prototypes,
        )

        coarse_logits = -coarse_distances
        coarse_pred = torch.argmax(coarse_logits, dim=1)

        # ----------------------------------------------------
        # Fine level: S / A / M
        # ----------------------------------------------------

        fine_prototypes = torch.stack([q_S, q_A, q_M], dim=0)

        fine_distances_all = pairwise_euclidean_distance(
            query_embeddings=query_embeddings,
            prototypes=fine_prototypes,
        )

        fine_logits_all = -fine_distances_all
        fine_pred_all = torch.argmax(fine_logits_all, dim=1)

        # ----------------------------------------------------
        # Final prediction
        # ----------------------------------------------------

        predictions = torch.zeros_like(coarse_pred, dtype=torch.long)
        anomaly_pred_mask = coarse_pred == 1
        predictions[anomaly_pred_mask] = fine_pred_all[anomaly_pred_mask] + 1

        output = {
            "q_0": q_0,
            "q_ano": q_ano,
            "q_S": q_S,
            "q_A": q_A,
            "q_M": q_M,
            "q_ano_empirical": prototypes["q_ano_empirical"],
            "q_S_empirical": prototypes["q_S_empirical"],
            "q_A_empirical": prototypes["q_A_empirical"],
            "q_M_empirical": prototypes["q_M_empirical"],
            "empirical_type_prototypes": prototypes[
                "empirical_type_prototypes"
            ],
            "meta_basis_mode": prototypes["meta_basis_mode"],
            "meta_basis_enabled": prototypes["meta_basis_enabled"],
            "q_ano_source": prototypes["q_ano_source"],
            "coarse_prototypes": coarse_prototypes,
            "fine_prototypes": fine_prototypes,
            "coarse_distances": coarse_distances,
            "fine_distances_all": fine_distances_all,
            "coarse_logits": coarse_logits,
            "fine_logits_all": fine_logits_all,
            "coarse_pred": coarse_pred,
            "fine_pred_all": fine_pred_all,
            "predictions": predictions,
        }

        if "basis_info" in prototypes:
            output["basis_info"] = prototypes["basis_info"]

            optional_basis_keys = [
                "refined_type_prototypes",
                "shared_attention",
                "specific_attention",
                "shared_components",
                "specific_components",
                "shared_component",
                "delta_types",
                "type_prototypes",
                "empirical_anomaly_prototype",
                "parent_delta_enabled",
            ]

            for key in optional_basis_keys:
                if key in prototypes:
                    output[key] = prototypes[key]

        # ----------------------------------------------------
        # Optional supervised losses
        # ----------------------------------------------------

        if query_labels is not None:
            coarse_targets = (query_labels != 0).long()

            coarse_loss = F.cross_entropy(
                coarse_logits,
                coarse_targets,
            )

            anomaly_query_mask = query_labels != 0

            if anomaly_query_mask.sum().item() > 0:
                fine_logits = fine_logits_all[anomaly_query_mask]
                fine_targets = query_labels[anomaly_query_mask] - 1

                fine_loss = F.cross_entropy(
                    fine_logits,
                    fine_targets,
                )
            else:
                fine_logits = fine_logits_all.new_empty(
                    (0, fine_logits_all.size(1))
                )
                fine_targets = query_labels.new_empty((0,))
                fine_loss = fine_logits_all.sum() * 0.0

            loss = (
                self.coarse_loss_weight * coarse_loss
                + self.fine_loss_weight * fine_loss
            )

            accuracy = (predictions == query_labels).float().mean()
            coarse_accuracy = (coarse_pred == coarse_targets).float().mean()

            if anomaly_query_mask.sum().item() > 0:
                fine_pred = fine_pred_all[anomaly_query_mask]
                fine_accuracy = (
                    fine_pred == fine_targets
                ).float().mean()
            else:
                fine_accuracy = torch.tensor(
                    0.0,
                    device=query_embeddings.device,
                    dtype=query_embeddings.dtype,
                )

            output.update(
                {
                    "query_labels": query_labels,
                    "coarse_targets": coarse_targets,
                    "fine_logits": fine_logits,
                    "fine_targets": fine_targets,
                    "coarse_loss": coarse_loss,
                    "fine_loss": fine_loss,
                    "loss": loss,
                    "accuracy": accuracy,
                    "coarse_accuracy": coarse_accuracy,
                    "fine_accuracy": fine_accuracy,
                }
            )

        return output


# ============================================================
# Flat Prototype Head
# ============================================================

class FlatPrototypeHead(nn.Module):
    """
    Non-hierarchical 4-class prototype head.

    This head is used only for the w/o hierarchical prototype ablation. It
    keeps the episodic prototype classifier but removes the normal-vs-anomaly
    coarse level and the anomaly-type fine level.
    """

    def __init__(
        self,
        distance: str = "euclidean",
        loss_weight: float = 1.0,
    ):
        super().__init__()

        if distance != "euclidean":
            raise ValueError(
                f"Unsupported distance: {distance}. "
                "Currently only 'euclidean' is supported."
            )

        self.distance = distance
        self.loss_weight = loss_weight

    def _mean_by_class(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_id: int,
    ) -> torch.Tensor:
        mask = labels == class_id
        if mask.sum().item() == 0:
            raise ValueError(
                f"No support samples found for flat prototype class {class_id}."
            )

        return embeddings[mask].mean(dim=0)

    def compute_flat_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        support_labels = support_labels.long().to(support_embeddings.device)

        prototypes = [
            self._mean_by_class(
                embeddings=support_embeddings,
                labels=support_labels,
                class_id=class_id,
            )
            for class_id in range(4)
        ]

        return torch.stack(prototypes, dim=0)

    def forward(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        query_embeddings: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = support_embeddings.device

        support_labels = support_labels.long().to(device)
        if query_labels is not None:
            query_labels = query_labels.long().to(device)

        class_prototypes = self.compute_flat_prototypes(
            support_embeddings=support_embeddings,
            support_labels=support_labels,
        )

        flat_distances = pairwise_euclidean_distance(
            query_embeddings=query_embeddings,
            prototypes=class_prototypes,
        )
        logits = -flat_distances
        predictions = torch.argmax(logits, dim=1)

        output = {
            "prototype_mode": "flat",
            "class_prototypes": class_prototypes,
            "flat_prototypes": class_prototypes,
            "flat_distances": flat_distances,
            "class_distances": flat_distances,
            "prototype_logits": logits,
            "logits": logits,
            "predictions": predictions,
        }

        if query_labels is not None:
            flat_loss = F.cross_entropy(logits, query_labels)
            loss = self.loss_weight * flat_loss
            accuracy = (predictions == query_labels).float().mean()
            zero_loss = logits.sum() * 0.0

            coarse_targets = (query_labels != 0).long()
            coarse_pred = (predictions != 0).long()
            coarse_accuracy = (coarse_pred == coarse_targets).float().mean()

            anomaly_query_mask = query_labels != 0
            if anomaly_query_mask.sum().item() > 0:
                fine_targets = query_labels[anomaly_query_mask] - 1
                fine_pred = predictions[anomaly_query_mask] - 1
                fine_accuracy = (fine_pred == fine_targets).float().mean()
            else:
                fine_targets = query_labels.new_empty((0,))
                fine_pred = predictions.new_empty((0,))
                fine_accuracy = torch.tensor(
                    0.0,
                    device=query_embeddings.device,
                    dtype=query_embeddings.dtype,
                )

            output.update(
                {
                    "query_labels": query_labels,
                    "flat_loss": flat_loss,
                    "loss": loss,
                    "accuracy": accuracy,
                    "coarse_targets": coarse_targets,
                    "coarse_pred": coarse_pred,
                    "fine_targets": fine_targets,
                    "fine_pred": fine_pred,
                    "coarse_loss": zero_loss,
                    "fine_loss": zero_loss,
                    "coarse_accuracy": coarse_accuracy,
                    "fine_accuracy": fine_accuracy,
                }
            )

        return output


# ============================================================
# Hierarchical ProtoNet Wrapper
# ============================================================

class HierarchicalProtoNet(nn.Module):
    """
    Hierarchical ProtoNet.

    This module wraps an encoder and the HierarchicalPrototypeHead.

    Encoder can be:
        - base CenterAwareSubgraphEncoder
        - EvidenceAwareSubgraphEncoder from V1-9
    """

    def __init__(
        self,
        encoder: nn.Module,
        prototype_head: nn.Module,
    ):
        super().__init__()

        self.encoder = encoder
        self.prototype_head = prototype_head

    def forward(
        self,
        support_subgraphs,
        support_labels: torch.Tensor,
        query_subgraphs,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward one episode.
        """
        support_embeddings = self.encoder(support_subgraphs)
        query_embeddings = self.encoder(query_subgraphs)

        output = self.prototype_head(
            support_embeddings=support_embeddings,
            support_labels=support_labels,
            query_embeddings=query_embeddings,
            query_labels=query_labels,
        )

        output["support_embeddings"] = support_embeddings
        output["query_embeddings"] = query_embeddings

        return output


# ============================================================
# Summary
# ============================================================

def summarize_hierarchical_output(
    output: Dict[str, torch.Tensor],
) -> Dict[str, object]:
    """
    Summarize hierarchical prototype output for dry-run printing.
    """
    summary = {
        "support_embedding_shape": tuple(output["support_embeddings"].shape),
        "query_embedding_shape": tuple(output["query_embeddings"].shape),
        "q_0_shape": tuple(output["q_0"].shape),
        "q_ano_shape": tuple(output["q_ano"].shape),
        "q_S_shape": tuple(output["q_S"].shape),
        "q_A_shape": tuple(output["q_A"].shape),
        "q_M_shape": tuple(output["q_M"].shape),
        "q_ano_empirical_shape": tuple(output["q_ano_empirical"].shape),
        "empirical_type_prototype_shape": tuple(
            output["empirical_type_prototypes"].shape
        ),
        "coarse_prototype_shape": tuple(output["coarse_prototypes"].shape),
        "fine_prototype_shape": tuple(output["fine_prototypes"].shape),
        "coarse_logits_shape": tuple(output["coarse_logits"].shape),
        "fine_logits_all_shape": tuple(output["fine_logits_all"].shape),
        "prediction_shape": tuple(output["predictions"].shape),
        "meta_basis_enabled": bool(
            int(output["meta_basis_enabled"].detach().cpu().item())
        ),
        "q_ano_source": output["q_ano_source"],
    }

    if "refined_type_prototypes" in output:
        summary["refined_type_prototype_shape"] = tuple(
            output["refined_type_prototypes"].shape
        )

    if "shared_attention" in output:
        summary["shared_attention_shape"] = tuple(
            output["shared_attention"].shape
        )

    if "specific_attention" in output:
        summary["specific_attention_shape"] = tuple(
            output["specific_attention"].shape
        )

    if "shared_components" in output:
        summary["shared_component_shape"] = tuple(
            output["shared_components"].shape
        )

    if "specific_components" in output:
        summary["specific_component_shape"] = tuple(
            output["specific_components"].shape
        )

    if "fine_logits" in output:
        summary["fine_logits_shape"] = tuple(output["fine_logits"].shape)

    if "coarse_targets" in output:
        summary["coarse_targets_shape"] = tuple(output["coarse_targets"].shape)

    if "fine_targets" in output:
        summary["fine_targets_shape"] = tuple(output["fine_targets"].shape)

    if "coarse_loss" in output:
        summary["coarse_loss"] = float(output["coarse_loss"].item())

    if "fine_loss" in output:
        summary["fine_loss"] = float(output["fine_loss"].item())

    if "loss" in output:
        summary["loss"] = float(output["loss"].item())

    if "accuracy" in output:
        summary["accuracy"] = float(output["accuracy"].item())

    if "coarse_accuracy" in output:
        summary["coarse_accuracy"] = float(output["coarse_accuracy"].item())

    if "fine_accuracy" in output:
        summary["fine_accuracy"] = float(output["fine_accuracy"].item())


    if "meta_basis_mode" in output:
        summary["meta_basis_mode"] = output["meta_basis_mode"]

    if "parent_delta_enabled" in output:
        value = output["parent_delta_enabled"]
        if isinstance(value, torch.Tensor):
            summary["parent_delta_enabled"] = bool(
                int(value.detach().cpu().item())
            )
        else:
            summary["parent_delta_enabled"] = bool(value)

    if "delta_types" in output:
        summary["delta_types_shape"] = tuple(output["delta_types"].shape)

    if "type_prototypes" in output:
        summary["type_prototypes_shape"] = tuple(output["type_prototypes"].shape)

    if "shared_component" in output:
        summary["shared_component_shape"] = tuple(output["shared_component"].shape)

    if "empirical_anomaly_prototype" in output:
        summary["empirical_anomaly_prototype_shape"] = tuple(
            output["empirical_anomaly_prototype"].shape
        )

    return summary
