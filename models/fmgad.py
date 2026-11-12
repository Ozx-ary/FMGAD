from models.type_aware_head import TypeAwareEvidenceScoringHead
from typing import Dict, Optional, List, Any

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models.multiview_encoder import MultiViewCenterAwareSubgraphEncoder
from models.encoder import CenterAwareSubgraphEncoder
from models.evidence import ExplicitAnomalyEvidence
from models.reconstruction import MatrixReconstructionEvidence
from models.fusion import AnomalyAwareFusion, EvidenceAwareSubgraphEncoder
from models.hierarchical_prototype import (
    FlatPrototypeHead,
    HierarchicalPrototypeHead,
    HierarchicalProtoNet,
)
from models.meta_basis import (
    MetaPrototypeBasis,
    ParentDeltaMetaPrototypeBasis,
)


class FMGAD(nn.Module):
    """Few-shot graph anomaly classification with evidence and hierarchical prototypes."""

    def __init__(
        self,
        config: Dict,
        input_dim: int,
    ):
        super().__init__()

        self.config = config

        project_config = config.get("project", {})
        self.model_stage = project_config.get("version", "unknown")
        self.project_stage = project_config.get("stage", "unknown")
        model_config = config.get("model", {})
        encoder_config = config.get("encoder", {})
        multiview_config = config.get("multiview_encoder", {})
        evidence_config = config.get("evidence", {})
        reconstruction_config = config.get("reconstruction", {})
        fusion_config = config.get("fusion", {})
        hierarchical_config = config.get("hierarchical_prototype", {})
        meta_basis_config = config.get("meta_basis", {})
        type_aware_config = config.get("type_aware_head", {})

        # ----------------------------------------------------
        # Model switches
        # ----------------------------------------------------

        self.use_evidence = model_config.get(
            "use_evidence",
            evidence_config.get("enabled", True),
        )

        self.use_fusion = model_config.get(
            "use_fusion",
            fusion_config.get("enabled", True),
        )

        self.use_hierarchy = model_config.get(
            "use_hierarchy",
            hierarchical_config.get("enabled", True),
        )

        self.use_meta_basis = model_config.get(
            "use_meta_basis",
            hierarchical_config.get("use_meta_basis", True),
        )

        self.use_reconstruction = model_config.get(
            "use_reconstruction",
            False,
        )

        self.use_type_aware_head = model_config.get(
            "use_type_aware_head",
            False,
        )

        if self.use_meta_basis and not self.use_hierarchy:
            raise ValueError(
                "model.use_meta_basis=True requires model.use_hierarchy=True."
            )




        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        encoder_type = encoder_config.get("type", "center_gcn")
        encoder_output_dim = encoder_config.get("output_dim", 64)

        if encoder_type == "center_gcn":
            self.base_encoder = CenterAwareSubgraphEncoder(
                input_dim=encoder_config.get("input_dim", input_dim),
                hidden_dim=encoder_config.get("hidden_dim", 64),
                output_dim=encoder_output_dim,
                num_layers=encoder_config.get("num_layers", 2),
                dropout=encoder_config.get("dropout", 0.5),
            )

        elif encoder_type == "multiview_center_gcn":
            if not multiview_config.get("enabled", True):
                raise ValueError(
                    "encoder.type='multiview_center_gcn' requires "
                    "multiview_encoder.enabled=True."
                )

            structural_config = multiview_config.get("structural", {})
            attribute_config = multiview_config.get("attribute", {})
            joint_config = multiview_config.get("joint", {})
            mv_fusion_config = multiview_config.get("fusion", {})
            pooling_config = multiview_config.get("pooling", {})
            raw_disabled_views = multiview_config.get("disabled_views", [])
            if isinstance(raw_disabled_views, str):
                raw_disabled_views = [raw_disabled_views]

            disabled_views = {
                str(view).lower()
                for view in raw_disabled_views
            }

            def view_enabled(name: str, branch_config: Dict) -> bool:
                return bool(
                    branch_config.get(
                        "enabled",
                        name not in disabled_views,
                    )
                )

            self.base_encoder = MultiViewCenterAwareSubgraphEncoder(
                input_dim=encoder_config.get("input_dim", input_dim),

                structural_hidden_dim=structural_config.get("hidden_dim", 64),
                structural_output_dim=structural_config.get("output_dim", 64),
                structural_num_layers=structural_config.get("num_layers", 2),
                structural_dropout=structural_config.get("dropout", 0.5),

                attribute_hidden_dim=attribute_config.get("hidden_dim", 64),
                attribute_output_dim=attribute_config.get("output_dim", 64),
                attribute_num_layers=attribute_config.get("num_layers", 2),
                attribute_dropout=attribute_config.get("dropout", 0.5),

                joint_hidden_dim=joint_config.get("hidden_dim", 64),
                joint_output_dim=joint_config.get("output_dim", 64),
                joint_num_layers=joint_config.get("num_layers", 2),
                joint_dropout=joint_config.get("dropout", 0.5),

                fusion_type=mv_fusion_config.get("type", "concat_mlp"),
                fusion_hidden_dim=mv_fusion_config.get("hidden_dim", 128),
                fusion_output_dim=mv_fusion_config.get(
                    "output_dim",
                    encoder_output_dim,
                ),
                fusion_dropout=mv_fusion_config.get("dropout", 0.1),
                normalize_fused=mv_fusion_config.get("normalize", False),

                pooling=pooling_config.get("type", "center"),
                use_structural_view=view_enabled("structural", structural_config),
                use_attribute_view=view_enabled("attribute", attribute_config),
                use_joint_view=view_enabled("joint", joint_config),
            )

            encoder_output_dim = mv_fusion_config.get(
                "output_dim",
                encoder_output_dim,
            )

        else:
            raise ValueError(
                f"Unsupported encoder.type: {encoder_type}. "
                "Supported: center_gcn, multiview_center_gcn."
            )

        # ----------------------------------------------------
        # Evidence and fusion
        # ----------------------------------------------------

        self.evidence_module = None
        self.fusion_module = None

        if self.use_evidence:
            if self.use_reconstruction:
                self.evidence_module = MatrixReconstructionEvidence(
                    normalize=reconstruction_config.get(
                        "normalize",
                        evidence_config.get("normalize", True),
                    ),
                    structure_weight=reconstruction_config.get(
                        "structure_weight",
                        1.0,
                    ),
                    attribute_weight=reconstruction_config.get(
                        "attribute_weight",
                        1.0,
                    ),
                    combined_mode=reconstruction_config.get(
                        "combined_mode",
                        "product",
                    ),
                    eps=reconstruction_config.get("eps", 1e-8),
                )
            else:
                self.evidence_module = ExplicitAnomalyEvidence(
                    normalize=evidence_config.get("normalize", True),
                )

        if self.use_fusion:
            if self.evidence_module is None:
                raise ValueError(
                    "use_fusion=True requires use_evidence=True in V2-3."
                )

            self.fusion_module = AnomalyAwareFusion(
                embedding_dim=fusion_config.get(
                    "embedding_dim",
                    encoder_output_dim,
                ),
                evidence_dim=fusion_config.get("evidence_dim", 3),
                hidden_dim=fusion_config.get("hidden_dim", 64),
                output_dim=fusion_config.get("output_dim", encoder_output_dim),
                fusion_type=fusion_config.get("type", "concat_mlp"),
                dropout=fusion_config.get("dropout", 0.1),
            )

            self.encoder = EvidenceAwareSubgraphEncoder(
                base_encoder=self.base_encoder,
                evidence_module=self.evidence_module,
                fusion_module=self.fusion_module,
            )

            prototype_embedding_dim = self.fusion_module.actual_output_dim

        else:
            self.encoder = self.base_encoder
            prototype_embedding_dim = encoder_output_dim


        self.type_aware_head = None
        self.type_aware_use_query_embeddings = type_aware_config.get(
            "use_query_embeddings",
            True,
        )

        if self.use_type_aware_head:
            if not type_aware_config.get("enabled", True):
                raise ValueError(
                    "model.use_type_aware_head=True requires "
                    "type_aware_head.enabled=True."
                )

            self.type_aware_head = TypeAwareEvidenceScoringHead(
                embedding_dim=type_aware_config.get(
                    "embedding_dim",
                    prototype_embedding_dim,
                ),
                evidence_dim=type_aware_config.get("evidence_dim", 3),
                prototype_distance_dim=type_aware_config.get(
                    "prototype_distance_dim",
                    5,
                ),
                hidden_dim=type_aware_config.get("hidden_dim", 64),
                num_classes=type_aware_config.get("num_classes", 4),
                dropout=type_aware_config.get("dropout", 0.1),
                lambda_type=type_aware_config.get("lambda_type", 1.0),
                use_query_embeddings=self.type_aware_use_query_embeddings,
            )

        # ----------------------------------------------------
        # Meta basis
        # ----------------------------------------------------

        self.meta_basis = None
        self.meta_basis_mode = "none"

        if self.use_meta_basis:
            if not meta_basis_config.get("enabled", True):
                raise ValueError(
                    "model.use_meta_basis=True requires meta_basis.enabled=True."
                )

            meta_basis_mode = meta_basis_config.get("mode", "refine")
            self.meta_basis_mode = meta_basis_mode

            if meta_basis_mode == "refine":
                self.meta_basis = MetaPrototypeBasis(
                    embedding_dim=meta_basis_config.get(
                        "embedding_dim",
                        prototype_embedding_dim,
                    ),
                    num_types=meta_basis_config.get("num_types", 3),
                    num_shared_basis=meta_basis_config.get("num_shared_basis", 4),
                    num_specific_basis=meta_basis_config.get(
                        "num_specific_basis",
                        2,
                    ),
                    lambda_empirical=meta_basis_config.get(
                        "lambda_empirical",
                        1.0,
                    ),
                    lambda_shared=meta_basis_config.get("lambda_shared", 0.5),
                    lambda_specific=meta_basis_config.get(
                        "lambda_specific",
                        0.5,
                    ),
                    normalize=meta_basis_config.get("normalize", True),
                )

            elif meta_basis_mode == "parent_delta":
                self.meta_basis = ParentDeltaMetaPrototypeBasis(
                    embedding_dim=meta_basis_config.get(
                        "embedding_dim",
                        prototype_embedding_dim,
                    ),
                    num_types=meta_basis_config.get("num_types", 3),
                    num_shared_basis=meta_basis_config.get("num_shared_basis", 4),
                    num_specific_basis=meta_basis_config.get(
                        "num_specific_basis",
                        2,
                    ),
                    lambda_empirical=meta_basis_config.get(
                        "lambda_empirical",
                        1.0,
                    ),
                    lambda_parent=meta_basis_config.get(
                        "lambda_parent",
                        1.0,
                    ),
                    lambda_delta=meta_basis_config.get(
                        "lambda_delta",
                        0.5,
                    ),
                    normalize=meta_basis_config.get("normalize", True),
                )

            else:
                raise ValueError(
                    f"Unsupported meta_basis.mode: {meta_basis_mode}. "
                    "Supported modes are: refine, parent_delta."
                )


        # ----------------------------------------------------
        # Hierarchical prototype head
        # ----------------------------------------------------

        if self.use_hierarchy:
            self.prototype_head = HierarchicalPrototypeHead(
                distance=hierarchical_config.get("distance", "euclidean"),
                coarse_loss_weight=hierarchical_config.get(
                    "coarse_loss_weight",
                    1.0,
                ),
                fine_loss_weight=hierarchical_config.get(
                    "fine_loss_weight",
                    1.0,
                ),
                meta_basis=self.meta_basis,
                use_meta_basis=self.use_meta_basis,
            )
        else:
            self.prototype_head = FlatPrototypeHead(
                distance=hierarchical_config.get("distance", "euclidean"),
                loss_weight=hierarchical_config.get("flat_loss_weight", 1.0),
            )

        self.classifier = HierarchicalProtoNet(
            encoder=self.encoder,
            prototype_head=self.prototype_head,
        )

    # ========================================================
    # Utilities
    # ========================================================

    def _zero_loss_like(self, reference: torch.Tensor) -> torch.Tensor:
        """
        Return differentiable zero scalar on the same device/dtype as reference.
        """
        return reference.sum() * 0.0

    def _parse_forward_inputs(
        self,
        episode: Optional[Dict[str, Any]] = None,
        support_subgraphs: Optional[List[Data]] = None,
        support_labels: Optional[torch.Tensor] = None,
        query_subgraphs: Optional[List[Data]] = None,
        query_labels: Optional[torch.Tensor] = None,
    ):
        """
        Support two calling styles:

            outputs = model(episode)

        or:

            outputs = model(
                support_subgraphs=...,
                support_labels=...,
                query_subgraphs=...,
                query_labels=...,
            )
        """
        if episode is not None:
            support_subgraphs = episode["support_subgraphs"]
            support_labels = episode["support_labels"]
            query_subgraphs = episode["query_subgraphs"]
            query_labels = episode.get("query_labels", None)

        if support_subgraphs is None:
            raise ValueError("support_subgraphs is required.")

        if support_labels is None:
            raise ValueError("support_labels is required.")

        if query_subgraphs is None:
            raise ValueError("query_subgraphs is required.")

        return support_subgraphs, support_labels, query_subgraphs, query_labels

    def _attach_type_aware_outputs(
        self,
        output: Dict[str, torch.Tensor],
        query_subgraphs: List[Data],
        query_labels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Attach type-aware evidence scoring outputs.

        Designed pipeline:

            fused query embedding
            + reconstruction/statistical evidence
            + hierarchical or flat prototype distances
            -> type-aware evidence vector
            -> type logits

        This follows the FMGAD architecture design.
        """
        if self.type_aware_head is None:
            return output

        uses_flat_distances = "flat_distances" in output
        required_keys = (
            ["flat_distances"]
            if uses_flat_distances
            else [
                "coarse_distances",
                "fine_distances_all",
            ]
        )

        if self.type_aware_head.use_query_embeddings:
            required_keys.insert(0, "query_embeddings")

        for key in required_keys:
            if key not in output:
                raise ValueError(
                    f"Type-aware head requires output['{key}'], "
                    "but it is missing from classifier output."
                )

        if self.evidence_module is None:
            raise ValueError(
                "Type-aware head requires evidence_module to compute "
                "query evidence."
            )

        if self.type_aware_head.use_query_embeddings:
            reference_tensor = output["query_embeddings"]
        elif uses_flat_distances:
            reference_tensor = output["flat_distances"]
        else:
            reference_tensor = output["coarse_distances"]

        if (
            not isinstance(reference_tensor, torch.Tensor)
            or reference_tensor.dim() != 2
        ):
            raise ValueError(
                "Type-aware head expected a 2D reference tensor for "
                "query count/device/dtype, got "
                f"{type(reference_tensor).__name__}."
            )

        num_query = reference_tensor.size(0)

        distance_items = (
            [("flat_distances", output["flat_distances"])]
            if uses_flat_distances
            else [
                ("coarse_distances", output["coarse_distances"]),
                ("fine_distances_all", output["fine_distances_all"]),
            ]
        )

        for key, value in distance_items:
            if not isinstance(value, torch.Tensor) or value.dim() != 2:
                raise ValueError(
                    f"Type-aware head requires output['{key}'] to be a "
                    f"2D tensor, got {type(value).__name__}."
                )

            if value.size(0) != num_query:
                raise ValueError(
                    f"Type-aware head batch size mismatch: reference has "
                    f"{num_query} queries, but output['{key}'] has "
                    f"{value.size(0)}."
                )

        query_evidence = self.evidence_module(query_subgraphs)
        query_evidence = query_evidence.to(
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )

        if query_evidence.dim() != 2 or query_evidence.size(0) != num_query:
            raise ValueError(
                "Type-aware head query_evidence shape mismatch: expected "
                f"[{num_query}, evidence_dim], got "
                f"{tuple(query_evidence.shape)}."
            )

        type_head_kwargs = {
            "query_evidence": query_evidence,
            "query_labels": query_labels,
        }

        if uses_flat_distances:
            type_head_kwargs["flat_distances"] = output["flat_distances"]
        else:
            type_head_kwargs["coarse_distances"] = output["coarse_distances"]
            type_head_kwargs["fine_distances_all"] = output["fine_distances_all"]

        if self.type_aware_head.use_query_embeddings:
            type_head_kwargs["query_embeddings"] = output["query_embeddings"]

        # Ablation: remove query embeddings from the type-aware evidence scoring head.
        type_output = self.type_aware_head(**type_head_kwargs)

        output.update(type_output)

        output["query_type_evidence"] = query_evidence
        output["type_aware_use_query_embeddings"] = (
            self.type_aware_head.use_query_embeddings
        )

        if "type_aware_loss" in type_output:
            lambda_type = self.type_aware_head.lambda_type

            output["prototype_loss"] = output["loss"]
            output["loss_type"] = type_output["type_aware_loss"]

            output["loss"] = (
                output["loss"]
                + lambda_type * type_output["type_aware_loss"]
            )

            output["loss_total"] = output["loss"]

        return output
    
    def _attach_evidence_outputs(
        self,
        output: Dict[str, torch.Tensor],
        support_subgraphs: List[Data],
        query_subgraphs: List[Data],
    ) -> Dict[str, torch.Tensor]:
        """
        Attach support/query evidence to output dict for logging/debugging.

        Note:
            EvidenceAwareSubgraphEncoder already computes evidence internally.
            This method recomputes evidence only for inspection. It does not
            change the forward classification result.
        """
        if self.evidence_module is None:
            return output

        with torch.no_grad():
            support_evidence = self.evidence_module(support_subgraphs)
            query_evidence = self.evidence_module(query_subgraphs)

        output["support_evidence"] = support_evidence
        output["query_evidence"] = query_evidence

        return output

    def _is_valid_four_class_logits(
        self,
        value: Any,
    ) -> bool:
        return (
            isinstance(value, torch.Tensor)
            and value.dim() == 2
            and value.size(-1) == 4
        )

    def _construct_four_class_logits(
        self,
        output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Build compatible 4-class logits from hierarchical coarse/fine logits.
        """
        if "coarse_logits" not in output or "fine_logits_all" not in output:
            raise ValueError(
                "Cannot standardize FMGAD output: no valid [N, 4] logits "
                "were found and coarse_logits/fine_logits_all are missing."
            )

        coarse_logits = output["coarse_logits"]
        fine_logits_all = output["fine_logits_all"]

        if (
            not isinstance(coarse_logits, torch.Tensor)
            or coarse_logits.dim() != 2
            or coarse_logits.size(-1) != 2
        ):
            shape = (
                tuple(coarse_logits.shape)
                if isinstance(coarse_logits, torch.Tensor)
                else type(coarse_logits).__name__
            )
            raise ValueError(
                "Cannot construct 4-class logits: expected coarse_logits "
                f"with shape [N, 2], got {shape}."
            )

        if (
            not isinstance(fine_logits_all, torch.Tensor)
            or fine_logits_all.dim() != 2
            or fine_logits_all.size(-1) != 3
        ):
            shape = (
                tuple(fine_logits_all.shape)
                if isinstance(fine_logits_all, torch.Tensor)
                else type(fine_logits_all).__name__
            )
            raise ValueError(
                "Cannot construct 4-class logits: expected fine_logits_all "
                f"with shape [N, 3], got {shape}."
            )

        if coarse_logits.size(0) != fine_logits_all.size(0):
            raise ValueError(
                "Cannot construct 4-class logits: coarse/fine batch sizes "
                f"differ ({coarse_logits.size(0)} vs {fine_logits_all.size(0)})."
            )

        normal_logits = coarse_logits[:, 0:1]
        anomaly_type_logits = coarse_logits[:, 1:2] + fine_logits_all

        return torch.cat([normal_logits, anomaly_type_logits], dim=-1)

    def _resolve_final_logits_and_predictions(
        self,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Resolve standardized final logits/probabilities/predictions.
        """
        final_logits = None
        final_predictions = None
        final_source = None

        if self.use_type_aware_head:
            if not self._is_valid_four_class_logits(output.get("type_logits")):
                raise ValueError(
                    "Type-aware head is enabled, but output['type_logits'] "
                    "is missing or is not shaped [N, 4]."
                )

            final_logits = output["type_logits"]
            final_source = "type_logits"

            if "type_pred" in output:
                final_predictions = output["type_pred"].long()
            else:
                final_predictions = torch.argmax(final_logits, dim=-1)

        else:
            candidate_keys = [
                "logits",
                "prototype_logits",
                "proto_logits",
                "class_logits",
                "final_logits",
            ]

            for key in candidate_keys:
                if self._is_valid_four_class_logits(output.get(key)):
                    final_logits = output[key]
                    final_source = key
                    break

            if final_logits is None:
                final_logits = self._construct_four_class_logits(output)
                final_source = "coarse_plus_fine_logits"

            if "predictions" in output:
                final_predictions = output["predictions"].long()
            else:
                final_predictions = torch.argmax(final_logits, dim=-1)

        if not self._is_valid_four_class_logits(final_logits):
            shape = (
                tuple(final_logits.shape)
                if isinstance(final_logits, torch.Tensor)
                else type(final_logits).__name__
            )
            raise ValueError(
                "Cannot standardize FMGAD output: final logits must have "
                f"shape [N, 4], got {shape}."
            )

        if (
            not isinstance(final_predictions, torch.Tensor)
            or final_predictions.dim() != 1
            or final_predictions.size(0) != final_logits.size(0)
        ):
            shape = (
                tuple(final_predictions.shape)
                if isinstance(final_predictions, torch.Tensor)
                else type(final_predictions).__name__
            )
            raise ValueError(
                "Cannot standardize FMGAD output: final predictions must "
                f"have shape [N], got {shape}."
            )

        probs = torch.softmax(final_logits, dim=-1)

        output["logits"] = final_logits
        output["probs"] = probs
        output["predictions"] = final_predictions
        output["anomaly_score"] = 1.0 - probs[:, 0]
        output["final_logits_source"] = final_source

        return output

    def _standardize_output_keys(
        self,
        output: Dict[str, torch.Tensor],
        support_labels: torch.Tensor,
        query_labels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Add standardized aliases for downstream training/evaluation.
        """

        # Existing keys from HierarchicalPrototypeHead:
        #   loss, coarse_loss, fine_loss
        #   coarse_logits, fine_logits_all
        #   coarse_pred, fine_pred_all
        #   predictions

        if "loss" in output:
            output["loss_total"] = output["loss"]

        if "coarse_loss" in output:
            output["loss_coarse"] = output["coarse_loss"]

        if "fine_loss" in output:
            output["loss_fine"] = output["fine_loss"]

        if "prototype_loss" in output:
            output["loss_prototype"] = output["prototype_loss"]

        if "type_aware_loss" in output:
            output["loss_type_aware"] = output["type_aware_loss"]

        if "type_accuracy" in output:
            output["accuracy_type_aware"] = output["type_accuracy"]

        if "loss_basis" not in output:
            if "loss" in output:
                output["loss_basis"] = self._zero_loss_like(output["loss"])
            elif "query_embeddings" in output:
                output["loss_basis"] = self._zero_loss_like(
                    output["query_embeddings"]
                )

        if "coarse_logits" in output:
            output["logits_coarse"] = output["coarse_logits"]

        if "fine_logits_all" in output:
            output["logits_fine_all"] = output["fine_logits_all"]

        if "fine_logits" in output:
            output["logits_fine"] = output["fine_logits"]

        if "coarse_pred" in output:
            output["pred_coarse"] = output["coarse_pred"]

        if "fine_pred_all" in output:
            output["pred_fine_all"] = output["fine_pred_all"]

        if "predictions" in output:
            output["pred_final"] = output["predictions"]
        
        if "type_logits" in output:
            output["logits_type_aware"] = output["type_logits"]

        if "type_pred" in output:
            output["pred_type_aware"] = output["type_pred"]

        output["support_labels"] = support_labels

        if query_labels is not None:
            output["query_labels"] = query_labels

        output = self._resolve_final_logits_and_predictions(output)
        output["pred_final"] = output["predictions"]
        output["logits_final"] = output["logits"]

        if self.use_type_aware_head:
            output["model_stage"] = self.model_stage
            output["project_stage"] = self.project_stage
            output["meta_basis_mode"] = self.meta_basis_mode
        elif self.use_reconstruction:
            output["model_stage"] = "V2-4"
        else:
            output["model_stage"] = "V2-3"
        output["use_evidence"] = self.use_evidence
        output["use_fusion"] = self.use_fusion
        output["use_hierarchy"] = self.use_hierarchy
        output["use_meta_basis"] = self.use_meta_basis
        output["use_reconstruction"] = self.use_reconstruction
        output["use_type_aware_head"] = self.use_type_aware_head
        output[
            "type_aware_use_query_embeddings"
        ] = self.type_aware_use_query_embeddings
        output["evidence_type"] = (
            "matrix_reconstruction"
            if self.use_reconstruction
            else "simple_statistical"
        )

        return output

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        episode: Optional[Dict[str, Any]] = None,
        support_subgraphs: Optional[List[Data]] = None,
        support_labels: Optional[torch.Tensor] = None,
        query_subgraphs: Optional[List[Data]] = None,
        query_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        (
            support_subgraphs,
            support_labels,
            query_subgraphs,
            query_labels,
        ) = self._parse_forward_inputs(
            episode=episode,
            support_subgraphs=support_subgraphs,
            support_labels=support_labels,
            query_subgraphs=query_subgraphs,
            query_labels=query_labels,
        )

        output = self.classifier(
            support_subgraphs=support_subgraphs,
            support_labels=support_labels,
            query_subgraphs=query_subgraphs,
            query_labels=query_labels,
        )

        output = self._attach_type_aware_outputs(
            output=output,
            query_subgraphs=query_subgraphs,
            query_labels=query_labels,
        )

        output = self._standardize_output_keys(
            output=output,
            support_labels=support_labels,
            query_labels=query_labels,
        )

        output = self._attach_evidence_outputs(
            output=output,
            support_subgraphs=support_subgraphs,
            query_subgraphs=query_subgraphs,
        )

        return output
