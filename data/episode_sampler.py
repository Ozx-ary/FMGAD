from typing import Dict, List, Optional

import torch

from data.subgraph_sampler import sample_multiple_center_aware_subgraphs


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


# ============================================================
# Episode Sampler
# ============================================================

class EpisodeSampler:
    """
    Episode sampler for few-shot graph anomaly detection.

    This sampler selects center nodes according to anomaly labels,
    then converts selected center nodes into center-aware subgraphs.

    Supported tasks:
        1. binary:
            label source:
                data.anomaly_label or data.anom_label

            class space:
                0 = normal
                1 = anomaly

        2. multiclass:
            label source:
                data.anomaly_type or data.anom_type

            class space:
                0 = normal
                1 = structural anomaly
                2 = attribute anomaly
                3 = hybrid anomaly

    Sampling rule:
        - Each episode samples k_shot support nodes per class.
        - Each episode samples q_query query nodes per class.
        - Support/query are non-overlapping within the same class and episode.
        - Different episodes may reuse nodes, which is standard in episodic training.
    """

    def __init__(
        self,
        data,
        num_hops: int = 2,
        relabel_nodes: bool = True,
        task: str = "binary",
        n_way: Optional[int] = None,
        k_shot: int = 5,
        q_query: int = 10,
        seed: int = 42,
        split_pools: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
        default_split: str = "train",
        dataset_name: str = "unknown",
        debug_diagnostics: bool = False,
        safe_subgraph_sampling: bool = False,
        allow_replacement: bool = False,
        max_nodes_per_subgraph: Optional[int] = None,
        max_edges_per_subgraph: Optional[int] = None,
        neighbor_sample_per_hop: Optional[int] = None,
    ):
        """
        Args:
            data:
                Full graph data object.

            num_hops:
                Number of hops for center-aware subgraph sampling.

            relabel_nodes:
                Whether to relabel nodes in sampled subgraphs.

            task:
                Episode task type.
                Supported:
                    - "binary"
                    - "multiclass"

            n_way:
                Number of classes in one episode.
                For current tasks:
                    - binary: n_way is treated as 2
                    - multiclass: n_way must be 4

            k_shot:
                Number of support samples per class.

            q_query:
                Number of query samples per class.

            seed:
                Random seed for episode sampling.
        """

        # ----------------------------------------------------
        # Basic configs
        # ----------------------------------------------------

        self.data = data
        self.num_hops = num_hops
        self.relabel_nodes = relabel_nodes
        self.task = task
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.seed = seed
        self.split_pools = split_pools
        self.default_split = default_split
        self.dataset_name = dataset_name
        self.debug_diagnostics = debug_diagnostics
        self.safe_subgraph_sampling = safe_subgraph_sampling
        self.allow_replacement = allow_replacement
        self.max_nodes_per_subgraph = max_nodes_per_subgraph
        self.max_edges_per_subgraph = max_edges_per_subgraph
        self.neighbor_sample_per_hop = neighbor_sample_per_hop
        self.num_nodes = self._resolve_num_nodes(data)

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # ----------------------------------------------------
        # Select label source and class space
        # ----------------------------------------------------
        # Important:
        #   The project currently may use two naming styles:
        #
        #   Preferred:
        #       data.anomaly_label
        #       data.anomaly_type
        #
        #   Legacy / short:
        #       data.anom_label
        #       data.anom_type
        #
        # This sampler supports both styles for robustness.

        if self.task == "binary":
            self.labels = self._get_binary_labels(data)
            self.labels = self._prepare_label_tensor(self.labels)

            # Binary task always uses two classes.
            self.n_way = 2
            self.classes = torch.tensor([0, 1], dtype=torch.long)

        elif self.task == "multiclass":
            self.labels = self._get_multiclass_labels(data)
            self.labels = self._prepare_label_tensor(self.labels)

            if self.n_way is None:
                self.n_way = 4

            if self.n_way != 4:
                raise ValueError(
                    "Current multiclass anomaly task expects n_way=4 "
                    "for classes [0, 1, 2, 3]."
                )

            self.classes = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        else:
            raise ValueError(
                f"Unsupported episode task: {self.task}. "
                "Supported tasks are: 'binary', 'multiclass'."
            )

        self.classes = self._prepare_index_tensor(
            self.classes,
            name="classes",
            check_bounds=False,
        )

        _debug_print(
            self.debug_diagnostics,
            (
                "EpisodeSampler init: "
                f"dataset={self.dataset_name}, task={self.task}, "
                f"num_nodes={self.num_nodes}, seed={self.seed}, "
                f"safe_subgraph_sampling={self.safe_subgraph_sampling}, "
                f"allow_replacement={self.allow_replacement}, "
                f"max_nodes_per_subgraph={self.max_nodes_per_subgraph}, "
                f"max_edges_per_subgraph={self.max_edges_per_subgraph}, "
                f"neighbor_sample_per_hop={self.neighbor_sample_per_hop}"
            ),
        )
        _debug_print(
            self.debug_diagnostics,
            f"listing available classes: {self.classes.detach().cpu().tolist()}",
        )

        # ----------------------------------------------------
        # Build class-to-node-index pools
        # ----------------------------------------------------
        # self.class_to_indices:
        #     {
        #         class_id: tensor([... node indices ...])
        #     }

        if self.split_pools is None:
            self.class_to_indices = self._build_class_to_indices()
        else:
            self.class_to_indices = None
            self._validate_split_pools()

        _debug_print(
            self.debug_diagnostics,
            f"class counts: {self._class_count_summary()}",
        )

    def _resolve_num_nodes(self, data) -> int:
        num_nodes = getattr(data, "num_nodes", None)
        if num_nodes is not None:
            return int(num_nodes)

        if hasattr(data, "x") and data.x is not None:
            return int(data.x.size(0))

        raise ValueError("EpisodeSampler cannot infer num_nodes from data.")

    def _prepare_label_tensor(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.device.type != "cpu":
            labels = labels.detach().cpu()

        if labels.dtype != torch.long:
            labels = labels.long()

        if not labels.is_contiguous():
            labels = labels.contiguous()

        if labels.dim() != 1:
            raise ValueError(
                f"Labels must be 1D, got shape {tuple(labels.shape)}."
            )

        if labels.numel() != self.num_nodes:
            raise ValueError(
                f"Labels length mismatch for dataset={self.dataset_name}: "
                f"labels={labels.numel()}, num_nodes={self.num_nodes}."
            )

        return labels

    def _prepare_index_tensor(
        self,
        indices,
        name: str,
        check_bounds: bool = True,
    ) -> torch.Tensor:
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices)

        if indices.device.type != "cpu":
            indices = indices.detach().cpu()

        if indices.dtype != torch.long:
            indices = indices.long()

        if not indices.is_contiguous():
            indices = indices.contiguous()

        if indices.dim() != 1:
            raise ValueError(
                f"{name} must be a 1D index tensor, got shape {tuple(indices.shape)}."
            )

        if check_bounds and indices.numel() > 0:
            min_value = int(indices.min().item())
            max_value = int(indices.max().item())
            if min_value < 0 or max_value >= self.num_nodes:
                raise ValueError(
                    f"{name} contains out-of-range node ids for "
                    f"dataset={self.dataset_name}: min={min_value}, "
                    f"max={max_value}, valid=[0, {self.num_nodes - 1}]."
                )

        return indices

    def _class_count_summary(self, split: Optional[str] = None) -> str:
        parts = []
        for class_id in self.classes.tolist():
            if split is None and self.split_pools is None:
                indices = self.class_to_indices[class_id]
            elif split is not None and self.split_pools is not None:
                indices = self.split_pools.get(split, {}).get(class_id, None)
            elif self.split_pools is None:
                indices = self.class_to_indices[class_id]
            else:
                counts = []
                for split_name, class_map in self.split_pools.items():
                    split_indices = class_map.get(class_id, None)
                    count = 0 if split_indices is None else int(split_indices.numel())
                    counts.append(f"{split_name}:{count}")
                parts.append(f"class {class_id} ({', '.join(counts)})")
                continue

            count = 0 if indices is None else int(indices.numel())
            parts.append(f"class {class_id}:{count}")

        return "; ".join(parts)

    # ========================================================
    # Label Source Helpers
    # ========================================================

    @staticmethod
    def _get_binary_labels(data) -> torch.Tensor:
        """
        Get binary anomaly labels from data.

        Returns:
            labels:
                Tensor of shape [num_nodes].
                0 = normal
                1 = anomaly
        """
        if hasattr(data, "anomaly_label"):
            return data.anomaly_label.long()

        if hasattr(data, "anom_label"):
            return data.anom_label.long()

        raise ValueError(
            "Binary task requires data.anomaly_label or data.anom_label."
        )

    @staticmethod
    def _get_multiclass_labels(data) -> torch.Tensor:
        """
        Get multiclass anomaly type labels from data.

        Returns:
            labels:
                Tensor of shape [num_nodes].
                0 = normal
                1 = structural anomaly
                2 = attribute anomaly
                3 = hybrid anomaly
        """
        if hasattr(data, "anomaly_type"):
            return data.anomaly_type.long()

        if hasattr(data, "anom_type"):
            return data.anom_type.long()

        raise ValueError(
            "Multiclass task requires data.anomaly_type or data.anom_type."
        )

    # ========================================================
    # Class Pool Construction
    # ========================================================

    def _build_class_to_indices(self) -> Dict[int, torch.Tensor]:
        """
        Build mapping from class id to node indices.

        Returns:
            class_to_indices:
                {
                    class_id: node indices with this class
                }
        """
        class_to_indices = {}

        required_per_class = self.k_shot + self.q_query

        for class_id in self.classes.tolist():
            indices = torch.where(self.labels == class_id)[0]
            indices = self._prepare_index_tensor(
                indices,
                name=f"class_to_indices[{class_id}]",
            )

            _debug_print(
                self.debug_diagnostics,
                (
                    "class pool built: "
                    f"dataset={self.dataset_name}, class={class_id}, "
                    f"indices={_tensor_stats(indices)}, "
                    f"requested_per_episode={required_per_class}"
                ),
            )

            if indices.numel() < required_per_class and not self.allow_replacement:
                raise ValueError(
                    f"Insufficient samples for dataset={self.dataset_name}, "
                    f"split=all, class={class_id}: available={indices.numel()}, "
                    f"requested={required_per_class}, k_shot={self.k_shot}, "
                    f"q_query={self.q_query}, replacement={self.allow_replacement}."
                )

            class_to_indices[class_id] = indices

        return class_to_indices


    def _validate_split_pools(self) -> None:
        """
        Validate split-aware class pools.

        Expected structure:
            {
                "train": {0: Tensor(...), 1: Tensor(...), ...},
                "val":   {0: Tensor(...), 1: Tensor(...), ...},
                "test":  {0: Tensor(...), 1: Tensor(...), ...},
            }
        """
        if self.split_pools is None:
            return

        required_per_class = self.k_shot + self.q_query

        _debug_print(
            self.debug_diagnostics,
            (
                "resolving split pools: "
                f"available_splits={list(self.split_pools.keys())}, "
                f"requested_per_class={required_per_class}"
            ),
        )

        for split_name, class_map in self.split_pools.items():
            for class_id in self.classes.tolist():
                if class_id not in class_map:
                    raise ValueError(
                        f"Split '{split_name}' does not contain class {class_id}."
                    )

                indices = class_map[class_id]

                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices, dtype=torch.long)

                indices = self._prepare_index_tensor(
                    indices,
                    name=f"split_pools[{split_name}][{class_id}]",
                )
                self.split_pools[split_name][class_id] = indices

                _debug_print(
                    self.debug_diagnostics,
                    (
                        "split class count: "
                        f"dataset={self.dataset_name}, split={split_name}, "
                        f"class={class_id}, indices={_tensor_stats(indices)}, "
                        f"requested_per_episode={required_per_class}"
                    ),
                )

                if indices.numel() < required_per_class and not self.allow_replacement:
                    raise ValueError(
                        f"Insufficient samples for dataset={self.dataset_name}, "
                        f"split={split_name}, class={class_id}: "
                        f"available={indices.numel()}, requested={required_per_class}, "
                        f"k_shot={self.k_shot}, q_query={self.q_query}, "
                        "replacement=false."
                    )

    # ========================================================
    # Node Sampling
    # ========================================================

    def _get_candidate_indices(
        self,
        class_id: int,
        split: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Get candidate node indices for a class.

        If split_pools is enabled, candidates come from split-specific pools.
        Otherwise, candidates come from the full class pool.
        """
        _debug_print(
            self.debug_diagnostics,
            (
                "resolving split pools for class: "
                f"dataset={self.dataset_name}, split={split}, class={class_id}"
            ),
        )

        if self.split_pools is None:
            indices = self.class_to_indices[class_id]
            _debug_print(
                self.debug_diagnostics,
                (
                    "resolved class pool: "
                    f"split=all, class={class_id}, indices={_tensor_stats(indices)}"
                ),
            )
            return indices

        if split is None:
            split = self.default_split

        if split not in self.split_pools:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Available splits: {list(self.split_pools.keys())}"
            )

        if class_id not in self.split_pools[split]:
            raise ValueError(
                f"Class {class_id} not found in split '{split}'."
            )

        indices = self._prepare_index_tensor(
            self.split_pools[split][class_id],
            name=f"candidate_indices[{split}][{class_id}]",
        )

        _debug_print(
            self.debug_diagnostics,
            (
                "resolved class pool: "
                f"split={split}, class={class_id}, indices={_tensor_stats(indices)}"
            ),
        )

        return indices

    def _sample_indices_from_class(
        self,
        class_id: int,
        split: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Sample support + query node indices from one class.

        Args:
            class_id:
                Class id in the episode class space.
            split:
                Optional split name: train / val / test.

        Returns:
            selected:
                Tensor of shape [k_shot + q_query].
                The first k_shot nodes are support nodes.
                The remaining q_query nodes are query nodes.
        """
        candidate_indices = self._get_candidate_indices(
            class_id=class_id,
            split=split,
        )

        required = self.k_shot + self.q_query

        _debug_print(
            self.debug_diagnostics,
            (
                "support/query requested per class: "
                f"dataset={self.dataset_name}, split={split}, class={class_id}, "
                f"k_shot={self.k_shot}, q_query={self.q_query}, "
                f"required={required}, available={candidate_indices.numel()}"
            ),
        )

        if candidate_indices.numel() == 0:
            raise ValueError(
                f"Cannot sample from empty class pool for dataset={self.dataset_name}, "
                f"split={split}, class={class_id}."
            )

        if candidate_indices.numel() < required and not self.allow_replacement:
            raise ValueError(
                f"Insufficient samples for dataset={self.dataset_name}, "
                f"split={split}, class={class_id}: "
                f"available={candidate_indices.numel()}, requested={required}, "
                f"k_shot={self.k_shot}, q_query={self.q_query}, "
                "replacement=false."
            )

        if candidate_indices.numel() < required:
            _debug_print(
                True,
                (
                    "[Warning] sampling with replacement: "
                    f"dataset={self.dataset_name}, split={split}, "
                    f"class={class_id}, available={candidate_indices.numel()}, "
                    f"requested={required}"
                ),
            )
            _debug_print(
                self.debug_diagnostics,
                (
                    "before torch.randint random sampling call: "
                    f"high={candidate_indices.numel()}, size={required}"
                ),
            )
            choice_positions = torch.randint(
                high=candidate_indices.numel(),
                size=(required,),
                generator=self.generator,
                dtype=torch.long,
            )
            _debug_print(
                self.debug_diagnostics,
                f"after torch.randint random sampling call: {_tensor_stats(choice_positions)}",
            )
        else:
            _debug_print(
                self.debug_diagnostics,
                (
                    "before torch.randperm random sampling call: "
                    f"n={candidate_indices.numel()}"
                ),
            )
            perm = torch.randperm(
                candidate_indices.numel(),
                generator=self.generator,
            )
            _debug_print(
                self.debug_diagnostics,
                f"after torch.randperm random sampling call: {_tensor_stats(perm)}",
            )
            choice_positions = perm[: required]

        choice_positions = self._prepare_index_tensor(
            choice_positions,
            name=f"choice_positions[{split}][{class_id}]",
            check_bounds=False,
        )
        selected = candidate_indices[choice_positions].long().contiguous()

        _debug_print(
            self.debug_diagnostics,
            (
                "sampled indices for class: "
                f"split={split}, class={class_id}, selected={_tensor_stats(selected)}"
            ),
        )

        return selected
    # ========================================================
    # Episode Sampling
    # ========================================================

    def sample_episode(
        self,
        split: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Sample one few-shot episode.

        Args:
            split:
                Optional split name. When split-aware sampling is enabled,
                it should be one of:
                    - "train"
                    - "val"
                    - "test"

                If None, self.default_split is used.

        Returns:
            episode:
                {
                    "split": str,
                    "classes": Tensor[num_classes],
                    "support_indices": Tensor[num_classes * k_shot],
                    "support_center_nodes": Tensor[num_classes * k_shot],
                    "support_labels": Tensor[num_classes * k_shot],
                    "support_subgraphs": List[Data],

                    "query_indices": Tensor[num_classes * q_query],
                    "query_center_nodes": Tensor[num_classes * q_query],
                    "query_labels": Tensor[num_classes * q_query],
                    "query_subgraphs": List[Data],
                }
        """
        if split is None:
            split = self.default_split

        _debug_print(
            self.debug_diagnostics,
            (
                "sample_episode start: "
                f"dataset={self.dataset_name}, split={split}, "
                f"available_classes={self.classes.detach().cpu().tolist()}, "
                f"class_counts={self._class_count_summary(split)}"
            ),
        )
        _debug_print(
            self.debug_diagnostics,
            f"chosen episode classes: {self.classes.detach().cpu().tolist()}",
        )

        support_indices = []
        support_labels = []
        query_indices = []
        query_labels = []

        # ----------------------------------------------------
        # Sample nodes class by class
        # ----------------------------------------------------

        for class_id in self.classes.tolist():
            selected = self._sample_indices_from_class(
                class_id=class_id,
                split=split,
            )

            class_support = selected[: self.k_shot]
            class_query = selected[self.k_shot : self.k_shot + self.q_query]
            class_support = self._prepare_index_tensor(
                class_support,
                name=f"support_indices[{split}][{class_id}]",
            )
            class_query = self._prepare_index_tensor(
                class_query,
                name=f"query_indices[{split}][{class_id}]",
            )

            _debug_print(
                self.debug_diagnostics,
                (
                    "support/query sampled indices shape and dtype: "
                    f"class={class_id}, support={_tensor_stats(class_support)}, "
                    f"query={_tensor_stats(class_query)}"
                ),
            )

            support_indices.append(class_support)
            query_indices.append(class_query)

            support_labels.append(
                torch.full(
                    size=(self.k_shot,),
                    fill_value=class_id,
                    dtype=torch.long,
                )
            )

            query_labels.append(
                torch.full(
                    size=(self.q_query,),
                    fill_value=class_id,
                    dtype=torch.long,
                )
            )

        support_indices = self._prepare_index_tensor(
            torch.cat(support_indices, dim=0),
            name=f"support_indices[{split}]",
        )
        query_indices = self._prepare_index_tensor(
            torch.cat(query_indices, dim=0),
            name=f"query_indices[{split}]",
        )
        support_labels = torch.cat(support_labels, dim=0).long().contiguous()
        query_labels = torch.cat(query_labels, dim=0).long().contiguous()

        _debug_print(
            self.debug_diagnostics,
            (
                "episode sampled tensors: "
                f"support={_tensor_stats(support_indices)}, "
                f"query={_tensor_stats(query_indices)}, "
                f"support_labels={_tensor_stats(support_labels)}, "
                f"query_labels={_tensor_stats(query_labels)}"
            ),
        )

        # ----------------------------------------------------
        # Convert center nodes into center-aware subgraphs
        # ----------------------------------------------------

        _debug_print(
            self.debug_diagnostics,
            "before support edge_index/subgraph extraction",
        )
        support_subgraphs = sample_multiple_center_aware_subgraphs(
            data=self.data,
            center_nodes=support_indices,
            num_hops=self.num_hops,
            relabel_nodes=self.relabel_nodes,
            max_samples=support_indices.numel(),
            debug_diagnostics=self.debug_diagnostics,
            safe_mode=self.safe_subgraph_sampling,
            seed=self.seed,
            max_nodes_per_subgraph=self.max_nodes_per_subgraph,
            max_edges_per_subgraph=self.max_edges_per_subgraph,
            neighbor_sample_per_hop=self.neighbor_sample_per_hop,
            context=f"dataset={self.dataset_name} split={split} support",
        )
        _debug_print(
            self.debug_diagnostics,
            f"after support edge_index/subgraph extraction: count={len(support_subgraphs)}",
        )

        _debug_print(
            self.debug_diagnostics,
            "before query edge_index/subgraph extraction",
        )
        query_subgraphs = sample_multiple_center_aware_subgraphs(
            data=self.data,
            center_nodes=query_indices,
            num_hops=self.num_hops,
            relabel_nodes=self.relabel_nodes,
            max_samples=query_indices.numel(),
            debug_diagnostics=self.debug_diagnostics,
            safe_mode=self.safe_subgraph_sampling,
            seed=self.seed,
            max_nodes_per_subgraph=self.max_nodes_per_subgraph,
            max_edges_per_subgraph=self.max_edges_per_subgraph,
            neighbor_sample_per_hop=self.neighbor_sample_per_hop,
            context=f"dataset={self.dataset_name} split={split} query",
        )
        _debug_print(
            self.debug_diagnostics,
            f"after query edge_index/subgraph extraction: count={len(query_subgraphs)}",
        )

        # ----------------------------------------------------
        # Build episode dictionary
        # ----------------------------------------------------

        episode = {
            "split": split,
            "classes": self.classes.clone(),

            "support_indices": support_indices,
            "support_center_nodes": support_indices,
            "support_labels": support_labels,
            "support_subgraphs": support_subgraphs,

            "query_indices": query_indices,
            "query_center_nodes": query_indices,
            "query_labels": query_labels,
            "query_subgraphs": query_subgraphs,
        }

        _debug_print(
            self.debug_diagnostics,
            (
                "returning episode batch: "
                f"split={split}, support_subgraphs={len(support_subgraphs)}, "
                f"query_subgraphs={len(query_subgraphs)}"
            ),
        )

        return episode

    def sample_episodes(
        self,
        num_episodes: int,
        split: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """
        Sample multiple episodes.

        Args:
            num_episodes:
                Number of episodes to sample.
            split:
                Optional split name: train / val / test.

        Returns:
            episodes:
                List of sampled episodes.
        """
        episodes = []

        _debug_print(
            self.debug_diagnostics,
            (
                "sample_episodes start: "
                f"dataset={self.dataset_name}, split={split}, "
                f"num_episodes={num_episodes}"
            ),
        )

        for episode_index in range(num_episodes):
            _debug_print(
                self.debug_diagnostics,
                f"before sampling episode[{episode_index}]",
            )
            episode = self.sample_episode(split=split)
            episodes.append(episode)
            _debug_print(
                self.debug_diagnostics,
                f"after sampling episode[{episode_index}]",
            )

        _debug_print(
            self.debug_diagnostics,
            f"returning episode batch list: count={len(episodes)}",
        )
        return episodes

    def sample_episodes_with_seed(
        self,
        num_episodes: int,
        seed: int,
        split: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """
        Sample episodes from an independent deterministic generator.

        This is intended for fixed validation/test episode generation. It
        preserves the sampler's advancing training generator state.
        """
        previous_seed = self.seed
        previous_generator = self.generator

        generator = torch.Generator()
        generator.manual_seed(int(seed))

        self.seed = int(seed)
        self.generator = generator

        try:
            return self.sample_episodes(
                num_episodes=num_episodes,
                split=split,
            )
        finally:
            self.seed = previous_seed
            self.generator = previous_generator


# ============================================================
# Episode Summary
# ============================================================

def summarize_episode(episode: Dict[str, object]) -> Dict[str, object]:
    """
    Summarize an episode.

    Args:
        episode:
            Episode dictionary.

    Returns:
        summary:
            Episode statistics for dry-run printing.
    """
    support_labels = episode["support_labels"]
    query_labels = episode["query_labels"]

    support_unique, support_counts = torch.unique(
        support_labels,
        return_counts=True,
    )

    query_unique, query_counts = torch.unique(
        query_labels,
        return_counts=True,
    )

    support_label_count = {
        int(k.item()): int(v.item())
        for k, v in zip(support_unique, support_counts)
    }

    query_label_count = {
        int(k.item()): int(v.item())
        for k, v in zip(query_unique, query_counts)
    }

    summary = {
        "support_size": int(support_labels.numel()),
        "query_size": int(query_labels.numel()),
        "support_label_count": support_label_count,
        "query_label_count": query_label_count,
    }

    # --------------------------------------------------------
    # Optional episode classes
    # --------------------------------------------------------

    if "classes" in episode:
        summary["classes"] = episode["classes"].detach().cpu().tolist()

    # --------------------------------------------------------
    # Center node indices
    # --------------------------------------------------------
    # Prefer explicit support_center_nodes / query_center_nodes.
    # Fall back to support_indices / query_indices for compatibility.

    if "support_center_nodes" in episode:
        summary["support_center_nodes"] = (
            episode["support_center_nodes"].detach().cpu().tolist()
        )
    elif "support_indices" in episode:
        summary["support_center_nodes"] = (
            episode["support_indices"].detach().cpu().tolist()
        )
    else:
        summary["support_center_nodes"] = []

    if "query_center_nodes" in episode:
        summary["query_center_nodes"] = (
            episode["query_center_nodes"].detach().cpu().tolist()
        )
    elif "query_indices" in episode:
        summary["query_center_nodes"] = (
            episode["query_indices"].detach().cpu().tolist()
        )
    else:
        summary["query_center_nodes"] = []

    return summary
