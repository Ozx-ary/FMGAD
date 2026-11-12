import argparse
import os
import time
from typing import Dict, List, Any, Optional, Sequence

import torch
import torch.nn as nn

from models.fmgad import FMGAD
from data.anomaly_injection import inject_multi_type_anomalies, summarize_anomalies
from data.datasets import (
    MAT_DATASETS,
    PLANETOID_DATASETS,
    attach_synthetic_label_aliases,
    load_mat_dataset,
    load_planetoid_dataset,
    print_dataset_load_summary,
)
from data.episode_sampler import EpisodeSampler
from data.episode_cache import get_fixed_eval_episodes_from_config
from data.split_manager import (
    build_or_load_split_pools,
    print_split_summary,
)
from utils.config import get_from_config, load_config, set_in_config
from utils.metrics import (
    compute_classification_metrics,
    compute_binary_anomaly_metrics,
    compute_metrics,
)
from utils.early_stopping import EarlyStopping
from utils.experiment_io import (
    append_metrics_to_csv,
    save_json,
    save_yaml,
)
from utils.seed import set_seed


SAFE_SAMPLING_DATASETS = {
    "blogcatalog",
    "flickr",
    "weibo",
    "tolokers",
    "questions",
    "reddit",
    "photo",
    "cs",
}


# ============================================================
# Argument Parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="FMGAD Training Pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/books.yaml",
        help="Path to YAML config file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override runtime seed.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override experiment output directory.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override experiment run name.",
    )

    parser.add_argument(
        "--debug_diagnostics",
        action="store_true",
        help="Print flushed diagnostics around major training stages.",
    )

    return parser.parse_args()


def apply_cli_overrides(
    config: Dict,
    args,
) -> Dict:
    """
    Apply command-line overrides to config.
    """
    if args.seed is not None:
        set_in_config(config, "runtime.seed", args.seed)
        set_in_config(config, "seed", args.seed)
        set_in_config(config, "episode.seed", args.seed)
        set_in_config(config, "split.seed", args.seed)

        anomaly_ratio = get_from_config(config, "anomaly.ratio", 0.05)
        ratio_tag = f"ratio{int(anomaly_ratio * 100):03d}"
        dataset_name = get_from_config(config, "dataset.name", "cora").lower()

        set_in_config(
            config,
            "split.save_path",
            f"splits/{dataset_name}_v2_8_{ratio_tag}_seed{args.seed}.json",
        )

    if args.output_dir is not None:
        set_in_config(config, "experiment.output_dir", args.output_dir)
        set_in_config(
            config,
            "training.checkpoint_dir",
            os.path.join(args.output_dir, "checkpoints"),
        )

    if args.run_name is not None:
        set_in_config(config, "experiment.run_name", args.run_name)

    return config


# ============================================================
# Runtime Utilities
# ============================================================

def resolve_device(device_config: str) -> torch.device:
    """
    Resolve runtime device.
    """
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_config == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_config)


def ensure_dir(path: str) -> None:
    """
    Create directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def move_episode_labels_to_device(
    episode: Dict,
    device: torch.device,
) -> Dict:
    """
    Move support/query labels and PyG subgraphs to target device.

    The function name is kept for backward compatibility.
    """
    if "support_labels" in episode and episode["support_labels"] is not None:
        episode["support_labels"] = episode["support_labels"].to(device)

    if "query_labels" in episode and episode["query_labels"] is not None:
        episode["query_labels"] = episode["query_labels"].to(device)

    if "support_subgraphs" in episode and episode["support_subgraphs"] is not None:
        episode["support_subgraphs"] = [
            subgraph.to(device)
            for subgraph in episode["support_subgraphs"]
        ]

    if "query_subgraphs" in episode and episode["query_subgraphs"] is not None:
        episode["query_subgraphs"] = [
            subgraph.to(device)
            for subgraph in episode["query_subgraphs"]
        ]

    return episode


def debug_print(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[Diagnostics] {message}", flush=True)


def tensor_debug_stats(value) -> str:
    if not isinstance(value, torch.Tensor):
        return f"type={type(value).__name__}"

    return (
        f"shape={tuple(value.shape)}, dtype={value.dtype}, "
        f"device={value.device}, numel={value.numel()}"
    )


def data_debug_stats(data) -> str:
    parts = []

    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None:
        parts.append(f"num_nodes={num_nodes}")

    for name in [
        "x",
        "edge_index",
        "y",
        "raw_y",
        "anomaly_label",
        "anomaly_type",
        "type_y",
        "coarse_y",
    ]:
        if hasattr(data, name):
            value = getattr(data, name)
            if value is not None:
                parts.append(f"{name}=({tensor_debug_stats(value)})")

    return "; ".join(parts) if parts else f"type={type(data).__name__}"


def split_pools_debug_stats(split_pools) -> str:
    if split_pools is None:
        return "none"

    parts = []
    for split_name, class_map in split_pools.items():
        class_parts = []
        for class_id, indices in class_map.items():
            if isinstance(indices, torch.Tensor):
                class_parts.append(f"{class_id}:{indices.numel()}")
            else:
                class_parts.append(f"{class_id}:{len(indices)}")
        parts.append(f"{split_name}={{{', '.join(class_parts)}}}")

    return "; ".join(parts)


def subgraph_list_debug_stats(value) -> str:
    if not isinstance(value, list):
        return f"type={type(value).__name__}"

    parts = [f"count={len(value)}"]
    if value:
        parts.append(f"first=({data_debug_stats(value[0])})")

    return "; ".join(parts)


def episode_debug_stats(episode: Dict) -> str:
    parts = [f"keys={sorted(episode.keys())}"]

    for name in [
        "classes",
        "support_indices",
        "query_indices",
        "support_labels",
        "query_labels",
    ]:
        if name in episode:
            parts.append(f"{name}=({tensor_debug_stats(episode[name])})")

    for name in ["support_subgraphs", "query_subgraphs"]:
        if name in episode:
            parts.append(f"{name}=({subgraph_list_debug_stats(episode[name])})")

    return "; ".join(parts)


def output_debug_stats(output: Dict) -> str:
    parts = []

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            parts.append(f"{key}=({tensor_debug_stats(value)})")
        else:
            parts.append(f"{key}=type({type(value).__name__})")

    return "; ".join(parts)


# ============================================================
# Data / Sampler Builder
# ============================================================

def build_data_and_sampler(
    config: Dict,
    seed: int,
    debug_diagnostics: bool = False,
):
    """
    Load dataset, inject anomalies, and build EpisodeSampler.
    """
    dataset_name = get_from_config(config, "dataset.name", "Cora")
    dataset_root = get_from_config(config, "dataset.root", "./data/raw")
    normalize_features = get_from_config(
        config,
        "dataset.normalize_features",
        True,
    )

    anomaly_config = config.get("anomaly", {})
    synthetic_anomaly_enabled = anomaly_config.get("enabled", False)

    dataset_key = dataset_name.strip().lower()
    debug_print(
        debug_diagnostics,
        (
            "before dataset loading: "
            f"name={dataset_name}, root={dataset_root}, "
            f"normalize_features={normalize_features}"
        ),
    )

    if dataset_key in PLANETOID_DATASETS:
        dataset, data = load_planetoid_dataset(
            name=dataset_name,
            root=dataset_root,
            normalize_features=normalize_features,
        )
    elif dataset_key in MAT_DATASETS:
        dataset, data = load_mat_dataset(
            root=dataset_root,
            name=dataset_name,
            normalize_features=normalize_features,
            debug_diagnostics=debug_diagnostics,
        )
    else:
        supported = sorted(PLANETOID_DATASETS | MAT_DATASETS)
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Supported datasets: {supported}."
        )

    debug_print(
        debug_diagnostics,
        f"after dataset loading: {data_debug_stats(data)}",
    )

    print_dataset_load_summary(
        dataset=dataset,
        data=data,
        synthetic_anomaly_enabled=synthetic_anomaly_enabled,
    )

    debug_print(
        debug_diagnostics,
        (
            "before anomaly injection: "
            f"enabled={synthetic_anomaly_enabled}, {data_debug_stats(data)}"
        ),
    )

    if synthetic_anomaly_enabled:
        data = inject_multi_type_anomalies(
            data=data,
            anomaly_ratio=anomaly_config.get("ratio", 0.05),
            type_ratio=anomaly_config.get("type_ratio", None),
            structural_edges_per_node=anomaly_config.get(
                "structural_edges_per_node",
                10,
            ),
            attribute_noise_scale=anomaly_config.get(
                "attribute_noise_scale",
                1.0,
            ),
            attribute_mode=anomaly_config.get(
                "attribute_mode",
                "gaussian",
            ),
            seed=seed,
        )

        data = attach_synthetic_label_aliases(data)

    debug_print(
        debug_diagnostics,
        f"after anomaly injection: {data_debug_stats(data)}",
    )

    subgraph_config = config.get("subgraph", {})
    episode_config = config.get("episode", {})
    split_config = config.get("split", {})
    large_graph_debug_safe_mode = (
        debug_diagnostics
        and dataset_key in SAFE_SAMPLING_DATASETS
    )
    safe_subgraph_sampling = bool(
        subgraph_config.get("safe_mode", large_graph_debug_safe_mode)
    )
    allow_replacement = bool(episode_config.get("allow_replacement", False))
    num_hops = int(subgraph_config.get("num_hops", 2))
    force_large_graph_hops = subgraph_config.get(
        "force_num_hops_for_large_graphs",
        None,
    )
    if dataset_key in SAFE_SAMPLING_DATASETS and force_large_graph_hops is not None:
        num_hops = int(force_large_graph_hops)

    max_nodes_per_subgraph = subgraph_config.get("max_nodes_per_subgraph", None)
    max_edges_per_subgraph = subgraph_config.get("max_edges_per_subgraph", None)
    neighbor_sample_per_hop = subgraph_config.get("neighbor_sample_per_hop", None)

    if max_nodes_per_subgraph is not None:
        max_nodes_per_subgraph = int(max_nodes_per_subgraph)

    if max_edges_per_subgraph is not None:
        max_edges_per_subgraph = int(max_edges_per_subgraph)

    if neighbor_sample_per_hop is not None:
        neighbor_sample_per_hop = int(neighbor_sample_per_hop)

    debug_print(
        debug_diagnostics,
        (
            "sampler options: "
            f"num_hops={num_hops}, "
            f"safe_subgraph_sampling={safe_subgraph_sampling}, "
            f"allow_replacement={allow_replacement}, "
            f"max_nodes_per_subgraph={max_nodes_per_subgraph}, "
            f"max_edges_per_subgraph={max_edges_per_subgraph}, "
            f"neighbor_sample_per_hop={neighbor_sample_per_hop}"
        ),
    )

    debug_print(
        debug_diagnostics,
        (
            "before split: "
            f"enabled={split_config.get('enabled', False)}, "
            f"seed={split_config.get('seed', seed)}"
        ),
    )

    split_pools = build_or_load_split_pools(
        data=data,
        split_config=split_config,
        default_seed=seed,
        classes=[0, 1, 2, 3],
    )

    debug_print(
        debug_diagnostics,
        f"after split: {split_pools_debug_stats(split_pools)}",
    )

    if split_pools is not None:
        print_split_summary(split_pools)

    sampler = EpisodeSampler(
        data=data,
        num_hops=num_hops,
        relabel_nodes=subgraph_config.get("relabel_nodes", True),
        task=episode_config.get("task", "multiclass"),
        n_way=episode_config.get("n_way", 4),
        k_shot=episode_config.get("k_shot", 5),
        q_query=episode_config.get("q_query", 10),
        seed=episode_config.get("seed", seed),
        split_pools=split_pools,
        default_split=episode_config.get("default_split", "train"),
        dataset_name=dataset_name,
        debug_diagnostics=debug_diagnostics,
        safe_subgraph_sampling=safe_subgraph_sampling,
        allow_replacement=allow_replacement,
        max_nodes_per_subgraph=max_nodes_per_subgraph,
        max_edges_per_subgraph=max_edges_per_subgraph,
        neighbor_sample_per_hop=neighbor_sample_per_hop,
    )

    return dataset, data, sampler


# ============================================================
# Metric Helpers
# ============================================================

def tensor_to_float(value):
    """
    Convert tensor/scalar to Python float.
    """
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def init_metric_accumulator() -> Dict[str, float]:
    return {
        "loss": 0.0,
        "coarse_loss": 0.0,
        "fine_loss": 0.0,
        "accuracy": 0.0,
        "coarse_accuracy": 0.0,
        "fine_accuracy": 0.0,
        "type_aware_loss": 0.0,
        "type_accuracy": 0.0,
    }


def update_metric_accumulator(
    accumulator: Dict[str, float],
    output: Dict,
) -> None:
    """
    Update metric accumulator.
    """
    required_keys = [
        "loss",
        "coarse_loss",
        "fine_loss",
        "accuracy",
        "coarse_accuracy",
        "fine_accuracy",
    ]

    for key in required_keys:
        accumulator[key] += tensor_to_float(output[key])

    optional_keys = [
        "type_aware_loss",
        "type_accuracy",
    ]

    for key in optional_keys:
        if key in output:
            accumulator[key] += tensor_to_float(output[key])


def average_metric_accumulator(
    accumulator: Dict[str, float],
    count: int,
) -> Dict[str, float]:
    if count <= 0:
        raise ValueError("count must be positive.")

    return {
        key: value / count
        for key, value in accumulator.items()
    }


def _to_cpu_list(tensor: torch.Tensor) -> List[int]:
    return tensor.detach().cpu().long().tolist()


def update_prediction_collector(
    collector: Dict[str, List],
    output: Dict[str, torch.Tensor],
    query_labels: torch.Tensor,
) -> None:
    """
    Collect predictions and targets for global metric computation.
    """
    query_labels_cpu = query_labels.detach().cpu().long()

    if "predictions" in output:
        collector["y_true"].extend(query_labels_cpu.tolist())
        predictions = output["predictions"].detach().cpu().long()
        collector["y_pred"].extend(predictions.tolist())

        probs = None
        if "probs" in output:
            probs = output["probs"].detach().cpu().float()
        elif "logits" in output:
            probs = torch.softmax(output["logits"].detach().cpu().float(), dim=-1)

        if (
            probs is not None
            and probs.dim() == 2
            and probs.size(0) == query_labels_cpu.size(0)
        ):
            collector.setdefault("y_prob", []).extend(probs.tolist())

        coarse_targets = (query_labels_cpu != 0).long()
        coarse_predictions = (predictions != 0).long()
        collector["coarse_y_true"].extend(coarse_targets.tolist())
        collector["coarse_y_pred"].extend(coarse_predictions.tolist())

        if "anomaly_score" in output:
            anomaly_score = output["anomaly_score"].detach().cpu().float()
            collector["coarse_y_score"].extend(anomaly_score.tolist())

    if "type_pred" in output:
        collector["type_y_true"].extend(query_labels_cpu.tolist())
        collector["type_y_pred"].extend(_to_cpu_list(output["type_pred"]))


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    sampler: EpisodeSampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    episodes_per_epoch: int,
    grad_clip: float = 0.0,
    split: str = "train",
    debug_diagnostics: bool = False,
) -> Dict[str, float]:
    """
    Train model for one epoch using episodic training.
    """
    model.train()

    metrics = init_metric_accumulator()

    debug_print(
        debug_diagnostics,
        (
            "before first episode sampling: "
            f"split={split}, num_episodes={episodes_per_epoch}"
        ),
    )
    for episode_index in range(episodes_per_epoch):
        episode = sampler.sample_episode(split=split)

        if debug_diagnostics and episode_index == 0:
            debug_print(
                True,
                f"after first episode sampling: {episode_debug_stats(episode)}",
            )

        episode = move_episode_labels_to_device(episode, device)

        optimizer.zero_grad()

        if debug_diagnostics and episode_index == 0:
            debug_print(
                True,
                f"before first forward/backward: {episode_debug_stats(episode)}",
            )

        output = model(episode)

        if debug_diagnostics and episode_index == 0:
            debug_print(
                True,
                f"after first forward: {output_debug_stats(output)}",
            )

        loss = output["loss"]

        loss.backward()

        if debug_diagnostics and episode_index == 0:
            debug_print(True, "after first backward")

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        update_metric_accumulator(metrics, output)

        del output
        del loss
        del episode

    return average_metric_accumulator(
        accumulator=metrics,
        count=episodes_per_epoch,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    sampler: EpisodeSampler,
    device: torch.device,
    num_episodes: int,
    num_classes: int = 4,
    normal_class_id: int = 0,
    split: str = "val",
    debug_diagnostics: bool = False,
    debug_label: str = "evaluation",
    fixed_episodes: Optional[Sequence[Dict[str, object]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on sampled episodes.

    V2-7:
        Besides episode-averaged losses, this function also collects
        predictions across all evaluation episodes and computes global
        macro-F1, weighted-F1, balanced accuracy, and confusion matrix.
    """
    model.eval()

    use_fixed_episodes = fixed_episodes is not None
    effective_num_episodes = (
        len(fixed_episodes)
        if fixed_episodes is not None
        else int(num_episodes)
    )
    if effective_num_episodes <= 0:
        raise ValueError("Evaluation requires at least one episode.")

    metrics = init_metric_accumulator()

    collector = {
        "y_true": [],
        "y_pred": [],
        "y_prob": [],
        "type_y_true": [],
        "type_y_pred": [],
        "coarse_y_true": [],
        "coarse_y_pred": [],
        "coarse_y_score": [],
    }

    debug_print(
        debug_diagnostics,
        (
            f"before evaluation: label={debug_label}, "
            f"split={split}, num_episodes={effective_num_episodes}, "
            f"fixed_episodes={use_fixed_episodes}"
        ),
    )
    debug_print(
        debug_diagnostics,
        (
            "before evaluation episode iteration: "
            f"split={split}, num_episodes={effective_num_episodes}, "
            f"fixed_episodes={use_fixed_episodes}"
        ),
    )
    for episode_index in range(effective_num_episodes):
        if fixed_episodes is None:
            episode = sampler.sample_episode(split=split)
        else:
            episode = fixed_episodes[episode_index]

        if debug_diagnostics and episode_index == 0:
            debug_print(
                True,
                f"after evaluation episode load: {episode_debug_stats(episode)}",
            )

        episode = move_episode_labels_to_device(episode, device)

        if debug_diagnostics and episode_index == 0:
            debug_print(
                True,
                f"before evaluation forward: {episode_debug_stats(episode)}",
            )

        output = model(episode)

        if debug_diagnostics and episode_index == 0:
            debug_print(
                True,
                f"after evaluation forward: {output_debug_stats(output)}",
            )

        update_metric_accumulator(metrics, output)

        query_labels = episode.get("query_labels", None)

        if query_labels is not None:
            update_prediction_collector(
                collector=collector,
                output=output,
                query_labels=query_labels,
            )

        del output
        del episode

    averaged_metrics = average_metric_accumulator(
        accumulator=metrics,
        count=effective_num_episodes,
    )

    if len(collector["y_true"]) > 0:
        class_metrics = compute_classification_metrics(
            y_true=collector["y_true"],
            y_pred=collector["y_pred"],
            num_classes=num_classes,
            prefix="",
            class_names=[
                "normal",
                "structural",
                "attribute",
                "hybrid",
            ],
        )
        averaged_metrics.update(class_metrics)

    if len(collector["type_y_true"]) > 0:
        type_metrics = compute_classification_metrics(
            y_true=collector["type_y_true"],
            y_pred=collector["type_y_pred"],
            num_classes=num_classes,
            prefix="type_",
            class_names=[
                "normal",
                "structural",
                "attribute",
                "hybrid",
            ],
        )
        averaged_metrics.update(type_metrics)

    if len(collector["coarse_y_true"]) > 0:
        coarse_metrics = compute_binary_anomaly_metrics(
            y_true=collector["coarse_y_true"],
            y_pred=collector["coarse_y_pred"],
            y_score=(
                collector["coarse_y_score"]
                if len(collector["coarse_y_score"]) > 0
                else None
            ),
            prefix="coarse_",
        )
        averaged_metrics.update(coarse_metrics)

    if len(collector["y_prob"]) > 0:
        official_metrics = compute_metrics(
            y_true=collector["y_true"],
            y_prob=collector["y_prob"],
            normal_class_id=normal_class_id,
        )
        averaged_metrics.update(official_metrics)

    debug_print(
        debug_diagnostics,
        f"after evaluation: label={debug_label}, keys={sorted(averaged_metrics.keys())}",
    )

    return averaged_metrics


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: Dict,
) -> None:
    """
    Save checkpoint.
    """
    ensure_dir(os.path.dirname(path))

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "config": config,
    }

    torch.save(checkpoint, path)


def is_better_metric(
    current: float,
    best: Optional[float],
    mode: str,
    min_delta: float = 0.0,
) -> bool:
    if best is None:
        return True

    if mode == "max":
        return current > best + min_delta

    if mode == "min":
        return current < best - min_delta

    raise ValueError(f"Unsupported mode: {mode}")


# ============================================================
# Main Training Pipeline
# ============================================================

def train(config: Dict, debug_diagnostics: bool = False) -> None:
    """
    Main training entry.
    """
    # --------------------------------------------------------
    # Runtime config
    # --------------------------------------------------------
    train_start_time = time.perf_counter()

    project_name = get_from_config(config, "project.name", "FMGAD")
    project_version = get_from_config(config, "project.version", "Unknown")
    project_stage = get_from_config(config, "project.stage", "Unknown")

    seed = get_from_config(config, "runtime.seed", 42)
    device_config = get_from_config(config, "runtime.device", "auto")
    device = resolve_device(device_config)

    set_seed(seed)

    training_config = config.get("training", {})
    model_config = config.get("model", {})
    experiment_config = config.get("experiment", {})
    evaluation_config = config.setdefault("evaluation", {})
    evaluation_config.setdefault("fixed_eval_episodes", True)
    evaluation_config.setdefault("episode_cache_dir", "outputs/episode_cache")
    evaluation_config.setdefault("regenerate_eval_episodes", False)

    split_config = config.get("split", {})
    train_split = split_config.get("train_split", "train")
    eval_split = split_config.get("eval_split", "val")

    use_type_aware_head = model_config.get("use_type_aware_head", False)

    if not training_config.get("enabled", True):
        print("Training is disabled in config.")
        return

    if not training_config.get("episode_training", True):
        raise NotImplementedError(
            "training.episode_training=False was requested, but train.py "
            "currently implements only episodic training."
        )

    run_name = experiment_config.get(
        "run_name",
        f"{project_version}_{project_stage}_seed{seed}",
    )

    output_root = experiment_config.get("output_root", "results/v2_7")
    output_dir = experiment_config.get(
        "output_dir",
        os.path.join(output_root, run_name),
    )

    model_name = training_config.get("model", "fmgad")
    model_name = model_name.replace("-", "_")

    num_epochs = training_config.get("num_epochs", 50)
    episodes_per_epoch = training_config.get("episodes_per_epoch", 20)
    eval_episodes = training_config.get("eval_episodes", 20)
    learning_rate = training_config.get("learning_rate", 1e-3)
    weight_decay = training_config.get("weight_decay", 5e-4)
    grad_clip = training_config.get("grad_clip", 5.0)
    log_interval = training_config.get("log_interval", 1)
    eval_interval = training_config.get("eval_interval", 5)
    checkpoint_interval = training_config.get("checkpoint_interval", 10)

    checkpoint_dir = training_config.get("checkpoint_dir", None)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(output_dir, "checkpoints")

    save_best = training_config.get("save_best", True)

    num_classes = evaluation_config.get("num_classes", 4)
    normal_class_id = int(evaluation_config.get("normal_class_id", 0))
    primary_metric = evaluation_config.get("primary_metric", "macro_f1")

    metric_for_best = training_config.get(
        "metric_for_best",
        primary_metric,
    )
    metric_mode = training_config.get("metric_mode", "max")

    ensure_dir(output_dir)
    ensure_dir(checkpoint_dir)

    metrics_csv_path = os.path.join(output_dir, "metrics_per_epoch.csv")
    best_metrics_path = os.path.join(output_dir, "best_metrics.json")
    final_metrics_path = os.path.join(output_dir, "final_metrics.json")
    run_config_path = os.path.join(output_dir, "config.yaml")

    save_yaml(run_config_path, config)

    print("========== FMGAD Training ==========")
    print(f"Project       : {project_name}")
    print(f"Version       : {project_version}")
    print(f"Stage         : {project_stage}")
    print(f"Model         : {model_name}")
    print(f"Seed          : {seed}")
    print(f"Device        : {device}")
    print(f"Epochs        : {num_epochs}")
    print(f"Episodes/epoch: {episodes_per_epoch}")
    print(f"Eval episodes : {eval_episodes}")
    print(f"LR            : {learning_rate}")
    print(f"Weight decay  : {weight_decay}")
    print(f"Run name      : {run_name}")
    print(f"Output dir    : {output_dir}")
    print(f"Best metric   : {metric_for_best} ({metric_mode})")

    if split_config.get("enabled", False):
        print(f"Split enabled : true")
        print(f"Train split   : {train_split}")
        print(f"Eval split    : {eval_split}")
    else:
        print(f"Split enabled : false")

    print("=======================================")

    # --------------------------------------------------------
    # Data and sampler
    # --------------------------------------------------------

    debug_print(debug_diagnostics, "before dataset/sampler build")
    dataset, data, sampler = build_data_and_sampler(
        config=config,
        seed=seed,
        debug_diagnostics=debug_diagnostics,
    )
    debug_print(
        debug_diagnostics,
        f"after dataset/sampler build: {data_debug_stats(data)}",
    )

    if hasattr(data, "anomaly_label"):
        anomaly_summary = summarize_anomalies(data)
        print("Anomaly Summary:")
        print(f"  Num anomalies : {anomaly_summary['num_anomalies']}")
        print(f"  Num normal    : {anomaly_summary['num_normal']}")
        print(f"  Structural    : {anomaly_summary['num_structural']}")
        print(f"  Attribute     : {anomaly_summary['num_attribute']}")
        print(f"  Hybrid        : {anomaly_summary['num_hybrid']}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if model_name not in ["hierarchical_meta", "fmgad"]:
        raise ValueError(
            f"Unsupported training model: {model_name}. "
            "Currently supported: 'hierarchical_meta', 'fmgad'."
        )

    debug_print(
        debug_diagnostics,
        (
            "before model init: "
            f"model={model_name}, input_dim={data.num_features}, device={device}"
        ),
    )
    model = FMGAD(
        config=config,
        input_dim=data.num_features,
    ).to(device)

    num_trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    debug_print(
        debug_diagnostics,
        f"after model init: trainable_params={num_trainable_params}, device={device}",
    )

    print("Model Summary:")
    print(f"  Trainable params: {num_trainable_params}")

    fixed_eval_episodes, eval_episode_cache_path = (
        get_fixed_eval_episodes_from_config(
            config=config,
            sampler=sampler,
            split_name=eval_split,
            num_episodes=eval_episodes,
            default_seed=seed,
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = None
    scheduler_config = training_config.get("scheduler", {})

    if scheduler_config.get("enabled", False):
        scheduler_type = scheduler_config.get("type", "reduce_on_plateau")

        if scheduler_type != "reduce_on_plateau":
            raise ValueError(
                f"Unsupported scheduler type: {scheduler_type}. "
                "Currently only reduce_on_plateau is supported."
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 10),
            min_lr=scheduler_config.get("min_lr", 1e-5),
        )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    early_stopper = None
    early_config = training_config.get("early_stopping", {})

    if early_config.get("enabled", False):
        early_stopper = EarlyStopping(
            patience=early_config.get("patience", 20),
            mode=early_config.get("mode", metric_mode),
            min_delta=early_config.get("min_delta", 0.0),
        )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_metric_value = None
    best_eval_metrics = None
    best_checkpoint_path = None
    last_eval_metrics = None
    stopped_early = False
    final_epoch = 0

    for epoch in range(1, num_epochs + 1):
        final_epoch = epoch

        train_metrics = train_one_epoch(
            model=model,
            sampler=sampler,
            optimizer=optimizer,
            device=device,
            episodes_per_epoch=episodes_per_epoch,
            grad_clip=grad_clip,
            split=train_split,
            debug_diagnostics=debug_diagnostics and epoch == 1,
        )

        if epoch % log_interval == 0:
            print(f"Epoch {epoch:03d}/{num_epochs}")
            print(f"  Train loss        : {train_metrics['loss']:.4f}")
            print(f"  Train coarse loss : {train_metrics['coarse_loss']:.4f}")
            print(f"  Train fine loss   : {train_metrics['fine_loss']:.4f}")
            print(f"  Train accuracy    : {train_metrics['accuracy']:.4f}")
            print(f"  Train coarse acc  : {train_metrics['coarse_accuracy']:.4f}")
            print(f"  Train fine acc    : {train_metrics['fine_accuracy']:.4f}")

            if use_type_aware_head:
                print(
                    f"  Train type loss   : "
                    f"{train_metrics['type_aware_loss']:.4f}"
                )
                print(
                    f"  Train type acc    : "
                    f"{train_metrics['type_accuracy']:.4f}"
                )

        if epoch % eval_interval == 0:
            eval_metrics = evaluate(
                model=model,
                sampler=sampler,
                device=device,
                num_episodes=eval_episodes,
                num_classes=num_classes,
                normal_class_id=normal_class_id,
                split=eval_split,
                debug_diagnostics=debug_diagnostics,
                debug_label=f"epoch_{epoch}_{eval_split}",
                fixed_episodes=fixed_eval_episodes,
            )

            if eval_episode_cache_path is not None:
                eval_metrics["episode_cache_path"] = eval_episode_cache_path

            last_eval_metrics = dict(eval_metrics)

            print("  Eval:")
            print(f"    Eval loss       : {eval_metrics['loss']:.4f}")
            print(f"    Eval coarse loss: {eval_metrics['coarse_loss']:.4f}")
            print(f"    Eval fine loss  : {eval_metrics['fine_loss']:.4f}")
            print(f"    Eval accuracy   : {eval_metrics['accuracy']:.4f}")
            print(f"    Eval coarse acc : {eval_metrics['coarse_accuracy']:.4f}")
            print(f"    Eval fine acc   : {eval_metrics['fine_accuracy']:.4f}")

            if "macro_f1" in eval_metrics:
                print(f"    Eval macro F1   : {eval_metrics['macro_f1']:.4f}")

            if "weighted_f1" in eval_metrics:
                print(f"    Eval weighted F1: {eval_metrics['weighted_f1']:.4f}")

            if "balanced_accuracy" in eval_metrics:
                print(
                    f"    Eval balanced acc: "
                    f"{eval_metrics['balanced_accuracy']:.4f}"
                )

            if use_type_aware_head:
                print(
                    f"    Eval type loss  : "
                    f"{eval_metrics['type_aware_loss']:.4f}"
                )
                print(
                    f"    Eval type acc   : "
                    f"{eval_metrics['type_accuracy']:.4f}"
                )

            # ------------------------------------------------
            # Save epoch metrics
            # ------------------------------------------------

            metric_row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }

            for key, value in train_metrics.items():
                metric_row[f"train_{key}"] = value

            for key, value in eval_metrics.items():
                metric_row[f"eval_{key}"] = value

            debug_print(
                debug_diagnostics,
                f"before metrics writing: path={metrics_csv_path}",
            )
            append_metrics_to_csv(
                path=metrics_csv_path,
                row=metric_row,
            )
            debug_print(
                debug_diagnostics,
                f"after metrics writing: path={metrics_csv_path}",
            )

            # ------------------------------------------------
            # Best checkpoint by selected metric
            # ------------------------------------------------

            if metric_for_best not in eval_metrics:
                raise KeyError(
                    f"Best metric '{metric_for_best}' not found in eval_metrics. "
                    f"Available keys: {list(eval_metrics.keys())}"
                )

            current_metric_value = float(eval_metrics[metric_for_best])

            if save_best and is_better_metric(
                current=current_metric_value,
                best=best_metric_value,
                mode=metric_mode,
            ):
                best_metric_value = current_metric_value
                best_eval_metrics = dict(eval_metrics)
                best_eval_metrics["best_epoch"] = epoch
                best_eval_metrics["best_metric"] = metric_for_best
                best_eval_metrics["best_metric_value"] = best_metric_value

                best_path = os.path.join(
                    checkpoint_dir,
                    "best_model.pt",
                )
                best_checkpoint_path = best_path

                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_metric=best_metric_value,
                    config=config,
                )

                debug_print(
                    debug_diagnostics,
                    f"before metrics writing: path={best_metrics_path}",
                )
                save_json(
                    path=best_metrics_path,
                    obj=best_eval_metrics,
                )
                debug_print(
                    debug_diagnostics,
                    f"after metrics writing: path={best_metrics_path}",
                )

                print(
                    f"    [Checkpoint] Saved best model to {best_path} "
                    f"({metric_for_best}={best_metric_value:.4f})"
                )

            # ------------------------------------------------
            # Scheduler step
            # ------------------------------------------------

            if scheduler is not None:
                scheduler_monitor = scheduler_config.get(
                    "monitor",
                    "loss",
                )

                if scheduler_monitor not in eval_metrics:
                    raise KeyError(
                        f"Scheduler monitor '{scheduler_monitor}' not found "
                        f"in eval_metrics. Available keys: "
                        f"{list(eval_metrics.keys())}"
                    )

                scheduler.step(float(eval_metrics[scheduler_monitor]))

            # ------------------------------------------------
            # Early stopping step
            # ------------------------------------------------

            if early_stopper is not None:
                should_stop = early_stopper.step(current_metric_value)

                if should_stop:
                    stopped_early = True
                    print(
                        f"    [EarlyStopping] Stop at epoch {epoch}. "
                        f"Best {metric_for_best}: "
                        f"{early_stopper.best_score:.4f}"
                    )
                    break

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        if epoch % checkpoint_interval == 0:
            latest_path = os.path.join(
                checkpoint_dir,
                "latest.pt",
            )

            save_checkpoint(
                path=latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_metric=(
                    best_metric_value
                    if best_metric_value is not None
                    else float("-inf")
                ),
                config=config,
            )

            print(f"  [Checkpoint] Saved latest model to {latest_path}")

    # --------------------------------------------------------
    # Final checkpoint
    # --------------------------------------------------------

    final_path = os.path.join(
        checkpoint_dir,
        "final.pt",
    )

    save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        epoch=final_epoch,
        best_metric=(
            best_metric_value
            if best_metric_value is not None
            else float("-inf")
        ),
        config=config,
    )

    # --------------------------------------------------------
    # Final test evaluation from best checkpoint
    # --------------------------------------------------------

    final_test_metrics = None
    test_metrics_path = os.path.join(output_dir, "test_metrics.json")
    compact_metrics_path = os.path.join(output_dir, "metrics.json")

    resolved_best_checkpoint_path = best_checkpoint_path

    if resolved_best_checkpoint_path is None:
        fallback_best_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(fallback_best_path):
            resolved_best_checkpoint_path = fallback_best_path

    if resolved_best_checkpoint_path is not None and os.path.exists(
        resolved_best_checkpoint_path
    ):
        checkpoint = torch.load(
            resolved_best_checkpoint_path,
            map_location=device,
        )

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            test_checkpoint_epoch = checkpoint.get("epoch", None)
            test_checkpoint_best_metric = checkpoint.get("best_metric", None)
        else:
            model.load_state_dict(checkpoint)
            test_checkpoint_epoch = None
            test_checkpoint_best_metric = None

        fixed_test_episodes, test_episode_cache_path = (
            get_fixed_eval_episodes_from_config(
                config=config,
                sampler=sampler,
                split_name="test",
                num_episodes=eval_episodes,
                default_seed=seed,
            )
        )

        final_test_metrics = evaluate(
            model=model,
            sampler=sampler,
            device=device,
            num_episodes=eval_episodes,
            num_classes=num_classes,
            normal_class_id=normal_class_id,
            split="test",
            debug_diagnostics=debug_diagnostics,
            debug_label="final_test",
            fixed_episodes=fixed_test_episodes,
        )

        final_test_metrics["eval_split"] = "test"
        final_test_metrics["num_eval_episodes"] = (
            len(fixed_test_episodes)
            if fixed_test_episodes is not None
            else eval_episodes
        )
        if test_episode_cache_path is not None:
            final_test_metrics["episode_cache_path"] = test_episode_cache_path
        final_test_metrics["checkpoint"] = resolved_best_checkpoint_path
        final_test_metrics["checkpoint_epoch"] = test_checkpoint_epoch
        final_test_metrics["checkpoint_best_metric"] = test_checkpoint_best_metric

        debug_print(
            debug_diagnostics,
            f"before metrics writing: path={test_metrics_path}",
        )
        save_json(
            path=test_metrics_path,
            obj=final_test_metrics,
        )
        debug_print(
            debug_diagnostics,
            f"after metrics writing: path={test_metrics_path}",
        )

        print("Final Test:")
        print(f"  Checkpoint      : {resolved_best_checkpoint_path}")
        print(f"  Test loss       : {final_test_metrics['loss']:.4f}")
        print(f"  Test accuracy   : {final_test_metrics['accuracy']:.4f}")

        if "macro_f1" in final_test_metrics:
            print(f"  Test macro F1   : {final_test_metrics['macro_f1']:.4f}")

        if "coarse_auc_roc" in final_test_metrics:
            print(f"  Test AUC-ROC    : {final_test_metrics['coarse_auc_roc']}")

        if "coarse_auc_pr" in final_test_metrics:
            print(f"  Test AUC-PR     : {final_test_metrics['coarse_auc_pr']}")

    else:
        print(
            "[Warning] Best checkpoint was not found. "
            "Skipping final test evaluation."
        )

    train_time_sec = float(time.perf_counter() - train_start_time)
    dataset_name = str(get_from_config(config, "dataset.name", "unknown")).lower()
    best_epoch = None
    if isinstance(best_eval_metrics, dict):
        best_epoch = best_eval_metrics.get("best_epoch")
    if best_epoch is None and final_test_metrics is not None:
        best_epoch = final_test_metrics.get("checkpoint_epoch")
    if best_epoch is None:
        best_epoch = final_epoch

    metric_source = final_test_metrics if isinstance(final_test_metrics, dict) else {}

    def metric_float(key: str) -> float:
        value = metric_source.get(key, float("nan"))
        if value is None:
            return float("nan")
        return float(value)

    def metric_float_any(*keys: str) -> float:
        for key in keys:
            value = metric_source.get(key)
            if value is not None:
                return float(value)
        return float("nan")

    compact_metrics = {
        "dataset": dataset_name,
        "model": "FMGAD",
        "seed": seed,
        "macro_f1": metric_float("macro_f1"),
        "macro_auroc": metric_float("macro_auroc"),
        "macro_auprc": metric_float("macro_auprc"),
        "binary_f1": metric_float_any(
            "binary_f1",
            "coarse_f1_class_1_anomaly",
        ),
        "binary_macro_f1": metric_float_any(
            "binary_macro_f1",
            "coarse_macro_f1",
        ),
        "binary_auroc": metric_float("binary_auroc"),
        "binary_auprc": metric_float("binary_auprc"),
        "train_time_sec": train_time_sec,
        "best_epoch": best_epoch,
    }

    final_metrics = {
        "best_metric": metric_for_best,
        "best_metric_value": best_metric_value,
        "best_eval_metrics": best_eval_metrics,
        "best_checkpoint": resolved_best_checkpoint_path,
        "last_eval_metrics": last_eval_metrics,
        "final_test_metrics": final_test_metrics,
        "final_epoch": final_epoch,
        "train_time_sec": train_time_sec,
        "best_epoch": best_epoch,
        "metrics_json": compact_metrics_path,
        "stopped_early": stopped_early,
        "final_checkpoint": final_path,
    }

    if early_stopper is not None:
        final_metrics["early_stopping"] = early_stopper.state_dict()

    debug_print(
        debug_diagnostics,
        f"before metrics writing: path={final_metrics_path}",
    )
    save_json(
        path=final_metrics_path,
        obj=final_metrics,
    )
    save_json(
        path=compact_metrics_path,
        obj=compact_metrics,
    )
    debug_print(
        debug_diagnostics,
        f"after metrics writing: path={final_metrics_path}",
    )

    print("=======================================")
    print("[OK] Training completed successfully.")

    if best_metric_value is not None:
        print(f"Best {metric_for_best}: {best_metric_value:.4f}")
    else:
        print(f"Best {metric_for_best}: N/A")

    print(f"Final checkpoint  : {final_path}")
    print(f"Metrics CSV       : {metrics_csv_path}")
    print(f"Best metrics JSON : {best_metrics_path}")
    print(f"Test metrics JSON : {test_metrics_path}")
    print(f"Metrics JSON      : {compact_metrics_path}")
    print(f"Final metrics JSON: {final_metrics_path}")


# ============================================================
# Entry Point
# ============================================================

def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)
    train(config, debug_diagnostics=args.debug_diagnostics)


if __name__ == "__main__":
    main()
