import argparse
import os
from typing import Any, Dict, List, Optional

import torch

from data.subgraph_sampler import sample_multiple_center_aware_subgraphs
from data.episode_cache import get_fixed_eval_episodes_from_config
from models.fmgad import FMGAD
from train import (
    build_data_and_sampler,
    evaluate as evaluate_episodes,
    resolve_device,
)
from utils.config import get_from_config, load_config, set_in_config
from utils.experiment_io import save_json
from utils.metrics import (
    compute_binary_anomaly_metrics,
    compute_classification_metrics,
    compute_metrics,
)
from utils.seed import set_seed


CLASS_NAMES = [
    "normal",
    "structural",
    "attribute",
    "hybrid",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate FMGAD checkpoints on episodic or full splits."
    )

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], required=True)
    parser.add_argument("--mode", type=str, choices=["episodic", "full"], required=True)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)

    return parser.parse_args()


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        return {
            "checkpoint_epoch": checkpoint.get("epoch", None),
            "checkpoint_best_metric": checkpoint.get("best_metric", None),
        }

    model.load_state_dict(checkpoint)
    return {
        "checkpoint_epoch": None,
        "checkpoint_best_metric": None,
    }


def _sample_fixed_train_support(
    sampler,
    seed: int,
) -> Dict[str, Any]:
    if sampler.split_pools is None:
        raise ValueError(
            "Full evaluation requires split-aware pools. "
            "Refusing to fall back to full-node random sampling."
        )

    if "train" not in sampler.split_pools:
        raise ValueError("Full evaluation requires a 'train' split pool.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    support_indices_by_class: Dict[int, List[int]] = {}
    support_indices = []
    support_labels = []

    for class_id in [0, 1, 2, 3]:
        class_pool = sampler.split_pools["train"][class_id].detach().cpu().long()

        if class_pool.numel() < sampler.k_shot:
            raise ValueError(
                f"Train split class {class_id} has only {class_pool.numel()} "
                f"nodes, but full evaluation requires k_shot={sampler.k_shot}."
            )

        perm = torch.randperm(class_pool.numel(), generator=generator)
        selected = class_pool[perm[: sampler.k_shot]].long()

        support_indices_by_class[class_id] = [
            int(v)
            for v in selected.tolist()
        ]
        support_indices.append(selected)
        support_labels.append(
            torch.full(
                size=(sampler.k_shot,),
                fill_value=class_id,
                dtype=torch.long,
            )
        )

    flat_support_indices = torch.cat(support_indices, dim=0)
    flat_support_labels = torch.cat(support_labels, dim=0)

    support_subgraphs = sample_multiple_center_aware_subgraphs(
        data=sampler.data,
        center_nodes=flat_support_indices,
        num_hops=sampler.num_hops,
        relabel_nodes=sampler.relabel_nodes,
        max_samples=flat_support_indices.numel(),
        safe_mode=sampler.safe_subgraph_sampling, seed=sampler.seed,
        max_nodes_per_subgraph=sampler.max_nodes_per_subgraph,
        max_edges_per_subgraph=sampler.max_edges_per_subgraph,
        neighbor_sample_per_hop=sampler.neighbor_sample_per_hop,
    )

    return {
        "support_indices": flat_support_indices,
        "support_labels": flat_support_labels,
        "support_subgraphs": support_subgraphs,
        "support_indices_by_class": support_indices_by_class,
    }


def _collect_query_nodes(
    sampler,
    split: str,
) -> Dict[str, torch.Tensor]:
    if sampler.split_pools is None:
        raise ValueError(
            "Full evaluation requires split-aware pools. "
            "Refusing to fall back to full-node random sampling."
        )

    if split not in sampler.split_pools:
        raise ValueError(
            f"Unknown query split '{split}'. "
            f"Available splits: {list(sampler.split_pools.keys())}"
        )

    query_indices = []
    query_labels = []

    for class_id in [0, 1, 2, 3]:
        class_indices = sampler.split_pools[split][class_id].detach().cpu().long()
        query_indices.append(class_indices)
        query_labels.append(
            torch.full(
                size=(class_indices.numel(),),
                fill_value=class_id,
                dtype=torch.long,
            )
        )

    return {
        "query_indices": torch.cat(query_indices, dim=0),
        "query_labels": torch.cat(query_labels, dim=0),
    }


def _to_cpu_list(tensor: torch.Tensor) -> List[int]:
    return tensor.detach().cpu().long().tolist()


def _class_counts(labels: torch.Tensor) -> Dict[str, int]:
    labels = labels.detach().cpu().long()
    counts = {}

    for class_id in [0, 1, 2, 3]:
        counts[str(class_id)] = int((labels == class_id).sum().item())

    return counts


@torch.no_grad()
def evaluate_full_split(
    model: torch.nn.Module,
    sampler,
    device: torch.device,
    split: str,
    seed: int,
    batch_size: int,
    num_classes: int = 4,
    normal_class_id: int = 0,
) -> Dict[str, Any]:
    if split not in ["val", "test"]:
        raise ValueError("Full evaluation split must be 'val' or 'test'.")

    model.eval()

    fixed_support = _sample_fixed_train_support(
        sampler=sampler,
        seed=seed,
    )
    query_data = _collect_query_nodes(
        sampler=sampler,
        split=split,
    )

    support_subgraphs = [
        subgraph.to(device)
        for subgraph in fixed_support["support_subgraphs"]
    ]
    support_labels = fixed_support["support_labels"].to(device)

    query_indices = query_data["query_indices"]
    query_labels = query_data["query_labels"]

    collector = {
        "y_true": [],
        "y_pred": [],
        "y_prob": [],
        "coarse_y_true": [],
        "coarse_y_pred": [],
        "coarse_y_score": [],
    }

    loss_sums: Dict[str, float] = {}
    num_batches = 0

    for start in range(0, query_indices.numel(), batch_size):
        end = min(start + batch_size, query_indices.numel())

        batch_query_indices = query_indices[start:end]
        batch_query_labels = query_labels[start:end]

        query_subgraphs = sample_multiple_center_aware_subgraphs(
            data=sampler.data,
            center_nodes=batch_query_indices,
            num_hops=sampler.num_hops,
            relabel_nodes=sampler.relabel_nodes,
            max_samples=batch_query_indices.numel(),
            safe_mode=sampler.safe_subgraph_sampling, seed=sampler.seed,
            max_nodes_per_subgraph=sampler.max_nodes_per_subgraph,
            max_edges_per_subgraph=sampler.max_edges_per_subgraph,
            neighbor_sample_per_hop=sampler.neighbor_sample_per_hop,
        )
        query_subgraphs = [
            subgraph.to(device)
            for subgraph in query_subgraphs
        ]

        episode = {
            "support_subgraphs": support_subgraphs,
            "support_labels": support_labels,
            "query_subgraphs": query_subgraphs,
            "query_labels": batch_query_labels.to(device),
        }

        output = model(episode)

        for key in [
            "loss",
            "coarse_loss",
            "fine_loss",
            "type_aware_loss",
            "accuracy",
            "coarse_accuracy",
            "fine_accuracy",
            "type_accuracy",
        ]:
            if key in output:
                loss_sums[key] = loss_sums.get(key, 0.0) + float(
                    output[key].detach().cpu().item()
                )

        predictions = output["predictions"].detach().cpu().long()
        probs = None
        if "probs" in output:
            probs = output["probs"].detach().cpu().float()
        elif "logits" in output:
            probs = torch.softmax(output["logits"].detach().cpu().float(), dim=-1)

        anomaly_score = output["anomaly_score"].detach().cpu().float()
        labels_cpu = batch_query_labels.detach().cpu().long()

        collector["y_true"].extend(labels_cpu.tolist())
        collector["y_pred"].extend(predictions.tolist())
        if (
            probs is not None
            and probs.dim() == 2
            and probs.size(0) == labels_cpu.size(0)
        ):
            collector["y_prob"].extend(probs.tolist())
        collector["coarse_y_true"].extend((labels_cpu != 0).long().tolist())
        collector["coarse_y_pred"].extend((predictions != 0).long().tolist())
        collector["coarse_y_score"].extend(anomaly_score.tolist())

        num_batches += 1

    metrics: Dict[str, Any] = {}

    if num_batches > 0:
        for key, value in loss_sums.items():
            metrics[key] = value / num_batches

    metrics.update(
        compute_classification_metrics(
            y_true=collector["y_true"],
            y_pred=collector["y_pred"],
            num_classes=num_classes,
            prefix="",
            class_names=CLASS_NAMES,
        )
    )

    metrics.update(
        compute_binary_anomaly_metrics(
            y_true=collector["coarse_y_true"],
            y_pred=collector["coarse_y_pred"],
            y_score=collector["coarse_y_score"],
            prefix="coarse_",
        )
    )

    metrics["auc_roc"] = metrics.get("coarse_auc_roc")
    metrics["auc_pr"] = metrics.get("coarse_auc_pr")
    metrics["average_precision"] = metrics.get("coarse_average_precision")

    if len(collector["y_prob"]) > 0:
        metrics.update(
            compute_metrics(
                y_true=collector["y_true"],
                y_prob=collector["y_prob"],
                normal_class_id=normal_class_id,
            )
        )

    metrics["mode"] = "full"
    metrics["support_split"] = "train"
    metrics["query_split"] = split
    metrics["support_seed"] = seed
    metrics["support_indices"] = _to_cpu_list(fixed_support["support_indices"])
    metrics["support_indices_by_class"] = fixed_support["support_indices_by_class"]
    metrics["query_indices"] = _to_cpu_list(query_indices)
    metrics["num_support_nodes"] = int(fixed_support["support_indices"].numel())
    metrics["num_query_nodes"] = int(query_indices.numel())
    metrics["support_class_counts"] = _class_counts(fixed_support["support_labels"])
    metrics["query_class_counts"] = _class_counts(query_labels)
    metrics["batch_size"] = int(batch_size)

    return metrics


def resolve_output_json(
    checkpoint_path: str,
    split: str,
    mode: str,
    output_json: Optional[str],
) -> str:
    if output_json is not None:
        return output_json

    run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    return os.path.join(run_dir, f"{split}_{mode}_metrics.json")


def main():
    args = parse_args()

    config = load_config(args.config)

    seed = args.seed
    if seed is None:
        seed = get_from_config(config, "runtime.seed", get_from_config(config, "seed", 42))

    set_seed(seed)
    set_in_config(config, "runtime.seed", seed)
    set_in_config(config, "seed", seed)
    set_in_config(config, "episode.seed", seed)

    device_config = args.device
    if device_config is None:
        device_config = get_from_config(config, "runtime.device", "auto")

    device = resolve_device(device_config)

    dataset, data, sampler = build_data_and_sampler(
        config=config,
        seed=seed,
    )

    model = FMGAD(
        config=config,
        input_dim=data.num_features,
    ).to(device)

    checkpoint_info = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    num_classes = get_from_config(config, "evaluation.num_classes", 4)
    normal_class_id = int(get_from_config(config, "evaluation.normal_class_id", 0))

    if args.mode == "episodic":
        num_episodes = args.num_episodes
        if num_episodes is None:
            num_episodes = get_from_config(config, "training.eval_episodes", 50)

        fixed_episodes, episode_cache_path = get_fixed_eval_episodes_from_config(
            config=config,
            sampler=sampler,
            split_name=args.split,
            num_episodes=num_episodes,
            default_seed=seed,
        )

        metrics = evaluate_episodes(
            model=model,
            sampler=sampler,
            device=device,
            num_episodes=num_episodes,
            num_classes=num_classes,
            normal_class_id=normal_class_id,
            split=args.split,
            fixed_episodes=fixed_episodes,
        )
        metrics["mode"] = "episodic"
        metrics["eval_split"] = args.split
        metrics["num_eval_episodes"] = (
            len(fixed_episodes)
            if fixed_episodes is not None
            else num_episodes
        )
        if episode_cache_path is not None:
            metrics["episode_cache_path"] = episode_cache_path

    else:
        metrics = evaluate_full_split(
            model=model,
            sampler=sampler,
            device=device,
            split=args.split,
            seed=seed,
            batch_size=args.batch_size,
            num_classes=num_classes,
            normal_class_id=normal_class_id,
        )

    metrics["dataset"] = config["dataset"]["name"].lower()
    metrics["model"] = "FMGAD"
    metrics["seed"] = seed
    metrics["checkpoint"] = args.checkpoint
    metrics.update(checkpoint_info)

    output_json = resolve_output_json(
        checkpoint_path=args.checkpoint,
        split=args.split,
        mode=args.mode,
        output_json=args.output_json,
    )

    save_json(
        path=output_json,
        obj=metrics,
    )

    print("========== FMGAD Evaluation ==========")
    print(f"Config     : {args.config}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Mode       : {args.mode}")
    print(f"Split      : {args.split}")
    print(f"Seed       : {seed}")
    print(f"Device     : {device}")

    for key in [
        "loss",
        "accuracy",
        "macro_f1",
        "macro_auroc",
        "macro_auprc",
        "binary_auroc",
        "binary_auprc",
        "weighted_f1",
        "coarse_auc_roc",
        "coarse_auc_pr",
        "coarse_average_precision",
    ]:
        if key in metrics:
            print(f"{key:<24s}: {metrics[key]}")

    print(f"Saved JSON : {output_json}")


if __name__ == "__main__":
    main()
