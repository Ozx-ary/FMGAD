import json
import os
from typing import Dict, Optional, Sequence

import torch


CLASS_NAMES = {
    0: "normal",
    1: "structural",
    2: "attribute",
    3: "hybrid",
}


def _ensure_dir_for_file(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _tensor_to_int_list(x: torch.Tensor):
    return [int(v) for v in x.detach().cpu().long().tolist()]


def _int_list_to_tensor(x) -> torch.Tensor:
    return torch.tensor([int(v) for v in x], dtype=torch.long)


def get_multiclass_labels(data) -> torch.Tensor:
    """
    Get multiclass anomaly labels.

    Labels:
        0 = normal
        1 = structural anomaly
        2 = attribute anomaly
        3 = hybrid anomaly
    """
    if hasattr(data, "anomaly_type"):
        return data.anomaly_type.detach().cpu().long()

    if hasattr(data, "anom_type"):
        return data.anom_type.detach().cpu().long()

    raise ValueError(
        "Split manager requires data.anomaly_type or data.anom_type."
    )


def build_classwise_split_pools(
    labels: torch.Tensor,
    split_ratios: Dict[str, float],
    seed: int = 42,
    classes: Optional[Sequence[int]] = None,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Build train/val/test split pools for each class independently.

    Args:
        labels:
            Tensor[num_nodes], class labels.
        split_ratios:
            Example:
                {
                    "train": 0.4,
                    "val": 0.2,
                    "test": 0.4,
                }
        seed:
            Random seed.
        classes:
            Class ids to split.

    Returns:
        split_pools:
            {
                "train": {0: Tensor(...), 1: Tensor(...), ...},
                "val":   {0: Tensor(...), 1: Tensor(...), ...},
                "test":  {0: Tensor(...), 1: Tensor(...), ...},
            }
    """
    if classes is None:
        classes = [0, 1, 2, 3]

    required_keys = ["train", "val", "test"]
    for key in required_keys:
        if key not in split_ratios:
            raise ValueError(f"split_ratios must contain key '{key}'.")

    ratio_sum = float(
        split_ratios["train"] + split_ratios["val"] + split_ratios["test"]
    )

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"split ratios must sum to 1.0, got {ratio_sum}."
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    split_pools: Dict[str, Dict[int, torch.Tensor]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    labels = labels.detach().cpu().long()

    for class_id in classes:
        class_indices = torch.where(labels == int(class_id))[0]
        num_class = int(class_indices.numel())

        if num_class == 0:
            raise ValueError(f"Class {class_id} has no samples.")

        perm = torch.randperm(num_class, generator=generator)
        shuffled = class_indices[perm]

        n_train = int(num_class * split_ratios["train"])
        n_val = int(num_class * split_ratios["val"])

        # Ensure each split has at least one sample when possible.
        # This is important for small anomaly classes such as hybrid.
        if num_class >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)

            if n_train + n_val >= num_class:
                n_val = max(1, num_class - n_train - 1)

        n_test = num_class - n_train - n_val

        if n_test <= 0:
            raise ValueError(
                f"Class {class_id} split failed: "
                f"num_class={num_class}, train={n_train}, val={n_val}, test={n_test}."
            )

        train_indices = shuffled[:n_train]
        val_indices = shuffled[n_train:n_train + n_val]
        test_indices = shuffled[n_train + n_val:]

        split_pools["train"][int(class_id)] = train_indices.long()
        split_pools["val"][int(class_id)] = val_indices.long()
        split_pools["test"][int(class_id)] = test_indices.long()

    return split_pools


def save_split_pools(
    split_pools: Dict[str, Dict[int, torch.Tensor]],
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save split pools to JSON.
    """
    _ensure_dir_for_file(path)

    obj = {
        "metadata": metadata or {},
        "splits": {},
    }

    for split_name, class_map in split_pools.items():
        obj["splits"][split_name] = {}
        for class_id, indices in class_map.items():
            obj["splits"][split_name][str(int(class_id))] = _tensor_to_int_list(indices)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_split_pools(path: str) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Load split pools from JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    raw_splits = obj["splits"]

    split_pools: Dict[str, Dict[int, torch.Tensor]] = {}

    for split_name, class_map in raw_splits.items():
        split_pools[split_name] = {}
        for class_id_str, indices in class_map.items():
            split_pools[split_name][int(class_id_str)] = _int_list_to_tensor(indices)

    return split_pools


def validate_split_pools_no_overlap(
    split_pools: Dict[str, Dict[int, torch.Tensor]],
) -> None:
    """
    Ensure train/val/test node sets are pairwise disjoint.
    """
    split_sets = {}

    for split_name in ["train", "val", "test"]:
        if split_name not in split_pools:
            raise ValueError(
                f"Split pools must contain '{split_name}'. "
                f"Available splits: {list(split_pools.keys())}."
            )

        nodes = []

        for class_id, indices in split_pools[split_name].items():
            indices = indices.detach().cpu().long()

            unique_indices = torch.unique(indices)
            if unique_indices.numel() != indices.numel():
                raise ValueError(
                    f"Split '{split_name}', class {class_id} contains "
                    "duplicate node indices."
                )

            nodes.extend(int(v) for v in indices.tolist())

        node_set = set(nodes)

        if len(node_set) != len(nodes):
            raise ValueError(
                f"Split '{split_name}' contains duplicate node indices "
                "across classes."
            )

        split_sets[split_name] = node_set

    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_sets[left] & split_sets[right]

        if overlap:
            examples = sorted(overlap)[:20]
            raise ValueError(
                f"Split overlap detected between '{left}' and '{right}': "
                f"{len(overlap)} nodes overlap. Examples: {examples}"
            )


def build_or_load_split_pools(
    data,
    split_config: Dict,
    default_seed: int = 42,
    classes: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Dict[int, torch.Tensor]]]:
    """
    Build or load split pools based on config.

    Config example:
        split:
          enabled: true
          seed: 42
          ratios:
            train: 0.4
            val: 0.2
            test: 0.4
          save_path: splits/cora_v2_8_ratio005_seed42.json
          load_existing: true
    """
    if not split_config or not split_config.get("enabled", False):
        return None

    seed = int(split_config.get("seed", default_seed))
    save_path = split_config.get("save_path", None)
    load_existing = bool(split_config.get("load_existing", True))

    if save_path is not None and load_existing and os.path.exists(save_path):
        print(f"[Split] Loading existing split pools from {save_path}")
        split_pools = load_split_pools(save_path)
        validate_split_pools_no_overlap(split_pools)
        return split_pools

    ratios = split_config.get(
        "ratios",
        {
            "train": 0.4,
            "val": 0.2,
            "test": 0.4,
        },
    )

    labels = get_multiclass_labels(data)

    split_pools = build_classwise_split_pools(
        labels=labels,
        split_ratios=ratios,
        seed=seed,
        classes=classes,
    )

    validate_split_pools_no_overlap(split_pools)

    if save_path is not None:
        metadata = {
            "seed": seed,
            "ratios": ratios,
            "class_names": CLASS_NAMES,
        }
        save_split_pools(
            split_pools=split_pools,
            path=save_path,
            metadata=metadata,
        )
        print(f"[Split] Saved split pools to {save_path}")

    return split_pools


def summarize_split_pools(
    split_pools: Dict[str, Dict[int, torch.Tensor]],
) -> Dict[str, Dict[int, int]]:
    """
    Return count summary for split pools.
    """
    summary: Dict[str, Dict[int, int]] = {}

    for split_name, class_map in split_pools.items():
        summary[split_name] = {}
        for class_id, indices in class_map.items():
            summary[split_name][int(class_id)] = int(indices.numel())

    return summary


def print_split_summary(
    split_pools: Dict[str, Dict[int, torch.Tensor]],
) -> None:
    """
    Print split summary.
    """
    print("Split Summary:")
    for split_name in ["train", "val", "test"]:
        if split_name not in split_pools:
            continue

        print(f"  [{split_name}]")
        for class_id in [0, 1, 2, 3]:
            if class_id not in split_pools[split_name]:
                continue

            count = int(split_pools[split_name][class_id].numel())
            class_name = CLASS_NAMES.get(class_id, "unknown")
            print(f"    class {class_id} ({class_name:>10s}): {count}")
