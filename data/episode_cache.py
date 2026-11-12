import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from utils.config import get_from_config

if TYPE_CHECKING:
    from data.episode_sampler import EpisodeSampler


DEFAULT_EPISODE_CACHE_DIR = "outputs/episode_cache"
EVAL_SEED_OFFSETS = {
    "val": 1000,
    "test": 2000,
}
METADATA_MISMATCH_MESSAGE = (
    "Episode cache metadata mismatch. Delete the cache or set "
    "evaluation.regenerate_eval_episodes=true."
)


def normalize_dataset_name(dataset_name: Any) -> str:
    return str(dataset_name).strip().lower()


def eval_generation_seed(split_name: str, episode_seed: int) -> int:
    if split_name in EVAL_SEED_OFFSETS:
        return int(episode_seed) + EVAL_SEED_OFFSETS[split_name]

    split_offset = 3000 + sum(
        (index + 1) * ord(char)
        for index, char in enumerate(str(split_name))
    )
    return int(episode_seed) + split_offset


def resolve_episode_cache_path(
    cache_dir: str,
    dataset_name: str,
    split_name: str,
    seed: int,
    split_seed: Optional[int] = None,
    episode_seed: Optional[int] = None,
) -> str:
    dataset_key = normalize_dataset_name(dataset_name)
    file_name = f"{split_name}_episodes.pt"

    if split_seed is None:
        return os.path.join(
            cache_dir,
            dataset_key,
            f"seed_{int(seed)}",
            file_name,
        )

    if episode_seed is None:
        episode_seed = seed

    return os.path.join(
        cache_dir,
        dataset_key,
        f"split_{int(split_seed)}",
        f"episode_{int(episode_seed)}",
        file_name,
    )


def _torch_load(path: str) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_episode_cache(path: str) -> Tuple[List[Dict[str, object]], Dict[str, Any]]:
    payload = _torch_load(path)

    if isinstance(payload, dict) and "episodes" in payload:
        episodes = payload["episodes"]
        metadata = payload.get("metadata", None)
        if metadata is None:
            metadata = {
                key: value
                for key, value in payload.items()
                if key != "episodes"
            }
        elif not isinstance(metadata, dict):
            raise ValueError(
                f"Episode cache at {path} has invalid metadata type: "
                f"{type(metadata).__name__}."
            )
    elif isinstance(payload, list):
        episodes = payload
        metadata = {}
    else:
        raise ValueError(
            f"Episode cache at {path} must contain an episode list or "
            "a dict with an 'episodes' field."
        )

    if not isinstance(episodes, list):
        raise ValueError(
            f"Episode cache at {path} has invalid episodes type: "
            f"{type(episodes).__name__}."
        )

    return episodes, metadata


def save_episode_cache(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)


def _metadata_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.detach().cpu().item())
        return [
            int(item)
            for item in value.detach().cpu().long().view(-1).tolist()
        ]

    if isinstance(value, bool) or value is None:
        return value

    if isinstance(value, int):
        return int(value)

    if isinstance(value, float):
        return float(value)

    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _metadata_value(item)
            for key, item in value.items()
        }

    return value


def build_episode_cache_metadata(
    sampler: "EpisodeSampler",
    split_name: str,
    num_episodes: int,
    dataset_name: str,
    split_seed: Optional[int],
    episode_seed: int,
) -> Dict[str, Any]:
    classes = [
        int(class_id)
        for class_id in sampler.classes.detach().cpu().long().tolist()
    ]
    n_way = int(getattr(sampler, "n_way", len(classes)))
    k_shot = int(getattr(sampler, "k_shot"))
    q_query = int(getattr(sampler, "q_query"))
    generation_seed = eval_generation_seed(
        split_name=split_name,
        episode_seed=episode_seed,
    )

    split_class_counts = {}
    for class_id in classes:
        indices = sampler._get_candidate_indices(  # noqa: SLF001
            class_id=class_id,
            split=split_name,
        )
        split_class_counts[str(class_id)] = int(indices.numel())

    metadata = {
        "metadata_version": 1,
        "dataset": normalize_dataset_name(dataset_name),
        "split": str(split_name),
        "split_seed": None if split_seed is None else int(split_seed),
        "episode_seed": int(episode_seed),
        "generation_seed": int(generation_seed),
        "num_episodes": int(num_episodes),
        "task": str(getattr(sampler, "task", "unknown")),
        "n_way": n_way,
        "classes": classes,
        "split_class_counts": split_class_counts,
        "k_shot": k_shot,
        "q_query": q_query,
        "support_size_per_episode": n_way * k_shot,
        "query_size_per_episode": n_way * q_query,
        "num_hops": int(getattr(sampler, "num_hops")),
        "relabel_nodes": bool(getattr(sampler, "relabel_nodes")),
        "safe_subgraph_sampling": bool(
            getattr(sampler, "safe_subgraph_sampling")
        ),
        "allow_replacement": bool(getattr(sampler, "allow_replacement")),
        "max_nodes_per_subgraph": _metadata_value(
            getattr(sampler, "max_nodes_per_subgraph")
        ),
        "max_edges_per_subgraph": _metadata_value(
            getattr(sampler, "max_edges_per_subgraph")
        ),
        "neighbor_sample_per_hop": _metadata_value(
            getattr(sampler, "neighbor_sample_per_hop")
        ),
    }
    return _metadata_value(metadata)


def validate_episode_cache_metadata(
    cache_path: str,
    saved_metadata: Dict[str, Any],
    expected_metadata: Dict[str, Any],
    actual_num_episodes: int,
) -> None:
    if not saved_metadata:
        raise ValueError(
            f"{METADATA_MISMATCH_MESSAGE} Cache has no saved metadata: "
            f"{cache_path}"
        )

    missing_keys = []
    mismatches = []

    for key, expected_value in expected_metadata.items():
        if key not in saved_metadata:
            missing_keys.append(key)
            continue

        saved_value = _metadata_value(saved_metadata[key])
        expected_value = _metadata_value(expected_value)
        if saved_value != expected_value:
            mismatches.append(
                f"{key}: expected={expected_value!r}, saved={saved_value!r}"
            )

    expected_count = int(expected_metadata["num_episodes"])
    if int(actual_num_episodes) != expected_count:
        mismatches.append(
            "episode_count: "
            f"expected={expected_count!r}, saved={int(actual_num_episodes)!r}"
        )

    if missing_keys or mismatches:
        details = []
        if missing_keys:
            details.append(f"missing_keys={missing_keys}")
        if mismatches:
            details.append(f"mismatches={mismatches}")
        raise ValueError(
            f"{METADATA_MISMATCH_MESSAGE} Cache: {cache_path}. "
            + "; ".join(details)
        )


def get_or_create_fixed_eval_episodes(
    sampler: "EpisodeSampler",
    split_name: str,
    num_episodes: int,
    cache_path: str,
    dataset_name: str,
    split_seed: Optional[int],
    episode_seed: int,
    regenerate: bool = False,
) -> List[Dict[str, object]]:
    dataset_key = normalize_dataset_name(dataset_name)
    expected_metadata = build_episode_cache_metadata(
        sampler=sampler,
        split_name=split_name,
        num_episodes=int(num_episodes),
        dataset_name=dataset_name,
        split_seed=split_seed,
        episode_seed=episode_seed,
    )

    if os.path.exists(cache_path) and not regenerate:
        print(
            f"[EpisodeCache] Loading {split_name} episodes from {cache_path}"
        )
        print(
            "[EpisodeCache] "
            f"dataset={dataset_key}, split_seed={split_seed}, "
            f"episode_seed={episode_seed}, action=loaded"
        )
        episodes, metadata = load_episode_cache(cache_path)
        validate_episode_cache_metadata(
            cache_path=cache_path,
            saved_metadata=metadata,
            expected_metadata=expected_metadata,
            actual_num_episodes=len(episodes),
        )
        print(
            f"[EpisodeCache] Loaded {len(episodes)} {split_name} episodes"
        )
        return episodes

    generation_seed = int(expected_metadata["generation_seed"])
    print(
        f"[EpisodeCache] Generating {split_name} episodes with seed "
        f"{generation_seed}"
    )
    print(
        "[EpisodeCache] "
        f"dataset={dataset_key}, split_seed={split_seed}, "
        f"episode_seed={episode_seed}, episodes={num_episodes}, "
        "action=generated"
    )
    episodes = sampler.sample_episodes_with_seed(
        num_episodes=int(num_episodes),
        seed=generation_seed,
        split=split_name,
    )

    payload = {
        "format_version": 2,
        "metadata": expected_metadata,
        "episodes": episodes,
    }

    print(
        f"[EpisodeCache] Saving {split_name} episodes to {cache_path}"
    )
    save_episode_cache(cache_path, payload)
    print(
        f"[EpisodeCache] Saved {len(episodes)} {split_name} episodes"
    )

    return episodes


def get_fixed_eval_episodes_from_config(
    config: Dict[str, Any],
    sampler: "EpisodeSampler",
    split_name: str,
    num_episodes: int,
    default_seed: int,
) -> Tuple[Optional[List[Dict[str, object]]], Optional[str]]:
    fixed_eval = bool(
        get_from_config(
            config,
            "evaluation.fixed_eval_episodes",
            True,
        )
    )
    if not fixed_eval:
        print(
            "[EpisodeCache] Fixed eval episodes disabled; "
            f"{split_name} episodes will be sampled dynamically."
        )
        return None, None

    dataset_name = get_from_config(config, "dataset.name", "unknown")
    cache_dir = get_from_config(
        config,
        "evaluation.episode_cache_dir",
        DEFAULT_EPISODE_CACHE_DIR,
    )
    regenerate = bool(
        get_from_config(
            config,
            "evaluation.regenerate_eval_episodes",
            False,
        )
    )
    episode_seed = int(
        get_from_config(
            config,
            "episode.seed",
            default_seed,
        )
    )
    split_seed = get_from_config(config, "split.seed", None)
    if split_seed is not None:
        split_seed = int(split_seed)

    cache_seed = int(episode_seed if split_seed is None else default_seed)
    cache_path = resolve_episode_cache_path(
        cache_dir=cache_dir,
        dataset_name=dataset_name,
        split_name=split_name,
        seed=cache_seed,
        split_seed=split_seed,
        episode_seed=episode_seed,
    )

    episodes = get_or_create_fixed_eval_episodes(
        sampler=sampler,
        split_name=split_name,
        num_episodes=int(num_episodes),
        cache_path=cache_path,
        dataset_name=dataset_name,
        split_seed=split_seed,
        episode_seed=episode_seed,
        regenerate=regenerate,
    )
    return episodes, cache_path
