import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def save_config(
    config: Dict[str, Any],
    config_path: str,
) -> None:
    """
    Save config dictionary to YAML file.
    """
    directory = os.path.dirname(config_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def get_from_config(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None,
) -> Any:
    """
    Get nested value from config using dot-style key path.
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def set_in_config(
    config: Dict[str, Any],
    key_path: str,
    value: Any,
) -> Dict[str, Any]:
    """
    Set nested value in config using dot-style key path.

    Example:
        set_in_config(config, "runtime.seed", 42)
    """
    keys = key_path.split(".")
    current = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value

    return config