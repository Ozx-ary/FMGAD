import csv
import json
import os
from typing import Dict, List, Any

import yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_yaml(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            obj,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def _flatten_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def append_metrics_to_csv(
    path: str,
    row: Dict[str, Any],
) -> None:
    """
    Append one metric row to CSV.

    If file does not exist, write header first.
    """
    ensure_dir(os.path.dirname(path))

    flattened_row = {
        key: _flatten_value(value)
        for key, value in row.items()
    }

    file_exists = os.path.exists(path)

    if file_exists:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
    else:
        existing_header = None

    fieldnames: List[str]

    if existing_header is not None:
        fieldnames = existing_header
    else:
        fieldnames = list(flattened_row.keys())

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(flattened_row)