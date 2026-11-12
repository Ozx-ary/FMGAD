import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected


PLANETOID_DATASETS = {"cora", "citeseer", "pubmed"}
MAT_DATASETS = {
    "acm",
    "blogcatalog",
    "flickr",
    "weibo",
    "tolokers",
    "questions",
    "disney",
    "books",
    "enron",
    "reddit",
    "photo",
    "computers",
    "cs",
}

MAT_ADJ_KEYS = ["Network", "A", "adj", "Adj", "network"]
MAT_FEATURE_KEYS = ["Attributes", "X", "features", "Features", "attr"]
MAT_LABEL_KEYS = ["Label", "label", "labels", "gnd", "y"]


@dataclass
class MatDataset:
    """
    Small dataset-like wrapper for one-graph .mat datasets.
    """

    name: str
    num_node_features: int
    num_classes: int = 0

    def __len__(self) -> int:
        return 1


def _canonical_mat_name(name: str) -> str:
    normalized = name.strip().lower()
    mapping = {
        "acm": "ACM",
        "blogcatalog": "BlogCatalog",
        "flickr": "Flickr",
        "weibo": "Weibo",
        "tolokers": "Tolokers",
        "questions": "Questions",
        "disney": "Disney",
        "book": "Books",
        "books": "Books",
        "enron": "Enron",
        "reddit": "Reddit",
        "photo": "Photo",
        "computer": "Computers",
        "computers": "Computers",
        "cs": "CS",
    }

    if normalized not in mapping:
        supported = ", ".join(sorted(mapping.values()))
        raise ValueError(
            f"Unsupported .mat dataset: {name}. Supported: {supported}."
        )

    return mapping[normalized]


def _available_mat_keys(mat: Dict) -> str:
    keys = sorted(k for k in mat.keys() if not k.startswith("__"))
    return ", ".join(keys)


def _find_first_key(mat: Dict, candidates, kind: str):
    for key in candidates:
        if key in mat:
            return key

    raise KeyError(
        f"Could not find {kind} in .mat file. Tried keys: "
        f"{candidates}. Available keys: {_available_mat_keys(mat)}."
    )


def _find_optional_key(mat: Dict, candidates) -> Optional[str]:
    for key in candidates:
        if key in mat:
            return key
    return None


def _resolve_mat_path(root: str, name: str) -> str:
    """
    Resolve .mat dataset path.

    Supported layouts include:

    1. Flat layout:
       root/ACM.mat
       root/acm.mat

    2. Raw subdirectory layout:
       root/raw/ACM/ACM.mat
       root/raw/BlogCatalog/BlogCatalog.mat
       root/raw/Flickr/Flickr.mat
       root/raw/Weibo/Weibo.mat

    3. Dataset subdirectory layout:
       root/ACM/ACM.mat
       root/BlogCatalog/BlogCatalog.mat
       root/Flickr/Flickr.mat

    The function keeps backward compatibility with the old flat layout.
    """
    canonical_name = _canonical_mat_name(name)
    requested_name = name.strip()
    lower_name = canonical_name.lower()

    # Candidate file names.
    candidate_filenames = []
    for filename in [
        f"{requested_name}.mat",
        f"{canonical_name}.mat",
        f"{lower_name}.mat",
    ]:
        if filename not in candidate_filenames:
            candidate_filenames.append(filename)

    # Candidate dataset directory names.
    candidate_dataset_dirs = []
    for dirname in [
        requested_name,
        canonical_name,
        lower_name,
    ]:
        if dirname not in candidate_dataset_dirs:
            candidate_dataset_dirs.append(dirname)

    normalized_root = os.path.normpath(root)

    # Candidate root directories.
    candidate_roots = [normalized_root]

    # If root is "./data/raw", also try "./data".
    if os.path.basename(normalized_root).lower() == "raw":
        parent_root = os.path.dirname(normalized_root)
        if parent_root not in candidate_roots:
            candidate_roots.append(parent_root)

    checked_paths = []

    for candidate_root in candidate_roots:
        # 1. Flat layout: root/ACM.mat
        for filename in candidate_filenames:
            path = os.path.join(candidate_root, filename)
            checked_paths.append(path)
            if os.path.exists(path):
                return path

        # 2. Dataset subdir layout: root/ACM/ACM.mat
        for dataset_dir in candidate_dataset_dirs:
            for filename in candidate_filenames:
                path = os.path.join(candidate_root, dataset_dir, filename)
                checked_paths.append(path)
                if os.path.exists(path):
                    return path

        # 3. Raw subdir layout: root/raw/ACM/ACM.mat
        for dataset_dir in candidate_dataset_dirs:
            for filename in candidate_filenames:
                path = os.path.join(candidate_root, "raw", dataset_dir, filename)
                checked_paths.append(path)
                if os.path.exists(path):
                    return path

    raise FileNotFoundError(
        "Could not find .mat dataset file. Checked: "
        + ", ".join(checked_paths)
    )


def _diagnostic_print(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[Diagnostics] {message}", flush=True)


def _safe_matrix_stats(matrix) -> str:
    parts = [f"type={type(matrix).__name__}"]

    shape = getattr(matrix, "shape", None)
    if shape is not None:
        parts.append(f"shape={tuple(shape)}")

    dtype = getattr(matrix, "dtype", None)
    if dtype is not None:
        parts.append(f"dtype={dtype}")

    nnz = getattr(matrix, "nnz", None)
    if nnz is not None:
        parts.append(f"nnz={nnz}")

    return ", ".join(parts)


def _safe_tensor_stats(tensor: torch.Tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, numel={tensor.numel()}"
    )


def _matrix_to_dense_float_tensor(
    matrix,
    debug_diagnostics: bool = False,
    label: str = "matrix",
) -> torch.Tensor:
    try:
        import scipy.sparse as sp
    except ImportError as exc:
        raise ImportError(
            "scipy is required to load .mat attributed graph datasets."
        ) from exc

    _diagnostic_print(
        debug_diagnostics,
        f"before feature sparse conversion ({label}): {_safe_matrix_stats(matrix)}",
    )

    if sp.issparse(matrix):
        array = matrix.toarray()
    else:
        array = np.asarray(matrix)

    array = np.asarray(array, dtype=np.float32)

    if array.ndim == 1:
        array = array.reshape(-1, 1)
    elif array.ndim != 2:
        array = np.squeeze(array)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        elif array.ndim != 2:
            raise ValueError(
                "Feature matrix must be 2D after squeeze, "
                f"got shape {array.shape}."
            )

    tensor = torch.from_numpy(array).float()
    _diagnostic_print(
        debug_diagnostics,
        f"after feature sparse conversion ({label}): {_safe_tensor_stats(tensor)}",
    )
    return tensor


def _adjacency_to_edge_index(
    adj,
    num_nodes: int,
    debug_diagnostics: bool = False,
) -> torch.Tensor:
    try:
        import scipy.sparse as sp
    except ImportError as exc:
        raise ImportError(
            "scipy is required to load .mat attributed graph datasets."
        ) from exc

    _diagnostic_print(
        debug_diagnostics,
        f"before adjacency sparse conversion: {_safe_matrix_stats(adj)}",
    )

    if sp.issparse(adj):
        coo = adj.tocoo()
        coo.eliminate_zeros()
        row = torch.from_numpy(coo.row.astype(np.int64, copy=False))
        col = torch.from_numpy(coo.col.astype(np.int64, copy=False))
    else:
        array = np.asarray(adj)
        if array.ndim != 2:
            raise ValueError(
                "Adjacency matrix must be 2D, "
                f"got shape {array.shape}."
            )
        row_np, col_np = np.nonzero(array)
        row = torch.from_numpy(row_np.astype(np.int64, copy=False))
        col = torch.from_numpy(col_np.astype(np.int64, copy=False))

    edge_index = torch.stack([row, col], dim=0).long()

    # Match the Planetoid convention used by this project: undirected edges.
    # This does not add self-loops.
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    _diagnostic_print(
        debug_diagnostics,
        f"after adjacency sparse conversion: {_safe_tensor_stats(edge_index)}",
    )

    return edge_index


def _labels_to_raw_y(labels, num_nodes: int) -> torch.Tensor:
    try:
        import scipy.sparse as sp
    except ImportError as exc:
        raise ImportError(
            "scipy is required to load .mat attributed graph datasets."
        ) from exc

    if sp.issparse(labels):
        array = labels.toarray()
    else:
        array = np.asarray(labels)

    array = np.asarray(array)
    array = np.squeeze(array)

    if array.ndim == 2:
        if array.shape[0] == num_nodes:
            array = np.argmax(array, axis=1)
        elif array.shape[1] == num_nodes:
            array = np.argmax(array, axis=0)
        else:
            raise ValueError(
                "Label matrix shape is incompatible with num_nodes="
                f"{num_nodes}: {array.shape}."
            )
    elif array.ndim != 1:
        raise ValueError(
            "Labels must be 1D or 2D after squeeze, "
            f"got shape {array.shape}."
        )

    if array.shape[0] != num_nodes:
        raise ValueError(
            "Label vector length is incompatible with num_nodes="
            f"{num_nodes}: {array.shape[0]}."
        )

    return torch.from_numpy(array.astype(np.int64, copy=False)).long()


def load_planetoid_dataset(
    name: str = "Cora",
    root: str = "./data/raw",
    normalize_features: bool = True,
) -> Tuple[object, object]:
    """
    Load a Planetoid dataset from PyTorch Geometric.

    Supported datasets:
        - Cora
        - Citeseer
        - Pubmed

    Args:
        name: Dataset name.
        root: Root directory for storing raw/processed data.
        normalize_features: Whether to normalize node features.

    Returns:
        dataset: PyG dataset object.
        data: PyG Data object.
    """
    supported = {"Cora", "Citeseer", "Pubmed"}
    if name not in supported:
        raise ValueError(f"Unsupported dataset: {name}. Supported: {supported}")

    transform = NormalizeFeatures() if normalize_features else None

    dataset = Planetoid(
        root=root,
        name=name,
        transform=transform,
    )

    data = dataset[0]

    return dataset, data


def load_mat_dataset(
    root: str,
    name: str,
    normalize_features: bool = True,
    debug_diagnostics: bool = False,
) -> Tuple[MatDataset, Data]:
    """
    Load an attributed graph dataset stored as a .mat file.

    Supported datasets:
        - ACM
        - BlogCatalog
        - Flickr
        - Weibo
        - Tolokers
        - Questions
        - Disney
        - Books
        - Enron
        - Reddit
        - Photo
        - Computers
        - CS

    Expected file examples:
        - data/ACM.mat
        - data/BlogCatalog.mat
        - data/Flickr.mat
        - data/Weibo.mat

    Original labels, when present, are stored as data.raw_y for inspection
    only. Synthetic anomaly experiments should use data.anomaly_type /
    data.anomaly_label produced by inject_multi_type_anomalies.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError(
            "scipy is required to load .mat attributed graph datasets."
        ) from exc

    canonical_name = _canonical_mat_name(name)
    mat_path = _resolve_mat_path(root=root, name=canonical_name)
    mat = loadmat(mat_path)

    adj_key = _find_first_key(mat, MAT_ADJ_KEYS, "adjacency matrix")
    feature_key = _find_first_key(mat, MAT_FEATURE_KEYS, "feature matrix")
    label_key = _find_optional_key(mat, MAT_LABEL_KEYS)

    adj = mat[adj_key]
    x = _matrix_to_dense_float_tensor(
        mat[feature_key],
        debug_diagnostics=debug_diagnostics,
        label=feature_key,
    )

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(
            "Adjacency matrix must be square, "
            f"got shape {adj.shape}."
        )

    num_nodes = int(adj.shape[0])

    if x.size(0) != num_nodes and x.dim() == 2 and x.size(1) == num_nodes:
        x = x.t().contiguous()

    if x.size(0) != num_nodes:
        raise ValueError(
            "Feature matrix row count must match adjacency size: "
            f"x has {x.size(0)} rows, adjacency has {num_nodes} nodes."
        )

    edge_index = _adjacency_to_edge_index(
        adj,
        num_nodes=num_nodes,
        debug_diagnostics=debug_diagnostics,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )

    num_classes = 0
    if label_key is not None:
        raw_y = _labels_to_raw_y(mat[label_key], num_nodes=num_nodes)
        data.raw_y = raw_y
        num_classes = int(torch.unique(raw_y).numel())

    if normalize_features:
        data = NormalizeFeatures()(data)

    dataset = MatDataset(
        name=canonical_name,
        num_node_features=int(data.x.size(1)),
        num_classes=num_classes,
    )

    return dataset, data


def print_dataset_load_summary(
    dataset,
    data,
    synthetic_anomaly_enabled: bool,
) -> None:
    """
    Print concise dataset loading information for training/dry-run logs.
    """
    raw_labels_exist = hasattr(data, "raw_y") and data.raw_y is not None

    print("Dataset Load:")
    print(f"  Name                         : {dataset.name}")
    print(f"  Num nodes                    : {data.num_nodes}")
    print(f"  Num edges                    : {data.edge_index.size(1)}")
    print(f"  Feature dim                  : {data.num_features}")
    print(f"  Raw labels exist             : {raw_labels_exist}")
    print(f"  Synthetic anomaly injection  : {synthetic_anomaly_enabled}")


def attach_synthetic_label_aliases(data):
    """
    Attach label aliases used by synthetic multi-type anomaly experiments.

    raw_y, when present, is preserved for inspection only. Training,
    validation, test metrics, and early stopping should use synthetic labels
    derived from inject_multi_type_anomalies.
    """
    if not hasattr(data, "anomaly_type") or data.anomaly_type is None:
        raise ValueError(
            "Synthetic label aliases require data.anomaly_type from "
            "inject_multi_type_anomalies."
        )

    if not hasattr(data, "anomaly_label") or data.anomaly_label is None:
        raise ValueError(
            "Synthetic label aliases require data.anomaly_label from "
            "inject_multi_type_anomalies."
        )

    data.type_y = data.anomaly_type.clone()
    data.coarse_y = data.anomaly_label.clone()
    data.y = data.type_y.clone()
    data.label_source = "synthetic_anomaly_type"

    if not torch.equal(data.y, data.anomaly_type):
        raise AssertionError(
            "Label source mismatch: data.y must equal synthetic "
            "data.anomaly_type when anomaly.enabled=true."
        )

    raw_y = getattr(data, "raw_y", None)
    if isinstance(raw_y, torch.Tensor) and raw_y.data_ptr() == data.y.data_ptr():
        raise AssertionError(
            "Raw label leakage detected: data.y shares storage with raw_y."
        )

    print(
        "[Labels] Using synthetic anomaly labels: "
        "data.y/type_y=data.anomaly_type, coarse_y=data.anomaly_label; "
        "raw_y is inspection-only."
    )

    return data


def print_data_summary(dataset, data) -> None:
    """
    Print basic information of a PyG dataset and graph data.

    Args:
        dataset: PyG dataset object.
        data: PyG Data object.
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    num_features = dataset.num_node_features
    num_classes = dataset.num_classes

    print("========== Dataset Summary ==========")
    print(f"Dataset name          : {dataset.name}")
    print(f"Number of graphs      : {len(dataset)}")
    print(f"Number of nodes       : {num_nodes}")
    print(f"Number of edges       : {num_edges}")
    print(f"Number of features    : {num_features}")
    print(f"Original num classes  : {num_classes}")

    if hasattr(data, "x") and data.x is not None:
        print(f"Feature matrix shape  : {tuple(data.x.shape)}")
        print(f"Feature dtype         : {data.x.dtype}")

    if hasattr(data, "edge_index") and data.edge_index is not None:
        print(f"Edge index shape      : {tuple(data.edge_index.shape)}")
        print(f"Edge index dtype      : {data.edge_index.dtype}")

    if hasattr(data, "y") and data.y is not None:
        print(f"Label shape           : {tuple(data.y.shape)}")
        print(f"Label dtype           : {data.y.dtype}")
        unique_labels = torch.unique(data.y).cpu().tolist()
        print(f"Original labels       : {unique_labels}")

    if hasattr(data, "raw_y") and data.raw_y is not None:
        print(f"Raw label shape       : {tuple(data.raw_y.shape)}")
        print(f"Raw label dtype       : {data.raw_y.dtype}")
        unique_raw_labels = torch.unique(data.raw_y).cpu().tolist()
        print(f"Raw labels            : {unique_raw_labels}")

    print("=====================================")

