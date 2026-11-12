from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def _safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def confusion_matrix_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute confusion matrix without sklearn.

    Rows are ground-truth labels.
    Columns are predicted labels.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(y_true, y_pred):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            cm[int(t), int(p)] += 1

    return cm


def compute_classification_metrics(
    y_true,
    y_pred,
    num_classes: int = 4,
    prefix: str = "",
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Compute multiclass classification metrics.

    This implementation avoids sklearn dependency and is suitable for
    reproducible lightweight evaluation.

    Args:
        y_true:
            Ground-truth labels.
        y_pred:
            Predicted labels.
        num_classes:
            Number of classes.
        prefix:
            Optional metric prefix, e.g. "eval_" or "coarse_".
        class_names:
            Optional class names for readability.

    Returns:
        Dictionary with accuracy, macro_f1, weighted_f1,
        balanced_accuracy, per-class metrics, and confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape[0] == 0:
        raise ValueError("y_true is empty.")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {y_true.shape[0]} and {y_pred.shape[0]}."
        )

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    if len(class_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, got {len(class_names)}."
        )

    cm = confusion_matrix_np(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=num_classes,
    )

    total = cm.sum()
    correct = np.trace(cm)

    accuracy = _safe_divide(float(correct), float(total))

    precisions = []
    recalls = []
    f1s = []
    supports = []

    metrics: Dict[str, object] = {}

    for class_idx in range(num_classes):
        tp = float(cm[class_idx, class_idx])
        fp = float(cm[:, class_idx].sum() - cm[class_idx, class_idx])
        fn = float(cm[class_idx, :].sum() - cm[class_idx, class_idx])
        support = float(cm[class_idx, :].sum())

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        class_name = class_names[class_idx]

        metrics[f"{prefix}precision_class_{class_idx}_{class_name}"] = precision
        metrics[f"{prefix}recall_class_{class_idx}_{class_name}"] = recall
        metrics[f"{prefix}f1_class_{class_idx}_{class_name}"] = f1
        metrics[f"{prefix}support_class_{class_idx}_{class_name}"] = support

    supports_np = np.asarray(supports, dtype=np.float64)
    f1s_np = np.asarray(f1s, dtype=np.float64)
    recalls_np = np.asarray(recalls, dtype=np.float64)

    macro_f1 = float(np.mean(f1s_np))
    weighted_f1 = float(
        _safe_divide(
            float((f1s_np * supports_np).sum()),
            float(supports_np.sum()),
        )
    )
    balanced_accuracy = float(np.mean(recalls_np))

    metrics[f"{prefix}accuracy"] = float(accuracy)
    metrics[f"{prefix}macro_f1"] = macro_f1
    metrics[f"{prefix}weighted_f1"] = weighted_f1
    metrics[f"{prefix}balanced_accuracy"] = balanced_accuracy
    metrics[f"{prefix}confusion_matrix"] = cm.tolist()

    return metrics


def _average_tied_ranks(values: np.ndarray) -> np.ndarray:
    """
    Return 1-based average ranks for ties.
    """
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1

        average_rank = 0.5 * (start + 1 + end)
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _binary_roc_auc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Optional[float]:
    """
    Compute binary ROC-AUC from ranks without sklearn.
    """
    positives = y_true == 1
    negatives = y_true == 0

    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())

    if n_pos == 0 or n_neg == 0:
        return None

    ranks = _average_tied_ranks(y_score)
    sum_pos_ranks = float(ranks[positives].sum())

    auc = (
        sum_pos_ranks - (n_pos * (n_pos + 1) / 2.0)
    ) / float(n_pos * n_neg)

    return float(auc)


def _binary_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Optional[float]:
    """
    Compute binary average precision without sklearn.
    """
    positives = y_true == 1
    n_pos = int(positives.sum())

    if n_pos == 0 or n_pos == y_true.shape[0]:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]

    true_positives = np.cumsum(sorted_true == 1)
    ranks = np.arange(1, sorted_true.shape[0] + 1, dtype=np.float64)
    precision = true_positives / ranks

    ap = precision[sorted_true == 1].sum() / float(n_pos)

    return float(ap)


def compute_binary_anomaly_metrics(
    y_true,
    y_pred,
    y_score=None,
    prefix: str = "coarse_",
) -> Dict[str, object]:
    """
    Compute binary normal-vs-anomaly metrics.

    Labels:
        0 = normal
        1 = anomaly
    """
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        num_classes=2,
        prefix=prefix,
        class_names=["normal", "anomaly"],
    )

    if y_score is None:
        return metrics

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float64)

    if y_true_np.shape[0] != y_score_np.shape[0]:
        raise ValueError(
            f"y_true and y_score must have the same length, "
            f"got {y_true_np.shape[0]} and {y_score_np.shape[0]}."
        )

    unique_labels = np.unique(y_true_np)

    if unique_labels.shape[0] < 2:
        auc_roc = None
        average_precision = None
    else:
        auc_roc = _binary_roc_auc_score(
            y_true=y_true_np,
            y_score=y_score_np,
        )
        average_precision = _binary_average_precision(
            y_true=y_true_np,
            y_score=y_score_np,
        )

    metrics[f"{prefix}auc_roc"] = auc_roc
    metrics[f"{prefix}auc_pr"] = average_precision
    metrics[f"{prefix}average_precision"] = average_precision

    return metrics


def _warn_metric(message: str) -> None:
    print(f"[WARN] {message}")


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def _ensure_probabilities(y_prob: np.ndarray) -> np.ndarray:
    if y_prob.ndim != 2:
        raise ValueError(f"Expected y_prob with shape [N, C], got {y_prob.shape}")

    if y_prob.shape[0] == 0:
        raise ValueError("y_prob is empty.")

    if not np.all(np.isfinite(y_prob)):
        _warn_metric("y_prob contains non-finite values; metrics may become NaN")

    row_sums = np.sum(y_prob, axis=1)
    looks_like_probabilities = (
        np.nanmin(y_prob) >= -1e-6
        and np.nanmax(y_prob) <= 1.0 + 1e-6
        and np.allclose(row_sums, 1.0, atol=1e-4, rtol=1e-4)
    )

    if looks_like_probabilities:
        return y_prob.astype(np.float64, copy=False)

    return _softmax_np(y_prob.astype(np.float64, copy=False))


def compute_metrics(y_true, y_prob, normal_class_id: int = 0) -> Dict[str, float]:
    """
    Compute the official multi-class anomaly-detection metrics.

    Args:
        y_true:
            Ground-truth class labels shaped [N].
        y_prob:
            Predicted class probabilities or logits shaped [N, C].
        normal_class_id:
            Class id used as the normal class for binary normal-vs-anomaly
            auxiliary metrics.

    Returns:
        Dictionary containing macro_f1, macro_auroc, macro_auprc,
        binary_auroc, and binary_auprc as Python floats or NaN.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = _ensure_probabilities(np.asarray(y_prob))

    if y_true.ndim != 1:
        raise ValueError(f"Expected y_true with shape [N], got {y_true.shape}")

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            "y_true and y_prob must have matching first dimensions, "
            f"got {y_true.shape[0]} and {y_prob.shape[0]}."
        )

    num_classes = y_prob.shape[1]
    classes = np.arange(num_classes)
    metrics: Dict[str, float] = {}

    y_pred = np.argmax(y_prob, axis=1)

    try:
        metrics["macro_f1"] = float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )
    except Exception as exc:
        _warn_metric(f"Failed to compute macro_f1: {exc}")
        metrics["macro_f1"] = float("nan")

    try:
        if normal_class_id < 0 or normal_class_id >= num_classes:
            raise ValueError(
                f"normal_class_id={normal_class_id} is outside [0, {num_classes})."
            )
        y_true_binary = (y_true != normal_class_id).astype(int)
        y_pred_binary = (y_pred != normal_class_id).astype(int)
        metrics["binary_f1"] = float(
            f1_score(
                y_true_binary,
                y_pred_binary,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        )
        metrics["binary_macro_f1"] = float(
            f1_score(
                y_true_binary,
                y_pred_binary,
                average="macro",
                zero_division=0,
            )
        )
        metrics["binary_weighted_f1"] = float(
            f1_score(
                y_true_binary,
                y_pred_binary,
                average="weighted",
                zero_division=0,
            )
        )
    except Exception as exc:
        _warn_metric(f"Failed to compute binary F1 metrics: {exc}")
        metrics["binary_f1"] = float("nan")
        metrics["binary_macro_f1"] = float("nan")
        metrics["binary_weighted_f1"] = float("nan")

    try:
        metrics["macro_auroc"] = float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
                labels=classes,
            )
        )
    except Exception as exc:
        _warn_metric(f"Failed to compute macro_auroc: {exc}")
        metrics["macro_auroc"] = float("nan")

    try:
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_true_bin.shape[1] != num_classes:
            fixed = np.zeros((len(y_true), num_classes), dtype=int)
            for j, cls in enumerate(classes):
                fixed[:, j] = (y_true == cls).astype(int)
            y_true_bin = fixed

        metrics["macro_auprc"] = float(
            average_precision_score(
                y_true_bin,
                y_prob,
                average="macro",
            )
        )
    except Exception as exc:
        _warn_metric(f"Failed to compute macro_auprc: {exc}")
        metrics["macro_auprc"] = float("nan")

    try:
        if normal_class_id < 0 or normal_class_id >= num_classes:
            raise ValueError(
                f"normal_class_id={normal_class_id} is outside [0, {num_classes})."
            )

        y_true_binary = (y_true != normal_class_id).astype(int)
        anomaly_score = 1.0 - y_prob[:, normal_class_id]

        if len(np.unique(y_true_binary)) < 2:
            _warn_metric(
                "Failed to compute binary_auroc: "
                "binary labels contain only one class"
            )
            metrics["binary_auroc"] = float("nan")
        else:
            metrics["binary_auroc"] = float(
                roc_auc_score(
                    y_true_binary,
                    anomaly_score,
                )
            )
    except Exception as exc:
        _warn_metric(f"Failed to compute binary_auroc: {exc}")
        metrics["binary_auroc"] = float("nan")

    try:
        if normal_class_id < 0 or normal_class_id >= num_classes:
            raise ValueError(
                f"normal_class_id={normal_class_id} is outside [0, {num_classes})."
            )

        y_true_binary = (y_true != normal_class_id).astype(int)
        anomaly_score = 1.0 - y_prob[:, normal_class_id]

        if len(np.unique(y_true_binary)) < 2:
            _warn_metric(
                "Failed to compute binary_auprc: "
                "binary labels contain only one class"
            )
            metrics["binary_auprc"] = float("nan")
        else:
            metrics["binary_auprc"] = float(
                average_precision_score(
                    y_true_binary,
                    anomaly_score,
                )
            )
    except Exception as exc:
        _warn_metric(f"Failed to compute binary_auprc: {exc}")
        metrics["binary_auprc"] = float("nan")

    return metrics
