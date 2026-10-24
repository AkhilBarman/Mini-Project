"""
Evaluation framework for hallucination detection in legal summaries.

Provides:
- compute_metrics: Accuracy, Precision, Recall, F1, ROC AUC, Average Precision, CM counts.
- best_f1_threshold: Optimistic threshold sweep finding maximal F1.
- per_error_type_recall: Recall breakdown across the 5 error categories.
- Timer: Context manager to measure runtime per summary and total.
- save_results & make_comparison_table: Formats, saves, and aggregates evaluation runs.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Timer:
    """Context manager to measure runtime in seconds."""

    def __init__(self, description: str = ""):
        self.description = description
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.elapsed_seconds: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_seconds = round(self.end_time - self.start_time, 4)


def compute_metrics(
    y_true: Union[List[int], np.ndarray, pd.Series],
    scores: Union[List[float], np.ndarray, pd.Series],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute classification and ranking metrics for hallucination scores.

    Convention:
      - Positive class (1) = Hallucinated sentence
      - Negative class (0) = Supported sentence
      - Higher score in [0, 1] means MORE LIKELY hallucinated.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        scores: Continuous hallucination scores in [0, 1].
        threshold: Decision threshold for binary classification (default 0.5).

    Returns:
        Dict of computed evaluation metrics.
    """
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(scores, dtype=float)

    if len(y_arr) != len(s_arr):
        raise ValueError(f"Length mismatch: y_true has {len(y_arr)} items, scores has {len(s_arr)} items.")

    if len(y_arr) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "roc_auc": 0.0,
            "average_precision": 0.0,
            "n_pos": 0,
            "n_neg": 0,
            "threshold": threshold,
            "confusion_matrix": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        }

    # Binary classification at threshold
    y_pred = (s_arr >= threshold).astype(int)

    acc = float(accuracy_score(y_arr, y_pred))
    prec = float(precision_score(y_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_arr, y_pred, zero_division=0))

    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))

    # Confusion matrix
    tp = int(np.sum((y_arr == 1) & (y_pred == 1)))
    fp = int(np.sum((y_arr == 0) & (y_pred == 1)))
    tn = int(np.sum((y_arr == 0) & (y_pred == 0)))
    fn = int(np.sum((y_arr == 1) & (y_pred == 0)))

    # Ranking metrics (ROC AUC, Average Precision)
    # Note: If y_arr contains only 0s or only 1s, ROC AUC is undefined
    if n_pos > 0 and n_neg > 0:
        try:
            auc = float(roc_auc_score(y_arr, s_arr))
        except Exception:
            auc = 0.5
        try:
            ap = float(average_precision_score(y_arr, s_arr))
        except Exception:
            ap = n_pos / len(y_arr)
    else:
        auc = 1.0 if (n_pos > 0 and np.all(s_arr == 1)) else 0.5
        ap = 1.0 if (n_pos > 0 and np.all(s_arr == 1)) else 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "average_precision": round(ap, 4),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "threshold": round(threshold, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }


def best_f1_threshold(
    y_true: Union[List[int], np.ndarray, pd.Series],
    scores: Union[List[float], np.ndarray, pd.Series],
    num_steps: int = 100
) -> Dict[str, Any]:
    """Sweep decision thresholds in [0.01, 0.99] to find maximum F1 score.

    Note:
      This is an OPTIMISTIC estimate because the threshold is selected and
      evaluated on the same test data. Always report default threshold (0.5)
      alongside optimistic threshold.
    """
    y_arr = np.asarray(y_true, dtype=int)
    s_arr = np.asarray(scores, dtype=float)

    if np.sum(y_arr == 1) == 0:
        return {
            "best_threshold": 0.5,
            "best_f1": 0.0,
            "note": "OPTIMISTIC: Evaluated and selected on same data (No positive labels found)."
        }

    candidate_thresholds = np.linspace(0.01, 0.99, num_steps)
    best_thresh = 0.5
    best_f1 = -1.0

    for thresh in candidate_thresholds:
        preds = (s_arr >= thresh).astype(int)
        f1 = f1_score(y_arr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return {
        "best_threshold": round(float(best_thresh), 4),
        "best_f1": round(float(best_f1), 4),
        "note": "OPTIMISTIC: Threshold tuned and evaluated on the same evaluation dataset."
    }


def per_error_type_recall(
    testset_df: pd.DataFrame,
    scores: Union[List[float], np.ndarray, pd.Series],
    threshold: float = 0.5
) -> Dict[str, Dict[str, Any]]:
    """Calculate recall specifically for each error category in the test set.

    Args:
        testset_df: DataFrame matching testset.csv containing 'label' and 'error_type'.
        scores: Aligned hallucination scores.
        threshold: Decision threshold.

    Returns:
        Dict mapping error_type -> {'total': int, 'detected': int, 'recall': float}
    """
    df = testset_df.copy()
    df["score"] = np.asarray(scores, dtype=float)
    df["predicted_hallucination"] = (df["score"] >= threshold).astype(int)

    # Filter to only hallucinated rows (label == 1)
    positives = df[df["label"].astype(int) == 1]
    results: Dict[str, Dict[str, Any]] = {}

    for etype, group in positives.groupby("error_type"):
        total = len(group)
        detected = int(group["predicted_hallucination"].sum())
        recall = round(detected / total, 4) if total > 0 else 0.0
        results[str(etype)] = {
            "total": total,
            "detected": detected,
            "recall": recall
        }

    return results


def save_results(
    metrics: Dict[str, Any],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save evaluation metrics and run metadata to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {},
        "metrics": metrics
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] Evaluation results saved to: {output_path}")


def make_comparison_table(results_dir: str) -> pd.DataFrame:
    """Merge all evaluation JSON files in results_dir into a consolidated DataFrame."""
    if not os.path.exists(results_dir):
        return pd.DataFrame()

    records = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                metr = data.get("metrics", {})
                rec = {
                    "result_file": fname,
                    "approach": meta.get("approach", "N/A"),
                    "model_name": meta.get("model_name", "N/A"),
                    "threshold": metr.get("threshold", 0.5),
                    "accuracy": metr.get("accuracy", None),
                    "precision": metr.get("precision", None),
                    "recall": metr.get("recall", None),
                    "f1": metr.get("f1", None),
                    "roc_auc": metr.get("roc_auc", None),
                    "average_precision": metr.get("average_precision", None),
                    "seconds_total": meta.get("seconds_total", None),
                    "seconds_per_summary": meta.get("seconds_per_summary", None),
                }
                records.append(rec)
            except Exception as e:
                print(f"Skipping corrupted result file {fname}: {e}")

    return pd.DataFrame(records)
