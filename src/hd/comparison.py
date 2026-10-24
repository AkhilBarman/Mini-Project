"""
Comparison and Benchmarking Module for Phase 5.

Combines, analyzes, and visualizes results across:
- Baselines (Oracle, Random)
- Approach A (SummaC Cross-Encoder NLI)
- Approach B (SelfCheckGPT Zero-Resource Self-Consistency)
- Hybrid Ensemble (Calibrated Fusion)
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .evaluation import compute_metrics, per_error_type_recall, best_f1_threshold


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert pandas DataFrame to a clean markdown table without external dependencies."""
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    col_widths = [max(len(h), max([len(str(v)) for v in df[col]], default=0)) for h, col in zip(headers, df.columns)]
    
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    sep_line = "| " + " | ".join("-" * w for w in col_widths) + " |"
    
    row_lines = []
    for _, row in df.iterrows():
        row_lines.append("| " + " | ".join(str(row[c]).ljust(w) for c, w in zip(df.columns, col_widths)) + " |")
        
    return "\n".join([header_line, sep_line] + row_lines)


def calibrate_score(score: Union[float, np.ndarray], center: float = 0.85) -> Union[float, np.ndarray]:
    """
    Piecewise linear calibration mapping threshold 'center' to 0.50.
    Values below center are mapped to [0.0, 0.50].
    Values above center are mapped to [0.50, 1.0].
    """
    arr = np.asarray(score, dtype=float)
    calibrated = np.where(
        arr < center,
        0.5 * (arr / max(center, 1e-6)),
        0.5 + 0.5 * ((arr - center) / max(1.0 - center, 1e-6))
    )
    return np.clip(calibrated, 0.0, 1.0)


def compute_hybrid_scores(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    method: str = "calibrated_max",
    theta_a: float = 0.85
) -> np.ndarray:
    """
    Compute hybrid ensemble hallucination scores from Approach A and Approach B.
    
    Methods:
    - 'calibrated_max': max(calibrated(score_a), score_b). Preserves high recall of A
      while leveraging high-confidence fabrications from B.
    - 'calibrated_mean': 0.5 * calibrated(score_a) + 0.5 * score_b.
    - 'linear_weighted': 0.6 * score_a + 0.4 * score_b.
    """
    s_a = np.asarray(scores_a, dtype=float)
    s_b = np.asarray(scores_b, dtype=float)
    
    if method == "calibrated_max":
        p_a = calibrate_score(s_a, center=theta_a)
        p_b = s_b
        return np.maximum(p_a, p_b)
    elif method == "calibrated_mean":
        p_a = calibrate_score(s_a, center=theta_a)
        p_b = s_b
        return 0.5 * p_a + 0.5 * p_b
    elif method == "linear_weighted":
        return 0.6 * s_a + 0.4 * s_b
    else:
        raise ValueError(f"Unknown hybrid method: {method}")


def build_comparison_dataframe(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of evaluated run metadata and metric dictionaries into a standardized DataFrame.
    """
    rows = []
    for run in runs:
        m = run.get("metrics", {})
        meta = run.get("metadata", {})
        cm = m.get("confusion_matrix", {})
        
        row = {
            "Model / Approach": run.get("name", "Unknown"),
            "Threshold": m.get("threshold", 0.5),
            "Accuracy": m.get("accuracy", 0.0),
            "Precision": m.get("precision", 0.0),
            "Recall": m.get("recall", 0.0),
            "F1": m.get("f1", 0.0),
            "ROC-AUC": m.get("roc_auc", 0.0),
            "Avg Precision": m.get("average_precision", 0.0),
            "TP": cm.get("tp", 0),
            "FP": cm.get("fp", 0),
            "TN": cm.get("tn", 0),
            "FN": cm.get("fn", 0),
            "Runtime (s)": round(meta.get("seconds_total", 0.0), 2)
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def build_error_breakdown_dataframe(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Aggregate per-error-type recall across all evaluated runs.
    """
    records = []
    for run in runs:
        name = run.get("name", "Unknown")
        breakdown = run.get("metadata", {}).get("per_error_type_recall", {})
        row = {"Model / Approach": name}
        for error_type, stats in breakdown.items():
            row[error_type] = stats.get("recall", 0.0)
        records.append(row)
    
    return pd.DataFrame(records)


def plot_comparison_dashboard(
    df_metrics: pd.DataFrame,
    df_errors: pd.DataFrame,
    output_path: str
) -> None:
    """
    Render and save a publication-quality 4-panel comparison dashboard.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comparative Evaluation: Hallucination Detection in Indian Court Judgments", fontsize=16, fontweight="bold", y=0.98)
    
    # 1. Overall Metrics Bar Chart (Accuracy, Precision, Recall, F1)
    ax1 = axes[0, 0]
    display_models = df_metrics["Model / Approach"].tolist()
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]
    
    x = np.arange(len(display_models))
    width = 0.2
    
    for i, metric in enumerate(metrics_to_plot):
        values = df_metrics[metric].tolist()
        ax1.bar(x + (i - 1.5) * width, values, width, label=metric, alpha=0.85)
    
    ax1.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="semibold")
    ax1.set_title("Core Classification Metrics Across Approaches", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_models, rotation=25, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. ROC-AUC vs Runtime (Latency Trade-Off)
    ax2 = axes[0, 1]
    # Filter out baselines for scatter
    eval_models = df_metrics[~df_metrics["Model / Approach"].str.contains("Baseline")].copy()
    
    colors = ["#2563eb", "#3b82f6", "#10b981", "#8b5cf6"]
    for idx, (_, row) in enumerate(eval_models.iterrows()):
        ax2.scatter(
            max(row["Runtime (s)"], 0.1),
            row["ROC-AUC"],
            s=220,
            label=row["Model / Approach"],
            c=colors[idx % len(colors)],
            edgecolors="black",
            linewidth=1.5,
            zorder=4
        )
        ax2.annotate(
            row["Model / Approach"],
            (max(row["Runtime (s)"], 0.1), row["ROC-AUC"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.5,
            fontweight="bold"
        )
    
    ax2.set_xscale("log")
    ax2.set_xlabel("Runtime in Seconds (Log Scale)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("ROC-AUC Score", fontsize=11, fontweight="semibold")
    ax2.set_title("Performance vs. Latency Trade-Off", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.2, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Error Category Breakdown (Heatmap / Grouped Bar)
    ax3 = axes[1, 0]
    error_cols = [c for c in df_errors.columns if c != "Model / Approach"]
    x_err = np.arange(len(error_cols))
    
    # Compare Approach A, Approach B, and Hybrid
    key_models = [m for m in df_errors["Model / Approach"].tolist() if "Approach" in m or "Hybrid" in m]
    width_err = 0.8 / max(len(key_models), 1)
    
    for i, model in enumerate(key_models):
        row = df_errors[df_errors["Model / Approach"] == model].iloc[0]
        vals = [row[c] for c in error_cols]
        ax3.bar(x_err + (i - len(key_models)/2 + 0.5) * width_err, vals, width_err, label=model, alpha=0.85)
    
    ax3.set_ylabel("Recall on Error Type (0.0 to 1.0)", fontsize=11, fontweight="semibold")
    ax3.set_title("Recall Across 5 Mutation Error Categories", fontsize=12, fontweight="bold")
    ax3.set_xticks(x_err)
    ax3.set_xticklabels(error_cols, rotation=20, ha="right", fontsize=9)
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc="upper right", frameon=True, fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # 4. Confusion Matrix Summary (True Positives vs False Positives)
    ax4 = axes[1, 1]
    tp_vals = df_metrics["TP"].tolist()
    fp_vals = df_metrics["FP"].tolist()
    
    ax4.bar(x - 0.15, tp_vals, width=0.3, label="True Positives (Max 16)", color="#10b981", alpha=0.9)
    ax4.bar(x + 0.15, fp_vals, width=0.3, label="False Positives (Lower is better)", color="#ef4444", alpha=0.8)
    
    ax4.set_ylabel("Count of Sentences", fontsize=11, fontweight="semibold")
    ax4.set_title("True Positives vs. False Positives (Testset = 462)", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(display_models, rotation=25, ha="right", fontsize=9)
    ax4.legend(loc="upper right", frameon=True)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
