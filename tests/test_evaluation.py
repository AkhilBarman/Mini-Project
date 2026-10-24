"""
Unit tests for hd.evaluation.

Tests hand-calculated metric formulas, oracle baseline perfection, random chance level,
and threshold sweep edge cases.
"""

import time
import numpy as np
import pandas as pd
import pytest
from hd.evaluation import (
    Timer,
    best_f1_threshold,
    compute_metrics,
    make_comparison_table,
    per_error_type_recall,
    save_results,
)


def test_compute_metrics_hand_calculated():
    # Hand-crafted toy dataset:
    # y_true: 4 samples [1, 1, 0, 0]
    # scores: [0.9, 0.4, 0.2, 0.8]
    # At threshold 0.5:
    # y_pred: [1, 0, 0, 1]
    # TP = 1 (idx 0), FN = 1 (idx 1), TN = 1 (idx 2), FP = 1 (idx 3)
    # Accuracy  = (1 + 1) / 4 = 0.50
    # Precision = 1 / (1 + 1) = 0.50
    # Recall    = 1 / (1 + 1) = 0.50
    # F1        = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.50
    y_true = [1, 1, 0, 0]
    scores = [0.9, 0.4, 0.2, 0.8]

    metrics = compute_metrics(y_true, scores, threshold=0.5)

    assert metrics["accuracy"] == 0.5000
    assert metrics["precision"] == 0.5000
    assert metrics["recall"] == 0.5000
    assert metrics["f1"] == 0.5000
    assert metrics["confusion_matrix"]["tp"] == 1
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["tn"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["n_pos"] == 2
    assert metrics["n_neg"] == 2


def test_oracle_perfect_scores():
    y_true = [1, 0, 1, 0, 0, 1]
    oracle_scores = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]

    metrics = compute_metrics(y_true, oracle_scores, threshold=0.5)
    assert metrics["accuracy"] == 1.0000
    assert metrics["precision"] == 1.0000
    assert metrics["recall"] == 1.0000
    assert metrics["f1"] == 1.0000
    assert metrics["roc_auc"] == 1.0000
    assert metrics["average_precision"] == 1.0000
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["fn"] == 0


def test_random_near_chance():
    rng = np.random.RandomState(42)
    # Large sample to verify statistical convergence towards 0.50 AUC
    y_true = rng.choice([0, 1], size=1000, p=[0.8, 0.2])
    scores = rng.uniform(0.0, 1.0, size=1000)

    metrics = compute_metrics(y_true, scores, threshold=0.5)
    # ROC AUC for random guessing must be approximately 0.50
    assert 0.45 <= metrics["roc_auc"] <= 0.55


def test_best_f1_threshold_sweep():
    # Dataset where positive samples have scores around 0.7-0.9
    # and negatives have scores around 0.1-0.3
    y_true = [1, 1, 1, 0, 0, 0]
    scores = [0.85, 0.75, 0.65, 0.25, 0.15, 0.05]

    res = best_f1_threshold(y_true, scores)
    assert res["best_f1"] == 1.0000
    assert 0.25 < res["best_threshold"] <= 0.65
    assert "OPTIMISTIC" in res["note"]


def test_edge_case_empty_or_single_class():
    # Empty inputs
    m_empty = compute_metrics([], [], threshold=0.5)
    assert m_empty["accuracy"] == 0.0

    # All negative class
    y_all_zero = [0, 0, 0]
    scores = [0.1, 0.2, 0.3]
    m_zero = compute_metrics(y_all_zero, scores, threshold=0.5)
    assert m_zero["precision"] == 0.0
    assert m_zero["recall"] == 0.0


def test_timer_context_manager():
    with Timer("test_task") as t:
        time.sleep(0.015)
    assert t.elapsed_seconds >= 0.01
    assert t.description == "test_task"


def test_per_error_type_recall():
    df_test = pd.DataFrame({
        "label": [1, 1, 1, 0],
        "error_type": ["DATE_NUMBER", "DATE_NUMBER", "NEGATION", ""]
    })
    # Scores: DATE_NUMBER: one 0.8 (detected), one 0.3 (missed); NEGATION: 0.9 (detected)
    scores = [0.8, 0.3, 0.9, 0.1]
    res = per_error_type_recall(df_test, scores, threshold=0.5)

    assert "DATE_NUMBER" in res
    assert res["DATE_NUMBER"]["total"] == 2
    assert res["DATE_NUMBER"]["detected"] == 1
    assert res["DATE_NUMBER"]["recall"] == 0.5

    assert "NEGATION" in res
    assert res["NEGATION"]["total"] == 1
    assert res["NEGATION"]["detected"] == 1
    assert res["NEGATION"]["recall"] == 1.0


def test_save_and_make_comparison_table(tmp_path):
    res_dir = tmp_path / "results"
    res_dir.mkdir()

    metrics = {"accuracy": 0.95, "f1": 0.88, "roc_auc": 0.98}
    meta = {"approach": "Approach A", "model_name": "SummaC-Conv"}
    save_results(metrics, str(res_dir / "res1.json"), metadata=meta)

    df_comp = make_comparison_table(str(res_dir))
    assert len(df_comp) == 1
    assert df_comp.iloc[0]["approach"] == "Approach A"
    assert df_comp.iloc[0]["f1"] == 0.88
