"""
Unit tests for comparison and ensembling module (Phase 5).
"""

import numpy as np
import pandas as pd
import pytest

from hd.comparison import (
    calibrate_score,
    compute_hybrid_scores,
    build_comparison_dataframe,
    build_error_breakdown_dataframe,
    dataframe_to_markdown
)


def test_calibrate_score_scalars():
    """Verify piecewise linear calibration at key boundary points."""
    assert calibrate_score(0.0, center=0.85) == pytest.approx(0.0)
    assert calibrate_score(0.85, center=0.85) == pytest.approx(0.50)
    assert calibrate_score(1.0, center=0.85) == pytest.approx(1.0)
    # Check midpoint of lower segment (0.425 -> 0.25)
    assert calibrate_score(0.425, center=0.85) == pytest.approx(0.25)


def test_calibrate_score_array():
    """Verify vectorised array calibration."""
    arr = np.array([0.0, 0.85, 1.0])
    calibrated = calibrate_score(arr, center=0.85)
    assert len(calibrated) == 3
    assert calibrated[0] == pytest.approx(0.0)
    assert calibrated[1] == pytest.approx(0.5)
    assert calibrated[2] == pytest.approx(1.0)


def test_compute_hybrid_scores_methods():
    """Verify different ensembling fusion strategies."""
    s_a = np.array([0.85, 0.0, 0.925])
    s_b = np.array([0.10, 0.90, 0.40])
    
    # calibrated_max
    h_max = compute_hybrid_scores(s_a, s_b, method="calibrated_max", theta_a=0.85)
    assert len(h_max) == 3
    # s_a[0]=0.85 -> cal=0.5, s_b[0]=0.10 -> max is 0.5
    assert h_max[0] == pytest.approx(0.50)
    # s_a[1]=0.0 -> cal=0.0, s_b[1]=0.90 -> max is 0.90
    assert h_max[1] == pytest.approx(0.90)
    
    # calibrated_mean
    h_mean = compute_hybrid_scores(s_a, s_b, method="calibrated_mean", theta_a=0.85)
    assert h_mean[0] == pytest.approx(0.30)
    
    # linear_weighted
    h_lin = compute_hybrid_scores(s_a, s_b, method="linear_weighted")
    assert h_lin[0] == pytest.approx(0.6 * 0.85 + 0.4 * 0.10)


def test_compute_hybrid_scores_invalid_method():
    """Verify ValueError on unknown method."""
    with pytest.raises(ValueError, match="Unknown hybrid method"):
        compute_hybrid_scores(np.array([0.5]), np.array([0.5]), method="unsupported_method")


def test_build_comparison_dataframe():
    """Verify DataFrame generation from mock evaluated runs."""
    mock_runs = [
        {
            "name": "Model Alpha",
            "metrics": {
                "threshold": 0.5,
                "accuracy": 0.88,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.67,
                "roc_auc": 0.92,
                "average_precision": 0.75,
                "confusion_matrix": {"tp": 5, "fp": 5, "tn": 40, "fn": 0}
            },
            "metadata": {"seconds_total": 12.34}
        }
    ]
    df = build_comparison_dataframe(mock_runs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["Model / Approach"] == "Model Alpha"
    assert df.iloc[0]["Accuracy"] == 0.88
    assert df.iloc[0]["TP"] == 5
    assert df.iloc[0]["Runtime (s)"] == 12.34


def test_build_error_breakdown_dataframe():
    """Verify per-error-type recall matrix formatting."""
    mock_runs = [
        {
            "name": "Model Beta",
            "metadata": {
                "per_error_type_recall": {
                    "DATE_NUMBER": {"recall": 1.0, "total": 3, "detected": 3},
                    "FABRICATED": {"recall": 0.5, "total": 4, "detected": 2}
                }
            }
        }
    ]
    df = build_error_breakdown_dataframe(mock_runs)
    assert len(df) == 1
    assert df.iloc[0]["Model / Approach"] == "Model Beta"
    assert df.iloc[0]["DATE_NUMBER"] == 1.0
    assert df.iloc[0]["FABRICATED"] == 0.5


def test_dataframe_to_markdown():
    """Verify markdown table output without tabulate."""
    sample_df = pd.DataFrame({
        "Model": ["M1", "M2"],
        "Score": [0.85, 0.92]
    })
    md = dataframe_to_markdown(sample_df)
    assert "| Model | Score |" in md
    assert "| M1    | 0.85  |" in md
    assert "| M2    | 0.92  |" in md
