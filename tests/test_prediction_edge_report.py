from __future__ import annotations

import numpy as np
import pandas as pd

from prediction_edge_report import build_edge_report


def _frame(probabilities: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "probability_up_session_close": probabilities,
        "target_up": labels,
        "horizon_minutes": np.where(np.arange(len(labels)) % 2 == 0, 30, 390),
    })


def test_random_predictions_do_not_claim_edge() -> None:
    rng = np.random.default_rng(21)
    frame = _frame(rng.uniform(0, 1, 800), rng.integers(0, 2, 800))
    report = build_edge_report(frame)
    assert report["overall"]["verdict"] in {"NO_EDGE_EVIDENCE", "WEAK_EDGE"}
    assert report["overall"]["n"] == 800


def test_informative_predictions_are_actionable() -> None:
    labels = np.array([0, 1] * 300)
    probabilities = np.where(labels == 1, 0.80, 0.20)
    report = build_edge_report(_frame(probabilities, labels))
    assert report["overall"]["verdict"] == "ACTIONABLE_EDGE"
    assert report["overall"]["roc_auc"] == 1.0
    assert report["overall"]["action"]["precision"] == 1.0


def test_small_sample_is_not_called_edge() -> None:
    labels = np.array([0, 1] * 20)
    probabilities = np.where(labels == 1, 0.90, 0.10)
    report = build_edge_report(_frame(probabilities, labels))
    assert report["overall"]["verdict"] == "INSUFFICIENT_SAMPLE"
