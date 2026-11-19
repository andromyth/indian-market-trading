from __future__ import annotations

import numpy as np

from probability_calibration import (
    apply_probability_calibration,
    brier_score,
    choose_threshold,
    fit_platt,
    probability_report,
    wilson_interval,
)


def _overconfident_sample(size: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=size)
    logits = 3.0 * (labels - 0.5) + rng.normal(0.0, 0.5, size=size)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return probabilities, labels


def test_platt_calibration_is_monotonic_and_reduces_brier() -> None:
    probabilities, labels = _overconfident_sample()
    spec = fit_platt(probabilities, labels)
    assert spec is not None
    calibrated = apply_probability_calibration(probabilities, spec)
    order = np.argsort(probabilities)
    assert np.all(np.diff(calibrated[order]) >= -1e-12)
    assert brier_score(calibrated, labels) <= brier_score(probabilities, labels) + 1e-9


def test_platt_returns_none_when_evidence_is_insufficient() -> None:
    assert fit_platt([0.5] * 20, [1] * 20) is None
    assert fit_platt([0.1, 0.2], [0, 1]) is None


def test_wilson_interval_contains_sample_rate() -> None:
    low, high = wilson_interval(7, 10)
    assert low is not None and high is not None
    assert low < 0.7 < high
    assert wilson_interval(0, 0) == (None, None)


def test_threshold_falls_back_without_evidence() -> None:
    rng = np.random.default_rng(11)
    probabilities = rng.uniform(0.0, 1.0, size=600)
    labels = rng.integers(0, 2, size=600)
    assert choose_threshold(probabilities, labels, "buy", default=0.60) == 0.60
    assert choose_threshold(probabilities, labels, "sell", default=0.40) == 0.40


def test_threshold_moves_when_precision_clears_null() -> None:
    rng = np.random.default_rng(13)
    probabilities = np.concatenate([
        rng.normal(0.65, 0.05, 300),
        rng.normal(0.35, 0.05, 300),
    ]).clip(0.01, 0.99)
    labels = np.concatenate([np.ones(300), np.zeros(300)]).astype(int)
    buy = choose_threshold(probabilities, labels, "buy", default=0.60)
    sell = choose_threshold(probabilities, labels, "sell", default=0.40)
    assert buy < 0.60
    assert sell > 0.40


def test_probability_report_uses_actions_not_hold_rows() -> None:
    probabilities = np.array([0.9, 0.8, 0.2, 0.1, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0, 1, 0])
    report = probability_report(probabilities, labels)
    assert report["n"] == 6
    assert report["buy"]["count"] == 2
    assert report["sell"]["count"] == 2
    assert report["action"]["count"] == 4
    assert report["action"]["precision"] == 1.0
