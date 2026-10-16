from __future__ import annotations

import pandas as pd
import pytest

from prediction_learning_engine import (
    _feature_error_associations,
    _fit_probability_calibrator,
    build_learning_report,
    chronological_split,
    deduplicate_forecasts,
    forecast_identity,
    probability_metrics,
)


def test_duplicate_forecasts_are_removed_using_explicit_key():
    records = [
        {"forecast_key": "same", "symbol": "SBIN", "signal": "BUY"},
        {"forecast_key": "same", "symbol": "SBIN", "signal": "BUY"},
        {"forecast_key": "other", "symbol": "RELIANCE", "signal": "SELL"},
    ]
    unique, report = deduplicate_forecasts(records)
    assert len(unique) == 2
    assert report["duplicate_rows_removed"] == 1
    assert report["duplicate_forecast_keys"] == 1
    assert forecast_identity(unique[0]) == "same"


def test_chronological_split_keeps_identical_target_minutes_together():
    frame = pd.DataFrame({
        "target_timestamp_utc": [
            "2026-10-16T04:00:00Z",
            "2026-10-16T04:00:00Z",
            "2026-10-16T04:01:00Z",
            "2026-10-16T04:02:00Z",
            "2026-10-16T04:02:00Z",
            "2026-10-16T04:03:00Z",
        ],
        "row": list(range(6)),
    })
    train, test, split_at = chronological_split(frame, train_fraction=0.5)
    assert split_at is not None
    assert not set(train["target_timestamp_utc"]).intersection(set(test["target_timestamp_utc"]))
    assert train["row"].max() < test["row"].min()
    assert len(train) + len(test) == len(frame)


def test_probability_metrics_match_known_brier_score():
    result = probability_metrics([0.9, 0.1], [1, 0])
    assert result["n"] == 2
    assert result["brier"] == pytest.approx(0.01)
    assert result["accuracy_at_0_5"] == 1.0


def test_probability_calibrator_does_not_fit_small_samples():
    train = pd.DataFrame({
        "probability_up": [0.6, 0.4, 0.7],
        "actual_return": [0.01, -0.01, 0.02],
        "target_timestamp_utc": pd.date_range("2026-10-01", periods=3, freq="min", tz="UTC"),
    })
    test = train.copy()
    result = _fit_probability_calibrator(train, test, "2026-10-01T00:02:00Z")
    assert result["status"] == "INSUFFICIENT_HISTORY"
    assert result["training_rows"] == 3


def test_feature_error_associations_preserve_row_alignment():
    frame = pd.DataFrame({
        "validation_status": ["VALIDATED"] * 5,
        "actual_return": [0.01, 0.02, 0.03, 0.04, 0.05],
        "expected_return": [0.0] * 5,
        "diagnostic_feature_snapshot": [
            {"trend": float(i), "constant": 1.0} for i in range(5)
        ],
    })
    result = _feature_error_associations(frame, min_samples=3)
    assert result
    assert result[0]["feature"] == "trend"
    assert result[0]["spearman_correlation_with_return_residual"] == pytest.approx(1.0)


def test_learning_report_never_auto_applies_corrections():
    n = 35
    target = pd.date_range("2026-10-01T04:00:00Z", periods=n, freq="min")
    actual = [0.01 if i % 2 == 0 else -0.01 for i in range(n)]
    signals = ["BUY" if i % 2 == 0 else "SELL" for i in range(n)]
    rows = pd.DataFrame({
        "validation_status": ["VALIDATED"] * n,
        "symbol": ["SBIN"] * n,
        "horizon_min": [5] * n,
        "target_timestamp_utc": target,
        "decision_timestamp_utc": target - pd.Timedelta(minutes=5),
        "signal": signals,
        "direction_match": [True] * n,
        "probability_up": [0.7 if v > 0 else 0.3 for v in actual],
        "actual_return": actual,
        "expected_return": actual,
        "predicted_price_abs_error": [0.2] * n,
        "baseline_price_abs_error": [0.5] * n,
        "brier_score": [0.09] * n,
        "signed_signal_return_before_costs": [0.01] * n,
        "model_version": ["test-model"] * n,
        "source": ["INDSTOCKS_API"] * n,
        "decision_time_bucket_30m": ["10:00-10:30"] * n,
        "feature_coverage": [1.0] * n,
        "confidence": ["MEDIUM"] * n,
        "input_age_seconds": [10] * n,
        "missing_model_features_count": [0] * n,
        "diagnostic_feature_snapshot": [{"ret_1": float(i)} for i in range(n)],
    })
    report, groups, bins = build_learning_report(
        rows,
        {"input_rows": n, "unique_forecasts": n, "duplicate_rows_removed": 0},
    )
    assert report["read_only"] is True
    assert report["production_model_changed"] is False
    assert report["production_thresholds_changed"] is False
    assert report["orders_placed"] is False
    assert "h5__test-model" in report["candidate_corrections"]
    assert not groups.empty
    assert not bins.empty



def test_feature_error_associations_work_without_return_regression_head():
    frame = pd.DataFrame({
        "validation_status": ["VALIDATED"] * 5,
        "actual_return": [0.01, -0.01, 0.01, -0.01, 0.01],
        "probability_up": [0.2, 0.8, 0.3, 0.7, 0.4],
        "expected_return": [float("nan")] * 5,
        "diagnostic_feature_snapshot": [
            {"feature_x": float(i)} for i in range(5)
        ],
    })
    result = _feature_error_associations(frame, min_samples=3)
    assert result
    assert all(
        row["residual_type"] == "classification_probability_residual"
        for row in result
    )


def test_runtime_diagnostic_snapshot_records_exact_inputs_and_nulls():
    from selfdriving.sd_v2_runtime import _diagnostic_feature_snapshot

    frame = pd.DataFrame({
        "price_return_1": [0.012345],
        "volume_ratio": [float("nan")],
    })
    result = _diagnostic_feature_snapshot(
        frame, ["price_return_1", "volume_ratio", "missing_column"]
    )
    assert result["price_return_1"] == pytest.approx(0.012345)
    assert result["volume_ratio"] is None
    assert result["missing_column"] is None
