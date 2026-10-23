from __future__ import annotations

import pandas as pd
import pytest

import all_nse_prediction_learning as learning
from all_nse_prediction_learning import (
    bounded_calibration_sample,
    match_matured_outcomes,
    prediction_manifest_is_clean,
    quote_manifest_is_clean,
    select_row_level_clean_predictions,
    calibration_model_version,
    is_usable_model_version,
    resolve_base_model_version,
)


def prediction_manifest(**overrides):
    base = {
        "status": "OK",
        "quality_gate_pass": True,
        "expected_symbol_count": 2,
        "row_count": 2,
        "prediction_count": 2,
        "missing_quote_count": 0,
        "data_limited_count": 0,
        "untrained_symbol_count": 0,
        "model_failure_count": 0,
        "trained_symbol_manifest_available": True,
        "history_ready": True,
        "history_seed_coverage_symbols": 2,
        "history_seed_failed_batches": 0,
    }
    return {**base, **overrides}


def quote_manifest(**overrides):
    base = {
        "status": "OK",
        "expected_symbol_count": 2,
        "row_count": 2,
        "quoted_symbol_count": 2,
        "missing_quote_count": 0,
        "unmapped_symbol_count": 0,
        "pending_snapshot_recovery": {"errors": []},
        "pending_manifest_recovery": {"errors": []},
    }
    return {**base, **overrides}


def test_complete_full_nse_prediction_manifest_is_eligible():
    assert prediction_manifest_is_clean(prediction_manifest())


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "PARTIAL"},
        {"quality_gate_pass": False},
        {"row_count": 1},
        {"prediction_count": 1},
        {"missing_quote_count": 1},
        {"data_limited_count": 1},
        {"untrained_symbol_count": 1},
        {"model_failure_count": 1},
        {"history_ready": False},
        {"history_seed_coverage_symbols": 1},
        {"history_seed_failed_batches": 1},
    ],
)
def test_partial_or_low_readiness_prediction_slot_is_quarantined(changes):
    assert not prediction_manifest_is_clean(prediction_manifest(**changes))


def test_complete_quote_manifest_is_eligible():
    assert quote_manifest_is_clean(quote_manifest())


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "PARTIAL"},
        {"row_count": 1},
        {"quoted_symbol_count": 1},
        {"missing_quote_count": 1},
        {"unmapped_symbol_count": 1},
        {"pending_snapshot_recovery": {"errors": ["upload failed"]}},
        {"pending_manifest_recovery": {"errors": ["upload failed"]}},
    ],
)
def test_partial_quote_slot_is_quarantined(changes):
    assert not quote_manifest_is_clean(quote_manifest(**changes))


def _prediction(target="2026-10-01T04:30:00Z"):
    return {
        "symbol": "ABC",
        "actual_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_market_value": 100.0,
        "actual_quote_status": "PASS",
        "actual_quote_age_seconds": 5.0,
        "forecast_target_timestamp_utc": target,
        "forecast_key": "ABC|2026-10-01T04:00:00Z|30|model-v1",
        "probability_up_30m": 0.62,
        "probability_up_30m_raw": 0.62,
        "expected_return_30m": 0.01,
        "expected_return_30m_raw": 0.01,
        "raw_model_signal": "BUY",
        "signal_raw": "BUY",
        "prediction_status": "PASS",
        "model_version": "all_nse_intraday__2026-09-30",
        "model_family": "logistic_regression+ridge",
        "data_readiness_score": 1.0,
    }


def test_matured_outcome_uses_nearest_clean_target_quote_within_tolerance():
    predictions = pd.DataFrame([_prediction()])
    quotes = pd.DataFrame([
        {
            "symbol": "ABC",
            "quote_observed_at_utc": "2026-10-01T04:30:15Z",
            "actual_observed_price": 102.0,
            "quote_status": "PASS",
        },
        {
            "symbol": "ABC",
            "quote_observed_at_utc": "2026-10-01T04:29:00Z",
            "actual_observed_price": 101.0,
            "quote_status": "PASS",
        },
        {
            "symbol": "ABC",
            "quote_observed_at_utc": "2026-10-01T04:30:20Z",
            "actual_observed_price": 999.0,
            "quote_status": "PARTIAL",
        },
    ])
    result = match_matured_outcomes(predictions, quotes)
    assert len(result) == 1
    assert result.loc[0, "validation_status"] == "VALIDATED"
    assert result.loc[0, "actual_target_price"] == pytest.approx(102.0)
    assert result.loc[0, "actual_return"] == pytest.approx(0.02)
    assert result.loc[0, "target_quote_distance_seconds"] == pytest.approx(15.0)
    assert result.loc[0, "model_version"] == "all_nse_intraday__2026-09-30"


def test_unmatched_or_too_distant_target_quote_is_not_scored():
    predictions = pd.DataFrame([_prediction()])
    quotes = pd.DataFrame([
        {
            "symbol": "ABC",
            "quote_observed_at_utc": "2026-10-01T04:35:00Z",
            "actual_observed_price": 102.0,
            "quote_status": "PASS",
        }
    ])
    result = match_matured_outcomes(predictions, quotes)
    assert result.loc[0, "validation_status"] == "PENDING_TARGET_QUOTE"
    assert pd.isna(result.loc[0, "actual_return"])


def test_matcher_never_uses_partial_quote_rows():
    predictions = pd.DataFrame([_prediction()])
    quotes = pd.DataFrame([
        {
            "symbol": "ABC",
            "quote_observed_at_utc": "2026-10-01T04:30:00Z",
            "actual_observed_price": 105.0,
            "quote_status": "QUOTE_REQUEST_ERROR",
        }
    ])
    result = match_matured_outcomes(predictions, quotes)
    assert result.loc[0, "validation_status"] == "PENDING_TARGET_QUOTE"
    assert pd.isna(result.loc[0, "actual_return"])


def test_calibration_sample_is_bounded_deterministic_and_stratified_by_target_slot():
    rows = []
    for day in ("2026-10-01", "2026-10-02"):
        for minute in (0, 5, 10):
            for stock in range(100):
                rows.append({
                    "symbol": f"S{stock:03d}",
                    "target_timestamp_utc": f"{day}T04:{minute:02d}:00Z",
                    "probability_up": 0.4 + (stock % 20) / 100,
                    "actual_return": 0.001 if stock % 2 else -0.001,
                })
    frame = pd.DataFrame(rows)
    one = bounded_calibration_sample(frame, max_rows_per_session=20, expected_slots_per_session=3)
    two = bounded_calibration_sample(frame, max_rows_per_session=20, expected_slots_per_session=3)
    assert len(one) <= 40
    assert one.equals(two)
    assert set(one["target_timestamp_utc"]).issubset(set(frame["target_timestamp_utc"]))
    assert one.groupby(pd.to_datetime(one["target_timestamp_utc"], utc=True).dt.strftime("%Y-%m-%d")).size().max() <= 20


def test_partial_slot_keeps_only_rows_matching_clean_same_slot_quotes():
    rows = [
        {
            **_prediction(target="2026-10-01T04:30:00Z"),
            "symbol": "AAA",
            "actual_observed_at_utc": "2026-10-01T04:00:02Z",
            "actual_market_value": 100.0,
            "actual_quote_status": "PASS",
            "actual_quote_age_seconds": 10.0,
            "trained_for_symbol": True,
            "data_limited": False,
            "data_readiness_score": 0.95,
        },
        {
            **_prediction(target="2026-10-01T04:30:00Z"),
            "symbol": "BBB",
            "actual_observed_at_utc": None,
            "actual_market_value": None,
            "actual_quote_status": "MISSING_QUOTE",
            "actual_quote_age_seconds": None,
            "trained_for_symbol": False,
            "data_limited": True,
            "data_readiness_score": 0.20,
            "prediction_status": "MISSING_CURRENT_QUOTE",
        },
        {
            **_prediction(target="2026-10-01T04:30:00Z"),
            "symbol": "CCC",
            "actual_observed_at_utc": "2026-10-01T04:00:04Z",
            "actual_market_value": 50.0,
            "actual_quote_status": "PASS",
            "actual_quote_age_seconds": 4.0,
            "trained_for_symbol": True,
            "data_limited": False,
            "data_readiness_score": 0.90,
        },
    ]
    predictions = pd.DataFrame(rows)
    contemporaneous_quotes = pd.DataFrame([
        {
            "symbol": "AAA",
            "quote_observed_at_utc": "2026-10-01T04:00:03Z",
            "actual_observed_price": 100.0,
            "quote_status": "PASS",
        },
        {
            "symbol": "BBB",
            "quote_observed_at_utc": None,
            "actual_observed_price": None,
            "quote_status": "MISSING_QUOTE",
        },
        {
            "symbol": "CCC",
            "quote_observed_at_utc": "2026-10-01T04:00:04Z",
            "actual_observed_price": 49.5,
            "quote_status": "PASS",
        },
    ])
    manifest = prediction_manifest(
        status="PARTIAL",
        quality_gate_pass=False,
        expected_symbol_count=3,
        row_count=3,
        prediction_count=1,
        missing_quote_count=1,
    )
    clean, gate = select_row_level_clean_predictions(
        predictions, contemporaneous_quotes, manifest
    )
    assert clean["symbol"].tolist() == ["AAA"]
    assert gate["slot_status"] == "PARTIAL"
    assert gate["input_rows"] == 3
    assert gate["quote_matched_rows"] == 1
    assert gate["clean_rows"] == 1
    assert gate["quarantined_rows"] == 2
    assert gate["condition_counts"]["trained_true"] == 2
    assert gate["condition_counts"]["trained_false"] == 1
    assert gate["condition_counts"]["quote_matched"] == 1
    assert gate["condition_counts"]["all_conditions"] == 1


def test_row_level_gate_accepts_pass_pooled_prediction_with_unknown_membership():
    pred = _prediction(target="2026-10-01T04:30:00Z")
    pred.update({
        "symbol": "AAA",
        "actual_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_market_value": 100.0,
        "actual_quote_age_seconds": 5.0,
        "trained_for_symbol": None,
        "data_limited": False,
        "data_readiness_score": 0.95,
    })
    quote = pd.DataFrame([{
        "symbol": "AAA",
        "quote_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_observed_price": 100.0,
        "quote_status": "PASS",
    }])
    clean, gate = select_row_level_clean_predictions(
        pd.DataFrame([pred]), quote, prediction_manifest(status="PARTIAL")
    )
    assert clean["symbol"].tolist() == ["AAA"]
    assert gate["condition_counts"]["trained_unknown"] == 1
    assert gate["condition_counts"]["all_non_quote_conditions"] == 1
    assert gate["condition_counts"]["all_conditions"] == 1


def test_row_level_gate_rejects_explicitly_untrained_prediction():
    pred = _prediction(target="2026-10-01T04:30:00Z")
    pred.update({
        "symbol": "AAA",
        "actual_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_market_value": 100.0,
        "actual_quote_age_seconds": 5.0,
        "prediction_status": "PASS_UNTRAINED_SYMBOL",
        "trained_for_symbol": False,
        "data_limited": False,
        "data_readiness_score": 0.95,
    })
    quote = pd.DataFrame([{
        "symbol": "AAA",
        "quote_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_observed_price": 100.0,
        "quote_status": "PASS",
    }])
    clean, gate = select_row_level_clean_predictions(
        pd.DataFrame([pred]), quote, prediction_manifest(status="PARTIAL")
    )
    assert clean.empty
    assert gate["condition_counts"]["trained_false"] == 1
    assert gate["condition_counts"]["prediction_status_pass"] == 0
    assert gate["condition_counts"]["all_conditions"] == 0


def test_row_level_gate_excludes_stale_same_symbol_quote_and_low_readiness():
    pred = _prediction(target="2026-10-01T04:30:00Z")
    pred.update({
        "symbol": "AAA",
        "actual_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_market_value": 100.0,
        "actual_quote_age_seconds": 200.0,
        "trained_for_symbol": True,
        "data_limited": False,
        "data_readiness_score": 0.50,
    })
    quote = pd.DataFrame([{
        "symbol": "AAA",
        "quote_observed_at_utc": "2026-10-01T04:03:30Z",
        "actual_observed_price": 100.0,
        "quote_status": "PASS",
    }])
    clean, gate = select_row_level_clean_predictions(
        pd.DataFrame([pred]), quote, prediction_manifest(status="PARTIAL")
    )
    assert clean.empty
    assert gate["clean_rows"] == 0
    assert gate["quarantined_rows"] == 1


def test_run_learning_reports_prediction_count_after_releasing_source_frames(tmp_path, monkeypatch):
    class FakeDriveStore:
        def upload_or_update(self, local_path, remote_relative_path, mime_type="application/octet-stream"):
            assert local_path.exists()
            return {
                "status": "OK",
                "action": "updated",
                "remote_path": remote_relative_path,
                "file_id": "fake-drive-id",
            }

    source_predictions = pd.DataFrame([
        {"symbol": "AAA", "probability_up_30m": 0.6}
    ])
    monkeypatch.setattr(learning, "GoogleDriveStore", FakeDriveStore)
    monkeypatch.setattr(
        learning,
        "load_clean_session_data",
        lambda store, start, end: (
            source_predictions.copy(),
            pd.DataFrame(),
            {"clean_prediction_rows": 1, "prediction_slots_seen": 1},
        ),
    )
    monkeypatch.setattr(
        learning,
        "match_matured_outcomes",
        lambda predictions, quotes: pd.DataFrame(),
    )

    report = learning.run_learning(
        pd.Timestamp("2026-10-01").date(),
        pd.Timestamp("2026-10-01").date(),
        output_dir=tmp_path,
    )
    assert report["outcome_summary"]["prediction_rows_in_clean_slots"] == 1
    assert report["outcome_summary"]["matched_validated_outcomes"] == 0
    assert report["recommendation_update"]["status"] == "WAITING_FOR_CLEAN_MATURED_OUTCOMES"
    assert report["production_model_changed"] is False


def test_data_limited_predictions_are_eligible_only_for_low_readiness_calibration():
    pred = _prediction(target="2026-10-01T04:30:00Z")
    pred.update({
        "symbol": "AAA",
        "actual_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_market_value": 100.0,
        "actual_quote_age_seconds": 5.0,
        "actual_quote_status": "PASS",
        "prediction_status": "PASS_DATA_LIMITED",
        "trained_for_symbol": True,
        "data_limited": True,
        "data_readiness_score": 0.5294,
        "base_model_version": "all_nse_intraday__artifact123",
    })
    quote = pd.DataFrame([{
        "symbol": "AAA",
        "quote_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_observed_price": 100.0,
        "quote_status": "PASS",
    }])
    clean, gate = select_row_level_clean_predictions(
        pd.DataFrame([pred]), quote, prediction_manifest(status="PARTIAL")
    )
    assert clean["symbol"].tolist() == ["AAA"]
    assert gate["condition_counts"]["prediction_status_pass_data_limited"] == 1
    assert gate["condition_counts"]["data_limited_true"] == 1
    assert gate["condition_counts"]["readiness_valid"] == 1
    assert gate["condition_counts"]["all_conditions"] == 1
    assert calibration_model_version("all_nse_intraday__artifact123", 0.5294) == (
        "all_nse_intraday__artifact123__READINESS_LOW"
    )
    assert calibration_model_version("all_nse_intraday__artifact123", 0.95) == (
        "all_nse_intraday__artifact123__READINESS_READY"
    )


def test_near_empty_readiness_and_explicitly_untrained_rows_remain_ineligible():
    pred = _prediction(target="2026-10-01T04:30:00Z")
    pred.update({
        "symbol": "AAA",
        "actual_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_market_value": 100.0,
        "actual_quote_age_seconds": 5.0,
        "actual_quote_status": "PASS",
        "prediction_status": "PASS_DATA_LIMITED",
        "trained_for_symbol": False,
        "data_limited": True,
        "data_readiness_score": 0.20,
    })
    quote = pd.DataFrame([{
        "symbol": "AAA",
        "quote_observed_at_utc": "2026-10-01T04:00:00Z",
        "actual_observed_price": 100.0,
        "quote_status": "PASS",
    }])
    clean, gate = select_row_level_clean_predictions(
        pd.DataFrame([pred]), quote, prediction_manifest(status="PARTIAL")
    )
    assert clean.empty
    assert gate["condition_counts"]["trained_false"] == 1
    assert gate["condition_counts"]["readiness_valid"] == 0
    assert gate["condition_counts"]["all_conditions"] == 0


def test_matcher_falls_back_to_model_values_when_raw_columns_exist_but_are_null():
    pred = _prediction(target="2026-10-01T04:30:00Z")
    # This is what older Drive Parquets look like after the loader has
    # materialized newer schema columns with null values.
    pred["probability_up_30m_raw"] = None
    pred["expected_return_30m_raw"] = None
    pred["probability_up_30m"] = 0.62
    pred["expected_return_30m"] = 0.01
    predictions = pd.DataFrame([pred])
    quotes = pd.DataFrame([
        {
            "symbol": "ABC",
            "quote_observed_at_utc": "2026-10-01T04:30:15Z",
            "actual_observed_price": 102.0,
            "quote_status": "PASS",
        }
    ])
    result = match_matured_outcomes(predictions, quotes)
    assert result.loc[0, "validation_status"] == "VALIDATED"
    assert result.loc[0, "probability_up"] == pytest.approx(0.62)
    assert result.loc[0, "expected_return"] == pytest.approx(0.01)
    assert result.loc[0, "actual_return"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    ("row_base", "row_version", "manifest_version", "expected", "source"),
    [
        (
            "LEGACY_UNKNOWN",
            "LEGACY_UNKNOWN__READINESS_READY",
            "all_nse_intraday__2026-10-22__classifier123_regressor456",
            "all_nse_intraday__2026-10-22__classifier123_regressor456",
            "manifest_model_version",
        ),
        (
            "all_nse_intraday__row-hash-abc",
            "all_nse_intraday__row-hash-abc__READINESS_READY",
            "all_nse_intraday__manifest-hash-xyz",
            "all_nse_intraday__row-hash-abc",
            "row_base_model_version",
        ),
        (
            "",
            "all_nse_intraday__row-hash-abc__READINESS_LOW",
            "all_nse_intraday__manifest-hash-xyz",
            "all_nse_intraday__row-hash-abc",
            "row_model_version",
        ),
        (
            "unversioned",
            "all_nse_intraday__unversioned__READINESS_READY",
            "all_nse_intraday__unversioned",
            "LEGACY_UNKNOWN",
            "legacy_unknown_fallback",
        ),
    ],
)
def test_model_version_resolution_skips_sentinels_and_uses_manifest_fallback(
    row_base, row_version, manifest_version, expected, source
):
    assert resolve_base_model_version(row_base, row_version, manifest_version) == (expected, source)


@pytest.mark.parametrize(
    "value",
    [
        None, "", "LEGACY_UNKNOWN", "UNKNOWN", "unversioned",
        "all_nse_intraday__unversioned",
        "LEGACY_UNKNOWN__READINESS_LOW",
        "LEGACY_UNKNOWN__READINESS_READY",
        "LEGACY_UNKNOWN__READINESS_UNKNOWN",
    ],
)
def test_unusable_model_version_sentinels_are_not_fit_eligible(value):
    assert not is_usable_model_version(value)


def test_versioned_artifact_with_readiness_suffix_is_fit_eligible():
    assert is_usable_model_version("all_nse_intraday__2026-10-22__abc__READINESS_READY")
