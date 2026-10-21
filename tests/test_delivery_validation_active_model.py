from __future__ import annotations

import pandas as pd

from delivery_validation_diagnostics import ACTIVE_DELIVERY_MODEL_VERSIONS, build_report


def _frames(n: int, model_version: str, quality_gate: bool):
    preds = []
    vals = []
    for i in range(n):
        pid = f"pred-{i}"
        preds.append({
            "prediction_id": pid,
            "symbol": f"SYM{i % 5}",
            "mode": "Delivery",
            "decision_timestamp": (pd.Timestamp("2026-01-01T15:30:00Z") + pd.Timedelta(days=i)).isoformat(),
            "probability_up_next_session": 0.9,
            "predicted_return_next_session": 0.01,
            "production_data_quality_gate": quality_gate,
            "production_data_quality_score": 1.0 if quality_gate else 0.0,
            "model_version": model_version,
            "decision_engine_version": "test-engine",
        })
        vals.append({
            "prediction_id": pid,
            "actual_direction_d1": "UP",
            "actual_d1_return": 0.01,
            "validation_status": "VALIDATED",
        })
    return pd.DataFrame(preds), pd.DataFrame(vals)


def test_legacy_model_successes_cannot_pass_active_model_scorecard():
    preds, vals = _frames(3, "2026.09-stage2", True)
    report = build_report(preds, vals)
    assert report["status"] == "NO_ACTIVE_MODEL_DATA"
    assert report["all_validated_all_versions_diagnostic_only"]["samples"] == 3
    assert report["active_model_scored_samples"] == 0
    assert report["all_validated"]["samples"] == 0


def test_active_model_with_zero_quality_approved_rows_cannot_pass():
    active_version = sorted(ACTIVE_DELIVERY_MODEL_VERSIONS)[0]
    preds, vals = _frames(12, active_version, False)
    report = build_report(preds, vals)
    assert report["status"] == "NO_QUALITY_APPROVED_DATA"
    assert report["active_model_scored_samples"] == 12
    assert report["quality_approved_active_samples"] == 0


def test_seven_active_rows_are_not_enough_to_claim_target_accuracy():
    active_version = sorted(ACTIVE_DELIVERY_MODEL_VERSIONS)[0]
    preds, vals = _frames(7, active_version, True)
    report = build_report(preds, vals)
    assert report["status"] == "INSUFFICIENT_HISTORY"
    assert report["history_sufficient"] is False
    assert report["scored_samples"] == 7


def test_future_active_targets_are_pending_not_missing_accuracy_data():
    active_version = sorted(ACTIVE_DELIVERY_MODEL_VERSIONS)[0]
    active_preds, _ = _frames(12, active_version, True)
    active_preds["prediction_id"] = "active-" + active_preds["prediction_id"].astype(str)
    active_preds["prediction_for_date"] = "2026-10-21"

    legacy_preds, legacy_vals = _frames(3, "2026.09-stage2", True)
    preds = pd.concat([active_preds, legacy_preds], ignore_index=True)
    report = build_report(
        preds,
        legacy_vals,
        latest_official_eod_date="2026-10-19",
    )

    assert report["status"] == "ACTIVE_MODEL_AWAITING_MATURITY"
    assert report["active_model_prediction_count"] == 12
    assert report["active_model_scored_samples"] == 0
    assert report["active_model_matured_prediction_count"] == 0
    assert report["active_model_waiting_for_target_count"] == 12
    assert report["all_validated_all_versions_diagnostic_only"]["samples"] == 3
    assert "unscored until their official target bars arrive" in report["diagnosis"]


def test_matured_active_targets_without_validations_remain_a_hard_gap():
    active_version = sorted(ACTIVE_DELIVERY_MODEL_VERSIONS)[0]
    preds, _ = _frames(4, active_version, True)
    preds["prediction_for_date"] = "2026-10-19"

    report = build_report(
        preds,
        pd.DataFrame(),
        latest_official_eod_date="2026-10-19",
    )

    assert report["status"] == "ACTIVE_MODEL_OUTCOME_GAP"
    assert report["active_model_prediction_count"] == 4
    assert report["active_model_matured_prediction_count"] == 4
    assert report["active_model_waiting_for_target_count"] == 0
    assert report["active_model_scored_samples"] == 0


def test_large_monitoring_sample_still_does_not_claim_holdout_target():
    active_version = sorted(ACTIVE_DELIVERY_MODEL_VERSIONS)[0]
    preds, vals = _frames(80, active_version, True)
    report = build_report(preds, vals)
    assert report["status"] == "OOS_HISTORY_AVAILABLE_TARGET_UNPROVEN"
    assert report["action_precision_target_met_on_monitoring_sample"] is True
    assert "later chronological holdout" in report["target_interpretation"]
