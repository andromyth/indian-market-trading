from __future__ import annotations

from delivery_policy_update import ACTIVE_DELIVERY_MODEL_VERSIONS
import delivery_policy_update as updater


def test_policy_join_excludes_legacy_model_and_unknown_quality(monkeypatch):
    active = sorted(ACTIVE_DELIVERY_MODEL_VERSIONS)[0]
    monkeypatch.setattr(updater, "load_records", lambda: [
        {
            "prediction_id": "legacy-1", "symbol": "RELIANCE", "mode": "Delivery",
            "decision_timestamp": "2026-01-05T15:30:00+05:30",
            "probability_up_next_session": 0.9, "predicted_return_next_session": 0.01,
            "production_data_quality_gate": True, "model_version": "2026.09-stage2",
        },
        {
            "prediction_id": "active-1", "symbol": "RELIANCE", "mode": "Delivery",
            "decision_timestamp": "2026-01-06T15:30:00+05:30",
            "probability_up_next_session": 0.9, "predicted_return_next_session": 0.01,
            "production_data_quality_gate": True, "model_version": active,
        },
    ])
    monkeypatch.setattr(updater, "load_validation_records", lambda: [
        {"prediction_id": "legacy-1", "actual_direction_d1": "UP", "validation_status": "VALIDATED", "actual_d1_return": 0.01},
        {"prediction_id": "active-1", "actual_direction_d1": "UP", "validation_status": "VALIDATED", "actual_d1_return": 0.01},
    ])
    joined = updater._join()
    assert joined["prediction_id"].tolist() == ["active-1"]
    assert joined["model_version"].iloc[0] == active

    monkeypatch.setattr(updater, "load_records", lambda: [{
        "prediction_id": "active-2", "symbol": "RELIANCE", "mode": "Delivery",
        "decision_timestamp": "2026-01-07T15:30:00+05:30",
        "probability_up_next_session": 0.9, "predicted_return_next_session": 0.01,
        "model_version": active,
    }])
    monkeypatch.setattr(updater, "load_validation_records", lambda: [
        {"prediction_id": "active-2", "actual_direction_d1": "UP", "validation_status": "VALIDATED", "actual_d1_return": 0.01},
    ])
    assert updater._join().empty
