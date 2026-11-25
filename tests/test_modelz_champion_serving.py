from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="autonomous_intraday_worker_service imports the Unix-only resource module"
)


def _champion_status(**overrides):
    features = [f"f{i}" for i in range(42)]
    status = {
        "status": "CHAMPION",
        "target": "target_session_close",
        "artifact_target": "target_session_close",
        "artifact_feature_module": "features_5m",
        "artifact_feature_count": 42,
        "artifact_features": features,
        "prediction_horizon": "session_close",
        "model_version": "v2026-11-20__beddd1883873",
        "trained_through": "2026-11-20",
        "feature_count": 42,
        "features": features,
        "manifest_sha256_match": True,
        "training_symbols": ["AAA", "BBB"],
        "probability_calibration_method": "isotonic",
        "serving_mode": "PAPER",
        "serving_reason": "paper until live evidence accumulates",
    }
    status.update(overrides)
    return status


def _fake_champion(status):
    features = list(status.get("features") or [])
    return SimpleNamespace(
        __name__="champion_model_api",
        model_status=lambda: status,
        model_target_contract=lambda: {
            "bar_interval_minutes": 5,
            "target": "target_session_close",
            "prediction_horizon": "session_close",
        },
        active_feature_names=lambda: features,
        _models=lambda: ("clf", "reg"),
    )


@pytest.fixture
def drive_ready(monkeypatch):
    import persistent_store

    monkeypatch.setattr(
        persistent_store,
        "drive_status",
        lambda require_write=False: {
            "configured": True,
            "reachable": True,
            "write_ready": True,
            "write_auth_method": "oauth",
        },
    )


def test_modelz_validates_the_champion_serving_path(monkeypatch, drive_ready):
    import autonomous_intraday_worker_service as worker

    status = _champion_status()
    monkeypatch.setattr(worker, "_serving_model_api", lambda: (_fake_champion(status), "champion"))

    code, payload = worker._model_bundle_diagnostic()

    assert code == 200
    assert payload["status"] == "ok"
    checks = payload["checks"]
    assert checks["model_api"] == "champion"
    assert checks["target"] == "target_session_close"
    assert checks["feature_count_manifest"] == checks["feature_count_runtime"] == 42
    assert checks["feature_list_matches_runtime"] is True
    assert checks["artifact_hashes_match"] is True
    assert checks["training_symbol_count"] == 2
    assert checks["serving_mode"] == "PAPER"


def test_modelz_rejects_champion_feature_drift(monkeypatch, drive_ready):
    import autonomous_intraday_worker_service as worker

    status = _champion_status(
        artifact_feature_count=41, artifact_features=[f"f{i}" for i in range(41)]
    )
    monkeypatch.setattr(worker, "_serving_model_api", lambda: (_fake_champion(status), "champion"))

    code, payload = worker._model_bundle_diagnostic()

    assert code == 503
    assert payload["status"] == "degraded"
    assert payload["checks"]["feature_count_manifest"] == 41


def test_modelz_rejects_champion_with_wrong_target(monkeypatch, drive_ready):
    import autonomous_intraday_worker_service as worker

    status = _champion_status(artifact_target="target_30m")
    monkeypatch.setattr(worker, "_serving_model_api", lambda: (_fake_champion(status), "champion"))

    code, payload = worker._model_bundle_diagnostic()

    assert code == 503
    assert payload["checks"]["target"] == "target_30m"


def test_modelz_blocks_when_drive_is_not_write_ready(monkeypatch):
    import autonomous_intraday_worker_service as worker
    import persistent_store

    monkeypatch.setattr(
        persistent_store,
        "drive_status",
        lambda require_write=False: {"configured": True, "reachable": True, "write_ready": False},
    )
    monkeypatch.setattr(
        worker, "_serving_model_api", lambda: (_fake_champion(_champion_status()), "champion")
    )

    code, payload = worker._model_bundle_diagnostic()

    assert code == 503
    assert payload["checks"]["drive_write_ready"] is False


def test_serving_model_api_honours_legacy_override(monkeypatch):
    import autonomous_intraday_worker_service as worker

    monkeypatch.setenv("ALL_NSE_MODEL_API", "legacy")

    model, api = worker._serving_model_api()

    assert api == "legacy"
    assert model.__name__ == "all_nse_intraday_model"
