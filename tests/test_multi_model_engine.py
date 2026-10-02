import pandas as pd

from multi_model_engine import (
    FullMultiModelRegistry,
    ModelRecord,
    classify_horizon,
    classify_role,
    classify_scope,
)


def test_classification():
    assert classify_horizon(
        "deployment_30m_logistic_classifier_2026.joblib"
    ) == "30m"

    assert classify_horizon(
        "deployment_next_session_logistic_classifier_2026.joblib"
    ) == "next_session"

    assert classify_horizon(
        "fno_only_classifier_7d_2026.joblib"
    ) == "7d"

    assert classify_scope(
        "GLOBAL_next_session_classifier.joblib"
    ) == "GLOBAL"

    assert classify_scope(
        "RELIANCE_next_session_classifier.joblib"
    ) == "SYMBOL:RELIANCE"

    assert classify_scope(
        "fno_only_classifier_1d_2026.joblib"
    ) == "FNO_ONLY"

    assert classify_role(
        "deployment_60m_hgb_return_regressor_2026.joblib"
    ) == "regressor"


def _record(scope="GLOBAL", horizon="next_session"):
    return ModelRecord(
        name="test.joblib",
        drive_id="test-id",
        relative_path="test.joblib",
        role="classifier",
        scope=scope,
        horizon=horizon,
        is_classifier=True,
        is_regressor=False,
        feature_count=2,
        feature_names=("a", "b"),
        metadata_status="PASS",
    )


def test_global_scope_is_universal():
    ok, reason = FullMultiModelRegistry.eligible(
        _record(),
        "IDEA",
        "next_session",
        pd.DataFrame([{"a": 1.0, "b": 2.0}]),
        {},
    )
    assert ok is True
    assert reason == "eligible_for_input_validation"


def test_symbol_scope_is_restricted():
    ok, _ = FullMultiModelRegistry.eligible(
        _record("SYMBOL:RELIANCE"),
        "SBIN",
        "next_session",
        pd.DataFrame([{"a": 1.0, "b": 2.0}]),
        {},
    )
    assert ok is False


def test_fno_requires_real_data():
    ok, reason = FullMultiModelRegistry.eligible(
        _record("FNO_ONLY"),
        "SBIN",
        "1d",
        pd.DataFrame([{"a": 1.0, "b": 2.0}]),
        {"fno": False},
    )
    assert ok is False
    assert reason == "fno_data_unavailable"
