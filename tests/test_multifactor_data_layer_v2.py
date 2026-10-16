import numpy as np
import pandas as pd
import pytest

from multifactor_data_layer_v2 import (
    FEATURES_59,
    assert_training_feature_coverage,
    attach_readiness_flags,
    build_feature_matrix,
    feature_coverage_report,
    merge_pit_layer,
    readiness_report,
    session_close_timestamp_utc,
)


def test_contract():
    assert len(FEATURES_59) == 59
    assert len(set(FEATURES_59)) == 59


def test_zero_is_known_state():
    df = pd.DataFrame({
        "symbol": ["ABC"],
        "decision_timestamp": pd.to_datetime(["2026-01-02"], utc=True),
        "ret_1": [0.01],
        "market_ret_1": [0.005],
        "news_available": [0],
        "event_available": [1],
        "fno_available": [0],
        "fundamental_eps": [5.0],
    })
    x = attach_readiness_flags(df)
    assert int(x.loc[0, "ready_news"]) == 0
    assert pd.notna(x.loc[0, "ready_news"])
    report = readiness_report(df)
    assert 0 <= report["mean_group_readiness_pct"] <= 100
    assert "feature_coverage" in report


def test_close_timestamp_is_1530_ist_not_1530_utc():
    ts = session_close_timestamp_utc(pd.Series(["2026-10-08", "2026-10-09"]))
    assert str(ts.iloc[0]) == "2026-10-08 10:00:00+00:00"
    assert str(ts.iloc[1]) == "2026-10-09 10:00:00+00:00"


def test_feature_matrix_preserves_exact_59_order_and_missing_values():
    data = {name: [1.0] for name in FEATURES_59}
    data["fno_iv_change"] = [np.nan]
    frame = pd.DataFrame(data)
    frame = frame[list(reversed(FEATURES_59))]
    matrix = build_feature_matrix(frame)
    assert list(matrix.columns) == FEATURES_59
    assert matrix.shape == (1, 59)
    assert pd.isna(matrix.loc[0, "fno_iv_change"])
    assert matrix.loc[0, "ret_1"] == 1.0


def test_feature_matrix_rejects_missing_contract_column():
    frame = pd.DataFrame({name: [1.0] for name in FEATURES_59 if name != "fno_iv_change"})
    with pytest.raises(ValueError, match="fno_iv_change"):
        build_feature_matrix(frame)


def test_coverage_report_detects_zero_observation_features():
    frame = pd.DataFrame({"symbol": ["ABC", "XYZ"], "ret_1": [0.01, 0.02]})
    report = feature_coverage_report(frame)
    assert "ret_1" not in report["features_with_zero_observations"]
    assert "fno_iv_change" in report["features_with_zero_observations"]
    assert "news_available" in report["features_missing_as_columns"]


def test_training_gate_blocks_silent_empty_or_rare_features():
    rows = 250
    frame = pd.DataFrame({name: np.arange(rows, dtype=float) for name in FEATURES_59})
    frame["fno_available"] = 1.0
    frame["fno_iv_change"] = np.nan
    with pytest.raises(RuntimeError, match="fno_iv_change"):
        assert_training_feature_coverage(frame)

    frame["fno_iv_change"] = np.arange(rows, dtype=float)
    frame.loc[1:, "fundamental_eps"] = np.nan
    with pytest.raises(RuntimeError, match="fundamental_eps"):
        assert_training_feature_coverage(frame)


def test_training_gate_counts_fno_features_only_when_fno_is_applicable():
    rows = 400
    frame = pd.DataFrame({name: np.arange(rows, dtype=float) for name in FEATURES_59})
    # F&O is relevant for only 120 rows. The feature is complete within that
    # applicable subset; the other 280 NaNs must not be misclassified as a
    # broken F&O feed.
    frame["fno_available"] = 0.0
    frame.loc[:119, "fno_available"] = 1.0
    frame["fno_iv_change"] = np.nan
    frame.loc[:119, "fno_iv_change"] = np.linspace(0.01, 0.12, 120)
    frame["ready_fundamentals"] = 1
    report = assert_training_feature_coverage(
        frame, min_observations=100, min_fraction=0.005
    )
    iv = report["coverage_scope_by_feature"]["fno_iv_change"]
    assert iv["scope"] == "fno_applicable_rows"
    assert iv["scope_rows"] == 120
    assert iv["observed_rows_in_scope"] == 120

def test_pit_merge_expires_old_observations_instead_of_forward_filling_forever():
    base = pd.DataFrame({
        "symbol": ["ABC", "ABC", "ABC"],
        "decision_timestamp": pd.to_datetime([
            "2026-01-01T10:00:00Z",
            "2026-01-05T10:00:00Z",
            "2026-01-12T10:00:00Z",
        ], utc=True),
    })
    layer = pd.DataFrame({
        "symbol": ["ABC"],
        "decision_timestamp": pd.to_datetime(["2026-01-01T10:00:00Z"], utc=True),
        "news_available": [1],
    })
    merged = merge_pit_layer(base, layer, ["news_available"], "news", max_age="7D")
    assert merged["news_available"].iloc[0] == 1
    assert merged["news_available"].iloc[1] == 1
    assert pd.isna(merged["news_available"].iloc[2])

def test_pit_merge_normalizes_microsecond_and_nanosecond_utc_keys():
    # Parquet can round-trip different timestamp resolutions from different
    # source files; the merge must normalize them before merge_asof.
    base_ts = pd.to_datetime(["2026-01-02T10:00:00Z"], utc=True).astype(
        "datetime64[ns, UTC]"
    )
    layer_ts = pd.to_datetime(["2026-01-01T10:00:00Z"], utc=True).astype(
        "datetime64[us, UTC]"
    )
    base = pd.DataFrame({"symbol": ["ABC"], "decision_timestamp": base_ts})
    layer = pd.DataFrame({
        "symbol": ["ABC"],
        "decision_timestamp": layer_ts,
        "news_available": [1],
    })
    merged = merge_pit_layer(base, layer, ["news_available"], "news")
    assert int(merged.loc[0, "news_available"]) == 1
    assert str(merged.loc[0, "decision_timestamp"].dtype) == "datetime64[ns, UTC]"

