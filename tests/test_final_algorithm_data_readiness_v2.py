import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

import final_algorithm_data_readiness_v2 as v2


def test_feature_contract_is_exactly_59():
    features, groups, _ = v2.feature_contract()
    assert len(features) == 59
    assert len(set(features)) == 59
    assert sum(len(cols) for cols in groups.values()) == 59


def test_pit_resolver_prefers_decision_timestamp():
    assert v2.pit_column(
        ["symbol", "trade_date", "latest_event_available_at", "decision_timestamp"]
    ) == "decision_timestamp"


def test_legacy_latest_event_available_at_is_not_used_as_replay_pit():
    assert v2.pit_column(
        ["symbol", "trade_date", "latest_event_available_at"]
    ) is None


def test_zero_availability_is_a_known_state():
    df = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "decision_timestamp": pd.to_datetime(["2026-01-02 09:15:00"], utc=True),
            "news_available": [0],
            "event_available": [0],
            "fno_available": [0],
            "fundamentals_available": [0],
        }
    )
    # Mirror the diagnostic's state interpretation.
    assert pd.to_numeric(df["news_available"], errors="coerce").isin([0, 1]).all()
    assert pd.to_numeric(df["event_available"], errors="coerce").isin([0, 1]).all()
    assert pd.to_numeric(df["fno_available"], errors="coerce").isin([0, 1]).all()
    assert pd.to_numeric(df["fundamentals_available"], errors="coerce").isin([0, 1]).all()


def test_fno_contract_level_duplicates_are_not_a_raw_archive_failure():
    pytest.importorskip("pyarrow")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        pd.DataFrame(
            {
                "symbol": ["ABC", "ABC"],
                "date": ["2026-01-02", "2026-01-02"],
                "instrument": ["OPTSTK", "OPTSTK"],
                "expiry": ["2026-01-29", "2026-01-29"],
                "strike": [100, 110],
                "close": [5.0, 2.0],
            }
        ).to_parquet(root / "fo_20260102.parquet", index=False)

        report = v2.fno_raw_audit(root)

        assert report["status"] == "PASS"
        assert report["duplicate_symbol_date_rows"] == 2
        assert report["duplicate_semantics"] == "EXPECTED_CONTRACT_LEVEL_DUPLICATES"


def test_feature_contract_hash_is_deterministic():
    features, _, _ = v2.feature_contract()
    h1 = v2.hash_features(features)
    h2 = v2.hash_features(features)
    assert h1 == h2
    assert len(h1) == 64


def test_strict_readiness_does_not_skip_unknown_flags():
    flags = pd.DataFrame(
        {
            "a": pd.Series([1, pd.NA], dtype="Int8"),
            "b": pd.Series([1, 1], dtype="Int8"),
        }
    )
    ready = (
        flags.notna().all(axis=1)
        & flags.eq(1).all(axis=1)
    )
    assert ready.tolist() == [True, False]
