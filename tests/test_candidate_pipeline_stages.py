import json

import numpy as np
import pandas as pd

import colab_build_all_nse_delivery_multifactor_candidate as candidate
from multifactor_data_layer_v2 import FEATURES_59


def test_universe_roster_uses_duckdb_counts_and_official_universe(tmp_path, monkeypatch):
    master = tmp_path / "cash_master.parquet"
    rows = []
    for symbol, dates in {
        "ABC": ["2026-01-01", "2026-01-02", "2026-01-05"],
        "XYZ": ["2026-01-01", "2026-01-02"],
        "NOP": ["2026-01-01"],
    }.items():
        for day in dates:
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp(day),
                "series": "EQ",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
            })
    # Duplicate listing row on an existing day must not inflate session count.
    rows.append({**rows[0]})
    pd.DataFrame(rows).to_parquet(master, index=False)
    monkeypatch.setattr(
        candidate, "official_nse_symbols",
        lambda force_refresh=False: ["ABC", "XYZ", "NOP", "NEVER_TRADED"],
    )

    roster = candidate._build_roster_from_canonical_master(master, min_history=2)

    assert roster["trainable_symbols"] == ["ABC", "XYZ"]
    assert roster["trainable_symbol_count"] == 2
    assert roster["symbols_with_bulk_history"] == 3
    assert roster["excluded_symbol_count"] == 2
    assert roster["history_sessions_by_symbol"]["ABC"] == 3
    assert "DuckDB" in roster["roster_method"]


def test_training_sample_labels_before_sampling_and_preserves_holdout(tmp_path):
    panel_path = tmp_path / "panel.parquet"
    sample_path = tmp_path / "samples" / "next_session.parquet"
    dates = pd.bdate_range("2024-01-01", periods=800)
    rows = []
    for symbol_index, symbol in enumerate(("ABC", "XYZ")):
        for i, day in enumerate(dates):
            record = {
                "symbol": symbol,
                "date": day,
                "decision_timestamp": day.tz_localize("Asia/Kolkata")
                    .replace(hour=15, minute=30).tz_convert("UTC"),
                "open": 100.0 + i * 0.1 + symbol_index,
                "high": 101.0 + i * 0.1 + symbol_index,
                "low": 99.0 + i * 0.1 + symbol_index,
                "close": 100.0 + i * 0.1 + symbol_index,
                "data_readiness_score": 0.75,
                "ready_market_price": 1,
                "ready_market_context": 1,
                "ready_news": 0,
                "ready_events": 0,
                "ready_fno": 0,
                "ready_fundamentals": 1,
                "news_available": 0,
                "event_available": 0,
                "event_high_impact": 0,
                "fno_available": 0,
            }
            for feature_index, feature in enumerate(FEATURES_59):
                record.setdefault(feature, float(i + feature_index + symbol_index))
            for feature in (
                "fno_pcr", "fno_oi_change", "fno_iv_change", "fno_fut_close",
                "fno_fut_basis", "fno_fut_return", "fno_fut_oi",
            ):
                record[feature] = np.nan
            rows.append(record)
    pd.DataFrame(rows).to_parquet(panel_path, index=False)

    stats = candidate._prepare_training_sample(
        panel_path, sample_path, "next_session", steps=1, max_training_rows=1000
    )
    sample = pd.read_parquet(sample_path)

    assert sample_path.exists()
    assert set(FEATURES_59).issubset(sample.columns)
    assert "target_return_next_session" in sample.columns
    assert "target_up_next_session" in sample.columns
    assert stats["full_60_day_holdout"] > 0
    assert stats["deterministic_history_sample"] > 0
    assert stats["sample_rows_total"] < stats["labelled_population_rows"]
    assert stats["sampling_policy"].startswith("all 60-day holdout rows")
