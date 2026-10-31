import json

import numpy as np
import pandas as pd

import colab_build_all_nse_delivery_multifactor_candidate as candidate


def test_fno_join_uses_history_and_current_archive_without_forward_fill(tmp_path, monkeypatch):
    base = tmp_path / "market"
    monkeypatch.setattr(candidate, "BASE", base)
    fno_root = base / "production_data/nse_official/fno"
    fno_root.mkdir(parents=True)
    current_path = fno_root / "fno_daily_features_2026.parquet"
    history_path = fno_root / "fno_daily_features_history_2016_2026.parquet"
    columns = {
        "fno_oi_change": [12.0],
        "fno_iv_change": [0.15],
        "fno_available": [1],
        "fno_fut_close": [101.0],
        "fno_fut_basis": [0.01],
        "fno_fut_return": [0.02],
        "fno_fut_oi": [25000.0],
    }
    pd.DataFrame({
        "symbol": ["ABC"],
        "date": pd.to_datetime(["2026-01-01"]),
        "fno_pcr": [1.25],
        **columns,
    }).to_parquet(current_path, index=False)
    pd.DataFrame({
        "symbol": ["ABC", "ABC"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "fno_pcr": [0.90, 1.50],
        "fno_oi_change": [8.0, 16.0],
        "fno_iv_change": [0.10, 0.20],
        "fno_available": [1, 1],
        "fno_fut_close": [99.0, 102.0],
        "fno_fut_basis": [0.01, 0.02],
        "fno_fut_return": [0.01, 0.03],
        "fno_fut_oi": [20000.0, 26000.0],
    }).to_parquet(history_path, index=False)

    rows = []
    for symbol in ("ABC", "XYZ"):
        for date in ("2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"):
            rows.append({
                "symbol": symbol,
                "date": pd.Timestamp(date),
                "decision_timestamp": pd.Timestamp(date, tz="Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30),
                "close": 100.0,
            })
    panel = pd.DataFrame(rows)
    panel["decision_timestamp"] = panel["decision_timestamp"].dt.tz_convert("UTC")

    fno = candidate._load_fno_source_frame()
    merged, audit = candidate._merge_broad_fno_pit(panel, fno)
    abc_jan2 = merged[(merged.symbol == "ABC") & (merged.date == pd.Timestamp("2026-01-02"))].iloc[0]
    xyz_jan2 = merged[(merged.symbol == "XYZ") & (merged.date == pd.Timestamp("2026-01-02"))].iloc[0]
    abc_jan5 = merged[(merged.symbol == "ABC") & (merged.date == pd.Timestamp("2026-01-05"))].iloc[0]
    abc_jan6 = merged[(merged.symbol == "ABC") & (merged.date == pd.Timestamp("2026-01-06"))].iloc[0]

    assert abc_jan2.fno_pcr == 1.25  # current archive wins the duplicate history row
    assert int(xyz_jan2.fno_available) == 0  # archive exists, but this symbol has no F&O contract
    assert abc_jan5.fno_pcr == 1.50  # historical archive is joined to the next exchange session
    assert pd.isna(abc_jan6.fno_available)  # no 5 Jan source archive; no forward-fill
    assert pd.isna(abc_jan6.fno_pcr)
    assert audit["covered_target_sessions"] == 2
    assert audit["source_paths"]["history"]["source_dates"] == 2
    assert audit["source_paths"]["current"]["source_dates"] == 1


def test_fundamental_availability_index_uses_full_catalog_independently_of_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, "FUND_AVAILABILITY", tmp_path / "availability.parquet")
    monkeypatch.setattr(candidate, "FUND_AVAILABILITY_META", tmp_path / "availability.meta.json")
    catalog_cache = tmp_path / "catalog.json"
    catalog_cache.write_text(json.dumps({
        "meta": {
            "coverage_start": "2025-01-01",
            "coverage_end": "2026-10-10",
            "coverage_complete": True,
        },
        "data": [],
    }))
    monkeypatch.setattr(candidate, "FUND_CATALOG_CACHE", catalog_cache)
    catalog = [
        {
            "type": "Integrated Filing- Financials",
            "symbol": "ABC",
            "xbrl": "https://example.test/abc-q1.xbrl",
            "qe_Date": "31-03-2026",
            "broadcast_Date": "10-05-2026",
            "consolidated": "Consolidated",
        },
        {
            "type": "Integrated Filing- Financials",
            "symbol": "ABC",
            "xbrl": "https://example.test/abc-q2.xbrl",
            "qe_Date": "30-06-2026",
            "broadcast_Date": "10-08-2026",
            "consolidated": "Consolidated",
        },
        {
            "type": "Integrated Filing- Financials",
            "symbol": "XYZ",
            "xbrl": "https://example.test/xyz-q1.xbrl",
            "qe_Date": "31-03-2026",
            "broadcast_Date": "15-05-2026",
            "consolidated": "Standalone",
        },
    ]
    report = candidate._build_fundamental_availability(catalog, ["ABC", "XYZ", "NOP"])
    index = pd.read_parquet(candidate.FUND_AVAILABILITY)

    assert report["coverage_complete"] is True
    assert report["target_symbol_count"] == 3
    assert set(index.symbol) == {"ABC", "XYZ"}
    assert len(index[index.symbol.eq("ABC")]) == 2
    assert "NOP" not in set(index.symbol)  # complete catalogue makes absence meaningful
