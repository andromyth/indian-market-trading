from __future__ import annotations

import pandas as pd

import audit_retraining_inputs as audit


def _write_equity(root, symbol, sessions):
    path = (
        root / "production_data/nse_official/intraday_seed/olo_5m_archive"
        / "symbols" / f"symbol={symbol}" / f"{symbol}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    index = pd.DatetimeIndex(
        [
            stamp
            for day in sessions
            for stamp in pd.date_range(f"{day} 09:20", f"{day} 15:30", freq="5min", tz=audit.IST)
        ],
        name="Datetime",
    )
    pd.DataFrame(
        {"Open": 1.0, "High": 1.1, "Low": 0.9, "Close": 1.0, "Volume": 10.0},
        index=index,
    ).to_parquet(path)


def test_missing_runs_groups_consecutive_sessions():
    days = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    runs = audit._missing_runs(days, {"2026-01-02", "2026-01-06"})
    assert runs == [{"start": "2026-01-05", "end": "2026-01-05", "sessions": 1},
                    {"start": "2026-01-07", "end": "2026-01-07", "sessions": 1}]


def test_largest_ready_window_respects_thresholds():
    days = ["2026-01-02", "2026-01-05", "2026-01-06"]
    largest = audit._largest_ready_window(days, {"2026-01-05", "2026-01-06"}, {"2026-01-02", "2026-01-05", "2026-01-06"})
    assert largest["sessions"] == 2
    assert largest["start"] == "2026-01-05"


def test_build_report_detects_nifty_gap(tmp_path):
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]
    _write_equity(tmp_path, "AAA", sessions)
    _write_equity(tmp_path, "BBB", sessions)

    fno = pd.DataFrame(
        {
            "date": pd.to_datetime(sessions),
            "symbol": ["AAA"] * len(sessions),
            "fno_pcr": [1.0] * len(sessions),
        }
    )
    fno_path = tmp_path / audit.FNO_DEFAULT
    fno_path.parent.mkdir(parents=True, exist_ok=True)
    fno.to_parquet(fno_path, index=False)

    nifty_path = tmp_path / audit.CANONICAL_NIFTY_REMOTE / audit.KAGGLE_NAME
    nifty_path.parent.mkdir(parents=True, exist_ok=True)
    nifty_index = pd.DatetimeIndex(
        [
            stamp
            for day in sessions[1:]
            for stamp in pd.date_range(f"{day} 09:20", f"{day} 15:30", freq="5min", tz=audit.IST)
        ]
    )
    pd.DataFrame({"Timestamp": nifty_index, "Close": 25000.0}).to_parquet(nifty_path, index=False)

    report = audit.build_report(tmp_path, max_equity_symbols=2)

    assert report["equity"]["sessions"] == 3
    assert report["fno"]["dates"] == 3
    assert report["nifty"]["sessions"] == 2
    assert report["nifty"]["missing_run_count"] == 1
    assert report["largest_ready_window"]["sessions"] == 2
    assert report["verdict"] == "INSUFFICIENT_COVERED_SESSIONS"
