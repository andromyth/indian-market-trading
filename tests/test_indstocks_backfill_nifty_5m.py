from __future__ import annotations

import json

import pandas as pd
import pytest

import indstocks_backfill_nifty_5m as nifty


def _open_stamped_session(day: str = "2026-05-20") -> pd.DataFrame:
    stamps = pd.date_range(f"{day} 09:15", f"{day} 15:25", freq="5min", tz=nifty.IST)
    index = pd.DatetimeIndex(stamps, name="Datetime")
    return pd.DataFrame(
        {
            "Open": 25000.0,
            "High": 25010.0,
            "Low": 24990.0,
            "Close": 25005.0,
            "Volume": 0.0,
        },
        index=index,
    )


def test_close_series_accepts_timestamp_column(tmp_path):
    path = tmp_path / "kaggle.parquet"
    pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2026-05-19 15:30:00+05:30"]),
            "Close": [25000.0],
        }
    ).to_parquet(path, index=False)

    series = nifty._close_series(pd.read_parquet(path))

    assert len(series) == 1
    assert float(series.iloc[0]) == 25000.0
    assert str(series.index.tz) == "Asia/Kolkata"


def test_session_bars_counts_full_and_partial_sessions():
    index = pd.DatetimeIndex(
        list(pd.date_range("2026-05-20 09:20", "2026-05-20 15:30", freq="5min", tz=nifty.IST))
        + list(pd.date_range("2026-05-21 09:20", "2026-05-21 10:00", freq="5min", tz=nifty.IST))
    )
    counts = nifty._session_bars(index)

    assert counts["2026-05-20"] == 75
    assert counts["2026-05-21"] == 9


def test_main_refuses_provider_access_outside_cloud_run(tmp_path, monkeypatch):
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("CLOUD_RUN_JOB", raising=False)
    monkeypatch.setattr(
        nifty,
        "resolve_index_scrip_code",
        lambda *_args, **_kwargs: pytest.fail("must not resolve or fetch outside Cloud Run"),
    )
    monkeypatch.setattr(
        nifty,
        "historical",
        lambda *_args, **_kwargs: pytest.fail("must not fetch outside Cloud Run"),
    )

    rc = nifty.main(
        [
            "--kaggle-parquet", str(tmp_path / nifty.KAGGLE_NAME),
            "--local-dir", str(tmp_path / "out"),
            "--no-upload",
        ]
    )

    assert rc == 2


def test_main_merges_new_five_minute_bars_with_kaggle_close(tmp_path, monkeypatch):
    monkeypatch.setenv("K_SERVICE", "indian-market-intraday-worker")
    kaggle = tmp_path / nifty.KAGGLE_NAME
    kaggle_index = pd.date_range(
        "2026-05-19 09:20", "2026-05-19 15:30", freq="5min", tz=nifty.IST
    )
    pd.DataFrame(
        {
            "Timestamp": kaggle_index,
            "Close": [25000.0] * len(kaggle_index),
        }
    ).to_parquet(kaggle, index=False)

    monkeypatch.setattr(
        nifty,
        "resolve_index_scrip_code",
        lambda name, exchange: "NSE_40000001",
    )
    monkeypatch.setattr(
        nifty,
        "historical",
        lambda code, start, end, interval, max_calls: _open_stamped_session(),
    )

    rc = nifty.main(
        [
            "--kaggle-parquet", str(kaggle),
            "--local-dir", str(tmp_path / "out"),
            "--start", "2026-05-20",
            "--end", "2026-05-20",
            "--no-upload",
        ]
    )

    assert rc == 0
    output = pd.read_parquet(tmp_path / "out" / nifty.MERGED_NAME)
    assert len(output) == 150
    assert output["Close"].notna().all()
    manifest = json.loads((tmp_path / "out" / nifty.MANIFEST_NAME).read_text())
    assert manifest["output_rows"] == 150
    assert manifest["sessions"] == 2
    assert manifest["incomplete_sessions"] == {}
