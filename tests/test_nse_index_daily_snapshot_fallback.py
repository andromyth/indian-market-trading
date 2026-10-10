from datetime import date

import pandas as pd

import nse_data_layer as layer


def _csv(day: str) -> bytes:
    return (
        "Index Name,Index Date,Open Index Value,High Index Value,Low Index Value,"
        "Closing Index Value,Points Change,Change(%),Volume,Turnover (Rs. Cr.),P/E,P/B,Div Yield\n"
        f"NIFTY 50,{day},22000,22100,21900,22050,50,0.2,123,456,20,3,1.2\n"
        f"NIFTY BANK,{day},48000,48200,47800,48100,100,0.2,456,789,15,2,0.8\n"
    ).encode("utf-8")


def test_parse_official_daily_snapshot_index_rows():
    frame = layer._parse_index_daily_snapshot(
        _csv("13-May-2026"),
        "^NSEI",
        date(2026, 5, 13),
    )

    assert len(frame) == 1
    assert frame.loc[0, "symbol"] == "^NSEI"
    assert float(frame.loc[0, "close"]) == 22050.0
    assert frame.loc[0, "source"] == "NSE_OFFICIAL_DAILY_SNAPSHOT"
    assert "ind_close_all_13052026.csv" in frame.loc[0, "provenance"]


def test_official_daily_snapshot_range_is_cached_and_coverage_checked(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "daily_snapshots"
    root.mkdir()
    for day, label in (
        (date(2026, 5, 13), "13-May-2026"),
        (date(2026, 5, 14), "14-May-2026"),
        (date(2026, 5, 15), "15-May-2026"),
    ):
        (root / f"ind_close_all_{day:%d%m%Y}.csv").write_bytes(_csv(label))

    monkeypatch.setattr(layer, "INDEX_DAILY_SNAPSHOT_ROOT", root)
    monkeypatch.setattr(layer, "INDEX_NORMALIZED_ROOT", tmp_path / "normalized")
    monkeypatch.setattr(layer, "_drive_upload", lambda *args, **kwargs: {"status": "OK"})

    frame = layer._fetch_index_history_from_daily_snapshots(
        "^NSEBANK",
        date(2026, 5, 13),
        date(2026, 5, 15),
        sync_drive=False,
    )

    assert len(frame) == 3
    assert set(frame["symbol"]) == {"^NSEBANK"}
    assert pd.to_datetime(frame["date"]).min().date() == date(2026, 5, 13)
    assert pd.to_datetime(frame["date"]).max().date() == date(2026, 5, 15)
    assert set(frame["source"]) == {"NSE_OFFICIAL_DAILY_SNAPSHOT"}


def test_fetch_index_history_uses_official_snapshot_after_api_json_failure(
    monkeypatch,
):
    dates = pd.to_datetime(["2026-05-13", "2026-05-14", "2026-05-15"])
    snapshot = pd.DataFrame({
        "symbol": ["^NSEI"] * 3,
        "date": dates,
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [0.0] * 3,
        "source": ["NSE_OFFICIAL_DAILY_SNAPSHOT"] * 3,
        "provenance": ["official archive"] * 3,
    })
    monkeypatch.setattr(layer, "_drive_index_history", lambda *args: pd.DataFrame())
    monkeypatch.setattr(
        layer,
        "_fetch_index_history_nse_api",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("NSE API returned non-JSON")
        ),
    )
    monkeypatch.setattr(
        layer,
        "_fetch_index_history_from_daily_snapshots",
        lambda *args, **kwargs: snapshot,
    )

    result = layer.fetch_index_history(
        "^NSEI",
        date(2026, 5, 13),
        date(2026, 5, 15),
        sync_drive=False,
    )

    assert len(result) == 3
    assert result["source"].eq("NSE_OFFICIAL_DAILY_SNAPSHOT").all()
