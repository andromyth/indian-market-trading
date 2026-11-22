from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

import autonomous_intraday_worker_service as service
import indstocks_backfill_all_nse_5m_history as backfill

IST = "Asia/Kolkata"
ROOT = Path(__file__).resolve().parents[1]


def _frame(closes: list[float], start: str = "2026-11-18 09:20") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq="5min", tz=IST, name="Datetime")
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [value - 1 for value in closes],
            "Close": closes,
            "Volume": [1000] * len(closes),
        },
        index=index,
    )


def test_pinned_universe_is_exactly_the_supplied_five_hundred_symbols():
    payload = json.loads((ROOT / "backfill_universe.json").read_text(encoding="utf-8"))
    symbols = payload["symbols"]
    assert len(symbols) == 500
    assert len(set(symbols)) == 500
    assert symbols[0] == "3BBLACKBIO"
    assert symbols[-1] == "ZUARI"
    assert all(symbol == symbol.strip().upper() for symbol in symbols)
    assert backfill._pinned_universe() == sorted(symbols)


def test_checkpoint_key_is_scoped_to_the_date_definition_schema():
    symbols = ["RELIANCE", "SBIN", "TCS"]
    legacy = hashlib.sha1("|".join(sorted(symbols)).encode()).hexdigest()[:16]
    assert backfill._checkpoint_key(symbols) != legacy
    assert backfill._checkpoint_key(symbols) == backfill._checkpoint_key(reversed(symbols))


def test_replacement_mode_lets_fetched_bars_override_legacy_rows():
    existing = _frame([10.0, 11.0])
    fetched = _frame([20.0, 21.0])

    replaced = backfill.merge_symbol_history(existing, fetched, fetched_wins=True)
    assert replaced["Close"].tolist() == [20.0, 21.0]

    preserved = backfill.merge_symbol_history(existing, fetched, fetched_wins=False)
    assert preserved["Close"].tolist() == [10.0, 11.0]

    only_fetched = backfill.merge_symbol_history(
        pd.DataFrame(columns=backfill.OHLCV), fetched, fetched_wins=True
    )
    assert only_fetched["Close"].tolist() == [20.0, 21.0]
    assert backfill.merge_symbol_history(None, None, fetched_wins=True).empty


def test_backfill_options_parse_end_date_and_replace_flag():
    options, error = service._parse_backfill_options({
        "end_date": "2026-11-18",
        "replace_existing_history": "auto",
    })
    assert error is None
    assert options == {"end_date": "2026-11-18", "replace_existing_history": False}

    options, error = service._parse_backfill_options({
        "end_date": "2026-11-18",
        "replace_existing_history": "yes",
    })
    assert error is None
    assert options["replace_existing_history"] is True

    options, error = service._parse_backfill_options({})
    assert (options, error) == ({}, None)


def test_backfill_options_parse_symbols():
    options, error = service._parse_backfill_options({"symbols": "RELIANCE,NHPC"})
    assert error is None
    assert options["symbols"] == ["NHPC", "RELIANCE"]

    options, error = service._parse_backfill_options({"symbols": ["AAA", "BBB", "AAA"]})
    assert error is None
    assert options["symbols"] == ["AAA", "BBB"]

    options, error = service._parse_backfill_options({"symbols": ""})
    assert (options, error) == ({}, None)


def test_main_uses_pinned_universe_end_date_and_one_time_replacement(monkeypatch, tmp_path):
    import indstocks_client
    import persistent_store

    captured: dict[str, object] = {}
    fetch_calls: list[tuple[object, list[str]]] = []

    monkeypatch.setattr(indstocks_client, "configured", lambda: True)
    monkeypatch.setattr(
        persistent_store,
        "drive_status",
        lambda require_write=False: {"configured": True, "reachable": True, "write_ready": True},
    )

    class FakeStore:
        root_id = "root"
        service = None

        def upload_or_update(self, local, remote, mime_type=None):
            return {"status": "OK", "action": "uploaded", "size": Path(local).stat().st_size}

    monkeypatch.setattr(persistent_store, "GoogleDriveStore", FakeStore)

    import delivery_training_universe
    monkeypatch.setattr(
        delivery_training_universe,
        "official_nse_symbols",
        lambda force_refresh=False: (_ for _ in ()).throw(
            AssertionError("pinned backfill must not consult the NSE roster")
        ),
    )

    monkeypatch.setattr(backfill, "_download_manifest_json", lambda store, root: {})
    monkeypatch.setattr(
        backfill,
        "_download_existing",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("replacement run must not read the legacy archive")
        ),
    )
    monkeypatch.setattr(
        backfill,
        "_load_recent_live_history",
        lambda *args, **kwargs: ({}, {"live_capture_files_read": 0, "live_capture_files_failed": 0}),
    )
    monkeypatch.setattr(
        backfill,
        "_write_manifest",
        lambda store, root, report: captured.update(report),
    )

    session = pd.date_range("2026-11-18 09:20", "2026-11-18 15:30", freq="5min", tz=IST)
    bars = pd.DataFrame(
        {
            "Open": 100.0, "High": 101.0, "Low": 99.0,
            "Close": 100.5, "Volume": 1000,
        },
        index=pd.DatetimeIndex(session, name="Datetime"),
    )

    def fake_fetch(group, start, end, **kwargs):
        fetch_calls.append((start, [symbol for symbol, _ in group]))
        return {symbol: bars.copy() for symbol, _ in group}, 1

    monkeypatch.setattr(backfill, "fetch_indstocks_5m_batch", fake_fetch)
    monkeypatch.setattr(
        backfill, "_instrument_codes",
        lambda: {symbol: f"NSE_{index}" for index, symbol in enumerate(backfill._pinned_universe())},
    )

    previous = dict(backfill.RUNTIME_OPTIONS)
    backfill.RUNTIME_OPTIONS = {"end_date": "2026-11-18", "replace_existing_history": False}
    try:
        exit_code = backfill.main(["--max-symbols", "5", "--work-root", str(tmp_path)])
    finally:
        backfill.RUNTIME_OPTIONS = previous

    assert exit_code == 2  # five symbols cannot satisfy the 200-session floor
    assert captured["universe_source"] == "PINNED_BACKFILL_UNIVERSE"
    assert captured["replace_existing_history"] is True
    assert captured["history_schema"] == backfill.HISTORY_SCHEMA
    assert captured["kaggle_excluded_from_backfill"] is True
    assert captured["official_symbols_requested"] == 5
    assert captured["range_end_ist"].startswith("2026-11-18T15:30")
    expected_start = backfill._align_to_monday(
        pd.Timestamp("2026-11-18 15:30", tz=IST).to_pydatetime()
        - pd.Timedelta(days=1000)
    )
    assert fetch_calls[0][0] == expected_start
    assert fetch_calls[0][1] == backfill._pinned_universe()[:5]
