from __future__ import annotations

import pandas as pd

from drive_all_nse_prediction_compiler import _read_history
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def test_read_history_normalizes_timestamps_and_keeps_last_duplicate(tmp_path):
    path = tmp_path / "history.parquet"
    pd.DataFrame({
        "Timestamp": [
            "2026-11-09 09:20:00",
            "2026-11-09 09:20:00",
            "2026-11-09 09:25:00",
        ],
        "Open": [100, 101, 102],
        "High": [101, 102, 103],
        "Low": [99, 100, 101],
        "Close": [100.5, 101.5, 102.5],
        "Volume": [1000, 1100, 1200],
    }).to_parquet(path, index=False)

    result = _read_history(path)

    assert str(result.index.tz) == "Asia/Kolkata"
    assert len(result) == 2
    assert result.iloc[0]["Close"] == 101.5
    assert result.index[-1] == pd.Timestamp("2026-11-09 09:25:00", tz=IST)


def test_read_history_filters_to_requested_symbol(tmp_path):
    path = tmp_path / "live.parquet"
    pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "Timestamp": ["2026-11-09 09:20:00", "2026-11-09 09:20:00"],
        "Open": [100, 200],
        "High": [101, 201],
        "Low": [99, 199],
        "Close": [100.5, 200.5],
        "Volume": [1000, 2000],
    }).to_parquet(path, index=False)

    result = _read_history(path, "BBB")

    assert len(result) == 1
    assert result.iloc[0]["Close"] == 200.5



def test_latest_capture_skips_blocked_pointer_and_uses_fresh_stored_minute_manifest(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import drive_all_nse_prediction_compiler as compiler

    now = pd.Timestamp("2026-11-09 14:45:30", tz=IST)
    snapshot_remote = "production_data/all_universe_minute_snapshots/2026/11/09/snapshot_20261109_1440.parquet"
    blocked = {
        "cycle_minute_ist": "2026-11-09T14:45:00+05:30",
        "status": "BLOCKED_PROVIDER_CREDENTIALS",
        "snapshot_persisted": False,
    }
    stored = {
        "cycle_minute_ist": "2026-11-09T14:40:00+05:30",
        "status": "STORED",
        "snapshot_persisted": True,
        "snapshot_drive_path": snapshot_remote,
        "row_count": 1,
    }
    items = [
        {"id": "blocked", "name": "manifest_20261109_1445.json"},
        {"id": "stored", "name": "manifest_20261109_1440.json"},
    ]

    def find_remote(store, remote):
        if remote.endswith("/2026/11/09"):
            return {"id": "day"}
        if remote.endswith("/latest_manifest.json"):
            return {"id": "latest", "name": "latest_manifest.json"}
        if remote == snapshot_remote:
            return {"id": "snapshot"}
        return None

    def download(store, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "blocked": blocked,
            "stored": stored,
            "latest": blocked,
        }.get(file_id)
        if payload is not None:
            destination.write_text(json.dumps(payload), encoding="utf-8")
        else:
            destination.write_bytes(b"parquet-placeholder")

    monkeypatch.setattr(compiler, "_find_remote", find_remote)
    monkeypatch.setattr(compiler, "_list_children", lambda store, parent_id: items)
    monkeypatch.setattr(compiler, "_download", download)
    monkeypatch.setattr(
        compiler.pd,
        "read_parquet",
        lambda path: pd.DataFrame({
            "symbol": ["RELIANCE"],
            "instrument_code": ["NSE_EQ|123"],
            "cycle_minute_ist": ["2026-11-09T14:40:00+05:30"],
            "quote_status": ["PASS"],
            "actual_observed_price": [1500.0],
            "quote_observed_at_ist": ["2026-11-09T14:40:05+05:30"],
            "quote_error": [None],
        }),
    )

    manifest, quotes, minute = compiler._latest_capture(SimpleNamespace(), now=now)

    assert manifest["status"] == "STORED"
    assert minute == pd.Timestamp("2026-11-09 14:40:00", tz=IST)
    assert quotes["symbol"].tolist() == ["RELIANCE"]


def test_latest_capture_rejects_stale_stored_manifest(monkeypatch):
    import json
    from types import SimpleNamespace

    import drive_all_nse_prediction_compiler as compiler

    now = pd.Timestamp("2026-11-09 14:45:30", tz=IST)
    stale = {
        "cycle_minute_ist": "2026-11-09T14:30:00+05:30",
        "status": "STORED",
        "snapshot_persisted": True,
        "snapshot_drive_path": "production_data/all_universe_minute_snapshots/2026/11/09/snapshot_20261109_1430.parquet",
        "row_count": 1,
    }
    blocked = {
        "cycle_minute_ist": "2026-11-09T14:45:00+05:30",
        "status": "BLOCKED_PROVIDER_CREDENTIALS",
        "snapshot_persisted": False,
    }

    def find_remote(store, remote):
        if remote.endswith("/2026/11/09"):
            return {"id": "day"}
        if remote.endswith("/latest_manifest.json"):
            return {"id": "latest", "name": "latest_manifest.json"}
        return None

    def download(store, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = stale if file_id == "stale" else blocked
        destination.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(compiler, "_find_remote", find_remote)
    monkeypatch.setattr(
        compiler,
        "_list_children",
        lambda store, parent_id: [{"id": "stale", "name": "manifest_20261109_1430.json"}],
    )
    monkeypatch.setattr(compiler, "_download", download)

    import pytest

    with pytest.raises(RuntimeError, match="No fresh, durably stored"):
        compiler._latest_capture(SimpleNamespace(), now=now)


def test_latest_capture_skips_persisted_but_ineligible_snapshot_and_uses_prior_valid_one(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import drive_all_nse_prediction_compiler as compiler

    now = pd.Timestamp("2026-11-09 14:45:30", tz=IST)
    newer_remote = "production_data/all_universe_minute_snapshots/2026/11/09/snapshot_20261109_1445.parquet"
    older_remote = "production_data/all_universe_minute_snapshots/2026/11/09/snapshot_20261109_1440.parquet"
    manifests = {
        "newer": {
            "cycle_minute_ist": "2026-11-09T14:45:00+05:30",
            "status": "STORED",
            "snapshot_persisted": True,
            "snapshot_drive_path": newer_remote,
            "row_count": 1,
        },
        "older": {
            "cycle_minute_ist": "2026-11-09T14:40:00+05:30",
            "status": "STORED",
            "snapshot_persisted": True,
            "snapshot_drive_path": older_remote,
            "row_count": 1,
        },
    }
    items = [
        {"id": "newer", "name": "manifest_20261109_1445.json"},
        {"id": "older", "name": "manifest_20261109_1440.json"},
    ]

    def find_remote(store, remote):
        if remote.endswith("/2026/11/09"):
            return {"id": "day"}
        if remote.endswith("/latest_manifest.json"):
            return {"id": "latest", "name": "latest_manifest.json"}
        if remote == newer_remote:
            return {"id": "snapshot_newer"}
        if remote == older_remote:
            return {"id": "snapshot_older"}
        return None

    def download(store, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if file_id in manifests:
            destination.write_text(json.dumps(manifests[file_id]), encoding="utf-8")
        elif file_id == "latest":
            destination.write_text(json.dumps(manifests["newer"]), encoding="utf-8")
        else:
            destination.write_bytes(b"parquet-placeholder")

    def read_parquet(path):
        if "1445" in str(path):
            return pd.DataFrame({
                "symbol": ["RELIANCE"],
                "instrument_code": ["NSE_EQ|123"],
                "cycle_minute_ist": ["2026-11-09T14:45:00+05:30"],
                "quote_status": ["TOKEN_REJECTED"],
                "actual_observed_price": [None],
                "quote_observed_at_ist": [None],
                "quote_error": ["provider token rejected"],
            })
        return pd.DataFrame({
            "symbol": ["RELIANCE"],
            "instrument_code": ["NSE_EQ|123"],
            "cycle_minute_ist": ["2026-11-09T14:40:00+05:30"],
            "quote_status": ["PASS"],
            "actual_observed_price": [1500.0],
            "quote_observed_at_ist": ["2026-11-09T14:40:05+05:30"],
            "quote_error": [None],
        })

    monkeypatch.setattr(compiler, "_find_remote", find_remote)
    monkeypatch.setattr(compiler, "_list_children", lambda store, parent_id: items)
    monkeypatch.setattr(compiler, "_download", download)
    monkeypatch.setattr(compiler.pd, "read_parquet", read_parquet)

    manifest, quotes, minute = compiler._latest_capture(SimpleNamespace(), now=now)

    assert minute == pd.Timestamp("2026-11-09 14:40:00", tz=IST)
    assert manifest["status"] == "STORED"
    assert quotes.iloc[0]["quote_status"] == "PASS"


def test_load_drive_history_retains_persisted_nifty_market_context(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    import drive_all_nse_prediction_compiler as compiler

    live_root = "production_data/nse_official/intraday_5m_live/2026/11/11"
    source = tmp_path / "source_capture.parquet"
    stamp = pd.Timestamp("2026-11-11 12:40:00", tz=IST)
    pd.DataFrame({
        "symbol": ["RELIANCE", "__NIFTY50__"],
        "Timestamp": [stamp, stamp],
        "Open": [100.0, 24000.0],
        "High": [102.0, 24020.0],
        "Low": [99.0, 23990.0],
        "Close": [101.0, 24010.0],
        "Volume": [1000.0, 50000.0],
    }).to_parquet(source, index=False)

    monkeypatch.setattr(
        compiler, "_find_remote",
        lambda store, remote: {"id": "live"} if remote == live_root else None,
    )
    monkeypatch.setattr(
        compiler, "_list_children",
        lambda store, parent: [{"id": "capture", "name": "capture_20261111_1240.parquet"}]
        if parent == "live" else [],
    )
    def copy_capture(store, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    monkeypatch.setattr(compiler, "_download", copy_capture)

    result = compiler._load_drive_history(
        SimpleNamespace(), {"RELIANCE"}, stamp, tmp_path / "work", 120
    )

    assert set(result) == {"RELIANCE", "__NIFTY50__"}
    assert result["RELIANCE"].iloc[-1]["Close"] == 101.0
    assert result["__NIFTY50__"].iloc[-1]["Close"] == 24010.0
    assert str(result["__NIFTY50__"].index.tz) == "Asia/Kolkata"



def test_prediction_already_stored_checks_per_slot_manifest(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    import drive_all_nse_prediction_compiler as compiler

    minute = pd.Timestamp("2026-11-11 14:00:00", tz=IST)
    requested = (
        "production_data/all_nse_minute_predictions/"
        "2026/11/11/manifest_20261111_1400.json"
    )
    monkeypatch.setattr(
        compiler,
        "_find_remote",
        lambda store, remote: {"id": "prediction"} if remote == requested else None,
    )

    def download(store, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"status": "STORED"}), encoding="utf-8")

    monkeypatch.setattr(compiler, "_download", download)

    assert compiler._prediction_already_stored(SimpleNamespace(), minute) is True

    monkeypatch.setattr(compiler, "_find_remote", lambda store, remote: None)
    assert compiler._prediction_already_stored(SimpleNamespace(), minute) is False
