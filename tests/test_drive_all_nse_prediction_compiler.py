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



def test_latest_capture_prefers_newest_fresh_eligible_slot(monkeypatch):
    import json
    from types import SimpleNamespace
    import drive_all_nse_prediction_compiler as compiler

    now = pd.Timestamp("2026-11-09 14:45:30", tz=IST)
    minutes = ["1440", "1445"]
    manifests = {
        minute: {
            "cycle_minute_ist": f"2026-11-09T{minute[:2]}:{minute[2:]}:00+05:30",
            "status": "STORED",
            "snapshot_persisted": True,
            "snapshot_drive_path": f"production_data/all_universe_minute_snapshots/2026/11/09/snapshot_20261109_{minute}.parquet",
            "row_count": 1,
        }
        for minute in minutes
    }
    items = [
        {"id": minute, "name": f"manifest_20261109_{minute}.json"}
        for minute in reversed(minutes)
    ]

    def find_remote(store, remote):
        if remote.endswith("/2026/11/09"):
            return {"id": "day"}
        if remote.endswith("/latest_manifest.json"):
            return {"id": "latest", "name": "latest_manifest.json"}
        if "/snapshot_20261109_1445.parquet" in remote:
            return {"id": "snapshot_1445"}
        if "/snapshot_20261109_1440.parquet" in remote:
            return {"id": "snapshot_1440"}
        return None

    def download(store, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if file_id in manifests:
            destination.write_text(json.dumps(manifests[file_id]), encoding="utf-8")
        elif file_id == "latest":
            destination.write_text(json.dumps(manifests["1445"]), encoding="utf-8")
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
            "cycle_minute_ist": ["2026-11-09T14:45:00+05:30" if "1445" in str(path) else "2026-11-09T14:40:00+05:30"],
            "quote_status": ["PASS"],
            "actual_observed_price": [1500.0],
            "quote_observed_at_ist": ["2026-11-09T14:45:05+05:30" if "1445" in str(path) else "2026-11-09T14:40:05+05:30"],
            "quote_error": [None],
        }),
    )

    _, _, minute = compiler._latest_capture(SimpleNamespace(), now=now)

    assert minute == pd.Timestamp("2026-11-09 14:45:00", tz=IST)


def test_latest_capture_rejects_snapshot_older_than_live_freshness_window(monkeypatch):
    import json
    from types import SimpleNamespace
    import pytest
    import drive_all_nse_prediction_compiler as compiler

    now = pd.Timestamp("2026-11-09 14:45:30", tz=IST)
    stale = {
        "cycle_minute_ist": "2026-11-09T14:30:00+05:30",
        "status": "STORED",
        "snapshot_persisted": True,
        "snapshot_drive_path": "production_data/all_universe_minute_snapshots/2026/11/09/snapshot_20261109_1430.parquet",
        "row_count": 1,
    }

    def find_remote(store, remote):
        if remote.endswith("/2026/11/09"):
            return {"id": "day"}
        if remote.endswith("/latest_manifest.json"):
            return {"id": "latest", "name": "latest_manifest.json"}
        return None

    def download(store, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(stale), encoding="utf-8")

    monkeypatch.setattr(compiler, "_find_remote", find_remote)
    monkeypatch.setattr(compiler, "_list_children", lambda store, parent_id: [
        {"id": "stale", "name": "manifest_20261109_1430.json"}
    ])
    monkeypatch.setattr(compiler, "_download", download)

    with pytest.raises(RuntimeError, match="No fresh, durably stored"):
        compiler._latest_capture(SimpleNamespace(), now=now)

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


def test_latest_capture_skips_sharded_snapshot_that_was_never_assembled(monkeypatch):
    import json
    from types import SimpleNamespace
    import drive_all_nse_prediction_compiler as compiler

    now = pd.Timestamp("2026-11-09 14:45:30", tz=IST)
    unassembled_remote = "production_data/all_universe_minute_snapshots/2026/11/09/snapshot_20261109_1445.parquet"
    assembled_remote = "production_data/all_universe_minute_snapshots/2026/11/09/snapshot_20261109_1440.parquet"
    manifests = {
        "unassembled": {
            "schema_version": "all-universe-minute-snapshot-v2",
            "cycle_minute_ist": "2026-11-09T14:45:00+05:30",
            "status": "STORED",
            "snapshot_persisted": True,
            "worker_count": 8,
            "shard_count_expected": 8,
            "shard_count_completed": 6,
            "assembly_status": "INCOMPLETE",
            "snapshot_drive_path": unassembled_remote,
            "row_count": 1,
            "artifact_kind": "SHARD",
        },
        "assembled": {
            "schema_version": "all-universe-minute-snapshot-v2",
            "cycle_minute_ist": "2026-11-09T14:40:00+05:30",
            "status": "STORED",
            "snapshot_persisted": True,
            "worker_count": 8,
            "shard_count_expected": 8,
            "shard_count_completed": 8,
            "assembly_status": "COMPLETE",
            "snapshot_drive_path": assembled_remote,
            "row_count": 1,
        },
    }
    items = [
        {"id": "unassembled", "name": "manifest_20261109_1445.json"},
        {"id": "assembled", "name": "manifest_20261109_1440.json"},
    ]

    def find_remote(store, remote):
        if remote.endswith("/2026/11/09"):
            return {"id": "day"}
        if remote.endswith("/latest_manifest.json"):
            return {"id": "latest", "name": "latest_manifest.json"}
        if remote == unassembled_remote:
            return {"id": "snapshot_unassembled"}
        if remote == assembled_remote:
            return {"id": "snapshot_assembled"}
        return None

    def download(store, file_id, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if file_id in manifests:
            destination.write_text(json.dumps(manifests[file_id]), encoding="utf-8")
        elif file_id == "latest":
            destination.write_text(json.dumps(manifests["unassembled"]), encoding="utf-8")
        else:
            destination.write_bytes(b"parquet-placeholder")

    def read_parquet(path):
        minute = "14:45" if "1445" in str(path) else "14:40"
        return pd.DataFrame({
            "symbol": ["RELIANCE"],
            "instrument_code": ["NSE_EQ|123"],
            "cycle_minute_ist": [f"2026-11-09T{minute}:00+05:30"],
            "quote_status": ["PASS"],
            "actual_observed_price": [1500.0],
            "quote_observed_at_ist": [f"2026-11-09T{minute}:05+05:30"],
            "quote_error": [None],
        })

    monkeypatch.setattr(compiler, "_find_remote", find_remote)
    monkeypatch.setattr(compiler, "_list_children", lambda store, parent_id: items)
    monkeypatch.setattr(compiler, "_download", download)
    monkeypatch.setattr(compiler.pd, "read_parquet", read_parquet)

    manifest, _, minute = compiler._latest_capture(SimpleNamespace(), now=now)

    assert minute == pd.Timestamp("2026-11-09 14:40:00", tz=IST)
    assert manifest["assembly_status"] == "COMPLETE"


def test_load_drive_history_downloads_seed_symbols_in_parallel(monkeypatch, tmp_path, capsys):
    import json
    import threading
    from types import SimpleNamespace

    import drive_all_nse_prediction_compiler as compiler

    minute = pd.Timestamp("2026-11-09 14:40:00", tz=IST)
    lookback = 2
    symbols = ["AAA", "BBB", "__NIFTY50__"]
    folder_rows = [
        {"id": f"folder_{symbol}", "name": f"symbol={symbol}", "mimeType": compiler.FOLDER_MIME}
        for symbol in symbols
    ]
    files_by_folder = {
        f"folder_{symbol}": [
            {
                "id": f"file_{symbol}",
                "name": f"{symbol}.parquet",
                "mimeType": "application/parquet",
                "modifiedTime": "2026-11-09T09:00:00Z",
                "size": 100,
            }
        ]
        for symbol in symbols
    }

    def fake_parquet(symbol):
        stamps = [
            pd.Timestamp("2026-11-09 14:35:00", tz=IST),
            pd.Timestamp("2026-11-09 14:40:00", tz=IST),
            pd.Timestamp("2026-11-09 14:45:00", tz=IST),
        ]
        frame = pd.DataFrame(
            {column: [1.0 + offset for offset in range(len(stamps))] for column in compiler.OHLCV},
            index=pd.DatetimeIndex(stamps, name="Datetime"),
        )
        return frame

    def find_remote(store, remote):
        return {"id": "archive"} if remote == compiler.HISTORY_ROOT else None

    def list_children(store, parent_id):
        if parent_id == "archive":
            return folder_rows + [
                {"id": "folder_unwanted", "name": "symbol=UNWANTED", "mimeType": compiler.FOLDER_MIME},
            ]
        return files_by_folder.get(parent_id, [])

    downloaded: list[str] = []

    def download(service, file_id, destination):
        downloaded.append(str(file_id))
        fake_parquet(str(file_id).removeprefix("file_")).to_parquet(destination)

    thread_services: list[SimpleNamespace] = []

    def fake_thread_service(store):
        marker = SimpleNamespace(thread=threading.current_thread().name)
        thread_services.append(marker)
        return marker

    monkeypatch.setattr(compiler, "_find_remote", find_remote)
    monkeypatch.setattr(compiler, "_list_children", list_children)
    monkeypatch.setattr(compiler, "_download", download)
    monkeypatch.setattr(compiler, "_thread_service", fake_thread_service)
    monkeypatch.setattr(compiler, "HISTORY_DOWNLOAD_WORKERS", 3)

    store = SimpleNamespace(credentials=None, service=None)
    history = compiler._load_drive_history(store, {"AAA", "BBB"}, minute, tmp_path, lookback)

    assert set(history) == {"AAA", "BBB", "__NIFTY50__"}
    for frame in history.values():
        assert len(frame) == lookback
        assert frame.index.max() <= minute
    assert sorted(downloaded) == [f"file_{symbol}" for symbol in symbols]
    assert thread_services, "parallel download path must hand out worker services"

    payload = None
    for line in capsys.readouterr().out.splitlines():
        if "HISTORY_SEED_LOADED" in line:
            payload = json.loads(line)
    assert payload is not None
    assert payload["symbols"] == 3
    assert payload["downloaded"] == 3
    assert payload["workers"] == 3
    assert (tmp_path / "history_cache" / "index.json").is_file()


def test_download_accepts_store_or_raw_per_thread_service(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import persistent_store

    import drive_all_nse_prediction_compiler as compiler

    seen: dict = {}

    def fake_retry(service, file_id, destination, operation=None, max_attempts=4):
        seen["service"] = service
        seen["file_id"] = file_id

    monkeypatch.setattr(persistent_store, "download_drive_file_with_retry", fake_retry)

    compiler._download(SimpleNamespace(service="STORE_SERVICE"), "fid", tmp_path / "a.bin")
    assert seen == {"service": "STORE_SERVICE", "file_id": "fid"}

    compiler._download("RAW_RESOURCE_SERVICE", "fid2", tmp_path / "b.bin")
    assert seen == {"service": "RAW_RESOURCE_SERVICE", "file_id": "fid2"}


def _capture_row(symbol: str, timestamp: pd.Timestamp, close: float) -> dict:
    return {
        "symbol": symbol,
        "Timestamp": timestamp,
        "Open": close,
        "High": close,
        "Low": close,
        "Close": close,
        "Volume": 1,
    }


def test_normalize_capture_frame_returns_long_rows_in_ist():
    import drive_all_nse_prediction_compiler as compiler

    frame = compiler._normalize_capture_frame(pd.DataFrame([
        {
            "symbol": " aaa ",
            "Timestamp": "2026-11-16 09:20:00",
            "Open": 1,
            "High": 2,
            "Low": 1,
            "Close": 1.5,
            "Volume": 10,
        }
    ]))

    assert list(frame.columns) == ["symbol", "Timestamp", "Open", "High", "Low", "Close", "Volume"]
    assert frame.loc[0, "symbol"] == "AAA"
    assert str(frame.loc[0, "Timestamp"].tz) == "Asia/Kolkata"


def test_live_history_state_ingests_only_captures_not_yet_merged(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    import drive_all_nse_prediction_compiler as compiler

    source = tmp_path / "source"
    source.mkdir()
    t0 = pd.Timestamp("2026-11-16 09:20:00", tz=IST)
    t1 = pd.Timestamp("2026-11-16 09:25:00", tz=IST)
    pd.DataFrame([
        _capture_row("AAA", t0, 1.0),
        _capture_row("BBB", t0, 5.0),
        _capture_row(compiler.MARKET_CONTEXT_SYMBOL, t0, 100.0),
    ]).to_parquet(source / "capture_0920.parquet", index=False)

    downloads: list[str] = []

    monkeypatch.setattr(
        compiler,
        "_find_remote",
        lambda store, path: {"id": "live"} if path.endswith("2026/11/16") else None,
    )
    monkeypatch.setattr(
        compiler,
        "_list_children",
        lambda store, folder_id: [
            {
                "id": path.name,
                "name": path.name,
                "modifiedTime": "2026-11-16T03:20:00Z",
                "size": str(path.stat().st_size),
            }
            for path in sorted(source.glob("capture_*.parquet"))
        ],
    )

    def fake_download(store, file_id, destination):
        downloads.append(file_id)
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source / file_id, destination)

    monkeypatch.setattr(compiler, "_download", fake_download)

    work = tmp_path / "work"
    history: dict = {}
    symbols = {"AAA", "BBB"}

    compiler._load_live_history_state(object(), symbols, t1, work, 120, history)
    assert downloads == ["capture_0920.parquet"]
    assert len(history["AAA"]) == 1
    assert len(history["BBB"]) == 1
    assert len(history[compiler.MARKET_CONTEXT_SYMBOL]) == 1

    pd.DataFrame([_capture_row("AAA", t1, 2.0)]).to_parquet(
        source / "capture_0925.parquet", index=False
    )
    downloads.clear()
    compiler._load_live_history_state(object(), symbols, t1, work, 120, history)
    assert downloads == ["capture_0925.parquet"]
    assert history["AAA"]["Close"].tolist() == [1.0, 2.0]
    assert len(history["BBB"]) == 1

    downloads.clear()
    compiler._load_live_history_state(object(), symbols, t1, work, 120, history)
    assert downloads == []


def test_load_live_history_state_keeps_seed_rows_when_capture_is_late(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    import drive_all_nse_prediction_compiler as compiler

    source = tmp_path / "source"
    source.mkdir()
    t0 = pd.Timestamp("2026-11-16 09:20:00", tz=IST)
    pd.DataFrame([_capture_row("AAA", t0, 1.0)]).to_parquet(
        source / "capture_0920.parquet", index=False
    )

    monkeypatch.setattr(
        compiler,
        "_find_remote",
        lambda store, path: {"id": "live"} if path.endswith("2026/11/16") else None,
    )
    monkeypatch.setattr(
        compiler,
        "_list_children",
        lambda store, folder_id: [
            {"id": path.name, "name": path.name, "modifiedTime": "x", "size": "1"}
            for path in sorted(source.glob("capture_*.parquet"))
        ],
    )

    def fake_download(store, file_id, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source / file_id, destination)

    monkeypatch.setattr(compiler, "_download", fake_download)

    seed_frame = pd.DataFrame(
        {"Open": [9.0], "High": [9.0], "Low": [9.0], "Close": [9.0], "Volume": [1]},
        index=pd.DatetimeIndex([t0], tz=IST, name="Datetime"),
    )
    history = {"AAA": seed_frame}

    compiler._load_live_history_state(object(), {"AAA"}, t0, tmp_path / "work", 120, history)

    assert history["AAA"]["Close"].tolist() == [1.0]
