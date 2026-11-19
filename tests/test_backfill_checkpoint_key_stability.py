from __future__ import annotations

import hashlib
import json
import shutil

import pandas as pd
import pytest

import indstocks_backfill_all_nse_5m_history as backfill

IST = "Asia/Kolkata"
GROUP = [("RELIANCE", "NSE_1"), ("SBIN", "NSE_2"), ("TCS", "NSE_3")]
START = pd.Timestamp("2026-10-26 00:00", tz=IST).to_pydatetime()   # a Monday
END = pd.Timestamp("2026-11-02 00:00", tz=IST).to_pydatetime()
WINDOW_KEY = "20261026T0000_20261102T0000"


def _key(symbols):
    payload = "|".join(sorted(symbols)) + "|" + backfill.CHECKPOINT_SCHEMA_VERSION
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


class Store:
    """Drive stand-in. File ids are their remote paths."""

    def __init__(self, root_id, files=None):
        self.root_id = root_id
        self.service = None
        self.files = dict(files or {})          # remote path -> local file
        self.uploads = []

    def _find_child(self, parent, name, mime_type=None):
        path = f"{parent}/{name}" if parent != self.root_id else name
        if mime_type == "application/vnd.google-apps.folder":
            hit = any(key.startswith(path + "/") for key in self.files)
            return {"id": path} if hit else None
        return {"id": path} if path in self.files else None

    def upload_or_update(self, local, remote, mime_type=None):
        self.uploads.append(remote)
        return {"status": "OK"}


class Response:
    @staticmethod
    def json():
        return {"success": True, "data": {"NSE_1": {"candles": [{
            "ts": int(pd.Timestamp("2026-10-28 09:15", tz=IST).timestamp()),
            "o": 100, "h": 101, "l": 99, "c": 100.5, "v": 1000}]}}}


@pytest.fixture
def events(monkeypatch):
    captured = []
    monkeypatch.setattr(backfill, "BACKFILL_PROGRESS_CALLBACK", captured.append)
    monkeypatch.setattr(backfill.time, "sleep", lambda _: None)
    return captured


def _write_checkpoint(directory, key, symbols):
    folder = directory / key
    folder.mkdir(parents=True, exist_ok=True)
    parquet = folder / f"{WINDOW_KEY}.parquet"
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-10-28 09:20", tz=IST)], name="Datetime")
    frame = pd.DataFrame(
        {"Symbol": [symbols[0]], "Open": [10.0], "High": [11.0],
         "Low": [9.0], "Close": [10.5], "Volume": [500.0]}, index=index)
    frame.to_parquet(parquet, index=True)
    manifest = folder / f"{WINDOW_KEY}.parquet.json"
    manifest.write_text(json.dumps({
        "status": "COMPLETE", "window_key": WINDOW_KEY, "symbols": sorted(symbols),
        "row_counts": {symbol: 1 if symbol == symbols[0] else 0 for symbol in symbols},
    }), encoding="utf-8")
    remote = f"{backfill.REMOTE_ROOT}/window_checkpoints/{key}/{WINDOW_KEY}.parquet"
    return {remote: parquet, remote + ".json": manifest}


def _fake_download(monkeypatch, store):
    import persistent_store

    def download(service, file_id, destination, operation=""):
        shutil.copy(store.files[file_id], destination)

    monkeypatch.setattr(persistent_store, "download_drive_file_with_retry", download)


def test_new_checkpoints_use_full_group_key_even_when_peers_are_complete(
        monkeypatch, tmp_path, events):
    import indstocks_client
    monkeypatch.setattr(indstocks_client, "_request", lambda *a, **k: Response())
    store = Store("root-key-stability")

    backfill.fetch_indstocks_5m_batch(
        GROUP, START, END, pause_seconds=0, store=store,
        checkpoint_root=tmp_path, active_symbols={"RELIANCE"},
    )

    full_key = _key(["RELIANCE", "SBIN", "TCS"])
    assert len(store.uploads) == 2
    assert all(f"/window_checkpoints/{full_key}/" in path for path in store.uploads)
    assert _key(["RELIANCE"]) != full_key
    manifest = json.loads(next(tmp_path.rglob("*.parquet.json")).read_text(encoding="utf-8"))
    assert manifest["symbols"] == ["RELIANCE", "SBIN", "TCS"]
    assert manifest["key_scheme"] == "FULL_BATCH_V2"


def test_legacy_active_subset_checkpoint_is_reused_without_a_provider_call(
        monkeypatch, tmp_path, events):
    import indstocks_client
    active = ["RELIANCE", "SBIN"]
    files = _write_checkpoint(tmp_path / "drive", _key(active), active)
    store = Store("root-legacy-reuse", files)
    _fake_download(monkeypatch, store)

    def no_provider_call(*args, **kwargs):
        raise AssertionError("completed legacy window must not be refetched")

    monkeypatch.setattr(indstocks_client, "_request", no_provider_call)

    frames, calls = backfill.fetch_indstocks_5m_batch(
        GROUP, START, END, pause_seconds=0, store=store,
        checkpoint_root=tmp_path / "work", active_symbols=set(active),
    )

    assert calls == 0
    assert store.uploads == []
    assert not frames["RELIANCE"].empty
    stages = [event.get("stage") for event in events]
    assert "CHECKPOINT_DOWNLOADED" in stages and "RESUMED_CHECKPOINT" in stages
    downloaded = next(e for e in events if e.get("stage") == "CHECKPOINT_DOWNLOADED")
    assert downloaded["checkpoint_source"] == "LEGACY_ACTIVE_SUBSET_KEY"
    assert downloaded["bytes"] > 0


def test_parquet_without_manifest_is_refetched_and_committed(monkeypatch, tmp_path, events):
    import indstocks_client
    full = ["RELIANCE", "SBIN", "TCS"]
    files = _write_checkpoint(tmp_path / "drive", _key(full), full)
    orphan_only = {k: v for k, v in files.items() if not k.endswith(".json")}
    store = Store("root-orphan", orphan_only)
    _fake_download(monkeypatch, store)
    monkeypatch.setattr(indstocks_client, "_request", lambda *a, **k: Response())

    _, calls = backfill.fetch_indstocks_5m_batch(
        GROUP, START, END, pause_seconds=0, store=store,
        checkpoint_root=tmp_path / "work", active_symbols=set(full),
    )

    assert calls == 1
    assert any(path.endswith(".parquet.json") for path in store.uploads)


def test_failed_provider_request_never_commits_a_checkpoint(monkeypatch, tmp_path, events):
    import indstocks_client

    def fail(*args, **kwargs):
        raise RuntimeError("HTTP 401 unauthorized")

    monkeypatch.setattr(indstocks_client, "_request", fail)
    store = Store("root-failure")

    with pytest.raises(RuntimeError):
        backfill.fetch_indstocks_5m_batch(
            GROUP, START, END, pause_seconds=0, store=store,
            checkpoint_root=tmp_path, active_symbols={"RELIANCE"},
        )

    assert store.uploads == []
    failed = [e for e in events if e.get("stage") == "PROVIDER_REQUEST_FAILED"]
    assert failed and failed[0]["retryable"] is False


def test_worker_counts_artifacts_by_stage_not_by_log_events(monkeypatch):
    import autonomous_intraday_worker_service as service
    monkeypatch.setattr(service, "_log", lambda *a, **k: None)
    service._reset_backfill_counters()
    with service.STATE_LOCK:
        service.STATE["five_minute_backfill"] = {"status": "IDLE"}

    for stage in (
        "LOADING_LIVE_HISTORY", "LIVE_HISTORY_READY", "CHECKPOINT_DOWNLOADED",
        "RESUMED_CHECKPOINT", "PROVIDER_REQUEST_STARTED", "PROVIDER_RESPONSE",
        "CHECKPOINT_PERSISTED", "SYMBOL_PARQUET_PERSISTED",
    ):
        service._backfill_progress({"stage": stage, "checkpoint_path": "c", "symbol_parquet_path": "s"})

    with service.STATE_LOCK:
        state = dict(service.STATE["five_minute_backfill"])
    assert state["counters"] == {
        "checkpoints_downloaded": 1, "checkpoints_reused": 1,
        "api_requests_attempted": 1, "api_requests_successful": 1,
        "checkpoints_written": 1, "symbol_files_written": 1,
    }
    # A final symbol file must no longer masquerade as a window checkpoint write.
    assert state["last_checkpoint_path"] == "c"
    assert state["last_symbol_parquet_path"] == "s"