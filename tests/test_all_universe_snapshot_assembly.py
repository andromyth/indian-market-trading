from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import all_universe_snapshot_assembly as assembly

IST = ZoneInfo("Asia/Kolkata")
SLOT = pd.Timestamp("2026-11-12 10:05:00", tz=IST)


def _frame(symbols, *, minute=SLOT, observed_offset_minutes=0, prices=None):
    observed = minute + pd.Timedelta(minutes=observed_offset_minutes, seconds=12)
    prices = prices or {}
    return pd.DataFrame({
        "symbol": list(symbols),
        "cycle_minute_ist": [minute.isoformat()] * len(symbols),
        "quote_observed_at_ist": [observed.isoformat()] * len(symbols),
        "quote_status": ["PASS"] * len(symbols),
        "actual_observed_price": [prices.get(s, 100.0) for s in symbols],
    })


def _manifest(slot, worker_id, worker_count, declared, frame, **overrides):
    payload = {
        "schema_version": "all-universe-minute-snapshot-v2",
        "artifact_kind": "SHARD",
        "status": "SHARD_STORED",
        "snapshot_persisted": True,
        "cycle_minute_ist": slot.isoformat(),
        "worker_id": worker_id,
        "worker_count": worker_count,
        "shard_symbols": list(declared),
        "shard_symbol_count": len(declared),
        "universe_symbol_count": overrides.pop("universe_symbol_count", 4),
        "row_count": len(frame),
        "quoted_symbol_count": int(frame["quote_status"].eq("PASS").sum()),
        "finished_at_utc": "2026-11-12T04:35:30+00:00",
    }
    payload.update(overrides)
    return payload


def test_shard_universe_is_disjoint_and_complete():
    symbols = ["DDD", "AAA", "CCC", "BBB", "EEE"]
    shards = assembly.shard_universe(symbols, 3)

    assert [sorted(shard) for shard in shards] == [["AAA", "DDD"], ["BBB", "EEE"], ["CCC"]]
    flattened = [symbol for shard in shards for symbol in shard]
    assert sorted(flattened) == sorted(set(symbols))
    assert len(flattened) == len(set(flattened))


def test_shard_artifact_names_keep_shards_out_of_canonical_namespace():
    names = assembly.shard_artifact_names(SLOT, 3, 8)

    assert names["snapshot_remote"].endswith(
        "/2026/11/12/shards/snapshot_20261112_1005_w03of08.parquet"
    )
    assert names["manifest_remote"].endswith(
        "/2026/11/12/shards/manifest_20261112_1005_w03of08.json"
    )
    canonical = assembly.canonical_artifact_names(SLOT)
    assert canonical["snapshot_remote"].endswith(
        "/2026/11/12/snapshot_20261112_1005.parquet"
    )
    assert canonical["manifest_remote"].endswith(
        "/2026/11/12/manifest_20261112_1005.json"
    )


def test_assembly_publishes_complete_canonical_frame():
    shards = [
        {"worker_id": 0, "manifest": _manifest(SLOT, 0, 2, ["AAA", "CCC"], _frame(["AAA", "CCC"])),
         "frame": _frame(["AAA", "CCC"])},
        {"worker_id": 1, "manifest": _manifest(SLOT, 1, 2, ["BBB", "DDD"], _frame(["BBB", "DDD"])),
         "frame": _frame(["BBB", "DDD"])},
    ]

    result = assembly.assemble_shards(
        cycle_minute=SLOT, worker_count=2, shards=shards,
        expected_universe=["AAA", "BBB", "CCC", "DDD"],
    )

    assert result["status"] == "VALIDATED"
    assert result["manifest"]["assembly_status"] == "COMPLETE"
    assert result["manifest"]["status"] == "STORED"
    assert result["manifest"]["snapshot_persisted"] is True
    assert result["manifest"]["shard_count_completed"] == 2
    assert result["manifest"]["row_count"] == 4
    assert list(result["frame"]["symbol"]) == ["AAA", "BBB", "CCC", "DDD"]
    assert result["manifest"]["validation"]["fatal_issue_count"] == 0


def test_assembly_refuses_snapshot_when_a_shard_is_missing():
    shard = {"worker_id": 0, "manifest": _manifest(SLOT, 0, 2, ["AAA", "CCC"], _frame(["AAA", "CCC"])),
             "frame": _frame(["AAA", "CCC"])}

    result = assembly.assemble_shards(
        cycle_minute=SLOT, worker_count=2, shards=[shard],
        expected_universe=["AAA", "BBB", "CCC", "DDD"],
    )

    assert result["status"] == "INCOMPLETE"
    assert result["frame"] is None
    assert result["manifest"]["status"] != "STORED"
    assert result["manifest"]["snapshot_persisted"] is False
    assert result["manifest"]["missing_shards"] == [1]
    assert any(issue["code"] == "MISSING_SHARD" for issue in result["issues"])


def test_assembly_rejects_duplicate_symbols_across_shards():
    frame_a = _frame(["AAA", "BBB"])
    frame_b = _frame(["BBB", "CCC"])
    shards = [
        {"worker_id": 0, "manifest": _manifest(SLOT, 0, 2, ["AAA", "BBB"], frame_a), "frame": frame_a},
        {"worker_id": 1, "manifest": _manifest(SLOT, 1, 2, ["BBB", "CCC"], frame_b), "frame": frame_b},
    ]

    result = assembly.assemble_shards(cycle_minute=SLOT, worker_count=2, shards=shards)

    assert result["status"] == "INCOMPLETE"
    assert result["manifest"]["duplicate_symbol_count"] == 1
    assert "BBB" in result["manifest"]["validation"]["duplicate_symbols"]


def test_assembly_rejects_row_count_mismatch():
    frame = _frame(["AAA"])
    manifest = _manifest(SLOT, 0, 1, ["AAA"], frame, row_count=99)
    shards = [{"worker_id": 0, "manifest": manifest, "frame": frame}]

    result = assembly.assemble_shards(cycle_minute=SLOT, worker_count=1, shards=shards)

    assert result["status"] == "INCOMPLETE"
    assert any(
        issue["code"] == "SHARD_ROW_COUNT_MISMATCH" for issue in result["issues"]
    )


def test_assembly_rejects_timestamp_mismatch():
    frame = _frame(["AAA"], minute=SLOT - pd.Timedelta(minutes=5))
    manifest = _manifest(SLOT, 0, 1, ["AAA"], frame)
    shards = [{"worker_id": 0, "manifest": manifest, "frame": frame}]

    result = assembly.assemble_shards(cycle_minute=SLOT, worker_count=1, shards=shards)

    assert result["status"] == "INCOMPLETE"
    assert any(issue["code"] == "SHARD_TIMESTAMP_MISMATCH" for issue in result["issues"])


def test_assembly_rejects_missing_symbol_inside_shard():
    frame = _frame(["AAA"])
    manifest = _manifest(SLOT, 0, 1, ["AAA", "BBB"], frame)
    shards = [{"worker_id": 0, "manifest": manifest, "frame": frame}]

    result = assembly.assemble_shards(cycle_minute=SLOT, worker_count=1, shards=shards)

    assert result["status"] == "INCOMPLETE"
    missing = next(issue for issue in result["issues"] if issue["code"] == "MISSING_SYMBOL")
    assert "BBB" in missing["symbols"]


class FakeStore:
    root_id = "root"

    def __init__(self):
        self.uploads = {}

    def upload_or_update(self, local_path, remote_path, mime_type=None):
        self.uploads[remote_path] = Path(local_path).read_bytes()
        return {"status": "OK", "remote_path": remote_path}


def test_write_canonical_snapshot_uploads_parquet_before_manifest(tmp_path):
    store = FakeStore()
    frame = _frame(["AAA", "BBB"])
    manifest = assembly.assemble_shards(
        cycle_minute=SLOT, worker_count=1,
        shards=[{"worker_id": 0, "manifest": _manifest(SLOT, 0, 1, ["AAA", "BBB"], frame), "frame": frame}],
    )["manifest"]

    result = assembly.write_canonical_snapshot(store, SLOT, frame, manifest, tmp_path)

    canonical = assembly.canonical_artifact_names(SLOT)
    assert canonical["snapshot_remote"] in store.uploads
    assert canonical["manifest_remote"] in store.uploads
    assert canonical["latest_manifest_remote"] in store.uploads
    saved = json.loads(store.uploads[canonical["manifest_remote"]].decode("utf-8"))
    assert saved["status"] == "STORED"
    assert saved["assembly_status"] == "COMPLETE"
    assert saved["row_count"] == 2
    assert result["snapshot_sha256"]


def test_write_assembly_manifest_only_never_publishes_a_snapshot(tmp_path):
    store = FakeStore()
    incomplete = assembly.assemble_shards(
        cycle_minute=SLOT, worker_count=2,
        shards=[{"worker_id": 0, "manifest": _manifest(SLOT, 0, 2, ["AAA"], _frame(["AAA"])), "frame": _frame(["AAA"])}],
    )

    written = assembly.write_assembly_manifest_only(
        store, SLOT, incomplete["manifest"], tmp_path, preserve_stored=False
    )

    canonical = assembly.canonical_artifact_names(SLOT)
    assert written["status"] == "WRITTEN"
    assert canonical["snapshot_remote"] not in store.uploads
    saved = json.loads(store.uploads[canonical["manifest_remote"]].decode("utf-8"))
    assert saved["status"] == "ASSEMBLY_INCOMPLETE"
    assert saved["snapshot_persisted"] is False


def test_sharded_capture_publishes_only_assembled_canonical_snapshot(tmp_path, monkeypatch):
    import sys

    import all_universe_minute_capture as capture

    master = pd.DataFrame([
        {"EXCH": "NSE", "SECURITY_ID": str(index), "TRADING_SYMBOL": symbol, "SERIES": "EQ"}
        for index, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"], start=1)
    ])
    quotes = {
        f"NSE_{index}": {"live_price": 100.0 + index}
        for index in range(1, 5)
    }

    class FakeResponse:
        def json(self):
            return {"status": "success", "data": quotes}

    class FakeDriveStore:
        def __init__(self):
            self.uploads = {}

        def upload_or_update(self, local_path, remote_path, mime_type=None):
            self.uploads[remote_path] = Path(local_path).read_bytes()
            return {"status": "OK", "remote_path": remote_path}

    fake_store = FakeDriveStore()
    fake_indstocks = SimpleNamespace(
        _load_instruments=lambda exchange: master,
        configured=lambda: True,
        _request=lambda method, path, params=None: FakeResponse(),
    )
    monkeypatch.setitem(sys.modules, "indstocks_client", fake_indstocks)
    monkeypatch.setattr(
        "prediction_universe.authoritative_roster",
        lambda: ["AAA", "BBB", "CCC", "DDD"] + [f"EXTRA{i:03d}" for i in range(496)],
    )
    monkeypatch.setattr("persistent_store.GoogleDriveStore", lambda: fake_store)

    slot = pd.Timestamp.now(tz=IST).floor("5min")
    stem = f"{slot:%Y%m%d_%H%M}"
    day = f"{slot:%Y/%m/%d}"
    result = capture.capture_all_universe_minute_snapshot_sharded(
        now=slot + pd.Timedelta(seconds=14),
        worker_root=tmp_path,
        daily_ledger=None,
        prediction_snapshots=[],
        universe=["AAA", "BBB", "CCC", "DDD"],
        worker_count=2,
        retry_failed_shards=0,
    )

    assert result["status"] == "STORED"
    assert result["snapshot_persisted"] is True
    assert result["shard_count_completed"] == 2
    assert result["shard_count_expected"] == 2
    assert result["row_count"] == 4
    paths = set(fake_store.uploads)
    assert any(f"/shards/snapshot_{stem}_w00of02.parquet" in path for path in paths)
    assert any(f"/shards/snapshot_{stem}_w01of02.parquet" in path for path in paths)
    assert any(path.endswith(f"/{day}/snapshot_{stem}.parquet") for path in paths)
    assert any(path.endswith(f"/{day}/manifest_{stem}.json") for path in paths)
    canonical = next(
        payload for path, payload in fake_store.uploads.items()
        if path.endswith(f"/{day}/snapshot_{stem}.parquet")
    )
    canonical_path = tmp_path / "canonical.parquet"
    canonical_path.write_bytes(canonical)
    saved = pd.read_parquet(canonical_path)
    assert sorted(saved["symbol"].tolist()) == ["AAA", "BBB", "CCC", "DDD"]


def test_sharded_capture_does_not_publish_when_a_shard_fails(tmp_path, monkeypatch):
    import sys

    import all_universe_minute_capture as capture

    master = pd.DataFrame([
        {"EXCH": "NSE", "SECURITY_ID": str(index), "TRADING_SYMBOL": symbol, "SERIES": "EQ"}
        for index, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"], start=1)
    ])

    class FakeResponse:
        def json(self):
            return {"status": "success", "data": {f"NSE_{index}": {"live_price": 100.0} for index in range(1, 5)}}

    class FakeDriveStore:
        def __init__(self):
            self.uploads = {}

        def upload_or_update(self, local_path, remote_path, mime_type=None):
            # Fail every upload for the second shard, including its manifest.
            if "_w01of02" in str(remote_path):
                return {"status": "ERROR", "remote_path": remote_path}
            self.uploads[remote_path] = Path(local_path).read_bytes()
            return {"status": "OK", "remote_path": remote_path}

    fake_store = FakeDriveStore()
    fake_indstocks = SimpleNamespace(
        _load_instruments=lambda exchange: master,
        configured=lambda: True,
        _request=lambda method, path, params=None: FakeResponse(),
    )
    monkeypatch.setitem(sys.modules, "indstocks_client", fake_indstocks)
    monkeypatch.setattr(
        "prediction_universe.authoritative_roster",
        lambda: ["AAA", "BBB", "CCC", "DDD"] + [f"EXTRA{i:03d}" for i in range(496)],
    )
    monkeypatch.setattr("persistent_store.GoogleDriveStore", lambda: fake_store)

    slot = pd.Timestamp.now(tz=IST).floor("5min")
    result = capture.capture_all_universe_minute_snapshot_sharded(
        now=slot + pd.Timedelta(seconds=14),
        worker_root=tmp_path,
        daily_ledger=None,
        prediction_snapshots=[],
        universe=["AAA", "BBB", "CCC", "DDD"],
        worker_count=2,
        retry_failed_shards=0,
    )

    assert result["status"] != "STORED"
    assert result["snapshot_persisted"] is False
    assert not any(
        path.endswith(f"/{slot:%Y/%m/%d}/snapshot_{slot:%Y%m%d_%H%M}.parquet")
        for path in fake_store.uploads
    )


def test_historical_repair_does_not_move_the_latest_manifest_pointer(tmp_path):
    store = FakeStore()
    frame = _frame(["AAA", "BBB"])
    manifest = assembly.assemble_shards(
        cycle_minute=SLOT, worker_count=1,
        shards=[{"worker_id": 0, "manifest": _manifest(SLOT, 0, 1, ["AAA", "BBB"], frame), "frame": frame}],
    )["manifest"]

    assembly.write_canonical_snapshot(
        store, SLOT, frame, manifest, tmp_path, write_latest=False
    )

    canonical = assembly.canonical_artifact_names(SLOT)
    assert canonical["snapshot_remote"] in store.uploads
    assert canonical["manifest_remote"] in store.uploads
    assert canonical["latest_manifest_remote"] not in store.uploads
