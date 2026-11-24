from __future__ import annotations

from datetime import date

import pandas as pd

import repair_missing_minute_snapshots as repair


def _shard_manifests(key: str, count: int) -> list[dict[str, str]]:
    return [
        {"name": f"manifest_{key}_w{index:02d}of{count:02d}.json"}
        for index in range(count)
    ]


def test_shard_worker_count_requires_every_shard_manifest():
    key = "20261016_1005"
    complete = _shard_manifests(key, 8)
    assert repair._shard_worker_count(complete, key) == 8
    assert repair._shard_worker_count(complete[:-1], key) is None
    assert repair._shard_worker_count(_shard_manifests("20261016_1010", 8), key) is None


def test_run_repair_reassembles_only_slots_without_a_stored_canonical(monkeypatch):
    key = "20261016_1005"
    calls: list[tuple[pd.Timestamp, int]] = []

    monkeypatch.setattr(
        "market_session_guard.is_nse_trading_day", lambda session_date: True
    )
    monkeypatch.setattr(
        repair, "_list_day_files",
        lambda store, session_date: [{"name": "manifest_20261016_0900.json"}],
    )
    monkeypatch.setattr(
        repair, "_list_shard_files",
        lambda store, session_date: _shard_manifests(key, 8),
    )
    monkeypatch.setattr(repair, "read_canonical_manifest", lambda store, minute, work: None)
    monkeypatch.setattr(
        repair, "repair_slot",
        lambda store, minute, *, worker_count, work_dir, expected_universe=None: (
            calls.append((minute, worker_count)) or {
                "status": "VALIDATED",
                "snapshot_persisted": True,
                "cycle_minute_ist": minute.isoformat(),
            }
        ),
    )
    monkeypatch.setattr(
        "persistent_store.GoogleDriveStore",
        lambda: object(),
    )

    report = repair.run_repair(date(2026, 10, 16), minutes=[key])

    assert calls == [(pd.Timestamp("2026-10-16 10:05:00", tz=repair.IST), 8)]
    assert report["repaired_count"] == 1
    assert report["skipped_count"] == 0
    assert report["repair_status"] == "COMPLETED"
