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


def _archive_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            "2026-10-15 15:25",
            "2026-10-16 09:20",
            "2026-10-16 09:25",
        ],
        tz=repair.IST,
    )
    return pd.DataFrame(
        {
            "Open": [99.0, 100.0, 102.0],
            "High": [100.0, 103.0, 104.0],
            "Low": [98.0, 99.0, 101.0],
            "Close": [100.0, 102.0, 103.0],
            "Volume": [10.0, 100.0, 50.0],
        },
        index=index,
    )


def test_build_archive_snapshot_frame_uses_bar_stamped_at_slot():
    minute = pd.Timestamp("2026-10-16 09:20", tz=repair.IST)
    frame = repair.build_archive_snapshot_frame(
        {"AAA": _archive_frame()}, minute, ["AAA"]
    )

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["symbol"] == "AAA"
    assert row["cycle_minute_ist"] == minute.isoformat()
    assert row["quote_observed_at_ist"] == minute.isoformat()
    assert row["quote_status"] == "PASS"
    assert row["quote_source"] == repair.ARCHIVE_REPAIR_SOURCE
    assert row["actual_observed_price"] == 102.0
    assert row["day_open"] == 100.0
    assert row["day_high"] == 103.0
    assert row["day_low"] == 99.0
    assert row["day_volume"] == 100.0
    assert row["previous_close"] == 100.0


def test_run_repair_falls_back_to_archive_when_no_shard_exists(monkeypatch):
    key = "20261016_0920"
    calls: list[pd.Timestamp] = []

    monkeypatch.setattr(
        "market_session_guard.is_nse_trading_day", lambda session_date: True
    )
    monkeypatch.setattr(repair, "_list_day_files", lambda store, session_date: [])
    monkeypatch.setattr(repair, "read_canonical_manifest", lambda store, minute, work: None)
    monkeypatch.setattr(
        repair, "load_archive_histories",
        lambda store, session_date, roster, work: {"AAA": _archive_frame()},
    )
    monkeypatch.setattr(
        repair,
        "repair_slot_from_archive",
        lambda store, minute, frame, *, roster_size, work_dir: (
            calls.append(minute) or {
                "status": "ARCHIVE_REPAIRED",
                "cycle_minute_ist": minute.isoformat(),
                "row_count": len(frame),
                "snapshot_persisted": True,
            }
        ),
    )
    monkeypatch.setattr("persistent_store.GoogleDriveStore", lambda: object())

    report = repair.run_repair(
        date(2026, 10, 16),
        minutes=[key],
        expected_universe=["AAA"],
    )

    assert calls == [pd.Timestamp("2026-10-16 09:20:00", tz=repair.IST)]
    assert report["archive_status"] == "AVAILABLE"
    assert report["archive_repaired_count"] == 1
    assert report["repaired_count"] == 1
    assert report["skipped_count"] == 0
    assert report["repair_status"] == "COMPLETED"
