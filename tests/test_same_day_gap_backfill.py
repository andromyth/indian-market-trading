from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import same_day_gap_backfill as gap
from consolidate_live_5m_into_archive import expected_session_slots


def test_missing_live_history_slots_reports_only_absent_captures():
    session_date = date(2026, 11, 23)
    slots = expected_session_slots()
    files = [
        {"name": f"capture_20261123_{slot:%H%M}.parquet"}
        for slot in slots
        if slot.strftime("%H%M") != "1210"
    ]

    assert gap.missing_live_history_slots(files, session_date) == ["1210"]


def test_warming_prediction_slots_ignores_latest_pointer_and_non_warming():
    manifests = [
        (
            "manifest_20261123_1210.json",
            {"prediction_count": 0, "prediction_status_counts": {"WARMING_HISTORY": 500}},
        ),
        (
            "manifest_20261123_1215.json",
            {"prediction_count": 12, "prediction_status_counts": {"PREDICTED": 12}},
        ),
        ("latest_manifest.json", {"prediction_count": 0}),
    ]

    assert gap.warming_prediction_slots(manifests) == ["20261123_1210"]


def test_discover_gap_state_uses_roster_only_when_a_gap_exists():
    session_date = date(2026, 11, 23)
    roster = ["AAA", "BBB", "CCC"]

    no_gap = gap.discover_gap_state(
        session_date=session_date,
        live_history_files=[
            {"name": f"capture_20261123_{slot:%H%M}.parquet"}
            for slot in expected_session_slots()
        ],
        prediction_manifests=[],
        roster=roster,
    )
    assert no_gap["needs_backfill"] is False
    assert no_gap["gap_symbol_count"] == 0
    assert no_gap["reason"] == "none"

    with_gap = gap.discover_gap_state(
        session_date=session_date,
        live_history_files=[],
        prediction_manifests=[],
        roster=roster,
    )
    assert with_gap["needs_backfill"] is True
    assert with_gap["gap_symbol_count"] == len(roster)
    assert with_gap["reason"] == "missing_live_history_slots"


def test_backfill_payload_is_provider_safe():
    payload = gap.backfill_payload(date(2026, 11, 23), ["AAA", "BBB"])

    assert payload == {
        "confirm": "START_NATIVE_5M_BACKFILL",
        "end_date": "2026-11-23",
        "replace_existing_history": "no",
        "symbols": "AAA,BBB",
    }


def test_large_symbol_list_falls_back_to_runner_gap_detection():
    symbols = [f"SYMBOL{index:03d}" for index in range(500)]

    assert gap.backfill_symbol_argument(symbols) == ""
    assert gap.backfill_payload(date(2026, 11, 23), symbols)["symbols"] == ""


def test_classify_backfill_status():
    assert gap.classify_backfill_status("COMPLETED") == "DONE"
    assert gap.classify_backfill_status("COMPLETED_WITH_GAPS") == "DONE"
    assert gap.classify_backfill_status("RUNNING") == "ACTIVE"
    assert gap.classify_backfill_status("PAUSED_FOR_LIVE_MARKET") == "ACTIVE"
    assert gap.classify_backfill_status("FAILED") == "FAILED"
    assert gap.classify_backfill_status("IDLE") == "IDLE"


def test_wait_ignores_previous_session_completion(monkeypatch):
    monkeypatch.setattr(
        gap,
        "backfill_state",
        lambda *args, **kwargs: {
            "status": "COMPLETED",
            "http_status": 200,
            "state": {"started_at_utc": "2026-11-22T10:35:00+00:00"},
        },
    )

    result = gap.wait_for_backfill(
        "https://example.test",
        "token",
        session_date=date(2026, 11, 23),
        max_wait_minutes=1,
    )

    assert result["status"] == "IDLE"


def test_wait_returns_current_session_failure(monkeypatch):
    monkeypatch.setattr(
        gap,
        "backfill_state",
        lambda *args, **kwargs: {
            "status": "FAILED",
            "http_status": 200,
            "state": {"started_at_utc": "2026-11-23T10:35:00+00:00"},
        },
    )

    result = gap.wait_for_backfill(
        "https://example.test",
        "token",
        session_date=date(2026, 11, 23),
        max_wait_minutes=1,
    )

    assert result["status"] == "FAILED"


def test_start_backfill_accepts_already_running(monkeypatch):
    monkeypatch.setattr(
        gap,
        "_request_json",
        lambda *args, **kwargs: (409, {"status": "BACKFILL_ALREADY_RUNNING"}),
    )

    result = gap.start_backfill(
        "https://example.test", "token", date(2026, 11, 23), ["AAA"]
    )

    assert result["status"] == "BACKFILL_ALREADY_RUNNING"


def test_start_backfill_defers_outside_allowed_window(monkeypatch):
    monkeypatch.setattr(
        gap,
        "_request_json",
        lambda *args, **kwargs: (
            423,
            {
                "status": "BACKFILL_OUTSIDE_ALLOWED_WINDOW",
                "next_allowed_time_ist": "16:00 IST today",
            },
        ),
    )

    result = gap.start_backfill(
        "https://example.test", "token", date(2026, 11, 23), ["AAA"]
    )

    assert result["http_status"] == 423
    assert result["status"] == "BACKFILL_OUTSIDE_ALLOWED_WINDOW"
    assert result["next_allowed_time_ist"] == "16:00 IST today"


def test_main_exits_zero_when_backfill_is_deferred(monkeypatch, capsys):
    import market_session_guard
    import persistent_store
    import prediction_universe

    monkeypatch.setattr(gap, "resolve_session_date", lambda raw=None: date(2026, 11, 23))
    monkeypatch.setattr(market_session_guard, "is_nse_trading_day", lambda _date: True)
    monkeypatch.setattr(persistent_store, "GoogleDriveStore", lambda: object())
    monkeypatch.setattr(prediction_universe, "authoritative_roster", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(gap, "_load_live_history_files", lambda store, session: [])
    monkeypatch.setattr(gap, "_load_prediction_manifests", lambda store, session: [])
    monkeypatch.setattr(gap, "_load_snapshot_files", lambda store, session: [])
    monkeypatch.setattr(
        gap,
        "discover_gap_state",
        lambda **kwargs: {
            "needs_backfill": True,
            "gap_symbol_count": 2,
            "reason": "missing_live_history_slots",
        },
    )
    monkeypatch.setattr(gap, "service_url", lambda: "https://example.test")
    monkeypatch.setattr(gap, "identity_token", lambda url: "token")
    monkeypatch.setattr(gap, "backfill_state", lambda url, token: {"status": "IDLE"})
    monkeypatch.setattr(
        gap,
        "start_backfill",
        lambda *args, **kwargs: {
            "http_status": 423,
            "status": "BACKFILL_OUTSIDE_ALLOWED_WINDOW",
            "next_allowed_time_ist": "16:00 IST today",
        },
    )

    monkeypatch.setattr(sys, "argv", ["same_day_gap_backfill.py", "--date", "2026-11-23"])
    code = gap.main()

    assert code == 0
    assert "SKIPPED_OUTSIDE_WINDOW" in capsys.readouterr().out


def _backfill_state(status: str, started: str, updated: str) -> dict:
    return {
        "status": status,
        "state": {"started_at_utc": started, "updated_at_utc": updated},
    }


def test_should_skip_existing_backfill_only_for_clean_current_session():
    now = datetime(2026, 11, 25, 13, 0, tzinfo=timezone.utc)
    current = _backfill_state(
        "COMPLETED",
        "2026-11-25T10:35:00+00:00",
        "2026-11-25T10:40:00+00:00",
    )

    assert gap.should_skip_existing_backfill(current, date(2026, 11, 25), now=now)
    assert not gap.should_skip_existing_backfill(
        current, date(2026, 11, 25), now=now, force=True
    )
    assert not gap.should_skip_existing_backfill(current, date(2026, 11, 24), now=now)


def test_should_retry_gapped_session_only_after_cooldown():
    now = datetime(2026, 11, 25, 13, 0, tzinfo=timezone.utc)
    recent = _backfill_state(
        "COMPLETED_WITH_GAPS",
        "2026-11-25T10:35:00+00:00",
        "2026-11-25T12:45:00+00:00",
    )
    stale = _backfill_state(
        "COMPLETED_WITH_GAPS",
        "2026-11-25T10:35:00+00:00",
        "2026-11-25T11:30:00+00:00",
    )

    assert gap.should_skip_existing_backfill(
        recent, date(2026, 11, 25), now=now, retry_minutes=30
    )
    assert not gap.should_skip_existing_backfill(
        stale, date(2026, 11, 25), now=now, retry_minutes=30
    )


def test_backfill_session_ownership_uses_target_end_date_before_start_time():
    current = {
        "status": "COMPLETED",
        "state": {
            "target_end_date": "2026-11-25",
            "started_at_utc": "2026-11-26T10:35:00+00:00",
        },
    }
    assert gap._belongs_to_session(current, date(2026, 11, 25))
    assert not gap._belongs_to_session(current, date(2026, 11, 26))


def test_backfill_session_ownership_falls_back_to_started_at_when_target_missing():
    current = {"state": {"started_at_utc": "2026-11-25T10:35:00+00:00"}}
    assert gap._belongs_to_session(current, date(2026, 11, 25))
    assert not gap._belongs_to_session(current, date(2026, 11, 26))


def test_missing_snapshot_slots_detects_absent_snapshots():
    from audit_all_universe_minute_snapshots import expected_minutes

    session_date = date(2026, 11, 23)
    expected = expected_minutes(session_date)
    all_files = [{"name": f"snapshot_{key}.parquet"} for key in expected]

    assert gap.missing_snapshot_slots(all_files, session_date) == []

    # One missing slot
    with_gap = [f for f in all_files if "1210" not in f["name"]]
    missing = gap.missing_snapshot_slots(with_gap, session_date)
    assert missing == ["20261123_1210"]


def test_discover_gap_state_detects_missing_snapshots():
    from audit_all_universe_minute_snapshots import expected_minutes

    session_date = date(2026, 11, 23)
    roster = ["AAA", "BBB", "CCC"]
    expected = expected_minutes(session_date)
    all_snapshots = [{"name": f"snapshot_{key}.parquet"} for key in expected]
    missing_one_snapshot = [f for f in all_snapshots if "1210" not in f["name"]]

    state = gap.discover_gap_state(
        session_date=session_date,
        live_history_files=[
            {"name": f"capture_20261123_{slot:%H%M}.parquet"}
            for slot in expected_session_slots()
        ],
        prediction_manifests=[],
        snapshot_files=missing_one_snapshot,
        roster=roster,
    )
    assert state["needs_backfill"] is True
    assert state["gap_symbol_count"] == len(roster)
    assert state["reason"] == "missing_snapshot_slots"
    assert state["missing_snapshot_slots"] == ["20261123_1210"]


def test_post_backfill_snapshot_repair_invoked_on_missing_snapshots(monkeypatch, capsys):
    import market_session_guard
    import persistent_store
    import prediction_universe

    monkeypatch.setattr(gap, "resolve_session_date", lambda raw=None: date(2026, 11, 23))
    monkeypatch.setattr(market_session_guard, "is_nse_trading_day", lambda _date: True)
    monkeypatch.setattr(persistent_store, "GoogleDriveStore", lambda: object())
    monkeypatch.setattr(prediction_universe, "authoritative_roster", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(gap, "_load_live_history_files", lambda store, session: [])
    monkeypatch.setattr(gap, "_load_prediction_manifests", lambda store, session: [])
    monkeypatch.setattr(gap, "_load_snapshot_files", lambda store, session: [])
    monkeypatch.setattr(
        gap,
        "discover_gap_state",
        lambda **kwargs: {
            "needs_backfill": True,
            "gap_symbol_count": 2,
            "reason": "missing_snapshot_slots",
            "missing_snapshot_slots": ["20261123_1210"],
        },
    )
    monkeypatch.setattr(gap, "service_url", lambda: "https://example.test")
    monkeypatch.setattr(gap, "identity_token", lambda url: "token")
    monkeypatch.setattr(
        gap,
        "backfill_state",
        lambda url, token: {
            "status": "COMPLETED",
            "state": {"started_at_utc": "2026-11-23T10:35:00+00:00"},
        },
    )

    repaired_calls = []
    import repair_missing_minute_snapshots
    monkeypatch.setattr(
        repair_missing_minute_snapshots,
        "run_repair",
        lambda session_date, minutes=None, expected_universe=None: (
            repaired_calls.append((session_date, minutes, expected_universe)) or {
                "repair_status": "COMPLETED",
                "repaired_count": 1,
            }
        ),
    )

    monkeypatch.setattr(sys, "argv", ["same_day_gap_backfill.py", "--date", "2026-11-23"])
    code = gap.main()

    assert code == 0
    assert len(repaired_calls) == 1
    assert repaired_calls[0][0] == date(2026, 11, 23)
    assert repaired_calls[0][1] == ["20261123_1210"]
    assert "POST_BACKFILL_SNAPSHOT_REPAIR=" in capsys.readouterr().out


def test_post_backfill_snapshot_repair_can_be_disabled(monkeypatch, capsys):
    import market_session_guard
    import persistent_store
    import prediction_universe

    monkeypatch.setattr(gap, "resolve_session_date", lambda raw=None: date(2026, 11, 23))
    monkeypatch.setattr(market_session_guard, "is_nse_trading_day", lambda _date: True)
    monkeypatch.setattr(persistent_store, "GoogleDriveStore", lambda: object())
    monkeypatch.setattr(prediction_universe, "authoritative_roster", lambda: ["AAA", "BBB"])
    monkeypatch.setattr(gap, "_load_live_history_files", lambda store, session: [])
    monkeypatch.setattr(gap, "_load_prediction_manifests", lambda store, session: [])
    monkeypatch.setattr(gap, "_load_snapshot_files", lambda store, session: [])
    monkeypatch.setattr(
        gap,
        "discover_gap_state",
        lambda **kwargs: {
            "needs_backfill": True,
            "gap_symbol_count": 2,
            "reason": "missing_snapshot_slots",
            "missing_snapshot_slots": ["20261123_1210"],
        },
    )
    monkeypatch.setattr(gap, "service_url", lambda: "https://example.test")
    monkeypatch.setattr(gap, "identity_token", lambda url: "token")
    monkeypatch.setattr(
        gap,
        "backfill_state",
        lambda url, token: {
            "status": "COMPLETED",
            "state": {"started_at_utc": "2026-11-23T10:35:00+00:00"},
        },
    )

    repaired_calls = []
    import repair_missing_minute_snapshots
    monkeypatch.setattr(
        repair_missing_minute_snapshots,
        "run_repair",
        lambda *args, **kwargs: repaired_calls.append(args),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["same_day_gap_backfill.py", "--date", "2026-11-23", "--no-repair-snapshots"],
    )
    code = gap.main()

    assert code == 0
    assert len(repaired_calls) == 0
    assert "POST_BACKFILL_SNAPSHOT_REPAIR=" not in capsys.readouterr().out
