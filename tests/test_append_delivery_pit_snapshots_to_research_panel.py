from __future__ import annotations

from datetime import date

from append_delivery_pit_snapshots_to_research_panel import plan_panel_extension
from nse_bulk_eod_ingest import business_days


def test_extension_plan_uses_nse_sessions_and_requires_all_three_sources():
    sessions = set(business_days(date(2026, 10, 1), date(2026, 10, 7)))
    inventory = {
        "eod": set(sessions),
        "delivery": set(sessions),
        "pit": {date(2026, 10, 1), date(2026, 10, 5)},
    }
    plan = plan_panel_extension(date(2026, 10, 1), date(2026, 10, 7), inventory)
    assert plan["expected_sessions"] == ["2026-10-05", "2026-10-06", "2026-10-07"]
    assert plan["missing_pit_dates"] == ["2026-10-06", "2026-10-07"]
    assert plan["status"] == "GAPS_REMAIN"


def test_extension_plan_blocks_missing_eod_and_delivery_archives():
    target = date(2026, 10, 7)
    sessions = set(business_days(date(2026, 10, 5), target))
    inventory = {
        "eod": set(sessions) - {target},
        "delivery": set(sessions),
        "pit": set(sessions),
    }
    plan = plan_panel_extension(date(2026, 10, 5), target, inventory)
    assert plan["missing_eod_dates"] == ["2026-10-07"]
    assert plan["status"] == "GAPS_REMAIN"


def test_extension_plan_ready_only_when_every_expected_nse_session_is_persisted():
    target = date(2026, 10, 7)
    sessions = set(business_days(date(2026, 10, 5), target))
    plan = plan_panel_extension(
        date(2026, 10, 5),
        target,
        {"eod": sessions, "delivery": sessions, "pit": sessions},
    )
    assert plan["status"] == "READY"
    assert plan["ready_to_append_dates"] == ["2026-10-06", "2026-10-07"]


def test_extension_plan_returns_no_missing_sessions_when_already_current():
    target = date(2026, 10, 30)
    plan = plan_panel_extension(
        target,
        target,
        {"eod": set(), "delivery": set(), "pit": set()},
    )
    assert plan["expected_sessions"] == []
    assert plan["status"] == "READY"
