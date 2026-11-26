from datetime import datetime
from zoneinfo import ZoneInfo

from market_session_guard import (
    latest_auditable_nse_session,
    latest_completed_nse_session,
    nse_session_label,
    parse_reference_date,
)

IST = ZoneInfo("Asia/Kolkata")


def dt(day: str, clock: str) -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}:00").replace(tzinfo=IST)


def test_saturday_never_reports_regular_session_open():
    now = dt("2026-10-10", "11:56")
    assert nse_session_label(now) == "MARKET CLOSED (WEEKEND)"
    assert latest_completed_nse_session(now).isoformat() == "2026-10-09"


def test_regular_session_uses_previous_completed_daily_close():
    now = dt("2026-10-09", "11:00")
    assert nse_session_label(now) == "REGULAR SESSION OPEN"
    assert latest_completed_nse_session(now).isoformat() == "2026-10-08"


def test_after_eod_window_expects_current_trading_day():
    now = dt("2026-10-09", "18:30")
    assert latest_completed_nse_session(now).isoformat() == "2026-10-09"


def test_holiday_never_reports_open_and_skips_to_prior_trading_day():
    now = dt("2026-10-02", "11:00")
    assert nse_session_label(now) == "MARKET CLOSED (NSE HOLIDAY)"
    assert latest_completed_nse_session(now).isoformat() == "2026-10-01"


def test_reference_date_parser_accepts_iso_daily_labels():
    assert parse_reference_date("2026-10-05 00:00:00") == datetime(2026, 10, 5).date()
    assert parse_reference_date(None) is None
    assert parse_reference_date("not-a-date") is None


def test_audit_delayed_before_next_open_uses_previous_session():
    # Overnight run at 01:34 IST on Nov 26 must audit Nov 25, not Nov 26.
    assert latest_auditable_nse_session(dt("2026-11-26", "01:34")).isoformat() == "2026-11-25"


def test_audit_after_close_uses_current_session():
    assert latest_auditable_nse_session(dt("2026-11-25", "16:05")).isoformat() == "2026-11-25"
    assert latest_auditable_nse_session(dt("2026-11-25", "15:59")).isoformat() == "2026-11-23"


def test_audit_weekend_and_holiday_resolve_to_prior_session():
    assert latest_auditable_nse_session(dt("2026-11-22", "10:00")).isoformat() == "2026-11-20"
    assert latest_auditable_nse_session(dt("2026-11-24", "17:00")).isoformat() == "2026-11-23"
