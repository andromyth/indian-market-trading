from datetime import datetime, date
from zoneinfo import ZoneInfo

from nse_eod_archive_audit import default_audit_end_date, expected_equity_dates

IST = ZoneInfo("Asia/Kolkata")


def test_before_market_close_audits_only_through_previous_calendar_date():
    # The failed 2026-10-16 audit ran at 06:29 IST; that day's EOD file
    # could not yet exist. The default audit must stop at 2026-10-15.
    now = datetime(2026, 10, 16, 6, 29, tzinfo=IST)
    assert default_audit_end_date(now) == date(2026, 10, 15)


def test_after_market_close_includes_current_session_date():
    now = datetime(2026, 10, 16, 21, 15, tzinfo=IST)
    assert default_audit_end_date(now) == date(2026, 10, 16)


def test_default_audit_cutoff_handles_utc_input_using_ist_clock():
    # 11:00 UTC is 16:30 IST, after the close buffer.
    now = datetime(2026, 10, 16, 11, 0, tzinfo=ZoneInfo("UTC"))
    assert default_audit_end_date(now) == date(2026, 10, 16)


def test_weekend_cutoff_still_resolves_to_last_expected_equity_session():
    now = datetime(2026, 10, 19, 7, 0, tzinfo=IST)  # Monday before market close
    end = default_audit_end_date(now)  # Sunday
    sessions = expected_equity_dates(date(2026, 10, 15), end)
    assert end == date(2026, 10, 18)
    assert sessions == [date(2026, 10, 15), date(2026, 10, 16)]
