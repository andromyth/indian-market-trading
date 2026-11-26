"""Session-aware intraday cycle freshness classification.

Covers every state the desk can render: live-ok, live-delayed, live-stale,
session-final (after close / pre-open / weekend / holiday), and
no-cycle-for-session.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from intraday_prediction_cycle import (
    LIVE_DELAYED,
    LIVE_OK,
    LIVE_STALE,
    NO_CYCLE_FOR_SESSION,
    SESSION_FINAL,
    classify_cycle_status,
)

IST = ZoneInfo("Asia/Kolkata")


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value).replace(tzinfo=IST)


# 2026-11-26 is a Thursday trading day. 2026-11-24 is an NSE holiday (Tuesday).
# 2026-11-21 is a Saturday. 2026-11-20 is a Friday trading day.

@pytest.mark.parametrize("slots", [0, 1, 2, 3])
def test_in_session_within_three_slots_is_live_ok(slots):
    result = classify_cycle_status(
        now_ist=_ts("2026-11-26T10:00:00"),
        cycle_ts=_ts("2026-11-26T09:45:00"),
        slots_behind=slots,
        last_completed_session=date(2026, 11, 25),
    )
    assert result.status == LIVE_OK
    assert result.show_numbers is True
    assert result.is_live is True
    assert result.slots_behind == slots


@pytest.mark.parametrize("slots", [4, 5, 6])
def test_in_session_four_to_six_slots_is_live_delayed(slots):
    result = classify_cycle_status(
        now_ist=_ts("2026-11-26T10:00:00"),
        cycle_ts=_ts("2026-11-26T09:45:00"),
        slots_behind=slots,
        last_completed_session=date(2026, 11, 25),
    )
    assert result.status == LIVE_DELAYED
    assert result.show_numbers is True
    assert result.is_delayed is True


@pytest.mark.parametrize("slots", [7, 12, 40])
def test_in_session_beyond_six_slots_is_live_stale(slots):
    result = classify_cycle_status(
        now_ist=_ts("2026-11-26T10:00:00"),
        cycle_ts=_ts("2026-11-26T09:00:00"),
        slots_behind=slots,
        last_completed_session=date(2026, 11, 25),
    )
    assert result.status == LIVE_STALE
    assert result.show_numbers is False
    assert result.is_stale is True


def test_in_session_with_previous_day_cycle_is_stale_even_if_slots_zero():
    result = classify_cycle_status(
        now_ist=_ts("2026-11-26T09:20:00"),
        cycle_ts=_ts("2026-11-25T15:25:00"),
        slots_behind=0,
        last_completed_session=date(2026, 11, 25),
    )
    assert result.status == LIVE_STALE
    assert result.show_numbers is False


def test_after_close_shows_session_final_no_banner():
    result = classify_cycle_status(
        now_ist=_ts("2026-11-26T16:05:00"),
        cycle_ts=_ts("2026-11-26T15:25:00"),
        slots_behind=0,
        last_completed_session=date(2026, 11, 26),
    )
    assert result.status == SESSION_FINAL
    assert result.show_numbers is True
    assert result.is_final is True


def test_pre_open_shows_previous_session_final():
    result = classify_cycle_status(
        now_ist=_ts("2026-11-27T08:10:00"),
        cycle_ts=_ts("2026-11-26T15:25:00"),
        slots_behind=0,
        last_completed_session=date(2026, 11, 26),
    )
    assert result.status == SESSION_FINAL
    assert result.show_numbers is True


def test_weekend_shows_friday_final():
    result = classify_cycle_status(
        now_ist=_ts("2026-11-21T11:00:00"),
        cycle_ts=_ts("2026-11-20T15:25:00"),
        slots_behind=0,
        last_completed_session=date(2026, 11, 20),
    )
    assert result.status == SESSION_FINAL
    assert result.show_numbers is True


def test_holiday_shows_previous_trading_day_final():
    # 2026-11-24 is an NSE holiday.
    result = classify_cycle_status(
        now_ist=_ts("2026-11-24T11:00:00"),
        cycle_ts=_ts("2026-11-23T15:25:00"),
        slots_behind=0,
        last_completed_session=date(2026, 11, 23),
    )
    assert result.status == SESSION_FINAL
    assert result.show_numbers is True


def test_no_cycle_when_cycle_is_older_than_last_completed_session():
    result = classify_cycle_status(
        now_ist=_ts("2026-11-26T16:05:00"),
        cycle_ts=_ts("2026-11-25T15:25:00"),
        slots_behind=0,
        last_completed_session=date(2026, 11, 26),
    )
    assert result.status == NO_CYCLE_FOR_SESSION
    assert result.show_numbers is False
    assert result.is_missing is True


def test_no_cycle_when_there_is_no_cycle_at_all():
    result = classify_cycle_status(
        now_ist=_ts("2026-11-26T16:05:00"),
        cycle_ts=None,
        slots_behind=0,
        last_completed_session=date(2026, 11, 26),
    )
    assert result.status == NO_CYCLE_FOR_SESSION


def test_no_cycle_when_last_completed_session_unknown():
    result = classify_cycle_status(
        now_ist=_ts("2026-11-26T16:05:00"),
        cycle_ts=_ts("2026-11-26T15:25:00"),
        slots_behind=0,
        last_completed_session=None,
    )
    assert result.status == NO_CYCLE_FOR_SESSION


def test_negative_slots_are_clamped_to_zero():
    result = classify_cycle_status(
        now_ist=_ts("2026-11-26T10:00:00"),
        cycle_ts=_ts("2026-11-26T10:00:00"),
        slots_behind=-5,
        last_completed_session=date(2026, 11, 25),
    )
    assert result.slots_behind == 0
    assert result.status == LIVE_OK
