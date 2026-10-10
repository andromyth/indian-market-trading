"""NSE session and Delivery prediction-freshness guards for the Streamlit UI."""
from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from production_data_freshness_gate import HOLIDAYS, previous_trading_day

IST = ZoneInfo("Asia/Kolkata")
OPEN = time(9, 15)
CLOSE = time(15, 30)
EOD_READY = time(18, 0)


def _to_ist(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=IST)
    return value.astimezone(IST)


def is_nse_trading_day(value: date) -> bool:
    return value.weekday() < 5 and value not in HOLIDAYS


def nse_session_label(value: datetime) -> str:
    """Return a session label that never marks a weekend/holiday as open."""
    current = _to_ist(value)
    if not is_nse_trading_day(current.date()):
        kind = "WEEKEND" if current.weekday() >= 5 else "NSE HOLIDAY"
        return f"MARKET CLOSED ({kind})"
    now = current.time()
    if now < OPEN:
        return "PRE-OPEN / BEFORE REGULAR SESSION"
    if now > CLOSE:
        return "MARKET CLOSED"
    if now == CLOSE:
        return "REGULAR SESSION CLOSE"
    return "REGULAR SESSION OPEN"


def latest_completed_nse_session(value: datetime) -> date:
    """Resolve the session whose daily close should back the current signal.

    Before the 18:00 IST EOD persistence window, use the previous trading
    session. After that window on a trading day, today's session is expected.
    On weekends and holidays, use the most recent preceding NSE session.
    """
    current = _to_ist(value)
    current_date = current.date()
    if is_nse_trading_day(current_date) and current.time() >= EOD_READY:
        return current_date
    return previous_trading_day(current_date)


def parse_reference_date(value: object) -> date | None:
    """Parse a prediction reference date without assuming timezone semantics."""
    if value is None:
        return None
    try:
        # Model date fields are daily-session labels; preserve the written date
        # instead of converting midnight UTC to a potentially different timezone.
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None
