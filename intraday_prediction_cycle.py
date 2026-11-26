"""Pick the newest five-minute cycle that already has a persisted prediction.

The intraday desk must never invent a forecast. It also must not fail just
because prediction persistence trails quote capture by a slot or two. These
pure helpers choose the newest cycle that has both a stored quote snapshot and
a stored prediction file, and report how far behind it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Iterable

from nse_trading_calendar import is_nse_trading_day

PREDICTION_STORED_STATUSES = frozenset({"STORED"})
DEFAULT_MAX_LOOKBACK_SLOTS = 6
FIVE_MINUTE_SECONDS = 300

# Session-aware freshness states for the intraday desk. Thresholds are counted
# in five-minute slots (not wall-clock minutes) so they stay aligned with the
# six-slot lookback used by ``select_prediction_complete_cycles``.
LIVE_OK = "LIVE_OK"
LIVE_DELAYED = "LIVE_DELAYED"
LIVE_STALE = "LIVE_STALE"
SESSION_FINAL = "SESSION_FINAL"
NO_CYCLE_FOR_SESSION = "NO_CYCLE_FOR_SESSION"

LIVE_OK_MAX_SLOTS_BEHIND = 3
LIVE_DELAYED_MAX_SLOTS_BEHIND = DEFAULT_MAX_LOOKBACK_SLOTS
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30)

DISPLAYABLE_STATUSES = frozenset({LIVE_OK, LIVE_DELAYED, SESSION_FINAL})


@dataclass(frozen=True)
class CycleFreshness:
    """Result of :func:`classify_cycle_status`."""

    status: str
    slots_behind: int

    @property
    def show_numbers(self) -> bool:
        return self.status in DISPLAYABLE_STATUSES

    @property
    def is_live(self) -> bool:
        return self.status in {LIVE_OK, LIVE_DELAYED}

    @property
    def is_delayed(self) -> bool:
        return self.status == LIVE_DELAYED

    @property
    def is_stale(self) -> bool:
        return self.status == LIVE_STALE

    @property
    def is_final(self) -> bool:
        return self.status == SESSION_FINAL

    @property
    def is_missing(self) -> bool:
        return self.status == NO_CYCLE_FOR_SESSION


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return value.date()
    except AttributeError:
        return None


def _as_time(value: Any) -> time | None:
    if isinstance(value, datetime):
        return value.time()
    try:
        return value.time()
    except AttributeError:
        return None


def classify_cycle_status(
    now_ist: Any,
    cycle_ts: Any,
    slots_behind: int,
    last_completed_session: date | None,
) -> CycleFreshness:
    """Classify the newest persisted prediction cycle for the intraday desk.

    ``now_ist`` is the current IST timestamp (``datetime``/``Timestamp``).
    ``cycle_ts`` is the timestamp of the newest fully persisted cycle (quote
    snapshot *and* prediction), or ``None`` when there is none. ``slots_behind``
    is the five-minute-slot gap between that cycle and the newest quote snapshot.
    ``last_completed_session`` is the most recent finished NSE session date.

    The classifier is pure: it never reads Drive, a clock, or config.
    """
    slots = max(0, int(slots_behind or 0))
    now_date = _as_date(now_ist)
    now_time = _as_time(now_ist)
    cycle_date = _as_date(cycle_ts)

    in_session = bool(
        now_date
        and is_nse_trading_day(now_date)
        and now_time is not None
        and SESSION_OPEN <= now_time < SESSION_CLOSE
    )

    if in_session:
        # A cycle from a previous day while the market is open means no fresh
        # cycle has been persisted for today's session yet.
        if cycle_date is None or cycle_date != now_date:
            return CycleFreshness(LIVE_STALE, slots)
        if slots <= LIVE_OK_MAX_SLOTS_BEHIND:
            return CycleFreshness(LIVE_OK, slots)
        if slots <= LIVE_DELAYED_MAX_SLOTS_BEHIND:
            return CycleFreshness(LIVE_DELAYED, slots)
        return CycleFreshness(LIVE_STALE, slots)

    # Outside the session the desk shows the last completed session's final
    # cycle, never a "stale" verdict for an intentionally frozen feed.
    if last_completed_session is None or cycle_date is None:
        return CycleFreshness(NO_CYCLE_FOR_SESSION, slots)
    if cycle_date == last_completed_session:
        return CycleFreshness(SESSION_FINAL, slots)
    if cycle_date < last_completed_session:
        return CycleFreshness(NO_CYCLE_FOR_SESSION, slots)
    return CycleFreshness(SESSION_FINAL, slots)


def _is_complete(candidate: dict[str, Any]) -> bool:
    return (
        bool(candidate.get("snapshot_stored"))
        and bool(candidate.get("prediction_file_stored"))
        and str(candidate.get("prediction_manifest_status") or "").strip().upper()
        in PREDICTION_STORED_STATUSES
    )


def select_prediction_complete_cycles(
    candidates: Iterable[dict[str, Any]],
    max_lookback_slots: int = DEFAULT_MAX_LOOKBACK_SLOTS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return complete cycles (newest first) and the minute keys skipped.

    ``candidates`` must be ordered newest first and describe stored quote
    snapshots. A cycle qualifies only when its ``predictions_<minute>.parquet``
    exists and its prediction manifest status is ``STORED``. The search is
    bounded to ``max_lookback_slots`` stored snapshots so a longer prediction
    outage surfaces as a diagnostic instead of silently serving a stale cycle.
    """
    limit = max(0, int(max_lookback_slots))
    window: list[dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.get("snapshot_stored"):
            continue
        window.append(candidate)
        if len(window) >= limit:
            break
    complete = [candidate for candidate in window if _is_complete(candidate)]
    skipped = [
        str(candidate.get("minute_key") or "")
        for candidate in window
        if not _is_complete(candidate)
    ]
    return complete, skipped


def slots_between(newest_minute_key: str | None, minute_key: str | None) -> int:
    """Five-minute slots between two ``YYYYMMDD_HHMM`` keys, 0 when unknown."""
    try:
        newest = datetime.strptime(str(newest_minute_key), "%Y%m%d_%H%M")
        chosen = datetime.strptime(str(minute_key), "%Y%m%d_%H%M")
    except (TypeError, ValueError):
        return 0
    return max(0, int((newest - chosen).total_seconds() // FIVE_MINUTE_SECONDS))
