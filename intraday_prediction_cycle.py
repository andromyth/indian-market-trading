"""Pick the newest five-minute cycle that already has a persisted prediction.

The intraday desk must never invent a forecast. It also must not fail just
because prediction persistence trails quote capture by a slot or two. These
pure helpers choose the newest cycle that has both a stored quote snapshot and
a stored prediction file, and report how far behind it is.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

PREDICTION_STORED_STATUSES = frozenset({"STORED"})
DEFAULT_MAX_LOOKBACK_SLOTS = 6
FIVE_MINUTE_SECONDS = 300


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
