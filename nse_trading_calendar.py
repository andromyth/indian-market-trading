"""Small, dependency-free NSE cash/equity trading calendar shared by workers and CI."""
from __future__ import annotations

from datetime import date

# 2026 NSE CM/equity holidays used by the existing ingestion layer.
HOLIDAYS = frozenset(
    date.fromisoformat(value)
    for value in (
        "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26",
        "2026-03-31", "2026-04-03", "2026-04-14", "2026-05-01",
        "2026-05-28", "2026-06-26", "2026-09-14", "2026-10-02",
        "2026-10-20", "2026-11-10", "2026-11-24", "2026-12-25",
    )
)


def is_nse_trading_day(value: date) -> bool:
    """Return True only for NSE weekdays not listed in the known holiday calendar."""
    return value.weekday() < 5 and value not in HOLIDAYS
