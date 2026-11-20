"""Cloud Run Job entrypoint for the native five-minute history backfill.

The Cloud Run service keeps its HTTP ``/backfill`` endpoint. This wrapper is
only used by the ``native-5m-backfill`` job, which has no K_SERVICE value and
therefore needs its own provider-auth allowance in ``indstocks_client``.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import indstocks_backfill_all_nse_5m_history as runner

IST = ZoneInfo("Asia/Kolkata")
JOB_NAME = "native-5m-backfill"
STOP_AT_IST = (8, 30)
NEXT_WINDOW_IST = (16, 0)


class BackfillStop(Exception):
    """Raised by the request guard to stop cleanly and resume from checkpoints."""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _guard_request_slot() -> None:
    """Stop before any provider call once the off-hours window has ended.

    Checkpoints for completed windows are already on Drive, so the next run
    resumes instead of re-fetching. This also guarantees the job never issues
    INDstocks history requests during the 09:15-15:59 NSE live window.
    """
    now = datetime.now(IST)
    current = (now.hour, now.minute)
    if STOP_AT_IST <= current < NEXT_WINDOW_IST:
        raise BackfillStop(
            f"off-hours window ended at {STOP_AT_IST[0]:02d}:{STOP_AT_IST[1]:02d} IST"
        )


def _inside_start_window(now: datetime, *, smoke: bool) -> bool:
    """Full runs start 16:00-08:29 IST; a manual smoke may run until 09:14."""
    minutes = now.hour * 60 + now.minute
    if minutes >= 16 * 60:
        return True
    return minutes < (9 * 60 + 15 if smoke else 8 * 60 + 30)


def _args(smoke: bool) -> list[str]:
    end_date = (os.getenv("BACKFILL_END_DATE") or "2026-11-18").strip()
    if smoke:
        # Three 7-day windows for five symbols: enough to prove job auth,
        # Drive writes, provider reads and the final symbol-parquet path.
        return [
            "--max-symbols=5",
            "--initial-calendar-days=14",
            "--calendar-days=14",
            "--sessions=500",
            "--minimum-usable-sessions=200",
            "--minimum-bars-per-session=50",
            f"--end-date={end_date}",
        ]
    max_symbols = int(os.getenv("BACKFILL_MAX_SYMBOLS") or "0")
    return [
        f"--max-symbols={max_symbols}",
        f"--end-date={end_date}",
    ]


def main() -> int:
    smoke = _truthy(os.getenv("BACKFILL_SMOKE"))
    now = datetime.now(IST)
    print(
        "BACKFILL_JOB_START="
        + repr(
            {
                "job": os.getenv("CLOUD_RUN_JOB", ""),
                "execution": os.getenv("CLOUD_RUN_EXECUTION", ""),
                "smoke": smoke,
                "ist": now.isoformat(),
            }
        ),
        flush=True,
    )
    if not _inside_start_window(now, smoke=smoke):
        print("BACKFILL_JOB_OUTSIDE_WINDOW", flush=True)
        return 3

    runner.REQUEST_GUARD = _guard_request_slot
    try:
        code = runner.main(_args(smoke))
    except BackfillStop as exc:
        print("BACKFILL_JOB_STOPPED_RESUMABLE " + str(exc), flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1

    if code == 2:
        # Below-200-session symbols are deliberately kept and flagged. That is
        # a data outcome, not a job failure, so the next run can top them up.
        print("BACKFILL_JOB_COMPLETED_WITH_GAPS", flush=True)
        return 0
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
