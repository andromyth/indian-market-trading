"""Cloud Run Job entrypoint for the native five-minute history backfill.

The Cloud Run service keeps its HTTP ``/backfill`` endpoint. This wrapper is
only used by the ``native-5m-backfill`` job, which has no K_SERVICE value and
therefore needs its own provider-auth allowance in ``indstocks_client``.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import indstocks_backfill_all_nse_5m_history as runner
import indstocks_client
from market_session_guard import is_nse_trading_day

IST = ZoneInfo("Asia/Kolkata")
JOB_NAME = "native-5m-backfill"
TOKEN_COOLDOWN_MAX_RETRIES = 5
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
    # On NSE non-trading days there is no live market to protect, so the job
    # may run continuously to finish the remaining batches.
    if not is_nse_trading_day(now.date()):
        return
    current = (now.hour, now.minute)
    if STOP_AT_IST <= current < NEXT_WINDOW_IST:
        raise BackfillStop(
            f"off-hours window ended at {STOP_AT_IST[0]:02d}:{STOP_AT_IST[1]:02d} IST"
        )


def _inside_start_window(now: datetime, *, smoke: bool) -> bool:
    """Full runs start 16:00-08:29 IST on trading days; all day on holidays.

    NSE non-trading days have no live snapshot to protect, so the backfill may
    run continuously to finish the remaining batches.
    """
    if not is_nse_trading_day(now.date()):
        return True
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
        # One recent session is enough to refresh the latest bars; the provider
        # already returns the full 500-session window, so a longer live-history
        # lookback only multiplies Drive reads across all 100 symbol batches.
        "--recent-session-lookback-days=1",
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
    last_exc = None
    for attempt in range(1, TOKEN_COOLDOWN_MAX_RETRIES + 1):
        try:
            code = runner.main(_args(smoke))
        except BackfillStop as exc:
            print("BACKFILL_JOB_STOPPED_RESUMABLE " + str(exc), flush=True)
            return 0
        except RuntimeError as exc:
            msg = str(exc)
            is_token_cooldown = (
                "token generation" in msg
                or "cooling down" in msg
                or "suppressed for" in msg
            )
            if is_token_cooldown and attempt < TOKEN_COOLDOWN_MAX_RETRIES:
                last_exc = exc
                wait_s = int(indstocks_client.TOKEN_GENERATION_FAILURE_COOLDOWN_SECONDS) + 5
                print(
                    "BACKFILL_TOKEN_COOLDOWN_RETRY attempt=%d/%d waiting %ds "
                    % (attempt, TOKEN_COOLDOWN_MAX_RETRIES, wait_s),
                    flush=True,
                )
                time.sleep(wait_s)
                continue
            traceback.print_exc()
            return 1
    if last_exc is not None:
        traceback.print_exc()
        print("BACKFILL_TOKEN_COOLDOWN_EXHAUSTED " + str(last_exc), flush=True)
        return 1

    if code == 2:
        # Below-200-session symbols are deliberately kept and flagged. That is
        # a data outcome, not a job failure, so the next run can top them up.
        print("BACKFILL_JOB_COMPLETED_WITH_GAPS", flush=True)
        return 0
    return int(code)


if __name__ == "__main__":
    sys.exit(main())
