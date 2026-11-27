#!/usr/bin/env python3
"""Private Cloud Run endpoint for scheduled, unattended NSE intraday ticks.

Cloud Scheduler sends one authenticated POST to /tick every five minutes during
NSE cash hours. The in-process worker and pooled-model cache are reused while
the Cloud Run instance is warm; duplicate requests in the same minute are ignored.
No browser, Streamlit process, order placement, or background CPU loop is used.
"""
from __future__ import annotations

import json
import os
import resource
import threading
import time
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from autonomous_intraday_worker import (
    AutonomousIntradayWorker,
    _clean_json,
    _log,
    all_nse_capture_slot,
)
from all_nse_minute_prediction_capture import AllNSEMinutePredictor
from intraday_prediction_cycle import slots_between
from market_session_guard import IST, is_nse_trading_day
from worker_tick_lease import acquire_lease

PORT = int(os.getenv("PORT", "8080"))
WORKER_ROOT = Path(os.getenv("AUTONOMOUS_INTRADAY_ROOT", "/tmp/autonomous_intraday"))
MAX_STOCKS = int(os.getenv("AUTONOMOUS_INTRADAY_MAX_STOCKS", "5"))
POLL_SECONDS = max(10, int(os.getenv("AUTONOMOUS_INTRADAY_INTERVAL_SECONDS", "300")))

TICK_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
STATE: dict[str, Any] = {
    "service": "indian-market-autonomous-intraday-worker",
    "service_started_at_utc": datetime.now(timezone.utc).isoformat(),
    "last_tick_at_utc": None,
    "last_tick_result": None,
    "last_error": None,
    "drive_preflight_status": "NOT_CHECKED",
    "drive_preflight": {"reason": "Drive startup preflight is pending."},
    "five_minute_backfill": {
        "status": "IDLE",
        "target_end_date": None,
        "started_at_utc": None,
        "updated_at_utc": None,
        "finished_at_utc": None,
        "progress": {},
        "error": None,
        "exit_code": None,
    },
}

WORKER = AutonomousIntradayWorker(
    segment="service",
    interval_seconds=POLL_SECONDS,
    refresh_watchlist_seconds=900,
    flush_every=5,
    max_stocks=MAX_STOCKS,
    root=WORKER_ROOT,
)

# Separate pooled ALL-NSE inference path; the existing news watchlist remains unchanged.
ALL_NSE_MINUTE_PREDICTOR = AllNSEMinutePredictor(WORKER_ROOT)
BACKFILL_START_LOCK = threading.Lock()
BACKFILL_THREAD: threading.Thread | None = None
BACKFILL_OPTIONS: dict[str, Any] = {}
BACKFILL_CONFIRMATION = "START_NATIVE_5M_BACKFILL"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# When enabled the Cloud Run tick persists ALL-NSE predictions for the slot it
# just captured instead of delegating inference to GitHub Actions. This keeps
# the five-minute slot -> prediction latency inside the session budget because
# the predictor, model cache and history are already warm in this process.
ALL_NSE_IN_WORKER_PREDICTION = _env_flag("ALL_NSE_IN_WORKER_PREDICTION", False)
ALL_NSE_IN_WORKER_DEADLINE_SECONDS = max(
    30.0,
    float(os.getenv("ALL_NSE_IN_WORKER_PREDICTION_DEADLINE_SECONDS") or 180),
)
# Backlog drain: on every tick, predict older same-session slots whose quote
# snapshot exists but whose prediction was never stored (timeout, error, or a
# tick skipped by a scheduler collision). This is the in-session "never lose a
# slot" guarantee; GitHub remains the post-close audit/backstop.
ALL_NSE_BACKLOG_DRAIN = _env_flag("ALL_NSE_BACKLOG_DRAIN", True)
ALL_NSE_BACKLOG_MAX_SLOTS = max(0, int(os.getenv("ALL_NSE_BACKLOG_MAX_SLOTS") or 12))
ALL_NSE_BACKLOG_DEADLINE_SECONDS = max(
    5.0, float(os.getenv("ALL_NSE_BACKLOG_DEADLINE_SECONDS") or 90)
)
# Optional per-slot lease (module worker_tick_lease.py). It only prevents two
# instances from double-ticking the same Drive slot. It does NOT make a second
# instance safe: the INDstocks API permits one active token/session per account,
# so --max-instances must stay at 1 and this flag must stay disabled (0).
ALL_NSE_TICK_LEASE = _env_flag("ALL_NSE_TICK_LEASE", False)
ALL_NSE_TICK_LEASE_TTL_SECONDS = max(
    30, int(os.getenv("ALL_NSE_TICK_LEASE_TTL_SECONDS") or 120)
)
PREDICTION_PENDING = "PREDICTION_PENDING"
WORKER_INSTANCE_ID = f"{os.getenv('HOSTNAME', 'worker')}-{os.getpid()}"


def _update_backfill_status(**fields: Any) -> None:
    with STATE_LOCK:
        current = dict(STATE.get("five_minute_backfill") or {})
        current.update(fields)
        current["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        current["last_status_update_at_utc"] = current["updated_at_utc"]
        STATE["five_minute_backfill"] = current


def _backfill_worker_heartbeat(**fields: Any) -> None:
    heartbeat = datetime.now(timezone.utc).isoformat()
    _update_backfill_status(
        last_heartbeat_at_utc=heartbeat,
        last_worker_heartbeat_at_utc=heartbeat,
        **fields,
    )


# Per-run counters of distinct backfill artifacts. These are not log-event
# counts: each counter increments only on the one stage that proves that artifact.
_BACKFILL_COUNTER_STAGES = {
    "PROVIDER_REQUEST_STARTED": "api_requests_attempted",
    "PROVIDER_RESPONSE": "api_requests_successful",
    "PROVIDER_REQUEST_FAILED": "api_requests_failed",
    "CHECKPOINT_DOWNLOADED": "checkpoints_downloaded",
    "CHECKPOINT_PERSISTED": "checkpoints_written",
    "RESUMED_CHECKPOINT": "checkpoints_reused",
    "CHECKPOINT_REFETCH_INVALID": "checkpoints_invalid",
    "SYMBOL_PARQUET_PERSISTED": "symbol_files_written",
}
_BACKFILL_COUNTERS: dict[str, int] = {}
_BACKFILL_COUNTER_LOCK = threading.Lock()


def _reset_backfill_counters() -> None:
    with _BACKFILL_COUNTER_LOCK:
        _BACKFILL_COUNTERS.clear()


def _count_backfill_stage(stage: str) -> dict[str, int]:
    name = _BACKFILL_COUNTER_STAGES.get(stage)
    with _BACKFILL_COUNTER_LOCK:
        if name:
            _BACKFILL_COUNTERS[name] = _BACKFILL_COUNTERS.get(name, 0) + 1
        return dict(_BACKFILL_COUNTERS)


def _backfill_progress(progress: dict[str, Any]) -> None:
    fields: dict[str, Any] = {
        "progress": progress,
        "status": "RUNNING",
        "phase": "DOWNLOADING",
        "last_progress_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    stage = str(progress.get("stage") or "").upper()
    fields["counters"] = _count_backfill_stage(stage)
    if stage == "PROVIDER_REQUEST_STARTED":
        fields["last_provider_request_started_at_utc"] = datetime.now(timezone.utc).isoformat()
    elif stage == "PROVIDER_RESPONSE":
        fields["last_provider_response_at_utc"] = datetime.now(timezone.utc).isoformat()
        fields["last_provider_window_end"] = progress.get("window_end")
    elif stage == "CHECKPOINT_PERSISTED":
        fields["last_checkpoint_write_at_utc"] = datetime.now(timezone.utc).isoformat()
        fields["last_checkpoint_path"] = progress.get("checkpoint_path")
    elif stage == "SYMBOL_PARQUET_PERSISTED":
        fields["last_symbol_parquet_write_at_utc"] = datetime.now(timezone.utc).isoformat()
        fields["last_symbol_parquet_path"] = progress.get("symbol_parquet_path")
    elif stage == "RESUMED_CHECKPOINT":
        fields["last_checkpoint_reused_at_utc"] = datetime.now(timezone.utc).isoformat()
    _backfill_worker_heartbeat(**fields)
    _log("indstocks_5m_backfill_progress", **progress)


def _is_nse_live_priority_window(now: datetime) -> bool:
    """Pause historical backfill from market open until the 16:00 IST backfill window."""
    return (
        is_nse_trading_day(now.date())
        and (now.hour, now.minute) >= (9, 15)
        and (now.hour, now.minute) < (16, 0)
    )


def _backfill_start_allowed(now: datetime) -> bool:
    """Allow starts outside the live-priority window.

    On NSE trading days historical backfill must not compete with the live
    market, so starts are blocked between 09:15 and 16:00 IST.  Before the
    open and after the close are safe; the runner itself also pauses during
    the live window.  This keeps delayed overnight schedules (which often
    slip past midnight) working instead of rejecting them with HTTP 423.
    The recovery workflow deploys this guard before it requests a backfill.
    """
    return not _is_nse_live_priority_window(now)


def _parse_backfill_options(payload: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Validate optional /backfill controls sent by the scheduled workflow.

    ``replace_existing_history`` accepts auto/yes/no; "auto" leaves the choice to
    the runner, which replaces once when the history schema marker changes.
    """
    options: dict[str, Any] = {}
    raw_end_date = payload.get("end_date")
    if raw_end_date not in (None, ""):
        try:
            options["end_date"] = date.fromisoformat(str(raw_end_date).strip()).isoformat()
        except (TypeError, ValueError):
            return {}, "BACKFILL_END_DATE_INVALID"
    raw_replace = payload.get("replace_existing_history")
    if raw_replace is not None:
        options["replace_existing_history"] = str(raw_replace).strip().lower() in {
            "1", "true", "yes", "on",
        }
    raw_symbols = payload.get("symbols")
    if raw_symbols not in (None, ""):
        values: list[str] = []
        if isinstance(raw_symbols, (list, tuple)):
            values = [str(value) for value in raw_symbols if str(value).strip()]
        else:
            values = [value.strip() for value in str(raw_symbols).split(",") if value.strip()]
        options["symbols"] = sorted({value.strip().upper() for value in values if value.strip()})
    return options, None


BACKFILL_MARKET_PAUSED = False


def _wait_for_backfill_request_slot() -> None:
    """Wait without historical API activity until the NSE live-priority window ends."""
    global BACKFILL_MARKET_PAUSED

    while _is_nse_live_priority_window(datetime.now(IST)):
        if not BACKFILL_MARKET_PAUSED:
            BACKFILL_MARKET_PAUSED = True
            _log("indstocks_5m_backfill_paused_for_live_market")
        # This heartbeat is local state only: no provider request or recovery check.
        _backfill_worker_heartbeat(status="PAUSED_FOR_LIVE_MARKET", phase="MARKET_HOURS_PAUSE")
        time.sleep(30)

    if BACKFILL_MARKET_PAUSED:
        BACKFILL_MARKET_PAUSED = False
        _update_backfill_status(status="RUNNING", phase="DOWNLOADING")
        _log("indstocks_5m_backfill_resumed_after_market")


def _run_five_minute_backfill() -> None:
    """Run in this process to share INDstocks TOTP token cache with live capture."""
    global BACKFILL_OPTIONS
    _reset_backfill_counters()
    _update_backfill_status(
        counters={},
        status="RUNNING",
        phase="DOWNLOADING",
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        target_end_date=(BACKFILL_OPTIONS.get("end_date") or datetime.now(IST).date().isoformat()),
        finished_at_utc=None,
        progress={},
        last_heartbeat_at_utc=datetime.now(timezone.utc).isoformat(),
        last_worker_heartbeat_at_utc=datetime.now(timezone.utc).isoformat(),
        last_progress_at_utc=None,
        last_provider_response_at_utc=None,
        last_checkpoint_write_at_utc=None,
        error=None,
        exit_code=None,
    )
    try:
        import indstocks_backfill_all_nse_5m_history as backfill
        from indstocks_client import configured
        if not configured():
            raise RuntimeError("INDstocks read-only credentials are not configured in Cloud Run.")
        backfill.REQUEST_GUARD = _wait_for_backfill_request_slot
        backfill.BACKFILL_PROGRESS_CALLBACK = _backfill_progress
        backfill.RUNTIME_OPTIONS = dict(BACKFILL_OPTIONS)
        # If a transient provider/network failure outlasts the per-request retry
        # budget, rerun the job from Drive checkpoints instead of requiring a
        # manual trigger. Existing windows are skipped after integrity checks.
        max_job_attempts = 3
        for job_attempt in range(1, max_job_attempts + 1):
            try:
                exit_code = backfill.main([])
                break
            except Exception as exc:
                message = str(exc).upper()
                retryable = any(
                    marker in message
                    for marker in (
                        "HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504",
                        "UNEXPECTED_EOF_WHILE_READING", "SSLEOFERROR",
                        "REMOTEENDCLOSEDERROR", "CONNECTION RESET", "CONNECTIONRESETERROR",
                        "CONNECTION ABORTED", "BROKEN PIPE", "CONNECTIONERROR",
                        "NETWORKEXCEPTION", "TIMEOUT",
                    )
                )
                if not retryable or job_attempt == max_job_attempts:
                    raise
                detail = f"{type(exc).__name__}: {str(exc)[:240]}"
                delay_seconds = 10 * job_attempt
                _update_backfill_status(
                    status="RUNNING",
                    phase="RETRY_WAIT",
                    error=f"Transient attempt {job_attempt}/{max_job_attempts} failed; resuming from Drive checkpoints in {delay_seconds}s: {detail}",
                )
                _log(
                    "indstocks_5m_backfill_retry_scheduled",
                    failed_attempt=job_attempt,
                    next_attempt=job_attempt + 1,
                    max_attempts=max_job_attempts,
                    delay_seconds=delay_seconds,
                    error=detail,
                )
                time.sleep(delay_seconds)
        _update_backfill_status(
            status="COMPLETED" if exit_code == 0 else "COMPLETED_WITH_GAPS",
            exit_code=exit_code,
            error=None,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _log("indstocks_5m_backfill_finished", status="COMPLETED" if exit_code == 0 else "COMPLETED_WITH_GAPS", exit_code=exit_code)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:240]}"
        _update_backfill_status(
            status="FAILED",
            error=detail,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _log("indstocks_5m_backfill_failed", error=detail)
    finally:
        try:
            import indstocks_backfill_all_nse_5m_history as backfill
            backfill.REQUEST_GUARD = None
            backfill.BACKFILL_PROGRESS_CALLBACK = None
            backfill.RUNTIME_OPTIONS = {"end_date": None, "replace_existing_history": False}
        except Exception:
            pass
        BACKFILL_OPTIONS = {}


def _run_all_nse_prediction_in_worker(capture: dict[str, Any]) -> dict[str, Any]:
    """Predict the just-persisted ALL-NSE slot in this process.

    Falls back to the GitHub Actions compiler when the in-process run is
    disabled, fails, or exceeds its deadline so a slot is never lost because
    of an inference error.
    """
    slot = capture.get("cycle_minute_ist")
    fallback = {
        "status": PREDICTION_PENDING,
        "cycle_minute_ist": slot,
        "source_snapshot_drive_path": capture.get("snapshot_drive_path"),
    }
    if not ALL_NSE_IN_WORKER_PREDICTION:
        fallback["status"] = "DELEGATED_TO_GITHUB_ACTIONS"
        fallback["reason"] = "ALL_NSE_IN_WORKER_PREDICTION is disabled."
        return fallback
    started = datetime.now(timezone.utc)
    outcome: dict[str, Any] = {}

    def _predict() -> None:
        try:
            outcome["result"] = ALL_NSE_MINUTE_PREDICTOR.run_for_capture(
                capture,
                worker_root=WORKER_ROOT,
                drive_only=False,
            )
        except Exception as exc:
            outcome["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"

    thread = threading.Thread(
        target=_predict,
        name="all-nse-in-worker-prediction",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=ALL_NSE_IN_WORKER_DEADLINE_SECONDS)
    elapsed = round((datetime.now(timezone.utc) - started).total_seconds(), 3)
    if thread.is_alive():
        _log(
            "all_nse_in_worker_prediction_timeout",
            cycle_minute_ist=slot,
            deadline_seconds=ALL_NSE_IN_WORKER_DEADLINE_SECONDS,
            elapsed_seconds=elapsed,
        )
        fallback.update({
            "prediction_latency_seconds": elapsed,
            "reason": (
                "In-worker prediction exceeded the "
                f"{ALL_NSE_IN_WORKER_DEADLINE_SECONDS:.0f}s deadline; "
                "queued for the backlog drain (GitHub remains the post-close backstop)."
            ),
        })
        return fallback
    if outcome.get("error"):
        _log(
            "all_nse_in_worker_prediction_failed",
            cycle_minute_ist=slot,
            error=outcome["error"],
            elapsed_seconds=elapsed,
        )
        fallback.update({
            "prediction_latency_seconds": elapsed,
            "error": outcome["error"],
            "reason": "In-worker prediction failed; queued for the backlog drain (GitHub remains the post-close backstop).",
        })
        return fallback
    result = dict(outcome.get("result") or {})
    result["prediction_mode"] = "CLOUD_RUN_IN_WORKER"
    result["prediction_latency_seconds"] = elapsed
    _log(
        "all_nse_in_worker_prediction_finished",
        cycle_minute_ist=slot,
        status=result.get("status"),
        prediction_count=result.get("prediction_count"),
        elapsed_seconds=elapsed,
    )
    return result


def _drive_lease_io() -> Any:
    """Build a Drive-backed lease adapter, importing Drive helpers lazily."""
    from drive_all_nse_prediction_compiler import _download, _find_remote
    from persistent_store import GoogleDriveStore

    store = GoogleDriveStore()
    scratch = WORKER_ROOT / "tick_lease"
    scratch.mkdir(parents=True, exist_ok=True)

    class _DriveLeaseIO:
        def read(self, remote_path: str) -> dict[str, Any] | None:
            item = _find_remote(store, remote_path)
            if not item:
                return None
            local = scratch / f"read_{os.getpid()}.json"
            _download(store, item["id"], local)
            try:
                return json.loads(local.read_text(encoding="utf-8"))
            except Exception:
                return None

        def write(self, remote_path: str, payload: dict[str, Any]) -> None:
            local = scratch / f"write_{os.getpid()}.json"
            local.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            store.upload_or_update(local, remote_path, mime_type="application/json")

    return _DriveLeaseIO()


def _acquire_tick_lease(slot_key: str) -> dict[str, Any]:
    """Best-effort per-slot lease; fails open if Drive is unavailable."""
    if not ALL_NSE_TICK_LEASE:
        return {"acquired": True, "reason": "lease disabled"}
    try:
        decision = acquire_lease(
            _drive_lease_io(),
            slot_key,
            WORKER_INSTANCE_ID,
            datetime.now(timezone.utc),
            ALL_NSE_TICK_LEASE_TTL_SECONDS,
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:180]}"
        _log("tick_lease_unavailable", slot_key=slot_key, error=detail)
        return {"acquired": True, "reason": "lease store unavailable; failing open", "error": detail}
    return {
        "acquired": decision.acquired,
        "reason": decision.reason,
        "existing_owner": decision.existing_owner,
    }


def _slot_key(value: Any) -> str | None:
    try:
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize(IST) if stamp.tzinfo is None else stamp.tz_convert(IST)
        return stamp.strftime("%Y%m%d_%H%M")
    except Exception:
        return None


def _count_missing_prediction_slots(store: Any, target: pd.Timestamp) -> dict[str, Any]:
    """Cheap today-level count of quote slots with no prediction manifest."""
    try:
        from drive_all_nse_prediction_compiler import _find_remote, _list_children

        quote_folder = _find_remote(
            store, f"production_data/all_universe_minute_snapshots/{target:%Y/%m/%d}"
        )
        prediction_folder = _find_remote(
            store, f"production_data/all_nse_minute_predictions/{target:%Y/%m/%d}"
        )

        def _names(folder: Any) -> set[str]:
            if not folder:
                return set()
            return {str(item.get("name") or "") for item in _list_children(store, folder["id"])}

        quote_names = {name for name in _names(quote_folder) if name.startswith("manifest_")}
        prediction_names = {name for name in _names(prediction_folder) if name.startswith("manifest_")}
        missing = sorted(quote_names - prediction_names)
        return {
            "quote_slots": len(quote_names),
            "prediction_slots": len(prediction_names),
            "prediction_slots_missing_today": len(missing),
            "missing_sample": missing[-6:],
        }
    except Exception as exc:
        return {
            "quote_slots": None,
            "prediction_slots": None,
            "prediction_slots_missing_today": None,
            "error": f"{type(exc).__name__}: {str(exc)[:180]}",
        }


def _drain_missing_prediction_slots(
    now: pd.Timestamp | datetime | None = None,
    max_slots: int | None = None,
    deadline_seconds: float | None = None,
) -> dict[str, Any]:
    """Predict older same-session slots whose prediction is not yet stored.

    Reuses ``_recovery_capture`` (the GitHub repair selector) so the worker and
    the post-close audit share one definition of "missing". Time-boxed so a
    large backlog can never push the current tick past its five-minute slot.
    """
    if not (ALL_NSE_IN_WORKER_PREDICTION and ALL_NSE_BACKLOG_DRAIN):
        return {"status": "DISABLED", "drained": 0, "attempted": 0, "results": []}
    stamp = pd.Timestamp.now(tz=IST) if now is None else pd.Timestamp(now)
    stamp = stamp.tz_localize(IST) if stamp.tzinfo is None else stamp.tz_convert(IST)
    target = stamp.normalize()
    limit = ALL_NSE_BACKLOG_MAX_SLOTS if max_slots is None else max(0, int(max_slots))
    budget = (
        ALL_NSE_BACKLOG_DEADLINE_SECONDS if deadline_seconds is None else float(deadline_seconds)
    )
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    drained = 0
    newest_stored: pd.Timestamp | None = None
    try:
        from drive_all_nse_prediction_compiler import _recovery_capture
        from persistent_store import GoogleDriveStore

        # _recovery_capture stages manifests/snapshots under /tmp/drive_all_nse.
        Path("/tmp/drive_all_nse").mkdir(parents=True, exist_ok=True)
        store = GoogleDriveStore()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:180]}"
        _log("backlog_drain_unavailable", error=detail)
        return {"status": "ERROR", "error": detail, "drained": 0, "attempted": 0, "results": []}
    # Cheap two-folder check first: on a healthy tick there is nothing to drain,
    # so never pay for the oldest-first manifest scan.
    counts = _count_missing_prediction_slots(store, target)
    missing = counts.get("prediction_slots_missing_today")
    if not isinstance(missing, int) or missing <= 0:
        report = {
            "status": "NO_BACKLOG" if counts.get("error") is None else "ERROR",
            "drained": 0,
            "attempted": 0,
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "newest_stored_cycle_minute_ist": None,
            "results": [],
        }
        report.update(counts)
        return report
    while len(results) < limit and (time.monotonic() - started) < budget:
        try:
            selection = _recovery_capture(store, target)
        except RuntimeError as exc:
            if "No quote snapshot folder" in str(exc):
                break
            detail = f"{type(exc).__name__}: {str(exc)[:180]}"
            _log("backlog_drain_scan_failed", error=detail)
            return {
                "status": "ERROR", "error": detail, "drained": drained,
                "attempted": len(results), "results": results,
            }
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:180]}"
            _log("backlog_drain_scan_failed", error=detail)
            return {
                "status": "ERROR", "error": detail, "drained": drained,
                "attempted": len(results), "results": results,
            }
        if selection is None:
            break
        manifest, _quotes, minute = selection
        try:
            # drive_only=True loads the persisted history archive for that slot,
            # so a backfilled prediction is point-in-time correct instead of
            # borrowing the warm live predictor's newer bars.
            result = ALL_NSE_MINUTE_PREDICTOR.run_for_capture(
                manifest, worker_root=WORKER_ROOT, drive_only=True
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {str(exc)[:180]}"
            results.append({"cycle_minute_ist": minute.isoformat(), "status": "ERROR", "error": detail})
            break
        status = str(result.get("status") or "").upper()
        results.append({
            "cycle_minute_ist": minute.isoformat(),
            "status": status,
            "prediction_count": result.get("prediction_count"),
        })
        if status in {"STORED", "OK", "PARTIAL"}:
            drained += 1
            newest_stored = minute
        else:
            # A failed slot is not retried in a tight loop; the next tick or the
            # post-close audit will pick it up again.
            break
    report = {
        "status": "OK" if results else "NO_BACKLOG",
        "drained": drained,
        "attempted": len(results),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "newest_stored_cycle_minute_ist": (
            newest_stored.isoformat() if newest_stored is not None else None
        ),
        "results": results[-6:],
    }
    report.update(counts)
    if isinstance(missing, int):
        report["prediction_slots_missing_today"] = max(0, missing - drained)
    if results:
        _log("backlog_drain_result", **report)
    return report


def _prediction_metrics(
    capture: dict[str, Any],
    prediction: dict[str, Any],
    drain: dict[str, Any] | None,
    timings: dict[str, Any],
) -> dict[str, Any]:
    capture_minute = capture.get("cycle_minute_ist")
    stored_minute = None
    if str(prediction.get("status") or "").upper() in {"STORED", "OK", "PARTIAL"}:
        stored_minute = prediction.get("cycle_minute_ist") or capture_minute
    elif drain and drain.get("newest_stored_cycle_minute_ist"):
        stored_minute = drain.get("newest_stored_cycle_minute_ist")
    lag = None
    capture_key = _slot_key(capture_minute) if capture_minute else None
    stored_key = _slot_key(stored_minute) if stored_minute else None
    if capture_key and stored_key:
        lag = slots_between(capture_key, stored_key)
    return {
        "tick_duration_seconds": timings.get("tick_duration_seconds"),
        "capture_seconds": timings.get("capture_seconds"),
        "prediction_seconds": timings.get("prediction_seconds"),
        "prediction_lag_slots": lag,
        "prediction_slots_missing_today": (drain or {}).get("prediction_slots_missing_today"),
    }


def _prepare_all_nse_history_before_capture(
    now: pd.Timestamp | datetime | None = None,
) -> dict[str, Any]:
    """Start same-slot history refresh before the slow full-universe quote fetch."""
    stamp = pd.Timestamp.now(tz=IST) if now is None else pd.Timestamp(now)
    stamp = stamp.tz_localize(IST) if stamp.tzinfo is None else stamp.tz_convert(IST)
    minute = stamp.floor("min")
    ALL_NSE_MINUTE_PREDICTOR.start()
    if all_nse_capture_slot(minute) is None:
        return {"status": "NOT_SCHEDULED", "minute": minute.isoformat()}
    try:
        result = ALL_NSE_MINUTE_PREDICTOR._schedule_current_session_history_refresh(minute)
        _log(
            "all_nse_pre_capture_history_refresh",
            minute=minute.isoformat(),
            refresh_status=result.get("status"),
            updated_symbols=result.get("updated_symbols"),
            pending_symbol_count=result.get("pending_symbol_count"),
            failed_batches=result.get("failed_batches"),
        )
        return result
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:180]}"
        _log(
            "all_nse_pre_capture_history_refresh_failed",
            minute=minute.isoformat(),
            error=detail,
        )
        # A failed refresh is surfaced, but must not block quote capture/write.
        return {"status": "ERROR", "minute": minute.isoformat(), "error": detail}


def _resource_usage() -> dict[str, Any]:
    """Peak RSS (MB) and process CPU seconds for cost right-sizing."""
    peak_rss_mb = None
    try:
        peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:  # pragma: no cover - getrusage is POSIX-only
        pass
    return {"peak_rss_mb": peak_rss_mb, "cpu_seconds": round(time.process_time(), 1)}


def _record_tick_result(result: dict[str, Any]) -> None:
    status = str(result.get("status", "")).upper()
    capture = result.get("all_universe_capture") or {}
    capture_status = str(capture.get("status") or "NOT_RUN").upper()
    capture_error = capture.get("error")
    all_nse = result.get("all_nse_minute_predictions") or {}
    all_nse_status = str(all_nse.get("status") or "NOT_RUN").upper()
    all_nse_error = all_nse.get("error")
    with STATE_LOCK:
        # Off-cadence/manual /tick requests still update the watchlist, but must
        # not erase the latest scheduled full-universe capture result.
        if capture_status == "NOT_SCHEDULED":
            capture = STATE.get("all_universe_capture_last_result") or {}
            capture_status = str(
                STATE.get("all_universe_capture_status") or capture.get("status") or "NOT_RUN"
            ).upper()
            capture_error = STATE.get("all_universe_capture_error")
        if capture.get("snapshot_persisted") is True:
            source_status = capture_status
            source_error = capture_error
            capture = {
                **capture,
                "source_snapshot_status": source_status,
                "source_snapshot_error": source_error,
                "status": "STORED",
                "error": None,
            }
            capture_status = "STORED"
            capture_error = None
        if all_nse_status == "NOT_SCHEDULED":
            all_nse = STATE.get("all_nse_minute_prediction_last_result") or {}
            all_nse_status = str(
                STATE.get("all_nse_minute_prediction_status") or all_nse.get("status") or "NOT_RUN"
            ).upper()
            all_nse_error = STATE.get("all_nse_minute_prediction_error")
        previous_capture_issues = {
            STATE.get("all_universe_capture_error"),
            STATE.get("all_universe_capture_warning"),
        }
        STATE["last_tick_at_utc"] = datetime.now(timezone.utc).isoformat()
        STATE["last_tick_result"] = _clean_json(result)
        STATE["tick_metrics"] = _clean_json(result.get("metrics") or {})
        STATE["all_universe_capture_status"] = capture_status
        STATE["all_nse_minute_prediction_status"] = all_nse_status
        STATE["all_nse_minute_prediction_last_result"] = _clean_json({
            key: all_nse.get(key)
            for key in (
                "cycle_minute_ist", "expected_symbol_count", "row_count", "prediction_count",
                "missing_quote_count", "data_limited_count", "model_failure_count",
                "untrained_symbol_count", "model_artifact_verified",
                "history_ready", "history_status", "history_seed_coverage_symbols",
                "history_seed_failed_batches", "history_refresh_batch_count",
                "history_refresh_failed_batches", "remote_path", "manifest_drive_path",
                "eligible_symbol_count", "skipped_missing_quote_count",
                "source_snapshot_drive_path", "prediction_policy", "status", "error", "reason",
                "prediction_mode", "prediction_latency_seconds",
            )
            if key in all_nse
        })
        if all_nse_status not in {"OK", "STORED", "PARTIAL", "NOT_RUN", "NO_ELIGIBLE_SYMBOLS", "DELEGATED_TO_GITHUB_ACTIONS", "PREDICTION_PENDING"}:
            STATE["all_nse_minute_prediction_error"] = str(
                all_nse_error or f"ALL-NSE minute prediction status is {all_nse_status}"
            )
        else:
            STATE["all_nse_minute_prediction_error"] = None
        STATE["all_universe_capture_last_result"] = _clean_json({
            key: capture.get(key)
            for key in (
                "cycle_minute_ist", "expected_symbol_count", "row_count", "quoted_symbol_count",
                "missing_quote_count", "unmapped_symbol_count", "forecast_symbol_count",
                "matched_outcome_count", "snapshot_drive_path", "remote_snapshot_path",
                "snapshot_persisted", "source_snapshot_status", "source_snapshot_error",
                "status", "error",
            )
            if key in capture
        })
        if capture_status in {
            "ERROR", "PERSISTENCE_ERROR", "BLOCKED_PROVIDER_CREDENTIALS", "NO_ROWS"
        }:
            STATE["all_universe_capture_error"] = str(
                capture_error or f"Full-universe capture status is {capture_status}"
            )
        elif capture_status in {"OK", "STORED", "PARTIAL"}:
            STATE["all_universe_capture_error"] = None
            STATE["all_universe_capture_warning"] = None
            if STATE.get("last_error") in previous_capture_issues:
                STATE["last_error"] = None
        elif capture_status == "NOT_RUN":
            STATE["all_universe_capture_warning"] = (
                None if status == "MARKET_CLOSED"
                else "Full-universe capture did not run on this tick."
            )

        if status in {
            "ERROR", "PARTIAL", "PERSISTENCE_ERROR", "PERSISTENCE_PENDING"
        }:
            detail = result.get("error") or result.get("reason")
            if not detail:
                summary = result.get("summary") or {}
                errors = summary.get("errors") or []
                if errors:
                    first = errors[0]
                    if isinstance(first, dict):
                        detail = first.get("error") or first.get("reason")
                        if not detail:
                            detail = json.dumps(_clean_json(first), sort_keys=True)
                    else:
                        detail = str(first)
            STATE["last_error"] = str(detail or status)
        elif (
            (result.get("flush") or {}).get("status") == "OK"
            or (
                status == "MARKET_CLOSED"
                and result.get("run_status") in {"STOPPED", "IDLE", "STARTING"}
            )
        ):
            # Keep the health endpoint degraded until both the regular worker
            # and the independent all-universe persistence path recover.
            STATE["last_error"] = STATE.get("all_universe_capture_error")


def _public_status() -> dict[str, Any]:
    with STATE_LOCK:
        state = dict(STATE)
    return _clean_json({
        **state,
        "worker_status": WORKER._status_payload(WORKER.run_status),
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
        "nse_time_ist": pd.Timestamp.now(tz=IST).isoformat(),
        "tick_interval_seconds": POLL_SECONDS,
        "max_stocks": MAX_STOCKS,
    })


def _diagnostic_age_seconds(value: Any) -> float | None:
    try:
        stamp = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(stamp):
            return None
        return float((pd.Timestamp.now(tz="UTC") - stamp).total_seconds())
    except Exception:
        return None


def _capture_diagnostic(path: str) -> tuple[int, dict[str, Any]]:
    """Report persistence and per-slot prediction outcomes; coverage is informational."""
    state = _public_status()
    now_utc = datetime.now(timezone.utc)

    if path == "/capturez":
        item = state.get("all_universe_capture_last_result") or {}
        drive_path = item.get("snapshot_drive_path") or item.get("remote_snapshot_path")
        minute = item.get("cycle_minute_ist")
        row_count = int(item.get("row_count") or 0)
        age = _diagnostic_age_seconds(minute)
        stored = bool(drive_path and row_count > 0)
        payload = {
            "status": "ok" if stored else "degraded",
            "probe": "all_nse_quote_capture",
            "checked_at_utc": now_utc.isoformat(),
            "cycle_minute_ist": minute,
            "age_seconds": None if age is None else round(age, 1),
            "expected_symbol_count": item.get("expected_symbol_count"),
            "row_count": row_count,
            "quoted_symbol_count": item.get("quoted_symbol_count"),
            "missing_quote_count": item.get("missing_quote_count"),
            "quote_coverage_is_informational": True,
            "drive_path": drive_path,
            "reasons": [] if stored else ["No persisted same-slot quote snapshot is recorded."],
        }
        return (200 if stored else 503), payload

    if path == "/predictionz":
        item = state.get("all_nse_minute_prediction_last_result") or {}
        drive_path = item.get("remote_path")
        manifest_path = item.get("manifest_drive_path")
        minute = item.get("cycle_minute_ist")
        row_count = int(item.get("row_count") or 0)
        predicted = int(item.get("prediction_count") or 0)
        eligible = int(item.get("eligible_symbol_count") or 0)
        item_status = str(item.get("status") or state.get("all_nse_minute_prediction_status") or "NOT_RUN").upper()
        age = _diagnostic_age_seconds(minute)

        if item_status == "DELEGATED_TO_GITHUB_ACTIONS":
            payload = {
                "status": "delegated",
                "probe": "all_nse_minute_prediction_actual",
                "checked_at_utc": now_utc.isoformat(),
                "cycle_minute_ist": minute,
                "source_snapshot_drive_path": item.get("source_snapshot_drive_path"),
                "reason": "GitHub Actions owns ALL-NSE inference and prediction persistence.",
            }
            return 202, payload

        if item_status == "PREDICTION_PENDING":
            metrics = state.get("tick_metrics") or {}
            payload = {
                "status": "pending",
                "probe": "all_nse_minute_prediction_actual",
                "checked_at_utc": now_utc.isoformat(),
                "cycle_minute_ist": minute,
                "age_seconds": None if age is None else round(age, 1),
                "source_snapshot_drive_path": item.get("source_snapshot_drive_path"),
                "prediction_lag_slots": metrics.get("prediction_lag_slots"),
                "prediction_slots_missing_today": metrics.get("prediction_slots_missing_today"),
                "reason": "This slot is queued for the in-session backlog drain; GitHub remains the post-close backstop.",
                "reasons": [],
            }
            return 202, payload

        # No valid quote is an expected no-op for that slot, not a global
        # coverage failure. The stored source snapshot remains the audit record.
        no_data = item_status == "NO_ELIGIBLE_SYMBOLS" or (
            eligible == 0 and predicted == 0 and not drive_path
        )
        if no_data:
            payload = {
                "status": "no_data",
                "probe": "all_nse_minute_prediction_actual",
                "checked_at_utc": now_utc.isoformat(),
                "cycle_minute_ist": minute,
                "age_seconds": None if age is None else round(age, 1),
                "expected_symbol_count": item.get("expected_symbol_count"),
                "row_count": row_count,
                "eligible_symbol_count": eligible,
                "prediction_count": 0,
                "source_snapshot_drive_path": item.get("source_snapshot_drive_path"),
                "reason": "No individually valid stock quote existed in this slot; no prediction was generated.",
                "reasons": [],
            }
            return 200, payload

        stored = bool(drive_path and manifest_path and row_count > 0)
        has_prediction = predicted > 0
        reasons = []
        if not stored:
            reasons.append("No persisted prediction artifact and manifest are recorded.")
        elif eligible > 0 and not has_prediction:
            reasons.append("Stock quotes were eligible, but the model emitted no predictions.")
        payload = {
            "status": "ok" if stored and has_prediction else "degraded",
            "probe": "all_nse_minute_prediction_actual",
            "checked_at_utc": now_utc.isoformat(),
            "cycle_minute_ist": minute,
            "age_seconds": None if age is None else round(age, 1),
            "expected_symbol_count": item.get("expected_symbol_count"),
            "snapshot_symbol_count": item.get("snapshot_symbol_count"),
            "row_count": row_count,
            "eligible_symbol_count": eligible,
            "skipped_missing_quote_count": item.get("skipped_missing_quote_count", item.get("missing_quote_count")),
            "prediction_count": predicted,
            "prediction_coverage_pct": item.get("prediction_coverage_pct"),
            "prediction_coverage_is_informational": True,
            "history_ready": item.get("history_ready"),
            "drive_path": drive_path,
            "manifest_drive_path": manifest_path,
            "prediction_lag_slots": (state.get("tick_metrics") or {}).get("prediction_lag_slots"),
            "prediction_slots_missing_today": (state.get("tick_metrics") or {}).get("prediction_slots_missing_today"),
            "reasons": reasons,
        }
        return (200 if stored and has_prediction else 503), payload

    return 404, {"status": "not_found"}




def _serving_model_api() -> tuple[Any, str]:
    """The model API the capture actually binds, in the same order it tries them.

    The capture prefers the nightly ``models/to_close`` champion and only falls back to the legacy
    pooled bundle. Validating the legacy bundle here blocked deploys on an artifact that serving never
    reads, so mirror the capture: champion first, legacy second.
    """
    mode = os.getenv("ALL_NSE_MODEL_API", "champion").strip().lower()
    if mode not in {"legacy", "pooled", "all_nse"}:
        try:
            from persistent_store import gdrive_config

            if gdrive_config().get("configured"):
                import champion_model_api as champion
                import daily_cycle

                champion.configure(daily_cycle.make_drive_store())
                return champion, "champion"
        except Exception:  # noqa: BLE001 - fall through to the legacy bundle
            pass
    import all_nse_intraday_model as legacy

    return legacy, "legacy"


def _model_bundle_diagnostic() -> tuple[int, dict[str, Any]]:
    """Validate the five-minute model serving uses, plus Drive writes, with the runtime identity."""
    checked_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "probe": "five_minute_model_bundle",
        "checked_at_utc": checked_at,
        "status": "degraded",
        "checks": {},
    }
    try:
        from persistent_store import drive_status

        model, model_api = _serving_model_api()
        status = model.model_status()
        contract = model.model_target_contract()
        drive = drive_status(require_write=True)
        runtime_features = model.active_feature_names()
        if model_api == "champion":
            # champion_model_api reports the contract itself; there is no on-disk manifest document.
            manifest_features = list(status.get("artifact_features") or status.get("features") or [])
            manifest = {
                "bar_interval_minutes": contract["bar_interval_minutes"],
                "target": status.get("artifact_target") or status.get("target"),
                "prediction_horizon": status.get("prediction_horizon"),
                "feature_count": status.get("artifact_feature_count") or len(manifest_features),
                "features": manifest_features,
                "training_symbol_count": len(status.get("training_symbols") or []),
            }
        else:
            manifest = model._manifest()
            manifest_features = list(manifest.get("features") or [])
        features_match = manifest_features == runtime_features
        checks = {
            "model_api": model_api,
            "model_status": status.get("status"),
            "model_version": status.get("model_version"),
            "trained_through": status.get("trained_through"),
            "serving_mode": status.get("serving_mode"),
            "serving_reason": status.get("serving_reason"),
            "bar_interval_minutes": manifest.get("bar_interval_minutes"),
            "target": manifest.get("target"),
            "prediction_horizon": manifest.get("prediction_horizon"),
            "feature_count_manifest": manifest.get("feature_count"),
            "feature_count_runtime": len(runtime_features),
            "feature_list_matches_runtime": features_match,
            "probability_calibration_method": status.get("probability_calibration_method"),
            "training_symbol_count": manifest.get("training_symbol_count"),
            "artifact_hashes_match": status.get("manifest_sha256_match") is True,
            "drive_configured": drive.get("configured") is True,
            "drive_reachable": drive.get("reachable") is True,
            "drive_write_ready": drive.get("write_ready") is True,
            "drive_auth_method": drive.get("write_auth_method") or drive.get("auth_method"),
        }
        payload["checks"] = checks
        valid = all([
            status.get("manifest_sha256_match") is True,
            manifest.get("bar_interval_minutes") == 5,
            manifest.get("target") == contract["target"],
            manifest.get("prediction_horizon") == contract["prediction_horizon"],
            manifest.get("feature_count") == len(runtime_features),
            features_match,
            int(manifest.get("training_symbol_count") or 0) > 0,
            drive.get("configured") is True,
            drive.get("reachable") is True,
            drive.get("write_ready") is True,
        ])
        if valid:
            # Verify deserialization too; hashes alone do not prove joblib.load works.
            model._models()
            payload["status"] = "ok"
            return 200, payload

        payload["reason"] = (
            f"{model_api} five-minute model contract, artifact hashes, or Drive write readiness failed."
        )
        return 503, payload
    except Exception as exc:
        # Safe diagnostic only: never return tokens, OAuth JSON, root IDs, or full provider responses.
        payload["error_type"] = type(exc).__name__
        payload["reason"] = str(exc).replace("\\n", " ")[:240]
        return 503, payload


def _indstocks_provider_diagnostic() -> tuple[int, dict[str, Any]]:
    """Verify the worker's own read-only market-data credentials without exposing secrets."""
    from indstocks_client import (
        configured,
        historical_5m,
        latest_quote,
        profile,
        _load_instruments,
    )

    checked_at = datetime.now(timezone.utc).isoformat()
    payload: dict[str, Any] = {
        "probe": "indstocks_provider_health",
        "checked_at_utc": checked_at,
        "status": "degraded",
        "checks": {},
    }
    if not configured():
        payload["reason"] = "INDstocks credentials are not configured in the worker runtime."
        payload["checks"]["credentials_configured"] = False
        return 503, payload
    payload["checks"]["credentials_configured"] = True

    # Never return profile fields, access tokens, quote values, or provider response bodies.
    try:
        profile()
        payload["checks"]["profile_authentication"] = "PASS"
    except Exception as exc:
        payload["checks"]["profile_authentication"] = "FAIL"
        payload["failed_check"] = "profile_authentication"
        payload["error_type"] = type(exc).__name__
        return 503, payload

    try:
        instruments = _load_instruments("NSE")
        if instruments is None or instruments.empty:
            raise RuntimeError("empty instrument master")
        payload["checks"]["nse_instrument_master"] = "PASS"
        payload["nse_instrument_count"] = int(len(instruments))
    except Exception as exc:
        payload["checks"]["nse_instrument_master"] = "FAIL"
        payload["failed_check"] = "nse_instrument_master"
        payload["error_type"] = type(exc).__name__
        return 503, payload

    try:
        quote = latest_quote("RELIANCE.NS")
        price = quote.get("live_price", quote.get("ltp"))
        if price is None or float(price) <= 0:
            raise RuntimeError("no valid live price")
        payload["checks"]["single_equity_quote"] = "PASS"
    except Exception as exc:
        payload["checks"]["single_equity_quote"] = "FAIL"
        payload["failed_check"] = "single_equity_quote"
        payload["error_type"] = type(exc).__name__
        return 503, payload

    try:
        candles = historical_5m("RELIANCE.NS", days=3)
        if candles is None or candles.empty:
            raise RuntimeError("no five-minute history returned")
        payload["checks"]["five_minute_history"] = "AVAILABLE_OPTIONAL"
        payload["five_minute_history_rows"] = int(len(candles))
        payload["latest_five_minute_candle_at"] = pd.Timestamp(candles.index.max()).isoformat()
    except Exception as exc:
        # Current stored quotes can still be evaluated with quote-only features.
        # Missing history is per-symbol context, not a whole-provider failure.
        payload["checks"]["five_minute_history"] = "UNAVAILABLE_OPTIONAL"
        payload["history_warning_type"] = type(exc).__name__

    payload["status"] = "ok"
    return 200, payload

def _preflight() -> dict[str, Any]:
    """Check Drive outside the listener's startup path and retain safe diagnostics.

    A credential/network error must never prevent the HTTP server from binding
    PORT. Failed preflight keeps /healthz degraded and blocks /tick; the monitor
    retries so a repaired OAuth/Drive configuration can recover without a new
    revision.
    """
    try:
        from persistent_store import drive_status

        status = drive_status(require_write=True)
        if not isinstance(status, dict):
            status = {
                "configured": False,
                "reachable": False,
                "write_ready": False,
                "reason": "drive_status returned a non-object result",
            }
    except Exception as exc:
        status = {
            "configured": False,
            "reachable": False,
            "write_ready": False,
            "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
        }

    safe = {
        key: status.get(key)
        for key in (
            "configured", "reachable", "write_ready",
            "auth_method", "write_auth_method", "reason",
        )
    }
    ready = bool(status.get("configured")) and status.get("reachable") is not False and status.get("write_ready") is not False

    with STATE_LOCK:
        STATE["drive_preflight_status"] = "OK" if ready else "ERROR"
        STATE["drive_preflight"] = safe
        if not ready:
            STATE["last_error"] = (
                "Google Drive preflight failed: "
                + str(status.get("reason") or "Drive read/write access is not ready")
            )
        elif str(STATE.get("last_error") or "").startswith("Google Drive preflight failed:"):
            STATE["last_error"] = None

    _log("google_drive_preflight", status="OK" if ready else "ERROR", **safe)
    return {"ready": ready, **safe}


def _drive_preflight_monitor() -> None:
    """Retry Drive readiness in the background after the HTTP listener is live."""
    while True:
        result = _preflight()
        # Retry unhealthy credentials quickly, and periodically revalidate
        # healthy credentials so revoked/expired OAuth access blocks new ticks.
        time.sleep(60 if result.get("ready") else 20)


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "IntradayWorker/1.0"

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(_clean_json(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/backfillz":
            with STATE_LOCK:
                backfill = dict(STATE.get("five_minute_backfill") or {})
            now = datetime.now(timezone.utc)
            backfill["worker_thread_alive"] = bool(
                BACKFILL_THREAD is not None and BACKFILL_THREAD.is_alive()
            )
            for field, timestamp_key in (
                ("worker_heartbeat_age_seconds", "last_worker_heartbeat_at_utc"),
                ("provider_response_age_seconds", "last_provider_response_at_utc"),
                ("checkpoint_write_age_seconds", "last_checkpoint_write_at_utc"),
            ):
                stamp = pd.to_datetime(backfill.get(timestamp_key), errors="coerce", utc=True)
                backfill[field] = (
                    None if pd.isna(stamp)
                    else round((pd.Timestamp(now) - stamp).total_seconds(), 1)
                )
            self._send_json(200, {"status": backfill.get("status", "IDLE"), "five_minute_backfill": backfill})
            return
        if self.path == "/modelz":
            code, payload = _model_bundle_diagnostic()
            self._send_json(code, payload)
            return
        if self.path == "/providerz":
            code, payload = _indstocks_provider_diagnostic()
            self._send_json(code, payload)
            return
        if self.path in {"/capturez", "/predictionz"}:
            code, payload = _capture_diagnostic(self.path)
            self._send_json(code, payload)
            return
        if self.path not in {"/", "/healthz", "/readyz"}:
            self.send_error(404)
            return
        status = _public_status()
        error = status.get("last_error")
        capture_status = status.get("all_universe_capture_status", "NOT_RUN")
        capture_warning = status.get("all_universe_capture_warning")
        drive_preflight_status = status.get("drive_preflight_status", "NOT_CHECKED")
        metrics = status.get("tick_metrics") or {}
        now_ist = pd.Timestamp.now(tz=IST)
        in_session = bool(
            is_nse_trading_day(now_ist.date())
            and pd.Timestamp("09:15").time() <= now_ist.time() < pd.Timestamp("15:30").time()
        )
        lag_slots = metrics.get("prediction_lag_slots")
        prediction_stalled = bool(
            in_session and isinstance(lag_slots, (int, float)) and lag_slots > 3
        )
        payload = {
            "status": "degraded" if (
                error
                or capture_status not in {"OK", "STORED", "PARTIAL", "NOT_RUN"}
                or capture_warning
                or drive_preflight_status != "OK"
                or status.get("all_nse_minute_prediction_status", "NOT_RUN") not in {"OK", "STORED", "PARTIAL", "NOT_RUN", "NO_ELIGIBLE_SYMBOLS", "DELEGATED_TO_GITHUB_ACTIONS", "PREDICTION_PENDING"}
                or prediction_stalled
            ) else "ok",
            "worker_status": status.get("worker_status", {}).get("run_status"),
            "last_tick_at_utc": status.get("last_tick_at_utc"),
            "last_error": error,
            "drive_preflight_status": drive_preflight_status,
            "drive_preflight": status.get("drive_preflight"),
            "all_universe_capture_status": capture_status,
            "all_universe_capture": status.get("all_universe_capture_last_result"),
            "all_universe_capture_warning": capture_warning,
            "all_nse_minute_prediction_status": status.get("all_nse_minute_prediction_status", "NOT_RUN"),
            "all_nse_minute_prediction": status.get("all_nse_minute_prediction_last_result"),
            "all_nse_minute_prediction_error": status.get("all_nse_minute_prediction_error"),
            "nse_time_ist": status.get("nse_time_ist"),
            "tick_interval_seconds": status.get("tick_interval_seconds"),
            "tick_duration_seconds": metrics.get("tick_duration_seconds"),
            "capture_seconds": metrics.get("capture_seconds"),
            "prediction_seconds": metrics.get("prediction_seconds"),
            "prediction_lag_slots": lag_slots,
            "prediction_slots_missing_today": metrics.get("prediction_slots_missing_today"),
            "prediction_stalled": prediction_stalled,
        }
        self._send_json(503 if payload["status"] == "degraded" else 200, payload)

    def do_POST(self) -> None:  # noqa: N802
        global BACKFILL_OPTIONS, BACKFILL_THREAD
        route = self.path.rstrip("/")
        if route == "/backfill":
            try:
                content_length = int(self.headers.get("Content-Length", "0") or "0")
            except (TypeError, ValueError):
                content_length = 0
            if content_length <= 0 or content_length > 1024:
                self._send_json(400, {
                    "status": "BACKFILL_CONFIRMATION_REQUIRED",
                    "required_confirmation": BACKFILL_CONFIRMATION,
                })
                return
            try:
                request_payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                request_payload = {}
            if not isinstance(request_payload, dict) or request_payload.get("confirm") != BACKFILL_CONFIRMATION:
                self._send_json(400, {
                    "status": "BACKFILL_CONFIRMATION_REQUIRED",
                    "required_confirmation": BACKFILL_CONFIRMATION,
                })
                return
            options, option_error = _parse_backfill_options(request_payload)
            if option_error:
                self._send_json(400, {"status": option_error})
                return
            if not BACKFILL_START_LOCK.acquire(blocking=False):
                self._send_json(409, {"status": "BACKFILL_START_IN_PROGRESS"})
                return
            try:
                now_ist = datetime.now(IST)
                if not _backfill_start_allowed(now_ist):
                    self._send_json(423, {
                        "status": "BACKFILL_OUTSIDE_ALLOWED_WINDOW",
                        "nse_time_ist": now_ist.isoformat(),
                        "next_allowed_time_ist": "16:00 IST today",
                    })
                    return

                thread_alive = BACKFILL_THREAD is not None and BACKFILL_THREAD.is_alive()
                with STATE_LOCK:
                    backfill_state = dict(STATE.get("five_minute_backfill") or {})
                    backfill_status = str(backfill_state.get("status") or "IDLE")
                if thread_alive:
                    self._send_json(409, {"status": "BACKFILL_ALREADY_RUNNING", "five_minute_backfill": backfill_state})
                    return
                if backfill_status in {"RUNNING", "PAUSED_FOR_LIVE_MARKET"}:
                    _log(
                        "indstocks_5m_backfill_stale_worker_recovery",
                        stale_status=backfill_status,
                        last_heartbeat_at_utc=backfill_state.get("last_heartbeat_at_utc"),
                    )
                    _update_backfill_status(
                        status="RECOVERING_STALE_WORKER",
                        phase="CHECKPOINT_RECOVERY",
                        previous_status=backfill_status,
                    )
                preflight = _preflight()
                if not all(
                    preflight.get(key) is True
                    for key in ("ready", "configured", "reachable", "write_ready")
                ):
                    self._send_json(503, {
                        "status": "DRIVE_PREFLIGHT_FAILED",
                        "reason": preflight.get("reason") or "Drive read/write preflight must explicitly pass.",
                        "checks": {
                            key: preflight.get(key)
                            for key in ("configured", "reachable", "write_ready")
                        },
                    })
                    return
                from indstocks_client import configured
                if not configured():
                    self._send_json(503, {"status": "INDSTOCKS_CREDENTIALS_NOT_CONFIGURED"})
                    return
                target_end_date = options.get("end_date") or now_ist.date().isoformat()
                options["end_date"] = target_end_date
                BACKFILL_OPTIONS = options
                _update_backfill_status(target_end_date=target_end_date)
                BACKFILL_THREAD = threading.Thread(
                    target=_run_five_minute_backfill,
                    name="indstocks-native-5m-backfill",
                    daemon=True,
                )
                BACKFILL_THREAD.start()
                self._send_json(202, {
                    "status": "BACKFILL_STARTED",
                    "status_endpoint": "/backfillz",
                    "end_date": options.get("end_date"),
                    "replace_existing_history": options.get("replace_existing_history"),
                    "symbols": options.get("symbols"),
                    "note": "Runs in the Cloud Run process to reuse its INDstocks token cache; historical requests pause from 09:15 through 15:59 IST on NSE trading days and resume after 16:00, or run throughout non-trading days.",
                })
                return
            finally:
                BACKFILL_START_LOCK.release()
        if route != "/tick":
            self.send_error(404)
            return
        # Cloud Run concurrency is set to one. This lock is a second guard
        # against duplicate/overlapping requests within the same process.
        if not TICK_LOCK.acquire(blocking=False):
            self._send_json(409, {"status": "TICK_ALREADY_RUNNING"})
            return
        try:
            # Never create predictions that cannot be durably stored. If the
            # background preflight has not succeeded (or has lost readiness),
            # retry it once in this request and return an explicit diagnostic
            # rather than silently dropping prediction/actual rows.
            with STATE_LOCK:
                drive_ready = STATE.get("drive_preflight_status") == "OK"
            if not drive_ready:
                preflight = _preflight()
                if not preflight.get("ready"):
                    error = str(
                        preflight.get("reason")
                        or "Google Drive read/write access is not ready"
                    )
                    _log("cloud_run_tick_blocked_drive_preflight", reason=error)
                    self._send_json(503, {
                        "status": "DRIVE_PREFLIGHT_FAILED",
                        "error": error,
                        "drive_preflight": {
                            key: preflight.get(key)
                            for key in (
                                "configured", "reachable", "write_ready",
                                "auth_method", "write_auth_method", "reason",
                            )
                        },
                    })
                    return

            slot_key = pd.Timestamp.now(tz=IST).floor("5min").strftime("%Y%m%d_%H%M")
            lease = _acquire_tick_lease(slot_key)
            if not lease.get("acquired"):
                _log("cloud_run_tick_lease_held", slot_key=slot_key, **lease)
                self._send_json(200, {
                    "status": "TICK_LEASE_HELD",
                    "slot_key": slot_key,
                    "reason": lease.get("reason"),
                    "existing_owner": lease.get("existing_owner"),
                })
                return

            # Drain the small scheduler payload, if any.
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                length = 0
            if length:
                self.rfile.read(min(length, 4096))

            tick_started = time.monotonic()
            # Seed and refresh history before the full-universe quote collector
            # starts. This lets fresh one-minute candles arrive during the quote
            # fetch instead of starting refresh only after quote collection ends.
            history_prefetch = _prepare_all_nse_history_before_capture()
            capture_started = time.monotonic()
            result = WORKER.tick_once()
            capture_seconds = round(time.monotonic() - capture_started, 3)
            result["all_nse_pre_capture_history_refresh"] = history_prefetch
            # Cloud Run owns INDstocks collection and durable snapshot/history writes.
            # When ALL_NSE_IN_WORKER_PREDICTION is enabled it also owns the
            # same-slot inference so predictions land within one minute of the
            # five-minute boundary; otherwise GitHub Actions compiles the
            # persisted Drive artifacts as before.
            capture_result = result.get("all_universe_capture") or {}
            prediction_started = time.monotonic()
            if capture_result.get("status") == "NOT_SCHEDULED":
                result["all_nse_minute_predictions"] = {
                    "status": "NOT_SCHEDULED",
                    "reason": "No scheduled five-minute quote snapshot was captured.",
                }
            else:
                result["all_nse_minute_predictions"] = _run_all_nse_prediction_in_worker(
                    capture_result
                )
            prediction_seconds = round(time.monotonic() - prediction_started, 3)
            # Drain older missed slots (time-boxed) so a timeout, error, or a
            # scheduler collision can never permanently lose a slot.
            drain = _drain_missing_prediction_slots()
            result["backlog_drain"] = drain
            result["metrics"] = _prediction_metrics(
                capture_result,
                result.get("all_nse_minute_predictions") or {},
                drain,
                {
                    "tick_duration_seconds": round(time.monotonic() - tick_started, 3),
                    "capture_seconds": capture_seconds,
                    "prediction_seconds": prediction_seconds,
                },
            )
            _record_tick_result(result)
            _log("cloud_run_tick_result", **result, **_resource_usage())
            # A tick completed even if one symbol/source failed; errors are
            # persisted in the response/logs and should not rerun the same minute.
            capture_status = str(
                (result.get("all_universe_capture") or {}).get("status") or ""
            ).upper()
            prediction_status = str(
                (result.get("all_nse_minute_predictions") or {}).get("status") or ""
            ).upper()
            # Retry only durability/execution failures; partial quote coverage
            # is informational and never prevents other symbols being predicted.
            capture_snapshot_persisted = (
                (result.get("all_universe_capture") or {}).get("snapshot_persisted") is True
            )
            retryable_persistence_failure = (
                (capture_status in {"ERROR", "PERSISTENCE_ERROR"} and not capture_snapshot_persisted)
                or prediction_status in {
                    "ERROR", "PERSISTENCE_ERROR",
                    "BLOCKED_SNAPSHOT_CAPTURE", "BLOCKED_SNAPSHOT_VALIDATION",
                }
            )
            if retryable_persistence_failure:
                _log(
                    "cloud_run_tick_retryable_persistence_failure",
                    capture_status=capture_status,
                    prediction_status=prediction_status,
                )
            self._send_json(503 if retryable_persistence_failure else 200, {
                "status": result.get("status"),
                "minute": result.get("minute"),
                "cycle": result.get("cycle"),
                "predictions_written": (
                    (result.get("summary") or {}).get("predictions_written")
                ),
                "prediction_rows_valid": (
                    (result.get("summary") or {}).get("prediction_rows_valid")
                ),
                "prediction_errors": (
                    (result.get("summary") or {}).get("prediction_errors")
                ),
                "prediction_abstains": (
                    (result.get("summary") or {}).get("prediction_abstains")
                ),
                "watchlist_symbols": (
                    (result.get("summary") or {}).get("watchlist_symbols")
                ),
                "reason": result.get("reason"),
                "error": result.get("error"),
                "flush": result.get("flush"),
                "all_universe_capture": result.get("all_universe_capture"),
            })
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            _record_tick_result({"status": "ERROR", "error": error})
            _log("cloud_run_tick_exception", error=error)
            self._send_json(500, {"status": "ERROR", "error": error})
        finally:
            TICK_LOCK.release()

    def log_message(self, format: str, *args: Any) -> None:
        _log("http_request", message=(format % args), path=self.path)


def main() -> None:
    # Bind and start the HTTP listener before any Google API/token refresh.
    # Cloud Run's startup probe must be able to connect even when Drive OAuth
    # is expired or temporarily unreachable; /tick remains gated on write-ready.
    server = ThreadingHTTPServer(("0.0.0.0", PORT), WorkerHandler)
    server.daemon_threads = True
    monitor = threading.Thread(
        target=_drive_preflight_monitor,
        name="drive-preflight-monitor",
        daemon=True,
    )
    monitor.start()
    def prediction_runtime_prewarm() -> None:
        # This instance is a market-data collector only; GitHub loads the model
        # and compiles predictions from the persisted Drive inputs.
        ALL_NSE_MINUTE_PREDICTOR.start()
        _log(
            "all_nse_history_collector_started",
            status=ALL_NSE_MINUTE_PREDICTOR.status().get("status"),
            predictor_status=ALL_NSE_MINUTE_PREDICTOR.status(),
        )
    threading.Thread(
        target=prediction_runtime_prewarm,
        name="all-nse-prediction-prewarm",
        daemon=True,
    ).start()
    _log(
        "cloud_run_worker_api_started",
        port=PORT,
        worker_root=str(WORKER_ROOT),
        tick_endpoint="/tick",
        backfill_endpoint="/backfill",
        backfill_status_endpoint="/backfillz",
        health_endpoint="/healthz",
        startup_gate="Drive preflight runs asynchronously; /tick is blocked until Drive is write-ready.",
        note="Cloud Scheduler calls /tick; service is private and must not be made public.",
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        _log("cloud_run_worker_api_stopped", worker_status=WORKER.run_status)


if __name__ == "__main__":
    main()
