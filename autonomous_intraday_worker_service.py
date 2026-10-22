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
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from autonomous_intraday_worker import AutonomousIntradayWorker, _clean_json, _log
from all_nse_minute_prediction_capture import AllNSEMinutePredictor
from market_session_guard import IST

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
        STATE["all_universe_capture_status"] = capture_status
        STATE["all_nse_minute_prediction_status"] = all_nse_status
        STATE["all_nse_minute_prediction_last_result"] = _clean_json({
            key: all_nse.get(key)
            for key in (
                "cycle_minute_ist", "expected_symbol_count", "row_count", "prediction_count",
                "missing_quote_count", "data_limited_count", "history_ready",
                "history_status", "remote_path", "manifest_drive_path",
                "quality_gate_pass", "status", "error",
            )
            if key in all_nse
        })
        if all_nse_status not in {"OK", "NOT_RUN"}:
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
        elif capture_status == "OK":
            STATE["all_universe_capture_error"] = None
            STATE["all_universe_capture_warning"] = None
            if STATE.get("last_error") in previous_capture_issues:
                STATE["last_error"] = None
        elif capture_status == "PARTIAL":
            STATE["all_universe_capture_error"] = None
            STATE["all_universe_capture_warning"] = (
                f"Quote coverage incomplete: {capture.get('quoted_symbol_count', 0)}/"
                f"{capture.get('expected_symbol_count', 0)} NSE symbols"
            )
            STATE["last_error"] = STATE["all_universe_capture_warning"]
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
    """Return a strict, machine-readable all-NSE capture readiness result."""
    state = _public_status()
    now_utc = datetime.now(timezone.utc).isoformat()
    if path == "/capturez":
        item = state.get("all_universe_capture_last_result") or {}
        prefix = "all_nse_quote_capture"
        expected = item.get("expected_symbol_count")
        row_count = item.get("row_count")
        valid = item.get("quoted_symbol_count")
        missing = item.get("missing_quote_count")
        drive_path = item.get("snapshot_drive_path") or item.get("remote_snapshot_path")
        minute = item.get("cycle_minute_ist")
        item_status = str(item.get("status") or "NOT_RUN").upper()
        age = _diagnostic_age_seconds(minute)
        reasons = []
        if state.get("drive_preflight_status") != "OK":
            reasons.append("Google Drive read/write preflight is not ready")
        if item_status != "OK":
            reasons.append(f"latest full-universe quote snapshot status is {item_status}")
        if age is None or age < 0 or age > 420:
            reasons.append("latest full-universe quote snapshot is missing or older than 420 seconds (5-minute cadence plus processing allowance)")
        if expected is None or int(expected or 0) <= 0:
            reasons.append("official NSE universe count is missing or zero")
        elif row_count is None or int(row_count) != int(expected):
            reasons.append("snapshot row count does not equal the expected NSE universe")
        if expected is not None and (valid is None or missing is None
                                     or int(valid) != int(expected) or int(missing) != 0):
            reasons.append("one or more NSE symbols lack a valid current quote")
        if not drive_path:
            reasons.append("confirmed Drive snapshot path is missing")
        payload = {
            "status": "ok" if not reasons else "degraded",
            "probe": prefix,
            "checked_at_utc": now_utc,
            "cycle_minute_ist": minute,
            "capture_status": item_status,
            "age_seconds": None if age is None else round(age, 1),
            "expected_symbol_count": expected,
            "row_count": row_count,
            "quoted_symbol_count": valid,
            "missing_quote_count": missing,
            "drive_path": drive_path,
            "reasons": reasons,
        }
    elif path == "/predictionz":
        item = state.get("all_nse_minute_prediction_last_result") or {}
        prefix = "all_nse_minute_prediction_actual"
        expected = item.get("expected_symbol_count")
        row_count = item.get("row_count")
        predicted = item.get("prediction_count")
        missing = item.get("missing_quote_count")
        limited = item.get("data_limited_count")
        minute = item.get("cycle_minute_ist")
        item_status = str(item.get("status") or state.get("all_nse_minute_prediction_status") or "NOT_RUN").upper()
        age = _diagnostic_age_seconds(minute)
        drive_path = item.get("remote_path")
        manifest_path = item.get("manifest_drive_path")
        reasons = []
        if state.get("drive_preflight_status") != "OK":
            reasons.append("Google Drive read/write preflight is not ready")
        if item_status != "OK":
            reasons.append(f"latest all-NSE prediction snapshot status is {item_status}")
        if age is None or age < 0 or age > 420:
            reasons.append("latest prediction snapshot is missing or older than 420 seconds (5-minute cadence plus processing allowance)")
        if expected is None or int(expected or 0) <= 0:
            reasons.append("expected NSE universe count is missing or zero")
        elif row_count is None or int(row_count) != int(expected):
            reasons.append("prediction row count does not equal the expected NSE universe")
        if expected is not None and (predicted is None or int(predicted) != int(expected)):
            reasons.append("not every expected NSE symbol has a model prediction")
        if missing is None or int(missing or 0) != 0:
            reasons.append("one or more symbols lack a valid current quote")
        if limited is None or int(limited or 0) != 0:
            reasons.append("one or more symbols are data-limited")
        if item.get("quality_gate_pass") is not True:
            reasons.append("prediction quality gate is not passing")
        if item.get("history_ready") is not True:
            reasons.append("historical feature data is not ready")
        if not drive_path or not manifest_path:
            reasons.append("confirmed Drive prediction or manifest path is missing")
        payload = {
            "status": "ok" if not reasons else "degraded",
            "probe": prefix,
            "checked_at_utc": now_utc,
            "cycle_minute_ist": minute,
            "capture_status": item_status,
            "age_seconds": None if age is None else round(age, 1),
            "expected_symbol_count": expected,
            "row_count": row_count,
            "prediction_count": predicted,
            "missing_quote_count": missing,
            "data_limited_count": limited,
            "quality_gate_pass": item.get("quality_gate_pass"),
            "history_ready": item.get("history_ready"),
            "drive_path": drive_path,
            "manifest_drive_path": manifest_path,
            "reasons": reasons,
        }
    else:
        return 404, {"status": "not_found"}
    return (200 if not reasons else 503), payload


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
        payload = {
            "status": "degraded" if (
                error
                or capture_status not in {"OK", "NOT_RUN"}
                or capture_warning
                or drive_preflight_status != "OK"
                or status.get("all_nse_minute_prediction_status", "NOT_RUN") not in {"OK", "NOT_RUN"}
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
        }
        self._send_json(503 if payload["status"] == "degraded" else 200, payload)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/tick":
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

            # Drain the small scheduler payload, if any.
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                length = 0
            if length:
                self.rfile.read(min(length, 4096))

            result = WORKER.tick_once()
            # Run pooled all-NSE inference only when the full-universe quote
            # capture actually ran on a scheduled five-minute slot.
            capture_result = result.get("all_universe_capture") or {}
            if capture_result.get("status") == "NOT_SCHEDULED":
                result["all_nse_minute_predictions"] = {
                    "status": "NOT_SCHEDULED",
                    "reason": "Full-NSE prediction snapshots are written on five-minute capture slots only.",
                }
            else:
                try:
                    prediction_capture = ALL_NSE_MINUTE_PREDICTOR.run_for_capture(
                        capture_result,
                        WORKER.root,
                    )
                    result["all_nse_minute_predictions"] = prediction_capture
                except Exception as prediction_exc:
                    prediction_error = f"{type(prediction_exc).__name__}: {str(prediction_exc)[:240]}"
                    result["all_nse_minute_predictions"] = {
                        "status": "ERROR",
                        "error": prediction_error,
                    }
                    _log("all_nse_minute_prediction_capture_failed", error=prediction_error)
            _record_tick_result(result)
            _log("cloud_run_tick_result", **result)
            # A tick completed even if one symbol/source failed; errors are
            # persisted in the response/logs and should not rerun the same minute.
            self._send_json(200, {
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
    _log(
        "cloud_run_worker_api_started",
        port=PORT,
        worker_root=str(WORKER_ROOT),
        tick_endpoint="/tick",
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
