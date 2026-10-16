#!/usr/bin/env python3
"""Private Cloud Run endpoint for scheduled, unattended NSE intraday ticks.

Cloud Scheduler sends one authenticated POST to /tick per minute during NSE
cash hours. The in-process worker object and model cache are reused while the
Cloud Run instance is warm; duplicate requests in the same minute are ignored.
No browser, Streamlit process, order placement, or background CPU loop is used.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from autonomous_intraday_worker import AutonomousIntradayWorker, _clean_json, _log
from market_session_guard import IST

PORT = int(os.getenv("PORT", "8080"))
WORKER_ROOT = Path(os.getenv("AUTONOMOUS_INTRADAY_ROOT", "/tmp/autonomous_intraday"))
MAX_STOCKS = int(os.getenv("AUTONOMOUS_INTRADAY_MAX_STOCKS", "5"))
POLL_SECONDS = max(10, int(os.getenv("AUTONOMOUS_INTRADAY_INTERVAL_SECONDS", "60")))

TICK_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
STATE: dict[str, Any] = {
    "service": "indian-market-autonomous-intraday-worker",
    "service_started_at_utc": datetime.now(timezone.utc).isoformat(),
    "last_tick_at_utc": None,
    "last_tick_result": None,
    "last_error": None,
}

WORKER = AutonomousIntradayWorker(
    segment="service",
    interval_seconds=POLL_SECONDS,
    refresh_watchlist_seconds=900,
    flush_every=5,
    max_stocks=MAX_STOCKS,
    root=WORKER_ROOT,
)


def _record_tick_result(result: dict[str, Any]) -> None:
    status = str(result.get("status", "")).upper()
    capture = result.get("all_universe_capture") or {}
    capture_status = str(capture.get("status") or "NOT_RUN").upper()
    capture_error = capture.get("error")
    with STATE_LOCK:
        STATE["last_tick_at_utc"] = datetime.now(timezone.utc).isoformat()
        STATE["last_tick_result"] = _clean_json(result)
        STATE["all_universe_capture_status"] = capture_status
        STATE["all_universe_capture_last_result"] = _clean_json({
            key: capture.get(key)
            for key in (
                "cycle_minute_ist", "expected_symbol_count", "quoted_symbol_count",
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
        elif capture_status == "PARTIAL":
            STATE["all_universe_capture_warning"] = (
                f"Quote coverage incomplete: {capture.get('quoted_symbol_count', 0)}/"
                f"{capture.get('expected_symbol_count', 0)} NSE symbols"
            )
            STATE["last_error"] = STATE["all_universe_capture_warning"]
        elif capture_status == "NOT_RUN":
            STATE["all_universe_capture_warning"] = "Full-universe capture did not run on this tick."

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


def _preflight() -> None:
    from persistent_store import drive_status

    status = drive_status(require_write=True)
    safe = {
        key: status.get(key)
        for key in (
            "configured", "reachable", "write_ready",
            "auth_method", "write_auth_method", "reason",
        )
    }
    _log("google_drive_preflight", **safe)
    if (
        not status.get("configured")
        or status.get("reachable") is False
        or status.get("write_ready") is False
    ):
        raise RuntimeError(
            "Google Drive read/write authentication is required for unattended operation: "
            + str(status.get("reason", "preflight failed"))
        )


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
        if self.path not in {"/", "/healthz", "/readyz"}:
            self.send_error(404)
            return
        status = _public_status()
        error = status.get("last_error")
        capture_status = status.get("all_universe_capture_status", "NOT_RUN")
        capture_warning = status.get("all_universe_capture_warning")
        payload = {
            "status": "degraded" if (error or capture_status not in {"OK", "NOT_RUN"} or capture_warning) else "ok",
            "worker_status": status.get("worker_status", {}).get("run_status"),
            "last_tick_at_utc": status.get("last_tick_at_utc"),
            "last_error": error,
            "all_universe_capture_status": capture_status,
            "all_universe_capture": status.get("all_universe_capture_last_result"),
            "all_universe_capture_warning": capture_warning,
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
            # Drain the small scheduler payload, if any.
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                length = 0
            if length:
                self.rfile.read(min(length, 4096))

            result = WORKER.tick_once()
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
    _preflight()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), WorkerHandler)
    server.daemon_threads = True
    _log(
        "cloud_run_worker_api_started",
        port=PORT,
        worker_root=str(WORKER_ROOT),
        tick_endpoint="/tick",
        health_endpoint="/healthz",
        note="Cloud Scheduler calls /tick; service is private and must not be made public.",
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        _log("cloud_run_worker_api_stopped", worker_status=WORKER.run_status)


if __name__ == "__main__":
    main()
