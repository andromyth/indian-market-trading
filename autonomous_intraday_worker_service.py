#!/usr/bin/env python3
"""Long-lived Cloud Run host for unattended NSE intraday collection/prediction.

A background thread calls the worker once per minute regardless of browser
traffic. Deploy with one minimum/maximum Cloud Run instance and CPU always
allocated. The HTTP server provides only health endpoints; no order endpoints
exist.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
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

STOP = threading.Event()
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


def _public_status() -> dict[str, Any]:
    with STATE_LOCK:
        state = dict(STATE)
    return _clean_json({
        **state,
        "worker_status": WORKER._status_payload(WORKER.run_status),
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
        "nse_time_ist": pd.Timestamp.now(tz=IST).isoformat(),
        "poll_interval_seconds": POLL_SECONDS,
        "max_stocks": MAX_STOCKS,
    })


def _worker_loop() -> None:
    _log(
        "cloud_run_worker_loop_started",
        poll_interval_seconds=POLL_SECONDS,
        worker_root=str(WORKER_ROOT),
        note="Polls independently of HTTP/browser traffic; skipped outside NSE session.",
    )
    last_bucket: str | None = None
    while not STOP.is_set():
        now = pd.Timestamp.now(tz=IST)
        bucket = now.floor("min").isoformat()
        if bucket != last_bucket:
            # Record the bucket before invoking the network-heavy cycle so an
            # exception cannot accidentally trigger repeated calls in one minute.
            last_bucket = bucket
            try:
                with TICK_LOCK:
                    result = WORKER.tick_once()
                with STATE_LOCK:
                    STATE["last_tick_at_utc"] = datetime.now(timezone.utc).isoformat()
                    STATE["last_tick_result"] = _clean_json(result)
                    result_status = str(result.get("status", "")).upper()
                    if result_status in {
                        "ERROR", "PARTIAL", "PERSISTENCE_ERROR", "PERSISTENCE_PENDING"
                    }:
                        STATE["last_error"] = str(
                            result.get("error") or result.get("reason") or result_status
                        )
                    elif (
                        (result.get("flush") or {}).get("status") == "OK"
                        or (
                            result_status == "MARKET_CLOSED"
                            and result.get("run_status") in {"STOPPED", "IDLE", "STARTING"}
                        )
                    ):
                        # Only a confirmed successful persistence flush clears a
                        # previous persistence alert; non-flush ticks don't hide it.
                        STATE["last_error"] = None
                _log("cloud_run_tick_result", **result)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                with STATE_LOCK:
                    STATE["last_tick_at_utc"] = datetime.now(timezone.utc).isoformat()
                    STATE["last_error"] = error
                    STATE["last_tick_result"] = {"status": "ERROR", "error": error}
                _log("cloud_run_tick_exception", error=error)

        # Wake up at least once a second to align on the next minute boundary.
        STOP.wait(1.0)

    try:
        if WORKER.run_status == "RUNNING":
            with TICK_LOCK:
                flush = WORKER._flush(final=True)
            _log("cloud_run_shutdown_flush", result=flush)
    except Exception as exc:
        _log("cloud_run_shutdown_flush_failed", error=f"{type(exc).__name__}: {exc}")


class HealthHandler(BaseHTTPRequestHandler):
    server_version = "IntradayWorkerHealth/1.0"

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in {"/", "/healthz", "/readyz"}:
            self.send_error(404)
            return
        status = _public_status()
        ready = not status.get("last_error")
        payload = {
            "status": "ok" if ready else "degraded",
            "worker_status": status.get("worker_status", {}).get("run_status"),
            "last_tick_at_utc": status.get("last_tick_at_utc"),
            "last_error": status.get("last_error"),
            "nse_time_ist": status.get("nse_time_ist"),
            "poll_interval_seconds": status.get("poll_interval_seconds"),
        }
        body = json.dumps(_clean_json(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(200 if ready else 503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep platform logs structured and compact; never log headers or secrets.
        _log("health_request", message=(format % args), path=self.path)


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


def main() -> None:
    _preflight()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), HealthHandler)
    server.daemon_threads = True
    thread = threading.Thread(
        target=_worker_loop,
        name="autonomous-intraday-worker",
        daemon=True,
    )
    thread.start()

    def request_shutdown(signum, frame) -> None:
        _log("cloud_run_shutdown_requested", signal=signum)
        STOP.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    _log("cloud_run_health_server_started", port=PORT)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        STOP.set()
        server.server_close()
        thread.join(timeout=15)
        _log("cloud_run_service_stopped", worker_status=WORKER.run_status)


if __name__ == "__main__":
    main()
