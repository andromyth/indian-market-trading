#!/usr/bin/env python3
"""Fill same-day five-minute history gaps after the close and before consolidation.

The live worker can miss a five-minute history capture while still writing the
quote snapshot. Drive-only recovery cannot create those bars. This script
discovers the affected slots, triggers the existing Cloud Run provider backfill
for the pinned roster, and waits for it to finish. It never writes snapshots or
predictions itself and never runs during the cash session.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IST = ZoneInfo("Asia/Kolkata")
PROJECT_ID = os.getenv("PROJECT_ID", "indian-market-trading")
REGION = os.getenv("REGION", "asia-south1")
SERVICE = os.getenv("SERVICE", "indian-market-intraday-worker")
LIVE_HISTORY_ROOT = "production_data/nse_official/intraday_5m_live"
PREDICTION_ROOT = "production_data/all_nse_minute_predictions"
CONFIRM = "START_NATIVE_5M_BACKFILL"
DONE_STATUSES = {"COMPLETED", "COMPLETED_WITH_GAPS"}
FAILED_STATUSES = {"FAILED", "ERROR"}
ACTIVE_STATUSES = {"RUNNING", "PAUSED_FOR_LIVE_MARKET"}
MANIFEST_RE = re.compile(r"^manifest_(\d{8}_\d{4})\.json$")


def resolve_session_date(raw: str | None = None) -> date:
    from market_session_guard import is_nse_trading_day

    if raw:
        return date.fromisoformat(raw)
    now = datetime.now(IST)
    session = now.date()
    if now.time() < clock_time(9, 15):
        session -= timedelta(days=1)
    while not is_nse_trading_day(session):
        session -= timedelta(days=1)
    return session


def missing_live_history_slots(files: list[dict[str, Any]], session_date: date) -> list[str]:
    from consolidate_live_5m_into_archive import expected_session_slots

    stamp = f"{session_date:%Y%m%d}"
    present = {str(item.get("name") or "") for item in files}
    missing: list[str] = []
    for slot in expected_session_slots():
        label = slot.strftime("%H%M")
        if f"capture_{stamp}_{label}.parquet" not in present:
            missing.append(label)
    return missing


def warming_prediction_slots(
    manifests: list[tuple[str, dict[str, Any]]],
) -> list[str]:
    slots: list[str] = []
    for name, manifest in manifests:
        match = MANIFEST_RE.match(name)
        if not match:
            continue
        counts = manifest.get("prediction_status_counts") or {}
        if int(manifest.get("prediction_count") or 0) == 0 and int(
            counts.get("WARMING_HISTORY") or 0
        ) > 0:
            slots.append(match.group(1))
    return sorted(set(slots))


def discover_gap_state(
    *,
    session_date: date,
    live_history_files: list[dict[str, Any]],
    prediction_manifests: list[tuple[str, dict[str, Any]]],
    roster: list[str],
) -> dict[str, Any]:
    missing_history = missing_live_history_slots(live_history_files, session_date)
    warming = warming_prediction_slots(prediction_manifests)
    needs_backfill = bool(missing_history or warming)
    return {
        "session_date": session_date.isoformat(),
        "needs_backfill": needs_backfill,
        "missing_live_history_slots": missing_history,
        "warming_prediction_slots": warming,
        "gap_symbol_count": len(roster) if needs_backfill else 0,
        "reason": (
            "missing_live_history_slots"
            if missing_history
            else "warming_prediction_slots"
            if warming
            else "none"
        ),
    }


def backfill_symbol_argument(symbols: list[str], max_chars: int = 800) -> str:
    joined = ",".join(symbols)
    return joined if len(joined) <= max_chars else ""


def backfill_payload(session_date: date, symbols: list[str]) -> dict[str, str]:
    return {
        "confirm": CONFIRM,
        "end_date": session_date.isoformat(),
        "replace_existing_history": "no",
        "symbols": backfill_symbol_argument(symbols),
    }


def classify_backfill_status(status: str) -> str:
    value = str(status or "").upper()
    if value in DONE_STATUSES:
        return "DONE"
    if value in FAILED_STATUSES:
        return "FAILED"
    if value in ACTIVE_STATUSES:
        return "ACTIVE"
    if value in {"IDLE", ""}:
        return "IDLE"
    return "UNKNOWN"


def _gcloud(*args: str) -> str:
    result = subprocess.run(
        ["gcloud", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.stdout.strip()


def service_url() -> str:
    configured = os.getenv("BACKFILL_SERVICE_URL", "").strip()
    if configured:
        return configured.rstrip("/")
    return _gcloud(
        "run",
        "services",
        "describe",
        SERVICE,
        f"--project={PROJECT_ID}",
        f"--region={REGION}",
        "--format=value(status.url)",
    )


def identity_token(audience: str) -> str:
    configured = os.getenv("BACKFILL_IDENTITY_TOKEN", "").strip()
    if configured:
        return configured
    return _gcloud(
        "auth",
        "print-identity-token",
        f"--audiences={audience}",
    )


def _request_json(
    url: str,
    token: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw or "{}")
        except json.JSONDecodeError:
            body = {"raw": raw}
        return exc.code, body


def start_backfill(url: str, token: str, session_date: date, symbols: list[str]) -> dict[str, Any]:
    code, body = _request_json(
        f"{url}/backfill",
        token,
        method="POST",
        payload=backfill_payload(session_date, symbols),
    )
    status = str(body.get("status") or "").upper()
    if code == 202 and status == "BACKFILL_STARTED":
        return {"http_status": code, "status": status}
    if code == 409 and status == "BACKFILL_ALREADY_RUNNING":
        return {"http_status": code, "status": status}
    raise RuntimeError(
        "BACKFILL_START_FAILED http=" + str(code) + " status=" + status
    )


def backfill_state(url: str, token: str) -> dict[str, Any]:
    code, body = _request_json(f"{url}/backfillz", token)
    if code != 200:
        return {"http_status": code, "status": "", "state": {}}
    state = body.get("five_minute_backfill") or {}
    return {
        "http_status": code,
        "status": str(body.get("status") or state.get("status") or "").upper(),
        "state": state,
    }


def _belongs_to_session(current: dict[str, Any], session_date: date) -> bool:
    started = str((current.get("state") or {}).get("started_at_utc") or "")
    return bool(started) and started[:10] == session_date.isoformat()


def wait_for_backfill(
    url: str,
    token: str,
    *,
    session_date: date,
    max_wait_minutes: int,
    interval_seconds: int = 15,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(1, int(max_wait_minutes)) * 60
    started = time.monotonic()
    saw_active = False
    last: dict[str, Any] = {"status": "UNKNOWN"}
    while time.monotonic() < deadline:
        current = backfill_state(url, token)
        last = current
        kind = classify_backfill_status(current.get("status", ""))
        print(
            "SAME_DAY_BACKFILL_STATE="
            + json.dumps(
                {
                    "status": current.get("status"),
                    "http_status": current.get("http_status"),
                    "phase": (current.get("state") or {}).get("phase"),
                    "updated_at_utc": (current.get("state") or {}).get("updated_at_utc"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if kind in {"DONE", "FAILED"}:
            if _belongs_to_session(current, session_date):
                return current
            return {"status": "IDLE", "http_status": current.get("http_status"), "state": current.get("state") or {}}
        if kind == "ACTIVE":
            saw_active = True
        if kind == "IDLE" and not saw_active and time.monotonic() - started >= 60:
            return current
        time.sleep(max(1, int(interval_seconds)))
    return {"status": "TIMEOUT", "http_status": last.get("http_status"), "state": last.get("state") or {}}


def _load_live_history_files(store: Any, session_date: date) -> list[dict[str, Any]]:
    from audit_all_universe_minute_snapshots import _find_folder, _list_children

    folder_id = _find_folder(
        store,
        LIVE_HISTORY_ROOT.split("/")
        + [f"{session_date:%Y}", f"{session_date:%m}", f"{session_date:%d}"],
    )
    return _list_children(store, folder_id) if folder_id else []


def _load_prediction_manifests(
    store: Any, session_date: date
) -> list[tuple[str, dict[str, Any]]]:
    from audit_all_universe_minute_snapshots import (
        _download_json,
        _find_folder,
        _list_children,
    )

    folder_id = _find_folder(
        store,
        PREDICTION_ROOT.split("/")
        + [f"{session_date:%Y}", f"{session_date:%m}", f"{session_date:%d}"],
    )
    files = _list_children(store, folder_id) if folder_id else []
    manifests: list[tuple[str, dict[str, Any]]] = []
    for item in files:
        name = str(item.get("name") or "")
        if not MANIFEST_RE.match(name):
            continue
        manifest = _download_json(store, item["id"])
        if manifest:
            manifests.append((name, manifest))
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="", help="NSE session date YYYY-MM-DD")
    parser.add_argument("--max-wait-minutes", type=int, default=12)
    parser.add_argument("--wait-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", default="", help="Comma-separated override")
    args = parser.parse_args()

    session_date = resolve_session_date(args.date or None)
    from market_session_guard import is_nse_trading_day

    if not is_nse_trading_day(session_date):
        print(
            "SAME_DAY_BACKFILL_JSON="
            + json.dumps(
                {"session_date": session_date.isoformat(), "status": "SKIPPED_NON_TRADING_DAY"},
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.wait_only:
        url = service_url()
        token = identity_token(url)
        current = wait_for_backfill(
            url,
            token,
            session_date=session_date,
            max_wait_minutes=args.max_wait_minutes,
        )
        status = str(current.get("status") or "").upper()
        print(
            "SAME_DAY_BACKFILL_JSON="
            + json.dumps({"mode": "wait", "status": status}, sort_keys=True),
            flush=True,
        )
        return 0 if classify_backfill_status(status) in {"DONE", "IDLE"} else 1

    from persistent_store import GoogleDriveStore
    from prediction_universe import authoritative_roster

    store = GoogleDriveStore()
    roster = authoritative_roster()
    state = discover_gap_state(
        session_date=session_date,
        live_history_files=_load_live_history_files(store, session_date),
        prediction_manifests=_load_prediction_manifests(store, session_date),
        roster=roster,
    )
    print(
        "SAME_DAY_BACKFILL_DISCOVERY="
        + json.dumps(state, sort_keys=True),
        flush=True,
    )
    if not state["needs_backfill"]:
        print(
            "SAME_DAY_BACKFILL_JSON="
            + json.dumps({"status": "NO_GAPS", **state}, sort_keys=True),
            flush=True,
        )
        return 0

    symbols = (
        [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
        if args.symbols
        else roster
    )
    if args.dry_run:
        print(
            "SAME_DAY_BACKFILL_JSON="
            + json.dumps(
                {"status": "DRY_RUN", "symbols": len(symbols), **state},
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    url = service_url()
    token = identity_token(url)
    symbols_mode = (
        "explicit" if backfill_symbol_argument(symbols) else "runner_gap_detection"
    )
    force = str(os.getenv("SAME_DAY_BACKFILL_FORCE") or "").strip().lower() in {
        "1", "true", "yes",
    }
    existing = backfill_state(url, token)
    already_completed = (
        not force
        and classify_backfill_status(str(existing.get("status") or "")) == "DONE"
        and _belongs_to_session(existing, session_date)
    )
    if already_completed:
        started = {
            "status": "SKIPPED_ALREADY_COMPLETED",
            "previous_status": existing.get("status"),
            "updated_at_utc": (existing.get("state") or {}).get("updated_at_utc"),
        }
        current = existing
    else:
        started = start_backfill(url, token, session_date, symbols)
        current = wait_for_backfill(
            url,
            token,
            session_date=session_date,
            max_wait_minutes=args.max_wait_minutes,
        )
    print(
        "SAME_DAY_BACKFILL_TRIGGER=" + json.dumps(started, sort_keys=True),
        flush=True,
    )
    status = str(current.get("status") or "").upper()
    print(
        "SAME_DAY_BACKFILL_JSON="
        + json.dumps(
            {
                "status": status,
                "trigger": started,
                "symbols": len(symbols),
                "symbols_mode": symbols_mode,
                **state,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if classify_backfill_status(status) == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
