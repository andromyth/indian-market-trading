#!/usr/bin/env python3
"""Backfill missed official NSE Delivery PIT snapshots and verify session continuity.

The script only rebuilds snapshots for previously missed NSE sessions. It does
not train/promote models, rewrite existing historical snapshots, or place trades.
The current target session is built by the normal production refresh step after
this backfill stage.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from nse_bulk_eod_ingest import business_days
from persistent_store import GoogleDriveStore

IST = ZoneInfo("Asia/Kolkata")
BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", "/content/drive/MyDrive/Indian_Market_Research"))
REPORT_PATH = BASE / "diagnostics" / "production" / "pit_snapshot_continuity_latest.json"
REMOTE_REPORT = "diagnostics/production/pit_snapshot_continuity_latest.json"
FOLDER_MIME = "application/vnd.google-apps.folder"

EOD_RE = re.compile(r"^BhavCopy_NSE_CM_0_0_0_(\d{8})_F_0000\.parquet$", re.IGNORECASE)
DELIVERY_RE = re.compile(r"^delivery_(\d{8})\.parquet$", re.IGNORECASE)
PIT_RE = re.compile(r"^DELIVERY_PIT_FEATURES_(20\d{2}-\d{2}-\d{2})\.parquet$", re.IGNORECASE)


def _resolve_folder(store: GoogleDriveStore, path: str) -> str:
    parent = store.root_id
    for part in path.strip("/").split("/"):
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            raise RuntimeError(f"Google Drive folder missing: {path} (component {part})")
        parent = hit["id"]
    return parent


def _list_files_recursive(store: GoogleDriveStore, folder_id: str, *, recursive: bool) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    queue: list[tuple[str, int]] = [(folder_id, 0)]
    while queue:
        current, depth = queue.pop(0)
        token = None
        while True:
            params: dict[str, Any] = {
                "q": f"'{current}' in parents and trashed = false",
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime)",
                "pageSize": 1000,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
            if token:
                params["pageToken"] = token
            response = store.service.files().list(**params).execute()
            for item in response.get("files", []):
                if item.get("mimeType") == FOLDER_MIME:
                    if recursive:
                        queue.append((str(item["id"]), depth + 1))
                else:
                    files.append(item)
            token = response.get("nextPageToken")
            if not token:
                break
        # The Drive folder tree is bounded by known path depth, but retain a
        # hard upper guard so an unexpected folder loop cannot scan indefinitely.
        if len(queue) > 10000:
            raise RuntimeError("Google Drive file traversal exceeded safety limit")
    return files


def _parsed_dates(items: Iterable[dict[str, Any]], pattern: re.Pattern[str], fmt: str) -> set[date]:
    found: set[date] = set()
    for item in items:
        name = str(item.get("name", ""))
        match = pattern.fullmatch(name)
        if not match:
            continue
        try:
            found.add(datetime.strptime(match.group(1), fmt).date())
        except ValueError:
            continue
    return found


def load_drive_inventory(store: GoogleDriveStore, *, max_attempts: int = 4) -> dict[str, set[date]]:
    """Read the three Drive inventories, retrying transient rate/server errors."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    retryable_statuses = {429, 500, 502, 503, 504}
    for attempt in range(1, max_attempts + 1):
        try:
            eod_folder = _resolve_folder(store, "production_data/nse_official/bulk_eod")
            delivery_folder = _resolve_folder(store, "production_data/nse_official/delivery_full")
            pit_folder = _resolve_folder(store, "features/daily")

            eod_files = _list_files_recursive(store, eod_folder, recursive=True)
            delivery_files = _list_files_recursive(store, delivery_folder, recursive=False)
            pit_files = _list_files_recursive(store, pit_folder, recursive=False)
            inventory = {
                "eod": _parsed_dates(eod_files, EOD_RE, "%Y%m%d"),
                "delivery": _parsed_dates(delivery_files, DELIVERY_RE, "%Y%m%d"),
                "pit": _parsed_dates(pit_files, PIT_RE, "%Y-%m-%d"),
            }
            print(
                "DRIVE_INVENTORY_READ_OK: "
                f"eod_dates={len(inventory['eod'])} "
                f"delivery_dates={len(inventory['delivery'])} "
                f"pit_dates={len(inventory['pit'])}",
                flush=True,
            )
            return inventory
        except Exception as exc:
            status = getattr(getattr(exc, "resp", None), "status", None)
            if status not in retryable_statuses or attempt >= max_attempts:
                raise
            delay = min(2 ** (attempt - 1), 8)
            print(
                "DRIVE_TRANSIENT_READ_RETRY: "
                f"http_status={status} attempt={attempt}/{max_attempts} "
                f"retry_in_seconds={delay} error={type(exc).__name__}",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("Drive inventory retry loop ended unexpectedly")


def expected_nse_sessions(start: date, end: date) -> list[date]:
    if start > end:
        return []
    return list(business_days(start, end))


def plan_snapshot_backfill(
    *,
    target_date: date,
    lookback_calendar_days: int,
    eod_dates: set[date],
    delivery_dates: set[date],
    pit_dates: set[date],
    max_backfills_per_run: int,
    include_target: bool = False,
) -> dict[str, Any]:
    """Build a deterministic plan from the NSE calendar and persisted source files."""
    if lookback_calendar_days < 1:
        raise ValueError("lookback_calendar_days must be >= 1")
    if max_backfills_per_run < 1:
        raise ValueError("max_backfills_per_run must be >= 1")

    start = target_date - timedelta(days=lookback_calendar_days)
    end = target_date if include_target else target_date - timedelta(days=1)
    expected = expected_nse_sessions(start, end)
    missing_eod = [d for d in expected if d not in eod_dates]
    missing_delivery = [d for d in expected if d not in delivery_dates]
    ready_sessions = [d for d in expected if d in eod_dates and d in delivery_dates]
    missing_pit = [d for d in ready_sessions if d not in pit_dates]
    backfill = missing_pit[:max_backfills_per_run]
    deferred = missing_pit[max_backfills_per_run:]

    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "expected_nse_sessions": [d.isoformat() for d in expected],
        "expected_session_count": len(expected),
        "missing_eod_dates": [d.isoformat() for d in missing_eod],
        "missing_delivery_dates": [d.isoformat() for d in missing_delivery],
        "missing_pit_dates_with_sources_ready": [d.isoformat() for d in missing_pit],
        "backfill_dates_this_run": [d.isoformat() for d in backfill],
        "deferred_backfill_dates": [d.isoformat() for d in deferred],
        "observed_pit_dates": [d.isoformat() for d in sorted(pit_dates) if start <= d <= end],
        "include_target": bool(include_target),
    }


def _write_report(store: GoogleDriveStore, report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    store.upload_or_update(
        REPORT_PATH, REMOTE_REPORT, mime_type="application/json"
    )


def _run_refresh(day: date) -> dict[str, Any]:
    command = [
        sys.executable,
        "-u",
        "delivery_production_refresh.py",
        "--lookback-days", "150",
        "--delivery-lookback-days", "150",
        "--source-wait-minutes", "0",
        "--expected-date", day.isoformat(),
    ]
    started = datetime.now(timezone.utc).isoformat()
    print(f"PIT_BACKFILL_START date={day.isoformat()} command={' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    result = {
        "date": day.isoformat(),
        "command": command,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "return_code": int(completed.returncode),
        "status": "PASS" if completed.returncode == 0 else "ERROR",
    }
    print(f"PIT_BACKFILL_RESULT {json.dumps(result, default=str)}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", required=True, help="Latest completed NSE session YYYY-MM-DD.")
    parser.add_argument("--lookback-calendar-days", type=int, default=14)
    parser.add_argument("--max-backfills-per-run", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true", help="Only verify recent EOD/Delivery/PIT continuity; do not build snapshots.")
    args = parser.parse_args()

    target = date.fromisoformat(args.target_date)
    today_ist = datetime.now(IST).date()
    if target.weekday() >= 5 or target not in expected_nse_sessions(target, target):
        raise ValueError(f"Target {target} is not a weekday NSE session in the configured calendar")
    if target > today_ist:
        raise ValueError(f"Target {target} is in the future relative to India date {today_ist}")

    store = GoogleDriveStore()
    inventory = load_drive_inventory(store)
    plan = plan_snapshot_backfill(
        target_date=target,
        lookback_calendar_days=args.lookback_calendar_days,
        eod_dates=inventory["eod"],
        delivery_dates=inventory["delivery"],
        pit_dates=inventory["pit"],
        max_backfills_per_run=args.max_backfills_per_run,
        include_target=args.audit_only,
    )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_at_ist": datetime.now(IST).isoformat(),
        "status": "AUDIT_ONLY" if args.audit_only else "BACKFILL_PLANNED",
        "target_date": target.isoformat(),
        "lookback_calendar_days": int(args.lookback_calendar_days),
        "max_backfills_per_run": int(args.max_backfills_per_run),
        "read_only_except_snapshot_backfill": True,
        "model_weights_changed": False,
        "model_promotion_performed": False,
        "orders_placed": False,
        "inventory_counts": {
            "eod_dates": len(inventory["eod"]),
            "delivery_dates": len(inventory["delivery"]),
            "pit_dates": len(inventory["pit"]),
        },
        "plan": plan,
        "backfill_results": [],
        "warnings": [],
    }

    if not args.audit_only:
        for text_day in plan["backfill_dates_this_run"]:
            result = _run_refresh(date.fromisoformat(text_day))
            report["backfill_results"].append(result)
            if result["status"] != "PASS":
                report["warnings"].append(
                    f"PIT snapshot backfill failed for {text_day} with return code {result['return_code']}."
                )
        # Re-query Drive after the writes; never assume a successful subprocess
        # means the expected filename was persisted.
        inventory = load_drive_inventory(store)
        report["inventory_counts_after"] = {
            "eod_dates": len(inventory["eod"]),
            "delivery_dates": len(inventory["delivery"]),
            "pit_dates": len(inventory["pit"]),
        }
        final_plan = plan_snapshot_backfill(
            target_date=target,
            lookback_calendar_days=args.lookback_calendar_days,
            eod_dates=inventory["eod"],
            delivery_dates=inventory["delivery"],
            pit_dates=inventory["pit"],
            max_backfills_per_run=args.max_backfills_per_run,
            include_target=False,
        )
        report["plan_after_backfill"] = final_plan
        if final_plan["deferred_backfill_dates"]:
            report["status"] = "BACKFILL_BATCH_COMPLETED_GAPS_REMAIN"
            report["warnings"].append(
                "More missing PIT sessions remain than this run's safe backfill limit. The post-refresh audit will report any remaining gaps; the next scheduled retry continues from the oldest missing session."
            )
        elif final_plan["missing_pit_dates_with_sources_ready"]:
            report["status"] = "BACKFILL_INCOMPLETE"
            report["warnings"].append("One or more source-ready sessions still have no persisted PIT snapshot.")
        else:
            report["status"] = "BACKFILL_COMPLETED"
    else:
        report["plan_after_backfill"] = plan
        unresolved = (
            plan["missing_eod_dates"]
            or plan["missing_delivery_dates"]
            or plan["missing_pit_dates_with_sources_ready"]
        )
        report["status"] = "PASS" if not unresolved else "GAPS_REMAIN"
        if plan["missing_eod_dates"]:
            report["warnings"].append("Expected NSE EOD archives are missing in the audit window.")
        if plan["missing_delivery_dates"]:
            report["warnings"].append("Expected NSE Delivery archives are missing in the audit window.")
        if plan["missing_pit_dates_with_sources_ready"]:
            report["warnings"].append("Expected NSE sessions lack persisted Delivery PIT snapshots.")

    _write_report(store, report)
    print(json.dumps(report, indent=2, default=str))
    if args.audit_only and report["status"] != "PASS":
        return 2
    # The pre-refresh backfill step is deliberately non-blocking: the normal
    # current-session PIT must still be published before the strict final audit.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
