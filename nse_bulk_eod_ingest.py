#!/usr/bin/env python3
"""Continuous NSE-first EOD ingestion with incremental catch-up and gap repair.

Normal scheduled mode is compatible with the existing workflow: when the
workflow supplies today's date, this script ignores the one-day range and
looks at Google Drive first, then fetches every missing NSE weekday through
India's current date. Manual historical backfills still use --start/--end.

This job is ingestion only. It does not train models or place orders.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from nse_data_layer import fetch_nse_eod_bhavcopy, nse_last_diagnostic
from persistent_store import GoogleDriveStore

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_BOOTSTRAP_START = date(2026, 1, 1)
DEFAULT_REPAIR_DAYS = 14
DATE_RE = re.compile(r"_(\d{8})_F_0000(?:\.csv\.zip|\.parquet)$", re.I)
LOCAL_SYNC_MANIFEST = (
    Path(__file__).resolve().parent
    / "data"
    / "nse_official"
    / "bulk_eod"
    / "sync_manifest.json"
)


def business_days(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def _remote_bhavcopy_dates(store: GoogleDriveStore) -> set[date]:
    """Return all NSE bulk-bhavcopy dates currently visible in Drive."""
    found: set[date] = set()
    token = None
    while True:
        kwargs = {
            "q": "trashed = false and name contains 'BhavCopy_NSE_CM_0_0_0_'",
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,mimeType)",
            "pageSize": 1000,
        }
        if token:
            kwargs["pageToken"] = token
        resp = store.service.files().list(**kwargs).execute()
        for item in resp.get("files", []):
            if item.get("mimeType") == "application/vnd.google-apps.folder":
                continue
            match = DATE_RE.search(str(item.get("name") or ""))
            if not match:
                continue
            try:
                found.add(datetime.strptime(match.group(1), "%Y%m%d").date())
            except ValueError:
                continue
        token = resp.get("nextPageToken")
        if not token:
            break
    return found


def _write_local_sync_manifest(payload: dict) -> None:
    LOCAL_SYNC_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_SYNC_MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _sync_manifest(store: GoogleDriveStore | None, payload: dict) -> None:
    _write_local_sync_manifest(payload)
    if store is None:
        return
    try:
        result = store.upload_or_update(
            LOCAL_SYNC_MANIFEST,
            "production_data/nse_official/bulk_eod/sync_manifest.json",
            mime_type="application/json",
        )
        payload["manifest_drive_sync"] = result
        _write_local_sync_manifest(payload)
    except Exception as exc:
        payload["manifest_drive_sync"] = {
            "status": "NOT_SYNCED",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_local_sync_manifest(payload)


def _now_ist() -> str:
    return datetime.now(IST).isoformat()


def _today_ist() -> date:
    return datetime.now(IST).date()


def _build_auto_candidates(
    remote_dates: set[date],
    today: date,
    repair_days: int,
    bootstrap_start: date,
) -> tuple[list[date], str | None]:
    if not remote_dates:
        return list(business_days(bootstrap_start, today)), f"BOOTSTRAP_FROM_{bootstrap_start.isoformat()}"

    latest = max(remote_dates)
    candidates: set[date] = set()

    # Primary catch-up: every missing weekday after the latest stored date.
    if latest < today:
        candidates.update(business_days(latest + timedelta(days=1), today))

    # Recent gap repair: recover missing weekdays even if later dates already exist.
    repair_start = max(bootstrap_start, today - timedelta(days=max(1, int(repair_days))))
    candidates.update(
        d for d in business_days(repair_start, today) if d not in remote_dates
    )

    return sorted(d for d in candidates if d not in remote_dates), None


def _fetch_one(d: date, sync_drive: bool) -> dict:
    try:
        df = fetch_nse_eod_bhavcopy(d, sync_drive=sync_drive)
        rows = int(len(df))
        if rows <= 0:
            return {"date": d.isoformat(), "status": "NO_DATA", "rows": rows}
        return {
            "date": d.isoformat(),
            "status": "FETCHED",
            "rows": rows,
            "source": "NSE_OFFICIAL_UDIFF_BHAVCOPY",
        }
    except Exception as exc:
        diag = nse_last_diagnostic()
        status_code = diag.get("status_code")
        # A 404 does not prove a holiday. Keep it unresolved so a later run can retry.
        if status_code == 404:
            return {
                "date": d.isoformat(),
                "status": "NO_ARCHIVE_FILE_RETRY",
                "rows": 0,
                "status_code": 404,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "date": d.isoformat(),
            "status": "ERROR",
            "rows": 0,
            "status_code": status_code,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--sync-drive", action="store_true")
    ap.add_argument("--repair-days", type=int, default=DEFAULT_REPAIR_DAYS)
    ap.add_argument("--bootstrap-start", default=DEFAULT_BOOTSTRAP_START.isoformat())
    args = ap.parse_args()

    if args.auto and (args.start or args.end):
        ap.error("--auto cannot be combined with --start/--end")
    if not args.auto and (not args.start or not args.end):
        ap.error("manual mode requires --start and --end, or use --auto")

    today = _today_ist()
    bootstrap_start = date.fromisoformat(args.bootstrap_start)
    store = None
    remote_dates: set[date] = set()

    if args.sync_drive:
        try:
            store = GoogleDriveStore()
            remote_dates = _remote_bhavcopy_dates(store)
        except Exception as exc:
            raise SystemExit(
                "DRIVE_DISCOVERY_ERROR: cannot inspect the NSE archive in Google Drive: "
                f"{type(exc).__name__}: {exc}"
            )

    # The existing scheduled workflow passes today's date as start=end. Treat
    # that invocation as automatic incremental mode so the workflow itself
    # does not need to know the Drive archive's latest date.
    scheduled_today_call = (
        not args.auto
        and args.sync_drive
        and date.fromisoformat(args.start) == today
        and date.fromisoformat(args.end) == today
    )

    if args.auto or scheduled_today_call:
        candidates, mode_reason = _build_auto_candidates(
            remote_dates=remote_dates,
            today=today,
            repair_days=args.repair_days,
            bootstrap_start=bootstrap_start,
        )
        mode = "AUTO"
    else:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        if end < start:
            raise SystemExit("--end cannot be before --start")
        candidates = list(business_days(start, end))
        mode_reason = "MANUAL_BACKFILL"
        mode = "MANUAL"

    run = {
        "version": "nse-eod-continuous-sync-v1",
        "started_at": _now_ist(),
        "mode": mode,
        "today_ist": today.isoformat(),
        "scheduled_today_call": scheduled_today_call,
        "repair_days": int(args.repair_days),
        "bootstrap_start": bootstrap_start.isoformat(),
        "remote_dates_seen_before_run": len(remote_dates),
        "latest_remote_date_before_run": max(remote_dates).isoformat() if remote_dates else None,
        "selection_reason": mode_reason,
        "candidate_dates": [d.isoformat() for d in candidates],
        "results": [],
    }

    print("NSE EOD CONTINUOUS SYNC")
    print(f"Mode             : {run['mode']}")
    print(f"Today IST        : {run['today_ist']}")
    print(f"Drive dates      : {run['remote_dates_seen_before_run']}")
    print(f"Latest Drive date: {run['latest_remote_date_before_run']}")
    print(f"Candidates       : {len(candidates)}")
    if mode_reason:
        print(f"Reason           : {mode_reason}")

    fetched = retryable_miss = errors = 0
    for d in candidates:
        result = _fetch_one(d, sync_drive=args.sync_drive)
        run["results"].append(result)
        status = result["status"]
        if status == "FETCHED":
            fetched += 1
            print(f"{d}: FETCHED rows={result['rows']:,}")
        elif status == "NO_ARCHIVE_FILE_RETRY":
            retryable_miss += 1
            print(f"{d}: NO_ARCHIVE_FILE_RETRY (404)")
        else:
            errors += 1
            print(f"{d}: ERROR {result.get('error', status)}")

    if args.sync_drive:
        try:
            after_dates = _remote_bhavcopy_dates(store)
        except Exception as exc:
            after_dates = set()
            run["drive_postcheck"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors += 1
        else:
            missing_after = sorted(d for d in candidates if d not in after_dates)
            run["drive_postcheck"] = {
                "status": "PASS" if not missing_after else "MISSING_REMOTE_FILES",
                "remote_dates_after_run": len(after_dates),
                "latest_remote_date_after_run": max(after_dates).isoformat() if after_dates else None,
                "missing_candidate_dates_after_run": [d.isoformat() for d in missing_after],
            }
            # 404/no-archive dates are unresolved and should make the run visible
            # as incomplete so the next scheduled run retries them.
            errors += len(missing_after)
    else:
        run["drive_postcheck"] = {"status": "SKIPPED"}

    run["summary"] = {
        "candidates": len(candidates),
        "fetched": fetched,
        "no_archive_file_retry": retryable_miss,
        "errors": errors,
    }
    run["finished_at"] = _now_ist()
    _sync_manifest(store, run)

    print("SUMMARY:", json.dumps(run["summary"], separators=(",", ":"), sort_keys=True))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
