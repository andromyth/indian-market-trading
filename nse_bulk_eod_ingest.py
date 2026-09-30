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

# NSE's 2026 CM/equity trading-holiday calendar, sourced from NSE's
# published 2026 equities holiday schedule. Weekends are handled separately.
NSE_EQUITY_HOLIDAYS_BY_YEAR = {
    2026: frozenset(
        date.fromisoformat(x)
        for x in (
            "2026-01-15",
            "2026-01-26",
            "2026-03-03",
            "2026-03-26",
            "2026-03-31",
            "2026-04-03",
            "2026-04-14",
            "2026-05-01",
            "2026-05-28",
            "2026-06-26",
            "2026-09-14",
            "2026-10-02",
            "2026-10-20",
            "2026-11-10",
            "2026-11-24",
            "2026-12-25",
        )
    )
}

DATE_RE = re.compile(r"(20\d{6})", re.I)
FOLDER_MIME = "application/vnd.google-apps.folder"
ARCHIVE_FOLDER_PATHS = (
    "production_data/nse_official/bulk_eod",
    "raw/nse_daily_bulk_2026",
    "raw/2026_expansion",
)
LOCAL_SYNC_MANIFEST = (
    Path(__file__).resolve().parent
    / "data"
    / "nse_official"
    / "bulk_eod"
    / "sync_manifest.json"
)


def business_days(start: date, end: date) -> Iterable[date]:
    """Yield NSE CM equity trading days, excluding weekends and known holidays."""
    holidays = NSE_EQUITY_HOLIDAYS_BY_YEAR.get(start.year, frozenset())
    d = start
    while d <= end:
        year_holidays = NSE_EQUITY_HOLIDAYS_BY_YEAR.get(d.year, holidays if d.year == start.year else frozenset())
        if d.weekday() < 5 and d not in year_holidays:
            yield d
        d += timedelta(days=1)


def _list_children(
    store: GoogleDriveStore,
    parent_id: str,
    *,
    max_pages: int = 10,
    page_size: int = 500,
) -> tuple[list[dict], bool]:
    """List immediate Drive children with bounded pagination."""
    items: list[dict] = []
    token = None
    pages = 0
    while True:
        pages += 1
        if pages > max_pages:
            return items, False
        kwargs = {
            "q": f"'{parent_id}' in parents and trashed = false",
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,mimeType)",
            "pageSize": page_size,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if token:
            kwargs["pageToken"] = token
        resp = store.service.files().list(**kwargs).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            return items, True

def _find_folder_path(store: GoogleDriveStore, relative_path: str) -> str | None:
    """Resolve a folder below the configured root without creating anything."""
    parent = store.root_id
    for part in [p for p in relative_path.replace("\\", "/").split("/") if p]:
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            return None
        parent = hit["id"]
    return parent


def _find_named_folders_globally(store: GoogleDriveStore, names: set[str]) -> list[str]:
    """Find exact archive folder names without scanning every Drive folder."""
    ids: list[str] = []
    for name in sorted({str(n).strip() for n in names if str(n).strip()}):
        kwargs = {
            "q": (
                "trashed = false and "
                "mimeType = 'application/vnd.google-apps.folder' and "
                f"name = '{store._q(name)}'"
            ),
            "spaces": "drive",
            "fields": "files(id,name,mimeType)",
            "pageSize": 100,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "corpora": "allDrives",
        }
        resp = store.service.files().list(**kwargs).execute()
        ids.extend(item["id"] for item in resp.get("files", []))
    return sorted(set(ids))


def _date_from_text(value: str) -> date | None:
    """Extract common NSE date tokens, including separated date formats."""
    text_value = str(value or "")
    patterns = (
        r"(?<!\\d)(20\\d{2})[-_./]?(0[1-9]|1[0-2])[-_./]?([0-3]\\d)(?!\\d)",
        r"(?<!\\d)([0-3]\\d)[-_./]?(0[1-9]|1[0-2])[-_./]?(20\\d{2})(?!\\d)",
    )
    for pattern, fmt in ((patterns[0], "%Y%m%d"), (patterns[1], "%d%m%Y")):
        for match in re.finditer(pattern, text_value):
            token = "".join(match.groups())
            try:
                return datetime.strptime(token, fmt).date()
            except ValueError:
                continue
    return None

def _date_from_path_parts(parts: list[str]) -> date | None:
    """Extract a YYYY/MM/DD-style date from ancestor folder names."""
    cleaned = [str(p).strip() for p in parts]
    for i in range(len(cleaned) - 2):
        try:
            y, m, d = int(cleaned[i]), int(cleaned[i + 1]), int(cleaned[i + 2])
            if 2000 <= y <= 2100:
                return date(y, m, d)
        except (TypeError, ValueError):
            continue
    return None


def _collect_dates_under_folder(
    store: GoogleDriveStore,
    folder_id: str,
    *,
    label: str,
    max_depth: int = 8,
    max_api_calls: int = 300,
    max_items: int = 30000,
) -> set[date]:
    """Walk a known archive subtree, but never allow unbounded Drive enumeration."""
    found: set[date] = set()
    queue: list[tuple[str, list[str], int]] = [(folder_id, [], 0)]
    api_calls = 0
    scanned_items = 0
    complete = True

    print(f"Drive discovery: scanning {label}", flush=True)

    while queue:
        current_id, ancestors, depth = queue.pop()
        if depth > max_depth:
            complete = False
            continue

        api_calls += 1
        if api_calls > max_api_calls:
            complete = False
            print(f"Drive discovery: LIMIT api_calls={max_api_calls} path={label}", flush=True)
            break

        children, page_complete = _list_children(
            store, current_id, max_pages=10, page_size=500
        )
        if not page_complete:
            complete = False

        scanned_items += len(children)
        if scanned_items > max_items:
            complete = False
            print(f"Drive discovery: LIMIT items={max_items} path={label}", flush=True)
            break

        for item in children:
            name = str(item.get("name") or "")
            if item.get("mimeType") == FOLDER_MIME:
                next_ancestors = ancestors + [name]
                folder_date = _date_from_text(name) or _date_from_path_parts(next_ancestors)
                if folder_date:
                    found.add(folder_date)
                    # A dated folder already proves the archive date. Do not
                    # recursively enumerate all files beneath that date.
                    continue
                queue.append((item["id"], next_ancestors, depth + 1))
                continue

            item_date = _date_from_text(name) or _date_from_path_parts(ancestors)
            if item_date:
                found.add(item_date)

        if api_calls % 25 == 0:
            print(
                f"Drive discovery: {label} folders_scanned={api_calls} dates={len(found)}",
                flush=True,
            )

    print(
        f"Drive discovery: {label} complete dates={len(found)} "
        f"api_calls={api_calls} items={scanned_items} complete={complete}",
        flush=True,
    )
    return found

def _remote_bhavcopy_dates(store: GoogleDriveStore) -> set[date]:
    """Return NSE EOD dates visible in Drive across current and legacy archive paths."""
    found: set[date] = set()

    # Prefer the known archive subtrees used by this project. This avoids
    # scanning unrelated Drive content and also works for legacy archives
    # whose filenames do not contain the full BhavCopy prefix.
    resolved_paths: list[str] = []
    for idx, relative_path in enumerate(ARCHIVE_FOLDER_PATHS):
        # raw/2026_expansion is a broad legacy tree. Only inspect it when
        # the two known NSE archive locations produce no dates.
        if idx == 2 and found:
            print(
                "Drive discovery: primary/legacy archive dates found; "
                "skipping broad expansion tree.",
                flush=True,
            )
            break

        folder_id = _find_folder_path(store, relative_path)
        if not folder_id:
            continue

        resolved_paths.append(relative_path)
        print(f"Drive archive path resolved: {relative_path}", flush=True)
        before = len(found)
        found.update(
            _collect_dates_under_folder(
                store,
                folder_id,
                label=relative_path,
            )
        )

        if len(found) == before:
            print(
                f"Drive discovery: {relative_path} added no date tokens.",
                flush=True,
            )

    # Fallback: if the configured root does not contain the expected path,
    # search for the archive folder names anywhere visible to this credential.
    # This handles older Drive layouts while still refusing to invent dates.
    if not found:
        fallback_ids = _find_named_folders_globally(
            store,
            {"nse_daily_bulk_2026", "2026_expansion", "bulk_eod"},
        )
        for folder_id in fallback_ids:
            found.update(
                _collect_dates_under_folder(
                    store,
                    folder_id,
                    label=f"global:{folder_id}",
                )
            )

    print(f"Drive archive path(s) resolved: {resolved_paths or 'none'}")
    print(f"Drive NSE archive dates discovered: {len(found)}")
    if found:
        print(f"Drive earliest date: {min(found).isoformat()}")
        print(f"Drive latest date  : {max(found).isoformat()}")

    # Search canonical filenames globally only when the known archive
    # subtrees/fallback folders did not yield any dates.
    if not found:
        token = None
        while True:
            kwargs = {
                "q": "trashed = false and name contains 'BhavCopy_NSE_CM_0_0_0_'",
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": 1000,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
                "corpora": "allDrives",
            }
            if token:
                kwargs["pageToken"] = token
            resp = store.service.files().list(**kwargs).execute()
            for item in resp.get("files", []):
                if item.get("mimeType") == FOLDER_MIME:
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
    allow_bootstrap: bool,
) -> tuple[list[date], str | None]:
    if not remote_dates:
        if allow_bootstrap:
            return list(business_days(bootstrap_start, today)), f"BOOTSTRAP_FROM_{bootstrap_start.isoformat()}"
        return [], "DRIVE_ARCHIVE_EMPTY_OR_UNDISCOVERED"

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
    ap.add_argument("--allow-bootstrap", action="store_true")
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
            root_meta = store.service.files().get(
                fileId=store.root_id,
                fields="id,name,mimeType,trashed,driveId",
                supportsAllDrives=True,
            ).execute()
            print(
                "Drive root:",
                json.dumps(
                    {
                        "id": root_meta.get("id"),
                        "name": root_meta.get("name"),
                        "mimeType": root_meta.get("mimeType"),
                        "driveId": root_meta.get("driveId"),
                    },
                    sort_keys=True,
                ),
            )
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
            allow_bootstrap=args.allow_bootstrap,
        )
        mode = "AUTO"
        if mode_reason == "DRIVE_ARCHIVE_EMPTY_OR_UNDISCOVERED":
            raise SystemExit(
                "DRIVE_ARCHIVE_EMPTY_OR_UNDISCOVERED: Google Drive is reachable, "
                "but no known NSE archive dates were found. Refusing to bootstrap "
                "2026 history automatically. Use an explicit manual backfill or "
                "--allow-bootstrap only for a genuinely empty archive."
            )
    else:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        if end < start:
            raise SystemExit("--end cannot be before --start")
        candidate_days = list(business_days(start, end))
        # Manual backfill is gap-filling when Drive is enabled. Do not
        # re-download dates that have already been archived successfully.
        if args.sync_drive:
            candidates = [d for d in candidate_days if d not in remote_dates]
        else:
            candidates = candidate_days
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
                "status": "PASS" if not missing_after else "INCOMPLETE_RETRY",
                "remote_dates_after_run": len(after_dates),
                "latest_remote_date_after_run": max(after_dates).isoformat() if after_dates else None,
                "missing_candidate_dates_after_run": [d.isoformat() for d in missing_after],
            }
    else:
        run["drive_postcheck"] = {"status": "SKIPPED"}

    unresolved = retryable_miss
    if args.sync_drive:
        unresolved = len(
            run.get("drive_postcheck", {}).get("missing_candidate_dates_after_run", [])
        )

    run["summary"] = {
        "candidates": len(candidates),
        "fetched": fetched,
        "no_archive_file_retry": retryable_miss,
        "unresolved_dates": unresolved,
        "errors": errors,
    }
    if errors:
        run["run_status"] = "FAILED"
    elif unresolved:
        run["run_status"] = "SUCCESS_WITH_RETRY_PENDING"
    else:
        run["run_status"] = "SUCCESS"

    run["finished_at"] = _now_ist()
    _sync_manifest(store, run)

    print(
        "SUMMARY:",
        json.dumps(
            {**run["summary"], "run_status": run["run_status"]},
            separators=(",", ":"),
            sort_keys=True,
        ),
    )
    # A retryable archive miss (normally today's not-yet-published EOD file)
    # is recorded and retried by the next run, but is not treated as a broken
    # ingestion pipeline. Only hard ingestion/Drive errors fail the workflow.
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
