#!/usr/bin/env python3
"""Fail closed unless the current NSE trading session is fully persisted.

The Delivery production refresh must never publish a new snapshot from an older
NSE session merely because today's official archive was not available when an
upstream scheduled job first ran. This gate is deliberately small and checks
only the two inputs that define the current Delivery session: official NSE EOD
and security-wise Delivery archives in Google Drive.
"""
from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from persistent_store import GoogleDriveStore

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"
EOD_RE = re.compile(r"BhavCopy_NSE_CM_0_0_0_(\d{8})_F_0000\.parquet$", re.I)
DELIVERY_RE = re.compile(r"delivery_(\d{8})\.parquet$", re.I)

# 2026 NSE CM/equity holidays used by the existing ingestion layer.
HOLIDAYS = {
    date.fromisoformat(x)
    for x in (
        "2026-01-15", "2026-01-26", "2026-03-03", "2026-03-26",
        "2026-03-31", "2026-04-03", "2026-04-14", "2026-05-01",
        "2026-05-28", "2026-06-26", "2026-09-14", "2026-10-02",
        "2026-10-20", "2026-11-10", "2026-11-24", "2026-12-25",
    )
}


def previous_trading_day(value: date) -> date:
    """Return the most recent NSE weekday that is not in the holiday set."""
    candidate = value - timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in HOLIDAYS:
        candidate -= timedelta(days=1)
    return candidate


def expected_session_date(now: datetime | None = None) -> date | None:
    """Resolve the latest completed NSE session for the hourly production window."""
    current = now.astimezone(IST) if now is not None else datetime.now(IST)
    d = current.date()

    # Saturday early morning is the overnight retry window for Friday's session.
    if d.weekday() == 5 and current.time() < time(12, 0):
        return previous_trading_day(d)

    # Sunday has no production session.
    if d.weekday() == 6:
        return None

    # A holiday evening has no new session to publish.
    if d in HOLIDAYS and current.time() >= time(18, 0):
        return None

    # Before 18:00 IST, the latest completed session is the previous NSE
    # trading day. This keeps overnight retries pinned to the same session.
    if current.time() < time(18, 0):
        return previous_trading_day(d)

    return d


def _child(store: GoogleDriveStore, parent: str, name: str):
    return store._find_child(parent, name, FOLDER_MIME)


def _resolve(store: GoogleDriveStore, path: str) -> str:
    parent = store.root_id
    for part in path.split("/"):
        hit = _child(store, parent, part)
        if not hit:
            raise RuntimeError(f"Drive folder missing: {path}")
        parent = hit["id"]
    return parent


def _names(store: GoogleDriveStore, folder_id: str, *, recursive: bool = False, max_depth: int = 5) -> list[str]:
    """Return names below a Drive folder; EOD archives are nested by date."""
    if not recursive:
        out: list[str] = []
        token = None
        while True:
            kwargs = {
                "q": f"'{folder_id}' in parents and trashed = false",
                "spaces": "drive",
                "fields": "nextPageToken,files(name,mimeType)",
                "pageSize": 1000,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
            if token:
                kwargs["pageToken"] = token
            resp = store.service.files().list(**kwargs).execute()
            out.extend(str(x.get("name", "")) for x in resp.get("files", []))
            token = resp.get("nextPageToken")
            if not token:
                return out

    found: list[str] = []
    queue: list[tuple[str, int]] = [(folder_id, 0)]
    while queue:
        current_id, depth = queue.pop()
        if depth > max_depth:
            continue
        token = None
        while True:
            kwargs = {
                "q": f"'{current_id}' in parents and trashed = false",
                "spaces": "drive",
                "fields": "nextPageToken,files(name,mimeType,id)",
                "pageSize": 1000,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }
            if token:
                kwargs["pageToken"] = token
            resp = store.service.files().list(**kwargs).execute()
            for item in resp.get("files", []):
                name = str(item.get("name", ""))
                found.append(name)
                if (
                    item.get("mimeType") == FOLDER_MIME
                    and depth < max_depth
                ):
                    queue.append((item["id"], depth + 1))
            token = resp.get("nextPageToken")
            if not token:
                break
    return found


def _dates(names: list[str], pattern: re.Pattern[str]) -> set[date]:
    found: set[date] = set()
    for name in names:
        m = pattern.match(name)
        if not m:
            continue
        try:
            found.add(datetime.strptime(m.group(1), "%Y%m%d").date())
        except ValueError:
            pass
    return found


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target-date",
        default="",
        help="Explicit NSE session date to verify (YYYY-MM-DD).",
    )
    args = ap.parse_args()

    now = datetime.now(IST)
    if args.target_date.strip():
        target = date.fromisoformat(args.target_date.strip())
        target_source = "EXPLICIT_TARGET_DATE"
    else:
        target = expected_session_date(now)
        target_source = "CLOCK_RESOLVED"

    print(f"Gate clock (IST)             : {now.isoformat()}")
    print(f"NSE freshness target session : {target}")
    print(f"Target resolution             : {target_source}")

    if target is None:
        raise SystemExit("NO_TRADING_SESSION: no NSE session is expected in the current retry window")

    if target.weekday() >= 5 or target in HOLIDAYS:
        raise SystemExit(f"NO_TRADING_SESSION: {target} is a weekend/NSE holiday; refusing refresh")

    store = GoogleDriveStore()
    eod_folder = _resolve(store, "production_data/nse_official/bulk_eod")
    delivery_folder = _resolve(store, "production_data/nse_official/delivery_full")

    eod_dates = _dates(_names(store, eod_folder, recursive=True, max_depth=5), EOD_RE)
    delivery_dates = _dates(_names(store, delivery_folder), DELIVERY_RE)

    eod_ok = target in eod_dates
    delivery_ok = target in delivery_dates

    print(f"EOD latest persisted date      : {max(eod_dates) if eod_dates else None}")
    print(f"Delivery latest persisted date : {max(delivery_dates) if delivery_dates else None}")
    print(f"TARGET EOD PRESENT             : {eod_ok}")
    print(f"TARGET DELIVERY PRESENT        : {delivery_ok}")

    if not eod_ok or not delivery_ok:
        missing = []
        if not eod_ok:
            missing.append("NSE EOD")
        if not delivery_ok:
            missing.append("NSE Delivery")
        raise SystemExit(
            "CURRENT_SESSION_NOT_READY: " + ", ".join(missing) +
            f" archive for {target} is not persisted; production refresh is blocked"
        )

    print("CURRENT_SESSION_READY: NSE EOD + Delivery are both persisted for target session")


if __name__ == "__main__":
    main()
