#!/usr/bin/env python3
"""Automatic NSE official Security-wise Delivery archive sync.

Runs from GitHub Actions after the NSE EOD sync. It fetches recent trading-day
MTO archives, persists each successful day as a parquet file, and mirrors the
files to Google Drive. Existing Drive files are skipped so normal daily runs
are incremental; a recent lookback repairs missed delivery days.
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from nse_bulk_eod_ingest import business_days
from nse_delivery_data_layer import fetch_mto_delivery_day
from persistent_store import GoogleDriveStore, drive_status

LOCAL_ROOT = Path(__file__).resolve().parent / "data" / "nse_official" / "delivery_full"
REMOTE_ROOT = "production_data/nse_official/delivery_full"
IST = ZoneInfo("Asia/Kolkata")


def _remote_exists(store: GoogleDriveStore, remote_relative_path: str) -> bool:
    parts = [p for p in remote_relative_path.replace("\\", "/").strip("/").split("/") if p]
    if not parts:
        return False
    parent = store.root_id
    for part in parts[:-1]:
        folder = store._find_child(parent, part, "application/vnd.google-apps.folder")
        if not folder:
            return False
        parent = folder["id"]
    return store._find_child(parent, parts[-1]) is not None


def _candidate_dates(lookback_days: int) -> list[date]:
    today = datetime.now(IST).date()
    start = today - timedelta(days=max(1, int(lookback_days)))
    # Weekdays are candidates; NSE holidays simply return MISSING from the
    # official MTO endpoint and are never fabricated as data.
    return list(business_days(start, today))


def sync_delivery(lookback_days: int, sleep_seconds: float, retries: int) -> None:
    cfg = drive_status()
    if not cfg.get("configured") or cfg.get("reachable") is False:
        raise RuntimeError(f"Google Drive authentication check failed: {cfg}")

    store = GoogleDriveStore()
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    dates = _candidate_dates(lookback_days)
    print("=" * 90)
    print("NSE OFFICIAL SECURITY-WISE DELIVERY DAILY SYNC")
    print("=" * 90)
    print(f"Trade-date window : {dates[0] if dates else 'none'} -> {dates[-1] if dates else 'none'}")
    print(f"Local root        : {LOCAL_ROOT}")
    print(f"Drive root        : {REMOTE_ROOT}")

    ok = missing = empty = skipped = 0

    for idx, trade_date in enumerate(dates, 1):
        filename = f"delivery_{trade_date:%Y%m%d}.parquet"
        local_path = LOCAL_ROOT / filename
        remote_path = f"{REMOTE_ROOT}/{filename}"

        if local_path.exists() or _remote_exists(store, remote_path):
            skipped += 1
            print(f"[{idx}/{len(dates)}] {trade_date} SKIP already persisted")
            continue

        frame, status, message = fetch_mto_delivery_day(
            trade_date,
            session=session,
            retries=retries,
        )

        if status == "OK":
            temp = local_path.with_suffix(".tmp.parquet")
            frame.to_parquet(temp, index=False)
            temp.replace(local_path)
            out = store.upload_or_update(
                local_path,
                remote_path,
                mime_type="application/octet-stream",
            )
            ok += 1
            print(
                f"[{idx}/{len(dates)}] {trade_date} OK rows={len(frame):,} "
                f"drive_action={out.get('action')}"
            )
        elif status == "MISSING":
            missing += 1
            print(f"[{idx}/{len(dates)}] {trade_date} MISSING {message}")
        elif status == "EMPTY":
            empty += 1
            print(f"[{idx}/{len(dates)}] {trade_date} EMPTY {message}")
        else:
            raise RuntimeError(
                f"NSE delivery sync failed for {trade_date}: {message}"
            )

        time.sleep(max(0.0, sleep_seconds))

    print("=" * 90)
    print(
        f"DELIVERY SYNC COMPLETE: ok={ok} skipped={skipped} "
        f"missing={missing} empty={empty}"
    )
    print("=" * 90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback-days", type=int, default=14)
    ap.add_argument("--sleep-seconds", type=float, default=1.0)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    sync_delivery(
        lookback_days=max(1, args.lookback_days),
        sleep_seconds=max(0.0, args.sleep_seconds),
        retries=max(1, args.retries),
    )


if __name__ == "__main__":
    main()
