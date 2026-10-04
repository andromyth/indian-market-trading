#!/usr/bin/env python3
"""Resumable Google Colab backfill of official NSE Security-wise Delivery data."""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from nse_delivery_data_layer import (
    DEFAULT_DELIVERY_ROOT,
    canonical_trading_dates,
    fetch_mto_delivery_day,
)


DEFAULT_MASTER = Path(
    "/content/drive/MyDrive/Indian_Market_Research/"
    "production_data/nse_official/"
    "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
)


def append_status(
    path: Path,
    trade_date,
    status: str,
    message: str,
    rows: int,
) -> None:
    exists = path.exists() and path.stat().st_size > 0

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.writer(fh)

        if not exists:
            writer.writerow(
                [
                    "trade_date",
                    "status",
                    "rows",
                    "message",
                    "updated_at_utc",
                ]
            )

        writer.writerow(
            [
                pd.Timestamp(trade_date).date().isoformat(),
                status,
                int(rows),
                message,
                datetime.utcnow().isoformat(),
            ]
        )


def existing_done_dates(status_path: Path) -> set:
    if not status_path.exists() or status_path.stat().st_size == 0:
        return set()

    try:
        d = pd.read_csv(status_path)
        if "trade_date" not in d.columns:
            return set()

        ok = d[
            d["status"].astype(str).isin(
                ["OK", "EMPTY", "MISSING"]
            )
        ]

        return set(
            pd.to_datetime(
                ok["trade_date"],
                errors="coerce",
            ).dt.date.dropna()
        )

    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cash-master",
        default=str(DEFAULT_MASTER),
    )
    ap.add_argument(
        "--delivery-root",
        default=str(DEFAULT_DELIVERY_ROOT),
    )
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N not-yet-completed dates.",
    )
    ap.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=3,
    )
    args = ap.parse_args()

    delivery_root = Path(args.delivery_root)
    delivery_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    status_path = (
        delivery_root /
        "delivery_backfill_status.csv"
    )

    start = (
        pd.Timestamp(args.start).date()
        if args.start
        else None
    )

    end = (
        pd.Timestamp(args.end).date()
        if args.end
        else None
    )

    dates = canonical_trading_dates(
        args.cash_master,
        start=start,
        end=end,
    )

    done = existing_done_dates(status_path)

    pending = [
        d for d in dates
        if d not in done
        and not (
            delivery_root /
            f"delivery_{d:%Y%m%d}.parquet"
        ).exists()
    ]

    if args.limit > 0:
        pending = pending[: args.limit]

    print("=" * 90)
    print("NSE OFFICIAL SECURITY-WISE DELIVERY BACKFILL")
    print("=" * 90)
    print(f"Canonical dates : {len(dates):,}")
    print(f"Already done    : {len(done):,}")
    print(f"Pending         : {len(pending):,}")
    print(f"Delivery root   : {delivery_root}")

    session = requests.Session()

    for i, trade_date in enumerate(pending, 1):

        print(
            f"
[{i:,}/{len(pending):,}] "
            f"{trade_date}",
            flush=True,
        )

        frame, status, message = fetch_mto_delivery_day(
            trade_date,
            session=session,
            retries=args.retries,
        )

        if status == "OK":
            path = (
                delivery_root /
                f"delivery_{trade_date:%Y%m%d}.parquet"
            )

            temp = path.with_suffix(".tmp.parquet")

            frame.to_parquet(
                temp,
                index=False,
            )

            temp.replace(path)

            print(
                f"  OK rows={len(frame):,} "
                f"file={path.name}",
                flush=True,
            )

            append_status(
                status_path,
                trade_date,
                status,
                message,
                len(frame),
            )

        elif status == "MISSING":
            print(
                f"  MISSING {message}",
                flush=True,
            )

            append_status(
                status_path,
                trade_date,
                status,
                message,
                0,
            )

        elif status == "EMPTY":
            print(
                f"  EMPTY {message}",
                flush=True,
            )

            append_status(
                status_path,
                trade_date,
                status,
                message,
                0,
            )

        else:
            print(
                f"  ERROR {message}",
                flush=True,
            )

            append_status(
                status_path,
                trade_date,
                status,
                message,
                0,
            )

        time.sleep(
            max(0.0, args.sleep_seconds)
        )

    print("
" + "=" * 90)
    print("BACKFILL RUN COMPLETE")
    print("=" * 90)
    print(
        f"Status file: {status_path}"
    )


if __name__ == "__main__":
    main()
