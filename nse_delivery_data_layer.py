#!/usr/bin/env python3
"""Official NSE Security-wise Delivery Positions data layer.

This module keeps NSE delivery data separate from the identity-preserved cash
master. Historical delivery archives are read from the official NSE MTO
endpoint and persisted as one parquet file per trade date. The research loader
projects them to the same continuous cash-equity series policy used by the
canonical cash layer and deterministically resolves duplicate symbol/date
records.
"""
from __future__ import annotations

import csv
import io
import time
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import requests


DEFAULT_DELIVERY_ROOT = Path(
    "/content/drive/MyDrive/Indian_Market_Research/"
    "production_data/nse_official/delivery_full"
)

DELIVERY_ARCHIVE_BASE = (
    "https://nsearchives.nseindia.com/archives/equities/mto"
)

RESEARCH_SERIES = ("EQ", "BE", "BZ", "SM", "ST", "SZ")
SERIES_PRIORITY = {
    "EQ": 0,
    "BE": 1,
    "BZ": 2,
    "SM": 3,
    "ST": 4,
    "SZ": 5,
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/153.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
}


def normalize_delivery_symbol(value: object) -> str:
    s = str(value or "").strip().upper()
    for suffix in (".NS", ".BO", "-EQ", ".EQ"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def delivery_archive_url(trade_date: date | datetime | pd.Timestamp) -> str:
    d = pd.Timestamp(trade_date).date()
    return (
        f"{DELIVERY_ARCHIVE_BASE}/"
        f"MTO_{d:%d%m%Y}.DAT"
    )


def parse_mto_bytes(
    payload: bytes,
    trade_date: date | datetime | pd.Timestamp,
    *,
    archive_url: str | None = None,
) -> pd.DataFrame:
    """Parse official NSE MTO Security-wise Delivery Position rows.

    MTO data rows use record type 20:
      20, Sr No, Security, Series, Quantity Traded,
      Deliverable Quantity, % Deliverable
    """
    if not payload:
        return pd.DataFrame()

    text = payload.decode(
        "utf-8-sig",
        errors="replace",
    )

    rows: list[dict] = []
    d = pd.Timestamp(trade_date).normalize()

    for raw in csv.reader(io.StringIO(text)):
        if not raw or str(raw[0]).strip() != "20":
            continue
        if len(raw) < 7:
            continue

        symbol = normalize_delivery_symbol(raw[2])
        series = str(raw[3]).strip().upper()

        if not symbol:
            continue

        def to_num(value: object) -> float:
            try:
                return float(
                    str(value)
                    .strip()
                    .replace(",", "")
                )
            except Exception:
                return np.nan

        rows.append(
            {
                "symbol": symbol,
                "date": d,
                "series": series,
                "traded_qty": to_num(raw[4]),
                "delivery_qty": to_num(raw[5]),
                "delivery_pct": to_num(raw[6]),
                "source": "NSE_OFFICIAL_MTO",
                "archive_url": archive_url or "",
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    ).dt.normalize()

    for c in (
        "traded_qty",
        "delivery_qty",
        "delivery_pct",
    ):
        out[c] = pd.to_numeric(
            out[c],
            errors="coerce",
        )

    out = out.dropna(
        subset=["symbol", "date"]
    ).copy()

    # A malformed archive must never create duplicate symbol/date/series
    # delivery records in the persisted daily layer.
    out["_series_priority"] = (
        out["series"]
        .map(SERIES_PRIORITY)
        .fillna(99)
    )

    out = (
        out.sort_values(
            ["symbol", "date", "_series_priority", "series"]
        )
        .drop_duplicates(
            ["symbol", "date", "series"],
            keep="first",
        )
        .drop(columns=["_series_priority"])
        .reset_index(drop=True)
    )

    return out


def fetch_mto_delivery_day(
    trade_date: date | datetime | pd.Timestamp,
    *,
    session: requests.Session | None = None,
    retries: int = 3,
    timeout: int = 30,
) -> tuple[pd.DataFrame, str, str]:
    """Fetch one official NSE MTO archive.

    Returns (frame, status, message). A 404 is reported as MISSING rather than
    raising, because a canonical trading-date list can contain a date for which
    NSE did not publish a separate MTO archive.
    """
    url = delivery_archive_url(trade_date)
    s = session or requests.Session()
    s.headers.update(HEADERS)

    last_error = ""

    for attempt in range(1, retries + 1):
        try:
            r = s.get(
                url,
                timeout=timeout,
            )

            if r.status_code == 200:
                frame = parse_mto_bytes(
                    r.content,
                    trade_date,
                    archive_url=url,
                )

                if frame.empty:
                    return (
                        frame,
                        "EMPTY",
                        "HTTP 200 but no record-type-20 delivery rows",
                    )

                return (
                    frame,
                    "OK",
                    f"rows={len(frame):,}",
                )

            if r.status_code == 404:
                return (
                    pd.DataFrame(),
                    "MISSING",
                    "HTTP 404",
                )

            if r.status_code in (403, 429) or r.status_code >= 500:
                last_error = (
                    f"HTTP {r.status_code}"
                )
                time.sleep(
                    min(8.0, float(attempt) * 1.5)
                )
                continue

            return (
                pd.DataFrame(),
                "ERROR",
                f"HTTP {r.status_code}",
            )

        except requests.RequestException as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(
                min(8.0, float(attempt) * 1.5)
            )

    return (
        pd.DataFrame(),
        "ERROR",
        last_error or "unknown fetch error",
    )


def _parquet_columns(root: Path) -> set[str]:
    pattern = str(root / "delivery_*.parquet")
    con = duckdb.connect()
    try:
        df = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?, union_by_name=True)",
            [pattern],
        ).fetchdf()
        return set(
            df["column_name"].astype(str)
        )
    except Exception:
        return set()
    finally:
        con.close()


def load_delivery_history(
    root: str | Path = DEFAULT_DELIVERY_ROOT,
    *,
    symbols: Iterable[str] | None = None,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
) -> pd.DataFrame:
    """Load persisted NSE delivery history as a deterministic research view.

    The source files may have either the minimal live schema
    (symbol/date/delivery fields) or the richer historical schema including
    series. The result is always one row per symbol/date, restricted to the
    continuous research-series policy.
    """
    root = Path(root)
    files = list(root.glob("delivery_*.parquet"))
    if not files:
        return pd.DataFrame()

    columns = _parquet_columns(root)
    if not {"symbol", "date", "delivery_qty", "delivery_pct"}.issubset(
        columns
    ):
        return pd.DataFrame()

    select_cols = [
        "symbol",
        "date",
        "delivery_qty",
        "delivery_pct",
    ]

    optional = [
        "series",
        "traded_qty",
        "source",
        "archive_url",
    ]

    select_cols.extend(
        c for c in optional if c in columns
    )

    where = [
        "symbol IS NOT NULL",
        "TRY_CAST(date AS DATE) IS NOT NULL",
    ]
    params: list[object] = []

    if start is not None:
        where.append("TRY_CAST(date AS DATE) >= ?")
        params.append(pd.Timestamp(start).date())

    if end is not None:
        where.append("TRY_CAST(date AS DATE) <= ?")
        params.append(pd.Timestamp(end).date())

    if symbols is not None:
        wanted = sorted(
            {
                normalize_delivery_symbol(s)
                for s in symbols
                if str(s).strip()
            }
        )
        if not wanted:
            return pd.DataFrame()

        where.append(
            "UPPER(TRIM(CAST(symbol AS VARCHAR))) IN "
            f"({','.join('?' for _ in wanted)})"
        )
        params.extend(wanted)

    pattern = str(root / "delivery_*.parquet").replace(
        "'",
        "''",
    )

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM read_parquet(
            '{pattern}',
            union_by_name=True
        )
        WHERE {" AND ".join(where)}
    """

    con = duckdb.connect()
    try:
        out = con.execute(
            query,
            params,
        ).df()
    finally:
        con.close()

    if out.empty:
        return out

    out["symbol"] = (
        out["symbol"]
        .astype(str)
        .map(normalize_delivery_symbol)
    )
    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    ).dt.normalize()

    for c in (
        "delivery_qty",
        "delivery_pct",
        "traded_qty",
    ):
        if c in out.columns:
            out[c] = pd.to_numeric(
                out[c],
                errors="coerce",
            )

    if "series" in out.columns:
        out["series"] = (
            out["series"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        out["_series_priority"] = (
            out["series"]
            .map(SERIES_PRIORITY)
            .fillna(99)
        )

        # Keep only the research series. Empty series is retained as a
        # fallback for the two existing live delivery snapshots that predate
        # the richer historical parser.
        out = out[
            out["series"].isin(RESEARCH_SERIES)
            | out["series"].eq("")
        ].copy()

        out = (
            out.sort_values(
                [
                    "symbol",
                    "date",
                    "_series_priority",
                    "series",
                ]
            )
            .drop_duplicates(
                ["symbol", "date"],
                keep="first",
            )
            .drop(columns=["_series_priority"])
        )
    else:
        out = out.drop_duplicates(
            ["symbol", "date"],
            keep="last",
        )

    return out.reset_index(drop=True)


def canonical_trading_dates(
    cash_master: str | Path,
    *,
    start: date | datetime | None = None,
    end: date | datetime | None = None,
) -> list[date]:
    """Return distinct canonical cash research dates for resumable backfill."""
    path = Path(cash_master)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Canonical cash master not found: {path}"
        )

    where = [
        "UPPER(TRIM(CAST(series AS VARCHAR))) IN "
        "('EQ','BE','BZ','SM','ST','SZ')",
        "TRY_CAST(date AS DATE) IS NOT NULL",
    ]
    params: list[object] = []

    if start is not None:
        where.append("TRY_CAST(date AS DATE) >= ?")
        params.append(pd.Timestamp(start).date())

    if end is not None:
        where.append("TRY_CAST(date AS DATE) <= ?")
        params.append(pd.Timestamp(end).date())

    master = str(path).replace("'", "''")

    query = f"""
        SELECT DISTINCT CAST(date AS DATE) AS trade_date
        FROM read_parquet('{master}')
        WHERE {" AND ".join(where)}
        ORDER BY trade_date
    """

    con = duckdb.connect()
    try:
        df = con.execute(query, params).df()
    finally:
        con.close()

    return [
        pd.Timestamp(x).date()
        for x in df["trade_date"].tolist()
    ]
