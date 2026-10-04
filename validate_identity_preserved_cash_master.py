#!/usr/bin/env python3
"""Read-only validation of the canonical NSE cash identity-preserved master.

The validator never rewrites the master. It checks row/identity integrity,
date coverage, symbol/series semantics, OHLCV sanity, and optional cross-layer
coverage. It is designed for Colab + Google Drive execution after the
identity-preserved NULL-series repair.

Exit status:
  0 = all hard gates pass
  1 = one or more hard gates fail
  2 = configuration / input error
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

DEFAULT_MASTER = Path(
    "/content/drive/MyDrive/Indian_Market_Research/production_data/"
    "nse_official/ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
)
DEFAULT_REPORT_DIR = Path(
    "/content/drive/MyDrive/Indian_Market_Research/diagnostics/"
    "canonical_cash_master_validation"
)

IDENTITY_COLUMNS = ("row_identity_key", "listing_id", "series", "symbol", "date")
NUMERIC_ALIASES = {
    "open": ("open", "open_price", "prev_open"),
    "high": ("high", "high_price", "prev_high"),
    "low": ("low", "low_price", "prev_low"),
    "close": ("close", "close_price", "last_price", "ltp"),
    "volume": ("volume", "trade_volume", "total_traded_quantity", "tottrdqty"),
    "delivery_qty": ("delivery_qty", "deliverable_qty", "delivery_quantity", "deliv_qty"),
    "delivery_pct": ("delivery_pct", "delivery_percentage", "deliverable_pct", "deliv_per"),
}
SERIES_EXPECTED = {"EQ", "BE", "BZ", "SM", "ST", "SZ", "GB", "RR", "NA"}

def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

def sql_str(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"

def resolve_col(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    exact = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in exact:
            return exact[alias.lower()]
    return None

def norm_symbol(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).strip().upper()
    for suffix in (".NS", ".BO", "-EQ", ".EQ"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s

def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False

def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument(
        "--expected-min-date", default="2016-01-04",
        help="Informational only unless the master starts later than this."
    )
    parser.add_argument(
        "--expected-max-date", default="2026-10-01",
        help="Informational only; future extensions are allowed."
    )
    parser.add_argument(
        "--fno-path",
        default="/content/drive/MyDrive/Indian_Market_Research/"
                "production_data/nse_official/fno/fno_daily_features_2026.parquet",
    )
    args = parser.parse_args()

    master = Path(args.master)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("NSE CASH — CANONICAL IDENTITY-PRESERVED MASTER VALIDATION")
    print("=" * 100)
    print("Master       :", master)
    print("Report dir   :", report_dir)

    if not master.exists() or master.stat().st_size == 0:
        print("STATUS       : FAIL — master missing or empty")
        return 2

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    schema = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_str(master)})"
    ).df()
    columns = [str(x) for x in schema["column_name"].tolist()]
    lower = {c.lower(): c for c in columns}

    required = {}
    for logical, aliases in {
        "symbol": ("symbol",),
        "date": ("date", "trade_date", "trading_date"),
        "series": ("series",),
        "listing_id": ("listing_id",),
        "row_identity_key": ("row_identity_key",),
    }.items():
        required[logical] = resolve_col(columns, aliases)

    missing_required = [k for k, v in required.items() if v is None]
    if missing_required:
        raise RuntimeError(f"Required master columns missing: {missing_required}")

    date_col = required["date"]
    symbol_col = required["symbol"]
    series_col = required["series"]
    listing_col = required["listing_id"]
    identity_col = required["row_identity_key"]

    numeric_cols = {
        k: resolve_col(columns, aliases) for k, aliases in NUMERIC_ALIASES.items()
    }

    total = int(con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_str(master)})"
    ).fetchone()[0])

    nulls = {}
    for logical, col in required.items():
        nulls[logical] = int(con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_str(master)}) "
            f"WHERE {qident(col)} IS NULL"
        ).fetchone()[0])

    dup_identity = int(con.execute(
        f"SELECT COUNT(*) FROM ("
        f" SELECT {qident(identity_col)}, COUNT(*) n"
        f" FROM read_parquet({sql_str(master)})"
        f" WHERE {qident(identity_col)} IS NOT NULL"
        f" GROUP BY {qident(identity_col)} HAVING COUNT(*) > 1"
        f")"
    ).fetchone()[0])

    invalid_identity = int(con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_str(master)}) "
        f"WHERE {qident(identity_col)} IS NULL OR TRIM(CAST({qident(identity_col)} AS VARCHAR)) = ''"
    ).fetchone()[0])

    min_date, max_date, distinct_dates = con.execute(
        f"SELECT MIN(TRY_CAST({qident(date_col)} AS DATE)), "
        f"MAX(TRY_CAST({qident(date_col)} AS DATE)), "
        f"COUNT(DISTINCT TRY_CAST({qident(date_col)} AS DATE)) "
        f"FROM read_parquet({sql_str(master)})"
    ).fetchone()
    invalid_date = int(con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_str(master)}) "
        f"WHERE TRY_CAST({qident(date_col)} AS DATE) IS NULL"
    ).fetchone()[0])

    distinct_symbols = int(con.execute(
        f"SELECT COUNT(DISTINCT NULLIF(TRIM(CAST({qident(symbol_col)} AS VARCHAR)), '')) "
        f"FROM read_parquet({sql_str(master)})"
    ).fetchone()[0])
    distinct_listing_ids = int(con.execute(
        f"SELECT COUNT(DISTINCT NULLIF(TRIM(CAST({qident(listing_col)} AS VARCHAR)), '')) "
        f"FROM read_parquet({sql_str(master)})"
    ).fetchone()[0])

    series_counts_df = con.execute(
        f"SELECT UPPER(TRIM(CAST({qident(series_col)} AS VARCHAR))) AS series, "
        f"COUNT(*) AS rows "
        f"FROM read_parquet({sql_str(master)}) "
        f"GROUP BY 1 ORDER BY 2 DESC"
    ).df()
    series_counts = {
        ("" if pd.isna(r["series"]) else str(r["series"])): int(r["rows"])
        for _, r in series_counts_df.iterrows()
    }

    unexpected_series = [
        s for s, n in series_counts.items()
        if s and s not in SERIES_EXPECTED
    ]
    blank_series = int(con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_str(master)}) "
        f"WHERE {qident(series_col)} IS NULL OR TRIM(CAST({qident(series_col)} AS VARCHAR)) = ''"
    ).fetchone()[0])

    # Symbol/date multiplicity is expected because a symbol can map to multiple
    # listing identities/series. It is reported, not treated as a defect.
    symbol_date_pairs_multi = int(con.execute(
        f"SELECT COUNT(*) FROM ("
        f" SELECT TRIM(CAST({qident(symbol_col)} AS VARCHAR)) symbol, "
        f" TRY_CAST({qident(date_col)} AS DATE) dt, COUNT(*) n"
        f" FROM read_parquet({sql_str(master)})"
        f" GROUP BY 1,2 HAVING COUNT(*) > 1"
        f")"
    ).fetchone()[0])

    price_checks = {}
    for logical in ("open", "high", "low", "close", "volume", "delivery_qty", "delivery_pct"):
        col = numeric_cols.get(logical)
        if not col:
            continue
        bad_numeric = int(con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_str(master)}) "
            f"WHERE {qident(col)} IS NOT NULL "
            f"AND TRY_CAST({qident(col)} AS DOUBLE) IS NULL"
        ).fetchone()[0])
        negative = int(con.execute(
            f"SELECT COUNT(*) FROM read_parquet({sql_str(master)}) "
            f"WHERE TRY_CAST({qident(col)} AS DOUBLE) < 0"
        ).fetchone()[0])
        price_checks[logical] = {
            "column": col,
            "non_numeric_non_null": bad_numeric,
            "negative_values": negative,
        }

    cross_price = {
        "high_below_low": None,
        "close_below_low": None,
        "close_above_high": None,
        "open_below_low": None,
        "open_above_high": None,
    }
    if all(numeric_cols.get(k) for k in ("open", "high", "low", "close")):
        oc, hc, lc, cc = (numeric_cols[k] for k in ("open", "high", "low", "close"))
        price_expr = (
            f" FROM read_parquet({sql_str(master)}) "
            f" WHERE TRY_CAST({qident(hc)} AS DOUBLE) IS NOT NULL "
            f" AND TRY_CAST({qident(lc)} AS DOUBLE) IS NOT NULL "
            f" AND TRY_CAST({qident(cc)} AS DOUBLE) IS NOT NULL"
        )
        cross_price["high_below_low"] = int(con.execute(
            f"SELECT COUNT(*){price_expr} "
            f"AND TRY_CAST({qident(hc)} AS DOUBLE) < TRY_CAST({qident(lc)} AS DOUBLE)"
        ).fetchone()[0])
        cross_price["close_below_low"] = int(con.execute(
            f"SELECT COUNT(*){price_expr} "
            f"AND TRY_CAST({qident(cc)} AS DOUBLE) < TRY_CAST({qident(lc)} AS DOUBLE)"
        ).fetchone()[0])
        cross_price["close_above_high"] = int(con.execute(
            f"SELECT COUNT(*){price_expr} "
            f"AND TRY_CAST({qident(cc)} AS DOUBLE) > TRY_CAST({qident(hc)} AS DOUBLE)"
        ).fetchone()[0])
        if numeric_cols.get("open"):
            cross_price["open_below_low"] = int(con.execute(
                f"SELECT COUNT(*){price_expr} "
                f"AND TRY_CAST({qident(oc)} AS DOUBLE) < TRY_CAST({qident(lc)} AS DOUBLE)"
            ).fetchone()[0])
            cross_price["open_above_high"] = int(con.execute(
                f"SELECT COUNT(*){price_expr} "
                f"AND TRY_CAST({qident(oc)} AS DOUBLE) > TRY_CAST({qident(hc)} AS DOUBLE)"
            ).fetchone()[0])

    # Validate the repaired NA series explicitly. It must exist as literal text,
    # not as NULL/blank. We do not assume every NA row is erroneous.
    na_rows = int(con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_str(master)}) "
        f"WHERE UPPER(TRIM(CAST({qident(series_col)} AS VARCHAR))) = 'NA'"
    ).fetchone()[0])
    na_null_identity = int(con.execute(
        f"SELECT COUNT(*) FROM read_parquet({sql_str(master)}) "
        f"WHERE UPPER(TRIM(CAST({qident(series_col)} AS VARCHAR))) = 'NA' "
        f"AND ({qident(identity_col)} IS NULL OR TRIM(CAST({qident(identity_col)} AS VARCHAR)) = '')"
    ).fetchone()[0])

    fno = {"exists": False, "path": str(args.fno_path), "cash_fno_symbol_overlap": None}
    fno_path = Path(args.fno_path)
    if fno_path.exists() and fno_path.stat().st_size > 0:
        try:
            fno_schema = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({sql_str(fno_path)})"
            ).df()
            fno_cols = [str(x) for x in fno_schema["column_name"].tolist()]
            fno_sym = resolve_col(fno_cols, ("symbol", "underlying", "undrlyg"))
            if fno_sym:
                cash_symbols_df = con.execute(
                    f"SELECT DISTINCT UPPER(TRIM(CAST({qident(symbol_col)} AS VARCHAR))) s "
                    f"FROM read_parquet({sql_str(master)}) "
                    f"WHERE {qident(symbol_col)} IS NOT NULL"
                ).df()
                fno_symbols_df = con.execute(
                    f"SELECT DISTINCT UPPER(TRIM(CAST({qident(fno_sym)} AS VARCHAR))) s "
                    f"FROM read_parquet({sql_str(fno_path)}) "
                    f"WHERE {qident(fno_sym)} IS NOT NULL"
                ).df()
                cash_set = set(cash_symbols_df["s"].astype(str))
                fno_set = set(fno_symbols_df["s"].astype(str))
                fno = {
                    "exists": True,
                    "path": str(fno_path),
                    "fno_symbol_count": len(fno_set),
                    "cash_fno_symbol_overlap": len(cash_set & fno_set),
                    "cash_symbols_with_fno_pct": pct(len(cash_set & fno_set), len(cash_set)),
                    "fno_symbol_column": fno_sym,
                }
            else:
                fno["status"] = "PRESENT_BUT_NO_SYMBOL_COLUMN"
        except Exception as exc:
            fno["status"] = f"AUDIT_ERROR: {type(exc).__name__}: {exc}"

    hard_failures = []
    if total <= 0:
        hard_failures.append("EMPTY_MASTER")
    if nulls["series"] != 0:
        hard_failures.append("NULL_SERIES_REMAINS")
    if nulls["row_identity_key"] != 0 or invalid_identity != 0:
        hard_failures.append("NULL_OR_BLANK_ROW_IDENTITY_REMAINS")
    if dup_identity != 0:
        hard_failures.append("DUPLICATE_ROW_IDENTITY_KEYS")
    if nulls["listing_id"] != 0:
        hard_failures.append("NULL_LISTING_ID_REMAINS")
    if invalid_date != 0:
        hard_failures.append("INVALID_DATES")
    if unexpected_series:
        hard_failures.append("UNEXPECTED_SERIES_CODES")
    for logical, chk in price_checks.items():
        if chk["non_numeric_non_null"] != 0:
            hard_failures.append(f"NON_NUMERIC_{logical.upper()}")
        if logical in {"open", "high", "low", "close", "volume", "delivery_qty"} and chk["negative_values"] != 0:
            hard_failures.append(f"NEGATIVE_{logical.upper()}")
    for k, n in cross_price.items():
        if n is not None and n != 0:
            hard_failures.append(k.upper())

    expected_min = pd.Timestamp(args.expected_min_date).date()
    expected_max = pd.Timestamp(args.expected_max_date).date()
    date_note = {
        "min_date": str(min_date) if min_date is not None else None,
        "max_date": str(max_date) if max_date is not None else None,
        "distinct_dates": int(distinct_dates),
        "expected_min_date": str(expected_min),
        "expected_max_date": str(expected_max),
        "starts_on_or_before_expected_min": bool(min_date is not None and min_date <= expected_min),
        "reaches_expected_max": bool(max_date is not None and max_date >= expected_max),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "master": str(master),
        "schema": columns,
        "rows": total,
        "identity": {
            "null_counts": nulls,
            "duplicate_identity_key_groups": dup_identity,
            "invalid_identity_rows": invalid_identity,
            "distinct_listing_ids": distinct_listing_ids,
            "distinct_symbols": distinct_symbols,
            "symbol_date_pairs_with_multiple_rows": symbol_date_pairs_multi,
            "note": "Multiple rows per symbol/date are permitted because listing/series identities are preserved.",
        },
        "dates": date_note,
        "series": {
            "counts": series_counts,
            "unexpected_series": unexpected_series,
            "blank_series_rows": blank_series,
            "literal_NA_rows": na_rows,
            "literal_NA_rows_with_null_identity": na_null_identity,
        },
        "numeric_columns": numeric_cols,
        "numeric_checks": price_checks,
        "cross_price_checks": cross_price,
        "cross_layer_fno": fno,
        "hard_failures": hard_failures,
        "status": "PASS" if not hard_failures else "FAIL",
    }

    report_path = report_dir / "identity_preserved_cash_master_validation_latest.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 100)
    print("CANONICAL CASH MASTER VALIDATION")
    print("=" * 100)
    print(f"Rows                    : {total:,}")
    print(f"Distinct symbols        : {distinct_symbols:,}")
    print(f"Distinct listing IDs    : {distinct_listing_ids:,}")
    print(f"Distinct trading dates : {distinct_dates:,}")
    print(f"Date range              : {min_date} -> {max_date}")
    print(f"NULL series             : {nulls['series']:,}")
    print(f"NULL listing_id         : {nulls['listing_id']:,}")
    print(f"NULL row_identity_key   : {nulls['row_identity_key']:,}")
    print(f"Duplicate identity keys : {dup_identity:,}")
    print(f"Blank/invalid identity  : {invalid_identity:,}")
    print(f"Literal NA series rows  : {na_rows:,}")
    print(f"NA rows with NULL ID    : {na_null_identity:,}")
    print(f"Unexpected series       : {unexpected_series}")
    print(f"Symbol/date multi-rows  : {symbol_date_pairs_multi:,} (reported, not a failure)")
    print(f"F&O layer available     : {fno.get('exists', False)}")
    if fno.get("cash_fno_symbol_overlap") is not None:
        print(f"Cash/F&O symbol overlap : {fno['cash_fno_symbol_overlap']:,}")
        print(f"Cash symbols with F&O % : {fno['cash_symbols_with_fno_pct']:.2f}%")
    print()
    print("Report                  :", report_path)
    if hard_failures:
        print("HARD FAILURES           :", ", ".join(hard_failures))
        print("STATUS                  : FAIL")
        return 1

    print("STATUS                  : PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
