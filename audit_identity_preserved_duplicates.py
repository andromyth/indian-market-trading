#!/usr/bin/env python3
"""
Read-only audit of duplicate row_identity_key values in the rebuilt NSE
identity-preserved cash master.

Purpose:
  Determine whether the 14,641 duplicate identity rows are:
    1) exact business duplicates that can be safely collapsed, or
    2) true data conflicts that require explicit resolution.

This script NEVER modifies the master.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

MASTER = Path(
    "/content/drive/MyDrive/Indian_Market_Research/"
    "production_data/nse_official/"
    "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
)
OUT = MASTER.parent / "cash_identity_duplicate_audit_2016_2026.csv"
SUMMARY = MASTER.parent / "cash_identity_duplicate_audit_2016_2026.json"

KEY = "row_identity_key"

BUSINESS_COLS = [
    "symbol", "date", "series", "isin",
    "economic_security_id", "listing_id", "security_class",
    "instrument_type", "source_format", "instrument_id",
    "instrument_name", "business_date", "segment", "source",
    "actual_expiry", "open", "high", "low", "close", "last",
    "prev_close", "last_price_zero", "last_price_for_research",
    "research_price", "volume", "traded_value", "num_trades",
]


def main():
    if not MASTER.exists():
        raise FileNotFoundError(f"Master not found: {MASTER}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    cols = ", ".join(f'"{c}"' for c in BUSINESS_COLS)

    print("=" * 100)
    print("NSE CASH — IDENTITY DUPLICATE AUDIT (READ-ONLY)")
    print("=" * 100)
    print("Master:", MASTER)
    print()

    total_rows = con.execute(
        f'SELECT COUNT(*) FROM read_parquet(?)',
        [str(MASTER)],
    ).fetchone()[0]

    dup_key_count = con.execute(
        f'''
        SELECT COUNT(*)
        FROM (
            SELECT "{KEY}"
            FROM read_parquet(?)
            WHERE "{KEY}" IS NOT NULL
            GROUP BY "{KEY}"
            HAVING COUNT(*) > 1
        )
        ''',
        [str(MASTER)],
    ).fetchone()[0]

    dup_row_count = con.execute(
        f'''
        SELECT COALESCE(SUM(n), 0)
        FROM (
            SELECT "{KEY}", COUNT(*) AS n
            FROM read_parquet(?)
            WHERE "{KEY}" IS NOT NULL
            GROUP BY "{KEY}"
            HAVING COUNT(*) > 1
        )
        ''',
        [str(MASTER)],
    ).fetchone()[0]

    print("Total master rows        :", f"{total_rows:,}")
    print("Duplicate identity keys  :", f"{dup_key_count:,}")
    print("Rows in duplicate groups :", f"{dup_row_count:,}")
    print()

    # A duplicate identity is an exact business duplicate when every business
    # field is identical within the group. raw_archive is intentionally
    # excluded because the same business row may have provenance differences.
    summary_sql = f'''
        WITH dup AS (
            SELECT *
            FROM read_parquet(?)
            QUALIFY COUNT(*) OVER (PARTITION BY "{KEY}") > 1
        ),
        g AS (
            SELECT
                "{KEY}" AS row_identity_key,
                COUNT(*) AS row_count,
                COUNT(DISTINCT md5(concat_ws('§',
                    {cols}
                ))) AS distinct_business_versions,
                COUNT(DISTINCT COALESCE(source_format, '')) AS source_formats,
                COUNT(DISTINCT COALESCE(raw_archive, '')) AS archives,
                MIN(date) AS date_min,
                MAX(date) AS date_max,
                MIN(symbol) AS symbol
            FROM dup
            GROUP BY "{KEY}"
        )
        SELECT *
        FROM g
        ORDER BY row_count DESC, row_identity_key
    '''

    detail = con.execute(summary_sql, [str(MASTER)]).df()
    detail.to_csv(OUT, index=False)

    if detail.empty:
        classification = {"duplicate_keys": 0}
        SUMMARY.write_text(json.dumps(classification, indent=2))
        print("No duplicate identity keys found.")
        return

    exact = int((detail["distinct_business_versions"] == 1).sum())
    conflicts = int((detail["distinct_business_versions"] > 1).sum())
    multi_archive = int((detail["archives"] > 1).sum())
    multi_source = int((detail["source_formats"] > 1).sum())

    rows_exact = int(
        detail.loc[
            detail["distinct_business_versions"] == 1,
            "row_count",
        ].sum()
    )
    rows_conflict = int(
        detail.loc[
            detail["distinct_business_versions"] > 1,
            "row_count",
        ].sum()
    )

    classification = {
        "master_rows": int(total_rows),
        "duplicate_identity_keys": int(dup_key_count),
        "duplicate_identity_rows": int(dup_row_count),
        "exact_business_duplicate_keys": exact,
        "exact_business_duplicate_rows": rows_exact,
        "conflicting_identity_keys": conflicts,
        "conflicting_identity_rows": rows_conflict,
        "duplicate_keys_spanning_multiple_archives": multi_archive,
        "duplicate_keys_spanning_multiple_source_formats": multi_source,
        "safe_to_collapse_exact_business_duplicates": bool(
            conflicts == 0
        ),
        "master": str(MASTER),
        "detail_csv": str(OUT),
    }

    SUMMARY.write_text(
        json.dumps(classification, indent=2),
        encoding="utf-8",
    )

    print("Exact business-duplicate keys :", f"{exact:,}")
    print("Exact business-duplicate rows :", f"{rows_exact:,}")
    print("Conflicting identity keys    :", f"{conflicts:,}")
    print("Conflicting identity rows    :", f"{rows_conflict:,}")
    print("Keys spanning archives        :", f"{multi_archive:,}")
    print("Keys spanning source formats  :", f"{multi_source:,}")
    print()
    print("SAFE TO COLLAPSE EXACT DUPLICATES:",
          "YES" if conflicts == 0 else "NO")
    print()
    print("Detail :", OUT)
    print("Summary:", SUMMARY)

    print()
    print("Top duplicate groups:")
    print(
        detail.head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
