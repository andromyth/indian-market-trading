#!/usr/bin/env python3
"""Safely reconcile the identity-preserved NSE cash master with recent official CM-UDiFF archives.

Default is a read-only dry run. Use --apply only after reviewing the report.
This script never trains/promotes models and never uses third-party prices.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from rebuild_nse_cash_history_identity_aware import (
    MASTER,
    META,
    add_identity,
    final_columns,
    normalize_udiff,
)

BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
NSE_ROOT = BASE / "production_data" / "nse_official"
BULK_ROOT = NSE_ROOT / "bulk_eod"
REPORT_PATH = BASE / "diagnostics" / "canonical_cash_master_validation" / "cash_master_incremental_refresh_latest.json"
ZIP_RE = re.compile(r"_(\d{8})_F_0000\.csv\.zip$", re.IGNORECASE)


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _master_stats(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Identity-preserved cash master is missing/empty: {path}")
    con = duckdb.connect(database=":memory:")
    try:
        q = _sql_path(path)
        row = con.execute(f"""
            SELECT COUNT(*),
                   COUNT(DISTINCT row_identity_key),
                   COUNT(*) FILTER (WHERE row_identity_key IS NULL),
                   COUNT(DISTINCT symbol),
                   COUNT(DISTINCT TRY_CAST(date AS DATE)),
                   MIN(TRY_CAST(date AS DATE)),
                   MAX(TRY_CAST(date AS DATE))
            FROM read_parquet({q})
        """).fetchone()
        return {
            "rows": int(row[0] or 0),
            "distinct_identity_keys": int(row[1] or 0),
            "null_identity_key_rows": int(row[2] or 0),
            "symbols": int(row[3] or 0),
            "trading_dates": int(row[4] or 0),
            "min_date": str(row[5]) if row[5] else None,
            "max_date": str(row[6]) if row[6] else None,
            "duplicate_identity_key_rows": int((row[0] or 0) - (row[1] or 0) - (row[2] or 0)),
        }
    finally:
        con.close()


def _source_archives(start: date, end: date) -> list[tuple[date, Path]]:
    if not BULK_ROOT.exists():
        return []
    found = []
    for path in BULK_ROOT.rglob("BhavCopy_NSE_CM_0_0_0_*_F_0000.csv.zip"):
        match = ZIP_RE.search(path.name)
        if not match:
            continue
        try:
            day = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if start <= day <= end:
            found.append((day, path))
    return sorted(found, key=lambda item: (item[0], str(item[1])))


def _read_identity_rows(archive: Path, expected_date: date) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/") and n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV payload in official NSE archive: {archive}")
        member = next((n for n in names if "bhav" in Path(n).name.lower()), names[0])
        import io
        raw = pd.read_csv(io.BytesIO(zf.read(member)), encoding="utf-8-sig")
    rows = normalize_udiff(raw, archive)
    if rows.empty:
        return pd.DataFrame(columns=final_columns(pd.DataFrame()).columns)
    rows = rows.dropna(subset=["symbol", "series", "date", "close"]).copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=["date"])
    actual_dates = sorted(pd.to_datetime(rows["date"]).dt.date.unique().tolist())
    if actual_dates and actual_dates != [expected_date]:
        raise RuntimeError(
            f"Archive/date mismatch for {archive.name}: expected {expected_date}, got {actual_dates[:5]}"
        )
    rows = add_identity(rows)
    rows["raw_archive"] = str(archive)
    rows = final_columns(rows)
    if rows["row_identity_key"].isna().any():
        raise RuntimeError(f"Archive generated null row identity keys: {archive.name}")
    return rows.drop_duplicates("row_identity_key", keep="last").reset_index(drop=True)


def _merge_master(master: Path, incoming: Path, output: Path) -> dict[str, Any]:
    con = duckdb.connect(database=":memory:")
    spill_dir = Path(tempfile.gettempdir()) / "cash_master_refresh_duckdb_tmp"
    spill_dir.mkdir(parents=True, exist_ok=True)
    try:
        con.execute("PRAGMA threads=2")
        con.execute("PRAGMA memory_limit='4GB'")
        con.execute(f"SET temp_directory={_sql_path(spill_dir)}")
        m_sql, i_sql, o_sql = _sql_path(master), _sql_path(incoming), _sql_path(output)
        columns = [
            row[0] for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({m_sql})"
            ).fetchall()
        ]
        if "row_identity_key" not in columns:
            raise RuntimeError("Existing master lacks row_identity_key; refusing to merge.")
        incoming_rows = int(con.execute(f"SELECT COUNT(*) FROM read_parquet({i_sql})").fetchone()[0])
        new_keys = int(con.execute(f"""
            SELECT COUNT(*) FROM (
              SELECT DISTINCT i.row_identity_key
              FROM read_parquet({i_sql}) i
              LEFT JOIN read_parquet({m_sql}) m USING (row_identity_key)
              WHERE m.row_identity_key IS NULL
            )
        """).fetchone()[0])
        overlapping_keys = int(con.execute(f"""
            SELECT COUNT(DISTINCT i.row_identity_key)
            FROM read_parquet({i_sql}) i
            INNER JOIN read_parquet({m_sql}) m USING (row_identity_key)
        """).fetchone()[0])
        projection = ", ".join(_quote_ident(c) for c in columns)
        con.execute(f"""
            COPY (
              WITH combined AS (
                SELECT *, 0 AS _refresh_priority FROM read_parquet({m_sql})
                UNION ALL BY NAME
                SELECT *, 1 AS _refresh_priority FROM read_parquet({i_sql})
              ),
              ranked AS (
                SELECT *,
                  ROW_NUMBER() OVER (
                    PARTITION BY row_identity_key
                    ORDER BY _refresh_priority DESC
                  ) AS _identity_rank
                FROM combined
              )
              SELECT {projection}
              FROM ranked
              WHERE _identity_rank = 1
              ORDER BY TRY_CAST(date AS DATE), listing_id
            )
            TO {o_sql} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        stats = _master_stats(output)
        if stats["null_identity_key_rows"] or stats["duplicate_identity_key_rows"]:
            raise RuntimeError(f"Merged master failed identity-key validation: {stats}")
        return {
            "incoming_rows": incoming_rows,
            "new_identity_keys": new_keys,
            "overlapping_identity_keys_refreshed_from_official_archive": overlapping_keys,
            "output_stats": stats,
        }
    finally:
        con.close()


def main() -> int:
    global BULK_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default=str(MASTER))
    parser.add_argument("--bulk-root", default=str(BULK_ROOT))
    parser.add_argument("--start", default="", help="First source date YYYY-MM-DD; defaults to master max date minus lookback.")
    parser.add_argument("--end", default="", help="Last source date YYYY-MM-DD; defaults to today's India date.")
    parser.add_argument("--lookback-calendar-days", type=int, default=30)
    parser.add_argument("--apply", action="store_true", help="Write validated update to canonical master. Default is dry-run.")
    args = parser.parse_args()

    BULK_ROOT = Path(args.bulk_root)
    master = Path(args.master)
    before = _master_stats(master)
    if args.lookback_calendar_days < 1:
        raise ValueError("--lookback-calendar-days must be >= 1")
    end_date = date.fromisoformat(args.end) if args.end else datetime.now(ZoneInfo("Asia/Kolkata")).date()
    default_start = date.fromisoformat(before["max_date"]) - timedelta(days=args.lookback_calendar_days)
    start_date = date.fromisoformat(args.start) if args.start else default_start
    if start_date > end_date:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")

    archives = _source_archives(start_date, end_date)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "DRY_RUN" if not args.apply else "APPLY_REQUESTED",
        "master_path": str(master),
        "bulk_root": str(BULK_ROOT),
        "source_window": {"start": str(start_date), "end": str(end_date)},
        "master_before": before,
        "official_archive_files_found": len(archives),
        "official_archive_dates_found": sorted({str(d) for d, _ in archives}),
        "official_archive_dates_after_master_max": sorted({str(d) for d, _ in archives if str(d) > str(before["max_date"])}),
        "archives_loaded": [],
        "warnings": [],
        "write_performed": False,
        "model_weights_changed": False,
        "model_promotion_performed": False,
    }
    if not archives:
        report["status"] = "BLOCKED_NO_OFFICIAL_ZIP_ARCHIVES"
        report["warnings"].append(
            "No official CM-UDiFF ZIP archives were found in the requested window. "
            "Do not rebuild from normalized symbol/date-only Parquet because it may have lost listing identity."
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps(report, indent=2, default=str))
        return 2

    parsed = []
    with tempfile.TemporaryDirectory(prefix="cash-master-refresh-") as tmp_dir:
        temp_root = Path(tmp_dir)
        for day, archive in archives:
            rows = _read_identity_rows(archive, day)
            report["archives_loaded"].append({
                "date": str(day), "file": str(archive), "rows": int(len(rows))
            })
            if not rows.empty:
                parsed.append(rows)
        if not parsed:
            report["status"] = "BLOCKED_NO_VALID_IDENTITY_ROWS"
            report["warnings"].append("Official archives were found, but none produced validated identity-preserved equity rows.")
        else:
            incoming = pd.concat(parsed, ignore_index=True, sort=False)
            incoming = incoming.drop_duplicates("row_identity_key", keep="last")
            incoming_path = temp_root / "incremental_identity_rows.parquet"
            incoming.to_parquet(incoming_path, index=False, compression="zstd")
            report["incoming_unique_identity_rows"] = int(len(incoming))
            report["incoming_min_date"] = str(pd.to_datetime(incoming["date"]).min().date())
            report["incoming_max_date"] = str(pd.to_datetime(incoming["date"]).max().date())

            # Count candidate new listing/date keys without altering the master.
            con = duckdb.connect(database=":memory:")
            try:
                new_keys = int(con.execute(f"""
                    SELECT COUNT(*) FROM (
                      SELECT DISTINCT i.row_identity_key
                      FROM read_parquet({_sql_path(incoming_path)}) i
                      LEFT JOIN read_parquet({_sql_path(master)}) m USING (row_identity_key)
                      WHERE m.row_identity_key IS NULL
                    )
                """).fetchone()[0])
                overlap = int(con.execute(f"""
                    SELECT COUNT(DISTINCT i.row_identity_key)
                    FROM read_parquet({_sql_path(incoming_path)}) i
                    INNER JOIN read_parquet({_sql_path(master)}) m USING (row_identity_key)
                """).fetchone()[0])
            finally:
                con.close()
            report["new_identity_keys_estimate"] = new_keys
            report["overlapping_identity_keys_to_refresh"] = overlap
            if args.apply and (new_keys or overlap):
                tmp_master = master.with_name(master.stem + ".incremental.tmp" + master.suffix)
                if tmp_master.exists():
                    tmp_master.unlink()
                merged = _merge_master(master, incoming_path, tmp_master)
                expected_max = max(pd.to_datetime(incoming["date"]).dt.date)
                if date.fromisoformat(merged["output_stats"]["max_date"]) < expected_max:
                    raise RuntimeError(
                        f"Merged master max date {merged['output_stats']['max_date']} is older than incoming max {expected_max}"
                    )
                backup_name = master.stem + ".before_incremental_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + master.suffix
                backup = master.with_name(backup_name)
                shutil.copy2(master, backup)
                tmp_master.replace(master)
                after = _master_stats(master)
                report["status"] = "APPLIED_AND_VALIDATED"
                report["write_performed"] = True
                report["backup_path"] = str(backup)
                report["merge"] = merged
                report["master_after"] = after
                report["metadata_path"] = str(META)
                if META.exists():
                    meta = json.loads(META.read_text(encoding="utf-8"))
                else:
                    meta = {}
                meta.update({
                    "last_incremental_refresh_utc": datetime.now(timezone.utc).isoformat(),
                    "incremental_refresh_source": "NSE_OFFICIAL_CM_UDIFF_BHAVCOPY_ZIP",
                    "incremental_refresh_start_date": str(start_date),
                    "incremental_refresh_end_date": str(end_date),
                    "rows": after["rows"],
                    "symbols": after["symbols"],
                    "dates": after["trading_dates"],
                    "min_date": after["min_date"],
                    "max_date": after["max_date"],
                    "duplicate_row_identity_keys": after["duplicate_identity_key_rows"],
                    "master": str(master),
                })
                META.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            elif args.apply:
                report["status"] = "NO_CHANGES_NEEDED"
                report["warnings"].append("All recent source identity keys already exist; master was not rewritten.")
            else:
                report["status"] = "DRY_RUN_READY"
                report["warnings"].append(
                    "No write was performed. Review the source archive dates and identity-key counts, then rerun with --apply to update the canonical master."
                )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["status"] in {"DRY_RUN_READY", "APPLIED_AND_VALIDATED", "NO_CHANGES_NEEDED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
