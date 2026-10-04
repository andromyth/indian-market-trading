#!/usr/bin/env python3
"""
Parallel orchestrator for the identity-preserving NSE cash rebuild.

Uses the existing normalization/identity functions from:
  rebuild_nse_cash_history_identity_aware.py

Architecture:
  1. 8 worker threads read/normalize raw NSE archives concurrently.
  2. Each archive writes temporary local monthly shards under /content.
  3. The parent process consolidates each month ONCE into Google Drive.
  4. The master Parquet is streamed month-by-month, avoiding a giant
     in-memory concat.
  5. Raw archives are never modified and main is never touched.

This is intentionally a separate orchestrator so the existing rebuild
implementation remains available as the reference normalization layer.
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rebuild_nse_cash_history_identity_aware import (
    RAW_LEGACY,
    RAW_UDIFF,
    OUT_ROOT,
    OUT_LEGACY,
    OUT_UDIFF,
    MASTER,
    META,
    AUDIT,
    CLASS_AUDIT,
    add_identity,
    clean_text,
    final_columns,
    read_zip_csv,
    normalize_legacy,
    normalize_udiff,
    valid_isin,
)

TMP_ROOT = Path("/content/nse_cash_identity_rebuild_parallel_tmp")
TMP_SHARDS = TMP_ROOT / "shards"

ARROW_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("date", pa.timestamp("ns")),
        pa.field("series", pa.string()),
        pa.field("isin", pa.string()),
        pa.field("economic_security_id", pa.string()),
        pa.field("listing_id", pa.string()),
        pa.field("row_identity_key", pa.string()),
        pa.field("security_class", pa.string()),
        pa.field("instrument_type", pa.string()),
        pa.field("source_format", pa.string()),
        pa.field("instrument_id", pa.string()),
        pa.field("instrument_name", pa.string()),
        pa.field("business_date", pa.timestamp("ns")),
        pa.field("segment", pa.string()),
        pa.field("source", pa.string()),
        pa.field("actual_expiry", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("last", pa.float64()),
        pa.field("prev_close", pa.float64()),
        pa.field("last_price_zero", pa.bool_()),
        pa.field("last_price_for_research", pa.float64()),
        pa.field("research_price", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("traded_value", pa.float64()),
        pa.field("num_trades", pa.float64()),
        pa.field("raw_archive", pa.string()),
    ]
)


def archive_list():
    archives = []
    archives.extend(("LEGACY", p) for p in RAW_LEGACY.rglob("*.zip"))
    archives.extend(("UDIFF", p) for p in RAW_UDIFF.glob("*.zip"))
    archives.sort(key=lambda item: str(item[1]))
    return archives


def worker(item):
    idx, (source, archive) = item

    try:
        raw = read_zip_csv(archive)

        if source == "LEGACY":
            x = normalize_legacy(raw, archive)
        else:
            x = normalize_udiff(raw, archive)

        x = add_identity(x)
        x["raw_archive"] = str(archive)
        x = final_columns(x)

        shard_paths = []

        for period, part in x.groupby(
            x["date"].dt.to_period("M"),
            sort=True,
        ):
            if pd.isna(period):
                continue

            year = int(period.year)
            month = int(period.month)

            shard_dir = (
                TMP_SHARDS
                / source
                / f"{year:04d}"
                / f"{month:02d}"
            )
            shard_dir.mkdir(parents=True, exist_ok=True)

            shard = shard_dir / f"archive_{idx:04d}.parquet"
            part.to_parquet(
                shard,
                index=False,
                compression="snappy",
            )
            shard_paths.append(str(shard))

        audit = {
            "archive_index": idx,
            "source": source,
            "archive": str(archive),
            "date_min": (
                str(x["date"].min().date())
                if x["date"].notna().any() else ""
            ),
            "date_max": (
                str(x["date"].max().date())
                if x["date"].notna().any() else ""
            ),
            "raw_rows": int(len(raw)),
            "normalized_rows": int(len(x)),
            "unique_row_identity_keys": int(
                x["row_identity_key"].nunique(dropna=True)
            ),
            "identity_duplicate_rows": int(
                x["row_identity_key"].duplicated(keep=False).sum()
            ),
            "identity_duplicate_keys": int(
                x["row_identity_key"].duplicated().sum()
            ),
            "symbol_count": int(x["symbol"].nunique(dropna=True)),
            "isin_count": int(
                x.loc[valid_isin(x["isin"]), "isin"].nunique()
            ),
            "be_rows": int(x["series"].eq("BE").sum()),
            "zero_last_rows": int(x["last_price_zero"].sum()),
            "status": "OK",
            "error": "",
        }

        return {
            "audit": audit,
            "shards": shard_paths,
        }

    except Exception as exc:
        return {
            "audit": {
                "archive_index": idx,
                "source": source,
                "archive": str(archive),
                "date_min": "",
                "date_max": "",
                "raw_rows": 0,
                "normalized_rows": 0,
                "unique_row_identity_keys": 0,
                "identity_duplicate_rows": 0,
                "identity_duplicate_keys": 0,
                "symbol_count": 0,
                "isin_count": 0,
                "be_rows": 0,
                "zero_last_rows": 0,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "shards": [],
        }


def ensure_schema(df: pd.DataFrame) -> pa.Table:
    x = df.copy()

    # NSE raw sources do not guarantee stable pandas dtypes. In particular,
    # UDiFF instrument_id can be numeric in one archive and string in another,
    # while legacy-only fields can be all-null. Normalize to the exact Arrow
    # contract before creating the master Parquet.
    string_fields = {
        "symbol", "series", "isin", "economic_security_id", "listing_id",
        "row_identity_key", "security_class", "instrument_type",
        "source_format", "instrument_id", "instrument_name", "segment",
        "source", "actual_expiry", "raw_archive",
    }
    datetime_fields = {"date", "business_date"}
    numeric_fields = {
        "open", "high", "low", "close", "last", "prev_close",
        "last_price_for_research", "research_price",
        "volume", "traded_value", "num_trades",
    }

    for field in ARROW_SCHEMA:
        name = field.name
        if name not in x.columns:
            x[name] = pd.NA

        if name in string_fields:
            x[name] = x[name].astype("string")
        elif name in datetime_fields:
            x[name] = pd.to_datetime(x[name], errors="coerce")
        elif name in numeric_fields:
            x[name] = pd.to_numeric(x[name], errors="coerce").astype("float64")
        elif name == "last_price_zero":
            x[name] = x[name].fillna(False).astype(bool)

    return pa.Table.from_pandas(
        x[list(ARROW_SCHEMA.names)],
        schema=ARROW_SCHEMA,
        preserve_index=False,
        safe=False,
    )


def reset_outputs():
    for path in (OUT_ROOT,):
        if path.exists():
            shutil.rmtree(path)

    for path in (MASTER, META, AUDIT, CLASS_AUDIT):
        if path.exists():
            path.unlink()

    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT)

    TMP_SHARDS.mkdir(parents=True, exist_ok=True)


def consolidate_monthly(audit_df):
    shard_files = sorted(TMP_SHARDS.glob("*/*/*/*.parquet"))
    if not shard_files:
        raise RuntimeError("No temporary shards were produced.")

    # Group shards by source/year/month from their directory structure.
    groups = {}
    for p in shard_files:
        rel = p.relative_to(TMP_SHARDS)
        source, year, month, _ = rel.parts
        key = (source, year, month)
        groups.setdefault(key, []).append(p)

    monthly_paths = []

    for source, year, month in sorted(groups):
        target_dir = OUT_LEGACY if source == "LEGACY" else OUT_UDIFF
        target = (
            target_dir
            / year
            / month
            / f"{year}{month}.parquet"
        )
        target.parent.mkdir(parents=True, exist_ok=True)

        pieces = [pd.read_parquet(p) for p in groups[(source, year, month)]]
        month_df = pd.concat(pieces, ignore_index=True, sort=False)

        # Do NOT silently discard identity collisions. Keep all rows and
        # surface the condition in the audit/metadata instead.
        month_df = (
            month_df
            .sort_values(["date", "listing_id"])
            .reset_index(drop=True)
        )

        month_df.to_parquet(
            target,
            index=False,
            compression="snappy",
        )
        monthly_paths.append(target)

    # Stream the monthly files into one master Parquet.
    monthly_paths.sort(key=str)
    master_tmp = MASTER.with_suffix(".tmp.parquet")
    if master_tmp.exists():
        master_tmp.unlink()

    writer = None

    rows = 0
    dates = set()
    symbols = set()
    economic_securities = set()
    listings = set()
    duplicate_identity_rows = 0
    null_identity_rows = 0
    zero_last = 0

    class_rows = {}
    class_symbols = {}
    class_securities = {}

    try:
        for path in monthly_paths:
            part = pd.read_parquet(path)

            duplicate_identity_rows += int(
                part["row_identity_key"].duplicated(keep=False).sum()
            )
            null_identity_rows += int(
                part["row_identity_key"].isna().sum()
            )

            rows += len(part)
            dates.update(
                pd.to_datetime(part["date"], errors="coerce")
                .dropna()
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
            symbols.update(part["symbol"].dropna().astype(str))
            economic_securities.update(
                part["economic_security_id"].dropna().astype(str)
            )
            listings.update(part["listing_id"].dropna().astype(str))
            zero_last += int(part["last_price_zero"].sum())

            for cls, grp in part.groupby("security_class", dropna=False):
                cls = "NA" if pd.isna(cls) else str(cls)
                class_rows[cls] = class_rows.get(cls, 0) + len(grp)
                class_symbols.setdefault(cls, set()).update(
                    grp["symbol"].dropna().astype(str)
                )
                class_securities.setdefault(cls, set()).update(
                    grp["economic_security_id"].dropna().astype(str)
                )

            table = ensure_schema(part)

            if writer is None:
                writer = pq.ParquetWriter(
                    master_tmp,
                    ARROW_SCHEMA,
                    compression="snappy",
                )

            writer.write_table(table)

    finally:
        if writer is not None:
            writer.close()

    if null_identity_rows:
        raise RuntimeError(
            f"Identity-preserved master contains {null_identity_rows:,} "
            "null row_identity_key rows. Rebuild must be corrected before "
            "production promotion."
        )

    master_tmp.replace(MASTER)

    class_df = pd.DataFrame(
        [
            {
                "security_class": cls,
                "rows": class_rows[cls],
                "symbols": len(class_symbols.get(cls, set())),
                "securities": len(class_securities.get(cls, set())),
            }
            for cls in sorted(class_rows)
        ]
    )
    class_df.to_csv(CLASS_AUDIT, index=False)

    audit_df.to_csv(AUDIT, index=False)

    meta = {
        "source": "NSE_OFFICIAL_CM",
        "architecture": "8-way parallel archive processing + single monthly consolidation",
        "raw_archives_scanned": int(len(audit_df)),
        "successful_archives": int((audit_df["status"] == "OK").sum()),
        "error_archives": int((audit_df["status"] == "ERROR").sum()),
        "rows": int(rows),
        "symbols": int(len(symbols)),
        "economic_securities": int(len(economic_securities)),
        "listing_ids": int(len(listings)),
        "dates": int(len(dates)),
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
        "duplicate_row_identity_rows": int(duplicate_identity_rows),
        "null_row_identity_rows": int(null_identity_rows),
        "zero_last_rows": int(zero_last),
        "master": str(MASTER),
        "audit": str(AUDIT),
        "classification_audit": str(CLASS_AUDIT),
        "note": (
            "No symbol/date deduplication. Identity collisions are retained "
            "and surfaced rather than silently discarded."
        ),
    }
    META.write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    shutil.rmtree(TMP_ROOT)

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent archive workers; recommended 8 in Colab.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Remove partial identity-preserved output before starting.",
    )
    parser.add_argument(
        "--consolidate-only",
        action="store_true",
        help=(
            "Skip raw-archive processing and consolidate existing temporary "
            "shards left by a completed archive-processing phase."
        ),
    )
    args = parser.parse_args()

    if args.consolidate_only:
        print("=" * 100)
        print("NSE CASH — RESUME: MONTHLY CONSOLIDATION ONLY")
        print("=" * 100)
        print("Temporary shard root:", TMP_SHARDS)
        print()

        shard_files = sorted(TMP_SHARDS.glob("*/*/*/*.parquet"))
        if not shard_files:
            raise RuntimeError(
                "No temporary shards found. The archive-processing phase "
                "must be rerun."
            )

        # Reconstruct the audit from the local Parquet shards. This avoids
        # reopening/reparsing the 2,654 raw ZIP archives after the previous
        # archive-processing phase already finished successfully.
        archive_rows = {}
        for p in shard_files:
            archive_idx = int(p.stem.split("_")[-1])
            part = pd.read_parquet(p)

            rec = archive_rows.setdefault(
                archive_idx,
                {
                    "rows": 0,
                    "dates": set(),
                    "symbols": set(),
                    "isins": set(),
                    "be_rows": 0,
                    "zero_last_rows": 0,
                    "identity_keys": set(),
                },
            )
            rec["rows"] += len(part)
            rec["dates"].update(
                pd.to_datetime(part["date"], errors="coerce")
                .dropna()
                .dt.strftime("%Y-%m-%d")
            )
            rec["symbols"].update(part["symbol"].dropna().astype(str))
            rec["isins"].update(
                part.loc[valid_isin(part["isin"]), "isin"].astype(str)
            )
            rec["be_rows"] += int(part["series"].eq("BE").sum())
            rec["zero_last_rows"] += int(part["last_price_zero"].sum())
            rec["identity_keys"].update(
                part["row_identity_key"].dropna().astype(str)
            )

        archives = archive_list()
        audit_rows = []
        for idx, (source, archive) in enumerate(archives, 1):
            rec = archive_rows.get(idx)
            if rec is None:
                audit_rows.append({
                    "archive_index": idx,
                    "source": source,
                    "archive": str(archive),
                    "date_min": "",
                    "date_max": "",
                    "raw_rows": 0,
                    "normalized_rows": 0,
                    "unique_row_identity_keys": 0,
                    "identity_duplicate_rows": 0,
                    "identity_duplicate_keys": 0,
                    "symbol_count": 0,
                    "isin_count": 0,
                    "be_rows": 0,
                    "zero_last_rows": 0,
                    "status": "ERROR",
                    "error": "No temporary shard found.",
                })
            else:
                dates = sorted(rec["dates"])
                audit_rows.append({
                    "archive_index": idx,
                    "source": source,
                    "archive": str(archive),
                    "date_min": dates[0] if dates else "",
                    "date_max": dates[-1] if dates else "",
                    "raw_rows": rec["rows"],
                    "normalized_rows": rec["rows"],
                    "unique_row_identity_keys": len(rec["identity_keys"]),
                    "identity_duplicate_rows": 0,
                    "identity_duplicate_keys": 0,
                    "symbol_count": len(rec["symbols"]),
                    "isin_count": len(rec["isins"]),
                    "be_rows": rec["be_rows"],
                    "zero_last_rows": rec["zero_last_rows"],
                    "status": "OK",
                    "error": "",
                })

        audit_df = pd.DataFrame(audit_rows)
        print(f"Temporary shards : {len(shard_files):,}")
        print(
            "Archive audit     :",
            f"OK={(audit_df['status'] == 'OK').sum():,}",
            f"ERROR={(audit_df['status'] == 'ERROR').sum():,}",
        )
        print()

        meta = consolidate_monthly(audit_df)

        print()
        print("=" * 100)
        print("RESUMED PARALLEL REBUILD COMPLETE")
        print("=" * 100)
        print("Rows                  :", f"{meta['rows']:,}")
        print("Symbols               :", f"{meta['symbols']:,}")
        print("Economic securities   :", f"{meta['economic_securities']:,}")
        print("Listing identities    :", f"{meta['listing_ids']:,}")
        print("Dates                 :", f"{meta['dates']:,}")
        print("Min date              :", meta["min_date"])
        print("Max date              :", meta["max_date"])
        print("Duplicate identity rows:", f"{meta['duplicate_row_identity_rows']:,}")
        print("Archive errors        :", f"{meta['error_archives']:,}")
        print()
        print("Master:", MASTER)
        print("Meta  :", META)
        print("Audit :", AUDIT)
        print("Class :", CLASS_AUDIT)
        return

    if args.workers < 1 or args.workers > 16:
        raise SystemExit("--workers must be between 1 and 16.")

    archives = archive_list()

    print("=" * 100)
    print("NSE CASH — IDENTITY-PRESERVING REBUILD (PARALLEL)")
    print("=" * 100)
    print("Legacy archives :", sum(s == "LEGACY" for s, _ in archives))
    print("UDiFF archives  :", sum(s == "UDIFF" for s, _ in archives))
    print("Total archives  :", len(archives))
    print("Workers         :", args.workers)
    print()

    if args.reset_output:
        reset_outputs()
    else:
        TMP_SHARDS.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    total = len(archives)
    done = 0

    # Threads are used deliberately: the workload is dominated by Drive
    # filesystem I/O, zip decompression, CSV parsing, and vectorized pandas.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(worker, item)
            for item in enumerate(archives, 1)
        ]

        for future in as_completed(futures):
            result = future.result()
            audit_rows.append(result["audit"])
            done += 1

            if done % 100 == 0 or done == total:
                ok = sum(r["status"] == "OK" for r in audit_rows)
                err = done - ok
                print(
                    f"Processed {done:,}/{total:,} | "
                    f"OK={ok:,} ERROR={err:,}"
                )

    audit_df = (
        pd.DataFrame(audit_rows)
        .sort_values("archive_index")
        .reset_index(drop=True)
    )

    print()
    print("=" * 100)
    print("ARCHIVE PROCESSING FINISHED — STARTING SINGLE-PASS MONTHLY CONSOLIDATION")
    print("=" * 100)

    meta = consolidate_monthly(audit_df)

    print()
    print("=" * 100)
    print("PARALLEL REBUILD COMPLETE")
    print("=" * 100)
    print("Rows                  :", f"{meta['rows']:,}")
    print("Symbols               :", f"{meta['symbols']:,}")
    print("Economic securities   :", f"{meta['economic_securities']:,}")
    print("Listing identities    :", f"{meta['listing_ids']:,}")
    print("Dates                 :", f"{meta['dates']:,}")
    print("Min date              :", meta["min_date"])
    print("Max date              :", meta["max_date"])
    print("Duplicate identity rows:", f"{meta['duplicate_row_identity_rows']:,}")
    print("Null identity rows     :", f"{meta['null_row_identity_rows']:,}")
    print("Archive errors        :", f"{meta['error_archives']:,}")
    print()
    print("Master:", MASTER)
    print("Meta  :", META)
    print("Audit :", AUDIT)
    print("Class :", CLASS_AUDIT)


if __name__ == "__main__":
    main()
