#!/usr/bin/env python3
"""
Repair parser-induced null NSE cash series in an already-built identity master.

The prior CSV parser allowed pandas to interpret the literal NSE series code
"NA" as missing. The affected rows therefore lost listing_id/row_identity_key.
This script repairs ONLY rows where series and row_identity_key are both NULL,
setting series to literal "NA", then recomputing the affected identity fields.

It does not download or reread the 2,654 raw archives.

Safety gates:
  - only NULL-series rows are changed;
  - no existing non-null series is changed;
  - the rewritten master must have zero NULL row_identity_key values;
  - the rewritten master must have zero duplicate row_identity_key values;
  - all expected row counts must be preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

ROOT = Path(
    "/content/drive/MyDrive/Indian_Market_Research/"
    "production_data/nse_official"
)
MASTER = ROOT / "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
TMP = ROOT / "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.repaired.tmp.parquet"
BACKUP = ROOT / "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.before_na_repair.parquet"
REPORT = ROOT / "cash_identity_na_series_repair_2016_2026.json"
META = ROOT / "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.meta.json"
AUDIT = ROOT / "cash_identity_preserved_rebuild_audit_2016_2026.csv"


def main():
    if not MASTER.exists():
        raise FileNotFoundError(MASTER)

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    total = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(MASTER)],
    ).fetchone()[0]

    null_series = con.execute(
        """
        SELECT COUNT(*)
        FROM read_parquet(?)
        WHERE series IS NULL
        """,
        [str(MASTER)],
    ).fetchone()[0]

    null_identity = con.execute(
        """
        SELECT COUNT(*)
        FROM read_parquet(?)
        WHERE row_identity_key IS NULL
        """,
        [str(MASTER)],
    ).fetchone()[0]

    print("=" * 100)
    print("NSE CASH — NULL SERIES / IDENTITY REPAIR")
    print("=" * 100)
    print("Master                 :", MASTER)
    print("Total rows             :", f"{total:,}")
    print("NULL series rows       :", f"{null_series:,}")
    print("NULL identity rows     :", f"{null_identity:,}")

    if null_series != null_identity:
        raise RuntimeError(
            "Safety gate failed: NULL series and NULL identity counts differ. "
            "Refusing to guess at affected rows."
        )

    if null_series == 0:
        print("Nothing to repair.")
        return

    # This repair is intentionally restricted to the exact parser-loss
    # signature observed in the completed build.
    check = con.execute(
        """
        SELECT
            COUNT(*) AS affected,
            COUNT(*) FILTER (WHERE series IS NOT NULL) AS nonnull_series,
            COUNT(*) FILTER (WHERE row_identity_key IS NOT NULL) AS nonnull_key
        FROM read_parquet(?)
        WHERE series IS NULL
          AND row_identity_key IS NULL
        """,
        [str(MASTER)],
    ).fetchone()

    if check[0] != null_series or check[1] != 0 or check[2] != 0:
        raise RuntimeError("Affected-row safety gate failed.")

    if BACKUP.exists():
        raise RuntimeError(
            f"Backup already exists: {BACKUP}. Refusing to overwrite it."
        )

    # Validate the target identity expression before rewriting the master.
    collisions = con.execute(
        """
        WITH repaired AS (
            SELECT
                CASE
                    WHEN series IS NULL AND row_identity_key IS NULL
                    THEN 'NA'
                    ELSE series
                END AS new_series,
                symbol,
                isin,
                date
            FROM read_parquet(?)
        ),
        identified AS (
            SELECT
                CASE
                    WHEN isin IS NOT NULL
                         AND TRIM(CAST(isin AS VARCHAR)) <> ''
                         AND UPPER(TRIM(CAST(isin AS VARCHAR))) NOT IN
                             ('NAN','NONE','<NA>')
                    THEN
                        'ISIN:' || UPPER(TRIM(CAST(isin AS VARCHAR))) ||
                        '|SERIES:' || UPPER(COALESCE(CAST(new_series AS VARCHAR), '')) ||
                        '|SYMBOL:' || UPPER(TRIM(CAST(symbol AS VARCHAR)))
                    ELSE
                        'SYMBOL:' || UPPER(TRIM(CAST(symbol AS VARCHAR))) ||
                        '|SERIES:' || UPPER(COALESCE(CAST(new_series AS VARCHAR), ''))
                END || '|DATE:' || STRFTIME(date, '%Y-%m-%d') AS new_key
            FROM repaired
        )
        SELECT COUNT(*)
        FROM (
            SELECT new_key
            FROM identified
            WHERE new_key IS NOT NULL
            GROUP BY new_key
            HAVING COUNT(*) > 1
        )
        """,
        [str(MASTER)],
    ).fetchone()[0]

    if collisions:
        raise RuntimeError(
            f"Safety gate failed: {collisions:,} duplicate identity keys "
            "would exist after repair."
        )

    print("Predicted post-repair identity collisions:", collisions)

    # A previous attempt may have created the backup and then failed during
    # the DuckDB COPY statement. Reuse that backup after validating it rather
    # than refusing to continue or overwriting evidence.
    import shutil
    if BACKUP.exists():
        backup_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(BACKUP)],
        ).fetchone()[0]
        if backup_rows != total:
            raise RuntimeError(
                f"Backup row-count mismatch: expected={total:,} "
                f"found={backup_rows:,}"
            )
        print("Existing validated backup reused:", BACKUP)
    else:
        print("Creating backup:", BACKUP)
        shutil.copy2(MASTER, BACKUP)

    if TMP.exists():
        TMP.unlink()

    # DuckDB parameters work for read_parquet(?) but not for COPY ... TO ?.
    # Use safely quoted SQL string literals for the two COPY file paths.
    master_sql = str(MASTER).replace("'", "''")
    tmp_sql = str(TMP).replace("'", "''")

    repair_sql = f"""
        COPY (
            SELECT
                symbol,
                date,
                CASE
                    WHEN series IS NULL AND row_identity_key IS NULL
                    THEN 'NA'
                    ELSE series
                END AS series,
                isin,
                economic_security_id,

                CASE
                    WHEN series IS NULL AND row_identity_key IS NULL THEN
                        CASE
                            WHEN isin IS NOT NULL
                                 AND TRIM(CAST(isin AS VARCHAR)) <> ''
                                 AND UPPER(TRIM(CAST(isin AS VARCHAR))) NOT IN
                                     ('NAN','NONE','<NA>')
                            THEN
                                'ISIN:' || UPPER(TRIM(CAST(isin AS VARCHAR))) ||
                                '|SERIES:NA|SYMBOL:' ||
                                UPPER(TRIM(CAST(symbol AS VARCHAR)))
                            ELSE
                                'SYMBOL:' || UPPER(TRIM(CAST(symbol AS VARCHAR))) ||
                                '|SERIES:NA'
                        END
                    ELSE listing_id
                END AS listing_id,

                CASE
                    WHEN series IS NULL AND row_identity_key IS NULL THEN
                        (
                            CASE
                                WHEN isin IS NOT NULL
                                     AND TRIM(CAST(isin AS VARCHAR)) <> ''
                                     AND UPPER(TRIM(CAST(isin AS VARCHAR))) NOT IN
                                         ('NAN','NONE','<NA>')
                                THEN
                                    'ISIN:' || UPPER(TRIM(CAST(isin AS VARCHAR))) ||
                                    '|SERIES:NA|SYMBOL:' ||
                                    UPPER(TRIM(CAST(symbol AS VARCHAR)))
                                ELSE
                                    'SYMBOL:' || UPPER(TRIM(CAST(symbol AS VARCHAR))) ||
                                    '|SERIES:NA'
                            END
                            || '|DATE:' || STRFTIME(date, '%Y-%m-%d')
                        )
                    ELSE row_identity_key
                END AS row_identity_key,

                security_class,
                instrument_type,
                source_format,
                instrument_id,
                instrument_name,
                business_date,
                segment,
                source,
                actual_expiry,
                open,
                high,
                low,
                close,
                last,
                prev_close,
                last_price_zero,
                last_price_for_research,
                research_price,
                volume,
                traded_value,
                num_trades,
                raw_archive
            FROM read_parquet('{master_sql}')
        )
        TO '{tmp_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(repair_sql)

    post_total = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(TMP)],
    ).fetchone()[0]

    post_null_series = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?) WHERE series IS NULL",
        [str(TMP)],
    ).fetchone()[0]

    post_null_identity = con.execute(
        "SELECT COUNT(*) FROM read_parquet(?) WHERE row_identity_key IS NULL",
        [str(TMP)],
    ).fetchone()[0]

    post_dupes = con.execute(
        """
        SELECT COUNT(*)
        FROM (
            SELECT row_identity_key
            FROM read_parquet(?)
            WHERE row_identity_key IS NOT NULL
            GROUP BY row_identity_key
            HAVING COUNT(*) > 1
        )
        """,
        [str(TMP)],
    ).fetchone()[0]

    if post_total != total:
        raise RuntimeError(
            f"Row-count gate failed: before={total:,} after={post_total:,}"
        )

    if post_null_series != 0 or post_null_identity != 0 or post_dupes != 0:
        raise RuntimeError(
            "Post-repair identity gates failed: "
            f"null_series={post_null_series:,}, "
            f"null_identity={post_null_identity:,}, "
            f"duplicate_keys={post_dupes:,}"
        )

    TMP.replace(MASTER)

    # Bring the persisted metadata in line with the repaired master.
    if META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
    else:
        meta = {}

    meta["rows"] = int(post_total)
    meta["duplicate_row_identity_rows"] = 0
    meta["null_row_identity_rows"] = 0
    meta["repaired_na_series_rows"] = int(null_series)
    meta["note"] = (
        "Identity-preserved cash layer. Literal NSE series code NA was "
        "restored for parser-induced NULL series rows."
    )
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Synchronize the per-archive audit without another DuckDB COPY-to-path
    # parameterization issue.
    if AUDIT.exists():
        audit_df = pd.read_csv(AUDIT)
        for col in ("identity_duplicate_rows", "identity_duplicate_keys"):
            if col in audit_df.columns:
                audit_df[col] = 0
        audit_df.to_csv(AUDIT, index=False)


    report = {
        "master": str(MASTER),
        "backup": str(BACKUP),
        "rows_before": int(total),
        "null_series_before": int(null_series),
        "null_identity_before": int(null_identity),
        "repaired_rows": int(null_series),
        "rows_after": int(post_total),
        "null_series_after": int(post_null_series),
        "null_identity_after": int(post_null_identity),
        "duplicate_identity_keys_after": int(post_dupes),
        "repair_value": "NA",
        "rule": (
            "Only rows where both series and row_identity_key were NULL "
            "were repaired."
        ),
        "status": "PASS",
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print("=" * 100)
    print("NULL SERIES REPAIR COMPLETE")
    print("=" * 100)
    print("Rows before            :", f"{total:,}")
    print("Rows repaired          :", f"{null_series:,}")
    print("Rows after             :", f"{post_total:,}")
    print("NULL series after      :", f"{post_null_series:,}")
    print("NULL identity after    :", f"{post_null_identity:,}")
    print("Duplicate identity keys:", f"{post_dupes:,}")
    print("Backup                 :", BACKUP)
    print("Report                 :", REPORT)
    print("STATUS                 : PASS")


if __name__ == "__main__":
    main()
