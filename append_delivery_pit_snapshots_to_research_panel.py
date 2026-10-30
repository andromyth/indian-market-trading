#!/usr/bin/env python3
"""Append verified daily PIT snapshots to the aggregate Delivery research panel.

The operation is append-only. It refuses source/PIT gaps and duplicate keys,
backs up the existing aggregate panel before replacing its Drive object, and
never trains or promotes models or places orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb

from backfill_missing_delivery_pit_snapshots import load_drive_inventory
from delivery_production_refresh import _download_file
from multifactor_data_layer_v2 import FEATURES_59
from nse_bulk_eod_ingest import business_days
from persistent_store import GoogleDriveStore

BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", "/content/drive/MyDrive/Indian_Market_Research"))
PANEL_REMOTE = (
    "models/all_nse_delivery_multifactor_replacement_2026/"
    "ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"
)
PANEL_LOCAL = BASE / PANEL_REMOTE
REPORT_LOCAL = BASE / "diagnostics/production/research_panel_incremental_update_latest.json"
REPORT_REMOTE = "diagnostics/production/research_panel_incremental_update_latest.json"
PIT_REMOTE_ROOT = "features/daily"
FOLDER_MIME = "application/vnd.google-apps.folder"
READINESS_COLUMNS = (
    "ready_market_price", "ready_market_context", "ready_news",
    "ready_events", "ready_fno", "ready_fundamentals",
)


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_panel_extension(
    panel_last_date: date,
    target_date: date,
    inventory: dict[str, set[date]],
) -> dict[str, Any]:
    """Return exact NSE-session continuity gaps before any aggregate write."""
    if target_date < panel_last_date:
        raise ValueError("target_date precedes panel_last_date")
    expected = list(business_days(panel_last_date + timedelta(days=1), target_date))
    eod = inventory.get("eod", set())
    delivery = inventory.get("delivery", set())
    pit = inventory.get("pit", set())
    missing_eod = [day.isoformat() for day in expected if day not in eod]
    missing_delivery = [day.isoformat() for day in expected if day not in delivery]
    missing_pit = [
        day.isoformat()
        for day in expected
        if day in eod and day in delivery and day not in pit
    ]
    ready = [
        day.isoformat()
        for day in expected
        if day in eod and day in delivery and day in pit
    ]
    return {
        "panel_last_date": panel_last_date.isoformat(),
        "target_date": target_date.isoformat(),
        "expected_sessions": [day.isoformat() for day in expected],
        "expected_session_count": len(expected),
        "missing_eod_dates": missing_eod,
        "missing_delivery_dates": missing_delivery,
        "missing_pit_dates": missing_pit,
        "ready_to_append_dates": ready,
        "status": (
            "READY"
            if not missing_eod and not missing_delivery and not missing_pit
            else "GAPS_REMAIN"
        ),
    }


def _resolve_parent(store: GoogleDriveStore, remote: str) -> tuple[str, str]:
    parts = [part for part in remote.strip("/").split("/") if part]
    if not parts:
        raise ValueError("Remote path is empty")
    parent = store.root_id
    for part in parts[:-1]:
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            raise RuntimeError(f"Google Drive folder missing: {'/'.join(parts[:-1])}")
        parent = hit["id"]
    return parent, parts[-1]


def _remote_metadata(store: GoogleDriveStore, remote: str) -> dict[str, Any]:
    parent, name = _resolve_parent(store, remote)
    hit = store._find_child(parent, name)
    if not hit:
        raise FileNotFoundError(f"Google Drive file missing: {remote}")
    return store.service.files().get(
        fileId=hit["id"],
        fields="id,name,size,md5Checksum,modifiedTime",
    ).execute()


def _write_report(store: GoogleDriveStore, report: dict[str, Any]) -> None:
    REPORT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_LOCAL.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, REPORT_LOCAL)
    store.upload_or_update(REPORT_LOCAL, REPORT_REMOTE, mime_type="application/json")


def _validate_parquet_sessions(
    con: duckdb.DuckDBPyConnection,
    paths: list[Path],
    expected_dates: list[date],
) -> list[int]:
    counts: list[int] = []
    required = {"symbol", "date", "close", *FEATURES_59, *READINESS_COLUMNS}
    for path, day in zip(paths, expected_dates):
        path_sql = _sql_path(path)
        columns = {
            str(row[0])
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet({path_sql})"
            ).fetchall()
        }
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(f"{path.name} is missing required PIT columns: {missing}")
        row_count, min_date, max_date, duplicate_groups = con.execute(
            f"""
            SELECT
                COUNT(*),
                MIN(TRY_CAST(date AS DATE)),
                MAX(TRY_CAST(date AS DATE)),
                COUNT(*) FILTER (WHERE duplicate_count > 1)
            FROM (
                SELECT
                    TRY_CAST(date AS DATE) AS date,
                    REGEXP_REPLACE(UPPER(TRIM(CAST(symbol AS VARCHAR))), '(\\\\.NS|\\\\.BO|-EQ|\\\\.EQ)$', '') AS symbol,
                    COUNT(*) AS duplicate_count
                FROM read_parquet({path_sql})
                GROUP BY 1, 2
            )
            """
        ).fetchone()
        if not row_count or min_date != day or max_date != day:
            raise RuntimeError(
                f"{path.name} date validation failed: rows={row_count}, "
                f"range={min_date}..{max_date}, expected={day}"
            )
        if duplicate_groups:
            raise RuntimeError(f"{path.name} has duplicate symbol/date keys.")
        counts.append(int(row_count))
    return counts


def append_panel(store: GoogleDriveStore, target_date: date) -> dict[str, Any]:
    inventory = load_drive_inventory(store)
    PANEL_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if not _download_file(store, PANEL_REMOTE, PANEL_LOCAL):
        raise FileNotFoundError(f"Could not download aggregate PIT panel: {PANEL_REMOTE}")

    work_dir = PANEL_LOCAL.parent / "incremental_append_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = work_dir / "duckdb_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_date": target_date.isoformat(),
        "panel_remote": PANEL_REMOTE,
        "panel_overwritten": False,
        "model_weights_changed": False,
        "model_promotion_performed": False,
        "orders_placed": False,
    }
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=2")
        con.execute("PRAGMA memory_limit='3GB'")
        con.execute(f"SET temp_directory={_sql_path(temp_dir)}")
        panel_sql = _sql_path(PANEL_LOCAL)
        columns = {
            str(row[0])
            for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet({panel_sql})").fetchall()
        }
        required = {"symbol", "date", "close", *FEATURES_59, *READINESS_COLUMNS}
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(f"Aggregate panel is missing required columns: {missing}")
        old_rows, old_last_date = con.execute(
            f"SELECT COUNT(*), MAX(TRY_CAST(date AS DATE)) FROM read_parquet({panel_sql})"
        ).fetchone()
        old_duplicate_groups = con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT
                    TRY_CAST(date AS DATE) AS date,
                    REGEXP_REPLACE(UPPER(TRIM(CAST(symbol AS VARCHAR))), '(\\\\.NS|\\\\.BO|-EQ|\\\\.EQ)$', '') AS symbol
                FROM read_parquet({panel_sql})
                GROUP BY 1, 2 HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        if old_duplicate_groups:
            raise RuntimeError(
                f"Aggregate panel already contains {old_duplicate_groups} duplicate symbol/date keys."
            )
        if old_last_date is None:
            raise RuntimeError("Aggregate panel has no valid session date.")
        plan = plan_panel_extension(old_last_date, target_date, inventory)
        report["plan"] = plan
        report["old_panel_rows"] = int(old_rows)
        report["old_panel_last_date"] = old_last_date.isoformat()
        report["drive_inventory_counts"] = {
            key: len(inventory.get(key, set())) for key in ("eod", "delivery", "pit")
        }
        if not plan["expected_sessions"]:
            report["status"] = "NO_EXTENSION_NEEDED"
            return report
        if plan["status"] != "READY":
            report["status"] = "BLOCKED_CONTINUITY_GAPS"
            return report

        expected_dates = [date.fromisoformat(value) for value in plan["expected_sessions"]]
        snapshot_paths = [
            BASE / PIT_REMOTE_ROOT / f"DELIVERY_PIT_FEATURES_{day.isoformat()}.parquet"
            for day in expected_dates
        ]
        for path in snapshot_paths:
            if not path.exists() or path.stat().st_size == 0:
                remote = f"{PIT_REMOTE_ROOT}/{path.name}"
                if not _download_file(store, remote, path):
                    raise FileNotFoundError(f"Missing dated PIT snapshot: {remote}")
        snapshot_counts = _validate_parquet_sessions(con, snapshot_paths, expected_dates)
        paths_sql = "[" + ", ".join(_sql_path(path) for path in snapshot_paths) + "]"
        temp_panel = work_dir / "aggregate_panel_incremental.tmp.parquet"
        temp_panel.unlink(missing_ok=True)
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet({panel_sql})
                UNION ALL BY NAME
                SELECT * FROM read_parquet({paths_sql}, union_by_name=true)
            )
            TO {_sql_path(temp_panel)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        temp_sql = _sql_path(temp_panel)
        new_rows_expected = sum(snapshot_counts)
        new_rows_actual = con.execute(
            f"SELECT COUNT(*) FROM read_parquet({paths_sql}, union_by_name=true)"
        ).fetchone()[0]
        final_rows, final_last_date = con.execute(
            f"SELECT COUNT(*), MAX(TRY_CAST(date AS DATE)) FROM read_parquet({temp_sql})"
        ).fetchone()
        final_duplicate_groups = con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT
                    TRY_CAST(date AS DATE) AS date,
                    REGEXP_REPLACE(UPPER(TRIM(CAST(symbol AS VARCHAR))), '(\\\\.NS|\\\\.BO|-EQ|\\\\.EQ)$', '') AS symbol
                FROM read_parquet({temp_sql})
                GROUP BY 1, 2 HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
        if int(new_rows_actual) != new_rows_expected:
            raise RuntimeError(
                f"Append row-count mismatch: expected={new_rows_expected}, actual={new_rows_actual}"
            )
        if int(final_rows) != int(old_rows) + new_rows_expected:
            raise RuntimeError("Final aggregate row count is not old rows plus appended snapshots.")
        if final_last_date != target_date:
            raise RuntimeError(f"Final aggregate max date {final_last_date} != target {target_date}.")
        if final_duplicate_groups:
            raise RuntimeError(f"Final aggregate panel has {final_duplicate_groups} duplicate keys.")

        old_sha = _sha256(PANEL_LOCAL)
        new_sha = _sha256(temp_panel)
        old_md5 = _md5(PANEL_LOCAL)
        backup_name = (
            f"{PANEL_LOCAL.stem}.backup_before_{target_date.isoformat()}_"
            f"{old_last_date.isoformat()}_{old_sha[:12]}.parquet"
        )
        backup_remote = str(Path(PANEL_REMOTE).with_name(backup_name))
        backup_parent, backup_leaf = _resolve_parent(store, backup_remote)
        existing_backup = store._find_child(backup_parent, backup_leaf)
        if existing_backup:
            backup_meta = store.service.files().get(
                fileId=existing_backup["id"], fields="id,name,size,md5Checksum"
            ).execute()
            if (
                int(backup_meta.get("size", -1)) != PANEL_LOCAL.stat().st_size
                or backup_meta.get("md5Checksum") != old_md5
            ):
                raise RuntimeError("Existing backup content mismatch; refusing to overwrite.")
            report["backup_status"] = "EXISTING_CONTENT_VERIFIED"
        else:
            store.upload_or_update(
                PANEL_LOCAL, backup_remote, mime_type="application/octet-stream"
            )
            backup_meta = _remote_metadata(store, backup_remote)
            if (
                int(backup_meta.get("size", -1)) != PANEL_LOCAL.stat().st_size
                or backup_meta.get("md5Checksum") != old_md5
            ):
                raise RuntimeError("Backup verification failed after upload.")
            report["backup_status"] = "CREATED_AND_VERIFIED"

        os.replace(temp_panel, PANEL_LOCAL)
        store.upload_or_update(
            PANEL_LOCAL, PANEL_REMOTE, mime_type="application/octet-stream"
        )
        remote_meta = _remote_metadata(store, PANEL_REMOTE)
        if (
            int(remote_meta.get("size", -1)) != PANEL_LOCAL.stat().st_size
            or remote_meta.get("md5Checksum") != _md5(PANEL_LOCAL)
        ):
            raise RuntimeError("Aggregate panel upload does not match validated local Parquet.")
        report.update({
            "status": "APPENDED_AND_VERIFIED",
            "panel_overwritten": True,
            "new_snapshot_dates": [day.isoformat() for day in expected_dates],
            "new_snapshot_row_counts": dict(zip(
                [day.isoformat() for day in expected_dates], snapshot_counts
            )),
            "new_rows": new_rows_expected,
            "final_rows": int(final_rows),
            "final_last_date": final_last_date.isoformat(),
            "final_duplicate_symbol_date_groups": int(final_duplicate_groups),
            "backup_remote": backup_remote,
            "old_panel_sha256": old_sha,
            "new_panel_sha256": new_sha,
            "new_panel_bytes": int(PANEL_LOCAL.stat().st_size),
            "remote_panel_metadata": remote_meta,
        })
        return report
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-date", required=True, help="Latest completed NSE session YYYY-MM-DD.")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.target_date)
    store = GoogleDriveStore()
    try:
        report = append_panel(store, target_date)
    except Exception as exc:
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_date": target_date.isoformat(),
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "panel_overwritten": False,
            "model_weights_changed": False,
            "model_promotion_performed": False,
            "orders_placed": False,
        }
        _write_report(store, report)
        raise
    _write_report(store, report)
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if report["status"] in {"APPENDED_AND_VERIFIED", "NO_EXTENSION_NEEDED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
