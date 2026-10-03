#!/usr/bin/env python3
"""Audit the persisted NSE CM-UDiFF EOD archive and publish a model-data gate.

This is an archive/data-quality audit only. It never trains or promotes a model.
The expected 2026 equity-session calendar is based on NSE's published 2026
equities holiday calendar. Muhurat Trading on Sunday 2026-11-08 is outside the
current Jan-1-through-audit-date range and therefore does not affect the
expected regular-session count.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from googleapiclient.http import MediaIoBaseDownload

from persistent_store import GoogleDriveStore

FOLDER_MIME = "application/vnd.google-apps.folder"
ARCHIVE_PATH = "production_data/nse_official/bulk_eod"
AUDIT_PATH = "production_data/nse_official/bulk_eod/audit"
DATE_RE = re.compile(
    r"(?<!\\d)(20\\d{2})[-_./]?(0[1-9]|1[0-2])[-_./]?([0-3]\\d)(?!\\d)"
)

# NSE 2026 EQUITIES holidays through the current audit horizon.
# Source: NSE India Market Timings & Holidays - 2026 Equities.
NSE_EQUITY_HOLIDAYS_2026 = {
    date(2026, 1, 15),
    date(2026, 1, 26),
    date(2026, 3, 3),
    date(2026, 3, 26),
    date(2026, 3, 31),
    date(2026, 4, 3),
    date(2026, 4, 14),
    date(2026, 5, 1),
    date(2026, 5, 28),
    date(2026, 6, 26),
    date(2026, 9, 14),
    date(2026, 10, 2),
    date(2026, 10, 20),
    date(2026, 11, 10),
    date(2026, 11, 24),
    date(2026, 12, 25),
}


def _date_from_text(value: str) -> date | None:
    for match in DATE_RE.finditer(str(value or "")):
        for token, fmt in (
            ("".join(match.groups()), "%Y%m%d"),
        ):
            try:
                return datetime.strptime(token, fmt).date()
            except ValueError:
                continue
    return None


def _date_from_parts(parts: list[str]) -> date | None:
    cleaned = [str(p).strip() for p in parts]
    for i in range(len(cleaned) - 2):
        try:
            y, m, d = map(int, cleaned[i : i + 3])
            if 2000 <= y <= 2100:
                return date(y, m, d)
        except (TypeError, ValueError):
            continue
    return None


def _list_children(store: GoogleDriveStore, parent_id: str) -> list[dict]:
    items: list[dict] = []
    token = None
    while True:
        kwargs = {
            "q": f"'{parent_id}' in parents and trashed = false",
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime)",
            "pageSize": 500,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if token:
            kwargs["pageToken"] = token
        resp = store.service.files().list(**kwargs).execute()
        items.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token:
            return items


def _resolve_folder(store: GoogleDriveStore, relative_path: str) -> str | None:
    parent = store.root_id
    for part in [x for x in relative_path.split("/") if x]:
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            return None
        parent = hit["id"]
    return parent


def _download_parquet(store: GoogleDriveStore, file_id: str) -> pd.DataFrame:
    buf = io.BytesIO()
    request = store.service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(buf, request, chunksize=4 * 1024 * 1024)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    return pd.read_parquet(buf)


def expected_equity_dates(start: date, end: date) -> list[date]:
    out: list[date] = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in NSE_EQUITY_HOLIDAYS_2026:
            out.append(d)
        d += timedelta(days=1)
    return out


def _collect_date_folders(store: GoogleDriveStore, root_id: str) -> dict[date, str]:
    """Find dated folders without descending into their file contents first."""
    found: dict[date, str] = {}
    queue: list[tuple[str, list[str]]] = [(root_id, [])]
    while queue:
        folder_id, ancestors = queue.pop()
        for item in _list_children(store, folder_id):
            if item.get("mimeType") != FOLDER_MIME:
                continue
            name = str(item.get("name") or "")
            next_ancestors = ancestors + [name]
            folder_date = _date_from_text(name) or _date_from_parts(next_ancestors)
            if folder_date:
                found.setdefault(folder_date, item["id"])
                continue
            queue.append((item["id"], next_ancestors))
    return found


def audit_archive(store: GoogleDriveStore, start: date, end: date) -> dict:
    archive_id = _resolve_folder(store, ARCHIVE_PATH)
    if not archive_id:
        raise RuntimeError(f"Archive path not found below Drive root: {ARCHIVE_PATH}")

    date_folders = _collect_date_folders(store, archive_id)
    expected = expected_equity_dates(start, end)
    expected_set = set(expected)
    observed_set = set(date_folders)

    missing_dates = sorted(expected_set - observed_set)
    unexpected_dates = sorted(observed_set - expected_set)

    per_day: list[dict] = []
    total_rows = 0
    total_symbols: set[str] = set()
    duplicate_symbol_dates = 0
    invalid_ohlc = 0
    negative_volume = 0
    null_required = 0
    wrong_date_rows = 0
    missing_parquet = []
    duplicate_parquet_artifacts = []
    missing_zip = []
    schema_errors = []

    for trade_date in sorted(observed_set & expected_set):
        folder_id = date_folders[trade_date]
        children = _list_children(store, folder_id)
        parquet_files = [
            x for x in children
            if str(x.get("name") or "").lower().endswith(".parquet")
        ]
        zip_files = [
            x for x in children
            if str(x.get("name") or "").lower().endswith(".zip")
        ]

        if not parquet_files:
            missing_parquet.append(trade_date.isoformat())
            per_day.append({
                "date": trade_date.isoformat(),
                "status": "MISSING_PARQUET",
                "parquet_files": 0,
                "zip_files": len(zip_files),
            })
            continue

        if len(parquet_files) > 1:
            duplicate_parquet_artifacts.append({
                "date": trade_date.isoformat(),
                "count": len(parquet_files),
                "files": [x.get("name") for x in parquet_files],
            })

        if not zip_files:
            missing_zip.append(trade_date.isoformat())

        chosen = sorted(parquet_files, key=lambda x: str(x.get("modifiedTime") or ""))[-1]
        try:
            df = _download_parquet(store, chosen["id"])
        except Exception as exc:
            schema_errors.append({
                "date": trade_date.isoformat(),
                "error": f"PARQUET_READ_ERROR: {type(exc).__name__}: {exc}",
            })
            continue

        cols = {str(c).strip().lower(): c for c in df.columns}
        required = {"symbol", "date", "open", "high", "low", "close", "volume"}
        missing_cols = sorted(required - set(cols))
        if missing_cols:
            schema_errors.append({
                "date": trade_date.isoformat(),
                "error": "MISSING_COLUMNS",
                "missing_columns": missing_cols,
            })
            continue

        x = df.rename(columns={v: k for k, v in cols.items()}).copy()
        for c in ["open", "high", "low", "close", "volume"]:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
        parsed_dates = pd.to_datetime(x["date"], errors="coerce", format="mixed", dayfirst=True).dt.date

        total_rows += len(x)
        total_symbols.update(x["symbol"].dropna().astype(str))
        null_required += int(x[["symbol", "open", "high", "low", "close", "volume"]].isna().any(axis=1).sum())
        wrong_date_rows += int((parsed_dates.notna() & (parsed_dates != trade_date)).sum())
        duplicate_symbol_dates += int(x.duplicated(subset=["symbol", "date"], keep=False).sum())
        invalid_ohlc += int(
            (
                (x["open"] <= 0)
                | (x["high"] <= 0)
                | (x["low"] <= 0)
                | (x["close"] <= 0)
                | (x["high"] < x[["open", "close"]].max(axis=1))
                | (x["low"] > x[["open", "close"]].min(axis=1))
                | (x["high"] < x["low"])
            ).fillna(False).sum()
        )
        negative_volume += int((x["volume"] < 0).fillna(False).sum())

        per_day.append({
            "date": trade_date.isoformat(),
            "status": "PASS",
            "rows": int(len(x)),
            "symbols": int(x["symbol"].nunique()),
            "parquet_files": len(parquet_files),
            "zip_files": len(zip_files),
            "duplicate_symbol_date_rows": int(
                x.duplicated(subset=["symbol", "date"], keep=False).sum()
            ),
            "invalid_ohlc_rows": int(
                (
                    (x["open"] <= 0)
                    | (x["high"] <= 0)
                    | (x["low"] <= 0)
                    | (x["close"] <= 0)
                    | (x["high"] < x[["open", "close"]].max(axis=1))
                    | (x["low"] > x[["open", "close"]].min(axis=1))
                    | (x["high"] < x["low"])
                ).fillna(False).sum()
            ),
            "negative_volume_rows": int((x["volume"] < 0).fillna(False).sum()),
        })

    hard_failures = (
        bool(missing_dates)
        or bool(missing_parquet)
        or bool(duplicate_parquet_artifacts)
        or bool(schema_errors)
        or duplicate_symbol_dates > 0
        or invalid_ohlc > 0
        or negative_volume > 0
        or null_required > 0
        or wrong_date_rows > 0
    )

    audit_status = "PASS" if not hard_failures else "FAIL"
    report = {
        "audit_version": "nse-eod-archive-audit-v1",
        "audited_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": "NSE_OFFICIAL_UDIFF_BHAVCOPY",
        "archive_path": ARCHIVE_PATH,
        "expected_calendar_source": "NSE 2026 Equities holiday calendar",
        "range": {"start": start.isoformat(), "end": end.isoformat()},
        "expected_trading_dates": len(expected),
        "observed_date_folders": len(observed_set),
        "observed_expected_dates": len(observed_set & expected_set),
        "missing_expected_dates": [d.isoformat() for d in missing_dates],
        "unexpected_dates": [d.isoformat() for d in unexpected_dates],
        "rows": total_rows,
        "unique_symbols": len(total_symbols),
        "duplicate_symbol_date_rows": duplicate_symbol_dates,
        "invalid_ohlc_rows": invalid_ohlc,
        "negative_volume_rows": negative_volume,
        "null_required_rows": null_required,
        "wrong_date_rows": wrong_date_rows,
        "missing_parquet_dates": missing_parquet,
        "duplicate_parquet_artifacts": duplicate_parquet_artifacts,
        "missing_zip_dates": missing_zip,
        "schema_errors": schema_errors,
        "per_day": per_day,
        "audit_status": audit_status,
        "model_data_gate": {
            "eod_archive_quality": audit_status,
            "next_session_delivery_candidate": (
                "READY_FOR_DATA_VALIDATION" if audit_status == "PASS" else "BLOCKED"
            ),
            "production_model_promotion": "NOT_PERMITTED",
            "wider_universe_validation": "REQUIRED",
            "point_in_time_enrichment": "REQUIRED_FOR_FULL_MULTIFACTOR_VALIDATION",
        },
    }
    return report


def save_report(store: GoogleDriveStore, report: dict, end: date) -> tuple[str, str]:
    out_dir = Path("/tmp/nse_eod_archive_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    dated = out_dir / f"NSE_EOD_ARCHIVE_AUDIT_{end:%Y%m%d}.json"
    latest = out_dir / "LATEST_NSE_EOD_ARCHIVE_AUDIT_2026.json"
    payload = json.dumps(report, indent=2, sort_keys=True, default=str)
    dated.write_text(payload, encoding="utf-8")
    latest.write_text(payload, encoding="utf-8")

    remote_dated = (
        f"{AUDIT_PATH}/NSE_EOD_ARCHIVE_AUDIT_{end:%Y%m%d}.json"
    )
    remote_latest = f"{AUDIT_PATH}/LATEST_NSE_EOD_ARCHIVE_AUDIT_2026.json"
    store.upload_or_update(dated, remote_dated, mime_type="application/json")
    store.upload_or_update(latest, remote_latest, mime_type="application/json")
    return str(dated), str(latest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start)
    if end < start:
        raise SystemExit("--end cannot be before --start")

    store = GoogleDriveStore()
    report = audit_archive(store, start, end)
    dated, latest = save_report(store, report, end)
    print(json.dumps({
        "audit_status": report["audit_status"],
        "expected_trading_dates": report["expected_trading_dates"],
        "observed_date_folders": report["observed_date_folders"],
        "missing_expected_dates": report["missing_expected_dates"],
        "rows": report["rows"],
        "unique_symbols": report["unique_symbols"],
        "duplicate_symbol_date_rows": report["duplicate_symbol_date_rows"],
        "invalid_ohlc_rows": report["invalid_ohlc_rows"],
        "negative_volume_rows": report["negative_volume_rows"],
        "schema_errors": len(report["schema_errors"]),
        "latest_report": latest,
    }, indent=2))
    raise SystemExit(0 if report["audit_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
