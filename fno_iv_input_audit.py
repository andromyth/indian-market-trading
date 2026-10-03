#!/usr/bin/env python3
"""Audit whether persisted NSE F&O raw archives contain enough inputs for ATM IV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nse_fno_udiff_ingest import (
    _atm_iv,
    _drive_download,
    _normalize,
    _read_zip,
    RAW_BASE,
    BASE,
)
from persistent_store import GoogleDriveStore


ARCHIVE_PREFIX = "BhavCopy_NSE_FO_0_0_0_"
ARCHIVE_SUFFIX = "_F_0000.csv.zip"


def _archive_path(day: pd.Timestamp) -> Path:
    day = pd.Timestamp(day).normalize()
    out_dir = RAW_BASE / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}"
    return out_dir / f"{ARCHIVE_PREFIX}{day:%Y%m%d}{ARCHIVE_SUFFIX}"


def _remote_archive(day: pd.Timestamp) -> str:
    day = pd.Timestamp(day).normalize()
    return (
        f"production_data/nse_official/fno/raw/"
        f"{day:%Y/%m/%d}/{ARCHIVE_PREFIX}{day:%Y%m%d}{ARCHIVE_SUFFIX}"
    )


def _materialize_drive_archive(path: Path, day: pd.Timestamp) -> tuple[bool, str]:
    """Materialize one known Drive archive and return (found, diagnostic)."""
    if path.exists() and path.stat().st_size > 0:
        return True, "LOCAL_CACHE"
    try:
        ok = _drive_download(_remote_archive(day), path)
    except Exception as exc:
        return False, f"DRIVE_DOWNLOAD_EXCEPTION:{type(exc).__name__}:{exc}"
    if ok and path.exists() and path.stat().st_size > 0:
        return True, "DRIVE_MATERIALIZED"
    return False, "DRIVE_PATH_NOT_FOUND_OR_UNREADABLE"


def _drive_discover_candidates(max_results: int = 50) -> tuple[list[dict], str]:
    """Inspect Drive recursively enough to determine where F&O raw archives live."""
    try:
        store = GoogleDriveStore()
    except Exception as exc:
        return [], f"DRIVE_INIT_FAILED:{type(exc).__name__}:{exc}"

    # First check the expected folder hierarchy from the production contract.
    try:
        parent = store.root_id
        for part in ["production_data", "nse_official", "fno", "raw"]:
            hit = store._find_child(parent, part)
            if not hit:
                return [], f"EXPECTED_FOLDER_MISSING:{part}"
            parent = hit["id"]

        direct = store.service.files().list(
            q=(
                f"'{parent}' in parents and trashed = false "
                "and name contains 'BhavCopy_NSE_FO_0_0_0_'"
            ),
            spaces="drive",
            fields="files(id,name,mimeType,size,modifiedTime,parents)",
            pageSize=max_results,
            orderBy="modifiedTime desc",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute().get("files", [])
        if direct:
            return direct, "EXPECTED_RAW_FOLDER_DIRECT_MATCH"
    except Exception as exc:
        return [], f"EXPECTED_TREE_CHECK_FAILED:{type(exc).__name__}:{exc}"

    # Fall back to a broad Drive filename search. This is diagnostic only.
    try:
        q = (
            "trashed = false and "
            "name contains 'BhavCopy_NSE_FO_0_0_0_'"
        )
        result = store.service.files().list(
            q=q,
            spaces="drive",
            fields="files(id,name,mimeType,size,modifiedTime,parents)",
            pageSize=max_results,
            orderBy="modifiedTime desc",
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute().get("files", [])
        return result, "BROAD_FILENAME_SEARCH"
    except Exception as exc:
        return [], f"BROAD_SEARCH_FAILED:{type(exc).__name__}:{exc}"


def audit_file(path: Path) -> dict:
    try:
        raw = _read_zip(path.read_bytes())
        day = pd.Timestamp(
            path.name.split("_0_0_0_")[1][:8], format="%Y%m%d"
        ).date()
        x = _normalize(raw, day)
        option = x[
            x["option_type"].isin(["CE", "PE", "CALL", "PUT"])
        ].copy()
        futures = x[
            x["instrument"].str.contains("FUT", na=False)
        ].copy()
        futures = futures.dropna(subset=["expiry"])
        futures = futures[futures["expiry"] >= x["date"].iloc[0]]
        futures = (
            futures.sort_values(["symbol", "expiry"])
            .drop_duplicates("symbol", keep="first")
        )

        with_strike = int(option["strike_price"].notna().sum())
        with_price = int(
            pd.to_numeric(option["close"], errors="coerce").gt(0).sum()
        )
        with_settle = int(
            pd.to_numeric(option["settle"], errors="coerce").gt(0).sum()
        )
        with_future = int(futures["symbol"].isin(option["symbol"]).sum())

        iv_symbols = 0
        iv_values = []
        for sym in sorted(option["symbol"].dropna().unique()):
            og = option[option["symbol"].eq(sym)]
            fg = futures[futures["symbol"].eq(sym)]
            fr = fg.iloc[0] if not fg.empty else None
            iv = _atm_iv(
                og,
                fr,
                pd.Timestamp(x["date"].iloc[0]),
            )
            if np.isfinite(iv):
                iv_symbols += 1
                iv_values.append(float(iv))

        return {
            "file": str(path),
            "date": str(x["date"].iloc[0].date()),
            "rows": int(len(x)),
            "symbols": int(x["symbol"].nunique()),
            "option_rows": int(len(option)),
            "option_rows_with_strike": with_strike,
            "option_rows_with_positive_close": with_price,
            "option_rows_with_positive_settle": with_settle,
            "symbols_with_future": with_future,
            "symbols_with_recoverable_atm_iv": iv_symbols,
            "median_atm_iv": (
                float(np.median(iv_values))
                if iv_values
                else np.nan
            ),
            "status": "PASS",
        }
    except Exception as exc:
        return {
            "file": str(path),
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _candidate_days(sample_days: int) -> list[pd.Timestamp]:
    end = min(
        pd.Timestamp.today().normalize(),
        pd.Timestamp("2026-12-31"),
    )
    start = pd.Timestamp("2026-01-01")
    dates = pd.date_range(start, end, freq="B")
    if sample_days > 0:
        dates = dates[-sample_days:]
    return [pd.Timestamp(d) for d in dates[::-1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default=str(RAW_BASE))
    ap.add_argument("--sample-days", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.raw_root)
    print("=" * 90)
    print("NSE F&O IV INPUT AUDIT")
    print("=" * 90)
    print("BASE:", BASE)
    print("RAW_BASE:", root)
    print("INDIAN_MARKET_RESEARCH_ROOT:", __import__("os").environ.get("INDIAN_MARKET_RESEARCH_ROOT", ""))

    try:
        store = GoogleDriveStore()
        print("DRIVE_STATUS: CONNECTED")
        print("DRIVE_ROOT_ID:", store.root_id)
    except Exception as exc:
        print(
            "DRIVE_STATUS: FAILED",
            f"{type(exc).__name__}: {exc}",
        )
        store = None

    found, discovery_status = _drive_discover_candidates(
        max_results=max(50, args.sample_days)
    ) if store is not None else ([], "SKIPPED_NO_DRIVE")
    print("DRIVE_DISCOVERY_STATUS:", discovery_status)
    print("DRIVE_CANDIDATE_COUNT:", len(found))
    if found:
        print("\nDrive candidates:")
        for f in found[:20]:
            print(
                f"- name={f.get('name')} "
                f"size={f.get('size')} "
                f"mime={f.get('mimeType')} "
                f"parents={f.get('parents')}"
            )

    files = sorted(
        root.rglob(f"{ARCHIVE_PREFIX}*{ARCHIVE_SUFFIX}")
    )
    print("LOCAL_ARCHIVE_COUNT_BEFORE_MATERIALIZE:", len(files))

    materialized = 0
    materialize_rows = []
    for day in _candidate_days(max(args.sample_days, 1)):
        path = _archive_path(day)
        ok, status = _materialize_drive_archive(path, day)
        materialize_rows.append(
            {"date": str(day.date()), "status": status, "path": str(path)}
        )
        if ok:
            materialized += 1

    files = sorted(
        root.rglob(f"{ARCHIVE_PREFIX}*{ARCHIVE_SUFFIX}")
    )
    print("DRIVE_ARCHIVES_MATERIALIZED_OR_LOCAL:", materialized)
    print("LOCAL_ARCHIVE_COUNT_AFTER_MATERIALIZE:", len(files))

    if not files:
        print("\nMaterialization attempts:")
        for row in materialize_rows[:20]:
            print(f"- {row['date']}: {row['status']}")
        raise SystemExit(
            "No NSE F&O raw archives found locally or through the configured "
            "Google Drive path. The discovery lines above distinguish a "
            "missing folder/path from a Drive-auth/access problem."
        )

    if args.sample_days > 0:
        files = sorted(files, reverse=True)[: args.sample_days]
        files = sorted(files)

    rows = [audit_file(p) for p in files]
    df = pd.DataFrame(rows)

    print("\nfiles_scanned:", len(df))
    good = df[df["status"].eq("PASS")].copy()
    if not good.empty:
        print("option_rows:", int(good["option_rows"].sum()))
        print(
            "option_rows_with_strike:",
            int(good["option_rows_with_strike"].sum()),
        )
        print(
            "option_rows_with_positive_close:",
            int(good["option_rows_with_positive_close"].sum()),
        )
        print(
            "option_rows_with_positive_settle:",
            int(good["option_rows_with_positive_settle"].sum()),
        )
        print("symbols_with_future:", int(good["symbols_with_future"].sum()))
        print(
            "symbols_with_recoverable_atm_iv:",
            int(
                good["symbols_with_recoverable_atm_iv"].sum()
            ),
        )
        display_cols = [
            "date",
            "symbols",
            "option_rows",
            "option_rows_with_strike",
            "option_rows_with_positive_close",
            "option_rows_with_positive_settle",
            "symbols_with_future",
            "symbols_with_recoverable_atm_iv",
            "median_atm_iv",
        ]
        print("\nPer-file summary:")
        print(good[display_cols].to_string(index=False))

    bad = df[df["status"].ne("PASS")]
    if not bad.empty:
        print("\nFailures:")
        print(bad.to_string(index=False))

    out = Path(
        "/content/drive/MyDrive/Indian_Market_Research/"
        "diagnostics/fno_iv_input_audit_20261003.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nOUTPUT:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
