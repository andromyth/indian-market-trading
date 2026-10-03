#!/usr/bin/env python3
"""Audit whether persisted NSE F&O UDiFF archives contain enough inputs for ATM IV."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nse_fno_udiff_ingest import _atm_iv, _normalize, _read_zip, UDIFF_RAW_BASE

ARCHIVE_PREFIX = "BhavCopy_NSE_FO_0_0_0_"
ARCHIVE_SUFFIX = "_F_0000.csv.zip"
BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", "/content/drive/MyDrive/Indian_Market_Research"))
OUTPUT = BASE / "diagnostics/fno_iv_input_audit_20261003.csv"

KNOWN_ROOTS = [
    BASE / "production_data/nse_official/fno/raw_udiff",
    BASE / "production_data/nse_official/fno",
    BASE / "fno/raw/2026",
    BASE / "raw/nse_fno",
]


def discover_archives() -> tuple[Path, list[tuple[Path, pd.Timestamp, str]]]:
    """Discover flat persisted F&O archives reliably on the mounted Drive."""
    found: dict[Path, tuple[pd.Timestamp, str]] = {}

    udiff_roots = [
        BASE / "production_data/nse_official/fno/raw_udiff",
        BASE / "production_data/nse_official/fno",
    ]

    for root in udiff_roots:
        if not root.exists():
            continue
        try:
            for pattern in (
                "BhavCopy_NSE_FO_*_F_0000.csv.zip",
                "BhavCopy_NSE_FO_0_0_0_*_F_0000.csv.zip",
            ):
                for p in root.glob(pattern):
                    if not p.is_file():
                        continue
                    try:
                        token = p.name.split(
                            "BhavCopy_NSE_FO_", 1
                        )[1][:8]
                        day = pd.Timestamp(
                            token,
                            format="%Y%m%d",
                        )
                        found[p] = (
                            day,
                            "NSE_UDIFF",
                        )
                    except Exception:
                        continue
        except OSError as exc:
            print(
                f"DISCOVERY_WARNING: {root}: "
                f"{type(exc).__name__}: {exc}"
            )

    # Legacy pre-UDiFF archive family retained for diagnostics only.
    legacy_root = BASE / "fno/raw/2026"
    if legacy_root.exists():
        try:
            for p in legacy_root.glob("fo_*.zip"):
                if not p.is_file():
                    continue
                try:
                    token = p.stem.split(
                        "fo_", 1
                    )[1][:8]
                    day = pd.Timestamp(
                        token,
                        format="%Y%m%d",
                    )
                    found[p] = (
                        day,
                        "LEGACY_FO",
                    )
                except Exception:
                    continue
        except OSError as exc:
            print(
                f"DISCOVERY_WARNING: {legacy_root}: "
                f"{type(exc).__name__}: {exc}"
            )

    items = [
        (path, day, source)
        for path, (day, source) in sorted(
            found.items(),
            key=lambda kv: (
                kv[1][0],
                str(kv[0]),
            ),
        )
    ]
    return BASE, items

def audit_file(path: Path) -> dict:
    try:
        raw = _read_zip(path.read_bytes())
        token = path.name.split("_0_0_0_")[1][:8]
        day = pd.Timestamp(token, format="%Y%m%d").date()
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
        with_close = int(
            pd.to_numeric(option["close"], errors="coerce").gt(0).sum()
        )
        with_settle = int(
            pd.to_numeric(option["settle"], errors="coerce").gt(0).sum()
        )
        with_underlying = int(
            pd.to_numeric(
                option["underlying_price"], errors="coerce"
            ).gt(0).sum()
        )
        symbols_with_future = int(
            futures["symbol"].isin(option["symbol"]).sum()
        )

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
            "date": str(day),
            "relative_to_research_root": str(path.relative_to(BASE))
            if str(path).startswith(str(BASE) + os.sep)
            else str(path),
            "rows": int(len(x)),
            "symbols": int(x["symbol"].nunique()),
            "option_rows": int(len(option)),
            "option_rows_with_strike": with_strike,
            "option_rows_with_positive_close": with_close,
            "option_rows_with_positive_settle": with_settle,
            "option_rows_with_positive_underlying": with_underlying,
            "symbols_with_future": symbols_with_future,
            "symbols_with_recoverable_atm_iv": iv_symbols,
            "median_atm_iv": float(np.median(iv_values))
            if iv_values
            else np.nan,
            "status": "PASS",
        }
    except Exception as exc:
        return {
            "file": str(path),
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-days", type=int, default=20)
    args = ap.parse_args()

    print("=" * 90)
    print("NSE F&O IV INPUT AUDIT")
    print("=" * 90)
    print("RESEARCH_ROOT:", BASE)
    print("CANONICAL_RAW_ROOT:", UDIFF_RAW_BASE)
    print(
        "INDIAN_MARKET_RESEARCH_ROOT:",
        os.environ.get("INDIAN_MARKET_RESEARCH_ROOT", ""),
    )

    _, files = discover_archives()
    print("ARCHIVE_DISCOVERY_PATHS:")
    for root in KNOWN_ROOTS:
        print(f"- {root}: {'EXISTS' if root.exists() else 'MISSING'}")
    print("RAW_UDIFF_ARCHIVES_FOUND_ANYWHERE_UNDER_RESEARCH_ROOT:", len(files))

    if not files:
        raise SystemExit(
            "No NSE F&O UDiFF archives found anywhere under the mounted "
            f"research root: {BASE}"
        )

    if args.sample_days > 0:
        files = sorted(files, reverse=True)[: args.sample_days]
        files = sorted(files)

    rows = [audit_file(path) for path in files]
    df = pd.DataFrame(rows)
    good = df[df["status"].eq("PASS")].copy()
    bad = df[df["status"].ne("PASS")].copy()

    print("FILES_SCANNED:", len(df))

    if not good.empty:
        for col in (
            "option_rows",
            "option_rows_with_strike",
            "option_rows_with_positive_close",
            "option_rows_with_positive_settle",
            "option_rows_with_positive_underlying",
            "symbols_with_future",
            "symbols_with_recoverable_atm_iv",
        ):
            print(f"{col.upper()}: {int(good[col].sum())}")

        print("\nPer-file summary:")
        print(
            good[
                [
                    "date",
                    "relative_to_research_root",
                    "symbols",
                    "option_rows",
                    "option_rows_with_strike",
                    "option_rows_with_positive_close",
                    "option_rows_with_positive_settle",
                    "option_rows_with_positive_underlying",
                    "symbols_with_future",
                    "symbols_with_recoverable_atm_iv",
                    "median_atm_iv",
                ]
            ].to_string(index=False)
        )

    if not bad.empty:
        print("\nFAILURES:")
        print(bad.to_string(index=False))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print("\nOUTPUT:", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
