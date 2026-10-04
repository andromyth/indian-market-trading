#!/usr/bin/env python3
"""Audit whether persisted NSE F&O UDiFF archives contain enough inputs for ATM IV."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from nse_fno_udiff_ingest import _atm_iv, _normalize, _read_zip, UDIFF_RAW_BASE

BASE = Path(
    os.getenv(
        "INDIAN_MARKET_RESEARCH_ROOT",
        "/content/drive/MyDrive/Indian_Market_Research",
    )
)
OUTPUT = BASE / "diagnostics/fno_iv_input_audit_20261003.csv"


def _parse_udiff_date(name: str) -> pd.Timestamp | None:
    """Parse YYYYMMDD from either flat NSE archive filename family."""
    marker = "BhavCopy_NSE_FO_"
    if marker in name:
        rest = name.split(marker, 1)[1]
        token = rest[:8]
    elif name.startswith("fo_"):
        token = name[3:11]
    else:
        return None
    try:
        return pd.Timestamp(token, format="%Y%m%d")
    except Exception:
        return None


def _record_archive(found: dict[Path, tuple[pd.Timestamp, str]], p: Path, source: str) -> None:
    if not p.is_file() or p.stat().st_size <= 0:
        return
    day = _parse_udiff_date(p.name)
    if day is not None:
        found[p] = (day, source)



def _archive_source(path: Path) -> str:
    if path.name.startswith("BhavCopy_NSE_FO_"):
        return "NSE_UDIFF"
    if path.name.startswith("fo_"):
        return "LEGACY_FO"
    return "UNKNOWN"


def discover_archives() -> tuple[Path, list[tuple[Path, pd.Timestamp, str]]]:
    """
    Discover persisted archives by probing exact expected paths for
    trading dates rather than recursively enumerating Google Drive.
    """
    from datetime import date, timedelta

    found: list[tuple[Path, pd.Timestamp, str]] = []

    # Work backwards so --sample-days can stop after finding the newest
    # available persisted dates.
    current = min(date.today(), date(2026, 12, 31))
    start = date(2026, 1, 1)

    while current >= start:
        if current.weekday() < 5:
            path = None

            # Importing this helper keeps audit and rebuild discovery
            # identical.
            from nse_fno_udiff_ingest import _existing_local_archive
            path = _existing_local_archive(current)

            if path is not None:
                day = pd.Timestamp(current)
                found.append((path, day, _archive_source(path)))

        current -= timedelta(days=1)

    found.sort(key=lambda x: (x[1], str(x[0])))
    return BASE, found


def _select_stock_futures(x: pd.DataFrame) -> pd.DataFrame:
    """Select nearest non-expired stock future using both UDiFF type/name evidence."""
    if x.empty:
        return x.iloc[0:0].copy()

    instrument = x["instrument"].astype(str).str.upper()
    fin_name = x.get(
        "fin_instrm_nm",
        pd.Series("", index=x.index),
    ).astype(str).str.upper()
    future_mask = (
        instrument.str.contains("FUT", na=False)
        | fin_name.str.contains(r"\bFUT\b|FUT", regex=True, na=False)
    )
    index_mask = x["symbol"].isin(
        {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
    )
    futures = x[future_mask & ~index_mask].copy()
    if futures.empty:
        return futures

    futures = futures.dropna(subset=["expiry"])
    futures = futures[futures["expiry"] >= x["date"].iloc[0]]
    return (
        futures.sort_values(["symbol", "expiry"])
        .drop_duplicates("symbol", keep="first")
    )


def audit_file(path: Path, day_hint: pd.Timestamp | None = None, source: str = "NSE_UDIFF") -> dict:
    try:
        raw = _read_zip(path.read_bytes())
        day = day_hint or _parse_udiff_date(path.name)
        if day is None:
            raise RuntimeError(f"Cannot determine trading date from archive: {path.name}")

        x = _normalize(raw, day.date())

        # Preserve UDiFF's instrument-name evidence when available.
        name_col = next(
            (c for c in raw.columns if "".join(ch for ch in str(c).strip().lower() if ch.isalnum()) == "fininstrmnm"),
            None,
        )
        if name_col:
            x["fin_instrm_nm"] = raw[name_col].astype(str).str.strip().str.upper()

        option_mask = x["option_type"].isin(["CE", "PE", "CALL", "PUT"])
        option = x[option_mask].copy()
        futures = _select_stock_futures(x)

        with_strike = int(option["strike_price"].notna().sum())
        with_close = int(
            pd.to_numeric(option["close"], errors="coerce").gt(0).sum()
        )
        with_settle = int(
            pd.to_numeric(option["settle"], errors="coerce").gt(0).sum()
        )
        with_underlying = int(
            pd.to_numeric(option["underlying_price"], errors="coerce").gt(0).sum()
        )
        symbols_with_future = int(
            futures["symbol"].isin(option["symbol"]).sum()
        )

        iv_symbols = 0
        iv_values: list[float] = []

        for sym in sorted(option["symbol"].dropna().unique()):
            og = option[option["symbol"].eq(sym)]
            fg = futures[futures["symbol"].eq(sym)]
            fr = fg.iloc[0] if not fg.empty else None
            iv = _atm_iv(og, fr, pd.Timestamp(day))
            if np.isfinite(iv):
                iv_symbols += 1
                iv_values.append(float(iv))

        instrument_counts = x["instrument"].value_counts().head(20).to_dict()
        option_type_counts = x["option_type"].value_counts().head(10).to_dict()

        return {
            "file": str(path),
            "date": str(day.date()),
            "source": source,
            "relative_to_research_root": str(path.relative_to(BASE)),
            "rows": int(len(x)),
            "symbols": int(x["symbol"].nunique()),
            "option_rows": int(len(option)),
            "option_rows_with_strike": with_strike,
            "option_rows_with_positive_close": with_close,
            "option_rows_with_positive_settle": with_settle,
            "option_rows_with_positive_underlying": with_underlying,
            "symbols_with_future": symbols_with_future,
            "symbols_with_recoverable_atm_iv": iv_symbols,
            "median_atm_iv": float(np.median(iv_values)) if iv_values else np.nan,
            "instrument_types": str(instrument_counts),
            "option_types": str(option_type_counts),
            "status": "PASS",
        }
    except Exception as exc:
        return {
            "file": str(path),
            "date": None,
            "source": source,
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
    print("EXACT_PATH_ARCHIVES_FOUND:", len(files))

    if not files:
        raise SystemExit(
            "No supported NSE F&O archives found under mounted research root: "
            f"{BASE}"
        )

    # Prefer canonical UDiFF archives for each date over legacy archives.
    by_day: dict[pd.Timestamp, tuple[Path, pd.Timestamp, str]] = {}
    for item in files:
        path, day, source = item
        key = day.normalize()
        previous = by_day.get(key)
        if previous is None or (
            source == "NSE_UDIFF" and previous[2] != "NSE_UDIFF"
        ):
            by_day[key] = item

    selected_days = sorted(by_day)
    if args.sample_days > 0:
        selected_days = selected_days[-args.sample_days:]
    selected = [by_day[d] for d in selected_days]

    print("FILES_SELECTED:", len(selected))
    if selected:
        print(
            "SELECTED_RANGE:",
            selected[0][1].date(),
            "->",
            selected[-1][1].date(),
        )

    rows = [audit_file(path, day, source) for path, day, source in selected]
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
            print(f"{col.upper()}:", int(good[col].sum()))

        print("\nPer-file summary:")
        print(
            good[
                [
                    "date",
                    "source",
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
