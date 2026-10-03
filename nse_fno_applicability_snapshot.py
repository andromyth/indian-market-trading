#!/usr/bin/env python3
"""Persist a current official NSE F&O underlying-universe snapshot.

This is a current applicability reference, not a historical PIT eligibility
table. Historical training must continue to use date-specific observed F&O
archives.
"""
from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

try:
    from nse_data_layer import HEADERS
except Exception:
    HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/html,*/*"}

from persistent_store import GoogleDriveStore

BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
URL = "https://www.nseindia.com/static/products-services/equity-derivatives-list-underlyings-information"
OUT = BASE / "production_data/nse_official/fno/fno_current_underlyings_2026.parquet"
META = BASE / "production_data/nse_official/fno/fno_current_underlyings_2026.meta.json"

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "NIFTYFPI150"}


def normalize_symbol(value: object) -> str:
    s = str(value or "").strip().upper()
    for suffix in (".NS", ".BO", "-EQ", ".EQ"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def main() -> None:
    if not BASE.exists():
        raise SystemExit(f"Google Drive research root not found: {BASE}")

    s = requests.Session()
    s.headers.update({**HEADERS, "Referer": "https://www.nseindia.com/", "Accept-Language": "en-GB,en;q=0.9"})
    r = s.get(URL, timeout=(10, 30))
    r.raise_for_status()

    tables = pd.read_html(io.StringIO(r.text))
    rows = []
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        symbol_col = next((c for k, c in cols.items() if "symbol" in k or "underlying" in k), None)
        if symbol_col is None:
            continue
        frame = table.copy()
        frame["symbol"] = frame[symbol_col].map(normalize_symbol)
        frame = frame[frame["symbol"].ne("")]
        frame = frame[~frame["symbol"].isin(INDEX_SYMBOLS)]
        for sym in frame["symbol"].drop_duplicates():
            rows.append({"symbol": sym, "source": "NSE_CURRENT_LIST_OF_UNDERLYINGS"})

    out = pd.DataFrame(rows).drop_duplicates("symbol").sort_values("symbol").reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Unable to parse any current NSE F&O individual-security symbols")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    meta = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_url": URL,
        "source_note": "Current official NSE List of Underlyings and Information; not a historical PIT eligibility table",
        "symbols": int(len(out)),
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    store = GoogleDriveStore()
    store.upload_or_update(OUT, "production_data/nse_official/fno/fno_current_underlyings_2026.parquet")
    store.upload_or_update(META, "production_data/nse_official/fno/fno_current_underlyings_2026.meta.json", mime_type="application/json")

    print(f"CURRENT NSE F&O UNDERLYINGS: {len(out):,}")
    print(f"OUTPUT: {OUT}")


if __name__ == "__main__":
    main()
