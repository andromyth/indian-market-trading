#!/usr/bin/env python3
"""Autonomous validation of stored Delivery predictions against official NSE EOD closes.

D+1 is the model-target-aligned validation. D+2..D+7 are preserved as
seven-session follow-through monitoring for each prediction. No retraining or
model promotion occurs here.

Validation consumes the shared official NSE bulk-EOD archive already prepared
by the production refresh. It deliberately does not bootstrap per-symbol
65-session histories.
"""
from __future__ import annotations

from datetime import date, timedelta
import re
from pathlib import Path

import pandas as pd

from prediction_ledger import load_records, load_validation_records, save_validation_results
from prediction_validation import ValidationConfig, validate_ledger, rolling_7day_scorecard
from model_health import assess_delivery_health

BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
EOD_LOCAL_ROOT = BASE / "production_data" / "nse_official" / "bulk_eod"


def _load_shared_eod(today: date, lookback_days: int = 370) -> pd.DataFrame:
    """Load the persisted official NSE bulk EOD archive once for all symbols."""
    parts = []
    if not EOD_LOCAL_ROOT.exists():
        return pd.DataFrame()

    rx = re.compile(r"BhavCopy_NSE_CM_0_0_0_(\d{8})_F_0000\.parquet$", re.I)
    start = today - timedelta(days=lookback_days)

    for path in EOD_LOCAL_ROOT.rglob("BhavCopy_NSE_CM_0_0_0_*_F_0000.parquet"):
        m = rx.match(path.name)
        if not m:
            continue
        try:
            d = pd.Timestamp(m.group(1)).date()
        except Exception:
            continue
        if not (start <= d <= today):
            continue

        try:
            x = pd.read_parquet(path)
        except Exception:
            continue
        if x.empty:
            continue

        cols = {str(c).strip().lower(): c for c in x.columns}
        def pick(names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        ren = {}
        for target, names in {
            "symbol": ["symbol", "tckrsymb", "ticker"],
            "date": ["date", "trade_date", "trad_dt"],
            "open": ["open", "open_price", "opnpric"],
            "high": ["high", "high_price", "hghpric"],
            "low": ["low", "low_price", "lwpric"],
            "close": ["close", "close_price", "clspric"],
            "volume": ["volume", "traded_volume", "ttltradgvol"],
        }.items():
            c = pick(names)
            if c is not None:
                ren[c] = target

        x = x.rename(columns=ren)
        if not {"symbol", "date", "close"}.issubset(x.columns):
            continue

        x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
        x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
        x["close"] = pd.to_numeric(x["close"], errors="coerce")

        keep = ["symbol", "date", "close"]
        for c in ("open", "high", "low", "volume"):
            if c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
                keep.append(c)

        x = x.dropna(subset=["symbol", "date", "close"])
        parts.append(x[keep])

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = (
        out.sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    return out


def _history_by_symbol(frame: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    if frame.empty or "symbol" not in frame.columns:
        return {s: pd.DataFrame() for s in symbols}

    wanted = {s.upper() for s in symbols}
    x = frame[frame["symbol"].isin(wanted)].copy()

    out = {}
    for symbol, group in x.groupby("symbol", sort=False):
        y = group.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        ).copy()
        y["Date"] = pd.to_datetime(y["Date"], errors="coerce")
        out[str(symbol).upper()] = (
            y.dropna(subset=["Date"])
            .set_index("Date")
            .sort_index()
        )

    return {s: out.get(s, pd.DataFrame()) for s in symbols}


def main() -> int:
    records = [
        r for r in load_records()
        if str(r.get("mode", "")).lower() == "delivery"
        and r.get("decision_timestamp")
    ]
    if not records:
        print("SUMMARY:", {"status": "NO_PREDICTIONS"})
        return 0

    existing = {str(r.get("prediction_id")): r for r in load_validation_records() if r.get("prediction_id")}
    pending = [
        r for r in records
        if (
            str(r.get("prediction_id")) not in existing
            or existing[str(r.get("prediction_id"))].get("validation_status") != "VALIDATED"
            or not all(existing[str(r.get("prediction_id"))].get(f"d{i}_actual_close") is not None for i in range(1, 8))
        )
    ]
    if not pending:
        current = pd.DataFrame(list(existing.values()))
        print("SUMMARY:", {**rolling_7day_scorecard(current), **assess_delivery_health(current)})
        return 0

    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    symbols = list(dict.fromkeys(str(r["symbol"]).upper() for r in pending))

    print(
        f"VALIDATION_HISTORY_SOURCE: shared_nse_bulk_eod symbols={len(symbols)} "
        f"as_of={today.isoformat()}",
        flush=True,
    )

    shared = _load_shared_eod(today)
    print(
        f"VALIDATION_SHARED_EOD_READY: rows={len(shared)} "
        f"symbols={shared['symbol'].nunique() if not shared.empty else 0} "
        f"last_date={shared['date'].max().date().isoformat() if not shared.empty else None}",
        flush=True,
    )

    by_symbol = _history_by_symbol(shared, symbols)
    ready = sum(not frame.empty for frame in by_symbol.values())
    print(
        f"VALIDATION_SYMBOL_HISTORIES_READY: {ready}/{len(symbols)}",
        flush=True,
    )

    def provider(symbol: str) -> pd.DataFrame:
        return by_symbol.get(symbol.upper(), pd.DataFrame())

    validated = validate_ledger(pending, provider, ValidationConfig(rolling_days=7))
    if not validated.empty:
        save_validation_results(validated.to_dict("records"), sync_drive=True)

    all_validations = {str(r.get("prediction_id")): r for r in load_validation_records() if r.get("prediction_id")}
    combined = pd.DataFrame(list(all_validations.values()))
    score = rolling_7day_scorecard(combined)
    health = assess_delivery_health(combined)

    print("AUTONOMOUS DELIVERY VALIDATION")
    print("Pending predictions:", len(pending))
    print("Validation result rows:", len(validated))
    print("7-day scorecard:", score)
    print("Model health:", health)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
