#!/usr/bin/env python3
"""Validate matured next-session Delivery predictions against official NSE EOD bars.

This command writes only prediction-validation records. It does not retrain or
alter model artifacts, predictions, policy thresholds, or trading behaviour.
"""
from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import pandas as pd

from delivery_prediction_csv_ledger import load_eod_history
from prediction_ledger import load_records, load_validation_records, save_validation_results
from prediction_validation import ValidationConfig, validate_ledger

BASE = os.getenv("INDIAN_MARKET_RESEARCH_ROOT", "/content/drive/MyDrive/Indian_Market_Research")


def _root(symbol: Any) -> str:
    value = str(symbol or "").strip().upper()
    for suffix in (".NS", ".BO"):
        if value.endswith(suffix):
            value = value[:-3]
    return value


def _normalise_eod(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return validated EOD rows with the exact columns prediction_validation expects."""
    columns = ["symbol", "date", "Open", "High", "Low", "Close"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    x = frame.copy()
    aliases = {str(c).strip().lower(): c for c in x.columns}
    required = {"symbol", "date", "open", "high", "low", "close"}
    if not required.issubset(aliases):
        return pd.DataFrame(columns=columns)
    x = x.rename(columns={
        aliases["symbol"]: "symbol",
        aliases["date"]: "date",
        aliases["open"]: "Open",
        aliases["high"]: "High",
        aliases["low"]: "Low",
        aliases["close"]: "Close",
    })
    x["symbol"] = x["symbol"].map(_root)
    parsed_dates = pd.to_datetime(x["date"], errors="coerce")
    if getattr(parsed_dates.dt, "tz", None) is not None:
        parsed_dates = parsed_dates.dt.tz_localize(None)
    x["date"] = parsed_dates.dt.normalize()
    for col in ("Open", "High", "Low", "Close"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna(subset=["symbol", "date", "Open", "High", "Low", "Close"])
    x = x.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return x[columns].reset_index(drop=True)


def _target_date(record: dict[str, Any]) -> date | None:
    raw = record.get("prediction_for_date")
    if raw in (None, ""):
        return None
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def main() -> int:
    history = _normalise_eod(load_eod_history())
    if history.empty:
        print(json.dumps({
            "status": "NO_OFFICIAL_EOD_HISTORY",
            "data_root": BASE,
            "validated": 0,
        }, indent=2, default=str))
        return 2

    latest_eod_date = history["date"].max().date()
    groups = {
        symbol: part.set_index("date")[["Open", "High", "Low", "Close"]].sort_index()
        for symbol, part in history.groupby("symbol", sort=False)
    }

    def history_provider(symbol: str) -> pd.DataFrame:
        return groups.get(_root(symbol), pd.DataFrame(columns=["Open", "High", "Low", "Close"])).copy()

    records = [
        row for row in load_records()
        if str(row.get("mode", "")).strip().lower() == "delivery"
        and row.get("prediction_id")
    ]
    validations = load_validation_records()
    latest_by_id: dict[str, dict[str, Any]] = {}
    for row in validations:
        if row.get("prediction_id"):
            latest_by_id[str(row["prediction_id"])] = row

    already_validated = {
        pid for pid, row in latest_by_id.items()
        if str(row.get("validation_status", "")).upper() == "VALIDATED"
        and pd.notna(pd.to_numeric(row.get("actual_d1_close"), errors="coerce"))
        and str(row.get("actual_d1_date", "")).strip()
    }

    due: list[dict[str, Any]] = []
    unkeyed_target = 0
    for row in records:
        pid = str(row["prediction_id"])
        if pid in already_validated:
            continue
        target = _target_date(row)
        if target is None:
            unkeyed_target += 1
            continue
        # No target bar should be scored until that official session has arrived.
        if target <= latest_eod_date:
            due.append(row)

    if not due:
        report = {
            "status": "NO_UNRESOLVED_MATURED_DELIVERY_PREDICTIONS",
            "latest_official_eod_date": latest_eod_date.isoformat(),
            "delivery_predictions_seen": len(records),
            "already_validated": len(already_validated),
            "unusable_target_dates": unkeyed_target,
            "validated": 0,
            "saved": False,
        }
        print(json.dumps(report, indent=2))
        return 0

    results = validate_ledger(due, history_provider, ValidationConfig())
    rows = results.to_dict("records")
    saved = save_validation_results(rows, sync_drive=True)
    counts = results["validation_status"].astype(str).str.upper().value_counts().to_dict() if not results.empty else {}
    report = {
        "status": "PASS" if saved.get("synced") is True else "PERSISTENCE_ERROR",
        "latest_official_eod_date": latest_eod_date.isoformat(),
        "delivery_predictions_seen": len(records),
        "already_validated": len(already_validated),
        "due_predictions": len(due),
        "validation_status_counts": counts,
        "saved_validation_rows": saved.get("rows"),
        "drive_synced": saved.get("synced"),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0 if saved.get("synced") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
