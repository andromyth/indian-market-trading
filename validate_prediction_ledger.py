#!/usr/bin/env python3
"""Autonomous validation of stored Delivery predictions against official NSE EOD closes.

D+1 is the model-target-aligned validation. D+2..D+7 are preserved as
seven-session follow-through monitoring for each prediction. No retraining or
model promotion occurs here.
"""
from __future__ import annotations

from datetime import date
import pandas as pd

from nse_data_layer import fetch_equity_history
from prediction_ledger import load_records, load_validation_records, save_validation_results
from prediction_validation import ValidationConfig, validate_ledger, rolling_7day_scorecard
from model_health import assess_delivery_health


def official_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    df = fetch_equity_history(symbol, start, end, sync_drive=True)
    if df.empty:
        return pd.DataFrame()
    x = df.rename(columns={"date": "Date", "close": "Close"}).copy()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
    return x.dropna(subset=["Date"]).set_index("Date").sort_index()


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

    by_symbol: dict[str, pd.DataFrame] = {}
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    for r in pending:
        symbol = str(r["symbol"]).upper()
        start = pd.Timestamp(r["decision_timestamp"]).date() - pd.Timedelta(days=2)
        # Fetch enough calendar time to cover seven NSE sessions, including holidays/weekends.
        by_symbol[symbol] = official_history(symbol, start, today)

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
    print("Validated rows:", len(validated))
    print("7-day scorecard:", score)
    print("Model health:", health)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
