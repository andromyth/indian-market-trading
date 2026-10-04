#!/usr/bin/env python3
"""Preflight checks for a live NSE Delivery/Intraday test.

Read-only checks by default. Optional 1-minute capture can verify the live NSE
chart path for selected NSE symbols. No orders are placed and no model is
modified.
"""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from all_nse_delivery_model import model_status as delivery_model_status
from persistent_store import drive_status
from prediction_ledger import load_records, load_validation_records
from production_model import model_status as intraday_model_status
from data_fetcher import market_is_open
from nse_data_layer import fetch_intraday_1m_current


IST = ZoneInfo("Asia/Kolkata")
BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
FEATURE_DIR = BASE / "features" / "daily"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _latest_snapshot() -> tuple[Path | None, str | None]:
    candidates = []
    for path in FEATURE_DIR.glob("DELIVERY_PIT_FEATURES_*.parquet"):
        try:
            d = path.stem.rsplit("_", 1)[-1]
            parsed = pd.Timestamp(d).date()
            candidates.append((parsed, path))
        except Exception:
            continue
    if not candidates:
        return None, None
    d, path = max(candidates, key=lambda x: x[0])
    return path, d.isoformat()


def _ledger_summary() -> dict:
    records = load_records()
    delivery = [
        r for r in records
        if str(r.get("mode", "")).lower() == "delivery"
    ]
    intraday = [
        r for r in records
        if str(r.get("mode", "")).lower() == "intraday"
    ]

    dates = []
    for r in delivery:
        ts = pd.to_datetime(r.get("decision_timestamp"), errors="coerce")
        if pd.notna(ts):
            dates.append(ts)

    return {
        "total_predictions": len(records),
        "delivery_predictions": len(delivery),
        "intraday_predictions": len(intraday),
        "latest_delivery_decision": str(max(dates)) if dates else None,
        "validation_records": len(load_validation_records()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols",
        default="BHARTIARTL,RELIANCE,SBIN",
        help="Comma-separated NSE symbols for optional live 1-minute capture.",
    )
    ap.add_argument(
        "--capture-1m",
        action="store_true",
        help="Perform a read-only live NSE 1-minute capture for the selected symbols.",
    )
    args = ap.parse_args()

    now = datetime.now(IST)
    print("LIVE TEST PREFLIGHT")
    print("local_time_ist:", now.isoformat())
    print("git_sha:", _git_sha())
    print("market_is_open:", market_is_open())

    failures = []

    try:
        drive = drive_status()
        print("drive_status:", drive)
        if drive.get("configured") and drive.get("reachable", True) is False:
            failures.append("Google Drive is configured but unreachable.")
    except Exception as exc:
        print("drive_status: ERROR", type(exc).__name__, str(exc))
        failures.append("Google Drive status check failed.")

    try:
        delivery = delivery_model_status()
        print(
            "delivery_model:",
            {
                "status": delivery.get("status"),
                "engine_version": delivery.get("engine_version"),
                "feature_count": delivery.get("feature_count"),
                "trainable_symbol_count": delivery.get("trainable_symbol_count"),
                "universe_symbol_count": delivery.get("universe_symbol_count"),
                "minimum_history_sessions": delivery.get("minimum_history_sessions"),
            },
        )
        if delivery.get("status") != "READY" or delivery.get("feature_count") != 59:
            failures.append("Delivery model is not READY with the locked 59-feature contract.")
    except Exception as exc:
        print("delivery_model: ERROR", type(exc).__name__, str(exc))
        failures.append("Delivery model status check failed.")

    try:
        intraday = intraday_model_status()
        print(
            "intraday_model:",
            {
                "target": intraday.get("target"),
                "feature_count": intraday.get("feature_count"),
                "manifest_sha256_match": intraday.get("manifest_sha256_match"),
            },
        )
        if intraday.get("feature_count") != 137 or not intraday.get("manifest_sha256_match"):
            failures.append("Intraday model is not verified with the 137-feature certified contract.")
    except Exception as exc:
        print("intraday_model: ERROR", type(exc).__name__, str(exc))
        failures.append("Intraday model status check failed.")

    snapshot_path, snapshot_date = _latest_snapshot()
    print("latest_delivery_snapshot:", snapshot_path)
    print("latest_delivery_snapshot_date:", snapshot_date)
    if snapshot_path is None:
        failures.append("No persisted Delivery PIT snapshot is available.")
    else:
        try:
            snap = pd.read_parquet(snapshot_path, columns=["symbol", "production_data_quality_gate"])
            gate_true = int(snap["production_data_quality_gate"].fillna(False).astype(bool).sum())
            print(
                "snapshot:",
                {
                    "rows": len(snap),
                    "symbols": snap["symbol"].nunique(),
                    "quality_gate_true": gate_true,
                    "quality_gate_pct": round(100 * gate_true / len(snap), 3) if len(snap) else 0.0,
                },
            )
            if len(snap) == 0 or gate_true != len(snap):
                failures.append("Latest Delivery PIT snapshot is not fully production-gated.")
        except Exception as exc:
            print("snapshot: ERROR", type(exc).__name__, str(exc))
            failures.append("Latest Delivery PIT snapshot could not be read.")

    try:
        ledger = _ledger_summary()
        print("ledger:", ledger)
        if ledger["delivery_predictions"] < 1:
            failures.append("No Delivery predictions are present in the persistent ledger.")
    except Exception as exc:
        print("ledger: ERROR", type(exc).__name__, str(exc))
        failures.append("Prediction ledger check failed.")

    if args.capture_1m:
        symbols = [
            s.strip().upper().replace(".NS", "").replace(".BO", "")
            for s in args.symbols.split(",")
            if s.strip()
        ]
        print("live_1m_capture_symbols:", symbols)
        for symbol in symbols:
            try:
                frame = fetch_intraday_1m_current(symbol, sync_drive=True)
                rows = 0 if frame is None else len(frame)
                print(
                    f"live_1m: {symbol} rows={rows} "
                    f"status={'READY' if rows else 'NO_DATA'}"
                )
                if rows == 0:
                    failures.append(f"No live NSE 1-minute data for {symbol}.")
            except Exception as exc:
                print(
                    f"live_1m: {symbol} ERROR "
                    f"{type(exc).__name__}: {exc}"
                )
                failures.append(f"Live NSE 1-minute capture failed for {symbol}.")

    print("PREFLIGHT_STATUS:", "PASS" if not failures else "FAIL")
    if failures:
        for item in failures:
            print(" -", item)
        return 1

    print(
        "READY_FOR_LIVE_TEST: "
        "Delivery uses the latest persisted PIT snapshot; "
        "Intraday uses the certified 137-feature live NSE 1-minute path."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
