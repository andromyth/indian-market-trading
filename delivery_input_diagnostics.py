#!/usr/bin/env python3
"""Read-only readiness audit for the active ALL-NSE 59-feature Delivery path.

The audit compares the official current NSE equity universe with the latest
persisted point-in-time (PIT) feature snapshot. It does not use the retired
three-symbol 53-feature adapter, emit predictions, retrain, or write to Drive.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import json

import pandas as pd

from delivery_training_universe import load_official_nse_universe
from all_nse_delivery_model import _latest_drive_pit_features, _load_manifest

IST = ZoneInfo("Asia/Kolkata")
MIN_COMPLETE_SYMBOL_COVERAGE_PCT = 90.0
CORE_READY_FLAGS = (
    "production_data_quality_gate",
    "ready_market_price",
    "ready_market_context",
)
KNOWN_STATE_FLAGS = (
    "ready_news",
    "ready_events",
    "ready_fno",
    "ready_fundamentals",
)


def _symbol_column(frame: pd.DataFrame) -> str:
    for col in ("_symbol", "symbol", "SYMBOL", "CH_SYMBOL", "SYMBOL_NAME"):
        if col in frame.columns:
            return col
    raise RuntimeError(f"No symbol column found; columns={list(frame.columns)}")


def _latest_per_symbol(frame: pd.DataFrame, as_of) -> pd.DataFrame:
    x = frame.copy()
    x["symbol"] = x[_symbol_column(x)].astype(str).str.strip().str.upper()
    x["symbol"] = x["symbol"].str.replace(r"\.(NS|BO)$", "", regex=True)
    date_col = next(
        (c for c in ("date", "prediction_date", "as_of_date", "feature_date") if c in x.columns),
        None,
    )
    if date_col:
        x["_audit_date"] = pd.to_datetime(x[date_col], errors="coerce").dt.normalize()
        x = x[x["_audit_date"].notna() & (x["_audit_date"] <= pd.Timestamp(as_of))]
        x = x.sort_values(["symbol", "_audit_date"])
    return x.drop_duplicates("symbol", keep="last").reset_index(drop=True)


def main() -> int:
    now = datetime.now(IST)
    as_of = now.date()
    universe_frame = load_official_nse_universe()
    universe_col = _symbol_column(universe_frame)
    universe = {
        str(v).strip().upper().removesuffix(".NS").removesuffix(".BO")
        for v in universe_frame[universe_col].dropna()
        if str(v).strip()
    }
    universe.discard("")
    if not universe:
        raise RuntimeError("Official NSE universe returned zero symbols.")

    manifest = _load_manifest()
    expected_features = list(manifest.get("features", []))
    if len(expected_features) != 59 or len(set(expected_features)) != 59:
        raise RuntimeError("Active ALL-NSE Delivery manifest does not expose 59 unique features.")

    pit, snapshot_date = _latest_drive_pit_features(as_of)
    if pit is None or pit.empty:
        print("ALL_NSE_DELIVERY_READINESS", {
            "status": "FAIL",
            "reason": "latest PIT feature snapshot unavailable or empty",
            "official_nse_symbols": len(universe),
            "snapshot_date": snapshot_date,
        }, flush=True)
        return 1

    pit = _latest_per_symbol(pit, as_of)
    pit_symbols = set(pit["symbol"]) & universe
    indexed = pit.set_index("symbol", drop=False)
    missing_columns = [
        c for c in (*expected_features, *CORE_READY_FLAGS, *KNOWN_STATE_FLAGS)
        if c not in pit.columns
    ]
    if missing_columns:
        print("ALL_NSE_DELIVERY_READINESS", {
            "status": "FAIL",
            "reason": "PIT schema missing required readiness flags",
            "missing_columns": missing_columns,
            "official_nse_symbols": len(universe),
            "pit_symbols_in_universe": len(pit_symbols),
            "snapshot_date": snapshot_date,
        }, flush=True)
        return 1

    complete_symbols: list[str] = []
    blocked_reasons: dict[str, int] = {}
    gate_reason_counts: dict[str, int] = {}
    readiness_value_counts: dict[str, dict[str, int]] = {}
    for flag in (*CORE_READY_FLAGS, *KNOWN_STATE_FLAGS):
        values = pd.to_numeric(pit[flag], errors="coerce")
        readiness_value_counts[flag] = {
            "ready_1": int(values.eq(1).sum()),
            "known_0": int(values.eq(0).sum()),
            "unknown": int(values.isna().sum()),
        }
    if "production_data_quality_gate_reasons" in pit.columns:
        for raw in pit["production_data_quality_gate_reasons"].dropna():
            try:
                reasons = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                reasons = ["INVALID_QUALITY_GATE_REASON_JSON"]
            if isinstance(reasons, list):
                for reason in reasons:
                    key = str(reason)
                    gate_reason_counts[key] = gate_reason_counts.get(key, 0) + 1
    for symbol in sorted(universe):
        if symbol not in indexed.index:
            blocked_reasons["NO_PIT_ROW"] = blocked_reasons.get("NO_PIT_ROW", 0) + 1
            continue
        row = indexed.loc[symbol]
        reasons = []
        for flag in CORE_READY_FLAGS:
            value = pd.to_numeric(pd.Series([row.get(flag)]), errors="coerce").iloc[0]
            if pd.isna(value) or value != 1:
                reasons.append(flag.upper() + "_NOT_READY")
        for flag in KNOWN_STATE_FLAGS:
            value = pd.to_numeric(pd.Series([row.get(flag)]), errors="coerce").iloc[0]
            if pd.isna(value):
                reasons.append(flag.upper() + "_UNKNOWN")
        if reasons:
            for reason in reasons:
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
        else:
            complete_symbols.append(symbol)

    universe_count = len(universe)
    pit_coverage_pct = round(len(pit_symbols) * 100.0 / universe_count, 3)
    complete_pct = round(len(complete_symbols) * 100.0 / universe_count, 3)
    passed = complete_pct >= MIN_COMPLETE_SYMBOL_COVERAGE_PCT
    report = {
        "status": "PASS" if passed else "NEEDS_ATTENTION",
        "as_of_ist": now.isoformat(),
        "snapshot_date": snapshot_date,
        "model_path": "active ALL-NSE 59-feature PIT Delivery",
        "feature_contract_count": len(expected_features),
        "feature_schema_complete": True,
        "official_nse_symbols": universe_count,
        "pit_symbols_in_universe": len(pit_symbols),
        "pit_symbol_coverage_pct": pit_coverage_pct,
        "complete_symbols": len(complete_symbols),
        "complete_symbol_coverage_pct": complete_pct,
        "required_complete_symbol_coverage_pct": MIN_COMPLETE_SYMBOL_COVERAGE_PCT,
        "blocked_reason_counts": blocked_reasons,
        "quality_gate_reason_counts": gate_reason_counts,
        "readiness_flag_value_counts": readiness_value_counts,
        "missing_pit_symbol_sample": sorted(universe - pit_symbols)[:30],
        "prediction_rows_written": 0,
        "drive_writes": 0,
    }
    print("ALL_NSE_DELIVERY_READINESS", report, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
