#!/usr/bin/env python3
"""Generate one idempotent ALL-NSE Delivery prediction batch from the PIT snapshot.

This stage scores the already-built 59-feature production snapshot. It does not
retrain, promote, or modify model artifacts. A prediction is persisted once per
(symbol, mode, horizon, decision timestamp, model version).
"""
from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from all_nse_delivery_model import _load_models, _signal_policy, _safe_float
from prediction_ledger import append_predictions


BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
FEATURE_DIR = BASE / "features" / "daily"


def _latest_snapshot_path(snapshot_date: str | None) -> tuple[Path, str]:
    if snapshot_date:
        path = FEATURE_DIR / f"DELIVERY_PIT_FEATURES_{snapshot_date}.parquet"
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"PIT snapshot not found: {path}")
        return path, snapshot_date

    candidates: list[tuple[date, Path]] = []
    rx = re.compile(r"DELIVERY_PIT_FEATURES_(20\\d{2}-\\d{2}-\\d{2})\\.parquet$")
    for path in FEATURE_DIR.glob("DELIVERY_PIT_FEATURES_*.parquet"):
        m = rx.match(path.name)
        if not m:
            continue
        try:
            d = pd.Timestamp(m.group(1)).date()
        except Exception:
            continue
        candidates.append((d, path))

    if not candidates:
        raise FileNotFoundError(f"No DELIVERY_PIT_FEATURES_*.parquet snapshots found in {FEATURE_DIR}")

    d, path = max(candidates, key=lambda x: x[0])
    return path, d.isoformat()


def _as_bool_gate(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "ready"}
    try:
        return bool(int(value))
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-date", default=None)
    ap.add_argument("--no-sync-drive", action="store_true")
    args = ap.parse_args()

    snapshot_path, snapshot_date = _latest_snapshot_path(args.snapshot_date)
    pit = pd.read_parquet(snapshot_path)
    if pit.empty:
        raise RuntimeError(f"PIT snapshot is empty: {snapshot_path}")
    if "symbol" not in pit.columns:
        raise RuntimeError("PIT snapshot lacks symbol column.")

    clf, reg, manifest = _load_models()
    features = list(manifest.get("features", []))
    if len(features) != 59:
        raise RuntimeError(f"Locked Delivery model exposes {len(features)} features; expected 59.")

    missing = [c for c in features if c not in pit.columns]
    if missing:
        raise RuntimeError(f"PIT snapshot is missing locked model features: {missing}")

    x = pit.copy()
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    x = x[x["symbol"].ne("")].drop_duplicates("symbol", keep="last").reset_index(drop=True)
    if x.empty:
        raise RuntimeError("No symbols remain after normalization.")

    matrix = x[features].apply(pd.to_numeric, errors="coerce")
    p_up = np.asarray(clf.predict_proba(matrix)[:, 1], dtype=float)
    expected_return = np.asarray(reg.predict(matrix), dtype=float)

    policy = _signal_policy()
    buy_p = float(policy["buy_probability_threshold"])
    sell_p = float(policy["sell_probability_threshold"])
    min_expected_return = float(policy["min_expected_return"])

    raw_gate = x["production_data_quality_gate"] if "production_data_quality_gate" in x.columns else pd.Series(False, index=x.index)
    gate = raw_gate.map(_as_bool_gate).to_numpy(dtype=bool)

    signals = np.where(
        (p_up >= buy_p) & (expected_return > min_expected_return),
        "BUY",
        np.where(
            (p_up <= sell_p) & (expected_return < -min_expected_return),
            "SELL",
            "HOLD",
        ),
    )
    signals = np.where(~gate & np.isin(signals, ["BUY", "SELL"]), "HOLD", signals)

    decision_ts = f"{snapshot_date}T15:30:00+05:30"
    model_version = str(manifest.get("engine_version", "2026-all-nse-delivery-multifactor-59-replacement-v1"))
    universe_count = int(manifest.get("universe_symbol_count", len(x)))
    trainable_count = int(manifest.get("trainable_symbol_count", len(x)))
    excluded_count = int(manifest.get("excluded_symbol_count", max(0, universe_count - trainable_count)))

    records: list[dict[str, Any]] = []
    for i, row in x.iterrows():
        symbol = str(row["symbol"])
        non_missing = int(matrix.iloc[i].notna().sum())
        records.append(
            {
                "symbol": symbol,
                "mode": "Delivery",
                "prediction_horizon": "next_session",
                "model_version": model_version,
                "engine_version": model_version,
                "decision_engine_version": model_version,
                "decision_timestamp": decision_ts,
                "reference_session": snapshot_date,
                "pit_snapshot_date": snapshot_date,
                "prediction_status": "PASS",
                "probability_up_next_session": _safe_float(p_up[i]),
                "base_model_probability_up_next_session": _safe_float(p_up[i]),
                "probability_down_next_session": _safe_float(1.0 - p_up[i]),
                "predicted_return_next_session": _safe_float(expected_return[i]),
                "signal": str(signals[i]),
                "confidence": (
                    "HIGH" if abs(p_up[i] - 0.5) >= 0.25
                    else "MEDIUM" if abs(p_up[i] - 0.5) >= 0.12
                    else "LOW"
                ),
                "feature_count": 59,
                "non_missing_feature_count": non_missing,
                "feature_non_missing_fraction": non_missing / 59.0,
                "data_readiness_score": non_missing / 59.0,
                "feature_source": "PERSISTED_PIT_SNAPSHOT",
                "production_data_quality_gate": bool(gate[i]),
                "production_data_quality_score": _safe_float(row.get("production_data_quality_score", np.nan)),
                "signal_policy": policy,
                "universe_symbol_count": universe_count,
                "trainable_symbol_count": trainable_count,
                "excluded_symbol_count": excluded_count,
                "minimum_history_sessions": int(manifest.get("minimum_history_sessions", 60)),
            }
        )

    sync_drive = not args.no_sync_drive
    result = append_predictions(records, sync_drive=sync_drive)

    signal_counts = pd.Series(signals).value_counts().to_dict()
    print("ALL-NSE DELIVERY PREDICTION CYCLE")
    print("snapshot_date:", snapshot_date)
    print("snapshot_path:", snapshot_path)
    print("symbols_scored:", len(records))
    print("production_quality_gate_true:", int(gate.sum()))
    print("signals:", signal_counts)
    print("append_result:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
