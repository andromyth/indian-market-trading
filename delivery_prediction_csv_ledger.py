"""Daywise Delivery prediction-vs-actual CSV ledger.

This CSV starts on 2026-10-06 and is a durable audit view derived from the
authoritative JSONL prediction and validation ledgers. One row represents one
NSE symbol's next-session Delivery prediction. Actual OHLC is filled only after
the next official NSE EOD session is available.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from persistent_store import GoogleDriveStore
from prediction_ledger import load_records, load_validation_records

DEFAULT_BASE = "/content/drive/MyDrive/Indian_Market_Research"
BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", DEFAULT_BASE))
EOD_LOCAL_ROOT = BASE / "production_data" / "nse_official" / "bulk_eod"

LEDGER_START_DATE = date(2026, 10, 6)
LOCAL_CSV = BASE / "data" / "prediction_ledger" / "delivery_prediction_actual_ledger.csv"
REMOTE_CSV = "production_data/predictions/delivery_prediction_actual_ledger.csv"


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def load_eod_history() -> pd.DataFrame:
    """Load the local NSE EOD archive once for batch prediction callers.

    Batch callers should load this once per process and reuse the result,
    rather than rescanning/re-reading every daily Parquet file for each symbol.
    """
    return _load_eod()


def _load_eod() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    rx = re.compile(r"BhavCopy_NSE_CM_0_0_0_(\d{8})_F_0000\.parquet$", re.I)

    if not EOD_LOCAL_ROOT.exists():
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close"])

    for path in EOD_LOCAL_ROOT.rglob("BhavCopy_NSE_CM_0_0_0_*_F_0000.parquet"):
        if not rx.match(path.name):
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

        rename = {}
        for target, names in {
            "symbol": ("symbol", "tckrsymb", "ticker"),
            "date": ("date", "trade_date", "trad_dt"),
            "open": ("open", "open_price", "opnpric"),
            "high": ("high", "high_price", "hghpric"),
            "low": ("low", "low_price", "lwpric"),
            "close": ("close", "close_price", "clspric"),
        }.items():
            c = pick(names)
            if c is not None:
                rename[c] = target

        x = x.rename(columns=rename)
        required = {"symbol", "date", "open", "high", "low", "close"}
        if not required.issubset(x.columns):
            continue

        x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
        x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
        for col in ("open", "high", "low", "close"):
            x[col] = pd.to_numeric(x[col], errors="coerce")

        x = x.dropna(subset=list(required))
        if not x.empty:
            parts.append(x[list(required)])

    if not parts:
        return pd.DataFrame(columns=["symbol", "date", "open", "high", "low", "close"])

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )


def _project_ohlc(symbol: str, reference_date: str, reference_close: Any,
                  predicted_close: Any, eod: pd.DataFrame) -> dict[str, Any]:
    try:
        ref_close = _safe_float(reference_close)
        pred_close = _safe_float(predicted_close)
        if not np.isfinite(ref_close):
            return {"status": "NO_DATA"}

        root = str(symbol).upper().replace(".NS", "").replace(".BO", "")
        x = eod[eod["symbol"].eq(root)].copy()
        if x.empty:
            return {"status": "NO_DATA"}

        ref_date = pd.Timestamp(reference_date).normalize()
        recent = x.loc[x["date"] < ref_date].tail(60).copy()
        if len(recent) < 20:
            return {"status": "INSUFFICIENT_HISTORY"}

        recent["prev_close"] = recent["close"].shift(1)
        recent["gap"] = recent["open"] / recent["prev_close"] - 1.0
        gaps = recent["gap"].dropna()
        if gaps.empty:
            return {"status": "NO_DATA"}

        median_gap = _safe_float(gaps.median())
        gap_std = _safe_float(gaps.std(ddof=1)) if len(gaps) >= 2 else np.nan
        expected_open = ref_close * (1.0 + median_gap)

        upper_base = recent[["open", "close"]].max(axis=1)
        lower_base = recent[["open", "close"]].min(axis=1)
        up_excursion = (
            (recent["high"] / upper_base - 1.0)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        down_excursion = (
            (1.0 - recent["low"] / lower_base)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        median_up = _safe_float(up_excursion.median()) if not up_excursion.empty else np.nan
        median_down = _safe_float(down_excursion.median()) if not down_excursion.empty else np.nan

        anchor_high = max(expected_open, pred_close) if np.isfinite(pred_close) else expected_open
        anchor_low = min(expected_open, pred_close) if np.isfinite(pred_close) else expected_open

        expected_high = (
            anchor_high * (1.0 + median_up) if np.isfinite(median_up) else np.nan
        )
        expected_low = (
            anchor_low * (1.0 - median_down) if np.isfinite(median_down) else np.nan
        )

        if np.isfinite(expected_high):
            expected_high = max(expected_high, expected_open, pred_close if np.isfinite(pred_close) else expected_open)
        if np.isfinite(expected_low):
            expected_low = min(expected_low, expected_open, pred_close if np.isfinite(pred_close) else expected_open)

        return {
            "status": "PASS",
            "predicted_open": expected_open,
            "predicted_high": expected_high,
            "predicted_low": expected_low,
            "predicted_close": pred_close,
            "predicted_open_low": (
                ref_close * (1.0 + median_gap - gap_std)
                if np.isfinite(gap_std) else np.nan
            ),
            "predicted_open_high": (
                ref_close * (1.0 + median_gap + gap_std)
                if np.isfinite(gap_std) else np.nan
            ),
            "ohlc_prediction_method": "60-session median gap + empirical daily high/low excursion around model close",
        }
    except Exception as exc:
        return {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}



def project_next_session_ohlc(
    symbol: str,
    reference_date: str,
    reference_close: Any,
    predicted_close: Any,
    eod_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Public wrapper used by the Delivery prediction cycle and CSV ledger.

    It uses the same statistical OHLC methodology as the Streamlit Delivery
    display, based only on completed sessions before the reference date.
    """
    history = _load_eod() if eod_history is None else eod_history
    return _project_ohlc(
        symbol,
        reference_date,
        reference_close,
        predicted_close,
        history,
    )


def _error_pct(actual: Any, predicted: Any) -> float:
    a = _safe_float(actual)
    p = _safe_float(predicted)
    if not np.isfinite(a) or not np.isfinite(p) or p == 0:
        return np.nan
    return a / p - 1.0


def _latest_revision_per_symbol_date(frame: pd.DataFrame) -> pd.DataFrame:
    """Choose the most recently created prediction revision per target date/symbol."""
    if frame.empty or not {"prediction_date", "symbol"}.issubset(frame.columns):
        return frame.copy()

    out = frame.copy()
    if "created_at" not in out.columns:
        out["created_at"] = ""
    out["_created_at_order"] = pd.to_datetime(
        out["created_at"], errors="coerce", utc=True
    )
    # Keep historical revisions in the authoritative JSONL ledger; this compact
    # CSV view surfaces only the newest revision for each symbol and target date.
    out = (
        out.sort_values(
            ["prediction_date", "symbol", "_created_at_order", "prediction_id"],
            kind="mergesort",
            na_position="first",
        )
        .drop_duplicates(["prediction_date", "symbol"], keep="last")
        .drop(columns=["_created_at_order"])
        .sort_values(["prediction_date", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def build_delivery_csv_ledger(eod_history: pd.DataFrame | None = None) -> pd.DataFrame:
    predictions = pd.DataFrame(load_records())
    validations = pd.DataFrame(load_validation_records())

    if predictions.empty or "mode" not in predictions.columns:
        return pd.DataFrame()

    predictions = predictions[
        predictions["mode"].astype(str).str.lower().eq("delivery")
    ].copy()
    if predictions.empty or "prediction_id" not in predictions.columns:
        return pd.DataFrame()

    predictions["prediction_id"] = predictions["prediction_id"].astype(str)
    predictions["prediction_for_date"] = pd.to_datetime(
        predictions.get("prediction_for_date", pd.Series(index=predictions.index)),
        errors="coerce",
    ).dt.date

    # Backward-compatible derivation for records created before prediction_for_date
    # existed. Those pre-start rows are filtered out by LEDGER_START_DATE anyway.
    missing_target = predictions["prediction_for_date"].isna()
    if missing_target.any():
        decisions = pd.to_datetime(
            predictions.loc[missing_target, "decision_timestamp"],
            errors="coerce",
            utc=True,
        )
        predictions.loc[missing_target, "prediction_for_date"] = (
            decisions.dt.tz_convert("Asia/Kolkata").dt.date
        )

    predictions["prediction_for_date"] = pd.to_datetime(
        predictions["prediction_for_date"], errors="coerce"
    ).dt.date
    predictions = predictions[
        predictions["prediction_for_date"].notna()
        & (predictions["prediction_for_date"] >= LEDGER_START_DATE)
    ].copy()

    if not validations.empty and "prediction_id" in validations.columns:
        validations["prediction_id"] = validations["prediction_id"].astype(str)
        validations = validations.drop_duplicates("prediction_id", keep="last").copy()
    else:
        validations = pd.DataFrame(columns=["prediction_id"])

    eod = _load_eod() if eod_history is None else eod_history
    rows: list[dict[str, Any]] = []

    for _, p in predictions.iterrows():
        pid = str(p.get("prediction_id", ""))
        match = (
            validations["prediction_id"].eq(pid)
            if not validations.empty and "prediction_id" in validations.columns
            else pd.Series(dtype=bool)
        )
        v = validations.loc[match].iloc[-1].to_dict() if bool(match.any()) else {}

        prediction_date = str(p["prediction_for_date"])
        reference_date = str(p.get("reference_session", ""))

        predicted_close = _safe_float(p.get("predicted_price_next_session"))
        projected_open = _safe_float(p.get("predicted_open"))
        projected_high = _safe_float(p.get("predicted_high"))
        projected_low = _safe_float(p.get("predicted_low"))
        method = str(p.get("ohlc_prediction_method", ""))

        if not all(np.isfinite(x) for x in (projected_open, projected_high, projected_low)):
            ohlc = _project_ohlc(
                str(p.get("symbol", "")),
                reference_date,
                p.get("reference_close"),
                predicted_close,
                eod,
            )
            if ohlc.get("status") == "PASS":
                projected_open = _safe_float(ohlc.get("predicted_open"))
                projected_high = _safe_float(ohlc.get("predicted_high"))
                projected_low = _safe_float(ohlc.get("predicted_low"))
                method = str(ohlc.get("ohlc_prediction_method", method))

        actual_date = str(v.get("actual_d1_date", "") or "")
        actual_open = _safe_float(v.get("actual_d1_open"))
        actual_high = _safe_float(v.get("actual_d1_high"))
        actual_low = _safe_float(v.get("actual_d1_low"))
        actual_close = _safe_float(v.get("actual_d1_close"))

        rows.append({
            "prediction_date": prediction_date,
            "actual_date": actual_date,
            "symbol": str(p.get("symbol", "")).upper(),
            "signal": str(p.get("signal", "")),
            "raw_model_signal": str(p.get("raw_model_signal", "")),
            "signal_reason": str(p.get("signal_reason", "")),
            "prediction_revision": str(p.get("prediction_revision", "")),
            "created_at": str(p.get("created_at", "")),
            "probability_up": _safe_float(p.get("probability_up_next_session")),
            "expected_return": _safe_float(p.get("predicted_return_next_session")),
            "reference_open": _safe_float(p.get("reference_open")),
            "reference_high": _safe_float(p.get("reference_high")),
            "reference_low": _safe_float(p.get("reference_low")),
            "reference_close": _safe_float(p.get("reference_close")),
            "predicted_open": projected_open,
            "predicted_high": projected_high,
            "predicted_low": projected_low,
            "predicted_close": predicted_close,
            "actual_open": actual_open,
            "actual_high": actual_high,
            "actual_low": actual_low,
            "actual_close": actual_close,
            "open_error": actual_open - projected_open if np.isfinite(actual_open) and np.isfinite(projected_open) else np.nan,
            "high_error": actual_high - projected_high if np.isfinite(actual_high) and np.isfinite(projected_high) else np.nan,
            "low_error": actual_low - projected_low if np.isfinite(actual_low) and np.isfinite(projected_low) else np.nan,
            "close_error": actual_close - predicted_close if np.isfinite(actual_close) and np.isfinite(predicted_close) else np.nan,
            "open_error_pct": _error_pct(actual_open, projected_open),
            "high_error_pct": _error_pct(actual_high, projected_high),
            "low_error_pct": _error_pct(actual_low, projected_low),
            "close_error_pct": _error_pct(actual_close, predicted_close),
            "actual_direction": v.get("actual_direction_d1", ""),
            "direction_match": v.get("direction_match_d1"),
            "validation_status": v.get("validation_status", "PENDING"),
            "data_gate": p.get("production_data_quality_gate"),
            "model_version": p.get("model_version", ""),
            "decision_engine_version": p.get("decision_engine_version", ""),
            "feature_count": p.get("feature_count", 59),
            "prediction_id": pid,
            "ohlc_prediction_method": method,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return _latest_revision_per_symbol_date(df)


def sync_delivery_csv_ledger(eod_history: pd.DataFrame | None = None) -> dict[str, Any]:
    df = build_delivery_csv_ledger(eod_history=eod_history)
    LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOCAL_CSV, index=False)

    store = GoogleDriveStore()
    result = store.upload_or_update(
        LOCAL_CSV,
        REMOTE_CSV,
        mime_type="text/csv",
    )

    return {
        "status": "SYNCED" if result.get("status") == "OK" else "ERROR",
        "rows": int(len(df)),
        "dates": int(df["prediction_date"].nunique()) if not df.empty else 0,
        "symbols": int(df["symbol"].nunique()) if not df.empty else 0,
        "remote": REMOTE_CSV,
        "sync_result": result,
    }


if __name__ == "__main__":
    print(json.dumps(sync_delivery_csv_ledger(), indent=2, default=str))
