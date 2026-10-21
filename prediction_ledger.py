"""
prediction_ledger.py
Unified minute-level prediction ledger and live microstructure feature capture
(volume z-score, VWAP distance, order-flow imbalance, volatility ratios, and event flags)
with strict Asia/Kolkata session boundary guards and zero-edit policy hot-reloading.
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import generated_alpha_policy
except ImportError:
    generated_alpha_policy = None

IST_TZ = "Asia/Kolkata"
SCHEMA_VERSION = "2.2"
MARKET_OPEN_TIME = time(9, 15, 0)
MARKET_LAST_VALID_TARGET_TIME = time(15, 29, 0)
MIN_FEATURE_COVERAGE = 0.70


def normalize_ist_timestamp(ts: Any) -> pd.Timestamp:
    """Parse and normalize any timestamp input to timezone-aware Asia/Kolkata."""
    parsed = pd.Timestamp(ts)
    if parsed.tzinfo is None:
        return parsed.tz_localize(IST_TZ)
    return parsed.tz_convert(IST_TZ)


def compute_target_timestamp(decision_ts: pd.Timestamp, horizon_min: int) -> pd.Timestamp:
    """Compute the realized evaluation timestamp for a given forecast horizon."""
    norm_ts = normalize_ist_timestamp(decision_ts)
    return norm_ts + pd.Timedelta(minutes=int(horizon_min))


def is_within_nse_session(decision_ts: pd.Timestamp, horizon_min: int) -> bool:
    """
    Verify that both the decision timestamp and target horizon timestamp fall
    within active NSE regular trading hours (09:15:00 to 15:29:00 IST) on the same date.
    """
    norm_decision = normalize_ist_timestamp(decision_ts)
    target_ts = compute_target_timestamp(norm_decision, horizon_min)

    if norm_decision.date() != target_ts.date():
        return False
    if norm_decision.weekday() >= 5:
        return False
    if not (MARKET_OPEN_TIME <= norm_decision.time() <= MARKET_LAST_VALID_TARGET_TIME):
        return False
    if not (MARKET_OPEN_TIME <= target_ts.time() <= MARKET_LAST_VALID_TARGET_TIME):
        return False
    return True


def extract_live_microstructure_features(bars_1m: pd.DataFrame | None) -> dict[str, Any]:
    """
    Extract real-time price, volume, and event features from trailing 1-minute NSE
    OHLCV bars to feed both inference and state capture.
    """
    if bars_1m is None or len(bars_1m) < 5:
        return {
            "volume_zscore_20m": 0.0,
            "vwap_distance_bps": 0.0,
            "price_momentum_5m_bps": 0.0,
            "order_flow_imbalance": 0.0,
            "volatility_ratio_15m": 1.0,
            "candle_body_to_wick": 0.5,
            "event_flags": [],
        }

    ts_col = "timestamp" if "timestamp" in bars_1m.columns else "datetime"
    df = bars_1m.sort_values(ts_col).copy() if ts_col in bars_1m.columns else bars_1m.copy()

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float) if "high" in df.columns else close
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float) if "low" in df.columns else close
    open_p = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float) if "open" in df.columns else close
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=float) if "volume" in df.columns else np.ones_like(close)

    curr_close = float(close[-1])
    vol_window = vol[-20:] if len(vol) >= 20 else vol
    vol_mean = float(np.mean(vol_window))
    vol_std = float(np.std(vol_window))
    vol_zscore = (float(vol[-1]) - vol_mean) / max(vol_std, 1.0)

    typical = (high + low + close) / 3.0
    cum_vwap = float(np.sum(typical * vol) / max(np.sum(vol), 1.0))
    vwap_dist_bps = ((curr_close - cum_vwap) / max(cum_vwap, 1e-4)) * 10000.0

    ref_5m = float(close[-5])
    mom_5m_bps = ((curr_close - ref_5m) / max(ref_5m, 1e-4)) * 10000.0

    bar_range = np.maximum(high - low, 1e-4)
    signed_body = (close - open_p) / bar_range
    ofi = float(np.sum(signed_body[-5:] * vol[-5:]) / max(np.sum(vol[-5:]), 1.0))
    body_to_wick = float(abs(close[-1] - open_p[-1]) / bar_range[-1])

    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    atr_short = float(np.mean(tr[-5:])) if len(tr) >= 5 else float(np.mean(tr))
    atr_long = float(np.mean(tr[-20:])) if len(tr) >= 20 else float(np.mean(tr))
    vol_ratio = atr_short / max(atr_long, 1e-4)

    events: list[str] = []
    if vol_zscore >= 2.0 and body_to_wick >= 0.65:
        events.append("INSTITUTIONAL_VOL_IGNITION")
    if vol_zscore >= 2.0 and body_to_wick <= 0.25:
        events.append("HIGH_VOL_ABSORPTION_PINBAR")
    if vol_ratio <= 0.65:
        events.append("VOLATILITY_COIL_COMPRESSION")
    if abs(vwap_dist_bps) >= 60.0 and np.sign(mom_5m_bps) != np.sign(vwap_dist_bps):
        events.append("VWAP_MEAN_REVERSION_SNAP")

    return {
        "volume_zscore_20m": round(vol_zscore, 4),
        "vwap_distance_bps": round(vwap_dist_bps, 2),
        "price_momentum_5m_bps": round(mom_5m_bps, 2),
        "order_flow_imbalance": round(ofi, 4),
        "volatility_ratio_15m": round(vol_ratio, 4),
        "candle_body_to_wick": round(body_to_wick, 4),
        "event_flags": events,
    }


@dataclass
class PredictionRecord:
    symbol: str
    decision_timestamp: str
    horizon_min: int
    signal: str
    probability_up: float
    reference_close: float
    predicted_price: float
    price_forecast_status: str = "OK"
    feature_coverage: float = 1.0
    calibration_profile_id: str = "RAW"
    atr_15m: float | None = None
    target_timestamp: str | None = None
    volume_zscore_20m: float = 0.0
    vwap_distance_bps: float = 0.0
    price_momentum_5m_bps: float = 0.0
    order_flow_imbalance: float = 0.0
    volatility_ratio_15m: float = 1.0
    candle_body_to_wick: float = 0.5
    event_flags: list[str] = field(default_factory=list)
    expert_votes: dict[str, float] = field(default_factory=dict)
    active_policy_version: str = "v1"
    schema_version: str = field(default=SCHEMA_VERSION)

    def __post_init__(self) -> None:
        self.symbol = str(self.symbol).strip().upper()
        self.horizon_min = int(self.horizon_min)

        norm_decision = normalize_ist_timestamp(self.decision_timestamp)
        target_ts = compute_target_timestamp(norm_decision, self.horizon_min)
        self.decision_timestamp = norm_decision.isoformat()
        self.target_timestamp = target_ts.isoformat()

        if self.reference_close is None or not math.isfinite(float(self.reference_close)) or float(self.reference_close) <= 0.0:
            raise ValueError(
                f"Invalid reference_close={self.reference_close} for {self.symbol} at {self.decision_timestamp}. "
                "reference_close must be a finite positive float."
            )
        self.reference_close = float(self.reference_close)
        self.predicted_price = float(self.predicted_price)
        self.probability_up = float(self.probability_up)
        self.feature_coverage = float(self.feature_coverage)

        if self.feature_coverage < MIN_FEATURE_COVERAGE:
            self.price_forecast_status = "LOW_FEATURE_COVERAGE"
        elif not is_within_nse_session(norm_decision, self.horizon_min):
            self.price_forecast_status = "OUT_OF_SESSION_HORIZON"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_record(
    ledger_path: str | Path,
    record: PredictionRecord | dict[str, Any],
    *,
    bars_1m: pd.DataFrame | None = None,
    auto_apply_intelligence: bool = True,
) -> PredictionRecord:
    """
    Zero-edit hook: automatically extracts live 1m microstructure features (if bars_1m is passed),
    hot-reloads generated_alpha_policy.py, upgrades the raw forecast, and appends to the ledger.
    """
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_dict = record.to_dict() if isinstance(record, PredictionRecord) else dict(record)

    if bars_1m is not None and not bars_1m.empty:
        micro_state = extract_live_microstructure_features(bars_1m)
        raw_dict.update(micro_state)
    else:
        micro_state = {
            "volume_zscore_20m": float(raw_dict.get("volume_zscore_20m", 0.0)),
            "vwap_distance_bps": float(raw_dict.get("vwap_distance_bps", 0.0)),
            "price_momentum_5m_bps": float(raw_dict.get("price_momentum_5m_bps", 0.0)),
            "order_flow_imbalance": float(raw_dict.get("order_flow_imbalance", 0.0)),
            "volatility_ratio_15m": float(raw_dict.get("volatility_ratio_15m", 1.0)),
            "candle_body_to_wick": float(raw_dict.get("candle_body_to_wick", 0.5)),
            "event_flags": list(raw_dict.get("event_flags", [])),
        }

    if (
        auto_apply_intelligence
        and generated_alpha_policy is not None
        and float(raw_dict.get("reference_close", 0.0)) > 0
    ):
        try:
            importlib.reload(generated_alpha_policy)
            evolved = generated_alpha_policy.apply_evolved_intelligence(
                reference_close=float(raw_dict["reference_close"]),
                raw_predicted_price=float(raw_dict["predicted_price"]),
                raw_probability_up=float(raw_dict.get("probability_up", 0.5)),
                horizon_min=int(raw_dict.get("horizon_min", 5)),
                state=micro_state,
            )
            raw_dict["predicted_price"] = evolved["predicted_price"]
            raw_dict["probability_up"] = evolved["probability_up"]
            raw_dict["expert_votes"] = evolved["expert_votes"]
            raw_dict["active_policy_version"] = evolved["active_policy_version"]
            raw_dict["calibration_profile_id"] = evolved["active_policy_version"]
        except Exception:
            pass

    validated = PredictionRecord(**raw_dict)
    row_dict = validated.to_dict()

    if path.suffix.lower() == ".jsonl":
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row_dict) + "\n")
    else:
        csv_row = dict(row_dict)
        csv_row["event_flags"] = json.dumps(csv_row.get("event_flags", []))
        csv_row["expert_votes"] = json.dumps(csv_row.get("expert_votes", {}))
        df_row = pd.DataFrame([csv_row])
        write_header = not path.exists() or path.stat().st_size == 0
        df_row.to_csv(path, mode="a", header=write_header, index=False)

    return validated


def load_records(
    ledger_path: str | Path,
    *,
    include_out_of_session: bool = False,
    min_feature_coverage: float = 0.0,
) -> pd.DataFrame:
    """
    Load historical prediction records, normalize timestamps to Asia/Kolkata,
    restore microstructure state columns, and filter out-of-session horizons.
    """
    path = Path(ledger_path)
    if not path.exists():
        return pd.DataFrame()

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        df = pd.read_csv(path)
        for json_col, default_val in [("event_flags", "[]"), ("expert_votes", "{}")]:
            if json_col in df.columns:
                df[json_col] = df[json_col].fillna(default_val).apply(
                    lambda x: json.loads(x) if isinstance(x, str) else x
                )

    if df.empty:
        return df

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["horizon_min"] = pd.to_numeric(df["horizon_min"], errors="coerce").fillna(5).astype(int)
    df["decision_timestamp"] = pd.to_datetime(df["decision_timestamp"], utc=True).dt.tz_convert(IST_TZ)

    if "target_timestamp" not in df.columns or df["target_timestamp"].isna().any():
        df["target_timestamp"] = df["decision_timestamp"] + pd.to_timedelta(df["horizon_min"], unit="m")
    else:
        df["target_timestamp"] = pd.to_datetime(df["target_timestamp"], utc=True).dt.tz_convert(IST_TZ)

    if "reference_close" not in df.columns:
        df["reference_close"] = float("nan")
    else:
        df["reference_close"] = pd.to_numeric(df["reference_close"], errors="coerce")

    if "price_forecast_status" not in df.columns:
        df["price_forecast_status"] = "OK"

    defaults = {
        "volume_zscore_20m": 0.0,
        "vwap_distance_bps": 0.0,
        "price_momentum_5m_bps": 0.0,
        "order_flow_imbalance": 0.0,
        "volatility_ratio_15m": 1.0,
        "candle_body_to_wick": 0.5,
    }
    for col, d_val in defaults.items():
        if col not in df.columns:
            df[col] = d_val
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(d_val)

    in_session_mask = [
        is_within_nse_session(dec_ts, h)
        for dec_ts, h in zip(df["decision_timestamp"], df["horizon_min"])
    ]
    df.loc[~pd.Series(in_session_mask, index=df.index), "price_forecast_status"] = "OUT_OF_SESSION_HORIZON"

    if not include_out_of_session:
        df = df[df["price_forecast_status"] != "OUT_OF_SESSION_HORIZON"].copy()

    if "feature_coverage" in df.columns and min_feature_coverage > 0.0:
        df["feature_coverage"] = pd.to_numeric(df["feature_coverage"], errors="coerce").fillna(1.0)
        df = df[df["feature_coverage"] >= min_feature_coverage].copy()

    return df.sort_values(["decision_timestamp", "symbol", "horizon_min"]).reset_index(drop=True)


def load_validation_records() -> list[dict[str, Any]]:
    """Load persisted Delivery validation rows, restoring the Drive copy when needed.

    This compatibility API is shared by Delivery diagnostics, policy updates,
    and the CSV prediction-vs-actual view. It deliberately remains separate
    from the minute-level prediction-record loader above.
    """
    local_validation = Path(__file__).resolve().parent / "data" / "prediction_ledger" / "prediction_validation.jsonl"
    remote_validation = "production_data/predictions/prediction_validation.jsonl"

    if not local_validation.exists() or local_validation.stat().st_size == 0:
        payload = _drive_file_bytes(remote_validation)
        if payload:
            local_validation.parent.mkdir(parents=True, exist_ok=True)
            local_validation.write_bytes(payload)

    if not local_validation.exists():
        return []

    rows: list[dict[str, Any]] = []
    with local_validation.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def save_validation_results(results: list[dict[str, Any]], sync_drive: bool = True) -> dict[str, Any]:
    """Persist the latest validation result per prediction ID, optionally syncing Drive."""
    local_validation = Path(__file__).resolve().parent / "data" / "prediction_ledger" / "prediction_validation.jsonl"
    remote_validation = "production_data/predictions/prediction_validation.jsonl"
    local_validation.parent.mkdir(parents=True, exist_ok=True)

    latest: dict[str, dict[str, Any]] = {}
    if local_validation.exists():
        with local_validation.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("prediction_id"):
                    latest[str(row["prediction_id"])] = row

    for result in results:
        if not isinstance(result, dict):
            continue
        prediction_id = result.get("prediction_id")
        if not prediction_id:
            continue
        row = dict(result)
        row["validated_at"] = _utc_timestamp()
        latest[str(prediction_id)] = row

    tmp = local_validation.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in latest.values():
            handle.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
    os.replace(tmp, local_validation)

    synced = False
    sync_error = None
    if sync_drive:
        try:
            from persistent_store import GoogleDriveStore
            result = GoogleDriveStore().upload_or_update(
                local_validation, remote_validation, mime_type="application/x-ndjson"
            )
            synced = result.get("status") == "OK"
            if not synced:
                sync_error = str(result)
        except Exception as exc:
            sync_error = f"{type(exc).__name__}: {exc}"

    result = {"status": "SAVED", "rows": len(latest), "synced": synced}
    if sync_error:
        result["sync_error"] = sync_error
    return result


def _utc_timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
