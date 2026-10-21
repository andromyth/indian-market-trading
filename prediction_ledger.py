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
import os
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

# Daily Delivery prediction ledger. The optional-path overload of load_records()
# preserves this legacy API while an explicit path selects the minute-level ledger.
LOCAL_DIR = Path(__file__).resolve().parent / "data" / "prediction_ledger"
LOCAL_LEDGER = LOCAL_DIR / "prediction_ledger.jsonl"
REMOTE_LEDGER = "production_data/predictions/prediction_ledger.jsonl"



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
    ledger_path: str | Path | None = None,
    *,
    include_out_of_session: bool = False,
    min_feature_coverage: float = 0.0,
) -> pd.DataFrame | list[dict[str, Any]]:
    """
    Load minute-level records when a path is supplied, or the shared daily
    Delivery/legacy prediction JSONL ledger when called with no arguments.
    """
    if ledger_path is None:
        return _load_delivery_prediction_records()
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
    # Hydrate first; otherwise a fresh runner could overwrite historical Drive rows.
    load_validation_records()
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



def _prediction_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic prediction content without volatile persistence metadata."""
    ignored = {
        "created_at", "created_timestamp", "prediction_id",
        "prediction_revision", "revision_fingerprint", "validated_at",
        "last_synced_at", "sync_status", "sync_error",
    }
    return {str(k): v for k, v in record.items() if str(k) not in ignored}


def prediction_revision_fingerprint(record: dict[str, Any]) -> str:
    """Hash model output and decision metadata, excluding creation-time noise."""
    payload = json.dumps(
        _prediction_payload(dict(record)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return __import__("hashlib").sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(record: dict[str, Any]) -> str:
    """Stable ID for an immutable prediction revision."""
    return prediction_revision_fingerprint(record)[:24]



def _drive_file_bytes(remote_relative_path: str) -> bytes | None:
    """Read a Drive file without conflating transport errors with a missing file.

    Returning None is reserved for an unconfigured Drive store or a path that
    does not exist. API/auth/download errors raise so a caller cannot mistake
    a transient read failure for an empty ledger and overwrite persisted history.
    """
    from persistent_store import gdrive_config

    cfg = gdrive_config()
    if not cfg.get("configured"):
        return None
    if not cfg.get("service_account_info") or not cfg.get("root_folder_id"):
        raise RuntimeError("Google Drive is marked configured but credentials/root folder are incomplete.")

    try:
        import io
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload

        credentials = service_account.Credentials.from_service_account_info(
            cfg["service_account_info"],
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        parts = remote_relative_path.strip("/").split("/")
        if not parts or any(not part for part in parts):
            raise ValueError(f"Invalid Drive-relative path: {remote_relative_path!r}")
        parent = cfg["root_folder_id"]

        for part in parts[:-1]:
            escaped = part.replace("\\", "\\\\").replace("'", "\\'")
            query = f"name = '{escaped}' and '{parent}' in parents and trashed = false"
            response = service.files().list(
                q=query,
                spaces="drive",
                fields="files(id,name,mimeType)",
                pageSize=10,
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            ).execute()
            found = response.get("files", [])
            if not found:
                return None
            parent = found[0]["id"]

        name = parts[-1]
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        query = f"name = '{escaped}' and '{parent}' in parents and trashed = false"
        response = service.files().list(
            q=query,
            spaces="drive",
            fields="files(id,name,mimeType)",
            pageSize=10,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        found = response.get("files", [])
        if not found:
            return None

        request = service.files().get_media(fileId=found[0]["id"], supportsAllDrives=True)
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read persisted Google Drive file {remote_relative_path!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

def _restore_remote_prediction_ledger() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    if LOCAL_LEDGER.exists() and LOCAL_LEDGER.stat().st_size > 0:
        return
    payload = _drive_file_bytes(REMOTE_LEDGER)
    if payload:
        LOCAL_LEDGER.write_bytes(payload)


def _load_delivery_prediction_records() -> list[dict[str, Any]]:
    """Load durable daily predictions, restoring the remote ledger when needed."""
    _restore_remote_prediction_ledger()
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    if not LOCAL_LEDGER.exists():
        return []
    rows: list[dict[str, Any]] = []
    with LOCAL_LEDGER.open("r", encoding="utf-8") as handle:
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


def append_predictions(
    records: list[dict[str, Any]],
    sync_drive: bool = True,
) -> dict[str, Any]:
    """Append idempotent Delivery prediction revisions and optionally sync the ledger."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_LEDGER.touch(exist_ok=True)
    existing = _load_delivery_prediction_records()
    existing_ids = {str(row.get("prediction_id")) for row in existing if row.get("prediction_id")}
    existing_fingerprints = {
        str(row.get("revision_fingerprint") or prediction_revision_fingerprint(row))
        for row in existing
    }

    rows_to_append: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        row = dict(record)
        row.setdefault("created_at", _utc_timestamp())
        fingerprint = prediction_revision_fingerprint(row)
        row["revision_fingerprint"] = fingerprint

        supplied_id = str(row.get("prediction_id") or "").strip()
        if fingerprint in existing_fingerprints:
            continue
        prediction_id = supplied_id
        if not prediction_id or prediction_id in existing_ids:
            prediction_id = _stable_id(row)
        if prediction_id in existing_ids:
            # Deterministic IDs must never silently overwrite a different version.
            prediction_id = __import__("hashlib").sha256(
                f"{prediction_id}|{fingerprint}".encode("utf-8")
            ).hexdigest()[:24]
        if prediction_id in existing_ids:
            continue

        row["prediction_id"] = prediction_id
        identity = (
            str(row.get("symbol", "")).strip().upper(),
            str(row.get("mode", "")).strip().lower(),
            str(row.get("prediction_horizon", "")),
            str(row.get("prediction_for_date", ""))[:10],
            str(row.get("model_version", "")),
        )
        prior_revisions = [
            max(1, int(prev.get("prediction_revision", 1) or 1))
            for prev in existing + rows_to_append
            if (
                str(prev.get("symbol", "")).strip().upper(),
                str(prev.get("mode", "")).strip().lower(),
                str(prev.get("prediction_horizon", "")),
                str(prev.get("prediction_for_date", ""))[:10],
                str(prev.get("model_version", "")),
            ) == identity
        ]
        row["prediction_revision"] = max(prior_revisions, default=0) + 1
        rows_to_append.append(row)
        existing_ids.add(prediction_id)
        existing_fingerprints.add(fingerprint)

    if rows_to_append:
        with LOCAL_LEDGER.open("a", encoding="utf-8") as handle:
            for row in rows_to_append:
                handle.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")

    synced = False
    sync_error = None
    if sync_drive:
        try:
            from persistent_store import GoogleDriveStore
            result = GoogleDriveStore().upload_or_update(
                LOCAL_LEDGER, REMOTE_LEDGER, mime_type="application/x-ndjson"
            )
            synced = result.get("status") == "OK"
            if not synced:
                sync_error = str(result)
        except Exception as exc:
            sync_error = f"{type(exc).__name__}: {exc}"

    status = "SYNCED" if synced else ("APPENDED_LOCAL_ONLY" if rows_to_append else "NO_NEW_ROWS")
    result = {
        "status": status,
        "rows_added": len(rows_to_append),
        "rows_total": len(existing) + len(rows_to_append),
        "synced": synced,
    }
    if sync_error:
        result["sync_error"] = sync_error
    return result


def append_prediction(record: dict[str, Any], sync_drive: bool = True) -> dict[str, Any]:
    """Compatibility wrapper for callers that append a single daily prediction."""
    return append_predictions([record], sync_drive=sync_drive)
