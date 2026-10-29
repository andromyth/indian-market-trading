"""All-NSE five-minute snapshot prediction plus contemporaneous actual-price persistence.

The full-universe collector persists one quote snapshot per NSE symbol every five
minutes. This module builds a shared 30-minute forecast from the prior snapshot
archive plus the same-slot snapshot that triggered inference, then persists one
auditable row per symbol. Live prediction does not call the one-minute candle
history endpoint. Sampled LTP is never represented as exchange OHLC; low-readiness
predictions may be recorded for analysis but the actionable signal remains HOLD.
"""
from __future__ import annotations

import io
import json
import threading
import time
from datetime import date, datetime, time as wall_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import os

from snapshot_5m_features import READINESS_FEATURES, SNAPSHOT_CADENCE_MINUTES

import calibration_overlay as calibration_overlay_module

IST = ZoneInfo("Asia/Kolkata")
REMOTE_ROOT = "production_data/all_nse_minute_predictions"
FNO_DAILY_REMOTE = "production_data/nse_official/fno/fno_daily_features_2026.parquet"
FNO_CONTEXT_MAX_AGE_DAYS = 7
FNO_CONTEXT_RETRY_SECONDS = 900
HISTORY_LOOKBACK_BARS = 120
HISTORY_BATCH_SIZE = 5
HISTORY_REQUEST_PAUSE_SECONDS = 0.22
HISTORY_REFRESH_REQUEST_SPACING_SECONDS = 0.24
HISTORY_REFRESH_WINDOW_MINUTES = 90
HISTORY_REFRESH_WORKERS = 8
MIN_SESSION_CANDLES_FOR_PREDICTION = 3
MAX_HISTORY_CANDLE_AGE_MINUTES = 10
MIN_SIGNAL_READINESS = 0.70
MAX_QUOTE_AGE_SECONDS = 120
MIN_PREDICTION_COVERAGE = 0.95
MAX_MISSING_QUOTE_FRACTION = 0.02
MAX_DATA_LIMITED_FRACTION = 0.10
MAX_UNTRAINED_SYMBOL_FRACTION = 0.10
MAX_MODEL_FAILURE_FRACTION = 0.01
MIN_HISTORY_SEED_COVERAGE = 0.95
MAX_HISTORY_FAILURE_FRACTION = 0.05
SNAPSHOT_HISTORY_LOOKBACK_BARS = 50


def _history_seed_batches(
    stock_entries: list[tuple[str, str]],
    nifty_code: str | None,
    batch_size: int = HISTORY_BATCH_SIZE,
) -> list[list[tuple[str, str]]]:
    """Batch equity history together, but keep the NIFTY index token separate.

    INDstocks can reject a mixed equity+index history request with HTTP 400.
    The index is context only and must never consume or invalidate an equity
    batch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batches = [
        stock_entries[offset:offset + batch_size]
        for offset in range(0, len(stock_entries), batch_size)
    ]
    if nifty_code:
        # Fetch benchmark context first so early progressive stock forecasts
        # have the same market-context input path as later rows.
        batches.insert(0, [("__NIFTY50__", nifty_code)])
    return batches


def _history_seed_meets_quality_gate(
    coverage_symbols: int,
    universe_symbols: int,
    failed_batches: int,
    batch_size: int = HISTORY_BATCH_SIZE,
) -> bool:
    """A seed is complete only when both history coverage and batch reliability pass."""
    universe_count = max(0, int(universe_symbols))
    if universe_count == 0:
        return False
    coverage_count = max(0, int(coverage_symbols))
    failures = max(0, int(failed_batches))
    requested_batches = max(1, int(np.ceil(universe_count / max(1, int(batch_size)))))
    coverage_target = int(np.ceil(universe_count * MIN_HISTORY_SEED_COVERAGE))
    return (
        coverage_count >= coverage_target
        and failures / requested_batches <= MAX_HISTORY_FAILURE_FRACTION
    )


def _history_seed_universe(
    official_symbols: list[str] | None,
    instrument_codes: dict[str, str],
) -> tuple[list[str], str]:
    """Resolve the history-seed roster, with a provider-master fallback.

    The live capture can continue using the official NSE roster, but the
    background feature warm-up must not be permanently blocked when NSE's
    EQUITY_L.csv endpoint returns 403. Its fallback is limited to NSE equity
    symbols already mapped by the validated INDstocks instrument master.
    """
    official = sorted({
        str(symbol).strip().upper().replace(".NS", "")
        for symbol in (official_symbols or [])
        if str(symbol).strip()
    })
    if official:
        return official, "NSE_OFFICIAL_EQUITY_L"

    fallback = sorted({
        str(symbol).strip().upper().replace(".NS", "")
        for symbol in instrument_codes
        if str(symbol).strip()
    })
    if fallback:
        return fallback, "INDSTOCKS_NSE_INSTRUMENT_MASTER_FALLBACK"
    raise RuntimeError(
        "Neither the official NSE equity roster nor the INDstocks NSE instrument master yielded symbols"
    )


def _latest_completed_fno_context(
    frame: pd.DataFrame | None,
    decision_date: date,
    *,
    max_age_days: int = FNO_CONTEXT_MAX_AGE_DAYS,
) -> dict[str, dict[str, Any]]:
    """Select latest completed-session F&O data per symbol, never same-day EOD data.

    The persisted NSE F&O bhavcopy is a daily close/settlement source, not a
    minute feed. For an intraday decision on D, only rows with date < D are
    eligible. Stale rows are omitted instead of being carried forward forever.
    """
    if frame is None or frame.empty:
        return {}
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if not {"symbol", "date"}.issubset(x.columns):
        return {}
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper().str.replace(
        r"\.NS$", "", regex=True
    )
    parsed = pd.to_datetime(x["date"], errors="coerce", utc=True)
    x["_fno_date"] = parsed.dt.tz_convert(IST).dt.date
    x = x.loc[
        x["symbol"].ne("")
        & x["_fno_date"].notna()
        & (x["_fno_date"] < decision_date)
        & x["_fno_date"].map(
            lambda value: (decision_date - value).days <= max_age_days
        )
    ].copy()
    if x.empty:
        return {}
    x = x.sort_values(["symbol", "_fno_date"]).drop_duplicates(
        "symbol", keep="last"
    )

    def first_value(row: pd.Series, *names: str) -> float | None:
        for name in names:
            if name in row.index:
                value = _safe_float(row.get(name))
                if value is not None:
                    return value
        return None

    out: dict[str, dict[str, Any]] = {}
    for _, row in x.iterrows():
        out[str(row["symbol"])] = {
            "pcr": first_value(row, "fno_pcr", "pcr"),
            "oi_change": first_value(row, "fno_oi_change", "oi_change"),
            "fut_return": first_value(row, "fno_fut_return", "fut_return"),
            "fut_oi": first_value(row, "fno_fut_oi", "fut_oi"),
            "basis": first_value(row, "fno_fut_basis", "fno_basis", "basis"),
            "source_date": row["_fno_date"].isoformat(),
        }
    return out


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(str(value).replace(",", "").strip())
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _as_ist_timestamp(value: Any, fallback: pd.Timestamp | None = None) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        ts = fallback or pd.Timestamp.now(tz=IST)
    ts = pd.Timestamp(ts)
    return ts.tz_localize(IST) if ts.tzinfo is None else ts.tz_convert(IST)


def _parse_ist_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        stamp = pd.Timestamp(parsed)
        return stamp.tz_localize(IST) if stamp.tzinfo is None else stamp.tz_convert(IST)
    except (TypeError, ValueError, OverflowError):
        return None


def _current_minute_row(
    *,
    symbol: str,
    snapshot_minute: pd.Timestamp,
    quote: dict[str, Any] | None,
    prior_history: pd.DataFrame,
    previous_day_volume: float | None,
    previous_volume_date: date | None,
    snapshot_date: date,
) -> tuple[dict[str, Any], dict[str, Any] | None, float | None]:
    """Normalize one quote and create a partial current-minute model input row."""
    quote = quote or {}
    actual_price = _safe_float(quote.get("actual_observed_price"))
    quote_status = str(quote.get("quote_status") or "MISSING_QUOTE").upper()
    observed_at = _as_ist_timestamp(
        quote.get("quote_observed_at_ist") or quote.get("quote_observed_at_utc"),
        fallback=snapshot_minute,
    )
    raw_day_volume = _safe_float(quote.get("day_volume"))
    quote_ok = actual_price is not None and actual_price > 0 and quote_status == "PASS"

    if previous_volume_date == snapshot_date and previous_day_volume is not None:
        volume_delta = raw_day_volume - previous_day_volume if raw_day_volume is not None else None
    else:
        same_day_history = prior_history.loc[
            pd.Index(prior_history.index.date) == snapshot_date
        ] if not prior_history.empty else pd.DataFrame()
        if not same_day_history.empty and "day_volume" in same_day_history:
            latest_history_ts = _as_ist_timestamp(same_day_history.index.max())
            if latest_history_ts == snapshot_minute - pd.Timedelta(minutes=SNAPSHOT_CADENCE_MINUTES):
                history_volume = _safe_float(same_day_history.iloc[-1].get("day_volume"))
            else:
                history_volume = np.nan
        else:
            history_volume = (
                pd.to_numeric(same_day_history.get("Volume"), errors="coerce").sum(min_count=1)
                if not same_day_history.empty and "Volume" in same_day_history else np.nan
            )
        volume_delta = (
            raw_day_volume - float(history_volume)
            if raw_day_volume is not None and pd.notna(history_volume)
            else None
        )
    if volume_delta is not None and (not np.isfinite(volume_delta) or volume_delta < 0):
        volume_delta = None

    price_row = None
    if quote_ok:
        # We know the current quote close and volume delta. We do not invent a
        # current-minute high/low/open from day-level OHLC values.
        price_row = {
            "Open": np.nan,
            "High": np.nan,
            "Low": np.nan,
            "Close": float(actual_price),
            "Volume": float(volume_delta) if volume_delta is not None else np.nan,
        }

    output = {
        "record_type": "all_nse_minute_prediction_actual",
        "schema_version": "all-nse-minute-prediction-v1",
        "snapshot_minute_ist": snapshot_minute.isoformat(),
        "symbol": symbol,
        "exchange": "NSE",
        "actual_market_value": actual_price,
        "actual_price_source": "INDSTOCKS_BATCH_QUOTE",
        "actual_observed_at_ist": observed_at.isoformat(),
        "actual_observed_at_utc": observed_at.tz_convert("UTC").isoformat(),
        "actual_timestamp_basis": "collector_observation_time_not_exchange_tick_timestamp",
        "actual_quote_status": quote_status,
        # Model execution follows Drive persistence. Measure collection lag from
        # the intended slot, not from the later processing time.
        "actual_quote_age_seconds": abs((observed_at - snapshot_minute).total_seconds()),
        "day_volume": raw_day_volume,
        "minute_volume_delta": volume_delta,
        "current_minute_ohlc_complete": False,
        "current_minute_bar_note": (
            "Only the live quote close and cumulative-volume delta are available "
            "from this snapshot; current-minute open/high/low are not fabricated."
        ),
        "prediction_status": "WARMING_HISTORY" if quote_ok else "MISSING_CURRENT_QUOTE",
        "raw_model_signal": None,
        "signal": "HOLD",
        "probability_up_30m": None,
        "expected_return_30m": None,
        "confidence": None,
        "model_family": "logistic_regression+ridge",
        "model_target": "target_30m",
        "model_coverage_symbols": None,
        "trained_for_symbol": None,
        "feature_count": len(READINESS_FEATURES) + 5,
        "non_missing_feature_count": 0,
        "readiness_feature_count": len(READINESS_FEATURES),
        "data_readiness_score": 0.0,
        "data_limited": True,
        "forecast_target_timestamp_utc": None,
        "forecast_key": None,
        "error": None,
    }
    return output, price_row, raw_day_volume


def _signal(probability_up: float, readiness: float) -> tuple[str, str]:
    """Return (raw model class signal, conservative manual signal)."""
    raw = "BUY" if probability_up >= 0.60 else ("SELL" if probability_up <= 0.40 else "HOLD")
    return raw, raw if readiness >= MIN_SIGNAL_READINESS else "HOLD"


def _is_auth_recovery_error(error: Any) -> bool:
    """Identify errors that mean the shared provider token is unusable.

    This deliberately targets authentication/token-generation failures rather
    than ordinary per-symbol no-data responses, which should still be retried
    and audited per batch.
    """
    detail = str(error or "").lower()
    return any(marker in detail for marker in (
        "token generation is cooling down",
        "token generation failed",
        "automatic re-authentication failed",
        "automatic reauthentication failed",
        "please wait a min",
        "access token may have expired",
        "invalid access token",
        "access_token is either incorrect",
        "access token is either incorrect",
        "http 401; the access token",
        "http 403; authentication or endpoint permissions",
    ))


def _primary_signal_with_readiness_guard(
    prediction_status: Any,
    data_limited: Any,
    candidate_signal: Any,
) -> str:
    """Never expose BUY/SELL as the primary action for a limited/unknown row."""
    status = str(prediction_status or "").strip().upper()
    ready_flag = isinstance(data_limited, (bool, np.bool_)) and not bool(data_limited)
    if status != "PASS" or not ready_flag:
        return "HOLD"
    signal = str(candidate_signal or "HOLD").strip().upper()
    return signal if signal in {"BUY", "SELL", "HOLD"} else "HOLD"


def _snapshot_quote_error(
    quote: dict[str, Any] | None,
    expected_minute: pd.Timestamp,
) -> str | None:
    """Validate one symbol's own quote against the exact capture slot."""
    quote = quote or {}
    if not str(quote.get("symbol") or "").strip():
        return "Symbol is blank."
    if not str(quote.get("instrument_code") or "").strip():
        return "Instrument code is missing."
    cycle_minute = _parse_ist_timestamp(quote.get("cycle_minute_ist"))
    if cycle_minute is None or cycle_minute != expected_minute:
        return "Quote does not belong to the requested snapshot minute."
    if str(quote.get("quote_status") or "").strip().upper() != "PASS":
        return "Quote status is not PASS."
    price = _safe_float(quote.get("actual_observed_price"))
    if price is None or price <= 0:
        return "Quote price is missing or invalid."
    observed_at = _parse_ist_timestamp(
        quote.get("quote_observed_at_ist") or quote.get("quote_observed_at_utc")
    )
    if observed_at is None:
        return "Quote observation timestamp is missing or invalid."
    if abs((observed_at - expected_minute).total_seconds()) > MAX_QUOTE_AGE_SECONDS:
        return "Quote observation timestamp is outside the freshness window."
    return None


def _capture_snapshot_validation_error(
    capture_manifest: dict[str, Any],
    quote_frame: pd.DataFrame,
    minute: pd.Timestamp,
) -> str | None:
    """Reject predictions unless the same-slot source snapshot was persisted and is structurally sound."""
    status = str(capture_manifest.get("status") or "").strip().upper()
    if status not in {"OK", "PARTIAL"}:
        return f"Capture status {status or 'MISSING'} is not eligible for prediction."
    expected_remote_path = (
        "production_data/all_universe_minute_snapshots/"
        f"{minute:%Y/%m/%d}/snapshot_{minute:%Y%m%d_%H%M}.parquet"
    )
    if capture_manifest.get("snapshot_drive_path") != expected_remote_path:
        return "Same-slot snapshot Drive persistence was not confirmed."
    try:
        expected_count = int(capture_manifest.get("expected_symbol_count") or 0)
        manifest_rows = int(capture_manifest.get("row_count") or 0)
    except (TypeError, ValueError):
        expected_count = manifest_rows = 0
    if expected_count <= 0 or manifest_rows != expected_count or len(quote_frame) != expected_count:
        return (
            "Snapshot row count does not match the capture manifest: "
            f"expected={expected_count}, manifest={manifest_rows}, actual={len(quote_frame)}."
        )
    required = {
        "symbol", "instrument_code", "cycle_minute_ist", "quote_status",
        "actual_observed_price", "quote_observed_at_ist",
    }
    missing = sorted(required - set(quote_frame.columns))
    if missing:
        return f"Snapshot is missing required columns: {', '.join(missing)}."
    symbols = quote_frame["symbol"].astype(str).str.strip().str.upper()
    if symbols.eq("").any() or symbols.duplicated().any():
        return "Snapshot has blank or duplicate symbol identities."
    slot_times = quote_frame["cycle_minute_ist"].map(_parse_ist_timestamp)
    if slot_times.isna().any() or not slot_times.eq(minute).all():
        return "Snapshot contains rows from a different or unparseable capture minute."
    if not any(
        _snapshot_quote_error(row, minute) is None
        for row in quote_frame.to_dict(orient="records")
    ):
        return "Snapshot contains no individually valid symbol quotes."
    return None


class AllNSEMinutePredictor:
    """Rate-limited history cache and one-row-per-symbol minute inference engine."""

    def __init__(self, worker_root: str | Path):
        self.worker_root = Path(worker_root)
        self.root = self.worker_root / "all_nse_minute_predictions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._seed_started = False
        self._seed_thread: threading.Thread | None = None
        self._snapshot_seed_minute: pd.Timestamp | None = None
        self._snapshot_seed_root: Path | None = None
        self._history_snapshot_slot_count = 0
        self._seed_ready = False
        # Readiness is progressive; completion is tracked separately so the
        # worker can forecast eligible symbols while remaining history is seeded.
        self._seed_complete = False
        self._seed_status = "NOT_STARTED"
        self._seed_error: str | None = None
        self._seed_universe_source = "NOT_SET"
        self._seed_coverage = 0
        self._seed_errors = 0
        self._seed_deferred_batches = 0
        self._symbols: list[str] = []
        self._codes: dict[str, str] = {}
        self._history: dict[str, pd.DataFrame] = {}
        self._pending_quote_rows: dict[str, list[tuple[pd.Timestamp, dict[str, Any]]]] = {}
        self._last_day_volume: dict[str, tuple[date, float]] = {}
        self._last_minute: str | None = None
        self._model_api = None
        self._classifier = None
        self._regressor = None
        self._model_status: dict[str, Any] = {}
        self._trained_symbols: set[str] = set()
        self._nifty_code: str | None = None
        self._market_history = pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([], tz=IST, name="Datetime"),
        )
        self._fno_context_raw: pd.DataFrame | None = None
        self._fno_context_loaded = False
        self._fno_context_last_attempt = 0.0
        self._fno_context_retry_after = 0.0
        self._fno_context_status = "NOT_LOADED"
        self._fno_context_error: str | None = None
        self._fno_context_source_max_date: str | None = None
        self._calibration = None
        self._calibration_init_error_logged = False
        self._history_request_lock = threading.Lock()
        self._next_history_request_at = 0.0
        self._history_refresh_run_lock = threading.Lock()
        self._history_refresh_scheduled = False
        self._last_history_refresh_minute: str | None = None
        self._last_history_refresh_summary: dict[str, Any] = {}
        # Symbols left stale by a provider/auth interruption are retried first
        # on the next history refresh rather than repeatedly falling at the tail.
        self._history_refresh_pending_symbols: set[str] = set()
        self._last_health: dict[str, Any] = {
            "status": "NOT_STARTED",
            "history_ready": False,
            "universe_symbols": 0,
            "history_symbols": 0,
            "last_capture_minute_ist": None,
            "last_prediction_count": 0,
            "last_persistence_status": None,
        }

    def start(self) -> None:
        """Start or resume a deferred history bootstrap; never duplicate a live seed thread."""
        with self._lock:
            if self._seed_complete:
                return
            if self._seed_thread is not None and self._seed_thread.is_alive():
                return
            self._seed_started = True
            self._seed_status = (
                "RESUMING_HISTORY" if self._seed_ready else "WARMING_HISTORY"
            )
            decision_minute = self._snapshot_seed_minute or pd.Timestamp.now(tz=IST).floor("5min")
            thread = threading.Thread(
                target=self._seed_snapshot_history,
                args=(decision_minute,),
                name="all-nse-five-minute-snapshot-history-seed",
                daemon=True,
            )
            self._seed_thread = thread
        thread.start()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return _clean({
                **self._last_health,
                "status": self._seed_status if self._last_health.get("status") in {"NOT_STARTED", "WARMING_HISTORY"} else self._last_health.get("status"),
                "history_ready": self._seed_ready,
                "history_seed_complete": self._seed_complete,
                "universe_symbols": len(self._symbols),
                "history_symbols": len(self._history),
                "history_seed_coverage_symbols": self._seed_coverage,
                "history_seed_failed_batches": self._seed_errors,
                "history_seed_deferred_batches": self._seed_deferred_batches,
                "history_seed_error": self._seed_error,
                "history_seed_universe_source": self._seed_universe_source,
                "history_refresh_pending_symbols": len(self._history_refresh_pending_symbols),
                "trained_symbol_count": len(self._trained_symbols),
            })

    def _production_model_version(self) -> str:
        trained_until = self._model_status.get("trained_until")
        version = self._model_status.get("model_version") or trained_until
        if version:
            return f"all_nse_intraday__{str(version).strip()}"
        return "all_nse_intraday__unversioned"

    def _apply_calibration_overlay(self, row: dict[str, Any]) -> dict[str, Any]:
        """Fail-open, version-specific calibration; SHADOW never changes live outputs."""
        mode = calibration_overlay_module.resolve_mode(
            os.getenv("ALL_NSE_CALIBRATION_MODE", "active")
        )
        if mode == "off":
            return row
        try:
            if self._calibration is None:
                def load_profile_document():
                    from persistent_store import GoogleDriveStore
                    from googleapiclient.http import MediaIoBaseDownload
                    store = GoogleDriveStore()
                    remote = "production_data/intraday_predictions/learning/calibration/active_calibration_profile.json"
                    parent = store.root_id
                    parts = remote.strip("/").split("/")
                    for part in parts[:-1]:
                        found = store._find_child(parent, part)
                        if not found:
                            return None
                        parent = found["id"]
                    found = store._find_child(parent, parts[-1])
                    if not found:
                        return None
                    buffer = io.BytesIO()
                    request = store.service.files().get_media(fileId=found["id"])
                    downloader = MediaIoBaseDownload(buffer, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    return json.loads(buffer.getvalue().decode("utf-8"))

                self._calibration = calibration_overlay_module.CalibrationOverlay(
                    load_profile_document, mode=mode, refresh_seconds=900
                )
            else:
                self._calibration.mode = mode
            return self._calibration.annotate(row)
        except Exception as exc:
            if not self._calibration_init_error_logged:
                self._log(
                    "all_nse_calibration_overlay_unavailable",
                    error=f"{type(exc).__name__}: {str(exc)[:240]}",
                    fail_open=True,
                )
                self._calibration_init_error_logged = True
            return row

    @staticmethod
    def _nifty_instrument_code() -> str | None:
        """Do not convert an INDstocks source=index underlying ID into an NSE scrip code.

        The source=index instrument master exposes index underlyings, not a
        documented NSE cash-equity token accepted by /market/quotes/full or
        /market/historical/1minute. The old NSE_<SECURITY_ID> conversion caused
        HTTP 400 on both REST paths. NIFTY context uses verified one-minute Yahoo
        candles as a fallback instead; no WebSocket token or fabricated OHLC is used.
        """
        return None

    @staticmethod
    def _fetch_yahoo_nifty_history(minute: pd.Timestamp) -> pd.DataFrame:
        """Fetch genuine, fresh one-minute NIFTY 50 candles with fallback diagnostics."""
        columns = ["Open", "High", "Low", "Close", "Volume"]
        empty = pd.DataFrame(
            columns=columns,
            index=pd.DatetimeIndex([], tz=IST, name="Datetime"),
        )
        cutoff = _as_ist_timestamp(minute).floor("min")
        attempts: list[dict[str, Any]] = []

        # Try the existing direct Yahoo chart adapter first, then the repository's
        # wider historical adapter (which can use cached/provider sources), and
        # finally yfinance's direct download path. This is deliberately market
        # context only; no proxy stock or synthetic OHLC is substituted for NIFTY.
        try:
            from data_fetcher import _normalize_historical_frame, _yahoo_chart_history, get_historical
        except Exception as exc:
            _normalize_historical_frame = None
            _yahoo_chart_history = None
            get_historical = None
            attempts.append({
                "source": "data_fetcher_import",
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            })

        candidates: list[tuple[str, Any]] = []
        if _yahoo_chart_history is not None:
            candidates.append((
                "yahoo_chart",
                lambda: _yahoo_chart_history("^NSEI", "5d", "1m"),
            ))
        if get_historical is not None:
            candidates.append((
                "get_historical",
                lambda: get_historical("^NSEI", period="5d", interval="1m", force_refresh=True),
            ))

        def normalize_candidate(candidate: Any) -> pd.DataFrame:
            if candidate is None or getattr(candidate, "empty", True):
                return empty.copy()

            # yfinance 1.x returns MultiIndex columns even for one ticker,
            # commonly (Price, Ticker). Passing that shape through the older
            # repository normalizer can raise TypeError or collapse all OHLCV
            # fields to the ticker label. Select OHLCV fields BEFORE normalizing.
            frame = candidate.copy()
            field_aliases = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adj close": "Adj Close",
                "adj_close": "Adj Close",
                "volume": "Volume",
                "vol": "Volume",
            }

            if isinstance(frame.columns, pd.MultiIndex):
                picked: dict[str, int] = {}
                column_tokens = [
                    [str(part).strip().lower() for part in column]
                    for column in frame.columns
                ]
                # Prefer a real Close over Adj Close; the latter is only a
                # fallback for chart adapters that omit the unadjusted field.
                for field in ("Open", "High", "Low", "Close", "Volume"):
                    matches = [
                        i for i, tokens in enumerate(column_tokens)
                        if field.lower() in tokens
                    ]
                    if not matches and field == "Close":
                        matches = [
                            i for i, tokens in enumerate(column_tokens)
                            if any(token in {"adj close", "adj_close"} for token in tokens)
                        ]
                    if matches:
                        picked[field] = matches[0]
                if "Close" not in picked:
                    return empty.copy()
                ordered_fields = [field for field in columns if field in picked]
                frame = frame.iloc[:, [picked[field] for field in ordered_fields]].copy()
                frame.columns = ordered_fields
            else:
                # Keep support for the repository's flat-column historical
                # normalizer, but don't feed it yfinance's MultiIndex frame.
                if _normalize_historical_frame is not None:
                    try:
                        normalized = _normalize_historical_frame(frame)
                        if normalized is not None and not normalized.empty:
                            frame = normalized.copy()
                    except (TypeError, ValueError, KeyError):
                        # This fallback parser is intentionally narrow and
                        # deterministic; it only accepts named OHLCV columns.
                        frame = candidate.copy()

                if isinstance(frame.columns, pd.MultiIndex):
                    # Some adapters wrap their result in MultiIndex columns.
                    picked: dict[str, int] = {}
                    tokens_by_column = [
                        [str(part).strip().lower() for part in column]
                        for column in frame.columns
                    ]
                    for field in ("Open", "High", "Low", "Close", "Volume"):
                        matches = [
                            i for i, tokens in enumerate(tokens_by_column)
                            if field.lower() in tokens
                        ]
                        if not matches and field == "Close":
                            matches = [
                                i for i, tokens in enumerate(tokens_by_column)
                                if any(token in {"adj close", "adj_close"} for token in tokens)
                            ]
                        if matches:
                            picked[field] = matches[0]
                    if "Close" not in picked:
                        return empty.copy()
                    ordered_fields = [field for field in columns if field in picked]
                    frame = frame.iloc[:, [picked[field] for field in ordered_fields]].copy()
                    frame.columns = ordered_fields
                else:
                    rename: dict[Any, str] = {}
                    for col in frame.columns:
                        token = str(col).strip().lower()
                        mapped = field_aliases.get(token)
                        if mapped:
                            # Preserve an existing unadjusted Close if both
                            # Close and Adj Close are supplied.
                            if mapped != "Adj Close":
                                rename[col] = mapped
                            elif "Close" not in {
                                field_aliases.get(str(existing).strip().lower(), str(existing))
                                for existing in frame.columns
                            }:
                                rename[col] = "Close"
                    frame = frame.rename(columns=rename)

            if frame is None or frame.empty:
                return empty.copy()
            # Drop duplicate normalized names rather than allowing frame["Close"]
            # to produce a DataFrame (which breaks boolean masks in pandas 3.x).
            frame = frame.loc[:, ~pd.Index(frame.columns).duplicated(keep="first")].copy()
            raw_index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="coerce"))
            if raw_index.tz is None:
                raw_index = raw_index.tz_localize(IST)
            else:
                raw_index = raw_index.tz_convert(IST)
            frame.index = raw_index
            frame.index.name = "Datetime"
            frame = frame.loc[~frame.index.isna()]
            for column in columns:
                if column not in frame.columns:
                    frame[column] = np.nan
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frame = frame.loc[
                (pd.Index(frame.index.date) == cutoff.date())
                & (frame.index < cutoff)
                & frame["Close"].notna()
                & frame["Close"].gt(0)
            ].sort_index()
            frame = frame.loc[~frame.index.duplicated(keep="last")]
            return frame[columns].tail(HISTORY_LOOKBACK_BARS)

        def try_source(source: str, loader: Any) -> pd.DataFrame:
            """Fetch and normalize one source while keeping failures stage-specific."""
            try:
                raw = loader()
            except Exception as exc:
                attempts.append({
                    "source": source,
                    "stage": "fetch",
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                })
                return empty.copy()

            try:
                frame = normalize_candidate(raw)
            except Exception as exc:
                attempts.append({
                    "source": source,
                    "stage": "normalize",
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                })
                return empty.copy()

            latest = frame.index.max().isoformat() if not frame.empty else None
            attempts.append({
                "source": source,
                "stage": "normalize",
                "status": "OK" if not frame.empty else "NO_FRESH_CURRENT_SESSION_CANDLES",
                "rows": int(len(frame)),
                "latest_candle_ist": latest,
            })
            if frame.empty:
                return empty.copy()

            frame.attrs["market_context_source"] = source
            AllNSEMinutePredictor._log(
                "all_nse_nifty_context_fetch",
                status="OK",
                source=source,
                rows=int(len(frame)),
                latest_candle_ist=latest,
                cutoff_ist=cutoff.isoformat(),
                attempts=attempts,
            )
            return frame

        for source, loader in candidates:
            frame = try_source(source, loader)
            if not frame.empty:
                return frame

        # yfinance.Ticker.history has a different wrapper path from yf.download.
        # Prefer it as the next fallback; it often yields ordinary OHLCV columns
        # where yf.download can return MultiIndex columns or fail in its internals.
        try:
            import yfinance as yf

            ticker = yf.Ticker("^NSEI")
            ticker_history = getattr(ticker, "history", None)
            if callable(ticker_history):
                frame = try_source(
                    "yfinance_ticker_history",
                    lambda: ticker_history(
                        period="1d",
                        interval="1m",
                        auto_adjust=False,
                        actions=False,
                        timeout=10,
                    ),
                )
                if not frame.empty:
                    return frame
            else:
                attempts.append({
                    "source": "yfinance_ticker_history",
                    "stage": "setup",
                    "status": "UNAVAILABLE",
                    "error": "Ticker.history is not callable",
                })
        except Exception as exc:
            attempts.append({
                "source": "yfinance_ticker_history",
                "stage": "setup",
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            })

        # Keep yf.download as the final adapter, and record whether failure occurs
        # during fetching or normalization. Never fabricate NIFTY candles.
        try:
            import yfinance as yf
        except Exception as exc:
            attempts.append({
                "source": "yfinance_download",
                "stage": "import",
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            })
        else:
            frame = try_source(
                "yfinance_download",
                lambda: yf.download(
                    "^NSEI",
                    period="1d",
                    interval="1m",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    timeout=10,
                ),
            )
            if not frame.empty:
                return frame

        AllNSEMinutePredictor._log(
            "all_nse_nifty_context_fetch",
            status="UNAVAILABLE",
            source="none",
            rows=0,
            cutoff_ist=cutoff.isoformat(),
            attempts=attempts,
            action="keep_market_features_missing_and_hold_low_readiness_rows",
        )
        return empty

    @staticmethod
    def _previous_trading_date(value: date) -> date:
        from market_session_guard import is_nse_trading_day

        d = value - timedelta(days=1)
        for _ in range(15):
            if is_nse_trading_day(d):
                return d
            d -= timedelta(days=1)
        return value - timedelta(days=1)

    @staticmethod
    def _market_history_before(
        market_history: pd.DataFrame | None,
        minute: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return only pre-decision benchmark rows; absent context must not block stock features.

        When the NIFTY token/history endpoint is unavailable, the initial empty
        DataFrame can carry a RangeIndex. Comparing that index with a Timestamp
        raises TypeError and had incorrectly marked almost every symbol
        FEATURE_BUILD_ERROR. Treat absent/malformed benchmark context as an
        empty timestamp-indexed OHLCV frame so its features become NaN and the
        model's existing readiness gate keeps the primary signal conservative.
        """
        columns = ["Open", "High", "Low", "Close", "Volume"]
        empty = pd.DataFrame(
            columns=columns,
            index=pd.DatetimeIndex([], tz=IST, name="Datetime"),
        )
        if (
            not isinstance(market_history, pd.DataFrame)
            or market_history.empty
            or not isinstance(market_history.index, pd.DatetimeIndex)
        ):
            return empty
        market = market_history.copy()
        idx = market.index
        market.index = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
        decision_minute = _as_ist_timestamp(minute).floor("min")
        before = market.loc[market.index < decision_minute].copy()
        if before.empty:
            return empty
        # Market context is as time-sensitive as the individual stock candle.
        # Do not count stale benchmark features toward readiness.
        latest_timestamp = pd.Timestamp(before.index.max())
        age_minutes = (decision_minute - latest_timestamp).total_seconds() / 60.0
        if age_minutes < 0 or age_minutes > MAX_HISTORY_CANDLE_AGE_MINUTES:
            return empty
        return before

    def _fresh_market_history_before(
        self,
        market_history: pd.DataFrame | None,
        minute: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return fresh pre-decision NIFTY candles, refreshing stale cache independently.

        Equity seeding/refresh can be delayed by provider authentication or quota
        errors. Keep the market-context feed independent so a stale NIFTY cache does
        not remove ten market/relative-return features from every stock at once.
        The fallback only accepts real, current-session one-minute candles strictly
        before the decision minute; it never fabricates index OHLC or uses future bars.
        """
        before = self._market_history_before(market_history, minute)
        if not before.empty:
            return before

        try:
            refreshed = self._fetch_yahoo_nifty_history(minute)
            fresh_before = self._market_history_before(refreshed, minute)
        except Exception as exc:
            fresh_before = pd.DataFrame(
                columns=["Open", "High", "Low", "Close", "Volume"],
                index=pd.DatetimeIndex([], tz=IST, name="Datetime"),
            )
            self._log(
                "all_nse_nifty_context_fast_refresh",
                status="UNAVAILABLE",
                cutoff_ist=_as_ist_timestamp(minute).isoformat(),
                error=f"{type(exc).__name__}: {str(exc)[:180]}",
            )
            return fresh_before

        if fresh_before.empty:
            self._log(
                "all_nse_nifty_context_fast_refresh",
                status="UNAVAILABLE",
                cutoff_ist=_as_ist_timestamp(minute).isoformat(),
                reason="no fresh real current-session candles returned",
            )
            return before

        with self._lock:
            existing = self._market_history.copy()
            if isinstance(existing.index, pd.DatetimeIndex) and not existing.empty:
                idx = existing.index
                existing.index = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
                # Keep only candles at or before the cutoff; the cache must never
                # retain an accidentally future-dated row as context for this slot.
                existing = existing.loc[existing.index < _as_ist_timestamp(minute)]
            else:
                existing = fresh_before.iloc[0:0].copy()
            merged = pd.concat([existing, fresh_before]).sort_index()
            merged = merged.loc[~merged.index.duplicated(keep="last")]
            self._market_history = merged.tail(HISTORY_LOOKBACK_BARS)

        self._log(
            "all_nse_nifty_context_fast_refresh",
            status="OK",
            source=str(refreshed.attrs.get("market_context_source") or "verified_yahoo_fallback"),
            cutoff_ist=_as_ist_timestamp(minute).isoformat(),
            latest_candle_ist=fresh_before.index.max().isoformat(),
            row_count=len(fresh_before),
        )
        return fresh_before

    @staticmethod
    def _parse_candles(candles: list[dict[str, Any]]) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        frame = pd.DataFrame(candles)
        needed = {"ts", "o", "h", "l", "c", "v"}
        if not needed.issubset(frame.columns):
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        idx = pd.to_datetime(
            pd.to_numeric(frame["ts"], errors="coerce"),
            unit="s",
            utc=True,
            errors="coerce",
        ).dt.tz_convert(IST)
        out = pd.DataFrame(
            {
                "Open": pd.to_numeric(frame["o"], errors="coerce").to_numpy(),
                "High": pd.to_numeric(frame["h"], errors="coerce").to_numpy(),
                "Low": pd.to_numeric(frame["l"], errors="coerce").to_numpy(),
                "Close": pd.to_numeric(frame["c"], errors="coerce").to_numpy(),
                "Volume": pd.to_numeric(frame["v"], errors="coerce").to_numpy(),
            },
            index=pd.DatetimeIndex(idx, name="Datetime"),
        )
        out = out.loc[~out.index.isna()].sort_index()
        out = out.loc[~out.index.duplicated(keep="last")]
        return out

    def _reserve_history_request_slot(self) -> None:
        """Coordinate all history calls under the provider's five-requests/second limit."""
        with self._history_request_lock:
            delay = self._next_history_request_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next_history_request_at = time.monotonic() + HISTORY_REFRESH_REQUEST_SPACING_SECONDS

    def _publish_history_seed_progress(
        self,
        updates: dict[str, pd.DataFrame],
        *,
        coverage_symbols: int,
        failed_batches: int,
        complete: bool = False,
    ) -> None:
        """Publish completed history batches immediately instead of waiting for the full universe."""
        with self._lock:
            for symbol, frame in updates.items():
                if frame.empty:
                    continue
                if symbol == "__NIFTY50__":
                    existing = self._market_history
                else:
                    existing = self._history.get(symbol)
                if existing is not None and not existing.empty:
                    # Seed results are fetched first; an already-refreshed live
                    # history frame is newer and wins on duplicate timestamps.
                    frame = pd.concat([frame, existing]).sort_index()
                    frame = frame.loc[~frame.index.duplicated(keep="last")]
                if symbol == "__NIFTY50__":
                    self._market_history = frame.tail(HISTORY_LOOKBACK_BARS)
                else:
                    self._history[symbol] = frame.tail(HISTORY_LOOKBACK_BARS)
            self._seed_coverage = int(coverage_symbols)
            self._seed_errors = int(failed_batches)
            self._seed_ready = bool(self._history)
            self._seed_complete = bool(complete)
            if not self._seed_ready:
                self._seed_status = "WARMING_HISTORY"
            elif complete and self._seed_coverage >= len(self._symbols) and failed_batches == 0:
                self._seed_status = "READY"
            else:
                self._seed_status = "READY_PARTIAL"

    def _seed_history(self) -> None:
        """Fetch one recent trading session plus the current session in batches of five."""
        while True:
            try:
                from all_universe_minute_capture import _instrument_map
                from delivery_training_universe import official_nse_symbols
                from indstocks_client import _request, configured

                # NIFTY is independent of the INDstocks equity instrument master.
                # Fetch benchmark history first so an INDstocks token failure cannot
                # suppress market-context features for otherwise valid stock quotes.
                market_now = pd.Timestamp.now(tz=IST)
                market_cutoff = market_now.floor("min")
                with self._lock:
                    cached_market = self._market_history.copy()
                market_needs_refresh = cached_market.empty
                if not cached_market.empty and isinstance(cached_market.index, pd.DatetimeIndex):
                    latest_market = _as_ist_timestamp(cached_market.index.max())
                    market_age = (market_cutoff - latest_market).total_seconds() / 60.0
                    market_needs_refresh = (
                        latest_market.date() != market_cutoff.date()
                        or market_age > MAX_HISTORY_CANDLE_AGE_MINUTES
                    )
                if market_needs_refresh:
                    try:
                        nifty_frame = self._fetch_yahoo_nifty_history(market_cutoff)
                        if not nifty_frame.empty:
                            with self._lock:
                                combined_market = pd.concat(
                                    [self._market_history, nifty_frame]
                                ).sort_index()
                                combined_market = combined_market.loc[
                                    ~combined_market.index.duplicated(keep="last")
                                ]
                                self._market_history = combined_market.tail(HISTORY_LOOKBACK_BARS)
                            self._log(
                                "all_nse_nifty_seed_before_provider_auth",
                                status="OK",
                                rows=int(len(nifty_frame)),
                                latest_candle_ist=nifty_frame.index.max().isoformat(),
                                cutoff_ist=market_cutoff.isoformat(),
                            )
                    except Exception as nifty_exc:
                        # NIFTY is optional context; this must never prevent stock
                        # quotes, history recovery, or snapshot persistence.
                        self._log(
                            "all_nse_nifty_seed_before_provider_auth",
                            status="UNAVAILABLE",
                            error_type=type(nifty_exc).__name__,
                        )

                if not configured():
                    raise RuntimeError("INDstocks credentials are not configured")
                # Load the validated provider instrument master first. If the NSE
                # archive roster is blocked (for example HTTP 403 on EQUITY_L.csv),
                # use the mapped NSE equity symbols for model warm-up instead of
                # leaving this revision with a zero-symbol predictor.
                codes = _instrument_map()
                official_universe_error = None
                try:
                    official_symbols = official_nse_symbols(force_refresh=False)
                except Exception as exc:
                    official_symbols = []
                    official_universe_error = exc
                symbols, universe_source = _history_seed_universe(
                    official_symbols,
                    codes,
                )
                if official_universe_error is not None or universe_source != "NSE_OFFICIAL_EQUITY_L":
                    self._log(
                        "all_nse_history_universe_fallback",
                        source=universe_source,
                        symbol_count=len(symbols),
                        official_error_type=(
                            type(official_universe_error).__name__
                            if official_universe_error is not None else None
                        ),
                    )
                try:
                    nifty_code = self._nifty_instrument_code()
                except Exception as exc:
                    # Index context is optional and must never prevent stock
                    # prediction warm-up from starting.
                    nifty_code = None
                    self._log(
                        "all_nse_nifty_instrument_lookup_failed",
                        error_type=type(exc).__name__,
                    )
                with self._lock:
                    self._symbols = symbols
                    self._codes = {s: codes[s] for s in symbols if s in codes}
                    self._nifty_code = nifty_code
                    self._seed_universe_source = universe_source

                now = pd.Timestamp.now(tz=IST)
                previous_day = self._previous_trading_date(now.date())
                start = pd.Timestamp(datetime.combine(previous_day, wall_time(9, 15), tzinfo=IST))
                end = now
                start_ms = int(start.timestamp() * 1000)
                end_ms = int(end.timestamp() * 1000)
                # Preserve real history already published by previous successful
                # batches or the current-session refresh. A resumed seed only needs
                # to fetch symbols for which no cached history exists.
                with self._lock:
                    histories: dict[str, pd.DataFrame] = {
                        symbol: frame.copy()
                        for symbol, frame in self._history.items()
                        if symbol in symbols and frame is not None and not frame.empty
                    }
                    if not self._market_history.empty:
                        histories["__NIFTY50__"] = self._market_history.copy()

                entries = [
                    (symbol, codes[symbol])
                    for symbol in symbols
                    if symbol in codes and (symbol not in histories or histories[symbol].empty)
                ]
                seed_nifty_code = (
                    nifty_code
                    if "__NIFTY50__" not in histories or histories["__NIFTY50__"].empty
                    else None
                )
                batches = _history_seed_batches(
                    entries,
                    seed_nifty_code,
                    HISTORY_BATCH_SIZE,
                )

                batch_errors = 0
                requests_made = 0
                auth_circuit_open = False
                auth_error = None
                deferred_batches = 0
                for batch_index, batch in enumerate(batches, start=1):
                    code_list = [code for _, code in batch]
                    body = None
                    batch_error = None
                    for attempt in range(3):
                        try:
                            self._reserve_history_request_slot()
                            body = _request(
                                "GET",
                                "/market/historical/1minute",
                                params={
                                    "scrip-codes": ",".join(code_list),
                                    "start_time": start_ms,
                                    "end_time": end_ms,
                                },
                            ).json()
                            requests_made += 1
                            success = isinstance(body, dict) and (
                                body.get("success") is True
                                or str(body.get("status") or "").strip().lower() == "success"
                            )
                            if not success:
                                raise RuntimeError("historical API response did not report success")
                            batch_error = None
                            break
                        except Exception as exc:
                            batch_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                            if _is_auth_recovery_error(batch_error):
                                # A rejected token is a shared-account failure, not
                                # a per-symbol data problem. Retrying every batch
                                # only repeats the same token-generation HTTP 400.
                                auth_circuit_open = True
                                auth_error = batch_error
                                break
                            if attempt < 2:
                                # Honor the provider's five-request/second limit and
                                # back off if a transient 429/5xx occurs.
                                time.sleep(0.6 * (attempt + 1))
                    is_nifty_batch = len(batch) == 1 and batch[0][0] == "__NIFTY50__"
                    if batch_error:
                        if is_nifty_batch:
                            # NIFTY context is optional to the stock-universe seed;
                            # a provider-specific index error must not count as a
                            # failed stock history batch.
                            self._log(
                                "all_nse_nifty_history_seed_failed",
                                error_type=batch_error.split(":", 1)[0],
                            )
                        else:
                            batch_errors += 1
                            self._log(
                                "all_nse_history_batch_failed",
                                batch_index=batch_index,
                                batch_size=len(batch),
                                error=batch_error,
                            )
                    else:
                        data = body.get("data") or {}
                        for symbol, code in batch:
                            instrument = data.get(code) or {}
                            frame = self._parse_candles(instrument.get("candles") or [])
                            if not frame.empty:
                                # Do not seed a still-forming candle; the live quote is
                                # appended as a separately timestamped partial row.
                                frame = frame.loc[frame.index < now.floor("min")]
                                if not frame.empty:
                                    histories[symbol] = frame.tail(HISTORY_LOOKBACK_BARS)
                    # Publish each successful batch as soon as it returns. Waiting
                    # until all ~500 universe batches finish left history coverage
                    # reported as zero throughout the morning and suppressed inference.
                    batch_updates = {
                        symbol: histories[symbol]
                        for symbol, _ in batch
                        if symbol in histories
                    }
                    batch_coverage = sum(
                        1 for symbol in symbols
                        if symbol in histories and not histories[symbol].empty
                    )
                    self._publish_history_seed_progress(
                        batch_updates,
                        coverage_symbols=batch_coverage,
                        failed_batches=batch_errors,
                    )
                    if auth_circuit_open:
                        deferred_batches = max(0, len(batches) - batch_index)
                        self._log(
                            "all_nse_history_seed_auth_deferred",
                            failed_batch_index=batch_index,
                            deferred_batches=deferred_batches,
                            history_symbols_ready=batch_coverage,
                            auth_error=auth_error,
                            action="stop_seed_requests_until_next_scheduled_tick",
                        )
                        break

                # Retry missing equity histories individually. The provider may
                # reject one malformed/stale code in a multi-symbol request; this
                # salvages the other symbols and brings coverage back above its
                # existing threshold when the source actually has candles.
                historical_coverage = sum(
                    1 for symbol in symbols
                    if symbol in histories and not histories[symbol].empty
                )
                coverage_target = min(
                    len(symbols),
                    int(np.ceil(len(symbols) * MIN_HISTORY_SEED_COVERAGE)),
                )
                targeted_attempts = 0
                targeted_recovered = 0
                targeted_errors = 0
                missing_entries = [
                    (symbol, code) for symbol, code in entries
                    if symbol not in histories or histories[symbol].empty
                ]
                max_targeted_attempts = min(40, len(missing_entries))
                for symbol, code in missing_entries[:max_targeted_attempts]:
                    if auth_circuit_open or historical_coverage >= coverage_target:
                        break
                    targeted_attempts += 1
                    try:
                        self._reserve_history_request_slot()
                        retry_body = _request(
                            "GET",
                            "/market/historical/1minute",
                            params={
                                "scrip-codes": code,
                                "start_time": start_ms,
                                "end_time": end_ms,
                            },
                        ).json()
                        requests_made += 1
                        retry_success = isinstance(retry_body, dict) and (
                            retry_body.get("success") is True
                            or str(retry_body.get("status") or "").strip().lower() == "success"
                        )
                        if not retry_success:
                            raise RuntimeError("single-symbol historical response did not report success")
                        retry_instrument = (retry_body.get("data") or {}).get(code) or {}
                        retry_frame = self._parse_candles(
                            retry_instrument.get("candles") or []
                        )
                        if not retry_frame.empty:
                            retry_frame = retry_frame.loc[
                                retry_frame.index < now.floor("min")
                            ]
                        if not retry_frame.empty:
                            histories[symbol] = retry_frame.tail(HISTORY_LOOKBACK_BARS)
                            historical_coverage += 1
                            targeted_recovered += 1
                    except Exception as exc:
                        targeted_errors += 1
                        # Avoid flooding Cloud Logging when a large tail of the
                        # official universe has no usable minute history.
                        if targeted_errors <= 3:
                            self._log(
                                "all_nse_history_targeted_retry_failed",
                                symbol=symbol,
                                error_type=type(exc).__name__,
                            )
                    self._publish_history_seed_progress(
                        {symbol: histories[symbol]} if symbol in histories else {},
                        coverage_symbols=sum(
                            1 for item in symbols
                            if item in histories and not histories[item].empty
                        ),
                        failed_batches=batch_errors,
                    )

                self._log(
                    "all_nse_history_seed_repair_summary",
                    initial_batch_errors=batch_errors,
                    targeted_attempts=targeted_attempts,
                    targeted_recovered=targeted_recovered,
                    targeted_errors=targeted_errors,
                    history_symbols_after_repair=historical_coverage,
                    history_coverage_target=coverage_target,
                )

                # If INDstocks cannot provide a documented REST index token,
                # use genuine Yahoo NIFTY 50 one-minute OHLCV candles for context.
                # This does not alter the stock history-coverage counter.
                if "__NIFTY50__" not in histories or histories["__NIFTY50__"].empty:
                    nifty_history = self._fetch_yahoo_nifty_history(now.floor("min"))
                    if not nifty_history.empty:
                        histories["__NIFTY50__"] = nifty_history
                    self._log(
                        "all_nse_nifty_yahoo_history_seed",
                        source="YAHOO_FINANCE_1M",
                        row_count=int(len(nifty_history)),
                        latest_candle_ist=(
                            nifty_history.index.max().isoformat()
                            if not nifty_history.empty else None
                        ),
                        status="OK" if not nifty_history.empty else "UNAVAILABLE",
                    )

                # Count only actual historical API results, not live quote rows
                # buffered while the potentially long bootstrap is in progress.
                historical_coverage = sum(
                    1 for symbol in symbols
                    if symbol in histories and not histories[symbol].empty
                )
                with self._lock:
                    # Merge point-in-time quotes observed while seed requests ran.
                    for symbol, pending in self._pending_quote_rows.items():
                        base = histories.get(symbol, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
                        add = []
                        for ts, values in pending:
                            add.append(pd.DataFrame([values], index=pd.DatetimeIndex([ts], name="Datetime")))
                        if add:
                            base = pd.concat([base, *add]).sort_index()
                            base = base.loc[~base.index.duplicated(keep="last")]
                        histories[symbol] = base.tail(HISTORY_LOOKBACK_BARS)
                    market_pending = self._pending_quote_rows.get("__NIFTY50__", [])
                    market_base = histories.get("__NIFTY50__", pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
                    if market_pending:
                        add = [
                            pd.DataFrame([values], index=pd.DatetimeIndex([ts], name="Datetime"))
                            for ts, values in market_pending
                        ]
                        market_base = pd.concat([market_base, *add]).sort_index()
                        market_base = market_base.loc[~market_base.index.duplicated(keep="last")]
                    # Keep the index's context history separate from stock histories;
                    # an available NIFTY series alone must not mark all-stock history ready.
                    # Merge with any newer frames published by the incremental
                    # refresh path while the initial seed was in progress.
                    for symbol, live_frame in self._history.items():
                        if symbol == "__NIFTY50__" or live_frame.empty:
                            continue
                        seeded_frame = histories.get(
                            symbol,
                            pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
                        )
                        if seeded_frame.empty:
                            histories[symbol] = live_frame
                        else:
                            merged = pd.concat([seeded_frame, live_frame]).sort_index()
                            histories[symbol] = merged.loc[
                                ~merged.index.duplicated(keep="last")
                            ].tail(HISTORY_LOOKBACK_BARS)
                    self._history = {
                        symbol: frame.tail(HISTORY_LOOKBACK_BARS)
                        for symbol, frame in histories.items()
                        if symbol != "__NIFTY50__" and not frame.empty
                    }
                    # Preserve fresher NIFTY candles fetched during the bootstrap.
                    if not self._market_history.empty:
                        market_base = pd.concat([market_base, self._market_history]).sort_index()
                        market_base = market_base.loc[
                            ~market_base.index.duplicated(keep="last")
                        ]
                    self._market_history = market_base.tail(HISTORY_LOOKBACK_BARS)
                    self._seed_coverage = historical_coverage
                    self._seed_errors = batch_errors
                    self._seed_deferred_batches = deferred_batches
                    self._seed_error = (
                        auth_error
                        if auth_circuit_open
                        else None if batch_errors == 0
                        else f"{batch_errors} historical batches did not return a success payload"
                    )
                    self._seed_ready = bool(self._history)
                    # An auth-rejected seed is deferred, not completed; start()
                    # can relaunch it on a later capture tick with cached history.
                    seed_quality_complete = _history_seed_meets_quality_gate(
                        historical_coverage,
                        len(symbols),
                        batch_errors,
                    )
                    self._seed_complete = not auth_circuit_open and seed_quality_complete
                    self._seed_status = (
                        "AUTH_DEFERRED" if auth_circuit_open
                        else "READY" if self._seed_ready and seed_quality_complete
                        else "READY_PARTIAL" if self._seed_ready
                        else "HISTORY_SEED_EMPTY"
                    )
                    self._pending_quote_rows.clear()
                if not self._seed_ready and not auth_circuit_open:
                    raise RuntimeError("Historical warm-up returned no usable OHLCV bars")
                self._log(
                    "all_nse_history_seed_completed",
                    universe_count=len(symbols),
                    mapped_symbol_count=len(self._codes),
                    history_symbol_count=self._seed_coverage,
                    history_universe_source=self._seed_universe_source,
                    historical_batch_count=requests_made,
                    failed_batches=batch_errors,
                    deferred_batches=deferred_batches,
                    auth_circuit_open=auth_circuit_open,
                    auth_error=auth_error,
                    status="AUTH_DEFERRED" if auth_circuit_open else self._seed_status,
                    nifty_history_rows=len(self._market_history),
                )
                return
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {str(exc)[:240]}"
                auth_deferred = _is_auth_recovery_error(error_text)
                with self._lock:
                    self._seed_error = error_text
                    self._seed_ready = bool(self._history)
                    self._seed_complete = False
                    if auth_deferred:
                        remaining_symbols = max(0, len(self._symbols) - self._seed_coverage)
                        self._seed_deferred_batches = int(
                            np.ceil(remaining_symbols / max(1, HISTORY_BATCH_SIZE))
                        )
                        self._seed_status = "AUTH_DEFERRED"
                    else:
                        self._seed_status = (
                            "READY_PARTIAL" if self._seed_ready else "WARMING_HISTORY_ERROR"
                        )
                if auth_deferred:
                    self._log(
                        "all_nse_history_seed_auth_deferred",
                        stage="bootstrap",
                        auth_error=error_text,
                        history_symbols_ready=self._seed_coverage,
                        deferred_batches=self._seed_deferred_batches,
                        action="stop_attempt_and_retry_on_next_scheduled_tick",
                    )
                    return
                self._log("all_nse_history_seed_failed", error=error_text)
                time.sleep(120)

    @staticmethod
    def _snapshot_filename_minute(name: str) -> pd.Timestamp | None:
        try:
            parsed = datetime.strptime(Path(name).stem, "snapshot_%Y%m%d_%H%M")
            return pd.Timestamp(parsed, tz=IST)
        except (TypeError, ValueError):
            return None

    def _snapshot_archive_paths(self, decision_minute: pd.Timestamp) -> list[Path]:
        """Load only persisted five-minute snapshots; no live one-minute history endpoint."""
        root = Path(self._snapshot_seed_root or self.worker_root)
        local_dir = root / "all_universe_minute_snapshots"
        local_dir.mkdir(parents=True, exist_ok=True)

        def prior_local_paths() -> list[Path]:
            found = []
            for path in local_dir.glob("snapshot_*.parquet"):
                slot = self._snapshot_filename_minute(path.name)
                if slot is not None and slot < decision_minute:
                    found.append((slot, path))
            return [path for _, path in sorted(found, key=lambda item: item[0])]

        local = prior_local_paths()
        if len(local) >= SNAPSHOT_HISTORY_LOOKBACK_BARS:
            return local[-SNAPSHOT_HISTORY_LOOKBACK_BARS:]

        try:
            from persistent_store import GoogleDriveStore
            from googleapiclient.http import MediaIoBaseDownload

            store = GoogleDriveStore()
            candidates: dict[str, tuple[pd.Timestamp, str]] = {}
            day_cursor = decision_minute.date()
            for _ in range(6):
                parent = store.root_id
                parts = [
                    "production_data", "all_universe_minute_snapshots",
                    f"{day_cursor:%Y}", f"{day_cursor:%m}", f"{day_cursor:%d}",
                ]
                try:
                    for part in parts:
                        found = store._find_child(parent, part, "application/vnd.google-apps.folder")
                        if not found:
                            parent = None
                            break
                        parent = found["id"]
                    if parent:
                        listing = store.service.files().list(
                            q=f"'{parent}' in parents and trashed = false",
                            spaces="drive",
                            fields="files(id,name,mimeType)",
                            pageSize=100,
                            includeItemsFromAllDrives=True,
                            supportsAllDrives=True,
                        ).execute()
                        for item in listing.get("files", []):
                            name = str(item.get("name") or "")
                            slot = self._snapshot_filename_minute(name)
                            if name.endswith(".parquet") and name.startswith("snapshot_") and slot is not None and slot < decision_minute:
                                candidates[name] = (slot, str(item["id"]))
                except Exception as exc:
                    self._log(
                        "all_nse_snapshot_history_folder_list_failed",
                        date=day_cursor.isoformat(),
                        error=f"{type(exc).__name__}: {str(exc)[:160]}",
                    )
                day_cursor = self._previous_trading_date(day_cursor)

            local_names = {path.name for path in local_dir.glob("snapshot_*.parquet")}
            need = max(0, SNAPSHOT_HISTORY_LOOKBACK_BARS - len(local))
            remote_order = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)
            downloaded = 0
            for name, (slot, file_id) in remote_order:
                if need <= 0:
                    break
                if name in local_names:
                    continue
                destination = local_dir / name
                buffer_path = destination.with_suffix(".download")
                request = store.service.files().get_media(fileId=file_id)
                with buffer_path.open("wb") as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                if buffer_path.stat().st_size == 0:
                    buffer_path.unlink(missing_ok=True)
                    continue
                buffer_path.replace(destination)
                downloaded += 1
                need -= 1
            self._log(
                "all_nse_snapshot_history_archive_loaded",
                local_prior_snapshots=len(local),
                remote_snapshots_downloaded=downloaded,
                requested_lookback=SNAPSHOT_HISTORY_LOOKBACK_BARS,
            )
        except Exception as exc:
            self._log(
                "all_nse_snapshot_history_archive_unavailable",
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
                action="use_only_already_persisted_local_five_minute_snapshots",
            )

        return prior_local_paths()[-SNAPSHOT_HISTORY_LOOKBACK_BARS:]

    def _seed_snapshot_history(self, decision_minute: pd.Timestamp) -> None:
        """Seed per-symbol history from the persisted five-minute snapshot archive only."""
        minute = _as_ist_timestamp(decision_minute).floor("5min")
        started = time.monotonic()
        try:
            paths = self._snapshot_archive_paths(minute)
            frames = []
            universe_symbols: set[str] = set()
            for path in paths:
                try:
                    frame = pd.read_parquet(path)
                except Exception as exc:
                    self._log(
                        "all_nse_snapshot_history_file_read_failed",
                        path=str(path),
                        error=f"{type(exc).__name__}: {str(exc)[:120]}",
                    )
                    continue
                required = {"symbol", "cycle_minute_ist", "quote_status", "actual_observed_price"}
                if not required.issubset(frame.columns):
                    continue
                frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
                universe_symbols.update(frame.loc[frame["symbol"].ne(""), "symbol"].tolist())
                frame["_snapshot_ts"] = pd.to_datetime(
                    frame["cycle_minute_ist"], errors="coerce", utc=True
                ).dt.tz_convert(IST)
                frame = frame.loc[frame["_snapshot_ts"].notna() & (frame["_snapshot_ts"] < minute)].copy()
                if frame.empty:
                    continue
                for col in ("day_volume", "day_open", "day_high", "day_low"):
                    if col not in frame:
                        frame[col] = np.nan
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
                frame["actual_observed_price"] = pd.to_numeric(frame["actual_observed_price"], errors="coerce")
                frames.append(frame[[
                    "symbol", "_snapshot_ts", "quote_status", "actual_observed_price",
                    "day_volume", "day_open", "day_high", "day_low",
                ]])
            if not frames:
                raise RuntimeError("No prior persisted five-minute snapshots could be loaded")
            all_rows = pd.concat(frames, ignore_index=True)
            slot_count = int(all_rows["_snapshot_ts"].nunique())
            rows = all_rows.sort_values(["symbol", "_snapshot_ts"]).copy()
            rows["_session_date"] = rows["_snapshot_ts"].dt.date
            rows["_quote_valid"] = (
                rows["quote_status"].astype(str).str.upper().eq("PASS")
                & rows["actual_observed_price"].gt(0)
            )
            rows["Close"] = rows["actual_observed_price"].where(rows["_quote_valid"])
            previous_ts = rows.groupby("symbol")["_snapshot_ts"].shift(1)
            previous_close = rows.groupby("symbol")["Close"].shift(1)
            previous_day_volume = rows.groupby("symbol")["day_volume"].shift(1)
            contiguous = (
                rows["_snapshot_ts"].sub(previous_ts).eq(pd.Timedelta(minutes=SNAPSHOT_CADENCE_MINUTES))
                & rows["_session_date"].eq(previous_ts.dt.date)
                & rows["_quote_valid"]
                & previous_close.notna()
            )
            volume_delta = rows["day_volume"] - previous_day_volume
            rows["Volume"] = volume_delta.where(
                contiguous & volume_delta.ge(0) & rows["day_volume"].notna()
            )

            histories: dict[str, pd.DataFrame] = {}
            for symbol, group in rows.groupby("symbol", sort=False):
                history = group.set_index("_snapshot_ts")[
                    ["Close", "Volume", "day_open", "day_high", "day_low", "day_volume"]
                ].sort_index()
                history = history.loc[~history.index.duplicated(keep="last")]
                history.index.name = "Datetime"
                histories[str(symbol)] = history.tail(SNAPSHOT_HISTORY_LOOKBACK_BARS)

            coverage = sum(
                1 for history in histories.values()
                if not history.empty and history["Close"].notna().any()
            )
            coverage_target = int(np.ceil(len(universe_symbols) * MIN_HISTORY_SEED_COVERAGE))
            complete = bool(
                slot_count >= SNAPSHOT_HISTORY_LOOKBACK_BARS
                and len(universe_symbols) > 0
                and coverage >= coverage_target
            )
            with self._lock:
                # Do not drop a newer slot appended while archive I/O was running.
                for symbol, old in self._history.items():
                    if old is not None and not old.empty:
                        merged = pd.concat([histories[symbol], old]).sort_index() if symbol in histories else old.copy()
                        histories[symbol] = merged.loc[~merged.index.duplicated(keep="last")].tail(SNAPSHOT_HISTORY_LOOKBACK_BARS)
                self._history = histories
                self._symbols = sorted(set(self._symbols) | universe_symbols)
                self._seed_coverage = sum(
                    1 for frame in histories.values()
                    if frame is not None and not frame.empty and "Close" in frame and frame["Close"].notna().any()
                )
                self._seed_errors = 0
                self._seed_deferred_batches = 0
                self._seed_universe_source = "PERSISTED_FIVE_MINUTE_SNAPSHOT_ARCHIVE"
                self._seed_ready = bool(histories)
                self._seed_complete = bool(
                    complete and self._seed_coverage >= int(np.ceil(len(self._symbols) * MIN_HISTORY_SEED_COVERAGE))
                )
                self._seed_status = (
                    "READY" if self._seed_complete
                    else "READY_PARTIAL" if self._seed_ready
                    else "WARMING_HISTORY_ERROR"
                )
                self._seed_error = None if self._seed_complete else (
                    f"Five-minute snapshot seed incomplete: slots={slot_count}, "
                    f"symbols_with_history={self._seed_coverage}/{len(self._symbols)}, "
                    f"required_slots={SNAPSHOT_HISTORY_LOOKBACK_BARS}"
                )
                self._history_snapshot_slot_count = slot_count
            self._log(
                "all_nse_five_minute_snapshot_history_seed_completed",
                cutoff_ist=minute.isoformat(),
                snapshot_slots_loaded=slot_count,
                universe_symbols=len(self._symbols),
                history_symbols=self._seed_coverage,
                required_history_slots=SNAPSHOT_HISTORY_LOOKBACK_BARS,
                status=self._seed_status,
                history_source=self._seed_universe_source,
                elapsed_seconds=round(time.monotonic() - started, 2),
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            with self._lock:
                self._seed_error = error
                self._seed_ready = bool(self._history)
                self._seed_complete = False
                self._seed_status = "READY_PARTIAL" if self._seed_ready else "WARMING_HISTORY_ERROR"
            self._log(
                "all_nse_five_minute_snapshot_history_seed_failed",
                cutoff_ist=minute.isoformat(),
                error=error,
            )

    def _append_five_minute_snapshot(self, quote_frame: pd.DataFrame, minute: pd.Timestamp) -> None:
        """Append every symbol row at the five-minute slot, preserving missing quotes as gaps."""
        current: dict[str, dict[str, Any]] = {}
        expected_previous = minute - pd.Timedelta(minutes=SNAPSHOT_CADENCE_MINUTES)
        for item in quote_frame.to_dict(orient="records"):
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            price = _safe_float(item.get("actual_observed_price"))
            quote_good = (
                str(item.get("quote_status") or "").strip().upper() == "PASS"
                and price is not None and price > 0
            )
            day_volume = _safe_float(item.get("day_volume")) if quote_good else np.nan
            day_open = _safe_float(item.get("day_open")) if quote_good else np.nan
            day_high = _safe_float(item.get("day_high")) if quote_good else np.nan
            day_low = _safe_float(item.get("day_low")) if quote_good else np.nan
            previous_day_volume = np.nan
            with self._lock:
                old = self._history.get(symbol)
                if old is not None and not old.empty and "day_volume" in old:
                    last_ts = _as_ist_timestamp(old.index.max())
                    if last_ts == expected_previous and last_ts.date() == minute.date():
                        previous_day_volume = _safe_float(old.iloc[-1].get("day_volume"))
            delta = (
                day_volume - previous_day_volume
                if np.isfinite(day_volume) and np.isfinite(previous_day_volume)
                else np.nan
            )
            if not np.isfinite(delta) or delta < 0:
                delta = np.nan
            current[symbol] = {
                "Close": price if quote_good else np.nan,
                "Volume": delta,
                "day_open": day_open,
                "day_high": day_high,
                "day_low": day_low,
                "day_volume": day_volume,
            }
        with self._lock:
            for symbol, values in current.items():
                row = pd.DataFrame(
                    [values],
                    index=pd.DatetimeIndex([minute], tz=IST, name="Datetime"),
                )
                old = self._history.get(symbol)
                combined = pd.concat([old, row]).sort_index() if old is not None and not old.empty else row
                self._history[symbol] = combined.loc[
                    ~combined.index.duplicated(keep="last")
                ].tail(SNAPSHOT_HISTORY_LOOKBACK_BARS)
            quote_symbols = (
                set(quote_frame["symbol"].astype(str).str.strip().str.upper())
                if "symbol" in quote_frame else set()
            )
            self._symbols = sorted(set(self._symbols) | {s for s in quote_symbols if s})
            self._history_snapshot_slot_count = max(
                self._history_snapshot_slot_count,
                len({ts for frame in self._history.values() if frame is not None for ts in frame.index}),
            )
            self._seed_coverage = sum(
                1 for frame in self._history.values()
                if frame is not None and not frame.empty and "Close" in frame and frame["Close"].notna().any()
            )
            coverage_target = int(np.ceil(len(self._symbols) * MIN_HISTORY_SEED_COVERAGE))
            if (
                self._history_snapshot_slot_count >= SNAPSHOT_HISTORY_LOOKBACK_BARS
                and len(self._symbols) > 0
                and self._seed_coverage >= coverage_target
            ):
                self._seed_ready = True
                self._seed_complete = True
                self._seed_status = "READY"
                self._seed_error = None

    @staticmethod
    def _log(event: str, **fields: Any) -> None:
        print(
            "[ALL_NSE_MINUTE_PREDICTOR] "
            + json.dumps(_clean({
                "event": event,
                "logged_at_utc": datetime.now(timezone.utc).isoformat(),
                **fields,
            }), sort_keys=True),
            flush=True,
        )

    def _ensure_model(self) -> None:
        if self._classifier is not None and self._regressor is not None:
            return
        import all_nse_intraday_model as model

        status = model.model_status()
        if not status.get("artifact_hashes_match"):
            raise RuntimeError("Pooled ALL-NSE intraday model classifier/regressor hashes do not match the manifest")
        if not status.get("feature_schema_match"):
            raise RuntimeError(
                "Pooled ALL-NSE intraday model is incompatible with the five-minute snapshot feature schema "
                f"{status.get('feature_schema_version') or 'missing'}; retrain and verify before deployment"
            )
        if not status.get("manifest_sha256_match"):
            raise RuntimeError("Pooled ALL-NSE intraday model manifest integrity validation failed")
        classifier, regressor = model._models()
        with self._lock:
            self._model_api = model
            self._classifier = classifier
            self._regressor = regressor
            self._model_status = status
            self._trained_symbols = {
                str(s).strip().upper().replace(".NS", "")
                for s in (status.get("training_symbols") or [])
            }

    def _load_completed_fno_context(
        self,
        decision_minute: pd.Timestamp,
    ) -> tuple[dict[str, dict[str, Any]], str, str | None]:
        """Load daily F&O data once per decision date, using only prior-date rows."""
        decision_date = decision_minute.date()
        now_mono = time.monotonic()
        with self._lock:
            raw = (
                self._fno_context_raw.copy()
                if self._fno_context_raw is not None
                else None
            )
            cached_date = getattr(self, "_fno_context_cache_decision_date", None)
            retry_after = self._fno_context_retry_after

        cached_context = _latest_completed_fno_context(raw, decision_date) if raw is not None else {}
        if cached_date == decision_date and cached_context:
            source_dates = [row["source_date"] for row in cached_context.values() if row.get("source_date")]
            latest_source_date = max(source_dates) if source_dates else None
            cached_status = (
                "STALE_CACHE_FALLBACK"
                if self._fno_context_status == "STALE_CACHE_FALLBACK"
                else "OK"
            )
            return cached_context, cached_status, latest_source_date
        if cached_date == decision_date and now_mono < retry_after:
            status = "STALE_CACHE_FALLBACK" if cached_context else self._fno_context_status
            source_dates = [row["source_date"] for row in cached_context.values() if row.get("source_date")]
            latest_source_date = max(source_dates) if source_dates else None
            return cached_context, status, latest_source_date

        cache_path = self.root / "fno_daily_features_2026.parquet"
        staging_path = cache_path.with_suffix(".download")
        try:
            import all_nse_intraday_model as model

            # Refresh once on a new decision date or after an empty/stale result.
            # Use a staging file so a failed read cannot corrupt the last good cache.
            staging_path.unlink(missing_ok=True)
            model._drive_download(FNO_DAILY_REMOTE, staging_path)
            if not staging_path.is_file() or staging_path.stat().st_size == 0:
                raise RuntimeError("Drive F&O feature download was empty")
            downloaded = pd.read_parquet(staging_path)
            if downloaded.empty or not {"symbol", "date"}.issubset(
                {str(c).strip().lower() for c in downloaded.columns}
            ):
                raise RuntimeError("Daily F&O feature parquet has no usable symbol/date rows")
            staging_path.replace(cache_path)
            raw = downloaded
            context = _latest_completed_fno_context(raw, decision_date)
            source_dates = [row["source_date"] for row in context.values() if row.get("source_date")]
            latest_source_date = max(source_dates) if source_dates else None
            status = "OK" if context else "NO_FRESH_PRIOR_SESSION_ROWS"
            with self._lock:
                self._fno_context_raw = raw.copy()
                self._fno_context_loaded = True
                self._fno_context_cache_decision_date = decision_date
                self._fno_context_status = status
                self._fno_context_source_max_date = latest_source_date
                self._fno_context_error = None if context else (
                    "No F&O feature rows from the previous seven calendar days before decision date"
                )
                self._fno_context_retry_after = (
                    0.0 if context else time.monotonic() + FNO_CONTEXT_RETRY_SECONDS
                )
            self._log(
                "all_nse_fno_context_ready",
                status=status,
                remote_path=FNO_DAILY_REMOTE,
                decision_date=decision_date.isoformat(),
                source_max_date=latest_source_date,
                symbols_with_context=len(context),
                max_age_days=FNO_CONTEXT_MAX_AGE_DAYS,
                point_in_time_rule="source date strictly before decision date",
            )
            return context, status, latest_source_date
        except Exception as exc:
            staging_path.unlink(missing_ok=True)
            message = f"{type(exc).__name__}: {str(exc)[:200]}"
            # Retain only genuine, still-fresh prior-date rows from the last cache.
            fallback = cached_context
            fallback_dates = [row["source_date"] for row in fallback.values() if row.get("source_date")]
            latest_source_date = max(fallback_dates) if fallback_dates else None
            status = "STALE_CACHE_FALLBACK" if fallback else "UNAVAILABLE"
            with self._lock:
                self._fno_context_status = status
                self._fno_context_error = message
                self._fno_context_cache_decision_date = decision_date
                self._fno_context_source_max_date = latest_source_date
                self._fno_context_retry_after = time.monotonic() + FNO_CONTEXT_RETRY_SECONDS
            self._log(
                "all_nse_fno_context_load",
                status=status,
                remote_path=FNO_DAILY_REMOTE,
                cached_symbols_used=len(fallback),
                cached_source_max_date=latest_source_date,
                error=message,
                retry_after_seconds=FNO_CONTEXT_RETRY_SECONDS,
            )
            return fallback, status, latest_source_date

    def _fetch_current_nifty_quote(self) -> tuple[float | None, pd.Timestamp | None, str | None]:
        """Return a fresh market-context close; never treat an index ID as an equity token."""
        quote_error = "INDstocks source=index IDs are not supported REST scrip-codes"
        if self._nifty_code:
            try:
                from indstocks_client import _request

                observed = pd.Timestamp.now(tz=IST)
                body = _request(
                    "GET",
                    "/market/quotes/full",
                    params={"scrip-codes": self._nifty_code},
                ).json()
                data = body.get("data") or {}
                quote = data.get(self._nifty_code) or {}
                price = _safe_float(quote.get("live_price"))
                if price is not None and price > 0:
                    return price, observed, None
                quote_error = "INDstocks NIFTY REST response had no positive live_price"
            except Exception as exc:
                quote_error = f"{type(exc).__name__}: {str(exc)[:160]}"

        # This fallback is the close of a real one-minute benchmark candle, not
        # an exchange-stamped live tick. Require a fresh candle.
        with self._lock:
            market = self._market_history.copy()
        if isinstance(market.index, pd.DatetimeIndex) and not market.empty and "Close" in market:
            market = market.loc[market.index <= pd.Timestamp.now(tz=IST)]
            close = pd.to_numeric(market["Close"], errors="coerce").dropna()
            if not close.empty:
                timestamp = pd.Timestamp(close.index[-1])
                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(IST)
                else:
                    timestamp = timestamp.tz_convert(IST)
                age_seconds = (pd.Timestamp.now(tz=IST) - timestamp).total_seconds()
                price = _safe_float(close.iloc[-1])
                if (
                    price is not None and price > 0 and 0 <= age_seconds
                    <= MAX_HISTORY_CANDLE_AGE_MINUTES * 60
                ):
                    return price, timestamp, None
        return None, None, quote_error

    def _schedule_current_session_history_refresh(
        self,
        minute: pd.Timestamp,
    ) -> dict[str, Any]:
        """Refresh already-seeded history in the background for the next slot.

        The full-universe quote capture already takes roughly two minutes. Running
        another 500+ provider requests synchronously after capture can push the
        whole Scheduler request past its 300-second deadline. During bootstrap,
        refresh only instruments whose history is already cached; the seed thread
        continues filling the remaining universe. Both paths use the same locked
        request-spacing limiter, and inference still enforces strict timestamp and
        freshness checks before any signal can be actionable.
        """
        minute = _as_ist_timestamp(minute).floor("min")
        minute_key = minute.isoformat()
        with self._lock:
            if self._last_history_refresh_minute == minute_key:
                return dict(self._last_history_refresh_summary)
            if self._history_refresh_scheduled:
                return {
                    "status": "IN_PROGRESS",
                    "minute": minute_key,
                    "batch_count": 0,
                    "failed_batches": 0,
                    "failed_symbols": [],
                    "updated_symbols": 0,
                    "current_session_candle_rows": 0,
                }
            # A partial seed is usable for maintenance: the refresh worker itself
            # restricts requests to cached symbols until bootstrap is complete.
            # Seed and refresh requests share the same request-spacing lock.
            if not self._seed_ready or not self._codes:
                return {
                    "status": "WARMING_SEED",
                    "minute": minute_key,
                    "batch_count": 0,
                    "failed_batches": 0,
                    "failed_symbols": [],
                    "updated_symbols": 0,
                    "current_session_candle_rows": 0,
                }
            self._history_refresh_scheduled = True

        def refresh_in_background() -> None:
            try:
                summary = self._refresh_current_session_history(minute)
                self._log(
                    "all_nse_background_history_refresh_finished",
                    **summary,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                with self._lock:
                    self._last_history_refresh_minute = minute_key
                    self._last_history_refresh_summary = {
                        "status": "ERROR",
                        "minute": minute_key,
                        "batch_count": 0,
                        "failed_batches": 1,
                        "failed_symbols": [],
                        "updated_symbols": 0,
                        "current_session_candle_rows": 0,
                        "error": error,
                    }
                self._log(
                    "all_nse_background_history_refresh_failed",
                    minute=minute_key,
                    error=error,
                )
            finally:
                with self._lock:
                    self._history_refresh_scheduled = False

        threading.Thread(
            target=refresh_in_background,
            name="all-nse-current-session-history-refresh",
            daemon=True,
        ).start()
        return {
            "status": "SCHEDULED_ASYNC",
            "minute": minute_key,
            "batch_count": 0,
            "failed_batches": 0,
            "failed_symbols": [],
            "updated_symbols": 0,
            "current_session_candle_rows": 0,
        }

    def _wait_for_current_session_history_refresh(
        self,
        minute: pd.Timestamp,
        *,
        timeout_seconds: float = 45.0,
    ) -> dict[str, Any]:
        """Wait briefly for a pre-capture refresh to finish before inference.

        The full-universe quote collector runs in parallel with history refresh.
        This bounded hand-off gives the refresh time to complete without allowing
        a slow provider request to hang Cloud Scheduler. If the wait expires, the
        stale-history guard still rejects outdated inputs.
        """
        minute = _as_ist_timestamp(minute).floor("min")
        minute_key = minute.isoformat()
        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        while True:
            with self._lock:
                in_progress = self._history_refresh_scheduled
                completed_minute = self._last_history_refresh_minute
                summary = dict(self._last_history_refresh_summary)
                pending_count = len(self._history_refresh_pending_symbols)
            if not in_progress:
                if completed_minute == minute_key:
                    return summary
                return {
                    "status": "NOT_CONFIRMED",
                    "minute": minute_key,
                    "last_completed_minute": completed_minute,
                    "pending_symbol_count": pending_count,
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {
                    "status": "WAIT_TIMEOUT",
                    "minute": minute_key,
                    "last_completed_minute": completed_minute,
                    "pending_symbol_count": pending_count,
                    "wait_timeout_seconds": max(0.0, float(timeout_seconds)),
                }
            time.sleep(min(0.25, remaining))

    def _refresh_current_session_history(self, minute: pd.Timestamp) -> dict[str, Any]:
        """Refresh genuine one-minute candles; never turn LTP snapshots into OHLC bars."""
        minute = _as_ist_timestamp(minute).floor("min")
        minute_key = minute.isoformat()
        with self._history_refresh_run_lock:
            if self._last_history_refresh_minute == minute_key:
                return dict(self._last_history_refresh_summary)
            with self._lock:
                seed_ready = self._seed_ready
                seed_complete = self._seed_complete
                codes = dict(self._codes)
                nifty_code = self._nifty_code
                history_symbols = set(self._history)
                pending_history_symbols = set(self._history_refresh_pending_symbols)
                if not seed_complete:
                    # Do not run hundreds of refresh requests concurrently with
                    # initial bootstrap. Refresh only symbols already seeded; the
                    # roster expands progressively as the seed thread succeeds.
                    codes = {
                        symbol: code for symbol, code in codes.items()
                        if symbol in history_symbols
                    }
            if not seed_ready or not codes:
                return {
                    "status": "WARMING_SEED", "minute": minute_key, "batch_count": 0,
                    "failed_batches": 0, "failed_symbols": [], "updated_symbols": 0,
                    "current_session_candle_rows": 0,
                }
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from datetime import datetime
            from indstocks_client import _request
            session_open = pd.Timestamp(datetime.combine(minute.date(), wall_time(9, 15), tzinfo=IST))
            window_start = max(session_open, minute - pd.Timedelta(minutes=HISTORY_REFRESH_WINDOW_MINUTES))
            # A refresh can take several minutes. Pull each batch up to the most
            # recent completed-minute boundary available when that batch starts,
            # so this cache is useful at the *next* scheduled slot. Inference
            # still filters history strictly before its own decision timestamp.
            batch_end_minutes: list[pd.Timestamp] = [minute]
            batch_end_lock = threading.Lock()
            entries = sorted((str(s), str(c)) for s, c in codes.items() if str(c).strip())
            # Retry prior interruptions first. Successful symbols remain in the
            # cache; the pending set only carries unresolved refresh work forward.
            if pending_history_symbols:
                entries = (
                    [entry for entry in entries if entry[0] in pending_history_symbols]
                    + [entry for entry in entries if entry[0] not in pending_history_symbols]
                )
            if nifty_code:
                entries.insert(0, ("__NIFTY50__", str(nifty_code)))
            batches = [entries[i:i + HISTORY_BATCH_SIZE] for i in range(0, len(entries), HISTORY_BATCH_SIZE)]

            def rate_limit_start() -> None:
                self._reserve_history_request_slot()

            auth_failure_event = threading.Event()
            auth_failure_lock = threading.Lock()
            auth_failure_detail: list[str | None] = [None]
            missing_candle_lock = threading.Lock()
            missing_fresh_candle_symbols: set[str] = set()

            def fetch_batch(batch):
                last_error = None
                if auth_failure_event.is_set():
                    return {}, "AUTH_CIRCUIT_OPEN", True
                for attempt in range(3):
                    if auth_failure_event.is_set():
                        return {}, "AUTH_CIRCUIT_OPEN", True
                    try:
                        rate_limit_start()
                        # Do not send queued requests into a known-bad token state.
                        if auth_failure_event.is_set():
                            return {}, "AUTH_CIRCUIT_OPEN", True
                        current_floor = pd.Timestamp.now(tz=IST).floor("min")
                        batch_end = max(
                            minute,
                            min(current_floor, minute + pd.Timedelta(minutes=5)),
                        )
                        batch_start = max(
                            session_open,
                            batch_end - pd.Timedelta(minutes=HISTORY_REFRESH_WINDOW_MINUTES),
                        )
                        with batch_end_lock:
                            batch_end_minutes.append(batch_end)
                        body = _request("GET", "/market/historical/1minute", params={
                            "scrip-codes": ",".join(code for _, code in batch),
                            "start_time": int(batch_start.timestamp() * 1000),
                            "end_time": int(batch_end.timestamp() * 1000),
                        }).json()
                        successful = isinstance(body, dict) and (
                            body.get("success") is True
                            or str(body.get("status") or "").strip().lower() == "success"
                        )
                        if not successful:
                            raise RuntimeError("historical refresh response did not report success")
                        data = body.get("data") or {}
                        result = {}
                        missing_in_batch = []
                        for symbol, code in batch:
                            frame = self._parse_candles((data.get(code) or {}).get("candles") or [])
                            if not frame.empty:
                                frame = frame.loc[
                                    (pd.Index(frame.index.date) == batch_end.date())
                                    & (frame.index < batch_end)
                                ].tail(HISTORY_REFRESH_WINDOW_MINUTES)
                            if not frame.empty:
                                result[symbol] = frame
                            elif symbol != "__NIFTY50__":
                                # A successful HTTP response without a fresh candle is
                                # not a fresh-history success for that equity. Keep it
                                # pending so the next cycle retries instead of leaving
                                # a stale cache row untracked.
                                missing_in_batch.append(symbol)
                        if missing_in_batch:
                            with missing_candle_lock:
                                missing_fresh_candle_symbols.update(missing_in_batch)
                        return result, None, False
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {str(exc)[:180]}"
                        if _is_auth_recovery_error(last_error):
                            # One auth/token failure invalidates the shared client
                            # for every queued batch. Fail fast; retries belong to
                            # the next scheduled cycle, after the provider cooldown.
                            with auth_failure_lock:
                                if auth_failure_detail[0] is None:
                                    auth_failure_detail[0] = last_error
                                auth_failure_event.set()
                            return {}, last_error, True
                        if attempt < 2:
                            time.sleep(0.25 * (attempt + 1))
                return {}, last_error or "historical refresh failed", False

            started = time.monotonic()
            updated, failed_symbols, errors, candle_rows = {}, [], [], 0
            deferred_batches = 0
            with ThreadPoolExecutor(max_workers=HISTORY_REFRESH_WORKERS) as pool:
                futures = {pool.submit(fetch_batch, batch): batch for batch in batches}
                for future in as_completed(futures):
                    batch = futures[future]
                    try:
                        result, error, auth_blocked = future.result()
                    except Exception as exc:
                        result, error, auth_blocked = {}, f"{type(exc).__name__}: {str(exc)[:180]}", False
                    if error:
                        if auth_blocked and error == "AUTH_CIRCUIT_OPEN":
                            deferred_batches += 1
                        else:
                            errors.append(error)
                            failed_symbols.extend(s for s, _ in batch if s != "__NIFTY50__")
                            if auth_blocked:
                                # Other active workers can observe the same cooldown.
                                deferred_batches += 1
                    for symbol, frame in result.items():
                        updated[symbol] = frame
                        candle_rows += len(frame)

            updated_symbols = set(updated) - {"__NIFTY50__"}
            equity_symbols = set(codes)
            with missing_candle_lock:
                missing_candle_symbols = set(missing_fresh_candle_symbols)
            if missing_candle_symbols:
                failed_symbols.extend(sorted(missing_candle_symbols))
                self._log(
                    "all_nse_history_refresh_missing_fresh_candles",
                    minute=minute_key,
                    symbol_count=len(missing_candle_symbols),
                    symbols=sorted(missing_candle_symbols)[:20],
                    action="queue_for_next_refresh_and_keep_prediction_freshness_guard",
                )
            with self._lock:
                if auth_failure_event.is_set():
                    # Every stock not refreshed successfully remains explicitly
                    # pending and will be attempted first on the next scheduled
                    # pass. Do not label it fresh merely because the batch ran.
                    self._history_refresh_pending_symbols = equity_symbols - updated_symbols
                else:
                    pending = set(self._history_refresh_pending_symbols)
                    pending.difference_update(updated_symbols)
                    pending.update(failed_symbols)
                    self._history_refresh_pending_symbols = pending

            refresh_cutoff = max(batch_end_minutes)
            nifty_history_source = "INDSTOCKS_HISTORICAL_1M" if "__NIFTY50__" in updated else None
            if "__NIFTY50__" not in updated or updated["__NIFTY50__"].empty:
                # Benchmark refresh follows the same current-time cutoff as the
                # equity batches. The next prediction still applies its own
                # strict pre-decision filter, so no future bar enters inference.
                fallback = self._fetch_yahoo_nifty_history(refresh_cutoff)
                if not fallback.empty:
                    fallback = fallback.loc[
                        (pd.Index(fallback.index.date) == refresh_cutoff.date())
                        & (fallback.index < refresh_cutoff)
                    ].tail(HISTORY_REFRESH_WINDOW_MINUTES)
                    if not fallback.empty:
                        updated["__NIFTY50__"] = fallback
                        candle_rows += len(fallback)
                        nifty_history_source = "YAHOO_FINANCE_1M_FALLBACK"
                if nifty_history_source is None:
                    nifty_history_source = "MISSING"

            def merge_history(existing, refreshed):
                columns = ["Open", "High", "Low", "Close", "Volume"]
                existing = existing.copy() if existing is not None else pd.DataFrame(columns=columns)
                if existing.empty:
                    prior, same_day = existing, existing
                else:
                    prior = existing.loc[pd.Index(existing.index.date) < minute.date()].tail(HISTORY_LOOKBACK_BARS)
                    same_day = existing.loc[
                        (pd.Index(existing.index.date) == minute.date()) & (existing.index < minute)
                    ]
                pieces = [p for p in (prior, same_day, refreshed) if p is not None and not p.empty]
                if not pieces:
                    return existing.tail(HISTORY_LOOKBACK_BARS)
                merged = pd.concat(pieces).sort_index()
                return merged.loc[~merged.index.duplicated(keep="last")].tail(HISTORY_LOOKBACK_BARS)

            with self._lock:
                for symbol, frame in updated.items():
                    self._history[symbol] = merge_history(self._history.get(symbol), frame)
                if "__NIFTY50__" in updated:
                    self._market_history = merge_history(self._market_history, updated["__NIFTY50__"])
                summary = {
                    "status": (
                        "AUTH_DEFERRED" if auth_failure_event.is_set()
                        else "OK" if not errors and not missing_candle_symbols else "PARTIAL"
                    ),
                    "minute": minute_key, "batch_count": len(batches),
                    "failed_batches": len(errors), "failed_symbols": sorted(set(failed_symbols)),
                    "missing_fresh_candle_symbols": len(missing_candle_symbols),
                    "missing_fresh_candle_symbol_samples": sorted(missing_candle_symbols)[:20],
                    "updated_symbols": len(updated_symbols),
                    "deferred_batches": deferred_batches,
                    "pending_symbol_count": len(self._history_refresh_pending_symbols),
                    "auth_circuit_open": auth_failure_event.is_set(),
                    "auth_error": auth_failure_detail[0],
                    "current_session_candle_rows": candle_rows,
                    "nifty_history_source": nifty_history_source,
                    "nifty_history_rows": int(len(updated.get("__NIFTY50__", pd.DataFrame()))),
                    "window_start_ist": window_start.isoformat(),
                    "window_end_ist": max(batch_end_minutes).isoformat(),
                    "batch_end_minute_span": {
                        "earliest": min(batch_end_minutes).isoformat(),
                        "latest": max(batch_end_minutes).isoformat(),
                    },
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                    "error_examples": errors[:5],
                }
                self._last_history_refresh_minute = minute_key
                self._last_history_refresh_summary = summary
            self._log("all_nse_current_session_history_refresh", **summary)
            return dict(summary)

    def _persist(self, records: list[dict[str, Any]], manifest: dict[str, Any], minute: pd.Timestamp) -> dict[str, Any]:
        from persistent_store import GoogleDriveStore

        day_dir = self.root / f"{minute:%Y}" / f"{minute:%m}" / f"{minute:%d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        filename = f"predictions_{minute:%Y%m%d_%H%M}.parquet"
        local_path = day_dir / filename
        tmp_path = local_path.with_suffix(".tmp.parquet")
        pd.DataFrame(records).to_parquet(tmp_path, index=False, compression="zstd")
        tmp_path.replace(local_path)
        remote = f"{REMOTE_ROOT}/{minute:%Y/%m/%d}/{filename}"
        store = GoogleDriveStore()
        try:
            uploaded = store.upload_or_update(
                local_path, remote, mime_type="application/vnd.apache.parquet"
            )
            if str(uploaded.get("status") or "").upper() != "OK":
                raise RuntimeError(f"Drive prediction snapshot write not confirmed: {uploaded}")
            receipt = {
                "status": "OK",
                "remote_path": remote,
                "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            local_path.with_suffix(".uploaded.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
            )
            manifest.update({
                "prediction_drive_path": remote,
                "prediction_drive_sync": _clean(uploaded),
                "status": "OK" if manifest.get("quality_gate_pass") is True else "PARTIAL",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as exc:
            manifest.update({
                "status": "PERSISTENCE_ERROR",
                "prediction_drive_path": remote,
                "prediction_drive_error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })

        manifest_name = f"manifest_{minute:%Y%m%d_%H%M}.json"
        manifest_path = day_dir / manifest_name
        manifest_path.write_text(json.dumps(_clean(manifest), indent=2, sort_keys=True), encoding="utf-8")
        manifest_remote = f"{REMOTE_ROOT}/{minute:%Y/%m/%d}/{manifest_name}"
        try:
            out = store.upload_or_update(manifest_path, manifest_remote, mime_type="application/json")
            if str(out.get("status") or "").upper() != "OK":
                raise RuntimeError(f"Drive per-minute manifest write not confirmed: {out}")
            manifest_path.with_suffix(".uploaded.json").write_text(
                json.dumps({
                    "status": "OK", "remote_path": manifest_remote,
                    "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
                }, sort_keys=True),
                encoding="utf-8",
            )
            latest_result = store.upload_or_update(
                manifest_path,
                f"{REMOTE_ROOT}/latest_manifest.json",
                mime_type="application/json",
            )
            if str(latest_result.get("status") or "").upper() != "OK":
                raise RuntimeError(f"Drive latest-manifest write not confirmed: {latest_result}")
        except Exception as exc:
            manifest["manifest_drive_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            manifest["status"] = "PERSISTENCE_ERROR"
            # Do not leave a stale per-minute manifest claiming success when the
            # latest pointer write failed. Retry the immutable minute manifest with
            # the degraded status recorded, without masking the initial error.
            try:
                manifest_path.write_text(
                    json.dumps(_clean(manifest), indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                retry = store.upload_or_update(
                    manifest_path, manifest_remote, mime_type="application/json"
                )
                if str(retry.get("status") or "").upper() == "OK":
                    manifest["manifest_recovery_sync"] = _clean(retry)
            except Exception as retry_exc:
                manifest["manifest_recovery_error"] = (
                    f"{type(retry_exc).__name__}: {str(retry_exc)[:200]}"
                )
        return {
            "status": manifest.get("status"),
            "remote_path": remote,
            "row_count": len(records),
            "prediction_count": manifest.get("prediction_count", 0),
            "missing_quote_count": manifest.get("missing_quote_count", 0),
            "data_limited_count": manifest.get("data_limited_count", 0),
            "manifest_drive_path": manifest_remote,
            "error": manifest.get("prediction_drive_error") or manifest.get("manifest_drive_error"),
        }

    def run_for_capture(
        self,
        capture_manifest: dict[str, Any] | None,
        worker_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Predict for the captured NSE universe and write one row per symbol to Drive."""
        capture_manifest = capture_manifest or {}
        if not capture_manifest.get("cycle_minute_ist"):
            return {
                "status": "NOT_RUN",
                "reason": "This worker tick did not produce a live full-universe quote capture.",
            }
        minute = _as_ist_timestamp(
            capture_manifest.get("cycle_minute_ist"),
            fallback=pd.Timestamp.now(tz=IST).floor("min"),
        ).floor("min")
        minute_key = minute.isoformat()
        capture_status = str(capture_manifest.get("status") or "").strip().upper()
        if capture_status not in {"OK", "PARTIAL"} or not capture_manifest.get("snapshot_drive_path"):
            error = (
                "Prediction blocked: the same-slot market snapshot was not confirmed "
                f"persisted to Drive (capture_status={capture_status or 'MISSING'})."
            )
            self._log(
                "all_nse_minute_prediction_capture_blocked",
                cycle_minute_ist=minute_key,
                status="BLOCKED_SNAPSHOT_CAPTURE",
                capture_status=capture_status or "MISSING",
                error=error,
            )
            return {
                "status": "BLOCKED_SNAPSHOT_CAPTURE",
                "cycle_minute_ist": minute_key,
                "prediction_count": 0,
                "quality_gate_pass": False,
                "error": error,
            }

        source_root = Path(worker_root or self.worker_root)
        snapshot_name = f"snapshot_{minute:%Y%m%d_%H%M}.parquet"
        local_quote_path = source_root / "all_universe_minute_snapshots" / snapshot_name
        try:
            quote_frame = (
                pd.read_parquet(local_quote_path)
                if local_quote_path.is_file()
                else pd.DataFrame()
            )
        except Exception as exc:
            quote_frame = pd.DataFrame()
            read_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        else:
            read_error = None
        snapshot_error = (
            read_error
            or (f"Same-slot local snapshot is missing or empty: {local_quote_path}" if quote_frame.empty else None)
            or _capture_snapshot_validation_error(capture_manifest, quote_frame, minute)
        )
        if snapshot_error:
            error = f"Prediction blocked: {snapshot_error}"
            self._log(
                "all_nse_minute_prediction_capture_blocked",
                cycle_minute_ist=minute_key,
                status="BLOCKED_SNAPSHOT_VALIDATION",
                snapshot_path=str(local_quote_path),
                error=error,
            )
            return {
                "status": "BLOCKED_SNAPSHOT_VALIDATION",
                "cycle_minute_ist": minute_key,
                "prediction_count": 0,
                "quality_gate_pass": False,
                "error": error,
                "snapshot_path": str(local_quote_path),
            }

        quote_frame["symbol"] = quote_frame["symbol"].astype(str).str.strip().str.upper()
        quote_by_symbol = {
            str(row["symbol"]): row
            for row in quote_frame.to_dict(orient="records")
        }

        self._snapshot_seed_minute = minute
        self._snapshot_seed_root = source_root
        self.start()
        if self._last_minute == minute_key:
            return {"status": "DUPLICATE_MINUTE_SKIPPED", "cycle_minute_ist": minute_key}

        # Seed and refresh from the exact persisted five-minute quote snapshots.
        # The live prediction path does not request one-minute stock candles.
        seed_thread = self._seed_thread
        if seed_thread is not None and seed_thread.is_alive():
            seed_thread.join(timeout=18.0)
        history_refresh = {
            "status": "SNAPSHOT_HISTORY_ONLY",
            "minute": minute_key,
            "batch_count": 0,
            "failed_batches": 0,
            "failed_symbols": [],
            "updated_symbols": len(self._history),
            "pending_symbol_count": 0,
            "elapsed_seconds": 0.0,
        }
        failed_refresh_symbols: set[str] = set()
        with self._lock:
            symbols = sorted(quote_by_symbol)
            history_ready = self._seed_ready
            history_seed_complete = self._seed_complete
            seed_status = self._seed_status
            seed_coverage_symbols = self._seed_coverage
            seed_failed_batches = self._seed_errors
            seed_deferred_batches = self._seed_deferred_batches
            seed_universe_source = self._seed_universe_source
            histories = {symbol: frame.copy() for symbol, frame in self._history.items()}
        market_history_before_minute = pd.DataFrame(
            columns=["Close"],
            index=pd.DatetimeIndex([], tz=IST, name="Datetime"),
        )
        nifty_price, nifty_observed_at, nifty_error = None, None, None

        try:
            if history_ready:
                self._ensure_model()
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {str(exc)[:240]}"
        else:
            model_error = None

        fno_context_by_symbol: dict[str, dict[str, Any]] = {}
        fno_context_status = "NOT_ATTEMPTED"
        fno_context_source_max_date = None
        if history_ready and model_error is None and self._model_api is not None:
            (
                fno_context_by_symbol,
                fno_context_status,
                fno_context_source_max_date,
            ) = self._load_completed_fno_context(minute)

        current_rows: list[dict[str, Any]] = []
        feature_rows: list[dict[str, float]] = []
        feature_symbols: list[str] = []
        quote_actuals: dict[str, dict[str, Any]] = {}
        missing_quotes = 0
        data_limited_count = 0
        model_failures = 0
        quote_timestamp = pd.Timestamp.now(tz=IST)

        for symbol in symbols:
            quote = quote_by_symbol.get(symbol)
            hist = histories.get(symbol, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
            hist = hist.copy()
            # Point-in-time safety: the actual quote and derived volume delta may
            # only use completed historical candles strictly before the slot.
            if not hist.empty and isinstance(hist.index, pd.DatetimeIndex):
                hist = hist.loc[hist.index < minute].tail(HISTORY_LOOKBACK_BARS)
            elif not hist.empty:
                hist = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
            previous_snapshot_volume = None
            previous_snapshot_date = None
            if not hist.empty and isinstance(hist.index, pd.DatetimeIndex) and "day_volume" in hist:
                previous_ts = _as_ist_timestamp(hist.index.max())
                if (
                    previous_ts == minute - pd.Timedelta(minutes=SNAPSHOT_CADENCE_MINUTES)
                    and previous_ts.date() == minute.date()
                ):
                    previous_snapshot_volume = _safe_float(hist.iloc[-1].get("day_volume"))
                    if np.isfinite(previous_snapshot_volume):
                        previous_snapshot_date = minute.date()
            quote_info, price_row, raw_day_volume = _current_minute_row(
                symbol=symbol,
                snapshot_minute=minute,
                quote=quote,
                prior_history=hist,
                previous_day_volume=previous_snapshot_volume,
                previous_volume_date=previous_snapshot_date,
                snapshot_date=minute.date(),
            )
            actual_price = quote_info.get("actual_market_value")
            actual_ts = _as_ist_timestamp(
                quote_info.get("actual_observed_at_ist"),
                fallback=minute,
            )
            quote_age = quote_info.get("actual_quote_age_seconds")
            quote_good = _snapshot_quote_error(quote, minute) is None
            if quote_good and raw_day_volume is not None:
                quote_actuals[symbol] = {"day_volume": raw_day_volume, "date": minute.date()}
            if not quote_good:
                missing_quotes += 1
                quote_info["prediction_status"] = (
                    "MISSING_CURRENT_QUOTE" if actual_price is None
                    else "STALE_OR_INVALID_QUOTE"
                )
            elif symbol in failed_refresh_symbols:
                quote_info["prediction_status"] = "HISTORY_REFRESH_FAILED"
                quote_info["error"] = "Five-minute snapshot history refresh failed for this symbol."
            elif not history_ready or hist.empty:
                quote_info["prediction_status"] = (
                    "INSUFFICIENT_HISTORY" if history_seed_complete else "WARMING_HISTORY"
                )
            elif model_error:
                quote_info["prediction_status"] = "MODEL_UNAVAILABLE"
                quote_info["error"] = model_error
                model_failures += 1
            elif price_row is not None:
                try:
                    snapshot_values = {
                        **price_row,
                        "day_open": _safe_float(quote.get("day_open")),
                        "day_high": _safe_float(quote.get("day_high")),
                        "day_low": _safe_float(quote.get("day_low")),
                        "day_volume": _safe_float(quote.get("day_volume")),
                    }
                    current_feature_row = pd.DataFrame(
                        [snapshot_values],
                        index=pd.DatetimeIndex([minute], tz=IST, name="Datetime"),
                    )
                    feature_history = pd.concat([hist, current_feature_row]).sort_index()
                    feature_history = feature_history.loc[
                        ~feature_history.index.duplicated(keep="last")
                    ].tail(SNAPSHOT_HISTORY_LOOKBACK_BARS)
                    session_hist = feature_history.loc[
                        pd.Index(feature_history.index.date) == minute.date()
                    ]
                    latest_feature_slot = _as_ist_timestamp(feature_history.index.max())
                    feature_age_minutes = max(
                        0.0, (minute - latest_feature_slot).total_seconds() / 60.0
                    )
                    if len(session_hist) < MIN_SESSION_CANDLES_FOR_PREDICTION:
                        quote_info["prediction_status"] = "INSUFFICIENT_SESSION_HISTORY"
                    elif feature_age_minutes > MAX_HISTORY_CANDLE_AGE_MINUTES:
                        quote_info["prediction_status"] = "STALE_HISTORY"
                        quote_info["error"] = (
                            "Latest five-minute snapshot input is missing or older than "
                            f"{MAX_HISTORY_CANDLE_AGE_MINUTES} minutes at the decision slot."
                        )
                    else:
                        model_feature_row = self._model_api._latest_features(
                            feature_history,
                            market_df=market_history_before_minute,
                            fno=fno_context_by_symbol.get(symbol),
                            decision_timestamp=actual_ts,
                        )
                        feature_rows.append(model_feature_row)
                        feature_symbols.append(symbol)
                        current_rows.append(quote_info)
                except Exception as exc:
                    quote_info["prediction_status"] = "FEATURE_BUILD_ERROR"
                    quote_info["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                    model_failures += 1
            else:
                quote_info["prediction_status"] = "MISSING_CURRENT_QUOTE"
            if symbol not in feature_symbols:
                current_rows.append(quote_info)

        prediction_count = 0
        feature_quality_diagnostics: dict[str, Any] = {
            "feature_row_count": len(feature_rows),
            "mean_feature_coverage_pct": None,
            "p10_feature_coverage_pct": None,
            "feature_group_coverage_pct": {},
            "missing_count_by_feature": {},
        }
        inference_frame = pd.DataFrame(
            feature_rows, columns=self._model_api.FEATURES
        ) if feature_rows and self._model_api is not None else pd.DataFrame()
        if not inference_frame.empty:
            row_coverage = inference_frame.notna().mean(axis=1) * 100.0
            feature_quality_diagnostics.update({
                "mean_feature_coverage_pct": round(float(row_coverage.mean()), 3),
                "p10_feature_coverage_pct": round(float(np.quantile(row_coverage, 0.10)), 3),
                "feature_group_coverage_pct": {
                    "price_technical": round(float(
                        inference_frame[[name for name in inference_frame.columns if name in {
                            "ret_5m", "ret_15m", "ret_25m", "ret_50m", "ret_75m", "ret_150m",
                            "rsi_70m", "ma_25m_ratio", "ma_50m_ratio", "ma_100m_ratio",
                            "ma_250m_ratio", "volume_ratio_100m", "volume_z_100m",
                            "session_return", "day_high_ratio", "day_low_ratio", "tod_sin", "tod_cos",
                        }]].notna().mean(axis=1).mean() * 100.0
                    ), 3) if any(name in inference_frame.columns for name in (
                        "ret_5m", "ret_15m", "ret_25m", "ret_50m", "ret_75m", "ret_150m",
                        "rsi_70m", "ma_25m_ratio", "ma_50m_ratio", "ma_100m_ratio",
                        "ma_250m_ratio", "volume_ratio_100m", "volume_z_100m",
                        "session_return", "day_high_ratio", "day_low_ratio", "tod_sin", "tod_cos",
                    )) else None,
                    "market_context": round(float(
                        inference_frame[[name for name in inference_frame.columns if name in {
                            "market_ret_1", "market_ret_3", "market_ret_5", "market_ret_10",
                            "market_ret_15", "market_ret_30", "market_rsi14",
                            "relative_ret_1", "relative_ret_5", "relative_ret_15",
                        }]].notna().mean(axis=1).mean() * 100.0
                    ), 3) if any(name in inference_frame.columns for name in (
                        "market_ret_1", "market_ret_3", "market_ret_5", "market_ret_10",
                        "market_ret_15", "market_ret_30", "market_rsi14",
                        "relative_ret_1", "relative_ret_5", "relative_ret_15",
                    )) else None,
                    "fno_context": round(float(
                        inference_frame[[name for name in inference_frame.columns if name in {
                            "fno_pcr", "fno_oi_change", "fno_fut_return", "fno_fut_oi",
                            "fno_basis", "fno_available",
                        }]].notna().mean(axis=1).mean() * 100.0
                    ), 3) if any(name in inference_frame.columns for name in (
                        "fno_pcr", "fno_oi_change", "fno_fut_return", "fno_fut_oi",
                        "fno_basis", "fno_available",
                    )) else None,
                },
                "missing_count_by_feature": {
                    str(name): int(inference_frame[name].isna().sum())
                    for name in inference_frame.columns
                },
            })
        if feature_rows and not model_error and history_ready:
            try:
                x = inference_frame
                probs = np.clip(self._classifier.predict_proba(x)[:, 1], 0, 1)
                expected_returns = np.asarray(self._regressor.predict(x), dtype=float)
                record_by_symbol = {str(row["symbol"]): row for row in current_rows}
                for idx, symbol in enumerate(feature_symbols):
                    row = record_by_symbol[symbol]
                    probability = float(probs[idx])
                    expected_return = float(expected_returns[idx]) if np.isfinite(expected_returns[idx]) else None
                    readiness = float(x.iloc[idx].reindex(READINESS_FEATURES).notna().mean())
                    decision_ts = _as_ist_timestamp(
                        row.get("actual_observed_at_ist"),
                        fallback=minute,
                    )
                    raw_signal, proposed_signal = _signal(probability, readiness)
                    trained_for_symbol = (
                        symbol in self._trained_symbols
                        if self._trained_symbols else None
                    )
                    data_limited = not np.isfinite(readiness) or readiness < MIN_SIGNAL_READINESS
                    signal = (
                        proposed_signal
                        if not data_limited and trained_for_symbol is not False
                        else "HOLD"
                    )
                    if data_limited:
                        prediction_status = "PASS_DATA_LIMITED"
                    elif trained_for_symbol is False:
                        prediction_status = "PASS_UNTRAINED_SYMBOL"
                    else:
                        prediction_status = "PASS"
                    row.update({
                        "prediction_status": prediction_status,
                        "raw_model_signal": raw_signal,
                        "signal": signal,
                        "trained_for_symbol": trained_for_symbol,
                        "probability_up_30m": probability,
                        "expected_return_30m": expected_return,
                        "confidence": (
                            "HIGH" if abs(probability - 0.5) >= 0.25
                            else "MEDIUM" if abs(probability - 0.5) >= 0.12
                            else "LOW"
                        ),
                        "model_family": "logistic_regression+ridge",
                        "model_target": "target_30m",
                        "model_coverage_symbols": int(
                            self._model_status.get("training_symbol_count")
                            or len(self._trained_symbols)
                        ),
                        "feature_count": len(self._model_api.FEATURES),
                        "non_missing_feature_count": int(x.iloc[idx].notna().sum()),
                        "readiness_feature_count": len(READINESS_FEATURES),
                        "data_readiness_score": readiness if np.isfinite(readiness) else None,
                        "data_limited": data_limited,
                        "forecast_target_timestamp_utc": (
                            (decision_ts + pd.Timedelta(minutes=30)).tz_convert("UTC").isoformat()
                        ),
                        "forecast_key": (
                            f"{symbol}|{decision_ts.tz_convert('UTC').isoformat()}|30|"
                            f"{self._model_status.get('trained_until') or 'all-nse-pooled'}"
                        ),
                        "error": None,
                    })

                    base_model_version = self._production_model_version()
                    readiness_segment = (
                        "READINESS_UNKNOWN" if not np.isfinite(readiness)
                        else "READINESS_LOW" if data_limited
                        else "READINESS_READY"
                    )
                    model_version = f"{base_model_version}__{readiness_segment}"
                    row["base_model_version"] = base_model_version
                    row["readiness_segment"] = readiness_segment
                    row["model_version"] = model_version
                    row["probability_up_30m_raw"] = probability
                    row["expected_return_30m_raw"] = expected_return
                    row["signal_raw"] = signal
                    reference_price = _safe_float(row.get("actual_market_value"))
                    predicted_price = (
                        reference_price * (1.0 + expected_return)
                        if reference_price is not None and expected_return is not None
                        and np.isfinite(expected_return) else None
                    )
                    overlay = self._apply_calibration_overlay({
                        "horizon_min": 30,
                        "model_version": model_version,
                        "probability_up": probability,
                        "expected_return": expected_return,
                        "predicted_price": predicted_price,
                        "reference_close": reference_price,
                        "signal": signal,
                    })
                    row["probability_up_30m_calibrated"] = overlay.get("probability_up_calibrated")
                    row["expected_return_30m_calibrated"] = overlay.get("expected_return_calibrated")
                    row["predicted_price_30m_calibrated"] = overlay.get("predicted_price_calibrated")
                    row["calibrated_signal_advisory"] = overlay.get("signal_calibrated")
                    # A data-limited inference must remain HOLD even if an
                    # active probability calibrator would otherwise suggest BUY/SELL.
                    row["signal_calibrated"] = _primary_signal_with_readiness_guard(
                        row.get("prediction_status"),
                        row.get("data_limited"),
                        overlay.get("signal_calibrated"),
                    )
                    row["calibration_profile_id"] = overlay.get("calibration_profile_id")
                    row["calibration_stage"] = overlay.get("calibration_stage")
                    row["calibration_applied"] = bool(overlay.get("calibration_applied", False))
                    if row["calibration_applied"]:
                        row["probability_up_30m"] = overlay.get("probability_up", probability)
                        row["expected_return_30m"] = overlay.get("expected_return", expected_return)
                        row["signal"] = _primary_signal_with_readiness_guard(
                            row.get("prediction_status"),
                            row.get("data_limited"),
                            overlay.get("signal", signal),
                        )
                        calibrated_probability = _safe_float(row.get("probability_up_30m"))
                        if calibrated_probability is not None:
                            row["confidence"] = (
                                "HIGH" if abs(calibrated_probability - 0.5) >= 0.25
                                else "MEDIUM" if abs(calibrated_probability - 0.5) >= 0.12
                                else "LOW"
                            )
                    if readiness < MIN_SIGNAL_READINESS:
                        data_limited_count += 1
                    prediction_count += 1
            except Exception as exc:
                model_failures += len(feature_rows)
                for row in current_rows:
                    if row.get("prediction_status") in {"WARMING_HISTORY", "PASS", "PASS_DATA_LIMITED"}:
                        row.update({
                            "prediction_status": "MODEL_INFERENCE_ERROR",
                            "signal": "HOLD",
                            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                        })

        # Append the current slot only after inference so predictions never use
        # their own decision-slot observation as historical input.
        self._append_five_minute_snapshot(quote_frame, minute)

        with self._lock:
            for symbol, item in quote_actuals.items():
                if item["day_volume"] is not None:
                    self._last_day_volume[symbol] = (item["date"], float(item["day_volume"]))

        # Ensure exactly one row per universe symbol even when data or inference failed.
        current_rows_by_symbol = {str(row["symbol"]): row for row in current_rows}
        records = []
        for symbol in symbols:
            row = current_rows_by_symbol.get(symbol)
            if row is None:
                row = {
                    "record_type": "all_nse_minute_prediction_actual",
                    "schema_version": "all-nse-minute-prediction-v1",
                    "snapshot_minute_ist": minute.isoformat(),
                    "symbol": symbol,
                    "exchange": "NSE",
                    "actual_market_value": None,
                    "actual_price_source": "INDSTOCKS_BATCH_QUOTE",
                    "actual_observed_at_ist": None,
                    "actual_observed_at_utc": None,
                    "actual_timestamp_basis": "collector_observation_time_not_exchange_tick_timestamp",
                    "actual_quote_status": "MISSING_QUOTE",
                    "actual_quote_age_seconds": None,
                    "day_volume": None,
                    "minute_volume_delta": None,
                    "current_minute_ohlc_complete": False,
                    "current_minute_bar_note": "No usable current quote row was available.",
                    "prediction_status": "MISSING_CURRENT_QUOTE",
                    "raw_model_signal": None,
                    "signal": "HOLD",
                    "probability_up_30m": None,
                    "expected_return_30m": None,
                    "confidence": None,
                    "model_family": "logistic_regression+ridge",
                    "model_target": "target_30m",
                    "model_coverage_symbols": None,
                    "trained_for_symbol": None,
                    "feature_count": len(READINESS_FEATURES) + 5,
                    "non_missing_feature_count": 0,
                    "readiness_feature_count": len(READINESS_FEATURES),
                    "data_readiness_score": 0.0,
                    "data_limited": True,
                    "forecast_target_timestamp_utc": None,
                    "forecast_key": None,
                    "error": read_error or "No current quote row",
                }
            records.append(row)

        prediction_status_counts: dict[str, int] = {}
        for row in records:
            status_key = str(row.get("prediction_status") or "UNKNOWN")
            prediction_status_counts[status_key] = prediction_status_counts.get(status_key, 0) + 1

        prediction_count = sum(
            1 for row in records
            if row.get("probability_up_30m") is not None
            and row.get("prediction_status") in {
                "PASS", "PASS_DATA_LIMITED", "PASS_UNTRAINED_SYMBOL"
            }
        )
        missing_quotes = sum(
            1 for row in records
            if row.get("actual_market_value") is None
            or row.get("actual_quote_status") != "PASS"
            or row.get("prediction_status") in {
                "MISSING_CURRENT_QUOTE", "STALE_OR_INVALID_QUOTE"
            }
        )
        data_limited_count = sum(
            1 for row in records
            if row.get("prediction_status") == "PASS_DATA_LIMITED"
        )
        untrained_symbol_count = sum(
            1 for row in records if row.get("trained_for_symbol") is False
        )
        trained_symbol_manifest_available = bool(self._trained_symbols)
        expected_count = max(1, len(symbols))
        seed_batch_count = max(1, (len(symbols) + HISTORY_BATCH_SIZE - 1) // HISTORY_BATCH_SIZE)
        refresh_batch_count = int(history_refresh.get("batch_count") or 0)
        refresh_failed_batches = int(history_refresh.get("failed_batches") or 0)
        prediction_coverage = prediction_count / expected_count
        missing_quote_fraction = missing_quotes / expected_count
        data_limited_fraction = data_limited_count / expected_count
        untrained_fraction = untrained_symbol_count / expected_count
        model_failure_fraction = model_failures / expected_count
        seed_coverage_fraction = seed_coverage_symbols / expected_count
        seed_failure_fraction = (
            seed_failed_batches + seed_deferred_batches
        ) / seed_batch_count
        refresh_failure_fraction = (
            refresh_failed_batches / refresh_batch_count
            if refresh_batch_count
            else 0.0 if history_refresh.get("status") in {
                "SCHEDULED_ASYNC", "IN_PROGRESS", "WARMING_SEED", "SNAPSHOT_HISTORY_ONLY"
            } else 1.0
        )
        model_artifact_verified = bool(self._model_status.get("manifest_sha256_match"))
        quality_gate_pass = bool(
            len(records) == len(symbols)
            and prediction_coverage >= MIN_PREDICTION_COVERAGE
            and missing_quote_fraction <= MAX_MISSING_QUOTE_FRACTION
            and data_limited_fraction <= MAX_DATA_LIMITED_FRACTION
            and untrained_fraction <= MAX_UNTRAINED_SYMBOL_FRACTION
            and model_failure_fraction <= MAX_MODEL_FAILURE_FRACTION
            and model_artifact_verified
            and trained_symbol_manifest_available
            and history_ready
            and history_seed_complete
            and seed_coverage_fraction >= MIN_HISTORY_SEED_COVERAGE
            and seed_failure_fraction <= MAX_HISTORY_FAILURE_FRACTION
            and refresh_failure_fraction <= MAX_HISTORY_FAILURE_FRACTION
        )
        manifest = {
            "audit_name": "all-nse-minute-prediction-actual-capture",
            "schema_version": "all-nse-minute-prediction-v1",
            "capture_interval_minutes": 5,
            "cycle_minute_ist": minute.isoformat(),
            "model_version": self._production_model_version(),
            "calibration_versioning": "per-row readiness segment; READINESS_LOW primary action remains HOLD",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_symbol_count": len(symbols),
            "row_count": len(records),
            "actual_quote_count": len(records) - missing_quotes,
            "missing_quote_count": missing_quotes,
            "prediction_count": prediction_count,
            "prediction_coverage_pct": round(100 * prediction_coverage, 3),
            "prediction_status_counts": prediction_status_counts,
            "data_limited_count": data_limited_count,
            "data_limited_fraction_pct": round(100 * data_limited_fraction, 3),
            "feature_quality_diagnostics": feature_quality_diagnostics,
            "model_failure_count": model_failures,
            "untrained_symbol_count": untrained_symbol_count,
            "trained_symbol_manifest_available": trained_symbol_manifest_available,
            "model_artifact_verified": model_artifact_verified,
            "model_target": "target_30m",
            "model_status": seed_status,
            "history_ready": history_ready,
            "history_seed_complete": history_seed_complete,
            "history_seed_universe_source": seed_universe_source,
            "history_snapshot_source": "PERSISTED_FIVE_MINUTE_SNAPSHOTS",
            "history_seed_coverage_symbols": seed_coverage_symbols,
            "history_seed_coverage_fraction": round(seed_coverage_fraction, 6),
            "history_seed_failed_batches": seed_failed_batches,
            "history_seed_deferred_batches": seed_deferred_batches,
            "history_seed_failure_fraction": round(seed_failure_fraction, 6),
            "history_refresh_status": history_refresh.get("status"),
            "history_refresh_mode": "SNAPSHOT_ONLY",
            "history_refresh_batch_count": refresh_batch_count,
            "history_refresh_failed_batches": refresh_failed_batches,
            "history_refresh_failed_symbols": len(failed_refresh_symbols),
            "history_refresh_failure_fraction": round(refresh_failure_fraction, 6),
            "history_refresh_elapsed_seconds": history_refresh.get("elapsed_seconds"),
            "min_session_candles_for_prediction": MIN_SESSION_CANDLES_FOR_PREDICTION,
            "snapshot_history_cadence_minutes": SNAPSHOT_CADENCE_MINUTES,
            "snapshot_history_slots_loaded": self._history_snapshot_slot_count,
            "model_coverage_symbols": (
                self._model_status.get("training_symbol_count") or len(self._trained_symbols)
            ),
            "nifty_context_status": "NOT_USED_SNAPSHOT_ONLY",
            "nifty_context_source": "NONE_SNAPSHOT_ONLY",
            "fno_context_status": fno_context_status,
            "fno_context_source_max_date": fno_context_source_max_date,
            "fno_context_symbol_count": len(fno_context_by_symbol),
            "nifty_context_error": nifty_error,
            "quote_source_snapshot": str(local_quote_path),
            "quote_snapshot_drive_path": capture_manifest.get("snapshot_drive_path"),
            "quote_snapshot_cycle_minute_ist": minute.isoformat(),
            "quote_snapshot_capture_status": capture_status,
            "quote_snapshot_expected_symbol_count": int(capture_manifest.get("expected_symbol_count") or 0),
            "quote_snapshot_rows_validated": True,
            "quote_snapshot_read_error": read_error,
            "current_minute_ohlc_complete": False,
            "quality_gate_thresholds": {
                "min_prediction_coverage": MIN_PREDICTION_COVERAGE,
                "max_missing_quote_fraction": MAX_MISSING_QUOTE_FRACTION,
                "max_data_limited_fraction": MAX_DATA_LIMITED_FRACTION,
                "max_untrained_symbol_fraction": MAX_UNTRAINED_SYMBOL_FRACTION,
                "max_model_failure_fraction": MAX_MODEL_FAILURE_FRACTION,
                "min_history_seed_coverage": MIN_HISTORY_SEED_COVERAGE,
                "max_history_failure_fraction": MAX_HISTORY_FAILURE_FRACTION,
            },
            "quality_gate_pass": quality_gate_pass,
            "status": "PARTIAL",
        }
        persisted = self._persist(records, manifest, minute)
        persistence_confirmed = str(persisted.get("status") or "").upper() in {"OK", "PARTIAL"}
        with self._lock:
            # A quality-gated PARTIAL result is still a durable snapshot and must
            # not be reprocessed. A persistence error is different: leave the
            # minute retryable so a transient Drive failure can recover in-place.
            if persistence_confirmed:
                self._last_minute = minute_key
            self._last_health = {
                **persisted,
                "status": persisted.get("status"),
                "history_ready": self._seed_ready,
                "history_seed_complete": history_seed_complete,
                "universe_symbols": len(self._symbols) or len(symbols),
                "history_symbols": seed_coverage_symbols,
                "last_capture_minute_ist": minute.isoformat(),
                "last_prediction_count": prediction_count,
                "last_persistence_status": persisted.get("status"),
                "data_limited_count": data_limited_count,
                "missing_quote_count": missing_quotes,
            }
        self._log(
            "all_nse_minute_prediction_actual_capture",
            cycle_minute_ist=minute.isoformat(),
            expected_symbol_count=len(symbols),
            prediction_count=prediction_count,
            missing_quote_count=missing_quotes,
            data_limited_count=data_limited_count,
            prediction_status_counts=prediction_status_counts,
            feature_quality_diagnostics=feature_quality_diagnostics,
            history_ready=history_ready,
            history_seed_complete=history_seed_complete,
            history_seed_coverage_symbols=seed_coverage_symbols,
            history_seed_failed_batches=seed_failed_batches,
            history_seed_deferred_batches=seed_deferred_batches,
            fno_context_status=fno_context_status,
            fno_context_source_max_date=fno_context_source_max_date,
            fno_context_symbol_count=len(fno_context_by_symbol),
            status=persisted.get("status"),
            remote_path=persisted.get("remote_path"),
            error=persisted.get("error"),
        )
        return {
            **persisted,
            "cycle_minute_ist": minute.isoformat(),
            "expected_symbol_count": len(symbols),
            "prediction_count": prediction_count,
            "missing_quote_count": missing_quotes,
            "data_limited_count": data_limited_count,
            "prediction_status_counts": prediction_status_counts,
            "quality_gate_pass": manifest["quality_gate_pass"],
            "history_ready": history_ready,
            "history_seed_complete": history_seed_complete,
            "history_status": seed_status,
            "fno_context_status": fno_context_status,
            "fno_context_source_max_date": fno_context_source_max_date,
            "fno_context_symbol_count": len(fno_context_by_symbol),
        }
