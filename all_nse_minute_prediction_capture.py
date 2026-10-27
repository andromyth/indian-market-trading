"""All-NSE minute-level prediction plus contemporaneous actual-price persistence.

The existing full-universe collector supplies real-time LTP snapshots. This module
adds a pooled 30-minute prediction for every NSE symbol in that snapshot and
persists a single auditable row per symbol. It does not submit or place orders.

Historical OHLCV is seeded in batches of at most five instruments per request.
For the current, still-forming minute, the actual LTP and cumulative-volume delta
are real; current-minute open/high/low are deliberately left missing because a
single REST quote is not an exchange OHLC candle. Low-readiness predictions are
saved for analysis but their actionable signal is forced to HOLD.
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

import calibration_overlay as calibration_overlay_module

IST = ZoneInfo("Asia/Kolkata")
REMOTE_ROOT = "production_data/all_nse_minute_predictions"
HISTORY_LOOKBACK_BARS = 120
HISTORY_BATCH_SIZE = 5
HISTORY_REQUEST_PAUSE_SECONDS = 0.22
HISTORY_REFRESH_REQUEST_SPACING_SECONDS = 0.24
HISTORY_REFRESH_WINDOW_MINUTES = 90
HISTORY_REFRESH_WORKERS = 8
MIN_SESSION_CANDLES_FOR_PREDICTION = 16
MAX_HISTORY_CANDLE_AGE_MINUTES = 5
MIN_SIGNAL_READINESS = 0.70
MAX_QUOTE_AGE_SECONDS = 120
MIN_PREDICTION_COVERAGE = 0.95
MAX_MISSING_QUOTE_FRACTION = 0.02
MAX_DATA_LIMITED_FRACTION = 0.10
MAX_UNTRAINED_SYMBOL_FRACTION = 0.10
MAX_MODEL_FAILURE_FRACTION = 0.01
MIN_HISTORY_SEED_COVERAGE = 0.90
MAX_HISTORY_FAILURE_FRACTION = 0.05


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
        "actual_quote_age_seconds": max(
            0.0, (observed_at - snapshot_minute).total_seconds()
        ),
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
        "feature_count": 34,
        "non_missing_feature_count": 0,
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


class AllNSEMinutePredictor:
    """Rate-limited history cache and one-row-per-symbol minute inference engine."""

    def __init__(self, worker_root: str | Path):
        self.worker_root = Path(worker_root)
        self.root = self.worker_root / "all_nse_minute_predictions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._seed_started = False
        self._seed_ready = False
        # Readiness is progressive; completion is tracked separately so the
        # worker can forecast eligible symbols while remaining history is seeded.
        self._seed_complete = False
        self._seed_status = "NOT_STARTED"
        self._seed_error: str | None = None
        self._seed_universe_source = "NOT_SET"
        self._seed_coverage = 0
        self._seed_errors = 0
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
        self._calibration = None
        self._calibration_init_error_logged = False
        self._history_request_lock = threading.Lock()
        self._next_history_request_at = 0.0
        self._history_refresh_run_lock = threading.Lock()
        self._history_refresh_scheduled = False
        self._last_history_refresh_minute: str | None = None
        self._last_history_refresh_summary: dict[str, Any] = {}
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
        """Start the history bootstrap once; no credentials are read at import time."""
        with self._lock:
            if self._seed_started:
                return
            self._seed_started = True
            self._seed_status = "WARMING_HISTORY"
        threading.Thread(
            target=self._seed_history,
            name="all-nse-minute-history-seed",
            daemon=True,
        ).start()

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
                "history_seed_error": self._seed_error,
                "history_seed_universe_source": self._seed_universe_source,
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
        """Fetch genuine NIFTY 50 one-minute candles strictly before a decision time."""
        columns = ["Open", "High", "Low", "Close", "Volume"]
        empty = pd.DataFrame(
            columns=columns,
            index=pd.DatetimeIndex([], tz=IST, name="Datetime"),
        )
        decision_minute = _as_ist_timestamp(minute).floor("min")
        try:
            from data_fetcher import _yahoo_chart_history

            frame = _yahoo_chart_history("^NSEI", "5d", "1m")
            if frame is None or frame.empty:
                return empty
            frame = frame.copy()
            aliases = {str(column).strip().lower(): column for column in frame.columns}
            rename = {}
            for required in columns:
                source = aliases.get(required.lower())
                if source is not None and source != required:
                    rename[source] = required
            frame = frame.rename(columns=rename)
            if "Close" not in frame.columns:
                return empty
            for required in columns:
                if required not in frame.columns:
                    frame[required] = np.nan
                frame[required] = pd.to_numeric(frame[required], errors="coerce")
            # Yahoo timestamps are epoch-derived and may arrive UTC-aware or
            # naive; normalize every row to an explicit IST DatetimeIndex.
            index = pd.DatetimeIndex(
                pd.to_datetime(frame.index, errors="coerce", utc=True)
            ).tz_convert(IST)
            frame.index = index
            frame.index.name = "Datetime"
            frame = frame.loc[~frame.index.isna()]
            frame = frame.loc[
                (frame.index < decision_minute)
                & frame["Close"].notna()
                & frame["Close"].gt(0)
            ].sort_index()
            frame = frame.loc[~frame.index.duplicated(keep="last")]
            return frame[columns].tail(HISTORY_LOOKBACK_BARS)
        except Exception:
            # Benchmark context is optional; keep stock persistence working and
            # let the existing feature-readiness gate force low-readiness HOLD.
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
                entries = [(symbol, codes[symbol]) for symbol in symbols if symbol in codes]
                batches = _history_seed_batches(entries, nifty_code, HISTORY_BATCH_SIZE)

                histories: dict[str, pd.DataFrame] = {}
                batch_errors = 0
                requests_made = 0
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
                    if historical_coverage >= coverage_target:
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
                    self._seed_error = None if batch_errors == 0 else (
                        f"{batch_errors} historical batches did not return a success payload"
                    )
                    self._seed_ready = self._seed_coverage > 0
                    self._seed_complete = True
                    self._seed_status = (
                        "READY"
                        if self._seed_ready and batch_errors == 0 and self._seed_coverage == len(symbols)
                        else "READY_PARTIAL" if self._seed_ready
                        else "HISTORY_SEED_EMPTY"
                    )
                    self._pending_quote_rows.clear()
                if not self._seed_ready:
                    raise RuntimeError("Historical warm-up returned no usable OHLCV bars")
                self._log(
                    "all_nse_history_seed_completed",
                    universe_count=len(symbols),
                    mapped_symbol_count=len(self._codes),
                    history_symbol_count=self._seed_coverage,
                    history_universe_source=self._seed_universe_source,
                    historical_batch_count=requests_made,
                    failed_batches=batch_errors,
                    nifty_history_rows=len(self._market_history),
                )
                return
            except Exception as exc:
                with self._lock:
                    self._seed_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                    self._seed_ready = bool(self._history)
                    self._seed_complete = False
                    self._seed_status = (
                        "READY_PARTIAL" if self._seed_ready else "WARMING_HISTORY_ERROR"
                    )
                self._log("all_nse_history_seed_failed", error=self._seed_error)
                time.sleep(120)

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
        if not status.get("manifest_sha256_match"):
            raise RuntimeError(
                "Pooled ALL-NSE intraday model is unavailable or its hashes do not match the manifest"
            )
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
        """Refresh history in the background so the five-minute tick can persist promptly.

        The full-universe quote capture already takes roughly two minutes. Running
        another 500+ provider requests synchronously after capture can push the
        whole Scheduler request past its 300-second deadline. This refresh updates
        the cache for the next slot; the current forecast uses only cached candles
        strictly before its decision minute and rejects stale history.
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
            # Initial seeding is already fetching a full history range. Do not
            # compete with it for provider quota; let the seeder finish first.
            if not self._seed_ready or not self._seed_complete or not self._codes:
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
            start_ms, end_ms = int(window_start.timestamp() * 1000), int(minute.timestamp() * 1000)
            entries = sorted((str(s), str(c)) for s, c in codes.items() if str(c).strip())
            if nifty_code:
                entries.append(("__NIFTY50__", str(nifty_code)))
            batches = [entries[i:i + HISTORY_BATCH_SIZE] for i in range(0, len(entries), HISTORY_BATCH_SIZE)]

            def rate_limit_start() -> None:
                self._reserve_history_request_slot()

            def fetch_batch(batch):
                last_error = None
                for attempt in range(3):
                    try:
                        rate_limit_start()
                        body = _request("GET", "/market/historical/1minute", params={
                            "scrip-codes": ",".join(code for _, code in batch),
                            "start_time": start_ms, "end_time": end_ms,
                        }).json()
                        successful = isinstance(body, dict) and (
                            body.get("success") is True
                            or str(body.get("status") or "").strip().lower() == "success"
                        )
                        if not successful:
                            raise RuntimeError("historical refresh response did not report success")
                        data = body.get("data") or {}
                        result = {}
                        for symbol, code in batch:
                            frame = self._parse_candles((data.get(code) or {}).get("candles") or [])
                            if not frame.empty:
                                frame = frame.loc[
                                    (pd.Index(frame.index.date) == minute.date())
                                    & (frame.index < minute)
                                ].tail(HISTORY_REFRESH_WINDOW_MINUTES)
                                if not frame.empty:
                                    result[symbol] = frame
                        return result, None
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {str(exc)[:180]}"
                        if attempt < 2:
                            time.sleep(0.25 * (attempt + 1))
                return {}, last_error or "historical refresh failed"

            started = time.monotonic()
            updated, failed_symbols, errors, candle_rows = {}, [], [], 0
            with ThreadPoolExecutor(max_workers=HISTORY_REFRESH_WORKERS) as pool:
                futures = {pool.submit(fetch_batch, batch): batch for batch in batches}
                for future in as_completed(futures):
                    result, error = future.result()
                    if error:
                        errors.append(error)
                        failed_symbols.extend(s for s, _ in futures[future] if s != "__NIFTY50__")
                    for symbol, frame in result.items():
                        updated[symbol] = frame
                        candle_rows += len(frame)

            nifty_history_source = "INDSTOCKS_HISTORICAL_1M" if "__NIFTY50__" in updated else None
            if "__NIFTY50__" not in updated or updated["__NIFTY50__"].empty:
                # Provider index IDs are not REST scrip-codes; use the same genuine
                # OHLCV fallback as bootstrap, excluding current/future minutes.
                fallback = self._fetch_yahoo_nifty_history(minute)
                if not fallback.empty:
                    fallback = fallback.loc[
                        (pd.Index(fallback.index.date) == minute.date())
                        & (fallback.index < minute)
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
                    "status": "OK" if not errors else "PARTIAL",
                    "minute": minute_key, "batch_count": len(batches),
                    "failed_batches": len(errors), "failed_symbols": sorted(set(failed_symbols)),
                    "updated_symbols": len([s for s in updated if s != "__NIFTY50__"]),
                    "current_session_candle_rows": candle_rows,
                    "nifty_history_source": nifty_history_source,
                    "nifty_history_rows": int(len(updated.get("__NIFTY50__", pd.DataFrame()))),
                    "window_start_ist": window_start.isoformat(),
                    "window_end_ist": minute.isoformat(),
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
        self.start()
        minute = _as_ist_timestamp(
            capture_manifest.get("cycle_minute_ist"),
            fallback=pd.Timestamp.now(tz=IST).floor("min"),
        ).floor("min")
        minute_key = minute.isoformat()
        if self._last_minute == minute_key:
            return {"status": "DUPLICATE_MINUTE_SKIPPED", "cycle_minute_ist": minute_key}

        # Do not put hundreds of history API requests on the prediction
        # critical path. Schedule a fresh candle cache for the next slot.
        # Capture is timestamp-aligned; prediction below filters every input
        # candle to strictly before this minute and rejects data older than the
        # explicit freshness threshold.
        refresh_target = max(minute, pd.Timestamp.now(tz=IST).floor("min"))
        history_refresh = self._schedule_current_session_history_refresh(refresh_target)
        failed_refresh_symbols = set(history_refresh.get("failed_symbols") or [])
        source_root = Path(worker_root or self.worker_root)
        snapshot_name = f"snapshot_{minute:%Y%m%d_%H%M}.parquet"
        local_quote_path = source_root / "all_universe_minute_snapshots" / snapshot_name
        try:
            quote_frame = pd.read_parquet(local_quote_path) if local_quote_path.is_file() else pd.DataFrame()
        except Exception as exc:
            quote_frame = pd.DataFrame()
            read_error = f"{type(exc).__name__}: {str(exc)[:200]}"
        else:
            read_error = None
        if not quote_frame.empty and "symbol" in quote_frame:
            quote_frame["symbol"] = quote_frame["symbol"].astype(str).str.strip().str.upper()
            quote_frame = quote_frame.drop_duplicates("symbol", keep="last")
            quote_by_symbol = {
                str(row["symbol"]): row
                for row in quote_frame.to_dict(orient="records")
            }
        else:
            quote_by_symbol = {}

        with self._lock:
            symbols = list(self._symbols) or sorted(quote_by_symbol)
            history_ready = self._seed_ready
            history_seed_complete = self._seed_complete
            seed_status = self._seed_status
            seed_coverage_symbols = self._seed_coverage
            seed_failed_batches = self._seed_errors
            seed_universe_source = self._seed_universe_source
            histories = {symbol: frame.copy() for symbol, frame in self._history.items()}
            market_history = self._market_history.copy()
        market_history_before_minute = self._market_history_before(
            market_history,
            minute,
        )
        # Capture every symbol in the official universe even if quote data is missing.
        for symbol in sorted(set(quote_by_symbol) - set(symbols)):
            symbols.append(symbol)
        symbols = sorted(set(symbols))

        nifty_price, nifty_observed_at, nifty_error = self._fetch_current_nifty_quote()
        try:
            if history_ready:
                self._ensure_model()
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {str(exc)[:240]}"
        else:
            model_error = None

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
            history_age_minutes = None
            if not hist.empty and isinstance(hist.index, pd.DatetimeIndex):
                latest_history_at = _as_ist_timestamp(hist.index.max())
                history_age_minutes = max(
                    0.0, (minute - latest_history_at).total_seconds() / 60.0
                )
            quote_info, price_row, raw_day_volume = _current_minute_row(
                symbol=symbol,
                snapshot_minute=minute,
                quote=quote,
                prior_history=hist,
                previous_day_volume=(
                    self._last_day_volume.get(symbol, (date.min, 0.0))[1]
                    if symbol in self._last_day_volume
                    and self._last_day_volume[symbol][0] == minute.date()
                    else None
                ),
                previous_volume_date=(
                    self._last_day_volume.get(symbol, (date.min, 0.0))[0]
                    if symbol in self._last_day_volume else None
                ),
                snapshot_date=minute.date(),
            )
            actual_price = quote_info.get("actual_market_value")
            actual_ts = _as_ist_timestamp(
                quote_info.get("actual_observed_at_ist"),
                fallback=minute,
            )
            quote_age = quote_info.get("actual_quote_age_seconds")
            quote_good = (
                actual_price is not None
                and str(quote_info.get("actual_quote_status") or "").upper() == "PASS"
                and quote_age is not None
                and quote_age <= MAX_QUOTE_AGE_SECONDS
            )
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
                quote_info["error"] = "Current-session one-minute candle refresh failed for this symbol."
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
                    session_hist = hist.loc[pd.Index(hist.index.date) == minute.date()] if not hist.empty else hist
                    if len(session_hist) < MIN_SESSION_CANDLES_FOR_PREDICTION:
                        quote_info["prediction_status"] = "INSUFFICIENT_SESSION_HISTORY"
                    elif hist.empty:
                        quote_info["prediction_status"] = "INSUFFICIENT_HISTORY"
                    elif history_age_minutes is None or history_age_minutes > MAX_HISTORY_CANDLE_AGE_MINUTES:
                        quote_info["prediction_status"] = "STALE_HISTORY"
                        quote_info["error"] = (
                            "Latest completed stock candle is missing or older than "
                            f"{MAX_HISTORY_CANDLE_AGE_MINUTES} minutes at the decision slot."
                        )
                    else:
                        model_feature_row = self._model_api._latest_features(
                            hist,
                            market_df=market_history_before_minute,
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
        if feature_rows and not model_error and history_ready:
            try:
                x = pd.DataFrame(feature_rows, columns=self._model_api.FEATURES)
                probs = np.clip(self._classifier.predict_proba(x)[:, 1], 0, 1)
                expected_returns = np.asarray(self._regressor.predict(x), dtype=float)
                record_by_symbol = {str(row["symbol"]): row for row in current_rows}
                for idx, symbol in enumerate(feature_symbols):
                    row = record_by_symbol[symbol]
                    probability = float(probs[idx])
                    expected_return = float(expected_returns[idx]) if np.isfinite(expected_returns[idx]) else None
                    readiness = float(x.iloc[idx].notna().mean())
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

        # Update the in-memory minute bar cache only after inference used the
        # preceding bars. The current LTP is real; missing OHLC fields stay NaN.
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
                    "feature_count": 34,
                    "non_missing_feature_count": 0,
                    "data_readiness_score": 0.0,
                    "data_limited": True,
                    "forecast_target_timestamp_utc": None,
                    "forecast_key": None,
                    "error": read_error or "No current quote row",
                }
            records.append(row)

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
        seed_failure_fraction = seed_failed_batches / seed_batch_count
        refresh_failure_fraction = (
            refresh_failed_batches / refresh_batch_count
            if refresh_batch_count
            else 0.0 if history_refresh.get("status") in {
                "SCHEDULED_ASYNC", "IN_PROGRESS", "WARMING_SEED"
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
            "data_limited_count": data_limited_count,
            "model_failure_count": model_failures,
            "untrained_symbol_count": untrained_symbol_count,
            "trained_symbol_manifest_available": trained_symbol_manifest_available,
            "model_artifact_verified": model_artifact_verified,
            "model_target": "target_30m",
            "model_status": seed_status,
            "history_ready": history_ready,
            "history_seed_complete": history_seed_complete,
            "history_seed_universe_source": seed_universe_source,
            "history_seed_coverage_symbols": seed_coverage_symbols,
            "history_seed_coverage_fraction": round(seed_coverage_fraction, 6),
            "history_seed_failed_batches": seed_failed_batches,
            "history_seed_failure_fraction": round(seed_failure_fraction, 6),
            "history_refresh_status": history_refresh.get("status"),
            "history_refresh_mode": "BACKGROUND_NONBLOCKING",
            "history_refresh_batch_count": refresh_batch_count,
            "history_refresh_failed_batches": refresh_failed_batches,
            "history_refresh_failed_symbols": len(failed_refresh_symbols),
            "history_refresh_failure_fraction": round(refresh_failure_fraction, 6),
            "history_refresh_elapsed_seconds": history_refresh.get("elapsed_seconds"),
            "min_session_candles_for_prediction": MIN_SESSION_CANDLES_FOR_PREDICTION,
            "model_coverage_symbols": (
                self._model_status.get("training_symbol_count") or len(self._trained_symbols)
            ),
            "nifty_context_status": (
                "HISTORICAL_ONE_MINUTE_CANDLES"
                if not market_history_before_minute.empty else "HISTORY_OR_MISSING"
            ),
            "nifty_context_source": (
                "YAHOO_FINANCE_1M_FALLBACK"
                if self._nifty_code is None and not market_history_before_minute.empty
                else "INDSTOCKS_REST" if self._nifty_code
                else "MISSING"
            ),
            "nifty_context_error": nifty_error,
            "quote_source_snapshot": str(local_quote_path),
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
            history_ready=history_ready,
            history_seed_complete=history_seed_complete,
            history_seed_coverage_symbols=seed_coverage_symbols,
            history_seed_failed_batches=seed_failed_batches,
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
            "quality_gate_pass": manifest["quality_gate_pass"],
            "history_ready": history_ready,
            "history_seed_complete": history_seed_complete,
            "history_status": seed_status,
        }
