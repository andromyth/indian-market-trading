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

IST = ZoneInfo("Asia/Kolkata")
REMOTE_ROOT = "production_data/all_nse_minute_predictions"
HISTORY_LOOKBACK_BARS = 120
HISTORY_BATCH_SIZE = 5
HISTORY_REQUEST_PAUSE_SECONDS = 0.22
MIN_SIGNAL_READINESS = 0.70
MAX_QUOTE_AGE_SECONDS = 120


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
        "actual_quote_age_seconds": max(
            0.0, (pd.Timestamp.now(tz=IST) - observed_at).total_seconds()
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


class AllNSEMinutePredictor:
    """Rate-limited history cache and one-row-per-symbol minute inference engine."""

    def __init__(self, worker_root: str | Path):
        self.worker_root = Path(worker_root)
        self.root = self.worker_root / "all_nse_minute_predictions"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._seed_started = False
        self._seed_ready = False
        self._seed_status = "NOT_STARTED"
        self._seed_error: str | None = None
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
        self._market_history = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
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
                "universe_symbols": len(self._symbols),
                "history_symbols": len(self._history),
                "history_seed_coverage_symbols": self._seed_coverage,
                "history_seed_failed_batches": self._seed_errors,
                "history_seed_error": self._seed_error,
                "trained_symbol_count": len(self._trained_symbols),
            })

    @staticmethod
    def _nifty_instrument_code() -> str | None:
        """Resolve the REST quote/history token; never reuse a WebSocket token."""
        from indstocks_client import _request

        response = _request("GET", "/market/instruments", params={"source": "index"})
        index_master = pd.read_csv(io.BytesIO(response.content), dtype=str)
        if index_master.empty or len(index_master.columns) < 3:
            return None
        # The source=index schema is positional: EXCH, SEGMENT(index name), SECURITY_ID.
        index_master = index_master.iloc[:, :3].copy()
        index_master.columns = ["EXCH", "INDEX_NAME", "SECURITY_ID"]
        index_master["EXCH"] = index_master["EXCH"].astype(str).str.strip().str.upper()
        index_master["INDEX_NAME"] = index_master["INDEX_NAME"].astype(str).str.strip().str.upper()
        index_master["SECURITY_ID"] = index_master["SECURITY_ID"].astype(str).str.strip()
        rows = index_master.loc[
            index_master["EXCH"].eq("NSE")
            & index_master["INDEX_NAME"].str.contains("NIFTY 50", regex=False)
            & index_master["SECURITY_ID"].str.fullmatch(r"\d+")
        ]
        if rows.empty:
            return None
        return f"NSE_{rows.iloc[0]['SECURITY_ID']}"

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

    def _seed_history(self) -> None:
        """Fetch one recent trading session plus the current session in batches of five."""
        while not self._seed_ready:
            try:
                from all_universe_minute_capture import _instrument_map
                from delivery_training_universe import official_nse_symbols
                from indstocks_client import _request, configured

                if not configured():
                    raise RuntimeError("INDstocks credentials are not configured")
                symbols = sorted({
                    str(s).strip().upper().replace(".NS", "")
                    for s in official_nse_symbols(force_refresh=False)
                    if str(s).strip()
                })
                codes = _instrument_map()
                nifty_code = self._nifty_instrument_code()
                with self._lock:
                    self._symbols = symbols
                    self._codes = {s: codes[s] for s in symbols if s in codes}
                    self._nifty_code = nifty_code

                now = pd.Timestamp.now(tz=IST)
                previous_day = self._previous_trading_date(now.date())
                start = pd.Timestamp(datetime.combine(previous_day, wall_time(9, 15), tzinfo=IST))
                end = now
                start_ms = int(start.timestamp() * 1000)
                end_ms = int(end.timestamp() * 1000)
                entries = [(symbol, codes[symbol]) for symbol in symbols if symbol in codes]
                if nifty_code:
                    entries.append(("__NIFTY50__", nifty_code))

                histories: dict[str, pd.DataFrame] = {}
                batch_errors = 0
                requests_made = 0
                for offset in range(0, len(entries), HISTORY_BATCH_SIZE):
                    batch = entries[offset:offset + HISTORY_BATCH_SIZE]
                    code_list = [code for _, code in batch]
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
                        batch_errors += 1
                    else:
                        data = body.get("data") or {}
                        for symbol, code in batch:
                            instrument = data.get(code) or {}
                            frame = self._parse_candles(instrument.get("candles") or [])
                            if not frame.empty:
                                # Do not seed a still-forming candle; the live quote is
                                # appended as a separately timestamped partial row.
                                frame = frame.loc[frame.index < now.floor("min")]
                                histories[symbol] = frame.tail(HISTORY_LOOKBACK_BARS)
                    if offset + HISTORY_BATCH_SIZE < len(entries):
                        time.sleep(HISTORY_REQUEST_PAUSE_SECONDS)

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
                    self._history = {
                        symbol: frame.tail(HISTORY_LOOKBACK_BARS)
                        for symbol, frame in histories.items()
                        if not frame.empty
                    }
                    self._market_history = market_base.tail(HISTORY_LOOKBACK_BARS)
                    self._seed_coverage = sum(
                        1 for symbol in symbols
                        if symbol in self._history and not self._history[symbol].empty
                    )
                    self._seed_errors = batch_errors
                    self._seed_error = None if batch_errors == 0 else (
                        f"{batch_errors} historical batches did not return a success payload"
                    )
                    self._seed_ready = bool(self._history)
                    self._seed_status = (
                        "READY" if self._seed_ready and batch_errors == 0
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
                    historical_batch_count=requests_made,
                    failed_batches=batch_errors,
                    nifty_history_rows=len(self._market_history),
                )
                return
            except Exception as exc:
                with self._lock:
                    self._seed_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                    self._seed_status = "WARMING_HISTORY_ERROR"
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
        if not self._nifty_code:
            return None, None, "NIFTY 50 index token was not resolved"
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
            if price is None or price <= 0:
                raise RuntimeError("INDstocks NIFTY quote contained no live_price")
            return price, observed, None
        except Exception as exc:
            return None, None, f"{type(exc).__name__}: {str(exc)[:160]}"

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
                "status": (
                    "OK" if manifest.get("missing_quote_count") == 0
                    and manifest.get("prediction_count") == manifest.get("expected_symbol_count")
                    else "PARTIAL"
                ),
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
            store.upload_or_update(
                manifest_path,
                f"{REMOTE_ROOT}/latest_manifest.json",
                mime_type="application/json",
            )
        except Exception as exc:
            manifest["manifest_drive_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
            manifest["status"] = "PERSISTENCE_ERROR"
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
        self.start()
        capture_manifest = capture_manifest or {}
        minute = _as_ist_timestamp(
            capture_manifest.get("cycle_minute_ist"),
            fallback=pd.Timestamp.now(tz=IST).floor("min"),
        ).floor("min")
        minute_key = minute.isoformat()
        if self._last_minute == minute_key:
            return {"status": "DUPLICATE_MINUTE_SKIPPED", "cycle_minute_ist": minute_key}

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
            seed_status = self._seed_status
            histories = self._history
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
        pending_additions: dict[str, tuple[pd.Timestamp, dict[str, Any]]] = {}
        missing_quotes = 0
        data_limited_count = 0
        model_failures = 0
        quote_timestamp = pd.Timestamp.now(tz=IST)

        for symbol in symbols:
            quote = quote_by_symbol.get(symbol)
            hist = histories.get(symbol, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
            hist = hist.copy()
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
            if not quote_good:
                missing_quotes += 1
                quote_info["prediction_status"] = (
                    "MISSING_CURRENT_QUOTE" if actual_price is None
                    else "STALE_OR_INVALID_QUOTE"
                )
            elif not history_ready:
                quote_info["prediction_status"] = "WARMING_HISTORY"
            elif model_error:
                quote_info["prediction_status"] = "MODEL_UNAVAILABLE"
                quote_info["error"] = model_error
                model_failures += 1
            else:
                # Use only same-session history to avoid treating an overnight
                # close-to-open gap as a 1-minute or 30-minute return.
                if not hist.empty:
                    hist = hist.loc[
                        (pd.Index(hist.index.date) == minute.date())
                        & (hist.index < minute)
                    ].tail(HISTORY_LOOKBACK_BARS)
                if price_row is not None:
                    history_with_quote = pd.concat([
                        hist,
                        pd.DataFrame([price_row], index=pd.DatetimeIndex([minute], name="Datetime")),
                    ]).sort_index()
                    history_with_quote = history_with_quote.loc[
                        ~history_with_quote.index.duplicated(keep="last")
                    ].tail(HISTORY_LOOKBACK_BARS)
                    try:
                        model_feature_row = self._model_api._latest_features(
                            history_with_quote,
                            market_df=(
                                pd.concat([
                                    self._market_history.loc[
                                        (pd.Index(self._market_history.index.date) == minute.date())
                                        & (self._market_history.index < minute)
                                    ],
                                    pd.DataFrame(
                                        [{
                                            "Open": np.nan,
                                            "High": np.nan,
                                            "Low": np.nan,
                                            "Close": float(nifty_price),
                                            "Volume": np.nan,
                                        }],
                                        index=pd.DatetimeIndex([minute], name="Datetime"),
                                    ),
                                ]).sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
                                if nifty_price is not None else self._market_history
                            ),
                            decision_timestamp=actual_ts,
                        )
                        feature_rows.append(model_feature_row)
                        feature_symbols.append(symbol)
                        current_rows.append(quote_info)
                        quote_actuals[symbol] = {
                            "day_volume": raw_day_volume,
                            "date": minute.date(),
                        }
                        pending_additions[symbol] = (minute, price_row)
                    except Exception as exc:
                        quote_info["prediction_status"] = "FEATURE_BUILD_ERROR"
                        quote_info["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
                        model_failures += 1
                else:
                    quote_info["prediction_status"] = "MISSING_CURRENT_QUOTE"
            if symbol not in feature_symbols:
                current_rows.append(quote_info)
                if quote_good and price_row is not None and not history_ready:
                    with self._lock:
                        self._pending_quote_rows.setdefault(symbol, []).append((minute, price_row))
                        # Bound bootstrap accumulation if an upstream seed call stalls.
                        self._pending_quote_rows[symbol] = self._pending_quote_rows[symbol][-180:]
                elif quote_good and price_row is not None and history_ready and not model_error:
                    pending_additions[symbol] = (minute, price_row)
                    quote_actuals[symbol] = {
                        "day_volume": raw_day_volume,
                        "date": minute.date(),
                    }

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
                    raw_signal, signal = _signal(probability, readiness)
                    row.update({
                        "prediction_status": "PASS" if readiness >= MIN_SIGNAL_READINESS else "PASS_DATA_LIMITED",
                        "raw_model_signal": raw_signal,
                        "signal": signal,
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
                        "trained_for_symbol": symbol in self._trained_symbols,
                        "feature_count": len(self._model_api.FEATURES),
                        "non_missing_feature_count": int(x.iloc[idx].notna().sum()),
                        "data_readiness_score": readiness,
                        "data_limited": readiness < MIN_SIGNAL_READINESS,
                        "forecast_target_timestamp_utc": (
                            (actual_ts + pd.Timedelta(minutes=30)).tz_convert("UTC").isoformat()
                        ),
                        "forecast_key": (
                            f"{symbol}|{actual_ts.tz_convert('UTC').isoformat()}|30|"
                            f"{self._model_status.get('trained_until') or 'all-nse-pooled'}"
                        ),
                        "error": None,
                    })
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
            for symbol, (ts, price_row) in pending_additions.items():
                hist = self._history.get(
                    symbol, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
                )
                new_row = pd.DataFrame([price_row], index=pd.DatetimeIndex([ts], name="Datetime"))
                hist = pd.concat([hist, new_row]).sort_index()
                hist = hist.loc[~hist.index.duplicated(keep="last")].tail(HISTORY_LOOKBACK_BARS)
                self._history[symbol] = hist
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
            and row.get("prediction_status") in {"PASS", "PASS_DATA_LIMITED"}
        )
        missing_quotes = sum(
            1 for row in records if row.get("actual_market_value") is None
            or row.get("actual_quote_status") != "PASS"
        )
        data_limited_count = sum(
            1 for row in records if row.get("prediction_status") == "PASS_DATA_LIMITED"
        )
        manifest = {
            "audit_name": "all-nse-minute-prediction-actual-capture",
            "schema_version": "all-nse-minute-prediction-v1",
            "cycle_minute_ist": minute.isoformat(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_symbol_count": len(symbols),
            "actual_quote_count": len(records) - missing_quotes,
            "missing_quote_count": missing_quotes,
            "prediction_count": prediction_count,
            "data_limited_count": data_limited_count,
            "model_failure_count": model_failures,
            "model_target": "target_30m",
            "model_status": self._seed_status,
            "history_ready": history_ready,
            "history_seed_coverage_symbols": self._seed_coverage,
            "history_seed_failed_batches": self._seed_errors,
            "model_coverage_symbols": (
                self._model_status.get("training_symbol_count") or len(self._trained_symbols)
            ),
            "nifty_context_status": "LIVE_QUOTE" if nifty_price is not None else "HISTORY_OR_MISSING",
            "nifty_context_error": nifty_error,
            "quote_source_snapshot": str(local_quote_path),
            "quote_snapshot_read_error": read_error,
            "current_minute_ohlc_complete": False,
            "quality_gate_pass": (
                len(records) == len(symbols) and missing_quotes == 0
                and prediction_count == len(symbols) and data_limited_count == 0
            ),
            "status": "PARTIAL",
        }
        persisted = self._persist(records, manifest, minute)
        with self._lock:
            self._last_minute = minute_key
            self._last_health = {
                **persisted,
                "status": persisted.get("status"),
                "history_ready": self._seed_ready,
                "universe_symbols": len(self._symbols) or len(symbols),
                "history_symbols": self._seed_coverage,
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
            "history_status": seed_status,
        }
