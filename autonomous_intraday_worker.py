#!/usr/bin/env python3
"""Unattended NSE intraday watchlist collector and prediction worker.

The worker runs independently of Streamlit, polls during the NSE cash session,
persists 1-minute watchlist bars to a dedicated Drive namespace, runs the saved
Beta V2 horizon models, and appends prediction/health records to a Drive ledger.

It scores the dynamic news-derived watchlist (top 5 by default), not thousands
of securities every minute. The existing nse_intraday_sync.yml job retains
ownership of the separate hourly full-universe archive.
"""
from __future__ import annotations

import argparse
import json
import os
import time as time_module
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_session_guard import CLOSE, OPEN, is_nse_trading_day
from delivery_training_universe import benchmark_for_symbol, load_official_nse_universe
from market_watchlist import build_stock_watchlist, fetch_watchlist_articles

IST = ZoneInfo("Asia/Kolkata")
DEFAULT_ROOT = Path("data/autonomous_intraday")
FLUSH_EVERY_DEFAULT = 5
WATCHLIST_REFRESH_SECONDS_DEFAULT = 15 * 60
MAX_DATA_AGE_SECONDS = 5 * 60
SEGMENT_STOP = {
    "morning": time(12, 15),
    "afternoon": time(15, 30),
    "once": time(15, 30),
}


def _log(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print("[AUTONOMOUS_INTRADAY] " + json.dumps(_clean_json(payload), sort_keys=True), flush=True)


def _clean_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _clean_json(value.item())
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


def normalize_bars(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Canonicalize OHLCV and timezone for the active minute feed."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    if isinstance(x.columns, pd.MultiIndex):
        # Prefer a column level with the actual OHLCV names (Yahoo can return
        # MultiIndex columns even for a single ticker).
        chosen = None
        for level in range(x.columns.nlevels):
            labels = {str(v).strip().lower() for v in x.columns.get_level_values(level)}
            if {"open", "high", "low", "close"}.issubset(labels):
                chosen = level
                break
        if chosen is not None:
            x.columns = x.columns.get_level_values(chosen)
        else:
            x.columns = [
                "_".join(str(v) for v in col if str(v) and str(v) != "nan")
                for col in x.columns
            ]

    aliases = {
        "o": "Open", "open": "Open",
        "h": "High", "high": "High",
        "l": "Low", "low": "Low",
        "c": "Close", "close": "Close", "ltp": "Close",
        "v": "Volume", "volume": "Volume", "vol": "Volume",
    }
    rename = {}
    for col in x.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in aliases:
            rename[col] = aliases[key]
    x = x.rename(columns=rename)

    if "_time" in x.columns:
        raw_ts = pd.to_datetime(x.pop("_time"), errors="coerce")
    elif "timestamp" in x.columns:
        raw_ts = pd.to_datetime(x.pop("timestamp"), errors="coerce")
    elif "datetime" in x.columns:
        raw_ts = pd.to_datetime(x.pop("datetime"), errors="coerce")
    else:
        raw_ts = pd.to_datetime(x.index, errors="coerce")

    idx = pd.DatetimeIndex(raw_ts)
    idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    x.index = idx
    x.index.name = "Datetime"

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.loc[~x.index.isna()].sort_index()
    x = x[~x.index.duplicated(keep="last")]
    x = x.dropna(subset=["Open", "High", "Low", "Close"])
    return x


def current_session_completed(frame: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Keep only today's completed bars and avoid using a still-forming candle."""
    x = normalize_bars(frame)
    if x.empty:
        return x
    today = now.tz_convert(IST).date()
    x = x.loc[(x.index.date == today) & (x.index < now.floor("min"))].copy()
    return x


def is_fresh_current_session(frame: pd.DataFrame, now: pd.Timestamp, max_age_seconds: int = MAX_DATA_AGE_SECONDS) -> tuple[bool, dict]:
    x = normalize_bars(frame)
    if x.empty:
        return False, {"status": "NO_DATA", "reason": "empty frame", "latest_candle_ist": None}
    latest = pd.Timestamp(x.index.max())
    if latest.tzinfo is None:
        latest = latest.tz_localize(IST)
    else:
        latest = latest.tz_convert(IST)
    local_now = now.tz_convert(IST)
    age = max(0.0, (local_now - latest).total_seconds())
    current_date = local_now.date() == latest.date()
    ok = current_date and age <= max_age_seconds
    return ok, {
        "status": "PASS" if ok else ("STALE" if current_date else "NOT_CURRENT_SESSION"),
        "latest_candle_ist": latest.isoformat(),
        "age_seconds": round(age, 2),
        "reason": None if ok else (
            f"latest candle age {age:.1f}s exceeds {max_age_seconds}s"
            if current_date else "latest candle is not from the current NSE session"
        ),
    }


def segment_should_stop(now: datetime, segment: str) -> bool:
    """Pure market-hours check used by the worker and unit tests."""
    current = now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)
    if not is_nse_trading_day(current.date()):
        return True
    if current.time() < OPEN or current.time() >= CLOSE:
        return True
    stop_at = SEGMENT_STOP[segment]
    return current.time() >= stop_at


def _watchlist(max_stocks: int) -> tuple[list[str], dict]:
    universe = load_official_nse_universe(force_refresh=False)
    sources = fetch_watchlist_articles()
    built = build_stock_watchlist(universe, sources, max_stocks=max_stocks)
    symbols = []
    for item in built.get("stocks", []) or []:
        root = str(item.get("symbol") or "").strip().upper().replace(".NS", "").replace(".BO", "")
        if root and root not in symbols:
            symbols.append(root)

    # Optional operator-defined symbols are a transparent fallback if news
    # source matching returns no stocks. They are not inferred or invented.
    fixed = [
        value.strip().upper().replace(".NS", "").replace(".BO", "")
        for value in os.getenv("AUTONOMOUS_INTRADAY_FIXED_SYMBOLS", "").split(",")
        if value.strip()
    ]
    if not symbols:
        symbols = list(dict.fromkeys(fixed))[:max_stocks]
    return symbols[:max_stocks], {
        "status": built.get("status", "UNKNOWN"),
        "checked_at_utc": built.get("checked_at_utc"),
        "article_count": built.get("article_count", 0),
        "source_status": built.get("sources", []),
        "dynamic_symbols": symbols,
        "message": built.get("message"),
    }


def _fetch_stock_bars(symbol: str) -> tuple[pd.DataFrame, str, dict]:
    root = symbol.replace(".NS", "").replace(".BO", "").upper()
    errors = []

    # Prefer the configured read-only broker feed when present.
    try:
        from indstocks_client import configured, historical_1m
        if configured():
            frame = normalize_bars(historical_1m(root, days=1))
            if not frame.empty:
                return frame, "INDSTOCKS_API", {"attempts": ["indstocks_historical_1m"]}
            errors.append("INDstocks returned no rows")
    except Exception as exc:
        errors.append(f"INDSTOCKS_API:{type(exc).__name__}:{exc}")

    # NSE official chart is the first public fallback.
    try:
        from nse_data_layer import fetch_intraday_1m_current
        frame = normalize_bars(fetch_intraday_1m_current(root, sync_drive=False, retries=1))
        if not frame.empty:
            return frame, "NSE_OFFICIAL_CHART", {"attempts": ["nse_official_chart"]}
        errors.append("NSE official chart returned no rows")
    except Exception as exc:
        errors.append(f"NSE_OFFICIAL_CHART:{type(exc).__name__}:{exc}")

    # The existing project's historical adapter supplies its configured
    # provider fallback (including Yahoo reference history when available).
    try:
        from data_fetcher import get_historical
        frame = normalize_bars(
            get_historical(f"{root}.NS", period="1d", interval="1m", force_refresh=True)
        )
        if not frame.empty:
            return frame, "HISTORICAL_ADAPTER_FALLBACK", {"attempts": ["indstocks", "nse_chart", "get_historical"], "prior_errors": errors}
        errors.append("historical adapter returned no rows")
    except Exception as exc:
        errors.append(f"HISTORICAL_ADAPTER:{type(exc).__name__}:{exc}")

    raise RuntimeError("All stock-data sources failed: " + " | ".join(errors[-5:]))


def _fetch_market_bars(benchmark: str) -> tuple[pd.DataFrame, str]:
    errors = []
    try:
        from data_fetcher import get_historical
        frame = normalize_bars(
            get_historical(benchmark, period="1d", interval="1m", force_refresh=True)
        )
        if not frame.empty:
            return frame, "HISTORICAL_ADAPTER"
        errors.append("historical adapter returned no rows")
    except Exception as exc:
        errors.append(f"historical adapter:{type(exc).__name__}:{exc}")

    try:
        import yfinance as yf
        from data_fetcher import _normalize_historical_frame
        raw = yf.download(
            benchmark, period="1d", interval="1m",
            auto_adjust=False, progress=False, threads=False, timeout=10,
        )
        frame = normalize_bars(_normalize_historical_frame(raw))
        if not frame.empty:
            return frame, "YFINANCE_DIRECT"
        errors.append("yfinance returned no rows")
    except Exception as exc:
        errors.append(f"yfinance:{type(exc).__name__}:{exc}")

    raise RuntimeError("All benchmark sources failed: " + " | ".join(errors[-4:]))


class AutonomousIntradayWorker:
    def __init__(
        self,
        *,
        segment: str,
        interval_seconds: int = 60,
        refresh_watchlist_seconds: int = WATCHLIST_REFRESH_SECONDS_DEFAULT,
        flush_every: int = FLUSH_EVERY_DEFAULT,
        max_stocks: int = 5,
        root: Path = DEFAULT_ROOT,
        max_cycles: int = 0,
    ) -> None:
        if segment not in SEGMENT_STOP:
            raise ValueError(f"Unsupported segment: {segment}")
        self.segment = segment
        self.interval_seconds = max(10, int(interval_seconds))
        self.refresh_watchlist_seconds = max(60, int(refresh_watchlist_seconds))
        self.flush_every = max(1, int(flush_every))
        self.max_stocks = max(1, min(10, int(max_stocks)))
        self.root = Path(root)
        self.max_cycles = max(0, int(max_cycles))
        self.cycle_count = 0
        self.symbols: list[str] = []
        self.watchlist_meta: dict = {}
        self.last_watchlist_refresh = 0.0
        self.session_frames: dict[str, tuple[pd.DataFrame, str, Path]] = {}
        self.daily_ledger: Path | None = None
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.run_status = "STARTING"
        self.last_cycle_summary: dict = {}

    def _ensure_paths(self, now: pd.Timestamp) -> None:
        d = now.tz_convert(IST)
        self.root.mkdir(parents=True, exist_ok=True)
        daily = self.root / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
        daily.mkdir(parents=True, exist_ok=True)
        self.daily_ledger = daily / f"predictions_{self.segment}.jsonl"

    def _persist_bars_locally(
        self, symbol: str, frame: pd.DataFrame, source: str
    ) -> tuple[Path, int]:
        """Append/upsert bars into the worker-owned local partition."""
        normalized = normalize_bars(frame)
        if normalized.empty:
            raise RuntimeError(f"No normalized bars to persist for {symbol}")
        now = pd.Timestamp.now(tz=IST)
        day_dir = self.root / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}" / "bars"
        day_dir.mkdir(parents=True, exist_ok=True)
        safe_symbol = "".join(
            char if char.isalnum() or char in {"_", "-", "^"} else "_"
            for char in str(symbol).upper()
        )
        path = day_dir / f"{safe_symbol}.parquet"
        if path.exists():
            try:
                old = normalize_bars(pd.read_parquet(path))
                normalized = pd.concat([old, normalized]).sort_index()
                normalized = normalized[~normalized.index.duplicated(keep="last")]
            except Exception as exc:
                raise RuntimeError(
                    f"Existing worker partition is unreadable for {symbol}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        normalized.to_parquet(path, index=True)
        self.session_frames[str(symbol)] = (normalized, source, path)
        return path, int(len(normalized))

    def _refresh_watchlist_if_due(self, force: bool = False) -> None:
        monotonic_now = time_module.monotonic()
        elapsed = monotonic_now - self.last_watchlist_refresh
        if self.symbols:
            due = force or elapsed >= self.refresh_watchlist_seconds
        else:
            # Avoid re-querying all five news publishers every single minute
            # while still retrying fairly quickly when no symbols have matched.
            due = force or self.last_watchlist_refresh == 0.0 or elapsed >= 300
        if not due:
            return
        try:
            symbols, meta = _watchlist(self.max_stocks)
            self.last_watchlist_refresh = monotonic_now
            self.watchlist_meta = meta
            if symbols:
                self.symbols = symbols
            _log("watchlist_refreshed", symbols=self.symbols, metadata=meta)
        except Exception as exc:
            # Retain the latest successful watchlist and apply a backoff after errors.
            self.last_watchlist_refresh = monotonic_now
            _log("watchlist_refresh_failed", error=f"{type(exc).__name__}: {exc}", retained_symbols=self.symbols)

    def _append_rows(self, rows: list[dict]) -> None:
        if not rows or self.daily_ledger is None:
            return
        with self.daily_ledger.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(_clean_json(row), sort_keys=True, separators=(",", ":")) + "\n")

    def _cycle(self) -> dict:
        now = pd.Timestamp.now(tz=IST)
        self._refresh_watchlist_if_due()
        summary = {
            "cycle": self.cycle_count + 1,
            "observed_at_utc": now.tz_convert("UTC").isoformat(),
            "observed_at_ist": now.isoformat(),
            "watchlist_symbols": list(self.symbols),
            "watchlist_status": self.watchlist_meta.get("status"),
            "stock_fetch_ok": 0,
            "stock_fetch_errors": 0,
            "fresh_benchmark_count": 0,
            "predictions_written": 0,
            "warmup_symbols": [],
            "stale_symbols": [],
            "errors": [],
        }
        if not self.symbols:
            summary["errors"].append("No news-matched watchlist symbols; no prediction was fabricated.")
            self.last_cycle_summary = summary
            _log("cycle_no_watchlist", **summary)
            return summary

        # Fetch each index context once per cycle, then share it among symbols.
        benchmarks = sorted({benchmark_for_symbol(f"{s}.NS") for s in self.symbols})
        market_frames: dict[str, tuple[pd.DataFrame, str, dict]] = {}
        for benchmark in benchmarks:
            try:
                frame, source = _fetch_market_bars(benchmark)
                completed = current_session_completed(frame, now)
                fresh, freshness = is_fresh_current_session(completed, now)
                if not fresh:
                    raise RuntimeError("benchmark is stale: " + str(freshness))
                market_frames[benchmark] = (completed, source, freshness)
                summary["fresh_benchmark_count"] += 1
                self._persist_bars_locally(benchmark, completed, source)
            except Exception as exc:
                summary["errors"].append({
                    "benchmark": benchmark,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                _log("benchmark_unavailable", benchmark=benchmark, error=f"{type(exc).__name__}: {exc}")

        prediction_rows: list[dict] = []
        for symbol in self.symbols:
            root = symbol.replace(".NS", "").replace(".BO", "").upper()
            benchmark = benchmark_for_symbol(f"{root}.NS")
            try:
                raw, source, source_meta = _fetch_stock_bars(root)
                completed = current_session_completed(raw, now)
                if completed.empty:
                    raise RuntimeError("no completed one-minute bars for current session")
                fresh, stock_freshness = is_fresh_current_session(completed, now)
                # Persist locally every cycle. Google Drive sync is batched every
                # five cycles so it does not turn each ticker's minute fetch into
                # a large number of uploads.
                local_path, local_rows = self._persist_bars_locally(
                    root, completed, source
                )
                summary["stock_fetch_ok"] += 1
                _log(
                    "stock_data_collected",
                    symbol=root,
                    source=source,
                    rows_current_session=int(len(completed)),
                    freshness=stock_freshness,
                    local_partition=str(local_path),
                    local_rows=int(local_rows),
                )

                if not fresh:
                    summary["stale_symbols"].append(root)
                    summary["errors"].append({
                        "symbol": root,
                        "error": "Stock candles failed current-session freshness guard",
                        "freshness": stock_freshness,
                    })
                    continue

                if benchmark not in market_frames:
                    summary["errors"].append({
                        "symbol": root,
                        "error": f"No fresh benchmark context ({benchmark}); prediction withheld",
                    })
                    continue

                if len(completed) < 80:
                    summary["warmup_symbols"].append({
                        "symbol": root,
                        "rows": int(len(completed)),
                        "required": 80,
                    })
                    continue

                from selfdriving.sd_v2_runtime import beta_intraday_predict
                predictions = beta_intraday_predict(
                    f"{root}.NS",
                    completed,
                    market_frames[benchmark][0],
                    decision_interval_minutes=1,
                )
                if not predictions:
                    summary["warmup_symbols"].append({
                        "symbol": root,
                        "rows": int(len(completed)),
                        "required": 80,
                        "reason": "runtime produced no eligible rows",
                    })
                for result in predictions:
                    record = {
                        **result,
                        "worker_observed_at_utc": now.tz_convert("UTC").isoformat(),
                        "worker_observed_at_ist": now.isoformat(),
                        "source": source,
                        "stock_latest_candle_ist": completed.index.max().isoformat(),
                        "benchmark_symbol": benchmark,
                        "benchmark_source": market_frames[benchmark][1],
                        "benchmark_latest_candle_ist": market_frames[benchmark][0].index.max().isoformat(),
                        "worker_segment": self.segment,
                    }
                    prediction_rows.append(record)
                    summary["predictions_written"] += 1
                _log(
                    "intraday_prediction_completed",
                    symbol=root,
                    prediction_rows=len(predictions),
                    latest_candle_ist=completed.index.max().isoformat(),
                    source=source,
                    horizons=[
                        {
                            "horizon_min": row.get("horizon_min"),
                            "signal": row.get("signal"),
                            "predicted_price": row.get("predicted_price"),
                            "price_forecast_status": row.get("price_forecast_status"),
                            "decision_timestamp": row.get("decision_timestamp"),
                        }
                        for row in predictions
                    ],
                )
            except Exception as exc:
                summary["stock_fetch_errors"] += 1
                summary["errors"].append({
                    "symbol": root,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                _log("symbol_cycle_failed", symbol=root, error=f"{type(exc).__name__}: {exc}")

        self._append_rows(prediction_rows)
        self.last_cycle_summary = summary
        _log("cycle_completed", **summary)
        return summary

    def _status_payload(self, status: str) -> dict:
        now = pd.Timestamp.now(tz=IST)
        return {
            "worker": "autonomous_intraday_worker",
            "version": "2026-10-13-v1",
            "run_status": status,
            "segment": self.segment,
            "started_at_utc": self.started_at_utc,
            "updated_at_utc": now.tz_convert("UTC").isoformat(),
            "updated_at_ist": now.isoformat(),
            "cycle_count": self.cycle_count,
            "watchlist_symbols": self.symbols,
            "watchlist_metadata": self.watchlist_meta,
            "last_cycle_summary": self.last_cycle_summary,
            "interval_seconds": self.interval_seconds,
            "drive_flush_every_cycles": self.flush_every,
            "prediction_ledger_local": str(self.daily_ledger) if self.daily_ledger else None,
        }

    def _flush(self, final: bool = False) -> dict:
        result = {"status": "OK", "data_partitions_synced": 0, "ledger_synced": False, "status_synced": False, "errors": []}
        synced_symbols = []
        try:
            from persistent_store import GoogleDriveStore
            store = GoogleDriveStore()
            now = pd.Timestamp.now(tz=IST)
            daily_dir = now.strftime("%Y/%m/%d")
            for symbol, (frame, source, local_path) in list(self.session_frames.items()):
                if frame is None or frame.empty or not local_path.exists():
                    continue
                safe_symbol = "".join(
                    char if char.isalnum() or char in {"_", "-", "^"} else "_"
                    for char in str(symbol).upper()
                )
                remote = (
                    f"production_data/intraday_live_worker/{daily_dir}/"
                    f"{safe_symbol}.parquet"
                )
                store.upload_or_update(
                    local_path,
                    remote,
                    mime_type="application/vnd.apache.parquet",
                )
                result["data_partitions_synced"] += 1
                synced_symbols.append(symbol)
            # Avoid repeatedly uploading already-synced partitions until their
            # contents change again. This namespace never races the full-universe
            # producer under production_data/intraday_1m.
            for symbol in synced_symbols:
                self.session_frames.pop(symbol, None)
        except Exception as exc:
            result["errors"].append({"data_sync_error": f"{type(exc).__name__}: {exc}"})

        try:
            from persistent_store import GoogleDriveStore
            store = GoogleDriveStore()
            now = pd.Timestamp.now(tz=IST)
            daily_dir = now.strftime("%Y/%m/%d")
            if self.daily_ledger is not None and self.daily_ledger.exists():
                store.upload_or_update(
                    self.daily_ledger,
                    f"production_data/intraday_predictions/{daily_dir}/predictions_{self.segment}.jsonl",
                    mime_type="application/x-ndjson",
                )
                result["ledger_synced"] = True

            local_status = self.root / "worker_status_latest.json"
            local_status.parent.mkdir(parents=True, exist_ok=True)
            local_status.write_text(
                json.dumps(_clean_json(self._status_payload("STOPPED" if final else "RUNNING")), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            store.upload_or_update(
                local_status,
                f"production_data/intraday_predictions/{daily_dir}/worker_status_{self.segment}.json",
                mime_type="application/json",
            )
            store.upload_or_update(
                local_status,
                "production_data/intraday_predictions/worker_status_latest.json",
                mime_type="application/json",
            )
            result["status_synced"] = True
        except Exception as exc:
            result["errors"].append({"drive_ledger_status_sync_error": f"{type(exc).__name__}: {exc}"})

        if result["errors"]:
            result["status"] = "PARTIAL"
        _log("drive_flush_completed", final=final, **result)
        return result

    def run(self) -> int:
        now_dt = datetime.now(IST)
        if segment_should_stop(now_dt, self.segment):
            self.run_status = "SKIPPED_OUTSIDE_MARKET_WINDOW"
            _log(
                "worker_skipped",
                segment=self.segment,
                now_ist=now_dt.isoformat(),
                nse_trading_day=is_nse_trading_day(now_dt.date()),
                reason="outside configured NSE trading window, weekend, or NSE holiday",
            )
            return 0

        now_ts = pd.Timestamp(now_dt)
        self._ensure_paths(now_ts)
        self.run_status = "RUNNING"
        _log(
            "worker_started",
            segment=self.segment,
            now_ist=now_dt.isoformat(),
            interval_seconds=self.interval_seconds,
            watchlist_refresh_seconds=self.refresh_watchlist_seconds,
            max_stocks=self.max_stocks,
            expected_stop_ist=SEGMENT_STOP[self.segment].isoformat(),
            note="Dashboard is not required to be open. This worker polls only the dynamic intraday watchlist.",
        )

        final_flush = {"status": "OK"}
        try:
            while True:
                now_dt = datetime.now(IST)
                if segment_should_stop(now_dt, self.segment):
                    break
                if self.max_cycles and self.cycle_count >= self.max_cycles:
                    break

                cycle_started = time_module.monotonic()
                self.cycle_count += 1
                self._cycle()

                if self.cycle_count % self.flush_every == 0:
                    self._flush(final=False)

                if self.max_cycles and self.cycle_count >= self.max_cycles:
                    break

                if segment_should_stop(datetime.now(IST), self.segment):
                    break

                # Align polls to the next minute wall-clock boundary. If a cycle
                # overruns, continue immediately rather than adding another 60 s.
                elapsed = time_module.monotonic() - cycle_started
                now_time = datetime.now(IST)
                delay_to_boundary = 60.0 - (now_time.second + now_time.microsecond / 1_000_000.0)
                delay = max(1.0, delay_to_boundary)
                if elapsed >= self.interval_seconds:
                    delay = min(delay, 1.0)
                time_module.sleep(delay)
        except KeyboardInterrupt:
            self.run_status = "INTERRUPTED"
            _log("worker_interrupted", cycles=self.cycle_count)
        finally:
            final_flush = self._flush(final=True)
            self.run_status = "STOPPED"
            _log(
                "worker_stopped",
                segment=self.segment,
                cycles=self.cycle_count,
                final_flush=final_flush,
                stopped_at_ist=datetime.now(IST).isoformat(),
            )

        return 1 if final_flush.get("status") != "OK" else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run unattended NSE intraday data/prediction worker.")
    parser.add_argument("--segment", choices=["morning", "afternoon", "once"], required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--refresh-watchlist-seconds", type=int, default=WATCHLIST_REFRESH_SECONDS_DEFAULT)
    parser.add_argument("--flush-every", type=int, default=FLUSH_EVERY_DEFAULT)
    parser.add_argument("--max-stocks", type=int, default=int(os.getenv("AUTONOMOUS_INTRADAY_MAX_STOCKS", "5")))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run until the configured segment ends.")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring notebook/runtime arguments:", unknown)

    if args.segment == "once" and args.max_cycles == 0:
        args.max_cycles = 1

    worker = AutonomousIntradayWorker(
        segment=args.segment,
        interval_seconds=args.interval_seconds,
        refresh_watchlist_seconds=args.refresh_watchlist_seconds,
        flush_every=args.flush_every,
        max_stocks=args.max_stocks,
        root=args.root,
        max_cycles=args.max_cycles,
    )
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())
