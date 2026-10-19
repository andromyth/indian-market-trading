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
import calibration_overlay as calibration_overlay_module

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"
DEFAULT_ROOT = Path("data/autonomous_intraday")
FLUSH_EVERY_DEFAULT = 5
WATCHLIST_REFRESH_SECONDS_DEFAULT = 15 * 60
MAX_DATA_AGE_SECONDS = 5 * 60
SEGMENT_STOP = {
    "morning": time(12, 15),
    "afternoon": time(15, 30),
    "once": time(15, 30),
    "service": time(15, 30),
}


def _log(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    print("[AUTONOMOUS_INTRADAY] " + json.dumps(_clean_json(payload), sort_keys=True), flush=True)


def _download_drive_file_if_present(store: Any, remote_path: str, local_path: Path) -> bool:
    """Download an existing Drive file without creating missing folders/files."""
    parts = [part for part in remote_path.replace("\\", "/").split("/") if part]
    if not parts:
        return False
    parent = store.root_id
    for part in parts[:-1]:
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            return False
        parent = hit["id"]
    hit = store._find_child(parent, parts[-1])
    if not hit:
        return False

    from googleapiclient.http import MediaIoBaseDownload

    local_path.parent.mkdir(parents=True, exist_ok=True)
    request = store.service.files().get_media(fileId=hit["id"])
    with local_path.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True


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
    # Yahoo can return duplicate OHLC field labels after its MultiIndex columns
    # are flattened (for example, multiple ticker levels). Keep the first field
    # deterministically so pd.to_numeric never receives a 2-D DataFrame.
    x = x.loc[:, ~x.columns.duplicated(keep="first")]

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
    try:
        universe = load_official_nse_universe(force_refresh=False)
    except Exception as exc:
        # A missing symbol master blocks symbol matching, but the independent
        # market-news collector continues on its own scheduled workflow.
        universe = pd.DataFrame()
        _log("watchlist_universe_unavailable", error=f"{type(exc).__name__}: {exc}")
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

def _fetch_yfinance_1m(ticker: str) -> pd.DataFrame:
    """Fetch Yahoo's latest one-minute bars without the shared dashboard adapter.

    The dashboard adapter can legitimately return an empty frame for symbols
    whose other providers fail. For this worker, make one explicit Yahoo
    attempt and normalize the raw yfinance shape at this boundary.
    """
    import yfinance as yf

    raw = yf.download(
        str(ticker).strip(),
        period="1d",
        interval="1m",
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=10,
        group_by="column",
    )
    return normalize_bars(raw)


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

    # Direct Yahoo fallback is separate from get_historical so a blocked NSE
    # chart or empty historical adapter does not prevent the final provider try.
    try:
        frame = _fetch_yfinance_1m(f"{root}.NS")
        if not frame.empty:
            return frame, "YFINANCE_DIRECT", {
                "attempts": ["indstocks", "nse_chart", "get_historical", "yfinance_download"],
                "prior_errors": errors,
            }
        errors.append("yfinance returned no rows")
    except Exception as exc:
        errors.append(f"YFINANCE_DIRECT:{type(exc).__name__}:{exc}")

    raise RuntimeError("All stock-data sources failed: " + " | ".join(errors[-6:]))


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
        raw = yf.download(
            benchmark, period="1d", interval="1m",
            auto_adjust=False, progress=False, threads=False, timeout=10,
            group_by="column",
        )
        # Normalize raw provider columns directly; the generic dashboard
        # adapter is unnecessary here and can fail on duplicate flattened fields.
        frame = normalize_bars(raw)
        if not frame.empty:
            return frame, "YFINANCE_DIRECT"
        errors.append("yfinance returned no rows")
    except Exception as exc:
        errors.append(f"yfinance:{type(exc).__name__}:{exc}")

    raise RuntimeError("All benchmark sources failed: " + " | ".join(errors[-4:]))


def _prediction_row_failure(symbol: str, prediction: dict[str, Any]) -> dict[str, Any] | None:
    """Return a health error when a row is not a usable live model inference.

    ABSTAIN is a valid policy output when the model has a valid probability and
    decision timestamp. An exception, absent probability, or absent timestamp is
    instead an inference failure and must degrade the worker health.
    """
    horizon = prediction.get("horizon_min")

    def failure(reason: str) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "horizon_min": horizon,
            "error": f"Prediction unavailable: {reason}",
        }

    reason = str(prediction.get("reason") or "").strip()
    if reason:
        return failure(reason)

    try:
        probability = float(prediction.get("probability_up"))
    except (TypeError, ValueError):
        return failure("missing_or_invalid_probability_up")
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return failure("missing_or_invalid_probability_up")

    decision_timestamp = pd.to_datetime(
        prediction.get("decision_timestamp"), errors="coerce", utc=True
    )
    if pd.isna(decision_timestamp):
        return failure("missing_or_invalid_decision_timestamp")

    return None


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
        self.last_tick_key: str | None = None
        self.last_minute_snapshot_rows: list[dict[str, Any]] = []
        self._hydrated_ledger_day: str | None = None
        self.last_news_context_refresh = 0.0
        self.news_context_by_symbol: dict[str, dict[str, Any]] = {}
        self.market_news_context: dict[str, Any] = {}
        self.news_manifest: dict[str, Any] = {}
        self.news_context_status = "UNKNOWN"
        self._calibration = None
        self._calibration_init_error_logged = False

    def _apply_calibration(self, rows: list[dict]) -> list[dict]:
        """Apply the optional fail-open calibration overlay to a batch of forecasts."""
        mode = calibration_overlay_module.resolve_mode(os.getenv("PREDICTION_CALIBRATION_MODE", "shadow"))
        if mode == "off" or not rows:
            return rows
        try:
            if self._calibration is None:
                def load_profile_document():
                    from persistent_store import GoogleDriveStore

                    store = GoogleDriveStore()
                    remote = "production_data/intraday_predictions/learning/calibration/active_calibration_profile.json"
                    local = self.root / "calibration" / "active_calibration_profile.json"
                    if not _download_drive_file_if_present(store, remote, local):
                        return None
                    return json.loads(local.read_text(encoding="utf-8"))

                self._calibration = calibration_overlay_module.CalibrationOverlay(
                    load_profile_document, mode=mode, refresh_seconds=900
                )
            else:
                self._calibration.mode = mode
            return [self._calibration.annotate(row) for row in rows]
        except Exception as exc:
            if not self._calibration_init_error_logged:
                _log("calibration_overlay_unavailable", error=f"{type(exc).__name__}: {exc}", fail_open=True)
                self._calibration_init_error_logged = True
            return rows

    def _ensure_paths(self, now: pd.Timestamp) -> None:
        d = now.tz_convert(IST)
        self.root.mkdir(parents=True, exist_ok=True)
        daily = self.root / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}"
        daily.mkdir(parents=True, exist_ok=True)
        self.daily_ledger = daily / f"predictions_{self.segment}.jsonl"

    def _hydrate_daily_ledger(self) -> None:
        """Restore the current day's append-only ledger before any Drive overwrite."""
        if self.daily_ledger is None:
            return
        ledger_parts = self.daily_ledger.relative_to(self.root).parts
        if len(ledger_parts) < 4:
            raise RuntimeError(f"Unexpected worker ledger path: {self.daily_ledger}")
        ledger_day = "/".join(ledger_parts[:3])
        if self._hydrated_ledger_day == ledger_day:
            return
        if self.daily_ledger.is_file() and self.daily_ledger.stat().st_size > 0:
            self._hydrated_ledger_day = ledger_day
            return

        from persistent_store import GoogleDriveStore

        remote = (
            f"production_data/intraday_predictions/{ledger_day}/"
            f"predictions_{self.segment}.jsonl"
        )
        try:
            found = _download_drive_file_if_present(
                GoogleDriveStore(), remote, self.daily_ledger
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot safely hydrate existing Drive prediction ledger before write: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._hydrated_ledger_day = ledger_day
        _log(
            "daily_prediction_ledger_hydrated",
            ledger_day=ledger_day,
            remote_path=remote,
            existing_remote_ledger_found=found,
            local_bytes=(
                self.daily_ledger.stat().st_size
                if self.daily_ledger.is_file() else 0
            ),
        )

    def _refresh_news_context_if_due(self, force: bool = False) -> None:
        """Load the latest publisher-news PIT archive after the 15-minute collection."""
        monotonic_now = time_module.monotonic()
        interval_due = force or not self.last_news_context_refresh or (
            monotonic_now - self.last_news_context_refresh >= 15 * 60
        )
        cache_dir = self.root / "news_cache"
        pit_path = cache_dir / "website_news_pit_2026.parquet"
        manifest_path = cache_dir / "website_news_manifest_2026.json"
        try:
            from persistent_store import GoogleDriveStore
            store = GoogleDriveStore()
            probe_path = cache_dir / "website_news_manifest_probe.json"
            try:
                probe_found = _download_drive_file_if_present(
                    store, "processed/live_news/website_news_manifest_2026.json", probe_path
                )
                remote_manifest = (
                    json.loads(probe_path.read_text(encoding="utf-8"))
                    if probe_found and probe_path.exists() else {}
                )
            except Exception as probe_exc:
                if not interval_due:
                    _log(
                        "website_news_manifest_probe_failed",
                        error=f"{type(probe_exc).__name__}: {probe_exc}",
                        retained_existing_context=True,
                    )
                    return
                probe_found = False
                remote_manifest = {}
                _log(
                    "website_news_manifest_probe_failed",
                    error=f"{type(probe_exc).__name__}: {probe_exc}",
                    will_attempt_full_refresh=True,
                )

            remote_checked_at = remote_manifest.get("checked_at_utc")
            cached_checked_at = self.news_manifest.get("checked_at_utc")
            manifest_changed = bool(
                remote_checked_at and remote_checked_at != cached_checked_at
            )
            if not interval_due and not manifest_changed:
                return

            self.last_news_context_refresh = monotonic_now
            if probe_found:
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    probe_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                manifest_found = True
            else:
                manifest_found = _download_drive_file_if_present(
                    store, "processed/live_news/website_news_manifest_2026.json", manifest_path
                )
            pit_found = _download_drive_file_if_present(
                store, "processed/live_news/website_news_pit_2026.parquet", pit_path
            )
            self.news_manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_found and manifest_path.exists() else {}
            )
            if not pit_found or not pit_path.exists():
                self.news_context_status = "MISSING"
                self.news_context_by_symbol = {}
                self.market_news_context = {}
                _log("website_news_context_unavailable", reason="PIT archive is not yet present in Google Drive")
                return
            frame = pd.read_parquet(pit_path)
            if frame.empty or "symbol" not in frame.columns or "decision_timestamp" not in frame.columns:
                self.news_context_status = "EMPTY_OR_INVALID"
                self.news_context_by_symbol = {}
                self.market_news_context = {}
                _log("website_news_context_unavailable", reason="PIT archive schema is empty or invalid")
                return
            frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
            frame["decision_timestamp"] = pd.to_datetime(
                frame["decision_timestamp"], errors="coerce", utc=True
            )
            now_utc = pd.Timestamp.now(tz="UTC")
            cutoff = now_utc - pd.Timedelta(days=7)
            frame = frame.dropna(subset=["decision_timestamp"])
            frame = frame.loc[
                frame["decision_timestamp"].le(now_utc)
                & frame["decision_timestamp"].ge(cutoff)
            ].copy()
            if "news_article_count" not in frame:
                frame["news_article_count"] = 1
            if "news_source" not in frame:
                frame["news_source"] = "UNKNOWN"
            if "news_sentiment" not in frame:
                frame["news_sentiment"] = np.nan
            if "news_direction_score" not in frame:
                frame["news_direction_score"] = frame["news_sentiment"]
            if "event_impact_score" not in frame:
                frame["event_impact_score"] = 0.25
            if "news_source_weight" not in frame:
                frame["news_source_weight"] = 0.5
            frame["news_article_count"] = pd.to_numeric(
                frame["news_article_count"], errors="coerce"
            ).fillna(1)
            frame["news_sentiment"] = pd.to_numeric(frame["news_sentiment"], errors="coerce")
            frame["news_direction_score"] = pd.to_numeric(frame["news_direction_score"], errors="coerce")
            frame["event_impact_score"] = pd.to_numeric(
                frame["event_impact_score"], errors="coerce"
            ).fillna(0.25).clip(0.0, 1.0)
            frame["news_source_weight"] = pd.to_numeric(
                frame["news_source_weight"], errors="coerce"
            ).fillna(0.5).clip(0.0, 1.0)
            self.news_context_by_symbol = {}
            if not frame.empty:
                for symbol, group in frame.loc[frame["symbol"].ne("MARKET")].groupby("symbol"):
                    latest = group.sort_values("decision_timestamp").iloc[-1]
                    source_values_7d = {
                        publisher.strip()
                        for value in group["news_source"].dropna().astype(str)
                        for publisher in value.split(";") if publisher.strip()
                    }
                    recent = group.loc[
                        group["decision_timestamp"].ge(now_utc - pd.Timedelta(hours=24))
                    ].copy()
                    recent["age_hours"] = (
                        now_utc - recent["decision_timestamp"]
                    ).dt.total_seconds() / 3600.0
                    recent = recent.loc[recent["age_hours"].between(0.0, 24.0)]
                    if not recent.empty:
                        weights = (
                            recent["news_source_weight"]
                            * (0.5 + 0.5 * recent["event_impact_score"])
                            * np.exp(-recent["age_hours"] / 18.0)
                        ).clip(lower=0.01)
                        direction = recent["news_direction_score"].fillna(
                            recent["news_sentiment"]
                        ).fillna(0.0)
                        weighted_score = float(np.average(direction, weights=weights))
                        positive_weight = float(weights.loc[direction.gt(0.08)].sum())
                        negative_weight = float(weights.loc[direction.lt(-0.08)].sum())
                        total_weight = max(float(weights.sum()), 1e-9)
                        directional_agreement = max(positive_weight, negative_weight) / total_weight
                        max_impact = float(recent["event_impact_score"].max())
                        source_values_24h = {
                            publisher.strip()
                            for value in recent["news_source"].dropna().astype(str)
                            for publisher in value.split(";") if publisher.strip()
                        }
                        article_count_24h = int(recent["news_article_count"].sum())
                    else:
                        weighted_score = 0.0
                        directional_agreement = 0.0
                        max_impact = 0.0
                        source_values_24h = set()
                        article_count_24h = 0
                    sentiments = group["news_sentiment"].dropna()
                    context_strength = float(np.clip(
                        0.70 * abs(weighted_score)
                        + 0.20 * max_impact
                        + 0.10 * min(len(source_values_24h) / 3.0, 1.0),
                        0.0, 1.0,
                    ))
                    self.news_context_by_symbol[str(symbol)] = {
                        "scope": "SYMBOL",
                        "article_count_7d": int(group["news_article_count"].sum()),
                        "article_count_24h": article_count_24h,
                        "source_count_7d": int(len(source_values_7d)),
                        "source_count_24h": int(len(source_values_24h)),
                        "sources_7d": sorted(source_values_7d)[:12],
                        "sources_24h": sorted(source_values_24h)[:12],
                        "mean_sentiment_7d": float(sentiments.mean()) if not sentiments.empty else None,
                        "news_direction_score_24h": weighted_score,
                        "news_directional_agreement_24h": float(directional_agreement),
                        "news_event_impact_24h": max_impact,
                        "news_signal_strength": context_strength,
                        "news_signal": "NEWS_INSUFFICIENT_DATA",
                        "news_signal_reason": "NEWS_SOURCE_STATUS_NOT_YET_CHECKED",
                        "latest_news_available_at_utc": latest["decision_timestamp"].isoformat(),
                        "latest_headline": str(latest.get("news_title") or "")[:500],
                        "latest_news_source": str(latest.get("news_source") or ""),
                    }
            market_rows = frame.loc[frame["symbol"].eq("MARKET")]
            if market_rows.empty:
                market_rows = frame
            market_sentiment = market_rows["news_sentiment"].dropna()
            market_sources = {
                publisher.strip()
                for value in market_rows["news_source"].dropna().astype(str)
                for publisher in value.split(";") if publisher.strip()
            }
            latest_market = (
                market_rows.sort_values("decision_timestamp").iloc[-1]
                if not market_rows.empty else None
            )
            self.market_news_context = {
                "scope": "MARKET",
                "article_count_7d": int(market_rows["news_article_count"].sum()),
                "source_count_7d": int(len(market_sources)),
                "sources_7d": sorted(market_sources)[:12],
                "mean_sentiment_7d": float(market_sentiment.mean()) if not market_sentiment.empty else None,
                "latest_news_available_at_utc": (
                    pd.Timestamp(latest_market["decision_timestamp"]).isoformat()
                    if latest_market is not None else None
                ),
                "latest_headline": str(latest_market.get("news_title") or "")[:500] if latest_market is not None else "",
                "latest_news_source": str(latest_market.get("news_source") or "") if latest_market is not None else "",
            }
            checked_at = pd.to_datetime(self.news_manifest.get("checked_at_utc"), errors="coerce", utc=True)
            age_minutes = (
                float((now_utc - checked_at).total_seconds() / 60.0)
                if pd.notna(checked_at) else None
            )
            feed_successes = int(self.news_manifest.get("successful_feed_count", 0) or 0)
            if self.news_manifest.get("collection_status") == "PASS" and age_minutes is not None and 0 <= age_minutes <= 45 and feed_successes >= 3:
                self.news_context_status = "FRESH"
            elif age_minutes is None:
                self.news_context_status = "UNKNOWN"
            else:
                self.news_context_status = "STALE_OR_PARTIAL"
            for context in self.news_context_by_symbol.values():
                score = float(context.get("news_direction_score_24h") or 0.0)
                agreement = float(context.get("news_directional_agreement_24h") or 0.0)
                count = int(context.get("article_count_24h") or 0)
                source_count = int(context.get("source_count_24h") or 0)
                impact = float(context.get("news_event_impact_24h") or 0.0)
                if self.news_context_status != "FRESH":
                    signal = "NEWS_INSUFFICIENT_DATA"
                    reason = "NEWS_FEED_ARCHIVE_STALE_OR_PARTIAL"
                elif count == 0:
                    signal = "NEWS_INSUFFICIENT_DATA"
                    reason = "NO_DIRECT_STOCK_NEWS_IN_LAST_24H"
                elif abs(score) < 0.12:
                    signal = "NEWS_NEUTRAL"
                    reason = "RECENT_NEWS_HAS_NO_CLEAR_DIRECTIONAL_EDGE"
                elif score >= 0.18 and agreement >= 0.60 and (
                    source_count >= 2 or count >= 2 or impact >= 0.85
                ):
                    signal = "NEWS_BUY_BIAS"
                    reason = "POSITIVE_FINANCIAL_NEWS_WITH_SUFFICIENT_SUPPORT"
                elif score <= -0.18 and agreement >= 0.60 and (
                    source_count >= 2 or count >= 2 or impact >= 0.85
                ):
                    signal = "NEWS_SELL_BIAS"
                    reason = "NEGATIVE_FINANCIAL_NEWS_WITH_SUFFICIENT_SUPPORT"
                else:
                    signal = "NEWS_INSUFFICIENT_DATA"
                    reason = "WEAK_OR_CONFLICTING_NEWS_EVIDENCE"
                context["news_signal"] = signal
                context["news_signal_reason"] = reason
            _log(
                "website_news_context_refreshed",
                status=self.news_context_status,
                pit_rows=len(frame),
                symbol_contexts=len(self.news_context_by_symbol),
                checked_at_utc=self.news_manifest.get("checked_at_utc"),
                manifest_age_minutes=age_minutes,
                successful_feed_count=feed_successes,
                collection_status=self.news_manifest.get("collection_status"),
                signal_counts={
                    status: sum(1 for value in self.news_context_by_symbol.values()
                                if value.get("news_signal") == status)
                    for status in ("NEWS_BUY_BIAS", "NEWS_SELL_BIAS",
                                   "NEWS_NEUTRAL", "NEWS_INSUFFICIENT_DATA")
                },
            )
        except Exception as exc:
            self.news_context_status = "ERROR"
            _log("website_news_context_refresh_failed", error=f"{type(exc).__name__}: {exc}")

    def _news_context_for_symbol(self, symbol: str) -> dict[str, Any]:
        symbol = str(symbol or "").strip().upper().replace(".NS", "").replace(".BO", "")
        context = self.news_context_by_symbol.get(symbol)
        if context:
            return {
                **context,
                "context_status": self.news_context_status,
                "used_by_locked_intraday_classifier": False,
                "signal_is_tradable": False,
            }
        return {
            **(self.market_news_context or {"scope": "MARKET", "article_count_7d": 0}),
            "scope": "MARKET_FALLBACK",
            "news_signal": "NEWS_INSUFFICIENT_DATA",
            "news_signal_reason": "NO_DIRECT_STOCK_NEWS_MATCH",
            "news_signal_strength": 0.0,
            "context_status": self.news_context_status,
            "used_by_locked_intraday_classifier": False,
            "signal_is_tradable": False,
        }

    def _persist_minute_snapshot(
        self, now: pd.Timestamp, tick_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Persist a compact, immutable prediction + observed-price shard every minute."""
        if not self.last_minute_snapshot_rows:
            return {"status": "SKIPPED", "reason": "no snapshot rows"}
        local_now = now.tz_convert(IST)
        day_dir = self.root / f"{local_now:%Y}" / f"{local_now:%m}" / f"{local_now:%d}"
        snapshot_dir = day_dir / "minute_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        filename = f"snapshot_{local_now:%Y%m%d_%H%M}.jsonl"
        local_path = snapshot_dir / filename

        rows = []
        for source_row in self.last_minute_snapshot_rows:
            row = dict(source_row)
            row["worker_cycle"] = self.cycle_count
            row["tick_status"] = tick_result.get("status")
            row["snapshot_filename"] = filename
            rows.append(row)
        with local_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(
                    json.dumps(_clean_json(row), sort_keys=True, separators=(",", ":"))
                    + "\n"
                )

        relative_day = f"{local_now:%Y/%m/%d}"
        remote = (
            f"production_data/intraday_minute_snapshots/{relative_day}/{filename}"
        )
        from persistent_store import GoogleDriveStore

        store = GoogleDriveStore()
        upload_result = store.upload_or_update(
            local_path, remote, mime_type="application/x-ndjson"
        )
        prediction_count = sum(
            len(row.get("predictions") or []) for row in rows
        )
        _log(
            "minute_prediction_actual_snapshot_persisted",
            remote_path=remote,
            snapshot_rows=len(rows),
            prediction_rows=prediction_count,
            status="OK",
        )
        return {
            "status": "OK",
            "remote_path": remote,
            "snapshot_rows": len(rows),
            "prediction_rows": prediction_count,
            "upload_result": _clean_json(upload_result),
        }

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
            # Keep RSS scraping and PIT publication on the same 15-minute cadence
            # even when no headline could be mapped to a symbol.
            due = force or self.last_watchlist_refresh == 0.0 or elapsed >= self.refresh_watchlist_seconds
        if not due:
            return
        try:
            symbols, meta = _watchlist(self.max_stocks)
            self.last_watchlist_refresh = monotonic_now
            self.watchlist_meta = meta
            if symbols:
                self.symbols = symbols
            _log(
                "watchlist_refreshed",
                symbols=self.symbols,
                metadata=meta,
                news_collection=meta.get("news_collection", {}),
            )
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
        self.last_minute_snapshot_rows = []
        self._refresh_watchlist_if_due()
        # Probe the Drive manifest each minute; if the 15-minute collector has
        # published a newer snapshot, reload it before this cycle's inference.
        self._refresh_news_context_if_due()
        summary = {
            "cycle": self.cycle_count,
            "observed_at_utc": now.tz_convert("UTC").isoformat(),
            "observed_at_ist": now.isoformat(),
            "watchlist_symbols": list(self.symbols),
            "watchlist_status": self.watchlist_meta.get("status"),
            "news_context_status": self.news_context_status,
            "news_context_checked_at_utc": self.news_manifest.get("checked_at_utc"),
            "stock_fetch_ok": 0,
            "stock_fetch_errors": 0,
            "fresh_benchmark_count": 0,
            "predictions_written": 0,
            "prediction_rows_valid": 0,
            "prediction_errors": 0,
            "prediction_abstains": 0,
            "warmup_symbols": [],
            "stale_symbols": [],
            "errors": [],
        }
        if not self.symbols:
            summary["errors"].append("No news-matched watchlist symbols; no prediction was fabricated.")
            self.last_minute_snapshot_rows = [{
                "record_type": "intraday_minute_observation",
                "observation_timestamp_utc": now.tz_convert("UTC").isoformat(),
                "observation_timestamp_ist": now.isoformat(),
                "worker_segment": self.segment,
                "watchlist_symbols": [],
                "data_quality_status": "BLOCKED",
                "status_reason": "NO_NEWS_MATCHED_WATCHLIST",
                "predictions": [],
            }]
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
        snapshot_rows: list[dict] = []
        for symbol in self.symbols:
            root = symbol.replace(".NS", "").replace(".BO", "").upper()
            benchmark = benchmark_for_symbol(f"{root}.NS")
            snapshot: dict[str, Any] | None = None
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

                benchmark_data = market_frames.get(benchmark)
                benchmark_frame = benchmark_data[0] if benchmark_data else pd.DataFrame()
                benchmark_latest = (
                    pd.Timestamp(benchmark_frame.index.max())
                    if not benchmark_frame.empty else None
                )
                snapshot = {
                    "record_type": "intraday_minute_observation",
                    "observation_id": f"{now:%Y%m%d_%H%M}|{root}",
                    "observation_timestamp_utc": now.tz_convert("UTC").isoformat(),
                    "observation_timestamp_ist": now.isoformat(),
                    "symbol": root,
                    "actual_observed_close": _clean_json(completed["Close"].iloc[-1]),
                    "actual_observed_timestamp_utc": (
                        completed.index.max().tz_convert("UTC").isoformat()
                    ),
                    "actual_observed_timestamp_ist": completed.index.max().isoformat(),
                    "actual_data_source": source,
                    "source_metadata": source_meta,
                    "stock_freshness": stock_freshness,
                    "benchmark_symbol": benchmark,
                    "benchmark_observed_close": (
                        _clean_json(benchmark_frame["Close"].iloc[-1])
                        if not benchmark_frame.empty else None
                    ),
                    "benchmark_observed_timestamp_utc": (
                        benchmark_latest.tz_convert("UTC").isoformat()
                        if benchmark_latest is not None else None
                    ),
                    "benchmark_observed_timestamp_ist": (
                        benchmark_latest.isoformat()
                        if benchmark_latest is not None else None
                    ),
                    "benchmark_data_source": benchmark_data[1] if benchmark_data else None,
                    "benchmark_freshness": benchmark_data[2] if benchmark_data else None,
                    "data_quality_status": "PASS" if fresh and benchmark_data else "BLOCKED",
                    "status_reason": None if fresh and benchmark_data else (
                        "STALE_STOCK_CANDLE" if not fresh else "FRESH_BENCHMARK_UNAVAILABLE"
                    ),
                    "website_news_context": self._news_context_for_symbol(root),
                    "predictions": [],
                }
                snapshot_rows.append(snapshot)

                if not fresh:
                    summary["stale_symbols"].append(root)
                    summary["errors"].append({
                        "symbol": root,
                        "error": "Stock candles failed current-session freshness guard",
                        "freshness": stock_freshness,
                    })
                    continue

                if benchmark not in market_frames:
                    snapshot["data_quality_status"] = "BLOCKED"
                    snapshot["status_reason"] = f"NO_FRESH_BENCHMARK_CONTEXT:{benchmark}"
                    summary["errors"].append({
                        "symbol": root,
                        "error": f"No fresh benchmark context ({benchmark}); prediction withheld",
                    })
                    continue

                if len(completed) < 80:
                    snapshot["data_quality_status"] = "BLOCKED"
                    snapshot["status_reason"] = "MODEL_WARMUP_INSUFFICIENT_HISTORY"
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
                predictions = self._apply_calibration(predictions)
                if not predictions:
                    empty_error = {
                        "symbol": root,
                        "error": "Intraday runtime produced no prediction rows after warmup",
                    }
                    summary["prediction_errors"] += 1
                    summary["errors"].append(empty_error)
                    _log("intraday_prediction_empty", **empty_error)
                    summary["warmup_symbols"].append({
                        "symbol": root,
                        "rows": int(len(completed)),
                        "required": 80,
                        "reason": "runtime produced no eligible rows",
                    })
                for result in predictions:
                    prediction_failure = _prediction_row_failure(root, result)
                    if prediction_failure:
                        summary["prediction_errors"] += 1
                        summary["errors"].append(prediction_failure)
                        _log("intraday_prediction_invalid", **prediction_failure)
                    else:
                        summary["prediction_rows_valid"] += 1
                        if str(result.get("signal") or "").upper() == "ABSTAIN":
                            summary["prediction_abstains"] += 1

                    decision_ts = pd.to_datetime(
                        result.get("decision_timestamp"), errors="coerce", utc=True
                    )
                    try:
                        horizon_min = int(result.get("horizon_min"))
                    except (TypeError, ValueError):
                        horizon_min = 0
                    target_ts = (
                        decision_ts + pd.Timedelta(minutes=horizon_min)
                        if not pd.isna(decision_ts) and horizon_min > 0
                        else pd.NaT
                    )
                    forecast_key = (
                        f"{root}|{decision_ts.isoformat() if not pd.isna(decision_ts) else now.tz_convert('UTC').isoformat()}"
                        f"|{horizon_min}|{result.get('model_version', '')}"
                    )
                    record = {
                        **result,
                        "symbol": root,
                        "forecast_key": forecast_key,
                        "worker_observed_at_utc": now.tz_convert("UTC").isoformat(),
                        "worker_observed_at_ist": now.isoformat(),
                        "source": source,
                        "stock_latest_candle_ist": completed.index.max().isoformat(),
                        "benchmark_symbol": benchmark,
                        "benchmark_source": market_frames[benchmark][1],
                        "benchmark_latest_candle_ist": market_frames[benchmark][0].index.max().isoformat(),
                        "worker_segment": self.segment,
                        "actual_observed_close": _clean_json(completed["Close"].iloc[-1]),
                        "actual_observed_timestamp_utc": completed.index.max().tz_convert("UTC").isoformat(),
                        "actual_observed_timestamp_ist": completed.index.max().isoformat(),
                        "actual_observed_is_horizon_outcome": False,
                        "target_timestamp_utc": (
                            target_ts.isoformat() if not pd.isna(target_ts) else None
                        ),
                        "outcome_status": (
                            "WAITING_FOR_HORIZON" if not pd.isna(target_ts)
                            else "NOT_VALIDATABLE"
                        ),
                        "actual_horizon_close": None,
                        "actual_horizon_timestamp_utc": None,
                        "website_news_context": self._news_context_for_symbol(root),
                    }
                    if prediction_failure:
                        record["prediction_health_status"] = "BLOCKED"
                        record["prediction_health_error"] = prediction_failure["error"]
                    else:
                        record["prediction_health_status"] = "PASS"
                        record["prediction_health_error"] = None
                    news_context = self._news_context_for_symbol(root)
                    record["website_news_context"] = news_context
                    record["news_overlay_signal"] = news_context.get(
                        "news_signal", "NEWS_INSUFFICIENT_DATA"
                    )
                    record["news_overlay_strength"] = news_context.get("news_signal_strength", 0.0)
                    record["news_overlay_reason"] = news_context.get("news_signal_reason")
                    record["news_overlay_only"] = True
                    prediction_rows.append(record)
                    snapshot["predictions"].append(record)
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
                            "signal_reason": row.get("signal_reason"),
                            "reason": row.get("reason"),
                            "decision_timestamp": row.get("decision_timestamp"),
                            "reference_close": row.get("reference_close"),
                            "expected_return": row.get("expected_return"),
                            "predicted_price": row.get("predicted_price"),
                            "price_forecast_status": row.get("price_forecast_status"),
                            "probability_up": row.get("probability_up"),
                            "feature_coverage": row.get("feature_coverage"),
                            "active_model_feature_count": row.get("active_model_feature_count"),
                            "missing_model_features": row.get("missing_model_features"),
                        }
                        for row in predictions
                    ],
                )
            except Exception as exc:
                summary["stock_fetch_errors"] += 1
                error_text = f"{type(exc).__name__}: {exc}"
                summary["errors"].append({
                    "symbol": root,
                    "error": error_text,
                })
                if snapshot is None:
                    snapshot_rows.append({
                        "record_type": "intraday_minute_observation",
                        "observation_id": f"{now:%Y%m%d_%H%M}|{root}",
                        "observation_timestamp_utc": now.tz_convert("UTC").isoformat(),
                        "observation_timestamp_ist": now.isoformat(),
                        "symbol": root,
                        "data_quality_status": "FETCH_ERROR",
                        "status_reason": error_text,
                        "predictions": [],
                    })
                else:
                    snapshot["data_quality_status"] = "PREDICTION_OR_PROCESSING_ERROR"
                    snapshot["status_reason"] = error_text
                _log("symbol_cycle_failed", symbol=root, error=error_text)

        self.last_minute_snapshot_rows = snapshot_rows
        self._append_rows(prediction_rows)
        self.last_cycle_summary = summary
        _log("cycle_completed", **summary)
        return summary

    def tick_once(self, now_value: datetime | None = None) -> dict[str, Any]:
        """Execute one prediction cycle for the persistent Cloud Run service.

        now_value is injectable for deterministic tests; production calls omit it.
        """
        now_dt = now_value or datetime.now(IST)
        now_dt = now_dt.replace(tzinfo=IST) if now_dt.tzinfo is None else now_dt.astimezone(IST)
        now = pd.Timestamp(now_dt)
        current_day = now.strftime("%Y/%m/%d")

        # Flush yesterday's data before replacing the active daily ledger path.
        if self.daily_ledger is not None:
            try:
                current_ledger_day = "/".join(self.daily_ledger.relative_to(self.root).parts[:3])
            except ValueError:
                current_ledger_day = current_day
            if current_ledger_day != current_day:
                old_flush = self._flush(final=True)
                if old_flush.get("status") != "OK":
                    _log("day_rollover_waiting_for_persistence", old_ledger_day=current_ledger_day,
                         current_day=current_day, flush=old_flush)
                    return {
                        "status": "PERSISTENCE_PENDING",
                        "reason": "Could not finish syncing the previous session; retrying before rollover.",
                        "flush": old_flush,
                    }
                self.session_frames.clear()
                self.symbols = []
                self.watchlist_meta = {}
                self.last_watchlist_refresh = 0.0
                self.cycle_count = 0
                self.daily_ledger = None
                self.run_status = "IDLE"
                self.last_tick_key = None

        self._ensure_paths(now)
        market_open = (
            is_nse_trading_day(now_dt.date())
            and OPEN <= now_dt.time() < CLOSE
        )
        if not market_open:
            local_ledger_has_rows = (
                self.daily_ledger is not None
                and self.daily_ledger.is_file()
                and self.daily_ledger.stat().st_size > 0
            )
            has_local_bar_frames = bool(self.session_frames)
            should_final_flush = (
                self.run_status in {"RUNNING", "PERSISTENCE_ERROR", "PERSISTENCE_PENDING"}
                or (
                    self.run_status in {"STARTING", "IDLE"}
                    and (local_ledger_has_rows or has_local_bar_frames)
                )
            )
            if should_final_flush:
                final_flush = self._flush(final=True)
                self.run_status = "STOPPED" if final_flush.get("status") == "OK" else "PERSISTENCE_ERROR"
                _log("market_session_worker_stopped", at_ist=now_dt.isoformat(), final_flush=final_flush)
            return {
                "status": "MARKET_CLOSED",
                "at_ist": now_dt.isoformat(),
                "nse_trading_day": is_nse_trading_day(now_dt.date()),
                "run_status": self.run_status,
            }

        minute_key = now.floor("min").isoformat()
        if self.last_tick_key == minute_key:
            return {"status": "DUPLICATE_MINUTE_SKIPPED", "minute": minute_key}

        try:
            self._hydrate_daily_ledger()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.run_status = "PERSISTENCE_ERROR"
            _log("daily_prediction_ledger_hydration_failed", minute=minute_key, error=error)
            return {"status": "PERSISTENCE_ERROR", "minute": minute_key, "error": error}

        self.last_tick_key = minute_key
        self.run_status = "RUNNING"
        self.cycle_count += 1
        _log("minute_tick_started", minute=minute_key, cycle=self.cycle_count,
             interval_seconds=self.interval_seconds)
        try:
            summary = self._cycle()
            self.last_cycle_summary = summary
            result: dict[str, Any] = {
                "status": "PASS" if not summary.get("errors") else "PARTIAL",
                "minute": minute_key,
                "cycle": self.cycle_count,
                "summary": summary,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.last_cycle_summary = {"status": "ERROR", "error": error}
            result = {"status": "ERROR", "minute": minute_key, "error": error}
            _log("minute_tick_failed", minute=minute_key, error=error)

        # Save the compact prediction + contemporaneous actual-price snapshot on
        # every minute tick. The larger bars and append-only daily ledger remain
        # batched to control Drive writes.
        if self.last_minute_snapshot_rows:
            try:
                snapshot_result = self._persist_minute_snapshot(now, result)
                result["minute_snapshot"] = snapshot_result
                if snapshot_result.get("status") != "OK":
                    result["status"] = "PERSISTENCE_ERROR"
                    self.run_status = "PERSISTENCE_ERROR"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                result["minute_snapshot"] = {"status": "ERROR", "error": error}
                result["status"] = "PERSISTENCE_ERROR"
                self.run_status = "PERSISTENCE_ERROR"
                _log("minute_snapshot_persist_failed", minute=minute_key, error=error)

        # Capture current market quotes for the full official NSE equity
        # universe independently of the smaller news-selected prediction watchlist.
        # The separate manifest reports coverage gaps and Drive persistence errors.
        try:
            from all_universe_minute_capture import capture_all_universe_minute_snapshot

            # Use the actual capture start time, not the tick-start time:
            # the model/watchlist cycle can cross a minute boundary.
            capture_now = pd.Timestamp.now(tz=IST)
            all_universe_capture = capture_all_universe_minute_snapshot(
                now=capture_now,
                worker_root=self.root,
                daily_ledger=self.daily_ledger,
                prediction_snapshots=self.last_minute_snapshot_rows,
            )
            result["all_universe_capture"] = all_universe_capture
            if all_universe_capture.get("status") != "OK":
                _log(
                    "all_universe_minute_capture_degraded",
                    minute=minute_key,
                    capture_status=all_universe_capture.get("status"),
                    expected_symbol_count=all_universe_capture.get("expected_symbol_count"),
                    quoted_symbol_count=all_universe_capture.get("quoted_symbol_count"),
                    missing_quote_count=all_universe_capture.get("missing_quote_count"),
                    snapshot_drive_path=all_universe_capture.get("snapshot_drive_path"),
                    error=all_universe_capture.get("error"),
                )
        except Exception as exc:
            capture_error = f"{type(exc).__name__}: {exc}"
            result["all_universe_capture"] = {
                "status": "ERROR",
                "error": capture_error,
                "cycle_minute_ist": minute_key,
            }
            _log(
                "all_universe_minute_capture_failed",
                minute=minute_key,
                error=capture_error,
            )

        # Persist immediately on any degraded tick so failed inference rows and
        # their health status survive even when the scheduler runs a one-off tick.
        # Cloud Run instances are ephemeral: force the final 15:29 IST prediction
        # batch to Drive in this same request instead of relying on a later request
        # that may be routed to a cold instance with no local ledger.
        final_minute_start = time(CLOSE.hour, max(0, CLOSE.minute - 1))
        final_session_tick = final_minute_start <= now_dt.time() < CLOSE
        should_flush = (
            self.cycle_count % self.flush_every == 0
            or result["status"] in {"ERROR", "PARTIAL", "PERSISTENCE_ERROR"}
            or final_session_tick
        )
        if should_flush:
            flush = self._flush(final=final_session_tick)
            result["flush"] = flush
            if flush.get("status") != "OK":
                result["status"] = "PERSISTENCE_ERROR"
                if final_session_tick:
                    self.run_status = "PERSISTENCE_ERROR"
            elif final_session_tick:
                self.run_status = "STOPPED"
                _log("market_session_final_flush_completed", final_flush=flush, at_ist=now_dt.isoformat())
        result["worker_status"] = self.run_status
        return result

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
            for symbol, (frame, source, local_path) in list(self.session_frames.items()):
                if frame is None or frame.empty or not local_path.exists():
                    continue
                relative_parts = local_path.relative_to(self.root).parts
                if len(relative_parts) < 4:
                    raise RuntimeError(f"Unexpected worker bar path: {local_path}")
                daily_dir = "/".join(relative_parts[:3])
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
            if self.daily_ledger is not None and self.daily_ledger.exists():
                ledger_parts = self.daily_ledger.relative_to(self.root).parts
                if len(ledger_parts) < 4:
                    raise RuntimeError(f"Unexpected worker ledger path: {self.daily_ledger}")
                ledger_day = "/".join(ledger_parts[:3])
                store.upload_or_update(
                    self.daily_ledger,
                    f"production_data/intraday_predictions/{ledger_day}/predictions_{self.segment}.jsonl",
                    mime_type="application/x-ndjson",
                )
                result["ledger_synced"] = True
            else:
                ledger_day = pd.Timestamp.now(tz=IST).strftime("%Y/%m/%d")

            local_status = self.root / "worker_status_latest.json"
            local_status.parent.mkdir(parents=True, exist_ok=True)
            local_status.write_text(
                json.dumps(_clean_json(self._status_payload("STOPPED" if final else "RUNNING")), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            store.upload_or_update(
                local_status,
                f"production_data/intraday_predictions/{ledger_day}/worker_status_{self.segment}.json",
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
