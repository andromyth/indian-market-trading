#!/usr/bin/env python3
"""Independent seven-session OOS validator for the Beta V2 intraday candidate.

Training is frozen at 2026-09-24. This validator evaluates only the seven
subsequent NSE trading sessions:
2026-09-25, 2026-09-28, 2026-09-29, 2026-09-30,
2026-10-01, 2026-10-05, 2026-10-06.

Production artifacts are never modified or promoted.

Data source:
  NSE charting platform historical API
    /v1/exchanges/symbolsDynamic
    /v1/charts/symbolHistoricalData

The seven-session OOS data is persisted separately under:
  selfdriving/beta_version2/intraday/oos_7day/

The validator resumes from completed raw and prediction Parquet files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from colab_build_all_nse_intraday_5m_multifactor import load_market
from delivery_training_universe import official_nse_symbols
from selfdriving.sd_intraday_fast_prepare import build_compact_symbol
from selfdriving.sd_intraday_v2 import _norm
from selfdriving.sd_walkforward import wilson_lower


HORIZONS = (5, 10, 15, 20, 25, 30)
OOS_SESSIONS = (
    "2026-09-25",
    "2026-09-28",
    "2026-09-29",
    "2026-09-30",
    "2026-10-01",
    "2026-10-05",
    "2026-10-06",
)
START_DATE = pd.Timestamp("2026-09-25 09:15", tz="Asia/Kolkata")
END_DATE = pd.Timestamp("2026-10-06 15:30", tz="Asia/Kolkata")
IST = "Asia/Kolkata"

SEARCH_URL = "https://charting.nseindia.com/v1/exchanges/symbolsDynamic"
HIST_URL = "https://charting.nseindia.com/v1/charts/symbolHistoricalData"

_thread_local = threading.local()


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is not None:
        return session
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/153.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Content-Type": "application/json",
            "Origin": "https://charting.nseindia.com",
            "Referer": "https://charting.nseindia.com/",
        }
    )
    _thread_local.session = session
    return session


def _request_json(url: str, payload: dict[str, Any], attempts: int = 5) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = _session().post(
                url,
                json=payload,
                timeout=(8, 30),
            )
            if response.status_code in (401, 403, 408, 425, 429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:250]}"
                )
            response.raise_for_status()
            obj = response.json()
            if not isinstance(obj, dict):
                raise RuntimeError(f"unexpected JSON type: {type(obj).__name__}")
            return obj
        except Exception as exc:
            last = exc
            if attempt < attempts:
                time.sleep(
                    min(20.0, 1.25 * (2 ** (attempt - 1)))
                    + random.random()
                )
    raise RuntimeError(f"request failed after {attempts} attempts: {last}") from last


def _resolve_symbol(symbol: str, segment: str = "EQ") -> dict[str, str]:
    obj = _request_json(
        SEARCH_URL,
        {"symbol": symbol, "segment": segment},
    )
    rows = obj.get("data")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"no charting match for {symbol}")

    wanted = symbol.strip().upper()
    choices: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_symbol = str(row.get("symbol", "")).strip()
        base_symbol = raw_symbol[:-3] if raw_symbol.upper().endswith("-EQ") else raw_symbol
        token = str(row.get("scripcode", "")).strip()
        if token:
            choices.append(
                {
                    "symbol": raw_symbol,
                    "scripcode": token,
                    "type": str(row.get("type", "Equity")),
                    "base": base_symbol.upper(),
                }
            )

    for row in choices:
        if row["base"] == wanted:
            return row
    for row in choices:
        if row["symbol"].upper() == wanted:
            return row
    raise RuntimeError(f"could not resolve NSE token for {symbol}")


def _fetch_history(
    symbol: str,
    token_info: dict[str, str],
    *,
    start: pd.Timestamp = START_DATE,
    end: pd.Timestamp = END_DATE,
) -> pd.DataFrame:
    payload = {
        "token": token_info["scripcode"],
        "fromDate": int(start.tz_convert("UTC").timestamp()),
        "toDate": int(end.tz_convert("UTC").timestamp()),
        "symbol": token_info["symbol"],
        "symbolType": token_info["type"],
        "chartType": "I",
        "timeInterval": 1,
    }
    obj = _request_json(HIST_URL, payload)
    rows = obj.get("data")
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()

    raw = pd.DataFrame(rows)
    raw = raw.rename(
        columns={
            "time": "ts",
            "timestamp": "ts",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
        }
    )
    required = {"ts", "open", "high", "low", "close"}
    if not required.issubset(raw.columns):
        raise RuntimeError(
            f"{symbol}: chart response missing columns {sorted(required - set(raw.columns))}"
        )

    ts_values = raw["ts"]
    if pd.api.types.is_numeric_dtype(ts_values):
        ts = pd.to_datetime(ts_values, unit="ms", utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(ts_values, utc=True, errors="coerce")

    frame = pd.DataFrame(
        {
            "ts": ts,
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
        }
    )
    frame["symbol"] = symbol
    frame = (
        frame.dropna(subset=["ts", "open", "high", "low", "close"])
        .drop_duplicates("ts", keep="last")
        .sort_values("ts")
    )

    local = frame["ts"].dt.tz_convert(IST)
    minute = local.dt.hour * 60 + local.dt.minute
    frame = frame[
        local.dt.strftime("%Y-%m-%d").isin(OOS_SESSIONS)
        & (minute >= 9 * 60 + 15)
        & (minute < 15 * 60 + 30)
    ].copy()
    return frame.reset_index(drop=True)


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _load_seed_symbol(root: Path, symbol: str) -> pd.DataFrame:
    seed_dir = (
        root
        / "production_data/nse_official/intraday_seed/olo_minute_archive/symbols"
        / f"symbol={symbol}"
    )
    parts: list[pd.DataFrame] = []
    if not seed_dir.exists():
        return pd.DataFrame()
    for path in sorted(seed_dir.rglob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
            if not frame.empty:
                parts.append(frame)
        except Exception:
            continue
    if not parts:
        return pd.DataFrame()
    return _norm(pd.concat(parts, ignore_index=True))


def _load_seed_market(root: Path) -> pd.DataFrame:
    try:
        return _norm(load_market(root))
    except Exception:
        return pd.DataFrame()


def _load_candidate(
    models_root: Path, horizon: int
) -> tuple[Any, dict[str, Any], Path]:
    registry_path = models_root / f"h{horizon}" / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"candidate registry missing: {registry_path}")

    state = json.loads(registry_path.read_text(encoding="utf-8"))
    champion = state.get("champion")
    if not champion:
        raise RuntimeError(f"no candidate registered in {registry_path}")

    artifact = models_root / f"h{horizon}" / str(champion["file"])
    if not artifact.exists():
        raise FileNotFoundError(f"candidate artifact missing: {artifact}")

    bundle = joblib.load(artifact)
    thresholds = champion.get("thresholds") or {}
    return bundle, thresholds, artifact


def _signal(probability: float, thresholds: dict[str, Any]) -> str:
    if thresholds.get("abstain_all", True):
        return "ABSTAIN"
    buy_p = float(thresholds["buy_p"])
    sell_p = float(thresholds["sell_p"])
    if probability >= buy_p:
        return "BUY"
    if probability <= sell_p:
        return "SELL"
    return "ABSTAIN"


def _prepare_prediction(
    root: Path,
    raw_root: Path,
    symbol: str,
    bundles: dict[int, Any],
    thresholds: dict[int, dict[str, Any]],
    market: pd.DataFrame,
) -> pd.DataFrame:
    oos_path = raw_root / f"symbol={symbol}.parquet"
    if not oos_path.exists():
        return pd.DataFrame()

    oos = _norm(pd.read_parquet(oos_path))
    if oos.empty:
        return pd.DataFrame()

    seed = _load_seed_symbol(root, symbol)
    combined = pd.concat([seed, oos], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    combined = (
        combined.sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )
    cutoff = END_DATE.tz_convert("UTC")
    combined = combined[combined["ts"] <= cutoff].copy()
    if len(combined) < 80:
        return pd.DataFrame()

    compact = build_compact_symbol(
        combined,
        market,
        HORIZONS,
        max_decisions=None,
    )
    if compact.empty:
        return pd.DataFrame()

    decision_ts = pd.to_datetime(
        compact["decision_timestamp"], utc=True, errors="coerce"
    )
    local = decision_ts.dt.tz_convert(IST)
    mask = (
        local.dt.strftime("%Y-%m-%d").isin(OOS_SESSIONS)
        & (local >= START_DATE)
        & (local <= END_DATE)
    )
    compact = compact.loc[mask].copy()
    if compact.empty:
        return pd.DataFrame()

    result = compact[
        ["symbol", "decision_timestamp", "date", "reference_close"]
    ].copy()

    for horizon in HORIZONS:
        bundle = bundles[horizon]
        probability = np.asarray(
            bundle.predict_proba_up(compact[bundle.features]),
            dtype=float,
        )
        actual = pd.to_numeric(
            compact[f"fwd_ret_{horizon}m"], errors="coerce"
        ).to_numpy()
        result[f"p_{horizon}m"] = probability
        result[f"fwd_ret_{horizon}m"] = actual
        result[f"signal_{horizon}m"] = [
            _signal(float(p), thresholds[horizon])
            if np.isfinite(p)
            else "ABSTAIN"
            for p in probability
        ]

    return result.reset_index(drop=True)


def _horizon_report(
    frame: pd.DataFrame,
    horizon: int,
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    pcol = f"p_{horizon}m"
    rcol = f"fwd_ret_{horizon}m"
    scol = f"signal_{horizon}m"

    detail = frame[
        ["symbol", "decision_timestamp", "date", pcol, rcol, scol]
    ].copy()
    detail = detail.rename(
        columns={pcol: "p", rcol: "actual_return", scol: "signal"}
    )
    detail["decision_timestamp"] = pd.to_datetime(
        detail["decision_timestamp"], utc=True, errors="coerce"
    )
    detail["date"] = pd.to_datetime(
        detail["date"], errors="coerce"
    ).dt.normalize()
    detail["p"] = pd.to_numeric(detail["p"], errors="coerce")
    detail["actual_return"] = pd.to_numeric(
        detail["actual_return"], errors="coerce"
    )
    detail = detail.dropna(
        subset=["decision_timestamp", "date", "p", "actual_return"]
    )

    # Market-neutral realised target, measured at the exact decision timestamp.
    cross_median = detail.groupby("decision_timestamp")[
        "actual_return"
    ].transform("median")
    detail["actual_excess_return"] = (
        detail["actual_return"] - cross_median
    )
    detail["actual_y"] = (
        detail["actual_excess_return"] > 0
    ).astype(int)

    acted = detail["signal"].isin(["BUY", "SELL"])
    detail["correct"] = np.where(
        detail["signal"].eq("BUY"),
        detail["actual_y"].eq(1),
        np.where(
            detail["signal"].eq("SELL"),
            detail["actual_y"].eq(0),
            False,
        ),
    )
    detail["signed_excess_return"] = np.where(
        detail["signal"].eq("BUY"),
        detail["actual_excess_return"],
        np.where(
            detail["signal"].eq("SELL"),
            -detail["actual_excess_return"],
            0.0,
        ),
    )
    detail["signed_absolute_return"] = np.where(
        detail["signal"].eq("BUY"),
        detail["actual_return"],
        np.where(
            detail["signal"].eq("SELL"),
            -detail["actual_return"],
            0.0,
        ),
    )

    n = int(len(detail))
    signal_count = int(acted.sum())
    hit_count = int(detail.loc[acted, "correct"].sum()) if signal_count else 0
    signal_accuracy = (
        float(hit_count / signal_count) if signal_count else np.nan
    )
    signal_wilson_lb = (
        wilson_lower(hit_count, signal_count)
        if signal_count
        else np.nan
    )

    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    auc = (
        float(roc_auc_score(detail["actual_y"], detail["p"]))
        if detail["actual_y"].nunique() > 1
        else np.nan
    )
    brier = float(
        brier_score_loss(detail["actual_y"], detail["p"])
    )
    logloss = float(
        log_loss(
            detail["actual_y"],
            np.clip(detail["p"], 1e-6, 1 - 1e-6),
        )
    )

    base_rate = (
        max(
            float(detail["actual_y"].mean()),
            float(1.0 - detail["actual_y"].mean()),
        )
        if n
        else np.nan
    )

    per_day = (
        detail.assign(
            signal_count=acted.astype(int),
            hit=detail["correct"].astype(int),
        )
        .groupby("date")
        .agg(
            rows=("symbol", "size"),
            signals=("signal_count", "sum"),
            hits=("hit", "sum"),
            signed_excess_return=("signed_excess_return", "sum"),
            signed_absolute_return=("signed_absolute_return", "sum"),
        )
        .reset_index()
    )
    per_day["signal_accuracy"] = np.where(
        per_day["signals"] > 0,
        per_day["hits"] / per_day["signals"],
        np.nan,
    )
    per_day["cumulative_excess_return"] = (
        per_day["signed_excess_return"].cumsum()
    )
    per_day["cumulative_absolute_return"] = (
        per_day["signed_absolute_return"].cumsum()
    )

    # Cost policy matches the training run: 12 bps per leg, 24 bps round trip.
    gross_bps = float(detail["signed_excess_return"].sum() * 1e4)
    net_bps = float(gross_bps - signal_count * 24.0)

    report = {
        "horizon_min": horizon,
        "rows": n,
        "dates": int(detail["date"].nunique()),
        "symbols": int(detail["symbol"].nunique()),
        "expected_oos_dates": list(OOS_SESSIONS),
        "all_7_sessions_present": bool(
            set(
                detail["date"].dt.strftime("%Y-%m-%d").dropna().unique()
            )
            == set(OOS_SESSIONS)
        ),
        "signals": signal_count,
        "coverage": float(signal_count / n) if n else np.nan,
        "signal_accuracy": signal_accuracy,
        "signal_wilson_lb": signal_wilson_lb,
        "auc": auc,
        "brier": brier,
        "logloss": logloss,
        "base_rate_majority": base_rate,
        "lift_vs_base_pp": (
            float((signal_accuracy - base_rate) * 100)
            if signal_count
            else np.nan
        ),
        "mean_actual_excess_return": (
            float(detail["actual_excess_return"].mean()) if n else np.nan
        ),
        "mean_signed_excess_return": (
            float(detail.loc[acted, "signed_excess_return"].mean())
            if signal_count
            else np.nan
        ),
        "sum_signed_excess_return": float(
            detail["signed_excess_return"].sum()
        ),
        "gross_signal_bps": gross_bps,
        "assumed_cost_bps_per_leg": 12.0,
        "net_signal_bps": net_bps,
        "thresholds": thresholds,
        "per_day": per_day.to_dict(orient="records"),
    }
    return report, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="/content/indian-market-trading",
    )
    ap.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/Indian_Market_Research",
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--download-only", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    drive_root = Path(args.drive_root)
    out = (
        drive_root
        / "selfdriving/beta_version2/intraday/oos_7day"
    )
    raw_root = out / "raw"
    prediction_root = out / "predictions"
    detail_root = out / "horizon_details"
    models_root = (
        drive_root / "selfdriving/beta_version2/intraday/models"
    )

    raw_root.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)
    detail_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "OOS_MANIFEST.json"
    state: dict[str, Any] = {
        "status": "RUNNING",
        "version": "beta-v2-intraday-7day-oos-20261007",
        "oos_sessions": list(OOS_SESSIONS),
        "train_cutoff": "2026-09-24",
        "source": "NSE_CHARTING_API",
        "source_endpoints": [SEARCH_URL, HIST_URL],
        "candidate_models_root": str(models_root),
        "production_promotion": False,
        "production_artifacts_changed": False,
    }

    def save_state() -> None:
        manifest_path.write_text(
            json.dumps(state, indent=2, default=str),
            encoding="utf-8",
        )

    coverage_file = (
        drive_root
        / "selfdriving/beta_version2/intraday/BETA_V2_INTRADAY_COVERAGE.parquet"
    )
    if coverage_file.exists():
        try:
            universe = sorted(
                pd.read_parquet(
                    coverage_file, columns=["symbol"]
                )["symbol"]
                .astype(str)
                .str.upper()
                .str.strip()
                .dropna()
                .unique()
                .tolist()
            )
        except Exception:
            universe = official_nse_symbols(force_refresh=False)
    else:
        universe = official_nse_symbols(force_refresh=False)

    if args.max_symbols > 0:
        universe = universe[: args.max_symbols]

    state["evaluation_universe"] = len(universe)
    state["candidate_only"] = True
    save_state()

    token_cache_path = out / "symbol_tokens.json"
    if token_cache_path.exists():
        try:
            token_cache = json.loads(
                token_cache_path.read_text(encoding="utf-8")
            )
        except Exception:
            token_cache = {}
    else:
        token_cache = {}

    if not args.evaluate_only:
        pending = [
            symbol
            for symbol in universe
            if not (raw_root / f"symbol={symbol}.parquet").exists()
        ]
        print(
            f"OOS DOWNLOAD: universe={len(universe):,}; "
            f"pending={len(pending):,}; sessions={len(OOS_SESSIONS)}",
            flush=True,
        )

        def download_one(symbol: str):
            info = token_cache.get(symbol)
            if not info:
                info = _resolve_symbol(symbol, "EQ")
            frame = _fetch_history(symbol, info)
            return symbol, info, frame

        results: dict[str, dict[str, Any]] = {}
        worker_count = max(1, min(int(args.workers), 2))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(download_one, symbol): symbol
                for symbol in pending
            }
            for i, future in enumerate(as_completed(futures), start=1):
                symbol = futures[future]
                try:
                    symbol, info, frame = future.result()
                    token_cache[symbol] = info
                    if frame.empty:
                        results[symbol] = {
                            "status": "EMPTY",
                            "rows": 0,
                        }
                    else:
                        _write_atomic(
                            frame,
                            raw_root / f"symbol={symbol}.parquet",
                        )
                        results[symbol] = {
                            "status": "OK",
                            "rows": int(len(frame)),
                            "dates": int(
                                frame["ts"]
                                .dt.tz_convert(IST)
                                .dt.strftime("%Y-%m-%d")
                                .nunique()
                            ),
                        }
                except Exception as exc:
                    results[symbol] = {
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                if i % 25 == 0 or i == len(pending):
                    token_cache_path.write_text(
                        json.dumps(token_cache, indent=2),
                        encoding="utf-8",
                    )
                    print(
                        f"OOS DOWNLOAD PROGRESS: {i:,}/{len(pending):,}",
                        flush=True,
                    )

        token_cache_path.write_text(
            json.dumps(token_cache, indent=2),
            encoding="utf-8",
        )
        state["download"] = {
            "pending_at_start": len(pending),
            "ok": int(
                sum(v["status"] == "OK" for v in results.values())
            ),
            "empty": int(
                sum(v["status"] == "EMPTY" for v in results.values())
            ),
            "errors": int(
                sum(v["status"] == "ERROR" for v in results.values())
            ),
            "raw_files_now": int(
                len(list(raw_root.glob("symbol=*.parquet")))
            ),
        }
        save_state()

    if args.download_only:
        state["status"] = "DOWNLOAD_COMPLETE"
        save_state()
        print(json.dumps(state, indent=2, default=str))
        return 0

    # Use the same NIFTY history source for market-relative features.
    # Some NSE charting responses do not expose NIFTY 50 through the generic
    # IDX search route even though the equity history route works. For OOS,
    # therefore prefer the repository's existing historical NIFTY series and
    # only fall back to direct charting API retrieval when necessary.
    market_path = raw_root / "NIFTY50.parquet"
    market_oos = pd.DataFrame()

    if market_path.exists():
        try:
            market_oos = _norm(pd.read_parquet(market_path))
        except Exception:
            market_oos = pd.DataFrame()

    if market_oos.empty:
        seed_market = _load_seed_market(root)
        if not seed_market.empty:
            seed_local = seed_market[
                seed_market["ts"]
                .dt.tz_convert(IST)
                .dt.strftime("%Y-%m-%d")
                .isin(OOS_SESSIONS)
            ].copy()
            if not seed_local.empty:
                market_oos = seed_local.copy()

    if market_oos.empty:
        # Final fallback: try the dedicated index search name variants.
        candidates = ("NIFTY 50", "NIFTY50", "NIFTY")
        info = token_cache.get("__NIFTY_50__")
        for candidate in candidates:
            try:
                info = info or _resolve_symbol(candidate, "IDX")
                nifty = _fetch_history(candidate, info)
                if not nifty.empty:
                    market_oos = nifty
                    break
            except Exception:
                continue

        if market_oos.empty:
            raise RuntimeError(
                "NIFTY 50 OOS history unavailable from seed, cached OOS file, "
                "or NSE charting index fallback"
            )

        token_cache["__NIFTY_50__"] = info
        _write_atomic(market_oos, market_path)
        token_cache_path.write_text(
            json.dumps(token_cache, indent=2),
            encoding="utf-8",
        )
    else:
        _write_atomic(market_oos, market_path)

    seed_market = _load_seed_market(root)
    market_parts = [x for x in (seed_market, market_oos) if not x.empty]
    if not market_parts:
        raise RuntimeError("no NIFTY market history available")
    market = (
        pd.concat(market_parts, ignore_index=True)
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
    )
    market = market[
        market["ts"]
        <= END_DATE.tz_convert("UTC")
    ].reset_index(drop=True)

    bundles: dict[int, Any] = {}
    thresholds: dict[int, dict[str, Any]] = {}
    artifacts: dict[int, str] = {}
    for horizon in HORIZONS:
        bundle, thr, artifact = _load_candidate(
            models_root, horizon
        )
        bundles[horizon] = bundle
        thresholds[horizon] = thr
        artifacts[horizon] = str(artifact)

    state["candidate_artifacts"] = artifacts
    save_state()

    pending_predictions = [
        symbol
        for symbol in universe
        if not (
            prediction_root / f"symbol={symbol}.parquet"
        ).exists()
    ]
    print(
        f"OOS EVALUATION: universe={len(universe):,}; "
        f"pending={len(pending_predictions):,}",
        flush=True,
    )

    def predict_one(symbol: str) -> tuple[str, pd.DataFrame]:
        return (
            symbol,
            _prepare_prediction(
                root,
                raw_root,
                symbol,
                bundles,
                thresholds,
                market,
            ),
        )

    worker_count = max(1, min(int(args.workers), 2))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(predict_one, symbol): symbol
            for symbol in pending_predictions
        }
        for i, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                symbol, prediction = future.result()
                if not prediction.empty:
                    _write_atomic(
                        prediction,
                        prediction_root
                        / f"symbol={symbol}.parquet",
                    )
            except Exception as exc:
                print(
                    f"OOS EVAL ERROR {symbol}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if (
                i % 25 == 0
                or i == len(pending_predictions)
            ):
                print(
                    f"OOS EVALUATION PROGRESS: "
                    f"{i:,}/{len(pending_predictions):,}",
                    flush=True,
                )

    prediction_files = sorted(
        prediction_root.glob("symbol=*.parquet")
    )
    if not prediction_files:
        raise RuntimeError("no OOS prediction files were produced")

    reports: dict[str, Any] = {}
    blocked = False

    for horizon in HORIZONS:
        pieces: list[pd.DataFrame] = []
        cols = [
            "symbol",
            "decision_timestamp",
            "date",
            f"p_{horizon}m",
            f"fwd_ret_{horizon}m",
            f"signal_{horizon}m",
        ]
        for path in prediction_files:
            try:
                part = pd.read_parquet(path, columns=cols)
                if not part.empty:
                    pieces.append(part)
            except Exception:
                continue

        if not pieces:
            reports[str(horizon)] = {
                "status": "BLOCKED",
                "reason": "no usable OOS prediction rows",
            }
            blocked = True
            continue

        combined = pd.concat(pieces, ignore_index=True)
        report, detail = _horizon_report(
            combined,
            horizon,
            thresholds[horizon],
        )
        report["status"] = (
            "COMPLETE"
            if report["all_7_sessions_present"]
            else "INCOMPLETE_7_SESSION_WINDOW"
        )
        if not report["all_7_sessions_present"]:
            blocked = True

        reports[str(horizon)] = report
        detail.to_parquet(
            detail_root / f"h{horizon}.parquet",
            index=False,
        )

        print(
            f"OOS +{horizon}M: dates={report['dates']} "
            f"rows={report['rows']:,} "
            f"signals={report['signals']:,} "
            f"accuracy={report['signal_accuracy']:.4f} "
            f"AUC={report['auc']:.4f} "
            f"net_bps={report['net_signal_bps']:.2f}",
            flush=True,
        )

    state["status"] = (
        "OOS_INCOMPLETE"
        if blocked
        else "OOS_COMPLETE"
    )
    state["results"] = reports
    report_payload = {
        "version": "beta-v2-intraday-7day-oos-20261007",
        "candidate_only": True,
        "production_changed": False,
        "train_cutoff": "2026-09-24",
        "oos_sessions": list(OOS_SESSIONS),
        "results": reports,
    }
    (out / "OOS_REPORT.json").write_text(
        json.dumps(
            report_payload,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    state["reports"] = {
        "manifest": str(manifest_path),
        "report_json": str(out / "OOS_REPORT.json"),
        "detail_dir": str(detail_root),
    }
    save_state()

    print(json.dumps(report_payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
