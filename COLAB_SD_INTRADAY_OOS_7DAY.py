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
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None

from colab_build_all_nse_intraday_5m_multifactor import load_market
from delivery_training_universe import official_nse_symbols
from nse_data_layer import fetch_index_history
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

# Public NSE charting clients use a short-lived CSRF token in addition to
# browser cookies. Keep the bootstrap token on the thread-local session.
_CSRF_HEADER_NAMES = ("x-nse-token", "x-csrftoken", "x-csrf-token")

_thread_local = threading.local()


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is not None:
        return session

    # NSE's current charting service expects browser-like transport. The
    # reference OpenChart client uses curl_cffi Chrome impersonation and
    # initializes both the NSE and charting domains before API calls.
    if curl_requests is not None:
        session = curl_requests.Session(impersonate="chrome120")
    else:
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
            "Origin": "https://charting.nseindia.com",
            "Referer": "https://charting.nseindia.com/",
        }
    )

    # Prime both domains so charting cookies / anti-bot context are present.
    bootstrap_pages = (
        ("https://www.nseindia.com", (8, 20)),
        ("https://charting.nseindia.com/", (8, 15)),
    )
    for bootstrap_url, timeout in bootstrap_pages:
        try:
            response = session.get(bootstrap_url, timeout=timeout)
            # Some NSE deployments expose a CSRF token in a response header.
            for name in _CSRF_HEADER_NAMES:
                token = response.headers.get(name)
                if token:
                    session.headers["X-NSE-Token"] = token
                    break
            if not session.headers.get("X-NSE-Token"):
                # Also accept the common cookie names used by the charting UI.
                for cookie_name in ("nseToken", "csrfToken", "XSRF-TOKEN"):
                    token = session.cookies.get(cookie_name)
                    if token:
                        session.headers["X-NSE-Token"] = token
                        break
        except Exception:
            pass

    _thread_local.session = session
    return session


def _request_json(
    url: str,
    payload: dict[str, Any],
    attempts: int = 5,
) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            # NSE charting accepts the request as a JSON POST. Include
            # the browser-origin headers and CSRF token when available.
            response = _session().post(
                url,
                json=payload,
                headers={
                    "Origin": "https://charting.nseindia.com",
                    "Referer": "https://charting.nseindia.com/",
                    **(
                        {"X-NSE-Token": _session().headers["X-NSE-Token"]}
                        if _session().headers.get("X-NSE-Token")
                        else {}
                    ),
                },
                timeout=(8, 30),
            )
            if response.status_code in (
                401, 403, 408, 425, 429, 500, 502, 503, 504
            ):
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:400]}"
                )
            response.raise_for_status()
            obj = response.json()
            if not isinstance(obj, dict):
                raise RuntimeError(
                    f"unexpected JSON type: {type(obj).__name__}"
                )
            return obj
        except Exception as exc:
            last = exc
            if attempt < attempts:
                time.sleep(
                    min(20.0, 1.25 * (2 ** (attempt - 1)))
                    + random.random()
                )
    raise RuntimeError(
        f"request failed after {attempts} attempts: {last}"
    ) from last


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
    allow_session_fallback: bool = False,
) -> pd.DataFrame:
    def fetch_window(
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> pd.DataFrame:
        payload = {
            "token": token_info["scripcode"],
            "fromDate": int(window_start.tz_convert("UTC").timestamp()),
            "toDate": int(window_end.tz_convert("UTC").timestamp()),
            "symbol": token_info["symbol"],
            "symbolType": token_info["type"],
            "chartType": "I",
            "timeInterval": 1,
        }
        obj = _request_json(HIST_URL, payload)
        status = obj.get("status")
        rows = obj.get("data")
        if not isinstance(rows, list):
            raise RuntimeError(
                f"{symbol}: chart API returned non-list data; "
                f"status={status!r} keys={sorted(obj.keys())} "
                f"message={obj.get('message')!r} error={obj.get('error')!r}"
            )
        if not rows:
            raise RuntimeError(
                f"{symbol}: chart API returned empty data; "
                f"status={status!r} keys={sorted(obj.keys())} "
                f"message={obj.get('message')!r} error={obj.get('error')!r}"
            )

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
                f"{symbol}: chart response missing columns "
                f"{sorted(required - set(raw.columns))}"
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

    # Prefer one request for the entire OOS window. This is the normal,
    # fast path for equities. Per-session fallback is intentionally opt-in
    # because applying it to 2,599 symbols can multiply API traffic by 8x.
    errors: list[str] = []
    try:
        frame = fetch_window(start, end)
    except Exception as exc:
        frame = pd.DataFrame()
        errors.append(f"window={type(exc).__name__}: {exc}")

    required_dates = set(OOS_SESSIONS)
    if not frame.empty:
        observed = set(
            frame["ts"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d").unique()
        )
        if required_dates.issubset(observed):
            return frame

    if not allow_session_fallback:
        detail = " | ".join(errors[-4:])
        observed_text = (
            sorted(observed)
            if not frame.empty
            else []
        )
        raise RuntimeError(
            f"{symbol}: OOS history incomplete from single window; "
            f"rows={len(frame):,} observed_sessions={observed_text}"
            + (f"; diagnostics={detail}" if detail else "")
        )

    # Deterministic per-session fallback. This is reserved for NIFTY 50,
    # where the multi-session minute request has previously returned empty.
    parts: list[pd.DataFrame] = [frame] if not frame.empty else []
    for session_date in OOS_SESSIONS:
        try:
            day_start = pd.Timestamp(
                f"{session_date} 09:15",
                tz=IST,
            )
            day_end = pd.Timestamp(
                f"{session_date} 15:30",
                tz=IST,
            )
            day = fetch_window(day_start, day_end)
            if not day.empty:
                parts.append(day)
        except Exception as exc:
            errors.append(
                f"session={session_date}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not parts:
        detail = " | ".join(errors[-8:])
        raise RuntimeError(
            f"{symbol}: no OOS history after window and per-session fallback"
            + (f"; diagnostics={detail}" if detail else "")
        )

    combined = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("ts", keep="last")
        .sort_values("ts")
        .reset_index(drop=True)
    )
    observed = set(
        combined["ts"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d").unique()
    )
    missing = [d for d in OOS_SESSIONS if d not in observed]
    if missing:
        detail = " | ".join(errors[-8:])
        raise RuntimeError(
            f"{symbol}: OOS history incomplete after per-session fallback; "
            f"missing_sessions={missing}"
            + (f"; diagnostics={detail}" if detail else "")
        )
    return combined


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
    ap.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent NSE charting workers; capped at 6.",
    )
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

    # The coverage checkpoint is a diagnostic/eligibility table, not the OOS
    # evaluation universe. The earlier implementation accidentally reduced
    # OOS to the 428 excluded/unavailable symbols. Evaluate against the same
    # full official NSE roster used by candidate training.
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
            frame = _fetch_history(
                symbol,
                info,
                allow_session_fallback=False,
            )
            return symbol, info, frame

        results: dict[str, dict[str, Any]] = {}
        worker_count = max(1, min(int(args.workers), 6))
        started_at = time.monotonic()
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
                    elapsed = max(0.001, time.monotonic() - started_at)
                    rate = i / elapsed
                    remaining = max(0, len(pending) - i)
                    eta_min = remaining / rate / 60.0 if rate > 0 else float("inf")
                    print(
                        f"OOS DOWNLOAD PROGRESS: {i:,}/{len(pending):,} "
                        f"rate={rate:.2f}/s eta={eta_min:.1f}m",
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

    # Prefer an already-complete cached minute NIFTY file. Otherwise obtain
    # official NSE daily index history for provenance and diagnostics, then
    # use the charting IDX lane for the required minute-level feature series.
    # NIFTY minute market context for OOS. Reuse a complete cached
    # minute series; otherwise resolve the NSE charting IDX instrument.
    market_path = raw_root / "NIFTY50.parquet"
    market_oos = pd.DataFrame()

    if market_path.exists():
        try:
            cached = _norm(pd.read_parquet(market_path))
            dates = (
                cached["ts"].dt.tz_convert(IST).dt.strftime("%Y-%m-%d")
                if not cached.empty else pd.Series(dtype=str)
            )
            if set(OOS_SESSIONS).issubset(set(dates.unique())):
                market_oos = cached
        except Exception:
            market_oos = pd.DataFrame()

    daily_diag = pd.DataFrame()
    try:
        daily_diag = fetch_index_history(
            "^NSEI",
            start=OOS_SESSIONS[0],
            end=OOS_SESSIONS[-1],
            sync_drive=True,
        )
        if not daily_diag.empty:
            print(
                "NIFTY DAILY OFFICIAL HISTORY: "
                f"rows={len(daily_diag):,} "
                f"range={daily_diag['date'].min().date()}..{daily_diag['date'].max().date()}",
                flush=True,
            )
    except Exception as exc:
        print(
            "NIFTY DAILY OFFICIAL HISTORY WARNING: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

    if market_oos.empty:
        candidates = ("NIFTY 50", "NIFTY50", "NIFTY")
        errors: list[str] = []
        info = token_cache.get("__NIFTY_50__")

        for candidate in candidates:
            try:
                candidate_info = (
                    info if info and candidate == "NIFTY 50"
                    else _resolve_symbol(candidate, "IDX")
                )
                print(
                    "NIFTY IDX RESOLVED: "
                    f"query={candidate} "
                    f"symbol={candidate_info.get('symbol')} "
                    f"type={candidate_info.get('type')} "
                    f"token={candidate_info.get('scripcode')}",
                    flush=True,
                )
                nifty = _fetch_history(
                    candidate,
                    candidate_info,
                    allow_session_fallback=True,
                )
                dates = (
                    set(
                        nifty["ts"].dt.tz_convert(IST)
                        .dt.strftime("%Y-%m-%d")
                        .unique()
                    )
                    if not nifty.empty else set()
                )
                missing = [d for d in OOS_SESSIONS if d not in dates]
                if dates and not missing:
                    market_oos = nifty
                    info = candidate_info
                    break
                errors.append(
                    f"{candidate}: rows={len(nifty):,} missing_sessions={missing}"
                )
            except Exception as exc:
                errors.append(
                    f"{candidate}: {type(exc).__name__}: {exc}"
                )

        if market_oos.empty:
            # Independent OOS fallback for the market-context series.
            # Yahoo Finance exposes recent 1-minute NIFTY 50 history and the
            # requested seven sessions are within its intraday retention
            # window. This fallback is used only for the market feature
            # series; equity OOS prices remain NSE charting data.
            try:
                import yfinance as yf

                # Yahoo Finance limits 1-minute history requests to a short
                # calendar window. The seven OOS sessions span more than
                # that limit, so request each exact session independently.
                yahoo_parts: list[pd.DataFrame] = []
                print(
                    "NIFTY OOS FALLBACK: requesting Yahoo Finance ^NSEI "
                    "1-minute market context in per-session chunks",
                    flush=True,
                )

                for session_date in OOS_SESSIONS:
                    try:
                        session_start = pd.Timestamp(
                            f"{session_date} 09:15",
                            tz=IST,
                        )
                        session_end = pd.Timestamp(
                            f"{session_date} 15:30",
                            tz=IST,
                        )
                        yf_start = (
                            session_start
                            .tz_convert("UTC")
                            .tz_localize(None)
                        )
                        yf_end = (
                            session_end
                            .tz_convert("UTC")
                            .tz_localize(None)
                        )
                        chunk = yf.download(
                            "^NSEI",
                            start=yf_start,
                            end=yf_end,
                            interval="1m",
                            auto_adjust=False,
                            progress=False,
                            threads=False,
                        )

                        if chunk is None or chunk.empty:
                            errors.append(
                                f"Yahoo ^NSEI {session_date}: empty response"
                            )
                            continue

                        if isinstance(chunk.columns, pd.MultiIndex):
                            chunk.columns = [
                                str(col[0]).lower()
                                for col in chunk.columns
                            ]
                        else:
                            chunk.columns = [
                                str(col).lower()
                                for col in chunk.columns
                            ]

                        chunk.index = pd.to_datetime(
                            chunk.index,
                            utc=True,
                            errors="coerce",
                        )
                        chunk = chunk.reset_index()

                        if "ts" not in chunk.columns:
                            chunk = chunk.rename(
                                columns={chunk.columns[0]: "ts"}
                            )

                        required = {
                            "ts",
                            "open",
                            "high",
                            "low",
                            "close",
                        }
                        if not required.issubset(chunk.columns):
                            errors.append(
                                f"Yahoo ^NSEI {session_date}: "
                                f"missing_columns="
                                f"{sorted(required - set(chunk.columns))}"
                            )
                            continue

                        part = chunk[
                            ["ts", "open", "high", "low", "close"]
                        ].copy()
                        part["symbol"] = "NIFTY 50"
                        part["ts"] = pd.to_datetime(
                            part["ts"],
                            utc=True,
                            errors="coerce",
                        )

                        for col in ("open", "high", "low", "close"):
                            part[col] = pd.to_numeric(
                                part[col],
                                errors="coerce",
                            )

                        part = part.dropna(
                            subset=[
                                "ts",
                                "open",
                                "high",
                                "low",
                                "close",
                            ]
                        )

                        local = part["ts"].dt.tz_convert(IST)
                        minute = local.dt.hour * 60 + local.dt.minute
                        part = part[
                            local.dt.strftime("%Y-%m-%d").eq(session_date)
                            & (minute >= 9 * 60 + 15)
                            & (minute < 15 * 60 + 30)
                        ].copy()

                        if part.empty:
                            errors.append(
                                f"Yahoo ^NSEI {session_date}: "
                                "no regular-session minute rows"
                            )
                            continue

                        yahoo_parts.append(part)
                        print(
                            "NIFTY OOS FALLBACK SESSION OK: "
                            f"{session_date} rows={len(part):,}",
                            flush=True,
                        )
                    except Exception as exc:
                        errors.append(
                            f"Yahoo ^NSEI {session_date}: "
                            f"{type(exc).__name__}: {exc}"
                        )

                if yahoo_parts:
                    market_oos = (
                        pd.concat(yahoo_parts, ignore_index=True)
                        .drop_duplicates("ts", keep="last")
                        .sort_values("ts")
                        .reset_index(drop=True)
                    )

                if not market_oos.empty:
                    observed = set(
                        market_oos["ts"]
                        .dt.tz_convert(IST)
                        .dt.strftime("%Y-%m-%d")
                        .unique()
                    )
                    missing = [
                        d for d in OOS_SESSIONS if d not in observed
                    ]
                    if missing:
                        market_oos = pd.DataFrame()
                        errors.append(
                            "Yahoo ^NSEI: missing_sessions="
                            + repr(missing)
                        )
                    else:
                        print(
                            "NIFTY OOS FALLBACK OK: "
                            f"rows={len(market_oos):,} "
                            f"sessions={len(observed)} "
                            "source=YAHOO_FINANCE",
                            flush=True,
                        )
                        state["nifty_market_source"] = (
                            "YAHOO_FINANCE_FALLBACK"
                        )
                        save_state()
            except Exception as exc:
                errors.append(
                    "Yahoo ^NSEI: "
                    f"{type(exc).__name__}: {exc}"
                )

        if market_oos.empty:
            raise RuntimeError(
                "NIFTY 50 OOS history unavailable from NSE charting "
                "or Yahoo Finance fallback. Attempt diagnostics: "
                + " | ".join(errors)
            )

        token_cache["__NIFTY_50__"] = info
        _write_atomic(market_oos, market_path)
        token_cache_path.write_text(
            json.dumps(token_cache, indent=2),
            encoding="utf-8",
        )
    else:
        _write_atomic(market_oos, market_path)

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
                market_oos,
            ),
        )

    worker_count = max(1, min(int(args.workers), 6))
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
