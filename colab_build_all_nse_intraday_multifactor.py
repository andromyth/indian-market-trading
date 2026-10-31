#!/usr/bin/env python3
"""Train the ALL-NSE 30-minute model on completed 5-minute OHLCV bars.

Historical minute-candle archives are resampled to complete five-minute bars
before feature/label creation. Live inference uses the same five-minute bar
contract. No orders are placed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

from all_nse_intraday_model import FEATURES

IST = "Asia/Kolkata"
PERSISTENT_INTRADAY_REL = Path("production_data/nse_official/intraday_1m")
SEED_REL = Path("production_data/nse_official/intraday_seed/olo_minute_archive/symbols")
NATIVE_5M_SEED_REL = Path("production_data/nse_official/intraday_seed/olo_5m_archive/symbols")
LIVE_5M_REL = Path("production_data/nse_official/intraday_5m_live")
MIN_SESSION_BARS = 50


def _norm(q: pd.DataFrame) -> pd.DataFrame:
    if q is None or q.empty:
        return pd.DataFrame()
    x = q.copy()
    x.columns = [str(c) for c in x.columns]
    aliases = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "adj close": "Close", "ltp": "Close", "volume": "Volume",
        "timestamp": "Timestamp", "datetime": "Timestamp",
        "date": "Timestamp", "time": "Timestamp",
    }
    x = x.rename(columns={c: aliases.get(str(c).strip().lower(), c) for c in x.columns})
    if "Timestamp" in x.columns:
        idx = pd.to_datetime(x.pop("Timestamp"), errors="coerce")
    else:
        idx = pd.to_datetime(x.index, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)
    x.index = idx
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in x:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[~x.index.isna()].sort_index()
    x = x[~x.index.duplicated(keep="last")]
    return x


def resample_completed_5m_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate genuine one-minute OHLCV to completed, close-timestamped 5m bars.

    Partial bars and bars with fewer than five source closes are dropped. The
    output index is the bar-close time, matching the live prediction decision
    timestamp and preventing a just-started candle from entering inference.
    """
    x = _norm(frame)
    if x.empty:
        return x
    idx = pd.DatetimeIndex(x.index)
    regular = (
        (idx.time >= pd.Timestamp("09:15").time())
        & (idx.time < pd.Timestamp("15:30").time())
    )
    x = x.loc[regular].copy()
    parts = []
    for _, session in x.groupby(pd.Index(x.index.date), sort=True):
        session = session.sort_index()
        grouped = session.resample(
            "5min", origin="start_day", closed="left", label="left"
        )
        bars = grouped.agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last"
        })
        source_counts = grouped["Close"].count()
        bars["Volume"] = grouped["Volume"].sum(min_count=5)
        bars = bars.loc[source_counts.eq(5) & bars[["Open", "High", "Low", "Close"]].notna().all(axis=1)]
        if bars.empty:
            continue
        bars.index = bars.index + pd.Timedelta(minutes=5)
        parts.append(bars)
    if not parts:
        return x.iloc[0:0].copy()
    result = pd.concat(parts).sort_index()
    return result.loc[~result.index.duplicated(keep="last")]


def session_aware_pct_change(c: pd.Series, n: int) -> pd.Series:
    """Match live returns: stay within a session and carry prior-session return during warm-up."""
    values = pd.to_numeric(c, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if values.empty:
        return result
    index = pd.DatetimeIndex(values.index)
    index = index.tz_localize(IST) if index.tz is None else index.tz_convert(IST)
    session_dates = pd.Index(index.date)
    previous_session_return = np.nan
    for day in pd.unique(session_dates):
        positions = np.flatnonzero(session_dates == day)
        session = values.iloc[positions].dropna()
        if session.empty:
            previous_session_return = np.nan
            continue
        if len(session) > n:
            returns = session.pct_change(n, fill_method=None)
            if np.isfinite(previous_session_return):
                returns.iloc[:n] = previous_session_return
            result.loc[session.index] = returns
            previous_session_return = float(returns.iloc[-1])
        else:
            if np.isfinite(previous_session_return):
                result.loc[session.index] = previous_session_return
            previous_session_return = np.nan
    return result


def rsi_series(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(g: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    x = _norm(g)
    if x.empty:
        return pd.DataFrame()
    c, h, l, o, v = x["Close"], x["High"], x["Low"], x["Open"], x["Volume"]
    out = pd.DataFrame(index=x.index)

    for n in (1, 3, 5, 10, 15, 30):
        out[f"ret_{n}"] = session_aware_pct_change(c, n)

    out["rsi14"] = rsi_series(c)
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    out["atr14_pct"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / c

    for n in (5, 10, 20, 50):
        out[f"ma{n}_ratio"] = c / c.rolling(n, min_periods=n).mean() - 1

    out["range_pct"] = (h - l) / c
    out["body_pct"] = (c - o) / o

    vm = v.rolling(20, min_periods=20).mean()
    vs = v.rolling(20, min_periods=20).std()
    out["volume_ratio"] = v / vm
    out["volume_z20"] = (v - vm) / vs

    mins = x.index.hour * 60 + x.index.minute
    out["tod_sin"] = np.sin(2 * np.pi * mins / 390.0)
    out["tod_cos"] = np.cos(2 * np.pi * mins / 390.0)

    if market is not None and not market.empty:
        m = market if isinstance(market.index, pd.DatetimeIndex) else _norm(market)
        mc = m["Close"].reindex(x.index, method="ffill", tolerance=pd.Timedelta(minutes=2))
        for n in (1, 3, 5, 10, 15, 30):
            market_returns = session_aware_pct_change(m["Close"], n)
            out[f"market_ret_{n}"] = market_returns.reindex(
                x.index, method="ffill", tolerance=pd.Timedelta(minutes=2)
            )
        out["market_rsi14"] = rsi_series(mc)
    else:
        for n in (1, 3, 5, 10, 15, 30):
            out[f"market_ret_{n}"] = np.nan
        out["market_rsi14"] = np.nan

    out["relative_ret_1"] = out["ret_1"] - out["market_ret_1"]
    out["relative_ret_5"] = out["ret_5"] - out["market_ret_5"]
    out["relative_ret_15"] = out["ret_15"] - out["market_ret_15"]

    # Strict timestamp target: use the actual +30 minute timestamp within a
    # one-minute tolerance rather than assuming 30 rows == 30 minutes.
    right = pd.DataFrame({"future_ts": x.index, "future_close": c.to_numpy()}).sort_values("future_ts")
    left = pd.DataFrame({"decision_ts": x.index, "future_ts": x.index + pd.Timedelta(minutes=30)}).sort_values("future_ts")
    merged = pd.merge_asof(
        left,
        right,
        on="future_ts",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=1),
    )
    future_close = merged.set_index("decision_ts")["future_close"].reindex(out.index)
    out["future_return_30m"] = future_close / c - 1
    out["target_30m"] = np.where(
        out["future_return_30m"].notna(),
        (out["future_return_30m"] > 0).astype(float),
        np.nan,
    )
    return out


def _merge_5m_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    clean = [_norm(part)[["Open", "High", "Low", "Close", "Volume"]]
             for part in parts if part is not None and not part.empty]
    if not clean:
        return pd.DataFrame()
    merged = pd.concat(clean).sort_index()
    return merged.loc[~merged.index.duplicated(keep="last")]


def _close_labeled_5m(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate true close-timestamped five-minute bars without resampling them."""
    x = _norm(frame)
    if x.empty:
        return x
    idx = pd.DatetimeIndex(x.index)
    mask = (
        (idx.time >= pd.Timestamp("09:20").time())
        & (idx.time <= pd.Timestamp("15:30").time())
        & (idx.minute % 5 == 0)
    )
    return x.loc[mask, ["Open", "High", "Low", "Close", "Volume"]].copy()


def load_live_history(root: Path) -> dict[str, pd.DataFrame]:
    """Read slot files from the active worker's durable five-minute archive once."""
    archive = root / LIVE_5M_REL
    grouped: dict[str, list[pd.DataFrame]] = {}
    if not archive.exists():
        return {}
    for path in sorted(archive.rglob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
            if frame is None or frame.empty or "symbol" not in frame.columns:
                continue
            frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
            for symbol, part in frame.groupby("symbol", sort=False):
                bars = _close_labeled_5m(part.drop(columns=["symbol", "source"], errors="ignore"))
                if not bars.empty:
                    grouped.setdefault(symbol, []).append(bars)
        except Exception as exc:
            print(f"LIVE_5M_ARCHIVE_READ_ERROR path={path} error={type(exc).__name__}:{exc}", flush=True)
    return {symbol: _merge_5m_parts(parts) for symbol, parts in grouped.items()}


def load_symbol(root: Path, sym: str, live_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    minute_parts = []
    errors = []

    # Legacy one-minute seed is preserved as-is and aggregated only from genuine
    # one-minute source bars; it is never mixed directly with native five-minute rows.
    seed_dir = root / SEED_REL / f"symbol={sym}"
    seed_files = sorted(seed_dir.rglob("*.parquet")) if seed_dir.exists() else []
    for path in seed_files:
        try:
            frame = pd.read_parquet(path)
            if frame is not None and not frame.empty:
                minute_parts.append(_norm(frame))
        except Exception as exc:
            errors.append(f"seed:{path.name}:{type(exc).__name__}:{exc}")

    official_dir = root / PERSISTENT_INTRADAY_REL
    if official_dir.exists():
        for path in sorted(official_dir.rglob(f"{sym}.parquet")):
            try:
                frame = pd.read_parquet(path)
                if frame is not None and not frame.empty:
                    minute_parts.append(_norm(frame))
            except Exception as exc:
                errors.append(f"official:{path}:{type(exc).__name__}:{exc}")

    five_minute_parts = []
    minute_parts = [part for part in minute_parts if part is not None and not part.empty]
    if minute_parts:
        minute_history = pd.concat(minute_parts).sort_index()
        minute_history = minute_history.loc[~minute_history.index.duplicated(keep="last")]
        aggregated = resample_completed_5m_bars(minute_history)
        if not aggregated.empty:
            five_minute_parts.append(aggregated)

    # Native Dhan five-minute history has its own namespace and timestamp
    # convention. It is merged as five-minute data, never passed through the
    # one-minute resampler.
    native_dir = root / NATIVE_5M_SEED_REL / f"symbol={sym}"
    if native_dir.exists():
        for path in sorted(native_dir.rglob("*.parquet")):
            try:
                frame = pd.read_parquet(path)
                bars = _close_labeled_5m(frame)
                if not bars.empty:
                    five_minute_parts.append(bars)
            except Exception as exc:
                errors.append(f"native_5m:{path}:{type(exc).__name__}:{exc}")

    if live_frame is not None and not live_frame.empty:
        live_bars = _close_labeled_5m(live_frame)
        if not live_bars.empty:
            five_minute_parts.append(live_bars)

    if errors and not five_minute_parts:
        raise RuntimeError(
            f"Could not read any intraday data for {sym}. "
            + " | ".join(errors[:3])
        )
    return _merge_5m_parts(five_minute_parts)


def load_market(root: Path, live_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    candidates = [
        root / "raw/2018_2025/NIFTY50-INDEX.parquet",
        root / "raw/2026/NIFTY50-INDEX.parquet",
    ]
    minute_parts = []
    for path in candidates:
        if path.exists():
            try:
                frame = pd.read_parquet(path)
                if not frame.empty:
                    minute_parts.append(_norm(frame))
            except Exception:
                pass
    five_minute_parts = []
    if minute_parts:
        minute_history = pd.concat(minute_parts).sort_index()
        minute_history = minute_history.loc[~minute_history.index.duplicated(keep="last")]
        aggregated = resample_completed_5m_bars(minute_history)
        if not aggregated.empty:
            five_minute_parts.append(aggregated)

    native_market_dir = root / NATIVE_5M_SEED_REL / "symbol=__NIFTY50__"
    if native_market_dir.exists():
        for path in sorted(native_market_dir.rglob("*.parquet")):
            try:
                bars = _close_labeled_5m(pd.read_parquet(path))
                if not bars.empty:
                    five_minute_parts.append(bars)
            except Exception:
                pass
    if live_frame is not None and not live_frame.empty:
        live_bars = _close_labeled_5m(live_frame)
        if not live_bars.empty:
            five_minute_parts.append(live_bars)
    return _merge_5m_parts(five_minute_parts)


def load_symbol(root: Path, sym: str) -> pd.DataFrame:
    parts = []
    errors = []

    # Historical seed consolidated into one Parquet file per symbol.
    seed_dir = root / SEED_REL / f"symbol={sym}"
    seed_files = sorted(seed_dir.glob("*.parquet")) if seed_dir.exists() else []
    for p in seed_files:
        try:
            z = pd.read_parquet(p)
            if z is None or z.empty:
                continue
            parts.append(_norm(z))
        except Exception as exc:
            errors.append(f"seed:{p.name}:{type(exc).__name__}:{exc}")

    # Official NSE incremental archive, if it already contains observations.
    official_dir = root / PERSISTENT_INTRADAY_REL
    if official_dir.exists():
        for p in sorted(official_dir.rglob(f"{sym}.parquet")):
            try:
                z = pd.read_parquet(p)
                if z is None or z.empty:
                    continue
                parts.append(_norm(z))
            except Exception as exc:
                errors.append(f"official:{p}:{type(exc).__name__}:{exc}")

    if errors and not parts:
        raise RuntimeError(
            f"Could not read any intraday data for {sym}. "
            + " | ".join(errors[:3])
        )

    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()

    q = pd.concat(parts).sort_index()
    q = q[~q.index.duplicated(keep="last")]
    return q

def load_market(root: Path) -> pd.DataFrame:
    cands = [
        root / "raw/2018_2025/NIFTY50-INDEX.parquet",
        root / "raw/2026/NIFTY50-INDEX.parquet",
    ]
    parts = []
    for p in cands:
        if p.exists():
            try:
                z = pd.read_parquet(p)
                if not z.empty:
                    parts.append(_norm(z))
            except Exception:
                pass
    if not parts:
        return pd.DataFrame()
    q = pd.concat(parts).sort_index()
    return q[~q.index.duplicated(keep="last")]


def load_symbol_5m(
    root: Path,
    sym: str,
    live_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge legacy one-minute history, native 5m history and live 5m bars."""
    parts = []
    minute_history = load_symbol(root, sym)
    if minute_history is not None and not minute_history.empty:
        aggregated = resample_completed_5m_bars(minute_history)
        if not aggregated.empty:
            parts.append(aggregated)

    native_dir = root / NATIVE_5M_SEED_REL / f"symbol={sym}"
    if native_dir.exists():
        for path in sorted(native_dir.rglob("*.parquet")):
            try:
                bars = _close_labeled_5m(pd.read_parquet(path))
                if not bars.empty:
                    parts.append(bars)
            except Exception as exc:
                print(f"NATIVE_5M_ARCHIVE_READ_ERROR symbol={sym} path={path} error={type(exc).__name__}:{exc}", flush=True)

    if live_frame is not None and not live_frame.empty:
        bars = _close_labeled_5m(live_frame)
        if not bars.empty:
            parts.append(bars)
    return _merge_5m_parts(parts)


def load_market_5m(root: Path, live_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return NIFTY context on the same close-timestamped five-minute cadence."""
    legacy = load_market(root)
    parts = []
    if legacy is not None and not legacy.empty:
        aggregated = resample_completed_5m_bars(legacy)
        if not aggregated.empty:
            parts.append(aggregated)

    native_dir = root / NATIVE_5M_SEED_REL / "symbol=__NIFTY50__"
    if native_dir.exists():
        for path in sorted(native_dir.rglob("*.parquet")):
            try:
                bars = _close_labeled_5m(pd.read_parquet(path))
                if not bars.empty:
                    parts.append(bars)
            except Exception:
                pass

    if live_frame is not None and not live_frame.empty:
        bars = _close_labeled_5m(live_frame)
        if not bars.empty:
            parts.append(bars)
    return _merge_5m_parts(parts)


def load_fno(root: Path) -> pd.DataFrame:
    p = root / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
    try:
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fno_join_features(feature_frame: pd.DataFrame, fno: pd.DataFrame, sym: str) -> pd.DataFrame:
    fno_cols = ["fno_pcr", "fno_oi_change", "fno_fut_return", "fno_fut_oi", "fno_basis"]
    if fno is None or fno.empty:
        for c in fno_cols:
            feature_frame[c] = np.nan
        feature_frame["fno_available"] = 0.0
        return feature_frame

    z = fno.copy()
    z.columns = [str(c) for c in z.columns]
    lower = {c.lower(): c for c in z.columns}
    datecol = next((lower[k] for k in ("date", "trade_date", "timestamp") if k in lower), None)
    symcol = next((lower[k] for k in ("symbol", "ticker", "underlying") if k in lower), None)
    if not datecol or not symcol:
        for c in fno_cols:
            feature_frame[c] = np.nan
        feature_frame["fno_available"] = 0.0
        return feature_frame

    fno_dates = pd.to_datetime(z[datecol], errors="coerce")
    if isinstance(fno_dates.dtype, pd.DatetimeTZDtype):
        fno_dates = fno_dates.dt.tz_convert(IST).dt.tz_localize(None)
    z["_date"] = fno_dates.dt.normalize().astype("datetime64[ns]")
    z = z[
        z[symcol].astype(str).str.upper().str.replace(".NS", "", regex=False).eq(sym.upper())
    ].sort_values("_date")
    if z.empty:
        for c in fno_cols:
            feature_frame[c] = np.nan
        feature_frame["fno_available"] = 0.0
        return feature_frame

    alias_map = {
        "fno_pcr": ("fno_pcr", "pcr"),
        "fno_oi_change": ("fno_oi_change", "oi_change"),
        "fno_fut_return": ("fno_fut_return", "fut_return"),
        "fno_fut_oi": ("fno_fut_oi", "fut_oi"),
        "fno_basis": ("fno_basis", "fut_basis", "fno_fut_basis"),
    }

    idx = pd.DatetimeIndex(feature_frame.index)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)

    # Point-in-time rule: for an intraday session, use only an F&O observation
    # from a strictly earlier trading date. merge_asof performs this in one
    # vectorized operation instead of scanning all available dates row-by-row.
    left = pd.DataFrame({
        "_row_ts": feature_frame.index,
        "_session_date": idx.normalize().tz_localize(None).astype("datetime64[ns]"),
    }).sort_values("_session_date")

    for outc, aliases in alias_map.items():
        src = next((lower[a] for a in aliases if a in lower), None)
        if not src:
            feature_frame[outc] = np.nan
            continue

        d = z[["_date", src]].copy()
        d["value"] = pd.to_numeric(d[src], errors="coerce")
        d = d.dropna(subset=["_date", "value"]).drop_duplicates("_date", keep="last")
        if d.empty:
            feature_frame[outc] = np.nan
            continue

        d = d[["_date", "value"]].sort_values("_date")
        # Shift F&O observations forward to the next trading session's intraday
        # rows by matching each session to the latest strictly prior F&O date.
        left_key = left[["_row_ts", "_session_date"]].copy()
        # Parquet may restore F&O dates at millisecond precision while the
        # five-minute feature index uses nanoseconds. merge_asof requires
        # identical key dtypes, so normalize both join keys explicitly.
        left_key["_session_date"] = pd.to_datetime(
            left_key["_session_date"], errors="coerce"
        ).astype("datetime64[ns]")
        d["_date"] = pd.to_datetime(d["_date"], errors="coerce").astype("datetime64[ns]")
        merged = pd.merge_asof(
            left_key,
            d,
            left_on="_session_date",
            right_on="_date",
            direction="backward",
            allow_exact_matches=False,
        )
        vals = merged.set_index("_row_ts")["value"].reindex(feature_frame.index)
        feature_frame[outc] = vals.to_numpy()

    feature_frame["fno_available"] = feature_frame[fno_cols].notna().any(axis=1).astype(float)
    return feature_frame

def sample_decisions(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if frame.empty or minutes <= 1:
        return frame
    # Use the last available observation in each clock bucket, which preserves
    # minute-scale features while limiting the training matrix size.
    bucket = frame.index.floor(f"{int(minutes)}min")
    return frame.groupby(bucket, sort=True).tail(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument("--out", default="/content/drive/MyDrive/Indian_Market_Research/models/all_nse_intraday_5mbar_multifactor_replacement_2026")
    ap.add_argument("--min-rows-symbol", type=int, default=300)
    ap.add_argument("--sample-minutes", type=int, default=5)
    ap.add_argument("--max-train-rows-per-symbol", type=int, default=2000)
    ap.add_argument("--min-usable-sessions", type=int, default=200)
    ap.add_argument("--max-sessions", type=int, default=500)
    ap.add_argument("--min-bars-per-session", type=int, default=50)
    args = ap.parse_args()
    if not 1 <= args.min_usable_sessions <= args.max_sessions <= 500:
        raise ValueError("Session limits must satisfy 1 <= min-usable-sessions <= max-sessions <= 500")
    if args.min_bars_per_session < 50:
        raise ValueError("--min-bars-per-session must be at least 50")

    root = Path(args.drive_root)
    out = Path(args.out)

    # Recovery may provide the historical seed universe from Drive when the
    # NSE endpoint is unreachable. Normal candidate training still refreshes
    # the official roster by default; the explicit override is audited in the
    # emitted manifest and must be a non-empty JSON list of symbols.
    universe_override = os.getenv("ALL_NSE_INTRADAY_TRAINING_UNIVERSE_JSON", "").strip()
    if universe_override:
        override_path = Path(universe_override)
        if not override_path.is_file():
            raise RuntimeError("Configured intraday training universe JSON is missing")
        payload = json.loads(override_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("Configured intraday training universe must be a JSON list")
        universe = sorted({str(s).strip().upper().replace(".NS", "").replace(".BO", "") for s in payload if str(s).strip()})
        universe_source = "GOOGLE_DRIVE_HISTORICAL_SEED_SYMBOL_PARTITIONS"
    else:
        from delivery_training_universe import official_nse_symbols
        universe = sorted(set(official_nse_symbols(force_refresh=True)))
        universe_source = "NSE_OFFICIAL_CURRENT_EQUITY_UNIVERSE"
    if not universe:
        raise RuntimeError("Intraday training universe is empty")

    seed_root = root / SEED_REL
    if not seed_root.exists() and not (root / NATIVE_5M_SEED_REL).exists():
        raise RuntimeError("No historical minute or native five-minute seed archive is available.")

    live_history = load_live_history(root)
    market = load_market_5m(root, live_history.get("__NIFTY50__"))
    fno = load_fno(root)

    # Colab RAM is limited, so never accumulate the complete five-minute-derived
    # feature frame for all symbols. Each symbol is sampled evenly across its
    # available decision history and only a bounded per-symbol quota is retained.
    # This keeps all-NSE symbol coverage while bounding the pooled training matrix.
    GLOBAL_RETAINED_ROWS = 1_000_000
    per_symbol_cap = min(
        args.max_train_rows_per_symbol,
        max(100, GLOBAL_RETAINED_ROWS // max(1, len(universe))),
    )

    frames = []
    coverage = []
    trainable_symbols = 0
    retained_rows = 0

    for i, sym in enumerate(universe, 1):
        raw = load_symbol_5m(root, sym, live_history.get(sym))
        if raw.empty:
            coverage.append((sym, 0, 0, "EXCLUDED_NO_HISTORY"))
            continue
        valid_bars = raw[["Open", "High", "Low", "Close"]].notna().all(axis=1)
        session_counts = valid_bars.groupby(pd.Index(raw.index.date)).sum()
        usable_dates = sorted(session_counts[session_counts >= args.min_bars_per_session].index)
        usable_sessions = len(usable_dates)
        if usable_sessions < args.min_usable_sessions:
            coverage.append((sym, len(raw), usable_sessions, "EXCLUDED_INSUFFICIENT_SESSIONS"))
            continue
        retained_date_list = usable_dates[-args.max_sessions:]
        retained_dates = {str(day) for day in retained_date_list}
        raw = raw.loc[pd.Index(raw.index.date).astype(str).isin(retained_dates)]
        usable_sessions = len(retained_date_list)
        if len(raw) < args.min_rows_symbol:
            coverage.append((sym, len(raw), usable_sessions, "EXCLUDED_INSUFFICIENT_ROWS"))
            continue

        feat = make_features(raw, market)
        if feat.empty:
            coverage.append((sym, len(raw), usable_sessions, "EXCLUDED_FEATURE_BUILD"))
            continue

        # Build features and labels from completed five-minute bars, then sample
        # only on the matching five-minute decision cadence.
        feat = sample_decisions(feat, args.sample_minutes)
        feat = fno_join_features(feat, fno, sym)
        feat = feat.replace([np.inf, -np.inf], np.nan)
        feat = feat.dropna(subset=["target_30m"])

        if feat.empty:
            coverage.append((sym, len(raw), usable_sessions, "EXCLUDED_NO_LABELS"))
            continue

        # Spread retained observations across the whole available history rather
        # than keeping only the latest rows. The per-symbol quota is bounded so
        # the total pooled matrix remains safe for a 12 GB Colab runtime.
        if len(feat) > per_symbol_cap:
            feat = feat.sample(
                n=per_symbol_cap,
                random_state=20261003 + i,
            ).sort_index()

        feat["_symbol"] = sym
        feat["decision_ts"] = feat.index
        frames.append(feat)
        trainable_symbols += 1
        retained_rows += len(feat)
        coverage.append((sym, len(raw), usable_sessions, "TRAINABLE"))

        # Release the symbol's intermediates before continuing.
        del raw, feat

        if i % 100 == 0:
            import gc
            gc.collect()
            print(
                f"prepared {i:,}/{len(universe):,} symbols; "
                f"trainable={trainable_symbols:,}; retained_rows={retained_rows:,}; "
                f"per_symbol_cap={per_symbol_cap:,}",
                flush=True,
            )

    if not frames:
        raise RuntimeError(
            "No ALL-NSE symbols have the required 200-500 usable five-minute sessions."
        )

    data = pd.concat(frames, axis=0, ignore_index=False)
    del frames

    data = data.sort_values("decision_ts")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["target_30m"])

    # Deterministic chronological split by decision timestamp.
    cut = data["decision_ts"].quantile(0.80)
    train = data[data["decision_ts"] < cut].copy()
    test = data[data["decision_ts"] >= cut].copy()

    ytr = train["target_30m"].astype(int)
    yte = test["target_30m"].astype(int)
    if ytr.nunique() != 2 or yte.nunique() != 2:
        raise RuntimeError("Target split does not contain both classes")

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=20261003,
        )),
    ])
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    clf.fit(train[FEATURES], ytr)
    reg.fit(train[FEATURES], train["future_return_30m"].astype(float))

    p = clf.predict_proba(test[FEATURES])[:, 1]
    pred = (p >= 0.5).astype(int)
    pred_ret = reg.predict(test[FEATURES])

    train_symbols = sorted(data["_symbol"].unique().tolist())
    universe_size = len(universe)
    trainable = len(train_symbols)

    manifest = {
        "status": "BRANCH_REPLACEMENT_ONLY",
        "model_version": "2026-5mbar-multifactor-v1",
        "bar_interval_minutes": 5,
        "model_family": "logistic_regression+ridge",
        "target": "target_30m",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "universe_size": universe_size,
        "universe_source": universe_source,
        "training_symbols": train_symbols,
        "training_symbol_count": trainable,
        "coverage_pct": round(100 * trainable / universe_size, 4) if universe_size else 0.0,
        "minimum_rows_per_symbol": args.min_rows_symbol,
        "minimum_usable_sessions": args.min_usable_sessions,
        "maximum_sessions_retained": args.max_sessions,
        "minimum_bars_per_usable_session": args.min_bars_per_session,
        "symbols_with_200_plus_sessions": int(sum(row[2] >= 200 for row in coverage)),
        "symbols_with_500_sessions": int(sum(row[2] >= 500 for row in coverage)),
        "sample_minutes": args.sample_minutes,
        "max_train_rows_per_symbol": args.max_train_rows_per_symbol,
        "effective_per_symbol_cap": int(per_symbol_cap),
        "global_retained_row_cap": int(GLOBAL_RETAINED_ROWS),
        "retained_rows_before_split": int(len(data)),
        "training_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "trained_until": str(train["decision_ts"].max()),
        "holdout_start": str(test["decision_ts"].min()),
        "validation": {
            "accuracy": float(accuracy_score(yte, pred)),
            "roc_auc": float(roc_auc_score(yte, p)),
            "mean_abs_return_error": float(
                mean_absolute_error(test["future_return_30m"], pred_ret)
            ),
        },
        "classifier_sha256": "",
        "regressor_sha256": "",
        "no_auto_orders": True,
        "training_data_sources": [
            "OLO_PUBLIC_NSE_MINUTE_ARCHIVE_RESAMPLED_TO_COMPLETE_5M_BARS",
            "NSE_OFFICIAL_INTRADAY_MINUTE_ARCHIVE_RESAMPLED_TO_COMPLETE_5M_BARS_WHEN_AVAILABLE",
            "DHAN_NSE_EQUITY_NATIVE_5M_HISTORICAL_ARCHIVE",
            "ACTIVE_WORKER_NATIVE_5M_LIVE_ARCHIVE",
            "NSE_OFFICIAL_NIFTY50_CONTEXT",
            "NSE_OFFICIAL_FNO_DAILY_WHEN_AVAILABLE",
        ],
        "historical_seed_source_url": "https://github.com/olo-labs/olo-testdata-nse-min-candle",
        "live_incremental_source": "NSE_OFFICIAL_CHART",
        "point_in_time_fno": True,
        "volume_policy": "Use source-reported volume only; missing volume stays NaN.",
    }

    out.mkdir(parents=True, exist_ok=True)
    cp = out / "deployment_all_nse_intraday_5mbar_multifactor_classifier_2026.joblib"
    rp = out / "deployment_all_nse_intraday_5mbar_multifactor_regressor_2026.joblib"

    joblib.dump(clf, cp)
    joblib.dump(reg, rp)

    manifest["classifier_sha256"] = hashlib.sha256(cp.read_bytes()).hexdigest()
    manifest["regressor_sha256"] = hashlib.sha256(rp.read_bytes()).hexdigest()

    (out / "ALL_NSE_INTRADAY_5MBAR_MULTIFACTOR_2026_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        coverage,
        columns=["symbol", "stored_rows", "usable_sessions", "status"],
    ).to_csv(out / "INTRADAY_TRAINING_COVERAGE.csv", index=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
