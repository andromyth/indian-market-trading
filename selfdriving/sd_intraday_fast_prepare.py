"""Fast, vectorized Beta V2 intraday compact-row preparation.

This module preserves the existing Beta V2 intraday feature/label contract but
avoids the per-decision/per-horizon Python-dict construction used by the
original preparation path. It is designed for Colab CPU execution and
read-only access to the canonical NSE minute seed.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from selfdriving.sd_intraday_v2 import (
    BASE_FEATURES,
    IST,
    OPEN_MIN,
    CLOSE_MIN,
    _norm,
)


_MARKET_NORM_CACHE: dict[int, pd.DataFrame] = {}
_MARKET_SESSION_CACHE: dict[int, dict[object, pd.DataFrame]] = {}


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100.0 - 100.0 / (1.0 + up / down.replace(0, np.nan))
    return out.where(down.ne(0), 100.0)


def _warm_market_cache(market: pd.DataFrame | None) -> dict[object, pd.DataFrame]:
    if market is None or market.empty:
        return {}

    key = id(market)
    cached = _MARKET_SESSION_CACHE.get(key)
    if cached is not None:
        return cached

    market_n = _MARKET_NORM_CACHE.get(key)
    if market_n is None:
        market_n = _norm(market)
        _MARKET_NORM_CACHE[key] = market_n

    if market_n.empty:
        return {}

    with_ist = market_n.assign(ist=market_n["ts"].dt.tz_convert(IST))
    sessions = {
        session: part.sort_values("ts")
        for session, part in with_ist.groupby(
            with_ist["ist"].dt.date,
            sort=False,
        )
    }
    _MARKET_SESSION_CACHE[key] = sessions
    return sessions


def warm_market_cache(market: pd.DataFrame | None) -> None:
    """Pre-build the immutable market-session lookup before worker threads start."""
    _warm_market_cache(market)


def _compact_one_session(
    d: pd.DataFrame,
    session,
    market_sessions: dict[object, pd.DataFrame],
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    d = d.sort_values("ts").copy()
    d["ist"] = d["ts"].dt.tz_convert(IST)

    minutes = (
        d["ist"].dt.hour.to_numpy(dtype=np.int16) * 60
        + d["ist"].dt.minute.to_numpy(dtype=np.int16)
    )
    mask = (minutes >= OPEN_MIN) & (minutes < CLOSE_MIN)
    d = d.loc[mask].copy()
    minutes = minutes[mask]

    if len(d) < 80:
        return pd.DataFrame()

    # Match the existing engine's duplicate-minute policy.
    if len(pd.Index(minutes).unique()) != len(minutes):
        keep = ~pd.Index(minutes).duplicated(keep="last")
        d = d.iloc[np.flatnonzero(keep)]
        minutes = minutes[keep]

    if len(d) < 80:
        return pd.DataFrame()

    idx = pd.DatetimeIndex(d["ts"])
    c = pd.to_numeric(d["close"], errors="coerce").to_numpy(dtype=float)
    h = pd.to_numeric(d["high"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(d["low"], errors="coerce").to_numpy(dtype=float)
    o = pd.to_numeric(d["open"], errors="coerce").to_numpy(dtype=float)

    close_s = pd.Series(c, index=idx)
    high_s = pd.Series(h, index=idx)
    low_s = pd.Series(lo, index=idx)
    open_s = pd.Series(o, index=idx)

    decision_mask = (
        (minutes >= OPEN_MIN + 15)
        & (((minutes - OPEN_MIN) % 5) == 0)
    )
    positions = np.flatnonzero(decision_mask)
    if positions.size == 0:
        return pd.DataFrame()

    # Causal price features. These are calculated over the complete session,
    # then sampled to decision timestamps, so max_decisions does not shorten
    # the feature lookback.
    logc = np.log(np.clip(c, 1e-12, None))
    log_by_min = pd.Series(logc, index=minutes)
    cur_log = logc

    r1 = pd.Series(cur_log, index=idx).diff().to_numpy()
    def lag_return(lag: int) -> np.ndarray:
        past = log_by_min.reindex(minutes - int(lag)).to_numpy()
        return cur_log - past

    r5 = lag_return(5)
    r15 = lag_return(15)
    r30 = lag_return(30)
    r60 = lag_return(60)

    rsi_delta = pd.Series(c, index=idx).diff()
    rsi7 = _rsi(pd.Series(c, index=idx), 7)
    rsi14 = _rsi(pd.Series(c, index=idx), 14)

    prev_close = close_s.shift(1)
    tr = pd.concat(
        [
            high_s - low_s,
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    ema8 = close_s.ewm(span=8, adjust=False, min_periods=8).mean()
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean()
    ma5 = close_s.rolling(5, min_periods=5).mean()
    ma10 = close_s.rolling(10, min_periods=10).mean()
    ma20 = close_s.rolling(20, min_periods=20).mean()
    ma50 = close_s.rolling(50, min_periods=50).mean()

    run_hi = high_s.cummax().to_numpy()
    run_lo = low_s.cummin().to_numpy()

    opening = (minutes < OPEN_MIN + 15)
    or_hi = float(np.nanmax(h[opening])) if np.any(opening) else np.nan
    or_lo = float(np.nanmin(lo[opening])) if np.any(opening) else np.nan

    mins_since = minutes.astype(np.int32) - OPEN_MIN
    current = c

    mins_s = pd.Series(minutes, index=idx)
    market_aligned = pd.Series(np.nan, index=idx, dtype=float)
    mn = market_sessions.get(session)
    if mn is not None and not mn.empty:
        ms = pd.Series(
            pd.to_numeric(mn["close"], errors="coerce").to_numpy(),
            index=pd.DatetimeIndex(mn["ts"]),
        )
        market_aligned = ms.reindex(
            idx,
            method="ffill",
            tolerance=pd.Timedelta(minutes=2),
        )

    market_by_min = pd.Series(
        np.log(pd.to_numeric(market_aligned, errors="coerce").to_numpy()),
        index=minutes,
    )
    market_has_data = bool(market_aligned.notna().any())

    if market_has_data:
        mcur = market_by_min.reindex(minutes).to_numpy()
        mret5 = mcur - market_by_min.reindex(minutes - 5).to_numpy()
        mret15 = mcur - market_by_min.reindex(minutes - 15).to_numpy()
        mret30 = mcur - market_by_min.reindex(minutes - 30).to_numpy()
    else:
        mret5 = np.full(len(d), np.nan, dtype=float)
        mret15 = np.full(len(d), np.nan, dtype=float)
        mret30 = np.full(len(d), np.nan, dtype=float)

    pos = positions
    cur = current[pos]
    rng = run_hi[pos] - run_lo[pos]

    data = {
        "symbol": str(d["symbol"].iloc[0]).strip().upper()
        if "symbol" in d.columns else "",
        "decision_timestamp": idx[pos],
        "date": pd.Timestamp(session),
        "reference_close": cur,
        "ret_1m": r1[pos],
        "ret_5m": r5[pos],
        "ret_15m": r15[pos],
        "ret_30m": r30[pos],
        "ret_60m": r60[pos],
        "rsi7": rsi7.to_numpy()[pos],
        "rsi14": rsi14.to_numpy()[pos],
        "atr14_pct": (atr14 / close_s.replace(0, np.nan)).to_numpy()[pos],
        "ema8_ratio": (close_s / ema8.replace(0, np.nan) - 1.0).to_numpy()[pos],
        "ema21_ratio": (close_s / ema21.replace(0, np.nan) - 1.0).to_numpy()[pos],
        "ma5_ratio": (close_s / ma5.replace(0, np.nan) - 1.0).to_numpy()[pos],
        "ma10_ratio": (close_s / ma10.replace(0, np.nan) - 1.0).to_numpy()[pos],
        "ma20_ratio": (close_s / ma20.replace(0, np.nan) - 1.0).to_numpy()[pos],
        "ma50_ratio": (close_s / ma50.replace(0, np.nan) - 1.0).to_numpy()[pos],
        "session_return": cur / float(o[0]) - 1.0 if o[0] else np.full(len(pos), np.nan),
        "pos_in_range": np.divide(
            cur - run_lo[pos],
            rng,
            out=np.full(len(pos), 0.5, dtype=float),
            where=rng > 0,
        ),
        "or_break": np.where(
            cur > or_hi,
            1.0,
            np.where(cur < or_lo, -1.0, 0.0),
        ),
        "or_dist": (cur - (or_hi + or_lo) / 2.0) / max(or_hi - or_lo, 1e-9),
        "range_pct": np.divide(
            h[pos] - lo[pos],
            cur,
            out=np.full(len(pos), np.nan, dtype=float),
            where=cur != 0,
        ),
        "body_pct": np.divide(
            cur - o[pos],
            o[pos],
            out=np.full(len(pos), np.nan, dtype=float),
            where=o[pos] != 0,
        ),
        "ret_accel_1_5": r5[pos] - r1[pos],
        "tod_sin": np.sin(2 * np.pi * mins_since[pos] / (CLOSE_MIN - OPEN_MIN)),
        "tod_cos": np.cos(2 * np.pi * mins_since[pos] / (CLOSE_MIN - OPEN_MIN)),
        "mins_since_open": mins_since[pos].astype(float),
        "market_ret_5m": mret5[pos],
        "market_ret_15m": mret15[pos],
        "market_ret_30m": mret30[pos],
        "relative_ret_5m": r5[pos] - mret5[pos],
        "relative_ret_15m": r15[pos] - mret15[pos],
        "relative_ret_30m": r30[pos] - mret30[pos],
    }

    out = pd.DataFrame(data)

    close_by_min = pd.Series(c, index=minutes)
    decision_minutes = minutes[pos]

    # Exact-minute future labels, matching the existing _session_rows contract.
    valid_horizon = np.zeros(len(out), dtype=bool)
    for horizon in horizons:
        future = close_by_min.reindex(decision_minutes + int(horizon)).to_numpy()
        out[f"future_close_{horizon}m"] = future
        out[f"fwd_ret_{horizon}m"] = future / cur - 1.0
        valid_horizon |= np.isfinite(future)

    return out.loc[valid_horizon].reset_index(drop=True)


def build_compact_symbol(
    g: pd.DataFrame,
    market: pd.DataFrame | None,
    horizons: Iterable[int],
    *,
    max_decisions: int | None = None,
) -> pd.DataFrame:
    """Build the compact wide checkpoint in one vectorized pass per session."""
    z = _norm(g)
    if z.empty:
        return pd.DataFrame()

    z["ist"] = z["ts"].dt.tz_convert(IST)
    z["session"] = z["ist"].dt.date
    market_sessions = _warm_market_cache(market)
    horizon_tuple = tuple(int(h) for h in horizons)

    chunks: list[pd.DataFrame] = []
    for session, d in z.groupby("session", sort=False):
        part = _compact_one_session(d, session, market_sessions, horizon_tuple)
        if not part.empty:
            chunks.append(part)

    if not chunks:
        return pd.DataFrame()

    out = pd.concat(chunks, ignore_index=True)
    out["decision_timestamp"] = pd.to_datetime(
        out["decision_timestamp"], utc=True, errors="coerce"
    )
    out = out.dropna(subset=["decision_timestamp"]).sort_values("decision_timestamp")
    out = out.drop_duplicates("decision_timestamp", keep="first").reset_index(drop=True)

    if max_decisions is not None and int(max_decisions) > 0 and len(out) > int(max_decisions):
        idx = np.linspace(0, len(out) - 1, int(max_decisions)).round().astype(int)
        out = out.iloc[np.unique(idx)].reset_index(drop=True)

    # Keep stable column ordering for checkpoints and downstream consumers.
    ordered = [
        "symbol",
        "decision_timestamp",
        "date",
        "reference_close",
    ] + BASE_FEATURES
    for horizon in horizon_tuple:
        ordered.extend([f"future_close_{horizon}m", f"fwd_ret_{horizon}m"])

    return out[[c for c in ordered if c in out.columns]]
