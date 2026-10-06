"""Volume-free Beta V2 intraday feature and label engine.

This module is deliberately independent of the current production 5-minute
model. It creates features that can actually be supplied by the NSE public
quote-chart live lane: price and time only, with no per-minute volume fields.
Labels are session-safe and market-neutral.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
OPEN_MIN = 9 * 60 + 15
CLOSE_MIN = 15 * 60 + 30

BASE_FEATURES = [
    "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_60m",
    "rsi7", "rsi14", "atr14_pct",
    "ema8_ratio", "ema21_ratio", "ma5_ratio", "ma10_ratio",
    "ma20_ratio", "ma50_ratio",
    "session_return", "pos_in_range", "or_break", "or_dist",
    "range_pct", "body_pct", "ret_accel_1_5",
    "tod_sin", "tod_cos", "mins_since_open",
    "market_ret_5m", "market_ret_15m", "market_ret_30m",
    "relative_ret_5m", "relative_ret_15m", "relative_ret_30m",
]

RANK_FEATURE_SOURCES = [
    "ret_5m", "ret_15m", "ret_30m", "ret_60m",
    "rsi14", "atr14_pct", "session_return", "pos_in_range",
]


def _norm(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    x = frame.copy()
    x.columns = [str(c).strip() for c in x.columns]
    aliases = {
        "timestamp": "ts", "datetime": "ts", "date": "ts", "time": "ts",
        "open": "open", "high": "high", "low": "low",
        "close": "close", "ltp": "close",
    }
    x = x.rename(columns={c: aliases.get(str(c).lower(), c) for c in x.columns})

    if "ts" in x.columns:
        raw = x.pop("ts")
    else:
        raw = x.index
    ts = pd.to_datetime(raw, errors="coerce", utc=True)
    x["ts"] = ts

    for c in ("open", "high", "low", "close"):
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts")
    return x.drop_duplicates("ts", keep="last").reset_index(drop=True)


def _rsi(c: pd.Series, n: int) -> float:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    if up.empty or pd.isna(up.iloc[-1]) or pd.isna(dn.iloc[-1]):
        return np.nan
    if dn.iloc[-1] == 0:
        return 100.0
    return float(100.0 - 100.0 / (1.0 + up.iloc[-1] / dn.iloc[-1]))


def _session_rows(g: pd.DataFrame, market: pd.DataFrame | None, horizons: Iterable[int]) -> list[dict]:
    z = _norm(g)
    if z.empty:
        return []

    z["ist"] = z["ts"].dt.tz_convert(IST)
    z["session"] = z["ist"].dt.date
    rows: list[dict] = []

    market_n = _norm(market) if market is not None else pd.DataFrame()

    for session, d in z.groupby("session", sort=False):
        d = d.sort_values("ts").copy()
        mod = d["ist"].dt.hour * 60 + d["ist"].dt.minute
        mask = (mod >= OPEN_MIN) & (mod < CLOSE_MIN)
        d = d.loc[mask].copy()
        if len(d) < 80:
            continue

        c = d["close"].to_numpy(float)
        h = d["high"].to_numpy(float)
        lo = d["low"].to_numpy(float)
        o = d["open"].to_numpy(float)
        ts = d["ts"].to_numpy()
        minutes = d["ist"].dt.hour.to_numpy() * 60 + d["ist"].dt.minute.to_numpy()

        # Exact minute-of-day lookup inside one session prevents an accidental
        # bridge over lunch/overnight gaps.
        minute_to_idx = {int(m): int(i) for i, m in enumerate(minutes)}
        run_hi = np.maximum.accumulate(h)
        run_lo = np.minimum.accumulate(lo)

        or_mask = minutes < OPEN_MIN + 15
        if not bool(or_mask.any()):
            continue
        or_hi = float(np.nanmax(h[or_mask]))
        or_lo = float(np.nanmin(lo[or_mask]))
        logc = np.log(np.clip(c, 1e-12, None))
        r1 = np.diff(logc, prepend=logc[0])

        market_close = None
        if not market_n.empty:
            mn = market_n.copy()
            mn["ist"] = mn["ts"].dt.tz_convert(IST)
            mn["session"] = mn["ist"].dt.date
            mn = mn[mn["session"].eq(session)].sort_values("ts")
            if not mn.empty:
                ms = mn.set_index("ts")["close"]
                market_close = ms.reindex(d["ts"], method="ffill", tolerance=pd.Timedelta(minutes=2))

        for i in range(len(d)):
            mins = int(minutes[i] - OPEN_MIN)
            if mins < 15 or mins % 5 != 0:
                continue

            def back(k: int) -> float:
                j = minute_to_idx.get(int(minutes[i] - k))
                return float(logc[i] - logc[j]) if j is not None else np.nan

            rng = float(run_hi[i] - run_lo[i])
            w = r1[max(1, i - 29): i + 1]
            rv30 = float(np.sqrt(np.sum(w ** 2))) if len(w) >= 10 else np.nan

            def safe_ratio(a: float, b: float) -> float:
                return float(a / b - 1.0) if np.isfinite(a) and np.isfinite(b) and b != 0 else np.nan

            ema8 = pd.Series(c[: i + 1]).ewm(span=8, adjust=False, min_periods=8).mean().iloc[-1]
            ema21 = pd.Series(c[: i + 1]).ewm(span=21, adjust=False, min_periods=21).mean().iloc[-1]
            ma5 = pd.Series(c[: i + 1]).rolling(5, min_periods=5).mean().iloc[-1]
            ma10 = pd.Series(c[: i + 1]).rolling(10, min_periods=10).mean().iloc[-1]
            ma20 = pd.Series(c[: i + 1]).rolling(20, min_periods=20).mean().iloc[-1]
            ma50 = pd.Series(c[: i + 1]).rolling(50, min_periods=50).mean().iloc[-1]
            tr = np.maximum.reduce([
                h[: i + 1] - lo[: i + 1],
                np.abs(h[: i + 1] - np.r_[np.nan, c[:i]]),
                np.abs(lo[: i + 1] - np.r_[np.nan, c[:i]]),
            ])
            atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1]

            market_r5 = market_r15 = market_r30 = np.nan
            if market_close is not None:
                mc = pd.to_numeric(market_close, errors="coerce").to_numpy(float)
                if i < len(mc) and np.isfinite(mc[i]):
                    def mback(k: int) -> float:
                        if i - k >= 0 and np.isfinite(mc[i-k]):
                            return float(np.log(mc[i]) - np.log(mc[i-k]))
                        return np.nan
                    market_r5, market_r15, market_r30 = mback(5), mback(15), mback(30)

            row = {
                "symbol": str(g["symbol"].iloc[0]) if "symbol" in g.columns else "",
                "decision_timestamp": pd.Timestamp(ts[i]),
                "date": pd.Timestamp(session),
                "ret_1m": back(1),
                "ret_5m": back(5),
                "ret_15m": back(15),
                "ret_30m": back(30),
                "ret_60m": back(60),
                "rsi7": _rsi(pd.Series(c[:i+1]), 7),
                "rsi14": _rsi(pd.Series(c[:i+1]), 14),
                "atr14_pct": safe_ratio(float(atr), float(c[i])),
                "ema8_ratio": safe_ratio(float(c[i]), float(ema8)),
                "ema21_ratio": safe_ratio(float(c[i]), float(ema21)),
                "ma5_ratio": safe_ratio(float(c[i]), float(ma5)),
                "ma10_ratio": safe_ratio(float(c[i]), float(ma10)),
                "ma20_ratio": safe_ratio(float(c[i]), float(ma20)),
                "ma50_ratio": safe_ratio(float(c[i]), float(ma50)),
                "session_return": safe_ratio(float(c[i]), float(o[0])),
                "pos_in_range": (c[i] - run_lo[i]) / rng if rng > 0 else 0.5,
                "or_break": 1.0 if c[i] > or_hi else (-1.0 if c[i] < or_lo else 0.0),
                "or_dist": (c[i] - (or_hi + or_lo) / 2.0) / max(or_hi - or_lo, 1e-9),
                "range_pct": safe_ratio(float(h[i] - lo[i]), float(c[i])),
                "body_pct": safe_ratio(float(c[i]), float(o[i])),
                "ret_accel_1_5": back(1) - back(5) / 5.0 if np.isfinite(back(1)) and np.isfinite(back(5)) else np.nan,
                "tod_sin": np.sin(2*np.pi*mins/(CLOSE_MIN-OPEN_MIN)),
                "tod_cos": np.cos(2*np.pi*mins/(CLOSE_MIN-OPEN_MIN)),
                "mins_since_open": float(mins),
                "market_ret_5m": market_r5,
                "market_ret_15m": market_r15,
                "market_ret_30m": market_r30,
                "relative_ret_5m": back(5) - market_r5 if np.isfinite(back(5)) and np.isfinite(market_r5) else np.nan,
                "relative_ret_15m": back(15) - market_r15 if np.isfinite(back(15)) and np.isfinite(market_r15) else np.nan,
                "relative_ret_30m": back(30) - market_r30 if np.isfinite(back(30)) and np.isfinite(market_r30) else np.nan,
            }

            for hmin in horizons:
                target_min = int(minutes[i] + hmin)
                j = minute_to_idx.get(target_min)
                if j is None:
                    continue
                row2 = row.copy()
                row2["horizon_min"] = hmin
                row2["reference_close"] = float(c[i])
                row2["future_close"] = float(c[j])
                row2["fwd_ret"] = float(c[j] / c[i] - 1.0)
                row2["target_timestamp"] = pd.Timestamp(ts[j])
                rows.append(row2)

    return rows


def add_cross_sectional_targets(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy()
    x["decision_timestamp"] = pd.to_datetime(x["decision_timestamp"], utc=True)
    x["fwd_ret"] = pd.to_numeric(x["fwd_ret"], errors="coerce")
    med = x.groupby(["decision_timestamp", "horizon_min"])["fwd_ret"].transform("median")
    x["fwd_ret_excess"] = x["fwd_ret"] - med
    x["y_up_excess"] = (x["fwd_ret_excess"] > 0).astype("int8")

    rank_cols = []
    for c in RANK_FEATURE_SOURCES:
        if c in x:
            name = c + "_xs_rank"
            x[name] = x.groupby(["decision_timestamp", "horizon_min"])[c].rank(pct=True)
            rank_cols.append(name)

    return x


def feature_columns() -> list[str]:
    return BASE_FEATURES + [c + "_xs_rank" for c in RANK_FEATURE_SOURCES]
