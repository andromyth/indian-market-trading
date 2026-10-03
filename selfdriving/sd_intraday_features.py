"""Volume-free intraday features.

The NSE public quote-chart lane has no per-minute volume (see repo docs), so a model
that depends on volume features is trained on one distribution and served another.
These features use price/time only and are identical in training and live serving.

Input: 1-minute bars [symbol, ts, open, high, low, close] with tz-aware `ts`.
Decisions are taken every `step_min` minutes; the label is the return over the next
`horizon_min` minutes and never crosses the session end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
OPEN_MIN, CLOSE_MIN = 9 * 60 + 15, 15 * 60 + 30


def _session_frame(g: pd.DataFrame, step: int, horizon: int, or_min: int = 15) -> pd.DataFrame:
    g = g.sort_values("ts")
    ts = g["ts"].dt.tz_convert(IST)
    mod = ts.dt.hour * 60 + ts.dt.minute
    g = g.assign(mod=mod.to_numpy())
    g = g[(g["mod"] >= OPEN_MIN) & (g["mod"] <= CLOSE_MIN)]
    if len(g) < 60:
        return pd.DataFrame()
    c = g["close"].to_numpy(float)
    h = g["high"].to_numpy(float)
    lo = g["low"].to_numpy(float)
    m = g["mod"].to_numpy()
    n = len(g)
    day_open = g["open"].iloc[0]
    run_hi, run_lo = np.maximum.accumulate(h), np.minimum.accumulate(lo)
    or_mask = m < OPEN_MIN + or_min
    or_hi, or_lo = h[or_mask].max(), lo[or_mask].min()
    logc = np.log(c)
    r1 = np.diff(logc, prepend=logc[0])
    rows = []
    for i in range(or_min, n):
        if (m[i] - OPEN_MIN) % step != 0:
            continue
        tgt = m[i] + horizon
        if tgt > CLOSE_MIN:
            continue
        j = np.searchsorted(m, tgt, side="left")
        if j >= n or m[j] != tgt:
            continue
        def back(k):
            t = m[i] - k
            jj = np.searchsorted(m, t, side="left")
            return logc[i] - logc[jj] if jj < n and m[jj] == t else np.nan
        w = r1[max(1, i - 29): i + 1]
        rv30 = float(np.sqrt(np.sum(w ** 2))) if len(w) >= 10 else np.nan
        rng = run_hi[i] - run_lo[i]
        rows.append({
            "ts": g["ts"].iloc[i],
            "ret_5m": back(5), "ret_15m": back(15), "ret_30m": back(30), "ret_60m": back(60),
            "rv30": rv30,
            "since_open": logc[i] - np.log(day_open),
            "pos_in_range": (c[i] - run_lo[i]) / rng if rng > 0 else 0.5,
            "or_break": 1.0 if c[i] > or_hi else (-1.0 if c[i] < or_lo else 0.0),
            "or_dist": (c[i] - (or_hi + or_lo) / 2) / max(or_hi - or_lo, 1e-9),
            "mins_since_open": float(m[i] - OPEN_MIN),
            "tod_sin": np.sin(2 * np.pi * (m[i] - OPEN_MIN) / (CLOSE_MIN - OPEN_MIN)),
            "tod_cos": np.cos(2 * np.pi * (m[i] - OPEN_MIN) / (CLOSE_MIN - OPEN_MIN)),
            "fwd_ret": logc[j] - logc[i],
        })
    return pd.DataFrame(rows)


def build_intraday_panel(bars: pd.DataFrame, market: pd.DataFrame | None = None,
                         step_min: int = 5, horizon_min: int = 30) -> pd.DataFrame:
    """Return decision rows with features, `fwd_ret` and `fwd_ret_excess` (vs cross-section median)."""
    b = bars.copy()
    b["ts"] = pd.to_datetime(b["ts"], utc=True)
    b["session"] = b["ts"].dt.tz_convert(IST).dt.date
    parts = []
    for (sym, sess), g in b.groupby(["symbol", "session"], sort=False):
        f = _session_frame(g, step_min, horizon_min)
        if f.empty:
            continue
        f["symbol"], f["session"] = sym, sess
        parts.append(f)
    if not parts:
        return pd.DataFrame()
    p = pd.concat(parts, ignore_index=True)
    p["decision_timestamp"] = p["ts"]
    if market is not None and not market.empty:
        mk = build_intraday_panel(market.assign(symbol="MKT"), None, step_min, horizon_min)
        mk = mk[["decision_timestamp", "ret_15m", "ret_60m", "since_open"]].rename(
            columns={"ret_15m": "mkt_ret_15m", "ret_60m": "mkt_ret_60m", "since_open": "mkt_since_open"})
        p = p.merge(mk, on="decision_timestamp", how="left")
        p["rel_15m"] = p["ret_15m"] - p["mkt_ret_15m"]
        p["rel_since_open"] = p["since_open"] - p["mkt_since_open"]
    grp = p.groupby("decision_timestamp")
    for c in ("ret_15m", "ret_60m", "since_open", "rv30"):
        p[f"xs_{c}_rank"] = grp[c].rank(pct=True)
    p["fwd_ret_excess"] = p["fwd_ret"] - grp["fwd_ret"].transform("median")
    p["y_up_excess"] = (p["fwd_ret_excess"] > 0).astype(int)
    p["date"] = pd.to_datetime(p["session"])
    return p


INTRADAY_FEATURES = ["ret_5m", "ret_15m", "ret_30m", "ret_60m", "rv30", "since_open", "pos_in_range",
                     "or_break", "or_dist", "mins_since_open", "tod_sin", "tod_cos",
                     "xs_ret_15m_rank", "xs_ret_60m_rank", "xs_since_open_rank", "xs_rv30_rank"]
