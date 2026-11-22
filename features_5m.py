"""Native 5-minute features and the single to-close label.

Import this SAME module from training and from the serving adapter, so a
feature can never be computed two different ways.

Conventions
  * input: one symbol, DatetimeIndex (IST, one row per 5-minute bar, stamped at bar CLOSE:
    first bar 09:20, last bar 15:30), columns Open High Low Close Volume
  * windows are counted in 5-minute BARS (ret_3b = 15 minutes, ret_12b = 60 minutes)
  * no feature crosses a session boundary: a row is valid only if the previous 12 bars are
    in the same session and exactly 5 minutes apart (a missing bar invalidates rows, it is never bridged),
    and the session's first bar is on time
  * default (strict) warm-up: first decision bar is the 13th of the session (10:20). With warmup=k < 12 the first
    decision is bar k+1 and long-window features are NaN until they are computable (the models take NaN natively)
  * last decision bar is 15:25
  * label: fr_close = Close[15:30] / Close[t] - 1, same session only; NaN if the 15:30 bar is missing.
    Causality: features at stamp t use only bars stamped <= t (closed at t); the label uses the bar
    stamped 15:30. This holds ONLY if stamps are bar-close times (unify_bars enforces this).
  * minutes_to_close and the six F&O features are part of FEATURES_5M (42 features)
  * F&O values are point-in-time: the latest daily row strictly BEFORE the session date,
    at most 7 days old. Serving must call the same function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WIN = 12
DEFAULT_WARMUP = 12      # strict: first decision is the 13th bar. warmup=3 allows NaN long-window features earlier.
# window class and first bar index at which each windowed feature is computable inside a session
WINDOWED = {
    "ret_1b": (1, 1), "ret_3b": (3, 3), "ret_6b": (6, 6), "ret_12b": (12, 12),
    "rsi6b": (6, 6), "rsi12b": (12, 12), "atr12b_pct": (12, 12),
    "ma6_ratio": (6, 5), "ma12_ratio": (12, 11), "vol_6b": (6, 6), "vol_12b": (12, 12),
    "volume_ratio_12": (12, 11), "volume_z12": (12, 11), "volume_trend": (12, 11), "ret_accel": (6, 6),
    "mkt_ret_1b": (1, 1), "mkt_ret_3b": (3, 3), "mkt_ret_6b": (6, 6), "mkt_ret_12b": (12, 12), "mkt_vol_12b": (12, 12),
    "rel_ret_1b": (1, 1), "rel_ret_3b": (3, 3), "rel_ret_6b": (6, 6), "rel_ret_12b": (12, 12),
}
CLOSE_MIN = 15 * 60 + 30            # 930: 15:30 IST
OPEN_MIN = 9 * 60 + 15              # 555: 09:15 IST
FNO_COLS = ["fno_pcr", "fno_oi_change", "fno_fut_return", "fno_fut_oi", "fno_basis"]
FNO_ALIASES = {
    "fno_pcr": ("fno_pcr", "pcr"),
    "fno_oi_change": ("fno_oi_change", "oi_change"),
    "fno_fut_return": ("fno_fut_return", "fut_return"),
    "fno_fut_oi": ("fno_fut_oi", "fut_oi"),
    "fno_basis": ("fno_basis", "fut_basis", "fno_fut_basis"),
}
FEATURES_5M = [
    "ret_1b", "ret_3b", "ret_6b", "ret_12b", "rsi6b", "rsi12b", "atr12b_pct",
    "ma6_ratio", "ma12_ratio", "vol_6b", "vol_12b",
    "session_return", "day_high_ratio", "day_low_ratio", "vwap_ratio",
    "range_pct", "body_pct", "volume_ratio_12", "volume_z12", "volume_trend", "ret_accel",
    "tod_sin", "tod_cos", "gap_pct", "prev_day_ret", "log_price",
    "mkt_ret_1b", "mkt_ret_3b", "mkt_ret_6b", "mkt_ret_12b", "mkt_vol_12b",
    "rel_ret_1b", "rel_ret_3b", "rel_ret_6b", "rel_ret_12b",
    "minutes_to_close", *FNO_COLS, "fno_available",
]


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _session_info(idx: pd.DatetimeIndex, warmup: int = DEFAULT_WARMUP):
    """Returns session keys, bar position in session, row validity, and per-window contiguity flags."""
    sess = np.asarray(idx.floor("D"))
    pos = pd.Series(0, index=idx).groupby(sess).cumcount().to_numpy()
    ts = pd.Series(idx, index=idx)
    contig = {w: (ts.diff(w) == pd.Timedelta(minutes=5 * w)).to_numpy() for w in (1, 3, 6, 12)}
    first_ts = ts.groupby(sess).transform("first")
    on_time = (first_ts.dt.hour * 60 + first_ts.dt.minute <= 9 * 60 + 20).to_numpy()
    wclass = max([w for w in (1, 3, 6, 12) if w <= max(warmup, 1)])
    valid = (pos >= warmup) & contig[wclass] & on_time
    return sess, pos, valid, contig


def _mask_windows(f: pd.DataFrame, pos: np.ndarray, contig: dict) -> pd.DataFrame:
    """NaN every windowed feature that would reach back across a session start or over a missing bar."""
    for col, (w, min_pos) in WINDOWED.items():
        if col in f.columns:
            f.loc[~((pos >= min_pos) & contig[w]), col] = np.nan
    return f


def prepare_fno(fno: pd.DataFrame | None, symbol: str) -> pd.DataFrame | None:
    """Filter the daily F&O table to one symbol and rename columns to FNO_COLS."""
    if fno is None or len(fno) == 0:
        return None
    z = fno.copy()
    lower = {str(c).strip().lower(): c for c in z.columns}
    dc = next((lower[k] for k in ("date", "trade_date", "timestamp") if k in lower), None)
    sc = next((lower[k] for k in ("symbol", "ticker", "underlying") if k in lower), None)
    if dc is None or sc is None:
        return None
    z = z[z[sc].astype(str).str.upper().str.replace(".NS", "", regex=False) == symbol.upper()]
    if z.empty:
        return None
    out = pd.DataFrame({"_date": pd.to_datetime(z[dc], errors="coerce").dt.normalize().dt.tz_localize(None)})
    for name, aliases in FNO_ALIASES.items():
        src = next((lower[a] for a in aliases if a in lower), None)
        out[name] = pd.to_numeric(z[src], errors="coerce").to_numpy() if src else np.nan
    out = out.dropna(subset=["_date"]).drop_duplicates("_date", keep="last").sort_values("_date")
    return out if len(out) else None


def _fno_asof(idx: pd.DatetimeIndex, fno_sym: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=range(len(idx)), columns=FNO_COLS)
    out["fno_available"] = 0.0
    if fno_sym is None:
        return out.set_index(idx)
    # Normalize both join keys to nanosecond resolution: parquet round-trips can
    # store datetimes as micro/millisecond units and pandas rejects mixed-unit
    # merge_asof keys.
    dtype = "datetime64[ns]"
    left = pd.DataFrame(
        {
            "d": pd.to_datetime(idx.floor("D")).tz_localize(None).astype(dtype),
            "i": np.arange(len(idx)),
        }
    ).sort_values("d")
    right = fno_sym.copy()
    right["_date"] = pd.to_datetime(right["_date"]).astype(dtype)
    m = pd.merge_asof(left, right, left_on="d", right_on="_date", direction="backward",
                      allow_exact_matches=False, tolerance=pd.Timedelta(days=7)).sort_values("i")
    vals = m[FNO_COLS].to_numpy()
    out = pd.DataFrame(vals, columns=FNO_COLS, index=idx)
    # Match serving (champion_model_api._latest_features): availability means at
    # least one FNO value is finite, not merely that a dated row exists.
    out["fno_available"] = np.isfinite(vals).any(axis=1).astype(float)
    return out


def market_features_5m(m: pd.DataFrame) -> pd.DataFrame:
    """Features of the benchmark (NIFTY 50) on its own 5-minute bars (close-stamped)."""
    m = m.sort_index()
    c = m["Close"].astype(float)
    sess, pos, valid, contig = _session_info(m.index, warmup=1)
    out = pd.DataFrame(index=m.index)
    for n in (1, 3, 6, 12):
        out[f"mkt_ret_{n}b"] = c.pct_change(n, fill_method=None)
    out["mkt_vol_12b"] = c.pct_change(fill_method=None).rolling(WIN).std()
    return _mask_windows(out, pos, contig)


def make_features_5m(x: pd.DataFrame, mkt: pd.DataFrame | None = None,
                     fno_sym: pd.DataFrame | None = None, warmup: int = DEFAULT_WARMUP) -> pd.DataFrame:
    """Return FEATURES_5M for every valid decision bar (stamp <= 15:25) of one symbol.

    `fno_sym` is the output of prepare_fno for this symbol (or None)."""
    x = x.sort_index()
    idx = x.index
    c, h, l, o = (x[k].astype(float) for k in ("Close", "High", "Low", "Open"))
    v = x["Volume"].astype(float)
    sess, pos, valid, contig = _session_info(idx, warmup)
    g = lambda s: s.groupby(sess)  # noqa: E731

    f = pd.DataFrame(index=idx)
    for n in (1, 3, 6, 12):
        f[f"ret_{n}b"] = c.pct_change(n, fill_method=None)
    f["rsi6b"], f["rsi12b"] = _rsi(c, 6), _rsi(c, 12)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    f["atr12b_pct"] = tr.rolling(WIN).mean() / c
    for n in (6, 12):
        f[f"ma{n}_ratio"] = c / c.rolling(n).mean() - 1
    r1 = c.pct_change(fill_method=None)
    f["vol_6b"], f["vol_12b"] = r1.rolling(6).std(), r1.rolling(WIN).std()
    f["session_return"] = c / g(o).transform("first") - 1
    f["day_high_ratio"] = c / g(h).cummax() - 1
    f["day_low_ratio"] = c / g(l).cummin() - 1
    pv = (h + l + c) / 3.0 * v
    f["vwap_ratio"] = c / (g(pv).cumsum() / g(v).cumsum()) - 1
    f["range_pct"], f["body_pct"] = (h - l) / c, (c - o) / o
    vm = v.rolling(WIN).mean()
    f["volume_ratio_12"] = v / vm
    f["volume_z12"] = (v - vm) / v.rolling(WIN).std()
    f["volume_trend"] = v.rolling(3).mean() / vm - 1
    f["ret_accel"] = f["ret_1b"] - f["ret_6b"] / 6.0
    clock = idx.hour * 60 + idx.minute
    ang = 2 * np.pi * np.clip(clock - OPEN_MIN, 0, 375) / 375.0
    f["tod_sin"], f["tod_cos"] = np.sin(ang), np.cos(ang)
    f["minutes_to_close"] = np.clip(CLOSE_MIN - clock, 0, None).astype(float)

    # Daily context known at the open: overnight gap and the previous session's return.
    day_open = o.groupby(sess).first()
    day_close = c.groupby(sess).last()
    stale = pd.Series(day_open.index, index=day_open.index).diff().dt.days > 5
    gap = (day_open / day_close.shift(1) - 1).mask(stale)
    prev_ret = day_close.pct_change(fill_method=None).shift(1).mask(stale)
    key = pd.Series(sess, index=idx)
    f["gap_pct"] = key.map(gap).to_numpy()
    f["prev_day_ret"] = key.map(prev_ret).to_numpy()
    f["log_price"] = np.log1p(c.clip(lower=0))

    if mkt is not None and len(mkt):
        mf = mkt.reindex(idx)
    else:
        mf = pd.DataFrame(np.nan, index=idx, columns=["mkt_ret_1b", "mkt_ret_3b", "mkt_ret_6b", "mkt_ret_12b", "mkt_vol_12b"])
    for col in mf.columns:
        f[col] = mf[col].to_numpy()
    for n in (1, 3, 6, 12):
        f[f"rel_ret_{n}b"] = f[f"ret_{n}b"] - f[f"mkt_ret_{n}b"]

    fn = _fno_asof(idx, fno_sym)
    for col in FNO_COLS + ["fno_available"]:
        f[col] = fn[col].to_numpy()

    f = _mask_windows(f[FEATURES_5M].copy(), pos, contig)
    f = f.loc[valid & (clock <= CLOSE_MIN - 5)]
    return f.replace([np.inf, -np.inf], np.nan)


def make_label_close(x: pd.DataFrame) -> pd.Series:
    """fr_close = Close[15:30 bar of the same session] / Close[t] - 1; NaN if that bar is missing."""
    x = x.sort_index()
    idx = x.index
    c = x["Close"].astype(float)
    clock = idx.hour * 60 + idx.minute
    day = np.asarray(idx.floor("D"))
    end = c.where(clock == CLOSE_MIN).groupby(day).transform("max")
    fr = end / c - 1.0
    return fr.where(clock < CLOSE_MIN).rename("fr_close")


def build_training_table_5m(bars: pd.DataFrame, market: pd.DataFrame | None, fno: pd.DataFrame | None = None,
                            stride: int = 3, warmup: int = DEFAULT_WARMUP) -> pd.DataFrame:
    """Wide table: symbol, decision_ts, FEATURES_5M (float32), fr_close.

    `stride` keeps every k-th decision bar. All rows of a session share the same label endpoint,
    so rows are heavily overlapping: the independent unit is the SESSION, not the row.
    """
    mkt = market_features_5m(market) if market is not None and len(market) else None
    parts = []
    for sym, g in bars.groupby("symbol", sort=True):
        x = g.set_index("Timestamp")[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        feats = make_features_5m(x, mkt, prepare_fno(fno, sym), warmup)
        if feats.empty:
            continue
        lab = make_label_close(x).reindex(feats.index)
        t = pd.concat([feats.astype("float32"), lab.astype("float32")], axis=1)
        pos = pd.Series(0, index=x.index).groupby(np.asarray(x.index.floor("D"))).cumcount().reindex(t.index)
        t = t.loc[(pos.to_numpy() % max(1, stride)) == 0]
        t.insert(0, "symbol", sym)
        t.insert(1, "decision_ts", t.index)
        parts.append(t.reset_index(drop=True))
    if not parts:
        raise RuntimeError("No rows produced: check bar frequency (must be 5 minutes) and session completeness")
    return pd.concat(parts, ignore_index=True)
