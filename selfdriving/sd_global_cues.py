"""Global market cues with explicit point-in-time availability.

Every bar gets an `available_at` timestamp (UTC) = exchange close + publication lag.
A decision at time D may only see bars with available_at <= D. Bars older than
`max_age_days` are returned as NaN (a data outage is never filled with stale values).
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass
from datetime import time as dtime
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Series:
    ticker: str          # Yahoo Finance ticker (verify each once in your environment)
    tz: str              # exchange timezone
    close: dtime         # local close time used for availability
    lag_min: int = 15    # publication lag after close
    kind: str = "equity"  # equity | fx | commodity | rate | vol


# Close times: futures/FX/rates trade ~24h, so their daily bar is treated as
# available only at the end of the local calendar day (conservative).
GLOBAL_SERIES: Dict[str, Series] = {
    "spx": Series("^GSPC", "America/New_York", dtime(16, 0)),
    "ndx": Series("^IXIC", "America/New_York", dtime(16, 0)),
    "dji": Series("^DJI", "America/New_York", dtime(16, 0)),
    "vix": Series("^VIX", "America/New_York", dtime(16, 15), kind="vol"),
    "nikkei": Series("^N225", "Asia/Tokyo", dtime(15, 0)),
    "hsi": Series("^HSI", "Asia/Hong_Kong", dtime(16, 0)),
    "shanghai": Series("000001.SS", "Asia/Shanghai", dtime(15, 0)),
    "ftse": Series("^FTSE", "Europe/London", dtime(16, 30)),
    "dax": Series("^GDAXI", "Europe/Berlin", dtime(17, 30)),
    "brent": Series("BZ=F", "America/New_York", dtime(23, 59), 0, "commodity"),
    "gold": Series("GC=F", "America/New_York", dtime(23, 59), 0, "commodity"),
    "usdinr": Series("INR=X", "America/New_York", dtime(23, 59), 0, "fx"),
    "dxy": Series("DX-Y.NYB", "America/New_York", dtime(23, 59), 0, "fx"),
    "us10y": Series("^TNX", "America/New_York", dtime(23, 59), 0, "rate"),
    "indiavix": Series("^INDIAVIX", "Asia/Kolkata", dtime(15, 30), 5, "vol"),
}
RISK_ON_KEYS = ["spx", "ndx", "nikkei", "hsi", "dax", "ftse"]


def fetch_global_daily(start: str, end: str, series: Optional[Dict[str, Series]] = None,
                       retries: int = 3, pause: float = 2.0) -> pd.DataFrame:
    """Download daily closes. Returns long frame [key, date, close, available_at].

    Failures are recorded in `df.attrs['failed']`, never silently filled.
    """
    import yfinance as yf  # lazy: only needed on the machine that downloads

    series = series or GLOBAL_SERIES
    frames, failed = [], {}
    for key, s in series.items():
        last_err = None
        for k in range(retries):
            try:
                h = yf.download(s.ticker, start=start, end=end, interval="1d",
                                auto_adjust=False, progress=False, threads=False)
                if h is None or h.empty:
                    raise ValueError("empty response")
                if isinstance(h.columns, pd.MultiIndex):
                    h.columns = h.columns.get_level_values(0)
                f = h[["Close"]].dropna().rename(columns={"Close": "close"}).reset_index()
                f = f.rename(columns={f.columns[0]: "date"})
                f["key"] = key
                frames.append(f)
                last_err = None
                break
            except Exception as e:  # network, schema, delisted ticker
                last_err = e
                _time.sleep(pause * (k + 1))
        if last_err is not None:
            failed[key] = str(last_err)
    if not frames:
        out = pd.DataFrame(columns=["key", "date", "close", "available_at"])
    else:
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
        out["available_at"] = out.apply(lambda r: availability(r["date"], series[r["key"]]), axis=1)
    out.attrs["failed"] = failed
    return out


def availability(date: pd.Timestamp, s: Series) -> pd.Timestamp:
    local = pd.Timestamp(date).normalize() + pd.Timedelta(hours=s.close.hour, minutes=s.close.minute + s.lag_min)
    return local.tz_localize(s.tz, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


def per_series_features(long: pd.DataFrame, series: Optional[Dict[str, Series]] = None) -> Dict[str, pd.DataFrame]:
    """Features computed on each series' own trading calendar (so they are PIT-safe)."""
    series = series or GLOBAL_SERIES
    out: Dict[str, pd.DataFrame] = {}
    for key, g in long.groupby("key"):
        g = g.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        c = pd.to_numeric(g["close"], errors="coerce")
        kind = series[key].kind if key in series else "equity"
        if kind in ("rate",):
            r1 = c.diff()
        else:
            r1 = np.log(c).diff()
        sd20 = r1.rolling(20, min_periods=10).std()
        f = pd.DataFrame({"available_at": g["available_at"]})
        f[f"g_{key}_r1"] = r1
        f[f"g_{key}_r5"] = r1.rolling(5, min_periods=5).sum()
        f[f"g_{key}_z1"] = r1 / sd20.replace(0, np.nan)
        if kind == "vol":
            f[f"g_{key}_lvl_pctile252"] = c.rolling(252, min_periods=60).apply(
                lambda x: (x[:-1] < x[-1]).mean() if len(x) > 1 else np.nan, raw=True)
        out[key] = f.dropna(subset=["available_at"]).sort_values("available_at")
    return out


def attach_global_features(panel: pd.DataFrame, long: pd.DataFrame, max_age_days: float = 5.0,
                           series: Optional[Dict[str, Series]] = None) -> pd.DataFrame:
    """As-of join global features onto panel rows by `decision_timestamp` (UTC).

    Adds g_* columns, g_*_age_h staleness, plus composites `g_risk_on`, `g_oil_z`,
    `g_inr_z`. Rows with no available bar (or stale) get NaN, not zero.
    """
    x = panel.copy()
    # Parquet/pandas versions may infer datetime64[us, UTC] for one source and
    # datetime64[ns, UTC] for another. merge_asof requires identical key dtypes,
    # so enforce nanosecond UTC precision on both sides of every as-of join.
    x["decision_timestamp"] = (
        pd.to_datetime(x["decision_timestamp"], errors="coerce", utc=True)
        .astype("datetime64[ns, UTC]")
    )
    order = np.argsort(x["decision_timestamp"].to_numpy(), kind="stable")
    base = x.iloc[order].reset_index(drop=True)
    feats = per_series_features(long, series)
    z_cols = {}
    for key, f in feats.items():
        f = f.copy()
        f["available_at"] = (
            pd.to_datetime(f["available_at"], errors="coerce", utc=True)
            .astype("datetime64[ns, UTC]")
        )
        f = f.dropna(subset=["available_at"]).sort_values("available_at")
        f["_avail"] = f["available_at"]
        m = pd.merge_asof(base[["decision_timestamp"]], f, left_on="decision_timestamp",
                          right_on="available_at", direction="backward")
        age_h = (m["decision_timestamp"] - m["_avail"]).dt.total_seconds() / 3600.0
        stale = age_h > max_age_days * 24
        cols = [c for c in f.columns if c.startswith("g_")]
        for c in cols:
            base[c] = m[c].where(~stale).to_numpy()
        base[f"g_{key}_age_h"] = age_h.where(~stale).to_numpy()
        zc = f"g_{key}_z1"
        if zc in base:
            z_cols[key] = zc
    risk = [z_cols[k] for k in RISK_ON_KEYS if k in z_cols]
    if risk:
        base["g_risk_on"] = base[risk].clip(-4, 4).mean(axis=1, skipna=True)
    if "brent" in z_cols:
        base["g_oil_z"] = base[z_cols["brent"]]
    if "usdinr" in z_cols:
        base["g_inr_z"] = base[z_cols["usdinr"]]
    res = base.copy()
    res.index = x.index[order]
    return res.sort_index()


def parse_fii_dii(payload) -> pd.DataFrame:
    """Parse NSE FII/DII cash-market JSON (endpoint unverified; keys matched loosely).

    Published after market hours, so available_at = date 19:30 IST. History must be
    archived forward by the daily job (no free backfill assumed).
    """
    rows = []
    for r in (payload or []):
        cat = str(r.get("category", "")).upper()
        who = "fii" if "FII" in cat or "FPI" in cat else ("dii" if "DII" in cat else None)
        if who is None:
            continue
        try:
            d = pd.to_datetime(r.get("date"), dayfirst=True, errors="coerce")
            net = float(str(r.get("netValue")).replace(",", ""))
        except Exception:
            continue
        if pd.isna(d):
            continue
        rows.append({"who": who, "date": d.normalize(), "net": net})
    if not rows:
        return pd.DataFrame(columns=["date", "g_fii_net", "g_dii_net", "available_at"])
    w = pd.DataFrame(rows).pivot_table(index="date", columns="who", values="net", aggfunc="last").reset_index()
    w = w.rename(columns={"fii": "g_fii_net", "dii": "g_dii_net"})
    for c in ("g_fii_net", "g_dii_net"):
        if c not in w:
            w[c] = np.nan
    w["available_at"] = (w["date"] + pd.Timedelta(hours=19, minutes=30)).dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
    return w.sort_values("date")
