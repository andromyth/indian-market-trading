"""F&O regime features derived from the daily symbol-level UDiFF layer.

Input: the daily frame (one row per symbol and trade `date`) that
nse_fno_udiff_ingest.py already produces (fno_fut_close, fno_fut_oi, fno_pcr,
fno_iv_change, fno_fut_basis, ...). Features are computed along each symbol's own
trade-date series. The caller must still apply the existing next-session 09:15
availability shift before joining to a decision clock.

Scope limit: these predict the UNDERLYING's direction and volatility regime. A
contract-level options model needs historical option-chain/IV history, which the
repo does not archive yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _z(s: pd.Series, n: int = 20) -> pd.Series:
    mu = s.rolling(n, min_periods=max(5, n // 2)).mean()
    sd = s.rolling(n, min_periods=max(5, n // 2)).std()
    return (s - mu) / sd.replace(0, np.nan)


def build_fno_regime_features(f: pd.DataFrame) -> pd.DataFrame:
    x = f.sort_values(["symbol", "date"]).copy()
    for c in ("fno_fut_close", "fno_fut_oi", "fno_pcr", "fno_iv_change", "fno_fut_basis"):
        if c not in x:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    g = x.groupby("symbol", sort=False)
    x["fut_ret"] = g["fno_fut_close"].pct_change()
    x["fut_oi_chg"] = g["fno_fut_oi"].pct_change()
    r, o = np.sign(x["fut_ret"]), np.sign(x["fut_oi_chg"])
    code = pd.Series(np.nan, index=x.index)
    code[(r > 0) & (o > 0)] = 2    # long build-up
    code[(r > 0) & (o < 0)] = 1    # short covering
    code[(r < 0) & (o < 0)] = -1   # long unwinding
    code[(r < 0) & (o > 0)] = -2   # short build-up
    x["fno_buildup"] = code
    x["fno_pcr_z20"] = g["fno_pcr"].transform(_z)
    x["fno_basis_z20"] = g["fno_fut_basis"].transform(_z)
    x["fno_oi_chg_z20"] = g["fut_oi_chg"].transform(_z)
    x["fno_iv_chg_z20"] = g["fno_iv_change"].transform(_z)
    x["fno_buildup_persist3"] = g["fno_buildup"].transform(lambda s: s.rolling(3, min_periods=2).sum())
    keep = ["symbol", "date", "fno_buildup", "fno_buildup_persist3", "fno_pcr_z20",
            "fno_basis_z20", "fno_oi_chg_z20", "fno_iv_chg_z20"]
    return x[keep]


def expected_move_pct(close: pd.Series, window: int = 20) -> pd.Series:
    """Realized-vol based 1-session expected absolute move (sizing/strike-distance input)."""
    r = np.log(close).diff()
    return r.rolling(window, min_periods=10).std() * np.sqrt(2 / np.pi)
