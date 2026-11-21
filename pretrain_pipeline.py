#!/usr/bin/env python3
"""Pre-training and continuous-retraining pipeline for the ALL-NSE 5-minute
multi-horizon model (andromyth/indian-market-trading).

Run offline (Colab or local), never inside Streamlit. Output is a CANDIDATE
bundle plus a JSON report. Nothing here touches production schedules, the
Drive backfill jobs, or the live collectors.

Stages
  Target: ONE label per 5-minute decision bar: is the 15:30 close above the current price (fr_close).
  A  unify_bars       backfill + accumulated live snapshots -> one clean 5m bar table
  B  label table      features_5m.py (native 5-minute features; import the same module in serving)
  C  walk_forward     session-level expanding window, embargo, untouched final holdout
  D  thresholds       chosen on out-of-fold predictions by NET expectancy after costs
  E  gate             promote only if the untouched holdout clears statistical tests
  F  append_day       idempotent daily accumulation of new sessions into the store

Bundle layout per horizon matches the existing serving adapter keys
(classifier_hgb, classifier_logistic, regressor_hgb, regressor_ridge, blend)
and adds `thresholds` and `gate`, which the adapter must be changed to honour.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from scipy.stats import t as student_t
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features_5m import DEFAULT_WARMUP, FEATURES_5M, build_training_table_5m

IST = "Asia/Kolkata"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
GATE_CONF = 0.99               # one-sided; stricter than 95% because the gate is re-run every night


def tcrit(n_sessions: int, conf: float = GATE_CONF) -> float:
    """Student-t critical value: sessions are few (tens), so the normal quantile would be too lenient."""
    return float(student_t.ppf(conf, max(n_sessions - 1, 1)))


# ----------------------------------------------------------------------------
# Stage A: data unification
# ----------------------------------------------------------------------------
def _ist(ts: pd.Series) -> pd.Series:
    s = pd.to_datetime(ts, errors="coerce")
    return s.dt.tz_localize(IST) if s.dt.tz is None else s.dt.tz_convert(IST)


def unify_bars(backfill: pd.DataFrame | None, live: pd.DataFrame | None,
               min_bars: int = 68, max_gap_min: int = 10, stamp: str = "close") -> tuple[pd.DataFrame, dict]:
    """Merge historical backfill and accumulated live bars into one table.

    Rules (deliberately conservative):
      * columns: symbol, Timestamp (bar CLOSE time), Open, High, Low, Close, Volume
      * on a duplicate (symbol, Timestamp) the backfill bar wins; live only fills gaps
      * invalid prices, off-grid timestamps and out-of-session bars are dropped and counted
      * a (symbol, session) with too few bars or a long gap is EXCLUDED, never interpolated,
        because rolling features over a gap silently change meaning
    """
    parts = []
    for rank, df in ((0, backfill), (1, live)):
        if df is None or len(df) == 0:
            continue
        x = df[["symbol", "Timestamp", *OHLCV]].copy()
        x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
        x["Timestamp"] = _ist(x["Timestamp"])
        if stamp == "open":                       # re-stamp bar-open times to bar-close times
            x["Timestamp"] = x["Timestamp"] + pd.Timedelta(minutes=5)
        for c in OHLCV:
            x[c] = pd.to_numeric(x[c], errors="coerce")
        x["_rank"] = rank
        parts.append(x)
    if not parts:
        raise ValueError("No bars supplied")
    bars = pd.concat(parts, ignore_index=True)
    rep: dict = {"input_rows": int(len(bars))}

    ohlc = bars[["Open", "High", "Low", "Close"]]
    bad = ohlc.isna().any(axis=1) | (ohlc <= 0).any(axis=1) | (bars["High"] < bars["Low"]) | (bars["Volume"] < 0)
    bad |= bars["Timestamp"].isna()
    rep["dropped_invalid"] = int(bad.sum())
    bars = bars.loc[~bad]

    ts = bars["Timestamp"]
    mod = ts.dt.hour * 60 + ts.dt.minute
    on_grid = (ts.dt.minute % 5 == 0) & (ts.dt.second == 0)
    in_session = (mod >= 9 * 60 + 15) & (mod <= 15 * 60 + 30) & (ts.dt.dayofweek < 5)
    rep["dropped_off_grid"] = int((~on_grid).sum())
    rep["dropped_out_of_session"] = int((on_grid & ~in_session).sum())
    bars = bars.loc[on_grid & in_session]

    # Source agreement check: if live and backfill disagree, features differ at serve time.
    b0, b1 = bars[bars["_rank"] == 0], bars[bars["_rank"] == 1]
    if len(b0) and len(b1):
        m = b0.merge(b1, on=["symbol", "Timestamp"], suffixes=("_b", "_l"))
        if len(m):
            d = (m["Close_l"] / m["Close_b"] - 1).abs() * 1e4
            rep["source_agreement"] = {"overlap_rows": int(len(m)),
                                       "median_abs_close_diff_bps": float(d.median()),
                                       "share_over_50bps": float((d > 50).mean())}

    bars = bars.sort_values(["symbol", "Timestamp", "_rank"], kind="mergesort")
    before = len(bars)
    bars = bars.drop_duplicates(["symbol", "Timestamp"], keep="first")
    rep["duplicates_removed"] = int(before - len(bars))

    bars["session"] = bars["Timestamp"].dt.floor("D")
    grp = bars.groupby(["symbol", "session"], sort=False)["Timestamp"]
    n = grp.transform("size")
    gap = grp.diff().dt.total_seconds() / 60.0
    maxgap = gap.groupby([bars["symbol"], bars["session"]], sort=False).transform("max").fillna(0)
    first = grp.transform("first")
    open_like = (first.dt.hour * 60 + first.dt.minute) == 9 * 60 + 15     # a close-stamped session cannot start at 09:15
    n_ss = max(int(bars.groupby(["symbol", "session"]).ngroups), 1)
    n_open = int(bars.loc[open_like].groupby(["symbol", "session"]).ngroups)
    rep["open_stamped_sessions"] = n_open
    if n_open / n_ss > 0.01:
        raise RuntimeError(f"{n_open}/{n_ss} symbol-sessions start at 09:15: bars look OPEN-stamped. A close-stamp model would see "
                           "the label bar's data (leakage). Pass stamp='open' to unify_bars to re-stamp, or fix the source.")
    first_mode = (first.dt.hour * 60 + first.dt.minute).mode()
    rep["first_bar_stamp_mode_min"] = int(first_mode.iat[0]) if len(first_mode) else None
    last = grp.transform("max")
    ends_at_close = (last.dt.hour * 60 + last.dt.minute) == 15 * 60 + 30     # the 15:30 bar is the label endpoint
    keep = (n >= min_bars) & (maxgap <= max_gap_min) & ends_at_close & ~open_like   # stray open-stamped sessions: excluded, counted
    rep["symbol_sessions_total"] = int(bars.groupby(["symbol", "session"]).ngroups)
    rep["rows_excluded_incomplete_session"] = int((~keep).sum())
    bars = bars.loc[keep].drop(columns=["_rank"]).reset_index(drop=True)
    if bars.empty:
        raise RuntimeError("No complete sessions survive (need >=68 bars, gaps <=10 min, final bar stamped 15:30)")

    spacing = bars.groupby(["symbol", "session"], sort=False)["Timestamp"].diff().dt.total_seconds().dropna() / 60
    rep["median_bar_spacing_min"] = float(spacing.median()) if len(spacing) else None
    if rep["median_bar_spacing_min"] not in (None, 5.0):
        raise RuntimeError("Bars are not 5-minute: training on another frequency breaks feature meaning")
    rep["output_rows"] = int(len(bars))
    rep["sessions"] = int(bars["session"].nunique())
    rep["symbols"] = int(bars["symbol"].nunique())
    return bars, rep


# ----------------------------------------------------------------------------
# Stage C helpers: horizon table, folds, models
# ----------------------------------------------------------------------------
def coverage_report(tbl: pd.DataFrame) -> dict:
    """How much of the table actually has NIFTY and F&O inputs. Check this before trusting any gate result."""
    mkt = [c for c in FEATURES_5M if c.startswith(("mkt_", "rel_"))]
    ts = pd.to_datetime(tbl["decision_ts"], utc=True).dt.tz_convert(IST)
    by_month = tbl["fno_available"].groupby(ts.dt.strftime("%Y-%m")).mean().round(3)
    return {"market_nonnull_share": {c: round(float(tbl[c].notna().mean()), 4) for c in mkt},
            "fno_available_share": round(float(tbl["fno_available"].mean()), 4),
            "fno_available_by_month": by_month.to_dict()}


def prepare_target(tbl: pd.DataFrame, features: list[str], *, max_abs_return: float = 0.20) -> pd.DataFrame:
    """Single to-close target from the wide table. Label is same-session by construction."""
    z = tbl.loc[tbl["fr_close"].notna()].copy()
    z["future_return"] = z["fr_close"].astype(float)
    z = z.loc[z["future_return"].abs() <= max_abs_return]     # NSE circuit limit; beyond is a bad tick
    z["decision_ts"] = _ist(z["decision_ts"])
    z["session"] = z["decision_ts"].dt.floor("D")
    z["target"] = (z["future_return"] > 0).astype(int)
    z["sess_id"] = pd.factorize(z["session"], sort=True)[0]
    cols = ["symbol", "decision_ts", "session", "sess_id", "future_return", "target", *features]
    return z[cols].sort_values(["session", "decision_ts"]).reset_index(drop=True)


def make_folds(n_sessions: int, n_splits: int, min_train: int, embargo: int = 1):
    """Expanding window over session indices. `embargo` whole sessions separate train from test."""
    n_splits = max(1, min(n_splits, (n_sessions - min_train) // 10))
    if n_sessions - min_train < 10:
        raise ValueError(f"Need at least {min_train + 10} sessions for CV, have {n_sessions}")
    size = (n_sessions - min_train) // n_splits
    folds = []
    for k in range(n_splits):
        te0 = min_train + k * size
        te1 = n_sessions if k == n_splits - 1 else te0 + size
        folds.append((np.arange(0, te0 - embargo), np.arange(te0, te1)))
    return folds


def _imp():
    return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)


def fit_models(X: pd.DataFrame, y: np.ndarray, fr: np.ndarray, w: np.ndarray, seed: int) -> dict:
    lo, hi = np.nanquantile(fr, [0.005, 0.995])
    frc = np.clip(fr, lo, hi)
    m = {
        "classifier_hgb": Pipeline([("imputer", _imp()), ("model", HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=400,
            l2_regularization=5.0, random_state=seed))]),
        "classifier_logistic": Pipeline([("imputer", _imp()), ("scale", StandardScaler()),
                                         ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed))]),
        "regressor_hgb": Pipeline([("imputer", _imp()), ("model", HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=200, learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=400, l2_regularization=5.0, random_state=seed))]),
        "regressor_ridge": Pipeline([("imputer", _imp()), ("scale", StandardScaler()), ("model", Ridge(alpha=5.0))]),
    }
    m["classifier_hgb"].fit(X, y, model__sample_weight=w)
    m["classifier_logistic"].fit(X, y, model__sample_weight=w)
    m["regressor_hgb"].fit(X, frc)
    m["regressor_ridge"].fit(X, frc)
    up, dn = fr[fr > 0], fr[fr < 0]
    m["blend"] = {"classifier_hgb_weight": 0.5, "regressor_hgb_weight": 0.5, "return_alpha": 1.0,
                  "return_beta_direction": 0.0,
                  "direction_up_return": float(up.mean()) if len(up) else 0.001,
                  "direction_down_return": float(dn.mean()) if len(dn) else -0.001}
    return m


def predict_raw(m: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Blended probability BEFORE calibration, and expected return. The serving adapter uses this."""
    b = m["blend"]
    cw, rw = b["classifier_hgb_weight"], b["regressor_hgb_weight"]
    p = cw * m["classifier_hgb"].predict_proba(X)[:, 1] + (1 - cw) * m["classifier_logistic"].predict_proba(X)[:, 1]
    er = rw * m["regressor_hgb"].predict(X) + (1 - rw) * m["regressor_ridge"].predict(X)
    return np.clip(p, 0, 1), er


def calibrate(p: np.ndarray, spec: dict | None) -> np.ndarray:
    """Same arithmetic as probability_calibration.apply_probability_calibration (isotonic = np.interp)."""
    if not spec or spec.get("method") != "isotonic":
        return p
    return np.interp(np.clip(p, 1e-6, 1 - 1e-6), spec["x"], spec["y"])


def predict(m: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p, er = predict_raw(m, X)
    return calibrate(p, m.get("calibration")), er


def train_weights(fr: np.ndarray, sess_id: np.ndarray, half_life: float | None, mtc: np.ndarray) -> np.ndarray:
    """Sign-classifier weights. A 1% move with 5 minutes left is far more informative than with 5 hours
    left, so magnitude is standardised by sqrt(minutes to close) before weighting."""
    z = np.abs(fr) / np.sqrt(np.clip(mtc, 5, None))
    w = np.clip(z / max(np.median(z), 1e-12), 0.2, 5.0)
    if half_life:
        age = sess_id.max() - sess_id
        w = w * 0.5 ** (age / half_life)
    return w


# ----------------------------------------------------------------------------
# Stage D/E: trading-aware evaluation
# ----------------------------------------------------------------------------
def signal_net(p, er, fr, t_buy: float, cost: float, use_edge: bool = True) -> np.ndarray:
    """Net return per trade, NaN where no trade. The validated rule must equal the served rule.
    use_edge=True : probability passes the threshold AND |expected return| exceeds the round-trip cost
                    (needs a small capture patch, see INTEGRATION.md)
    use_edge=False: probability thresholds only, which is what the capture's `_signal` does today."""
    if use_edge:
        long_m = (p >= t_buy) & (er > cost)
        short_m = (p <= 1.0 - t_buy) & (er < -cost)
    else:
        long_m = p >= t_buy
        short_m = p <= 1.0 - t_buy
    net = np.full(len(p), np.nan)
    net[long_m] = fr[long_m] - cost
    net[short_m] = -fr[short_m] - cost
    return net


def _cluster_stat(values: np.ndarray, sessions: np.ndarray) -> dict:
    """Mean per-trade value with a session-clustered standard error (sessions are the independent units)."""
    ok = ~np.isnan(values)
    if ok.sum() == 0:
        return {"n": 0, "mean": np.nan, "lower": np.nan, "t": np.nan, "se": np.nan, "sessions": 0}
    s = pd.Series(values[ok]).groupby(sessions[ok]).mean()
    if len(s) < 5:
        return {"n": int(ok.sum()), "mean": float(values[ok].mean()), "lower": np.nan, "t": np.nan, "se": np.nan, "sessions": int(len(s))}
    se = s.std(ddof=1) / np.sqrt(len(s))
    mean = float(s.mean())
    crit = tcrit(len(s))
    return {"n": int(ok.sum()), "mean": mean, "se": float(se), "lower": float(mean - crit * se),
            "t": float(mean / se) if se > 0 else np.nan, "sessions": int(len(s))}


def session_auc(p, y, sessions, min_rows: int = 200) -> dict:
    vals = []
    for sid in np.unique(sessions):
        m = sessions == sid
        if m.sum() > min_rows and 0 < y[m].mean() < 1:
            vals.append(roc_auc_score(y[m], p[m]))
    if len(vals) < 5:
        return {"mean": np.nan, "lower": np.nan, "sessions": len(vals)}
    v = np.array(vals)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return {"mean": float(v.mean()), "lower": float(v.mean() - tcrit(len(v)) * se), "sessions": len(v)}


def bucket_auc(p, y, sessions, mtc) -> dict:
    """Session-mean AUC by time left to close. Pooled AUC can hide a model that only works late in the day."""
    out = {}
    for name, lo, hi in (("over_240m", 240, 1e9), ("120_to_240m", 120, 240), ("under_120m", 0, 120)):
        m = (mtc > lo) & (mtc <= hi) if lo else (mtc <= hi)
        out[name] = session_auc(p[m], y[m], sessions[m], min_rows=100) if m.sum() else None
    return out


def choose_threshold(p, er, fr, sessions, cost: float, min_trades: int, min_sessions: int = 20, use_edge: bool = True) -> dict | None:
    """One parameter (symmetric BUY/SELL) to limit overfitting. Picks the best session-clustered t-stat."""
    best = None
    for t in np.arange(0.52, 0.701, 0.01):
        st = _cluster_stat(signal_net(p, er, fr, t, cost, use_edge), sessions)
        if st["n"] >= min_trades and st["sessions"] >= min_sessions and np.isfinite(st["t"]) and st["mean"] > 0:
            if best is None or st["t"] > best["t"]:
                best = {"t_buy": round(float(t), 3), **st}
    return best


def run_target(z: pd.DataFrame, features: list[str], a: argparse.Namespace) -> tuple[dict, dict]:
    """Walk-forward, threshold choice, holdout gate and deployment refit for the single to-close target."""
    h = "close"
    n_sess = int(z["sess_id"].max()) + 1
    if n_sess <= a.holdout_sessions + a.min_train_sessions + 10:
        raise ValueError(f"to-close: {n_sess} sessions is too few for holdout={a.holdout_sessions}, min_train={a.min_train_sessions}")
    cv_n = n_sess - a.holdout_sessions - 1                     # one embargo session before holdout
    sid = z["sess_id"].to_numpy()
    cv = z[sid < cv_n]
    ho = z[sid >= n_sess - a.holdout_sessions]

    def xy(df):
        return df[features].astype("float32"), df["target"].to_numpy(int), df["future_return"].to_numpy(float)

    def cap(df):
        return df.sample(a.max_train_rows, random_state=7) if len(df) > a.max_train_rows else df

    oof = []
    fold_rows = []
    for k, (tr_s, te_s) in enumerate(make_folds(cv_n, a.n_splits, a.min_train_sessions)):
        tr = cap(cv[cv["sess_id"].isin(tr_s)])
        te = cv[cv["sess_id"].isin(te_s)]
        Xt, yt, ft = xy(tr)
        m = fit_models(Xt, yt, ft, train_weights(ft, tr["sess_id"].to_numpy(), a.half_life, tr["minutes_to_close"].to_numpy()), seed=20261100 + k)
        Xe, ye, fe = xy(te)
        p, er = predict(m, Xe)
        oof.append(pd.DataFrame({"p": p, "er": er, "fr": fe, "y": ye, "sid": te["sess_id"].to_numpy(), "fold": k}))
    oof = pd.concat(oof, ignore_index=True)
    # Probability calibration on out-of-fold predictions only. The weighted, class-balanced models are
    # overconfident; thresholds and the served `probability_up` must both be on the calibrated scale.
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof["p"], oof["y"])
    cal = {"method": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}   # repo spec format
    oof["p"] = calibrate(oof["p"].to_numpy(), cal)

    thr = choose_threshold(oof["p"].to_numpy(), oof["er"].to_numpy(), oof["fr"].to_numpy(),
                           oof["sid"].to_numpy(), a.cost, a.min_trades_cv, a.min_trade_sessions_cv,
                           getattr(a, "use_edge", True))

    # Frozen chain on the untouched holdout: models fit on CV sessions only.
    tr = cap(cv)
    Xt, yt, ft = xy(tr)
    m_eval = fit_models(Xt, yt, ft, train_weights(ft, tr["sess_id"].to_numpy(), a.half_life, tr["minutes_to_close"].to_numpy()), seed=20261199)
    m_eval["calibration"] = cal
    Xh, yh, fh = xy(ho)
    ph, erh = predict(m_eval, Xh)
    sh = ho["sess_id"].to_numpy()

    res = {"horizon": h, "cv_sessions": int(cv_n), "holdout_sessions": int(a.holdout_sessions),
           "cv_rows": int(len(cv)), "holdout_rows": int(len(ho)),
           "oof_auc_overall": float(roc_auc_score(oof["y"], oof["p"])),
           "holdout_auc": session_auc(ph, yh, sh),
           "holdout_brier": float(brier_score_loss(yh, ph)),
           "holdout_brier_baseline": float(brier_score_loss(yh, np.full(len(yh), cv["target"].mean()))),   # training base rate, not the oracle holdout rate
           "holdout_auc_by_time_left": bucket_auc(ph, yh, sh, ho["minutes_to_close"].to_numpy()),
           "cost_assumed": a.cost, "cv_threshold": thr}
    gate = {"auc_lower_gt_0.5": bool(np.isfinite(res["holdout_auc"]["lower"]) and res["holdout_auc"]["lower"] > 0.5)}
    if thr is None:
        res["holdout_trading"] = None
        gate.update({"net_expectancy_lower_gt_0": False, "min_trades": False, "trade_sessions": False, "cv_fold_stability": False})
    else:
        ue = getattr(a, "use_edge", True)
        st = _cluster_stat(signal_net(ph, erh, fh, thr["t_buy"], a.cost, ue), sh)
        net = signal_net(ph, erh, fh, thr["t_buy"], a.cost, ue)
        taken = ~np.isnan(net)
        res["holdout_trading"] = {**st, "mean_net_bps": st["mean"] * 1e4, "lower_net_bps": st["lower"] * 1e4, "se_bps": st["se"] * 1e4,
                                  "coverage": float(taken.mean()),
                                  "gross_hit_rate": float(((net[taken] + a.cost) > 0).mean()) if taken.any() else None}
        pos = []
        for k in sorted(oof["fold"].unique()):
            f = oof[oof["fold"] == k]
            s = _cluster_stat(signal_net(f["p"].to_numpy(), f["er"].to_numpy(), f["fr"].to_numpy(), thr["t_buy"], a.cost, ue), f["sid"].to_numpy())
            pos.append(bool(np.isfinite(s["mean"]) and s["mean"] > 0))
        res["cv_folds_positive"] = f"{sum(pos)}/{len(pos)}"
        gate["net_expectancy_lower_gt_0"] = bool(np.isfinite(st["lower"]) and st["lower"] > 0)
        gate["min_trades"] = st["n"] >= a.min_trades_holdout
        gate["trade_sessions"] = st["sessions"] >= a.min_trade_sessions_holdout    # rows overlap; sessions are the real sample
        gate["cv_fold_stability"] = (sum(pos) / len(pos)) >= 0.6
    res["gate"] = gate
    res["verdict"] = "PASS" if all(gate.values()) else "FAIL"

    # Deployment refit on every session (CV + holdout) with the frozen recipe.
    allz = cap(z)
    Xa, ya, fa = xy(allz)
    deploy = fit_models(Xa, ya, fa, train_weights(fa, allz["sess_id"].to_numpy(), a.half_life, allz["minutes_to_close"].to_numpy()), seed=20261299)
    deploy["calibration"] = cal
    deploy["training_symbols"] = sorted(z["symbol"].unique().tolist())
    deploy["feature_warmup"] = int(getattr(a, "warmup_bars", DEFAULT_WARMUP))
    deploy["thresholds"] = {"buy_p": thr["t_buy"] if thr else None, "sell_p": round(1 - thr["t_buy"], 3) if thr else None,
                            "min_edge": a.cost if getattr(a, "use_edge", True) else None}
    deploy["gate"] = res["verdict"]
    deploy["features"] = features
    deploy["feature_module"] = "features_5m"
    return res, deploy


# ----------------------------------------------------------------------------
# Stage F: daily accumulation (idempotent, partitioned by session date)
# ----------------------------------------------------------------------------
def append_day(store_root: str | Path, day: str, bars: pd.DataFrame) -> Path:
    """Write one session's validated 5m bars. Re-running a day overwrites that partition only."""
    d = Path(store_root) / "bars" / f"date={day}"
    d.mkdir(parents=True, exist_ok=True)
    tmp, final = d / "part.parquet.tmp", d / "part.parquet"
    bars.to_parquet(tmp, index=False)
    tmp.replace(final)
    return final


def load_store(store_root: str | Path) -> pd.DataFrame:
    files = sorted((Path(store_root) / "bars").glob("date=*/part.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backfill-parquet", help="historical 5m bars: symbol,Timestamp,Open..Volume")
    ap.add_argument("--store", help="accumulated live-bar store written by append_day")
    ap.add_argument("--market-parquet", required=True, help="NIFTY 50 bars with Timestamp, Open..Volume (5-minute, or finer: it is resampled)")
    ap.add_argument("--fno-parquet", help="daily F&O features (date, symbol, pcr, oi_change, fut_return, fut_oi, fut_basis)")
    ap.add_argument("--warmup-bars", type=int, default=DEFAULT_WARMUP, help="12 = strict (first decision 10:20); 3 = NaN warm-up, first decision 09:35")
    ap.add_argument("--no-edge-rule", dest="use_edge", action="store_false", help="probability-only signals (matches the capture as it is today)")
    ap.add_argument("--stamp", choices=["close", "open"], default="close", help="are the bar timestamps bar-close or bar-open times")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3, help="keep every k-th 5m decision bar")
    ap.add_argument("--max-train-rows", type=int, default=2_500_000)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--min-train-sessions", type=int, default=100, help="use ~80 when only 200 sessions exist")
    ap.add_argument("--holdout-sessions", type=int, default=40, help="use ~60 once 500 sessions exist")
    ap.add_argument("--half-life", type=float, default=None, help="recency half-life in sessions")
    ap.add_argument("--cost", type=float, default=0.0005, help="ASSUMED round-trip cost+slippage as a fraction; replace with your contract-note figure")
    ap.add_argument("--min-trades-cv", type=int, default=1500)
    ap.add_argument("--min-trades-holdout", type=int, default=500)
    ap.add_argument("--min-trade-sessions-cv", type=int, default=20)
    ap.add_argument("--min-trade-sessions-holdout", type=int, default=24)
    a = ap.parse_args()

    bf = pd.read_parquet(a.backfill_parquet) if a.backfill_parquet else None
    lv = load_store(a.store) if a.store else None
    bars, rep_a = unify_bars(bf, lv, stamp=a.stamp)
    market = load_market(a.market_parquet)
    fno = pd.read_parquet(a.fno_parquet) if a.fno_parquet else None
    tbl = build_training_table_5m(bars, market, fno, stride=a.stride, warmup=a.warmup_bars)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    z = prepare_target(tbl, FEATURES_5M)
    res, deploy = run_target(z, list(FEATURES_5M), a)
    joblib.dump({"close": deploy}, out / "candidate_bundle.joblib")
    (out / "training_report.json").write_text(json.dumps({"data": rep_a, "table_rows": int(len(tbl)), "coverage": coverage_report(tbl),
                                                          "result": res}, indent=2, default=str))
    print(f"to-close  {res['verdict']}  holdout AUC={res['holdout_auc']['mean']:.4f}  trading={res['holdout_trading']}")
    print("Candidate written. A FAIL verdict means serving must stay on HOLD.")


def load_market(path: str) -> pd.DataFrame:
    m = pd.read_parquet(path)
    m.columns = [str(c).strip().capitalize() if str(c).lower() != "timestamp" else "Timestamp" for c in m.columns]
    m = m.set_index(pd.DatetimeIndex(_ist(m.pop("Timestamp")))).sort_index()
    m = m[~m.index.duplicated(keep="last")]
    if len(m) > 2 and m.index.to_series().diff().median() >= pd.Timedelta(minutes=5):      # already 5-minute: check stamps
        first = m.index.to_series().groupby(m.index.floor("D")).first()
        if ((first.dt.hour * 60 + first.dt.minute) == 9 * 60 + 15).mean() > 0.5:
            raise RuntimeError("Market frame looks OPEN-stamped (sessions start 09:15); re-stamp to bar-close first")
    if len(m) > 2 and m.index.to_series().diff().median() < pd.Timedelta(minutes=5):
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        m = m.resample("5min", label="right", closed="right").agg(agg).dropna(subset=["Close"])
    return m


if __name__ == "__main__":
    main()
