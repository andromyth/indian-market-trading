"""Purged walk-forward training, calibration, evaluation and abstaining signals.

Design choices aimed at real out-of-sample accuracy rather than a flattering number:
  * Target is market-neutral by default (stock beats the same-day cross-sectional median),
    so the score is not just "was the whole market up".
  * Expanding-window folds with a purge (label horizon) and an embargo between train and test.
  * HistGradientBoosting (native NaN handling, strong regularisation) blended with a
    regularised logistic model; blend weight and isotonic calibration are fit on a
    time-ordered validation tail inside each training window, never on the test fold.
  * Accuracy is always reported next to the base rate, with date-block bootstrap CIs.
  * Thresholds for acting are chosen from out-of-fold predictions using the Wilson lower
    bound of precision, so a lucky band cannot be selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------- targets
def add_targets(panel: pd.DataFrame, ret_col: str = "y_return", date_col: str = "date") -> pd.DataFrame:
    """Adds y_excess (return minus same-day median) and y_up_excess (0/1)."""
    x = panel.copy()
    x[date_col] = pd.to_datetime(x[date_col]).dt.normalize()
    med = x.groupby(date_col)[ret_col].transform("median")
    x["y_excess"] = x[ret_col] - med
    x["y_up_excess"] = (x["y_excess"] > 0).astype(int)
    x["y_up_abs"] = (x[ret_col] > 0).astype(int)
    return x


def add_cross_sectional_ranks(panel: pd.DataFrame, cols: Sequence[str], date_col: str = "date",
                              suffix: str = "_csr") -> pd.DataFrame:
    """Percentile rank of each feature within the same-day cross-section (regime robust)."""
    x = panel.copy()
    g = x.groupby(date_col)
    for c in cols:
        if c in x:
            x[c + suffix] = g[c].rank(pct=True)
    return x


# ----------------------------------------------------------------------------- folds
@dataclass
class FoldSpec:
    n_folds: int = 8
    test_days: int = 21
    min_train_days: int = 252
    horizon_days: int = 1     # label horizon: rows this close to the test start are purged
    embargo_days: int = 1


def purged_folds(dates: Sequence, spec: FoldSpec) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_dates, test_dates), most recent fold last, expanding train window."""
    d = np.array(sorted(pd.to_datetime(pd.Series(dates)).dt.normalize().unique()))
    n = len(d)
    need = spec.min_train_days + spec.n_folds * spec.test_days + spec.horizon_days + spec.embargo_days
    if n < need:
        raise ValueError(f"Need >= {need} sessions for {spec.n_folds} folds, have {n}")
    first_test = n - spec.n_folds * spec.test_days
    for k in range(spec.n_folds):
        t0 = first_test + k * spec.test_days
        t1 = t0 + spec.test_days
        train_end = t0 - spec.horizon_days - spec.embargo_days
        yield d[:train_end], d[t0:t1]


# ----------------------------------------------------------------------------- model
@dataclass
class ModelConfig:
    name: str = "hgb_logit_blend"
    hgb_params: Dict = field(default_factory=lambda: dict(
        max_depth=4, learning_rate=0.04, max_iter=250, min_samples_leaf=200,
        l2_regularization=5.0, max_bins=63, early_stopping=False, random_state=7))
    logit_C: float = 0.05
    blend_grid: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)   # weight on HGB
    val_frac: float = 0.15
    calibrate: bool = True


def _logit_pipe(C: float) -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=400, random_state=7)),
    ])


@dataclass
class Bundle:
    features: List[str]
    hgb: HistGradientBoostingClassifier
    logit: Pipeline
    w_hgb: float
    iso: Optional[IsotonicRegression]
    meta: Dict
    return_model: Optional[HistGradientBoostingRegressor] = None

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features].apply(pd.to_numeric, errors="coerce")
        return self.w_hgb * self.hgb.predict_proba(X)[:, 1] + (1 - self.w_hgb) * self.logit.predict_proba(X)[:, 1]

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        p = self.raw(X)
        return np.clip(self.iso.predict(p), 0.001, 0.999) if self.iso is not None else p

    def predict_expected_return(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Predict continuous raw forward return when this bundle has a return head.

        Old joblib bundles lack this head; returning None lets the UI explicitly
        report that a price forecast requires a retrained/retrofitted bundle.
        """
        model = getattr(self, "return_model", None)
        if model is None:
            return None
        matrix = _clean_model_matrix(X, self.features)
        values = np.asarray(model.predict(matrix), dtype=float)
        values[~np.isfinite(values)] = np.nan
        return values


def _clean_model_matrix(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Convert model inputs to finite numeric values without fabricating data."""
    x = df[list(features)].apply(pd.to_numeric, errors="coerce")
    float32_max = float(np.finfo(np.float32).max)
    for c in x.columns:
        v = x[c]
        x[c] = v.mask(~np.isfinite(v) | (v.abs() > float32_max), np.nan)
    return x.astype("float32")


def fit_return_regressor(
    df: pd.DataFrame,
    features: Sequence[str],
    target_col: str,
    date_col: str = "date",
    X: Optional[pd.DataFrame] = None,
) -> Tuple[Optional[HistGradientBoostingRegressor], Dict]:
    """Fit a continuous forward-return head and report chronological-tail diagnostics.

    The classifier/action thresholds are not involved in this head. Validation
    compares model MAE against a training-period median-return baseline on later
    dates. This is a diagnostic, not a substitute for purged walk-forward tests.
    """
    empty_meta = {
        "target": target_col,
        "status": "UNAVAILABLE",
        "validation_method": "chronological_tail_vs_training_median_baseline",
        "validation_rows": 0,
        "validation_mae": None,
        "baseline_mae": None,
        "mae_skill_vs_baseline": None,
        "validation_directional_accuracy": None,
    }
    if target_col not in df.columns or date_col not in df.columns or df.empty:
        return None, {**empty_meta, "reason": "missing target/date or empty training frame"}

    y_series = pd.to_numeric(df[target_col], errors="coerce")
    y = y_series.to_numpy(dtype=float, na_value=np.nan)
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    date_values = dates.to_numpy()
    valid = np.isfinite(y) & dates.notna().to_numpy()
    valid_positions = np.flatnonzero(valid)
    if len(valid_positions) < 50:
        return None, {**empty_meta, "reason": f"only {len(valid_positions)} valid return labels"}
    if X is None:
        X = _clean_model_matrix(df, features)
    elif list(X.columns) != list(features):
        X = _clean_model_matrix(X, features)

    unique_dates = np.array(sorted(pd.Series(dates[valid]).dropna().unique()))
    if len(unique_dates) >= 2:
        split_pos = max(1, min(len(unique_dates) - 1, int(len(unique_dates) * 0.85)))
        cutoff = unique_dates[split_pos]
        train_mask = valid & (date_values < cutoff)
        valid_mask = valid & (date_values >= cutoff)
    else:
        # Same-day-only data is not enough for a credible chronological check.
        return None, {
            **empty_meta,
            "reason": "at least two distinct sessions required for chronological validation",
        }

    train_positions = np.flatnonzero(train_mask)
    validation_positions = np.flatnonzero(valid_mask)
    if len(train_positions) < 30 or len(validation_positions) < 30:
        return None, {
            **empty_meta,
            "reason": (
                f"insufficient chronological split: train={len(train_positions)}, "
                f"validation={len(validation_positions)}"
            ),
        }

    def _params(n_rows: int) -> Dict:
        return {
            "max_depth": 3,
            "learning_rate": 0.04,
            "max_iter": 180,
            "min_samples_leaf": max(5, min(120, n_rows // 30)),
            "l2_regularization": 8.0,
            "max_bins": 63,
            "early_stopping": False,
            "random_state": 37,
        }

    validation_model = HistGradientBoostingRegressor(**_params(len(train_positions)))
    validation_model.fit(X.iloc[train_positions], y[train_positions])
    validation_prediction = np.asarray(
        validation_model.predict(X.iloc[validation_positions]), dtype=float
    )
    actual = y[validation_positions]
    baseline = float(np.median(y[train_positions]))
    model_mae = float(np.mean(np.abs(actual - validation_prediction)))
    baseline_mae = float(np.mean(np.abs(actual - baseline)))
    skill = (
        float(1.0 - model_mae / baseline_mae)
        if np.isfinite(baseline_mae) and baseline_mae > 0 else None
    )
    direction_acc = float(
        np.mean(np.sign(validation_prediction) == np.sign(actual))
    )
    status = (
        "MAE_BEATS_MEDIAN_BASELINE"
        if skill is not None and skill > 0
        else "MAE_NOT_BETTER_THAN_MEDIAN_BASELINE"
    )

    final_model = HistGradientBoostingRegressor(**_params(len(valid_positions)))
    final_model.fit(X.iloc[valid_positions], y[valid_positions])
    diagnostics = {
        "target": target_col,
        "status": status,
        "validation_method": "chronological_tail_vs_training_median_baseline",
        "train_rows": int(len(train_positions)),
        "validation_rows": int(len(validation_positions)),
        "validation_sessions": int(pd.Series(dates.iloc[validation_positions]).nunique()),
        "validation_mae": model_mae,
        "baseline_mae": baseline_mae,
        "mae_skill_vs_baseline": skill,
        "validation_directional_accuracy": direction_acc,
        "validation_start": pd.Timestamp(unique_dates[split_pos]).date().isoformat(),
        "validation_end": pd.Timestamp(unique_dates[-1]).date().isoformat(),
        "note": (
            "Chronological-tail diagnostic. It is not an untouched holdout or "
            "a guarantee of live forecast accuracy."
        ),
    }
    return final_model, diagnostics


def fit_bundle(df: pd.DataFrame, features: Sequence[str], target: str, cfg: ModelConfig,
               date_col: str = "date") -> Bundle:
    features = list(features)
    d = np.array(sorted(pd.to_datetime(df[date_col]).dt.normalize().unique()))
    cut = d[int(len(d) * (1 - cfg.val_frac))]
    dd = pd.to_datetime(df[date_col]).dt.normalize()
    tr, va = df[dd < cut], df[dd >= cut]
    Xtr = _clean_model_matrix(tr, features)
    ytr = tr[target].astype(int)
    Xva = _clean_model_matrix(va, features)
    yva = va[target].astype(int)
    hgb = HistGradientBoostingClassifier(**cfg.hgb_params).fit(Xtr, ytr)
    lg = _logit_pipe(cfg.logit_C).fit(Xtr, ytr)
    ph, pl = hgb.predict_proba(Xva)[:, 1], lg.predict_proba(Xva)[:, 1]
    best_w, best_ll = 0.5, np.inf
    for w in cfg.blend_grid:
        ll = log_loss(yva, np.clip(w * ph + (1 - w) * pl, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_w, best_ll = w, ll
    pv = best_w * ph + (1 - best_w) * pl
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(pv, yva) if cfg.calibrate else None
    # refit base learners on the full window (calibration map stays from the held-out tail)
    Xall = _clean_model_matrix(df, features)
    yall = df[target].astype(int)
    hgb = HistGradientBoostingClassifier(**cfg.hgb_params).fit(Xall, yall)
    lg = _logit_pipe(cfg.logit_C).fit(Xall, yall)
    # Attach an independent continuous-return head only when a raw forward-return
    # target is part of the training frame. Intraday uses fwd_ret; Delivery uses y_return.
    return_target = next(
        (name for name in ("fwd_ret", "y_return") if name in df.columns),
        None,
    )
    return_model = None
    return_metrics = {
        "status": "UNAVAILABLE",
        "reason": "training frame contains neither fwd_ret nor y_return",
    }
    if return_target is not None:
        return_model, return_metrics = fit_return_regressor(
            df, features, return_target, date_col=date_col, X=Xall
        )

    meta = {"config": cfg.name, "target": target, "n_rows": int(len(df)), "val_logloss": float(best_ll),
            "w_hgb": float(best_w), "cfg": {"hgb_params": cfg.hgb_params, "logit_C": cfg.logit_C, "name": cfg.name, "val_frac": cfg.val_frac}, "train_start": pd.Timestamp(d[0]).date().isoformat(),
            "train_end": pd.Timestamp(d[-1]).date().isoformat(),
            "return_forecast": return_metrics}
    return Bundle(features, hgb, lg, float(best_w), iso, meta, return_model)


# ----------------------------------------------------------------------------- run
def run_walkforward(panel: pd.DataFrame, features: Sequence[str], target: str = "y_up_excess",
                    ret_col: str = "y_return", cfg: Optional[ModelConfig] = None,
                    spec: Optional[FoldSpec] = None, date_col: str = "date") -> pd.DataFrame:
    """Out-of-fold predictions. Columns: date, symbol, fold, p, y, ret, ret_excess."""
    cfg, spec = cfg or ModelConfig(), spec or FoldSpec()
    dts = pd.to_datetime(panel[date_col]).dt.normalize()
    outs = []
    for k, (tr_d, te_d) in enumerate(purged_folds(dts, spec)):
        tr = panel[dts.isin(tr_d)]
        te = panel[dts.isin(te_d)]
        if tr[target].nunique() < 2 or te.empty:
            continue
        b = fit_bundle(tr, features, target, cfg, date_col)
        o = pd.DataFrame({
            "date": pd.to_datetime(te[date_col]).dt.normalize().to_numpy(),
            "symbol": te["symbol"].to_numpy() if "symbol" in te else "X",
            "fold": k, "p": b.predict_proba_up(te), "p_raw": b.raw(te),
            "y": te[target].astype(int).to_numpy(), "ret": te[ret_col].to_numpy(),
        })
        o["ret_excess"] = o["ret"] - o.groupby("date")["ret"].transform("median")
        outs.append(o)
    if not outs:
        raise RuntimeError("No usable folds")
    return pd.concat(outs, ignore_index=True)


# ----------------------------------------------------------------------------- stats
def wilson_lower(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    ph = k / n
    den = 1 + z * z / n
    c = ph + z * z / (2 * n)
    a = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return float((c - a) / den)


def block_bootstrap_ci(values: pd.DataFrame, n_boot: int = 500, seed: int = 11) -> Tuple[float, float]:
    """values: per-date frame with columns hit, n. Resamples DATES (not rows) for the CI."""
    rng = np.random.default_rng(seed)
    h, n = values["hit"].to_numpy(float), values["n"].to_numpy(float)
    m = len(h)
    if m < 5:
        return (np.nan, np.nan)
    acc = []
    for _ in range(n_boot):
        i = rng.integers(0, m, m)
        acc.append(h[i].sum() / max(n[i].sum(), 1))
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def evaluate(pred: pd.DataFrame, cost_bps: float = 30.0, bands: Sequence[float] = (0, .02, .04, .06, .08, .10, .15)) -> Dict:
    """Metrics for out-of-fold predictions. cost_bps is the ROUND-TRIP cost per leg (assumption)."""
    y, p = pred["y"].to_numpy(), pred["p"].to_numpy()
    hit = ((p >= 0.5).astype(int) == y)
    per_date = pd.DataFrame({"date": pred["date"], "hit": hit.astype(float)}).groupby("date").agg(hit=("hit", "sum"), n=("hit", "size"))
    lo, hi = block_bootstrap_ci(per_date)
    base = max(y.mean(), 1 - y.mean())
    res: Dict = {
        "rows": int(len(pred)), "dates": int(pred["date"].nunique()),
        "accuracy": float(hit.mean()), "accuracy_ci95": [lo, hi], "base_rate_majority": float(base),
        "lift_vs_base_pp": float((hit.mean() - base) * 100),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "binom_p_vs_base": float(binomtest(int(hit.sum()), len(hit), base, alternative="greater").pvalue),
    }
    ic = pred.groupby("date").apply(
        lambda g: spearmanr(g["p"], g["ret_excess"])[0] if g["p"].nunique() > 2 else np.nan, include_groups=False).dropna()
    res["ic_mean"] = float(ic.mean()); res["ic_std"] = float(ic.std(ddof=1))
    res["ic_t"] = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 2 and ic.std(ddof=1) > 0 else np.nan
    res["ic_pos_frac"] = float((ic > 0).mean())
    sel = []
    for b in bands:
        m = np.abs(p - 0.5) >= b
        if m.sum() == 0:
            continue
        k = int(hit[m].sum()); n = int(m.sum())
        sel.append({"min_conf": b, "coverage": float(n / len(p)), "n": n, "accuracy": k / n, "wilson_lb": wilson_lower(k, n)})
    res["selective"] = sel
    # decile spread, net of cost: long top decile, short bottom decile, daily turnover assumed 100%
    def dec(g):
        if len(g) < 20:
            return np.nan
        q = g["p"].rank(pct=True)
        return g.loc[q >= 0.9, "ret_excess"].mean() - g.loc[q <= 0.1, "ret_excess"].mean()
    spread = pred.groupby("date").apply(dec, include_groups=False).dropna()
    res["decile_spread_gross_bps"] = float(spread.mean() * 1e4)
    res["decile_spread_net_bps"] = float(spread.mean() * 1e4 - 2 * cost_bps)
    res["cost_bps_assumed_roundtrip_per_leg"] = cost_bps
    res["fold_auc"] = [float(roc_auc_score(g["y"], g["p"])) for _, g in pred.groupby("fold") if g["y"].nunique() > 1]
    res["per_date_hit"] = per_date  # for paired comparison; drop before JSON dumping
    return res


def choose_thresholds(pred: pd.DataFrame, target_precision: float = 0.56, min_coverage: float = 0.02,
                      min_n: int = 500) -> Dict:
    """Smallest |p-0.5| band whose Wilson lower-bound precision >= target, on OOF data.

    If no band qualifies the system must ABSTAIN (returns {'abstain_all': True}).
    """
    p, y = pred["p"].to_numpy(), pred["y"].to_numpy()
    hit = ((p >= 0.5).astype(int) == y)
    for b in np.arange(0.0, 0.30, 0.01):
        m = np.abs(p - 0.5) >= b
        n = int(m.sum())
        if n < min_n or n / len(p) < min_coverage:
            break
        k = int(hit[m].sum())
        lb = wilson_lower(k, n)
        if lb >= target_precision:
            return {"abstain_all": False, "min_conf": float(round(b, 2)), "buy_p": float(round(0.5 + b, 3)),
                    "sell_p": float(round(0.5 - b, 3)), "coverage": n / len(p), "oof_accuracy": k / n,
                    "wilson_lb": lb, "n": n}
    return {"abstain_all": True, "reason": f"no band reached Wilson LB >= {target_precision}"}


def to_signal(p: float, thr: Dict, market_p: Optional[float] = None, market_gate: float = 0.47) -> str:
    """BUY/SELL only outside the validated band, with an optional market-direction gate."""
    if thr.get("abstain_all", True):
        return "ABSTAIN"
    if p >= thr["buy_p"]:
        return "BUY" if (market_p is None or market_p >= market_gate) else "HOLD"
    if p <= thr["sell_p"]:
        return "SELL" if (market_p is None or market_p <= 1 - market_gate) else "HOLD"
    return "HOLD"


def paired_logloss_gain(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 500, seed: int = 5) -> Dict:
    """Challenger b vs champion a on identical rows: mean log-loss reduction with date-block CI."""
    m = a.merge(b, on=["date", "symbol"], suffixes=("_a", "_b"))
    la = -(m["y_a"] * np.log(np.clip(m["p_a"], 1e-6, 1)) + (1 - m["y_a"]) * np.log(np.clip(1 - m["p_a"], 1e-6, 1)))
    lb = -(m["y_b"] * np.log(np.clip(m["p_b"], 1e-6, 1)) + (1 - m["y_b"]) * np.log(np.clip(1 - m["p_b"], 1e-6, 1)))
    d = pd.DataFrame({"date": m["date"], "gain": la - lb}).groupby("date")["gain"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"mean_gain": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "dates": int(len(d)), "frac_dates_better": float((d > 0).mean())}
