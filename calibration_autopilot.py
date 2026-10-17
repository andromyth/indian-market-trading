#!/usr/bin/env python3
"""Evidence-gated probability and return calibration for intraday forecasts.

Pure calibration logic only. It never edits model artifacts or places orders.
The worker overlay defaults to shadow mode and fails open.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
SCHEMA_VERSION = 1
LEARNING_ROOT = "production_data/intraday_predictions/learning"
REMOTE_DIR = f"{LEARNING_ROOT}/calibration"
REGISTRY_NAME = "calibration_registry.json"
ACTIVE_PROFILE_NAME = "active_calibration_profile.json"
STAGE_SHADOW = "SHADOW"
STAGE_ACTIVE = "ACTIVE"
LIVE_STAGES = (STAGE_SHADOW, STAGE_ACTIVE)
_EPS = 1e-6


@dataclass(frozen=True)
class Rules:
    min_rows: int = 600
    min_dates: int = 15
    max_input_age_seconds: float = 120.0
    min_train_dates: int = 8
    n_folds: int = 4
    embargo_dates: int = 1
    min_fold_train_rows: int = 200
    n_boot: int = 400
    alpha: float = 0.05
    ridge_lambda: float = 0.02
    slope_bounds: tuple = (0.2, 2.0)
    shift_bound: float = 1.0
    p_floor: float = 0.02
    p_ceil: float = 0.98
    max_ece_worsen: float = 0.002
    max_mean_abs_change: float = 0.25
    large_change_flag: float = 0.10
    min_fold_win_rate: float = 0.6
    ph_delta: float = 0.002
    ph_threshold: float = 0.05
    isotonic_min_rows: int = 2500
    isotonic_min_dates: int = 30
    min_return_rows: int = 600
    return_slope_bounds: tuple = (0.0, 2.0)
    cost_bps_roundtrip: float = 12.0
    band_grid: tuple = tuple(round(x, 2) for x in np.arange(0.02, 0.201, 0.01))
    band_looks: int = 4
    min_action_rows: int = 300
    min_action_dates: int = 10
    min_precision_lb: float = 0.52
    shadow_min_dates: int = 8
    shadow_max_dates: int = 20
    shadow_alpha: float = 0.10
    demote_min_dates: int = 5
    cooldown_days: int = 5
    valid_for_days: int = 35
    refit_after_days: int = 7
    refresh_max_param_distance: float = 0.03
    psi_alarm: float = 0.25


class AutopilotStateError(RuntimeError):
    """Persisted state is not safe to overwrite."""


def profile_key(horizon_min: Any, model_version: Any) -> str:
    try:
        horizon_text = str(int(horizon_min))
    except (TypeError, ValueError):
        horizon_text = "unknown"
    version = "UNKNOWN" if model_version is None else str(model_version)
    return f"h{horizon_text}__{version.replace('/', '_')}"


def _logit(p: Any) -> np.ndarray:
    v = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(v / (1 - v))


def _sigmoid(z: Any) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(np.asarray(z, dtype=float), -35, 35)))


def _row_logloss(p: Any, y: Any) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _seed(key: str, tag: str) -> int:
    return int(hashlib.sha1(f"{key}|{tag}".encode()).hexdigest()[:8], 16)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _utc(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_iso(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def date_block_bootstrap(values: Any, dates: Any, *, n_boot: int, alpha: float, seed: int) -> dict[str, Any]:
    values, dates = np.asarray(values, float), np.asarray(dates)
    ok = np.isfinite(values)
    values, dates = values[ok], dates[ok]
    if not len(values):
        return {"mean": None, "lo": None, "hi": None, "dates": 0}
    per = pd.DataFrame({"d": dates, "v": values}).groupby("d")["v"].mean().to_numpy()
    n = len(per)
    result = {"mean": float(per.mean()), "lo": None, "hi": None, "dates": int(n)}
    if n < 2:
        return result
    rng = np.random.default_rng(seed)
    samples = per[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    result.update(lo=float(np.percentile(samples, 100 * alpha / 2)),
                  hi=float(np.percentile(samples, 100 * (1 - alpha / 2))))
    return result


def date_block_ratio_bootstrap(hits: Any, counts: Any, *, n_boot: int, alpha: float, seed: int) -> dict[str, Any]:
    h, c = np.asarray(hits, float), np.asarray(counts, float)
    if not len(h) or c.sum() <= 0:
        return {"ratio": None, "lo": None, "hi": None, "dates": 0}
    result = {"ratio": float(h.sum() / c.sum()), "lo": None, "hi": None, "dates": int(len(h))}
    if len(h) < 2:
        return result
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(h), size=(n_boot, len(h)))
    boot = h[idx].sum(axis=1) / np.maximum(c[idx].sum(axis=1), 1)
    result.update(lo=float(np.percentile(boot, 100 * alpha / 2)),
                  hi=float(np.percentile(boot, 100 * (1 - alpha / 2))))
    return result


def _lo(summary: dict[str, Any]) -> float:
    return float(summary["lo"]) if summary.get("lo") is not None else float("-inf")


def _hi(summary: dict[str, Any]) -> float:
    return float(summary["hi"]) if summary.get("hi") is not None else float("inf")


def _fit_logit_affine(p: Any, y: Any, *, fit_slope: bool, rules: Rules) -> tuple[float, float]:
    z, yy = _logit(p), np.asarray(y, float)
    lam, lower, upper = rules.ridge_lambda, *rules.slope_bounds
    a, b = 0.0, 1.0

    def objective(aa: float, bb: float) -> float:
        q = np.clip(_sigmoid(aa + bb * z), _EPS, 1 - _EPS)
        loss = -np.mean(yy * np.log(q) + (1 - yy) * np.log(1 - q))
        return float(loss + 0.5 * lam * (aa * aa + ((bb - 1) ** 2 if fit_slope else 0)))

    current = objective(a, b)
    for _ in range(80):
        q = _sigmoid(a + b * z)
        residual, weight = q - yy, np.maximum(q * (1 - q), 1e-9)
        ga, haa = float(residual.mean() + lam * a), float(weight.mean() + lam)
        if fit_slope:
            gb = float((residual * z).mean() + lam * (b - 1))
            hab, hbb = float((weight * z).mean()), float((weight * z * z).mean() + lam)
            det = haa * hbb - hab * hab
            if det <= 1e-12:
                break
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        else:
            da, db = ga / haa, 0.0
        step, accepted = 1.0, False
        for _ in range(25):
            na = float(np.clip(a - step * da, -rules.shift_bound, rules.shift_bound))
            nb = float(np.clip(b - step * db, lower, upper)) if fit_slope else 1.0
            score = objective(na, nb)
            if score <= current + 1e-12:
                moved = max(abs(na - a), abs(nb - b))
                a, b, current, accepted = na, nb, score, True
                break
            step *= 0.5
        if not accepted or moved < 1e-9:
            break
    return float(a), float(b)


def fit_calibrator(kind: str, p: Any, y: Any, rules: Rules) -> dict[str, Any]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    if kind == "identity":
        return {"type": "identity"}
    if kind == "shift":
        a, _ = _fit_logit_affine(p, y, fit_slope=False, rules=rules)
        return {"type": "shift", "a": a, "b": 1.0}
    if kind == "platt":
        a, b = _fit_logit_affine(p, y, fit_slope=True, rules=rules)
        return {"type": "platt", "a": a, "b": b}
    if kind == "base_rate":
        return {"type": "base_rate", "rate": float(np.clip(y.mean() if len(y) else .5, rules.p_floor, rules.p_ceil))}
    if kind == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        model = IsotonicRegression(y_min=rules.p_floor, y_max=rules.p_ceil, out_of_bounds="clip")
        model.fit(p, y)
        xs, ys = np.asarray(model.X_thresholds_, float), np.asarray(model.y_thresholds_, float)
        if len(xs) > 100:
            keep = np.unique(np.linspace(0, len(xs) - 1, 100).astype(int))
            xs, ys = xs[keep], ys[keep]
        return {"type": "isotonic", "x": xs.tolist(), "y": ys.tolist()}
    raise ValueError(f"unknown calibrator kind: {kind}")


def apply_calibrator(cal: dict[str, Any] | None, p: Any, rules: Rules = Rules()) -> np.ndarray:
    arr = np.asarray(p, float)
    out = np.full(arr.shape, np.nan, float)
    ok = np.isfinite(arr) & (arr >= 0) & (arr <= 1)
    if not ok.any():
        return out
    kind = (cal or {}).get("type", "identity")
    if kind in ("shift", "platt"):
        out[ok] = _sigmoid(float(cal["a"]) + float(cal["b"]) * _logit(arr[ok]))
    elif kind == "isotonic":
        out[ok] = np.interp(arr[ok], np.asarray(cal["x"], float), np.asarray(cal["y"], float))
    elif kind == "base_rate":
        out[ok] = float(cal["rate"])
    else:
        out[ok] = arr[ok]
    finite = np.isfinite(out)
    out[finite] = np.clip(out[finite], rules.p_floor, rules.p_ceil)
    return out


def expected_calibration_error(p: Any, y: Any, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if not len(p):
        return float("nan")
    bins = max(2, min(bins, len(p) // 5)) if len(p) < bins * 5 else bins
    total = 0.0
    for ids in np.array_split(np.argsort(p), bins):
        if len(ids):
            total += len(ids) / len(p) * abs(float(p[ids].mean() - y[ids].mean()))
    return float(total)


def psi(expected: Any, actual: Any, bins: int = 10) -> float:
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < bins * 5 or len(a) < bins * 5:
        return float("nan")
    cuts = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return float("nan")
    cuts[0], cuts[-1] = -np.inf, np.inf
    pe, pa = np.histogram(e, cuts)[0] / len(e), np.histogram(a, cuts)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def page_hinkley(series: Any, delta: float, threshold: float) -> dict[str, Any]:
    vals = np.asarray(series, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 6:
        return {"status": "INSUFFICIENT_DAYS", "alarm": False, "alarm_index": None}
    mean, cumulative, minimum, alarm_at = 0.0, 0.0, 0.0, None
    for i, value in enumerate(vals):
        mean += (value - mean) / (i + 1)
        cumulative += value - mean - delta
        minimum = min(minimum, cumulative)
        if alarm_at is None and cumulative - minimum > threshold:
            alarm_at = i
    return {"status": "OK", "alarm": alarm_at is not None, "alarm_index": alarm_at, "observations": int(len(vals))}


def session_date_ist(timestamps: Any) -> pd.Series:
    parsed = pd.to_datetime(pd.Series(timestamps).reset_index(drop=True), errors="coerce", utc=True)
    return parsed.dt.tz_convert(IST).dt.strftime("%Y-%m-%d")


def prepare_fit_frame(group: pd.DataFrame, rules: Rules) -> tuple[pd.DataFrame, dict[str, Any]]:
    g = group.reset_index(drop=True)
    x = pd.DataFrame({
        "session_date": session_date_ist(g["target_timestamp_utc"]),
        "p": pd.to_numeric(g["probability_up"], errors="coerce"),
        "ret": pd.to_numeric(g["actual_return"], errors="coerce"),
        "er": pd.to_numeric(g["expected_return"], errors="coerce") if "expected_return" in g else np.nan,
        "age": pd.to_numeric(g["input_age_seconds"], errors="coerce") if "input_age_seconds" in g else np.nan,
    })
    rows_in = len(x)
    x = x.loc[x["session_date"].notna() & x["p"].between(0, 1) & x["ret"].notna()].copy()
    stale = x["age"] > rules.max_input_age_seconds
    stale_n = int(stale.sum())
    x = x.loc[~stale].copy()
    x["y"] = (x["ret"] > 0).astype(float)
    return x.reset_index(drop=True), {"rows_in": int(rows_in), "rows_after_validity": int(rows_in - int((~(x.index.isin([]))).sum()) if False else len(x) + stale_n), "stale_input_rows_excluded": stale_n, "rows_used": int(len(x)), "sessions_used": int(x["session_date"].nunique())}


def rolling_origin_folds(dates: Any, rules: Rules) -> list[tuple[list[str], list[str]]]:
    d = sorted({str(v) for v in dates})
    n = len(d)
    avail = n - rules.min_train_dates - rules.embargo_dates
    if avail < 2:
        return []
    k = min(rules.n_folds, avail)
    block = max(1, avail // k)
    first_test = n - k * block
    folds = []
    for i in range(k):
        t0 = first_test + i * block
        train_end = t0 - rules.embargo_dates
        if train_end >= rules.min_train_dates:
            folds.append((d[:train_end], d[t0:t0 + block]))
    return folds


def _oof(df: pd.DataFrame, kinds: list[str], rules: Rules) -> pd.DataFrame | None:
    parts = []
    for fold_no, (train_dates, test_dates) in enumerate(rolling_origin_folds(df["session_date"], rules)):
        tr, te = df[df["session_date"].isin(train_dates)], df[df["session_date"].isin(test_dates)]
        if len(tr) < rules.min_fold_train_rows or te.empty or tr["y"].nunique() < 2:
            continue
        out = te.copy()
        out["fold"] = fold_no
        for kind in kinds:
            out[f"p_{kind}"] = apply_calibrator(fit_calibrator(kind, tr["p"], tr["y"], rules), te["p"].to_numpy(), rules)
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else None


def _auc_with_ci(oof: pd.DataFrame, key: str, rules: Rules, n_boot: int = 150) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score
    sample = oof.iloc[::max(1, int(math.ceil(len(oof) / 30000)))] if len(oof) > 30000 else oof
    if sample["y"].nunique() < 2:
        return {"auc": None, "lo": None, "hi": None}
    point = float(roc_auc_score(sample["y"], sample["p"]))
    groups = [g for _, g in sample.groupby("session_date")]
    if len(groups) < 3:
        return {"auc": point, "lo": None, "hi": None}
    rng, vals = np.random.default_rng(_seed(key, "auc")), []
    for _ in range(n_boot):
        pick = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if pick["y"].nunique() > 1:
            vals.append(roc_auc_score(pick["y"], pick["p"]))
    if len(vals) < 20:
        return {"auc": point, "lo": None, "hi": None}
    return {"auc": point, "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5))}


def evaluate_probability_candidates(df: pd.DataFrame, key: str, rules: Rules) -> tuple[dict[str, Any], dict[str, Any] | None]:
    n_rows, n_dates = int(len(df)), int(df["session_date"].nunique()) if len(df) else 0
    info = {"n_rows": n_rows, "n_dates": n_dates, "minimum_rows": rules.min_rows, "minimum_dates": rules.min_dates}
    if n_rows < rules.min_rows or n_dates < rules.min_dates:
        return {**info, "status": "INSUFFICIENT_HISTORY"}, None
    if df["y"].nunique() < 2:
        return {**info, "status": "INSUFFICIENT_CLASS_VARIATION"}, None
    kinds = ["shift", "platt"]
    if n_rows >= rules.isotonic_min_rows and n_dates >= rules.isotonic_min_dates:
        kinds.append("isotonic")
    oof = _oof(df, kinds + ["base_rate"], rules)
    if oof is None or oof["session_date"].nunique() < 3 or oof["fold"].nunique() < 2:
        return {**info, "status": "INSUFFICIENT_HISTORY", "reason": "walk-forward folds could not be formed"}, None
    y, p, dates = oof["y"].to_numpy(), oof["p"].to_numpy(), oof["session_date"].to_numpy()
    ll_raw = _row_logloss(p, y)
    brier_raw, ece_raw = float(np.mean((p - y) ** 2)), expected_calibration_error(p, y)
    alpha = rules.alpha / len(kinds)
    cands = {}
    for kind in kinds:
        pk = oof[f"p_{kind}"].to_numpy()
        gain = date_block_bootstrap(ll_raw - _row_logloss(pk, y), dates, n_boot=rules.n_boot, alpha=alpha, seed=_seed(key, f"prob_{kind}"))
        brier_k, ece_k = float(np.mean((pk-y)**2)), expected_calibration_error(pk, y)
        change = float(np.mean(np.abs(pk-p)))
        wins = [float(np.mean((g[f"p_{kind}"]-g["y"])**2)) < float(np.mean((g["p"]-g["y"])**2)) for _, g in oof.groupby("fold")]
        win_rate = float(np.mean(wins)) if wins else 0.0
        passes = bool(_lo(gain)>0 and brier_k<=brier_raw and ece_k<=ece_raw+rules.max_ece_worsen and change<=rules.max_mean_abs_change and win_rate>=rules.min_fold_win_rate)
        cands[kind] = {"logloss_gain":gain,"brier":brier_k,"ece":ece_k,"mean_abs_probability_change":change,"fold_win_rate_vs_raw":win_rate,"folds":len(wins),"passes_gate":passes,"adjusted_alpha":alpha}
    passing = [k for k in kinds if cands[k]["passes_gate"]]
    chosen = None
    if passing:
        best = max(cands[k]["logloss_gain"]["mean"] for k in passing)
        for kind in ("shift", "platt", "isotonic"):
            if kind in passing and cands[kind]["logloss_gain"]["mean"] >= .9 * best:
                chosen = kind
                break
    auc = _auc_with_ci(oof, key, rules)
    no_skill = auc.get("lo") is not None and auc["lo"] <= .5
    base_gain = date_block_bootstrap(ll_raw - _row_logloss(oof["p_base_rate"].to_numpy(), y), dates, n_boot=rules.n_boot, alpha=rules.alpha, seed=_seed(key, "prob_base"))
    worse_than_base = bool(_lo(base_gain) > 0)
    result = {**info,"status":"REJECTED_BY_GATE","kinds_tested":kinds,"oof_sessions":int(oof["session_date"].nunique()),"oof_rows":int(len(oof)),"raw_oof":{"logloss":float(ll_raw.mean()),"brier":brier_raw,"ece":ece_raw,"auc":auc},"candidates":cands,"no_discrimination_detected":bool(no_skill),"base_rate_logloss_gain_vs_raw":base_gain,"raw_worse_than_base_rate":worse_than_base,"retrain_recommended":bool(no_skill or worse_than_base),"chosen":None}
    if not chosen:
        return result, None
    if no_skill:
        result.update(status="REJECTED_NO_DISCRIMINATION", rejected_candidate=chosen)
        return result, None
    final = fit_calibrator(chosen, df["p"], df["y"], rules)
    if chosen == "platt" and final["b"] <= rules.slope_bounds[0] + 1e-6:
        result.update(status="REJECTED_UNINFORMATIVE_MODEL", retrain_recommended=True)
        return result, None
    full_change=float(np.mean(np.abs(apply_calibrator(final,df["p"].to_numpy(),rules)-df["p"].to_numpy())))
    if full_change > rules.max_mean_abs_change:
        result["status"]="REJECTED_EXCESSIVE_CHANGE"
        return result, None
    result.update(status="ELIGIBLE",chosen=chosen,large_correction_flag=full_change>rules.large_change_flag,full_sample_mean_abs_change=full_change)
    return result, {"calibrator":final,"oof":oof}


def _huber_slope(x: np.ndarray, target: np.ndarray, rules: Rules, iters: int = 40) -> float:
    w=np.ones_like(x); beta=0.0
    for _ in range(iters):
        denom=float(np.sum(w*x*x))
        if denom<=1e-18: return 0.0
        beta=float(np.sum(w*x*target)/denom)
        r=target-beta*x
        scale=max(1.4826*float(np.median(np.abs(r-np.median(r)))),1e-12)
        delta=1.345*scale
        w=np.where(np.abs(r)<=delta,1.0,delta/np.maximum(np.abs(r),1e-18))
    return float(np.clip(beta,*rules.return_slope_bounds))


def fit_return_correction(kind: str, er: Any, ret: Any, rules: Rules) -> dict[str, Any]:
    x,t=np.asarray(er,float),np.asarray(ret,float)
    if kind=="zero": return {"type":"zero","a":0.0,"b":0.0}
    b=_huber_slope(x,t,rules)
    if kind=="scale": return {"type":"scale","a":0.0,"b":b}
    if kind=="affine": return {"type":"affine","a":float(np.median(t-b*x)),"b":b}
    raise ValueError(f"unknown return correction kind: {kind}")


def apply_return_correction(correction: dict[str, Any] | None, er: Any) -> np.ndarray:
    x=np.asarray(er,float)
    if not correction: return x
    return float(correction.get("a",0.0))+float(correction.get("b",1.0))*x


def evaluate_return_candidates(df: pd.DataFrame, key: str, rules: Rules) -> dict[str, Any]:
    r=df.loc[df["er"].notna()].copy()
    info={"n_rows":int(len(r)),"n_dates":int(r["session_date"].nunique()) if len(r) else 0}
    if info["n_rows"]<rules.min_return_rows or info["n_dates"]<rules.min_dates:
        return {**info,"status":"INSUFFICIENT_HISTORY","correction":None,"price_forecast_reliable":None}
    kinds=["zero","scale","affine"]; parts=[]
    for fold_no,(train_dates,test_dates) in enumerate(rolling_origin_folds(r["session_date"],rules)):
        tr,te=r[r["session_date"].isin(train_dates)],r[r["session_date"].isin(test_dates)]
        if len(tr)<rules.min_fold_train_rows or te.empty: continue
        out=te.copy()
        for kind in kinds:
            cor=fit_return_correction(kind,tr["er"],tr["ret"],rules)
            out[f"er_{kind}"]=apply_return_correction(cor,te["er"].to_numpy())
        parts.append(out)
    if not parts: return {**info,"status":"INSUFFICIENT_HISTORY","correction":None,"price_forecast_reliable":None}
    oof=pd.concat(parts,ignore_index=True); dates=oof["session_date"].to_numpy(); ret=oof["ret"].to_numpy()
    raw_err=np.abs(ret-oof["er"].to_numpy()); zero_err=np.abs(ret)
    raw_vs_zero=date_block_bootstrap(zero_err-raw_err,dates,n_boot=rules.n_boot,alpha=rules.alpha,seed=_seed(key,"ret_raw_zero"))
    cands={}
    for kind in kinds:
        err=np.abs(ret-oof[f"er_{kind}"].to_numpy())
        vs_raw=date_block_bootstrap(raw_err-err,dates,n_boot=rules.n_boot,alpha=rules.alpha/len(kinds),seed=_seed(key,f"ret_{kind}"))
        vs_zero=date_block_bootstrap(zero_err-err,dates,n_boot=rules.n_boot,alpha=rules.alpha,seed=_seed(key,f"ret0_{kind}"))
        cands[kind]={"mae":float(err.mean()),"gain_vs_raw":vs_raw,"gain_vs_zero":vs_zero,"passes_gate":bool(_lo(vs_raw)>0)}
    passing=[k for k in kinds if cands[k]["passes_gate"]]
    informative=[k for k in ("scale","affine") if k in passing and _lo(cands[k]["gain_vs_zero"])>0]
    chosen=None
    if informative:
        best=max(cands[k]["gain_vs_raw"]["mean"] for k in informative)
        for kind in ("scale","affine"):
            if kind in informative and cands[kind]["gain_vs_raw"]["mean"] >= .9*best: chosen=kind; break
    elif passing:
        best=max(cands[k]["gain_vs_raw"]["mean"] for k in passing)
        for kind in kinds:
            if kind in passing and cands[kind]["gain_vs_raw"]["mean"] >= .9*best: chosen=kind; break
    correction=fit_return_correction(chosen,r["er"],r["ret"],rules) if chosen else None
    reliable=bool(_lo(cands[chosen]["gain_vs_zero"])>0) if chosen else bool(_lo(raw_vs_zero)>0)
    return {**info,"status":"ELIGIBLE" if chosen else "REJECTED_BY_GATE","raw_mae":float(raw_err.mean()),"no_change_mae":float(zero_err.mean()),"raw_vs_no_change":raw_vs_zero,"candidates":cands,"chosen":chosen,"correction":correction,"price_forecast_reliable":reliable}


def fit_action_band(oof: pd.DataFrame, prob_col: str, key: str, rules: Rules) -> dict[str, Any]:
    p,ret,dates=oof[prob_col].to_numpy(float),oof["ret"].to_numpy(float),oof["session_date"].to_numpy()
    gross=np.where(p>=.5,1.0,-1.0)*ret; cost=rules.cost_bps_roundtrip/1e4
    trace=[]
    for band in rules.band_grid:
        mask=np.abs(p-.5)>=band; n=int(mask.sum()); nd=int(pd.Series(dates[mask]).nunique())
        if n<rules.min_action_rows or nd<rules.min_action_dates: break
        frame=pd.DataFrame({"d":dates[mask],"hit":(gross[mask]>0).astype(float),"net":gross[mask]-cost})
        per=frame.groupby("d").agg(hit=("hit","sum"),n=("hit","size"))
        precision=date_block_ratio_bootstrap(per["hit"],per["n"],n_boot=rules.n_boot,alpha=rules.alpha/rules.band_looks,seed=_seed(key,f"band_p_{band}"))
        net=date_block_bootstrap(frame["net"],frame["d"],n_boot=rules.n_boot,alpha=rules.alpha/rules.band_looks,seed=_seed(key,f"band_n_{band}"))
        trace.append({"min_conf":band,"rows":n,"dates":nd,"precision":precision["ratio"],"precision_lb":precision["lo"],"net_bps_mean":None if net["mean"] is None else net["mean"]*1e4,"net_bps_lb":None if net["lo"] is None else net["lo"]*1e4})
        if _lo(precision)>=rules.min_precision_lb and _lo(net)>0:
            return {"abstain_all":False,"min_conf":float(band),"buy_p":round(.5+band,4),"sell_p":round(.5-band,4),"coverage":n/len(p),"precision":precision["ratio"],"precision_lb":precision["lo"],"net_bps_mean":net["mean"]*1e4,"net_bps_lb":net["lo"]*1e4,"cost_bps_roundtrip_assumed":rules.cost_bps_roundtrip,"rows":n,"dates":nd}
    return {"abstain_all":True,"reason":"no band met precision and net-of-cost gates","trace_tail":trace[-3:],"cost_bps_roundtrip_assumed":rules.cost_bps_roundtrip}


def signal_from_thresholds(p: float, thresholds: dict[str, Any] | None) -> str:
    if not thresholds or thresholds.get("abstain_all",True): return "ABSTAIN"
    if p>=thresholds["buy_p"]: return "BUY"
    if p<=thresholds["sell_p"]: return "SELL"
    return "HOLD"


def live_monitor(group: pd.DataFrame, profile_id: str | list[str] | set[str] | tuple[str, ...] | None, rules: Rules, key: str) -> dict[str, Any]:
    empty={"dates":0,"rows":0,"logloss_gain":{"mean":None,"lo":None,"hi":None,"dates":0},"brier_ok":True}
    if not profile_id or "calibration_profile_id" not in group.columns or "probability_up_calibrated" not in group.columns: return empty
    if isinstance(profile_id,(list,tuple,set)):
        ids={str(x) for x in profile_id if x}; tag=sorted(ids)[-1] if ids else ""
    else: ids={str(profile_id)}; tag=str(profile_id)
    if not ids: return empty
    g=group.loc[group["calibration_profile_id"].astype(str).isin(ids)].reset_index(drop=True)
    if g.empty: return empty
    p=pd.to_numeric(g["probability_up"],errors="coerce").to_numpy()
    pc=pd.to_numeric(g["probability_up_calibrated"],errors="coerce").to_numpy()
    ret=pd.to_numeric(g["actual_return"],errors="coerce").to_numpy()
    age=pd.to_numeric(g["input_age_seconds"],errors="coerce").to_numpy() if "input_age_seconds" in g else np.full(len(g),np.nan)
    ok=np.isfinite(p)&np.isfinite(pc)&np.isfinite(ret)&~(age>rules.max_input_age_seconds)
    if not ok.any(): return empty
    g,p,pc,ret=g.loc[ok].reset_index(drop=True),p[ok],pc[ok],ret[ok]
    y=(ret>0).astype(float); dates=session_date_ist(g["target_timestamp_utc"]).to_numpy()
    gain=date_block_bootstrap(_row_logloss(p,y)-_row_logloss(pc,y),dates,n_boot=rules.n_boot,alpha=rules.shadow_alpha,seed=_seed(key,f"live_{tag}"))
    brier_raw,brier_cal=float(np.mean((p-y)**2)),float(np.mean((pc-y)**2))
    out={"dates":gain["dates"],"rows":int(len(y)),"logloss_gain":gain,"brier_raw":brier_raw,"brier_calibrated":brier_cal,"brier_ok":bool(brier_cal<=brier_raw)}
    if "expected_return_calibrated" in g and "expected_return" in g:
        er=pd.to_numeric(g["expected_return"],errors="coerce").to_numpy(); erc=pd.to_numeric(g["expected_return_calibrated"],errors="coerce").to_numpy()
        m=np.isfinite(er)&np.isfinite(erc)
        if m.sum()>=50:
            out["return_gain"]=date_block_bootstrap(np.abs(ret[m]-er[m])-np.abs(ret[m]-erc[m]),dates[m],n_boot=rules.n_boot,alpha=rules.shadow_alpha,seed=_seed(key,f"liveret_{tag}"))
    return out


def new_state() -> dict[str, Any]:
    return {"schema_version":SCHEMA_VERSION,"updated_utc":None,"entries":{}}


def load_state(payload: bytes | None) -> dict[str, Any]:
    if payload is None: return new_state()
    try: obj=json.loads(payload.decode("utf-8"))
    except Exception as exc: raise AutopilotStateError(f"registry is not valid JSON: {type(exc).__name__}: {exc}") from exc
    if not isinstance(obj,dict) or not isinstance(obj.get("entries"),dict): raise AutopilotStateError("registry has no 'entries' mapping")
    if obj.get("schema_version")!=SCHEMA_VERSION: raise AutopilotStateError(f"unsupported registry schema_version: {obj.get('schema_version')!r}")
    return obj


def _expired(profile: dict[str, Any], now: datetime) -> bool:
    until=_parse_iso(profile.get("valid_until_utc"))
    return until is None or now>=until


def _param_distance(a: dict[str, Any], b: dict[str, Any], rules: Rules) -> float:
    grid=np.linspace(.05,.95,19)
    return float(np.max(np.abs(apply_calibrator(a.get("calibrator"),grid,rules)-apply_calibrator(b.get("calibrator"),grid,rules))))


def _event(kind: str, profile: dict[str, Any] | None, now: datetime, **detail: Any) -> dict[str, Any]:
    return {"event":kind,"at_utc":_iso(now),"profile_id":(profile or {}).get("profile_id"),**detail}


def _retire(entry: dict[str, Any], profile: dict[str, Any], status: str, now: datetime, cooldown_days: int | None = None) -> None:
    entry.setdefault("history",[]).append({**profile,"stage":status,"retired_utc":_iso(now)})
    entry["history"]=entry["history"][-8:]; entry["champion"]=None
    if cooldown_days: entry["cooldown_until_utc"]=_iso(now+timedelta(days=cooldown_days))


def advance_entry(entry: dict[str, Any], *, candidate: dict[str, Any] | None, live: dict[str, Any], now: datetime, rules: Rules) -> list[dict[str, Any]]:
    events=[]; champ=entry.get("champion")
    if champ and _expired(champ,now):
        events.append(_event("EXPIRED",champ,now)); _retire(entry,champ,"EXPIRED",now); champ=None
    if champ:
        n=int(live.get("dates",0) or 0); gain=live.get("logloss_gain",{}); brier_ok=bool(live.get("brier_ok",True))
        if champ["stage"]==STAGE_SHADOW:
            if n>=rules.shadow_min_dates and _lo(gain)>0 and brier_ok:
                champ["stage"]=STAGE_ACTIVE; champ["promoted_utc"]=_iso(now); champ["live_evidence_at_promotion"]=_clean(live)
                events.append(_event("PROMOTED_TO_ACTIVE",champ,now,live_dates=n,gain_lo=gain.get("lo")))
            elif n>=rules.shadow_min_dates and _hi(gain)<0:
                events.append(_event("REJECTED_IN_SHADOW_WORSE",champ,now,live_dates=n,gain_hi=gain.get("hi"))); _retire(entry,champ,"REJECTED",now,rules.cooldown_days); champ=None
            elif n>=rules.shadow_max_dates:
                events.append(_event("REJECTED_SHADOW_TIMEOUT",champ,now,live_dates=n)); _retire(entry,champ,"REJECTED",now,rules.cooldown_days); champ=None
        elif champ["stage"]==STAGE_ACTIVE:
            worse=_hi(gain)<0 or (gain.get("mean") is not None and gain["mean"]<0 and not brier_ok)
            if n>=rules.demote_min_dates and worse:
                events.append(_event("DEMOTED_LIVE_WORSE_THAN_RAW",champ,now,live_dates=n,gain_mean=gain.get("mean"),gain_hi=gain.get("hi"))); _retire(entry,champ,"DEMOTED",now,rules.cooldown_days); champ=None
        if champ and champ.get("return_correction"):
            rg=live.get("return_gain") or {}
            if int(rg.get("dates",0) or 0)>=rules.demote_min_dates and _hi(rg)<0:
                champ["return_correction"]=None; champ["price_forecast_reliable"]=False; events.append(_event("RETURN_CORRECTION_DROPPED_LIVE_WORSE",champ,now))
    if candidate and candidate.get("profile"):
        new=copy.deepcopy(candidate["profile"]); cooling=_parse_iso(entry.get("cooldown_until_utc"))
        if champ is None:
            if cooling and now<cooling: events.append(_event("CANDIDATE_HELD_BY_COOLDOWN",new,now,until=entry["cooldown_until_utc"]))
            else:
                new["stage"]=STAGE_SHADOW; entry["champion"]=new; events.append(_event("SHADOW_STARTED",new,now))
        elif champ["stage"]==STAGE_ACTIVE:
            created=_parse_iso(champ.get("created_utc")) or now
            if now-created>=timedelta(days=rules.refit_after_days):
                dist=_param_distance(champ,new,rules)
                if dist<=rules.refresh_max_param_distance:
                    new["stage"]=champ["stage"]
                    for carried in ("promoted_utc","live_evidence_at_promotion"):
                        if carried in champ: new[carried]=champ[carried]
                    new["lineage"]=(champ.get("lineage") or [])[-4:]+[champ["profile_id"]]
                    entry["champion"]=new; events.append(_event("REFRESHED_IN_PLACE",new,now,parameter_distance=dist))
                else:
                    entry["challenger_note"]={"profile_id":new["profile_id"],"parameter_distance":dist,"seen_utc":_iso(now),"note":"regime-shift candidate; not installed while current profile is healthy"}
                    events.append(_event("CHALLENGER_NOTED_NOT_INSTALLED",new,now,parameter_distance=dist))
    entry["events"]=(entry.get("events") or [])+events; entry["events"]=entry["events"][-40:]
    return events


def build_profile(key: str, horizon: Any, version: Any, prob: dict[str, Any], cal: dict[str, Any], ret: dict[str, Any] | None, band: dict[str, Any], fit_window: dict[str, Any], now: datetime, rules: Rules) -> dict[str, Any]:
    body={"key":key,"calibrator":cal,"return_correction":(ret or {}).get("correction"),"fit_window":fit_window}
    digest=hashlib.sha1(json.dumps(_clean(body),sort_keys=True).encode()).hexdigest()[:10]
    chosen=prob["chosen"]
    return _clean({"profile_id":f"{key}|{digest}","key":key,"horizon_min":int(horizon) if pd.notna(horizon) else None,"model_version":str(version),"stage":STAGE_SHADOW,"created_utc":_iso(now),"valid_until_utc":_iso(now+timedelta(days=rules.valid_for_days)),"fit_window":fit_window,"calibrator":cal,"return_correction":(ret or {}).get("correction"),"price_forecast_reliable":(ret or {}).get("price_forecast_reliable"),"thresholds":band,"walk_forward_evidence":{"chosen":chosen,"gain":prob["candidates"][chosen]["logloss_gain"],"raw_oof":prob["raw_oof"],"candidate_oof":{k:prob["candidates"][chosen][k] for k in ("brier","ece")}}})


def build_active_profile_doc(state: dict[str, Any], now: datetime) -> dict[str, Any]:
    profiles=[]
    for entry in state.get("entries",{}).values():
        champ=entry.get("champion")
        if champ and champ.get("stage") in LIVE_STAGES and not _expired(champ,now):
            profiles.append({k:champ.get(k) for k in ("profile_id","key","horizon_min","model_version","stage","valid_until_utc","calibrator","return_correction","price_forecast_reliable","thresholds")})
    return _clean({"schema_version":SCHEMA_VERSION,"generated_utc":_iso(now),"profiles":profiles})


def run_autopilot(frame: pd.DataFrame | None, state: dict[str, Any], *, now: datetime | None = None, rules: Rules = Rules()) -> tuple[dict[str, Any],dict[str, Any],dict[str, Any]]:
    now=_utc(now); state=copy.deepcopy(state); entries=state.setdefault("entries",{})
    report={"status":"COMPLETED","rules":asdict(rules),"keys":{},"events":[],"assumptions":[f"round-trip action-band cost assumed at {rules.cost_bps_roundtrip} bps","label is actual_return > 0; flat counts as not-up","confidence intervals resample whole IST session dates"]}
    validated=pd.DataFrame()
    if frame is not None and not frame.empty and "validation_status" in frame:
        validated=frame.loc[frame["validation_status"].eq("VALIDATED")].copy()
    if validated.empty: report["status"]="NO_MATURED_OUTCOMES"
    seen=set()
    if not validated.empty:
        for (horizon,version),group in validated.groupby(["horizon_min","model_version"],dropna=False):
            key=profile_key(horizon,version); seen.add(key)
            fit_df,prep=prepare_fit_frame(group,rules)
            prob,art=evaluate_probability_candidates(fit_df,key,rules)
            ret_res,band,candidate=None,None,None
            if art is not None:
                ret_res=evaluate_return_candidates(fit_df,key,rules); band=fit_action_band(art["oof"],f"p_{prob['chosen']}",key,rules)
                window={"first_session":fit_df["session_date"].min(),"last_session":fit_df["session_date"].max(),"rows":int(len(fit_df)),"sessions":int(fit_df["session_date"].nunique())}
                candidate={"profile":build_profile(key,horizon,version,prob,art["calibrator"],ret_res,band,window,now,rules)}
            champ_obj=entries.setdefault(key,{"champion":None,"history":[],"events":[]}).get("champion") or {}
            ids=[*(champ_obj.get("lineage") or []),champ_obj.get("profile_id")] if champ_obj.get("profile_id") else None
            live=live_monitor(group,ids,rules,key)
            events=advance_entry(entries[key],candidate=candidate,live=live,now=now,rules=rules)
            dates=sorted(fit_df["session_date"].unique()); drift=None
            if len(dates)>=12:
                recent=fit_df.loc[fit_df["session_date"].isin(dates[-5:]),"p"]; prior=fit_df.loc[~fit_df["session_date"].isin(dates[-5:]),"p"]
                pv=psi(prior,recent); daily=fit_df.assign(b=(fit_df["p"]-fit_df["y"])**2).groupby("session_date")["b"].mean().sort_index()
                ph=page_hinkley(daily.to_numpy(),rules.ph_delta,rules.ph_threshold)
                drift={"psi_raw_probability":None if math.isnan(pv) else pv,"alarm":bool(not math.isnan(pv) and pv>=rules.psi_alarm),"brier_page_hinkley":ph}
                if drift["alarm"]: events.append(_event("DRIFT_ALARM_RAW_PROBABILITY",None,now,psi=pv))
                if ph.get("alarm"): events.append(_event("DRIFT_ALARM_RAW_BRIER_DETERIORATION",None,now))
                entries[key]["events"]=(entries[key].get("events") or [])+events[-2:]
            report["keys"][key]=_clean({"horizon_min":horizon,"model_version":str(version),"preparation":prep,"probability":prob,"return_correction":ret_res,"action_band":band,"live_monitor":live,"drift":drift,"champion_stage":(entries[key].get("champion") or {}).get("stage"),"events":events})
    for key,entry in entries.items():
        if key not in seen:
            events=advance_entry(entry,candidate=None,live={"dates":0},now=now,rules=rules)
            if events: report["keys"][key]=_clean({"note":"no matured rows in this window","events":events,"champion_stage":(entry.get("champion") or {}).get("stage")})
    for key,val in report["keys"].items():
        for ev in val.get("events",[]) or []: report["events"].append({"key":key,**ev})
    state["updated_utc"]=_iso(now); active_doc=build_active_profile_doc(state,now)
    report["summary"]={"keys_evaluated":len(report["keys"]),"profiles_live":len(active_doc["profiles"]),"stages":{k:(v.get("champion") or {}).get("stage") for k,v in entries.items()}}
    return _clean(state),active_doc,_clean(report)
