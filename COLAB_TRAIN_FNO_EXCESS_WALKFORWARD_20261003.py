#!/usr/bin/env python3
"""F&O specialist walk-forward candidate trainer.

Design:
- Separate candidate directory; never overwrites/promotes the existing model.
- Uses only rows with an explicit PIT-matched F&O observation.
- Builds specialist features from the persisted daily NSE F&O layer.
- Targets future stock return relative to the cross-sectional median future return
  of the observed F&O universe on the same decision date.
- Uses date-based purged/embargoed walk-forward folds so overlapping horizons do
  not leak future-label information across train/validation boundaries.
- Trains two classifiers (logistic + histogram gradient boosting), one regressor,
  and a probability-average ensemble.
- No automatic orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
BASELINE_DIR = BASE / "models" / "all_nse_delivery_multifactor_replacement_2026"
OUT_DIR = BASE / "models" / "all_nse_delivery_fno_walkforward_2026"
FNO_PATH = BASE / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
PANEL_PATH = BASELINE_DIR / "ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"

try:
    from multifactor_data_layer_v2 import FEATURES_59
except Exception as exc:
    raise SystemExit(f"Cannot import FEATURES_59 from repository: {exc}")


FNO_FEATURES_8 = [
    "fno_atm_iv",
    "fno_atm_iv_z20",
    "fno_pcr_z20",
    "fno_basis_z20",
    "fno_oi_chg_z20",
    "fno_iv_chg_z20",
    "fno_buildup",
    "fno_buildup_persist3",
]
FEATURES_67 = list(FEATURES_59) + FNO_FEATURES_8
HORIZONS = [("next_session", 1), ("3d", 3), ("5d", 5), ("7d", 7)]
N_FOLDS = 3
VALIDATION_DAYS = 20
EMBARGO_SESSIONS = 1
MIN_TRAIN_DATES = 40


def norm_symbol(value: object) -> str:
    x = str(value or "").strip().upper()
    for suffix in (".NS", ".BO", "-EQ", ".EQ"):
        if x.endswith(suffix):
            x = x[: -len(suffix)]
    return x


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_fno_specialist_features(fno: pd.DataFrame) -> pd.DataFrame:
    x = fno.copy()
    required = {"symbol", "date"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"F&O layer missing required columns: {sorted(missing)}")

    x["symbol"] = x["symbol"].map(norm_symbol)
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    numeric = [
        "fno_atm_iv", "fno_pcr", "fno_fut_basis", "fno_fut_oi",
        "fno_fut_close", "fno_oi_change", "fno_iv_change"
    ]
    for c in numeric:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = (
        x.dropna(subset=["symbol", "date"])
        .sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .copy()
    )
    g = x.groupby("symbol", sort=False)

    def z20(s: pd.Series) -> pd.Series:
        mu = s.rolling(20, min_periods=10).mean()
        sd = s.rolling(20, min_periods=10).std()
        return (s - mu) / sd.replace(0, np.nan)

    x["fno_atm_iv_z20"] = g["fno_atm_iv"].transform(z20)
    x["fno_pcr_z20"] = g["fno_pcr"].transform(z20)
    x["fno_basis_z20"] = g["fno_fut_basis"].transform(z20)

    oi_pct = g["fno_fut_oi"].pct_change()
    x["fno_oi_chg_z20"] = oi_pct.groupby(x["symbol"]).transform(z20)
    x["fno_iv_chg_z20"] = g["fno_iv_change"].transform(z20)

    fut_ret = g["fno_fut_close"].pct_change()
    oi_ret = g["fno_fut_oi"].pct_change()
    r = np.sign(fut_ret)
    o = np.sign(oi_ret)

    x["fno_buildup"] = np.nan
    x.loc[(r > 0) & (o > 0), "fno_buildup"] = 2.0
    x.loc[(r > 0) & (o < 0), "fno_buildup"] = 1.0
    x.loc[(r < 0) & (o < 0), "fno_buildup"] = -1.0
    x.loc[(r < 0) & (o > 0), "fno_buildup"] = -2.0
    x["fno_buildup_persist3"] = (
        x.groupby("symbol", sort=False)["fno_buildup"]
        .transform(lambda s: s.rolling(3, min_periods=2).sum())
    )
    return x


def attach_fno_and_subset(panel: pd.DataFrame, fno: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    p = panel.copy()
    p["symbol"] = p["symbol"].map(norm_symbol)
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p = p.dropna(subset=["symbol", "date"]).copy()

    f = add_fno_specialist_features(fno)
    sessions = pd.Series(sorted(p["date"].dropna().unique()))
    if sessions.empty:
        raise RuntimeError("Panel has no trading sessions")

    next_map = pd.Series(sessions.to_numpy(), index=sessions.to_numpy()).shift(-1).to_dict()
    f["decision_date"] = f["date"].map(next_map)
    f = f.dropna(subset=["decision_date"]).copy()
    keep = ["symbol", "decision_date"] + FNO_FEATURES_8
    f = f[keep].drop_duplicates(["symbol", "decision_date"], keep="last")

    merged = p.merge(
        f,
        left_on=["symbol", "date"],
        right_on=["symbol", "decision_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["decision_date"])

    observed = merged[FNO_FEATURES_8].notna().any(axis=1)
    specialist = merged.loc[observed].copy()
    specialist = specialist.sort_values(["date", "symbol"]).reset_index(drop=True)

    audit = {
        "fno_source": str(FNO_PATH),
        "fno_source_rows": int(len(fno)),
        "fno_source_symbols": int(fno["symbol"].nunique()),
        "panel_rows": int(len(p)),
        "specialist_rows": int(len(specialist)),
        "specialist_coverage_pct": float(observed.mean() * 100),
        "specialist_symbols": int(specialist["symbol"].nunique()),
        "pit_rule": "F&O trade date -> next actual panel trading session",
        "fno_decision_min_date": str(pd.to_datetime(f["decision_date"]).min().date()),
        "fno_decision_max_date": str(pd.to_datetime(f["decision_date"]).max().date()),
    }
    return merged, specialist, audit


def make_targets(frame: pd.DataFrame, steps: int, horizon_name: str) -> pd.DataFrame:
    x = frame.sort_values(["symbol", "date"]).copy()
    future = x.groupby("symbol", sort=False)["close"].shift(-steps) / x["close"] - 1.0
    x[f"future_return_{horizon_name}"] = future.replace([np.inf, -np.inf], np.nan)

    # Cross-sectional benchmark is computed from future labels only. It is
    # therefore unavailable to features at decision time and is used solely
    # to define an excess-return target.
    benchmark = x.groupby("date")[f"future_return_{horizon_name}"].transform("median")
    x[f"benchmark_future_return_{horizon_name}"] = benchmark
    x[f"excess_return_{horizon_name}"] = (
        x[f"future_return_{horizon_name}"] - benchmark
    )
    x[f"target_up_{horizon_name}"] = (
        x[f"excess_return_{horizon_name}"] > 0
    ).astype(int)
    return x.dropna(subset=[f"excess_return_{horizon_name}"]).copy()


def ordered_dates(frame: pd.DataFrame) -> np.ndarray:
    return np.array(sorted(pd.to_datetime(frame["date"]).dropna().dt.normalize().unique()))


def make_walkforward_folds(frame: pd.DataFrame, horizon_steps: int) -> list[dict]:
    dates = ordered_dates(frame)
    if len(dates) < MIN_TRAIN_DATES + VALIDATION_DAYS:
        raise RuntimeError(
            f"Not enough specialist dates: {len(dates)}; "
            f"need at least {MIN_TRAIN_DATES + VALIDATION_DAYS}"
        )

    # Three chronological validation windows taken from the final ~60 sessions.
    n_val = VALIDATION_DAYS
    windows = []
    for i in range(N_FOLDS, 0, -1):
        end_idx = len(dates) - (i - 1) * n_val
        start_idx = max(0, end_idx - n_val)
        if start_idx < MIN_TRAIN_DATES:
            continue
        windows.append((dates[start_idx], dates[end_idx - 1]))
    windows = sorted(set(windows), key=lambda z: z[0])

    folds = []
    for k, (val_start, val_end) in enumerate(windows, start=1):
        val_mask = frame["date"].between(val_start, val_end)
        val_dates = frame.loc[val_mask, "date"].dropna().unique()
        if len(val_dates) == 0:
            continue

        # Purge training rows whose horizon can overlap the validation window.
        val_start_ts = pd.Timestamp(val_start)
        purge_before = val_start_ts - pd.Timedelta(days=max(2, horizon_steps * 2))
        train_mask = frame["date"] < purge_before

        # Explicit session-count embargo immediately preceding the validation start.
        prior_dates = dates[dates < val_start_ts]
        if len(prior_dates):
            embargo_dates = set(prior_dates[-EMBARGO_SESSIONS:])
            train_mask &= ~frame["date"].isin(embargo_dates)

        train_dates = frame.loc[train_mask, "date"].dropna().unique()
        if len(train_dates) < MIN_TRAIN_DATES:
            continue

        folds.append({
            "fold": k,
            "train_min_date": str(pd.Timestamp(train_dates.min()).date()),
            "train_max_date": str(pd.Timestamp(train_dates.max()).date()),
            "validation_min_date": str(pd.Timestamp(val_start).date()),
            "validation_max_date": str(pd.Timestamp(val_end).date()),
            "train_dates": int(len(train_dates)),
            "validation_dates": int(len(val_dates)),
            "purge_rule_days": int(max(2, horizon_steps * 2)),
            "embargo_sessions": EMBARGO_SESSIONS,
        })
    if not folds:
        raise RuntimeError("No viable walk-forward folds were constructed")
    return folds


def pipelines() -> dict[str, tuple[Pipeline, Pipeline | None]]:
    def imp():
        return SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )

    logistic = Pipeline([
        ("imputer", imp()),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=0.5, class_weight="balanced", solver="lbfgs",
            max_iter=2500, random_state=42
        )),
    ])
    hgb_clf = Pipeline([
        ("imputer", imp()),
        ("model", HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.035, max_leaf_nodes=15,
            l2_regularization=2.5, random_state=42
        )),
    ])
    hgb_reg = Pipeline([
        ("imputer", imp()),
        ("model", HistGradientBoostingRegressor(
            max_iter=220, learning_rate=0.035, max_leaf_nodes=15,
            l2_regularization=2.5, random_state=42
        )),
    ])
    return {"logistic": (logistic, None), "hgb": (hgb_clf, hgb_reg)}


def fit_one_fold(train: pd.DataFrame, val: pd.DataFrame, horizon: str, features: list[str]) -> list[dict]:
    Xtr = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xva = val[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ytr = train[f"target_up_{horizon}"].astype(int)
    yva = val[f"target_up_{horizon}"].astype(int)
    rtr = train[f"excess_return_{horizon}"].astype(float)
    rva = val[f"excess_return_{horizon}"].astype(float)

    fitted = []
    probs = {}
    returns = {}
    for name, (clf, reg) in pipelines().items():
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xva)[:, 1]
        probs[name] = prob
        returns[name] = reg.predict(Xva) if reg is not None else np.zeros(len(val))
        roc = float(roc_auc_score(yva, prob)) if yva.nunique() == 2 else None
        fitted.append({
            "model": name,
            "accuracy": float(np.mean((prob >= 0.5) == yva.to_numpy())),
            "balanced_accuracy": float(balanced_accuracy_score(yva, (prob >= 0.5).astype(int))),
            "roc_auc": roc,
        })

    ensemble_prob = (probs["logistic"] + probs["hgb"]) / 2.0
    hgb_reg = pipelines()["hgb"][1]
    hgb_reg.fit(Xtr, rtr)
    ensemble_ret = hgb_reg.predict(Xva)

    fitted.append({
        "model": "ensemble_logistic_hgb_classifier",
        "accuracy": float(np.mean((ensemble_prob >= 0.5) == yva.to_numpy())),
        "balanced_accuracy": float(balanced_accuracy_score(yva, (ensemble_prob >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(yva, ensemble_prob)) if yva.nunique() == 2 else None,
        "return_mae": float(mean_absolute_error(rva, ensemble_ret)),
        "validation_rows": int(len(val)),
        "validation_up_rate": float(yva.mean()),
        "predicted_up_rate": float((ensemble_prob >= 0.5).mean()),
    })
    return fitted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--fno", default=str(FNO_PATH))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if len(FEATURES_59) != 59:
        raise RuntimeError(f"Expected 59 base features, found {len(FEATURES_59)}")

    panel = pd.read_parquet(args.panel)
    fno = pd.read_parquet(args.fno)
    required = {"symbol", "date", "close"} | set(FEATURES_59)
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RuntimeError(f"Panel missing required columns: {missing}")

    merged, specialist, audit = attach_fno_and_subset(panel, fno)
    specialist.to_parquet(out / "FNO_WALKFORWARD_SPECIALIST_PANEL_2026.parquet", index=False)

    report = {
        "experiment_version": "2026-10-03-fno-excess-return-walkforward-v1",
        "status": "CANDIDATE_ONLY",
        "production_promotion": False,
        "self_driving_touched": False,
        "target_definition": "future stock return minus same-date cross-sectional median future return within the F&O-observed specialist universe",
        "base_features": 59,
        "fno_features": 8,
        "feature_count": 67,
        "horizons": [x[0] for x in HORIZONS],
        "walkforward": {
            "folds_requested": N_FOLDS,
            "validation_days_per_fold": VALIDATION_DAYS,
            "min_train_dates": MIN_TRAIN_DATES,
            "embargo_sessions": EMBARGO_SESSIONS,
        },
        "audit": audit,
        "fold_results": {},
    }

    for horizon, steps in HORIZONS:
        d = make_targets(specialist, steps, horizon)
        folds = make_walkforward_folds(d, steps)
        fold_rows = []

        print("=" * 100)
        print(f"F&O EXCESS-RETURN WALK-FORWARD: {horizon}")
        print("=" * 100)

        for meta in folds:
            tr_min = pd.Timestamp(meta["train_min_date"])
            tr_max = pd.Timestamp(meta["train_max_date"])
            va_min = pd.Timestamp(meta["validation_min_date"])
            va_max = pd.Timestamp(meta["validation_max_date"])
            train_mask = d["date"].between(tr_min, tr_max)
            val_mask = d["date"].between(va_min, va_max)
            train = d.loc[train_mask].copy()
            val = d.loc[val_mask].copy()

            results = fit_one_fold(train, val, horizon, FEATURES_67)
            row = {**meta, "results": results}
            fold_rows.append(row)
            print(
                f"fold={meta['fold']} train={len(train):,} val={len(val):,} "
                f"window={meta['validation_min_date']}..{meta['validation_max_date']}",
                flush=True
            )
            for rr in results:
                print(
                    f"  {rr['model']}: accuracy={rr.get('accuracy'):.6f} "
                    f"balanced={rr.get('balanced_accuracy'):.6f} "
                    f"roc_auc={rr.get('roc_auc')}",
                    flush=True
                )

        report["fold_results"][horizon] = fold_rows

    # Train final candidate artifacts on all available specialist observations
    # after the last fold's validation window, only for deployment packaging.
    # These remain candidates and are not copied into the production model dir.
    for horizon, steps in HORIZONS:
        d = make_targets(specialist, steps, horizon)
        final_dir = out / f"candidate_{horizon}"
        final_dir.mkdir(parents=True, exist_ok=True)
        X = d[FEATURES_67].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = d[f"target_up_{horizon}"].astype(int)
        r = d[f"excess_return_{horizon}"].astype(float)

        logit = pipelines()["logistic"][0]
        hgb_clf, hgb_reg = pipelines()["hgb"]
        logit.fit(X, y)
        hgb_clf.fit(X, y)
        hgb_reg.fit(X, r)

        artifacts = {
            "logistic_classifier": final_dir / "fno_excess_logistic_classifier_2026.joblib",
            "hgb_classifier": final_dir / "fno_excess_hgb_classifier_2026.joblib",
            "hgb_regressor": final_dir / "fno_excess_hgb_regressor_2026.joblib",
        }
        joblib.dump(logit, artifacts["logistic_classifier"])
        joblib.dump(hgb_clf, artifacts["hgb_classifier"])
        joblib.dump(hgb_reg, artifacts["hgb_regressor"])

        report.setdefault("final_candidate_artifacts", {})[horizon] = {
            "status": "CANDIDATE_ONLY",
            "training_rows": int(len(d)),
            "training_min_date": str(d["date"].min().date()),
            "training_max_date": str(d["date"].max().date()),
            "classifier_features": FEATURES_67,
            "classifier_feature_count": 67,
            "artifacts": {k: str(v) for k, v in artifacts.items()},
            "sha256": {k: sha256_file(v) for k, v in artifacts.items()},
        }

    report["feature_contract_hash"] = hashlib.sha256(
        json.dumps(FEATURES_67, separators=(",", ":")).encode()
    ).hexdigest()
    report_path = out / "FNO_EXCESS_RETURN_WALKFORWARD_REPORT_2026.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("F&O EXCESS-RETURN WALK-FORWARD COMPLETE")
    print("=" * 100)
    print(json.dumps(audit, indent=2))
    print(f"REPORT: {report_path}")


if __name__ == "__main__":
    main()
