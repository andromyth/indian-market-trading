#!/usr/bin/env python3
"""Train a separate ALL-NSE F&O-enhanced Delivery candidate on the repaired daily F&O layer.

This is an experiment/candidate only:
- does not modify or promote the existing 59-feature candidate
- does not modify the self-driving layer
- does not place orders
- preserves missing F&O values as NaN
- applies F&O PIT: trade-date F&O becomes usable on the next panel trading session
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
BASELINE_DIR = BASE / "models" / "all_nse_delivery_multifactor_replacement_2026"
OUT_DIR = BASE / "models" / "all_nse_delivery_multifactor_fno_enhanced_2026"
FNO_PATH = BASE / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
PANEL_PATH = BASELINE_DIR / "ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"

try:
    from multifactor_data_layer_v2 import FEATURES_59
except Exception as exc:
    raise SystemExit(f"Cannot import FEATURES_59 from repository: {exc}")


FNO_EXPERIMENTAL = [
    "fno_atm_iv",
    "fno_atm_iv_z20",
    "fno_pcr_z20",
    "fno_basis_z20",
    "fno_oi_chg_z20",
    "fno_iv_chg_z20",
    "fno_buildup",
    "fno_buildup_persist3",
]

HORIZONS = [
    ("next_session", 1),
    ("3d", 3),
    ("5d", 5),
    ("7d", 7),
]


def norm_symbol(s: object) -> str:
    x = str(s or "").strip().upper()
    for suf in (".NS", ".BO", "-EQ", ".EQ"):
        if x.endswith(suf):
            x = x[: -len(suf)]
    return x


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_fno_specialist_features(fno: pd.DataFrame) -> pd.DataFrame:
    x = fno.copy()
    x["symbol"] = x["symbol"].map(norm_symbol)
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()

    numeric = [
        "fno_atm_iv",
        "fno_pcr",
        "fno_fut_basis",
        "fno_fut_oi",
        "fno_fut_close",
        "fno_oi_change",
        "fno_iv_change",
    ]
    for c in numeric:
        if c not in x:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.dropna(subset=["symbol", "date"]).sort_values(["symbol", "date"]).copy()
    g = x.groupby("symbol", sort=False)

    def z20(s: pd.Series) -> pd.Series:
        mu = s.rolling(20, min_periods=10).mean()
        sd = s.rolling(20, min_periods=10).std()
        return (s - mu) / sd.replace(0, np.nan)

    x["fno_atm_iv_z20"] = g["fno_atm_iv"].transform(z20)
    x["fno_pcr_z20"] = g["fno_pcr"].transform(z20)
    x["fno_basis_z20"] = g["fno_fut_basis"].transform(z20)
    x["fno_oi_chg_z20"] = g["fno_fut_oi"].pct_change().groupby(x["symbol"]).transform(z20)
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


def attach_fno_pit(panel: pd.DataFrame, fno: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = panel.copy()
    p["symbol"] = p["symbol"].map(norm_symbol)
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()

    f = add_fno_specialist_features(fno)
    if f.empty:
        raise RuntimeError("F&O derived layer is empty after normalization")

    # F&O trade-date information becomes usable at the next actual panel session.
    sessions = pd.Series(sorted(p["date"].dropna().unique()))
    next_map = pd.Series(sessions.to_numpy(), index=sessions.to_numpy()).shift(-1).to_dict()
    f["decision_date"] = f["date"].map(next_map)
    f = f.dropna(subset=["decision_date"]).copy()

    keep = ["symbol", "decision_date"] + FNO_EXPERIMENTAL
    f = f[keep].drop_duplicates(["symbol", "decision_date"], keep="last")

    before_cols = len(p.columns)
    p = p.merge(
        f,
        left_on=["symbol", "date"],
        right_on=["symbol", "decision_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["decision_date"])

    coverage = {
        c: float(p[c].notna().mean())
        for c in FNO_EXPERIMENTAL
    }

    audit = {
        "source": str(FNO_PATH),
        "fno_source_rows": int(len(fno)),
        "fno_source_symbols": int(fno["symbol"].nunique()),
        "merged_rows": int(len(p)),
        "panel_columns_before": before_cols,
        "panel_columns_after": int(len(p.columns)),
        "fno_pit_rule": "F&O trade date -> next actual panel trading session",
        "experimental_feature_coverage": coverage,
        "fno_source_min_date": str(pd.to_datetime(fno["date"]).min().date()),
        "fno_source_max_date": str(pd.to_datetime(fno["date"]).max().date()),
        "fno_decision_min_date": str(pd.to_datetime(f["decision_date"]).min().date()),
        "fno_decision_max_date": str(pd.to_datetime(f["decision_date"]).max().date()),
    }
    return p, audit


def add_targets(panel: pd.DataFrame, steps: int, name: str) -> pd.DataFrame:
    x = panel.sort_values(["symbol", "date"]).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        x[f"target_return_{name}"] = (
            x.groupby("symbol")["close"].shift(-steps) / x["close"] - 1.0
        )
    x[f"target_return_{name}"] = x[f"target_return_{name}"].replace(
        [np.inf, -np.inf], np.nan
    )
    x[f"target_up_{name}"] = (x[f"target_return_{name}"] > 0).astype(int)
    return x.dropna(subset=[f"target_return_{name}"])


def fit_models(frame: pd.DataFrame, horizon: str, feature_set: list[str]) -> tuple[dict, dict, dict]:
    cutoff = pd.to_datetime(frame["date"]).max() - pd.Timedelta(days=30)
    tr = frame[pd.to_datetime(frame["date"]) < cutoff].copy()
    te = frame[pd.to_datetime(frame["date"]) >= cutoff].copy()
    if tr.empty or te.empty:
        raise RuntimeError(f"{horizon}: empty chronological train/holdout split")

    Xtr = (
        tr[feature_set]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    Xte = (
        te[feature_set]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    y = tr[f"target_up_{horizon}"].astype(int)
    yr = tr[f"target_return_{horizon}"].astype(float)

    clf = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=0.5,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2500,
            random_state=42,
        )),
    ])
    reg = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )),
        ("model", HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.04,
            max_leaf_nodes=19,
            l2_regularization=2.0,
            random_state=42,
        )),
    ])

    clf.fit(Xtr, y)
    reg.fit(Xtr, yr)

    imp_c = clf.named_steps["imputer"]
    imp_r = reg.named_steps["imputer"]
    learned_c = int(imp_c.n_features_in_)
    learned_r = int(imp_r.n_features_in_)
    if learned_c != len(feature_set) or learned_r != len(feature_set):
        raise RuntimeError(
            f"{horizon}: model contract mismatch: "
            f"classifier={learned_c}, regressor={learned_r}, expected={len(feature_set)}"
        )

    p = clf.predict_proba(Xte)[:, 1]
    pr = reg.predict(Xte)
    yte = te[f"target_up_{horizon}"].to_numpy()
    rte = te[f"target_return_{horizon}"].to_numpy()

    metrics = {
        "train_rows": int(len(tr)),
        "holdout_rows": int(len(te)),
        "train_min_date": str(pd.to_datetime(tr["date"]).min().date()),
        "train_max_date": str(pd.to_datetime(tr["date"]).max().date()),
        "holdout_min_date": str(pd.to_datetime(te["date"]).min().date()),
        "holdout_max_date": str(pd.to_datetime(te["date"]).max().date()),
        "holdout_accuracy": float(np.mean((p >= 0.5) == yte)),
        "mean_abs_return_error": float(np.mean(np.abs(pr - rte))),
        "holdout_up_rate": float(np.mean(yte)),
        "feature_count": len(feature_set),
    }
    return clf, reg, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--fno", default=str(FNO_PATH))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if len(FEATURES_59) != 59 or len(set(FEATURES_59)) != 59:
        raise RuntimeError(f"Expected exactly 59 unique base features, found {len(FEATURES_59)}")

    panel = pd.read_parquet(args.panel)
    fno = pd.read_parquet(args.fno)

    required_panel = {"symbol", "date", "close"} | set(FEATURES_59)
    missing = sorted(required_panel - set(panel.columns))
    if missing:
        raise RuntimeError(f"Baseline panel missing required columns: {missing}")

    panel, fno_audit = attach_fno_pit(panel, fno)

    # Baseline = the existing 59-feature contract, evaluated on exactly the same
    # rows and chronological split as the F&O-enhanced experiment.
    experiments = {
        "baseline_59": list(FEATURES_59),
        "fno_enhanced_67": list(FEATURES_59) + FNO_EXPERIMENTAL,
    }

    panel.to_parquet(out / "FNO_ENHANCED_PIT_PANEL_2026.parquet", index=False)

    all_results = {
        "experiment_version": "2026-10-03-fno-enhanced-candidate-v1",
        "status": "CANDIDATE_ONLY",
        "production_promotion": False,
        "self_driving_touched": False,
        "base_feature_count": 59,
        "fno_feature_count_added": len(FNO_EXPERIMENTAL),
        "features_added": FNO_EXPERIMENTAL,
        "fno_audit": fno_audit,
        "model_dir": str(out),
        "artifacts": {},
    }

    for model_name, features in experiments.items():
        model_dir = out / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        all_results["artifacts"][model_name] = {}

        for horizon, steps in HORIZONS:
            d = add_targets(panel, steps, horizon)
            try:
                clf, reg, metrics = fit_models(d, horizon, features)
                cp = model_dir / f"deployment_{model_name}_{horizon}_classifier_2026.joblib"
                rp = model_dir / f"deployment_{model_name}_{horizon}_regressor_2026.joblib"
                joblib.dump(clf, cp)
                joblib.dump(reg, rp)
                all_results["artifacts"][model_name][horizon] = {
                    "status": "CANDIDATE_READY",
                    "classifier": str(cp),
                    "regressor": str(rp),
                    "classifier_sha256": sha256_file(cp),
                    "regressor_sha256": sha256_file(rp),
                    "metrics": metrics,
                }
                print(
                    f"{model_name} {horizon}: "
                    f"accuracy={metrics['holdout_accuracy']:.6f} "
                    f"MAE={metrics['mean_abs_return_error']:.6f} "
                    f"train={metrics['train_rows']:,} "
                    f"holdout={metrics['holdout_rows']:,}",
                    flush=True,
                )
            except Exception as exc:
                all_results["artifacts"][model_name][horizon] = {
                    "status": "BLOCKED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"{model_name} {horizon}: BLOCKED "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    all_results["feature_contract_hash"] = hashlib.sha256(
        json.dumps(FEATURES_59 + FNO_EXPERIMENTAL, separators=(",", ":")).encode()
    ).hexdigest()
    (out / "FNO_ENHANCED_TRAINING_REPORT_2026.json").write_text(
        json.dumps(all_results, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(all_results["fno_audit"], indent=2))
    print(f"REPORT: {out / 'FNO_ENHANCED_TRAINING_REPORT_2026.json'}")


if __name__ == "__main__":
    main()
