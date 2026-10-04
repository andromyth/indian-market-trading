#!/usr/bin/env python3
"""Train an F&O-specialist Delivery candidate on the repaired NSE daily F&O layer.

Experiment only:
- does not modify/promote the existing 59-feature candidate
- does not modify the self-driving layer
- does not place orders
- trains only on panel rows with PIT-matched F&O observations
- preserves missing values and uses explicit imputation indicators
- evaluates a baseline 59-feature model on the same F&O subset
- evaluates 59 + repaired/derived F&O features
- includes an 8-feature F&O-only diagnostic model
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
OUT_DIR = BASE / "models" / "all_nse_delivery_fno_specialist_2026"
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
    """Recompute derived specialist features from the persisted daily F&O layer."""
    x = fno.copy()
    required = {"symbol", "date"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"F&O layer missing required columns: {sorted(missing)}")

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


def attach_fno_pit(
    panel: pd.DataFrame,
    fno: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Attach F&O trade-date information to the next actual panel session."""
    p = panel.copy()
    p["symbol"] = p["symbol"].map(norm_symbol)
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p = p.dropna(subset=["symbol", "date"]).copy()

    f = add_fno_specialist_features(fno)
    if f.empty:
        raise RuntimeError("F&O derived layer is empty after normalization")

    sessions = pd.Series(sorted(p["date"].dropna().unique()))
    next_map = (
        pd.Series(sessions.to_numpy(), index=sessions.to_numpy())
        .shift(-1)
        .to_dict()
    )
    f["decision_date"] = f["date"].map(next_map)
    f = f.dropna(subset=["decision_date"]).copy()

    keep = ["symbol", "decision_date"] + FNO_FEATURES_8
    f = f[keep].drop_duplicates(["symbol", "decision_date"], keep="last")

    p = p.merge(
        f,
        left_on=["symbol", "date"],
        right_on=["symbol", "decision_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["decision_date"])

    observed_mask = p[FNO_FEATURES_8].notna().any(axis=1)
    specialist = p.loc[observed_mask].copy()

    coverage = {
        c: float(p[c].notna().mean()) for c in FNO_FEATURES_8
    }
    audit = {
        "source": str(FNO_PATH),
        "fno_source_rows": int(len(fno)),
        "fno_source_symbols": int(fno["symbol"].nunique()),
        "merged_panel_rows": int(len(p)),
        "specialist_rows_with_any_fno_feature": int(observed_mask.sum()),
        "specialist_coverage_pct": float(observed_mask.mean() * 100),
        "specialist_symbols": int(specialist["symbol"].nunique()),
        "panel_columns_before": int(panel.shape[1]),
        "panel_columns_after": int(p.shape[1]),
        "fno_pit_rule": "F&O trade date -> next actual panel trading session",
        "experimental_feature_coverage": coverage,
        "fno_source_min_date": str(pd.to_datetime(fno["date"]).min().date()),
        "fno_source_max_date": str(pd.to_datetime(fno["date"]).max().date()),
        "fno_decision_min_date": str(pd.to_datetime(f["decision_date"]).min().date()),
        "fno_decision_max_date": str(pd.to_datetime(f["decision_date"]).max().date()),
    }
    return p, specialist, audit


def add_targets(panel: pd.DataFrame, steps: int, name: str) -> pd.DataFrame:
    x = panel.sort_values(["symbol", "date"]).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        x[f"target_return_{name}"] = (
            x.groupby("symbol")["close"].shift(-steps) / x["close"] - 1.0
        )
    x[f"target_return_{name}"] = x[f"target_return_{name}"].replace(
        [np.inf, -np.inf], np.nan
    )
    return x.dropna(subset=[f"target_return_{name}"])


def build_pipelines():
    clf = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        ),
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2500,
                random_state=42,
            ),
        ),
    ])
    reg = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        ),
        (
            "model",
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.04,
                max_leaf_nodes=19,
                l2_regularization=2.0,
                random_state=42,
            ),
        ),
    ])
    return clf, reg


def fit_models(
    frame: pd.DataFrame,
    horizon: str,
    feature_set: list[str],
) -> tuple[object, object, dict]:
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

    ytr = (tr[f"target_return_{horizon}"] > 0).astype(int)
    rtr = tr[f"target_return_{horizon}"].astype(float)

    clf, reg = build_pipelines()
    clf.fit(Xtr, ytr)
    reg.fit(Xtr, rtr)

    p = clf.predict_proba(Xte)[:, 1]
    pr = reg.predict(Xte)
    yte = (te[f"target_return_{horizon}"] > 0).to_numpy()
    rte = te[f"target_return_{horizon}"].to_numpy()

    learned_c = int(clf.named_steps["imputer"].n_features_in_)
    learned_r = int(reg.named_steps["imputer"].n_features_in_)
    if learned_c != len(feature_set) or learned_r != len(feature_set):
        raise RuntimeError(
            f"{horizon}: contract mismatch classifier={learned_c}, "
            f"regressor={learned_r}, expected={len(feature_set)}"
        )

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
        "feature_count": int(len(feature_set)),
        "holdout_symbols": int(te["symbol"].nunique()),
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

    merged_panel, specialist_panel, fno_audit = attach_fno_pit(panel, fno)

    # Persist both the full merged diagnostic panel and the actual F&O-specialist
    # training subset. These live in a separate candidate directory.
    merged_panel.to_parquet(out / "FNO_SPECIALIST_PIT_PANEL_2026.parquet", index=False)
    specialist_panel.to_parquet(
        out / "FNO_SPECIALIST_TRAINING_SUBSET_2026.parquet",
        index=False,
    )

    experiments = {
        # Control: same rows, same dates, no extra F&O features.
        "baseline_59_on_fno_subset": list(FEATURES_59),
        # Main experiment: holistic baseline plus repaired/derived F&O information.
        "fno_enhanced_67_specialist": list(FEATURES_59) + FNO_FEATURES_8,
        # Diagnostic only: isolates the predictive content of F&O features.
        "fno_only_8_diagnostic": list(FNO_FEATURES_8),
    }

    report = {
        "experiment_version": "2026-10-03-fno-specialist-candidate-v1",
        "status": "CANDIDATE_ONLY",
        "production_promotion": False,
        "self_driving_touched": False,
        "base_feature_count": 59,
        "fno_feature_count_added": len(FNO_FEATURES_8),
        "features_added": FNO_FEATURES_8,
        "specialist_training_definition": "rows with at least one PIT-matched F&O specialist feature",
        "fno_audit": fno_audit,
        "model_dir": str(out),
        "artifacts": {},
    }

    for model_name, features in experiments.items():
        model_dir = out / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        report["artifacts"][model_name] = {}

        for horizon, steps in HORIZONS:
            d = add_targets(specialist_panel, steps, horizon)
            try:
                clf, reg, metrics = fit_models(d, horizon, features)

                cp = model_dir / (
                    f"deployment_{model_name}_{horizon}_classifier_2026.joblib"
                )
                rp = model_dir / (
                    f"deployment_{model_name}_{horizon}_regressor_2026.joblib"
                )
                joblib.dump(clf, cp)
                joblib.dump(reg, rp)

                report["artifacts"][model_name][horizon] = {
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
                    f"holdout={metrics['holdout_rows']:,} "
                    f"symbols={metrics['holdout_symbols']:,}",
                    flush=True,
                )
            except Exception as exc:
                report["artifacts"][model_name][horizon] = {
                    "status": "BLOCKED",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                print(
                    f"{model_name} {horizon}: BLOCKED "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

    report["feature_contract_hash"] = hashlib.sha256(
        json.dumps(
            FEATURES_59 + FNO_FEATURES_8,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    report_path = out / "FNO_SPECIALIST_TRAINING_REPORT_2026.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=" * 100)
    print("F&O SPECIALIST CANDIDATE TRAINING")
    print("=" * 100)
    print(json.dumps(fno_audit, indent=2))
    print(f"REPORT: {report_path}")


if __name__ == "__main__":
    main()
