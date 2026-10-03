#!/usr/bin/env python3
"""F&O specialist v2: purged walk-forward excess-return experiment.

Important:
1. Future-return labels are computed on the FULL equity panel before the F&O
   subset is selected. Thus 1/3/5/7d means actual panel trading sessions.
2. The cross-sectional benchmark uses only decisions with explicit PIT-matched
   F&O data on the decision date.
3. Walk-forward folds are built from actual panel sessions, with purge equal
   to the forecast horizon plus a one-session embargo.
4. Baseline 59, F&O-enhanced 67, and F&O-only 8 controls are evaluated on the
   exact same specialist rows/folds.
5. Everything is written to a separate candidate directory. Nothing is
   promoted to production and the self-driving layer is untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
OUT_DIR = BASE / "models" / "all_nse_delivery_fno_walkforward_v2_2026"
PANEL_PATH = BASELINE_DIR / "ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"
FNO_PATH = BASE / "production_data/nse_official/fno/fno_daily_features_2026.parquet"

try:
    from multifactor_data_layer_v2 import FEATURES_59
except Exception as exc:
    raise SystemExit(f"Cannot import 59-feature contract: {exc}")

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
VALIDATION_SESSIONS = 20
MIN_TRAIN_SESSIONS = 40
EMBARGO_SESSIONS = 1


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


def add_fno_features(fno: pd.DataFrame) -> pd.DataFrame:
    x = fno.copy()
    required = {"symbol", "date"}
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"F&O layer missing required columns: {missing}")

    x["symbol"] = x["symbol"].map(norm_symbol)
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()

    numeric = [
        "fno_atm_iv", "fno_pcr", "fno_fut_basis", "fno_fut_oi",
        "fno_fut_close", "fno_iv_change",
    ]
    for col in numeric:
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = (
        x.dropna(subset=["symbol", "date"])
        .sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .copy()
    )
    g = x.groupby("symbol", sort=False)

    def z20(series: pd.Series) -> pd.Series:
        mean = series.rolling(20, min_periods=10).mean()
        std = series.rolling(20, min_periods=10).std()
        return (series - mean) / std.replace(0, np.nan)

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


def merge_fno_pit(panel: pd.DataFrame, fno: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = panel.copy()
    p["symbol"] = p["symbol"].map(norm_symbol)
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p = p.dropna(subset=["symbol", "date"]).copy()

    f = add_fno_features(fno)
    sessions = np.array(sorted(p["date"].dropna().unique()))
    if sessions.size == 0:
        raise RuntimeError("Panel contains no trading sessions")

    next_map = pd.Series(sessions, index=sessions).shift(-1).to_dict()
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
    merged["fno_specialist_observed"] = observed.astype("int8")

    audit = {
        "panel_rows": int(len(merged)),
        "panel_symbols": int(merged["symbol"].nunique()),
        "panel_sessions": int(len(sessions)),
        "fno_source_rows": int(len(fno)),
        "fno_source_symbols": int(fno["symbol"].nunique()),
        "specialist_rows": int(observed.sum()),
        "specialist_coverage_pct": float(observed.mean() * 100),
        "specialist_symbols": int(merged.loc[observed, "symbol"].nunique()),
        "fno_source_min_date": str(pd.Timestamp(fno["date"].min()).date()),
        "fno_source_max_date": str(pd.Timestamp(fno["date"].max()).date()),
        "fno_decision_min_date": str(pd.Timestamp(f["decision_date"].min()).date()),
        "fno_decision_max_date": str(pd.Timestamp(f["decision_date"].max()).date()),
        "pit_rule": "F&O trade date -> next actual panel trading session",
    }
    return merged, audit


def build_targets(merged: pd.DataFrame, steps: int, horizon: str) -> pd.DataFrame:
    # Labels are calculated on the full panel BEFORE selecting specialist rows.
    x = merged.sort_values(["symbol", "date"]).copy()
    x[f"future_return_{horizon}"] = (
        x.groupby("symbol", sort=False)["close"].shift(-steps) / x["close"] - 1.0
    ).replace([np.inf, -np.inf], np.nan)

    observed = x["fno_specialist_observed"].eq(1)
    benchmark = (
        x.loc[observed, ["date", f"future_return_{horizon}"]]
        .groupby("date")[f"future_return_{horizon}"]
        .median()
    )
    x[f"benchmark_future_return_{horizon}"] = x["date"].map(benchmark)
    x[f"excess_return_{horizon}"] = (
        x[f"future_return_{horizon}"] - x[f"benchmark_future_return_{horizon}"]
    )
    x[f"target_up_{horizon}"] = (x[f"excess_return_{horizon}"] > 0).astype("int8")

    keep = observed & x[f"excess_return_{horizon}"].notna()
    return x.loc[keep].sort_values(["date", "symbol"]).reset_index(drop=True)


def make_folds(
    frame: pd.DataFrame,
    panel_sessions: np.ndarray,
    horizon_steps: int,
) -> list[dict]:
    sessions = np.array(sorted(pd.to_datetime(panel_sessions)))
    if len(sessions) < MIN_TRAIN_SESSIONS + VALIDATION_SESSIONS:
        raise RuntimeError(
            f"Only {len(sessions)} panel sessions; "
            f"need at least {MIN_TRAIN_SESSIONS + VALIDATION_SESSIONS}"
        )

    windows = []
    for i in range(N_FOLDS, 0, -1):
        end = len(sessions) - (i - 1) * VALIDATION_SESSIONS
        start = end - VALIDATION_SESSIONS
        if start >= MIN_TRAIN_SESSIONS:
            windows.append((sessions[start], sessions[end - 1]))

    folds = []
    for fold_no, (val_start, val_end) in enumerate(windows, start=1):
        val_mask = frame["date"].between(val_start, val_end)
        if not bool(val_mask.any()):
            continue

        val_start_idx = int(np.searchsorted(sessions, pd.Timestamp(val_start)))
        train_cut_idx = val_start_idx - horizon_steps - EMBARGO_SESSIONS
        if train_cut_idx <= 0:
            continue
        last_train_date = sessions[train_cut_idx - 1]

        train_mask = frame["date"] <= last_train_date
        train_dates = frame.loc[train_mask, "date"].unique()
        val_dates = frame.loc[val_mask, "date"].unique()
        if len(train_dates) < MIN_TRAIN_SESSIONS:
            continue

        folds.append({
            "fold": fold_no,
            "train_min_date": str(pd.Timestamp(train_dates.min()).date()),
            "train_max_date": str(pd.Timestamp(train_dates.max()).date()),
            "validation_min_date": str(pd.Timestamp(val_start).date()),
            "validation_max_date": str(pd.Timestamp(val_end).date()),
            "train_sessions": int(len(train_dates)),
            "validation_sessions": int(len(val_dates)),
            "purge_horizon_sessions": int(horizon_steps),
            "embargo_sessions": int(EMBARGO_SESSIONS),
        })

    if not folds:
        raise RuntimeError("No viable walk-forward folds")
    return folds


def make_models() -> dict[str, Pipeline]:
    def imp() -> SimpleImputer:
        return SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )

    return {
        "logistic": Pipeline([
            ("imputer", imp()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.5,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2500,
                random_state=42,
            )),
        ]),
        "hgb_classifier": Pipeline([
            ("imputer", imp()),
            ("model", HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.035,
                max_leaf_nodes=15,
                l2_regularization=2.5,
                random_state=42,
            )),
        ]),
    }


def make_regressor() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )),
        ("model", HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=15,
            l2_regularization=2.5,
            random_state=42,
        )),
    ])


def evaluate_feature_set(
    train: pd.DataFrame,
    val: pd.DataFrame,
    horizon: str,
    feature_set_name: str,
    features: list[str],
) -> list[dict]:
    Xtr = train[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xva = val[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ytr = train[f"target_up_{horizon}"].astype(int)
    yva = val[f"target_up_{horizon}"].astype(int)
    rtr = train[f"excess_return_{horizon}"].astype(float)
    rva = val[f"excess_return_{horizon}"].astype(float)

    out = []
    probs = []
    models = make_models()
    for model_name, model in models.items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        probs.append(p)
        out.append({
            "feature_set": feature_set_name,
            "model": model_name,
            "accuracy": float(np.mean((p >= 0.5) == yva.to_numpy())),
            "balanced_accuracy": float(
                balanced_accuracy_score(yva, (p >= 0.5).astype(int))
            ),
            "roc_auc": (
                float(roc_auc_score(yva, p))
                if yva.nunique() == 2 else None
            ),
            "return_mae": None,
            "validation_rows": int(len(val)),
            "validation_up_rate": float(yva.mean()),
            "predicted_up_rate": float((p >= 0.5).mean()),
        })

    ensemble_prob = np.mean(np.vstack(probs), axis=0)
    reg = make_regressor()
    reg.fit(Xtr, rtr)
    pred_return = reg.predict(Xva)

    out.append({
        "feature_set": feature_set_name,
        "model": "logistic_hgb_ensemble",
        "accuracy": float(np.mean((ensemble_prob >= 0.5) == yva.to_numpy())),
        "balanced_accuracy": float(
            balanced_accuracy_score(yva, (ensemble_prob >= 0.5).astype(int))
        ),
        "roc_auc": (
            float(roc_auc_score(yva, ensemble_prob))
            if yva.nunique() == 2 else None
        ),
        "return_mae": float(mean_absolute_error(rva, pred_return)),
        "validation_rows": int(len(val)),
        "validation_up_rate": float(yva.mean()),
        "predicted_up_rate": float((ensemble_prob >= 0.5).mean()),
    })
    return out


def aggregate_rows(rows: list[dict]) -> pd.DataFrame:
    flat = []
    for row in rows:
        meta = {k: v for k, v in row.items() if k != "results"}
        for result in row["results"]:
            flat.append({**meta, **result})
    df = pd.DataFrame(flat)
    if df.empty:
        return df

    summary = (
        df.groupby(["feature_set", "model"], dropna=False)
        .agg(
            folds=("fold", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_return_mae=("return_mae", "mean"),
            std_return_mae=("return_mae", "std"),
            total_validation_rows=("validation_rows", "sum"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL_PATH))
    ap.add_argument("--fno", default=str(FNO_PATH))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if len(FEATURES_59) != 59 or len(set(FEATURES_59)) != 59:
        raise RuntimeError(f"Invalid 59-feature contract: {len(FEATURES_59)}")

    panel = pd.read_parquet(args.panel)
    fno = pd.read_parquet(args.fno)

    required = {"symbol", "date", "close"} | set(FEATURES_59)
    missing = sorted(required - set(panel.columns))
    if missing:
        raise RuntimeError(f"Panel missing required columns: {missing}")

    merged, audit = merge_fno_pit(panel, fno)
    panel_sessions = np.array(sorted(pd.to_datetime(panel["date"]).dropna().dt.normalize().unique()))

    merged.to_parquet(out / "FNO_WALKFORWARD_V2_MERGED_PANEL_2026.parquet", index=False)

    feature_sets = {
        "baseline_59_control": list(FEATURES_59),
        "fno_enhanced_67": list(FEATURES_67),
        "fno_only_8_diagnostic": list(FNO_FEATURES_8),
    }

    report = {
        "experiment_version": "2026-10-03-fno-excess-return-walkforward-v2",
        "status": "CANDIDATE_ONLY",
        "production_promotion": False,
        "self_driving_touched": False,
        "target_definition": (
            "Future stock return over actual panel sessions minus the same-date "
            "cross-sectional median future return among F&O-observed decisions."
        ),
        "horizon_definition": "1/3/5/7 actual panel trading sessions",
        "feature_sets": {k: len(v) for k, v in feature_sets.items()},
        "walkforward": {
            "folds_requested": N_FOLDS,
            "validation_sessions_per_fold": VALIDATION_SESSIONS,
            "min_train_sessions": MIN_TRAIN_SESSIONS,
            "purge_horizon_sessions": "equal to forecast horizon",
            "embargo_sessions": EMBARGO_SESSIONS,
        },
        "audit": audit,
        "fold_results": {},
        "promotion_gate": (
            "No promotion based on this run alone. Require stable multi-fold "
            "improvement versus the same-fold 59-feature control and separate "
            "final holdout validation."
        ),
    }

    all_fold_rows = []

    for horizon, steps in HORIZONS:
        d = build_targets(merged, steps, horizon)
        folds = make_folds(d, panel_sessions, steps)
        fold_rows = []

        print("=" * 100)
        print(f"F&O SPECIALIST V2 WALK-FORWARD: {horizon}")
        print("=" * 100)

        for fold in folds:
            tr0 = pd.Timestamp(fold["train_min_date"])
            tr1 = pd.Timestamp(fold["train_max_date"])
            va0 = pd.Timestamp(fold["validation_min_date"])
            va1 = pd.Timestamp(fold["validation_max_date"])

            train = d.loc[d["date"].between(tr0, tr1)].copy()
            val = d.loc[d["date"].between(va0, va1)].copy()

            results = []
            for set_name, features in feature_sets.items():
                results.extend(
                    evaluate_feature_set(
                        train, val, horizon, set_name, features
                    )
                )

            row = {**fold, "results": results}
            fold_rows.append(row)
            all_fold_rows.append({
                "horizon": horizon,
                **row,
            })

            print(
                f"fold={fold['fold']} "
                f"train_rows={len(train):,} val_rows={len(val):,} "
                f"window={fold['validation_min_date']}..{fold['validation_max_date']}",
                flush=True,
            )
            for rr in results:
                print(
                    f"  {rr['feature_set']} {rr['model']}: "
                    f"acc={rr['accuracy']:.6f} "
                    f"bal={rr['balanced_accuracy']:.6f} "
                    f"auc={rr['roc_auc']} "
                    f"mae={rr['return_mae']}",
                    flush=True,
                )

        report["fold_results"][horizon] = fold_rows

        summary = aggregate_rows(fold_rows)
        summary.to_csv(
            out / f"FNO_V2_{horizon.upper()}_WALKFORWARD_SUMMARY.csv",
            index=False,
        )

    overall = aggregate_rows(all_fold_rows)
    overall.to_csv(out / "FNO_V2_ALL_HORIZONS_WALKFORWARD_SUMMARY.csv", index=False)

    # Package only the principal 67-feature specialist candidate. It remains
    # isolated from the existing production/replacement directory.
    report["final_candidate_artifacts"] = {}
    for horizon, steps in HORIZONS:
        d = build_targets(merged, steps, horizon)
        final_dir = out / f"candidate_{horizon}"
        final_dir.mkdir(parents=True, exist_ok=True)

        X = d[FEATURES_67].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = d[f"target_up_{horizon}"].astype(int)
        r = d[f"excess_return_{horizon}"].astype(float)

        models = make_models()
        reg = make_regressor()
        artifacts = {}

        for name, model in models.items():
            model.fit(X, y)
            path = final_dir / f"fno_v2_{horizon}_{name}_classifier_2026.joblib"
            joblib.dump(model, path)
            artifacts[name] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

        reg.fit(X, r)
        reg_path = final_dir / f"fno_v2_{horizon}_hgb_regressor_2026.joblib"
        joblib.dump(reg, reg_path)
        artifacts["hgb_regressor"] = {
            "path": str(reg_path),
            "sha256": sha256_file(reg_path),
        }

        report["final_candidate_artifacts"][horizon] = {
            "status": "CANDIDATE_ONLY",
            "training_rows": int(len(d)),
            "training_symbols": int(d["symbol"].nunique()),
            "training_min_date": str(d["date"].min().date()),
            "training_max_date": str(d["date"].max().date()),
            "feature_count": 67,
            "artifacts": artifacts,
        }

    report["feature_contract_hash"] = hashlib.sha256(
        json.dumps(FEATURES_67, separators=(",", ":")).encode()
    ).hexdigest()

    report_path = out / "FNO_V2_EXCESS_RETURN_WALKFORWARD_REPORT_2026.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print("=" * 100)
    print("F&O SPECIALIST V2 COMPLETE")
    print("=" * 100)
    print(json.dumps(audit, indent=2))
    print(f"REPORT: {report_path}")


if __name__ == "__main__":
    main()
