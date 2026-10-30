#!/usr/bin/env python3
"""Read-only purged walk-forward and untouched-holdout audit for the 59-feature Delivery model.

Models are fitted in memory only. This script does not save or promote weights,
update a registry, mutate a prediction ledger, or change a signal policy.
The aggregate panel cannot independently certify historical exchange-universe
membership, so this limitation is explicitly reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from multifactor_data_layer_v2 import FEATURES_59  # noqa: E402

DEFAULT_REMOTE_PANEL = "models/all_nse_delivery_multifactor_replacement_2026/ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"
IST = "Asia/Kolkata"
COVERAGE_FLOOR = 0.70
DEFAULT_RESEARCH_START_DATE = date(2025, 6, 1)
READINESS_COLUMNS = (
    "ready_market_price", "ready_market_context", "ready_news",
    "ready_events", "ready_fno", "ready_fundamentals",
)
HOLDOUT_FRACTION = 0.20
MIN_TRAIN_SESSIONS = 60
OOF_FOLDS = 3
OOF_DAYS_PER_FOLD = 10
MIN_ACTIONS = 30
MIN_ACTION_DATES = 10
TARGET_PRECISION = 0.80
ROUND_TRIP_COST = 0.003  # 30 bps scenario; actual execution costs are not simulated.
MAX_FIT_ROWS = 25000
AUDIT_HGB_MAX_ITER = 45


def parse_holidays() -> set[date]:
    from production_data_freshness_gate import HOLIDAYS
    out = set()
    for value in HOLIDAYS:
        try:
            out.add(pd.Timestamp(value).date())
        except Exception:
            pass
    return out



def summarize_production_inventory(inventory: dict[str, set[date]], target_date: date | None) -> dict[str, Any]:
    """Summarize actual production archive/snapshot dates separately from the research panel."""
    eod_dates = inventory.get("eod", set())
    delivery_dates = inventory.get("delivery", set())
    pit_dates = inventory.get("pit", set())
    target = target_date.isoformat() if target_date is not None else None
    eod_present = target_date in eod_dates if target_date is not None else None
    delivery_present = target_date in delivery_dates if target_date is not None else None
    pit_present = target_date in pit_dates if target_date is not None else None

    if target_date is None:
        status = "NO_NSE_SESSION_EXPECTED"
    elif eod_present and delivery_present and pit_present:
        status = "CURRENT_SESSION_PIT_AND_SOURCES_READY"
    elif eod_present and delivery_present and not pit_present:
        status = "CURRENT_SESSION_PIT_MISSING_SOURCES_READY"
    else:
        status = "CURRENT_SESSION_SOURCE_ARCHIVES_MISSING"

    def latest(values: set[date]) -> str | None:
        return max(values).isoformat() if values else None

    return {
        "status": status,
        "target_session": target,
        "target_eod_archive_present": eod_present,
        "target_delivery_archive_present": delivery_present,
        "target_pit_snapshot_present": pit_present,
        "latest_eod_archive_date": latest(eod_dates),
        "latest_delivery_archive_date": latest(delivery_dates),
        "latest_production_pit_snapshot_date": latest(pit_dates),
        "eod_archive_dates_seen": len(eod_dates),
        "delivery_archive_dates_seen": len(delivery_dates),
        "production_pit_snapshot_dates_seen": len(pit_dates),
        "read_only": True,
    }


def audit_live_production_inventory(now: datetime | None = None) -> dict[str, Any]:
    """Read current production archive/PIT filenames from Drive without modifying them."""
    try:
        from backfill_missing_delivery_pit_snapshots import load_drive_inventory
        from persistent_store import GoogleDriveStore
        from production_data_freshness_gate import expected_session_date

        clock = now or datetime.now(ZoneInfo(IST))
        target = expected_session_date(clock)
        inventory = load_drive_inventory(GoogleDriveStore())
        result = summarize_production_inventory(inventory, target)
        result["checked_at_ist"] = clock.astimezone(ZoneInfo(IST)).isoformat()
        result["source"] = "Google Drive production_data/nse_official and features/daily inventory"
        result["research_panel_is_live_production_snapshot"] = False
        return result
    except Exception as exc:
        return {
            "status": "PRODUCTION_INVENTORY_AUDIT_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "read_only": True,
            "research_panel_is_live_production_snapshot": False,
        }


def safe_gate(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(np.isfinite(value) and float(value) == 1.0)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass", "ready"}
    return False


def quality_gate_from_readiness_flags(row: pd.Series) -> bool:
    """Reconstruct the existing production gate only from explicit PIT readiness states."""
    from delivery_quality_gate import quality_gate_reasons

    return not quality_gate_reasons(row)


def session_successors(session_dates: list[pd.Timestamp], holidays: set[date]):
    """Map to the next observed exchange session; skip labels across a missing weekday."""
    observed = {pd.Timestamp(value).normalize() for value in session_dates}
    successors = {}
    gaps = []
    if not observed:
        return successors, gaps
    maximum = max(observed)
    for ref in sorted(observed):
        candidate = ref + pd.Timedelta(days=1)
        while candidate <= maximum:
            # An observed weekend date is allowed (e.g. an official special session).
            if candidate in observed:
                successors[ref] = candidate
                break
            if candidate.weekday() >= 5 or candidate.date() in holidays:
                candidate += pd.Timedelta(days=1)
                continue
            successors[ref] = None
            gaps.append({
                "reference_date": ref.date().isoformat(),
                "missing_expected_session": candidate.date().isoformat(),
                "status": "SKIPPED_MISSING_EXPECTED_SESSION",
            })
            break
        else:
            successors[ref] = None
    return successors, gaps


def load_panel_symbol_sample(panel_path: Path, symbol_limit: int = 500):
    """Read a deterministic symbol subset via DuckDB instead of materialising the full panel in pandas."""
    import duckdb

    if symbol_limit < 50:
        raise ValueError("symbol_limit must be at least 50 for a useful cross-sectional audit.")
    path_sql = "'" + str(panel_path).replace("'", "''") + "'"
    work_temp = Path("/tmp/strict_delivery_audit_duckdb_tmp")
    work_temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=2")
        con.execute("PRAGMA memory_limit='2GB'")
        temp_sql = "'" + str(work_temp).replace("'", "''") + "'"
        con.execute(f"SET temp_directory={temp_sql}")
        description = con.execute(f"DESCRIBE SELECT * FROM read_parquet({path_sql})").fetchall()
        columns = {str(row[0]) for row in description}
        required = {"symbol", "date", "close", *FEATURES_59}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"Persisted panel missing required columns: {missing[:20]}")
        symbol_expr = r"""REGEXP_REPLACE(UPPER(TRIM(CAST("symbol" AS VARCHAR))), '(\.NS|\.BO|-EQ|\.EQ)$', '')"""
        full_meta = con.execute(f"""
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT {symbol_expr}) AS symbols,
                   COUNT(DISTINCT TRY_CAST("date" AS DATE)) AS sessions,
                   MIN(TRY_CAST("date" AS DATE)) AS first_date,
                   MAX(TRY_CAST("date" AS DATE)) AS last_date
            FROM read_parquet({path_sql})
        """).fetchone()
        sessions = con.execute(f"""
            SELECT DISTINCT TRY_CAST("date" AS DATE) AS session_date
            FROM read_parquet({path_sql})
            WHERE TRY_CAST("date" AS DATE) IS NOT NULL
            ORDER BY session_date
        """).fetchdf()["session_date"].tolist()
        con.execute(f"""
            CREATE TEMP TABLE audit_symbol_sample AS
            WITH symbols AS (
                SELECT DISTINCT {symbol_expr} AS symbol
                FROM read_parquet({path_sql})
                WHERE "symbol" IS NOT NULL
            )
            SELECT symbol FROM symbols
            WHERE symbol <> ''
            ORDER BY HASH(symbol), symbol
            LIMIT {int(symbol_limit)}
        """)
        selected_symbols = [row[0] for row in con.execute(
            "SELECT symbol FROM audit_symbol_sample ORDER BY symbol"
        ).fetchall()]
        use_columns = ["symbol", "date", "close"]
        for optional in ("decision_timestamp", "production_data_quality_gate", *READINESS_COLUMNS):
            if optional in columns:
                use_columns.append(optional)
        use_columns.extend(list(FEATURES_59))
        projection = ", ".join('"' + name.replace('"', '""') + '"' for name in dict.fromkeys(use_columns))
        panel = con.execute(f"""
            SELECT {projection}
            FROM read_parquet({path_sql})
            WHERE {symbol_expr} IN (SELECT symbol FROM audit_symbol_sample)
        """).fetchdf()
        meta = {
            "method": "deterministic symbol-hash sample read by DuckDB; no full wide-panel materialisation in pandas",
            "full_panel_rows": int(full_meta[0] or 0),
            "full_panel_symbols": int(full_meta[1] or 0),
            "full_panel_sessions": int(full_meta[2] or 0),
            "full_panel_first_date": str(full_meta[3]) if full_meta[3] else None,
            "full_panel_last_date": str(full_meta[4]) if full_meta[4] else None,
            "requested_symbol_limit": int(symbol_limit),
            "sampled_symbol_count": int(len(selected_symbols)),
            "sampled_symbols": selected_symbols,
            "sampled_panel_rows": int(len(panel)),
            "sample_fraction_of_symbols_pct": round(100.0 * len(selected_symbols) / max(1, int(full_meta[1] or 0)), 3),
            "row_sampling": "all available dates for a deterministic symbol subset; no sampling based on outcome",
        }
        return panel, meta, sessions
    finally:
        con.close()


def prepare_labeled_panel(
    panel: pd.DataFrame,
    holidays: set[date],
    all_session_dates: list | None = None,
    research_start_date: date | None = DEFAULT_RESEARCH_START_DATE,
):
    """Reconstruct exact-session labels inside a fixed feature-availability research cohort."""
    if panel is None or panel.empty:
        raise ValueError("PIT panel is empty.")
    required = {"symbol", "date", "close", *FEATURES_59}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"PIT panel is missing required columns: {missing[:20]}")

    input_rows = len(panel)
    x = panel.copy()
    x["symbol"] = (
        x["symbol"].astype(str).str.strip().str.upper()
        .str.replace(".NS", "", regex=False).str.replace(".BO", "", regex=False)
        .str.replace("-EQ", "", regex=False).str.replace(".EQ", "", regex=False)
    )
    x["reference_date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = x.dropna(subset=["symbol", "reference_date", "close"])
    x = x[x["symbol"].ne("") & x["close"].gt(0)].copy()
    x["reference_date"] = pd.to_datetime(x["reference_date"]).dt.normalize()

    rows_before_cohort = len(x)
    cohort_start = pd.Timestamp(research_start_date).normalize() if research_start_date is not None else None
    if cohort_start is not None:
        x = x.loc[x["reference_date"] >= cohort_start].copy()
        if x.empty:
            raise ValueError(f"No PIT rows remain on or after research start date {cohort_start.date()}.")

    rows_excluded_before_research_cohort = rows_before_cohort - len(x)
    duplicate_rows = int(x.duplicated(["symbol", "reference_date"], keep=False).sum())
    if duplicate_rows:
        raise ValueError(
            f"PIT panel has {duplicate_rows} rows in duplicated (symbol, date) groups; refusing arbitrary deduplication."
        )

    sessions = sorted(
        pd.Timestamp(v).normalize()
        for v in (all_session_dates if all_session_dates is not None else x["reference_date"].unique())
    )
    if cohort_start is not None:
        sessions = [session for session in sessions if session >= cohort_start]
    successors, gaps = session_successors(sessions, holidays)
    future = x[["symbol", "reference_date", "close"]].rename(
        columns={"reference_date": "target_date", "close": "actual_close"}
    )
    x["target_date"] = x["reference_date"].map(successors)
    x = x.dropna(subset=["target_date"]).copy()
    x["target_date"] = pd.to_datetime(x["target_date"]).dt.normalize()
    rows_with_successor = len(x)
    labeled = x.merge(future, on=["symbol", "target_date"], how="left", validate="many_to_one")
    labeled["actual_close"] = pd.to_numeric(labeled["actual_close"], errors="coerce")
    labeled["y_return"] = (labeled["actual_close"] / labeled["close"] - 1.0).replace([np.inf, -np.inf], np.nan)
    labeled = labeled.dropna(subset=["actual_close", "y_return"])
    labeled = labeled[(labeled["actual_close"] > 0) & (labeled["target_date"] > labeled["reference_date"])].copy()
    labeled["y_up"] = (labeled["y_return"] > 0).astype("int8")

    ref_local = pd.to_datetime(labeled["reference_date"].dt.strftime("%Y-%m-%d") + " 15:30:00")
    labeled["_reference_close_utc"] = ref_local.dt.tz_localize(IST).dt.tz_convert("UTC")
    target_local = pd.to_datetime(labeled["target_date"].dt.strftime("%Y-%m-%d") + " 15:30:00")
    labeled["_target_timestamp"] = target_local.dt.tz_localize(IST).dt.tz_convert("UTC")
    if "decision_timestamp" in labeled:
        labeled["_decision_timestamp"] = pd.to_datetime(labeled["decision_timestamp"], errors="coerce", utc=True)
    else:
        labeled["_decision_timestamp"] = labeled["_reference_close_utc"]
    invalid_clock = (
        labeled["_decision_timestamp"].isna()
        | (labeled["_decision_timestamp"] > labeled["_reference_close_utc"])
        | (labeled["_decision_timestamp"] >= labeled["_target_timestamp"])
    )
    invalid_clock_rows = int(invalid_clock.sum())
    labeled = labeled.loc[~invalid_clock].copy()
    if labeled.empty:
        raise ValueError("No rows remain after exact-session target and PIT clock checks.")

    matrix = labeled[FEATURES_59].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    labeled["feature_coverage"] = np.isfinite(matrix.to_numpy(dtype=float, na_value=np.nan)).mean(axis=1)
    if "production_data_quality_gate" in labeled:
        labeled["quality_gate"] = labeled["production_data_quality_gate"].map(safe_gate)
        gate_source = "production_data_quality_gate"
    elif set(READINESS_COLUMNS).issubset(labeled.columns):
        labeled["quality_gate"] = labeled[list(READINESS_COLUMNS)].apply(
            quality_gate_from_readiness_flags, axis=1
        )
        gate_source = "reconstructed_from_pit_readiness_flags"
    else:
        labeled["quality_gate"] = False
        gate_source = "MISSING_GATE_AND_READINESS_FLAGS_BLOCK_ACTIONS"
    labeled["eligible_for_action"] = labeled["quality_gate"] & (labeled["feature_coverage"] >= COVERAGE_FLOOR)

    feature_groups = {
        "price_volume_technical": [name for name in FEATURES_59 if not name.startswith(("market_", "relative_", "delivery_", "news_", "event_", "fno_", "fundamental_", "fund_"))],
        "market_relative": [name for name in FEATURES_59 if name.startswith(("market_", "relative_"))],
        "delivery": [name for name in FEATURES_59 if name.startswith("delivery_")],
        "news_events": [name for name in FEATURES_59 if name.startswith(("news_", "event_"))],
        "fno": [name for name in FEATURES_59 if name.startswith("fno_")],
        "fundamentals": [name for name in FEATURES_59 if name.startswith(("fundamental_", "fund_"))],
    }
    feature_availability = {
        name: round(float(matrix[name].notna().mean() * 100), 3)
        for name in FEATURES_59
    }
    feature_group_availability = {
        group: round(float(matrix[columns].notna().to_numpy().mean() * 100), 3) if columns else None
        for group, columns in feature_groups.items()
    }
    feature_coverage_by_year = []
    for year, group in labeled.groupby(labeled["reference_date"].dt.year, sort=True):
        feature_coverage_by_year.append({
            "year": int(year),
            "rows": int(len(group)),
            "median_feature_coverage_pct": round(float(group["feature_coverage"].median() * 100), 3),
            "rows_ge_70pct_coverage_pct": round(float((group["feature_coverage"] >= COVERAGE_FLOOR).mean() * 100), 3),
            "quality_gate_pass_pct": round(float(group["quality_gate"].mean() * 100), 3),
            "joint_action_eligibility_pct": round(float(group["eligible_for_action"].mean() * 100), 3),
        })

    stats = {
        "input_rows": int(input_rows),
        "research_cohort_start_date": cohort_start.date().isoformat() if cohort_start is not None else None,
        "rows_excluded_before_research_cohort": int(rows_excluded_before_research_cohort),
        "readiness_columns_present": [name for name in READINESS_COLUMNS if name in labeled.columns],
        "clean_symbol_date_rows": int(len(x)),
        "labeled_rows": int(len(labeled)),
        "symbols": int(labeled["symbol"].nunique()),
        "observed_sessions": int(len(sessions)),
        "first_observed_session": sessions[0].date().isoformat(),
        "last_observed_session": sessions[-1].date().isoformat(),
        "duplicate_symbol_session_rows": duplicate_rows,
        "missing_expected_session_gaps": int(len(gaps)),
        "session_gap_examples": gaps[:20],
        "rows_without_exact_next_session_price_or_invalid_clock": int(max(0, rows_with_successor - len(labeled))),
        "rows_dropped_for_invalid_decision_clock": invalid_clock_rows,
        "quality_gate_source": gate_source,
        "quality_gate_column_present": "production_data_quality_gate" in labeled.columns,
        "quality_gate_reconstructed_from_readiness": gate_source == "reconstructed_from_pit_readiness_flags",
        "quality_gate_pass_rows": int(labeled["quality_gate"].sum()),
        "quality_gate_fail_or_missing_rows": int((~labeled["quality_gate"]).sum()),
        "quality_gate_pass_pct": round(float(labeled["quality_gate"].mean() * 100), 3),
        "rows_eligible_for_action": int(labeled["eligible_for_action"].sum()),
        "feature_availability_by_feature_pct": feature_availability,
        "feature_group_availability_pct": feature_group_availability,
        "feature_coverage_by_reference_year": feature_coverage_by_year,
        "feature_coverage_ge_70_pct": round(float((labeled["feature_coverage"] >= COVERAGE_FLOOR).mean() * 100), 3),
        "joint_action_eligibility_pct": round(float(labeled["eligible_for_action"].mean() * 100), 3),
        "feature_coverage_median_pct": round(float(labeled["feature_coverage"].median() * 100), 3),
        "feature_coverage_p10_pct": round(float(labeled["feature_coverage"].quantile(0.10) * 100), 3),
        "raw_abs_return_ge_20pct_rows": int(labeled["y_return"].abs().ge(0.20).sum()),
        "return_definition": "raw close(next exact observed NSE session)/close(reference session)-1; large corporate-action moves are flagged, not silently adjusted",
    }
    return labeled.sort_values(["reference_date", "symbol"]).reset_index(drop=True), stats


def clean_matrix(frame: pd.DataFrame, features: list[str]):
    x = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    upper = float(np.finfo(np.float32).max)
    for col in x.columns:
        x[col] = x[col].mask(~np.isfinite(x[col]) | (x[col].abs() > upper), np.nan)
    return x.astype("float32")


def deterministic_training_sample(train: pd.DataFrame, max_rows: int = MAX_FIT_ROWS) -> pd.DataFrame:
    """Bound fit time while retaining training rows from across every reference session."""
    if len(train) <= max_rows:
        return train.copy()
    if "reference_date" not in train.columns:
        return train.sample(n=max_rows, random_state=42).copy()
    dates = sorted(pd.to_datetime(train["reference_date"]).dt.normalize().unique())
    per_date = max(1, max_rows // max(1, len(dates)))
    parts = []
    normalized_dates = pd.to_datetime(train["reference_date"]).dt.normalize()
    for day in dates:
        group = train.loc[normalized_dates.eq(pd.Timestamp(day))]
        if len(group) > per_date:
            seed = int(pd.Timestamp(day).strftime("%Y%m%d")) % (2**32 - 1)
            group = group.sample(n=per_date, random_state=seed)
        parts.append(group)
    sampled = pd.concat(parts, ignore_index=False)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=42)
    return sampled.sort_values(["reference_date", "symbol"]).reset_index(drop=True)


def fit_production_family(train: pd.DataFrame, features: list[str]):
    """Fit a time-bounded, in-memory sample of the production model family."""
    input_rows = int(len(train))
    if input_rows < 1000:
        raise ValueError(f"Too few input fit rows: {input_rows}")
    train = deterministic_training_sample(train)
    fit_rows = int(len(train))
    print(f"MODEL FIT: input_rows={input_rows:,}; sampled_fit_rows={fit_rows:,}; features={len(features)}", flush=True)
    y_return = pd.to_numeric(train["y_return"], errors="coerce")
    valid = y_return.notna()
    train = train.loc[valid].copy()
    y_return = y_return.loc[valid]
    y_up = (y_return > 0).astype("int8")
    if y_up.nunique() < 2:
        raise ValueError("Training fold contains only one direction class.")
    X = clean_matrix(train, features)
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(C=0.5, class_weight="balanced", solver="lbfgs", max_iter=2500, random_state=42)),
    ])
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", HistGradientBoostingRegressor(max_iter=AUDIT_HGB_MAX_ITER, learning_rate=0.04, max_leaf_nodes=19, l2_regularization=2.0, random_state=42)),
    ])
    clf.fit(X, y_up)
    reg.fit(X, y_return.astype(float))
    fit_stats = {
        "input_rows": input_rows,
        "sampled_fit_rows": fit_rows,
        "sampling_policy": "deterministic per-reference-session sampling, fixed seed, max 25000 rows per fit",
        "regressor_max_iter": AUDIT_HGB_MAX_ITER,
    }
    return (clf, reg), fit_stats


def predict_rows(pair, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    clf, reg = pair
    X = clean_matrix(frame, features)
    return pd.DataFrame({
        "reference_date": frame["reference_date"].to_numpy(),
        "target_date": frame["target_date"].to_numpy(),
        "symbol": frame["symbol"].astype(str).to_numpy(),
        "reference_close": pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
        "actual_close": pd.to_numeric(frame["actual_close"], errors="coerce").to_numpy(),
        "actual_return": pd.to_numeric(frame["y_return"], errors="coerce").to_numpy(),
        "actual_up": pd.to_numeric(frame["y_up"], errors="coerce").to_numpy(dtype=int),
        "probability_up": clf.predict_proba(X)[:, 1],
        "predicted_return": reg.predict(X),
        "feature_coverage": pd.to_numeric(frame["feature_coverage"], errors="coerce").to_numpy(),
        "quality_gate": frame["quality_gate"].astype(bool).to_numpy(),
        "eligible_for_action": frame["eligible_for_action"].astype(bool).to_numpy(),
    })


def policy_signals(pred, buy, sell, min_return, abstain_all=False):
    p = pd.to_numeric(pred["probability_up"], errors="coerce").to_numpy()
    r = pd.to_numeric(pred["predicted_return"], errors="coerce").to_numpy()
    eligible = pred["eligible_for_action"].astype(bool).to_numpy()
    signal = np.full(len(pred), "ABSTAIN", dtype=object)
    if abstain_all:
        return signal
    signal[eligible] = "HOLD"
    signal[eligible & (p >= buy) & (r > min_return)] = "BUY"
    signal[eligible & (p <= sell) & (r < -min_return)] = "SELL"
    return signal


def date_block_ci(pred, signals, n_boot=500, seed=42):
    x = pred.copy()
    x["signal"] = signals
    x = x[x["signal"].isin(["BUY", "SELL"])].copy()
    if x.empty:
        return {"dates_with_actions": 0, "daily_precision_mean": None, "ci95": [None, None]}
    x["correct"] = (
        (x["signal"].eq("BUY") & pd.to_numeric(x["actual_return"], errors="coerce").gt(0))
        | (x["signal"].eq("SELL") & pd.to_numeric(x["actual_return"], errors="coerce").lt(0))
    ).astype(float)
    daily = x.groupby("target_date", sort=True)["correct"].mean().to_numpy(dtype=float)
    if len(daily) == 1:
        return {"dates_with_actions": 1, "daily_precision_mean": float(daily.mean()), "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(daily, size=len(daily), replace=True).mean() for _ in range(n_boot)])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"dates_with_actions": int(len(daily)), "daily_precision_mean": float(daily.mean()), "ci95": [float(low), float(high)]}


def evaluate_policy(pred, policy, n_boot=1000):
    signals = policy_signals(
        pred, float(policy.get("buy_probability_threshold", 0.60)),
        float(policy.get("sell_probability_threshold", 0.40)),
        float(policy.get("min_expected_return", 0.0015)), bool(policy.get("abstain_all", False)),
    )
    x = pred.copy()
    x["signal"] = signals
    y = pd.to_numeric(x["actual_up"], errors="coerce").to_numpy(dtype=int)
    p = pd.to_numeric(x["probability_up"], errors="coerce").to_numpy(dtype=float)
    r_pred = pd.to_numeric(x["predicted_return"], errors="coerce").to_numpy(dtype=float)
    actual = pd.to_numeric(x["actual_return"], errors="coerce").to_numpy(dtype=float)
    directional = (p >= 0.5).astype(int)
    actions = x["signal"].isin(["BUY", "SELL"]).to_numpy()
    eligible = x["eligible_for_action"].astype(bool).to_numpy()
    correct = ((x["signal"].eq("BUY") & (actual > 0)) | (x["signal"].eq("SELL") & (actual < 0))).to_numpy()
    signed = np.where(x["signal"].eq("BUY"), actual, np.where(x["signal"].eq("SELL"), -actual, np.nan))
    if len(y) and np.unique(y).size > 1:
        auc = float(roc_auc_score(y, p))
        balanced = float(balanced_accuracy_score(y, directional))
    else:
        auc, balanced = None, None
    cal = []
    for i in range(10):
        low, high = i / 10, (i + 1) / 10
        mask = (p >= low) & ((p < high) if i < 9 else (p <= high))
        if mask.any():
            cal.append({"bin": f"{low:.1f}-{high:.1f}", "count": int(mask.sum()),
                        "mean_probability": float(p[mask].mean()), "observed_up_rate": float(y[mask].mean())})
    n_actions = int(actions.sum())
    ci = date_block_ci(x, signals, n_boot)
    signed_action = signed[actions]
    net = signed_action - ROUND_TRIP_COST if n_actions else np.array([], dtype=float)
    return {
        "rows_scored": int(len(x)), "reference_sessions": int(x["reference_date"].nunique()),
        "target_sessions": int(x["target_date"].nunique()), "symbols": int(x["symbol"].nunique()),
        "positive_return_rate": float(y.mean()) if len(y) else None,
        "directional_accuracy_p50": float(np.mean(directional == y)) if len(y) else None,
        "balanced_accuracy_p50": balanced, "roc_auc": auc,
        "brier_score": float(brier_score_loss(y, p)) if len(y) else None,
        "majority_class_baseline_accuracy": float(max(y.mean(), 1 - y.mean())) if len(y) else None,
        "predicted_return_mae_percentage_points": float(np.mean(np.abs(r_pred - actual)) * 100) if len(y) else None,
        "predicted_return_bias_percentage_points": float(np.mean(r_pred - actual) * 100) if len(y) else None,
        "calibration_bins": cal, "action_count": n_actions, "action_dates": ci["dates_with_actions"],
        "action_precision": float(correct[actions].mean()) if n_actions else None,
        "action_precision_date_block_bootstrap_95pct_ci": ci["ci95"],
        "date_equal_weighted_action_precision": ci["daily_precision_mean"],
        "action_coverage_pct_all_rows": float(n_actions / len(x) * 100) if len(x) else 0.0,
        "action_coverage_pct_eligible_rows": float(n_actions / eligible.sum() * 100) if eligible.any() else 0.0,
        "eligible_row_count": int(eligible.sum()), "eligible_row_pct": float(eligible.mean() * 100) if len(x) else 0.0,
        "buy_count": int(x["signal"].eq("BUY").sum()), "sell_count": int(x["signal"].eq("SELL").sum()),
        "hold_count": int(x["signal"].eq("HOLD").sum()), "abstain_count": int(x["signal"].eq("ABSTAIN").sum()),
        "mean_gross_signed_return_pct_per_action": float(signed_action.mean() * 100) if n_actions else None,
        "mean_net_signed_return_pct_per_action_after_assumed_30bps_cost": float(net.mean() * 100) if n_actions else None,
        "assumed_round_trip_cost_bps": 30, "policy": policy,
        "action_target_precision_met_on_holdout": bool(
            n_actions >= MIN_ACTIONS and ci["dates_with_actions"] >= MIN_ACTION_DATES
            and ci["ci95"][0] is not None and ci["ci95"][0] >= TARGET_PRECISION
        ),
    }


def tune_policy_on_oof(oof):
    """Threshold search uses only purged OOF predictions inside the development window."""
    qualified = oof.loc[oof["eligible_for_action"].astype(bool)].copy()
    if qualified.empty:
        policy = {"abstain_all": True, "reason": "NO_OOF_ROWS_PASSING_QUALITY_AND_70PCT_COVERAGE_GATES", "live_use_approved": False}
        return policy, {"candidate_count": 0, "qualified_oof_rows": 0, "best_candidate": None}
    actual = pd.to_numeric(qualified["actual_return"], errors="coerce").to_numpy(dtype=float)
    candidates = []
    for buy in (0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80):
        for sell in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            for min_ret in (0.0015, 0.003):
                policy = {"abstain_all": False, "buy_probability_threshold": float(buy),
                          "sell_probability_threshold": float(sell), "min_expected_return": float(min_ret),
                          "selection_source": "purged expanding-window OOF predictions before final holdout"}
                signals = policy_signals(qualified, buy, sell, min_ret)
                action = np.isin(signals, ["BUY", "SELL"])
                n = int(action.sum())
                n_dates = int(qualified.loc[action, "target_date"].nunique()) if n else 0
                if n < MIN_ACTIONS or n_dates < MIN_ACTION_DATES:
                    continue
                correct = ((signals == "BUY") & (actual > 0)) | ((signals == "SELL") & (actual < 0))
                daily_df = pd.DataFrame({"target_date": qualified.loc[action, "target_date"].to_numpy(),
                                         "correct": correct[action].astype(float)})
                daily = daily_df.groupby("target_date", sort=True)["correct"].mean().to_numpy(dtype=float)
                rng = np.random.default_rng(42)
                draws = np.array([rng.choice(daily, size=len(daily), replace=True).mean() for _ in range(300)])
                lo, hi = np.quantile(draws, [0.025, 0.975])
                candidates.append({
                    "policy": policy, "actions": n, "action_dates": n_dates,
                    "action_precision": float(correct[action].mean()),
                    "date_equal_weighted_precision": float(daily.mean()),
                    "date_block_bootstrap_95pct_ci": [float(lo), float(hi)],
                    "action_coverage_pct_eligible_oof": float(action.mean() * 100),
                    "target_precision_gate_pass": bool(lo >= TARGET_PRECISION),
                    "selection_score": float(lo),
                })
    candidates.sort(key=lambda v: (v["selection_score"], v["date_equal_weighted_precision"], v["actions"]), reverse=True)
    if not candidates:
        policy = {"abstain_all": True, "reason": "NO_OOF_POLICY_MET_MINIMUM_ACTION_COUNTS", "live_use_approved": False}
        return policy, {"candidate_count": 0, "qualified_oof_rows": int(len(qualified)), "best_candidate": None}
    best = candidates[0]
    policy = dict(best["policy"])
    policy.update({
        "oof_action_precision": best["action_precision"],
        "oof_date_equal_weighted_precision": best["date_equal_weighted_precision"],
        "oof_date_block_bootstrap_95pct_ci": best["date_block_bootstrap_95pct_ci"],
        "oof_actions": best["actions"], "oof_action_dates": best["action_dates"],
        "target_precision_gate_pass": best["target_precision_gate_pass"],
        "live_use_approved": best["target_precision_gate_pass"],
        "approval_reason": "OOF_DATE_BLOCK_BOOTSTRAP_LOWER_BOUND_AT_LEAST_80PCT" if best["target_precision_gate_pass"] else "OOF_DATE_BLOCK_BOOTSTRAP_LOWER_BOUND_BELOW_80PCT",
    })
    return policy, {"candidate_count": len(candidates), "qualified_oof_rows": int(len(qualified)),
                    "target_precision": TARGET_PRECISION, "best_candidate": best, "top_candidates": candidates[:10]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_drive_file(remote_path: str, local_path: Path):
    """Download the one requested PIT panel without writing to Google Drive."""
    from googleapiclient.http import MediaIoBaseDownload
    from persistent_store import GoogleDriveStore
    store = GoogleDriveStore()
    parent = store.root_id
    item = None
    for part in [p for p in remote_path.strip("/").split("/") if p]:
        item = store._find_child(parent, part)
        if not item:
            raise FileNotFoundError(f"Google Drive input missing: {remote_path}; missing component {part!r}")
        parent = item["id"]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp = local_path.with_suffix(local_path.suffix + ".tmp")
    with temp.open("wb") as fh:
        dl = MediaIoBaseDownload(fh, store.service.files().get_media(fileId=item["id"]))
        done = False
        while not done:
            _, done = dl.next_chunk()
    temp.replace(local_path)
    return {"remote_path": remote_path, "drive_file_id": str(item.get("id")),
            "drive_file_name": str(item.get("name")), "drive_modified_time": item.get("modifiedTime"),
            "size_bytes": int(local_path.stat().st_size), "sha256": sha256_file(local_path), "local_cache_reused": False}


def render_summary(report):
    fixed = report["untouched_holdout"]["fixed_policy"]
    tuned = report["untouched_holdout"]["oof_selected_candidate_policy"]
    cert = report["strict_oos_certification"]
    lines = [
        "# Strict Delivery Walk-Forward Audit", "",
        f"- Generated UTC: {report['generated_at_utc']}",
        f"- Input SHA-256: {report['input_panel_sha256']}",
        f"- Data range: {report['data']['first_observed_session']} to {report['data']['last_observed_session']}",
        f"- Labeled rows: {report['data']['labeled_rows']:,}; sampled symbols: {report['data']['symbols']:,}",
        f"- Source panel rows/symbols: {report['panel_sampling']['full_panel_rows']:,} / {report['panel_sampling']['full_panel_symbols']:,}",
        f"- Stable symbol sample: {report['panel_sampling']['sampled_symbol_count']:,} ({report['panel_sampling']['sample_fraction_of_symbols_pct']}% of source symbols)",
        f"- Development sessions: {report['split']['development_sessions']}; holdout sessions: {report['split']['holdout_sessions']}",
        f"- Holdout cutoff: {report['split']['holdout_start_reference_date']}",
        f"- Training label purge passed: {report['split']['training_targets_strictly_before_holdout']}",
        f"- Holdout outcomes used in threshold selection: {report['split']['holdout_targets_used_for_selection_or_tuning']}",
        "", "## Untouched chronological holdout", "", "### Fixed existing threshold policy",
        f"- Directional accuracy: {fixed['directional_accuracy_p50']}",
        f"- Balanced accuracy: {fixed['balanced_accuracy_p50']}",
        f"- ROC-AUC: {fixed['roc_auc']}", f"- Brier score: {fixed['brier_score']}",
        f"- Action precision: {fixed['action_precision']}",
        f"- Action coverage: {fixed['action_coverage_pct_all_rows']:.3f}%",
        f"- Date-block 95% action-precision CI: {fixed['action_precision_date_block_bootstrap_95pct_ci']}",
        f"- Mean net signed return after assumed 30 bps: {fixed['mean_net_signed_return_pct_per_action_after_assumed_30bps_cost']}",
        "", "### Candidate policy selected on development OOF only",
        f"- Policy: {json.dumps(tuned['policy'], sort_keys=True)}",
        f"- Directional accuracy: {tuned['directional_accuracy_p50']}",
        f"- Balanced accuracy: {tuned['balanced_accuracy_p50']}",
        f"- ROC-AUC: {tuned['roc_auc']}", f"- Action precision: {tuned['action_precision']}",
        f"- Action coverage: {tuned['action_coverage_pct_all_rows']:.3f}%",
        f"- Date-block 95% action-precision CI: {tuned['action_precision_date_block_bootstrap_95pct_ci']}",
        f"- OOF 80% lower-bound gate passed: {report['interpretation']['oof_policy_live_use_approved']}",
        "", "## Certification limitations",
        f"- Temporal purged holdout executed: {cert['chronological_purged_holdout_executed']}",
        f"- Tuning separated from holdout: {cert['model_selection_and_threshold_tuning_separate_from_holdout']}",
        f"- Fully PIT-universe certified: {cert['certified_as_fully_unbiased_point_in_time_universe_oos']}",
        cert["reason"],
        f"- Research panel last observed session: {report['source_freshness']['panel_last_session']} (historical research panel, not the live production snapshot).",
        f"- Live production inventory status: {report.get('production_inventory', {}).get('status')}",
        f"- Production target session: {report.get('production_inventory', {}).get('target_session')}",
        f"- Latest persisted production PIT snapshot: {report.get('production_inventory', {}).get('latest_production_pit_snapshot_date')}",
        f"- Target EOD/Delivery/PIT presence: {report.get('production_inventory', {}).get('target_eod_archive_present')} / {report.get('production_inventory', {}).get('target_delivery_archive_present')} / {report.get('production_inventory', {}).get('target_pit_snapshot_present')}",
        "", "Weights were not saved or promoted; no production registry, policy, prediction ledger, or Drive source was changed.",
    ]
    return "\n".join(lines) + "\n"


def run_audit(
    panel_path: Path,
    out: Path,
    research_start_date: date = DEFAULT_RESEARCH_START_DATE,
):
    print(
        f"SELECT SYMBOL SAMPLE: path={panel_path}; bytes={panel_path.stat().st_size:,}; "
        f"research_start_date={research_start_date.isoformat()}",
        flush=True,
    )
    panel, panel_sample, all_session_dates = load_panel_symbol_sample(panel_path, symbol_limit=500)
    print(
        f"PANEL SAMPLE READY: full_rows={panel_sample['full_panel_rows']:,}; "
        f"full_symbols={panel_sample['full_panel_symbols']:,}; sampled_symbols={panel_sample['sampled_symbol_count']:,}; "
        f"sampled_rows={panel_sample['sampled_panel_rows']:,}",
        flush=True,
    )
    labeled, stats = prepare_labeled_panel(
        panel, parse_holidays(), all_session_dates=all_session_dates,
        research_start_date=research_start_date,
    )
    del panel
    print(
        f"LABEL BUILD COMPLETE: labeled_rows={len(labeled):,}; "
        f"sampled_symbols={labeled['symbol'].nunique():,}; sessions={labeled['reference_date'].nunique():,}",
        flush=True,
    )
    sessions = sorted(pd.Timestamp(v).normalize() for v in labeled["reference_date"].unique())
    if len(sessions) < 100:
        raise ValueError(f"Need at least 100 labeled sessions; found {len(sessions)}.")
    split_idx = min(max(int(math.floor(len(sessions) * (1 - HOLDOUT_FRACTION))), MIN_TRAIN_SESSIONS + 5), len(sessions) - 20)
    holdout_start = sessions[split_idx]
    dev = labeled[(labeled["reference_date"] < holdout_start) & (labeled["target_date"] < holdout_start)].copy()
    hold = labeled[labeled["reference_date"] >= holdout_start].copy()
    dev_sessions = sorted(pd.Timestamp(v).normalize() for v in dev["reference_date"].unique())
    hold_sessions = sorted(pd.Timestamp(v).normalize() for v in hold["reference_date"].unique())
    if len(dev_sessions) < MIN_TRAIN_SESSIONS + OOF_FOLDS * OOF_DAYS_PER_FOLD or hold.empty:
        raise ValueError(f"Insufficient after split: dev sessions={len(dev_sessions)}, holdout rows={len(hold)}.")
    if dev["target_date"].max() >= holdout_start:
        raise AssertionError("Training target reaches holdout cutoff.")
    if hold["reference_date"].min() < holdout_start:
        raise AssertionError("Holdout decision precedes cutoff.")

    fold_dates = dev_sessions[-(OOF_FOLDS * OOF_DAYS_PER_FOLD):]
    oof_parts, folds = [], []
    for k in range(OOF_FOLDS):
        dates = fold_dates[k * OOF_DAYS_PER_FOLD:(k + 1) * OOF_DAYS_PER_FOLD]
        start = dates[0]
        tr = dev[(dev["reference_date"] < start) & (dev["target_date"] < start)].copy()
        va = dev[dev["reference_date"].isin(dates) & (dev["target_date"] < holdout_start)].copy()
        if len(tr) < 1000 or va.empty:
            raise ValueError(f"OOF fold {k + 1} insufficient: train={len(tr)}, test={len(va)}.")
        if tr["target_date"].max() >= start:
            raise AssertionError(f"OOF fold {k + 1} has target leakage.")
        print(f"OOF FOLD {k + 1}/{OOF_FOLDS} START: train_rows={len(tr):,}; validation_rows={len(va):,}; cutoff={start.date()}", flush=True)
        pair, fit_stats = fit_production_family(tr, list(FEATURES_59))
        predictions = predict_rows(pair, va, list(FEATURES_59))
        print(f"OOF FOLD {k + 1}/{OOF_FOLDS} COMPLETE: predictions={len(predictions):,}; sampled_fit_rows={fit_stats['sampled_fit_rows']:,}", flush=True)
        predictions["fold"] = k + 1
        oof_parts.append(predictions)
        folds.append({
            "fold": k + 1, "train_rows": int(len(tr)), "validation_rows": int(len(va)),
            "train_start": tr["reference_date"].min().date().isoformat(),
            "train_end": tr["reference_date"].max().date().isoformat(),
            "latest_train_target": tr["target_date"].max().date().isoformat(),
            "validation_start": va["reference_date"].min().date().isoformat(),
            "validation_end": va["reference_date"].max().date().isoformat(),
            "purge_check": "PASS",
            "fit_sampling": fit_stats,
        })
        del pair
    oof = pd.concat(oof_parts, ignore_index=True)
    policy, tuning = tune_policy_on_oof(oof)

    print(f"FINAL DEVELOPMENT FIT START: train_rows={len(dev):,}; holdout_rows={len(hold):,}; cutoff={holdout_start.date()}", flush=True)
    final_pair, final_fit_stats = fit_production_family(dev, list(FEATURES_59))
    hold_pred = predict_rows(final_pair, hold, list(FEATURES_59))
    del final_pair
    print(f"FINAL HOLDOUT SCORING COMPLETE: rows={len(hold_pred):,}", flush=True)

    fixed = {"abstain_all": False, "buy_probability_threshold": 0.60, "sell_probability_threshold": 0.40,
             "min_expected_return": 0.0015, "policy_source": "fixed existing policy; not tuned on holdout"}
    metrics_fixed = evaluate_policy(hold_pred, fixed)
    metrics_tuned = evaluate_policy(hold_pred, policy)
    metrics_oof_fixed = evaluate_policy(oof, fixed, n_boot=300)
    metrics_oof_tuned = evaluate_policy(oof, policy, n_boot=300)

    last_date = pd.Timestamp(stats["last_observed_session"]).date()
    now = pd.Timestamp.now(tz=IST)
    try:
        from market_session_guard import latest_completed_nse_session
        expected_latest = latest_completed_nse_session(now.to_pydatetime())
    except Exception:
        expected_latest = None

    report = {
        "audit_schema_version": "1.0", "status": "COMPLETE",
        "audit_type": "read-only expanding-window OOF plus untouched chronological holdout",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": os.getenv("GITHUB_SHA"),
        "input_panel_sha256": sha256_file(panel_path),
        "model_weights_saved": False, "model_registry_changed": False,
        "prediction_ledger_changed": False, "signal_policy_changed": False,
        "model_family": {
            "classifier": "median imputation + missing indicators + scaled LogisticRegression(C=0.5, class_weight=balanced)",
            "regressor": f"median imputation + missing indicators + HistGradientBoostingRegressor({AUDIT_HGB_MAX_ITER} iter, learning_rate=0.04, max_leaf_nodes=19, L2=2.0)",
            "features": list(FEATURES_59), "target": "positive raw return to next exact observed NSE session",
            "audit_fit_is_sampled_proxy_not_exact_production_artifact": True,
            "sample_cap_per_fit": MAX_FIT_ROWS,
            "reason": "Runtime-bounded research refit; scores describe this sampled model-family walk-forward, not the exact locked production weights.",
        },
        "fit_policy": {
            "max_rows_per_fit": MAX_FIT_ROWS,
            "hgb_max_iter": AUDIT_HGB_MAX_ITER,
            "sampling": "fixed-seed per-reference-session sampling",
        },
        "data": stats,
        "panel_sampling": panel_sample,
        "split": {
            "method": "chronological 80/20 session split, with target-date purge at cutoff, within the fixed feature-availability cohort",
            "research_cohort_start_date": stats.get("research_cohort_start_date"),
            "holdout_start_reference_date": holdout_start.date().isoformat(),
            "development_first_reference_date": dev_sessions[0].date().isoformat(),
            "development_last_reference_date": dev_sessions[-1].date().isoformat(),
            "holdout_first_reference_date": hold_sessions[0].date().isoformat(),
            "holdout_last_reference_date": hold_sessions[-1].date().isoformat(),
            "development_rows": int(len(dev)), "holdout_rows": int(len(hold)),
            "development_sessions": int(len(dev_sessions)), "holdout_sessions": int(len(hold_sessions)),
            "latest_development_target_date": dev["target_date"].max().date().isoformat(),
            "training_targets_strictly_before_holdout": bool(dev["target_date"].max() < holdout_start),
            "holdout_targets_used_for_selection_or_tuning": 0,
            "holdout_touched_before_final_evaluation": False,
        },
        "oof_walk_forward_folds": folds,
        "oof_fixed_policy_metrics": metrics_oof_fixed,
        "oof_tuned_policy_metrics": metrics_oof_tuned,
        "threshold_tuning": tuning,
        "untouched_holdout": {"fixed_policy": metrics_fixed, "oof_selected_candidate_policy": metrics_tuned},
        "source_freshness": {
            "panel_last_session": last_date.isoformat(),
            "latest_completed_session_expected": expected_latest.isoformat() if hasattr(expected_latest, "isoformat") else str(expected_latest) if expected_latest is not None else None,
            "panel_matches_latest_completed_session": None,
            "interpretation": (
                "The audited file is an aggregate historical research/training panel, not the dated live production PIT snapshot. "
                "Its end date must not be used as a production-ingestion freshness verdict; see production_inventory."
            ),
            "research_panel_is_live_production_snapshot": False,
        },
        "production_inventory": audit_live_production_inventory(now.to_pydatetime()),
        "strict_oos_certification": {
            "chronological_purged_holdout_executed": True,
            "model_selection_and_threshold_tuning_separate_from_holdout": True,
            "aggregate_panel_decision_clock_checked": True,
            "historical_universe_membership_pit_verified": False,
            "all_optional_source_availability_reconstructed_from_raw_logs": False,
            "certified_as_fully_unbiased_point_in_time_universe_oos": False,
            "reason": (
                "Temporal holdout is leakage-controlled, but this aggregate panel does not independently prove "
                "historical exchange-universe membership or reconstruct every optional-source availability join. "
                "Treat scores as a chronological research holdout on the persisted PIT panel, not fully certified "
                "point-in-time-universe performance."
            ),
        },
        "interpretation": {
            "target_action_precision": TARGET_PRECISION,
            "oof_policy_live_use_approved": bool(policy.get("live_use_approved", False)),
            "no_80_to_90_percent_accuracy_claim_without_holdout_evidence": True,
            "assumed_round_trip_cost_bps": 30,
            "execution_costs_not_fully_simulated": True,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "strict_delivery_walkforward_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    oof.to_csv(out / "strict_delivery_walkforward_oof_predictions.csv", index=False)
    hold_csv = hold_pred.copy()
    hold_csv["signal_fixed_policy"] = policy_signals(hold_csv, fixed["buy_probability_threshold"], fixed["sell_probability_threshold"], fixed["min_expected_return"])
    hold_csv["signal_oof_selected_policy"] = policy_signals(
        hold_csv, policy.get("buy_probability_threshold", 0.60), policy.get("sell_probability_threshold", 0.40),
        policy.get("min_expected_return", 0.0015), bool(policy.get("abstain_all", False)),
    )
    hold_csv.to_csv(out / "strict_delivery_walkforward_holdout_predictions.csv", index=False)
    (out / "strict_delivery_walkforward_summary.md").write_text(render_summary(report), encoding="utf-8")
    return report


def render_summary(report):
    fixed = report["untouched_holdout"]["fixed_policy"]
    tuned = report["untouched_holdout"]["oof_selected_candidate_policy"]
    cert = report["strict_oos_certification"]
    feature_availability = report["data"].get("feature_availability_by_feature_pct", {})
    lowest_coverage = sorted(feature_availability.items(), key=lambda item: item[1])[:10]
    lowest_coverage_text = ", ".join(f"{name}={coverage:.1f}%" for name, coverage in lowest_coverage)
    return "\n".join([
        "# Strict Delivery Walk-Forward Audit", "",
        f"- Data range: {report['data']['first_observed_session']} to {report['data']['last_observed_session']}",
        f"- Labeled rows: {report['data']['labeled_rows']:,}; symbols: {report['data']['symbols']:,}",
        f"- Development sessions: {report['split']['development_sessions']}; holdout sessions: {report['split']['holdout_sessions']}",
        f"- Holdout cutoff: {report['split']['holdout_start_reference_date']}",
        f"- Training label purge passed: {report['split']['training_targets_strictly_before_holdout']}",
        f"- Holdout labels used in threshold tuning: {report['split']['holdout_targets_used_for_selection_or_tuning']}",
        f"- Quality-gate column present: {report['data'].get('quality_gate_column_present')}; pass rate: {report['data']['quality_gate_pass_pct']}%",
        f"- Rows eligible for actions: {report['data'].get('rows_eligible_for_action', 0):,}",
        f"- Rows with >=70% feature coverage: {report['data']['feature_coverage_ge_70_pct']}%",
        f"- Lowest-coverage features: {lowest_coverage_text}",
        f"- Feature-group availability: {json.dumps(report['data'].get('feature_group_availability_pct', {}), sort_keys=True)}",
        "", "## Fixed-policy holdout",
        f"- Directional accuracy: {fixed['directional_accuracy_p50']}",
        f"- Balanced accuracy: {fixed['balanced_accuracy_p50']}",
        f"- ROC-AUC: {fixed['roc_auc']}", f"- Brier score: {fixed['brier_score']}",
        f"- Action precision: {fixed['action_precision']}",
        f"- Action coverage: {fixed['action_coverage_pct_all_rows']:.3f}%",
        f"- Date-block 95% action-precision CI: {fixed['action_precision_date_block_bootstrap_95pct_ci']}",
        "", "## OOF-selected threshold candidate, scored once on holdout",
        f"- Policy: {json.dumps(tuned['policy'], sort_keys=True)}",
        f"- Directional accuracy: {tuned['directional_accuracy_p50']}",
        f"- Balanced accuracy: {tuned['balanced_accuracy_p50']}",
        f"- ROC-AUC: {tuned['roc_auc']}", f"- Action precision: {tuned['action_precision']}",
        f"- Action coverage: {tuned['action_coverage_pct_all_rows']:.3f}%",
        f"- Date-block 95% action-precision CI: {tuned['action_precision_date_block_bootstrap_95pct_ci']}",
        f"- OOF policy live-use approval gate passed: {report['interpretation']['oof_policy_live_use_approved']}",
        "", "## Limitations",
        f"- Temporal purged holdout executed: {cert['chronological_purged_holdout_executed']}",
        f"- Selection/tuning separated from holdout: {cert['model_selection_and_threshold_tuning_separate_from_holdout']}",
        f"- Fully PIT-universe certified: {cert['certified_as_fully_unbiased_point_in_time_universe_oos']}",
        cert["reason"],
        f"- Panel last session: {report['source_freshness']['panel_last_session']}; latest expected: {report['source_freshness']['latest_completed_session_expected']}",
        f"- Audit fit cap: {report['fit_policy']['max_rows_per_fit']} rows per fit; regressor iterations: {report['fit_policy']['hgb_max_iter']}",
        "", "This is a sampled model-family research holdout, not the exact active production weights. No production model, registry, policy, ledger, or Drive source was changed.",
    ]) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--remote-panel", default=DEFAULT_REMOTE_PANEL)
    ap.add_argument("--local-panel", default="/tmp/strict-delivery-audit/PIT_PANEL.parquet")
    ap.add_argument("--output-dir", default="strict-delivery-audit-out")
    ap.add_argument(
        "--research-start-date",
        default=DEFAULT_RESEARCH_START_DATE.isoformat(),
        help="First session in the fixed feature-availability research cohort (YYYY-MM-DD).",
    )
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel_path = Path(args.local_panel)
    try:
        if panel_path.exists() and panel_path.stat().st_size:
            provenance = {"remote_path": args.remote_panel, "local_cache_reused": True,
                          "size_bytes": int(panel_path.stat().st_size), "sha256": sha256_file(panel_path)}
            print(f"REUSING LOCAL PIT PANEL: bytes={provenance['size_bytes']:,}", flush=True)
        else:
            print(f"DOWNLOADING PIT PANEL FROM DRIVE: {args.remote_panel}", flush=True)
            provenance = download_drive_file(args.remote_panel, panel_path)
            print(f"PANEL DOWNLOAD COMPLETE: bytes={provenance['size_bytes']:,}; sha256={provenance['sha256']}", flush=True)
    except Exception as exc:
        blocked = {
            "audit_schema_version": "1.0", "status": "BLOCKED_INPUT_UNAVAILABLE",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_panel": args.remote_panel, "error": f"{type(exc).__name__}: {exc}",
            "production_artifacts_changed": False,
            "next_step": "Verify the PIT panel path and GitHub Actions Google Drive secrets. Historical prediction outputs are not substituted for a time-split model test.",
        }
        (out / "strict_delivery_walkforward_audit.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        (out / "strict_delivery_walkforward_summary.md").write_text(
            "# Strict Delivery Walk-Forward Audit\n\nRun blocked before model fitting because the PIT input panel could not be read.\n\n"
            f"- Input: {args.remote_panel}\n- Error: {blocked['error']}\n- No production artifact changed.\n", encoding="utf-8"
        )
        print(json.dumps(blocked, indent=2))
        return 2
    try:
        report = run_audit(panel_path, out, research_start_date=date.fromisoformat(args.research_start_date))
        report["input_provenance"] = provenance
        (out / "strict_delivery_walkforward_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({
            "status": report["status"], "data": report["data"], "split": report["split"],
            "untouched_holdout": report["untouched_holdout"],
            "strict_oos_certification": report["strict_oos_certification"],
            "source_freshness": report["source_freshness"],
            "production_inventory": report["production_inventory"],
        }, indent=2, default=str))
        return 0
    except Exception as exc:
        blocked = {
            "audit_schema_version": "1.0", "status": "BLOCKED_AUDIT_ERROR",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_provenance": provenance, "error": f"{type(exc).__name__}: {exc}",
            "production_artifacts_changed": False,
        }
        (out / "strict_delivery_walkforward_audit.json").write_text(json.dumps(blocked, indent=2), encoding="utf-8")
        (out / "strict_delivery_walkforward_summary.md").write_text(
            "# Strict Delivery Walk-Forward Audit\n\nAudit failed closed.\n\n"
            f"- Error: {blocked['error']}\n- No production model, policy, ledger, or source file changed.\n", encoding="utf-8"
        )
        print(json.dumps(blocked, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
