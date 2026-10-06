"""Colab trainer for the isolated Beta V2 volume-free 5-minute Intraday challenger.

The current production 5-minute model is untouched. This trainer uses only price/time
features that are available from the NSE public quote-chart stream, market-neutral
future-return labels, purged walk-forward folds, calibration and a per-horizon
champion/challenger registry.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from colab_build_all_nse_intraday_5m_multifactor import load_market, load_symbol
from delivery_training_universe import official_nse_symbols
from selfdriving.sd_intraday_v2 import (
    add_cross_sectional_targets,
    feature_columns,
    _norm,
    _session_rows,
)
from selfdriving.sd_registry import GateRules, Registry, gate
from selfdriving.sd_walkforward import (
    FoldSpec,
    ModelConfig,
    fit_bundle,
    evaluate,
    paired_logloss_gain,
    run_walkforward,
    choose_thresholds,
)


IST = "Asia/Kolkata"
HORIZONS = (5, 10, 15, 20, 25, 30)

# Beta V2 Intraday historical-data policy.
PREFERRED_SESSIONS = 750
MIN_PREFERRED_SESSIONS = 500
MIN_USABLE_SESSIONS = 200

# Each compact checkpoint has one row per decision timestamp and six future
# returns, rather than six duplicate feature rows. This preserves history while
# keeping Drive storage and per-horizon training RAM bounded.
DEFAULT_MAX_DECISIONS_PER_SYMBOL = 3000
DEFAULT_TRAIN_ROWS_PER_SYMBOL = 500
DEFAULT_GLOBAL_ROW_CAP = 1_500_000


def _even_sample_frame(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(frame) <= int(n):
        return frame
    idx = np.linspace(0, len(frame) - 1, int(n)).round().astype(int)
    return frame.iloc[np.unique(idx)].copy()


def _compact_generated_rows(generated: list[dict]) -> pd.DataFrame:
    if not generated:
        return pd.DataFrame()

    long = pd.DataFrame(generated)
    if long.empty:
        return pd.DataFrame()

    long["decision_timestamp"] = pd.to_datetime(
        long["decision_timestamp"], utc=True, errors="coerce"
    )
    long["date"] = pd.to_datetime(long["date"], errors="coerce").dt.normalize()
    long = long.dropna(subset=["decision_timestamp", "date"])
    long = long.sort_values("decision_timestamp")

    base_cols = [
        "symbol",
        "decision_timestamp",
        "date",
        "reference_close",
    ] + feature_columns()

    base = (
        long[base_cols]
        .drop_duplicates("decision_timestamp", keep="first")
        .reset_index(drop=True)
    )

    for horizon in HORIZONS:
        part = (
            long[long["horizon_min"].eq(horizon)][
                ["decision_timestamp", "future_close", "fwd_ret"]
            ]
            .drop_duplicates("decision_timestamp", keep="last")
            .rename(
                columns={
                    "future_close": f"future_close_{horizon}m",
                    "fwd_ret": f"fwd_ret_{horizon}m",
                }
            )
        )
        base = base.merge(part, on="decision_timestamp", how="left")

    return base


def _history_sessions(frame: pd.DataFrame) -> int:
    if frame.empty or "date" not in frame.columns:
        return 0
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize()
    return int(dates.nunique())


def _make_horizon_training_rows(
    symbol_paths: list[Path],
    horizon: int,
    per_symbol_rows: int,
    global_row_cap: int,
) -> tuple[pd.DataFrame, dict]:
    target_col = f"fwd_ret_{horizon}m"
    cols = (
        ["symbol", "decision_timestamp", "date", target_col]
        + feature_columns()
    )

    frames = []
    history = []
    eligible = []
    retained = 0

    for path in symbol_paths:
        try:
            frame = pd.read_parquet(path, columns=cols)
        except Exception as exc:
            history.append({
                "symbol": path.stem,
                "status": "CHECKPOINT_READ_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        if frame.empty:
            continue

        sym = str(frame["symbol"].iloc[0]).strip().upper()
        sessions = _history_sessions(frame)
        row = {
            "symbol": sym,
            "sessions": sessions,
        }

        if sessions < MIN_USABLE_SESSIONS:
            row["status"] = "EXCLUDED_INSUFFICIENT_HISTORY"
            history.append(row)
            continue

        if sessions < MIN_PREFERRED_SESSIONS:
            row["status"] = "LIMITED_HISTORY"
        else:
            row["status"] = "PREFERRED_HISTORY"

        frame = frame.dropna(subset=[target_col]).sort_values("decision_timestamp")
        if frame.empty:
            row["status"] = "EXCLUDED_NO_LABELS"
            history.append(row)
            continue

        frame = _even_sample_frame(frame, per_symbol_rows)

        remaining = int(global_row_cap) - int(retained)
        if remaining <= 0:
            row["status"] = "EXCLUDED_GLOBAL_TRAINING_CAP"
            history.append(row)
            continue

        if len(frame) > remaining:
            frame = _even_sample_frame(frame, remaining)

        frames.append(frame)
        retained += len(frame)
        row["training_rows"] = int(len(frame))
        history.append(row)
        eligible.append(sym)

    if not frames:
        return pd.DataFrame(), {
            "eligible_symbols": 0,
            "preferred_symbols": 0,
            "limited_symbols": 0,
            "retained_rows": 0,
            "history": history,
        }

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper().str.strip()
    data["decision_timestamp"] = pd.to_datetime(
        data["decision_timestamp"], utc=True, errors="coerce"
    )
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["date", "symbol", "decision_timestamp", target_col])

    meta = {
        "eligible_symbols": int(len(eligible)),
        "preferred_symbols": int(
            sum(1 for r in history if r.get("status") == "PREFERRED_HISTORY")
        ),
        "limited_symbols": int(
            sum(1 for r in history if r.get("status") == "LIMITED_HISTORY")
        ),
        "retained_rows": int(len(data)),
        "history": history,
    }
    return data, meta


def train_horizon(frame: pd.DataFrame, horizon: int, registry_root: Path) -> dict:
    h = frame[frame["horizon_min"].eq(horizon)].copy()
    h = add_cross_sectional_targets(h)
    h = h[
        h["fwd_ret"].notna()
        & h["y_up_excess"].notna()
        & (pd.to_numeric(h["cross_section_n"], errors="coerce") >= 20)
    ].copy()

    features = feature_columns()
    missing = [c for c in features if c not in h.columns]
    if missing:
        return {"status": "BLOCKED", "reason": f"missing features: {missing}"}

    h["date"] = pd.to_datetime(h["date"], errors="coerce").dt.normalize()
    h["symbol"] = h["symbol"].astype(str).str.upper().str.strip()
    h = h.dropna(subset=["date", "symbol", "fwd_ret", "y_up_excess"])

    # Keep the fold sizes bounded while retaining every symbol in the broad
    # universe. The cap is applied symmetrically by symbol before this function.
    spec = FoldSpec(
        n_folds=5,
        test_days=10,
        min_train_days=150,
        horizon_days=1,
        embargo_days=1,
    )
    cfg = ModelConfig(
        name=f"beta_v2_intraday_h{horizon}",
        hgb_params=dict(
            max_depth=3,
            learning_rate=0.04,
            max_iter=220,
            min_samples_leaf=120,
            l2_regularization=6.0,
            max_bins=63,
            early_stopping=False,
            random_state=17,
        ),
        logit_C=0.05,
        blend_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        val_frac=0.15,
        calibrate=True,
    )

    pred = run_walkforward(
        h,
        features,
        target="y_up_excess",
        ret_col="fwd_ret",
        cfg=cfg,
        spec=spec,
    )
    metrics = evaluate(pred, cost_bps=12.0)
    thresholds = choose_thresholds(pred, target_precision=0.56, min_coverage=0.02, min_n=300)

    reg = Registry(registry_root)
    champ = reg.champion_bundle()
    champ_metrics = None
    paired = None

    if champ is not None:
        try:
            champ_pred = run_walkforward(
                h,
                champ.features,
                target="y_up_excess",
                ret_col="fwd_ret",
                cfg=cfg,
                spec=spec,
            )
            champ_metrics = evaluate(champ_pred, cost_bps=12.0)
            paired = paired_logloss_gain(champ_pred, pred)
        except Exception as exc:
            paired = {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}"}

    decision = gate(champ_metrics, metrics, paired, GateRules(
        min_dates=40,
        min_auc=0.505,
        min_ic_t=1.5,
        min_pos_fold_frac=0.60,
        max_brier_worsen=0.0005,
        require_ci_excludes_zero=True,
    ))

    bundle = None
    if decision["promote"]:
        bundle = fit_bundle(
            h.dropna(subset=["y_up_excess"]),
            features,
            "y_up_excess",
            cfg,
        )
        entry = reg.register(f"beta_v2_intraday_h{horizon}", bundle, metrics, decision, thresholds)
    else:
        entry = None

    return {
        "status": "PROMOTED" if decision["promote"] else "REJECTED",
        "horizon_min": horizon,
        "rows": int(len(h)),
        "dates": int(h["date"].nunique()),
        "symbols": int(h["symbol"].nunique()),
        "features": features,
        "feature_count": len(features),
        "gate": decision,
        "metrics": {k: v for k, v in metrics.items() if k != "per_date_hit"},
        "thresholds": thresholds,
        "paired_vs_champion": paired,
        "champion_metrics": (
            {k: v for k, v in champ_metrics.items() if k != "per_date_hit"}
            if champ_metrics else None
        ),
        "registry_entry": entry,
    }



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive/Indian_Market_Research",
    )
    ap.add_argument(
        "--out",
        default=(
            "/content/drive/MyDrive/Indian_Market_Research/"
            "selfdriving/beta_version2/intraday"
        ),
    )
    ap.add_argument(
        "--max-decisions-per-symbol",
        type=int,
        default=DEFAULT_MAX_DECISIONS_PER_SYMBOL,
        help=(
            "Maximum decision timestamps retained per symbol after computing "
            "features over the full available history."
        ),
    )
    ap.add_argument(
        "--train-rows-per-symbol",
        type=int,
        default=DEFAULT_TRAIN_ROWS_PER_SYMBOL,
        help=(
            "Maximum compact decision rows contributed by one symbol to each "
            "horizon training run."
        ),
    )
    ap.add_argument(
        "--global-row-cap",
        type=int,
        default=DEFAULT_GLOBAL_ROW_CAP,
    )
    args = ap.parse_args()

    root = Path(args.drive_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    universe = sorted(set(official_nse_symbols(force_refresh=True)))
    if not universe:
        raise RuntimeError("Official NSE universe is empty")

    market = load_market(root)
    if market is None or market.empty:
        raise RuntimeError("NIFTY 50 1-minute market context could not be loaded")

    # New preparation namespace. The old 60-session long-format checkpoints
    # are intentionally not reused because the target policy now requires
    # materially deeper history.
    prep_dir = out / "compact_prepared_v3"
    symbol_parts_dir = prep_dir / "symbols"
    batch_dir = prep_dir / "batches"
    symbol_parts_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    coverage = []
    batch_size = 100

    for batch_start in range(0, len(universe), batch_size):
        batch_end = min(len(universe), batch_start + batch_size)
        part_no = batch_start // batch_size + 1
        batch_path = batch_dir / f"part_{part_no:04d}_{batch_start+1:04d}_{batch_end:04d}.parquet"

        # A non-empty batch checkpoint is reusable only when every symbol
        # checkpoint in its range also exists. This avoids accepting a stale
        # aggregate from a shorter-history run.
        symbol_paths = [
            symbol_parts_dir / f"{i:04d}_{sym}.parquet"
            for i, sym in enumerate(universe[batch_start:batch_end], batch_start + 1)
        ]
        if batch_path.exists() and all(
            p.exists() and p.stat().st_size > 0 for p in symbol_paths
        ):
            print(
                f"prepared batch {part_no}: reused compact checkpoints "
                f"symbols {batch_start+1:,}-{batch_end:,}",
                flush=True,
            )
            continue

        # If a previous batch marker is stale, recreate it from symbol files.
        if batch_path.exists():
            batch_path.unlink()

        batch_tables = []
        for i, sym in enumerate(universe[batch_start:batch_end], batch_start + 1):
            symbol_path = symbol_parts_dir / f"{i:04d}_{sym}.parquet"

            if symbol_path.exists() and symbol_path.stat().st_size > 0:
                print(
                    f"symbol {i:,}/{len(universe):,} {sym}: "
                    "reused compact symbol checkpoint",
                    flush=True,
                )
                continue

            raw = load_symbol(root, sym)
            if raw is None or raw.empty:
                coverage.append({
                    "symbol": sym,
                    "status": "NO_DATA",
                    "raw_rows": 0,
                    "sessions": 0,
                    "rows": 0,
                })
                continue

            try:
                frame = raw.reset_index()
                frame["symbol"] = sym
                frame = frame.rename(columns={frame.columns[0]: "ts"})

                generated = _session_rows(
                    frame,
                    market,
                    HORIZONS,
                    max_decisions=args.max_decisions_per_symbol,
                )
                compact = _compact_generated_rows(generated)

                sessions = _history_sessions(compact)
                status = (
                    "PREFERRED_HISTORY"
                    if sessions >= MIN_PREFERRED_SESSIONS
                    else (
                        "LIMITED_HISTORY"
                        if sessions >= MIN_USABLE_SESSIONS
                        else (
                            "EXCLUDED_INSUFFICIENT_HISTORY"
                            if sessions > 0
                            else "NO_FEATURE_ROWS"
                        )
                    )
                )

                tmp = symbol_path.with_suffix(".parquet.tmp")
                compact.to_parquet(tmp, index=False)
                tmp.replace(symbol_path)

                coverage.append({
                    "symbol": sym,
                    "status": status,
                    "raw_rows": int(len(raw)),
                    "sessions": int(sessions),
                    "rows": int(len(compact)),
                })

                print(
                    f"symbol {i:,}/{len(universe):,} {sym}: "
                    f"history_sessions={sessions:,}; "
                    f"compact_rows={len(compact):,}",
                    flush=True,
                )

                del raw, frame, generated, compact

            except Exception as exc:
                coverage.append({
                    "symbol": sym,
                    "status": "ERROR",
                    "raw_rows": int(len(raw)),
                    "sessions": 0,
                    "rows": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(
                    f"symbol {i:,}/{len(universe):,} {sym}: ERROR "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        # Rebuild aggregate batch from symbol checkpoints. This is intentionally
        # compact wide format and is used only as a resumable artifact.
        for p in symbol_paths:
            if p.exists() and p.stat().st_size > 0:
                try:
                    z = pd.read_parquet(p)
                    if not z.empty:
                        batch_tables.append(z)
                except Exception as exc:
                    print(
                        f"WARNING: cannot read symbol checkpoint {p.name}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

        if batch_tables:
            batch = pd.concat(batch_tables, ignore_index=True)
        else:
            batch = pd.DataFrame()

        tmp_batch = batch_path.with_suffix(".parquet.tmp")
        batch.to_parquet(tmp_batch, index=False)
        tmp_batch.replace(batch_path)

        pd.DataFrame(coverage).to_parquet(
            prep_dir / "COVERAGE_PROGRESS.parquet",
            index=False,
        )

        print(
            f"prepared batch {part_no}: saved compact rows={len(batch):,}; "
            f"symbols={batch_end-batch_start:,}",
            flush=True,
        )

        del batch_tables, batch

    # Fail early rather than wasting hours training on a short-history archive.
    all_symbol_paths = sorted(symbol_parts_dir.glob("*.parquet"))
    session_counts = []
    for p in all_symbol_paths:
        try:
            z = pd.read_parquet(p, columns=["date"])
            if not z.empty:
                session_counts.append(
                    int(pd.to_datetime(z["date"], errors="coerce").dt.normalize().nunique())
                )
        except Exception:
            continue

    max_sessions = max(session_counts) if session_counts else 0
    preferred_count = sum(1 for n in session_counts if n >= MIN_PREFERRED_SESSIONS)
    usable_count = sum(1 for n in session_counts if n >= MIN_USABLE_SESSIONS)

    print(
        f"INTRADAY HISTORY AUDIT: universe={len(universe):,}; "
        f"max_sessions={max_sessions:,}; "
        f"preferred_500+={preferred_count:,}; usable_200+={usable_count:,}",
        flush=True,
    )

    if max_sessions < MIN_USABLE_SESSIONS:
        raise RuntimeError(
            f"Intraday archive has only {max_sessions} sessions at best. "
            f"Beta V2 requires at least {MIN_USABLE_SESSIONS} usable sessions "
            f"and prefers {MIN_PREFERRED_SESSIONS}-{PREFERRED_SESSIONS}. "
            "Run the expanded historical seed/backfill first."
        )

    symbol_paths = sorted(symbol_parts_dir.glob("*.parquet"))

    results = {}
    registry_root = out / "models"

    for horizon in HORIZONS:
        print(f"TRAINING HORIZON +{horizon}M", flush=True)

        horizon_data, history_meta = _make_horizon_training_rows(
            symbol_paths,
            horizon,
            args.train_rows_per_symbol,
            args.global_row_cap,
        )

        if horizon_data.empty:
            results[str(horizon)] = {
                "status": "BLOCKED",
                "reason": "No horizon-specific training rows after history/label filters",
                "history": history_meta,
            }
            continue

        horizon_dir = registry_root / f"h{horizon}"
        result = train_horizon(horizon_data, horizon, horizon_dir)
        result["history_policy"] = {
            "preferred_sessions": PREFERRED_SESSIONS,
            "minimum_preferred_sessions": MIN_PREFERRED_SESSIONS,
            "minimum_usable_sessions": MIN_USABLE_SESSIONS,
            "train_rows_per_symbol": args.train_rows_per_symbol,
            "global_row_cap": args.global_row_cap,
        }
        result["history_summary"] = {
            "eligible_symbols": history_meta["eligible_symbols"],
            "preferred_symbols": history_meta["preferred_symbols"],
            "limited_symbols": history_meta["limited_symbols"],
            "retained_rows": history_meta["retained_rows"],
        }
        results[str(horizon)] = result

        del horizon_data
        import gc
        gc.collect()

    # Persist coverage after preparation/training.
    pd.DataFrame(coverage).to_parquet(
        out / "BETA_V2_INTRADAY_COVERAGE.parquet",
        index=False,
    )

    manifest = {
        "version": "2026-10-06-beta-v2-intraday-history-v3",
        "status": "CANDIDATE_ONLY",
        "production_promotion": False,
        "production_artifacts_changed": False,
        "universe_symbols": len(universe),
        "preferred_training_history_sessions": PREFERRED_SESSIONS,
        "minimum_preferred_training_history_sessions": MIN_PREFERRED_SESSIONS,
        "minimum_usable_sessions": MIN_USABLE_SESSIONS,
        "max_sessions_observed": max_sessions,
        "preferred_symbol_count": preferred_count,
        "usable_symbol_count": usable_count,
        "max_decisions_per_symbol": args.max_decisions_per_symbol,
        "train_rows_per_symbol": args.train_rows_per_symbol,
        "global_row_cap_per_horizon": args.global_row_cap,
        "checkpoint_format": "one_decision_row_with_six_future_horizon_returns",
        "features": feature_columns(),
        "feature_count": len(feature_columns()),
        "volume_features_present": False,
        "horizons": list(HORIZONS),
        "results": results,
        "design": {
            "history_policy": "750 preferred / 500 preferred minimum / 200 hard usability minimum",
            "feature_history": "full available causal intraday history",
            "training_sampling": (
                "deterministic evenly spaced decision rows per symbol after "
                "full-history feature calculation; no random row deletion"
            ),
            "target": (
                "future return minus cross-sectional median future return at "
                "the same decision timestamp and horizon"
            ),
            "validation": "purged walk-forward with one-session embargo",
            "probability_calibration": "isotonic",
            "action_thresholds": "Wilson lower-bound precision",
            "abstention": True,
        },
    }

    (out / "BETA_V2_INTRADAY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
