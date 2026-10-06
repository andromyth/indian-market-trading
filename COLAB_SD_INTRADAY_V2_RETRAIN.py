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


def cap_symbol_rows(rows: list[dict], max_decisions: int) -> list[dict]:
    if len(rows) <= max_decisions * len(HORIZONS):
        return rows
    df = pd.DataFrame(rows)
    # sorted() returns a Python list; NumPy integer-array indexing into
    # that list raises "only integer scalar arrays can be converted to a
    # scalar index". Convert to a NumPy array before positional selection.
    timestamps = np.asarray(
        sorted(pd.to_datetime(df["decision_timestamp"], utc=True).unique())
    )
    keep_n = min(int(max_decisions), len(timestamps))
    idx = np.linspace(0, len(timestamps) - 1, keep_n).round().astype(int)
    wanted = set(timestamps[idx].tolist())
    return df[df["decision_timestamp"].isin(wanted)].to_dict("records")


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
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument(
        "--out",
        default="/content/drive/MyDrive/Indian_Market_Research/selfdriving/beta_version2/intraday",
    )
    ap.add_argument(
        "--max-decisions-per-symbol",
        type=int,
        default=60,
        help="Decision timestamps retained per symbol across all six horizons. "
             "Keeps the full NSE universe while bounding Colab RAM.",
    )
    ap.add_argument(
        "--global-row-cap",
        type=int,
        default=900000,
        help="Hard cap on generated rows; with 60 timestamps/symbol and 2,382 symbols "
             "the normal panel remains below this cap.",
    )
    args = ap.parse_args()

    root = Path(args.drive_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    universe = sorted(set(official_nse_symbols(force_refresh=True)))
    market = load_market(root)
    if market is None or market.empty:
        raise RuntimeError("NIFTY 50 1-minute market context could not be loaded")

    parts_dir = out / "prepared_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Fine-grained resume layer: one checkpoint per symbol. This is separate
    # from the historical batch checkpoints so an interrupted batch can resume
    # from the last completed symbol without repeating earlier work.
    symbol_parts_dir = parts_dir / "symbols"
    symbol_parts_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    coverage = []

    # Checkpoint in 100-symbol batches. The previous implementation retained
    # every generated row for the entire NSE universe in one Python list,
    # creating unnecessary RAM pressure on 12-GB Colab runtimes.
    batch_size = 100

    def parquet_rows(path: Path) -> int:
        try:
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return 0

    for batch_start in range(0, len(universe), batch_size):
        batch_end = min(len(universe), batch_start + batch_size)
        part_no = batch_start // batch_size + 1
        part_path = parts_dir / f"part_{part_no:04d}_{batch_start+1:04d}_{batch_end:04d}.parquet"

        if part_path.exists() and part_path.stat().st_size > 0:
            # A prior failed implementation wrote empty schema-only parquet
            # files as "completed" batches. Those are NOT reusable work.
            # Reuse only checkpoints that actually contain feature rows.
            try:
                checkpoint_rows = int(__import__("pyarrow.parquet", fromlist=[""]).ParquetFile(part_path).metadata.num_rows)
            except Exception:
                checkpoint_rows = 0

            if checkpoint_rows > 0:
                print(
                    f"prepared batch {part_no}: reused symbols {batch_start+1:,}-{batch_end:,} "
                    f"(rows={checkpoint_rows:,})",
                    flush=True,
                )
                continue

            print(
                f"prepared batch {part_no}: empty checkpoint detected; rebuilding "
                f"symbols {batch_start+1:,}-{batch_end:,}",
                flush=True,
            )
            try:
                part_path.unlink()
            except FileNotFoundError:
                pass

        batch_rows = []
        batch_checkpoint_paths = []

        for i, sym in enumerate(universe[batch_start:batch_end], batch_start + 1):
            symbol_part_path = symbol_parts_dir / f"{i:04d}_{sym}.parquet"

            # Reuse a completed symbol checkpoint, including a valid zero-row
            # checkpoint for a symbol that genuinely has insufficient data.
            if symbol_part_path.exists() and symbol_part_path.stat().st_size > 0:
                print(
                    f"symbol {i:,}/{len(universe):,} {sym}: reused symbol checkpoint",
                    flush=True,
                )
                batch_checkpoint_paths.append(symbol_part_path)
                continue

            try:
                raw = load_symbol(root, sym)
                if raw is None or raw.empty:
                    generated = []
                    coverage.append({
                        "symbol": sym,
                        "status": "NO_DATA",
                        "decision_rows": 0,
                        "output_rows": 0,
                    })
                    pd.DataFrame().to_parquet(symbol_part_path, index=False)
                    batch_checkpoint_paths.append(symbol_part_path)
                    print(
                        f"symbol {i:,}/{len(universe):,} {sym}: "
                        "generated=0; status=NO_DATA",
                        flush=True,
                    )
                    continue

                frame = raw.reset_index()
                frame["symbol"] = sym
                frame = frame.rename(columns={frame.columns[0]: "ts"})
                generated = _session_rows(frame, market, HORIZONS)
                generated = cap_symbol_rows(
                    generated,
                    args.max_decisions_per_symbol,
                )

                # Persist the symbol BEFORE moving on. A crash after this
                # point cannot force the symbol to be recomputed.
                symbol_panel = pd.DataFrame(generated)
                tmp_symbol = symbol_part_path.with_suffix(".parquet.tmp")
                symbol_panel.to_parquet(tmp_symbol, index=False)
                tmp_symbol.replace(symbol_part_path)
                batch_checkpoint_paths.append(symbol_part_path)

                coverage.append({
                    "symbol": sym,
                    "status": "OK" if generated else "NO_FEATURE_ROWS",
                    "decision_rows": int(
                        len(set(
                            pd.to_datetime(
                                x["decision_timestamp"],
                                utc=True,
                            )
                            for x in generated
                        ))
                    ),
                    "output_rows": int(len(generated)),
                })

                batch_rows.extend(generated)
                print(
                    f"symbol {i:,}/{len(universe):,} {sym}: "
                    f"generated={len(generated):,}; batch_rows={len(batch_rows):,}",
                    flush=True,
                )

                del raw, frame, generated, symbol_panel

            except Exception as exc:
                coverage.append({
                    "symbol": sym,
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(
                    f"symbol {i:,}/{len(universe):,} {sym}: ERROR "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        # Build the batch checkpoint from the fine-grained symbol checkpoints.
        # This works both for a clean batch and after resuming mid-batch.
        batch_tables = []
        for symbol_part_path in batch_checkpoint_paths:
            if parquet_rows(symbol_part_path) > 0:
                batch_tables.append(pd.read_parquet(symbol_part_path))

        if batch_tables:
            batch_panel = pd.concat(batch_tables, ignore_index=True)
        else:
            batch_panel = pd.DataFrame(
                columns=[
                    "symbol", "decision_timestamp", "date",
                    "horizon_min", "reference_close", "future_close", "fwd_ret",
                ]
            )

        tmp_batch = part_path.with_suffix(".parquet.tmp")
        batch_panel.to_parquet(tmp_batch, index=False)
        tmp_batch.replace(part_path)

        print(
            f"prepared batch {part_no}: saved rows={len(batch_panel):,} "
            f"symbols={batch_end-batch_start:,}",
            flush=True,
        )

        del batch_rows, batch_panel, batch_tables, batch_checkpoint_paths

        # Persist batch-level coverage after every completed batch.
        pd.DataFrame(coverage).to_parquet(
            out / "BETA_V2_INTRADAY_COVERAGE_PROGRESS.parquet",
            index=False,
        )


    part_files = sorted(parts_dir.glob("part_*.parquet"))
    if len(part_files) != int(np.ceil(len(universe) / batch_size)):
        raise RuntimeError(
            f"Prepared checkpoint count incomplete: {len(part_files)} files for "
            f"{len(universe)} symbols"
        )

    panel_path = out / "BETA_V2_INTRADAY_PANEL.parquet"
    if not panel_path.exists() or panel_path.stat().st_size == 0:
        glob = str(parts_dir / "part_*.parquet").replace("'", "''")
        tmp = out / "BETA_V2_INTRADAY_PANEL.parquet.tmp"
        con = __import__("duckdb").connect()
        try:
            con.execute(
                f"""
                COPY (
                    SELECT *
                    FROM read_parquet('{glob}')
                )
                TO '{str(tmp).replace("'", "''")}'
                (FORMAT PARQUET, COMPRESSION ZSTD);
                """
            )
        finally:
            con.close()
        tmp.replace(panel_path)

    panel = pd.read_parquet(panel_path)
    if panel.empty:
        raise RuntimeError("No Beta V2 intraday feature rows were built")

    if len(panel) > args.global_row_cap:
        raise RuntimeError(
            f"Prepared intraday panel has {len(panel):,} rows, exceeding the configured "
            f"global cap of {args.global_row_cap:,}. Lower --max-decisions-per-symbol "
            "rather than arbitrarily deleting timestamps."
        )

    panel = add_cross_sectional_targets(panel)
    panel.to_parquet(panel_path, index=False)
    pd.DataFrame(coverage).to_parquet(
        out / "BETA_V2_INTRADAY_COVERAGE.parquet",
        index=False,
    )

    results = {}
    registry_root = out / "models"
    for horizon in HORIZONS:
        print(f"TRAINING HORIZON +{horizon}M", flush=True)
        horizon_dir = registry_root / f"h{horizon}"
        results[str(horizon)] = train_horizon(panel, horizon, horizon_dir)

    manifest = {
        "version": "2026-10-06-beta-v2-intraday",
        "status": "CANDIDATE_ONLY",
        "production_promotion": False,
        "production_artifacts_changed": False,
        "universe_symbols": len(universe),
        "panel_rows": len(panel),
        "features": feature_columns(),
        "feature_count": len(feature_columns()),
        "volume_features_present": any(c.startswith("volume") for c in feature_columns()),
        "horizons": list(HORIZONS),
        "results": results,
        "design": {
            "feature_semantics": "price_time_only",
            "live_volume_dependency": False,
            "target": "future return minus cross-sectional median future return at the same decision timestamp and horizon",
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
