#!/usr/bin/env python3
"""Retrofit continuous-return heads onto existing Beta V2 Intraday champions.

This targeted Colab script reuses the exact compact prepared symbol checkpoints
created by COLAB_SD_INTRADAY_V2_RETRAIN.py. It does NOT retrain the directional
classifiers or change their blend/calibration/thresholds; it only adds a model
for raw forward return so inference can calculate an estimated future price.

Default checkpoint root:
  /content/drive/MyDrive/Indian_Market_Research/selfdriving/beta_version2/intraday

The script never searches Drive recursively. It reads only the explicit
compact_prepared_v3/symbols folder and writes registry bundles under models/hN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from COLAB_SD_INTRADAY_V2_RETRAIN import (
    DEFAULT_GLOBAL_ROW_CAP,
    DEFAULT_TRAIN_ROWS_PER_SYMBOL,
    HORIZONS,
    _make_horizon_training_rows,
)
from selfdriving.sd_intraday_v2 import feature_columns
from selfdriving.sd_registry import Registry
from selfdriving.sd_walkforward import fit_return_regressor


DEFAULT_ROOT = Path(
    "/content/drive/MyDrive/Indian_Market_Research/"
    "selfdriving/beta_version2/intraday"
)


def retrofit_horizon(
    root: Path,
    horizon: int,
    *,
    train_rows_per_symbol: int,
    global_row_cap: int,
    force: bool = False,
) -> dict[str, Any]:
    symbol_dir = root / "compact_prepared_v3" / "symbols"
    registry_dir = root / "models" / f"h{horizon}"

    # Exact-path check; intentionally no Drive-wide search.
    if not symbol_dir.is_dir():
        return {
            "horizon_min": horizon,
            "status": "BLOCKED",
            "reason": (
                f"Prepared checkpoint directory not found: {symbol_dir}. "
                "Run COLAB_SD_INTRADAY_V2_RETRAIN.py once to create compact_prepared_v3."
            ),
        }
    symbol_paths = sorted(symbol_dir.glob("*.parquet"))
    if not symbol_paths:
        return {
            "horizon_min": horizon,
            "status": "BLOCKED",
            "reason": f"No symbol checkpoints in the expected directory: {symbol_dir}",
        }

    registry = Registry(registry_dir)
    champion_entry = registry.state.get("champion") or {}
    champion = registry.champion_bundle()
    if champion is None or not champion_entry:
        return {
            "horizon_min": horizon,
            "status": "SKIPPED_NO_CHAMPION",
            "reason": f"No validated directional champion exists in {registry_dir}",
        }

    if not force and getattr(champion, "return_model", None) is not None:
        return {
            "horizon_min": horizon,
            "status": "ALREADY_HAS_RETURN_HEAD",
            "champion_file": champion_entry.get("file"),
            "return_forecast": (
                (getattr(champion, "meta", {}) or {}).get("return_forecast", {})
            ),
            "classifier_thresholds_unchanged": True,
        }

    target_name = f"fwd_ret_{horizon}m"
    frame, history_meta = _make_horizon_training_rows(
        symbol_paths,
        horizon,
        train_rows_per_symbol,
        global_row_cap,
    )
    if frame is None or frame.empty or target_name not in frame.columns:
        return {
            "horizon_min": horizon,
            "status": "BLOCKED",
            "reason": f"No labelled horizon training data for {target_name}",
            "history": history_meta,
        }

    frame = frame.rename(columns={target_name: "fwd_ret"})
    features = feature_columns()
    missing = [name for name in features if name not in frame.columns]
    if missing:
        return {
            "horizon_min": horizon,
            "status": "BLOCKED",
            "reason": f"Prepared checkpoints lack required features: {missing[:10]}",
        }

    model, validation = fit_return_regressor(
        frame,
        features,
        "fwd_ret",
        date_col="date",
    )
    if model is None:
        return {
            "horizon_min": horizon,
            "status": "BLOCKED",
            "reason": validation.get("reason", "return model could not be fitted"),
            "return_forecast": validation,
            "history": {
                "retained_rows": history_meta.get("retained_rows"),
                "eligible_symbols": history_meta.get("eligible_symbols"),
            },
        }

    # Attach only the new continuous-return estimator. Existing directional
    # components and policy are preserved byte-for-byte in model semantics.
    champion.return_model = model
    if not isinstance(getattr(champion, "meta", None), dict):
        champion.meta = {}
    champion.meta["return_forecast"] = validation
    champion.meta["return_head_retrofit_only"] = True
    champion.meta["return_head_target"] = "raw_forward_return"
    champion.meta["return_head_horizon_minutes"] = int(horizon)

    old_thresholds = champion_entry.get("thresholds") or {"abstain_all": True}
    old_metrics = champion_entry.get("metrics") or {}
    new_entry = registry.register(
        f"beta_v2_intraday_h{horizon}_return_head",
        champion,
        old_metrics,
        {
            "promote": True,
            "type": "RETURN_HEAD_ONLY",
            "reasons": [
                "Only the continuous-return model head was attached.",
                "Directional classifier, probability calibration and action thresholds were not changed.",
            ],
        },
        old_thresholds,
    )

    # Defensive contract check: the registry still contains the original
    # action thresholds and original classifier metadata after the retrofit.
    final_thresholds = (registry.state.get("champion") or {}).get("thresholds")
    if final_thresholds != old_thresholds:
        raise RuntimeError(
            f"Threshold contract changed unexpectedly during horizon {horizon} retrofit"
        )

    return {
        "horizon_min": horizon,
        "status": "RETROFITTED",
        "champion_file": new_entry.get("file"),
        "return_forecast": validation,
        "classifier_thresholds_unchanged": True,
        "classifier_metrics_unchanged": True,
        "retained_rows": history_meta.get("retained_rows"),
        "eligible_symbols": history_meta.get("eligible_symbols"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add price-regression heads to existing Beta V2 intraday champions."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--train-rows-per-symbol", type=int, default=DEFAULT_TRAIN_ROWS_PER_SYMBOL)
    parser.add_argument("--global-row-cap", type=int, default=DEFAULT_GLOBAL_ROW_CAP)
    parser.add_argument("--force", action="store_true", help="Refit heads even when one already exists.")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("Ignoring notebook/kernel arguments:", unknown)

    bad_horizons = sorted(set(args.horizons) - set(HORIZONS))
    if bad_horizons:
        parser.error(f"Unsupported horizons: {bad_horizons}; allowed: {list(HORIZONS)}")

    results = []
    for horizon in args.horizons:
        print(f"RETROFIT RETURN HEAD: +{horizon} MIN", flush=True)
        result = retrofit_horizon(
            args.root,
            horizon,
            train_rows_per_symbol=args.train_rows_per_symbol,
            global_row_cap=args.global_row_cap,
            force=args.force,
        )
        results.append(result)
        print(json.dumps(result, default=str, sort_keys=True), flush=True)

    print("=== RETURN-HEAD RETROFIT SUMMARY ===")
    print(json.dumps(results, indent=2, default=str))
    failures = [
        result for result in results
        if result.get("status") in {"BLOCKED", "ERROR"}
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
