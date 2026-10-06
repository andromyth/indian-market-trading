"""Colab trainer for the isolated Beta V2 Delivery challenger.

This script does NOT touch the current production Delivery artifacts.
It consumes the existing ALL-NSE PIT panel, enriches it with point-in-time
global/news/F&O features, derives a causal next-session excess-return target,
and trains through the self-driving champion/challenger registry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from delivery_training_universe import training_roster
from nse_data_layer import load_canonical_cash_research_history
from nse_delivery_data_layer import DEFAULT_DELIVERY_ROOT, load_delivery_history
from colab_build_all_nse_delivery_multifactor_candidate import make_panel
from selfdriving.sd_cycle import train_stage
from selfdriving.sd_global_cues import attach_global_features
from selfdriving.sd_v2_panel import build_beta_delivery_panel, split_training_inference


DEFAULT_BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
DEFAULT_PANEL = (
    DEFAULT_BASE
    / "models/all_nse_delivery_multifactor_replacement_2026"
    / "ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"
)
MIN_WALKFORWARD_SESSIONS = 422


def load_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--force-rebuild-base",
        action="store_true",
        help="Force rebuilding the full-history Beta V2 base panel instead of reusing its Drive cache.",
    )
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out) if args.out else base / "selfdriving/beta_version2"
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.panel)
    panel_sessions = int(pd.to_datetime(panel.get("date"), errors="coerce").dt.normalize().nunique()) if "date" in panel.columns else 0

    # The original candidate panel is a 2026-only research artifact in some
    # Drive snapshots. Eight purged walk-forward folds require at least 422
    # distinct sessions with the current FoldSpec (252 train + 8*21 test +
    # purge/embargo). Never silently shrink the validation design just to make
    # a short panel train. Rebuild the Beta V2 source panel from the canonical
    # identity-preserved NSE cash master when the supplied panel is too short.
    if panel_sessions < MIN_WALKFORWARD_SESSIONS:
        master = base / "production_data/nse_official/ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
        if not master.exists() or master.stat().st_size == 0:
            raise RuntimeError(
                f"Input panel has only {panel_sessions} sessions; Beta V2 requires "
                f">= {MIN_WALKFORWARD_SESSIONS}. Canonical NSE cash master is "
                f"missing: {master}. Build the full-history canonical cash master first."
            )

        source_copy = out / "BETA_V2_DELIVERY_BASE_PANEL_FULL_HISTORY.parquet"
        if (
            not args.force_rebuild_base
            and source_copy.exists()
            and source_copy.stat().st_size > 0
        ):
            panel = pd.read_parquet(source_copy)
            rebuilt_sessions = int(
                pd.to_datetime(panel["date"], errors="coerce")
                .dt.normalize()
                .nunique()
            )
            print(
                f"Reusing cached full-history Beta V2 base panel: "
                f"rows={len(panel):,}, symbols={panel['symbol'].nunique():,}, "
                f"sessions={rebuilt_sessions:,}; cached={source_copy}",
                flush=True,
            )
        else:
            print(
                f"Input panel has only {panel_sessions} sessions. Rebuilding the Beta V2 "
                f"base panel from the canonical NSE cash master for full-history walk-forward training.",
                flush=True,
            )
            bulk = load_canonical_cash_research_history(master)
            roster = training_roster(bulk, min_history=60, force_refresh=False)
            trainable = list(roster["trainable_symbols"])
            delivery = load_delivery_history(
                DEFAULT_DELIVERY_ROOT,
                symbols=trainable,
                start=bulk["date"].min(),
                end=bulk["date"].max(),
            )
            # The downstream code sorts by symbol/date where needed. Avoid the
            # expensive global sort here because this panel can contain millions
            # of rows.
            rebuilt = make_panel(
                bulk,
                trainable,
                sort_output=False,
                delivery=delivery,
            )
            if rebuilt.empty:
                raise RuntimeError("Full-history Beta V2 base-panel rebuild produced no rows")

            rebuilt_sessions = int(
                pd.to_datetime(rebuilt["date"], errors="coerce")
                .dt.normalize()
                .nunique()
            )
            if rebuilt_sessions < MIN_WALKFORWARD_SESSIONS:
                raise RuntimeError(
                    f"Canonical source rebuild produced only {rebuilt_sessions} sessions; "
                    f"Beta V2 requires >= {MIN_WALKFORWARD_SESSIONS}. "
                    "The available historical source is not yet sufficient for the configured OOS design."
                )
            panel = rebuilt
            panel.to_parquet(source_copy, index=False)
            print(
                f"Full-history base panel ready: rows={len(panel):,}, "
                f"symbols={panel['symbol'].nunique():,}, sessions={rebuilt_sessions:,}; "
                f"saved={source_copy}",
                flush=True,
            )

    global_path = base / "selfdriving/global_cues.parquet"
    news_path = base / "selfdriving/raw_news.parquet"
    fno_path = base / "production_data/nse_official/fno/fno_daily_features_2026.parquet"

    global_cues = load_optional(global_path)
    raw_news = load_optional(news_path)
    fno_daily = load_optional(fno_path)

    beta_panel, audit = build_beta_delivery_panel(
        panel,
        global_cues=global_cues,
        raw_news=raw_news,
        fno_daily=fno_daily,
    )

    beta_panel.to_parquet(out / "BETA_V2_DELIVERY_PANEL.parquet", index=False)

    training, inference, realised = split_training_inference(beta_panel)
    training.to_parquet(out / "BETA_V2_DELIVERY_TRAINING_PANEL.parquet", index=False)
    inference.to_parquet(out / "BETA_V2_DELIVERY_INFERENCE_PANEL.parquet", index=False)
    realised.to_parquet(out / "BETA_V2_DELIVERY_REALISED_RETURNS.parquet", index=False)

    result = train_stage(
        training,
        out / "models",
    )

    status = {
        "version": "2026-10-06-beta-v2-delivery",
        "status": result.get("status"),
        "audit": audit,
        "training_rows": int(len(training)),
        "inference_rows": int(len(inference)),
        "realised_rows": int(len(realised)),
        "groups": result.get("groups"),
        "gate": result.get("gate"),
        "thresholds": result.get("thresholds"),
        "challenger": result.get("challenger"),
        "paired_vs_champion": result.get("paired_vs_champion"),
        "production_promotion": False,
        "production_artifacts_changed": False,
    }
    (out / "status_train.json").write_text(
        json.dumps(status, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(status, indent=2, default=str))
    return 0 if result.get("status") in {"PROMOTED", "REJECTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
