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

from selfdriving.sd_cycle import train_stage
from selfdriving.sd_global_cues import attach_global_features
from selfdriving.sd_v2_panel import build_beta_delivery_panel, split_training_inference


DEFAULT_BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
DEFAULT_PANEL = (
    DEFAULT_BASE
    / "models/all_nse_delivery_multifactor_replacement_2026"
    / "ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"
)


def load_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = Path(args.base)
    out = Path(args.out) if args.out else base / "selfdriving/beta_version2"
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.panel)

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
