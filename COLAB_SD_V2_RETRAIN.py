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

import duckdb

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


def build_full_history_panel_chunked(
    base: Path,
    out: Path,
    master: Path,
    *,
    chunk_size: int = 100,
) -> pd.DataFrame:
    """Build the full-history Beta V2 base panel in resumable symbol chunks.

    The previous implementation held 2,382 per-symbol DataFrames in memory and
    then performed one large concat. On Colab this can be interrupted after the
    final progress message during the peak-memory concat. This version writes
    each completed symbol chunk to Drive, resumes existing chunks, and uses
    DuckDB to assemble the completed Parquet parts without creating another
    giant Python-side concat.
    """
    parts_dir = out / "base_panel_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    combined = out / "BETA_V2_DELIVERY_BASE_PANEL_FULL_HISTORY.parquet"

    # Fast resume path: if the combined parquet already exists, do not reload
    # the canonical 4M+ row source just to discover that the work is complete.
    if combined.exists() and combined.stat().st_size > 0:
        print(
            f"Full-history base panel already assembled: {combined} "
            f"({combined.stat().st_size / 1024 / 1024:.1f} MB); loading it.",
            flush=True,
        )
        panel = pd.read_parquet(combined)
        sessions = int(
            pd.to_datetime(panel["date"], errors="coerce").dt.normalize().nunique()
        )
        if sessions < MIN_WALKFORWARD_SESSIONS:
            raise RuntimeError(
                f"Cached full-history base panel has only {sessions} sessions; "
                f"Beta V2 requires >= {MIN_WALKFORWARD_SESSIONS}."
            )
        print(
            f"Cached full-history base panel loaded: rows={len(panel):,}, "
            f"symbols={panel['symbol'].nunique():,}, sessions={sessions:,}",
            flush=True,
        )
        return panel

    # Second fast resume path: when every symbol chunk is already present,
    # assemble directly. No canonical source or Delivery reread is needed.
    final_part = parts_dir / "part_0024_2301_2382.parquet"
    existing_parts = sorted(parts_dir.glob("part_*.parquet"))
    if final_part.exists() and len(existing_parts) >= 24:
        print(
            f"All {len(existing_parts)} checkpoint files are present; "
            "assembling cached chunks directly.",
            flush=True,
        )
    else:
        existing_parts = []

    bulk = None
    trainable = None
    delivery = None
    if not existing_parts:
        bulk = load_canonical_cash_research_history(master)
        if bulk.empty:
            raise RuntimeError("Canonical NSE cash research view is empty")

        roster = training_roster(bulk, min_history=60, force_refresh=False)
        trainable = list(roster["trainable_symbols"])
        if not trainable:
            raise RuntimeError("No trainable NSE symbols found in canonical cash history")

        delivery = load_delivery_history(
            DEFAULT_DELIVERY_ROOT,
            symbols=trainable,
            start=bulk["date"].min(),
            end=bulk["date"].max(),
        )

        # Prepare a cheap symbol lookup so each batch filters the large source
    # once, rather than copying the complete 4M+ row frame.
    bulk["symbol"] = bulk["symbol"].astype(str).str.upper().str.strip()
    delivery = delivery.copy()
    if not delivery.empty and "symbol" in delivery.columns:
        delivery["symbol"] = delivery["symbol"].astype(str).str.upper().str.strip()

    if not existing_parts:
        total = len(trainable)
        for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        chunk_index = start // chunk_size + 1
        part = parts_dir / f"part_{chunk_index:04d}_{start+1:04d}_{end:04d}.parquet"

        if part.exists() and part.stat().st_size > 0:
            print(
                f"base panel chunk {chunk_index}: reused symbols {start+1:,}-{end:,}; "
                f"file={part.name}",
                flush=True,
            )
            continue

        symbols = trainable[start:end]
        symbol_set = set(str(x).upper().strip() for x in symbols)
        bulk_chunk = bulk[bulk["symbol"].isin(symbol_set)].copy()
        delivery_chunk = (
            delivery[delivery["symbol"].isin(symbol_set)].copy()
            if not delivery.empty and "symbol" in delivery.columns
            else delivery
        )

        print(
            f"base panel chunk {chunk_index}: building symbols {start+1:,}-{end:,} "
            f"({len(symbols):,} symbols)",
            flush=True,
        )
        chunk_panel = make_panel(
            bulk_chunk,
            symbols,
            sort_output=False,
            delivery=delivery_chunk,
        )
        if chunk_panel.empty:
            raise RuntimeError(
                f"Base panel chunk {chunk_index} produced no rows "
                f"for symbols {start+1}-{end}"
            )
        chunk_panel.to_parquet(part, index=False)
        print(
            f"base panel chunk {chunk_index}: saved rows={len(chunk_panel):,} "
            f"symbols={chunk_panel['symbol'].nunique():,}",
            flush=True,
        )
        del bulk_chunk, delivery_chunk, chunk_panel

    # Assemble via DuckDB directly into one Parquet file. This avoids the
    # peak-memory pandas concat that repeatedly interrupted the Colab run.
    combined = out / "BETA_V2_DELIVERY_BASE_PANEL_FULL_HISTORY.parquet"
    glob = str(parts_dir / "part_*.parquet").replace("'", "''")
    tmp = out / "BETA_V2_DELIVERY_BASE_PANEL_FULL_HISTORY.parquet.tmp"
    if not combined.exists() or combined.stat().st_size == 0:
        con = duckdb.connect()
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
        tmp.replace(combined)

    panel = pd.read_parquet(combined)
    sessions = int(
        pd.to_datetime(panel["date"], errors="coerce").dt.normalize().nunique()
    )
    if sessions < MIN_WALKFORWARD_SESSIONS:
        raise RuntimeError(
            f"Chunked canonical source build produced only {sessions} sessions; "
            f"Beta V2 requires >= {MIN_WALKFORWARD_SESSIONS}."
        )
    print(
        f"Full-history base panel ready: rows={len(panel):,}, "
        f"symbols={panel['symbol'].nunique():,}, sessions={sessions:,}; "
        f"saved={combined}",
        flush=True,
    )
    return panel


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
            panel = build_full_history_panel_chunked(
                base,
                out,
                master,
                chunk_size=100,
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
