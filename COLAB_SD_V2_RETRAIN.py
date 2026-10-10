"""Colab trainer for the isolated Beta V2 Delivery challenger.

This trainer is designed for the full NSE universe while remaining resumable on
CPU-only Colab runtimes. It never changes production model artefacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd

from selfdriving.sd_cycle import train_stage
from selfdriving.sd_global_cues import attach_global_features
from selfdriving.sd_v2_panel import (
    add_delivery_targets,
    merge_beta_fno,
    merge_beta_news,
    split_training_inference,
)
from selfdriving.sd_global_cues import attach_global_features


DEFAULT_BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
DEFAULT_PANEL = (
    DEFAULT_BASE
    / "models/all_nse_delivery_multifactor_replacement_2026"
    / "ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"
)
MIN_WALKFORWARD_SESSIONS = 422
DEFAULT_LOOKBACK_SESSIONS = 600
CHUNK_SIZE = 100


def load_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


def file_fingerprint(path: Path, hash_content: bool = False) -> dict:
    """Capture source identity so enriched-panel caches cannot silently go stale."""
    if not path.exists() or not path.is_file():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    result = {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def parquet_stats(path: Path) -> dict:
    con = duckdb.connect()
    try:
        p = str(path).replace("'", "''")
        row = con.execute(
            f"""
            SELECT
                COUNT(*) AS rows,
                COUNT(DISTINCT symbol) AS symbols,
                COUNT(DISTINCT CAST(date AS DATE)) AS sessions,
                MIN(CAST(date AS DATE)) AS min_date,
                MAX(CAST(date AS DATE)) AS max_date
            FROM read_parquet('{p}')
            """
        ).fetchone()
        return {
            "rows": int(row[0]),
            "symbols": int(row[1]),
            "sessions": int(row[2]),
            "min_date": str(row[3]),
            "max_date": str(row[4]),
        }
    finally:
        con.close()


def sql_recent_panel(path: Path, lookback_sessions: int) -> pd.DataFrame:
    """Load only the most recent N sessions from a Parquet panel."""
    p = str(path).replace("'", "''")
    con = duckdb.connect()
    try:
        return con.execute(
            f"""
            WITH sessions AS (
                SELECT CAST(date AS DATE) AS date
                FROM read_parquet('{p}')
                GROUP BY 1
                ORDER BY 1 DESC
                LIMIT {int(lookback_sessions)}
            )
            SELECT p.*
            FROM read_parquet('{p}') p
            INNER JOIN sessions s
                ON CAST(p.date AS DATE) = s.date
            ORDER BY p.symbol, p.date
            """
        ).df()
    finally:
        con.close()


def latest_rows(path: Path) -> pd.DataFrame:
    p = str(path).replace("'", "''")
    con = duckdb.connect()
    try:
        return con.execute(
            f"""
            WITH latest AS (
                SELECT MAX(CAST(date AS DATE)) AS date
                FROM read_parquet('{p}')
            )
            SELECT p.*
            FROM read_parquet('{p}') p
            INNER JOIN latest l
                ON CAST(p.date AS DATE) = l.date
            ORDER BY p.symbol
            """
        ).df()
    finally:
        con.close()


def build_beta_panel_chunked(
    out: Path,
    base_parts_dir: Path,
    global_cues: pd.DataFrame,
    raw_news: pd.DataFrame,
    fno_daily: pd.DataFrame,
    *,
    lookback_sessions: int,
) -> Path:
    """Enrich the already checkpointed base panel without loading 4M rows."""
    beta_parts = out / "beta_panel_parts"
    beta_parts.mkdir(parents=True, exist_ok=True)

    base_parts = sorted(base_parts_dir.glob("part_*.parquet"))
    if not base_parts:
        raise FileNotFoundError(f"No base panel checkpoints found under {base_parts_dir}")

    # Determine the cutoff once from the complete checkpoint set. Only the
    # configured recent training window is enriched; the full base history
    # remains preserved separately on Drive.
    glob = str(base_parts_dir / "part_*.parquet").replace("'", "''")
    con = duckdb.connect()
    try:
        cutoff_row = con.execute(
            f"""
            WITH dates AS (
                SELECT DISTINCT CAST(date AS DATE) AS date
                FROM read_parquet('{glob}')
            )
            SELECT MIN(date)
            FROM (
                SELECT date
                FROM dates
                ORDER BY date DESC
                LIMIT {int(lookback_sessions)}
            )
            """
        ).fetchone()
    finally:
        con.close()
    cutoff_date = cutoff_row[0]
    if cutoff_date is None:
        raise RuntimeError("Unable to determine Beta V2 training-window cutoff")

    print(
        f"Enriching only the latest {lookback_sessions} sessions "
        f"(cutoff={cutoff_date}) from the preserved full-history checkpoints.",
        flush=True,
    )

    total = len(base_parts)
    for idx, base_part in enumerate(base_parts, 1):
        beta_part = beta_parts / base_part.name.replace("part_", "beta_part_")

        if beta_part.exists() and beta_part.stat().st_size > 0:
            print(
                f"beta panel chunk {idx}/{total}: reused {beta_part.name}",
                flush=True,
            )
            continue

        print(
            f"beta panel chunk {idx}/{total}: enriching {base_part.name}",
            flush=True,
        )

        part_sql = str(base_part).replace("'", "''")
        con = duckdb.connect()
        try:
            chunk = con.execute(
                f"""
                SELECT *
                FROM read_parquet('{part_sql}')
                WHERE CAST(date AS DATE) >= DATE '{cutoff_date}'
                ORDER BY symbol, date
                """
            ).df()
        finally:
            con.close()

        if chunk.empty:
            # Keep an explicit empty checkpoint so future runs know this part
            # was processed for the chosen window.
            pd.DataFrame().to_parquet(beta_part, index=False)
            continue

        x = add_delivery_targets(chunk)

        if global_cues is not None and not global_cues.empty:
            x = attach_global_features(x, global_cues)

        x, news_audit = merge_beta_news(x, raw_news)
        x, fno_audit = merge_beta_fno(x, fno_daily)

        x["beta_v2_panel_version"] = "2026-10-06-beta-v2"
        x.to_parquet(beta_part, index=False)

        print(
            f"beta panel chunk {idx}/{total}: saved rows={len(x):,} "
            f"symbols={x['symbol'].nunique():,}; "
            f"news={news_audit.get('status')} fno={fno_audit.get('status')}",
            flush=True,
        )
        del chunk, x

    combined = out / f"BETA_V2_DELIVERY_PANEL_{lookback_sessions}S_FULL.parquet"
    if not combined.exists() or combined.stat().st_size == 0:
        glob = str(beta_parts / "beta_part_*.parquet").replace("'", "''")
        tmp = out / "BETA_V2_DELIVERY_PANEL_FULL_HISTORY.parquet.tmp"
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

    stats = parquet_stats(combined)
    print(
        f"Beta V2 full panel ready: rows={stats['rows']:,}, "
        f"symbols={stats['symbols']:,}, sessions={stats['sessions']:,}, "
        f"saved={combined}",
        flush=True,
    )
    return combined


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--panel", default=str(DEFAULT_PANEL))
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--lookback-sessions",
        type=int,
        default=DEFAULT_LOOKBACK_SESSIONS,
        help="Recent sessions used for OOS training. Must be >= 422; all symbols remain included.",
    )
    ap.add_argument(
        "--force-rebuild-panel",
        action="store_true",
        help="Rebuild the cached enriched Beta V2 panel even when source fingerprints match.",
    )
    args = ap.parse_args()

    if args.lookback_sessions < MIN_WALKFORWARD_SESSIONS:
        raise ValueError(
            f"--lookback-sessions must be >= {MIN_WALKFORWARD_SESSIONS}; "
            f"received {args.lookback_sessions}"
        )

    base = Path(args.base)
    out = Path(args.out) if args.out else base / "selfdriving/beta_version2"
    out.mkdir(parents=True, exist_ok=True)

    source_panel = Path(args.panel)
    source_stats = parquet_stats(source_panel)
    print(
        f"Input panel: rows={source_stats['rows']:,}, "
        f"symbols={source_stats['symbols']:,}, sessions={source_stats['sessions']:,}",
        flush=True,
    )

    if source_stats["sessions"] < MIN_WALKFORWARD_SESSIONS:
        full_base = out / "BETA_V2_DELIVERY_BASE_PANEL_FULL_HISTORY.parquet"
        base_parts = out / "base_panel_parts"

        if full_base.exists() and full_base.stat().st_size > 0:
            base_stats = parquet_stats(full_base)
            print(
                f"Using cached full-history base panel: rows={base_stats['rows']:,}, "
                f"symbols={base_stats['symbols']:,}, sessions={base_stats['sessions']:,}; "
                f"cached={full_base}",
                flush=True,
            )
        elif len(list(base_parts.glob("part_*.parquet"))) >= 24:
            print(
                "Using completed base-panel checkpoints; assembling them without "
                "rebuilding the canonical NSE source.",
                flush=True,
            )
            glob = str(base_parts / "part_*.parquet").replace("'", "''")
            tmp = out / "BETA_V2_DELIVERY_BASE_PANEL_FULL_HISTORY.parquet.tmp"
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
            tmp.replace(full_base)
            base_stats = parquet_stats(full_base)
            print(
                f"Full-history base assembled: rows={base_stats['rows']:,}, "
                f"symbols={base_stats['symbols']:,}, sessions={base_stats['sessions']:,}",
                flush=True,
            )
        else:
            raise RuntimeError(
                "The 2026 panel is too short and the full-history checkpoint set is incomplete. "
                "Rebuild the base panel before training."
            )

        if base_stats["sessions"] < MIN_WALKFORWARD_SESSIONS:
            raise RuntimeError(
                f"Full-history base has only {base_stats['sessions']} sessions; "
                f"Beta V2 requires >= {MIN_WALKFORWARD_SESSIONS}."
            )
    else:
        full_base = source_panel

    # These paths are the same archives updated by the Beta V2 daily job.
    # Do not train from the older legacy selfdriving/ copies.
    global_path = base / "selfdriving/beta_version2/global_cues.parquet"
    news_path = base / "selfdriving/beta_version2/raw_news.parquet"
    fno_path = base / "production_data/nse_official/fno/fno_daily_features_2026.parquet"

    global_cues = load_optional(global_path)
    raw_news = load_optional(news_path)
    fno_daily = load_optional(fno_path)

    beta_full = out / f"BETA_V2_DELIVERY_PANEL_{args.lookback_sessions}S_FULL.parquet"
    beta_parts = out / "beta_panel_parts"
    source_manifest_path = out / "BETA_V2_PANEL_SOURCE_MANIFEST.json"

    fingerprint_payload = {
        "schema": "beta-v2-panel-source-contract-2026-10-10-v1",
        "lookback_sessions": int(args.lookback_sessions),
        "base_panel": file_fingerprint(source_panel, hash_content=False),
        "global_cues": file_fingerprint(global_path, hash_content=True),
        "raw_news": file_fingerprint(news_path, hash_content=True),
        "fno_daily": file_fingerprint(fno_path, hash_content=True),
    }
    signature = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    previous_manifest = {}
    if source_manifest_path.exists():
        try:
            previous_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            previous_manifest = {}
    cache_exists = beta_full.exists() or beta_parts.exists()
    source_signature_changed = previous_manifest.get("signature_sha256") != signature
    if args.force_rebuild_panel or (cache_exists and source_signature_changed):
        if beta_full.exists():
            beta_full.unlink()
        if beta_parts.exists():
            shutil.rmtree(beta_parts)
        print(
            "Beta V2 enriched-panel cache invalidated because source fingerprints changed "
            "or --force-rebuild-panel was requested.",
            flush=True,
        )

    if beta_full.exists() and beta_full.stat().st_size > 0:
        beta_stats = parquet_stats(beta_full)
        print(
            f"Reusing cached Beta V2 full panel: rows={beta_stats['rows']:,}, "
            f"symbols={beta_stats['symbols']:,}, sessions={beta_stats['sessions']:,}",
            flush=True,
        )
    elif beta_parts.exists() and len(list(beta_parts.glob("beta_part_*.parquet"))) >= 24:
        print("All Beta V2 enrichment checkpoints exist; assembling them.", flush=True)
        glob = str(beta_parts / "beta_part_*.parquet").replace("'", "''")
        tmp = out / "BETA_V2_DELIVERY_PANEL_FULL_HISTORY.parquet.tmp"
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
        tmp.replace(beta_full)
    else:
        beta_full = build_beta_panel_chunked(
            out,
            out / "base_panel_parts",
            global_cues,
            raw_news,
            fno_daily,
            lookback_sessions=args.lookback_sessions,
        )

    source_manifest = {
        **fingerprint_payload,
        "signature_sha256": signature,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "enriched_panel": str(beta_full),
        "enriched_panel_size_bytes": int(beta_full.stat().st_size),
    }
    source_manifest_path.write_text(
        json.dumps(source_manifest, indent=2, default=str),
        encoding="utf-8",
    )

    # Do not load the entire multi-million-row Beta panel for model fitting.
    # Keep all NSE symbols but use a recent, contiguous session window for
    # walk-forward training. This preserves the required 8 folds while keeping
    # memory bounded on CPU-only Colab. The full panel remains persisted.
    train_panel = sql_recent_panel(beta_full, args.lookback_sessions)

    train_dates = pd.to_datetime(train_panel["date"], errors="coerce").dt.normalize()
    n_sessions = int(train_dates.nunique())
    if n_sessions < MIN_WALKFORWARD_SESSIONS:
        raise RuntimeError(
            f"Training window has only {n_sessions} sessions; "
            f"Beta V2 requires >= {MIN_WALKFORWARD_SESSIONS}."
        )

    # Recompute cross-sectional availability globally in the selected training
    # window. This prevents chunk-local counts from becoming a training gate.
    train_panel["cross_section_n"] = (
        train_panel.groupby("date")["y_return"].transform("count")
    )

    training, _, realised = split_training_inference(train_panel)
    latest = latest_rows(beta_full)
    inference = latest.copy()

    print(
        f"Training window ready: rows={len(training):,}, "
        f"symbols={training['symbol'].nunique():,}, sessions={training['date'].nunique():,}",
        flush=True,
    )

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
        "lookback_sessions": int(args.lookback_sessions),
        "full_panel": parquet_stats(beta_full),
        "training_rows": int(len(training)),
        "inference_rows": int(len(inference)),
        "realised_rows": int(len(realised)),
        "panel_source_signature": signature,
        "panel_source_manifest": str(source_manifest_path),
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
