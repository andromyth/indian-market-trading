#!/usr/bin/env python3
"""Cloud Run Job entrypoint for the single to-close retraining run.

The job downloads the native 5-minute archive, NIFTY context and F&O features
from Google Drive into ephemeral local storage, audits the requested window,
assembles the long training frame, runs ``pretrain_pipeline.py`` and uploads a
candidate bundle plus its report to ``models/to_close/runs/<run_id>/``.

It never changes the live champion. Promotion remains a separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ARCHIVE_REL = "production_data/nse_official/intraday_seed/olo_5m_archive/symbols"
FNO_REL = "production_data/nse_official/fno"
NIFTY_SYMBOL = "__NIFTY50__"
NIFTY_FILES = (
    "nifty50_5m_merged_close_only.parquet",
    "nifty50_kaggle_5m_close_only.parquet",
)
FNO_FILES = (
    "fno_daily_features_2024_2026.parquet",
    "fno_daily_features_2025_2026.parquet",
    "fno_daily_features_2024_2025.parquet",
    "fno_daily_features_2026.parquet",
)
INDEX_SYMBOLS = {
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
    "NIFTY50",
}
REMOTE_MODELS = "models/to_close"
FOLDER_MIME = "application/vnd.google-apps.folder"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_symbol(value: object) -> str:
    text = str(value or "").strip().upper()
    for suffix in (".NS", ".BO"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_window(sessions: int) -> tuple[int, int]:
    """Return (min_train_sessions, holdout_sessions) for the available window."""
    if sessions >= 500:
        return 150, 60
    if sessions >= 200:
        return 80, 40
    raise ValueError(f"only {sessions} covered sessions; at least 200 are required")


def choose_universe(
    available: dict[str, Any],
    fno: pd.DataFrame,
    *,
    universe: str = "fno",
    max_symbols: int = 300,
) -> list[str]:
    """Pick trainable archive symbols, defaulting to the liquid F&O universe."""
    names = set(available)
    if universe == "fno":
        if "symbol" not in fno.columns:
            raise RuntimeError("F&O parquet has no symbol column")
        counts = fno["symbol"].map(_norm_symbol)
        counts = counts[counts.ne("")]
        ordered = counts.value_counts().index.tolist()
        chosen = [s for s in ordered if s in names and s not in INDEX_SYMBOLS]
    elif universe == "all":
        chosen = sorted(s for s in names if s not in INDEX_SYMBOLS)
    else:
        raise ValueError("universe must be 'fno' or 'all'")
    if max_symbols > 0:
        chosen = chosen[:max_symbols]
    return chosen


def _run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN " + " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def _local_first(root: Path, relative_dir: str, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / relative_dir / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _remote_first(store: Any, relative_dir: str, names: tuple[str, ...]) -> Path | None:
    import rebuild_all_nse_intraday_5m_model_from_drive as bridge

    folder_id = bridge._resolve_folder(store, relative_dir)
    if not folder_id:
        return None
    present = {
        str(item.get("name"))
        for item in bridge._list_children(store, folder_id)
        if item.get("mimeType") != FOLDER_MIME
    }
    for name in names:
        if name in present:
            remote = f"{relative_dir}/{name}"
            if bridge._download_optional_path(store, remote, Path(store.local_root)):
                return Path(store.local_root) / remote
    return None


def _symbol_folders(store: Any) -> dict[str, str]:
    import rebuild_all_nse_intraday_5m_model_from_drive as bridge

    folder_id = bridge._resolve_folder(store, ARCHIVE_REL)
    if not folder_id:
        raise RuntimeError(f"Drive archive folder is missing: {ARCHIVE_REL}")
    out: dict[str, str] = {}
    for item in bridge._list_children(store, folder_id):
        if item.get("mimeType") != FOLDER_MIME:
            continue
        name = str(item.get("name") or "")
        if not name.startswith("symbol="):
            continue
        symbol = _norm_symbol(name.split("=", 1)[1])
        if symbol and "__NIFTY" not in symbol and symbol not in INDEX_SYMBOLS:
            out[symbol] = item["id"]
    return out


def _download_inputs(root: Path, args: argparse.Namespace) -> tuple[Path, Path, list[str]]:
    from persistent_store import GoogleDriveStore, drive_status

    preflight = drive_status(require_write=True)
    print(
        "DRIVE_PREFLIGHT="
        + json.dumps(
            {
                key: preflight.get(key)
                for key in (
                    "configured",
                    "reachable",
                    "write_ready",
                    "auth_method",
                    "write_auth_method",
                    "root_name",
                    "reason",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if (
        not preflight.get("configured")
        or preflight.get("reachable") is not True
        or preflight.get("write_ready") is not True
    ):
        raise RuntimeError("Google Drive read/write preflight failed")

    store = GoogleDriveStore()
    store.local_root = root  # used only by the download helpers below
    import rebuild_all_nse_intraday_5m_model_from_drive as bridge

    fno_path = _remote_first(store, FNO_REL, FNO_FILES)
    if fno_path is None:
        raise RuntimeError(
            "No multi-year F&O parquet found in Drive. Run "
            "nse_fno_udiff_ingest.py --rebuild-derived-start 2024-11-01 first."
        )
    nifty_path = _remote_first(
        store, f"{ARCHIVE_REL}/symbol={NIFTY_SYMBOL}", NIFTY_FILES
    )
    if nifty_path is None:
        raise RuntimeError(
            "NIFTY 50 five-minute context is missing. Run "
            "indstocks_backfill_nifty_5m.py first."
        )

    available = _symbol_folders(store)
    fno = pd.read_parquet(fno_path)
    chosen = choose_universe(
        available,
        fno,
        universe=args.universe,
        max_symbols=args.max_symbols,
    )
    if len(chosen) < 20:
        raise RuntimeError(
            f"only {len(chosen)} archive symbols overlap the {args.universe} universe; "
            "refusing to train"
        )

    print(
        "TRAIN_UNIVERSE="
        + json.dumps(
            {
                "mode": args.universe,
                "archive_symbols": len(available),
                "fno_symbols": len(chosen),
                "max_symbols": args.max_symbols,
                "first_symbols": chosen[:10],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    downloaded = 0
    for index, symbol in enumerate(chosen, 1):
        _, size = bridge._download_tree(
            store,
            available[symbol],
            f"{ARCHIVE_REL}/symbol={symbol}",
            root,
        )
        if size > 0:
            downloaded += 1
        if index % 25 == 0:
            print(
                f"DOWNLOAD_PROGRESS symbols={index}/{len(chosen)} "
                f"with_data={downloaded}",
                flush=True,
            )
    if downloaded < 20:
        raise RuntimeError(f"only {downloaded} symbol archives downloaded")
    return fno_path, nifty_path, chosen


def _upload_run(
    store: Any,
    run_id: str,
    status: dict[str, Any],
    artifacts: list[Path],
    status_path: Path,
) -> dict[str, Any]:
    remote_base = f"{REMOTE_MODELS}/runs/{run_id}"
    uploaded = []
    for path in artifacts:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        remote = f"{remote_base}/{path.name}"
        result = store.upload_or_update(
            path,
            remote,
            mime_type="application/json" if path.suffix == ".json" else "application/octet-stream",
        )
        uploaded.append(
            {
                "name": path.name,
                "remote_path": remote,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "drive_status": result.get("status"),
            }
        )
    latest = {
        **status,
        "updated_at_utc": _now(),
        "remote_base": remote_base,
        "artifacts": uploaded,
    }
    latest_path = status_path.parent / "latest.json"
    latest_path.write_text(json.dumps(latest, indent=2, sort_keys=True), encoding="utf-8")
    store.upload_or_update(
        latest_path,
        f"{REMOTE_MODELS}/latest.json",
        mime_type="application/json",
    )
    return {"uploaded": uploaded, "latest": f"{REMOTE_MODELS}/latest.json"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("fno", "all"), default="fno")
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--max-symbols", type=int, default=300)
    parser.add_argument("--cost", type=float, default=0.0011)
    parser.add_argument("--warmup-bars", type=int, default=3)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--max-train-rows", type=int, default=2_500_000)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp/to_close"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--drive-root", type=Path, default=None, help="Local mounted Drive for debugging")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args(argv)

    work = args.work_root.resolve()
    data_root = args.drive_root.resolve() if args.drive_root else work / "drive"
    out = (args.out or work / "out").resolve()
    log_dir = work / "logs"
    if not args.drive_root:
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        data_root.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    status_path = work / "retrain_status.json"
    status: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": _now(),
        "universe": args.universe,
        "sessions_requested": args.sessions,
        "max_symbols": args.max_symbols,
        "cost": args.cost,
        "warmup_bars": args.warmup_bars,
        "status": "STARTED",
    }
    store = None
    artifacts: list[Path] = []
    log_path = log_dir / "retrain.log"

    def write_status(**updates: Any) -> None:
        status.update(updates)
        status["updated_at_utc"] = _now()
        status_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")

    def finish(code: int) -> int:
        write_status(finished_at_utc=_now())
        if status_path not in artifacts:
            artifacts.append(status_path)
        if store is not None and not args.no_upload:
            try:
                uploaded = _upload_run(store, run_id, status, artifacts, status_path)
                status.update(uploaded)
                status_path.write_text(
                    json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
                )
            except Exception as exc:
                status["upload_error"] = f"{type(exc).__name__}: {exc}"
                status_path.write_text(
                    json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
                )
                print("UPLOAD_FAILED " + status["upload_error"], flush=True)
                return 1
        print("RETRAIN_STATUS=" + json.dumps(status, sort_keys=True), flush=True)
        return code

    try:
        if args.drive_root:
            fno_path = _local_first(data_root, FNO_REL, FNO_FILES)
            nifty_path = _local_first(
                data_root, f"{ARCHIVE_REL}/symbol={NIFTY_SYMBOL}", NIFTY_FILES
            )
            if fno_path is None or nifty_path is None:
                raise RuntimeError("local Drive root is missing the F&O or NIFTY input")
            available = {
                _norm_symbol(path.name.split("=", 1)[1]): path
                for path in (data_root / ARCHIVE_REL).glob("symbol=*")
                if path.is_dir()
                and "__NIFTY" not in path.name.upper()
                and _norm_symbol(path.name.split("=", 1)[1]) not in INDEX_SYMBOLS
            }
            chosen = choose_universe(
                available,
                pd.read_parquet(fno_path),
                universe=args.universe,
                max_symbols=args.max_symbols,
            )
        else:
            if args.no_upload:
                raise RuntimeError("--no-upload is only valid with --drive-root")
            from persistent_store import GoogleDriveStore

            store = GoogleDriveStore()
            fno_path, nifty_path, chosen = _download_inputs(data_root, args)
            store.local_root = data_root

        status.update(
            {
                "fno_parquet": str(fno_path),
                "nifty_parquet": str(nifty_path),
                "symbols_selected": len(chosen),
            }
        )
        write_status()

        audit_json = work / "input_audit.json"
        audit_md = work / "input_audit.md"
        artifacts.extend([audit_json, audit_md])
        audit_cmd = [
            sys.executable,
            "audit_retraining_inputs.py",
            "--drive-root",
            str(data_root),
            "--fno-parquet",
            str(fno_path),
            "--nifty-parquet",
            str(nifty_path),
            "--max-equity-symbols",
            "40",
            "--json-out",
            str(audit_json),
            "--md-out",
            str(audit_md),
        ]
        if _run(audit_cmd, log_dir / "audit.log") != 0:
            write_status(status="DATA_NOT_READY", reason="input audit did not pass")
            return finish(2)

        report = json.loads(audit_json.read_text(encoding="utf-8"))
        ready = report.get("largest_ready_window") or {}
        sessions = int(ready.get("sessions") or 0)
        status.update(
            {
                "audit_verdict": report.get("verdict"),
                "covered_sessions": sessions,
                "covered_start": ready.get("start"),
                "covered_end": ready.get("end"),
            }
        )
        if sessions < 200:
            write_status(status="DATA_NOT_READY", reason="fewer than 200 covered sessions")
            return finish(2)
        min_train, holdout = training_window(sessions)
        status.update({"min_train_sessions": min_train, "holdout_sessions": holdout})

        bars_dir = work / "bars_5m"
        assemble_cmd = [
            sys.executable,
            "assemble_native_5m_bars.py",
            "--drive-root",
            str(data_root),
            "--out-dir",
            str(bars_dir),
            "--sessions",
            str(args.sessions),
            "--clean",
            "--symbols",
            ",".join(chosen),
        ]
        if _run(assemble_cmd, log_dir / "assemble.log") != 0:
            write_status(status="ASSEMBLE_FAILED")
            return finish(1)
        bars_manifest = bars_dir / "manifest.json"
        artifacts.append(bars_manifest)
        if not list((bars_dir / "parts").glob("part-*.parquet")):
            raise RuntimeError("assembler produced no training parts")

        train_cmd = [
            sys.executable,
            "pretrain_pipeline.py",
            "--backfill-parquet",
            str(bars_dir / "parts"),
            "--market-parquet",
            str(nifty_path),
            "--fno-parquet",
            str(fno_path),
            "--out",
            str(out),
            "--cost",
            str(args.cost),
            "--warmup-bars",
            str(args.warmup_bars),
            "--min-train-sessions",
            str(min_train),
            "--holdout-sessions",
            str(holdout),
            "--n-splits",
            str(args.n_splits),
            "--stride",
            str(args.stride),
            "--max-train-rows",
            str(args.max_train_rows),
        ]
        if _run(train_cmd, log_dir / "train.log") != 0:
            write_status(status="TRAINING_FAILED")
            return finish(1)

        training_report = out / "training_report.json"
        bundle = out / "candidate_bundle.joblib"
        artifacts.extend([training_report, bundle, log_path])
        if not bundle.is_file() or not training_report.is_file():
            raise RuntimeError("trainer did not write the candidate bundle and report")
        result = json.loads(training_report.read_text(encoding="utf-8")).get("result", {})
        gate = result.get("gate") or {}
        status.update(
            {
                "status": "COMPLETED_PASS" if result.get("verdict") == "PASS" else "COMPLETED_FAIL",
                "training_verdict": result.get("verdict"),
                "gate": gate,
                "holdout_auc": result.get("holdout_auc"),
                "holdout_trading": result.get("holdout_trading"),
                "candidate_bundle_sha256": _sha256(bundle),
                "training_report_sha256": _sha256(training_report),
            }
        )
        write_status()
        return finish(0)
    except Exception as exc:
        traceback.print_exc()
        write_status(
            status="FAILED",
            reason=f"{type(exc).__name__}: {exc}",
        )
        artifacts.extend([log_path])
        return finish(1)


if __name__ == "__main__":
    raise SystemExit(main())
