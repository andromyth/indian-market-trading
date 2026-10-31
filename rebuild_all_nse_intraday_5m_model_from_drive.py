#!/usr/bin/env python3
"""Rebuild the exact ALL-NSE five-minute model bundle from Drive seed data.

Downloads the existing historical seed into a temporary workspace, runs the
repository's canonical trainer, validates the 34-feature/5-minute/target_30m
contract and estimator hashes, then uploads the bundle to its canonical Drive
path. It never changes production traffic or places orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_REMOTE = "models/all_nse_intraday_5mbar_multifactor_replacement_2026"
NATIVE_5M_SEED_REMOTE = "production_data/nse_official/intraday_seed/olo_5m_archive/symbols"
NATIVE_5M_SEED_REL = Path(NATIVE_5M_SEED_REMOTE)
LIVE_5M_REMOTE = "production_data/nse_official/intraday_5m_live"
MARKET_FILES = (
    "raw/2018_2025/NIFTY50-INDEX.parquet",
    "raw/2026/NIFTY50-INDEX.parquet",
)
FNO_FILE = "production_data/nse_official/fno/fno_daily_features_2026.parquet"
MANIFEST_NAME = "ALL_NSE_INTRADAY_5MBAR_MULTIFACTOR_2026_MODEL_MANIFEST.json"
CLASSIFIER_NAME = "deployment_all_nse_intraday_5mbar_multifactor_classifier_2026.joblib"
REGRESSOR_NAME = "deployment_all_nse_intraday_5mbar_multifactor_regressor_2026.joblib"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _list_children(store: Any, folder_id: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_token = None
    while True:
        params: dict[str, Any] = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime)",
            "pageSize": 1000,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "orderBy": "name",
        }
        if page_token:
            params["pageToken"] = page_token
        response = store.service.files().list(**params).execute()
        items.extend(response.get("files") or [])
        page_token = response.get("nextPageToken")
        if not page_token:
            return items


def _resolve_folder(store: Any, relative_path: str) -> str | None:
    parent = store.root_id
    for component in (p for p in relative_path.strip("/").split("/") if p):
        found = store._find_child(
            parent, component, "application/vnd.google-apps.folder"
        )
        if not found:
            return None
        parent = found["id"]
    return parent


def _download_file(store: Any, file_meta: dict[str, Any], destination: Path) -> int:
    from googleapiclient.http import MediaIoBaseDownload

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".download")
    try:
        request = store.service.files().get_media(fileId=file_meta["id"])
        with staging.open("wb") as stream:
            downloader = MediaIoBaseDownload(stream, request, chunksize=8 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        if not staging.is_file() or staging.stat().st_size == 0:
            raise RuntimeError(f"Drive returned an empty file for {destination.name}")
        staging.replace(destination)
        return destination.stat().st_size
    finally:
        staging.unlink(missing_ok=True)


def _download_tree(store: Any, folder_id: str, remote_root: str, local_root: Path) -> tuple[int, int]:
    """Download Parquet files below a folder, preserving relative paths."""
    folders = [(folder_id, Path())]
    file_count = 0
    bytes_total = 0
    while folders:
        current_id, relative = folders.pop()
        for item in _list_children(store, current_id):
            name = str(item.get("name") or "")
            item_type = item.get("mimeType")
            child_relative = relative / name
            if item_type == "application/vnd.google-apps.folder":
                folders.append((item["id"], child_relative))
            elif name.lower().endswith(".parquet"):
                remote = Path(remote_root) / child_relative
                destination = local_root / remote
                if not destination.exists() or destination.stat().st_size == 0:
                    bytes_total += _download_file(store, item, destination)
                else:
                    bytes_total += destination.stat().st_size
                file_count += 1
    return file_count, bytes_total


def _download_optional_path(store: Any, remote_path: str, local_root: Path) -> bool:
    parts = remote_path.strip("/").split("/")
    parent = store.root_id
    for component in parts[:-1]:
        found = store._find_child(parent, component, "application/vnd.google-apps.folder")
        if not found:
            return False
        parent = found["id"]
    found = store._find_child(parent, parts[-1])
    if not found or found.get("mimeType") == "application/vnd.google-apps.folder":
        return False
    destination = local_root / remote_path
    _download_file(store, found, destination)
    return True


def _download_optional_tree(store: Any, remote_path: str, root: Path) -> tuple[int, int]:
    folder_id = _resolve_folder(store, remote_path)
    if not folder_id:
        return 0, 0
    return _download_tree(store, folder_id, remote_path, root)


def _download_native_5m_seed(store: Any, root: Path, symbols: set[str]) -> tuple[int, int, int]:
    symbols_folder = _resolve_folder(store, NATIVE_5M_SEED_REMOTE)
    if not symbols_folder:
        raise FileNotFoundError(
            f"Required native five-minute training seed folder is missing in Drive: {NATIVE_5M_SEED_REMOTE}"
        )

    children = _list_children(store, symbols_folder)
    by_symbol: dict[str, dict[str, Any]] = {}
    for item in children:
        name = str(item.get("name") or "")
        if name.startswith("symbol=") and item.get("mimeType") == "application/vnd.google-apps.folder":
            symbol = name.split("=", 1)[1].upper().replace(".NS", "")
            if symbol in symbols:
                by_symbol[symbol] = item
        elif name.lower().endswith(".parquet"):
            symbol = Path(name).stem.upper().replace(".NS", "")
            if symbol in symbols:
                by_symbol[symbol] = item

    if not by_symbol:
        raise RuntimeError(
            "Drive native five-minute seed folder contains no partitions matching the official NSE universe."
        )

    base = root / NATIVE_5M_SEED_REL
    downloaded_symbols = 0
    file_count = 0
    bytes_total = 0
    for position, (symbol, item) in enumerate(sorted(by_symbol.items()), 1):
        target_dir = base / f"symbol={symbol}"
        if item.get("mimeType") == "application/vnd.google-apps.folder":
            count, size = _download_tree(store, item["id"], str(NATIVE_5M_SEED_REL / f"symbol={symbol}"), root)
            if count:
                downloaded_symbols += 1
                file_count += count
                bytes_total += size
        else:
            size = _download_file(store, item, target_dir / str(item["name"]))
            downloaded_symbols += 1
            file_count += 1
            bytes_total += size
        if position % 100 == 0 or position == len(by_symbol):
            print(
                f"NATIVE_5M_DOWNLOAD progress={position}/{len(by_symbol)} "
                f"native_5m_symbols_with_files={downloaded_symbols} parquet_files={file_count}",
                flush=True,
            )
    return downloaded_symbols, file_count, bytes_total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--min-symbol-coverage", type=float, default=0.90)
    parser.add_argument("--min-trainable-symbols", type=int, default=100)
    parser.add_argument("--min-seed-universe-symbols", type=int, default=500)
    args = parser.parse_args()

    if not os.getenv("GOOGLE_OAUTH_TOKEN_JSON", "").strip():
        raise RuntimeError("GOOGLE_OAUTH_TOKEN_JSON is required for Drive access and upload.")
    if not (
        os.getenv("GOOGLE_DRIVE_ROOT_ID", "").strip()
        or os.getenv("GDRIVE_ROOT_FOLDER_ID", "").strip()
    ):
        raise RuntimeError("Google Drive root folder ID is not configured.")

    from persistent_store import GoogleDriveStore, drive_status
    from all_nse_intraday_model import FEATURES, model_target_contract

    preflight = drive_status(require_write=True)
    print("DRIVE_PREFLIGHT=" + json.dumps({
        key: preflight.get(key)
        for key in ("configured", "reachable", "write_ready", "auth_method", "write_auth_method", "root_name", "reason")
    }, sort_keys=True))
    if not preflight.get("configured") or preflight.get("reachable") is not True or preflight.get("write_ready") is not True:
        raise RuntimeError("Google Drive read/write preflight failed.")

    root = Path(args.work_root).resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "trained_model"
    output.mkdir(parents=True, exist_ok=True)

    store = GoogleDriveStore()
    from delivery_training_universe import official_nse_symbols
    universe = sorted({
        str(value).strip().upper().replace(".NS", "").replace(".BO", "")
        for value in official_nse_symbols(force_refresh=True)
        if str(value).strip()
    })
    if len(universe) < args.min_seed_universe_symbols:
        raise RuntimeError(
            f"Official NSE equity universe is incomplete: {len(universe)} symbols; "
            f"minimum is {args.min_seed_universe_symbols}. No model was trained or uploaded."
        )
    print("TRAINING_UNIVERSE=" + json.dumps({
        "symbol_count": len(universe),
        "source": "NSE_OFFICIAL_CURRENT_EQUITY_UNIVERSE",
        "official_current_roster_refresh": "PASS",
    }, sort_keys=True))

    native_symbols, native_files, native_bytes = _download_native_5m_seed(
        store, root, set(universe)
    )
    live_files, live_bytes = _download_optional_tree(store, LIVE_5M_REMOTE, root)
    coverage = native_symbols / len(universe)
    required_symbols = max(
        args.min_trainable_symbols,
        math.ceil(len(universe) * args.min_symbol_coverage),
    )
    print("SEED_INPUT_SUMMARY=" + json.dumps({
        "official_nse_symbols": len(universe),
        "native_5m_symbols_with_files": native_symbols,
        "native_5m_parquet_files": native_files,
        "native_5m_bytes": native_bytes,
        "live_5m_slot_files": live_files,
        "live_5m_bytes": live_bytes,
        "native_5m_symbol_coverage_pct": round(coverage * 100, 4),
    }, sort_keys=True))
    if native_symbols < required_symbols:
        raise RuntimeError(
            f"Native five-minute archive coverage is inadequate: {native_symbols}/{len(universe)} "
            f"symbols; minimum is {required_symbols}. Legacy one-minute data is not counted."
        )
    market_found = [_download_optional_path(store, path, root) for path in MARKET_FILES]
    if not any(market_found):
        raise RuntimeError("Neither expected historical NIFTY50 market-context Parquet file exists in Drive.")
    fno_found = _download_optional_path(store, FNO_FILE, root)
    print("TRAINING_INPUTS=" + json.dumps({
        "nifty_market_context_files": [path for path, ok in zip(MARKET_FILES, market_found) if ok],
        "fno_daily_features_available": fno_found,
        "native_5m_symbols_with_files": native_symbols,
        "native_5m_parquet_files": native_files,
        "live_5m_slot_files": live_files,
    }, sort_keys=True))

    # The canonical builder imports the 34-feature runtime contract and performs
    # the chronological holdout. Train locally first; do not write into the live
    # Drive model folder until every validation below has passed.
    universe_json = root / "intraday_training_universe.json"
    universe_json.write_text(json.dumps(sorted(universe)), encoding="utf-8")
    os.environ["ALL_NSE_INTRADAY_TRAINING_UNIVERSE_JSON"] = str(universe_json)
    from colab_build_all_nse_intraday_multifactor import main as train_main
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "colab_build_all_nse_intraday_multifactor.py",
            "--drive-root", str(root),
            "--out", str(output),
            "--min-rows-symbol", "300",
            "--sample-minutes", "5",
            "--max-train-rows-per-symbol", "2000",
            "--min-usable-sessions", "200",
            "--max-sessions", "500",
            "--min-bars-per-session", "50",
        ]
        train_main()
    finally:
        sys.argv = original_argv

    manifest_path = output / MANIFEST_NAME
    classifier_path = output / CLASSIFIER_NAME
    regressor_path = output / REGRESSOR_NAME
    for path in (manifest_path, classifier_path, regressor_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Training output missing or empty: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = model_target_contract()
    expected = {
        "bar_interval_minutes": 5,
        "target": "target_30m",
        "feature_count": len(FEATURES),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Trained manifest contract mismatch for {key}: {manifest.get(key)!r}")
    if manifest.get("features") != contract["features"]:
        raise RuntimeError("Trained manifest feature list does not match the live runtime.")
    if manifest.get("classifier_sha256") != _sha256(classifier_path):
        raise RuntimeError("Classifier SHA-256 does not match the trained manifest.")
    if manifest.get("regressor_sha256") != _sha256(regressor_path):
        raise RuntimeError("Regressor SHA-256 does not match the trained manifest.")
    if int(manifest.get("training_symbol_count") or 0) < required_symbols:
        raise RuntimeError("Trainable symbol coverage is below the configured safety threshold.")
    for metric in ("accuracy", "roc_auc", "mean_abs_return_error"):
        value = manifest.get("validation", {}).get(metric)
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"Holdout validation metric {metric} is missing or non-finite.")

    import joblib
    import numpy as np
    import pandas as pd

    classifier = joblib.load(classifier_path)
    regressor = joblib.load(regressor_path)
    smoke = pd.DataFrame([np.zeros(len(FEATURES), dtype=float)], columns=FEATURES)
    probabilities = classifier.predict_proba(smoke)
    returns = regressor.predict(smoke)
    if probabilities.shape != (1, 2) or len(returns) != 1 or not np.isfinite(probabilities).all() or not np.isfinite(returns).all():
        raise RuntimeError("Serialized estimators failed the inference smoke test.")

    # Upload only after validation. If any upload fails, staged deployment is
    # never requested and the production revision remains untouched.
    uploads = []
    for path in (manifest_path, classifier_path, regressor_path):
        remote = f"{MODEL_REMOTE}/{path.name}"
        result = store.upload_or_update(
            path,
            remote,
            mime_type="application/json" if path.suffix == ".json" else "application/octet-stream",
        )
        uploads.append({
            "name": path.name,
            "remote_path": remote,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "drive_status": result.get("status"),
            "drive_action": result.get("action"),
            "file_id": result.get("file_id"),
        })
    if len(uploads) != 3 or any(item["drive_status"] != "OK" for item in uploads):
        raise RuntimeError("One or more model artifacts did not persist to Google Drive.")

    report = {
        "status": "MODEL_BUNDLE_UPLOADED_AWAITING_STAGED_DEPLOYMENT_VALIDATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_remote_folder": MODEL_REMOTE,
        "model_version": manifest.get("model_version"),
        "target": manifest.get("target"),
        "bar_interval_minutes": manifest.get("bar_interval_minutes"),
        "feature_count": manifest.get("feature_count"),
        "universe_size": manifest.get("universe_size"),
        "training_symbol_count": manifest.get("training_symbol_count"),
        "coverage_pct": manifest.get("coverage_pct"),
        "universe_source": manifest.get("universe_source"),
        "training_rows": manifest.get("training_rows"),
        "holdout_rows": manifest.get("holdout_rows"),
        "trained_until": manifest.get("trained_until"),
        "holdout_start": manifest.get("holdout_start"),
        "validation": manifest.get("validation"),
        "classifier_sha256": manifest.get("classifier_sha256"),
        "regressor_sha256": manifest.get("regressor_sha256"),
        "native_5m_symbols_with_files": native_symbols,
        "native_5m_parquet_files": native_files,
        "native_5m_symbol_coverage_pct": round(coverage * 100, 4),
        "no_auto_orders": True,
        "uploads": uploads,
    }
    report_path = root / "MODEL_REBUILD_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_remote = (
        "production_data/intraday_predictions/model_rebuild_audits/"
        + datetime.now(timezone.utc).strftime("%Y/%m/")
        + f"all_nse_intraday_5m_model_rebuild_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_upload = store.upload_or_update(report_path, report_remote, mime_type="application/json")
    report["audit_report_path"] = report_remote
    report["audit_report_drive_status"] = report_upload.get("status")
    print("MODEL_BUNDLE_UPLOAD=PASS")
    print("MODEL_REBUILD_REPORT=" + json.dumps(report, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
