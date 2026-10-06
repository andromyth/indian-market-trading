"""Autonomous Beta V2 daily prediction/validation job for GitHub Actions.

This job is deliberately isolated under selfdriving/beta_version2 in Google Drive.
It never edits the current production model or main-branch artifacts.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/tmp/Indian_Market_Research/selfdriving/beta_version2")
ROOT.mkdir(parents=True, exist_ok=True)

REMOTE_BASE = "selfdriving/beta_version2"
PIT_DIR = "features/daily"


def drive():
    from persistent_store import GoogleDriveStore
    return GoogleDriveStore()


def download_file(store, remote: str, local: Path) -> bool:
    parts = [p for p in remote.strip("/").split("/") if p]
    parent = store.root_id
    for part in parts[:-1]:
        hit = store._find_child(parent, part)
        if not hit:
            return False
        parent = hit["id"]
    hit = store._find_child(parent, parts[-1])
    if not hit:
        return False

    from googleapiclient.http import MediaIoBaseDownload
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + ".tmp")
    with tmp.open("wb") as fh:
        dl = MediaIoBaseDownload(
            fh,
            store.service.files().get_media(fileId=hit["id"]),
        )
        done = False
        while not done:
            _, done = dl.next_chunk()
    tmp.replace(local)
    return True


def upload(store, local: Path, remote: str, mime: str = "application/octet-stream") -> None:
    store.upload_or_update(local, remote, mime_type=mime)


def pull_latest_pit(store) -> Path | None:
    parent = store.root_id
    for part in ("features", "daily"):
        hit = store._find_child(parent, part, "application/vnd.google-apps.folder")
        if not hit:
            return None
        parent = hit["id"]

    resp = store.service.files().list(
        q=(
            f"'{parent}' in parents and "
            "name contains 'DELIVERY_PIT_FEATURES_' and trashed = false"
        ),
        spaces="drive",
        fields="files(id,name,modifiedTime)",
        pageSize=1000,
        orderBy="name desc",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()

    files = [
        x for x in resp.get("files", [])
        if str(x.get("name", "")).endswith(".parquet")
    ]
    if not files:
        return None

    chosen = files[0]
    local = ROOT / "current_pit.parquet"
    from googleapiclient.http import MediaIoBaseDownload
    with local.open("wb") as fh:
        dl = MediaIoBaseDownload(
            fh,
            store.service.files().get_media(fileId=chosen["id"]),
        )
        done = False
        while not done:
            _, done = dl.next_chunk()
    return local


def pull_recent_pit_snapshots(store, limit: int = 100) -> tuple[pd.DataFrame, Path | None]:
    """Pull recent persisted official NSE PIT snapshots for live inference and realised returns."""
    parent = store.root_id
    for part in ("features", "daily"):
        hit = store._find_child(parent, part, "application/vnd.google-apps.folder")
        if not hit:
            return pd.DataFrame(), None
        parent = hit["id"]

    resp = store.service.files().list(
        q=(
            f"'{parent}' in parents and "
            "name contains 'DELIVERY_PIT_FEATURES_' and trashed = false"
        ),
        spaces="drive",
        fields="files(id,name)",
        pageSize=1000,
        orderBy="name desc",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    files = [
        x for x in resp.get("files", [])
        if str(x.get("name", "")).endswith(".parquet")
    ][: int(limit)]

    frames = []
    latest_path = None
    from googleapiclient.http import MediaIoBaseDownload

    for i, item in enumerate(files):
        local = ROOT / "pit_history" / str(item["name"])
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists() or local.stat().st_size == 0:
            with local.open("wb") as fh:
                dl = MediaIoBaseDownload(
                    fh,
                    store.service.files().get_media(fileId=item["id"]),
                )
                done = False
                while not done:
                    _, done = dl.next_chunk()

        try:
            x = pd.read_parquet(local)
            if {"symbol", "close"}.issubset(x.columns):
                if "date" not in x.columns:
                    name = str(item["name"])
                    m = name.replace("DELIVERY_PIT_FEATURES_", "").replace(".parquet", "")
                    x["date"] = pd.Timestamp(m)
                x = x[["symbol", "date", "close"]].copy()
                x["symbol"] = (
                    x["symbol"].astype(str).str.upper().str.strip()
                    .str.replace(".NS", "", regex=False)
                )
                x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
                x["close"] = pd.to_numeric(x["close"], errors="coerce")
                frames.append(x.dropna(subset=["symbol", "date", "close"]))
                if i == 0:
                    latest_path = local
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(), latest_path

    hist = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    hist["y_return"] = (
        hist.groupby("symbol", sort=False)["close"].shift(-1) / hist["close"] - 1.0
    )
    realised = (
        hist.dropna(subset=["y_return"])[["symbol", "date", "y_return"]]
        .drop_duplicates(["symbol", "date"], keep="last")
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    realised.to_parquet(ROOT / "BETA_V2_DELIVERY_REALISED_RETURNS.parquet", index=False)
    return realised, latest_path


def pull_sources(store) -> dict:
    sources = {
        "global_cues": (f"{REMOTE_BASE}/global_cues.parquet", ROOT / "global_cues.parquet"),
        "raw_news": (f"{REMOTE_BASE}/raw_news.parquet", ROOT / "raw_news.parquet"),
        "fno_daily": (
            "production_data/nse_official/fno/fno_daily_features_2026.parquet",
            ROOT / "fno_daily_features_2026.parquet",
        ),
        "training_panel": (
            f"{REMOTE_BASE}/BETA_V2_DELIVERY_TRAINING_PANEL.parquet",
            ROOT / "BETA_V2_DELIVERY_TRAINING_PANEL.parquet",
        ),
        "realised": (
            f"{REMOTE_BASE}/BETA_V2_DELIVERY_REALISED_RETURNS.parquet",
            ROOT / "BETA_V2_DELIVERY_REALISED_RETURNS.parquet",
        ),
        "registry": (
            f"{REMOTE_BASE}/models/registry.json",
            ROOT / "models/registry.json",
        ),
        "ledger": (
            f"{REMOTE_BASE}/ledger/predictions.parquet",
            ROOT / "ledger/predictions.parquet",
        ),
    }
    result = {}
    for key, (remote, local) in sources.items():
        result[key] = "OK" if download_file(store, remote, local) else "MISSING"
    pit = pull_latest_pit(store)
    result["current_pit"] = "OK" if pit else "MISSING"
    return result


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    status = {
        "version": "2026-10-06-beta-v2-daily",
        "started": started,
        "stages": {},
        "production_promotion": False,
        "production_artifacts_changed": False,
    }

    store = drive()
    status["stages"]["pull"] = pull_sources(store)
    realised, latest_pit = pull_recent_pit_snapshots(store, limit=100)
    status["stages"]["pit_history"] = {
        "status": "PASS" if len(realised) else "NO_DATA",
        "realised_rows": int(len(realised)),
        "latest_pit": str(latest_pit) if latest_pit else None,
    }

    # Refresh global cues and headlines incrementally. Both layers are explicitly
    # point-in-time; failures remain visible rather than being imputed.
    from selfdriving.sd_global_cues import fetch_global_daily

    gp = ROOT / "global_cues.parquet"
    old = pd.read_parquet(gp) if gp.exists() else pd.DataFrame()
    if len(old):
        start = (pd.to_datetime(old["date"]).max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    else:
        start = "2018-01-01"
    global_new = fetch_global_daily(
        start,
        (datetime.now(timezone.utc) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if len(global_new):
        merged = (
            pd.concat([old, global_new], ignore_index=True)
            .drop_duplicates(["key", "date"], keep="last")
            .sort_values(["key", "date"])
        )
        merged.to_parquet(gp, index=False)
    status["stages"]["global_cues"] = {
        "rows": int(len(global_new)),
        "failed_series": global_new.attrs.get("failed", {}),
    }

    from selfdriving.sd_news_collect import build_alias_map, collect
    try:
        from nse_data_layer import fetch_nse_equity_universe
        universe = fetch_nse_equity_universe(force_refresh=False)
        news_path = ROOT / "raw_news.parquet"
        news_result = collect(build_alias_map(universe), news_path)
        status["stages"]["news"] = news_result
    except Exception as exc:
        status["stages"]["news"] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}

    pit_path = ROOT / "current_pit.parquet"
    # Prefer the latest snapshot pulled above. If it was not separately
    # materialised, use the newest local PIT file.
    if latest_pit is not None:
        pit_path = latest_pit

    if not pit_path.exists():
        status["stages"]["inference"] = {
            "status": "BLOCKED",
            "reason": "No current official NSE Delivery PIT snapshot found in Drive",
        }
    else:
        from selfdriving.sd_v2_panel import build_beta_delivery_panel, split_training_inference
        pit = pd.read_parquet(pit_path)
        global_cues = pd.read_parquet(gp) if gp.exists() else pd.DataFrame()
        news = (
            pd.read_parquet(ROOT / "raw_news.parquet")
            if (ROOT / "raw_news.parquet").exists()
            else pd.DataFrame()
        )
        fno = (
            pd.read_parquet(ROOT / "fno_daily_features_2026.parquet")
            if (ROOT / "fno_daily_features_2026.parquet").exists()
            else pd.DataFrame()
        )

        inference_panel, audit = build_beta_delivery_panel(
            pit,
            global_cues=global_cues,
            raw_news=news,
            fno_daily=fno,
        )
        _, inference, _ = split_training_inference(inference_panel)
        inference.to_parquet(
            ROOT / "BETA_V2_DELIVERY_INFERENCE_PANEL.parquet",
            index=False,
        )

        from selfdriving.sd_cycle import predict_stage, validate_stage
        registry_path = ROOT / "models"
        reg_file = registry_path / "registry.json"

        if reg_file.exists():
            try:
                predictions = predict_stage(inference, registry_path)
                ledger_path = ROOT / "ledger" / "predictions.parquet"
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                if ledger_path.exists():
                    old = pd.read_parquet(ledger_path)
                    predictions = (
                        pd.concat([old, predictions], ignore_index=True)
                        .drop_duplicates(["symbol", "date"], keep="last")
                        .sort_values(["date", "symbol"])
                    )
                predictions.to_parquet(ledger_path, index=False)

                latest = predictions[predictions["date"].eq(predictions["date"].max())].copy()
                status["stages"]["inference"] = {
                    "status": "PASS",
                    "audit": audit,
                    "rows_latest": int(len(latest)),
                    "rows_ledger": int(len(predictions)),
                    "signals_latest": latest["signal"].value_counts().to_dict(),
                    "prediction_date": str(predictions["date"].max()) if len(predictions) else None,
                }

                realised_path = ROOT / "BETA_V2_DELIVERY_REALISED_RETURNS.parquet"
                if realised_path.exists() and not realised.empty:
                    validation = validate_stage(
                        predictions,
                        realised,
                        registry_path,
                    )
                    (ROOT / "status_validate.json").write_text(
                        json.dumps(validation, indent=2, default=str),
                        encoding="utf-8",
                    )
                    status["stages"]["validation"] = validation
                else:
                    status["stages"]["validation"] = {
                        "status": "WAITING_FOR_REALISED_RETURNS"
                    }
            except Exception as exc:
                status["stages"]["inference"] = {
                    "status": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                status["stages"]["validation"] = {
                    "status": "BLOCKED_BY_INFERENCE"
                }
        else:
            status["stages"]["inference"] = {
                "status": "NO_CHAMPION",
                "reason": "Run COLAB_SD_V2_RETRAIN.py successfully before live beta predictions",
            }
            status["stages"]["validation"] = {
                "status": "NO_CHAMPION"
            }

    status["finished"] = datetime.now(timezone.utc).isoformat()
    status_path = ROOT / "status_cycle.json"
    status_path.write_text(
        json.dumps(status, indent=2, default=str),
        encoding="utf-8",
    )

    # Persist the daily artefacts under the isolated beta path.
    upload(store, gp, f"{REMOTE_BASE}/global_cues.parquet")
    if (ROOT / "raw_news.parquet").exists():
        upload(store, ROOT / "raw_news.parquet", f"{REMOTE_BASE}/raw_news.parquet")
    ledger_local = ROOT / "ledger" / "predictions.parquet"
    if ledger_local.exists():
        upload(store, ledger_local, f"{REMOTE_BASE}/ledger/predictions.parquet")
    for name in ("BETA_V2_DELIVERY_INFERENCE_PANEL.parquet", "status_validate.json", "status_cycle.json"):
        p = ROOT / name
        if p.exists():
            upload(store, p, f"{REMOTE_BASE}/{name}")
    print(json.dumps(status, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
