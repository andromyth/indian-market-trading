"""Autonomous Beta V2 daily predict/validate cycle.

This job is isolated from the current production models. It reads the newest
official NSE Delivery PIT snapshot, durable Beta V2 training/realised artefacts,
and point-in-time global/news/F&O layers, then writes only under the
selfdriving/beta_version2 Drive prefix.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/tmp/Indian_Market_Research/selfdriving/beta_version2")
ROOT.mkdir(parents=True, exist_ok=True)
REMOTE_BASE = "selfdriving/beta_version2"


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


def upload(store, local: Path, remote: str,
           mime: str = "application/octet-stream") -> None:
    store.upload_or_update(local, remote, mime_type=mime)


def pull_latest_pit_snapshot(store) -> Path | None:
    """Pull exactly one newest official NSE Delivery PIT snapshot."""
    parent = store.root_id
    for part in ("features", "daily"):
        hit = store._find_child(
            parent,
            part,
            "application/vnd.google-apps.folder",
        )
        if not hit:
            return None
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
        item for item in resp.get("files", [])
        if str(item.get("name", "")).endswith(".parquet")
    ][:1]
    if not files:
        return None

    item = files[0]
    local = ROOT / "pit_history" / str(item["name"])
    local.parent.mkdir(parents=True, exist_ok=True)

    if not local.exists() or local.stat().st_size == 0:
        from googleapiclient.http import MediaIoBaseDownload

        with local.open("wb") as fh:
            dl = MediaIoBaseDownload(
                fh,
                store.service.files().get_media(fileId=item["id"]),
            )
            done = False
            while not done:
                _, done = dl.next_chunk()

    return local


def pull_sources(store) -> dict:
    sources = {
        "global_cues": (
            f"{REMOTE_BASE}/global_cues.parquet",
            ROOT / "global_cues.parquet",
        ),
        "raw_news": (
            f"{REMOTE_BASE}/raw_news.parquet",
            ROOT / "raw_news.parquet",
        ),
        "fno_daily": (
            "production_data/nse_official/fno/fno_daily_features_2026.parquet",
            ROOT / "fno_daily_features_2026.parquet",
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
        result[key] = (
            "OK" if download_file(store, remote, local) else "MISSING"
        )

    # The registry points to the actual champion bundle filename. Pull that
    # bundle too; Registry.champion_bundle() will otherwise fail at inference
    # even though registry.json itself is present.
    registry_path = ROOT / "models" / "registry.json"
    if registry_path.exists():
        try:
            state = json.loads(registry_path.read_text(encoding="utf-8"))
            champion = state.get("champion") or {}
            bundle_name = str(champion.get("file", "")).strip()
            if bundle_name:
                result["champion_bundle"] = (
                    "OK"
                    if download_file(
                        store,
                        f"{REMOTE_BASE}/models/{bundle_name}",
                        ROOT / "models" / bundle_name,
                    )
                    else "MISSING"
                )
            else:
                result["champion_bundle"] = "NO_CHAMPION"
        except Exception as exc:
            result["champion_bundle"] = f"ERROR:{type(exc).__name__}"

    latest_pit = pull_latest_pit_snapshot(store)
    result["current_pit"] = "OK" if latest_pit else "MISSING"
    return result, latest_pit


def refresh_global(root: Path) -> dict:
    from selfdriving.sd_global_cues import fetch_global_daily

    path = root / "global_cues.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    start = (
        (
            pd.to_datetime(old["date"]).max()
            - pd.Timedelta(days=10)
        ).strftime("%Y-%m-%d")
        if len(old)
        else "2018-01-01"
    )

    fresh = fetch_global_daily(
        start,
        (datetime.now(timezone.utc) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if len(fresh):
        merged = (
            pd.concat([old, fresh], ignore_index=True)
            .drop_duplicates(["key", "date"], keep="last")
            .sort_values(["key", "date"])
        )
        merged.to_parquet(path, index=False)

    return {
        "rows": int(len(fresh)),
        "failed_series": fresh.attrs.get("failed", {}),
    }


def refresh_news(root: Path) -> dict:
    from selfdriving.sd_news_collect import build_alias_map, collect
    from nse_data_layer import fetch_nse_equity_universe

    path = root / "raw_news.parquet"
    universe = fetch_nse_equity_universe(force_refresh=False)
    return collect(build_alias_map(universe), path)


def main() -> int:
    status = {
        "version": "2026-10-06-beta-v2-daily",
        "started": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "production_promotion": False,
        "production_artifacts_changed": False,
    }

    store = drive()
    pull_status, latest_pit = pull_sources(store)
    status["stages"]["pull"] = pull_status

    try:
        status["stages"]["global_cues"] = refresh_global(ROOT)
    except Exception as exc:
        status["stages"]["global_cues"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        status["stages"]["news"] = refresh_news(ROOT)
    except Exception as exc:
        status["stages"]["news"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    pit_path = latest_pit
    if pit_path is None:
        status["stages"]["inference"] = {
            "status": "BLOCKED",
            "reason": "No current official NSE Delivery PIT snapshot found in Drive",
        }
    else:
        try:
            from selfdriving.sd_v2_panel import (
                build_beta_delivery_panel,
                split_training_inference,
            )
            from selfdriving.sd_cycle import predict_stage, validate_stage

            pit = pd.read_parquet(pit_path)
            global_path = ROOT / "global_cues.parquet"
            news_path = ROOT / "raw_news.parquet"
            fno_path = ROOT / "fno_daily_features_2026.parquet"
            realised_path = ROOT / "BETA_V2_DELIVERY_REALISED_RETURNS.parquet"
            registry_path = ROOT / "models"

            global_cues = (
                pd.read_parquet(global_path)
                if global_path.exists()
                else pd.DataFrame()
            )
            news = (
                pd.read_parquet(news_path)
                if news_path.exists()
                else pd.DataFrame()
            )
            fno = (
                pd.read_parquet(fno_path)
                if fno_path.exists()
                else pd.DataFrame()
            )

            panel, audit = build_beta_delivery_panel(
                pit,
                global_cues=global_cues,
                raw_news=news,
                fno_daily=fno,
            )
            _, inference, _ = split_training_inference(panel)

            inference_path = ROOT / "BETA_V2_DELIVERY_INFERENCE_PANEL.parquet"
            inference.to_parquet(inference_path, index=False)

            reg_file = registry_path / "registry.json"
            if not reg_file.exists():
                status["stages"]["inference"] = {
                    "status": "NO_CHAMPION",
                    "reason": (
                        "Run COLAB_SD_V2_RETRAIN.py successfully "
                        "before Beta V2 live inference."
                    ),
                }
            else:
                predictions = predict_stage(inference, registry_path)

                ledger_path = ROOT / "ledger" / "predictions.parquet"
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                if ledger_path.exists():
                    old_ledger = pd.read_parquet(ledger_path)
                    predictions = (
                        pd.concat(
                            [old_ledger, predictions],
                            ignore_index=True,
                        )
                        .drop_duplicates(
                            ["symbol", "date"],
                            keep="last",
                        )
                        .sort_values(["date", "symbol"])
                    )
                predictions.to_parquet(ledger_path, index=False)

                status["stages"]["inference"] = {
                    "status": "PASS",
                    "audit": audit,
                    "rows_latest": int(
                        predictions[predictions["date"].eq(
                            predictions["date"].max()
                        )].shape[0]
                    ) if len(predictions) else 0,
                    "rows_ledger": int(len(predictions)),
                    "signals_latest": (
                        predictions["signal"].value_counts().to_dict()
                        if len(predictions) else {}
                    ),
                    "prediction_date": (
                        str(predictions["date"].max())
                        if len(predictions) else None
                    ),
                }

                if realised_path.exists():
                    realised = pd.read_parquet(realised_path)
                    if not realised.empty:
                        status["stages"]["validation"] = validate_stage(
                            predictions,
                            realised,
                            registry_path,
                        )
                    else:
                        status["stages"]["validation"] = {
                            "status": "WAITING_FOR_REALISED_RETURNS"
                        }
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

    status["finished"] = datetime.now(timezone.utc).isoformat()
    status_path = ROOT / "status_cycle.json"
    status_path.write_text(
        json.dumps(status, indent=2, default=str),
        encoding="utf-8",
    )

    upload(
        store,
        ROOT / "global_cues.parquet",
        f"{REMOTE_BASE}/global_cues.parquet",
    )
    if (ROOT / "raw_news.parquet").exists():
        upload(
            store,
            ROOT / "raw_news.parquet",
            f"{REMOTE_BASE}/raw_news.parquet",
        )

    ledger = ROOT / "ledger" / "predictions.parquet"
    if ledger.exists():
        upload(
            store,
            ledger,
            f"{REMOTE_BASE}/ledger/predictions.parquet",
        )

    for name in (
        "BETA_V2_DELIVERY_INFERENCE_PANEL.parquet",
        "status_validate.json",
        "status_cycle.json",
    ):
        path = ROOT / name
        if path.exists():
            upload(store, path, f"{REMOTE_BASE}/{name}")

    print(json.dumps(status, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
