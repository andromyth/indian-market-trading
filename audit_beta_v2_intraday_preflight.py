"""Read-only Google Drive preflight for Beta V2 intraday retraining.

This audits source archive presence and horizon champion registries. It never
writes to Drive, changes a registry, trains a model, or promotes a candidate.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import PurePosixPath

from googleapiclient.http import MediaIoBaseDownload

from delivery_training_universe import official_nse_symbols
from persistent_store import GoogleDriveStore

FOLDER_MIME = "application/vnd.google-apps.folder"
SOURCE_PATH = (
    "production_data/nse_official/intraday_seed/"
    "olo_minute_archive/symbols"
)
BETA_ROOT = "selfdriving/beta_version2/intraday"
HORIZONS = (5, 10, 15, 20, 25, 30)


def child(store, parent_id: str, name: str, mime_type: str | None = None):
    return store._find_child(parent_id, name, mime_type)


def resolve_folder(store, path: str):
    parent = store.root_id
    for part in (p for p in path.split("/") if p):
        hit = child(store, parent, part, FOLDER_MIME)
        if not hit:
            return None
        parent = hit["id"]
    return parent


def list_children(store, folder_id: str):
    token = None
    while True:
        response = store.service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            spaces="drive",
            fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)",
            pageSize=1000,
            pageToken=token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        yield from response.get("files", [])
        token = response.get("nextPageToken")
        if not token:
            break


def parquet_inventory(store, folder_id: str):
    """Count source parquet files without downloading market history."""
    stack = [folder_id]
    parquet_files = 0
    total_bytes = 0
    folders = 0
    errors = []
    while stack:
        current = stack.pop()
        try:
            for item in list_children(store, current):
                if item.get("mimeType") == FOLDER_MIME:
                    folders += 1
                    stack.append(item["id"])
                elif item.get("name", "").lower().endswith(".parquet"):
                    parquet_files += 1
                    total_bytes += int(item.get("size") or 0)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return {
        "parquet_files": parquet_files,
        "total_bytes": total_bytes,
        "nested_folders": folders,
        "listing_errors": errors,
    }


def read_json(store, file_id: str):
    request = store.service.files().get_media(fileId=file_id)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return json.loads(buffer.getvalue().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="beta_v2_intraday_preflight.json")
    args = parser.parse_args()

    report = {
        "audit": "beta_v2_intraday_retraining_preflight",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "READ_ONLY_NO_TRAINING_NO_PROMOTION",
        "source_path": SOURCE_PATH,
        "beta_root": BETA_ROOT,
        "horizons": {},
    }

    store = GoogleDriveStore()
    report["drive"] = {
        "root_accessible": True,
        "root_name": store.service.files().get(
            fileId=store.root_id, fields="name"
        ).execute().get("name"),
    }

    universe = official_nse_symbols(force_refresh=True)
    report["official_nse_universe"] = {
        "symbol_count": len(universe),
        "source": "official_nse_symbols(force_refresh=True)",
    }
    if not universe:
        raise RuntimeError("Official NSE universe is empty; refusing to assess coverage.")

    source_folder = resolve_folder(store, SOURCE_PATH)
    if source_folder is None:
        report["history_archive"] = {
            "status": "MISSING",
            "eligible_symbol_folders": 0,
            "coverage_pct": 0.0,
        }
    else:
        children = list(list_children(store, source_folder))
        symbol_folders = {
            str(item["name"]).removeprefix("symbol=").strip().upper(): item
            for item in children
            if item.get("mimeType") == FOLDER_MIME
            and str(item.get("name", "")).startswith("symbol=")
        }
        official = set(universe)
        covered = official.intersection(symbol_folders)
        report["history_archive"] = {
            "status": "PRESENT",
            "symbol_folders_found": len(symbol_folders),
            "official_symbols_with_folder": len(covered),
            "official_symbols_without_folder": sorted(official - covered),
            "folder_coverage_pct": round(100 * len(covered) / len(official), 3),
            "coverage_warning": (
                "Folder presence is not proof of 200+ usable sessions. "
                "The trainer must verify rows, session continuity and labels."
            ),
            "per_symbol_file_inventory": {},
        }
        for sym in sorted(covered):
            inv = parquet_inventory(store, symbol_folders[sym]["id"])
            report["history_archive"]["per_symbol_file_inventory"][sym] = inv

    for horizon in HORIZONS:
        entry = {
            "registry_path": f"{BETA_ROOT}/models/h{horizon}/registry.json",
            "registry_status": "MISSING",
            "champion_present": False,
            "champion_bundle_present": False,
        }
        folder = resolve_folder(store, f"{BETA_ROOT}/models/h{horizon}")
        if folder:
            registry_file = child(store, folder, "registry.json")
            if registry_file:
                try:
                    registry = read_json(store, registry_file["id"])
                    champion = registry.get("champion")
                    entry["registry_status"] = "VALID_JSON"
                    entry["champion_present"] = bool(champion)
                    if champion:
                        bundle_name = str(champion.get("file") or "").strip()
                        bundle = child(store, folder, bundle_name) if bundle_name else None
                        entry.update({
                            "champion_name": champion.get("name"),
                            "champion_created": champion.get("created"),
                            "champion_metrics": champion.get("metrics"),
                            "champion_gate": champion.get("gate"),
                            "champion_bundle_name": bundle_name or None,
                            "champion_bundle_present": bool(bundle),
                        })
                except Exception as exc:
                    entry["registry_status"] = "INVALID"
                    entry["error"] = f"{type(exc).__name__}: {exc}"
        report["horizons"][str(horizon)] = entry

    history = report.get("history_archive", {})
    missing_champions = [
        h for h, x in report["horizons"].items()
        if not x["champion_present"] or not x["champion_bundle_present"]
    ]
    listing_errors = sum(
        len(v.get("listing_errors", []))
        for v in history.get("per_symbol_file_inventory", {}).values()
    )
    report["missing_or_invalid_champion_horizons"] = missing_champions
    report["history_archive_listing_errors"] = listing_errors
    history_complete = (
        history.get("status") == "PRESENT"
        and history.get("folder_coverage_pct", 0) >= 90.0
        and listing_errors == 0
    )
    report["status"] = (
        "INVENTORY_COMPLETE_REQUIRES_SESSION_VALIDATION"
        if history_complete and not missing_champions
        else "NEEDS_ATTENTION"
    )
    report["retraining_authorized_by_this_audit"] = False
    report["promotion_performed"] = False

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    # Exit 0 means the audit ran and wrote a report; readiness is in report.status.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
