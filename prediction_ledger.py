"""Persistent, idempotent prediction ledger with optional Google Drive sync."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

LOCAL_DIR = Path(__file__).resolve().parent / "data" / "prediction_ledger"
LOCAL_LEDGER = LOCAL_DIR / "prediction_ledger.jsonl"
REMOTE_LEDGER = "production_data/predictions/prediction_ledger.jsonl"
LOCAL_VALIDATION = LOCAL_DIR / "prediction_validation.jsonl"
REMOTE_VALIDATION = "production_data/predictions/prediction_validation.jsonl"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(record: dict[str, Any]) -> str:
    keys = [
        str(record.get("symbol", "")),
        str(record.get("mode", "")),
        str(record.get("prediction_horizon", "")),
        str(record.get("decision_timestamp", "")),
        str(record.get("model_version", "")),
    ]
    return hashlib.sha256("|".join(keys).encode("utf-8")).hexdigest()[:24]


def _ensure_local() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_LEDGER.touch(exist_ok=True)


def _drive_file_bytes(remote_relative_path: str) -> bytes | None:
    """Download a Drive file using the same OAuth/service-account resolver as the store."""
    try:
        from persistent_store import GoogleDriveStore

        g = GoogleDriveStore()
        parts = remote_relative_path.strip("/").split("/")
        parent = g.root_id

        for part in parts[:-1]:
            found = g._find_child(parent, part)
            if not found:
                return None
            parent = found["id"]

        name = parts[-1]
        found = g._find_child(parent, name)
        if not found:
            return None

        from googleapiclient.http import MediaIoBaseDownload
        import io

        request = g.service.files().get_media(fileId=found["id"])
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        return buf.getvalue()
    except Exception:
        return None


def _restore_remote_prediction_ledger() -> None:
    _ensure_local()
    if LOCAL_LEDGER.stat().st_size:
        return
    payload = _drive_file_bytes(REMOTE_LEDGER)
    if payload:
        LOCAL_LEDGER.write_bytes(payload)


def load_records() -> list[dict[str, Any]]:
    _restore_remote_prediction_ledger()
    _ensure_local()
    records: list[dict[str, Any]] = []
    with LOCAL_LEDGER.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def append_prediction(record: dict[str, Any], sync_drive: bool = True) -> dict[str, Any]:
    _ensure_local()
    x = dict(record)
    x.setdefault("created_at", _utc())
    x["prediction_id"] = x.get("prediction_id") or _stable_id(x)

    existing = {r.get("prediction_id") for r in load_records()}
    if x["prediction_id"] in existing:
        return {"status": "EXISTS", "prediction_id": x["prediction_id"], "synced": False}

    with LOCAL_LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, default=str, separators=(",", ":")) + "\n")

    synced = False
    sync_error = None
    if sync_drive:
        try:
            from persistent_store import GoogleDriveStore
            g = GoogleDriveStore()
            remote = g.upload_or_update(LOCAL_LEDGER, REMOTE_LEDGER, mime_type="application/x-ndjson")
            synced = remote.get("status") == "OK"
        except Exception as exc:
            sync_error = f"{type(exc).__name__}: {exc}"

    out = {"status": "APPENDED", "prediction_id": x["prediction_id"], "synced": synced}
    if sync_error:
        out["sync_error"] = sync_error
    return out


def save_validation_results(results: list[dict[str, Any]], sync_drive: bool = True) -> dict[str, Any]:
    """Persist the latest validation result per prediction ID as an append-safe state file."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    latest: dict[str, dict[str, Any]] = {}
    if LOCAL_VALIDATION.exists():
        with LOCAL_VALIDATION.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("prediction_id"):
                        latest[str(r["prediction_id"])] = r
                except json.JSONDecodeError:
                    continue
    for r in results:
        pid = r.get("prediction_id")
        if pid:
            x = dict(r)
            x["validated_at"] = _utc()
            latest[str(pid)] = x
    tmp = LOCAL_VALIDATION.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in latest.values():
            f.write(json.dumps(r, default=str, separators=(",", ":")) + "\n")
    os.replace(tmp, LOCAL_VALIDATION)
    synced = False
    err = None
    if sync_drive:
        try:
            from persistent_store import GoogleDriveStore
            g = GoogleDriveStore()
            out = g.upload_or_update(LOCAL_VALIDATION, REMOTE_VALIDATION, mime_type="application/x-ndjson")
            synced = out.get("status") == "OK"
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
    result = {"status": "SAVED", "rows": len(latest), "synced": synced}
    if err:
        result["sync_error"] = err
    return result


def load_validation_records() -> list[dict[str, Any]]:
    if not LOCAL_VALIDATION.exists() or LOCAL_VALIDATION.stat().st_size == 0:
        payload = _drive_file_bytes(REMOTE_VALIDATION)
        if payload:
            LOCAL_VALIDATION.write_bytes(payload)
    if not LOCAL_VALIDATION.exists():
        return []
    rows = []
    with LOCAL_VALIDATION.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def dataframe() -> pd.DataFrame:
    rows = load_records()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summary(records: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(records) if records is not None else load_records()
    return {
        "total_predictions": len(rows),
        "delivery_predictions": sum(str(r.get("mode", "")).lower() == "delivery" for r in rows),
        "intraday_predictions": sum(str(r.get("mode", "")).lower() == "intraday" for r in rows),
        "last_prediction_at": max((str(r.get("created_at", "")) for r in rows), default=None),
    }
