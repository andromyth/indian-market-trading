"""Persistent, idempotent prediction ledger with optional Google Drive sync."""
from __future__ import annotations

import hashlib
import json
import math
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


# Forecast/output fields participate in identity so a recomputed prediction for
# the same symbol, target date, decision timestamp and model is stored as a
# distinct auditable revision when its signal, price forecast or readiness changes.
_REVISION_FIELDS = (
    "prediction_for_date",
    "pit_snapshot_date",
    "reference_session",
    "signal",
    "raw_model_signal",
    "signal_status",
    "signal_reason",
    "quality_gate_block_reasons",
    "production_data_quality_gate",
    "production_data_quality_score",
    "probability_up_next_session",
    "probability_down_next_session",
    "predicted_return_next_session",
    "predicted_price_next_session",
    "predicted_price_change_next_session",
    "predicted_open",
    "predicted_high",
    "predicted_low",
    "confidence",
    "feature_count",
    "non_missing_feature_count",
    "feature_non_missing_fraction",
    "data_readiness_score",
)


def _canonical_revision_value(value: Any) -> Any:
    """Convert prediction output values to deterministic JSON-safe primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _canonical_revision_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_revision_value(item) for item in value]
    try:
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        return round(numeric, 8)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def prediction_revision_fingerprint(record: dict[str, Any]) -> str:
    """Short content fingerprint for idempotent, auditable forecast revisions."""
    payload = {
        key: _canonical_revision_value(record[key])
        for key in _REVISION_FIELDS
        if key in record
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _stable_id(record: dict[str, Any]) -> str:
    keys = [
        str(record.get("symbol", "")),
        str(record.get("mode", "")),
        str(record.get("prediction_horizon", "")),
        str(record.get("decision_timestamp", "")),
        str(record.get("model_version", "")),
        prediction_revision_fingerprint(record),
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
    x.setdefault("prediction_revision", prediction_revision_fingerprint(x))
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


def append_predictions(records: list[dict[str, Any]], sync_drive: bool = True) -> dict[str, Any]:
    """Append many predictions efficiently and sync the ledger once."""
    _ensure_local()
    existing_records = load_records()
    existing = {str(r.get("prediction_id")) for r in existing_records if r.get("prediction_id")}
    to_write: list[dict[str, Any]] = []
    seen: set[str] = set()

    for record in records:
        x = dict(record)
        x.setdefault("created_at", _utc())
        x.setdefault("prediction_revision", prediction_revision_fingerprint(x))
        x["prediction_id"] = x.get("prediction_id") or _stable_id(x)
        pid = str(x["prediction_id"])
        if pid in existing or pid in seen:
            continue
        seen.add(pid)
        to_write.append(x)

    if to_write:
        with LOCAL_LEDGER.open("a", encoding="utf-8") as f:
            for x in to_write:
                f.write(json.dumps(x, default=str, separators=(",", ":")) + "\n")

    synced = False
    sync_error = None
    if sync_drive:
        # Also retry syncing when this batch contains no new IDs. A prior run
        # can append locally and fail the Drive upload; without this retry path,
        # the next idempotent run would skip the rows forever.
        try:
            from persistent_store import GoogleDriveStore
            g = GoogleDriveStore()
            remote = g.upload_or_update(
                LOCAL_LEDGER,
                REMOTE_LEDGER,
                mime_type="application/x-ndjson",
            )
            synced = remote.get("status") == "OK"
        except Exception as exc:
            sync_error = f"{type(exc).__name__}: {exc}"

    out = {
        "status": "APPENDED" if to_write else "NO_NEW_PREDICTIONS",
        "new_rows": len(to_write),
        "existing_rows_seen": len(existing),
        "synced": synced,
    }
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



def sync_delivery_prediction_csv_ledger() -> dict[str, Any]:
    """Build and sync the daywise Delivery prediction-vs-actual CSV ledger."""
    from delivery_prediction_csv_ledger import sync_delivery_csv_ledger
    return sync_delivery_csv_ledger()
