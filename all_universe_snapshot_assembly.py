"""Explicit assembly and validation for sharded five-minute NSE snapshots.

The capture path collects the fixed owner-selected roster in non-overlapping shards.
Each shard writes its own persisted Parquet file plus a shard manifest. This
module is the single, explicit assembly step:

1. verify that every expected shard for the slot completed and persisted;
2. validate duplicate symbols, missing symbols, timestamps and row counts;
3. only then combine the shard rows into one canonical snapshot and publish it
   as the assembled artifact that the prediction compiler consumes.

A canonical snapshot is never published when any shard is missing or invalid.
Re-running assembly for the same slot is idempotent: it rewrites the same
deterministic paths from the same shard inputs.

This module is a Drive consumer/producer only. It never imports or calls
``indstocks_client``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"
QUOTE_ROOT = "production_data/all_universe_minute_snapshots"
SHARD_SUBDIR = "shards"
SCHEMA_VERSION = "all-universe-minute-snapshot-v2"
SHARD_STATUS = "SHARD_STORED"
DEFAULT_WORKER_COUNT = 8
# A shard may finish collecting a little after the slot boundary when the
# coordinated tick itself started late. Anything beyond this window means the
# observation does not belong to the slot being assembled.
OBSERVED_AT_TOLERANCE = pd.Timedelta(minutes=20)

# Issue codes that make the assembly invalid. Warning codes are recorded in the
# manifest but do not block publication.
FATAL_ISSUE_CODES = {
    "MISSING_SHARD",
    "SHARD_MINUTE_MISMATCH",
    "SHARD_IDENTITY_MISMATCH",
    "SHARD_NOT_STORED",
    "SHARD_NOT_PERSISTED",
    "SHARD_SYMBOLS_MISSING",
    "SHARD_ROW_COUNT_MISMATCH",
    "SHARD_TIMESTAMP_MISMATCH",
    "SHARD_OBSERVED_AT_OUT_OF_WINDOW",
    "DUPLICATE_SYMBOL",
    "MISSING_SYMBOL",
    "UNEXPECTED_SYMBOL",
    "UNIVERSE_COUNT_MISMATCH",
    "EMPTY_ASSEMBLY",
}
WARNING_ISSUE_CODES = {
    "ROSTER_DRIFT_MISSING",
    "ROSTER_DRIFT_EXTRA",
}


def _json_clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def normalized_cycle_minute(value: Any) -> pd.Timestamp:
    """Return the slot timestamp in IST, floored to the five-minute grid."""
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize(IST) if stamp.tzinfo is None else stamp.tz_convert(IST)
    return stamp.floor("5min")


def shard_universe(symbols: Iterable[str], worker_count: int) -> list[list[str]]:
    """Split a sorted symbol universe into ``worker_count`` disjoint shards.

    Round-robin assignment (``symbols[i::worker_count]``) keeps shard sizes
    balanced even when the universe is not a multiple of the worker count. Every
    symbol is assigned to exactly one shard, so the union is the selected roster
    and no provider request is duplicated.
    """
    count = max(1, int(worker_count))
    ordered = sorted({str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()})
    return [ordered[index::count] for index in range(count)]


def shard_artifact_names(
    cycle_minute: pd.Timestamp, worker_id: int, worker_count: int
) -> dict[str, str]:
    minute = normalized_cycle_minute(cycle_minute)
    count = max(1, int(worker_count))
    index = int(worker_id)
    if count == 1:
        stem = f"{minute:%Y%m%d_%H%M}"
        return {
            "snapshot_name": f"snapshot_{stem}.parquet",
            "manifest_name": f"manifest_{stem}.json",
            "snapshot_remote": f"{QUOTE_ROOT}/{minute:%Y/%m/%d}/snapshot_{stem}.parquet",
            "manifest_remote": f"{QUOTE_ROOT}/{minute:%Y/%m/%d}/manifest_{stem}.json",
            "sharded": False,
        }
    stem = f"{minute:%Y%m%d_%H%M}_w{index:02d}of{count:02d}"
    shard_dir = f"{QUOTE_ROOT}/{minute:%Y/%m/%d}/{SHARD_SUBDIR}"
    return {
        "snapshot_name": f"snapshot_{stem}.parquet",
        "manifest_name": f"manifest_{stem}.json",
        "snapshot_remote": f"{shard_dir}/snapshot_{stem}.parquet",
        "manifest_remote": f"{shard_dir}/manifest_{stem}.json",
        "sharded": True,
    }


def canonical_artifact_names(cycle_minute: pd.Timestamp) -> dict[str, str]:
    minute = normalized_cycle_minute(cycle_minute)
    stem = f"{minute:%Y%m%d_%H%M}"
    return {
        "snapshot_name": f"snapshot_{stem}.parquet",
        "manifest_name": f"manifest_{stem}.json",
        "snapshot_remote": f"{QUOTE_ROOT}/{minute:%Y/%m/%d}/snapshot_{stem}.parquet",
        "manifest_remote": f"{QUOTE_ROOT}/{minute:%Y/%m/%d}/manifest_{stem}.json",
        "latest_manifest_remote": f"{QUOTE_ROOT}/latest_manifest.json",
    }


def _issue(code: str, detail: str, *, worker_id: int | None = None, **extra: Any) -> dict[str, Any]:
    payload = {"code": code, "detail": detail}
    if worker_id is not None:
        payload["worker_id"] = int(worker_id)
    payload.update(extra)
    return payload


def _symbol_set(frame: pd.DataFrame | None) -> set[str]:
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return set()
    return {
        str(symbol).strip().upper()
        for symbol in frame["symbol"].tolist()
        if str(symbol).strip()
    }


def _declared_symbols(manifest: dict[str, Any]) -> list[str] | None:
    declared = manifest.get("shard_symbols")
    if not isinstance(declared, list):
        return None
    return [str(symbol).strip().upper() for symbol in declared if str(symbol).strip()]


def validate_shard(
    *,
    cycle_minute: pd.Timestamp,
    worker_count: int,
    worker_id: int,
    manifest: dict[str, Any] | None,
    frame: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Validate one shard against the slot and its own manifest."""
    minute = normalized_cycle_minute(cycle_minute)
    if manifest is None:
        return [_issue("MISSING_SHARD", "Shard manifest is absent from Drive", worker_id=worker_id)]

    issues: list[dict[str, Any]] = []

    declared_minute_raw = manifest.get("cycle_minute_ist")
    try:
        declared_minute = normalized_cycle_minute(declared_minute_raw)
    except Exception:
        declared_minute = None
    if declared_minute is None or declared_minute != minute:
        issues.append(_issue(
            "SHARD_MINUTE_MISMATCH",
            f"manifest cycle_minute_ist={declared_minute_raw!r} does not match {minute.isoformat()}",
            worker_id=worker_id,
        ))

    if int(manifest.get("worker_id", -1)) != int(worker_id):
        issues.append(_issue(
            "SHARD_IDENTITY_MISMATCH",
            f"manifest worker_id={manifest.get('worker_id')!r} expected {worker_id}",
            worker_id=worker_id,
        ))
    if int(manifest.get("worker_count", -1)) != int(worker_count):
        issues.append(_issue(
            "SHARD_IDENTITY_MISMATCH",
            f"manifest worker_count={manifest.get('worker_count')!r} expected {worker_count}",
            worker_id=worker_id,
        ))

    status = str(manifest.get("status") or "").upper()
    if status not in {SHARD_STATUS, "STORED"}:
        issues.append(_issue(
            "SHARD_NOT_STORED",
            f"manifest status={status or 'MISSING'}",
            worker_id=worker_id,
        ))
    if manifest.get("snapshot_persisted") is not True:
        issues.append(_issue(
            "SHARD_NOT_PERSISTED",
            "manifest does not confirm a persisted shard snapshot",
            worker_id=worker_id,
        ))

    declared = _declared_symbols(manifest)
    if declared is None:
        issues.append(_issue(
            "SHARD_SYMBOLS_MISSING",
            "manifest does not declare the symbols assigned to this shard",
            worker_id=worker_id,
        ))
    elif frame is None:
        issues.append(_issue("MISSING_SHARD", "Shard snapshot could not be read", worker_id=worker_id))
    else:
        observed = _symbol_set(frame)
        declared_set = set(declared)
        missing = sorted(declared_set - observed)
        unexpected = sorted(observed - declared_set)
        if missing:
            issues.append(_issue(
                "MISSING_SYMBOL",
                f"{len(missing)} declared shard symbols are absent from the shard snapshot",
                worker_id=worker_id,
                symbols=missing[:50],
            ))
        if unexpected:
            issues.append(_issue(
                "UNEXPECTED_SYMBOL",
                f"{len(unexpected)} symbols in the shard snapshot were not declared",
                worker_id=worker_id,
                symbols=unexpected[:50],
            ))
        declared_rows = manifest.get("row_count", manifest.get("shard_row_count"))
        if declared_rows is not None and int(declared_rows) != int(len(frame)):
            issues.append(_issue(
                "SHARD_ROW_COUNT_MISMATCH",
                f"manifest row_count={declared_rows} parquet rows={len(frame)}",
                worker_id=worker_id,
            ))
        if "cycle_minute_ist" in frame.columns:
            observed_minutes = pd.to_datetime(frame["cycle_minute_ist"], errors="coerce", utc=True)
            expected = minute.tz_convert("UTC")
            mismatched = int((observed_minutes != expected).sum())
            if mismatched:
                issues.append(_issue(
                    "SHARD_TIMESTAMP_MISMATCH",
                    f"{mismatched} rows carry a cycle_minute_ist other than {minute.isoformat()}",
                    worker_id=worker_id,
                ))
        if "quote_observed_at_ist" in frame.columns:
            observed_at = pd.to_datetime(
                frame["quote_observed_at_ist"], errors="coerce", utc=True
            ).dt.tz_convert(IST)
            lower = minute - pd.Timedelta(minutes=1)
            upper = minute + pd.Timedelta(minutes=5) + OBSERVED_AT_TOLERANCE
            out_of_window = observed_at.notna().to_numpy() & (
                (observed_at < lower).to_numpy() | (observed_at > upper).to_numpy()
            )
            if int(out_of_window.sum()):
                issues.append(_issue(
                    "SHARD_OBSERVED_AT_OUT_OF_WINDOW",
                    f"{int(out_of_window.sum())} quotes were observed outside the slot window",
                    worker_id=worker_id,
                ))
    return issues


def assemble_shards(
    *,
    cycle_minute: pd.Timestamp,
    worker_count: int,
    shards: Sequence[dict[str, Any]],
    expected_universe: Iterable[str] | None = None,
    strict_universe: bool = False,
) -> dict[str, Any]:
    """Validate all shards and return the canonical assembly result.

    ``shards`` entries must be dicts with ``worker_id``, ``manifest`` (dict or
    None), ``frame`` (DataFrame or None) and optionally ``snapshot_sha256``.
    """
    minute = normalized_cycle_minute(cycle_minute)
    count = max(1, int(worker_count))
    by_id: dict[int, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    for shard in shards:
        try:
            worker_id = int(shard.get("worker_id"))
        except (TypeError, ValueError):
            issues.append(_issue("MISSING_SHARD", "Shard entry has no worker_id"))
            continue
        if worker_id in by_id:
            issues.append(_issue(
                "DUPLICATE_SYMBOL",
                f"Shard {worker_id} was supplied more than once",
                worker_id=worker_id,
            ))
            continue
        by_id[worker_id] = shard

    expected_ids = set(range(count))
    missing_ids = sorted(expected_ids - set(by_id))
    for worker_id in missing_ids:
        issues.append(_issue(
            "MISSING_SHARD",
            f"Shard {worker_id} of {count} was not completed for this slot",
            worker_id=worker_id,
        ))

    frames: list[pd.DataFrame] = []
    shard_summaries: list[dict[str, Any]] = []
    universe_counts: set[int] = set()
    for worker_id in sorted(expected_ids & set(by_id)):
        shard = by_id[worker_id]
        manifest = shard.get("manifest")
        frame = shard.get("frame")
        shard_issues = validate_shard(
            cycle_minute=minute,
            worker_count=count,
            worker_id=worker_id,
            manifest=manifest,
            frame=frame,
        )
        issues.extend(shard_issues)
        if manifest:
            universe_count = manifest.get("selected_symbol_count", manifest.get("universe_symbol_count"))
            if universe_count is not None:
                try:
                    universe_counts.add(int(universe_count))
                except (TypeError, ValueError):
                    pass
        declared = _declared_symbols(manifest or {})
        summary = {
            "worker_id": worker_id,
            "status": str((manifest or {}).get("status") or "MISSING"),
            "snapshot_remote_path": (manifest or {}).get("snapshot_drive_path")
            or (manifest or {}).get("remote_snapshot_path")
            or shard.get("snapshot_remote_path"),
            "manifest_remote_path": shard.get("manifest_remote_path"),
            "row_count": int(len(frame)) if frame is not None else None,
            "declared_symbol_count": len(declared) if declared is not None else None,
            "quoted_symbol_count": int((manifest or {}).get("quoted_symbol_count") or 0),
            "snapshot_sha256": shard.get("snapshot_sha256"),
            "finished_at_utc": (manifest or {}).get("finished_at_utc"),
            "issue_count": len(shard_issues),
        }
        shard_summaries.append(summary)
        if frame is not None and not frame.empty:
            frames.append(frame)

    if len(universe_counts) > 1:
        issues.append(_issue(
            "UNIVERSE_COUNT_MISMATCH",
            f"Shards disagree about the universe size: {sorted(universe_counts)}",
        ))

    frame = pd.concat(frames, ignore_index=True) if frames else None
    duplicate_symbols: list[str] = []
    missing_symbols: list[str] = []
    if frame is not None and not frame.empty:
        if "symbol" not in frame.columns:
            issues.append(_issue("EMPTY_ASSEMBLY", "Assembled rows have no symbol column"))
        else:
            frame = frame.copy()
            frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
            counts = frame["symbol"].value_counts()
            duplicate_symbols = sorted(counts[counts > 1].index.tolist())
            if duplicate_symbols:
                issues.append(_issue(
                    "DUPLICATE_SYMBOL",
                    f"{len(duplicate_symbols)} symbols appear in more than one shard",
                    symbols=duplicate_symbols[:50],
                ))
    else:
        issues.append(_issue("EMPTY_ASSEMBLY", "No shard rows were available to assemble"))

    union_symbols: set[str] = set()
    if frame is not None and not frame.empty and "symbol" in frame.columns:
        union_symbols = set(frame["symbol"].unique().tolist())

    if expected_universe is not None:
        expected = {
            str(symbol).strip().upper()
            for symbol in expected_universe
            if str(symbol).strip()
        }
        absent = sorted(expected - union_symbols)
        extra = sorted(union_symbols - expected)
        if absent:
            issues.append(_issue(
                "UNIVERSE_DRIFT_MISSING",
                f"{len(absent)} selected-roster symbols are absent from the assembly",
                symbols=absent[:50],
            ))
            if strict_universe:
                issues.append(_issue(
                    "MISSING_SYMBOL",
                    f"{len(absent)} official universe symbols are absent from the assembly",
                    symbols=absent[:50],
                ))
        if extra:
            issues.append(_issue(
                "UNIVERSE_DRIFT_EXTRA",
                f"{len(extra)} assembled symbols are outside the official universe",
                symbols=extra[:50],
            ))

    fatal = [issue for issue in issues if issue.get("code") in FATAL_ISSUE_CODES]
    if not missing_symbols:
        missing_symbols = sorted({
            str(symbol).strip().upper()
            for issue in issues
            if issue.get("code") == "MISSING_SYMBOL"
            for symbol in (issue.get("symbols") or [])
            if str(symbol).strip()
        })
    assembled = frame is not None and not frame.empty and not fatal
    if frame is not None and not frame.empty and duplicate_symbols:
        assembled = False
    if frame is not None and not frame.empty and not duplicate_symbols:
        frame = frame.sort_values("symbol", kind="stable").reset_index(drop=True)

    expected_symbol_count = (
        max(universe_counts) if universe_counts else len(union_symbols)
    )
    quoted_symbol_count = 0
    missing_quote_count = 0
    if frame is not None and not frame.empty and "quote_status" in frame.columns:
        quoted_symbol_count = int(frame["quote_status"].eq("PASS").sum())
        missing_quote_count = int(frame["quote_status"].ne("PASS").sum())

    coverage_pct = (
        round(100.0 * quoted_symbol_count / expected_symbol_count, 3)
        if expected_symbol_count else 0.0
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": "INDSTOCKS_BATCH_QUOTE_SHARDED_ASSEMBLY",
        "capture_interval_minutes": 5,
        "cycle_minute_ist": minute.isoformat(),
        "assembly_status": "COMPLETE" if assembled else "INCOMPLETE",
        "status": "STALE" if not assembled else "STORED",
        "snapshot_persisted": bool(assembled),
        "worker_count": count,
        "shard_count_expected": count,
        "shard_count_completed": int(len(expected_ids & set(by_id))),
        "shard_count_valid": int(
            sum(1 for summary in shard_summaries if summary["issue_count"] == 0)
        ),
        "shards": shard_summaries,
        "missing_shards": missing_ids,
        "expected_symbol_count": int(expected_symbol_count),
        "selected_symbol_count": int(expected_symbol_count),
        "row_count": int(len(frame)) if frame is not None else 0,
        "quoted_symbol_count": quoted_symbol_count,
        "missing_quote_count": missing_quote_count,
        "complete_quote_symbol_count": quoted_symbol_count,
        "complete_quote_symbol_coverage_pct": coverage_pct,
        "coverage_is_informational": True,
        "duplicate_symbol_count": len(duplicate_symbols),
        "missing_symbol_count": len(missing_symbols),
        "validation": {
            "duplicate_symbols": duplicate_symbols[:200],
            "missing_symbols": missing_symbols[:200],
            "issues": _json_clean(issues),
            "fatal_issue_count": len(fatal),
        },
        "assembly_issues": _json_clean(issues),
        "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not assembled:
        manifest["error"] = (
            "Sharded snapshot was not assembled: "
            + "; ".join(
                f"{issue.get('code')}"
                + (f"[w{issue['worker_id']}]" if issue.get("worker_id") is not None else "")
                for issue in fatal[:8]
            )
        )
    return {
        "status": "VALIDATED" if assembled else "INCOMPLETE",
        "frame": frame if assembled else None,
        "manifest": manifest,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Drive helpers (assembly is a Drive producer/consumer only)
# ---------------------------------------------------------------------------
def _find_remote(store: Any, remote: str) -> dict[str, Any] | None:
    parent = store.root_id
    parts = [part for part in remote.strip("/").split("/") if part]
    found: dict[str, Any] | None = None
    for index, part in enumerate(parts):
        mime = FOLDER_MIME if index < len(parts) - 1 else None
        found = store._find_child(parent, part, mime)
        if not found:
            return None
        if index < len(parts) - 1:
            parent = found["id"]
    return found


def _download(store: Any, file_id: str, destination: Path) -> None:
    from persistent_store import download_drive_file_with_retry

    destination.parent.mkdir(parents=True, exist_ok=True)
    download_drive_file_with_retry(
        store.service,
        file_id,
        destination,
        operation="all_universe_shard_assembly",
    )


def load_shard_payloads(
    store: Any,
    cycle_minute: pd.Timestamp,
    worker_count: int,
    work_dir: Path,
) -> list[dict[str, Any]]:
    """Download every expected shard snapshot and manifest for one slot."""
    minute = normalized_cycle_minute(cycle_minute)
    count = max(1, int(worker_count))
    work_dir = Path(work_dir)
    payloads: list[dict[str, Any]] = []
    for worker_id in range(count):
        names = shard_artifact_names(minute, worker_id, count)
        payload: dict[str, Any] = {
            "worker_id": worker_id,
            "manifest": None,
            "frame": None,
            "snapshot_remote_path": names["snapshot_remote"],
            "manifest_remote_path": names["manifest_remote"],
            "snapshot_sha256": None,
        }
        manifest_item = _find_remote(store, names["manifest_remote"])
        snapshot_item = _find_remote(store, names["snapshot_remote"])
        if manifest_item:
            manifest_path = work_dir / "shards" / names["manifest_name"]
            _download(store, manifest_item["id"], manifest_path)
            try:
                payload["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload["manifest"] = None
        if snapshot_item:
            snapshot_path = work_dir / "shards" / names["snapshot_name"]
            _download(store, snapshot_item["id"], snapshot_path)
            try:
                payload["frame"] = pd.read_parquet(snapshot_path)
                payload["snapshot_sha256"] = hashlib.sha256(
                    snapshot_path.read_bytes()
                ).hexdigest()
            except Exception:
                payload["frame"] = None
        payloads.append(payload)
    return payloads


def assemble_from_drive(
    store: Any,
    cycle_minute: pd.Timestamp,
    worker_count: int,
    work_dir: Path,
    *,
    expected_universe: Iterable[str] | None = None,
    strict_universe: bool = False,
    write_canonical: bool = True,
) -> dict[str, Any]:
    """Load shards from Drive, validate them, and optionally publish canonical output."""
    minute = normalized_cycle_minute(cycle_minute)
    payloads = load_shard_payloads(store, minute, worker_count, work_dir)
    result = assemble_shards(
        cycle_minute=minute,
        worker_count=worker_count,
        shards=payloads,
        expected_universe=expected_universe,
        strict_universe=strict_universe,
    )
    if result["status"] == "VALIDATED" and write_canonical:
        write_result = write_canonical_snapshot(
            store,
            minute,
            result["frame"],
            result["manifest"],
            work_dir,
        )
        result["manifest"].update(write_result)
    result["manifest"]["work_dir"] = str(work_dir)
    return result


def write_canonical_snapshot(
    store: Any,
    cycle_minute: pd.Timestamp,
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    work_dir: Path,
) -> dict[str, Any]:
    """Persist the assembled canonical snapshot, then its manifest.

    The Parquet file is uploaded before the manifest so a compiler can never
    observe a manifest that points at a missing or partial snapshot.
    """
    minute = normalized_cycle_minute(cycle_minute)
    names = canonical_artifact_names(minute)
    local_dir = Path(work_dir) / "all_universe_minute_snapshots"
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = local_dir / names["snapshot_name"]
    temp_path = snapshot_path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp_path, index=False, compression="zstd")
    temp_path.replace(snapshot_path)
    snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    payload = dict(manifest)
    payload.update({
        "schema_version": SCHEMA_VERSION,
        "status": "STORED",
        "assembly_status": "COMPLETE",
        "snapshot_persisted": True,
        "snapshot_drive_path": names["snapshot_remote"],
        "remote_snapshot_path": names["snapshot_remote"],
        "snapshot_bytes": int(snapshot_path.stat().st_size),
        "snapshot_sha256": snapshot_sha256,
        "row_count": int(len(frame)),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    })

    snapshot_upload = store.upload_or_update(
        snapshot_path,
        names["snapshot_remote"],
        mime_type="application/vnd.apache.parquet",
    )
    if snapshot_upload.get("status") != "OK":
        raise RuntimeError(
            f"Canonical snapshot upload was not confirmed: {snapshot_upload}"
        )

    encoded = json.dumps(_json_clean(payload), indent=2, sort_keys=True)
    manifest_path = local_dir / names["manifest_name"]
    manifest_path.write_text(encoded, encoding="utf-8")
    latest_path = local_dir / "latest_manifest.json"
    latest_path.write_text(encoded, encoding="utf-8")

    minute_upload = store.upload_or_update(
        manifest_path, names["manifest_remote"], mime_type="application/json"
    )
    if minute_upload.get("status") != "OK":
        raise RuntimeError(
            f"Canonical manifest upload was not confirmed: {minute_upload}"
        )
    latest_upload = store.upload_or_update(
        latest_path, names["latest_manifest_remote"], mime_type="application/json"
    )
    if latest_upload.get("status") != "OK":
        raise RuntimeError(
            f"Canonical latest-manifest upload was not confirmed: {latest_upload}"
        )
    snapshot_path.with_suffix(".uploaded.json").write_text(
        json.dumps({
            "status": "OK",
            "remote_path": names["snapshot_remote"],
            "sha256": snapshot_sha256,
            "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "snapshot_sha256": snapshot_sha256,
        "snapshot_drive_sync": snapshot_upload,
        "manifest_drive_sync": {
            "minute_manifest": minute_upload,
            "latest_manifest": latest_upload,
        },
    }


def read_canonical_manifest(
    store: Any, cycle_minute: pd.Timestamp, work_dir: Path
) -> dict[str, Any] | None:
    """Return the existing canonical manifest for a slot, if any."""
    names = canonical_artifact_names(cycle_minute)
    item = _find_remote(store, names["manifest_remote"])
    if not item:
        return None
    local = Path(work_dir) / "canonical_existing" / names["manifest_name"]
    _download(store, item["id"], local)
    try:
        return json.loads(local.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_assembly_manifest_only(
    store: Any,
    cycle_minute: pd.Timestamp,
    manifest: dict[str, Any],
    work_dir: Path,
    *,
    preserve_stored: bool = True,
) -> dict[str, Any] | None:
    """Persist an audit manifest for an incomplete assembly attempt.

    No snapshot is written, so the slot can never look complete. When a valid
    canonical snapshot already exists for the slot, the audit write is skipped so
    a late retry cannot downgrade a published snapshot.
    """
    minute = normalized_cycle_minute(cycle_minute)
    if preserve_stored:
        existing = read_canonical_manifest(store, minute, work_dir)
        if existing and str(existing.get("status") or "").upper() == "STORED":
            return None
    names = canonical_artifact_names(minute)
    local_dir = Path(work_dir) / "all_universe_minute_snapshots"
    local_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(manifest)
    payload.update({
        "schema_version": SCHEMA_VERSION,
        "status": "ASSEMBLY_INCOMPLETE",
        "assembly_status": "INCOMPLETE",
        "snapshot_persisted": False,
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    encoded = json.dumps(_json_clean(payload), indent=2, sort_keys=True)
    manifest_path = local_dir / names["manifest_name"]
    manifest_path.write_text(encoded, encoding="utf-8")
    latest_path = local_dir / "latest_manifest.json"
    latest_path.write_text(encoded, encoding="utf-8")
    minute_upload = store.upload_or_update(
        manifest_path, names["manifest_remote"], mime_type="application/json"
    )
    latest_upload = store.upload_or_update(
        latest_path, names["latest_manifest_remote"], mime_type="application/json"
    )
    return {
        "status": "WRITTEN",
        "minute_manifest": minute_upload,
        "latest_manifest": latest_upload,
    }


def assembly_worker_count_from_env(default: int = DEFAULT_WORKER_COUNT) -> int:
    try:
        return max(1, int(os.getenv("ALL_NSE_CAPTURE_WORKERS", str(default)) or default))
    except (TypeError, ValueError):
        return max(1, int(default))


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble and validate sharded five-minute NSE snapshots from Google Drive."
    )
    parser.add_argument("--minute", required=True, help="Slot minute in IST, e.g. 2026-11-12T10:05")
    parser.add_argument("--worker-count", type=int, default=assembly_worker_count_from_env())
    parser.add_argument("--work-dir", default="/tmp/all_universe_assembly")
    parser.add_argument("--strict-universe", action="store_true")
    parser.add_argument("--no-write", action="store_true", help="Validate only; do not publish canonical files")
    parser.add_argument("--expected-universe", default="", help="Optional comma-separated official universe")
    args = parser.parse_args(argv)

    from persistent_store import GoogleDriveStore

    minute = normalized_cycle_minute(args.minute)
    expected = (
        [part.strip() for part in args.expected_universe.split(",") if part.strip()]
        if args.expected_universe else None
    )
    store = GoogleDriveStore()
    result = assemble_from_drive(
        store,
        minute,
        args.worker_count,
        Path(args.work_dir),
        expected_universe=expected,
        strict_universe=args.strict_universe,
        write_canonical=not args.no_write,
    )
    print(json.dumps(_json_clean(result["manifest"]), indent=2, sort_keys=True))
    return 0 if result["status"] == "VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(_main())
