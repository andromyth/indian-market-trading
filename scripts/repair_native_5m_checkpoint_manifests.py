#!/usr/bin/env python3
"""Safely reconstruct missing INDstocks five-minute checkpoint commit markers."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import PurePosixPath
from zoneinfo import ZoneInfo

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"
PARQUET_MIME = "application/vnd.apache.parquet"


def main() -> int:
    token_text = os.environ.get("GOOGLE_OAUTH_TOKEN_JSON", "")
    root_id = os.environ.get("GOOGLE_DRIVE_ROOT_ID", "").strip()
    if not token_text or not root_id:
        raise RuntimeError("Missing GOOGLE_OAUTH_TOKEN_JSON or GOOGLE_DRIVE_ROOT_ID")
    info = json.loads(token_text)
    creds = Credentials.from_authorized_user_info(
        info, scopes=["https://www.googleapis.com/auth/drive"]
    )
    if not creds.valid:
        creds.refresh(Request())
    drive = build("drive", "v3", credentials=creds, cache_discovery=False)

    def children(parent_id: str, name: str | None = None) -> list[dict]:
        q = f"'{parent_id}' in parents and trashed = false"
        if name is not None:
            escaped = name.replace("\\", "\\\\").replace("'", "\\'")
            q += f" and name = '{escaped}'"
        rows, page = [], None
        while True:
            response = drive.files().list(
                q=q, pageSize=1000, pageToken=page,
                fields="nextPageToken,files(id,name,mimeType,size,modifiedTime)"
            ).execute()
            rows.extend(response.get("files", []))
            page = response.get("nextPageToken")
            if not page:
                return rows

    def descend(parent_id: str, parts: list[str]) -> str:
        current = parent_id
        for part in parts:
            matches = [x for x in children(current, part) if x["mimeType"] == FOLDER_MIME]
            if not matches:
                raise RuntimeError("Missing Drive folder: " + part)
            current = matches[0]["id"]
        return current

    def download(file_id: str) -> bytes:
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, drive.files().get_media(fileId=file_id))
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()

    def upsert_json(parent_id: str, name: str, document: dict) -> None:
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        media = MediaIoBaseUpload(io.BytesIO(payload), mimetype="application/json", resumable=False)
        matches = [x for x in children(parent_id, name) if x["mimeType"] != FOLDER_MIME]
        if matches:
            drive.files().update(
                fileId=matches[0]["id"], media_body=media, fields="id,name,size,modifiedTime"
            ).execute()
        else:
            drive.files().create(
                body={"name": name, "mimeType": "application/json", "parents": [parent_id]},
                media_body=media, fields="id,name,size,modifiedTime"
            ).execute()

    archive_id = descend(root_id, [
        "production_data", "nse_official", "intraday_seed", "olo_5m_archive"
    ])
    symbols_id = descend(archive_id, ["symbols"])
    checkpoint_id = descend(symbols_id, ["window_checkpoints"])
    batch_folders = [x for x in children(checkpoint_id) if x["mimeType"] == FOLDER_MIME]
    repaired = 0
    already_present = 0
    unsafe = []
    parquet_total = 0

    for batch_folder in sorted(batch_folders, key=lambda x: x["name"]):
        batch_key = batch_folder["name"]
        files = children(batch_folder["id"])
        parquet_files = sorted(
            [x for x in files if x["name"].endswith(".parquet") and not x["name"].endswith(".parquet.json")],
            key=lambda x: x["name"],
        )
        parquet_total += len(parquet_files)
        if not parquet_files:
            continue

        decoded = []
        union_symbols: set[str] = set()
        invalid_reason = None
        for item in parquet_files:
            manifest_name = item["name"] + ".json"
            if any(x["name"] == manifest_name and x["mimeType"] != FOLDER_MIME for x in files):
                already_present += 1
            try:
                frame = pd.read_parquet(io.BytesIO(download(item["id"])))
                if "Symbol" not in frame.columns:
                    if not frame.empty:
                        raise ValueError("non-empty checkpoint has no Symbol column")
                    symbols_in_file = set()
                else:
                    symbols_in_file = {
                        str(v).strip().upper()
                        for v in frame["Symbol"].dropna().unique()
                        if str(v).strip()
                    }
                if len(symbols_in_file) > 5:
                    raise ValueError("checkpoint contains more than five symbols")
                union_symbols.update(symbols_in_file)
                decoded.append((item, frame, symbols_in_file))
            except Exception as exc:
                invalid_reason = f"{item['name']}: {type(exc).__name__}: {exc}"
                break

        expected_symbols = sorted(union_symbols)
        expected_hash = hashlib.sha1("|".join(expected_symbols).encode("utf-8")).hexdigest()[:16]
        if invalid_reason:
            unsafe.append({"batch": batch_key, "reason": invalid_reason})
            continue
        if not expected_symbols or len(expected_symbols) > 5 or expected_hash != batch_key:
            unsafe.append({
                "batch": batch_key,
                "reason": "cannot prove full symbol batch from checkpoint contents and SHA-1 batch key",
                "symbols_inferred": expected_symbols,
                "hash_from_inferred_symbols": expected_hash,
            })
            continue

        batch_writes = 0
        for item, frame, symbols_in_file in decoded:
            manifest_name = item["name"] + ".json"
            if any(x["name"] == manifest_name and x["mimeType"] != FOLDER_MIME for x in files):
                continue
            if not symbols_in_file.issubset(set(expected_symbols)):
                invalid_reason = f"unexpected symbols in {item['name']}"
                break
            window_key = PurePosixPath(item["name"]).stem
            match = re.fullmatch(r"(\d{8}T\d{4})_(\d{8}T\d{4})", window_key)
            if not match:
                invalid_reason = f"unrecognized window filename: {item['name']}"
                break
            start = datetime.strptime(match.group(1), "%Y%m%dT%H%M").replace(tzinfo=IST)
            end = datetime.strptime(match.group(2), "%Y%m%dT%H%M").replace(tzinfo=IST)
            row_counts = {symbol: 0 for symbol in expected_symbols}
            if not frame.empty:
                counts = frame["Symbol"].astype(str).str.strip().str.upper().value_counts()
                for symbol, count in counts.items():
                    row_counts[symbol] = int(count)
            # The Parquet is written only after the full API response is parsed and
            # its upload is confirmed. The batch hash proves the exact symbol set.
            manifest = {
                "status": "COMPLETE",
                "window_key": window_key,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "symbols": expected_symbols,
                "row_counts": row_counts,
                "checkpoint_remote": (
                    "production_data/nse_official/intraday_seed/olo_5m_archive/"
                    f"symbols/window_checkpoints/{batch_key}/{item['name']}"
                ),
                "reconstructed_from_verified_checkpoint": True,
                "reconstructed_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
            }
            upsert_json(batch_folder["id"], manifest_name, manifest)
            repaired += 1
            batch_writes += 1
        if invalid_reason:
            unsafe.append({"batch": batch_key, "reason": invalid_reason})
        print(json.dumps({
            "batch": batch_key,
            "inferred_symbols": expected_symbols,
            "checkpoint_parquets": len(parquet_files),
            "manifests_reconstructed": batch_writes,
        }, sort_keys=True))

    print("CHECKPOINT_REPAIR_SUMMARY=" + json.dumps({
        "checkpoint_batch_folders": len(batch_folders),
        "checkpoint_parquets": parquet_total,
        "manifests_reconstructed": repaired,
        "preexisting_manifests_counted": already_present,
        "unsafe_batch_count": len(unsafe),
        "unsafe_batches": unsafe,
        "root_backfill_manifest_created": False,
        "status": "PASS" if not unsafe else "PARTIAL",
    }, sort_keys=True))
    # Partial is non-fatal: the backfill may safely re-fetch any ambiguous windows.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"CHECKPOINT_REPAIR_FATAL={type(exc).__name__}: {exc}", file=sys.stderr)
        raise
