"""Persistent historical market-data store.

The Streamlit container is treated as a cache, not as the system of record.
1-minute observations are partitioned by IST trading date and mirrored to
Google Drive using OAuth when available, with the existing service-account
configuration retained as a fallback.

Configuration for Streamlit Cloud:

[gdrive]
root_folder_id = "<Google Drive root folder ID>"
service_account_json = '''{...service-account-json...}'''
oauth_token_json = '''{...authorized-user-token-json...}'''

GitHub Actions should provide GOOGLE_DRIVE_ROOT_ID and
GOOGLE_OAUTH_TOKEN_JSON. The service-account variables remain supported as
a fallback for existing deployments.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PERSIST_ROOT = BASE_DIR / "data" / "persistent_market_data"
INTRADAY_ROOT = PERSIST_ROOT / "intraday_1m"
MANIFEST_PATH = PERSIST_ROOT / "manifest.json"
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dirs() -> None:
    INTRADAY_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_manifest() -> dict:
    ensure_dirs()
    m = _safe_json_read(MANIFEST_PATH, {})
    if not isinstance(m, dict):
        m = {}
    m.setdefault("version", "persistent-market-store-v1")
    m.setdefault("updated_at", None)
    m.setdefault("files", {})
    return m


def save_manifest(m: dict) -> None:
    m["updated_at"] = _now_utc()
    _atomic_json(MANIFEST_PATH, m)


def normalize_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s.rsplit(".", 1)[0]
    return s.replace("/", "_")


def _normalize_1m(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x = df.copy()
    rename: dict[Any, str] = {}
    for c in x.columns:
        lc = str(c).strip().lower().replace(" ", "_")
        if lc in {"timestamp", "datetime", "date", "time", "dt", "_time"}:
            rename[c] = "_time"
        elif lc in {"open", "o"}:
            rename[c] = "Open"
        elif lc in {"high", "h"}:
            rename[c] = "High"
        elif lc in {"low", "l"}:
            rename[c] = "Low"
        elif lc in {"close", "c", "ltp"}:
            rename[c] = "Close"
        elif lc in {"volume", "vol", "v"}:
            rename[c] = "Volume"
    x = x.rename(columns=rename)
    if "_time" in x.columns:
        idx = pd.to_datetime(x.pop("_time"), errors="coerce")
    else:
        idx = pd.to_datetime(x.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    else:
        idx = pd.DatetimeIndex(idx)
    x.index = pd.DatetimeIndex(idx)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[~x.index.isna()].sort_index()
    x = x[~x.index.duplicated(keep="last")]
    x = x.dropna(subset=["Open", "High", "Low", "Close"])
    return x


def _partition_path(symbol: str, day: pd.Timestamp) -> Path:
    root = normalize_symbol(symbol)
    d = pd.Timestamp(day).normalize()
    return INTRADAY_ROOT / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}" / f"{root}.parquet"


def local_partition_files(symbol: str | None = None) -> list[Path]:
    ensure_dirs()
    files = sorted(INTRADAY_ROOT.glob("*/*/*/*.parquet"))
    if symbol:
        stem = normalize_symbol(symbol)
        files = [p for p in files if p.stem.upper() == stem]
    return files


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def gdrive_config() -> dict:
    """Resolve Google Drive configuration.

    OAuth is preferred when GOOGLE_OAUTH_TOKEN_JSON is available because
    the user's OAuth credentials use the user's own Drive storage quota.
    The existing service-account configuration remains as a fallback.
    """
    root_id = (
        os.getenv("GDRIVE_ROOT_FOLDER_ID", "").strip()
        or os.getenv("GOOGLE_DRIVE_ROOT_ID", "").strip()
    )
    service_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    oauth_json = os.getenv("GOOGLE_OAUTH_TOKEN_JSON", "").strip()

    try:
        import streamlit as st  # lazy: module also works outside Streamlit
        if hasattr(st, "secrets"):
            sec = st.secrets.get("gdrive", {})
            if not root_id:
                root_id = str(sec.get("root_folder_id", "")).strip()
            if not service_json:
                service_json = str(sec.get("service_account_json", "")).strip()
            if not oauth_json:
                oauth_json = str(sec.get("oauth_token_json", "")).strip()
    except Exception:
        pass

    if not root_id:
        return {
            "configured": False,
            "reason": "missing root_folder_id",
        }

    # Prefer user's OAuth credentials.
    if oauth_json:
        try:
            oauth_info = json.loads(oauth_json)
        except Exception as exc:
            return {
                "configured": False,
                "reason": f"invalid OAuth token JSON: {type(exc).__name__}",
            }

        if not isinstance(oauth_info, dict):
            return {
                "configured": False,
                "reason": "OAuth token JSON is not an object",
            }

        if not oauth_info.get("refresh_token"):
            return {
                "configured": False,
                "reason": "OAuth token JSON has no refresh_token",
            }

        return {
            "configured": True,
            "auth_method": "oauth",
            "root_folder_id": root_id,
            "oauth_token_info": oauth_info,
        }

    # Existing service-account fallback.
    if not service_json:
        return {
            "configured": False,
            "reason": "missing OAuth token JSON and service-account JSON",
        }

    try:
        info = json.loads(service_json)
    except Exception as exc:
        return {
            "configured": False,
            "reason": f"invalid service-account JSON: {type(exc).__name__}",
        }

    return {
        "configured": True,
        "auth_method": "service_account",
        "root_folder_id": root_id,
        "service_account_info": info,
        "client_email": info.get("client_email"),
    }

def _drive_credentials(cfg: dict):
    """Build authenticated Drive credentials using OAuth first, then service account."""
    if cfg.get("auth_method") == "oauth":
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        credentials = Credentials.from_authorized_user_info(
            cfg["oauth_token_info"],
            scopes=DRIVE_SCOPES,
        )

        if not credentials.valid:
            if not credentials.refresh_token:
                raise RuntimeError(
                    "OAuth credentials are invalid and have no refresh token"
                )
            credentials.refresh(Request())

        return credentials

    from google.oauth2 import service_account

    return service_account.Credentials.from_service_account_info(
        cfg["service_account_info"],
        scopes=DRIVE_SCOPES,
    )


def drive_status() -> dict:
    cfg = gdrive_config()
    if not cfg.get("configured"):
        return {"configured": False, "reason": cfg.get("reason"), "persistent_local_root": _relative_or_absolute(PERSIST_ROOT)}
    try:
        from googleapiclient.discovery import build

        credentials = _drive_credentials(cfg)
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        meta = service.files().get(fileId=cfg["root_folder_id"], fields="id,name,mimeType,trashed").execute()
        if meta.get("trashed"):
            return {"configured": False, "reason": "configured root folder is in Trash"}
        if meta.get("mimeType") != "application/vnd.google-apps.folder":
            return {"configured": False, "reason": "configured root ID is not a folder"}
        return {
            "configured": True,
            "root_folder_id": cfg["root_folder_id"],
            "auth_method": cfg.get("auth_method"),
            "client_email": cfg.get("client_email"),
            "root_name": meta.get("name"),
        }
    except Exception as exc:
        return {"configured": True, "reachable": False, "reason": f"Google Drive check failed: {type(exc).__name__}: {exc}", "client_email": cfg.get("client_email")}


class GoogleDriveStore:
    def __init__(self) -> None:
        cfg = gdrive_config()
        if not cfg.get("configured"):
            raise RuntimeError(cfg.get("reason", "Google Drive is not configured"))
        try:
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload
        except Exception as exc:
            raise RuntimeError("Google Drive dependencies are not installed") from exc
        self.root_id = cfg["root_folder_id"]
        self.credentials = _drive_credentials(cfg)
        self.service = build("drive", "v3", credentials=self.credentials, cache_discovery=False)
        self.MediaFileUpload = MediaFileUpload

    @staticmethod
    def _q(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    def _find_child(self, parent_id: str, name: str, mime_type: str | None = None) -> dict | None:
        q = f"name = '{self._q(name)}' and '{parent_id}' in parents and trashed = false"
        if mime_type:
            q += f" and mimeType = '{mime_type}'"
        resp = self.service.files().list(
            q=q,
            spaces="drive",
            fields="files(id,name,mimeType)",
            pageSize=10,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        files = resp.get("files", [])
        return files[0] if files else None

    def _ensure_folder(self, parent_id: str, name: str) -> str:
        hit = self._find_child(parent_id, name, "application/vnd.google-apps.folder")
        if hit:
            return hit["id"]
        meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
        return self.service.files().create(body=meta, fields="id").execute()["id"]

    def ensure_path(self, relative_folder: str) -> str:
        parent = self.root_id
        parts = [p for p in relative_folder.replace("\\", "/").split("/") if p]
        for part in parts:
            parent = self._ensure_folder(parent, part)
        return parent

    def upload_or_update(self, local_path: Path, remote_relative_path: str, mime_type: str = "application/octet-stream") -> dict:
        remote = remote_relative_path.replace("\\", "/").strip("/")
        parts = remote.split("/")
        name = parts[-1]
        folder = "/".join(parts[:-1])
        parent_id = self.ensure_path(folder)
        existing = self._find_child(parent_id, name)
        media = self.MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
        if existing:
            result = self.service.files().update(fileId=existing["id"], media_body=media, fields="id,name,modifiedTime,size").execute()
            action = "updated"
        else:
            result = self.service.files().create(body={"name": name, "parents": [parent_id]}, media_body=media, fields="id,name,createdTime,size").execute()
            action = "created"
        return {"status": "OK", "action": action, "remote_path": remote, "file_id": result.get("id"), "size": result.get("size")}


def persist_1m_frame(symbol: str, df: pd.DataFrame, source: str = "unknown", sync_drive: bool = True) -> dict:
    """Merge 1-minute observations into date partitions and optionally mirror to Drive."""
    ensure_dirs()
    x = _normalize_1m(df)
    if x.empty:
        return {"status": "NO_DATA", "symbol": normalize_symbol(symbol), "rows_added": 0, "rows_total": 0, "partitions": []}
    root = normalize_symbol(symbol)
    manifest = load_manifest()
    partitions = []
    total_added = 0
    total_rows = 0
    for day, part in x.groupby(x.index.normalize()):
        path = _partition_path(root, day)
        before_idx: set[int] | None = None
        if path.exists():
            try:
                old = pd.read_parquet(path)
                old = _normalize_1m(old)
                before_idx = set(pd.Index(old.index).view("int64"))
                merged = pd.concat([old, part]).sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
            except Exception:
                merged = part.sort_index()
        else:
            merged = part.sort_index()
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=True)
        idx_after = set(pd.Index(merged.index).view("int64"))
        added = len(idx_after - (before_idx or set()))
        total_added += added
        total_rows += len(merged)
        rel = str(path.relative_to(BASE_DIR)).replace("\\", "/")
        key = f"intraday_1m:{root}:{day:%Y-%m-%d}"
        entry = {
            "kind": "intraday_1m_daily_partition",
            "symbol": root,
            "date": f"{day:%Y-%m-%d}",
            "source": source,
            "path": rel,
            "rows": int(len(merged)),
            "rows_added_last_write": int(added),
            "sha256": _sha256(path),
            "updated_at": _now_utc(),
            "drive_sync": "pending",
        }
        if sync_drive:
            try:
                g = GoogleDriveStore()
                remote = f"production_data/intraday_1m/{day:%Y}/{day:%m}/{day:%d}/{root}.parquet"
                out = g.upload_or_update(path, remote)
                entry["drive_sync"] = "OK"
                entry["drive"] = out
            except Exception as exc:
                entry["drive_sync"] = "NOT_SYNCED"
                entry["drive_error"] = f"{type(exc).__name__}: {exc}"
        manifest["files"][key] = entry
        partitions.append(entry)
    save_manifest(manifest)
    return {"status": "OK", "symbol": root, "rows_added": total_added, "rows_total": total_rows, "partitions": partitions}


def persist_local_1m_file(path: str | Path, symbol: str, source: str = "local-file", sync_drive: bool = True) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    return persist_1m_frame(symbol, df, source=source, sync_drive=sync_drive)


def load_persistent_1m(symbol: str, start=None, end=None) -> pd.DataFrame:
    """Load merged persistent daily partitions for a symbol."""
    files = local_partition_files(symbol)
    if not files:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    frames = []
    for path in files:
        try:
            day = pd.Timestamp(path.parent.name + "-" + path.parent.parent.name + "-" + path.parent.parent.parent.name)
        except Exception:
            day = None
        if day is not None:
            if start_ts is not None and end_ts is not None and (day.normalize() < start_ts.normalize() or day.normalize() > end_ts.normalize()):
                continue
            if start_ts is not None and end_ts is None and day.normalize() < start_ts.normalize():
                continue
            if end_ts is not None and start_ts is None and day.normalize() > end_ts.normalize():
                continue
        try:
            x = pd.read_parquet(path)
            x = _normalize_1m(x)
            if start_ts is not None:
                x = x[x.index >= start_ts]
            if end_ts is not None:
                x = x[x.index <= end_ts]
            if not x.empty:
                frames.append(x)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def scan_coverage(symbols: Iterable[str] | None = None) -> pd.DataFrame:
    files = local_partition_files()
    wanted = {normalize_symbol(s) for s in symbols} if symbols else None
    rows = []
    for path in files:
        root = path.stem.upper()
        if wanted and root not in wanted:
            continue
        try:
            x = pd.read_parquet(path)
            x = _normalize_1m(x)
            rows.append({
                "dataset": "persistent_intraday_1m",
                "symbol": root,
                "trading_date": path.parent.name,
                "month": path.parent.parent.name,
                "year": path.parent.parent.parent.name,
                "rows": int(len(x)),
                "first_timestamp": str(x.index.min()) if not x.empty else None,
                "last_timestamp": str(x.index.max()) if not x.empty else None,
                "file_mb": round(path.stat().st_size / (1024 * 1024), 3),
                "path": str(path.relative_to(BASE_DIR)).replace("\\", "/"),
            })
        except Exception as exc:
            rows.append({"dataset": "persistent_intraday_1m", "symbol": root, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def sync_pending(max_files: int = 100) -> dict:
    """Mirror local persistent partitions to Drive. Existing remote files are updated."""
    files = local_partition_files()[:max_files]
    if not files:
        return {"status": "NO_DATA", "attempted": 0, "synced": 0, "errors": []}
    try:
        g = GoogleDriveStore()
    except Exception as exc:
        return {"status": "NOT_CONFIGURED", "attempted": len(files), "synced": 0, "errors": [f"{type(exc).__name__}: {exc}"]}
    synced = 0
    errors = []
    manifest = load_manifest()
    for path in files:
        rel = path.relative_to(INTRADAY_ROOT)
        if len(rel.parts) != 4:
            continue
        year, month, day, name = rel.parts
        symbol = Path(name).stem
        remote = f"production_data/intraday_1m/{year}/{month}/{day}/{symbol}.parquet"
        key = f"intraday_1m:{symbol}:{year}-{month}-{day}"
        try:
            out = g.upload_or_update(path, remote)
            synced += 1
            if key in manifest.get("files", {}):
                manifest["files"][key]["drive_sync"] = "OK"
                manifest["files"][key]["drive"] = out
        except Exception as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    save_manifest(manifest)
    return {"status": "OK" if not errors else "PARTIAL", "attempted": len(files), "synced": synced, "errors": errors}


def storage_summary() -> dict:
    ensure_dirs()
    total = 0
    files = 0
    for p in INTRADAY_ROOT.rglob("*.parquet"):
        files += 1
        total += p.stat().st_size
    m = load_manifest()
    drive = drive_status()
    return {
        "local_root": _relative_or_absolute(PERSIST_ROOT),
        "local_files": files,
        "local_gb": round(total / (1024 ** 3), 4),
        "manifest_files": len(m.get("files", {})),
        "google_drive": drive,
    }
