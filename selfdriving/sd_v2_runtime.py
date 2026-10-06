"""Streamlit/runtime adapters for Beta V2.

All artefacts live in the isolated Google Drive prefix
selfdriving/beta_version2. These helpers are opt-in and never alter the current
production model artefacts.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .sd_cycle import predict_stage
from .sd_v2_panel import build_beta_delivery_panel, split_training_inference
from .sd_intraday_v2 import _session_rows, _norm as norm_intraday, feature_columns

CACHE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "beta_version2"
)
REMOTE_ROOT = "selfdriving/beta_version2"


def _drive():
    from persistent_store import GoogleDriveStore
    return GoogleDriveStore()


def _download(remote: str, local: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    store = _drive()
    parts = remote.strip("/").split("/")
    parent = store.root_id
    for part in parts[:-1]:
        hit = store._find_child(parent, part)
        if not hit:
            raise FileNotFoundError(remote)
        parent = hit["id"]
    hit = store._find_child(parent, parts[-1])
    if not hit:
        raise FileNotFoundError(remote)

    local.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    dl = MediaIoBaseDownload(
        buf,
        store.service.files().get_media(fileId=hit["id"]),
    )
    done = False
    while not done:
        _, done = dl.next_chunk()
    local.write_bytes(buf.getvalue())


def _ensure(remote: str, local: Path) -> Path:
    if not local.exists() or local.stat().st_size == 0:
        _download(remote, local)
    return local


def beta_status() -> dict[str, Any]:
    out: dict[str, Any] = {
        "version": "2026-10-06-beta-v2",
        "delivery": {"status": "UNAVAILABLE"},
        "intraday": {"status": "UNAVAILABLE"},
    }
    try:
        p = _ensure(
            f"{REMOTE_ROOT}/status_train.json",
            CACHE / "status_train.json",
        )
        out["delivery"] = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        out["delivery"] = {"status": "NOT_TRAINED", "error": f"{type(exc).__name__}: {exc}"}

    try:
        p = _ensure(
            f"{REMOTE_ROOT}/intraday/BETA_V2_INTRADAY_MANIFEST.json",
            CACHE / "intraday_manifest.json",
        )
        out["intraday"] = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        out["intraday"] = {"status": "NOT_TRAINED", "error": f"{type(exc).__name__}: {exc}"}
    return out


def beta_delivery_predict(symbol: str, global_cues: pd.DataFrame | None = None,
                          raw_news: pd.DataFrame | None = None,
                          fno_daily: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run the isolated Beta V2 Delivery champion for one current symbol."""
    store = _drive()
    # Find newest PIT snapshot directly from Drive.
    parent = store.root_id
    for part in ("features", "daily"):
        hit = store._find_child(parent, part, "application/vnd.google-apps.folder")
        if not hit:
            raise RuntimeError("Drive features/daily folder unavailable")
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
    files = [x for x in resp.get("files", []) if str(x.get("name", "")).endswith(".parquet")]
    if not files:
        raise RuntimeError("No current Delivery PIT snapshot available")

    latest = files[0]
    pit_local = CACHE / str(latest["name"])
    if not pit_local.exists():
        from googleapiclient.http import MediaIoBaseDownload
        pit_local.parent.mkdir(parents=True, exist_ok=True)
        with pit_local.open("wb") as fh:
            dl = MediaIoBaseDownload(
                fh,
                store.service.files().get_media(fileId=latest["id"]),
            )
            done = False
            while not done:
                _, done = dl.next_chunk()

    pit = pd.read_parquet(pit_local)
    if global_cues is None:
        gp = _ensure(f"{REMOTE_ROOT}/global_cues.parquet", CACHE / "global_cues.parquet")
        global_cues = pd.read_parquet(gp)
    if raw_news is None:
        npth = _ensure(f"{REMOTE_ROOT}/raw_news.parquet", CACHE / "raw_news.parquet")
        raw_news = pd.read_parquet(npth)
    if fno_daily is None:
        try:
            fp = _ensure(
                "production_data/nse_official/fno/fno_daily_features_2026.parquet",
                CACHE / "fno_daily.parquet",
            )
            fno_daily = pd.read_parquet(fp)
        except Exception:
            fno_daily = pd.DataFrame()

    panel, audit = build_beta_delivery_panel(
        pit,
        global_cues=global_cues,
        raw_news=raw_news,
        fno_daily=fno_daily,
    )
    _, inference, _ = split_training_inference(panel)
    inference = inference[
        inference["symbol"].eq(
            str(symbol).upper().replace(".NS", "").replace(".BO", "")
        )
    ].copy()
    if inference.empty:
        return {
            "prediction_status": "BLOCKED",
            "symbol": str(symbol).upper(),
            "reason": "symbol_not_present_in_latest_pit_snapshot",
            "audit": audit,
        }

    registry_root = CACHE / "models"
    registry_root.mkdir(parents=True, exist_ok=True)

    # Download registry and champion bundle referenced by it.
    reg_local = registry_root / "registry.json"
    _ensure(f"{REMOTE_ROOT}/models/registry.json", reg_local)
    state = json.loads(reg_local.read_text(encoding="utf-8"))
    champion = state.get("champion")
    if not champion:
        return {
            "prediction_status": "ABSTAIN",
            "symbol": str(symbol).upper(),
            "reason": "beta_v2_delivery_champion_not_trained",
        }
    bundle_name = str(champion["file"])
    _ensure(f"{REMOTE_ROOT}/models/{bundle_name}", registry_root / bundle_name)

    pred = predict_stage(inference, registry_root)
    row = pred.iloc[0].to_dict()
    row["prediction_status"] = "PASS"
    row["beta_v2_audit"] = audit
    return row


def beta_intraday_predict(
    symbol: str,
    price_df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Run all trained Beta V2 intraday horizons for one live symbol."""
    root = str(symbol).upper().replace(".NS", "").replace(".BO", "")
    rows = []
    latest_rows = _session_rows(
        price_df.assign(symbol=root),
        market_df,
        (5, 10, 15, 20, 25, 30),
        include_future=False,
    )
    if not latest_rows:
        return []

    frame = pd.DataFrame(latest_rows)
    latest_ts = pd.to_datetime(frame["decision_timestamp"], utc=True).max()
    frame = frame[frame["decision_timestamp"].eq(latest_ts)].copy()
    features = feature_columns()

    for horizon in (5, 10, 15, 20, 25, 30):
        h = frame[frame["horizon_min"].eq(horizon)].copy()
        registry_root = CACHE / "intraday" / "models" / f"h{horizon}"
        try:
            _ensure(
                f"{REMOTE_ROOT}/intraday/models/h{horizon}/registry.json",
                registry_root / "registry.json",
            )
            state = json.loads((registry_root / "registry.json").read_text(encoding="utf-8"))
            champion = state.get("champion")
            if not champion:
                rows.append({
                    "symbol": root,
                    "horizon_min": horizon,
                    "signal": "ABSTAIN",
                    "reason": "NO_CHAMPION",
                })
                continue
            bundle = str(champion["file"])
            _ensure(
                f"{REMOTE_ROOT}/intraday/models/h{horizon}/{bundle}",
                registry_root / bundle,
            )
            bundle_obj = joblib.load(registry_root / bundle)
            X = h[features].apply(pd.to_numeric, errors="coerce")
            p = float(bundle_obj.predict_proba_up(X)[0])
            thr = state["champion"].get("thresholds", {"abstain_all": True})
            from .sd_walkforward import to_signal
            signal = to_signal(p, thr)
            rows.append({
                "symbol": root,
                "horizon_min": horizon,
                "decision_timestamp": str(latest_ts),
                "reference_close": float(h["reference_close"].iloc[0]),
                "probability_up": p,
                "signal": signal,
                "signal_reason": (
                    "VALIDATED_THRESHOLD"
                    if signal != "ABSTAIN" else "NO_VALIDATED_ACTION_BAND"
                ),
            })
        except Exception as exc:
            rows.append({
                "symbol": root,
                "horizon_min": horizon,
                "signal": "ABSTAIN",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    return rows
