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


def _load_optional_parquet(
    remotes: tuple[str, ...],
    local: Path,
) -> tuple[pd.DataFrame, str]:
    """Load an optional Drive-backed parquet with legacy-path fallback.

    Returns the loaded frame and a provenance label. A genuinely absent optional
    archive becomes an empty frame so the feature builder can flag NO_DATA;
    authentication, transport, and corrupt-file errors still propagate.
    """
    if local.exists() and local.stat().st_size > 0:
        return pd.read_parquet(local), "LOCAL_CACHE"

    for remote in remotes:
        try:
            _download(remote, local)
            if local.exists() and local.stat().st_size > 0:
                return pd.read_parquet(local), remote
        except FileNotFoundError:
            continue

    return pd.DataFrame(), "MISSING"


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

    # This file is small enough to refresh on each cache miss. Unlike a PIT
    # snapshot, it is mutable status and must not stay stale in a long-lived app.
    try:
        audit_path = CACHE / "status_data_audit.json"
        _download(f"{REMOTE_ROOT}/status_data_audit.json", audit_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        global_audit = audit.get("global_market_fx", {}) or {}
        news_audit = audit.get("news", {}) or {}
        fno_audit = audit.get("fno", {}) or {}
        snapshot_audit = audit.get("snapshot", {}) or {}
        out["data_audit"] = {
            "overall_status": audit.get("overall_status", "UNKNOWN"),
            "generated_at_utc": audit.get("generated_at_utc"),
            "expected_reference_date": audit.get("expected_reference_date"),
            "observed_reference_date": audit.get("observed_reference_date"),
            "snapshot_fresh": audit.get("snapshot_fresh"),
            "snapshot": {
                "rows": snapshot_audit.get("rows"),
                "symbols": snapshot_audit.get("symbols"),
                "duplicate_symbol_rows": snapshot_audit.get("duplicate_symbol_rows"),
                "mean_feature_non_null_pct": snapshot_audit.get("mean_feature_non_null_pct"),
                "production_quality_gate_pct": snapshot_audit.get("production_quality_gate_pct"),
            },
            "stockwise_source_coverage": audit.get("stockwise_source_coverage", {}),
            "feature_groups": audit.get("feature_groups", {}),
            "volume": audit.get("volume", {}),
            "global_market_fx": {
                "rows": global_audit.get("rows"),
                "series": global_audit.get("series", []),
                "duplicate_key_date_rows": global_audit.get("duplicate_key_date_rows"),
                "integrated_feature_coverage": global_audit.get("integrated_feature_coverage", {}),
                "latest_available_by_series": global_audit.get("latest_available_by_series", {}),
            },
            "news": {
                "rows": news_audit.get("rows"),
                "symbols": news_audit.get("symbols"),
                "duplicate_headline_rows": news_audit.get("duplicate_headline_rows"),
                "latest_first_seen_at": news_audit.get("latest_first_seen_at"),
                "rows_first_seen_last_24h": news_audit.get("rows_first_seen_last_24h"),
                "integrated_feature_coverage": news_audit.get("integrated_feature_coverage", {}),
            },
            "fno": {
                "rows": fno_audit.get("rows"),
                "symbols": fno_audit.get("symbols"),
                "duplicate_symbol_date_rows": fno_audit.get("duplicate_symbol_date_rows"),
                "latest_trade_date": fno_audit.get("latest_trade_date"),
                "integrated_feature_coverage": fno_audit.get("integrated_feature_coverage", {}),
            },
            "hard_failures": audit.get("hard_failures", []),
            "warnings": audit.get("warnings", []),
        }
    except Exception as exc:
        out["data_audit"] = {
            "overall_status": "NOT_AVAILABLE",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return out


def beta_delivery_predict(symbol: str, global_cues: pd.DataFrame | None = None,
                          raw_news: pd.DataFrame | None = None,
                          fno_daily: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run the isolated Beta V2 Delivery champion for one current symbol."""
    store = _drive()

    # Check model readiness before touching optional feature archives. A fresh
    # deployment can legitimately have no Beta V2 training artefacts yet; that
    # state must produce an explicit ABSTAIN rather than a misleading
    # FileNotFoundError for a missing optional global/news cache.
    registry_root = CACHE / "models"
    registry_root.mkdir(parents=True, exist_ok=True)
    reg_local = registry_root / "registry.json"
    try:
        _ensure(
            f"{REMOTE_ROOT}/models/registry.json",
            reg_local,
        )
    except FileNotFoundError:
        return {
            "prediction_status": "ABSTAIN",
            "symbol": str(symbol).upper(),
            "reason": "beta_v2_delivery_champion_not_trained",
            "runtime_state": "NO_REGISTRY",
        }

    try:
        state = json.loads(reg_local.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "prediction_status": "ERROR",
            "symbol": str(symbol).upper(),
            "reason": f"invalid_beta_v2_registry:{type(exc).__name__}:{exc}",
        }

    champion = state.get("champion")
    if not champion:
        return {
            "prediction_status": "ABSTAIN",
            "symbol": str(symbol).upper(),
            "reason": "beta_v2_delivery_champion_not_trained",
            "runtime_state": "NO_CHAMPION",
        }

    bundle_name = str(champion.get("file", "")).strip()
    if not bundle_name:
        return {
            "prediction_status": "ERROR",
            "symbol": str(symbol).upper(),
            "reason": "beta_v2_registry_champion_file_missing",
        }

    try:
        _ensure(
            f"{REMOTE_ROOT}/models/{bundle_name}",
            registry_root / bundle_name,
        )
    except FileNotFoundError:
        return {
            "prediction_status": "ERROR",
            "symbol": str(symbol).upper(),
            "reason": f"beta_v2_champion_bundle_missing:{bundle_name}",
        }

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
    global_source = "PROVIDED"
    news_source = "PROVIDED"
    if global_cues is None:
        global_cues, global_source = _load_optional_parquet(
            (
                f"{REMOTE_ROOT}/global_cues.parquet",
                "selfdriving/global_cues.parquet",
            ),
            CACHE / "global_cues.parquet",
        )
    if raw_news is None:
        raw_news, news_source = _load_optional_parquet(
            (
                f"{REMOTE_ROOT}/raw_news.parquet",
                "selfdriving/raw_news.parquet",
            ),
            CACHE / "raw_news.parquet",
        )
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

    pred = predict_stage(inference, registry_root)
    row = pred.iloc[0].to_dict()
    row["prediction_status"] = "PASS"
    audit["global_source"] = global_source
    audit["news_source"] = news_source
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
            float32_max = float(__import__("numpy").finfo(__import__("numpy").float32).max)
            X = X.replace([float("inf"), float("-inf")], __import__("numpy").nan)
            for c in X.columns:
                X[c] = X[c].mask(
                    ~__import__("numpy").isfinite(X[c])
                    | (X[c].abs() > float32_max),
                    __import__("numpy").nan,
                ).astype("float32")
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
