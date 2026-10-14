"""Streamlit/runtime adapters for Beta V2.

All artefacts live in the isolated Google Drive prefix
selfdriving/beta_version2. These helpers are opt-in and never alter the current
production model artefacts.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import time as _perf_time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .sd_cycle import predict_stage
from .sd_v2_panel import build_beta_delivery_panel, split_training_inference
from .sd_intraday_v2 import _session_rows, _norm as norm_intraday, feature_columns
from live_signal_guards import MIN_LIVE_FEATURE_COVERAGE, feature_coverage_fraction
from .runtime_timeframe_diagnostics import (
    report_runtime_cadence,
    runtime_diagnostics_enabled,
    runtime_event,
)

CACHE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "beta_version2"
)
REMOTE_ROOT = "selfdriving/beta_version2"

# The unattended service evaluates several symbols every minute. Re-downloading
# the same six Drive registries/bundles for every symbol wastes the latency
# budget. Revalidate model artifacts on a short TTL so updated return heads
# arrive automatically without forcing a service restart.
_INTRADAY_BUNDLE_CACHE: dict[int, tuple[float, dict[str, Any], Any]] = {}


def _load_intraday_champion(horizon: int) -> tuple[dict[str, Any], Any, float, bool]:
    ttl_seconds = max(
        0.0,
        float(os.getenv("BETA_V2_MODEL_CACHE_TTL_SECONDS", "180")),
    )
    now_mono = _perf_time.monotonic()
    cached = _INTRADAY_BUNDLE_CACHE.get(int(horizon))
    if cached is not None and now_mono - cached[0] < ttl_seconds:
        return cached[1], cached[2], 0.0, True

    started = _perf_time.perf_counter()
    registry_root = CACHE / "intraday" / "models" / f"h{int(horizon)}"
    _ensure(
        f"{REMOTE_ROOT}/intraday/models/h{int(horizon)}/registry.json",
        registry_root / "registry.json",
    )
    state = json.loads(
        (registry_root / "registry.json").read_text(encoding="utf-8")
    )
    champion = state.get("champion")
    bundle_obj = None
    if champion:
        bundle_name = str(champion.get("file") or "").strip()
        if not bundle_name:
            raise RuntimeError(
                f"Intraday h{horizon} registry champion is missing its bundle filename"
            )
        _ensure(
            f"{REMOTE_ROOT}/intraday/models/h{int(horizon)}/{bundle_name}",
            registry_root / bundle_name,
        )
        bundle_obj = joblib.load(registry_root / bundle_name)

    _INTRADAY_BUNDLE_CACHE[int(horizon)] = (
        _perf_time.monotonic(),
        state,
        bundle_obj,
    )
    return state, bundle_obj, _perf_time.perf_counter() - started, False


def _drive():
    from persistent_store import GoogleDriveStore
    return GoogleDriveStore()


def _remote_signature(metadata: dict[str, Any]) -> dict[str, Any]:
    """Stable Drive file identity used to detect source updates without redownloading it."""
    return {
        key: metadata.get(key)
        for key in ("id", "name", "mimeType", "size", "modifiedTime", "md5Checksum")
    }


def _cache_metadata_path(local: Path) -> Path:
    return local.with_name(local.name + ".drive-meta.json")


def _cache_matches_remote(local: Path, metadata: dict[str, Any]) -> bool:
    """Return True only when the local bytes are known to match this Drive version."""
    if not local.exists() or local.stat().st_size == 0:
        return False
    try:
        saved = json.loads(_cache_metadata_path(local).read_text(encoding="utf-8"))
        if saved != _remote_signature(metadata):
            return False
        remote_size = metadata.get("size")
        return remote_size is None or int(remote_size) == local.stat().st_size
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _resolve_remote_file(remote: str):
    store = _drive()
    parts = [part for part in remote.strip("/").split("/") if part]
    if not parts:
        raise FileNotFoundError(remote)
    parent = store.root_id
    folder_mime = "application/vnd.google-apps.folder"
    for part in parts[:-1]:
        hit = store._find_child(parent, part, folder_mime)
        if not hit:
            raise FileNotFoundError(remote)
        parent = hit["id"]
    hit = store._find_child(parent, parts[-1])
    if not hit:
        raise FileNotFoundError(remote)
    metadata = store.service.files().get(
        fileId=hit["id"],
        fields="id,name,mimeType,size,modifiedTime,md5Checksum",
    ).execute()
    return store, metadata


def _download(remote: str, local: Path) -> None:
    """Refresh a Drive-backed cache when the remote file version has changed.

    Mutable data archives and same-named PIT snapshots are checked on every
    inference request. An unchanged remote file costs only a metadata lookup;
    changed bytes are downloaded to a temporary file and atomically replaced.
    A missing or unreachable remote source is never silently replaced by stale
    local data.
    """
    store, metadata = _resolve_remote_file(remote)
    if _cache_matches_remote(local, metadata):
        return

    from googleapiclient.http import MediaIoBaseDownload

    local.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=local.name + ".", suffix=".tmp", dir=str(local.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            downloader = MediaIoBaseDownload(
                fh,
                store.service.files().get_media(fileId=metadata["id"]),
            )
            done = False
            while not done:
                _, done = downloader.next_chunk()
        if Path(temporary_name).stat().st_size == 0:
            raise RuntimeError(f"Drive source downloaded as an empty file: {remote}")
        os.replace(temporary_name, local)

        sidecar = _cache_metadata_path(local)
        sidecar_fd, sidecar_tmp = tempfile.mkstemp(
            prefix=sidecar.name + ".", suffix=".tmp", dir=str(sidecar.parent)
        )
        try:
            with os.fdopen(sidecar_fd, "w", encoding="utf-8") as fh:
                json.dump(_remote_signature(metadata), fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(sidecar_tmp, sidecar)
        finally:
            if os.path.exists(sidecar_tmp):
                os.unlink(sidecar_tmp)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _ensure(remote: str, local: Path) -> Path:
    # Metadata is rechecked even when local bytes exist. This is critical for
    # current PIT snapshots, the model registry and mutable global/news/F&O files.
    _download(remote, local)
    return local


def _load_optional_parquet(
    remotes: tuple[str, ...],
    local: Path,
) -> tuple[pd.DataFrame, str]:
    """Load the newest available Drive version, with legacy-path fallback.

    A missing optional archive becomes NO_DATA. Permission, transport, and
    corrupt-file errors still propagate rather than being hidden as missing.
    """
    for remote in remotes:
        try:
            _download(remote, local)
            if not local.exists() or local.stat().st_size == 0:
                raise RuntimeError(f"Drive source is empty after download: {remote}")
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
        active_model_audit = audit.get("active_model_contract", {}) or {}
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
            "active_model_contract": {
                "status": active_model_audit.get("status", "UNKNOWN"),
                "model_file": active_model_audit.get("model_file"),
                "feature_count": active_model_audit.get("feature_count"),
                "feature_names": active_model_audit.get("feature_names", []),
                "missing_columns_in_current_panel": active_model_audit.get("missing_columns_in_current_panel", []),
                "feature_groups": active_model_audit.get("feature_groups", {}),
                "current_panel_feature_coverage": active_model_audit.get("current_panel_feature_coverage", {}),
                "usd_inr": active_model_audit.get("usd_inr", {}),
            },
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
    # A same-named PIT file may be repaired/rebuilt for the same reference date.
    # Check its Drive version instead of trusting the existence of local bytes.
    _download(f"features/daily/{latest['name']}", pit_local)
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
    coverage = pd.to_numeric(row.get("feature_coverage"), errors="coerce")
    row["minimum_active_model_feature_coverage"] = MIN_LIVE_FEATURE_COVERAGE
    if pd.isna(coverage) or not float(coverage) >= MIN_LIVE_FEATURE_COVERAGE:
        row["prediction_status"] = "BLOCKED"
        row["signal"] = "ABSTAIN"
        row["signal_reason"] = "INSUFFICIENT_ACTIVE_MODEL_FEATURE_COVERAGE"
        row["reason"] = (
            "Active model feature coverage is "
            + (f"{float(coverage):.1%}" if pd.notna(coverage) else "unavailable")
            + f"; at least {MIN_LIVE_FEATURE_COVERAGE:.0%} is required."
        )
    else:
        row["prediction_status"] = "PASS"

    # Expose the target, actual calculation time, and model input cutoff.
    # The existing champion was trained on prior-session point-in-time rows,
    # so current post-close news/global data is not backfilled into that row.
    reference = pd.to_datetime(row.get("date"), errors="coerce")
    if pd.notna(reference):
        from .sd_data_quality import next_nse_session
        next_session = next_nse_session(reference.date())
        row["target_session_date"] = (
            next_session.isoformat() if next_session is not None else None
        )
    else:
        row["target_session_date"] = None
    row["forecast_as_of_utc"] = datetime.now(timezone.utc).isoformat()
    cutoff = inference.iloc[0].get("decision_timestamp")
    row["feature_cutoff_timestamp"] = (
        pd.Timestamp(cutoff).isoformat() if cutoff is not None and not pd.isna(cutoff) else None
    )

    audit["global_source"] = global_source
    audit["news_source"] = news_source
    row["beta_v2_audit"] = audit
    return row


def beta_intraday_predict(
    symbol: str,
    price_df: pd.DataFrame,
    market_df: pd.DataFrame | None = None,
    *,
    decision_interval_minutes: int = 1,
) -> list[dict[str, Any]]:
    """Run trained intraday horizons on fresh bars, defaulting to 1-minute inference anchors.

    Historical training remains 5-minute anchored unless explicitly changed.
    """
    root = str(symbol).upper().replace(".NS", "").replace(".BO", "")
    diagnostics = runtime_diagnostics_enabled()
    call_started_perf = _perf_time.perf_counter()
    price_input_report = {}
    market_input_report = {}
    if diagnostics:
        price_input_report = report_runtime_cadence(
            "intraday_price_df_received", getattr(price_df, "index", [])
        )
        if market_df is not None and not market_df.empty:
            market_input_report = report_runtime_cadence(
                "intraday_market_df_received", getattr(market_df, "index", [])
            )

    feature_build_started = _perf_time.perf_counter()
    rows = []
    latest_rows = _session_rows(
        price_df.assign(symbol=root),
        market_df,
        (5, 10, 15, 20, 25, 30),
        include_future=False,
        decision_interval_minutes=decision_interval_minutes,
    )
    feature_build_seconds = _perf_time.perf_counter() - feature_build_started
    if diagnostics:
        report_runtime_cadence(
            "intraday_derived_decision_timestamps",
            [item.get("decision_timestamp") for item in latest_rows],
        )
        runtime_event(
            "intraday_feature_construction_completed",
            symbol=root,
            input_price_rows=int(len(price_df)) if price_df is not None else 0,
            input_market_rows=int(len(market_df)) if market_df is not None else 0,
            derived_rows=int(len(latest_rows)),
            feature_construction_seconds=round(feature_build_seconds, 6),
            decision_schedule_minutes=decision_interval_minutes,
        )
    if not latest_rows:
        if diagnostics:
            runtime_event(
                "intraday_prediction_empty",
                symbol=root,
                reason="No completed decision rows available from current in-memory frames",
                total_runtime_seconds=round(_perf_time.perf_counter() - call_started_perf, 6),
            )
        return []

    frame = pd.DataFrame(latest_rows)
    latest_ts = pd.to_datetime(frame["decision_timestamp"], utc=True).max()
    frame = frame[frame["decision_timestamp"].eq(latest_ts)].copy()
    features = feature_columns()

    input_to_decision_gap_minutes = None
    if diagnostics:
        latest_input_value = price_input_report.get("latest_timestamp_utc")
        try:
            latest_input_ts = pd.Timestamp(latest_input_value)
            decision_anchor_ts = pd.Timestamp(latest_ts)
            input_to_decision_gap_minutes = (
                latest_input_ts - decision_anchor_ts
            ).total_seconds() / 60.0
        except Exception:
            latest_input_value = None
        runtime_event(
            "intraday_decision_anchor",
            symbol=root,
            latest_price_candle_utc=latest_input_value,
            latest_market_candle_utc=market_input_report.get("latest_timestamp_utc"),
            selected_model_decision_utc=pd.Timestamp(latest_ts).isoformat(),
            selected_model_decision_ist=pd.Timestamp(latest_ts).tz_convert("Asia/Kolkata").isoformat(),
            latest_candle_to_decision_gap_minutes=(
                round(float(input_to_decision_gap_minutes), 3)
                if input_to_decision_gap_minutes is not None else None
            ),
            prediction_horizons_minutes=[5, 10, 15, 20, 25, 30],
            decision_schedule_minutes=decision_interval_minutes,
            note="Inference selects fresh minute-aligned decisions; training rows retain their default five-minute anchor cadence.",
        )

    for horizon in (5, 10, 15, 20, 25, 30):
        h = frame[frame["horizon_min"].eq(horizon)].copy()
        horizon_started_perf = _perf_time.perf_counter()
        model_cache_hit = False
        try:
            state, bundle_obj, artifact_load_seconds, model_cache_hit = (
                _load_intraday_champion(horizon)
            )
            champion = state.get("champion")
            if not champion or bundle_obj is None:
                rows.append({
                    "symbol": root,
                    "horizon_min": horizon,
                    "decision_timestamp": str(latest_ts),
                    "signal": "ABSTAIN",
                    "reason": "NO_CHAMPION",
                })
                continue
            matrix_started_perf = _perf_time.perf_counter()
            X = h[features].apply(pd.to_numeric, errors="coerce")
            float32_max = float(__import__("numpy").finfo(__import__("numpy").float32).max)
            X = X.replace([float("inf"), float("-inf")], __import__("numpy").nan)
            for c in X.columns:
                X[c] = X[c].mask(
                    ~__import__("numpy").isfinite(X[c])
                    | (X[c].abs() > float32_max),
                    __import__("numpy").nan,
                ).astype("float32")

            # Use the registered bundle's actual fitted feature contract. Missing
            # columns count as missing; they are never synthesized to pass the gate.
            model_features = list(getattr(bundle_obj, "features", []) or features)
            missing_model_features = [name for name in model_features if name not in X.columns]
            if missing_model_features:
                rows.append({
                    "symbol": root,
                    "horizon_min": horizon,
                    "signal": "ABSTAIN",
                    "reason": "MODEL_FEATURE_CONTRACT_MISSING",
                    "missing_model_features": missing_model_features[:10],
                    "active_model_feature_count": len(model_features),
                })
                continue

            coverage = feature_coverage_fraction(X, model_features)
            if coverage < MIN_LIVE_FEATURE_COVERAGE:
                rows.append({
                    "symbol": root,
                    "horizon_min": horizon,
                    "decision_timestamp": str(latest_ts),
                    "signal": "ABSTAIN",
                    "signal_reason": "INSUFFICIENT_ACTIVE_MODEL_FEATURE_COVERAGE",
                    "reason": (
                        f"Active model feature coverage is {coverage:.1%}; "
                        f"at least {MIN_LIVE_FEATURE_COVERAGE:.0%} is required."
                    ),
                    "feature_coverage": coverage,
                    "active_model_feature_count": len(model_features),
                })
                continue

            feature_matrix_seconds = _perf_time.perf_counter() - matrix_started_perf
            model_predict_started = _perf_time.perf_counter()
            p = float(bundle_obj.predict_proba_up(X)[0])
            model_prediction_seconds = _perf_time.perf_counter() - model_predict_started
            thr = state["champion"].get("thresholds", {"abstain_all": True})
            from .sd_walkforward import to_signal
            signal = to_signal(p, thr)

            reference_close = pd.to_numeric(
                h["reference_close"].iloc[0], errors="coerce"
            )
            expected_return = np.nan
            predicted_price = np.nan
            price_forecast_status = "RETRAIN_REQUIRED"
            price_forecast_validation = {}
            predict_return = getattr(bundle_obj, "predict_expected_return", None)
            if callable(predict_return):
                try:
                    return_prediction = predict_return(X)
                    if return_prediction is not None and len(return_prediction):
                        expected_return = float(return_prediction[0])
                        if np.isfinite(expected_return) and pd.notna(reference_close) and float(reference_close) > 0:
                            candidate_price = float(reference_close) * (1.0 + expected_return)
                            if np.isfinite(candidate_price) and candidate_price > 0:
                                predicted_price = candidate_price
                                model_meta = getattr(bundle_obj, "meta", {}) or {}
                                price_forecast_validation = (
                                    model_meta.get("return_forecast", {}) or {}
                                )
                                price_forecast_status = str(
                                    price_forecast_validation.get(
                                        "status", "REGRESSION_HEAD_PRESENT_UNVALIDATED"
                                    )
                                )
                            else:
                                price_forecast_status = "INVALID_PRICE_FROM_RETURN_HEAD"
                        else:
                            price_forecast_status = "INVALID_RETURN_FORECAST"
                except Exception as return_exc:
                    price_forecast_status = (
                        f"RETURN_HEAD_ERROR:{type(return_exc).__name__}"
                    )

            prediction_ready_at = pd.Timestamp.now(tz="UTC")
            row_out = {
                "symbol": root,
                "horizon_min": horizon,
                "decision_timestamp": str(latest_ts),
                "reference_close": float(reference_close) if pd.notna(reference_close) else np.nan,
                "expected_return": expected_return,
                "predicted_price": predicted_price,
                "price_forecast_status": price_forecast_status,
                "price_forecast_validation": price_forecast_validation,
                "probability_up": p,
                "signal": signal,
                "feature_coverage": coverage,
                "active_model_feature_count": len(model_features),
                "minimum_active_model_feature_coverage": MIN_LIVE_FEATURE_COVERAGE,
                "signal_reason": (
                    "VALIDATED_THRESHOLD"
                    if signal != "ABSTAIN" else "NO_VALIDATED_ACTION_BAND"
                ),
            }
            if diagnostics:
                input_value = price_input_report.get("latest_timestamp_utc")
                try:
                    input_age_seconds = (prediction_ready_at - pd.Timestamp(input_value)).total_seconds()
                except Exception:
                    input_age_seconds = None
                metrics = {
                    "latest_input_timestamp_utc": input_value,
                    "decision_timestamp_utc": pd.Timestamp(latest_ts).isoformat(),
                    "prediction_ready_at_utc": prediction_ready_at.isoformat(),
                    "input_age_at_prediction_ready_seconds": (
                        round(float(input_age_seconds), 3) if input_age_seconds is not None else None
                    ),
                    "latest_input_to_decision_gap_minutes": (
                        round(float(input_to_decision_gap_minutes), 3)
                        if input_to_decision_gap_minutes is not None else None
                    ),
                    "feature_construction_seconds": round(feature_build_seconds, 6),
                    "model_artifact_load_seconds": round(artifact_load_seconds, 6),
                    "model_artifact_cache_hit": bool(model_cache_hit),
                    "feature_matrix_preparation_seconds": round(feature_matrix_seconds, 6),
                    "model_prediction_seconds": round(model_prediction_seconds, 6),
                    "horizon_total_seconds": round(_perf_time.perf_counter() - horizon_started_perf, 6),
                    "input_cadence_median_minutes": price_input_report.get("median_interval_minutes"),
                    "decision_schedule_minutes": 5,
                }
                row_out["runtime_diagnostics"] = metrics
                runtime_event(
                    "intraday_horizon_timing",
                    symbol=root,
                    horizon_min=horizon,
                    signal=signal,
                    probability_up=p,
                    feature_count=len(model_features),
                    **metrics,
                )
            rows.append(row_out)
        except Exception as exc:
            rows.append({
                "symbol": root,
                "horizon_min": horizon,
                "signal": "ABSTAIN",
                "reason": f"{type(exc).__name__}: {exc}",
            })
    if diagnostics:
        runtime_event(
            "intraday_prediction_call_completed",
            symbol=root,
            horizons_attempted=[5, 10, 15, 20, 25, 30],
            rows_returned=int(len(rows)),
            prediction_ready_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
            total_runtime_seconds=round(_perf_time.perf_counter() - call_started_perf, 6),
        )
    return rows