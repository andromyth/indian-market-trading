"""Five-minute quote snapshots for the fixed owner-selected 500-symbol roster.

This module records observed INDstocks prices only for the pinned roster. It
deliberately distinguishes a provider quote snapshot from an exchange-timestamped
OHLCV candle. Predictions are attached only when the prediction engine emitted them.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from all_universe_snapshot_assembly import (
    DEFAULT_WORKER_COUNT,
    SHARD_STATUS,
    SHARD_SUBDIR,
    assemble_shards,
    canonical_artifact_names,
    normalized_cycle_minute,
    shard_artifact_names,
    shard_universe,
    write_assembly_manifest_only,
    write_canonical_snapshot,
)

IST = ZoneInfo("Asia/Kolkata")
BATCH_SIZE = 500  # Below the 1,000-symbol API limit and avoids oversized GET URLs.


def _snapshot_coverage_pct(complete_symbols: int, expected_symbols: int) -> float:
    """Return selected-roster quote coverage as an informational metric."""
    expected = max(0, int(expected_symbols))
    complete = max(0, int(complete_symbols))
    return round(100.0 * complete / expected, 3) if expected else 0.0


def _selected_roster(universe: Iterable[str] | None = None) -> list[str]:
    """Return only symbols in the owner-pinned 500-symbol roster."""
    from prediction_universe import authoritative_roster

    pinned = authoritative_roster()
    if not pinned:
        raise RuntimeError("backfill_universe.json is missing or has no selected symbols")
    if universe is None:
        return pinned
    requested = {
        str(symbol).strip().upper().replace(".NS", "").replace(".BO", "")
        for symbol in universe if str(symbol).strip()
    }
    selected = [symbol for symbol in pinned if symbol in requested]
    if not selected:
        raise RuntimeError("Requested capture symbols do not intersect the pinned 500-symbol roster")
    return selected


QUOTE_REQUEST_MAX_ATTEMPTS = 3
QUOTE_FALLBACK_BATCH_SIZE = 50
QUOTE_RETRY_BACKOFF_SECONDS = 0.35
REMOTE_ROOT = "production_data/all_universe_minute_snapshots"


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return _clean(value.item())
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


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _instrument_map() -> dict[str, str]:
    """Map canonical NSE symbols to INDstocks NSE security codes."""
    from indstocks_client import _load_instruments

    master = _load_instruments("NSE")
    if master is None or master.empty:
        raise RuntimeError("INDstocks returned an empty NSE instrument master")
    frame = master.copy()
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    required = {"TRADING_SYMBOL", "SECURITY_ID"}
    if not required.issubset(frame.columns):
        raise RuntimeError(
            "INDstocks instrument master lacks TRADING_SYMBOL/SECURITY_ID"
        )
    if "EXCH" in frame.columns:
        frame = frame.loc[frame["EXCH"].astype(str).str.upper().eq("NSE")].copy()
    if "SERIES" in frame.columns:
        # The official NSE roster also includes eligible BE/BZ/SM/ST/SZ/GB/RR
        # series. Keep supported equity series instead of dropping all of them
        # whenever the provider master contains any EQ-series rows. Prefer EQ
        # for duplicate trading symbols, then retain the first supported series.
        from delivery_training_universe import EQUITY_SERIES

        frame["SERIES"] = frame["SERIES"].astype(str).str.strip().str.upper()
        frame = frame.loc[
            frame["SERIES"].isin(EQUITY_SERIES) | frame["SERIES"].eq("")
        ].copy()
        frame["_series_priority"] = frame["SERIES"].ne("EQ").astype("int8")
    else:
        frame["_series_priority"] = 1
    frame["TRADING_SYMBOL"] = frame["TRADING_SYMBOL"].astype(str).str.strip().str.upper()
    frame["SECURITY_ID"] = frame["SECURITY_ID"].astype(str).str.strip()
    frame = frame.loc[frame["SECURITY_ID"].str.fullmatch(r"\d+")]
    frame = frame.sort_values("_series_priority", kind="stable")
    frame = frame.drop_duplicates("TRADING_SYMBOL", keep="first")
    return {
        str(row.TRADING_SYMBOL): f"NSE_{row.SECURITY_ID}"
        for row in frame[["TRADING_SYMBOL", "SECURITY_ID"]].itertuples(index=False)
        if str(row.TRADING_SYMBOL).strip()
    }


def _is_shared_quote_auth_failure(error: Any) -> bool:
    """Recognize shared INDstocks auth/token failures that small-batch retries cannot heal."""
    detail = str(error or "").lower()
    return any(marker in detail for marker in (
        "token generation",
        "automatic re-authentication failed",
        "automatic reauthentication failed",
        "freshly generated indstocks token was rejected",
        "access token may have expired",
        "invalid access token",
        "access_token is either incorrect",
        "http 401; the access token",
    ))


def _fetch_quotes_resilient(
    request_fn: Any,
    codes: list[str],
    *,
    max_attempts: int = QUOTE_REQUEST_MAX_ATTEMPTS,
    fallback_batch_size: int = QUOTE_FALLBACK_BATCH_SIZE,
    retry_backoff_seconds: float = QUOTE_RETRY_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Fetch quotes with bounded retries and small-batch fallback for gaps.

    A transient HTTP/provider failure for one 500-code batch must not erase
    hundreds of otherwise available observations for that five-minute slot.
    Successful codes are retained; only unresolved codes are retried. When a
    large request continues to fail or returns only a partial map, unresolved
    codes are retried in smaller groups. Symbols still absent from the provider
    response remain explicitly missing; no prices are fabricated.
    """
    requested = list(dict.fromkeys(str(code) for code in codes if str(code)))
    quotes: dict[str, dict[str, Any]] = {}
    last_error_by_code: dict[str, str] = {}
    error_examples: list[str] = []
    attempts_total = 0
    fallback_batch_count = 0
    auth_failure_detail: str | None = None

    def fetch_group(group: list[str], *, allow_split: bool) -> None:
        nonlocal attempts_total, fallback_batch_count, auth_failure_detail
        pending = list(group)
        if auth_failure_detail is not None:
            for code in pending:
                last_error_by_code[code] = auth_failure_detail
            return
        last_error: str | None = None
        for attempt in range(max(1, int(max_attempts))):
            if not pending:
                break
            attempts_total += 1
            try:
                response = request_fn(
                    "GET",
                    "/market/quotes/full",
                    params={"scrip-codes": ",".join(pending)},
                )
                body = response.json()
                data = body.get("data", {}) if isinstance(body, dict) else {}
                if not isinstance(data, dict):
                    raise RuntimeError("INDstocks quote response has no data object")
                success = (
                    body.get("success") is True
                    or str(body.get("status", "")).strip().lower() == "success"
                )
                if not success:
                    raise RuntimeError(
                        str(body.get("message") or body.get("error") or "quote request not successful")
                    )
                for code in pending:
                    quote = data.get(code)
                    if isinstance(quote, dict):
                        quotes[code] = quote
                        last_error_by_code.pop(code, None)
                pending = [code for code in pending if code not in quotes]
                # If the provider returned a valid but incomplete map, retry only
                # missing codes rather than repeatedly downloading successful ones.
                last_error = None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                if len(error_examples) < 8:
                    error_examples.append(last_error)
                if _is_shared_quote_auth_failure(last_error):
                    auth_failure_detail = last_error
                    for code in pending:
                        last_error_by_code[code] = last_error
                    return
            if not pending:
                return
            if auth_failure_detail is not None:
                for code in pending:
                    last_error_by_code[code] = auth_failure_detail
                return
            if attempt + 1 < max(1, int(max_attempts)):
                time.sleep(max(0.0, float(retry_backoff_seconds)) * (attempt + 1))

        if (
            pending
            and auth_failure_detail is None
            and allow_split
            and len(pending) > max(1, int(fallback_batch_size))
        ):
            fallback_batch_count += (len(pending) + fallback_batch_size - 1) // fallback_batch_size
            for offset in range(0, len(pending), fallback_batch_size):
                fetch_group(pending[offset:offset + fallback_batch_size], allow_split=False)
                if auth_failure_detail is not None:
                    break
            return
        if last_error:
            for code in pending:
                last_error_by_code[code] = last_error

    fetch_group(requested, allow_split=True)
    unresolved = [code for code in requested if code not in quotes]
    return {
        "quotes": quotes,
        "requested_code_count": len(requested),
        "matched_code_count": len(quotes),
        "unresolved_codes": unresolved,
        "attempt_count": attempts_total,
        "fallback_batch_count": fallback_batch_count,
        "last_error_by_code": last_error_by_code,
        "error_examples": error_examples,
        "auth_failure": auth_failure_detail is not None,
        "auth_error": auth_failure_detail,
    }


def _load_forecast_rows(ledger_path: Path | None) -> list[dict[str, Any]]:
    if ledger_path is None or not Path(ledger_path).is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with Path(ledger_path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("forecast_key"):
                    rows.append(row)
    except OSError:
        return []
    return rows


def _prediction_map(
    snapshot_rows: list[dict[str, Any]] | None,
    cycle_minute: pd.Timestamp,
) -> dict[str, list[dict[str, Any]]]:
    """Include only forecasts actually emitted during this collector minute."""
    mapped: dict[str, list[dict[str, Any]]] = {}
    for snapshot in snapshot_rows or []:
        symbol = str(snapshot.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        predictions = snapshot.get("predictions") or []
        if not isinstance(predictions, list):
            continue
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            observed_at = pd.to_datetime(
                pred.get("worker_observed_at_utc")
                or pred.get("worker_observed_at_ist"),
                errors="coerce",
                utc=True,
            )
            if pd.isna(observed_at):
                continue
            # A tick may run a little after its anchor; forecasts observed
            # anywhere inside the slot bucket still belong to that slot.
            observed_minute = observed_at.tz_convert(IST).floor("min")
            if not (
                cycle_minute <= observed_minute
                < cycle_minute + pd.Timedelta(minutes=5)
            ):
                continue
            mapped.setdefault(symbol, []).append(_clean(pred))
    return mapped


def _outcomes_for_observation(
    symbol: str,
    actual_price: float | None,
    observed_at: pd.Timestamp,
    forecasts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match only forecasts whose intended target falls in this observed minute."""
    if actual_price is None:
        return []
    obs_minute_utc = observed_at.tz_convert("UTC").floor("min")
    outcomes: list[dict[str, Any]] = []
    for forecast in forecasts:
        if str(forecast.get("symbol") or "").strip().upper().replace(".NS", "") != symbol:
            continue
        target = pd.to_datetime(
            forecast.get("target_timestamp_utc"), errors="coerce", utc=True
        )
        if pd.isna(target) or target.floor("min") != obs_minute_utc:
            continue
        reference = _as_float(
            forecast.get("reference_close", forecast.get("actual_observed_close"))
        )
        predicted_price = _as_float(forecast.get("predicted_price"))
        signal = str(forecast.get("signal") or "").strip().upper()
        actual_return_pct = (
            (actual_price / reference - 1.0) * 100.0
            if reference is not None and reference > 0 else None
        )
        directional_correct = None
        if actual_return_pct is not None:
            if signal == "BUY":
                directional_correct = actual_return_pct > 0.0
            elif signal == "SELL":
                directional_correct = actual_return_pct < 0.0
        outcomes.append({
            "forecast_key": forecast.get("forecast_key"),
            "symbol": symbol,
            "horizon_min": forecast.get("horizon_min"),
            "target_timestamp_utc": target.isoformat(),
            "actual_quote_observed_at_utc": observed_at.tz_convert("UTC").isoformat(),
            "actual_target_price": actual_price,
            "actual_target_price_source": "INDSTOCKS_BATCH_QUOTE",
            "actual_price_timestamp_basis": "collector_observation_time_not_exchange_tick_timestamp",
            "reference_close": reference,
            "actual_return_pct_from_reference": actual_return_pct,
            "predicted_price": predicted_price,
            "absolute_price_error": (
                abs(actual_price - predicted_price)
                if predicted_price is not None else None
            ),
            "signal": signal or None,
            "directional_correct_before_costs": directional_correct,
            "outcome_status": "MATCHED_TARGET_MINUTE_QUOTE",
            "validation_note": (
                "Point-in-time quote observed during the target minute; this is not "
                "a substitute for an exchange-stamped one-minute candle close."
            ),
        })
    return outcomes


def _quote_row(
    symbol: str,
    instrument_code: str | None,
    quote: dict[str, Any] | None,
    observed_at: pd.Timestamp,
    cycle_minute: pd.Timestamp,
    prediction_map: dict[str, list[dict[str, Any]]],
    forecast_rows: list[dict[str, Any]],
    provider_error: str | None = None,
) -> dict[str, Any]:
    quote = quote or {}
    ltp = _as_float(quote.get("live_price", quote.get("ltp")))
    if ltp is not None and ltp <= 0:
        provider_error = provider_error or "Provider returned a non-positive live price"
        ltp = None
    forecasts = prediction_map.get(symbol, [])
    outcomes = _outcomes_for_observation(symbol, ltp, observed_at, forecast_rows)
    status = (
        "PASS" if ltp is not None
        else "QUOTE_REQUEST_ERROR" if provider_error
        else "MISSING_QUOTE"
    )
    return {
        "record_type": "all_universe_minute_market_observation",
        "schema_version": "all-universe-minute-snapshot-v1",
        "symbol": symbol,
        "exchange": "NSE",
        "instrument_code": instrument_code,
        "cycle_minute_ist": cycle_minute.isoformat(),
        "quote_observed_at_utc": observed_at.tz_convert("UTC").isoformat(),
        "quote_observed_at_ist": observed_at.isoformat(),
        "quote_timestamp_basis": "collector_observation_time_not_exchange_tick_timestamp",
        "quote_freshness_status": (
            "PROVIDER_PRICE_PRESENT_TIMESTAMP_NOT_SUPPLIED"
            if ltp is not None else "UNAVAILABLE"
        ),
        "quote_source": "INDSTOCKS_BATCH_QUOTE",
        "quote_status": status,
        "quote_error": provider_error,
        "actual_observed_price": ltp,
        "day_open": _as_float(quote.get("day_open")),
        "day_high": _as_float(quote.get("day_high")),
        "day_low": _as_float(quote.get("day_low")),
        "previous_close": _as_float(quote.get("prev_close")),
        "day_volume": _as_float(quote.get("volume")),
        "day_change_pct": _as_float(quote.get("day_change_percentage")),
        "is_one_minute_ohlcv_candle": False,
        "forecast_status": "AVAILABLE" if forecasts else "NO_FORECAST_EMITTED_FOR_SYMBOL",
        "forecast_count": len(forecasts),
        "current_forecasts_json": json.dumps(_clean(forecasts), separators=(",", ":")),
        "matched_outcome_count": len(outcomes),
        "matched_outcomes_json": json.dumps(_clean(outcomes), separators=(",", ":")),
    }


def _write_manifest(
    store: Any,
    local_dir: Path,
    payload: dict[str, Any],
    *,
    remote_dir: str | None = None,
    name: str | None = None,
    write_latest: bool = True,
) -> dict[str, Any]:
    """Persist both a minute-specific audit record and an optional latest pointer.

    Shard capture writes into the ``shards`` subfolder with a shard-qualified
    name and must not update the canonical ``latest_manifest.json`` pointer;
    only the explicit assembly step does that.
    """
    path = local_dir / "latest_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_clean(payload), indent=2, sort_keys=True)

    minute = pd.to_datetime(payload.get("cycle_minute_ist"), errors="coerce")
    if pd.isna(minute):
        raise RuntimeError("Cannot persist a per-minute manifest without cycle_minute_ist")
    minute_name = name or f"manifest_{minute:%Y%m%d_%H%M}.json"
    minute_path = local_dir / minute_name
    minute_path.write_text(encoded, encoding="utf-8")
    minute_remote = f"{remote_dir or f'{REMOTE_ROOT}/{minute:%Y/%m/%d}'}/{minute_name}"

    minute_result = store.upload_or_update(
        minute_path, minute_remote, mime_type="application/json"
    )
    if minute_result.get("status") != "OK":
        raise RuntimeError(f"Drive per-minute manifest write was not confirmed: {minute_result}")

    minute_path.with_suffix(".uploaded.json").write_text(
        json.dumps({
            "status": "OK",
            "remote_path": minute_remote,
            "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
        }, sort_keys=True),
        encoding="utf-8",
    )

    if not write_latest:
        return {"minute_manifest": minute_result, "latest_manifest": None}

    path.write_text(encoded, encoding="utf-8")
    latest_result = store.upload_or_update(
        path, f"{REMOTE_ROOT}/latest_manifest.json", mime_type="application/json"
    )
    if latest_result.get("status") != "OK":
        raise RuntimeError(f"Drive latest-manifest write was not confirmed: {latest_result}")
    return {"minute_manifest": minute_result, "latest_manifest": latest_result}



def _remote_for_local_artifact(local_path: Path) -> str | None:
    """Map a local snapshot/manifest file name to its Drive path.

    Supports both canonical slot artifacts and shard-qualified artifacts such as
    ``snapshot_20261112_1005_w03of08.parquet``.
    """
    match = re.fullmatch(
        r"(snapshot|manifest)_(\d{8})_(\d{4})(?:_w(\d{2})of(\d{2}))?\.(?:parquet|json)",
        local_path.name,
    )
    if not match:
        return None
    kind, yyyymmdd, hhmm, shard_index, shard_count = match.groups()
    day = f"{yyyymmdd[:4]}/{yyyymmdd[4:6]}/{yyyymmdd[6:]}"
    if shard_index is not None and shard_count is not None and int(shard_count) > 1:
        return f"{REMOTE_ROOT}/{day}/shards/{local_path.name}"
    return f"{REMOTE_ROOT}/{day}/{local_path.name}"


def _upload_pending_snapshots(store: Any, local_dir: Path, current_name: str) -> dict[str, Any]:
    """Retry local minute shards whose Drive upload was not confirmed."""
    result: dict[str, Any] = {"attempted": 0, "recovered": 0, "errors": []}
    for local_path in sorted(local_dir.glob("snapshot_*.parquet")):
        if local_path.name == current_name:
            continue
        marker_path = local_path.with_suffix(".uploaded.json")
        if marker_path.is_file():
            continue
        try:
            remote = _remote_for_local_artifact(local_path)
            if not remote:
                raise RuntimeError(f"Unrecognized snapshot file name: {local_path.name}")
            result["attempted"] += 1
            upload = store.upload_or_update(
                local_path, remote, mime_type="application/vnd.apache.parquet"
            )
            if upload.get("status") != "OK":
                raise RuntimeError(f"Drive retry was not confirmed: {upload}")
            marker_path.write_text(
                json.dumps({
                    "status": "OK",
                    "remote_path": remote,
                    "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
                }, sort_keys=True),
                encoding="utf-8",
            )
            result["recovered"] += 1
        except Exception as exc:
            result["errors"].append({
                "local_path": str(local_path),
                "error": f"{type(exc).__name__}: {exc}",
            })
    return result


def _upload_pending_manifests(
    store: Any, local_dir: Path, current_name: str
) -> dict[str, Any]:
    """Retry local per-minute manifests whose Drive writes were not confirmed."""
    result: dict[str, Any] = {"attempted": 0, "recovered": 0, "errors": []}
    for local_path in sorted(local_dir.glob("manifest_*.json")):
        # Success receipts are named manifest_YYYYMMDD_HHMM.uploaded.json and
        # also match the broad glob above. Only retry canonical minute payloads,
        # including shard-qualified ones.
        if not re.fullmatch(
            r"manifest_\d{8}_\d{4}(?:_w\d{2}of\d{2})?\.json", local_path.name
        ):
            continue
        if local_path.name == current_name:
            continue
        marker_path = local_path.with_suffix(".uploaded.json")
        if marker_path.is_file():
            continue
        try:
            remote = _remote_for_local_artifact(local_path)
            if not remote:
                raise RuntimeError(f"Unrecognized manifest file name: {local_path.name}")
            result["attempted"] += 1
            upload = store.upload_or_update(
                local_path, remote, mime_type="application/json"
            )
            if upload.get("status") != "OK":
                raise RuntimeError(f"Drive manifest retry was not confirmed: {upload}")
            marker_path.write_text(
                json.dumps({
                    "status": "OK",
                    "remote_path": remote,
                    "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
                }, sort_keys=True),
                encoding="utf-8",
            )
            result["recovered"] += 1
        except Exception as exc:
            result["errors"].append({
                "local_path": str(local_path),
                "error": f"{type(exc).__name__}: {exc}",
            })
    return result


def capture_all_universe_minute_snapshot(
    *,
    now: datetime | pd.Timestamp,
    worker_root: Path,
    daily_ledger: Path | None,
    prediction_snapshots: list[dict[str, Any]] | None = None,
    worker_id: int = 0,
    worker_count: int = 1,
    universe: Iterable[str] | None = None,
    instruments: dict[str, str] | None = None,
    return_frame: bool = False,
) -> dict[str, Any]:
    """Collect one slice of the official NSE universe and persist it for a slot.

    With ``worker_count > 1`` the universe is split into disjoint shards and
    this worker collects only ``symbols[worker_id::worker_count]``, writing a
    shard-qualified snapshot and manifest under the slot's ``shards`` folder.
    Shards are combined by the explicit assembly step
    (:mod:`all_universe_snapshot_assembly`); a shard write is never presented as
    a complete canonical snapshot. With ``worker_count == 1`` the behavior is
    the original single-artifact capture.
    """
    worker_count = max(1, int(worker_count))
    worker_id = max(0, min(int(worker_id), worker_count - 1))
    from indstocks_client import _request, configured
    from persistent_store import GoogleDriveStore

    now_ts = pd.Timestamp(now)
    now_ts = now_ts.tz_localize(IST) if now_ts.tzinfo is None else now_ts.tz_convert(IST)
    cycle_minute = now_ts.floor("5min")
    artifact_names = shard_artifact_names(cycle_minute, worker_id, worker_count)
    local_dir = Path(worker_root) / "all_universe_minute_snapshots"
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_day = cycle_minute.strftime("%Y/%m/%d")
    remote_path = artifact_names["snapshot_remote"]
    remote_manifest_dir = (
        f"{REMOTE_ROOT}/{remote_day}/{SHARD_SUBDIR}"
        if worker_count > 1
        else f"{REMOTE_ROOT}/{remote_day}"
    )
    manifest: dict[str, Any] = {
        "schema_version": "all-universe-minute-snapshot-v2",
        "artifact_kind": "SHARD" if worker_count > 1 else "CANONICAL_DIRECT",
        "capture_interval_minutes": 5,
        "cycle_minute_ist": cycle_minute.isoformat(),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "INDSTOCKS_BATCH_QUOTE",
        "status": "STARTING",
        "snapshot_persisted": False,
        "worker_id": worker_id,
        "worker_count": worker_count,
        "expected_symbol_count": 0,
        "universe_symbol_count": 0,
        "shard_symbol_count": 0,
        "quoted_symbol_count": 0,
        "missing_quote_count": 0,
        "unmapped_symbol_count": 0,
        "forecast_symbol_count": 0,
        "matched_outcome_count": 0,
        "remote_snapshot_path": remote_path,
    }

    store = GoogleDriveStore()
    pending_recovery = _upload_pending_snapshots(
        store, local_dir, artifact_names["snapshot_name"]
    )
    pending_manifest_recovery = _upload_pending_manifests(
        store, local_dir, artifact_names["manifest_name"]
    )
    manifest["pending_snapshot_recovery"] = pending_recovery
    manifest["pending_manifest_recovery"] = pending_manifest_recovery
    if not configured():
        manifest.update({
            "status": "BLOCKED_PROVIDER_CREDENTIALS",
            "error": (
                "INDstocks TOTP credentials are not configured in the Cloud Run service's "
                "secret-backed environment. Only Cloud Run may access INDstocks."
            ),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        try:
            _write_manifest(
                store, local_dir, manifest,
                remote_dir=remote_manifest_dir,
                name=artifact_names["manifest_name"],
                write_latest=(worker_count == 1),
            )
        except Exception as exc:
            manifest["manifest_sync_error"] = f"{type(exc).__name__}: {exc}"
        return _clean(manifest)

    universe_symbols = _selected_roster(universe)
    universe_filter: dict[str, Any] = {
        "status": "PINNED_500_ROSTER",
        "selected_symbol_count": len(universe_symbols),
    }
    if instruments is None:
        instruments = _instrument_map()
    assigned_symbols = shard_universe(universe_symbols, worker_count)[worker_id]
    prediction_map = _prediction_map(prediction_snapshots, cycle_minute)
    forecast_rows = _load_forecast_rows(daily_ledger)
    manifest["universe_filter"] = universe_filter
    manifest["expected_symbol_count"] = len(universe_symbols)
    manifest["universe_symbol_count"] = len(universe_symbols)
    manifest["shard_symbol_count"] = len(assigned_symbols)
    manifest["shard_symbols"] = assigned_symbols
    manifest["unmapped_symbol_count"] = sum(
        1 for symbol in assigned_symbols if symbol not in instruments
    )
    manifest["forecast_symbol_count"] = sum(
        1 for symbol in assigned_symbols if prediction_map.get(symbol)
    )

    rows: list[dict[str, Any]] = []
    request_errors = 0
    batch_diagnostics: list[dict[str, Any]] = []
    provider_auth_circuit_open = False
    provider_auth_error: str | None = None
    all_codes = [
        (symbol, instruments[symbol])
        for symbol in assigned_symbols
        if symbol in instruments
    ]

    for batch_index, offset in enumerate(range(0, len(all_codes), BATCH_SIZE), start=1):
        batch = all_codes[offset:offset + BATCH_SIZE]
        codes = [code for _, code in batch]
        auth_circuit_blocked = provider_auth_circuit_open
        if auth_circuit_blocked:
            fetched = {
                "quotes": {},
                "requested_code_count": len(codes),
                "matched_code_count": 0,
                "unresolved_codes": list(codes),
                "attempt_count": 0,
                "fallback_batch_count": 0,
                "last_error_by_code": {
                    code: provider_auth_error or "Shared INDstocks authentication circuit is open"
                    for code in codes
                },
                "error_examples": [provider_auth_error] if provider_auth_error else [],
                "auth_failure": True,
                "auth_error": provider_auth_error,
            }
        else:
            fetched = _fetch_quotes_resilient(_request, codes)
            if fetched.get("auth_failure"):
                provider_auth_circuit_open = True
                provider_auth_error = str(
                    fetched.get("auth_error") or "Shared INDstocks authentication failed"
                )
        data = fetched["quotes"]
        unresolved = set(fetched["unresolved_codes"])
        error_by_code = fetched["last_error_by_code"]
        matched_code_count = sum(1 for _, code in batch if isinstance(data.get(code), dict))
        if any(code in error_by_code for code in unresolved):
            request_errors += 1
        if auth_circuit_blocked:
            batch_status = "AUTH_CIRCUIT_BLOCKED"
        elif fetched.get("auth_failure"):
            batch_status = "AUTH_FAILURE"
        else:
            batch_status = "OK" if not unresolved else "PARTIAL"
        batch_diagnostics.append({
            "batch_index": batch_index,
            "requested_code_count": len(codes),
            "response_data_key_count": matched_code_count,
            "matched_response_code_count": matched_code_count,
            "status": batch_status,
            "request_attempt_count": fetched["attempt_count"],
            "fallback_batch_count": fetched["fallback_batch_count"],
            "unresolved_code_count": len(unresolved),
            "auth_failure": bool(fetched.get("auth_failure")),
            "auth_circuit_blocked": auth_circuit_blocked,
            "unresolved_symbol_samples": [
                symbol for symbol, code in batch if code in unresolved
            ][:10],
            "error_examples": fetched["error_examples"],
        })
        observed_at = pd.Timestamp.now(tz=IST)
        for symbol, code in batch:
            quote = data.get(code)
            provider_error = error_by_code.get(code)
            if quote is None and code in unresolved and not provider_error:
                # A successful provider response without this code means the
                # quote is missing, not that the whole HTTP request failed.
                provider_error = None
            rows.append(_quote_row(
                symbol, code, quote if isinstance(quote, dict) else None,
                observed_at, cycle_minute, prediction_map, forecast_rows,
                provider_error=provider_error,
            ))

    for symbol in assigned_symbols:
        if symbol not in instruments:
            rows.append(_quote_row(
                symbol, None, None, pd.Timestamp.now(tz=IST), cycle_minute,
                prediction_map, forecast_rows,
                provider_error="Symbol missing from INDstocks instrument master",
            ))

    manifest["quote_request_error_count"] = request_errors
    manifest["provider_auth_circuit_open"] = provider_auth_circuit_open
    manifest["provider_auth_error"] = provider_auth_error
    manifest["quote_batch_diagnostics"] = batch_diagnostics

    frame = pd.DataFrame(rows)
    if frame.empty:
        manifest["status"] = "NO_ROWS"
        manifest["error"] = "The all-universe quote collector returned no rows."
    else:
        # Count coverage before attempting Drive upload, so failure reports retain
        # the real collector result rather than dropping coverage statistics.
        manifest["quoted_symbol_count"] = int(frame["quote_status"].eq("PASS").sum())
        manifest["missing_quote_count"] = int(frame["quote_status"].ne("PASS").sum())
        manifest["capture_data_status"] = (
            "BLOCKED_PROVIDER_AUTH"
            if provider_auth_circuit_open and manifest["quoted_symbol_count"] == 0
            else "COMPLETE_SHARD"
            if manifest["quoted_symbol_count"] == manifest["shard_symbol_count"]
            else "PARTIAL"
        )
        manifest["matched_outcome_count"] = int(frame["matched_outcome_count"].sum())
        manifest["row_count"] = int(len(frame))
        manifest["shard_row_count"] = int(len(frame))
        complete_coverage_pct = _snapshot_coverage_pct(
            manifest["quoted_symbol_count"], manifest["expected_symbol_count"]
        )
        manifest["complete_quote_symbol_count"] = manifest["quoted_symbol_count"]
        manifest["complete_quote_symbol_coverage_pct"] = complete_coverage_pct
        manifest["coverage_is_informational"] = True
        manifest["forecast_record_count"] = int(frame["forecast_count"].sum())
        local_path = local_dir / artifact_names["snapshot_name"]
        temp_path = local_path.with_suffix(".tmp.parquet")
        frame.to_parquet(temp_path, index=False, compression="zstd")
        temp_path.replace(local_path)
        try:
            drive_result = store.upload_or_update(
                local_path, remote_path, mime_type="application/vnd.apache.parquet"
            )
            manifest["snapshot_drive_sync"] = drive_result
            if drive_result.get("status") != "OK":
                raise RuntimeError(f"Drive snapshot upload was not confirmed: {drive_result}")
            local_path.with_suffix(".uploaded.json").write_text(
                json.dumps({
                    "status": "OK",
                    "remote_path": remote_path,
                    "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
                }, sort_keys=True),
                encoding="utf-8",
            )
            manifest["status"] = SHARD_STATUS if worker_count > 1 else "STORED"
            manifest["snapshot_persisted"] = True
            manifest["snapshot_bytes"] = int(local_path.stat().st_size)
            manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["snapshot_drive_path"] = remote_path
        except Exception as exc:
            manifest.update({
                "status": "PERSISTENCE_ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "local_snapshot_path": str(local_path),
                "row_count": int(len(frame)),
            })

    if (
        pending_recovery.get("errors") or pending_manifest_recovery.get("errors")
    ) and manifest.get("status") not in {
        "BLOCKED_PROVIDER_CREDENTIALS", "NO_ROWS", "PERSISTENCE_ERROR"
    }:
        manifest["status"] = "PERSISTENCE_ERROR"
        manifest["recovery_error"] = (
            "One or more previous minute snapshots or manifests still failed Drive "
            "recovery; inspect pending_snapshot_recovery and pending_manifest_recovery."
        )
        if not manifest.get("error"):
            manifest["error"] = manifest["recovery_error"]

    manifest_write_kwargs = {
        "remote_dir": remote_manifest_dir,
        "name": artifact_names["manifest_name"],
        "write_latest": worker_count == 1,
    }
    try:
        manifest["manifest_drive_sync"] = _write_manifest(
            store, local_dir, manifest, **manifest_write_kwargs
        )
    except Exception as exc:
        manifest["manifest_sync_error"] = f"{type(exc).__name__}: {exc}"
        manifest["status"] = "PERSISTENCE_ERROR"
        # Make a best-effort second write of the per-minute audit showing the
        # failed manifest sync; never mask the original failure in the result.
        try:
            manifest["manifest_recovery_sync"] = _write_manifest(
                store, local_dir, manifest, **manifest_write_kwargs
            )
        except Exception as retry_exc:
            manifest["manifest_recovery_sync_error"] = (
                f"{type(retry_exc).__name__}: {retry_exc}"
            )

    cleaned = _clean(manifest)
    if return_frame:
        cleaned["_frame"] = frame
    return cleaned


def _shard_result_is_persisted(result: dict[str, Any]) -> bool:
    if result.get("_frame") is None:
        return False
    return (
        result.get("snapshot_persisted") is True
        and str(result.get("status") or "").upper() in {SHARD_STATUS, "STORED"}
    )


def capture_all_universe_minute_snapshot_sharded(
    *,
    now: datetime | pd.Timestamp,
    worker_root: Path,
    daily_ledger: Path | None,
    prediction_snapshots: list[dict[str, Any]] | None = None,
    worker_count: int = DEFAULT_WORKER_COUNT,
    universe: Iterable[str] | None = None,
    instruments: dict[str, str] | None = None,
    store: Any | None = None,
    retry_failed_shards: int = 1,
    expected_universe: Iterable[str] | None = None,
    strict_universe: bool = False,
) -> dict[str, Any]:
    """Coordinated sharded capture followed by one explicit assembly.

    One coordinated process fans the universe out to ``worker_count`` shard
    collectors that share the process-wide INDstocks request gate. Every shard
    persists its own artifact. Only when all shards are present and valid does
    the assembly step publish the canonical snapshot that the prediction
    compiler consumes. If any shard is missing or invalid, no canonical snapshot
    is written for the slot; an incomplete audit manifest is recorded instead.
    """
    cycle_minute = normalized_cycle_minute(now)
    count = max(1, int(worker_count))
    worker_root = Path(worker_root)

    if count == 1:
        # Degenerate non-sharded path: the single collector already writes the
        # canonical artifact, so no separate assembly pass is required.
        return capture_all_universe_minute_snapshot(
            now=cycle_minute,
            worker_root=worker_root,
            daily_ledger=daily_ledger,
            prediction_snapshots=prediction_snapshots,
            worker_id=0,
            worker_count=1,
            universe=universe,
            instruments=instruments,
        )

    from indstocks_client import configured
    from persistent_store import GoogleDriveStore

    if not configured():
        return _clean({
            "schema_version": "all-universe-minute-snapshot-v2",
            "mode": "SHARDED_ASSEMBLY",
            "status": "BLOCKED_PROVIDER_CREDENTIALS",
            "snapshot_persisted": False,
            "cycle_minute_ist": cycle_minute.isoformat(),
            "worker_count": count,
            "error": (
                "INDstocks TOTP credentials are not configured in the Cloud Run "
                "service's secret-backed environment."
            ),
        })

    universe_symbols = _selected_roster(universe)
    if instruments is None:
        instruments = _instrument_map()
    store = store or GoogleDriveStore()

    def run_shard(index: int) -> dict[str, Any]:
        try:
            return capture_all_universe_minute_snapshot(
                now=cycle_minute,
                worker_root=worker_root,
                daily_ledger=daily_ledger,
                prediction_snapshots=prediction_snapshots,
                worker_id=index,
                worker_count=count,
                universe=universe_symbols,
                instruments=instruments,
                return_frame=True,
            )
        except Exception as exc:
            names = shard_artifact_names(cycle_minute, index, count)
            return {
                "status": "ERROR",
                "snapshot_persisted": False,
                "worker_id": index,
                "worker_count": count,
                "cycle_minute_ist": cycle_minute.isoformat(),
                "remote_snapshot_path": names["snapshot_remote"],
                "error": f"{type(exc).__name__}: {exc}",
            }

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="nse-shard") as pool:
        results = list(pool.map(run_shard, range(count)))

    for attempt in range(max(0, int(retry_failed_shards))):
        retry_ids = [
            index for index, result in enumerate(results)
            if not _shard_result_is_persisted(result)
        ]
        if not retry_ids:
            break
        time.sleep(min(2.0, 0.5 * (attempt + 1)))
        with ThreadPoolExecutor(
            max_workers=len(retry_ids), thread_name_prefix="nse-shard-retry"
        ) as pool:
            retried = list(pool.map(run_shard, retry_ids))
        for index, retried_result in zip(retry_ids, retried):
            if _shard_result_is_persisted(retried_result):
                results[index] = retried_result

    shard_payloads: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        names = shard_artifact_names(cycle_minute, index, count)
        payload = dict(result)
        frame = payload.pop("_frame", None)
        shard_payloads.append({
            "worker_id": index,
            "manifest": payload,
            "frame": frame,
            "snapshot_remote_path": names["snapshot_remote"],
            "manifest_remote_path": names["manifest_remote"],
        })

    assembly = assemble_shards(
        cycle_minute=cycle_minute,
        worker_count=count,
        shards=shard_payloads,
        expected_universe=universe_symbols,
        strict_universe=True,
    )
    assembly_manifest = assembly["manifest"]
    canonical_names = canonical_artifact_names(cycle_minute)
    write_result: dict[str, Any] = {}
    status = "INCOMPLETE"
    if assembly["status"] == "VALIDATED":
        try:
            write_result = write_canonical_snapshot(
                store,
                cycle_minute,
                assembly["frame"],
                assembly_manifest,
                worker_root,
            )
            status = "STORED"
        except Exception as exc:
            status = "PERSISTENCE_ERROR"
            assembly_manifest["snapshot_persisted"] = False
            assembly_manifest["status"] = "ASSEMBLY_WRITE_FAILED"
            assembly_manifest["error"] = f"{type(exc).__name__}: {exc}"
    else:
        try:
            write_result["incomplete_manifest"] = write_assembly_manifest_only(
                store, cycle_minute, assembly_manifest, worker_root
            )
        except Exception as exc:
            write_result["incomplete_manifest_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    if write_result:
        assembly_manifest["assembly_write"] = _clean(write_result)

    shard_summaries = [
        {
            key: _clean(value)
            for key, value in {
                **result,
            }.items()
            if key != "_frame"
        }
        for result in results
    ]
    combined = {
        "schema_version": "all-universe-minute-snapshot-v2",
        "mode": "SHARDED_ASSEMBLY",
        "status": status,
        "snapshot_persisted": bool(status == "STORED"),
        "cycle_minute_ist": cycle_minute.isoformat(),
        "worker_count": count,
        "shard_count_expected": count,
        "shard_count_completed": int(
            sum(1 for result in results if _shard_result_is_persisted(result))
        ),
        "shards": shard_summaries,
        "assembly": assembly_manifest,
        "snapshot_drive_path": (
            canonical_names["snapshot_remote"] if status == "STORED" else None
        ),
        "remote_snapshot_path": canonical_names["snapshot_remote"],
        "expected_symbol_count": assembly_manifest.get("expected_symbol_count"),
        "row_count": assembly_manifest.get("row_count"),
        "quoted_symbol_count": assembly_manifest.get("quoted_symbol_count"),
        "missing_quote_count": assembly_manifest.get("missing_quote_count"),
        "complete_quote_symbol_coverage_pct": assembly_manifest.get(
            "complete_quote_symbol_coverage_pct"
        ),
        "coverage_is_informational": True,
    }
    if status != "STORED":
        combined["error"] = assembly_manifest.get("error") or (
            f"Only {combined['shard_count_completed']} of {count} shards completed; "
            "no canonical snapshot was published."
        )
    return _clean(combined)
