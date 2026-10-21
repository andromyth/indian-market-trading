#!/usr/bin/env python3
"""Read-only outcome audit for the autonomous Cloud Run intraday worker.

Reads the worker-owned prediction ledger and 1-minute bar partitions from the
Indian_Market_Research Google Drive tree. It joins each prediction to the first
persisted close at or after decision_timestamp + horizon_min, within a strict
time tolerance. It does not update ledgers, retrain models, change signals, or
place orders.

Example:
  ./py.sh audit_autonomous_intraday_outcomes.py --start-date 2026-10-14 --end-date 2026-10-14
"""
from __future__ import annotations

import argparse
import io
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
FOLDER_MIME = "application/vnd.google-apps.folder"
WORKER_LEDGER_ROOT = "production_data/intraday_predictions"
WORKER_BARS_ROOT = "production_data/intraday_live_worker"


def _root_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            symbol = symbol[:-3]
    return symbol


def _safe_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9_\-^]", "_", _root_symbol(value))


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _retryable_drive_error(exc: Exception) -> bool:
    """Return true for transient Drive/API failures, not for missing objects."""
    status = getattr(getattr(exc, "resp", None), "status", None)
    if status in {408, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in (
        "internal error", "backend error", "temporarily unavailable",
        "rate limit", "timed out", "timeout",
    ))


def _upload_drive_with_retry(
    store: Any,
    local_path: Path,
    remote_path: str,
    *,
    mime_type: str,
    attempts: int = 5,
) -> dict[str, Any]:
    """Retry transient Drive uploads without treating timeouts as missing reports."""
    import random
    import time as time_module

    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            return store.upload_or_update(
                local_path, remote_path, mime_type=mime_type
            )
        except Exception as exc:
            if not _retryable_drive_error(exc) or attempt + 1 >= attempts:
                raise
            delay = min(8.0, 0.4 * (2 ** attempt) + random.random() * 0.25)
            status = getattr(getattr(exc, "resp", None), "status", None)
            print(
                "DRIVE_UPLOAD_RETRY: "
                f"path={remote_path}, status={status if status is not None else 'network'}, "
                f"attempt={attempt + 1}/{attempts}, "
                f"error={type(exc).__name__}: {exc}, sleep_seconds={delay:.2f}",
                flush=True,
            )
            time_module.sleep(delay)
    raise RuntimeError(f"Drive upload retry loop ended unexpectedly: {remote_path}")


def _find_child_retry(store: Any, parent: str, name: str, mime_type: str | None = None,
                      attempts: int = 5) -> dict[str, Any] | None:
    """Retry only transient Drive listing failures; a confirmed miss remains a miss."""
    import random
    import time as time_module

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return store._find_child(parent, name, mime_type)
        except Exception as exc:
            last_error = exc
            if not _retryable_drive_error(exc) or attempt + 1 >= attempts:
                raise
            time_module.sleep(min(8.0, (0.4 * (2 ** attempt)) + random.random() * 0.25))
    if last_error:
        raise last_error
    return None


def _resolve_drive_id(store: Any, remote_path: str, *, folder: bool = False) -> str | None:
    parts = [part for part in remote_path.replace("\\", "/").split("/") if part]
    if not parts:
        return None
    parent = store.root_id
    for part in parts[:-1]:
        hit = _find_child_retry(store, parent, part, FOLDER_MIME)
        if not hit:
            return None
        parent = hit["id"]
    hit = _find_child_retry(store, parent, parts[-1], FOLDER_MIME if folder else None)
    return str(hit["id"]) if hit else None


def _download_drive_bytes(store: Any, remote_path: str) -> bytes | None:
    """Resolve and download a Drive file with bounded retries for transient failures."""
    import random
    import time as time_module
    from googleapiclient.http import MediaIoBaseDownload

    file_id = _resolve_drive_id(store, remote_path)
    if not file_id:
        return None
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            buffer = io.BytesIO()
            request = store.service.files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buffer.getvalue()
        except Exception as exc:
            last_error = exc
            if not _retryable_drive_error(exc) or attempt >= 4:
                raise
            time_module.sleep(min(8.0, 0.4 * (2 ** attempt) + random.random() * 0.25))
    if last_error:
        raise last_error
    return None


def _read_ledger(payload: bytes) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    for line in payload.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
            else:
                malformed += 1
        except json.JSONDecodeError:
            malformed += 1
    return records, malformed


def _normalise_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["close"], index=pd.DatetimeIndex([], tz="UTC"))
    x = frame.copy()
    close_column = next((c for c in x.columns if str(c).strip().lower() == "close"), None)
    if close_column is None:
        close_column = next((c for c in x.columns if str(c).strip().lower() in {"c", "ltp"}), None)
    if close_column is None:
        return pd.DataFrame(columns=["close"], index=pd.DatetimeIndex([], tz="UTC"))

    timestamp_column = next(
        (c for c in x.columns if str(c).strip().lower() in {"timestamp", "datetime", "_time"}),
        None,
    )
    raw_index = x.pop(timestamp_column) if timestamp_column is not None else x.index
    parsed = pd.DatetimeIndex(pd.to_datetime(raw_index, errors="coerce"))
    if parsed.tz is None:
        parsed = parsed.tz_localize(IST)
    parsed = parsed.tz_convert("UTC")
    x.index = parsed
    x.index.name = "timestamp_utc"
    close = pd.to_numeric(x[close_column], errors="coerce")
    out = pd.DataFrame({"close": close.to_numpy()}, index=x.index)
    out = out.loc[~out.index.isna()].dropna(subset=["close"]).sort_index()
    return out.loc[~out.index.duplicated(keep="last")]


def evaluate_record(
    record: dict[str, Any],
    bars: pd.DataFrame,
    tolerance_seconds: int = 60,
) -> dict[str, Any]:
    """Validate one matured forecast against the first non-pretarget close."""
    symbol = _root_symbol(record.get("symbol"))
    out: dict[str, Any] = {
        "symbol": symbol,
        "horizon_min": None,
        "decision_timestamp_utc": None,
        "target_timestamp_utc": None,
        "signal": str(record.get("signal") or "HOLD").upper(),
        "probability_up": _number(record.get("probability_up")),
        "reference_close": _number(record.get("reference_close")),
        "predicted_price": _number(record.get("predicted_price")),
        "actual_price": np.nan,
        "actual_return": np.nan,
        "actual_direction": None,
        "direction_match": None,
        "brier_score": np.nan,
        "predicted_price_abs_error": np.nan,
        "baseline_price_abs_error": np.nan,
        "signed_signal_return_before_costs": np.nan,
        "actual_timestamp_utc": None,
        "target_alignment_seconds": np.nan,
        "validation_status": "INVALID_PREDICTION",
        "validation_reason": None,
    }

    try:
        parsed_decision = pd.to_datetime(
            record.get("decision_timestamp"), errors="coerce", utc=True
        )
        if pd.isna(parsed_decision):
            out["validation_reason"] = "missing_or_invalid_decision_timestamp"
            return out
        decision = pd.Timestamp(parsed_decision)
        horizon = int(record.get("horizon_min"))
        if horizon <= 0 or horizon > 240:
            out["validation_reason"] = "horizon_min_outside_supported_range"
            return out
    except Exception as exc:
        out["validation_reason"] = f"invalid_decision_timestamp_or_horizon:{type(exc).__name__}"
        return out

    target = decision + pd.Timedelta(minutes=horizon)
    out["horizon_min"] = horizon
    out["decision_timestamp_utc"] = decision.isoformat()
    out["target_timestamp_utc"] = target.isoformat()

    target_ist = target.tz_convert(IST)
    if target_ist.time() > time(15, 30) or target_ist.time() < time(9, 15):
        out["validation_status"] = "TARGET_OUTSIDE_CASH_SESSION"
        out["validation_reason"] = "nominal horizon target falls outside regular NSE cash-session hours"
        return out

    if bars is None or bars.empty:
        out["validation_status"] = "NO_BAR_PARTITION"
        out["validation_reason"] = "no persisted 1-minute bar partition for this symbol/date"
        return out

    reference = out["reference_close"]
    if not np.isfinite(reference) or reference <= 0:
        prior = bars.loc[bars.index <= decision, "close"]
        if prior.empty:
            out["validation_status"] = "NO_REFERENCE_CLOSE"
            out["validation_reason"] = "no persisted close at or before decision timestamp"
            return out
        reference = float(prior.iloc[-1])
        out["reference_close"] = reference

    seconds_after_target = (bars.index - target).total_seconds()
    eligible = (seconds_after_target >= 0) & (seconds_after_target <= max(0, int(tolerance_seconds)))
    if not eligible.any():
        out["validation_status"] = "WAITING_FOR_TARGET_BAR"
        out["validation_reason"] = "no persisted close at/after target within configured tolerance"
        return out

    position = int(np.argmin(np.where(eligible, seconds_after_target, np.inf)))
    actual_timestamp = bars.index[position]
    actual_price = float(bars["close"].iloc[position])
    actual_return = actual_price / reference - 1.0
    signal = out["signal"]
    direction_match: bool | None = None
    signed_return = np.nan
    if signal == "BUY":
        direction_match = bool(actual_return > 0)
        signed_return = actual_return
    elif signal == "SELL":
        direction_match = bool(actual_return < 0)
        signed_return = -actual_return

    p_up = out["probability_up"]
    actual_label = 1.0 if actual_return > 0 else 0.0
    predicted_price = out["predicted_price"]

    out.update({
        "actual_price": actual_price,
        "actual_return": actual_return,
        "actual_direction": "UP" if actual_return > 0 else "DOWN_FLAT",
        "direction_match": direction_match,
        "brier_score": (p_up - actual_label) ** 2 if np.isfinite(p_up) and 0 <= p_up <= 1 else np.nan,
        "predicted_price_abs_error": abs(predicted_price - actual_price) if np.isfinite(predicted_price) else np.nan,
        "baseline_price_abs_error": abs(reference - actual_price),
        "signed_signal_return_before_costs": signed_return,
        "actual_timestamp_utc": actual_timestamp.isoformat(),
        "target_alignment_seconds": float((actual_timestamp - target).total_seconds()),
        "validation_status": "VALIDATED",
        "validation_reason": None,
    })
    return out


def _metrics(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {
            "predictions_seen": 0, "validated": 0, "action_predictions": 0,
            "action_direction_accuracy": None, "buy_precision": None, "sell_precision": None,
            "mean_brier_score": None, "model_price_mae": None, "reference_price_baseline_mae": None,
            "model_beats_reference_price_baseline": None, "mean_signed_signal_return_before_costs": None,
        }

    valid = rows.loc[rows["validation_status"].eq("VALIDATED")].copy()
    action = valid.loc[valid["signal"].isin(["BUY", "SELL"])]
    buy = valid.loc[valid["signal"].eq("BUY")]
    sell = valid.loc[valid["signal"].eq("SELL")]

    def mean_numeric(frame: pd.DataFrame, column: str) -> float | None:
        values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    model_mae = mean_numeric(valid, "predicted_price_abs_error")
    baseline_mae = mean_numeric(valid, "baseline_price_abs_error")
    return {
        "predictions_seen": int(len(rows)),
        "validated": int(len(valid)),
        "waiting_for_target_bar": int(rows["validation_status"].eq("WAITING_FOR_TARGET_BAR").sum()),
        "outside_cash_session_targets": int(rows["validation_status"].eq("TARGET_OUTSIDE_CASH_SESSION").sum()),
        "other_unvalidated": int((~rows["validation_status"].isin({"VALIDATED", "WAITING_FOR_TARGET_BAR", "TARGET_OUTSIDE_CASH_SESSION"})).sum()),
        "action_predictions": int(len(action)),
        "buy_predictions": int(len(buy)),
        "sell_predictions": int(len(sell)),
        "action_direction_accuracy": mean_numeric(action, "direction_match"),
        "buy_precision": mean_numeric(buy, "direction_match"),
        "sell_precision": mean_numeric(sell, "direction_match"),
        "mean_brier_score": mean_numeric(valid, "brier_score"),
        "model_price_mae": model_mae,
        "reference_price_baseline_mae": baseline_mae,
        "model_beats_reference_price_baseline": (
            bool(model_mae < baseline_mae) if model_mae is not None and baseline_mae is not None else None
        ),
        "mean_signed_signal_return_before_costs": mean_numeric(action, "signed_signal_return_before_costs"),
        "warning": "Descriptive signal metrics only: minute-by-minute predictions overlap; the signed return is before all transaction costs and is not a portfolio backtest.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", help="First worker-ledger date in YYYY-MM-DD; default is six days ago (IST).")
    parser.add_argument("--end-date", help="Last worker-ledger date in YYYY-MM-DD; default is today (IST).")
    parser.add_argument("--segment", default="service", choices=["service", "morning", "afternoon", "once"])
    parser.add_argument("--tolerance-seconds", type=int, default=60)
    parser.add_argument("--out-dir", default="diagnostics/autonomous_intraday_outcomes")
    parser.add_argument("--sync-drive", action="store_true", help="Upload the generated outcome CSV and JSON report to Google Drive.")
    args = parser.parse_args()

    today = datetime.now(IST).date()
    try:
        start_date = date.fromisoformat(args.start_date) if args.start_date else today - timedelta(days=6)
        end_date = date.fromisoformat(args.end_date) if args.end_date else today
    except ValueError:
        parser.error("dates must use YYYY-MM-DD")
    if end_date < start_date:
        parser.error("--end-date must be on or after --start-date")
    if args.tolerance_seconds < 0:
        parser.error("--tolerance-seconds cannot be negative")

    from persistent_store import GoogleDriveStore

    store = GoogleDriveStore()
    report_rows: list[dict[str, Any]] = []
    load_errors: list[str] = []
    malformed_rows = 0
    ledger_days_found = 0
    ledger_rows_loaded = 0
    bars_cache: dict[tuple[str, str], pd.DataFrame] = {}
    daily_records: list[tuple[str, dict[str, Any]]] = []

    current = start_date
    while current <= end_date:
        day_path = f"{current:%Y/%m/%d}"
        ledger_path = f"{WORKER_LEDGER_ROOT}/{day_path}/predictions_{args.segment}.jsonl"
        payload = _download_drive_bytes(store, ledger_path)
        if payload is None:
            current += timedelta(days=1)
            continue
        ledger_days_found += 1
        records, malformed = _read_ledger(payload)
        malformed_rows += malformed
        ledger_rows_loaded += len(records)
        for record in records:
            daily_records.append((current.isoformat(), record))
        current += timedelta(days=1)

    if not daily_records:
        print(json.dumps({
            "status": "NO_WORKER_PREDICTIONS_FOUND",
            "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "ledger_days_found": ledger_days_found,
            "ledger_rows_loaded": ledger_rows_loaded,
            "note": "No files found in production_data/intraday_predictions/YYYY/MM/DD/predictions_<segment>.jsonl. Verify Drive configuration and segment.",
        }, indent=2))
        return 2

    for ledger_date, record in daily_records:
        symbol = _root_symbol(record.get("symbol"))
        if not symbol:
            report_rows.append({
                "symbol": None,
                "validation_status": "INVALID_PREDICTION",
                "validation_reason": "prediction row has no symbol",
                "ledger_date": ledger_date,
            })
            continue
        cache_key = (ledger_date, symbol)
        if cache_key not in bars_cache:
            day_path = ledger_date.replace("-", "/")
            safe_symbol = _safe_symbol(symbol)
            bar_path = f"{WORKER_BARS_ROOT}/{day_path}/{safe_symbol}.parquet"
            bar_payload = _download_drive_bytes(store, bar_path)
            if bar_payload is None:
                bars_cache[cache_key] = pd.DataFrame()
            else:
                try:
                    bars_cache[cache_key] = _normalise_bars(pd.read_parquet(io.BytesIO(bar_payload)))
                except Exception as exc:
                    load_errors.append(f"{ledger_date}/{symbol} bars: {type(exc).__name__}: {exc}")
                    bars_cache[cache_key] = pd.DataFrame()

        evaluated = evaluate_record(record, bars_cache[cache_key], args.tolerance_seconds)
        evaluated["ledger_date"] = ledger_date
        evaluated["model_version"] = record.get("model_version")
        evaluated["worker_observed_at_utc"] = record.get("worker_observed_at_utc")
        report_rows.append(evaluated)

    details = pd.DataFrame(report_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{start_date.isoformat()}_to_{end_date.isoformat()}_{args.segment}"
    csv_path = out_dir / f"outcomes_{tag}.csv"
    json_path = out_dir / f"outcomes_{tag}.json"
    details.to_csv(csv_path, index=False)

    if details.empty:
        overall = _metrics(details)
    else:
        overall = _metrics(details)
    by_horizon = {}
    if not details.empty and "horizon_min" in details:
        for horizon, group in details.loc[details["horizon_min"].notna()].groupby("horizon_min"):
            by_horizon[str(int(horizon))] = _metrics(group)

    validated_count = int(details["validation_status"].eq("VALIDATED").sum()) if not details.empty else 0
    status = "PASS_WITH_MATURED_OUTCOMES" if validated_count else "NO_MATURED_OUTCOMES"
    report = {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_range": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "segment": args.segment,
        "source": "Google Drive autonomous worker prediction ledger + worker 1-minute bar partitions",
        "read_only": True,
        "model_weights_changed": False,
        "orders_placed": False,
        "ledger_days_found": ledger_days_found,
        "ledger_rows_loaded": ledger_rows_loaded,
        "malformed_ledger_rows": malformed_rows,
        "outcome_rows": int(len(details)),
        "validated_rows": validated_count,
        "bar_partition_cache_entries": len(bars_cache),
        "bar_load_errors": load_errors,
        "overall_metrics": overall,
        "metrics_by_horizon_min": by_horizon,
        "details_csv": str(csv_path),
        "details_json": str(json_path),
        "interpretation": [
            "Only outcomes whose target candle exists at or after the nominal horizon are scored.",
            "No target bar yet is reported as waiting, not scored as wrong.",
            "Targets outside the regular NSE cash session are excluded from accuracy metrics.",
            "The reference-price baseline predicts no price change; model price MAE must be lower to beat it.",
            "BUY/SELL accuracy and signed returns use only matured observations; signed returns ignore all trading costs.",
            "Overlapping minute-by-minute forecasts are not independent trades and must not be interpreted as portfolio returns.",
        ],
    }
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if args.sync_drive:
        remote_root = (
            f"{WORKER_LEDGER_ROOT}/outcomes_audit/"
            f"{end_date:%Y/%m/%d}"
        )
        remote_csv = f"{remote_root}/{csv_path.name}"
        remote_json = f"{remote_root}/{json_path.name}"
        csv_upload = _upload_drive_with_retry(
            store, csv_path, remote_csv, mime_type="text/csv"
        )
        report["google_drive_sync"] = {
            "status": "CSV_UPLOADED_JSON_PENDING",
            "csv_path": remote_csv,
            "json_path": remote_json,
            "csv_upload_result": str(csv_upload),
        }
        json_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        json_upload = _upload_drive_with_retry(
            store, json_path, remote_json, mime_type="application/json"
        )
        report["google_drive_sync"]["status"] = "OK"
        report["google_drive_sync"]["json_upload_result"] = str(json_upload)
        json_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        store.upload_or_update(
            json_path, remote_json, mime_type="application/json"
        )
        print("OUTCOME_REPORTS_UPLOADED_TO_DRIVE:", remote_csv, remote_json)
    print(json.dumps(report, indent=2, default=str))
    return 0 if validated_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
