"""Audits and realised-outcome helpers for the isolated autonomous Delivery challenger.

This module does not change the locked production model contract. It measures
what was actually available to the challenger and derives labels only from
consecutive official NSE point-in-time snapshots.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd

PIT_NAME_RE = re.compile(r"^DELIVERY_PIT_FEATURES_(20\d{2}-\d{2}-\d{2})\.parquet$")
GLOBAL_MAX_AGE_DAYS = 5.0


def parse_pit_date(name: str) -> date | None:
    match = PIT_NAME_RE.fullmatch(str(name or "").strip())
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def normalize_symbol(value: object) -> str:
    symbol = str(value or "").strip().upper()
    for suffix in (".NS", ".BO", "-EQ", ".EQ"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)]
    return symbol


def next_nse_session(day: date) -> date:
    """Next weekday excluding the repository's known official NSE holidays."""
    try:
        from production_data_freshness_gate import HOLIDAYS
    except Exception:
        HOLIDAYS = set()
    candidate = day + timedelta(days=1)
    while candidate.weekday() >= 5 or candidate in HOLIDAYS:
        candidate += timedelta(days=1)
    return candidate


def _snapshot_close_frame(day: date, snapshot: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if snapshot is None or snapshot.empty:
        return pd.DataFrame(columns=["symbol", "close"]), 0
    if not {"symbol", "close"}.issubset(snapshot.columns):
        raise ValueError(f"{day}: PIT snapshot must contain symbol and close")
    rows = snapshot[["symbol", "close"]].copy()
    rows["symbol"] = rows["symbol"].map(normalize_symbol)
    rows["close"] = pd.to_numeric(rows["close"], errors="coerce")
    duplicate_rows = int(rows.duplicated("symbol", keep=False).sum())
    rows = rows.dropna(subset=["symbol", "close"])
    rows = rows[rows["symbol"].ne("") & rows["close"].gt(0)]
    rows = rows.drop_duplicates("symbol", keep="last")
    return rows, duplicate_rows


def derive_realised_returns(
    snapshots: Iterable[tuple[date, pd.DataFrame]],
) -> tuple[pd.DataFrame, list[dict]]:
    """Build close-to-close labels only across consecutive NSE sessions.

    The date is the reference/prediction date D; y_return is official
    close(D+1)/close(D)-1, available only when the next PIT snapshot exists.
    Missing-session gaps are skipped, not mislabeled as next-session returns.
    """
    ordered = sorted(snapshots, key=lambda item: item[0])
    normalized = []
    for day, frame in ordered:
        close_frame, _ = _snapshot_close_frame(day, frame)
        normalized.append((day, close_frame))

    out = []
    gaps = []
    for (day0, left), (day1, right) in zip(normalized, normalized[1:]):
        expected = next_nse_session(day0)
        if day1 != expected:
            gaps.append({
                "from_date": day0.isoformat(),
                "to_date": day1.isoformat(),
                "expected_next_session": expected.isoformat(),
                "status": "SKIPPED_SESSION_GAP",
            })
            continue
        joined = left.merge(right, on="symbol", how="inner", suffixes=("_from", "_to"))
        if joined.empty:
            continue
        joined["y_return"] = joined["close_to"] / joined["close_from"] - 1.0
        joined["date"] = day0
        joined = joined.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_return"])
        out.append(joined[["symbol", "date", "y_return"]])

    if not out:
        return pd.DataFrame(columns=["symbol", "date", "y_return"]), gaps
    result = pd.concat(out, ignore_index=True)
    result["symbol"] = result["symbol"].map(normalize_symbol)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["y_return"] = pd.to_numeric(result["y_return"], errors="coerce")
    result = result.dropna(subset=["symbol", "date", "y_return"])
    result = result.drop_duplicates(["symbol", "date"], keep="last")
    result = result.sort_values(["date", "symbol"]).reset_index(drop=True)
    return result, gaps


def merge_realised_returns(old: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merge official labels by (symbol, date), retaining older history."""
    required = ["symbol", "date", "y_return"]

    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame(columns=required)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Realised return table missing columns: {missing}")
        out = frame[required].copy()
        out["symbol"] = out["symbol"].map(normalize_symbol)
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out["y_return"] = pd.to_numeric(out["y_return"], errors="coerce")
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        return out[out["symbol"].ne("")].drop_duplicates(["symbol", "date"], keep="last")

    before = clean(old)
    updates = clean(fresh)
    keys_before = set(zip(before["symbol"], before["date"]))
    new_keys = set(zip(updates["symbol"], updates["date"]))
    overlap = len(keys_before & new_keys)
    merged = pd.concat([before, updates], ignore_index=True)
    merged = merged.drop_duplicates(["symbol", "date"], keep="last").sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)
    return merged, {
        "existing_rows": int(len(before)),
        "derived_rows": int(len(updates)),
        "new_keys": int(len(new_keys - keys_before)),
        "refreshed_keys": int(overlap),
        "rows_after_merge": int(len(merged)),
        "latest_realised_date": str(merged["date"].max().date()) if len(merged) else None,
    }


def _coverage(frame: pd.DataFrame, cols: list[str]) -> dict:
    present = [column for column in cols if column in frame.columns]
    if not present or frame.empty:
        return {"feature_count": len(present), "row_coverage_pct": 0.0, "feature_coverage_pct": {}}
    numeric = frame[present].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return {
        "feature_count": len(present),
        "row_coverage_pct": round(float(numeric.notna().any(axis=1).mean() * 100), 3),
        "mean_feature_non_null_pct": round(float(numeric.notna().mean().mean() * 100), 3),
        "feature_coverage_pct": {
            column: round(float(numeric[column].notna().mean() * 100), 3)
            for column in present
        },
    }


def build_input_audit(
    pit: pd.DataFrame,
    panel: pd.DataFrame,
    global_cues: pd.DataFrame,
    raw_news: pd.DataFrame,
    fno_daily: pd.DataFrame,
    *,
    expected_date: date | None,
    observed_date: date | None,
    realised_update: dict | None = None,
    source_status: dict | None = None,
    now: pd.Timestamp | None = None,
) -> dict:
    """Create JSON-safe stockwise/source-integrity diagnostics; never imputes data."""
    from multifactor_data_layer_v2 import FEATURES_59

    now_utc = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")
    source_status = source_status or {}
    hard_failures: list[str] = []
    warnings: list[str] = []

    symbols = pit["symbol"].map(normalize_symbol) if "symbol" in pit.columns else pd.Series(dtype=str)
    dupes = int(symbols.duplicated(keep=False).sum()) if len(symbols) else 0
    missing_features = [feature for feature in FEATURES_59 if feature not in pit.columns]
    if pit.empty or "symbol" not in pit.columns or symbols.nunique() == 0:
        hard_failures.append("PIT_EMPTY_OR_NO_SYMBOLS")
    if missing_features:
        hard_failures.append("PIT_MISSING_LOCKED_59_FEATURES")
    if dupes:
        hard_failures.append("PIT_DUPLICATE_SYMBOL_ROWS")
    if expected_date is not None and observed_date != expected_date:
        hard_failures.append("PIT_REFERENCE_DATE_MISMATCH")
    if not {"close", "volume_ratio", "volume_z20"}.issubset(pit.columns):
        warnings.append("PRICE_VOLUME_FIELDS_INCOMPLETE")

    feature_coverage = _coverage(pit, list(FEATURES_59))
    groups = {
        "price_volume_delivery": [
            c for c in FEATURES_59
            if c.startswith(("ret_", "rsi", "atr", "ma", "volume", "delivery", "range_", "body_", "up_bar"))
        ],
        "nse_market_context": [c for c in FEATURES_59 if c.startswith(("market_", "relative_ret_"))],
        "news": [c for c in FEATURES_59 if c.startswith("news_")],
        "corporate_events": [c for c in FEATURES_59 if c.startswith("event_")],
        "fno": [c for c in FEATURES_59 if c.startswith("fno_")],
        "fundamentals": [c for c in FEATURES_59 if c.startswith(("fundamental_", "fund_"))],
        "global_market_fx": [
            c for c in panel.columns if c.startswith("g_") and not c.endswith("_age_h")
        ] if not panel.empty else [],
        "scored_news": [
            c for c in panel.columns
            if c.startswith(("news_", "mkt_news_")) and c != "news_available"
        ] if not panel.empty else [],
    }
    group_coverage = {
        name: _coverage(
            pit if name in {
                "price_volume_delivery", "nse_market_context", "news",
                "corporate_events", "fno", "fundamentals"
            } else panel,
            cols,
        )
        for name, cols in groups.items()
    }

    lowest_coverage = []
    if not pit.empty and "symbol" in pit.columns:
        numeric_features = [c for c in FEATURES_59 if c in pit.columns]
        if numeric_features:
            matrix = pit[numeric_features].apply(pd.to_numeric, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            row_cov = matrix.notna().mean(axis=1)
            lowest_coverage = sorted(
                [
                    {"symbol": str(symbols.iloc[idx]), "feature_non_null_pct": round(float(row_cov.iloc[idx] * 100), 3)}
                    for idx in range(len(pit))
                ],
                key=lambda item: item["feature_non_null_pct"],
            )[:20]

    if "volume_ratio" in pit.columns:
        vr = pd.to_numeric(pit["volume_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid_volume = int(vr.notna().sum())
        invalid_volume = int((vr.notna() & vr.le(0)).sum())
    else:
        valid_volume = invalid_volume = 0
    volume_audit = {
        "volume_column_present": "volume" in pit.columns,
        "volume_ratio_valid_rows": valid_volume,
        "volume_ratio_coverage_pct": round(valid_volume / len(pit) * 100, 3) if len(pit) else 0.0,
        "non_positive_volume_ratio_rows": invalid_volume,
        "minute_volume_is_not_a_daily_volume_proxy": True,
    }
    if len(pit) and volume_audit["volume_ratio_coverage_pct"] < 60:
        warnings.append("LOW_DAILY_VOLUME_FEATURE_COVERAGE")

    def _row_value(row: pd.Series | None, column: str):
        if row is None or column not in row.index or pd.isna(row[column]):
            return None
        value = row[column]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value)

    def _any_known(row: pd.Series | None, columns: list[str]) -> bool:
        return bool(row is not None and any(
            column in row.index and pd.notna(row[column]) for column in columns
        ))

    panel_by_symbol = {}
    panel_symbol_count = 0
    if not panel.empty and "symbol" in panel.columns:
        px = panel.copy()
        px["__symbol"] = px["symbol"].map(normalize_symbol)
        px = px.drop_duplicates("__symbol", keep="last")
        panel_by_symbol = {str(row["__symbol"]): row for _, row in px.iterrows()}
        panel_symbol_count = len(panel_by_symbol)

    stockwise_input_readiness = []
    if not pit.empty and "symbol" in pit.columns:
        pit_x = pit.copy()
        pit_x["__symbol"] = pit_x["symbol"].map(normalize_symbol)
        numeric_features = [column for column in FEATURES_59 if column in pit_x.columns]
        feature_matrix = pit_x[numeric_features].apply(pd.to_numeric, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ) if numeric_features else pd.DataFrame(index=pit_x.index)
        coverage_by_index = feature_matrix.notna().mean(axis=1) if numeric_features else pd.Series(0.0, index=pit_x.index)
        global_columns = groups["global_market_fx"]
        scored_news_columns = groups["scored_news"]
        delivery_columns = [
            "delivery_pct", "delivery_pct_change_1", "delivery_pct_ratio20",
            "delivery_pct_z20", "delivery_qty_ratio20", "delivery_qty_z20",
        ]
        fno_columns = [column for column in pit_x.columns if column.startswith("fno_")]
        fundamental_columns = [column for column in pit_x.columns if column.startswith(("fundamental_", "fund_"))]
        for idx, row in pit_x.iterrows():
            symbol = str(row["__symbol"])
            live_row = panel_by_symbol.get(symbol)
            global_present = [
                column for column in global_columns
                if live_row is not None and column in live_row.index and pd.notna(live_row[column])
            ]
            news_present = [
                column for column in scored_news_columns
                if live_row is not None and column in live_row.index and pd.notna(live_row[column])
            ]
            stockwise_input_readiness.append({
                "symbol": symbol,
                "model_feature_non_null_pct": round(float(coverage_by_index.loc[idx] * 100), 3),
                "daily_volume_ratio_known": bool(pd.notna(row.get("volume_ratio", np.nan))),
                "daily_volume_zscore_known": bool(pd.notna(row.get("volume_z20", np.nan))),
                "delivery_features_known": _any_known(row, delivery_columns),
                "news_feature_known": _any_known(live_row, ["news_cnt_1", "news_sent_ewm", "news_available"]),
                "news_available_flag": _row_value(live_row, "news_available"),
                "scored_news_features_present": len(news_present),
                "scored_news_features_expected": len(scored_news_columns),
                "event_feature_known": _any_known(row, ["event_available", "event_high_impact"]),
                "event_available_flag": _row_value(row, "event_available"),
                "fno_features_known": _any_known(live_row, fno_columns + ["fno_beta_observed"]),
                "fundamental_features_known": _any_known(row, fundamental_columns + ["fundamental_age_days"]),
                "global_features_present": len(global_present),
                "global_features_expected": len(global_columns),
                "usd_inr_feature_available": bool(
                    live_row is not None and "g_inr_z" in live_row.index and pd.notna(live_row["g_inr_z"])
                ),
            })

    stockwise_source_coverage = {
        "symbols_in_pit": len(stockwise_input_readiness),
        "symbols_with_usdinr_feature": sum(bool(row["usd_inr_feature_available"]) for row in stockwise_input_readiness),
        "symbols_with_daily_volume_ratio": sum(bool(row["daily_volume_ratio_known"]) for row in stockwise_input_readiness),
        "symbols_with_delivery_features": sum(bool(row["delivery_features_known"]) for row in stockwise_input_readiness),
        "symbols_with_known_news_features": sum(bool(row["news_feature_known"]) for row in stockwise_input_readiness),
        "symbols_with_event_features": sum(bool(row["event_feature_known"]) for row in stockwise_input_readiness),
        "symbols_with_fno_features": sum(bool(row["fno_features_known"]) for row in stockwise_input_readiness),
        "symbols_with_fundamental_features": sum(bool(row["fundamental_features_known"]) for row in stockwise_input_readiness),
        "symbols_with_any_global_features": sum(row["global_features_present"] > 0 for row in stockwise_input_readiness),
        "symbols_with_scored_news_features": sum(row["scored_news_features_present"] > 0 for row in stockwise_input_readiness),
        "panel_symbols": panel_symbol_count,
    }

    global_duplicate_rows = (
        int(global_cues.duplicated(["key", "date"], keep=False).sum())
        if {"key", "date"}.issubset(global_cues.columns) else 0
    )
    global_audit = {
        "rows": int(len(global_cues)),
        "series": sorted(global_cues["key"].dropna().astype(str).str.lower().unique().tolist()) if "key" in global_cues.columns else [],
        "duplicate_key_date_rows": global_duplicate_rows,
        "latest_available_by_series": {},
        "integrated_feature_coverage": _coverage(panel, groups["global_market_fx"]),
    }
    if global_duplicate_rows:
        warnings.append("DUPLICATE_GLOBAL_SERIES_DATE_ROWS")
    if not global_cues.empty and {"key", "available_at"}.issubset(global_cues.columns):
        gx = global_cues.copy()
        gx["available_at"] = pd.to_datetime(gx["available_at"], errors="coerce", utc=True)
        gx = gx[gx["available_at"].notna() & gx["available_at"].le(now_utc)]
        for key in global_audit["series"]:
            rows = gx[gx["key"].astype(str).str.lower().eq(key)]
            if rows.empty:
                global_audit["latest_available_by_series"][key] = {"status": "NOT_YET_AVAILABLE"}
            else:
                latest = rows.sort_values("available_at").iloc[-1]
                age_h = max(0.0, (now_utc - latest["available_at"]).total_seconds() / 3600)
                row_date = pd.to_datetime(latest.get("date"), errors="coerce")
                global_audit["latest_available_by_series"][key] = {
                    "date": row_date.date().isoformat() if pd.notna(row_date) else None,
                    "available_at": latest["available_at"].isoformat(),
                    "age_hours": round(age_h, 2),
                }
    usdinr = global_audit["latest_available_by_series"].get("usdinr")
    if not usdinr or usdinr.get("status") == "NOT_YET_AVAILABLE":
        warnings.append("USDINR_SERIES_NOT_AVAILABLE_AS_OF_AUDIT")
    elif usdinr.get("age_hours", 0) > GLOBAL_MAX_AGE_DAYS * 24:
        warnings.append("USDINR_SERIES_STALE")
    if global_cues.empty:
        warnings.append("GLOBAL_CUES_ARCHIVE_MISSING_OR_EMPTY")
    elif "usdinr" not in global_audit["series"]:
        warnings.append("USDINR_SERIES_MISSING_FROM_ARCHIVE")

    news_duplicate_rows = (
        int(raw_news.duplicated(["symbol", "published_at", "text"], keep=False).sum())
        if {"symbol", "published_at", "text"}.issubset(raw_news.columns) else 0
    )
    news_audit = {
        "rows": int(len(raw_news)),
        "symbols": int(raw_news["symbol"].map(normalize_symbol).nunique()) if "symbol" in raw_news.columns else 0,
        "market_wide_rows": int(raw_news["symbol"].astype(str).str.upper().eq("MARKET").sum()) if "symbol" in raw_news.columns else 0,
        "duplicate_headline_rows": news_duplicate_rows,
        "latest_first_seen_at": None,
        "rows_first_seen_last_24h": 0,
        "integrated_feature_coverage": _coverage(panel, groups["scored_news"]),
    }
    if news_duplicate_rows:
        warnings.append("DUPLICATE_NEWS_HEADLINE_ROWS")
    if not raw_news.empty and "published_at" in raw_news.columns:
        nts = pd.to_datetime(raw_news["published_at"], errors="coerce", utc=True).dropna()
        if len(nts):
            latest_seen = nts.max()
            news_audit["latest_first_seen_at"] = latest_seen.isoformat()
            news_audit["rows_first_seen_last_24h"] = int((nts >= now_utc - pd.Timedelta(hours=24)).sum())
            if (now_utc - latest_seen).total_seconds() > 72 * 3600:
                warnings.append("RSS_ARCHIVE_HAS_NO_HEADLINE_SEEN_WITHIN_72H")
        else:
            warnings.append("RSS_ARCHIVE_TIMESTAMP_INVALID")
    else:
        warnings.append("RSS_ARCHIVE_MISSING_OR_EMPTY")

    fno_duplicate_rows = (
        int(fno_daily.duplicated(["symbol", "date"], keep=False).sum())
        if {"symbol", "date"}.issubset(fno_daily.columns) else 0
    )
    fno_audit = {
        "rows": int(len(fno_daily)),
        "symbols": int(fno_daily["symbol"].map(normalize_symbol).nunique()) if "symbol" in fno_daily.columns else 0,
        "duplicate_symbol_date_rows": fno_duplicate_rows,
        "latest_trade_date": None,
        "integrated_feature_coverage": _coverage(panel, groups["fno"]),
    }
    if fno_duplicate_rows:
        warnings.append("DUPLICATE_FNO_SYMBOL_DATE_ROWS")
    if not fno_daily.empty and "date" in fno_daily.columns:
        fdates = pd.to_datetime(fno_daily["date"], errors="coerce").dropna()
        if len(fdates):
            latest_fno = fdates.max().date()
            fno_audit["latest_trade_date"] = latest_fno.isoformat()
            if expected_date and (expected_date - latest_fno).days > 5:
                warnings.append("FNO_SOURCE_OLDER_THAN_5_CALENDAR_DAYS")
    else:
        warnings.append("FNO_ARCHIVE_MISSING_OR_EMPTY")

    if not pit.empty and "production_data_quality_gate" in pit.columns:
        gate = pit["production_data_quality_gate"].astype(str).str.strip().str.lower().isin(
            {"true", "1", "yes", "pass"}
        )
        gate_pct = round(float(gate.mean() * 100), 3)
    else:
        gate_pct = None
        warnings.append("PIT_PRODUCTION_QUALITY_GATE_MISSING")

    global_refresh = source_status.get("global_refresh", {}) if isinstance(source_status, dict) else {}
    failed_series = global_refresh.get("failed_series", {}) if isinstance(global_refresh, dict) else {}
    if isinstance(failed_series, dict) and len(failed_series) > max(1, len(global_audit["series"]) // 2):
        warnings.append("MORE_THAN_HALF_GLOBAL_SERIES_FAILED_CURRENT_REFRESH")

    news_refresh = source_status.get("news_refresh", {}) if isinstance(source_status, dict) else {}
    dead_feeds = news_refresh.get("dead_feeds", []) if isinstance(news_refresh, dict) else []
    try:
        from selfdriving.sd_news_collect import FEEDS
        feed_count = len(FEEDS)
    except Exception:
        feed_count = 0
    if feed_count and len(dead_feeds) >= feed_count:
        warnings.append("ALL_RSS_FEEDS_EMPTY_OR_FAILED_DURING_CURRENT_REFRESH")

    report = {
        "audit_version": "2026-10-10-v1",
        "generated_at_utc": now_utc.isoformat(),
        "expected_reference_date": expected_date.isoformat() if expected_date else None,
        "observed_reference_date": observed_date.isoformat() if observed_date else None,
        "snapshot_fresh": bool(expected_date is not None and observed_date == expected_date),
        "snapshot": {
            "rows": int(len(pit)),
            "symbols": int(symbols.nunique()) if len(symbols) else 0,
            "duplicate_symbol_rows": dupes,
            "missing_locked_features": missing_features,
            "mean_feature_non_null_pct": feature_coverage.get("mean_feature_non_null_pct"),
            "production_quality_gate_pct": gate_pct,
            "lowest_feature_coverage_symbols": lowest_coverage,
        },
        "feature_groups": group_coverage,
        "stockwise_source_coverage": stockwise_source_coverage,
        "stockwise_input_readiness": stockwise_input_readiness,
        "volume": volume_audit,
        "global_market_fx": global_audit,
        "news": news_audit,
        "fno": fno_audit,
        "realised_outcomes": realised_update or {},
        "source_status": source_status,
        "hard_failures": hard_failures,
        "warnings": sorted(set(warnings)),
    }
    report["overall_status"] = "FAIL" if hard_failures else "WARN" if warnings else "PASS"
    return report
