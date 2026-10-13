"""Direct multi-horizon daily OHLC forecast from historical OHLCV.

This is a separate model from the existing Beta V2 directional Delivery model.
It trains one horizon-conditioned gradient-boosted regressor per OHLC target,
validates on a chronological holdout with a seven-session purge, and reports
skill against a training-median baseline. It never places orders.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TARGETS = ("open", "high", "low", "close")
HORIZONS = tuple(range(1, 8))


def _normalise_daily(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    x = frame.copy()
    if isinstance(x.columns, pd.MultiIndex):
        chosen = None
        for level in range(x.columns.nlevels):
            labels = {str(v).strip().lower() for v in x.columns.get_level_values(level)}
            if {"open", "high", "low", "close"}.issubset(labels):
                chosen = level
                break
        if chosen is not None:
            x.columns = x.columns.get_level_values(chosen)
        else:
            x.columns = [
                "_".join(str(v) for v in col if str(v) and str(v).lower() != "nan")
                for col in x.columns
            ]

    rename = {}
    for col in x.columns:
        key = str(col).strip().lower().replace("_", " ")
        aliases = {
            "o": "open", "open": "open",
            "h": "high", "high": "high",
            "l": "low", "low": "low",
            "c": "close", "close": "close", "adj close": "close",
            "v": "volume", "vol": "volume", "volume": "volume",
            "date": "date", "datetime": "date", "timestamp": "date",
        }
        if key in aliases:
            rename[col] = aliases[key]
    x = x.rename(columns=rename)
    x = x.loc[:, ~x.columns.duplicated(keep="first")]

    if "date" in x.columns:
        dates = pd.to_datetime(x.pop("date"), errors="coerce")
    else:
        dates = pd.Series(pd.to_datetime(x.index, errors="coerce"), index=x.index)

    try:
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass

    x.index = pd.DatetimeIndex(dates).normalize()
    x.index.name = "date"
    for col in (*TARGETS, "volume"):
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.loc[~x.index.isna()].sort_index()
    x = x[~x.index.duplicated(keep="last")]
    x = x.dropna(subset=list(TARGETS))
    x = x.loc[(x[list(TARGETS)] > 0).all(axis=1)]
    if not x.empty:
        x["volume"] = x["volume"].where(x["volume"] >= 0)
    return x[list(TARGETS) + ["volume"]]


def _features(daily: pd.DataFrame) -> pd.DataFrame:
    o = daily["open"].astype(float)
    h = daily["high"].astype(float)
    lo = daily["low"].astype(float)
    c = daily["close"].astype(float)
    v = daily["volume"].astype(float)
    logret = np.log(c).diff()
    range_pct = (h - lo) / c.replace(0, np.nan)
    prev_close = c.shift(1)

    f = pd.DataFrame(index=daily.index)
    for lag in (1, 2, 3, 5, 10, 20):
        f[f"log_return_{lag}"] = np.log(c / c.shift(lag))
    f["overnight_gap"] = np.log(o / prev_close)
    f["intraday_body"] = np.log(c / o)
    f["range_pct"] = range_pct
    f["close_location"] = (c - lo) / (h - lo).replace(0, np.nan)

    for window in (5, 10, 20, 50):
        f[f"ma_deviation_{window}"] = c / c.rolling(window).mean() - 1.0
        f[f"volatility_{window}"] = logret.rolling(window).std()
        f[f"mean_range_{window}"] = range_pct.rolling(window).mean()

    log_volume = np.log1p(v.where(v >= 0))
    f["volume_change_1"] = log_volume.diff()
    f["volume_zscore_20"] = (
        (log_volume - log_volume.rolling(20).mean())
        / log_volume.rolling(20).std().replace(0, np.nan)
    )

    weekday = pd.Series(f.index.dayofweek, index=f.index, dtype=float)
    f["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 5.0)
    f["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 5.0)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f


def _new_model(n_rows: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.035,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=max(8, min(25, int(n_rows) // 100)),
        l2_regularization=8.0,
        early_stopping=False,
        random_state=41,
    )



def _cohere_log_return_predictions(
    predicted_log_returns: np.ndarray,
    anchor_close: np.ndarray,
) -> np.ndarray:
    """Apply the same OHLC price-coherence rule to every scored and shown forecast."""
    values = np.asarray(predicted_log_returns, dtype=float)
    anchors = np.asarray(anchor_close, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(TARGETS):
        raise ValueError(f"Expected an (n, {len(TARGETS)}) log-return matrix.")
    if len(values) != len(anchors):
        raise ValueError("Forecast rows and anchor closes must have the same length.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prices = anchors[:, None] * np.exp(values)
    if (
        not np.isfinite(prices).all()
        or not np.isfinite(anchors).all()
        or (anchors <= 0).any()
        or (prices <= 0).any()
    ):
        return np.full_like(values, np.nan)

    # TARGETS order: open, high, low, close.
    prices[:, 1] = np.maximum.reduce([prices[:, 1], prices[:, 0], prices[:, 3]])
    prices[:, 2] = np.minimum.reduce([prices[:, 2], prices[:, 0], prices[:, 3]])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(prices / anchors[:, None])


def guard_daily_forecast_for_display(
    result: dict[str, Any],
    *,
    expected_latest_session: object,
    now_ist: object,
    target_sessions: list[object],
) -> dict[str, Any]:
    """Fail closed on stale source sessions or an already-started first target session.

    A forecast calculated successfully is not necessarily current or eligible for
    trading. This gate checks the exact issue-time contract before the UI renders it.
    The first target may be today's session only before the 09:15 IST open.
    """
    out = dict(result or {})
    if str(out.get("status", "")).upper() != "CALCULATED":
        out.setdefault("trade_eligibility", "NOT_ELIGIBLE")
        return out

    out["trade_eligibility"] = "NOT_ELIGIBLE_PAPER_TRIAL_REQUIRED"
    out.setdefault("forecast_status", "CALCULATED")
    out.setdefault("data_status", "PASS")
    out.setdefault("validation_status", "LOW_VALIDATION_SKILL")

    def _as_date(value: object) -> object | None:
        if value is None:
            return None
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(stamp):
            return None
        return stamp.date()

    observed_date = _as_date(out.get("as_of_date"))
    expected_date = _as_date(expected_latest_session)

    if expected_date is None or observed_date != expected_date:
        out.update({
            "status": "STALE_HISTORY",
            "data_status": "STALE",
            "forecast_status": "WITHHELD",
            "validation_status": "NOT_CURRENT_DATA",
            "reason": (
                f"Daily history through {observed_date or 'an unknown date'} does not match "
                f"the latest completed NSE session expected ({expected_date or 'unknown'})."
            ),
            "forecasts": [],
        })
        return out

    forecasts = out.get("forecasts", []) or []
    if not forecasts:
        out.update({
            "status": "NO_FORECAST_ROWS",
            "forecast_status": "WITHHELD",
            "reason": "The model returned no forecast rows; nothing can be displayed as a valid outlook.",
            "forecasts": [],
        })
        return out

    target_dates = []
    try:
        target_dates = [pd.Timestamp(value).date() for value in target_sessions]
    except (TypeError, ValueError):
        target_dates = []

    if forecasts and (
        len(target_dates) < len(forecasts)
        or any(target_dates[i] >= target_dates[i + 1] for i in range(len(target_dates) - 1))
    ):
        out.update({
            "status": "TARGET_DATES_UNAVAILABLE",
            "forecast_status": "WITHHELD",
            "reason": "The forecast target-session dates are missing, incomplete, or not strictly increasing.",
            "forecasts": [],
        })
        return out

    try:
        current = pd.Timestamp(now_ist)
        if current.tzinfo is None:
            current = current.tz_localize("Asia/Kolkata")
        else:
            current = current.tz_convert("Asia/Kolkata")
    except (TypeError, ValueError):
        out.update({
            "status": "ISSUE_TIME_INVALID",
            "forecast_status": "WITHHELD",
            "reason": "Unable to validate the forecast issue time in Asia/Kolkata.",
            "forecasts": [],
        })
        return out

    if forecasts and target_dates:
        first_target = target_dates[0]
        same_day_preopen = (
            first_target == current.date()
            and current.time() < pd.Timestamp("09:15").time()
        )
        if first_target <= current.date() and not same_day_preopen:
            out.update({
                "status": "STALE_TARGET_DATE",
                "forecast_status": "WITHHELD",
                "reason": (
                    f"The first forecast target ({first_target}) has already started or passed "
                    f"at issue time {current.strftime('%Y-%m-%d %H:%M %Z')}. "
                    "Wait for the current session's completed daily data before issuing a new seven-session outlook."
                ),
                "forecasts": [],
            })
            return out

    out["target_sessions"] = [value.isoformat() for value in target_dates[:len(forecasts)]]
    out["status"] = "CALCULATED"
    out.setdefault("reason", None)
    return out


def forecast_next_sessions(
    frame: pd.DataFrame | None,
    n_sessions: int = 7,
) -> dict[str, Any]:
    """Return direct OHLC forecasts for the next NSE sessions and holdout skill."""
    n_sessions = max(1, min(7, int(n_sessions)))
    daily = _normalise_daily(frame)
    if len(daily) < 160:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "calculation_status": "NOT_CALCULATED",
            "data_status": "INSUFFICIENT_HISTORY",
            "forecast_status": "WITHHELD",
            "validation_status": "NOT_EVALUATED",
            "trade_eligibility": "NOT_ELIGIBLE",
            "reason": f"Need at least 160 valid daily OHLC rows; received {len(daily)}.",
            "history_rows": int(len(daily)),
            "forecasts": [],
        }

    feature_frame = _features(daily)
    eligible = feature_frame["ma_deviation_50"].notna() & feature_frame["volatility_20"].notna()
    features = feature_frame.loc[eligible].copy()
    if len(features) < 100:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "calculation_status": "NOT_CALCULATED",
            "data_status": "INSUFFICIENT_HISTORY",
            "forecast_status": "WITHHELD",
            "validation_status": "NOT_EVALUATED",
            "trade_eligibility": "NOT_ELIGIBLE",
            "reason": f"Only {len(features)} rows remain after feature warm-up.",
            "history_rows": int(len(daily)),
            "forecasts": [],
        }

    source_positions = daily.index.get_indexer(features.index)
    samples: list[pd.DataFrame] = []
    for horizon in range(1, n_sessions + 1):
        block = features.copy()
        block["forecast_horizon"] = float(horizon)
        block["asof_position"] = source_positions
        anchor = daily["close"].reindex(features.index).astype(float)
        block["anchor_close"] = anchor.to_numpy(dtype=float)
        for target in TARGETS:
            future = daily[target].shift(-horizon).reindex(features.index).astype(float)
            block[f"target_{target}"] = np.log(future / anchor)
        block = block.replace([np.inf, -np.inf], np.nan)
        block = block.dropna(subset=[f"target_{name}" for name in TARGETS])
        samples.append(block)

    stacked = pd.concat(samples, axis=0).sort_index()
    if len(stacked) < 300:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "calculation_status": "NOT_CALCULATED",
            "data_status": "INSUFFICIENT_HISTORY",
            "forecast_status": "WITHHELD",
            "validation_status": "NOT_EVALUATED",
            "trade_eligibility": "NOT_ELIGIBLE",
            "reason": f"Only {len(stacked)} labelled horizon samples are available.",
            "history_rows": int(len(daily)),
            "forecasts": [],
        }

    feature_names = list(features.columns) + ["forecast_horizon"]
    validation_start_feature_pos = max(1, min(len(features) - 20, int(len(features) * 0.80)))
    validation_start_asof_pos = int(source_positions[validation_start_feature_pos])
    training_asof_cutoff = validation_start_asof_pos - n_sessions - 1

    train = stacked.loc[stacked["asof_position"] < training_asof_cutoff]
    valid = stacked.loc[stacked["asof_position"] >= validation_start_asof_pos]
    if len(train) < 200 or len(valid) < 60:
        return {
            "status": "VALIDATION_UNAVAILABLE",
            "calculation_status": "NOT_CALCULATED",
            "data_status": "PASS",
            "forecast_status": "WITHHELD",
            "validation_status": "NOT_EVALUATED",
            "trade_eligibility": "NOT_ELIGIBLE",
            "reason": f"Chronological split too small after purge: train={len(train)}, validation={len(valid)}.",
            "history_rows": int(len(daily)),
            "forecasts": [],
        }

    validation_by_horizon: dict[int, dict[str, Any]] = {}
    target_skill_by_horizon: dict[int, dict[str, float | None]] = {
        horizon: {} for horizon in range(1, n_sessions + 1)
    }

    # Fit on the chronological training slice, then score the exact coherent OHLC
    # values that will be displayed. The median baseline receives the identical
    # OHLC coherence transformation, so the comparison is like-for-like.
    raw_model_matrix = np.column_stack([
        _new_model(len(train)).fit(
            train[feature_names], train[f"target_{target}"]
        ).predict(valid[feature_names])
        for target in TARGETS
    ])
    horizon_values = valid["forecast_horizon"].to_numpy(dtype=int)
    baseline_matrix = np.full_like(raw_model_matrix, np.nan, dtype=float)
    for row_idx, horizon in enumerate(horizon_values):
        horizon_train = train.loc[train["forecast_horizon"].eq(float(horizon))]
        for target_idx, target in enumerate(TARGETS):
            values = horizon_train[f"target_{target}"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values):
                baseline_matrix[row_idx, target_idx] = float(np.median(values))

    anchors = valid["anchor_close"].to_numpy(dtype=float)
    model_scored = _cohere_log_return_predictions(raw_model_matrix, anchors)
    baseline_scored = _cohere_log_return_predictions(baseline_matrix, anchors)
    actual_matrix = valid[[f"target_{target}" for target in TARGETS]].to_numpy(dtype=float)

    if (
        not np.isfinite(model_scored).all()
        or not np.isfinite(baseline_scored).all()
        or not np.isfinite(actual_matrix).all()
    ):
        return {
            "status": "INVALID_VALIDATION_VALUES",
            "calculation_status": "NOT_CALCULATED",
            "data_status": "INVALID",
            "forecast_status": "WITHHELD",
            "validation_status": "NOT_EVALUATED",
            "trade_eligibility": "NOT_ELIGIBLE",
            "reason": "Non-finite OHLC values appeared in the coherent validation comparison.",
            "history_rows": int(len(daily)),
            "forecasts": [],
        }

    for horizon in range(1, n_sessions + 1):
        mask = horizon_values == horizon
        if not mask.any():
            for target in TARGETS:
                target_skill_by_horizon[horizon][target] = None
            continue
        for target_idx, target in enumerate(TARGETS):
            actual_h = actual_matrix[mask, target_idx]
            pred_h = model_scored[mask, target_idx]
            base_h = baseline_scored[mask, target_idx]
            model_mae = float(np.mean(np.abs(actual_h - pred_h)))
            baseline_mae = float(np.mean(np.abs(actual_h - base_h)))
            skill = (
                float(1.0 - model_mae / baseline_mae)
                if np.isfinite(baseline_mae) and baseline_mae > 0
                else None
            )
            target_skill_by_horizon[horizon][target] = skill
            validation_by_horizon.setdefault(horizon, {})[target] = {
                "model_mae_log_return_after_ohlc_coherence": model_mae,
                "median_baseline_mae_log_return_after_ohlc_coherence": baseline_mae,
                "mae_skill_vs_median": skill,
                "validation_rows": int(mask.sum()),
            }

    validation = []
    for horizon in range(1, n_sessions + 1):
        metrics = target_skill_by_horizon[horizon]
        numeric = [float(v) for v in metrics.values() if v is not None and np.isfinite(v)]
        mean_skill = float(np.mean(numeric)) if numeric else None
        beats = sum(value is not None and value > 0 for value in metrics.values())
        status = (
            "BEATS_MEDIAN_BASELINE"
            if mean_skill is not None and mean_skill > 0 and beats >= 3 and len(numeric) == len(TARGETS)
            else "LOW_VALIDATION_SKILL"
        )
        validation.append({
            "horizon": horizon,
            "mean_mae_skill_vs_median": mean_skill,
            "targets_beating_baseline": int(beats),
            "targets_evaluated": int(len(numeric)),
            "status": status,
            "target_metrics": validation_by_horizon.get(horizon, {}),
        })

    # Refit each OHLC target on every labelled row only after the chronological
    # holdout metrics have been recorded. The holdout is not used for model choice.
    models: dict[str, HistGradientBoostingRegressor] = {}
    for target in TARGETS:
        model = _new_model(len(stacked))
        model.fit(stacked[feature_names], stacked[f"target_{target}"])
        models[target] = model

    latest_features = features.iloc[[-1]].copy()
    latest_close = float(daily["close"].iloc[-1])
    forecasts = []
    for horizon in range(1, n_sessions + 1):
        query = latest_features.copy()
        query["forecast_horizon"] = float(horizon)
        raw_returns = np.asarray([
            float(models[target].predict(query[feature_names])[0])
            for target in TARGETS
        ], dtype=float).reshape(1, -1)
        coherent_returns = _cohere_log_return_predictions(
            raw_returns,
            np.asarray([latest_close], dtype=float),
        )
        if not np.isfinite(coherent_returns).all():
            return {
                "status": "INVALID_FORECAST",
                "calculation_status": "NOT_CALCULATED",
                "data_status": "INVALID",
                "forecast_status": "WITHHELD",
                "validation_status": "NOT_EVALUATED",
                "trade_eligibility": "NOT_ELIGIBLE",
                "reason": f"Invalid or non-finite coherent OHLC estimate for horizon +{horizon}.",
                "history_rows": int(len(daily)),
                "forecasts": [],
            }
        prices = {
            target: float(latest_close * np.exp(coherent_returns[0, idx]))
            for idx, target in enumerate(TARGETS)
        }
        metric = validation[horizon - 1]
        forecasts.append({
            "horizon": horizon,
            **prices,
            "mean_mae_skill_vs_median": metric["mean_mae_skill_vs_median"],
            "targets_beating_baseline": metric["targets_beating_baseline"],
            "targets_evaluated": metric["targets_evaluated"],
            "validation_status": metric["status"],
        })

    all_skill = [
        row["mean_mae_skill_vs_median"]
        for row in validation
        if row["mean_mae_skill_vs_median"] is not None
    ]
    overall_skill = float(np.mean(all_skill)) if all_skill else None
    validation_start_date = features.index[validation_start_feature_pos].date().isoformat()
    validation_end_position = int(valid["asof_position"].max())
    validation_end_date = daily.index[validation_end_position].date().isoformat()
    validation_outcome_end_date = daily.index[-1].date().isoformat()
    all_horizons_beat_baseline = bool(validation) and all(
        row["status"] == "BEATS_MEDIAN_BASELINE" for row in validation
    )
    global_validation_status = (
        "BASELINE_EDGE_ON_CHRONOLOGICAL_HOLDOUT"
        if all_horizons_beat_baseline and overall_skill is not None and overall_skill > 0
        else "LOW_VALIDATION_SKILL"
    )
    return {
        "status": "CALCULATED",
        "calculation_status": "CALCULATED",
        "data_status": "PASS",
        "forecast_status": "CALCULATED",
        "validation_status": global_validation_status,
        "trade_eligibility": "NOT_ELIGIBLE_PAPER_TRIAL_REQUIRED",
        "validation_gate": (
            "PASS_ON_SINGLE_CHRONOLOGICAL_HOLDOUT_ONLY"
            if global_validation_status == "BASELINE_EDGE_ON_CHRONOLOGICAL_HOLDOUT"
            else "FAIL_BASELINE_COMPARISON"
        ),
        "model": "horizon-conditioned HistGradientBoostingRegressor",
        "history_rows": int(len(daily)),
        "training_samples": int(len(train)),
        "validation_samples": int(len(valid)),
        "purge_sessions": int(n_sessions),
        "as_of_date": daily.index[-1].date().isoformat(),
        "anchor_close": latest_close,
        "overall_mean_mae_skill_vs_median": overall_skill,
        "validation_start": validation_start_date,
        "validation_end": validation_end_date,
        "validation_outcome_end": validation_outcome_end_date,
        "validation": validation,
        "forecasts": forecasts,
        "reason": None if global_validation_status == "BASELINE_EDGE_ON_CHRONOLOGICAL_HOLDOUT" else (
            "Validation did not beat the baseline at every forecast horizon. "
            "Forecasts may be reviewed for research but are not qualified for trading."
        ),
        "note": (
            "Direct multi-horizon OHLC estimates from historical OHLCV. The chronological "
            "holdout is scored after the same OHLC-coherence transformation applied to the "
            "shown forecasts. A single holdout is not trading approval; low-skill forecasts "
            "must not be treated as trading-grade targets."
        ),
    }
