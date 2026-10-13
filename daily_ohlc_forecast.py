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
            "reason": f"Chronological split too small after purge: train={len(train)}, validation={len(valid)}.",
            "history_rows": int(len(daily)),
            "forecasts": [],
        }

    validation_by_horizon: dict[int, dict[str, Any]] = {}
    target_skill_by_horizon: dict[int, dict[str, float | None]] = {
        horizon: {} for horizon in range(1, n_sessions + 1)
    }
    for target in TARGETS:
        target_col = f"target_{target}"
        model = _new_model(len(train))
        model.fit(train[feature_names], train[target_col])
        predicted = np.asarray(model.predict(valid[feature_names]), dtype=float)
        actual = valid[target_col].to_numpy(dtype=float)

        for horizon in range(1, n_sessions + 1):
            mask = valid["forecast_horizon"].to_numpy(dtype=int) == horizon
            if not mask.any():
                target_skill_by_horizon[horizon][target] = None
                continue
            actual_h = actual[mask]
            pred_h = predicted[mask]
            training_h = train.loc[
                train["forecast_horizon"].eq(float(horizon)), target_col
            ].to_numpy(dtype=float)
            baseline = float(np.nanmedian(training_h)) if len(training_h) else 0.0
            model_mae = float(np.mean(np.abs(actual_h - pred_h)))
            baseline_mae = float(np.mean(np.abs(actual_h - baseline)))
            skill = (
                float(1.0 - model_mae / baseline_mae)
                if np.isfinite(baseline_mae) and baseline_mae > 0
                else None
            )
            target_skill_by_horizon[horizon][target] = skill
            validation_by_horizon.setdefault(horizon, {})[target] = {
                "model_mae_log_return": model_mae,
                "median_baseline_mae_log_return": baseline_mae,
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
            if mean_skill is not None and mean_skill > 0 and beats >= 3
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
        prices: dict[str, float] = {}
        for target in TARGETS:
            predicted_log_return = float(models[target].predict(query[feature_names])[0])
            projected = latest_close * float(np.exp(predicted_log_return))
            if not np.isfinite(projected) or projected <= 0:
                return {
                    "status": "INVALID_FORECAST",
                    "reason": f"Invalid {target} estimate for horizon +{horizon}.",
                    "history_rows": int(len(daily)),
                    "forecasts": [],
                }
            prices[target] = projected

        # Keep the OHLC bar internally coherent without fabricating a direction:
        # the predicted high cannot be below the predicted open/close, and the
        # predicted low cannot be above either.
        prices["high"] = max(prices["high"], prices["open"], prices["close"])
        prices["low"] = min(prices["low"], prices["open"], prices["close"])
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
    return {
        "status": "PASS",
        "model": "horizon-conditioned HistGradientBoostingRegressor",
        "history_rows": int(len(daily)),
        "training_samples": int(len(train)),
        "validation_samples": int(len(valid)),
        "purge_sessions": int(n_sessions),
        "as_of_date": daily.index[-1].date().isoformat(),
        "anchor_close": latest_close,
        "overall_mean_mae_skill_vs_median": overall_skill,
        "validation_start": validation_start_date,
        "validation_end": features.index[-1].date().isoformat(),
        "validation": validation,
        "forecasts": forecasts,
        "note": (
            "Direct multi-horizon OHLC estimates from historical OHLCV. Holdout skill is "
            "reported against a horizon-specific training-median baseline; positive skill "
            "is not a guarantee of profit and low-skill forecasts must not be treated as trade signals."
        ),
    }
