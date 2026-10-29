"""Five-minute snapshot feature contract shared by live inference and training."""
from __future__ import annotations
from datetime import datetime, time
from typing import Any
import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
SNAPSHOT_CADENCE_MINUTES = 5
FEATURE_SCHEMA_VERSION = "nse_5m_snapshot_close_v1"
FEATURES = [
    "ret_5m", "ret_15m", "ret_25m", "ret_50m", "ret_75m", "ret_150m",
    "rsi_70m", "ma_25m_ratio", "ma_50m_ratio", "ma_100m_ratio", "ma_250m_ratio",
    "volume_ratio_100m", "volume_z_100m", "session_return", "day_high_ratio", "day_low_ratio",
    "tod_sin", "tod_cos", "fno_pcr", "fno_oi_change", "fno_fut_return", "fno_fut_oi", "fno_basis", "fno_available",
]
OPTIONAL_FNO_FEATURES = (
    "fno_pcr", "fno_oi_change", "fno_fut_return", "fno_fut_oi", "fno_basis",
)
# Readiness means core five-minute price/technical/context availability.
# F&O fields are optional for most NSE cash equities and must not make a
# healthy cash snapshot look data-limited when fno_available is explicitly 0.
READINESS_FEATURES = tuple(name for name in FEATURES if name not in OPTIONAL_FNO_FEATURES)
RETURN_LAGS = {"ret_5m": 1, "ret_15m": 3, "ret_25m": 5, "ret_50m": 10, "ret_75m": 15, "ret_150m": 30}


def _safe_float(value: Any) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError, OverflowError):
        return np.nan


def normalise_price_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Normalise columns without manufacturing OHLC from sampled LTP values."""
    columns = ["Open", "High", "Low", "Close", "Volume", "day_open", "day_high", "day_low", "day_volume"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz=IST, name="Datetime"))
    x = frame.copy()
    aliases = {
        "open":"Open", "high":"High", "low":"Low", "close":"Close", "adj close":"Close", "ltp":"Close",
        "volume":"Volume", "vol":"Volume", "day_open":"day_open", "day open":"day_open",
        "day_high":"day_high", "day high":"day_high", "day_low":"day_low", "day low":"day_low",
        "day_volume":"day_volume", "day volume":"day_volume", "timestamp":"Timestamp", "datetime":"Timestamp",
        "date":"Timestamp", "time":"Timestamp",
    }
    x.columns = [aliases.get(str(c).strip().lower(), str(c).strip()) for c in x.columns]
    x = x.loc[:, ~x.columns.duplicated(keep="last")]
    idx = pd.DatetimeIndex(pd.to_datetime(x.pop("Timestamp"), errors="coerce")) if "Timestamp" in x else pd.DatetimeIndex(pd.to_datetime(x.index, errors="coerce"))
    idx = idx.tz_localize(IST) if idx.tz is None else idx.tz_convert(IST)
    x.index = idx
    for col in columns:
        if col not in x:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.loc[~x.index.isna()].sort_index()
    return x.loc[~x.index.duplicated(keep="last"), columns]


def resample_minute_candles_to_snapshot_5m(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Align completed minute candles to live five-minute decision endpoints."""
    x = normalise_price_frame(frame)
    if x.empty or x["Close"].notna().sum() == 0:
        return pd.DataFrame(columns=["Close", "Volume", "day_open", "day_high", "day_low"])
    output = []
    for session_date, day in x.groupby(pd.Index(x.index.date), sort=True):
        day = day.loc[(day.index.time >= time(9, 15)) & (day.index.time < time(15, 30))]
        if day.empty:
            continue
        start = pd.Timestamp(datetime.combine(session_date, time(9, 15)), tz=IST)
        end = pd.Timestamp(datetime.combine(session_date, time(15, 30)), tz=IST)
        opens, closes = day["Open"].dropna(), day["Close"].dropna()
        day_open = _safe_float(opens.iloc[0]) if not opens.empty else (_safe_float(closes.iloc[0]) if not closes.empty else np.nan)
        for slot in pd.date_range(start + pd.Timedelta(minutes=5), end - pd.Timedelta(minutes=5), freq="5min"):
            window = day.loc[(day.index >= slot - pd.Timedelta(minutes=5)) & (day.index < slot)]
            expected_minutes = pd.date_range(
                slot - pd.Timedelta(minutes=5), periods=5, freq="min"
            )
            if not window.index.equals(expected_minutes):
                continue
            close_values = window["Close"].dropna()
            if close_values.empty:
                continue
            close = _safe_float(close_values.iloc[-1])
            if not np.isfinite(close) or close <= 0:
                continue
            prior = day.loc[day.index < slot]
            high, low = pd.to_numeric(prior["High"], errors="coerce").dropna(), pd.to_numeric(prior["Low"], errors="coerce").dropna()
            output.append({
                "Timestamp": slot, "Close": close,
                "Volume": _safe_float(pd.to_numeric(window["Volume"], errors="coerce").sum(min_count=1)),
                "day_open": day_open,
                "day_high": _safe_float(high.max()) if not high.empty else close,
                "day_low": _safe_float(low.min()) if not low.empty else close,
            })
    if not output:
        return pd.DataFrame(columns=["Close", "Volume", "day_open", "day_high", "day_low"])
    out = pd.DataFrame(output).set_index("Timestamp").sort_index()
    out.index.name = "Datetime"
    return out.loc[~out.index.duplicated(keep="last")]


def _rsi_series(
    close: pd.Series,
    period: int = 14,
    session_key: pd.Series | None = None,
) -> pd.Series:
    c = pd.to_numeric(close, errors="coerce")
    delta = c.diff()
    # Keep the rolling RSI state across sessions, but never interpret the
    # overnight close-to-open discontinuity as an intraday return.
    if session_key is not None:
        delta = delta.where(session_key.eq(session_key.shift(1)))
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    result = 100 - 100 / (1 + up / down.replace(0, np.nan))
    return result.where(down.ne(0), 100.0)


def _fill_session_prefix_from_previous_session(values: pd.Series, session_key: pd.Series) -> pd.Series:
    """Fill current-session warm-up prefix with the previous session's last known feature."""
    out = values.copy()
    previous_last = np.nan
    for _, labels in session_key.groupby(session_key, sort=True).groups.items():
        labels = list(labels)
        group = out.loc[labels]
        valid = group.dropna()
        if np.isfinite(previous_last):
            if valid.empty:
                out.loc[labels] = group.fillna(previous_last)
            else:
                prefix = labels[:labels.index(valid.index[0])]
                if prefix:
                    out.loc[prefix] = out.loc[prefix].fillna(previous_last)
        after = out.loc[labels].dropna()
        if not after.empty:
            previous_last = _safe_float(after.iloc[-1])
    return out


def make_snapshot_feature_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Build features on a complete five-minute clock grid, preserving missing slots.

    The grid begins at the NSE open and ends at the latest supplied slot for each
    session. Missing quote rows stay missing; row-count shifts must never masquerade
    as five-minute returns when a source snapshot is absent.
    """
    source = normalise_price_frame(frame)
    if source.empty or source["Close"].notna().sum() == 0:
        return pd.DataFrame(columns=FEATURES, index=source.index)
    sessions = []
    for session_date, day in source.groupby(pd.Index(source.index.date), sort=True):
        day = day.loc[(day.index.time >= time(9, 15)) & (day.index.time < time(15, 30))]
        if day.empty:
            continue
        session_open = pd.Timestamp(datetime.combine(session_date, time(9, 15)), tz=IST)
        latest = day.index.max().floor("5min")
        if latest < session_open:
            continue
        grid = pd.date_range(session_open, latest, freq="5min")
        sessions.append(day.reindex(grid))
    if not sessions:
        return pd.DataFrame(columns=FEATURES, index=source.index)
    x = pd.concat(sessions).sort_index()
    idx = pd.DatetimeIndex(x.index)
    session_key = pd.Series(idx.date, index=x.index)
    close = pd.to_numeric(x["Close"], errors="coerce")
    volume = pd.to_numeric(x["Volume"], errors="coerce")
    out = pd.DataFrame(index=x.index)

    for name, lag in RETURN_LAGS.items():
        out[name] = close.groupby(session_key, sort=False).pct_change(periods=lag, fill_method=None)
    # Long rolling indicators use only observed five-minute samples, including
    # previous sessions. Short-horizon return features above remain session-local.
    out["rsi_70m"] = _rsi_series(close, 14, session_key)

    for minutes, window in ((25, 5), (50, 10), (100, 20), (250, 50)):
        moving_average = close.rolling(window, min_periods=window).mean()
        out[f"ma_{minutes}m_ratio"] = close / moving_average - 1.0

    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std()
    out["volume_ratio_100m"] = volume / volume_mean
    out["volume_z_100m"] = (volume - volume_mean) / volume_std.replace(0, np.nan)

    day_open = x["day_open"].copy()
    missing_open = day_open.isna()
    fallback_open = (
        x["Open"].groupby(session_key, sort=False).transform("first")
        if x["Open"].notna().any()
        else close.groupby(session_key, sort=False).transform("first")
    )
    day_open.loc[missing_open] = fallback_open.loc[missing_open]
    fallback_high = (
        x["High"].groupby(session_key, sort=False).cummax()
        if x["High"].notna().any()
        else close.groupby(session_key, sort=False).cummax()
    )
    fallback_low = (
        x["Low"].groupby(session_key, sort=False).cummin()
        if x["Low"].notna().any()
        else close.groupby(session_key, sort=False).cummin()
    )
    day_high = x["day_high"].fillna(fallback_high)
    day_low = x["day_low"].fillna(fallback_low)
    out["session_return"] = close / day_open - 1.0
    out["day_high_ratio"] = close / day_high - 1.0
    out["day_low_ratio"] = close / day_low - 1.0

    elapsed = np.clip(idx.hour * 60 + idx.minute - 555, 0, 389)
    out["tod_sin"] = np.sin(2 * np.pi * elapsed / 390.0)
    out["tod_cos"] = np.cos(2 * np.pi * elapsed / 390.0)
    for name in ("fno_pcr", "fno_oi_change", "fno_fut_return", "fno_fut_oi", "fno_basis"):
        out[name] = np.nan
    out["fno_available"] = 0.0
    return out.reindex(columns=FEATURES)


def latest_snapshot_features(price_df: pd.DataFrame, fno: dict[str, Any] | None = None) -> dict[str, float]:
    """Return latest snapshot features plus point-in-time daily F&O context."""
    frame = make_snapshot_feature_frame(price_df)
    if frame.empty:
        raise ValueError("No usable five-minute NSE snapshot history")
    row = frame.iloc[-1].to_dict()
    context = fno or {}
    aliases = {
        "fno_pcr": ("pcr","fno_pcr"), "fno_oi_change": ("oi_change","fno_oi_change"),
        "fno_fut_return": ("fut_return","fno_fut_return"), "fno_fut_oi": ("fut_oi","fno_fut_oi"),
        "fno_basis": ("basis","fno_basis"),
    }
    for target, names in aliases.items():
        value = next((context[name] for name in names if context.get(name) is not None), None)
        row[target] = _safe_float(value)
    row["fno_available"] = 1.0 if any(np.isfinite(_safe_float(row[k])) for k in aliases) else 0.0
    return {name: _safe_float(row.get(name)) for name in FEATURES}
