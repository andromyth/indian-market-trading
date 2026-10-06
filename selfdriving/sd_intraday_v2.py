"""Volume-free Beta V2 intraday feature/label engine.

The current production 5-minute model is intentionally not modified here.
Beta V2 uses only price/time inputs that are actually available from the NSE
public quote-chart lane: no minute-volume feature exists anywhere in this
contract. Labels are exact-session and market-neutral.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

IST = "Asia/Kolkata"
OPEN_MIN = 9 * 60 + 15
CLOSE_MIN = 15 * 60 + 30

BASE_FEATURES = [
    "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_60m",
    "rsi7", "rsi14", "atr14_pct",
    "ema8_ratio", "ema21_ratio", "ma5_ratio", "ma10_ratio",
    "ma20_ratio", "ma50_ratio",
    "session_return", "pos_in_range", "or_break", "or_dist",
    "range_pct", "body_pct", "ret_accel_1_5",
    "tod_sin", "tod_cos", "mins_since_open",
    "market_ret_5m", "market_ret_15m", "market_ret_30m",
    "relative_ret_5m", "relative_ret_15m", "relative_ret_30m",
]

# Cross-sectional targets are used for training labels, but cross-sectional
# feature ranks are not part of the live contract because a single-symbol
# Streamlit request cannot truthfully know the full universe at that instant.
RANK_FEATURE_SOURCES: list[str] = []


def _dedupe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    x.columns = [str(c).strip() for c in x.columns]
    if not x.columns.duplicated().any():
        return x

    out = pd.DataFrame(index=x.index)
    for name in list(dict.fromkeys(x.columns)):
        part = x.loc[:, x.columns.eq(name)]
        out[name] = part.iloc[:, 0] if part.shape[1] == 1 else part.bfill(axis=1).iloc[:, 0]
    return out


def _norm(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    x = _dedupe_columns(frame)
    aliases = {
        "timestamp": "ts", "datetime": "ts", "date": "ts", "time": "ts",
        "open": "open", "high": "high", "low": "low",
        "close": "close", "ltp": "close",
    }
    x = x.rename(columns={c: aliases.get(str(c).lower(), c) for c in x.columns})

    raw_ts = x.pop("ts") if "ts" in x.columns else x.index
    if isinstance(raw_ts, pd.MultiIndex):
        candidates = []
        for level in range(raw_ts.nlevels):
            parsed = pd.to_datetime(raw_ts.get_level_values(level), errors="coerce", utc=True)
            candidates.append((int(parsed.notna().sum()), parsed))
        ts = max(candidates, key=lambda z: z[0])[1] if candidates else pd.DatetimeIndex([])
    else:
        ts = pd.to_datetime(raw_ts, errors="coerce", utc=True)

    x["ts"] = pd.DatetimeIndex(ts)
    for c in ("open", "high", "low", "close"):
        if c not in x.columns:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = (
        x.dropna(subset=["ts", "open", "high", "low", "close"])
         .sort_values("ts")
         .drop_duplicates("ts", keep="last")
         .reset_index(drop=True)
    )
    return x


def _lookup_return(logc_by_min: pd.Series, mins: pd.Series, lag: int) -> np.ndarray:
    target = (mins - int(lag)).astype(int)
    values = logc_by_min.reindex(target.to_numpy()).to_numpy()
    return logc_by_min.reindex(mins.to_numpy()).to_numpy() - values


def _session_rows(
    g: pd.DataFrame,
    market: pd.DataFrame | None,
    horizons: Iterable[int],
    *,
    include_future: bool = True,
    max_decisions: int | None = None,
) -> list[dict]:
    z = _norm(g)
    if z.empty:
        return []

    z["ist"] = z["ts"].dt.tz_convert(IST)
    z["session"] = z["ist"].dt.date
    market_n = _norm(market) if market is not None else pd.DataFrame()

    # For training we only retain a small, evenly spaced set of decision
    # timestamps per symbol. Discover those timestamps before constructing
    # row dictionaries so the expensive Python row-building loop is bounded.
    selected_timestamps: set[pd.Timestamp] | None = None
    if max_decisions is not None and int(max_decisions) > 0:
        candidates: list[pd.Timestamp] = []
        horizon_list = tuple(int(h) for h in horizons)

        for _, d0 in z.groupby("session", sort=False):
            d0 = d0.sort_values("ts").copy()
            minutes0 = (
                d0["ist"].dt.hour.to_numpy(dtype=int) * 60
                + d0["ist"].dt.minute.to_numpy(dtype=int)
            )
            mask0 = (minutes0 >= OPEN_MIN) & (minutes0 < CLOSE_MIN)
            d0 = d0.loc[mask0].copy()
            minutes0 = minutes0[mask0]
            if len(d0) < 80:
                continue

            if len(pd.Index(minutes0).unique()) != len(minutes0):
                keep0 = ~pd.Index(minutes0).duplicated(keep="last")
                d0 = d0.iloc[np.flatnonzero(keep0)]
                minutes0 = minutes0[keep0]

            decision_positions = np.flatnonzero(
                (minutes0 >= OPEN_MIN + 15)
                & (((minutes0 - OPEN_MIN) % 5) == 0)
            )
            if len(decision_positions) == 0:
                continue

            minute_set = set(int(m) for m in minutes0)
            for pos0 in decision_positions:
                base_minute = int(minutes0[pos0])
                if any(base_minute + h in minute_set for h in horizon_list):
                    candidates.append(pd.Timestamp(d0["ts"].iloc[pos0]))

        candidates = sorted(set(candidates))
        if len(candidates) > int(max_decisions):
            idx0 = np.linspace(
                0, len(candidates) - 1, int(max_decisions)
            ).round().astype(int)
            candidates = [candidates[int(i)] for i in idx0]
        selected_timestamps = set(candidates)

    rows: list[dict] = []
    for session, d in z.groupby("session", sort=False):
        d = d.sort_values("ts").copy()
        minutes = (
            d["ist"].dt.hour.to_numpy(dtype=int) * 60
            + d["ist"].dt.minute.to_numpy(dtype=int)
        )
        mask = (minutes >= OPEN_MIN) & (minutes < CLOSE_MIN)
        d = d.loc[mask].copy()
        minutes = minutes[mask]
        if len(d) < 80:
            continue

        c = d["close"].to_numpy(float)
        h = d["high"].to_numpy(float)
        lo = d["low"].to_numpy(float)
        o = d["open"].to_numpy(float)
        close_by_min = pd.Series(c, index=minutes)
        logc = np.log(np.clip(c, 1e-12, None))
        log_by_min = pd.Series(logc, index=minutes)

        # Require one unique observation per minute. Duplicate minutes would
        # make future-label alignment ambiguous.
        if not close_by_min.index.is_unique:
            keep = ~pd.Index(minutes).duplicated(keep="last")
            d = d.iloc[np.flatnonzero(keep)]
            minutes = minutes[keep]
            c, h, lo, o = [arr[keep] for arr in (c, h, lo, o)]
            close_by_min = pd.Series(c, index=minutes)
            logc = np.log(np.clip(c, 1e-12, None))
            log_by_min = pd.Series(logc, index=minutes)

        decision_mask = (
            (minutes >= OPEN_MIN + 15)
            & (((minutes - OPEN_MIN) % 5) == 0)
        )
        if not bool(decision_mask.any()):
            continue

        if selected_timestamps is not None:
            decision_mask = np.asarray(
                [
                    bool(v) and pd.Timestamp(ts) in selected_timestamps
                    for v, ts in zip(decision_mask, d["ts"])
                ],
                dtype=bool,
            )
            if not bool(decision_mask.any()):
                continue

        idx = pd.DatetimeIndex(d["ts"])
        close_s = pd.Series(c, index=idx)
        open_s = pd.Series(o, index=idx)
        high_s = pd.Series(h, index=idx)
        low_s = pd.Series(lo, index=idx)

        ret1 = log_by_min.diff()
        rsi_delta = pd.Series(c, index=idx).diff()
        up = rsi_delta.clip(lower=0).ewm(
            alpha=1/14, adjust=False, min_periods=14
        ).mean()
        down = (-rsi_delta.clip(upper=0)).ewm(
            alpha=1/14, adjust=False, min_periods=14
        ).mean()
        rsi14 = 100.0 - 100.0 / (1.0 + up / down.replace(0, np.nan))
        rsi14 = rsi14.where(down.ne(0), 100.0)

        prev_close = close_s.shift(1)
        tr = pd.concat(
            [
                high_s - low_s,
                (high_s - prev_close).abs(),
                (low_s - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()

        ema8 = close_s.ewm(span=8, adjust=False, min_periods=8).mean()
        ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean()
        ma5 = close_s.rolling(5, min_periods=5).mean()
        ma10 = close_s.rolling(10, min_periods=10).mean()
        ma20 = close_s.rolling(20, min_periods=20).mean()
        ma50 = close_s.rolling(50, min_periods=50).mean()

        run_hi = high_s.cummax()
        run_lo = low_s.cummin()
        or_data = pd.DataFrame({"h": h, "l": lo}, index=idx).loc[minutes < OPEN_MIN + 15]
        or_hi = float(or_data["h"].max())
        or_lo = float(or_data["l"].min())

        mins_s = pd.Series(minutes, index=idx)
        dmask = pd.Series(decision_mask, index=idx)

        market_aligned = pd.Series(np.nan, index=idx, dtype=float)
        if not market_n.empty:
            mn = market_n.copy()
            mn["ist"] = mn["ts"].dt.tz_convert(IST)
            mn["session"] = mn["ist"].dt.date
            mn = mn[mn["session"].eq(session)].sort_values("ts")
            if not mn.empty:
                ms = pd.Series(
                    pd.to_numeric(mn["close"], errors="coerce").to_numpy(),
                    index=pd.DatetimeIndex(mn["ts"]),
                )
                market_aligned = ms.reindex(
                    idx, method="ffill", tolerance=pd.Timedelta(minutes=2)
                )

        market_log = np.log(pd.to_numeric(market_aligned, errors="coerce"))
        market_by_min = pd.Series(
            market_log.to_numpy(),
            index=minutes,
        ) if market_aligned.notna().any() else pd.Series(dtype=float)

        mret5 = mret15 = mret30 = pd.Series(np.nan, index=idx)
        if not market_by_min.empty:
            cur_m = market_by_min.reindex(minutes).to_numpy()
            mret5_vals = cur_m - market_by_min.reindex(minutes - 5).to_numpy()
            mret15_vals = cur_m - market_by_min.reindex(minutes - 15).to_numpy()
            mret30_vals = cur_m - market_by_min.reindex(minutes - 30).to_numpy()
            mret5 = pd.Series(mret5_vals, index=idx)
            mret15 = pd.Series(mret15_vals, index=idx)
            mret30 = pd.Series(mret30_vals, index=idx)

        ret1_arr = ret1.to_numpy()
        mret5_arr = np.asarray(mret5, dtype=float)
        mret15_arr = np.asarray(mret15, dtype=float)
        mret30_arr = np.asarray(mret30, dtype=float)
        r5 = _lookup_return(log_by_min, pd.Series(minutes, index=idx), 5)
        r15 = _lookup_return(log_by_min, pd.Series(minutes, index=idx), 15)
        r30 = _lookup_return(log_by_min, pd.Series(minutes, index=idx), 30)
        r60 = _lookup_return(log_by_min, pd.Series(minutes, index=idx), 60)

        rv30 = pd.Series(ret1_arr, index=idx).rolling(30, min_periods=10).apply(
            lambda a: float(np.sqrt(np.sum(a * a))), raw=True
        )

        symbol = str(g["symbol"].iloc[0]) if "symbol" in g.columns else ""

        for pos in np.flatnonzero(decision_mask):
            mins_since = int(minutes[pos] - OPEN_MIN)
            dt = pd.Timestamp(idx[pos])

            def v(series) -> float:
                values = np.asarray(series)
                value = values[pos]
                return float(value) if np.isfinite(value) else np.nan

            current = float(c[pos])
            prev = float(o[pos])
            range_now = float(run_hi.iloc[pos] - run_lo.iloc[pos])

            row = {
                "symbol": symbol,
                "decision_timestamp": dt,
                "date": pd.Timestamp(session),
                "ret_1m": float(ret1_arr[pos]) if np.isfinite(ret1_arr[pos]) else np.nan,
                "ret_5m": float(r5[pos]) if np.isfinite(r5[pos]) else np.nan,
                "ret_15m": float(r15[pos]) if np.isfinite(r15[pos]) else np.nan,
                "ret_30m": float(r30[pos]) if np.isfinite(r30[pos]) else np.nan,
                "ret_60m": float(r60[pos]) if np.isfinite(r60[pos]) else np.nan,
                "rsi7": v((100.0 - 100.0 / (1.0 + (
                    rsi_delta.clip(lower=0).ewm(alpha=1/7, adjust=False, min_periods=7).mean()
                    / (-rsi_delta.clip(upper=0)).ewm(alpha=1/7, adjust=False, min_periods=7).mean().replace(0, np.nan)
                )))),
                "rsi14": v(rsi14),
                "atr14_pct": v(atr14 / close_s.replace(0, np.nan)),
                "ema8_ratio": v(close_s / ema8.replace(0, np.nan) - 1.0),
                "ema21_ratio": v(close_s / ema21.replace(0, np.nan) - 1.0),
                "ma5_ratio": v(close_s / ma5.replace(0, np.nan) - 1.0),
                "ma10_ratio": v(close_s / ma10.replace(0, np.nan) - 1.0),
                "ma20_ratio": v(close_s / ma20.replace(0, np.nan) - 1.0),
                "ma50_ratio": v(close_s / ma50.replace(0, np.nan) - 1.0),
                "session_return": current / float(o[0]) - 1.0 if o[0] else np.nan,
                "pos_in_range": (current - float(run_lo.iloc[pos])) / range_now if range_now > 0 else 0.5,
                "or_break": 1.0 if current > or_hi else (-1.0 if current < or_lo else 0.0),
                "or_dist": (current - (or_hi + or_lo) / 2.0) / max(or_hi - or_lo, 1e-9),
                "range_pct": (float(h[pos] - lo[pos]) / current) if current else np.nan,
                "body_pct": (current - prev) / prev if prev else np.nan,
                "ret_accel_1_5": float(r5[pos] - ret1_arr[pos]) if np.isfinite(r5[pos]) and np.isfinite(ret1_arr[pos]) else np.nan,
                "tod_sin": np.sin(2*np.pi*mins_since/(CLOSE_MIN-OPEN_MIN)),
                "tod_cos": np.cos(2*np.pi*mins_since/(CLOSE_MIN-OPEN_MIN)),
                "mins_since_open": float(mins_since),
                "market_ret_5m": float(mret5_arr[pos]) if np.isfinite(mret5_arr[pos]) else np.nan,
                "market_ret_15m": float(mret15_arr[pos]) if np.isfinite(mret15_arr[pos]) else np.nan,
                "market_ret_30m": float(mret30_arr[pos]) if np.isfinite(mret30_arr[pos]) else np.nan,
                "relative_ret_5m": float(r5[pos] - mret5_arr[pos]) if np.isfinite(r5[pos]) and np.isfinite(mret5_arr[pos]) else np.nan,
                "relative_ret_15m": float(r15[pos] - mret15_arr[pos]) if np.isfinite(r15[pos]) and np.isfinite(mret15_arr[pos]) else np.nan,
                "relative_ret_30m": float(r30[pos] - mret30_arr[pos]) if np.isfinite(r30[pos]) and np.isfinite(mret30_arr[pos]) else np.nan,
            }

            for hmin in horizons:
                target_min = int(minutes[pos] + int(hmin))
                future = close_by_min.get(target_min, np.nan)

                if include_future and not np.isfinite(future):
                    continue

                row2 = row.copy()
                row2["horizon_min"] = int(hmin)
                row2["reference_close"] = current

                if np.isfinite(future):
                    row2["future_close"] = float(future)
                    row2["fwd_ret"] = (
                        float(future / current - 1.0) if current else np.nan
                    )
                    future_idx = np.where(minutes == target_min)[0]
                    row2["target_timestamp"] = (
                        pd.Timestamp(idx[future_idx[0]])
                        if len(future_idx)
                        else pd.NaT
                    )
                else:
                    row2["future_close"] = np.nan
                    row2["fwd_ret"] = np.nan
                    row2["target_timestamp"] = pd.NaT

                rows.append(row2)

    return rows


def add_cross_sectional_targets(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy()
    x["decision_timestamp"] = pd.to_datetime(x["decision_timestamp"], utc=True)
    x["fwd_ret"] = pd.to_numeric(x["fwd_ret"], errors="coerce")
    grp = x.groupby(["decision_timestamp", "horizon_min"])["fwd_ret"]
    x["cross_section_n"] = grp.transform("count")
    med = grp.transform("median")
    x["fwd_ret_excess"] = x["fwd_ret"] - med
    x["y_up_excess"] = np.where(
        x["fwd_ret_excess"].notna(),
        (x["fwd_ret_excess"] > 0).astype("int8"),
        np.nan,
    )

    for c in RANK_FEATURE_SOURCES:
        if c in x:
            x[c + "_xs_rank"] = x.groupby(
                ["decision_timestamp", "horizon_min"]
            )[c].rank(pct=True)

    return x


def feature_columns() -> list[str]:
    return list(BASE_FEATURES)
