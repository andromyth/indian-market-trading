#!/usr/bin/env python3
"""Build the next-session Delivery candidate from the full current NSE equity universe.

This is a research/training job only. It does not modify the bundled production
artifacts or promote a candidate model. The training roster comes from NSE's
official securities-available-for-trading file, while price observations come
from the persistent official NSE CM-UDiFF EOD archive.

Design rules:
- all current NSE equity symbols are part of the roster;
- only symbols with sufficient clean historical sessions are trainable;
- labels are strictly next-session and point-in-time;
- broad-market features use official NSE NIFTY 50 / NIFTY BANK history;
- unavailable optional news/F&O/fundamental fields remain explicitly missing;
- candidate artifacts are written only to the separate candidate directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from delivery_training_universe import benchmark_for_symbol, training_roster
from nse_data_layer import fetch_index_history

FEATURES = [
    "ret_1","ret_3","ret_6","ret_12","ret_24","ret_48","rsi14","atr14",
    "ma20_ratio","ma50_ratio","volume_ratio","volume_z20","range_pct","body_pct","up_bar",
    "market_ret_1","market_ret_3","market_ret_6","market_ret_12","market_rsi14",
    "relative_ret_1","relative_ret_6","relative_ret_12","market_mood_proxy","market_stress_proxy",
    "news_available","event_available","event_high_impact","fno_pcr","fno_oi_change",
    "fno_iv_change","fno_available","fno_fut_close","fno_fut_return","fno_fut_oi",
    "fundamental_revenue","fundamental_pat","fundamental_eps","fundamental_ebitda",
    "fundamental_finance_cost","fundamental_total_equity","fundamental_total_debt",
    "fundamental_cash","fund_revenue_growth","fund_pat_growth","fund_eps_growth",
    "fund_ebitda_growth","fund_pat_margin","fund_debt_to_equity","fund_cash_to_debt",
    "fundamental_age_days","fundamental_stale_90d","fundamental_stale_180d",
]

BANK_SYMBOLS = {
    "HDFCBANK","ICICIBANK","SBIN","AXISBANK","KOTAKBANK",
    "INDUSINDBK","BANKBARODA","PNB",
}


def load_bulk(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("**/BhavCopy_NSE_CM_0_0_0_*.parquet"))
    if not files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    x = pd.concat(frames, ignore_index=True)
    x["symbol"] = x["symbol"].astype(str).str.strip().str.upper()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    for col in ["open","high","low","close","volume"]:
        if col not in x.columns:
            x[col] = np.nan
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.dropna(subset=["symbol","date","open","high","low","close"])
    return (
        x.sort_values(["symbol","date"])
        .drop_duplicates(["symbol","date"], keep="last")
        .reset_index(drop=True)
    )


def _rsi(s: pd.Series, n: int = 14) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < n + 1:
        return np.nan
    d = x.diff()
    gain = d.clip(lower=0).rolling(n).mean().iloc[-1]
    loss = (-d.clip(upper=0)).rolling(n).mean().iloc[-1]
    if not np.isfinite(gain) or not np.isfinite(loss):
        return np.nan
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _atr(x: pd.DataFrame, n: int = 14) -> float:
    c = pd.to_numeric(x["close"], errors="coerce")
    h = pd.to_numeric(x["high"], errors="coerce")
    l = pd.to_numeric(x["low"], errors="coerce")
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    v = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean().iloc[-1]
    return float(v) if np.isfinite(v) else np.nan


def _ret(s: pd.Series, n: int) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) <= n:
        return np.nan
    return float(x.iloc[-1] / x.iloc[-1-n] - 1.0)


def stock_feature_row(hist: pd.DataFrame) -> dict[str, float]:
    c = hist["close"]
    h = hist["high"]
    l = hist["low"]
    o = hist["open"]
    v = hist["volume"]

    last = float(c.iloc[-1])
    last_open = float(o.iloc[-1])
    ma20 = float(c.rolling(20).mean().iloc[-1]) if len(c) >= 20 else np.nan
    ma50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else np.nan
    vma20 = float(v.rolling(20).mean().iloc[-1]) if len(v) >= 20 else np.nan
    vstdev20 = float(v.rolling(20).std().iloc[-1]) if len(v) >= 20 else np.nan

    return {
        "ret_1": _ret(c, 1),
        "ret_3": _ret(c, 3),
        "ret_6": _ret(c, 6),
        "ret_12": _ret(c, 12),
        "ret_24": _ret(c, 24),
        "ret_48": _ret(c, 48),
        "rsi14": _rsi(c, 14),
        "atr14": _atr(hist, 14),
        "ma20_ratio": last / ma20 - 1.0 if np.isfinite(ma20) and ma20 else np.nan,
        "ma50_ratio": last / ma50 - 1.0 if np.isfinite(ma50) and ma50 else np.nan,
        "volume_ratio": float(v.iloc[-1] / vma20) if np.isfinite(vma20) and vma20 else np.nan,
        "volume_z20": float((v.iloc[-1] - vma20) / vstdev20) if np.isfinite(vma20) and np.isfinite(vstdev20) and vstdev20 else np.nan,
        "range_pct": float((h.iloc[-1] - l.iloc[-1]) / last) if last else np.nan,
        "body_pct": float((last - last_open) / last_open) if last_open else np.nan,
        "up_bar": 1.0 if last > last_open else 0.0,
    }


def market_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = x.dropna(subset=["date","close"]).sort_values("date").drop_duplicates("date", keep="last")

    c = x["close"]
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    market_rsi = 100.0 - 100.0 / (1.0 + gain / loss)
    market_rsi = market_rsi.where(loss != 0, 100.0)

    returns = c.pct_change()
    out = pd.DataFrame({"date": x["date"]})
    out["market_ret_1"] = c.pct_change(1)
    out["market_ret_3"] = c.pct_change(3)
    out["market_ret_6"] = c.pct_change(6)
    out["market_ret_12"] = c.pct_change(12)
    out["market_rsi14"] = market_rsi
    out["market_mood_proxy"] = out["market_ret_1"]
    out["market_stress_proxy"] = returns.rolling(20).std()
    return out


def build_panel(
    bulk: pd.DataFrame,
    market_tables: dict[str, pd.DataFrame],
    trainable_symbols: list[str],
    min_history: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    trainable_set = set(trainable_symbols)
    selected = bulk[bulk["symbol"].isin(trainable_set)].copy()

    for symbol, group in selected.groupby("symbol", sort=False):
        g = group.sort_values("date").reset_index(drop=True)
        benchmark = benchmark_for_symbol(symbol)
        market = market_tables.get(benchmark, pd.DataFrame())

        if len(g) < min_history + 1:
            continue

        market_by_date = (
            market.set_index("date").to_dict("index")
            if not market.empty
            else {}
        )

        for i in range(min_history, len(g) - 1):
            hist = g.iloc[: i + 1]
            current_date = pd.Timestamp(g.loc[i, "date"]).normalize()
            base = stock_feature_row(hist)

            m = market_by_date.get(current_date, {})
            base["market_ret_1"] = float(m.get("market_ret_1", np.nan))
            base["market_ret_3"] = float(m.get("market_ret_3", np.nan))
            base["market_ret_6"] = float(m.get("market_ret_6", np.nan))
            base["market_ret_12"] = float(m.get("market_ret_12", np.nan))
            base["market_rsi14"] = float(m.get("market_rsi14", np.nan))
            base["market_mood_proxy"] = float(m.get("market_mood_proxy", np.nan))
            base["market_stress_proxy"] = float(m.get("market_stress_proxy", np.nan))

            for n in (1, 6, 12):
                stock_key = f"ret_{n}"
                market_key = f"market_ret_{n}"
                base[f"relative_ret_{n}"] = (
                    base[stock_key] - base[market_key]
                    if np.isfinite(base[stock_key]) and np.isfinite(base[market_key])
                    else np.nan
                )

            # Optional information is deliberately missing unless a separate
            # point-in-time dataset is explicitly merged. This preserves the
            # fixed 53-column contract without substituting current/future data.
            for col in FEATURES:
                base.setdefault(col, np.nan)

            base["symbol"] = symbol
            base["date"] = current_date
            next_close = float(g.loc[i + 1, "close"])
            base["y_return"] = next_close / float(g.loc[i, "close"]) - 1.0
            base["y_up"] = int(base["y_return"] > 0)
            rows.append(base)

    return pd.DataFrame(rows)


def fit_models(panel: pd.DataFrame):
    if panel.empty:
        raise RuntimeError("Training panel is empty.")

    dates = sorted(pd.to_datetime(panel["date"]).dropna().unique())
    if len(dates) < 5:
        raise RuntimeError("Not enough distinct sessions for a chronological holdout.")

    cutoff = dates[max(1, int(len(dates) * 0.80) - 1)]
    train = panel[panel["date"] <= cutoff].copy()
    test = panel[panel["date"] > cutoff].copy()

    if train["y_up"].nunique() < 2:
        raise RuntimeError("Training split contains only one class.")
    if test.empty or test["y_up"].nunique() < 2:
        raise RuntimeError("Holdout split lacks sufficient class diversity.")

    Xtr = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    Xte = test[FEATURES].apply(pd.to_numeric, errors="coerce")
    ytr = train["y_up"].astype(int)
    yte = test["y_up"].astype(int)

    clf = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )),
        ("scale", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(
            C=0.5,
            max_iter=500,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    reg = Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True,
            keep_empty_features=True,
        )),
        ("model", HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.5,
            random_state=42,
        )),
    ])

    clf.fit(Xtr, ytr)
    reg.fit(Xtr, train["y_return"].astype(float))

    p = clf.predict_proba(Xte)[:, 1]
    pr = reg.predict(Xte)

    metrics = {
        "holdout_accuracy": float(accuracy_score(yte, (p >= 0.5).astype(int))),
        "holdout_auc": float(roc_auc_score(yte, p)),
        "holdout_mae_return": float(mean_absolute_error(test["y_return"], pr)),
        "train_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "holdout_start": str(pd.Timestamp(test["date"].min()).date()),
        "holdout_end": str(pd.Timestamp(test["date"].max()).date()),
    }
    return clf, reg, metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bulk-root",
        default="/content/drive/MyDrive/Indian_Market_Research/processed/universe_expansion_2026_v4/bulk_eod",
    )
    ap.add_argument(
        "--out",
        default="/content/drive/MyDrive/Indian_Market_Research/models/all_nse_delivery_candidate_2026",
    )
    ap.add_argument("--min-history", type=int, default=60)
    ap.add_argument(
        "--allow-missing-market-context",
        action="store_true",
        help="Continue training when an official NSE benchmark history cannot be loaded.",
    )
    args = ap.parse_args()

    bulk_root = Path(args.bulk_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bulk = load_bulk(bulk_root)
    if bulk.empty:
        raise SystemExit(
            "No official NSE bulk EOD parquet files found. Run the automatic NSE EOD "
            "backfill/sync first; do not substitute a public fallback for training."
        )

    roster = training_roster(bulk, min_history=args.min_history)
    trainable = roster["trainable_symbols"]
    if not trainable:
        raise SystemExit("NSE official roster contains no symbols with sufficient clean history.")

    bulk_dates = pd.to_datetime(bulk["date"]).dropna()
    start = (bulk_dates.min() - pd.Timedelta(days=90)).date()
    end = bulk_dates.max().date()

    market_tables: dict[str, pd.DataFrame] = {}
    market_status: dict[str, dict] = {}

    for benchmark in ("^NSEI", "^NSEBANK"):
        try:
            market_raw = fetch_index_history(
                benchmark,
                start,
                end,
                sync_drive=True,
            ).rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                }
            )
            market_raw = market_raw.rename(
                columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"}
            )
            market_tables[benchmark] = market_feature_table(market_raw.rename(columns={"Close":"close"}) if "Close" in market_raw.columns else market_raw)
            latest_market = market_tables[benchmark]["date"].max().date() if not market_tables[benchmark].empty else None
            bulk_latest = pd.Timestamp(bulk["date"].max()).date()
            if latest_market is None or latest_market < bulk_latest - pd.Timedelta(days=10):
                raise RuntimeError(
                    f"Official NSE benchmark history is stale for {benchmark}: "
                    f"latest={latest_market}, bulk_latest={bulk_latest}"
                )
            market_status[benchmark] = {
                "status": "PASS",
                "rows": int(len(market_tables[benchmark])),
                "latest": str(latest_market),
            }
        except Exception as exc:
            market_status[benchmark] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            market_tables[benchmark] = pd.DataFrame()

    if not args.allow_missing_market_context:
        failed = {k:v for k,v in market_status.items() if v.get("status") != "PASS"}
        if failed:
            raise SystemExit(
                "Official NSE benchmark history is unavailable. "
                "Use --allow-missing-market-context only for a diagnostic candidate run: "
                + json.dumps(failed, default=str)
            )

    panel = build_panel(bulk, market_tables, trainable, args.min_history)
    if panel.empty:
        raise SystemExit("No training rows could be built from the all-NSE roster.")

    clf, reg, metrics = fit_models(panel)

    joblib.dump(clf, out / "deployment_all_nse_delivery_classifier_2026.joblib")
    joblib.dump(reg, out / "deployment_all_nse_delivery_regressor_2026.joblib")

    feature_coverage = {
        col: float(panel[col].notna().mean())
        for col in FEATURES
    }

    manifest = {
        "engine_version": "2026-all-nse-delivery-candidate-2",
        "model_family": "logistic_regression + hgb_return_regressor",
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "universe_source": roster["universe_source"],
        "universe_as_of": roster["universe_as_of"],
        "universe_symbol_count": roster["universe_symbol_count"],
        "symbols_with_bulk_history": roster["symbols_with_bulk_history"],
        "trainable_symbol_count": roster["trainable_symbol_count"],
        "trainable_symbols": trainable,
        "excluded_symbol_count": roster["excluded_symbol_count"],
        "minimum_history_sessions": args.min_history,
        "training_panel_rows": int(len(panel)),
        "market_context": market_status,
        "feature_coverage": feature_coverage,
        "optional_information_policy": (
            "News/F&O/fundamentals remain explicitly missing unless separate "
            "point-in-time datasets are merged. Current/future values are never substituted."
        ),
        "source": "NSE_OFFICIAL_EQUITY_L + NSE_OFFICIAL_UDIFF_BHAVCOPY + NSE_OFFICIAL_INDEX_HISTORY",
        "metrics": metrics,
        "promotion_status": "CANDIDATE_ONLY",
        "production_artifacts_changed": False,
    }
    (out / "ALL_NSE_DELIVERY_MODEL_MANIFEST_2026.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "universe_symbol_count": roster["universe_symbol_count"],
                "symbols_with_bulk_history": roster["symbols_with_bulk_history"],
                "trainable_symbol_count": roster["trainable_symbol_count"],
                "training_panel_rows": len(panel),
                "metrics": metrics,
                "market_context": market_status,
                "promotion_status": "CANDIDATE_ONLY",
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
