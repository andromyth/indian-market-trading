#!/usr/bin/env python3
"""Train the pooled ALL-NSE 30-minute model exclusively from persisted five-minute snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

from snapshot_5m_features import (
    FEATURES, FEATURE_SCHEMA_VERSION, SNAPSHOT_CADENCE_MINUTES,
    make_snapshot_feature_frame, normalise_price_frame,
)

IST = "Asia/Kolkata"
SNAPSHOT_REL = Path("production_data/all_universe_minute_snapshots")
_SNAPSHOT_CACHE_ROOT: Path | None = None


def _norm(q: pd.DataFrame) -> pd.DataFrame:
    if q is None or q.empty:
        return pd.DataFrame()
    x = q.copy()
    x.columns = [str(c) for c in x.columns]
    aliases = {
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "adj close": "Close", "ltp": "Close", "volume": "Volume",
        "timestamp": "Timestamp", "datetime": "Timestamp",
        "date": "Timestamp", "time": "Timestamp",
    }
    x = x.rename(columns={c: aliases.get(str(c).strip().lower(), c) for c in x.columns})
    if "Timestamp" in x.columns:
        idx = pd.to_datetime(x.pop("Timestamp"), errors="coerce")
    else:
        idx = pd.to_datetime(x.index, errors="coerce")
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)
    x.index = idx
    for c in ("Open", "High", "Low", "Close", "Volume"):
        if c not in x:
            x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x[~x.index.isna()].sort_index()
    x = x[~x.index.duplicated(keep="last")]
    return x


def rsi_series(c: pd.Series, n: int = 14) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def make_features(g: pd.DataFrame, market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the 30-minute target on actual persisted five-minute snapshot rows."""
    del market
    x = normalise_price_frame(g)
    if x.empty:
        return pd.DataFrame()
    out = make_snapshot_feature_frame(x)
    close = pd.to_numeric(x["Close"], errors="coerce")
    right = pd.DataFrame({
        "future_ts": x.index,
        "future_close": close.to_numpy(),
    }).sort_values("future_ts")
    left = pd.DataFrame({
        "decision_ts": x.index,
        "future_ts": x.index + pd.Timedelta(minutes=30),
    }).sort_values("future_ts")
    merged = pd.merge_asof(
        left,
        right,
        on="future_ts",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=1),
    )
    future_close = merged.set_index("decision_ts")["future_close"].reindex(out.index)
    out["future_close"] = future_close.to_numpy()
    out["future_return_30m"] = out["future_close"] / close - 1.0
    out["target_30m"] = np.where(
        out["future_return_30m"].notna(),
        (out["future_return_30m"] > 0).astype(float),
        np.nan,
    )
    return out


def build_snapshot_training_cache(root: Path, cache_root: Path) -> dict[str, Any]:
    """Create symbol-partitioned rows from saved five-minute full-universe snapshots only."""
    import duckdb

    snapshot_root = root / SNAPSHOT_REL
    files = sorted(snapshot_root.rglob("snapshot_*.parquet")) if snapshot_root.exists() else []
    if not files:
        raise RuntimeError(
            f"No persisted five-minute quote snapshots found under {snapshot_root}; "
            "the one-minute candle archive is not an allowed training fallback."
        )
    cache_root = Path(cache_root)
    shutil.rmtree(cache_root, ignore_errors=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    file_list = "[" + ", ".join(
        "'" + str(path.resolve()).replace("\\", "/").replace("'", "''") + "'"
        for path in files
    ) + "]"
    partition_root = cache_root / "by_symbol"
    destination = str(partition_root.resolve()).replace("\\", "/").replace("'", "''")
    query = f"""
        COPY (
            WITH raw AS (
                SELECT
                    UPPER(TRIM(CAST(symbol AS VARCHAR))) AS symbol,
                    TRY_CAST(cycle_minute_ist AS TIMESTAMPTZ) AS snapshot_ts,
                    UPPER(TRIM(CAST(quote_status AS VARCHAR))) AS quote_status,
                    TRY_CAST(actual_observed_price AS DOUBLE) AS observed_price,
                    TRY_CAST(day_volume AS DOUBLE) AS day_volume,
                    TRY_CAST(day_open AS DOUBLE) AS day_open,
                    TRY_CAST(day_high AS DOUBLE) AS day_high,
                    TRY_CAST(day_low AS DOUBLE) AS day_low
                FROM read_parquet({file_list}, union_by_name=true)
            ),
            eligible AS (
                SELECT *
                FROM raw
                WHERE symbol <> ''
                  AND snapshot_ts IS NOT NULL
                  AND EXTRACT(MINUTE FROM (snapshot_ts AT TIME ZONE 'Asia/Kolkata')) % 5 = 0
                  AND CAST(snapshot_ts AT TIME ZONE 'Asia/Kolkata' AS TIME) >= TIME '09:15:00'
                  AND CAST(snapshot_ts AT TIME ZONE 'Asia/Kolkata' AS TIME) < TIME '15:30:00'
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY symbol, snapshot_ts
                    ORDER BY quote_status DESC
                ) = 1
            ),
            ordered AS (
                SELECT *,
                    LAG(snapshot_ts) OVER w AS previous_ts,
                    LAG(quote_status) OVER w AS previous_quote_status,
                    LAG(observed_price) OVER w AS previous_price,
                    LAG(day_volume) OVER w AS previous_day_volume
                FROM eligible
                WINDOW w AS (
                    PARTITION BY symbol, CAST(snapshot_ts AT TIME ZONE 'Asia/Kolkata' AS DATE)
                    ORDER BY snapshot_ts
                )
            ),
            features AS (
                SELECT
                    symbol,
                    snapshot_ts AT TIME ZONE 'Asia/Kolkata' AS "Timestamp",
                    CASE WHEN quote_status = 'PASS' AND observed_price > 0
                         THEN observed_price ELSE NULL END AS "Close",
                    CASE
                        WHEN quote_status = 'PASS' AND observed_price > 0
                         AND previous_quote_status = 'PASS' AND previous_price > 0
                         AND snapshot_ts - previous_ts = INTERVAL '5 minutes'
                         AND day_volume IS NOT NULL
                         AND previous_day_volume IS NOT NULL
                         AND day_volume >= previous_day_volume
                        THEN day_volume - previous_day_volume
                        ELSE NULL
                    END AS "Volume",
                    CASE WHEN quote_status = 'PASS' AND observed_price > 0 THEN day_open ELSE NULL END AS day_open,
                    CASE WHEN quote_status = 'PASS' AND observed_price > 0 THEN day_high ELSE NULL END AS day_high,
                    CASE WHEN quote_status = 'PASS' AND observed_price > 0 THEN day_low ELSE NULL END AS day_low,
                    CASE WHEN quote_status = 'PASS' AND observed_price > 0 THEN day_volume ELSE NULL END AS day_volume
                FROM ordered
            )
            SELECT * FROM features
            ORDER BY symbol, "Timestamp"
        ) TO '{destination}'
        (FORMAT PARQUET, PARTITION_BY (symbol), COMPRESSION ZSTD)
    """
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone='UTC'")
        con.execute(query)
    finally:
        con.close()
    manifest = {
        "training_input_source": "PERSISTED_FIVE_MINUTE_NSE_QUOTE_SNAPSHOTS_ONLY",
        "snapshot_cadence_minutes": SNAPSHOT_CADENCE_MINUTES,
        "snapshot_file_count": len(files),
        "snapshot_root": str(snapshot_root),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "cache_root": str(cache_root),
    }
    (cache_root / "_snapshot_training_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def load_symbol(root: Path, sym: str) -> pd.DataFrame:
    """Read one symbol's ordered history from the one-pass snapshot-only cache."""
    cache_root = _SNAPSHOT_CACHE_ROOT or (Path(root) / "snapshot_training_cache")
    symbol_dir = cache_root / "by_symbol" / f"symbol={str(sym).strip().upper()}"
    if not symbol_dir.exists():
        return pd.DataFrame()
    parts = []
    for path in sorted(symbol_dir.rglob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
            if not frame.empty:
                parts.append(normalise_price_frame(frame))
        except Exception:
            continue
    parts = [frame for frame in parts if frame is not None and not frame.empty]
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts).sort_index()
    return merged.loc[~merged.index.duplicated(keep="last")]


def load_market(root: Path) -> pd.DataFrame:
    cands = [
        root / "raw/2018_2025/NIFTY50-INDEX.parquet",
        root / "raw/2026/NIFTY50-INDEX.parquet",
    ]
    parts = []
    for p in cands:
        if p.exists():
            try:
                z = pd.read_parquet(p)
                if not z.empty:
                    parts.append(_norm(z))
            except Exception:
                pass
    if not parts:
        return pd.DataFrame()
    q = pd.concat(parts).sort_index()
    return q[~q.index.duplicated(keep="last")]


def load_fno(root: Path) -> pd.DataFrame:
    p = root / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
    try:
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fno_join_features(feature_frame: pd.DataFrame, fno: pd.DataFrame, sym: str) -> pd.DataFrame:
    fno_cols = ["fno_pcr", "fno_oi_change", "fno_fut_return", "fno_fut_oi", "fno_basis"]
    if fno is None or fno.empty:
        for c in fno_cols:
            feature_frame[c] = np.nan
        feature_frame["fno_available"] = 0.0
        return feature_frame

    z = fno.copy()
    z.columns = [str(c) for c in z.columns]
    lower = {c.lower(): c for c in z.columns}
    datecol = next((lower[k] for k in ("date", "trade_date", "timestamp") if k in lower), None)
    symcol = next((lower[k] for k in ("symbol", "ticker", "underlying") if k in lower), None)
    if not datecol or not symcol:
        for c in fno_cols:
            feature_frame[c] = np.nan
        feature_frame["fno_available"] = 0.0
        return feature_frame

    z["_date"] = pd.to_datetime(z[datecol], errors="coerce").dt.normalize()
    z = z[
        z[symcol].astype(str).str.upper().str.replace(".NS", "", regex=False).eq(sym.upper())
    ].sort_values("_date")
    if z.empty:
        for c in fno_cols:
            feature_frame[c] = np.nan
        feature_frame["fno_available"] = 0.0
        return feature_frame

    alias_map = {
        "fno_pcr": ("fno_pcr", "pcr"),
        "fno_oi_change": ("fno_oi_change", "oi_change"),
        "fno_fut_return": ("fno_fut_return", "fut_return"),
        "fno_fut_oi": ("fno_fut_oi", "fut_oi"),
        "fno_basis": ("fno_basis", "fut_basis", "fno_fut_basis"),
    }

    idx = pd.DatetimeIndex(feature_frame.index)
    if idx.tz is None:
        idx = idx.tz_localize(IST)
    else:
        idx = idx.tz_convert(IST)

    # Point-in-time rule: for an intraday session, use only an F&O observation
    # from a strictly earlier trading date. merge_asof performs this in one
    # vectorized operation instead of scanning all available dates row-by-row.
    left = pd.DataFrame({
        "_row_ts": feature_frame.index,
        "_session_date": idx.normalize().tz_localize(None),
    }).sort_values("_session_date")

    for outc, aliases in alias_map.items():
        src = next((lower[a] for a in aliases if a in lower), None)
        if not src:
            feature_frame[outc] = np.nan
            continue

        d = z[["_date", src]].copy()
        d["value"] = pd.to_numeric(d[src], errors="coerce")
        d = d.dropna(subset=["_date", "value"]).drop_duplicates("_date", keep="last")
        if d.empty:
            feature_frame[outc] = np.nan
            continue

        d = d[["_date", "value"]].sort_values("_date")
        # Shift F&O observations forward to the next trading session's intraday
        # rows by matching each session to the latest strictly prior F&O date.
        left_key = left[["_row_ts", "_session_date"]].copy()
        merged = pd.merge_asof(
            left_key,
            d,
            left_on="_session_date",
            right_on="_date",
            direction="backward",
            allow_exact_matches=False,
        )
        vals = merged.set_index("_row_ts")["value"].reindex(feature_frame.index)
        feature_frame[outc] = vals.to_numpy()

    feature_frame["fno_available"] = feature_frame[fno_cols].notna().any(axis=1).astype(float)
    return feature_frame

def sample_decisions(frame: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if frame.empty or minutes <= 1:
        return frame
    # Use the last available observation in each clock bucket, which preserves
    # minute-scale features while limiting the training matrix size.
    bucket = frame.index.floor(f"{int(minutes)}min")
    return frame.groupby(bucket, sort=True).tail(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument("--out", default="/content/drive/MyDrive/Indian_Market_Research/models/all_nse_intraday_multifactor_replacement_2026")
    ap.add_argument("--min-rows-symbol", type=int, default=300)
    ap.add_argument("--sample-minutes", type=int, default=5)
    ap.add_argument("--max-train-rows-per-symbol", type=int, default=2000)
    ap.add_argument("--snapshot-cache-dir", default="/content/all_nse_intraday_snapshot_training_cache")
    args = ap.parse_args()

    root = Path(args.drive_root)
    out = Path(args.out)
    global _SNAPSHOT_CACHE_ROOT
    _SNAPSHOT_CACHE_ROOT = Path(args.snapshot_cache_dir)

    from delivery_training_universe import official_nse_symbols
    universe = sorted(set(official_nse_symbols(force_refresh=True)))
    if not universe:
        raise RuntimeError("Official NSE universe is empty")

    snapshot_root = root / SNAPSHOT_REL
    if not snapshot_root.exists() or not any(snapshot_root.rglob("snapshot_*.parquet")):
        raise RuntimeError(
            f"Persisted five-minute snapshot archive is empty: {snapshot_root}. "
            "Training will not fall back to one-minute candle history."
        )
    snapshot_cache_manifest = build_snapshot_training_cache(root, _SNAPSHOT_CACHE_ROOT)
    market = pd.DataFrame()
    fno = load_fno(root)

    # Colab RAM is limited, so never accumulate the complete minute-derived
    # feature frame for all symbols. Each symbol is sampled evenly across its
    # available decision history and only a bounded per-symbol quota is retained.
    # This keeps all-NSE symbol coverage while bounding the pooled training matrix.
    GLOBAL_RETAINED_ROWS = 1_000_000
    per_symbol_cap = min(
        args.max_train_rows_per_symbol,
        max(100, GLOBAL_RETAINED_ROWS // max(1, len(universe))),
    )

    frames = []
    coverage = []
    trainable_symbols = 0
    retained_rows = 0

    for i, sym in enumerate(universe, 1):
        raw = load_symbol(root, sym)
        if len(raw) < args.min_rows_symbol:
            coverage.append((sym, len(raw), "EXCLUDED_INSUFFICIENT_HISTORY"))
            continue

        feat = make_features(raw, market)
        if feat.empty:
            coverage.append((sym, len(raw), "EXCLUDED_FEATURE_BUILD"))
            continue

        # Features are already computed on the shared five-minute endpoint
        # grid. Keep the explicit cadence sampler as a contract check before the
        # daily point-in-time F&O join; no minute-cadence inference features remain.
        feat = sample_decisions(feat, args.sample_minutes)
        feat = fno_join_features(feat, fno, sym)
        feat = feat.replace([np.inf, -np.inf], np.nan)
        feat = feat.dropna(subset=["target_30m"])

        if feat.empty:
            coverage.append((sym, len(raw), "EXCLUDED_NO_LABELS"))
            continue

        # Spread retained observations across the whole available history rather
        # than keeping only the latest rows. The per-symbol quota is bounded so
        # the total pooled matrix remains safe for a 12 GB Colab runtime.
        if len(feat) > per_symbol_cap:
            feat = feat.sample(
                n=per_symbol_cap,
                random_state=20261003 + i,
            ).sort_index()

        feat["_symbol"] = sym
        feat["decision_ts"] = feat.index
        frames.append(feat)
        trainable_symbols += 1
        retained_rows += len(feat)
        coverage.append((sym, len(raw), "TRAINABLE"))

        # Release the large 1-minute intermediates before continuing.
        del raw, feat

        if i % 100 == 0:
            import gc
            gc.collect()
            print(
                f"prepared {i:,}/{len(universe):,} symbols; "
                f"trainable={trainable_symbols:,}; retained_rows={retained_rows:,}; "
                f"per_symbol_cap={per_symbol_cap:,}",
                flush=True,
            )

    if not frames:
        raise RuntimeError(
            "No ALL-NSE symbols have enough persisted five-minute snapshots to build a training candidate."
        )

    data = pd.concat(frames, axis=0, ignore_index=False)
    del frames

    data = data.sort_values("decision_ts")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["target_30m"])

    # Deterministic chronological split by decision timestamp.
    cut = data["decision_ts"].quantile(0.80)
    train = data[data["decision_ts"] < cut].copy()
    test = data[data["decision_ts"] >= cut].copy()

    ytr = train["target_30m"].astype(int)
    yte = test["target_30m"].astype(int)
    if ytr.nunique() != 2 or yte.nunique() != 2:
        raise RuntimeError("Target split does not contain both classes")

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=20261003,
        )),
    ])
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])

    clf.fit(train[FEATURES], ytr)
    reg.fit(train[FEATURES], train["future_return_30m"].astype(float))

    p = clf.predict_proba(test[FEATURES])[:, 1]
    pred = (p >= 0.5).astype(int)
    pred_ret = reg.predict(test[FEATURES])

    train_symbols = sorted(data["_symbol"].unique().tolist())
    universe_size = len(universe)
    trainable = len(train_symbols)

    manifest = {
        "status": "BRANCH_REPLACEMENT_ONLY",
        "model_family": "logistic_regression+ridge",
        "target": "target_30m",
        "feature_count": len(FEATURES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "input_cadence_minutes": SNAPSHOT_CADENCE_MINUTES,
        "features": FEATURES,
        "universe_size": universe_size,
        "training_symbols": train_symbols,
        "training_symbol_count": trainable,
        "coverage_pct": round(100 * trainable / universe_size, 4) if universe_size else 0.0,
        "minimum_rows_per_symbol": args.min_rows_symbol,
        "sample_minutes": args.sample_minutes,
        "max_train_rows_per_symbol": args.max_train_rows_per_symbol,
        "effective_per_symbol_cap": int(per_symbol_cap),
        "global_retained_row_cap": int(GLOBAL_RETAINED_ROWS),
        "retained_rows_before_split": int(len(data)),
        "training_rows": int(len(train)),
        "holdout_rows": int(len(test)),
        "trained_until": str(train["decision_ts"].max()),
        "holdout_start": str(test["decision_ts"].min()),
        "validation": {
            "accuracy": float(accuracy_score(yte, pred)),
            "roc_auc": float(roc_auc_score(yte, p)),
            "mean_abs_return_error": float(
                mean_absolute_error(test["future_return_30m"], pred_ret)
            ),
        },
        "classifier_sha256": "",
        "regressor_sha256": "",
        "no_auto_orders": True,
         "training_data_sources": [
            "PERSISTED_FIVE_MINUTE_NSE_QUOTE_SNAPSHOTS_ONLY",
            "NSE_OFFICIAL_FNO_DAILY_WHEN_AVAILABLE",
        ],
        "snapshot_training_cache": snapshot_cache_manifest,
        "live_incremental_source": "PERSISTED_FIVE_MINUTE_NSE_SNAPSHOTS",
        "point_in_time_fno": True,
        "volume_policy": "Training and live volume use the difference in provider cumulative day volume between adjacent five-minute snapshots. Missing slots, invalid quotes, or volume resets stay NaN.",
    }

    out.mkdir(parents=True, exist_ok=True)
    cp = out / "deployment_all_nse_intraday_multifactor_classifier_2026.joblib"
    rp = out / "deployment_all_nse_intraday_multifactor_regressor_2026.joblib"

    joblib.dump(clf, cp)
    joblib.dump(reg, rp)

    manifest["classifier_sha256"] = hashlib.sha256(cp.read_bytes()).hexdigest()
    manifest["regressor_sha256"] = hashlib.sha256(rp.read_bytes()).hexdigest()

    (out / "ALL_NSE_INTRADAY_MULTIFACTOR_2026_MODEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        coverage,
        columns=["symbol", "stored_rows", "status"],
    ).to_csv(out / "INTRADAY_TRAINING_COVERAGE.csv", index=False)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
