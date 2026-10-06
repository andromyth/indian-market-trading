"""Beta V2 self-driving panel preparation.

This module is intentionally isolated from the current production model paths.
It converts the existing ALL-NSE Delivery PIT panel into a training/inference
panel for the beta self-driving challenger:

- target = next-session stock return
- market-neutral target = stock return minus same-date cross-sectional median
- global cues joined by explicit available_at
- news features joined forward-only from the archived headlines
- F&O features joined only after the source trade date
- historical source gaps remain NaN/unknown; no synthetic history is invented
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .sd_fno_features import build_fno_regime_features
from .sd_global_cues import attach_global_features
from .sd_news_features import build_news_features, score_news_frame


def normalize_delivery_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if panel is None or panel.empty:
        raise ValueError("Delivery panel is empty")

    x = panel.copy()
    if "symbol" not in x.columns or "date" not in x.columns:
        raise ValueError("Delivery panel requires symbol and date")

    x["symbol"] = (
        x["symbol"].astype(str).str.strip().str.upper()
        .str.replace(".NS", "", regex=False)
        .str.replace(".BO", "", regex=False)
        .str.replace("-EQ", "", regex=False)
        .str.replace(".EQ", "", regex=False)
    )
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()

    if "decision_timestamp" not in x.columns:
        x["decision_timestamp"] = (
            x["date"] + pd.Timedelta(hours=15, minutes=30)
        ).dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")
    else:
        x["decision_timestamp"] = pd.to_datetime(
            x["decision_timestamp"], errors="coerce", utc=True
        )

    for c in ("open", "high", "low", "close", "volume"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    x = (
        x.dropna(subset=["symbol", "date", "decision_timestamp", "close"])
        .sort_values(["symbol", "date"])
        .drop_duplicates(["symbol", "date"], keep="last")
        .reset_index(drop=True)
    )
    return x


def add_delivery_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Create next-session labels from actual panel sessions, not calendar days."""
    x = normalize_delivery_panel(panel)

    x["y_return"] = (
        x.groupby("symbol", sort=False)["close"].shift(-1) / x["close"] - 1.0
    )
    x["y_return"] = x["y_return"].replace([np.inf, -np.inf], np.nan)
    x["y_excess"] = (
        x["y_return"]
        - x.groupby("date", sort=False)["y_return"].transform("median")
    )
    x["y_up_excess"] = (x["y_excess"] > 0).astype("int8")
    x["y_up_abs"] = (x["y_return"] > 0).astype("int8")
    return x


def merge_beta_fno(panel: pd.DataFrame, fno_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join F&O regime features after their source trading date."""
    x = normalize_delivery_panel(panel)
    if fno_daily is None or fno_daily.empty:
        return x, {"status": "NO_DATA", "coverage_pct": 0.0}

    f = fno_daily.copy()
    if not {"symbol", "date"}.issubset(f.columns):
        return x, {"status": "INVALID", "coverage_pct": 0.0}

    f["symbol"] = (
        f["symbol"].astype(str).str.strip().str.upper()
        .str.replace(".NS", "", regex=False)
        .str.replace(".BO", "", regex=False)
    )
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
    f = f.dropna(subset=["symbol", "date"])

    f = build_fno_regime_features(f)
    if f.empty:
        return x, {"status": "NO_FEATURES", "coverage_pct": 0.0}

    f["decision_timestamp"] = (
        f["date"] + pd.Timedelta(days=1, hours=9, minutes=15)
    ).dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC")

    keep = [
        "symbol", "decision_timestamp",
        "fno_buildup", "fno_buildup_persist3",
        "fno_pcr_z20", "fno_basis_z20",
        "fno_oi_chg_z20", "fno_iv_chg_z20",
    ]
    f = f[[c for c in keep if c in f.columns]].drop_duplicates(
        ["symbol", "decision_timestamp"], keep="last"
    )

    pieces = []
    for symbol, lp in x.groupby("symbol", sort=False):
        rp = f[f["symbol"].eq(symbol)]
        if rp.empty:
            pieces.append(lp)
            continue
        m = pd.merge_asof(
            lp.sort_values("decision_timestamp"),
            rp.drop(columns=["symbol"]).sort_values("decision_timestamp"),
            on="decision_timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        pieces.append(m)

    out = pd.concat(pieces, ignore_index=True)
    cols = [c for c in keep if c.startswith("fno_") and c in out.columns]
    out["fno_beta_observed"] = out[cols].notna().any(axis=1).astype("Int8") if cols else 0
    return out, {
        "status": "PASS",
        "source_rows": int(len(f)),
        "source_symbols": int(f["symbol"].nunique()),
        "observed_rows": int(out["fno_beta_observed"].eq(1).sum()),
        "coverage_pct": float(out["fno_beta_observed"].mean() * 100),
        "pit_rule": "F&O trade date becomes available from next session 09:15 IST",
    }


def merge_beta_news(panel: pd.DataFrame, raw_news: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = normalize_delivery_panel(panel)
    if raw_news is None or raw_news.empty:
        return x, {"status": "NO_DATA", "archive_start": None}

    required = {"symbol", "published_at", "text"}
    if not required.issubset(raw_news.columns):
        return x, {"status": "INVALID", "archive_start": None}

    n = score_news_frame(raw_news)
    archive_start = pd.to_datetime(n["published_at"], utc=True).min()
    nf = build_news_features(
        n,
        x[["symbol", "decision_timestamp"]],
        archive_start=(
            str(archive_start.date()) if pd.notna(archive_start) else None
        ),
    )

    news_cols = [
        c for c in nf.columns if c.startswith(("news_", "mkt_news_"))
    ]
    x = x.drop(columns=[c for c in news_cols if c in x.columns], errors="ignore")
    x = x.merge(
        nf[["symbol", "decision_timestamp"] + news_cols],
        on=["symbol", "decision_timestamp"],
        how="left",
    )
    return x, {
        "status": "PASS",
        "archive_start": str(archive_start) if pd.notna(archive_start) else None,
        "news_rows": int(len(n)),
        "news_symbols": int(n["symbol"].nunique()),
    }


def build_beta_delivery_panel(
    base_panel: pd.DataFrame,
    global_cues: Optional[pd.DataFrame] = None,
    raw_news: Optional[pd.DataFrame] = None,
    fno_daily: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the complete Beta V2 Delivery panel and explicit source audit."""
    x = add_delivery_targets(base_panel)
    audit = {
        "base_rows": int(len(x)),
        "base_symbols": int(x["symbol"].nunique()),
        "base_sessions": int(x["date"].nunique()),
    }

    if global_cues is not None and not global_cues.empty:
        x = attach_global_features(x, global_cues)
        audit["global"] = {"status": "PASS", "rows": int(len(global_cues))}
    else:
        audit["global"] = {"status": "NO_DATA"}

    x, news_audit = merge_beta_news(x, raw_news)
    audit["news"] = news_audit

    x, fno_audit = merge_beta_fno(x, fno_daily)
    audit["fno"] = fno_audit

    # Preserve a strict distinction between missing source history and known zero.
    if "news_available" in x:
        x["news_available"] = pd.to_numeric(
            x["news_available"], errors="coerce"
        )
    x["beta_v2_panel_version"] = "2026-10-06-beta-v2"

    audit["final_rows"] = int(len(x))
    audit["final_columns"] = int(len(x.columns))
    audit["target_coverage_pct"] = float(x["y_return"].notna().mean() * 100)
    return x, audit


def split_training_inference(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = normalize_delivery_panel(panel)
    training = x[x["y_return"].notna()].copy()
    latest_date = x["date"].max()
    inference = x[x["date"].eq(latest_date)].copy()

    realised = (
        training[["symbol", "date", "y_return"]]
        .dropna()
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    return training, inference, realised
