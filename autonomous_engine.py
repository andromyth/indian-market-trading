"""Single entry point for the dashboard's daily/intraday decision flow."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from all_nse_delivery_model import (
    predict_symbol as predict_all_nse_delivery,
    _build_latest_features as build_all_nse_features,
)
from directional_ensemble import enrich_delivery_prediction
from production_model import build_feature_rows as build_certified_intraday_features, predict as predict_certified_intraday
from prediction_ledger import append_prediction





@lru_cache(maxsize=1)
def _load_persisted_live_news() -> pd.DataFrame:
    """Load the latest persisted RSS headline snapshot from Drive."""
    local = Path(__file__).resolve().parent / "data" / "live_news" / "news_latest.parquet"
    remote = "processed/live_news/news_latest.parquet"
    try:
        if not local.exists() or local.stat().st_size == 0:
            from persistent_store import GoogleDriveStore
            from googleapiclient.http import MediaIoBaseDownload
            store = GoogleDriveStore()
            parts = remote.split("/")
            parent = store.root_id
            for part in parts[:-1]:
                found = store._find_child(parent, part)
                if not found:
                    return pd.DataFrame()
                parent = found["id"]
            found = store._find_child(parent, parts[-1])
            if not found:
                return pd.DataFrame()
            local.parent.mkdir(parents=True, exist_ok=True)
            temp = local.with_suffix(".tmp")
            with temp.open("wb") as fh:
                dl = MediaIoBaseDownload(
                    fh,
                    store.service.files().get_media(fileId=found["id"]),
                )
                done = False
                while not done:
                    _, done = dl.next_chunk()
            temp.replace(local)
        x = pd.read_parquet(local)
        if x.empty:
            return x
        x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
        x["published_at"] = pd.to_datetime(x["published_at"], errors="coerce", utc=True)
        return x.dropna(subset=["symbol", "published_at"])
    except Exception:
        return pd.DataFrame()


def _news_for_prediction(symbol: str, as_of: Any, provided: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if provided:
        return list(provided)
    x = _load_persisted_live_news()
    if x.empty:
        return []
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Kolkata")
    ts = ts.tz_convert("UTC")
    root = str(symbol).upper().replace(".NS", "").replace(".BO", "")
    hit = x[x["symbol"].isin([root, "MARKET"]) & x["published_at"].le(ts)].copy()
    return hit.to_dict("records")

def delivery_prediction(item: dict[str, Any], sync_drive: bool = True) -> dict[str, Any]:
    """Run the active ALL-NSE 59-feature Delivery candidate.

    The model artifact remains in Google Drive and is verified against the
    locked Step 5K hashes by all_nse_delivery_model.py. No retraining occurs.
    """
    symbol = str(item["symbol"])
    result = predict_all_nse_delivery(
        symbol,
        item.get("decision_timestamp"),
    )

    if result.get("prediction_status") == "PASS":

        # Add compatible auxiliary-model and multi-source directional evidence
        # before persisting the prediction. The locked ALL-NSE model remains
        # the primary trained output.
        try:
            price_df = item.get("price_df")
            market_df = item.get("market_df")

            if (
                price_df is not None
                and not price_df.empty
            ):

                feature_frame = build_all_nse_features(
                    price_df,
                    market_df,
                )

                result = enrich_delivery_prediction(
                    result,
                    symbol,
                    feature_frame,
                    price_df,
                    market_df,
                    item.get(
                        "decision_timestamp"
                    ),
                    news_items=_news_for_prediction(
                        symbol,
                        item.get("decision_timestamp"),
                        item.get("matched_news"),
                    ),
                    announcements=item.get(
                        "announcements",
                        [],
                    ),
                    fno=item.get(
                        "fo",
                        {},
                    ),
                )

        except Exception as exc:
            result["ensemble_context_status"] = "ERROR"
            result["ensemble_context_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        result["mode"] = "Delivery"
        result["prediction_horizon"] = "next_session"
        result["model_version"] = result.get(
            "model_version",
            "2026-all-nse-delivery-candidate-5-market-context-fixed",
        )

        persisted = append_prediction(
            result,
            sync_drive=sync_drive,
        )

        result["prediction_id"] = persisted.get(
            "prediction_id"
        )
        result["prediction_ledger_status"] = persisted.get(
            "status"
        )
        result["prediction_ledger_synced"] = bool(
            persisted.get("synced")
        )

        if persisted.get("sync_error"):
            result["prediction_ledger_sync_error"] = (
                persisted["sync_error"]
            )

    return result


def intraday_predictions(items: list[dict[str, Any]], sync_drive: bool = True) -> list[dict[str, Any]]:
    results = []
    for item in items:
        try:
            price_df = item["price_df"]
            decision_ts = item.get("decision_timestamp")
            if decision_ts is None and price_df is not None and not price_df.empty:
                decision_ts = price_df.index[-1]
            feature_frame = build_certified_intraday_features([{
                "symbol": item["symbol"],
                "price_df": price_df,
                "market_df": item.get("market_df"),
                "matched_news": item.get("matched_news", []),
                "announcements": item.get("announcements", []),
                "event_risk": item.get("event_risk", 0.0),
                "fo": item.get("fo") or {},
                "decision_ts": decision_ts,
            }])
            predictions = predict_certified_intraday(feature_frame)
            if not predictions:
                raise RuntimeError("Certified Intraday model returned no prediction row")
            r = predictions[0]
            r["prediction_status"] = "PASS"
            r["mode"] = "Intraday"
            r["prediction_horizon"] = "30m"
            r["model_version"] = "2026-certified-30m-137-feature"
        except Exception as exc:
            r = {
                "prediction_status": "ERROR",
                "symbol": item.get("symbol"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if r.get("prediction_status", "PASS") == "PASS":
            r["mode"] = "Intraday"
            r["prediction_horizon"] = "30m"
            r["model_version"] = "2026-all-nse-intraday-multifactor-replacement"
            append_prediction(r, sync_drive=sync_drive)
        results.append(r)
    return results
