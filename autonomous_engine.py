"""Single entry point for the dashboard's daily/intraday decision flow."""
from __future__ import annotations

from typing import Any

from all_nse_delivery_model import (
    predict_symbol as predict_all_nse_delivery,
    _build_latest_features as build_all_nse_features,
)
from directional_ensemble import enrich_delivery_prediction
from production_model import build_feature_rows as build_certified_intraday_features, predict as predict_certified_intraday
from prediction_ledger import append_prediction


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
                    news_items=item.get(
                        "matched_news",
                        [],
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
