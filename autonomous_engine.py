"""Single entry point for the dashboard's daily/intraday decision flow."""
from __future__ import annotations

from typing import Any

from all_nse_delivery_model import predict_symbol as predict_all_nse_delivery
from production_model import build_feature_rows, predict as predict_intraday
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
    feature_items = []
    for item in items:
        feature_items.append({
            "symbol": item["symbol"], "price_df": item["price_df"],
            "market_df": item.get("market_df"), "matched_news": item.get("matched_news", []),
            "announcements": item.get("announcements", []), "event_risk": item.get("event_risk", 0),
            "fo": item.get("fo", {}),
        })
    frame = build_feature_rows(feature_items)
    results = predict_intraday(frame)
    for r in results:
        if r.get("prediction_status", "PASS") == "PASS":
            r["mode"] = "Intraday"
            r["prediction_horizon"] = "30m"
            r["model_version"] = "production-2026.09"
            append_prediction(r, sync_drive=sync_drive)
    return results
