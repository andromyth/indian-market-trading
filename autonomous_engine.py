"""Single entry point for the dashboard's daily/intraday decision flow."""
from __future__ import annotations

from typing import Any

from delivery_model import build_feature_row, predict as predict_delivery
from production_model import build_feature_rows, predict as predict_intraday
from prediction_ledger import append_prediction


def delivery_prediction(item: dict[str, Any], sync_drive: bool = True) -> dict[str, Any]:
    symbol = str(item["symbol"])
    frame = build_feature_row(
        item["price_df"], item.get("market_df"), news=item.get("news"),
        event=item.get("event"), fno=item.get("fno"), fundamentals=item.get("fundamentals"),
    )
    result = predict_delivery(frame, symbol, item.get("decision_timestamp"))
    if result.get("prediction_status") == "PASS":
        result["mode"] = "Delivery"
        result["prediction_horizon"] = "next_session"
        result["model_version"] = "2026.09-stage2"
        persisted = append_prediction(result, sync_drive=sync_drive)
        result['prediction_id'] = persisted.get('prediction_id')
        result['prediction_ledger_status'] = persisted.get('status')
        result['prediction_ledger_synced'] = bool(persisted.get('synced'))
        if persisted.get('sync_error'):
            result['prediction_ledger_sync_error'] = persisted['sync_error']
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
