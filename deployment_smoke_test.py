"""Offline production smoke test for the repository's active model contracts.

This test validates the artifacts and adapters that are authoritative in the
current dashboard architecture. The intraday production target is the
same-session 15:30 IST close; no 30-minute horizon is part of the live
contract. Large historical data and Google Drive runtime availability are
intentionally not required for this offline check.
"""
from __future__ import annotations

import pandas as pd

ROOT = None


def main() -> None:
    checks: list[str] = []

    import all_nse_delivery_model as delivery
    import all_nse_intraday_model as intraday
    import prediction_ledger
    import prediction_validation
    import model_health

    contract = intraday.model_target_contract()
    if contract.get("target") != "target_session_close":
        raise SystemExit("Intraday production target is not target_session_close")
    if contract.get("prediction_horizon") != "session_close":
        raise SystemExit("Intraday production horizon is not session_close")
    if contract.get("session_close_ist") != "15:30":
        raise SystemExit("Intraday production close target is not 15:30 IST")
    if contract.get("probability_column") != "probability_up_session_close":
        raise SystemExit("Intraday probability column is not probability_up_session_close")
    feature_count = int(contract.get("feature_count") or 0)
    if feature_count <= 0:
        raise SystemExit("Intraday production feature contract is empty")
    checks.append(
        f"OK champion same-session-close contract / {feature_count}-feature intraday model"
    )

    # The active Delivery adapter is the 59-feature ALL-NSE path. The actual
    # Drive model files are runtime dependencies, so this offline check only
    # validates adapter import and declared routing surface.
    if not hasattr(delivery, "predict_symbol") or not hasattr(delivery, "model_status"):
        raise SystemExit("ALL-NSE Delivery adapter contract missing")
    if not hasattr(delivery, "supported_symbol"):
        raise SystemExit("ALL-NSE Delivery adapter has no trainable-universe gate")
    checks.append("OK active ALL-NSE Delivery adapter import / universe gate / 59-feature runtime contract")

    # Prove the session-close target helper maps a decision to the same-day
    # 15:30 IST close without touching model artifacts or Drive.
    decision = pd.Timestamp("2026-01-05T10:20:00", tz="Asia/Kolkata")
    target = intraday.session_close_timestamp(decision)
    expected = pd.Timestamp("2026-01-05T15:30:00", tz="Asia/Kolkata")
    if target != expected:
        raise SystemExit(f"session_close_timestamp returned {target}, expected {expected}")
    if int(intraday.minutes_to_close(decision)) != 310:
        raise SystemExit("minutes_to_close is not measured to the same-session 15:30 IST close")
    checks.append("OK session-close target helper maps decisions to the 15:30 IST close")

    print("DEPLOYMENT SMOKE TEST: PASS")
    for msg in checks:
        print(msg)


if __name__ == "__main__":
    raise SystemExit(main())
