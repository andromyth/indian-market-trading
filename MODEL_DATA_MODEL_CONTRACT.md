# Production data-model contract

## Live capture contract: `target_session_close`

The live ALL-NSE capture predicts whether the **same-session 15:30 IST close** will be above the decision price. Rows carry:

- `model_target = target_session_close`
- `target_timestamp` = the same trading day at 15:30 IST
- `horizon_minutes` / `minutes_to_close` = minutes remaining in the session
- `probability_up_session_close`, `expected_return_session_close`

The dashboard, Paper Trading and Live Test read these stored rows. They do not run a second on-demand model. The legacy `probability_up_30m` / `expected_return_30m` columns are read only as a fallback for archived rows written before the session-close migration.

## Legacy certified artifact: `target_30m`

The older certified deployment artifact is a binary Logistic Regression model with target `target_30m` and 137 named input features. It remains available for historical replay and audit, but it is **not** the live capture contract.

### Direction semantics

`target_30m = 1` means the future 30-minute return is **positive (UP)**.
`target_30m = 0` means the future 30-minute return is **non-positive (DOWN/FLAT)**.

The production artifact's classifier classes are `[0, 1]`, so `predict_proba(... )[:, 1]` is the probability of the UP class.

The dashboard therefore uses:

- BUY when `P(UP) >= 0.55`
- SELL when `P(UP) <= 0.45`
- HOLD otherwise

Historical replay uses the same causal definition for the actual outcome:

`future_close / decision_close - 1`

UP is positive, DOWN is negative, FLAT is zero.

A historical case where the model says BUY and the actual move is DOWN is an **incorrect model prediction**, not evidence that BUY/SELL labels have been inverted. The dashboard now exposes this contract explicitly so a label inversion cannot be silently introduced.

## Data-limited rule

The model probability can still be produced when optional information groups are missing because the certified pipeline contains an imputer. Historical and live screens must distinguish the model probability from overall information completeness. In the historical batch audit, rows with less than 70% information-group readiness are retained for traceability but excluded from the headline eligible directional-accuracy calculation.
