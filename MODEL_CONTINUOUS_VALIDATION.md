# Continuous Prediction Validation

The application now treats every production prediction as a persistent record.

## Delivery

A daily `next_session` prediction is stored with its model version, probability,
predicted return and signal. Once the next trading-session close is available,
the validator compares the predicted price/return with the actual D+1 close.

The dashboard also records the subsequent D+2 through D+7 price path as monitoring
observations. These later observations are **not** incorrectly counted as
additional `next_session` model targets.

A rolling seven-completed-prediction scorecard reports directional accuracy and
price-error statistics. This is the continuous seven-day monitoring requirement:
each day contributes a newly validated prediction as the next actual session
arrives.

## Intraday

The existing production contract remains unchanged: `target_30m`, class 1 = UP,
class 0 = DOWN/FLAT, and BUY/SELL/HOLD are derived from `P(UP)` using the certified
thresholds. The production model is never retrained inside Streamlit.

## Error handling

A prediction miss is recorded. It does not trigger immediate model replacement.
The health monitor can raise a `retraining_candidate` flag after enough completed
predictions. Any replacement still requires a separate full retraining,
walk-forward and validation process.
