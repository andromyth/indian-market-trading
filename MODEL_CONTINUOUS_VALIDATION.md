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

The live capture contract is `target_session_close`: class 1 = the price at the
same session's 15:30 IST close is above the decision price, class 0 otherwise.
The signal is refreshed every five minutes through 15:25 and is validated only
against the stored 15:30 target. The separately certified historical-replay
artifact remains `target_30m`; it is not the live capture contract. The
production model is never retrained inside Streamlit.

## Error handling

A prediction miss is recorded. It does not trigger immediate model replacement.
The health monitor can raise a `retraining_candidate` flag after enough completed
predictions. Any replacement still requires a separate full retraining,
walk-forward and validation process.
