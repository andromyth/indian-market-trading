# Calibration autopilot

A post-hoc calibration layer for autonomous intraday forecasts. It learns from matured forecast outcomes, proposes bounded calibration changes, evaluates them before use, and keeps a profile in shadow until fresh live evidence supports promotion.

It never edits the underlying model artifact and never places orders. The overlay defaults to shadow mode. Any loading or calibration error fails open to the raw forecast.

## Components

| File | Role |
|---|---|
| \`calibration_autopilot.py\` | Pure candidate fitting, walk-forward validation, return correction, advisory action band, drift checks, and profile lifecycle. |
| \`calibration_overlay.py\` | Worker-side overlay; loads active profile document and annotates or applies each forecast row. |
| \`prediction_learning_engine.py\` | Scores matured outcomes on the raw forecast, runs the autopilot, and persists registry and profile documents. |
| \`autonomous_intraday_worker.py\` | Calls the overlay after raw inference; failures do not block raw forecasts. |
| \`tests/test_calibration_autopilot.py\` | Regression tests covering fitting, lifecycle, overlay, persistence, engine integration, and worker integration. |

## Lifecycle

\`\`\`
no profile -> SHADOW -> ACTIVE -> DEMOTED or EXPIRED
                  \-> REJECTED
\`\`\`

1. **Fit and gate.** Candidate calibrators use expanding-window folds by whole IST session, with an embargo session. The raw forecast remains the incumbent. A candidate must improve held-out log loss, avoid Brier and calibration-error regressions, meet fold consistency, and stay within bounded parameter changes.
2. **SHADOW.** The worker writes calibrated fields next to raw fields. Primary forecast values are unchanged. At least 8 live sessions are required before promotion can be considered.
3. **ACTIVE.** Promotion requires paired live evidence: the session-block log-loss improvement interval must be above zero and Brier score must not be worse. The profile is applied to primary fields only when \`PREDICTION_CALIBRATION_MODE=active\`.
4. **Demotion.** An active profile is removed when live evidence shows harmful performance, after the configured minimum number of live sessions. A 5 day cooldown blocks immediate retries. Profiles expire after 35 days unless a validated refit refreshes them.
5. **Refit.** In-place refit applies only to ACTIVE profiles. After 7 days, a close fit may replace an active profile while retaining its promotion evidence and profile lineage. A distant candidate is recorded for review rather than installed. A SHADOW profile is never replaced during its trial.

The module also reports a robust return-head correction, an advisory BUY/SELL band, PSI and Page-Hinkley drift indicators, and a retraining recommendation when the raw probability model has no detectable discrimination or loses to a constant base-rate diagnostic.

## Modes

Set \`PREDICTION_CALIBRATION_MODE\` in the Cloud Run worker environment. Unset defaults to \`shadow\`.

| Mode | Effect |
|---|---|
| \`off\` | No loading or annotation. |
| \`shadow\` | Adds calibrated probability, expected return, signal, profile ID, and stage fields. Primary fields remain unchanged. |
| \`active\` | Applies ACTIVE profiles to primary fields only when the profile has a validated advisory band. Raw values remain in \`*_raw\` fields. SHADOW profiles are never applied. |

To roll back, set the mode to \`shadow\` or \`off\` and redeploy. Deleting \`active_calibration_profile.json\` disables new application when the worker refreshes; a worker retains the last known-good document for transient refresh failures, until profile expiry.

## Raw scoring contract

The outcome ledger keeps \`probability_up_raw\`, \`expected_return_raw\`, \`predicted_price_raw\`, and \`signal_raw\` when calibration is applied. Outcome scoring reconstructs those raw values, so a profile cannot be trained on its own adjusted output. The same live rows provide a paired raw-versus-calibrated comparison for promotion and demotion.

## Drive layout

Under \`production_data/intraday_predictions/learning/calibration/\`:

- \`calibration_registry.json\`: state, lifecycle events, and recent retired profiles.
- \`active_calibration_profile.json\`: compact profile document read by the worker.

The registry is not replaced when its existing content cannot be read or validated. A failed registry read yields \`STATE_LOAD_FAILED\`; failed remote persistence yields \`UPLOAD_FAILED\`. Local diagnostics are written under \`diagnostics/prediction_learning/calibration/\`.

## Evidence and assumptions

- The advisory action band uses an assumed round-trip cost of 12 basis points, not a measured broker-specific cost.
- Label is \`actual_return > 0\`; flat counts as not-up, matching the production contract.
- Thresholds, slope bounds, 8 and 20 shadow-session limits, 35 day expiry, fold win rate, and confidence intervals are engineering gates, not empirically optimized values.
- The expected earliest activation is roughly 23 valid sessions: 15 matured sessions to form a candidate and 8 new live sessions for shadow verification. More history is preferable; a 60 calendar day learning window is recommended over the default 30 day window.

## Not verified

Synthetic tests cannot establish real-market accuracy. Let the profile remain in shadow mode for several weeks, inspect prediction-versus-actual outcomes and transaction costs, and only then decide whether active mode is warranted. This calibration layer does not retrain the underlying multi-factor model and cannot repair a model that lacks directional skill.
