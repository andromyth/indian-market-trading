# beta_version2 — self-driving prediction challenger

This branch is isolated from the current production paths. It does not replace
the current ALL-NSE Delivery model, the certified 30-minute Intraday model, or
the current 5-minute Intraday production model.

## Delivery fixes
- Causal next-observed-session return labels.
- Market-neutral excess-return target: stock return minus same-date cross-sectional median.
- Global macro features joined by explicit point-in-time availability timestamps.
- News sentiment, count, novelty and impact features from the forward-only archive.
- F&O regime features shifted to their next-session availability time.
- Corporate-event fields participate as a separate feature group when coverage is sufficient.
- Purged walk-forward validation with calibration and Wilson-bound action thresholds.
- ABSTAIN when there is no validated action band or insufficient feature coverage.
- A challenger cannot be promoted without a valid out-of-fold comparison against an existing champion.

## Intraday fixes
- New Beta V2 intraday contract uses price/time features only.
- No minute-volume feature is used, eliminating the current NSE quote-chart train/serve volume mismatch.
- No raw price-level feature is required for the new contract.
- Labels never cross the NSE regular-session boundary.
- Targets are market-neutral by decision timestamp and horizon.
- Independent +5/+10/+15/+20/+25/+30 minute models.
- Separate per-horizon registries.

## Data honesty controls
- Missing historical information remains unknown; it is not replaced with synthetic values.
- News features remain unknown before the real archive start. The group is only enabled after sufficient historical coverage.
- Global series are point-in-time joined; stale data becomes unavailable rather than silently carried forward beyond the configured age.
- FII/DII is not promoted into the model until the existing parser/endpoint is independently verified.

## Operational integration
- `COLAB_SD_V2_RETRAIN.py`: trains the isolated Delivery challenger.
- `COLAB_SD_INTRADAY_V2_RETRAIN.py`: trains the isolated volume-free Intraday challengers.
- `selfdriving/sd_v2_runtime.py`: optional runtime adapters for the beta champions.
- `selfdriving/sd_github_job_v2.py`: isolated daily prediction/validation cycle using Drive-persisted artefacts.
- `.github/workflows/autonomous_delivery_cycle_v2.yml`: weekday scheduled beta cycle plus manual dispatch.
- Existing candidate CI is configured to run on `beta_version2`.

## Validation status
The branch changes the architecture and wiring, but no claim of improved predictive accuracy is made until the
Delivery and Intraday challengers are actually trained on the available historical data and pass out-of-sample gates.
Production promotion is deliberately absent from this branch.
