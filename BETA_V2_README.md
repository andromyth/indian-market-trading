# beta_version2 — self-driving prediction challenger

This branch is an isolated challenger implementation. It does not replace the
current production Delivery model, the current certified 30-minute Intraday
model, or the current 5-minute production model.

## Problems addressed

### Delivery
- Labels are derived as next observed-session stock return and market-neutral excess return.
- Global macro features are joined with explicit available_at timestamps.
- Archived news is scored for sentiment, intensity, novelty and impact instead of a single availability flag.
- Daily F&O regime features are joined only after their source trading session.
- Training uses purged walk-forward folds, isotonic calibration, Wilson lower-bound action thresholds and ABSTAIN.
- Champion/challenger promotion requires a valid paired comparison; a missing comparison cannot silently promote a candidate.
- Training, inference and realised-return panels are exported as persistent parquet artefacts.

### Intraday
- Beta V2 uses price/time features only; there are no minute-volume features.
- Future labels never cross the NSE regular-session boundary.
- Future return is measured relative to the cross-sectional median at the same decision timestamp and horizon.
- +5/+10/+15/+20/+25/+30 minute horizons are trained independently.
- Each horizon uses purged walk-forward evaluation, calibration, Wilson-bound thresholds and its own champion registry.

## Operational flow

1. COLAB_SD_V2_RETRAIN.py builds the Delivery Beta V2 panel and trains the Delivery challenger.
2. COLAB_SD_INTRADAY_V2_RETRAIN.py builds the volume-free Intraday challenger for all current NSE symbols available in the persisted intraday archive.
3. .github/workflows/autonomous_delivery_cycle_v2.yml runs the Beta V2 daily prediction/validation loop at 14:30 UTC on weekdays and can also be dispatched manually.
4. Streamlit production remains unchanged until a later, explicit promotion decision.

## Important limitation

The Beta V2 code fixes the architecture and wiring issues, but it does not claim improved live accuracy until the new challengers have actually been trained on the available historical data and pass their out-of-sample gates.
