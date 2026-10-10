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

## Autonomous data controls (2026-10-10)

The Beta V2 weekday job runs at 07:15 IST, after the overnight Production PIT refresh retry window. Before scoring, it checks that the newest saved PIT filename matches the latest completed NSE session expected by the repository's session calendar. Missing or stale PIT data blocks inference and makes the Actions run fail visibly.

Each run updates and saves:
- `selfdriving/beta_version2/status_data_audit.json`: point-in-time PIT date, locked-feature coverage, lowest-coverage and full stockwise coverage, volume/Delivery features, scored-news and event availability, F&O coverage, global series availability, USD/INR availability and age, duplicate source rows, and source refresh failures.
- `selfdriving/beta_version2/BETA_V2_DELIVERY_REALISED_RETURNS.parquet`: next-session close-to-close outcomes reconstructed from consecutive official NSE PIT snapshots. A missing NSE session is skipped rather than mislabeled as a one-session result.
- `selfdriving/beta_version2/status_cycle.json`: pull, source-refresh, audit, inference, and validation results.

The first signal issued for each symbol/reference-date is retained during reruns. Data corrections can refresh the realised outcome table, but do not silently rewrite the already-issued prediction.

The Colab trainer reads the same Beta V2 global-cues and RSS-news archives updated by the daily job. Its source manifest fingerprints the base panel plus global, news and F&O inputs; when those sources change, stale enriched-panel caches are discarded before retraining. Use `--force-rebuild-panel` to request an explicit rebuild.

The audit separately records the **registered champion's actual feature list** and its per-feature/per-stock coverage. This distinguishes three states that must not be conflated: a source was downloaded, a transformed feature is present/populated, and the currently registered model actually selected that feature. In particular, `active_model_contract.usd_inr.selected_by_active_champion` states whether any recognized USD/INR input is among the champion's fitted features. The recognized family includes `g_inr_z` and the raw transformed `g_usdinr_r1`, `g_usdinr_r5`, and `g_usdinr_z1` features; `available_in_current_panel` reports data availability independently. If there is no registered champion, the audit explicitly reports `NO_CHAMPION` rather than claiming these inputs are used by a live Beta V2 model.

Global USD/INR, other global series and scored-news features are still subject to point-in-time availability and Beta V2 coverage/date gates. The RSS archive starts when the autonomous collector begins seeing headlines; missing historical headlines are not fabricated. Richer source presence alone does not mean the trained challenger uses every group. Inspect `status_data_audit.json` and the training manifest before interpreting a prediction. These changes do not alter or promote the locked production model.

## Validation status
The branch changes the architecture and wiring, but no claim of improved predictive accuracy is made until the
Delivery and Intraday challengers are actually trained on the available historical data and pass out-of-sample gates.
Production promotion is deliberately absent from this branch.
