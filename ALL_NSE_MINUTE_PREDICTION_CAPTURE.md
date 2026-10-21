# ALL-NSE Minute Prediction + Actual-Price Capture

## Drive layout

Every worker minute writes a Parquet file containing one row per symbol in the current official NSE equity universe:

`production_data/all_nse_minute_predictions/YYYY/MM/DD/predictions_YYYYMMDD_HHMM.parquet`

The same folder receives `manifest_YYYYMMDD_HHMM.json` and the current pointer `latest_manifest.json`. A matching `.uploaded.json` receipt is written locally when Drive confirms an upload.

## Row contents

Rows record the NSE symbol, snapshot minute, actual observed market value (INDstocks live quote), quote-observation timestamps, quote quality, pooled-model 30-minute probability and expected return, raw model signal, conservative manual signal, feature-readiness score, training-coverage indicator, forecast target timestamp/key, and an explicit inference or persistence status.

A row is still written when price/history/inference is unavailable. In that case its status explains what failed and the corresponding prediction or actual-price fields remain null. The capture must not silently invent values.

## Interpretation and safeguards

- Prediction horizon: 30 minutes (`target_30m`), recomputed once per minute for all symbols with usable inputs.
- Actual price: the live quote observed by the collector at the stored timestamp. The timestamp is the collector-observation time, not an exchange-stamped tick timestamp.
- Minute OHLC: the batch quote only supplies a current LTP and day-level OHLC. The pipeline does not mislabel day-level open/high/low as the current minute's OHLC; unknown current-minute OHLC fields remain null.
- Actionable manual signal is `HOLD` when feature readiness is below 70%, or where the model manifest explicitly shows the symbol was outside the training-symbol list. The raw model signal and probability remain available for audit. This module does not place orders.
- `OK` requires full quote and prediction coverage, adequate feature readiness, no untrained symbols or model errors, full historical seed coverage, and confirmed Drive persistence. Otherwise the minute manifest is `PARTIAL` or `PERSISTENCE_ERROR`.
- The service keeps one Cloud Run instance and CPU available between scheduled minute requests to preserve its pooled-model and historical-window cache. This improves continuity but incurs ongoing Cloud Run cost.

## Useful manifest fields

`expected_symbol_count`, `actual_quote_count`, `missing_quote_count`, `prediction_count`, `data_limited_count`, `untrained_symbol_count`, `model_failure_count`, `history_seed_coverage_symbols`, `history_seed_failed_batches`, `quality_gate_pass`, `status`, and `prediction_drive_path`.
