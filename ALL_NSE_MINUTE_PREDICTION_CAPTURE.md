# ALL-NSE Five-Minute Prediction + Actual-Price Capture

## Drive layout

Each worker five-minute slot stores its observed quote snapshot at:

`production_data/all_universe_minute_snapshots/YYYY/MM/DD/snapshot_YYYYMMDD_HHMM.parquet`

When at least one symbol has a valid stored quote, its per-stock prediction rows are stored at:

`production_data/all_nse_minute_predictions/YYYY/MM/DD/predictions_YYYYMMDD_HHMM.parquet`

The prediction folder also receives `manifest_YYYYMMDD_HHMM.json` and the `latest_manifest.json` pointer for slots that run prediction. If no symbol has a valid quote, no prediction rows are generated; the source quote snapshot remains the record for that slot. Upload receipts are retained locally when Drive confirms writes.

## Per-stock eligibility

- Each five-minute slot is evaluated stock by stock.
- A stock is sent to prediction only if its quote exists in that slot's persisted snapshot and has a valid symbol, slot timestamp, quote status, and positive price.
- If a particular stock's quote is missing or invalid, that stock is skipped. Missing data for other stocks does not block forecasts for eligible stocks.
- Historical five-minute bars enrich the features when available. If stock history is absent, the model can use the stored current quote with missing OHLC fields left as null; lower feature readiness may keep the manual signal at `HOLD`.
- A usable model artifact is required for inference. Prediction and manifest files must be successfully written to Drive. These are actual execution/storage requirements, not universe-wide coverage thresholds.
- `STORED` means the relevant artifact was persisted. It does not mean that every NSE symbol produced a prediction. Quote coverage, missing-quote counts, and history-seed counts are informational metrics, not global pass/fail gates.
- If no valid stock quote exists in a slot, the predictor records `NO_ELIGIBLE_SYMBOLS` and generates no prediction rows for that slot.

## Row contents and safeguards

Prediction rows preserve the stock symbol, five-minute slot, actual observed price, quote timestamps, probability and expected return for the 30-minute horizon, raw/manual signals, feature-readiness score, model version, forecast target timestamp/key, and the row's inference status. Missing stocks do not receive fabricated predictions.

The quote timestamp is the collector-observation time, not an exchange-stamped tick timestamp. A live quote supplies the observed price, but does not establish current-slot open/high/low, so unavailable OHLC fields remain null.

The manual signal remains `HOLD` when feature readiness is below 70% or where the model manifest explicitly identifies the symbol as outside the training-symbol list. This system never submits or places orders.

## Useful manifest fields

`cycle_minute_ist`, `expected_symbol_count`, `snapshot_symbol_count`, `eligible_symbol_count`, `skipped_missing_quote_count`, `actual_quote_count`, `prediction_count`, `prediction_coverage_is_informational`, `data_limited_count`, `model_failure_count`, `history_seed_coverage_symbols`, `snapshot_drive_path`, `prediction_drive_path`, and `status`.
