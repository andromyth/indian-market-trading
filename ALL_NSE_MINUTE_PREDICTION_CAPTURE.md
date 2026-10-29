# ALL-NSE Five-Minute Snapshot Prediction + Actual-Price Capture

## Drive layout

On each five-minute NSE session anchor, the collector persists the full-universe quote snapshot:

`production_data/all_universe_minute_snapshots/YYYY/MM/DD/snapshot_YYYYMMDD_HHMM.parquet`

The prediction worker then writes one row per NSE symbol to:

`production_data/all_nse_minute_predictions/YYYY/MM/DD/predictions_YYYYMMDD_HHMM.parquet`

The prediction folder also receives `manifest_YYYYMMDD_HHMM.json` and `latest_manifest.json`. A matching `.uploaded.json` receipt is written locally when Drive confirms an upload.

## Five-minute-only live input

- The live prediction cadence is five minutes, matching the persisted full-universe quote snapshots.
- History is loaded from up to 50 prior persisted five-minute snapshot files. The current slot's valid quote is included in a temporary feature frame for inference and is appended to the in-memory history only after inference completes.
- The active live prediction path does not request per-stock `/market/historical/1minute` candles or use one-minute stock-history freshness as a prediction gate.
- Missing or invalid symbol rows remain explicit gaps on the five-minute time grid. Returns and volume features do not bridge a missing snapshot. Volume delta is accepted only when the immediately preceding five-minute row has a valid cumulative day-volume value.
- The sampled LTP is treated as a close-like observation, not a true exchange OHLC candle. The system never fabricates current-snapshot open/high/low values.

## Model and target contract

- Forecast target: 30 minutes ahead (`target_30m`).
- Live features use schema `nse_5m_snapshot_close_v1`, with a declared input cadence of five minutes. Low-readiness forecasts may be recorded for analysis, but their actionable manual signal remains `HOLD` when feature readiness is below 70% or the symbol is outside the training-symbol manifest.
- The training candidate uses the same five-minute endpoint alignment and feature schema. Historical one-minute OHLCV is aggregated to completed five-minute endpoints only as an offline training source; it is not a live inference dependency.
- The previous one-minute-trained artifact is intentionally rejected unless its manifest contains the exact five-minute schema version, cadence, feature list, and matching artifact hashes. Retrain and verify a compatible bundle before promoting or deploying this change.
- The worker writes a row even when quote/history/inference is unavailable. Failed fields remain null and the row status records why; missing values are never fabricated.
- The prediction manifest is `OK` only when coverage, model/schema integrity, feature readiness and persistence quality gates pass. Otherwise it remains `PARTIAL` or `PERSISTENCE_ERROR`. The minimum prediction-coverage gate remains 95%.

## Useful manifest fields

`expected_symbol_count`, `actual_quote_count`, `missing_quote_count`, `prediction_count`, `data_limited_count`, `untrained_symbol_count`, `model_failure_count`, `history_seed_coverage_symbols`, `history_seed_failed_batches`, `history_snapshot_source`, `snapshot_history_slots_loaded`, `snapshot_history_cadence_minutes`, `quality_gate_pass`, `status`, and `prediction_drive_path`.
