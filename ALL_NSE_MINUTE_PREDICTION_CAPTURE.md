# ALL-NSE Five-Minute Prediction + Actual-Price Capture

## Acquisition → assembly → prediction

The five-minute cycle is a single coordinated pipeline:

1. **Sharded acquisition.** One coordinated Cloud Run tick fans the official NSE
   universe out to `ALL_NSE_CAPTURE_WORKERS` shard collectors (default 8). Every
   shard owns a disjoint slice (`symbols[worker_id::worker_count]`) and all
   shards share the process-wide INDstocks request gate. Shards are worker
   threads inside the one acquisition process; they are not independent
   deployed instances, and no local file lock is used as distributed
   coordination.
2. **Shard persistence.** Each shard writes its own snapshot and manifest:
   `production_data/all_universe_minute_snapshots/YYYY/MM/DD/shards/snapshot_YYYYMMDD_HHMM_wNNofMM.parquet`
   and `.../shards/manifest_YYYYMMDD_HHMM_wNNofMM.json`.
3. **Explicit assembly + validation.** `all_universe_snapshot_assembly.py`
   requires every shard for the slot, then validates shard identity, slot
   timestamps, duplicate symbols across shards, missing symbols inside a shard,
   and declared-vs-actual row counts. If any shard is missing or invalid, no
   canonical snapshot is published; an `ASSEMBLY_INCOMPLETE` manifest is written
   instead.
4. **One canonical snapshot.** Only a validated assembly is published at
   `production_data/all_universe_minute_snapshots/YYYY/MM/DD/snapshot_YYYYMMDD_HHMM.parquet`
   with `manifest_YYYYMMDD_HHMM.json` and the `latest_manifest.json` pointer.
5. **One prediction publisher.** The GitHub Actions compiler
   (`drive_all_nse_prediction_compiler.py`) reads only the canonical assembled
   snapshot and keeps its existing parallel inference + single-writer design.
   It skips shard artifacts and any manifest whose assembly is not `COMPLETE`.

Assembly is idempotent: a retry rewrites the same deterministic shard and
canonical paths for the same slot. It can also be re-run independently, which
only reads Drive:

```
python all_universe_snapshot_assembly.py --minute 2026-11-12T10:05 --worker-count 8
```

## Scheduling robustness

- Cloud Scheduler fires every five minutes from 09:15 through 15:25 IST.
- The worker maps a tick to its five-minute anchor and applies
  `ALL_NSE_CAPTURE_GRACE_MINUTES` (default 3) of lateness, so a tick delayed by
  a slow prior slot still fills the correct anchor instead of silently skipping
  the whole five-minute bucket.
- The Cloud Run request timeout is 900s and the Scheduler attempt deadline is
  540s, so a single slow tick can no longer be killed at the five-minute
  boundary and cascade into permanently missing slots.
- Cloud Run stays at concurrency 1 and max instances 1: exactly one process owns
  the INDstocks token and the assembly write.

## Same-slot prediction inside the Cloud Run tick

- `ALL_NSE_IN_WORKER_PREDICTION=1` makes the tick that just persisted the
  snapshot run `AllNSEMinutePredictor.run_for_capture` immediately, using the
  already-warm model/history cache. The slot then lands on Drive roughly one
  minute after the five-minute boundary instead of the GitHub Actions lane's
  ~3.5 minutes (runner setup + pip + pytest + Drive auth).
- The in-worker run uses `drive_only=False`, so the NIFTY benchmark refresh is
  allowed and market/relative features are not dropped for being 10 minutes
  old, unlike a Drive-only compile that must stay inside
  `MAX_HISTORY_CANDLE_AGE_MINUTES = 5`.
- The call is bounded by `ALL_NSE_IN_WORKER_PREDICTION_DEADLINE_SECONDS`
  (default 180). On timeout or error the tick returns
  `DELEGATED_TO_GITHUB_ACTIONS`, so the compiler and post-close recovery
  workflows still own the slot; the compiler skips slots that are already
  `STORED`.
- `prediction_mode` and `prediction_latency_seconds` are recorded on the tick
  result (and surfaced through `/healthz`), so the slot -> prediction latency
  is measurable from logs and the session-health record.

## Drive layout

Canonical quote snapshot (the only input the prediction compiler accepts):

`production_data/all_universe_minute_snapshots/YYYY/MM/DD/snapshot_YYYYMMDD_HHMM.parquet`

Shard artifacts for the same slot:

`production_data/all_universe_minute_snapshots/YYYY/MM/DD/shards/snapshot_YYYYMMDD_HHMM_wNNofMM.parquet`

Per-slot shard manifests:

`production_data/all_universe_minute_snapshots/YYYY/MM/DD/shards/manifest_YYYYMMDD_HHMM_wNNofMM.json`

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

Canonical (assembled) snapshot manifest: `cycle_minute_ist`, `status`,
`assembly_status`, `snapshot_persisted`, `worker_count`, `shard_count_expected`,
`shard_count_completed`, `shards`, `expected_symbol_count`, `row_count`,
`quoted_symbol_count`, `missing_quote_count`, `validation`,
`snapshot_drive_path`.

Shard manifest: `worker_id`, `worker_count`, `shard_symbols`,
`shard_symbol_count`, `universe_symbol_count`, `row_count`, `status`
(`SHARD_STORED`), `snapshot_drive_path`.

Per-stock prediction manifest: `cycle_minute_ist`, `expected_symbol_count`,
`snapshot_symbol_count`, `eligible_symbol_count`, `skipped_missing_quote_count`,
`actual_quote_count`, `prediction_count`,
`prediction_coverage_is_informational`, `data_limited_count`,
`model_failure_count`, `history_seed_coverage_symbols`, `snapshot_drive_path`,
`prediction_drive_path`, and `status`.
