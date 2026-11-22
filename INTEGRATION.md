# Integration checklist: single to-close target (target_session_close)

Files go in the repo root: `features_5m.py`, `pretrain_pipeline.py`, `daily_cycle.py`, `champion_model_api.py`.
Nothing here changes a schedule, restarts a backfill, or touches Delivery. Everything below marked PATCH is an edit
to `all_nse_minute_prediction_capture.py` that you apply and review; I have not modified your repo.

## Execution: GitHub Actions -> Cloud Run Job (recommended)

Do NOT retrain in Colab. Colab has no reliable long runtime, and pasting an INDstocks token there can invalidate the
token the live Cloud Run worker uses. The supported path is already wired:

- `.github/workflows/retrain_to_close.yml` — trigger manually with `workflow_dispatch`.
  It builds the image, deploys the `to-close-retrain` Cloud Run Job (8 vCPU / 32 GiB / 24h task timeout), executes it,
  and waits for completion. A commented `schedule: "45 14 * * 1-5"` (20:15 IST, after the F&O sync) can be enabled
  after the first successful manual run.
- `retrain_to_close_job.py` — the Job entrypoint. It reads Drive via `GoogleDriveStore` (OAuth secret), downloads the
  selected native 5m symbols plus NIFTY context and F&O, audits the covered window, assembles the long training frame,
  runs `pretrain_pipeline.py`, and uploads a **candidate bundle** to `models/to_close/runs/<run_id>/` plus
  `models/to_close/latest.json`. It never touches the live champion.

Defaults: `--universe fno` (liquid F&O underlyings), `--max-symbols 300`, `--sessions 500`, `--cost 0.0011`,
`--warmup-bars 3`. The job auto-selects `--min-train-sessions`/`--holdout-sessions` from the audit: 150/60 for
500+ covered sessions and 80/40 for 200-499. **Below 200 it still continues** by operator request, using a smaller
window (50/30 for 120-199, 35/20 for 80-119, 20/12 for 45-79), and records `insufficient_history: true` plus a
`history_warning` in `retrain_status.json`/`latest.json`. Only below the technical floor of 45 covered sessions does
it stop with `DATA_NOT_READY` (a model cannot be fit on less). A result of `COMPLETED_FAIL` is a normal, uploaded
outcome; serving stays HOLD. The 6h GitHub wrapper timeout is only for the wait — the Cloud Run task can run up to 24h.

Local debug without Drive auth (uses a mounted/copied research root, no upload):
```
python retrain_to_close_job.py --drive-root /path/to/Indian_Market_Research \
  --work-root /tmp/to_close --out /tmp/to_close/out --no-upload \
  --universe fno --max-symbols 300 --sessions 500
```

## 0. Data readiness (run first)

INDstocks is **Cloud Run only**. Colab, Kaggle and local notebooks must not import
`indstocks_client`, set `INDSTOCKS_ACCESS_TOKEN`, or call the provider: a second login can
invalidate the token Cloud Run uses for live capture. The NIFTY 50 five-minute history and the
per-symbol native 5m archive are already produced by the Cloud Run backfill and written to Drive,
so the retrain job only reads those artifacts.

The retrain job itself performs the audit, assembly and training steps. This section is only for
**preparing the inputs in Drive** before the first retrain:

1. **Check what is in Drive** (read-only). Run on a machine with a mounted research root, or via
   any existing read-only audit workflow:
   ```bash
   python audit_retraining_inputs.py --drive-root /path/to/Indian_Market_Research \
     --json-out out/input_audit.json --md-out out/input_audit.md
   ```
   The verdict must be `READY` (or `READY_FOR_A_SMALLER_WINDOW`) for the window you intend to train.

2. **NIFTY 5m** — if it is missing or stale, do NOT fetch it locally. Trigger the private backfill:
   GitHub Actions -> "Start Cloud Run native five-minute history backfill"
   (`.github/workflows/start_cloud_run_5m_backfill.yml`, `confirm_backfill=yes`), then poll
   `.github/workflows/check_native_5m_backfill_status.yml`. It publishes
   `.../symbols/symbol=__NIFTY50__/nifty50_5m_merged_close_only.parquet`. Cloud Run pauses
   history during 09:15-15:59 IST on trading days.

3. **F&O multi-year file** — only needed if the 500-session window starts before 2026.
   Run `.github/workflows/nse_fno_sync.yml` with `rebuild_2026=true` for 2026, or extend it to
   build `fno_daily_features_2024_2026.parquet` via
   `nse_fno_udiff_ingest.py --rebuild-derived-start 2024-11-01 --rebuild-derived-end <today>`.
   The retrain job prefers the multi-year file and falls back to `fno_daily_features_2026.parquet`.

If the audit's largest ready window is below 200, the retrain job still runs with a smaller window and flags
`insufficient_history` instead of aborting; below 45 sessions it will not train. `fno_available_share` below ~0.90
means the F&O history is incomplete.

## 1. First model (GitHub Actions -> Cloud Run)

Trigger GitHub Actions -> "Retrain to-close champion (Cloud Run)" (`retrain_to_close.yml`) with the
defaults: `universe=all`, `max_symbols=500`, `sessions=500`, `cost=0.0011`, `warmup_bars=3`
(`universe=all` trains every backfilled symbol; `universe=fno` restricts to F&O underlyings).
It does the audit, assembly and training in the job and uploads the candidate to
`models/to_close/runs/<run_id>/`. Watch the run summary and "Show recent training logs"; then read
`models/to_close/latest.json` in Drive for the verdict and metrics.

For a direct local run (no Cloud Run, mounted/copied Drive input):
```
python retrain_to_close_job.py --drive-root /path/to/Indian_Market_Research \
  --work-root /tmp/to_close --out /tmp/to_close/out --no-upload \
  --universe fno --max-symbols 300 --sessions 500 --cost 0.0011 --warmup-bars 3
```
Or call the trainer directly if the long frame is already assembled:
```
python pretrain_pipeline.py --backfill-parquet out/bars_5m/parts \
  --market-parquet ".../olo_5m_archive/symbols/symbol=__NIFTY50__/nifty50_5m_merged_close_only.parquet" \
  --fno-parquet ".../fno/fno_daily_features_2024_2026.parquet" --out out \
  --cost 0.0011 --warmup-bars 3 --min-train-sessions 80 --holdout-sessions 40 --n-splits 4   # 200 sessions
  # 500 sessions: --min-train-sessions 150 --holdout-sessions 60
```
`--backfill-parquet` may point at a directory of Parquet parts. `--cost 0.0011` is the Zerodha
₹1-lakh round-trip assumption including 3 bps slippage; stress it at 0.0015.
Read `training_report.json` before anything else: `holdout_trading.se_bps` (can a real 2 to 5 bps edge be seen at all?),
`holdout_auc_by_time_left`, `open_stamped_sessions`, `source_agreement`, `coverage.fno_available_by_month`. A FAIL is a valid result.

## 1b. Promotion (candidate -> live champion)

The retrain job never moves the live champion. Run GitHub Actions ->
"Promote to-close champion (Drive)" (`promote_to_close_champion.yml`) after reviewing a run:

* default `dry_run=true` prints `PROMOTE_RESULT` and writes nothing;
* set `dry_run=false` to publish. A `FAIL` verdict returns `KEEP_PREVIOUS` (exit 3) and never
  touches `champion.json`; `--force` publishes a FAIL but keeps serving in HOLD;
* a `PASS` candidate is copied to `production_data/all_nse_champion/versions/<v>/bundle.joblib`,
  read back and SHA-256 checked, then `champion.json` is written last;
* serving leaves HOLD only after `activate_after=3` consecutive PASS nights (`active: true`), so
  the first two promotions publish a bundle that `champion_model_api` reports as HOLD.

```
python promote_to_close_candidate.py --dry-run          # inspect the newest candidate
python promote_to_close_candidate.py --run-id 20261122T011557Z
```
The nightly retrain runs at 20:15 IST (`45 14 * * 1-5`); set repo variable `RETRAIN_SCHEDULE=off`
to pause it without editing code.

## 2. Capture changes (PATCH)

1. **Use the champion API.** In `_ensure_model`, `import champion_model_api as model` instead of `all_nse_intraday_model`.
   Keep the constants/helpers (`HORIZON_NAME`, `TARGET_NAME`, `minutes_to_close`, `session_close_timestamp`) imported from
   `all_nse_intraday_model`. At startup call `champion_model_api.configure(daily_cycle.make_drive_store(), holidays=[...NSE holidays...])`.
2. **History length.** `HISTORY_LOOKBACK_BARS = 120` is too short: `prev_day_ret` mismatched training from about 14:05 onward in
   the parity test; 240 matched on every feature at 10:20, 12:35 and 15:25. Set it to 240 (confirm the provider returns that many
   5-minute candles), or drop `prev_day_ret` from `FEATURES_5M`.
3. **Closed candle only.** The adapter rejects any decision bar that is not a closed, bar-close-stamped candle on the 5-minute grid.
   The `FRESH_5M_HISTORY_PLUS_CURRENT_QUOTE` path appends a synthetic current-quote bar at an off-grid time; that row would be
   rejected (data-limited), and if it were accepted it would differ from training. Predict after the candle closes, from candles only.
4. **One calibration.** The champion's isotonic map is applied once by `_model_calibrate_probability` (the adapter's
   `calibrate_probability`). Set `ALL_NSE_CALIBRATION_MODE=off` so `calibration_overlay` does not calibrate a second time.
5. **Rebind per bar.** At the start of each decision bar, before the symbol batch:
   ```python
   if getattr(self._model_api, "champion_changed", lambda: False)():
       self._classifier = self._regressor = None
   self._ensure_model()
   ```
   `_ensure_model` caches forever today, so without this a swapped champion is never picked up and rows carry a stale `model_version`.
6. **Serving mode.** Write `model.serving_mode()` ("LIVE" / "PAPER" / "HOLD") into every prediction row. PAPER rows are signals for
   evaluation, not for money. HOLD mode already returns thresholds that cannot fire.
7. **Edge rule (only if trained with the edge rule, the default).** In `_signal`, also require the expected return to clear
   `model.signal_min_edge()` in the signal's direction. Otherwise train with `--no-edge-rule` so the validated rule equals the served rule.
8. **Log the served features** (`FEATURES_5M` values) with each prediction. `daily_cycle.feature_parity` compares them with a recomputation
   from stored bars. Column names `probability_up_session_close` and `expected_return_session_close` are already aliased.

## 3. Nightly job

`daily_cycle.run_nightly(bars_store, market, store, cfg, predictions, backfill, fno)` after the day's bars are final (about 16:15 IST,
timing to be measured on one real run first). It ingests, scores yesterday's predictions into the ledger, refits, gates, publishes
`versions/<v>/bundle.joblib` then `champion.json` last, and reports the serving policy. Use a no-cancel concurrency group and a timeout.
Scheduling is yours to approve. The first retrain is handled by `retrain_to_close.yml` / `retrain_to_close_job.py` above, which
downloads its own inputs from Drive (the Drive bridge IS written there). `daily_cycle.run_nightly` still expects a local bar store
plus `market` and `fno` frames passed in by the caller; wiring that nightly caller to the same Drive download bridge is a separate
step after the first candidate is validated.

## 4. Safety behaviour (tested)

- Missing, unreadable or hash-mismatched champion: `model_status()` reports unavailable, `_models()` raises, thresholds cannot fire.
- Staleness uses the wall clock (weekdays minus the holiday list), so a silently failed nightly job ages the model: more than 1 missed
  session means HOLD. Raise `max_lag_sessions` only deliberately.
- A new horizon of trust: gate PASS three nights in a row, then PAPER until 10 live sessions with trades, then LIVE only after 20 sessions
  with a positive one-sided 95% t-bound; negative live mean over the window halts to HOLD.

## 5. Drive notes

Change stamp is `modifiedTime` plus `size` (already returned by `list_folder`); SHA-256 gives integrity, so no `md5Checksum` field change.
`make_drive_store` was exercised against a mocked Drive only.

## 6. Tests to add in the repo

42-feature contract; per-session stamp guard (open-stamped input raises); strict vs NaN warm-up equality on shared rows; calibration
applied exactly once; champion swap picked up after rebind; stale, tampered and missing champion fail closed; F&O point-in-time
(day D uses day D-1) against your real parquet column names.

## 7. Still open

Real round-trip cost; warm-up A/B result on real data; real-data runtime of one nightly refit; the F&O dict the capture passes
must equal the previous-session daily row (parity audit will show if not).
