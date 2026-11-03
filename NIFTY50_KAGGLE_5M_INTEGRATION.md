# Kaggle NIFTY 50 five-minute market context

The Kaggle NIFTY 50 five-minute CSV is an approved **historical index-context source** for the ALL-NSE five-minute / 30-minute intraday model.

## What enters the model

Only NIFTY Close is enabled. The trainer derives five-minute market returns and RSI from that close series. The Kaggle Open, High, Low and Volume fields are not supplied to the feature builder: opening-bar OHLC showed larger disagreement in the comparison audit, and source volume is zero throughout.

The source uses bar-start timestamps. The importer converts the remaining complete-session records to the trainer's completed-bar convention by adding five minutes to each timestamp. It requires the expected 75-bar regular NSE session grid, drops incomplete/special-session days, and excludes the 09:15 opening bar.

The historical file ends on 18 May 2026. It is not a live feed. During current-session predictions, all_nse_minute_prediction_capture.py must continue to use a fresh current-session NIFTY feed and its existing five-minute freshness check. The Kaggle archive must never be used to bypass that check or fill a stale current-session gap.

## Import into Google Drive

Mount Google Drive in Colab first, then place the CSV somewhere accessible, for example /content/drive/MyDrive/NIFTY_50_5minute.csv.

Run from the repository root:

\`\`\`bash
python ingest_kaggle_nifty_5m_context.py \\
  --input /content/drive/MyDrive/NIFTY_50_5minute.csv \\
  --drive-root /content/drive/MyDrive/Indian_Market_Research
\`\`\`

The importer writes the close-only Parquet file and a provenance manifest to:

\`\`\`
production_data/nse_official/intraday_seed/olo_5m_archive/symbols/symbol=__NIFTY50__/
\`\`\`

## Rebuild and prediction safety

rebuild_all_nse_intraday_5m_model_from_drive.py downloads this special NIFTY partition separately from the equity-symbol partitions. The canonical trainer already reads it through load_market_5m() and derives NIFTY close returns/RSI. Equity stock partitions are unchanged.

The rebuild writes the model bundle to the canonical artifact path consumed by `all_nse_intraday_model.py`. Therefore, do not run an ad-hoc rebuild/upload outside the guarded deployment sequence.

Use the existing `.github/workflows/rebuild_intraday_5m_model_bundle.yml` workflow via **workflow_dispatch** after the data import. It rebuilds and validates the five-minute `target_30m` bundle, uploads the artifacts, and then requests `.github/workflows/deploy_intraday_worker_recovery.yml`. That workflow stages a new Cloud Run revision at zero traffic, validates `/modelz` (five-minute target, feature list, estimator hashes, and Drive readiness), and shifts production traffic only after those checks pass. The normal current-session NIFTY freshness guard remains active.

The data source is a public Kaggle dataset, not an official NSE feed. Its directional agreement with the sampled public secondary reference was strong, but that does not establish exchange-level ground truth. The Kaggle history is historical through 18 May 2026; it is not a substitute for current-session live NIFTY data.
