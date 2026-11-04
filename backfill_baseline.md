# Step 0 baseline — Indian Market Trading historical equity backfill

**Status: PASS**  
**Audit date:** 4 November 2026  
**Repository:** [andromyth/indian-market-trading](https://github.com/andromyth/indian-market-trading)  
**Branch:** `backfill-roadmap`  
**Branch HEAD before this final Step 0 record:** `8d3c8079d3a77f93cee8b3af7c38738fc558233e`

## 1. Repository and scope

- Read `BACKFILL_ROADMAP.md` directly from the requested branch.
- Inspected the existing five-minute INDstocks backfill, INDstocks read-only client, Drive storage implementation, Cloud Run worker service, NIFTY context documentation and production model manifest.
- Working environment clarification: the user confirms the project is maintained on GitHub and there is no local laptop checkout. Local `git status` is not applicable; GitHub remote branch HEAD is authoritative.
- Frozen scope: native five-minute NSE equity OHLCV; Kaggle equity history is primary only after source/data-integrity audit; INDstocks fills only verified missing coverage; NIFTY remains separate market-context data. No retraining or production model/feature/threshold/horizon changes.

## 2. Google Drive archive inventory (user-provided read-only Colab output)

**Archive root**

`/content/drive/MyDrive/Indian_Market_Research/production_data/nse_official/intraday_seed/olo_5m_archive`

| Measure | Observed value |
|---|---:|
| Parquet files | 2,821 |
| Recognized `symbol=` partitions | 188 |
| NIFTY context partition | 1 (`__NIFTY50__`) |
| Other symbol partitions | 187 |
| Total Parquet bytes | 122,786,077 |
| Latest Parquet modification | 2026-11-03 15:25:26 UTC |

These are archive inventory counts, not a completeness or session-quality audit. No complete-session coverage is inferred from file counts.

### Manifest finding — important exception

Repository helper `dhan_backfill_all_nse_5m_history.py` defines the expected equity archive manifest as:

`production_data/nse_official/intraday_seed/olo_5m_archive/backfill_manifest.json`

The recursive JSON inventory supplied from Colab did **not** find this archive-root file. It found only:

`symbols/symbol=__NIFTY50__/nifty50_kaggle_5m_close_only_manifest.json`

That is the separate NIFTY context manifest, not the historical equity backfill manifest. Accordingly, Step 0 records the equity backfill manifest as **NOT FOUND** and makes no manifest-based claim about per-symbol sessions or coverage. Carry this exception into Step 3/Step 6 for investigation; do not fabricate or reconstruct coverage from partition counts.

## 3. Production model integrity

The actual Drive model file was read and SHA-256 hashed.

- Model: Logistic Regression
- Feature count in repository manifest: 137
- Actual Drive file size: 9,617 bytes
- Manifest SHA-256: `c2d7067f5ea3daa006eaa0ecb0c5aca75f3084591712ea27b6486ef4873d4271`
- Actual Drive SHA-256: `c2d7067f5ea3daa006eaa0ecb0c5aca75f3084591712ea27b6486ef4873d4271`
- **Result: MATCH**

No production model artifact was changed.

## 4. Active backfill / live service check

- GitHub Actions status-filtered queries for `in_progress`, `queued` and `waiting`: zero runs.
- Cloud Run Jobs in `asia-south1`: zero listed.
- Cloud Run service `indian-market-intraday-worker`: Ready.
- Latest ready revision: `indian-market-intraday-worker-00075-gmb`; 100% production traffic.
- Four existing intraday Scheduler jobs are ENABLED. They are the regular live collection schedule and were left enabled.
- The supplied last-24-hour Cloud Run log query, filtered for backfill/historical/INDSTOCKS/progress/start/complete/pause terms, returned no matching lines.

**Conclusion:** no active GitHub Actions run, Cloud Run Job, or matching historical-backfill log was observed in the supplied checks. The live intraday service and schedules remain operational. The empty filtered result is recorded accurately; it is not described as proof that every unfiltered service log is empty.

## 5. Safety / no-change record

- No historical backfill was started.
- No market-data file was written or modified.
- No code or credential was changed as part of the inspection.
- No model training, model replacement, feature-contract change, threshold change or prediction-horizon change was made.
- Live intraday Cloud Run service and Scheduler jobs were not stopped or disabled.

## 6. Step 0 acceptance

| Acceptance item | Result |
|---|---|
| Repository, branch and commit verified | PASS |
| Relevant repository components inspected | PASS |
| Canonical archive root and inventory recorded | PASS |
| Active backfill surfaces checked | PASS |
| Production model actual checksum matches manifest | PASS |
| Baseline reports saved | PASS |
| Local working tree | Not applicable; no local checkout |

**Step 0 result: PASS.** The absent archive-root `backfill_manifest.json` is a recorded baseline exception, not a claim of data completeness. It must be investigated in the later archive-inventory/gap-audit steps before any historical backfill is considered.

## 7. Next allowed step

**Step 1 — Locate and inventory the original Kaggle equity archive.**

The original large archive on the user's device must be inspected. Do not substitute the previously uploaded approximately 11 MB partial/sample file. Do not start any historical backfill.
