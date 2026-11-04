# Step 0 baseline — Indian Market Trading historical equity backfill

**Status: BLOCKED (partial remote inspection completed; Drive/Cloud Run baseline not yet verifiable)**  
**Audit timestamp:** 2026-11-04 16:00 UTC  
**Repository:** [andromyth/indian-market-trading](https://github.com/andromyth/indian-market-trading)  
**Branch:** `backfill-roadmap`  
**Commit at inspection start:** [`542cf790d2472a4be1493114785b142019e37364`](https://github.com/andromyth/indian-market-trading/commit/542cf790d2472a4be1493114785b142019e37364)

## Completed and verified remotely

- Read `BACKFILL_ROADMAP.md` directly from `backfill-roadmap`. Initial roadmap blob SHA: `762e2b520d8d5a27537508f1e67cd99a3aebc59f`.
- Inspected the current branch versions of `indstocks_backfill_all_nse_5m_history.py`, `indstocks_client.py`, `persistent_store.py`, `autonomous_intraday_worker_service.py`, the production model manifest and `NIFTY50_KAGGLE_5M_INTEGRATION.md`.
- Confirmed scope: native 5-minute equity OHLCV; Kaggle is primary only after audit; INDstocks fills audited gaps only; NIFTY remains separate; no retraining or model/threshold/prediction-horizon changes.
- Queried the latest 100 GitHub Actions runs: none in that returned page were queued/in progress. No run was returned for `backfill-roadmap`. The latest visible native 5-minute backfill run was [run 41796785284](https://github.com/andromyth/indian-market-trading/actions/runs/41796785284), on `main`, completed with failure on 2026-11-04 08:05 UTC. This is not proof of Cloud Run inactivity.
- Captured repository manifest identifiers: `production_model/DEPLOYMENT_MODEL_MANIFEST_2026.json` blob SHA `6f3157890986b6d2651c3bcf5a7b27af45a9e5de`; declared model SHA-256 `c2d7067f5ea3daa006eaa0ecb0c5aca75f3084591712ea27b6486ef4873d4271`; model family Logistic Regression; 137 features. The actual Drive model file could not be re-hashed.
- No code, model, deployment, credential or market-data file was changed during inspection. No backfill or training was started.

## Not verified — blocks Step 0 PASS

This chat has no connected Google Drive or Google Cloud/Cloud Run tool. Therefore the current archive root, exact current manifest, symbol/file counts, byte counts, last-modified times, current per-symbol coverage metadata, Drive model checksum and Cloud Run active-job state remain **unverified**. The path below is the previously recorded expected location, not a fresh observation:

`Google Drive / Indian_Market_Research/production_data/nse_official/intraday_seed/olo_5m_archive`

An older 2026-10-31 report mentioned 187 readable symbol files and 6,397,281 valid OHLC rows. Those historical figures are not substituted for the current baseline. The user confirmed that no local checkout exists and all project files/work are maintained on GitHub; therefore local working-tree status is not applicable. The remote branch HEAD is the authoritative repository state.

## Required action to unblock

1. Open the existing authorized project runtime (Colab with the project Drive mounted, or the approved environment already used for this archive).
2. Run a **read-only** inventory of the archive and manifest, and verify the current production model SHA-256 against the manifest. Do not launch a backfill or rewrite any data.
3. Check Cloud Run/Cloud Scheduler and recent logs for an active historical backfill; do not start another worker.
4. Paste the output here. Do not paste credentials, tokens, OAuth JSON or secrets.

**Next allowed step:** stay on Step 0 until the missing read-only evidence is supplied and the baseline can be completed. Step 1 (Kaggle archive inventory) is not authorized yet.
