# Indian Market Trading — Historical Equity Backfill Roadmap

**Purpose:** Complete and independently verify the project's historical **five-minute NSE equity OHLCV** archive, using validated Kaggle equity history as the primary historical source and INDstocks only to fill missing coverage. Keep the NIFTY 50 CSV as a separate market-context dataset.

**Execution mode:** Run exactly one numbered step per separate development-agent chat. Do not skip ahead. Each step has its own objective, constraints, deliverables, acceptance checks and copy/paste prompt.

**Current status:** ROADMAP PREPARED — execution has not been performed by this document.

---

## 0. Non-negotiable scope and decisions

These decisions are the project contract for this backfill. A development agent must not reinterpret or expand them without explicit approval.

1. **Backfill target is native five-minute equity OHLCV.** Do not introduce a one-minute historical backfill in this workstream. Do not change prediction horizons or model targets.
2. **Kaggle equity history is the primary historical source only after it passes source and data-integrity audits.** The original archive must be inspected, not inferred from a small sample or a partial upload.
3. **INDstocks is the gap-filling provider.** Fetch only the missing coverage established by the gap audit. Do not re-download complete intervals unnecessarily.
4. **NIFTY 50 five-minute CSV is separate.** It is an index/market-context input, not an equity symbol archive and must never be merged into stock OHLCV files.
5. **Google Drive remains the durable storage layer.** Keep source archives, normalized source files, per-symbol Parquet files, manifests, checkpoints and audit reports there.
6. **Preserve source provenance.** Never silently mix Kaggle, INDstocks, Dhan, NSE or Yahoo rows without retaining source identity and a documented conflict policy. Preserve original source files unchanged.
7. **The scope is data completion only.** Do not retrain, replace, flip, recalibrate or otherwise modify production models, feature definitions, thresholds or prediction policies. No XGBoost/LightGBM, Triple-Barrier, Polars rewrite, Redis, Compute Engine migration or new async architecture is part of this roadmap.
8. **Keep existing working infrastructure unless a verified defect requires a small targeted correction.** In particular, preserve the existing INDstocks client, Cloud Run authentication ownership, Drive integration and resumable backfill mechanism where possible.
9. **Respect provider limits.** The repository's INDstocks documentation describes a maximum of five instruments and seven calendar days per historical request, with a five-data-requests-per-second ceiling. The existing backfill uses five-symbol batches and a conservative one-request-per-second pause. Do not increase request rate or concurrency without verified provider documentation and a tested rate limiter.
10. **Do not backfill during regular NSE cash-market hours.** Historical requests must yield to live capture from 09:15 to 15:30 IST on NSE trading days. Use the exchange calendar/session guard, not weekday-only assumptions.
11. **Completion means verified data, not a successful process exit.** A run is not complete until the Drive manifest and independent audits meet the acceptance criteria in Step 11.
12. **Do not fabricate candles.** Missing OHLCV remains missing and is reported. No interpolation, forward-fill, proxy ticker, synthetic candle or invented volume is permitted.

### Target acceptance contract

- Target coverage: **200 to 500 complete usable sessions per eligible symbol**; retain no more than the latest 500 usable sessions in the canonical research archive unless a separately approved retention rule says otherwise.
- At least **90% of the explicitly defined eligible target universe** must meet the minimum 200-session requirement.
- A session is complete only when all bars expected for that specific NSE session are present and valid. A normal full regular NSE session generally has 75 five-minute bars; special/short sessions must use the correct exchange session grid rather than being forced to 75.
- Every accepted bar has a valid timestamp, positive OHLC, non-negative volume where volume is supplied, valid OHLC relationships, and a documented source.
- Duplicate timestamps are resolved deterministically and reported. Conflicting source values are reported, not silently hidden.
- The final report separately counts: eligible symbols, ineligible/newly listed symbols, unmapped symbols, symbols with 200+ complete sessions, symbols with 500 sessions, gaps, stale latest data, source conflicts, and failed downloads.
- The 90% denominator must be frozen and explained. Do not silently remove hard-to-map or poor-coverage symbols to make the percentage pass. Symbols that genuinely could not yet have 200 sessions because of listing date must be reported separately with evidence.

> **Important:** The repository's existing INDstocks backfill currently uses a minimum-bars-per-session threshold of 50 for its `usable_session_summary`. That is not, by itself, proof that a session has every expected five-minute candle. The gap audit must define and verify complete sessions against the correct session grid before a symbol is counted as meeting this roadmap's complete-session target.

---

## 1. Existing repository components to preserve and inspect

Repository: `https://github.com/andromyth/indian-market-trading`

| Component | Existing role | Roadmap treatment |
|---|---|---|
| `indstocks_backfill_all_nse_5m_history.py` | INDstocks five-minute historical backfill; batches instruments, walks seven-day windows, checkpoints windows in Drive and writes `backfill_manifest.json` | Reuse and extend only where the gap audit identifies a verified deficiency |
| `dhan_backfill_all_nse_5m_history.py` | Shared archive constants/helpers and Dhan historical backfill implementation | Inspect shared helpers before changing; do not accidentally alter Dhan behaviour |
| `indstocks_client.py` | INDstocks authentication and read-only API client | Preserve existing Cloud Run TOTP ownership; do not create a second TOTP owner |
| `persistent_store.py` | Google Drive storage/read-write integration | Reuse for durable files and checkpoint commits |
| `autonomous_intraday_worker_service.py` | Private Cloud Run worker service; shares process-local INDstocks token cache and pauses historical backfill during live market hours | Keep live collection priority and existing authentication model |
| `autonomous_intraday_worker.py` | Live intraday collection/worker logic | Do not rewrite for this historical-only project |
| `data_fetcher.py` | Data-source selection for dashboard/research requests | Do not change unless a verified source-selection defect blocks the backfill |
| `ingest_kaggle_nifty_5m_context.py` | NIFTY CSV market-context importer | Keep separate from equity backfill |
| `NIFTY50_KAGGLE_5M_INTEGRATION.md` | Defines NIFTY-only context inputs and timestamp rules | Treat as separate and already scoped context work |

**Known existing INDstocks backfill behaviour to verify before modifying:** maximum five scrip codes and seven calendar days per request; pauses between requests; normalizes candle-start timestamps to completed-bar timestamps; reads existing Drive symbol files and a separate `source=KAGGLE` file; writes window Parquet checkpoints and a JSON `COMPLETE` marker; writes a final Drive `backfill_manifest.json`; uses a 200-session minimum, maximum 500 sessions and a 90% symbol-coverage gate by default. Confirm against the current branch before relying on these details.

**Known prior audit result (historical, not current):** an earlier archive audit reported 187 readable symbol files, 175 symbols with at least 200 usable sessions, 6,397,281 valid OHLC rows and 93.58% coverage against that audit's denominator. These figures are a baseline reference only; they are not proof of current coverage or full candle completeness.

---

## 2. How to run this roadmap using ChatGPT and GitHub across separate chats

This section is the operating procedure. Follow it for every numbered step. The GitHub copy of `BACKFILL_ROADMAP.md` is the shared source of truth between chats; do not rely on an old downloaded copy or on ChatGPT remembering another conversation.

### 2.1 One-time GitHub setup

**Repository:** `https://github.com/andromyth/indian-market-trading`

1. Download this roadmap and add it to the repository root with the exact filename `BACKFILL_ROADMAP.md`.
2. Create one dedicated working branch named `backfill-roadmap` from the latest `main` branch. Do not do this work directly on `main`.
3. Commit the roadmap to that branch. Keep the same branch for the whole roadmap so each separate chat sees the previous step's code, reports and execution-ledger updates.
4. Do not upload the original Kaggle archive, large Parquet datasets, OAuth files, API tokens, MPIN, TOTP secret or `.env` files to GitHub. Large/source data stays in the agreed device or Google Drive locations; secrets stay in the approved secret store.

**Using the GitHub website:** open the repository → select the branch dropdown → create `backfill-roadmap` from `main` → use **Add file → Upload files** → upload this file as `BACKFILL_ROADMAP.md` at the repository root → commit to `backfill-roadmap`.

**Using a local Git checkout:** run these commands from the folder where you want the repository cloned. If you already have a checkout, enter that folder and use the update/branch commands rather than cloning again.

```bash
git clone https://github.com/andromyth/indian-market-trading.git
cd indian-market-trading
git switch main
git pull --ff-only origin main
git switch -c backfill-roadmap
# Copy the downloaded roadmap into this repository root as BACKFILL_ROADMAP.md.
git add BACKFILL_ROADMAP.md
git commit -m "docs: add historical backfill roadmap"
git push -u origin backfill-roadmap
```

If the branch already exists, use `git switch backfill-roadmap` and `git pull --ff-only origin backfill-roadmap`; do not create another branch with the same purpose. If Git reports uncommitted changes, stop and inspect them. Do not discard or overwrite unrelated work.

### 2.2 Start a fresh ChatGPT chat for each step

Use one separate ChatGPT conversation for each step (Step 0 gets its own chat, Step 1 gets another, and so on).

At the beginning of **every** new chat:

1. Open the current `backfill-roadmap` branch in GitHub and confirm its latest commit.
2. In ChatGPT, use the connected GitHub repository/integration if it is available, and identify the repository explicitly as `andromyth/indian-market-trading`, branch `backfill-roadmap`.
3. Make the current `BACKFILL_ROADMAP.md` available to the chat. Prefer asking ChatGPT to fetch the file from the current GitHub branch. If GitHub access is not connected in that chat, upload the latest copy downloaded from the branch. **Do not upload a stale copy from an earlier step.**
4. Paste the checkpoint/handoff from the immediately preceding step. For Step 0, write `No previous step; begin with Step 0`.
5. Paste the matching **Chat prompt** from the numbered step in Section 3. State clearly: `Execute Step N only`.

Do not paste credentials into ChatGPT. Do not paste a full OAuth JSON, refresh token, INDstocks access token, MPIN or TOTP secret. Configure credentials only through the approved GitHub Actions secrets, Google Cloud Secret Manager, or the existing local/Colab secure environment.

### 2.3 Copy this opening message into each new ChatGPT chat

Replace `[N]` with the step number and paste the previous step's checkpoint below it. Then paste the full step-specific prompt from Section 3.

```text
I am continuing the Indian Market Trading historical five-minute equity backfill roadmap in a NEW chat.

Repository: https://github.com/andromyth/indian-market-trading
Working branch: backfill-roadmap
Roadmap file: BACKFILL_ROADMAP.md
Current task: Execute Step [N] only.

First, read the current BACKFILL_ROADMAP.md from the backfill-roadmap branch. Verify the branch HEAD and inspect the actual repository files and the previous step's evidence. Do not rely only on my handoff text. Confirm that all prerequisite steps passed before doing any work.

Stay strictly within this step's scope. Preserve the existing INDstocks client, Google Drive archive, Cloud Run live collection, source provenance and production model artifacts. Do not retrain or modify any production model, target, feature contract, signal threshold or prediction horizon. Do not start a backfill workflow unless this step explicitly authorizes it. Do not run historical requests during NSE cash-market hours.

If this step requires access to my original device-only Kaggle archive, Google Drive, Colab, GitHub Actions or Google Cloud and you do not actually have that access, say exactly what I must open/run and give me the exact safe command or UI path. Do not claim to have inspected, executed or verified something you could not access. I will run it and provide the output in this chat.

Before editing, report the current branch, HEAD commit, working-tree status and files you intend to change. Make the smallest necessary change. Preserve unrelated user changes. Run the step's tests and acceptance checks. Do not mark the step PASS merely because code was written or a workflow started.

Only if every acceptance criterion passes, update the Execution ledger in BACKFILL_ROADMAP.md on the backfill-roadmap branch and commit the step's deliverables. Do not commit raw datasets, large Parquet files, credentials or secrets. If any criterion fails, leave the status FAILED, BLOCKED or NEEDS ATTENTION, explain the root cause and stop without starting the next step.

At the end, provide the exact end-of-chat handoff format from Section 5, including commit SHA, artifact paths, tests, evidence, unresolved issues and the next allowed step. I will carry that handoff into the next separate chat.
```

### 2.4 What ChatGPT can and cannot execute

Use the correct execution location for each action. A GitHub connection does not automatically grant access to files stored only on your computer, Google Drive, Colab runtime or Google Cloud project.

| Work required | Where to do it | What to bring back into the chat |
|---|---|---|
| Read/inspect repository source code | ChatGPT with the GitHub repository connected, or a local Git checkout | File paths, commit SHA and relevant command output |
| Edit Python/scripts/tests/Markdown | GitHub-connected coding agent or local development environment | Commit SHA, diff summary and test output |
| Inspect the original Kaggle archive on your device | Your computer's file manager/terminal, or a runtime where the full original archive is mounted | Archive path, byte size, SHA-256, member listing and audit report; do not substitute a partial upload |
| Inspect or read/write historical files in Google Drive | Colab or the already-authorized project runtime | Exact Drive paths, manifests, counts, checksums and logs |
| Run model/data audit code against Drive | Colab | Full cell output and saved report paths |
| Run scheduled/manual repository workflows | GitHub → **Actions** | Workflow run URL, logs, artifacts and resulting Drive manifest |
| Inspect or operate Cloud Run | Google Cloud Console / `gcloud` in the authorized environment | Service/revision, logs, status and Drive write evidence |

If a normal ChatGPT chat cannot access a required environment, it must prepare the command or code for you to run; it must not claim that the operation has been completed. A green GitHub Actions result alone is not proof of a complete Drive write or successful coverage audit.

### 2.5 GitHub branch and commit rules for every step

- Keep the work on the single `backfill-roadmap` branch. Do not commit directly to `main` during execution.
- At the beginning of a step, inspect the current branch, HEAD and working tree. Fetch/pull the latest branch before editing. If another chat has added commits, continue from the latest HEAD.
- Make one small, reviewable commit per completed step. Use a message such as `backfill: complete step 03 drive inventory`.
- Commit only code, tests, small audit reports and the updated roadmap ledger that belong to the current step. Put large datasets and bulky manifests in Google Drive; record their exact Drive paths and checksums in the report/ledger.
- Do not overwrite or delete source archives. Do not force-push. Do not resolve conflicts by discarding another chat's work.
- After committing, verify that the commit is visible on GitHub and that the roadmap ledger reflects the verified state. Carry the new commit SHA into the next chat.
- After Step 12 passes, review the complete branch and open a pull request from `backfill-roadmap` to `main`. Merge only after checking the full diff and final completion report.

### 2.6 When to use GitHub Actions, Cloud Run and Colab

- **Steps 0–6:** inspection, source verification, data inventory, staging design and read-only gap detection. Do not trigger a full backfill.
- **Step 7:** implement and test checkpoint/resume behaviour. Test with temporary/local fixtures first. Do not run the full-universe job.
- **Step 8:** run only the explicitly approved tiny pilot, using the execution route verified in Step 7. Check the resulting Parquet files and Drive manifest, not just the job exit code.
- **Step 9:** run the full backfill only after Step 8 passes, with the frozen eligible universe and an approved rate/market-hours guard. Use the existing supported GitHub Actions or Cloud Run entry point verified in Step 7; do not invent a new endpoint or start a second concurrent worker.
- **Steps 10–12:** reconcile source outputs, independently audit coverage and freeze the validated dataset. Colab may be used for read-only validation; model training remains prohibited under this roadmap.

Before using **GitHub → Actions → Run workflow**, verify the workflow name and inputs from the current branch. Do not guess the workflow filename, input names or provider mode. For any INDstocks operation, use the authentication method approved in Step 7; do not generate a second TOTP token or change Cloud Run's token ownership. Check the current NSE session/holiday guard and active jobs before starting.

### 2.7 End each chat with a verified handoff

1. Ask ChatGPT to print the end-of-chat report format in Section 5.
2. Check the commit SHA and open the changed files/commit on GitHub.
3. Check that every report path exists in the stated environment (GitHub, Drive, Colab or local machine).
4. Confirm that a step marked PASS meets every acceptance item in Section 3. A script created, workflow queued, API request sent or partial file uploaded is not a PASS.
5. Copy the handoff text into the next chat and start the next numbered step only when the previous step is PASS.
6. If the next chat cannot see the prior chat's files, upload the latest roadmap and the required report artifacts. Do not assume separate chats share attachments or memory.

### Mandatory rule for every step

**One chat = one numbered step = one verified handoff.** The next chat must independently verify the previous step's evidence before proceeding. If a prerequisite fails or access is missing, stop at that step and report exactly what is needed; never skip ahead to keep the sequence moving.

---

# 3. Step-by-step execution plan

## Step 0 — Freeze the scope and record the baseline

**Execution record (2026-11-04): BLOCKED.** Remote repository inspection completed and recorded in `backfill_baseline.json` and `backfill_baseline.md`. GitHub Actions query showed no queued/in-progress run in the latest 100 returned runs; latest visible native 5-minute backfill run 41796785284 on `main` had completed with failure. This does not establish Cloud Run inactivity. Google Drive and Google Cloud/Cloud Run are not connected to this chat, so the current archive inventory/manifest, live job state, and actual Drive model checksum could not be verified. Do not mark Step 0 PASS or begin Step 1 until those read-only checks are supplied and the baseline is completed. No backfill or training was started.

**Objective:** Establish exactly which repository version and Drive archive are being worked on, and protect the live prediction system from incidental changes.

**Actions:**
- Verify repository, current branch, HEAD commit and working-tree status.
- Read the current backfill scripts, INDstocks setup, Drive storage implementation, Cloud Run deployment and NIFTY context documentation.
- Locate the current canonical five-minute archive root and current backfill manifest in Drive.
- Record current production model manifest/version and confirm it will not be touched.
- Record whether a backfill is already running. Do not start a second concurrent run.
- Capture a read-only baseline of file counts, symbol counts, byte counts, manifest status and last update times. Do not rewrite data.

**Acceptance:** Repository/commit and archive paths are known; active jobs are identified; production model artifacts are confirmed untouched; baseline report is saved.

**Deliverables:** `backfill_baseline.json` and a short `backfill_baseline.md` report. No market-data files changed.

**Chat prompt:**

```text
Execute ROADMAP Step 0 only. Read BACKFILL_ROADMAP.md. Inspect the current repository and Google Drive backfill archive without changing any market data, source code, credentials, deployment, or model artifacts. Verify the current branch/commit/working tree, identify the canonical five-minute archive and manifest paths, detect any active backfill job, record per-symbol file counts/coverage metadata available from existing manifests, and record the active production model version for a no-change guard. Produce backfill_baseline.json and backfill_baseline.md with exact evidence and paths. Do not start a backfill. Do not infer missing data from file counts alone. Pass only if the baseline is reproducible and no production artifact or data file was modified.
```

**Checkpoint to carry forward:** repository URL, branch, commit SHA, working-tree state, active-job status, Drive archive root, manifest path, baseline report paths, production model version/hash.

---

## Step 1 — Locate and inventory the original Kaggle equity archive

**Objective:** Inspect the actual original archive, not a partial or reduced upload.

**Actions:**
- Locate the original Kaggle archive supplied for the project. The previously mentioned large original archive on the user's device must be distinguished from any small sample/partial upload (including the previously uploaded approximately 11 MB file, if that is not the original).
- Record file name, exact byte size, SHA-256, archive members, formats, compression and folder structure.
- Identify whether the archive contains equity data, index data, metadata, documentation or unrelated files.
- Do not extract the entire archive into memory. Inventory first and use bounded samples for structural inspection.
- Do not assume CSV/Parquet is OHLCV until verified.

**Acceptance:** Original file identity and checksum are recorded; contents and formats are inventoried; the audit clearly states whether the original archive is accessible and whether the small uploaded file is complete or merely a sample.

**Deliverables:** `kaggle_archive_inventory.json` and `kaggle_archive_inventory.md`. No imported market data.

**Chat prompt:**

```text
Execute ROADMAP Step 1 only. Read BACKFILL_ROADMAP.md and the Step 0 checkpoint. Locate the ORIGINAL Kaggle equity archive in the available user-provided files/device-mounted paths. Do not substitute a small sample or partial upload for the original. Inventory the archive without modifying it: exact path, filename, byte size, SHA-256, archive member list, file formats, compression, folder structure and obvious metadata. Use streaming/listing and bounded samples; do not load the full archive into RAM. Determine whether the original large archive is actually accessible. If it is not accessible, stop and report exactly which original file must be supplied; do not claim the small upload represents it. Produce kaggle_archive_inventory.json and .md. No extraction into the production Drive archive, no code changes, no model work.
```

**Checkpoint:** original archive path, byte size, SHA-256, archive-member inventory, completeness/sample determination, unresolved access blockers.

---

## Step 2 — Validate Kaggle equity data and source integrity

**Objective:** Decide whether the Kaggle data is genuinely usable for five-minute equity OHLCV backfill.

**Actions:**
- Inspect representative files across archive members, not just one convenient file.
- Verify symbol/ticker field, exchange/market, timestamp field, timezone or timestamp convention, interval, OHLC fields and volume field.
- Verify timestamp range, per-symbol row counts, duplicate timestamps, missing values, zero/negative prices, negative volume, invalid OHLC relationships and anomalous intervals.
- Determine whether prices are adjusted or unadjusted, and identify split/bonus adjustment risks. Do not combine adjusted and unadjusted series without a documented policy.
- Check for multiple files/versions per symbol and whether rows are daily, minute, five-minute or mixed frequency.
- Compare sample sessions against an independent known source where practical; clearly label any comparison as a sample, not full validation.
- Record data lineage, Kaggle dataset title/version/source URL if included in the archive metadata.

**Acceptance:** A written PASS/FAIL decision for each data family and for the archive as a whole. Only data proven to contain usable five-minute equity OHLCV can be approved for the five-minute historical archive. Invalid or ambiguous members are quarantined, not guessed.

**Deliverables:** `kaggle_source_quality_audit.json`, `kaggle_source_quality_audit.md`, and a list of approved/rejected archive members. No production data writes.

**Chat prompt:**

```text
Execute ROADMAP Step 2 only. Read BACKFILL_ROADMAP.md and verify the Step 1 archive checksum before inspecting the archive. Perform a read-only source/data-quality audit of the Kaggle equity files. Establish whether each file is genuine five-minute equity OHLCV; verify symbol mapping, exchange, timestamps/timezone, candle timestamp convention, interval, OHLCV columns, adjustment status, date coverage, duplicates, missing/invalid values, OHLC invariants, volume validity and per-symbol coverage. Inspect samples across all distinct file formats/layouts. Do not silently repair ambiguous data. Do not import to the canonical Drive archive or alter production files. Produce machine-readable JSON and a human-readable report with approved and rejected members, checksums and explicit limitations. If timestamp semantics, frequency or adjustment status cannot be established, mark that portion NOT APPROVED. Do not proceed to normalization or model training.
```

**Checkpoint:** approved files, rejected files and reasons, verified interval, timezone/timestamp convention, adjustment status, symbol count, date range, quality metrics, archive SHA-256.

---

## Step 3 — Audit existing Google Drive archives and source provenance

**Objective:** Know exactly what is already stored before downloading anything again.

**Actions:**
- Inventory existing per-symbol five-minute Parquet files under the current archive root.
- Inventory separate Kaggle source files, existing INDstocks/Dhan/NSE files, older seed archives and current manifests.
- Read Parquet schema and row counts; inspect timestamp min/max and duplicate counts by symbol.
- Identify whether a file is source-specific or already merged. Do not assume a merged file's provenance can be reconstructed.
- Record existing window checkpoints and validate the data file plus its JSON commit marker before treating a window as complete.
- Identify active or stale partial runs and any parallel archive roots that could cause duplicate work.

**Acceptance:** A source-aware inventory exists for every readable symbol file. Unreadable files, unknown provenance, overlapping roots and incomplete checkpoints are explicitly listed. Nothing has been overwritten.

**Deliverables:** `drive_archive_inventory.json` and `.md`, including exact Drive paths and source classifications.

**Chat prompt:**

```text
Execute ROADMAP Step 3 only. Read BACKFILL_ROADMAP.md and verify Steps 0–2 evidence. Perform a READ-ONLY inventory of the existing Google Drive five-minute equity archives and manifests. Inspect every relevant Parquet file's schema, symbol, row count, timestamp range, duplicate count and source/provenance metadata. Include Kaggle source-specific files, current combined per-symbol files, INDstocks/Dhan/NSE archives, window checkpoints and backfill manifests. Detect overlapping archive roots, incomplete checkpoint pairs (Parquet without valid COMPLETE JSON marker or vice versa), unreadable files and unknown provenance. Do not write, rename, merge, delete or download replacement data. Produce drive_archive_inventory.json and .md. The report must distinguish verified source-specific data from merged/unknown-source data.
```

**Checkpoint:** inventory report paths, canonical archive root, per-source counts/bytes/symbols, unknown provenance, corrupt files, incomplete checkpoints, duplicate roots.

---

## Step 4 — Define canonical five-minute schema and safely stage Kaggle data

**Objective:** Convert only approved Kaggle members into a reproducible source-specific staging layer.

**Actions:**
- Define canonical columns and dtypes: symbol, completed-bar timestamp in `Asia/Kolkata`, Open, High, Low, Close, Volume, source ID and source file/version reference.
- Define candle timestamp convention. Convert source candle-start timestamps to completed-bar timestamps only when the source audit proves that convention.
- Normalize symbol names to official NSE trading symbols using a documented mapping table; preserve the original source ticker alongside the canonical symbol.
- Preserve the original archive and raw extracted files unchanged. Write normalized outputs to a separate `source=KAGGLE` staging location; never overwrite INDstocks or existing canonical files in this step.
- Partition by symbol or another repository-compatible layout; use Parquet and a deterministic compression setting.
- Produce checksums and row/session statistics per output file.

**Acceptance:** Every normalized row can be traced to an approved source member; no invalid/ambiguous rows are silently fabricated; source archive remains unchanged; staging output is reproducible.

**Deliverables:** Source-specific Kaggle Parquet staging files plus `kaggle_staging_manifest.json` and audit report. No merged archive and no INDstocks fetch.

**Chat prompt:**

```text
Execute ROADMAP Step 4 only. Read BACKFILL_ROADMAP.md and verify Steps 1–3. Implement the smallest reproducible importer for ONLY the Kaggle members approved in Step 2. Use the existing repository schema and storage conventions where compatible. Preserve raw original files unchanged and write outputs only to a clearly separate Kaggle source staging path (do not overwrite canonical per-symbol files or INDstocks data). Include original ticker, canonical NSE symbol, completed-bar timestamp in Asia/Kolkata, OHLCV, source ID, source dataset/version and source-file reference. Apply timestamp conversion only where the Step 2 audit proved the source convention. Emit per-file checksum, rows, symbol/date coverage and quality counts in kaggle_staging_manifest.json. Test normal, empty, duplicate-timestamp, invalid-OHLC and unmapped-symbol cases. Do not fetch INDstocks data, merge sources, retrain or deploy. Report exact tests and paths.
```

**Checkpoint:** importer file(s), test results, staging path, staging manifest, source-file checksums, approved symbol count and all rejected/unmapped records.

---

## Step 5 — Reconcile symbols, dates and source overlap

**Objective:** Establish the exact eligible stock universe and decide which source supplies each historical interval.

**Actions:**
- Compare approved Kaggle staging files with existing Drive symbol files and current official NSE equity universe/instrument mapping.
- Normalize only for comparison; retain all source-specific raw files.
- For each symbol/date/time, classify rows as Kaggle-only, existing-archive-only, overlapping-identical, overlapping-within-tolerance or conflicting.
- Define and document deterministic source precedence. Project agreement: use valid Kaggle equity history as the primary historical source and INDstocks for missing coverage. Preserve already-valid existing source data and do not silently discard it. Any official-source or existing-source conflict requires an explicit, auditable rule.
- Identify new listings/ineligible symbols that could not yet have 200 historical sessions. Keep them in the report; do not silently remove them from the denominator.
- Freeze the target symbol universe and eligibility denominator before calculating the 90% coverage target.

**Acceptance:** Every target symbol has a recorded source status; overlap conflicts are measured; the eligibility denominator and rationale are frozen; no source has been overwritten.

**Deliverables:** `source_reconciliation.json`, `source_conflicts.csv`, `eligible_universe.csv`, and report.

**Chat prompt:**

```text
Execute ROADMAP Step 5 only. Read BACKFILL_ROADMAP.md and verify Steps 0–4. Reconcile approved Kaggle staging data against existing Drive archives and the official NSE equity/instrument universe. Preserve every source-specific input unchanged. For each symbol/date/timestamp, quantify Kaggle-only, existing-source-only, identical overlap, tolerable overlap and conflicting overlap. Propose a deterministic documented precedence consistent with the project agreement: valid Kaggle history is the primary historical source; INDstocks fills missing coverage; valid existing data is retained unless a conflict rule is explicitly approved. Do not silently choose a source for material conflicts. Freeze the exact eligible symbol list and 90% denominator, explicitly accounting for listing dates and symbols that could not yet have 200 sessions. Produce the reconciliation JSON, conflict CSV, eligible universe CSV and report. Do not call INDstocks or merge/write canonical data.
```

**Checkpoint:** frozen universe file and checksum, denominator and eligibility rules, source precedence, conflict counts/list, symbol mappings and listing-date exclusions.

---

## Step 6 — Build an exact missing-session and missing-candle audit

**Objective:** Replace approximate start-date-only decisions with a precise manifest of missing intervals.

**Actions:**
- Build `audit_gaps.py` (or the smallest equivalent utility) using the frozen eligible universe and source reconciliation outputs.
- Construct expected five-minute timestamps from an authoritative NSE trading calendar/session schedule, including holidays and special/short sessions. Normal full sessions generally have 75 bars; do not force 75 on a session whose official schedule differs.
- Compare expected timestamps against the union of approved historical sources, using validated OHLCV rows only.
- Identify missing whole sessions, partial sessions, missing internal candles, invalid bars, duplicate timestamps and stale tails separately.
- Generate gap tasks in provider-compatible windows of at most seven calendar days, with no more than five symbols per INDstocks request. Group only compatible ranges; avoid requesting already-complete intervals.
- Manifest task identity must be stable across restarts and must include provider/source, symbol(s), exact start/end, interval, calendar/session rule version and input archive checksums.
- Do not label a chunk complete merely because the API returned HTTP success. It becomes complete only after the downloaded rows are validated and durably committed.

**Acceptance:** Repeated audits against unchanged inputs generate the same task IDs and gap counts. Complete sessions and incomplete sessions are distinguished accurately. A deliberately empty dataset, one missing candle, a missing date, a holiday and a special-session example are handled correctly in tests.

**Deliverables:** `audit_gaps.py`, tests, `backfill_manifest.json` in PLANNED/read-only state, and `gap_audit_report.md`.

**Chat prompt:**

```text
Execute ROADMAP Step 6 only. Read BACKFILL_ROADMAP.md and verify Steps 0–5. Build a read-only exact five-minute gap auditor using the frozen eligible_universe.csv and source-specific archive inventory. Construct expected candle timestamps from the project's verified NSE calendar/session rules, accounting for holidays and special/short sessions. A normal full NSE session generally has 75 bars, but do not hard-code 75 for every date. Detect missing sessions, missing internal candles, partial sessions, duplicate timestamps, invalid OHLCV, stale tails and unknown provenance. Generate stable task IDs and seven-calendar-day task windows compatible with the INDstocks maximum of five instruments per request. Do not fetch market data or mutate source archives. Add tests for complete data, empty data, one missing candle, one missing session, holiday, special session, duplicate and invalid OHLC. Re-running on identical inputs must produce identical task IDs and counts. Produce audit_gaps.py, tests, a PLANNED backfill_manifest.json and gap_audit_report.md. No model changes.
```

**Checkpoint:** gap auditor path and commit, tests and results, manifest path/checksum, counts by gap type, expected calendar version and exact task count.

---

## Step 7 — Implement durable, resumable checkpoint execution

**Objective:** Make INDstocks fill only manifest-approved missing gaps and survive process restarts without corrupting data.

**Actions:**
- Reuse `indstocks_backfill_all_nse_5m_history.py`, `indstocks_client.py` and `persistent_store.py` rather than building a parallel downloader.
- Read only PLANNED/PENDING tasks from the approved gap manifest.
- Enforce five instruments/request, seven calendar days/request and conservative one-request-per-second pacing unless current provider documentation requires stricter limits.
- Retry only bounded, classified transient errors (429/5xx/network/TLS/timeouts); use backoff and persist failure details. Do not retry permanent mapping/schema/authentication errors indefinitely.
- Persist each chunk using a safe commit protocol: write Parquet to a temporary/staging object, verify it can be read and passes schema/range/quality checks, persist it to Drive, then write a JSON completion marker as the commit record. On resume, skip only when both data and completion marker exist and their identities/checksums validate.
- Track PENDING, RUNNING, COMPLETED, FAILED and PAUSED_FOR_MARKET states. Do not overwrite COMPLETED with RUNNING on duplicate triggers.
- Use a single-run lock/lease to prevent concurrent workers processing the same manifest/task. Expired leases must be recoverable.
- Keep Cloud Run as the existing TOTP token owner. If the worker is invoked through the existing Cloud Run process, reuse its token cache. Do not independently mint a new TOTP token from GitHub Actions, Colab or a second service.
- Pause out of hours only; live market capture has priority. Store checkpoints in Drive, not only in container `/tmp` or process memory.
- Do not mark a provider-omitted instrument/window as a valid full session. Record `NO_CANDLES`/`PROVIDER_OMITTED` and leave the corresponding gap unresolved unless another validated source fills it.

**Acceptance:** Unit/integration tests prove restart recovery, duplicate-trigger exclusion, bad/missing marker handling, incomplete Parquet rejection, retry limits, rate pacing and market-hours suspension. A failed upload cannot be reported COMPLETED.

**Deliverables:** Minimal code patch, tests and a design note for the commit protocol. No full-universe execution yet.

**Chat prompt:**

```text
Execute ROADMAP Step 7 only. Read BACKFILL_ROADMAP.md and verify Step 6. Extend the existing INDstocks backfill/checkpoint implementation with the smallest change needed to consume the approved gap manifest. Do not create a second API client or parallel backfill architecture. Preserve the provider limits (max five instruments and seven calendar days per request; conservative one request/second pacing) and the existing Cloud Run TOTP token ownership. Implement durable task states and a single-run lock. A task may be marked COMPLETED only after its Parquet output is durably written/read back/validated and its JSON commit marker is durably written with matching task ID, range, symbols, schema and checksum. Resume must verify both artifacts and skip only a valid completed task. Missing/provider-omitted candles remain gaps. Pause during NSE cash-market hours using the exchange calendar. Add tests for process restart, duplicate trigger, corrupt/missing marker, incomplete Parquet, retry exhaustion, auth/permanent errors, rate pacing and live-market pause. Do not run a large backfill, retrain, change production model files or deploy. Report files changed and test evidence.
```

**Checkpoint:** code commit, modified files, tests, lock semantics, checkpoint paths/schema, retry policy, rate-limit evidence and Cloud Run token ownership verification.

---

## Step 8 — Run a controlled pilot on a tiny symbol set

**Objective:** Verify real provider requests, gap-only behaviour and Drive commits before scaling.

**Actions:**
- Choose a small, representative pilot (for example, 3 symbols) from the frozen eligible universe; include at least one with clear missing coverage and one with existing coverage.
- Run only during NSE non-trading hours.
- Confirm the pilot requests only manifest-listed intervals; no request for a verified complete interval.
- Inspect raw provider response, normalized bars, timestamp conversion, OHLCV validity, duplicate resolution and Drive commit markers.
- Interrupt one test run deliberately at a safe checkpoint and resume it. Verify completed tasks are skipped and unfinished tasks resume.
- Independently rerun `audit_gaps.py` and compare before/after gaps.

**Acceptance:** Pilot fills only eligible missing intervals; checkpoint recovery works; no existing valid rows are lost; no source data is fabricated; post-pilot gap audit reflects the actual additions; request count and pacing are documented.

**Deliverables:** `backfill_pilot_report.md`, before/after manifests, task logs with secrets redacted, test symbol list and Drive file checksums.

**Chat prompt:**

```text
Execute ROADMAP Step 8 only. Read BACKFILL_ROADMAP.md and verify Steps 0–7, including the exact gap manifest checksum and code commit. Run a tiny 3-symbol pilot only, during NSE non-trading hours, using the approved manifest and the existing Cloud Run/INDstocks token owner. One pilot symbol should have a known gap and one should have existing complete coverage. Prove that no complete interval is requested. Validate returned timestamps, OHLCV, session completeness, duplicate/conflict handling and Drive commit markers. Exercise restart recovery on a controlled unfinished test task without corrupting production data. Run the independent gap audit again and quantify the change. Stop after the pilot; do not scale to the full universe. Produce a report with exact request count, timestamps, files, checksums, before/after gap counts and any remaining issue. Do not expose tokens or alter model artifacts.
```

**Checkpoint:** pilot symbols, execution dates, before/after manifest checksums, number of tasks/requests, recovered task evidence, uploaded files/checksums and pass/fail.

---

## Step 9 — Execute the full eligible-universe backfill

**Objective:** Fill all approved historical gaps at a controlled rate without disrupting live collection.

**Preconditions (all required):** Steps 0–8 PASS; no other backfill is active; frozen universe checksum matches; provider credentials and Drive write preflight pass; Cloud Run token ownership is verified; the gap manifest is current and approved.

**Actions:**
- Execute only PENDING/FAILED-retryable gap tasks in the approved manifest.
- Run the backfill outside NSE market hours and allow the Cloud Run market-hours guard to pause if a session begins.
- Persist progress and checkpoint state frequently in Drive.
- Preserve completed tasks across restart; process only unfinished tasks on resume.
- Monitor 429/5xx/TLS/timeouts, request rate, Drive writes, provider omissions, unmapped instruments and per-symbol progress.
- Stop safely on authentication failures, repeated rate limits, unexpected schema changes, Drive write failures, task identity mismatch or unexpected overwrite risk.
- Do not loosen completeness gates to finish faster.

**Acceptance:** Every task is either COMPLETED with verified data/commit evidence or explicitly FAILED/UNRESOLVED with a reason. No task is falsely marked complete. Live capture was not starved. No production model changed.

**Deliverables:** Updated `backfill_manifest.json`, run logs/progress summary and `full_backfill_execution_report.md`.

**Chat prompt:**

```text
Execute ROADMAP Step 9 only. Read BACKFILL_ROADMAP.md and independently verify that Steps 0–8 passed. Verify the approved universe checksum, gap manifest checksum, code commit, Drive write preflight, INDstocks authentication owner and absence of another active backfill. Execute only approved pending/retryable gap tasks, outside NSE cash-market hours, using the existing guarded Cloud Run process and conservative provider rate. Do not request complete intervals. Persist/recover each task using Step 7 commit rules. Stop safely on auth, rate-limit, schema, Drive-write, manifest-integrity or overwrite errors. Do not lower acceptance thresholds, retrain, deploy model changes or switch to one-minute data. At the end, report every task state and symbol outcome; a successful workflow exit alone is not PASS. Produce the updated Drive manifest and a detailed execution report. Do not claim completion until Step 11's independent audit passes.
```

**Checkpoint:** final task counts by state, symbol coverage, API calls, retry counts, unresolved gaps, latest Drive manifest checksum, execution log/report paths and production model unchanged evidence.

---

## Step 10 — Reconcile and consolidate source-specific archives

**Objective:** Produce a consistent canonical research archive while retaining immutable source-specific copies and provenance.

**Actions:**
- Re-read Kaggle, pre-existing Drive and INDstocks source-specific files.
- Reconcile on `(symbol, completed_bar_timestamp, interval)`.
- Apply the approved source precedence from Step 5. Kaggle supplies approved primary historical rows; INDstocks supplies missing intervals; valid prior sources are retained according to the documented conflict rule.
- Never silently replace a conflicting OHLCV row. Write conflict records with both source values, source paths and resolution decision.
- Deduplicate timestamps deterministically and report before/after counts.
- Store per-row or per-partition provenance sufficient to identify the source and original object/file. Preserve raw source-specific Parquet files separately.
- Retain the latest 500 usable complete sessions per symbol, while keeping raw source data immutable and outside the capped canonical view if required for audit.

**Acceptance:** Canonical data is deterministic across repeated runs; every canonical row has traceable source lineage; overlap conflicts are accounted for; source-specific inputs remain unchanged; there are no invalid or fabricated bars.

**Deliverables:** Canonical per-symbol Parquet archive, source reconciliation manifest, conflict report, checksums and consolidation report.

**Chat prompt:**

```text
Execute ROADMAP Step 10 only. Read BACKFILL_ROADMAP.md and verify Steps 0–9. Build the canonical five-minute research view from approved Kaggle source files, retained existing Drive sources and newly completed INDstocks gap fills. Keep each raw/source-specific archive immutable and separate. Reconcile by symbol, completed-bar timestamp and interval. Apply only the documented Step 5 source precedence; never silently overwrite material conflicts. Record source lineage for each row or partition, conflict values/resolutions, duplicate counts, row counts, date ranges and checksums. Enforce the latest-500-complete-usable-session retention rule only on the canonical view, not by deleting raw source archives. Prove deterministic output by running consolidation twice on the same inputs and comparing checksums/row counts. Do not train or deploy models. Produce the canonical files and a consolidation report.
```

**Checkpoint:** canonical archive path, source provenance design, manifest/checksums, conflict resolutions, duplicate counts, reproducibility test and raw-source immutability evidence.

---

## Step 11 — Independent final completeness and integrity audit

**Objective:** Establish whether the historical backfill truly meets the project's acceptance criteria.

**Actions:**
- Use an independent read-only audit path; do not rely solely on the downloader's own manifest calculations.
- Reconstruct expected session grids and compare every canonical symbol/date against expected five-minute timestamps.
- Check OHLCV invariants, timestamp timezone/convention, missing/duplicate/out-of-session bars, data freshness, source lineage, source conflicts and Parquet readability.
- Count complete usable sessions per symbol, capped at 500 for the canonical view.
- Recalculate the eligible-universe denominator from the frozen Step 5 list and report all exclusions explicitly.
- Check the 90% coverage criterion and minimum 200 complete sessions per covered symbol.
- Verify there is no accidental NIFTY index data in equity files and no equity data in the NIFTY context file.
- Verify no production model manifest, model weights, thresholds or feature contract changed during the backfill.

**Acceptance:** Overall PASS only if at least 90% of the frozen eligible universe has 200–500 complete usable sessions and all accepted sessions pass exact-grid integrity checks. Any unresolved source conflict, unknown provenance, invalid session or missing data must be visible in the report. If coverage is below 90%, report NEEDS_ATTENTION; do not lower the threshold.

**Deliverables:** `final_backfill_audit.json`, `final_backfill_audit.md`, per-symbol CSV and exception lists. This is the authoritative completion decision.

**Chat prompt:**

```text
Execute ROADMAP Step 11 only. Read BACKFILL_ROADMAP.md and verify Steps 0–10. Perform an independent READ-ONLY final audit of the canonical five-minute equity archive and source manifests. Do not trust the backfill's own PASS flag without independently recomputing it. Reconstruct expected session grids using the correct NSE trading calendar and special-session rules. Verify every accepted session has every expected valid candle, all OHLCV/timestamp/source-provenance rules, duplicate and conflict counts, freshness and Parquet readability. Count 200–500 complete usable sessions per symbol. Recalculate coverage against the frozen Step 5 eligible-universe denominator and explicitly list ineligible/new listings, unmapped symbols, failures and exclusions. Confirm NIFTY index rows remain separate from equity data. Compare production model manifests/weights/feature contracts/thresholds against the Step 0 baseline and prove they were not changed. Return PASS only if the full acceptance contract is met; otherwise return NEEDS_ATTENTION with exact failing symbols and gaps. Produce final_backfill_audit.json, .md, per-symbol CSV and exception lists. No repair or retraining in this step.
```

**Checkpoint:** authoritative PASS/NEEDS_ATTENTION, eligible denominator, symbols with 200+ complete sessions, coverage percentage, per-symbol exceptions, audit code/commit, final file checksums, production model unchanged evidence.

---

## Step 12 — Freeze the validated dataset and close the backfill workstream

**Objective:** Prevent accidental mutation and provide a handoff for later, separately approved research/model work.

**Actions:**
- If Step 11 is PASS, mark the canonical archive version as frozen and record its manifest/checksums.
- Preserve source-specific Kaggle and INDstocks archives and the complete audit trail.
- Stop/disable only temporary backfill activity that is no longer needed; do not stop live collection services.
- Record remaining gaps and symbols below threshold even if the overall gate passes.
- Produce a concise handoff stating that data is ready for a separate model-research decision. Do not start model training in this step.
- If Step 11 is NEEDS_ATTENTION, do not freeze as fully validated; return to the exact failed gap tasks only.

**Acceptance:** Frozen archive version and checksums are documented; no retraining has occurred; all residual exceptions are recorded; live collection remains operational.

**Deliverables:** `BACKFILL_FREEZE_MANIFEST.json`, `BACKFILL_COMPLETION_REPORT.md`, and updated execution ledger.

**Chat prompt:**

```text
Execute ROADMAP Step 12 only. Read BACKFILL_ROADMAP.md and verify Step 11's final independent audit artifact and checksum. If and only if Step 11 is PASS, freeze/version the canonical dataset and record manifest/checksums, source archives, code commit, coverage results and residual exceptions. Preserve all raw source files and audit artefacts. Stop only temporary historical backfill execution if no tasks remain; do not stop live market collection. Do not retrain, redeploy, alter model thresholds or change the prediction engine. If Step 11 is NEEDS_ATTENTION, do not declare completion or freeze as validated; identify the precise failed tasks to return to. Produce BACKFILL_FREEZE_MANIFEST.json and BACKFILL_COMPLETION_REPORT.md.
```

**Final checkpoint:** frozen dataset ID/version, canonical Drive path, checksums, final audit status, coverage percentage, remaining exceptions, live collection status and confirmation that no model training/replacement occurred.

---

# 4. Execution ledger

Update this ledger only after the evidence and acceptance checks for that step have been independently verified. Use the exact commit SHA and artifact paths. Do not mark a step PASS for code-only completion, a started workflow, or an unverified Drive upload.

| Step | Description | Status | Commit / evidence | Artifact path | Date |
|---:|---|---|---|---|---|
| 0 | Freeze scope and baseline | BLOCKED | `542cf790d2472a4be1493114785b142019e37364` (inspection start); baseline commits recorded on branch | `backfill_baseline.json`; `backfill_baseline.md` | 2026-11-04 |
| 1 | Original Kaggle archive inventory | NOT STARTED | — | — | — |
| 2 | Kaggle source/data-quality audit | NOT STARTED | — | — | — |
| 3 | Existing Drive archive inventory | NOT STARTED | — | — | — |
| 4 | Canonical schema and Kaggle staging | NOT STARTED | — | — | — |
| 5 | Source reconciliation and eligible universe | NOT STARTED | — | — | — |
| 6 | Exact gap audit and planned manifest | NOT STARTED | — | — | — |
| 7 | Durable resumable checkpoint execution | NOT STARTED | — | — | — |
| 8 | Controlled pilot | NOT STARTED | — | — | — |
| 9 | Full eligible-universe backfill | NOT STARTED | — | — | — |
| 10 | Source reconciliation and canonical consolidation | NOT STARTED | — | — | — |
| 11 | Independent final completeness audit | NOT STARTED | — | — | — |
| 12 | Freeze and handoff | NOT STARTED | — | — | — |

Allowed statuses: `NOT STARTED`, `IN PROGRESS`, `BLOCKED`, `FAILED`, `NEEDS ATTENTION`, `PASS`.

## 5. Required end-of-chat report format

Every development-agent chat must end with this compact handoff. Copy it into the next chat along with the roadmap.

```text
ROADMAP STEP: [number and title]
STATUS: [PASS / FAILED / BLOCKED / NEEDS ATTENTION]
REPOSITORY: andromyth/indian-market-trading
BRANCH:
COMMIT BEFORE:
COMMIT AFTER:
WORKING TREE:
FILES CHANGED:
COMMANDS RUN:
TESTS AND RESULTS:
DRIVE PATHS READ:
DRIVE PATHS WRITTEN:
MANIFEST / CHECKSUMS:
SYMBOLS / ROWS / SESSIONS / GAPS AFFECTED:
RATE-LIMIT / MARKET-HOURS CHECK:
PRODUCTION MODEL UNCHANGED: [YES/NO + evidence]
ROOT CAUSE OF ANY FAILURE:
UNRESOLVED ITEMS:
NEXT ALLOWED STEP:
```

## 6. Explicitly out of scope until this roadmap is complete

- Retraining or replacing Logistic Regression with XGBoost/LightGBM.
- Changing `target_30m`, adding Triple-Barrier labels or changing forecast horizons.
- Rewriting Pandas feature construction in Polars.
- Replacing the existing INDstocks client with an unverified `aiohttp` implementation.
- Adding Redis, Compute Engine or a new persistent cache service.
- Downloading one-minute historical data as part of this five-minute backfill.
- Rewriting Cloud Run's live worker or changing TOTP token ownership.
- Merging NIFTY index CSV rows into individual equity OHLCV data.
- Starting any model retraining before Step 11 independently reports PASS and a separate training plan is approved.

---

**End state:** a source-audited, provenance-preserving, gap-filled, resumable and independently validated five-minute NSE equity dataset in Google Drive, with at least 200 and no more than 500 complete usable sessions for at least 90% of the frozen eligible universe, while the live prediction system and its production models remain unchanged.