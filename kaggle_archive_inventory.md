# Step 1 inventory — original Kaggle equity archive

**Status: BLOCKED**  
**Audit date:** 4 November 2026  
**Repository:** [andromyth/indian-market-trading](https://github.com/andromyth/indian-market-trading)  
**Branch:** `backfill-roadmap`  
**Branch HEAD before Step 1:** `3cbaa1080033bb2bf49e1628b7458a52866b397e`

## Result

The original full Kaggle equity archive could not be accessed from this ChatGPT session. The Step 0 handoff says the original is stored on the user's device and specifically warns not to substitute the previously referenced approximately 11 MB sample/partial upload.

The current runtime had no archive candidates in the mounted locations checked (`/mnt/data`, `/home/oai/share` and `/tmp`). A search of the available conversation and Library files also returned no compressed archive candidate. This session does not have direct access to the user's device filesystem.

Accordingly, **no claim is made about the original archive's filename, path, exact byte size, SHA-256, compression, members, folder structure or data contents**. These fields are intentionally left unknown in the JSON inventory rather than inferred.

## Inventory findings

| Required item | Finding |
|---|---|
| Original archive accessible | No |
| Exact path / filename | Not available |
| Exact byte size | Not measured |
| SHA-256 | Not calculated |
| Archive format / compression | Not determined |
| Member list and count | Not inspected |
| File formats / folder structure | Not inspected |
| Equity vs index vs metadata contents | Not determined |
| Previously referenced ~11 MB upload | Not present in this runtime; not treated as the original; completeness not determined |

## Work actually completed

1. Read `BACKFILL_ROADMAP.md` directly from `backfill-roadmap`.
2. Read `backfill_baseline.md` and `backfill_baseline.json` and confirmed the Step 0 handoff.
3. Verified the branch head before this Step 1 record: `3cbaa1080033bb2bf49e1628b7458a52866b397e`.
4. Checked the mounted runtime locations listed above and searched the available conversation/Library files for a compressed archive candidate.
5. Created this report and `kaggle_archive_inventory.json` with missing measurements explicitly marked as unknown.

## Scope and safety

No archive was extracted or modified. No Google Drive path was read or written. No market data was imported. No code, model, deployment, Cloud Run service or Scheduler job was changed. No historical backfill or model training was started.

## Blocker and required action

Please attach the **original full Kaggle equity archive** to this conversation, or make it accessible in a runtime that can be inspected and provide the exact path. Do not send the previously referenced approximately 11 MB sample in place of the original.

Once the original is accessible, resume **Step 1 only**: calculate its exact size and SHA-256, list archive members, identify formats/compression/folder structure, and inspect bounded samples without loading the entire archive into RAM. Step 2 must remain unstarted until that inventory is complete and Step 1 passes.

## Handoff

- **Step:** 1 — Locate and inventory the original Kaggle equity archive
- **Status:** BLOCKED
- **Evidence:** `kaggle_archive_inventory.json`, `kaggle_archive_inventory.md`
- **Next allowed action:** Supply the original archive and finish Step 1; do not start Step 2 or any backfill.
