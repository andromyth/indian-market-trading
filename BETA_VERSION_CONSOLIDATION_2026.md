# Beta Version Consolidation

Created: 2026-10-03

## Base

`beta_version` is based on the current `main` branch. All production-critical ALL-NSE Delivery/Intraday/F&O/EOD/news-events work already present in main is therefore retained.

## Branch-only work included in beta

### feature/context-direction-layer
- `directional_context.py`
- `tests/test_directional_context.py`

This is an independent causal evidence layer. It does not replace or modify the locked 59-feature production classifier.

### feature/eod-archive-audit-gate
- `nse_eod_archive_audit.py`
- `.github/workflows/nse_eod_archive_audit.yml`

The obsolete modification to the earlier 53-feature candidate builder was not copied because the current production path uses the ALL-NSE 59-feature builder.

### colab-gpu-handoff-20261003
- `COLAB_GPU_BOOTSTRAP_20261003.ipynb`
- `COLAB_GPU_HANDOFF_20261003.txt`

### colab-handoff-20261002-a7f3
- `COLAB_HANDOFF_A7F3_20261002.txt`

## Branches already represented by main

The historical delivery/EOD/F&O/intraday branches and the ALL-NSE replacement branch contain no commits that are absent from the current main history, so they were not duplicated into beta.

Older unique implementation commits such as the earlier Drive-folder discovery and early NSE index-history loader were superseded by later main-branch implementations and were intentionally not reintroduced.

## Production rule

`main` remains the production branch. `beta_version` is the consolidated development/experimental branch.
