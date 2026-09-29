# Release — 2026-09-29

## Purpose

Add autonomous Delivery prediction serving and continuous prediction-vs-actual
validation around the existing production system without restarting the prior
model-development cycle.

## Included

- Existing 137-feature `target_30m` production model remains unchanged.
- Existing next-session Delivery classifier/regressor is bundled and served.
- Delivery is the default dashboard mode.
- Current validated Delivery universe is explicitly limited to the three trained
  symbols in the Delivery model manifest.
- Prediction ledger is idempotent: one logical prediction is stored once.
- Validation ledger persists the latest validation state per prediction.
- Delivery D+1 is target-aligned; D+2..D+7 are preserved as follow-through
  observations.
- Rolling seven-completed-prediction scorecard and model-health flag added.
- No automatic order placement.

## Deployment note

Historical datasets are intentionally not included in this release archive.
Keep them in Google Drive. Google Drive credentials/secrets are not included.

Run:

```bash
python deployment_smoke_test.py
```

before publishing the repository update.


## Hotfix 2 — provider OHLC(V) normalization

- Normalize Yahoo/NSE/Dhan history with flat or MultiIndex columns before indicator computation.
- Preserve missing Volume as NaN rather than fabricating it; price indicators remain available when OHLC is valid.
- Prevent malformed provider history from crashing the Streamlit dashboard.
- Normalize live model input frames using the same provider-safe contract.
