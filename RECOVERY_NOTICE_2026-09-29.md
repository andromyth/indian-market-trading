# Stable recovery baseline — 2026-09-29

This package restores the last known-good dashboard core before the recent
watchlist/data-loader hotfix changes.

Core files restored exactly from the previously working project baseline:
- app.py
- data_fetcher.py
- signals.py

No new watchlist form, MultiIndex normalizer, or live-provider routing changes
are applied in this recovery baseline.

Purpose: first re-establish the previously working stock analysis screen.
The autonomous delivery/7-day validation work will be integrated incrementally
after this baseline is confirmed working.
