# Self-driving add-on (beta_version)

Additive layer. It does not touch `main`, the certified 30-minute production model or the
59-feature delivery candidate. It places no orders.

## What the review of the beta found (from the files, not from measured accuracy)

1. Global inputs do not exist. No US/Europe/Asia indices, crude, USD/INR, DXY, US 10Y, India VIX
   or FII/DII appear anywhere in the Python sources.
2. In the 59-feature delivery contract, news is a single flag (`news_available`). Sentiment,
   intensity and novelty are not model inputs.
3. Delivery training uses one 80/20 chronological split, a logistic classifier thresholded at
   0.5, and a label of "next close above today's close". That label mostly measures whether the
   whole market rose.
4. The certified intraday model lists raw price levels (`open`, `high`, `low`, `close`,
   `nifty_close`) and many `*_extra` duplicates among 137 features, and was trained on 3 stocks.
   It also uses `volume`/`volume_ratio`, while the repo's own docs say the NSE quote-chart lane
   has no per-minute volume. Training and live serving would then see different inputs.
5. Promotion is manual. Nothing automatically retrains, compares and promotes or rolls back.
   `autonomous_delivery_cycle.yml` is dispatch-only.

## What this add-on adds

| Module | Purpose |
|---|---|
| `sd_global_cues.py` | 15 global series with explicit `available_at`; as-of join; stale bars become NaN |
| `sd_news_features.py`, `sd_news_collect.py` | forward-only headline archive, scored features, 0 vs unknown handled |
| `sd_fno_features.py` | OI/price build-up regime, PCR/basis/IV z-scores for the underlying |
| `sd_intraday_features.py` | volume-free intraday features; label never crosses session end |
| `sd_walkforward.py` | market-neutral target, purged walk-forward, blend + isotonic calibration, Wilson-bound thresholds, abstain |
| `sd_registry.py` | champion/challenger gate on identical folds, rollback, PSI drift, live accuracy check |
| `sd_cycle.py` | train / predict / validate stages and automatic feature-group gating |

Feature groups switch themselves on only when populated (default: >= 60% coverage and enough
distinct dates), so a news archive that started last week adds nothing until it has history.

## Daily loop

```
GitHub Actions weekdays 14:30 UTC (after your NSE syncs)
  global cues -> headline archive -> predict (champion) -> validate yesterday -> status.json
Colab weekly, or when status_validate.json has retrain=true
  panel + global + news -> walk-forward -> gate vs champion -> register or reject
Streamlit
  reads champion thresholds and the prediction ledger from Drive; shows ABSTAIN rows as ABSTAIN
```

## Integration points I could not verify end to end (do these first)

1. Export the training panel. In the base delivery builder, after the panel DataFrame is built,
   add `panel.to_parquet(out / "PANEL_2026.parquet", index=False)`. Required columns: `symbol`,
   `date`, `decision_timestamp` (UTC), `y_return` (next-session close/close - 1), plus features.
2. Produce an inference parquet for the latest session (same columns, no label) and a realised
   returns parquet (`symbol`, `date`, `y_return`). Then replace the "skipped" stage in
   `sd_github_job.py` with `predict_stage` and `validate_stage` calls.
3. Confirm each Yahoo ticker in `GLOBAL_SERIES` returns data (especially `^INDIAVIX`, `DX-Y.NYB`,
   `^TNX`), and each RSS URL in `sd_news_collect.FEEDS`. Dead ones are logged, not fatal.
4. The NSE FII/DII parser (`parse_fii_dii`) assumes JSON keys I could not confirm. Test it once.

## Cost assumptions (editable)

`evaluate(cost_bps=30)` is a round-trip cost per leg for delivery: 20 bps is STT (0.1% on buy and
on sell) and 10 bps is an assumed allowance for brokerage, exchange fees, GST, stamp duty and
slippage. For intraday use roughly 12 bps (STT 0.025% on the sell side plus the same allowance).
STT rates were checked against several broker and tax sites (futures 0.05%, options premium 0.15%
from 1 April 2026); confirm against the NSE circular before relying on F&O costs.

## Tests

`python -m pytest -q tests/test_selfdriving.py` (11 tests). Run on synthetic data only, with
scikit-learn 1.8.0 and pandas 3.0.2; your pinned scikit-learn 1.6.1 was not run.
