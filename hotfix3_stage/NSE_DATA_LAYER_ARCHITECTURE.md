# NSE-first production architecture

## Source policy

NSE official web interfaces are the primary source for exchange data exposed by
NSE's public pages/reports:

- Security-wise historical equity price/volume/delivery data.
- Current equity quote and trade-information snapshots.
- Corporate actions and corporate announcements.
- Historical F&O contract data where the public interface exposes the required contract parameters.
- Current quote-chart prints for the public intraday chart.

The current NSE site exposes CSV downloads on its historical security-wise price/
volume and historical F&O interfaces. See the official pages in `README.md`.
The software uses a cautious session/cookie pattern and rate-limited requests;
it does not bypass CAPTCHA/anti-bot controls.

## Storage

Google Drive is the durable data store. Streamlit's filesystem is only a cache.

```text
Indian_Market_Research/
└── production_data/
    └── nse_official/
        ├── equity/<SYMBOL>/YYYYMMDD_YYYYMMDD.parquet
        ├── intraday_1m/YYYY/MM/DD/<SYMBOL>.parquet
        ├── fno/<SYMBOL>/...
        ├── raw/equity/<SYMBOL>/...
        └── raw/fno/<SYMBOL>/...
```

The local cache mirrors that structure under `data/` and records provenance in
`data/nse_official/manifest.json`.

## Training/deployment separation

**Google Colab** is the training/build environment. It mounts Google Drive,
constructs point-in-time features, trains/validates the candidate model, and
writes a small release bundle:

```text
nse_production_candidate_30m.joblib
NSE_PRODUCTION_MODEL_MANIFEST_2026.json
MODEL_TARGET_SEMANTICS_TEST.txt
```

The **private GitHub repository** contains source code, requirements, model
artifact/release manifest, and no large historical datasets or secrets.

**Streamlit** is the serving/runtime environment. It reads the small model
artifact from the private repo and reads current/historical data from the
Google Drive persistent store through the configured service account. It does
not train or recompile a model.

## Target semantics (fixed)

```text
future_return_30m = close(t+30m) / close(t) - 1
class 1 = future_return_30m > 0  -> UP
class 0 = future_return_30m <= 0 -> DOWN/FLAT
P(UP) = predict_proba()[:,1]
BUY = P(UP) >= 0.55
SELL = P(UP) <= 0.45
HOLD = otherwise
```

Historical replay must use the exact same target definition and must never use
information newer than the selected timestamp.

## Live intraday note

NSE's public equity quote chart can be converted to 1-minute LTP-based OHLC,
but it is not a broker websocket and does not provide a reliable per-minute
volume field in the public chart response. The application therefore does not
invent volume. A broker websocket/API (e.g. Dhan) may be added later strictly
for live execution-grade fields while retaining NSE as the authoritative
exchange-data layer for historical/EOD data.

## Source precedence

1. Stored NSE-official archive in Google Drive.
2. Fresh NSE official pull.
3. Dhan live/historical only where the NSE public interface cannot supply the required live field, and the provenance is shown.
4. Other public providers are reference-only and never silently presented as NSE.
