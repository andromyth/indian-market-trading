# Indian Stock Signal Dashboard — upgraded

A Streamlit dashboard for NSE/BSE cash-equity decision support with current
stock news, corporate-event triggers and F&O context. It generates ML-backed
Delivery and Intraday signals for the currently validated universe, records every
production prediction, continuously checks completed Delivery predictions against
actual next-session prices, and **never places orders**.

## What changed in this version

### 1. Causal backtest
The old backtest entered at the same close used to create the signal. That is a
look-ahead problem. The upgraded engine creates the signal on bar `i` and
enters at the next bar's open.

Stops and targets can trigger intrabar. When both are touched in one bar, the
backtest assumes the stop happened first, which is conservative.

### 2. Risk/exit engine
Delivery and intraday use different ATR-based risk settings:

| Mode | Trend | Stop | Target | Max holding |
|---|---|---:|---:|---:|
| Delivery | 20/50 | 1.8 ATR | 3.0 ATR | 20 bars |
| Intraday | 9/21 | 1.2 ATR | 2.0 ATR | 30 bars |

All values can be changed from the sidebar.

### 3. Production ML delivery + intraday routing

The dashboard now routes Delivery mode to the existing certified `next_session`
classifier/regressor and Intraday mode to the certified `target_30m` production
model. A signal is displayed directly after a stock is searched. Unknown symbols
are blocked instead of receiving an unvalidated score.

### 4. Continuous seven-prediction validation

Every production Delivery prediction is written to an idempotent prediction ledger.
When the next trading-session actual becomes available, the system compares the
predicted return/price with the real close, records the validation, and preserves
D+2 through D+7 follow-through observations as monitoring. A rolling seven-
completed-prediction scorecard is shown in the dashboard. A miss does not cause
immediate model replacement; the model-health monitor can raise a retraining
candidate flag after enough completed observations.

### 5. More evidence, not more certainty
The live score now combines:

* price trend, MACD, RSI, volume, breakout and volatility information;
* broad-market regime using NIFTY, or BANKNIFTY for common bank stocks;
* stock-specific current news with recency weighting and high-impact tags;
* corporate-event risk that reduces conviction instead of inventing direction.

F&O PCR, Max Pain and OI reads remain contextual because the available live
snapshot alone is not enough for an honest historical predictive test.

### 6. Better news matching
Common Indian-company aliases are supported. For example, `BHARTIARTL.NS`
will match `Bharti Airtel`, `Airtel`, and related naming variants rather than
only looking for the ticker string.

### 7. More honest validation
The dashboard reports:

* net strategy return after configurable cost/slippage assumptions;
* win rate, profit factor, average holding period and max drawdown;
* a matched-duration random benchmark;
* a last-30%-of-history holdout run.

The historical test does **not** pretend to validate live news sentiment,
because a dependable historical archive of the exact news feed is not provided
by the free data sources used here.

## Data sources and limitations

Equity price data uses `yfinance`, which is convenient for a local research
project but is not an exchange-certified tick feed. It can differ from a
broker's live quote. For execution-timing work, use a broker websocket later.

F&O and corporate announcements use NSE's public website endpoints. These are
not documented/guaranteed APIs and may be rate-limited or changed by NSE.

Current news comes from public RSS feeds from Economic Times Markets and
Moneycontrol. RSS freshness and coverage depend on the publishers; this is not
a complete market-news firehose.

## Run

```bash
cd stock_dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Important interpretation note

`BUY`, `SELL`, and `HOLD` are model classifications for decision support, not
guarantees of return. Delivery's trained universe is currently limited to the
validated symbols in the bundled delivery model. Unknown or unvalidated symbols
are blocked rather than silently generalized. No order execution is implemented.

## Next upgrade path

For a genuinely production-grade system, the next step is a broker websocket
for live quotes plus a proper historical event/news archive. With that data,
the next model layer can be walk-forward calibrated on out-of-sample data and
can separately evaluate cash equity and F&O strategies.

## Offline self-test

After installing dependencies, run:

```bash
python self_test.py
```

It uses synthetic OHLCV data to test indicator construction, signal scoring,
causal next-bar execution, risk exits, holdout calculation, matched-duration
random benchmarking, and the intraday no-overnight rule. It does not claim to
validate profitability on real securities.

## 1-Year Prediction Audit

The dashboard includes a causal one-year prediction audit for a diversified NSE universe. It replays the signal at each daily close using only information available through that date, enters at the next day's open, and measures direction over 1, 3, 5, 10 and 20 trading days. BUY is counted correct when forward return is positive; SELL is counted correct when forward return is negative; HOLD is excluded from directional hit rate.

The audit uses roughly two years of daily price history so the first year has enough warm-up data for the 20/50-day indicators. It also reports per-stock results, stock-size/volatility buckets, score-to-return Spearman correlation, average signed edge, and sensitivity to signal thresholds 0.20–0.40.

The audit intentionally does **not** inject current RSS headlines into historical dates. Therefore it validates the causal price + market portion of the model. A full historical news audit requires a timestamped historical news dataset.

Run it from the sidebar with **Run 1-Year Prediction Audit**. You can download the full prediction replay CSV from the dashboard.


## Historical Prediction Replay & Audit

The dashboard now has a **Historical Replay** view. Pick a past date, a stock, a mode and a prediction/holding horizon. The engine reconstructs the selected point-in-time input set, produces a prediction, then compares it with the subsequent **actual market move**.

For Intraday replay, the certified `target_30m` production model is run on the historical 1-minute price history through the selected timestamp. The result includes the ML signal, probability, confidence, data-readiness, reference price, ATR-based target/stop and the actual outcome after the selected minute horizon.

For Delivery replay, the existing causal rule engine is evaluated at the selected EOD date, with a configurable 1/3/5/10/20-trading-day forward horizon. The comparison uses the next trading day's open as the historical entry reference and reports subsequent return, target/stop touches and directional match.

The replay never substitutes **today's** news, F&O chain or corporate-event state into a past prediction. Historical headlines are retrieved on a best-effort basis from Google News RSS date search; this is not a complete timestamped news archive. Historical option-chain/OI/IV data is not yet connected to the replay, so the F&O mode is explicitly labelled **underlying replay** rather than presented as a contract-level historical F&O forecast.

Each replay can be appended to a session ledger and exported as CSV for later analysis. The replay is designed to become the foundation for a larger prediction-vs-actual ledger and repeated daily audit without rebuilding the production ML model.

The Historical Replay view also includes a **Batch Historical Prediction Audit**. It can replay the certified 30-minute Intraday model across a selected stock universe, date range, and decision interval. Cross-sectional features are built across the selected universe at each historical timestamp, and each prediction is joined to the next 30-minute actual outcome. Rows with incomplete optional information remain in the ledger and are flagged as data-limited rather than being silently discarded.

The file `MODEL_DATA_MODEL_CONTRACT.md` documents the production target semantics: class 1 is UP (future 30-minute return > 0), class 0 is DOWN/FLAT, and `predict_proba()[:,1]` is therefore the UP probability. A BUY followed by a DOWN actual is a model miss; the dashboard does not invert the labels to make the prediction appear correct.

## Free Historical Data Manager

The dashboard now includes a **Free Historical Data** manager. It downloads public research archives directly to the laptop; no broker API is required for the historical equity bootstrap.

### Public archives used

* `ganeshbiyer/Nse_Historical_Data`: 1-minute Nifty 500 history from 2018 through 31 Dec 2025.
* `ganeshbiyer/Nse_Historical_Data_2026`: separate 2026 1-minute Nifty 500 archive.
* `voletiramu/nse-fno-1min-data`: 214 F&O underlying stocks, 1-minute spot OHLCV, Apr 2024 through Apr 2026; about 464 MB compressed and about 3 GB uncompressed.

These are public research datasets provided by their maintainers. The dashboard stores the downloaded files locally and records source URLs, status, size, checksum and retrieval time in `data/historical/free_sources/manifest.json`. It does not bundle or redistribute the third-party data.

### What the downloader does

* downloads files itself with retries and streaming;
* skips files already present;
* supports the current watchlist, the 19-stock audit core, or discovery of the NIFTY 500 file list;
* merges 2018-2025 and 2026 local archives for historical research;
* resamples local 1-minute data to 5-minute or daily bars;
* has a local-first history provider for the one-year audit, falling back to yfinance when the local archive is missing;
* downloads and safely extracts the public F&O underlying archive;
* scans local coverage and exports a CSV manifest.

### CLI alternative

From the project directory:

```bash
source venv/bin/activate
python free_data.py --symbols SBIN BHARTIARTL RELIANCE --sources 2018_2025 2026 --scan
```

For the 19-stock core universe:

```bash
python free_data.py --universe core --sources 2018_2025 2026
```

For the free F&O archive:

```bash
python free_data.py --download-fno
```

The software intentionally does **not** silently download the entire NIFTY 500 archive on application startup because the archive can require substantial bandwidth and disk space. A user-initiated download is required.

### Source notes

The public Nifty 500 repositories state that the 2018-2025 archive contains 1-minute data from 2018 to 31 Dec 2025 and that 2026 is maintained separately. The free F&O repository states that its stock-underlying 1-minute archive covers 214 symbols from Apr 2024 to Apr 2026 and is distributed as a ~464 MB compressed release. NSE's current public report catalogue lists CM security-wise delivery positions, CM-UDiFF bhavcopy, and multiple F&O/OI reports; those public NSE reports remain candidates for a second, authoritative EOD reconciliation layer.

## Data authenticity and NSE reconciliation

The free intraday archives in this project are treated as **third-party research datasets**, not as exchange-authoritative truth. The current public sources are the `ganeshbiyer/Nse_Historical_Data` repository (2018-2025), its 2026 companion repository, and the `voletiramu/nse-fno-1min-data` release. The latter states that its 1-minute stock-underlying data is sourced from the Zerodha Kite API and provided as-is for research/backtesting.

The dashboard now includes:

* OHLCV integrity checks: duplicate timestamps, ordering, weekend rows, market-hour violations, OHLC consistency, non-positive prices, missing expected intraday slots, session coverage and large-move flags.
* A source-provenance table distinguishing **authoritative NSE** from **third-party public archives**.
* A best-effort **NSE reconciliation** step for the local equity archive. It aggregates local 1-minute bars to daily OHLCV and compares them to NSE's official historical-equity endpoint, recording price/volume mismatches and any NSE HTTP/rate-limit error.
* A CSV-exportable quality report. A failed NSE fetch is recorded as `ERROR`, never silently treated as validated.

### Command line

Validate local data:

```bash
python free_data.py --validate --quality-symbols SBIN BHARTIARTL RELIANCE
```

Validate and reconcile the recent local window against NSE:

```bash
python free_data.py --reconcile-nse --quality-symbols SBIN BHARTIARTL RELIANCE
```

NSE is the primary source for EOD reconciliation; the exchange's historical-report catalogue exposes security-wise equity price/volume archives and contract-wise derivatives archives. The public 1-minute GitHub archives are used as bulk intraday research inputs, not as the ground truth.

## DhanHQ integration

This build can use a user's DhanHQ account as the preferred NSE cash-data source when credentials are configured locally. It does not place orders.

### What Dhan provides to this integration

* NSE equity instrument/security-ID resolution from Dhan's official instrument master.
* Historical daily OHLCV.
* Historical intraday 1/5/15/25/60-minute OHLCV and OI where applicable. Dhan documents up to five years of intraday history, with a maximum 90-day polling window per request; this project automatically chunks long ranges.
* Current LTP through Dhan's market quote API.
* Expired options history (OHLC, IV, volume, OI and spot) through Dhan's rolling expired-options endpoint when the account/data plan permits it.
* Dhan profile/data-plan status check.

The Dhan documentation states that API requests require an access token. API key + secret authentication is a three-step consent flow which yields an access token; the API key/secret are valid for 12 months while access tokens are short-lived. See the official DhanHQ authentication and historical-data documentation.

### Local setup

1. Create an API-key/secret application in Dhan.
2. Set the redirect URL to:

```text
http://127.0.0.1:8765/callback
```

3. Copy `.env.example` to `.env` and fill in the local values:

```text
DHAN_CLIENT_ID=your_client_id
DHAN_API_KEY=your_api_key
DHAN_API_SECRET=your_api_secret
DHAN_REDIRECT_URL=http://127.0.0.1:8765/callback
```

Never put these values in source control or paste them into chat.

4. Generate the access token from the local project:

```bash
python dhan_auth.py login
python dhan_auth.py status
```

The login opens a browser, completes Dhan's credential/2FA flow, captures the localhost `tokenId`, exchanges it for an access token, and stores the token locally in `data/dhan/auth.json` with restrictive permissions.

The Streamlit sidebar also has **Authenticate / refresh Dhan token** once key/secret/client-id are configured.

### Data provider priority

For NSE symbols ending in `.NS`, the dashboard now attempts Dhan first when a valid local Dhan token is available. If Dhan is unavailable or a request fails, the app falls back to the existing yfinance/free-source path rather than silently claiming Dhan data were used.

No order endpoint is called by this project. Dhan data-plan status is visible in the sidebar via the profile API.

## DhanHQ preferred data provider

This edition integrates DhanHQ v2 as the preferred NSE cash-data provider when a valid local Dhan access token is available. The app remains signals-only and does not call Dhan order endpoints.

Dhan documents that its intraday historical endpoint supports 1/5/15/25/60-minute OHLCV/OI for active instruments for up to five years, with a maximum 90-day polling window per request. This project automatically chunks a longer range and stores the result locally. Dhan also documents daily history and expired-options data. See the official documentation before enabling a paid Data API plan. 

### Required local credentials

The Dhan API-key flow requires all three of these locally:

```text
DHAN_CLIENT_ID
DHAN_API_KEY
DHAN_API_SECRET
```

The API key/secret must be generated in Dhan's API access page. Do not paste credentials into ChatGPT or commit them to source control.

Copy the template:

```bash
cp .env.example .env
```

Then edit `.env` on your laptop.

### Redirect URL

For the built-in local browser authentication helper, configure the Dhan API application redirect URL as:

```text
http://127.0.0.1:8765/callback
```

Then run:

```bash
python dhan_auth.py login
```

The helper calls Dhan's documented consent endpoint, opens the Dhan browser login, captures the localhost `tokenId` callback, exchanges it for an access token, and saves the token locally in `data/dhan/auth.json`.

Check the account/data-plan status with:

```bash
python dhan_auth.py status
```

The profile response includes Dhan's data-plan status/validity. This application treats a missing/invalid Dhan token as an unavailable Dhan source rather than reporting it as a successful connection.

### Dhan historical downloader

From the project directory:

```bash
python dhan_download.py --symbols SBIN BHARTIARTL RELIANCE --start 2021-01-01 --end 2026-09-26 --interval 5m
```

For 1-minute data:

```bash
python dhan_download.py --symbols SBIN BHARTIARTL RELIANCE --start 2024-09-26 --end 2026-09-26 --interval 1m
```

The Streamlit sidebar also has a **DhanHQ data provider** section and a **Download selected history from Dhan** control. The downloaded Parquet files are stored under `data/historical/dhan/NSE_EQ/` with a local manifest.

### Provider priority

For NSE symbols ending in `.NS`, `data_fetcher.py` attempts Dhan first when a valid local access token is available. If that call fails, the application falls back to the existing free/yfinance route. The UI text should therefore show when Dhan is connected; it does not silently claim Dhan was used if it was not.

### Live layer

Current LTP can be fetched through Dhan's Market Quote API. The separate Dhan WebSocket feed remains the next live-streaming layer; this build deliberately does not place orders and does not require trading API permissions. Dhan documents real-time LTP/quote APIs and a separate WebSocket live market feed.

## Permanent Python environment (recommended)

This project is designed so you do **not** need to activate a virtual environment manually.

Run once:

```bash
./setup.sh
```

Then start the dashboard any time with:

```bash
./run.sh
```

Run Python scripts through the private environment with:

```bash
./py.sh self_test.py
./py.sh dhan_auth.py status
```

Check the environment:

```bash
./doctor.sh
```

Repair the private environment if it is ever damaged:

```bash
./repair_python.sh
```

The scripts always use `.venv/bin/python`, so a missing/broken `venv/bin/activate` or an activated system Python cannot cause the recurring `ModuleNotFoundError` problem.

## Local historical data you provide

Place CSV, CSV.GZ, or Parquet historical datasets under:

```text
data/inbox/
```

Do not put API keys, API secrets, access tokens, or passwords in this directory.
### Python environment reliability

The project uses a private `.venv` and the launchers `./run` and `./py.sh`; do not rely on shell activation. If `.venv` is missing or incomplete, the setup/repair scripts recreate it automatically. The audit Spearman calculation is implemented without SciPy, so SciPy is not an undeclared runtime dependency. Paths containing spaces (such as `~/Documents/Direct Applications`) are handled by the setup script.

### Environment note
Use `./run` / `./run.sh` for the dashboard and `./py.sh <script>` for Python commands. The project uses `.venv` automatically and checks requirements by hash. The fast/deep Parquet audit requires `fsspec[http]`, which is declared in `requirements.txt`.

## Step 52 model-readiness gate

The 2026-09-25 PIT walk-forward run covered the frozen pilot symbols
BHARTIARTL, RELIANCE and SBIN (41,145 joined rows and 87 predictors). Across
six expanding-window holdouts, ROC AUC ranged from 0.486 to 0.522 and balanced
accuracy from 0.488 to 0.529. Those results are close to chance, so this model
is marked **research only** and is not connected to live dashboard signals.

The dashboard's live BUY/SELL/HOLD label remains a rule-based blend of price,
current RSS headline sentiment, broad-market regime and event-risk dampening.
The headline sentiment is shown as a perception score with its coverage and
freshness; it is not a probability. The 1-year historical audit remains a
price/market-only validation because the live RSS feed has no corresponding
point-in-time news archive. No orders are placed.

## Permanent 1-minute historical store (v4)

The deployed app now treats the Streamlit filesystem as an ephemeral cache and maintains an append-oriented local store under `data/persistent_market_data/intraday_1m/YYYY/MM/DD/SYMBOL.parquet`. The same store can be mirrored to Google Drive.

For Streamlit Cloud, configure a Drive service account in App Settings -> Secrets using the structure in `STREAMLIT_SECRETS_GDRIVE.example.toml`, and share the chosen Drive root folder with that service-account email as Editor. See `PERSISTENT_DATA_SETUP.md` for the full procedure.

The new sidebar controls are:
- Capture current 1-minute data
- Sync local 1-minute archive to Google Drive
- Scan permanent 1-minute archive

Dhan 1-minute downloads and current-year free 1-minute archive downloads are automatically partitioned into the persistent store. Historical replay reads the persistent store first.

## All-NSE Delivery training policy

Future next-session Delivery candidate training uses the current **official NSE
securities-available-for-trading equity universe** as the training roster. The
training job reads historical observations from the persistent official NSE
CM-UDiFF bulk EOD archive and includes every roster symbol with sufficient clean
history; symbols lacking enough history are reported as excluded rather than
silently replaced by another data source.

The candidate builder also restores official NSE NIFTY 50 / NIFTY BANK daily market
context and creates the relative-return, market-RSI, mood-proxy and stress-proxy
features point-in-time. News, F&O and fundamentals remain explicit missing inputs
until timestamped point-in-time datasets are available.

All-NSE candidate artifacts are written to a separate candidate directory and are
**not automatically promoted** to production. Production continues to use the
small certified artifact until the broader candidate has passed the required
walk-forward and live D+1 validation gates.

## NSE-first data layer (v5)

The production data policy is now **NSE-first**. The current NSE site exposes
CSV downloads for security-wise historical price/volume data and historical
contract-wise F&O data, plus corporate-actions/announcements interfaces.
The dashboard therefore treats NSE as the primary exchange-data source and
stores normalized NSE observations in the persistent Google Drive archive.
See `NSE_DATA_LAYER_ARCHITECTURE.md`.

### Training/deployment split

- **Google Drive:** historical NSE data and durable prediction/data ledgers.
- **Google Colab:** training, compilation and validation only.
- **Private GitHub:** source code + small model artifacts/manifests; no large datasets and no secrets.
- **Streamlit:** model serving, dashboard, live/current acquisition and replay; no training.

`colab_build_nse_model.py` prepares a chronological 30-minute Logistic Regression
candidate with an explicit and tested target convention. It does not auto-promote
it to production.

### Official-source caveat

NSE's public website/report interfaces are the primary source in this project,
but public web endpoints can be rate-limited or changed and are not a substitute
for a separately licensed market-data feed where NSE's terms require one.
The project caches, slows and records provenance, and does not attempt to bypass
anti-bot controls.
