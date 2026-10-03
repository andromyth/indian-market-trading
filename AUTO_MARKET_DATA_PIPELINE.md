# ALL-NSE automatic market-data pipeline

This branch is the replacement-production branch. The data plane is designed
around the complete current NSE equity universe rather than the original
three-stock pilot.

## Daily cash/equity

The existing nse_eod_sync.yml keeps the official NSE CM-UDiFF daily archive
current under production_data/nse_official/bulk_eod. The ALL-NSE Delivery
builder consumes that archive.

NSE currently lists the CM-UDiFF Common Bhavcopy Final as the current cash-market
bhavcopy format and the older cash bhavcopy as discontinued.

## Intraday

nse_intraday_all_universe_sync.py walks the current official NSE equity
universe and captures the public NSE quote-chart series as 1-minute OHLC.
persistent_store.py appends each symbol/day partition under
production_data/intraday_1m/YYYY/MM/DD/SYMBOL.parquet.

The public NSE chart path is LTP/time-series based. Minute volume is therefore
kept missing when it is not supplied; no volume is fabricated. This lane is
required for the ALL-NSE Intraday training archive.

## F&O

nse_fno_udiff_ingest.py preserves the official NSE F&O UDiFF Common Bhavcopy
Final archive and maintains the broad symbol-level
fno_daily_features_2026.parquet layer.

The derived layer includes option PCR/OI change and near-future price/OI/basis/
return where the published fields are present.

NSE currently lists F&O UDiFF Common Bhavcopy Final as the current derivatives
bhavcopy format and the older F&O bhavcopy as discontinued.

## Compilation

Delivery training consumes the full EOD universe and available point-in-time
F&O/fundamental/event layers.

Intraday training consumes the persistent 1-minute archive as it accumulates
across the NSE universe.

The model artifacts remain in Google Drive. GitHub stores source code and small
manifests. No order-placement code is introduced.

New NSE data is persisted before model compilation so Colab restarts do not
erase accumulated history.
