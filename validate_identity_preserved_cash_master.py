#!/usr/bin/env python3
"""Read-only validation of the canonical NSE cash identity-preserved master."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

DEFAULT_MASTER = Path(
    "/content/drive/MyDrive/Indian_Market_Research/production_data/"
    "nse_official/ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
)
DEFAULT_REPORT_DIR = Path(
    "/content/drive/MyDrive/Indian_Market_Research/diagnostics/"
    "canonical_cash_master_validation"
)

# The historical CM master intentionally contains multiple NSE security-series
# classes. Equity-model eligibility is handled downstream; this validator must
# not reject legitimate non-EQ historical series.
KNOWN_HISTORICAL_SERIES = {
    "EQ","BE","BZ","SM","ST","SZ","GB","RR","NA","GS","TB","SG","MF","IV","ID","SO",
    "ME","RT","EX","EY","EZ","E0","E1","E2","E3","P1","P2","Q1","Q2","X1","X2","Y0",
    "Y1","Y2","Y3","Y4","Y5","Y6","Y7","Y8","Y9","Z0","Z1","Z2","Z3","Z4","Z5","Z6",
    "Z7","Z8","Z9",
}

def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

def sql_str(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"

def resolve_col(columns, aliases) -> str | None:
    exact = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in exact:
            return exact[alias.lower()]
    return None

def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 4) if d else 0.0

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=str(DEFAULT_MASTER))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument(
        "--fno-path",
        default="/content/drive/MyDrive/Indian_Market_Research/"
                "production_data/nse_official/fno/fno_daily_features_2026.parquet",
    )
    args = parser.parse_args()

    master = Path(args.master)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("NSE CASH — CANONICAL IDENTITY-PRESERVED MASTER VALIDATION")
    print("=" * 100)
    print("Master       :", master)
    print("Report dir   :", report_dir)

    if not master.exists() or master.stat().st_size == 0:
        print("STATUS       : FAIL — master missing or empty")
        return 2

    con = duckdb.connect()
    con.execute("PRAGMA threads=8")

    src = f"read_parquet({sql_str(master)})"
    schema = con.execute(f"DESCRIBE SELECT * FROM {src}").df()
    columns = [str(x) for x in schema["column_name"].tolist()]

    required = {
        "symbol": resolve_col(columns, ("symbol",)),
        "date": resolve_col(columns, ("date","trade_date","trading_date")),
        "series": resolve_col(columns, ("series",)),
        "listing_id": resolve_col(columns, ("listing_id",)),
        "row_identity_key": resolve_col(columns, ("row_identity_key",)),
    }
    missing = [k for k,v in required.items() if v is None]
    if missing:
        raise RuntimeError(f"Required master columns missing: {missing}")

    sc, dc, sec, lic, ric = (
        required["symbol"], required["date"], required["series"],
        required["listing_id"], required["row_identity_key"]
    )

    total = int(con.execute(f"SELECT COUNT(*) FROM {src}").fetchone()[0])
    nulls = {
        name: int(con.execute(
            f"SELECT COUNT(*) FROM {src} WHERE {qident(col)} IS NULL"
        ).fetchone()[0])
        for name,col in required.items()
    }

    dup_identity_groups = int(con.execute(
        f"SELECT COUNT(*) FROM (SELECT {qident(ric)} FROM {src} "
        f"WHERE {qident(ric)} IS NOT NULL GROUP BY {qident(ric)} "
        f"HAVING COUNT(*) > 1)"
    ).fetchone()[0])

    blank_identity = int(con.execute(
        f"SELECT COUNT(*) FROM {src} WHERE {qident(ric)} IS NULL "
        f"OR TRIM(CAST({qident(ric)} AS VARCHAR))=''"
    ).fetchone()[0])

    min_date, max_date, distinct_dates, invalid_dates = con.execute(
        f"SELECT MIN(TRY_CAST({qident(dc)} AS DATE)), "
        f"MAX(TRY_CAST({qident(dc)} AS DATE)), "
        f"COUNT(DISTINCT TRY_CAST({qident(dc)} AS DATE)), "
        f"SUM(CASE WHEN TRY_CAST({qident(dc)} AS DATE) IS NULL THEN 1 ELSE 0 END) "
        f"FROM {src}"
    ).fetchone()

    distinct_symbols = int(con.execute(
        f"SELECT COUNT(DISTINCT NULLIF(TRIM(CAST({qident(sc)} AS VARCHAR)),'')) FROM {src}"
    ).fetchone()[0])
    distinct_listing_ids = int(con.execute(
        f"SELECT COUNT(DISTINCT NULLIF(TRIM(CAST({qident(lic)} AS VARCHAR)),'')) FROM {src}"
    ).fetchone()[0])

    series_df = con.execute(
        f"SELECT UPPER(TRIM(CAST({qident(sec)} AS VARCHAR))) AS series, COUNT(*) AS row_count "
        f"FROM {src} GROUP BY 1 ORDER BY 2 DESC"
    ).df()
    series_counts = {
        ("" if pd.isna(r["series"]) else str(r["series"])): int(r["row_count"])
        for _,r in series_df.iterrows()
    }
    unexpected_series = [
        s for s in series_counts
        if s and s not in KNOWN_HISTORICAL_SERIES
    ]

    cols = {k: resolve_col(columns, aliases) for k,aliases in {
        "open":("open","open_price"),
        "high":("high","high_price"),
        "low":("low","low_price"),
        "close":("close","close_price","last_price","ltp"),
        "volume":("volume","trade_volume","total_traded_quantity","tottrdqty"),
        "delivery_qty":("delivery_qty","deliverable_qty","delivery_quantity","deliv_qty"),
        "delivery_pct":("delivery_pct","delivery_percentage","deliverable_pct","deliv_per"),
    }.items()}

    numeric_checks = {}
    for logical,col in cols.items():
        if not col:
            continue
        non_numeric = int(con.execute(
            f"SELECT COUNT(*) FROM {src} WHERE {qident(col)} IS NOT NULL "
            f"AND TRY_CAST({qident(col)} AS DOUBLE) IS NULL"
        ).fetchone()[0])
        negative = int(con.execute(
            f"SELECT COUNT(*) FROM {src} "
            f"WHERE TRY_CAST({qident(col)} AS DOUBLE) < 0"
        ).fetchone()[0])
        numeric_checks[logical] = {
            "column": col,
            "non_numeric_non_null": non_numeric,
            "negative_values": negative,
        }

    cross_price = {k: None for k in (
        "high_below_low","close_below_low","close_above_high",
        "open_below_low","open_above_high"
    )}
    if all(cols.get(k) for k in ("open","high","low","close")):
        oc,hc,lc,cc = (cols[k] for k in ("open","high","low","close"))
        common = (
            f" FROM {src} WHERE TRY_CAST({qident(hc)} AS DOUBLE) IS NOT NULL "
            f"AND TRY_CAST({qident(lc)} AS DOUBLE) IS NOT NULL "
            f"AND TRY_CAST({qident(cc)} AS DOUBLE) IS NOT NULL"
        )
        cross_price["high_below_low"] = int(con.execute(
            f"SELECT COUNT(*){common} AND TRY_CAST({qident(hc)} AS DOUBLE) < TRY_CAST({qident(lc)} AS DOUBLE)"
        ).fetchone()[0])
        cross_price["close_below_low"] = int(con.execute(
            f"SELECT COUNT(*){common} AND TRY_CAST({qident(cc)} AS DOUBLE) < TRY_CAST({qident(lc)} AS DOUBLE)"
        ).fetchone()[0])
        cross_price["close_above_high"] = int(con.execute(
            f"SELECT COUNT(*){common} AND TRY_CAST({qident(cc)} AS DOUBLE) > TRY_CAST({qident(hc)} AS DOUBLE)"
        ).fetchone()[0])
        cross_price["open_below_low"] = int(con.execute(
            f"SELECT COUNT(*){common} AND TRY_CAST({qident(oc)} AS DOUBLE) < TRY_CAST({qident(lc)} AS DOUBLE)"
        ).fetchone()[0])
        cross_price["open_above_high"] = int(con.execute(
            f"SELECT COUNT(*){common} AND TRY_CAST({qident(oc)} AS DOUBLE) > TRY_CAST({qident(hc)} AS DOUBLE)"
        ).fetchone()[0])

    na_rows = int(con.execute(
        f"SELECT COUNT(*) FROM {src} WHERE UPPER(TRIM(CAST({qident(sec)} AS VARCHAR)))='NA'"
    ).fetchone()[0])
    na_bad_identity = int(con.execute(
        f"SELECT COUNT(*) FROM {src} WHERE UPPER(TRIM(CAST({qident(sec)} AS VARCHAR)))='NA' "
        f"AND ({qident(ric)} IS NULL OR TRIM(CAST({qident(ric)} AS VARCHAR))='')"
    ).fetchone()[0])

    # IMPORTANT: many historical listings use symbol/date duplicates across
    # preserved series/listing identities. This is not a defect.
    symbol_date_multi = int(con.execute(
        f"SELECT COUNT(*) FROM (SELECT TRIM(CAST({qident(sc)} AS VARCHAR)), "
        f"TRY_CAST({qident(dc)} AS DATE) FROM {src} GROUP BY 1,2 HAVING COUNT(*)>1)"
    ).fetchone()[0])

    # The F&O layer is contextual, not a completeness gate for cash history.
    fno = {"exists":False}
    fno_path = Path(args.fno_path)
    if fno_path.exists() and fno_path.stat().st_size:
        try:
            fschema = con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_str(fno_path)})").df()
            fcols=[str(x) for x in fschema["column_name"].tolist()]
            fsc=resolve_col(fcols,("symbol","underlying","undrlyg"))
            if fsc:
                cash_set=set(con.execute(
                    f"SELECT DISTINCT UPPER(TRIM(CAST({qident(sc)} AS VARCHAR))) FROM {src} "
                    f"WHERE {qident(sc)} IS NOT NULL"
                ).df().iloc[:,0].astype(str))
                fno_set=set(con.execute(
                    f"SELECT DISTINCT UPPER(TRIM(CAST({qident(fsc)} AS VARCHAR))) FROM read_parquet({sql_str(fno_path)}) "
                    f"WHERE {qident(fsc)} IS NOT NULL"
                ).df().iloc[:,0].astype(str))
                fno={
                    "exists":True,
                    "fno_symbol_count":len(fno_set),
                    "cash_fno_symbol_overlap":len(cash_set & fno_set),
                    "cash_symbols_with_fno_pct":pct(len(cash_set & fno_set),len(cash_set)),
                }
        except Exception as exc:
            fno={"exists":True,"status":f"AUDIT_ERROR: {type(exc).__name__}: {exc}"}

    hard_failures=[]
    if total<=0: hard_failures.append("EMPTY_MASTER")
    for k in ("series","listing_id","row_identity_key"):
        if nulls[k]!=0: hard_failures.append(f"NULL_{k.upper()}_REMAINS")
    if dup_identity_groups!=0: hard_failures.append("DUPLICATE_ROW_IDENTITY_KEYS")
    if blank_identity!=0: hard_failures.append("BLANK_ROW_IDENTITY")
    if invalid_dates!=0: hard_failures.append("INVALID_DATES")
    # Series codes are informational. Do not fail the historical master merely
    # because NSE used an instrument series not needed by the equity model.
    for logical,chk in numeric_checks.items():
        if chk["non_numeric_non_null"]!=0:
            hard_failures.append(f"NON_NUMERIC_{logical.upper()}")
        if logical in {"open","high","low","close","volume","delivery_qty"} and chk["negative_values"]!=0:
            hard_failures.append(f"NEGATIVE_{logical.upper()}")
    for k,n in cross_price.items():
        if n not in (None,0): hard_failures.append(k.upper())

    # Write a small sample of rows participating in OHLC contradictions so
    # investigation does not require rerunning a full scan.
    anomaly_path = report_dir / "ohlc_anomalies_sample.csv"
    if any(v not in (None,0) for v in cross_price.values()) and all(cols.get(k) for k in ("open","high","low","close")):
        oc,hc,lc,cc=(cols[k] for k in ("open","high","low","close"))
        where=(
            f"TRY_CAST({qident(cc)} AS DOUBLE)<TRY_CAST({qident(lc)} AS DOUBLE) "
            f"OR TRY_CAST({qident(cc)} AS DOUBLE)>TRY_CAST({qident(hc)} AS DOUBLE) "
            f"OR TRY_CAST({qident(oc)} AS DOUBLE)<TRY_CAST({qident(lc)} AS DOUBLE) "
            f"OR TRY_CAST({qident(oc)} AS DOUBLE)>TRY_CAST({qident(hc)} AS DOUBLE)"
        )
        con.execute(
            f"SELECT {qident(sc)} AS symbol,{qident(dc)} AS trade_date,{qident(sec)} AS series,"
            f"{qident(oc)} AS open_price,{qident(hc)} AS high_price,{qident(lc)} AS low_price,{qident(cc)} AS close_price,"
            f"{qident(lic)} AS listing_identity FROM {src} WHERE {where} LIMIT 5000"
        ).df().to_csv(anomaly_path,index=False)

    report={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "master":str(master),
        "rows":total,
        "distinct_symbols":distinct_symbols,
        "distinct_listing_ids":distinct_listing_ids,
        "distinct_trading_dates":int(distinct_dates),
        "date_range":{"min":str(min_date),"max":str(max_date)},
        "identity":{"null_counts":nulls,"duplicate_identity_key_groups":dup_identity_groups,"blank_identity_rows":blank_identity},
        "series":{
            "counts":series_counts,
            "known_historical_series_allowlist":sorted(KNOWN_HISTORICAL_SERIES),
            "unexpected_series_in_allowlist_check":unexpected_series,
            "literal_NA_rows":na_rows,
            "literal_NA_rows_with_null_identity":na_bad_identity,
            "note":"Historical CM contains multiple security-series classes; downstream equity universe filtering must use explicit security-class policy."
        },
        "symbol_date_multirow_groups":symbol_date_multi,
        "numeric_checks":numeric_checks,
        "cross_price_checks":cross_price,
        "ohlc_anomaly_sample":str(anomaly_path) if anomaly_path.exists() else None,
        "fno_context":fno,
        "hard_failures":hard_failures,
        "status":"PASS" if not hard_failures else "FAIL",
    }
    report_path=report_dir/"identity_preserved_cash_master_validation_latest.json"
    report_path.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")

    print()
    print("="*100)
    print("CANONICAL CASH MASTER VALIDATION")
    print("="*100)
    print(f"Rows                    : {total:,}")
    print(f"Distinct symbols        : {distinct_symbols:,}")
    print(f"Distinct listing IDs    : {distinct_listing_ids:,}")
    print(f"Distinct trading dates : {int(distinct_dates):,}")
    print(f"Date range              : {min_date} -> {max_date}")
    print(f"NULL series             : {nulls['series']:,}")
    print(f"NULL listing_id         : {nulls['listing_id']:,}")
    print(f"NULL row_identity_key   : {nulls['row_identity_key']:,}")
    print(f"Duplicate identity keys : {dup_identity_groups:,}")
    print(f"Blank/invalid identity  : {blank_identity:,}")
    print(f"Literal NA series rows  : {na_rows:,}")
    print(f"NA rows with NULL ID    : {na_bad_identity:,}")
    print(f"Historical series outside validator allowlist: {unexpected_series}")
    print(f"Symbol/date multi-rows  : {symbol_date_multi:,} (reported)")
    print(f"F&O layer available     : {fno.get('exists',False)}")
    if fno.get("cash_fno_symbol_overlap") is not None:
        print(f"Cash/F&O symbol overlap : {fno['cash_fno_symbol_overlap']:,}")
        print(f"Cash symbols with F&O % : {fno['cash_symbols_with_fno_pct']:.2f}%")
    print(f"OHLC anomaly sample     : {anomaly_path if anomaly_path.exists() else 'not created'}")
    print(f"Report                  : {report_path}")
    if hard_failures:
        print("HARD FAILURES           :", ", ".join(hard_failures))
        print("STATUS                  : FAIL")
        return 1
    print("STATUS                  : PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
