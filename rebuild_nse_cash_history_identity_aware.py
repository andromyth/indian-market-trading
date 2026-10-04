#!/usr/bin/env python3
"""
Identity-preserving rebuild of the historical NSE cash layer.

Source:
  * 2016-01-01 .. 2024-07-05 : NSE legacy CM Bhavcopy
  * 2024-07-08 .. onward      : NSE CM-UDiFF Common Bhavcopy Final

Reads ONLY already-downloaded raw archives from Google Drive.
Does not download, train models, modify production model artifacts, or push
anything to main.

Why this exists:
The earlier normalized layer deduplicated on (symbol, date). Raw-audit V2
found 68,640 symbol/date identity collisions and 3,484 ISIN/date collisions.
This rebuild preserves each raw security/listing row and creates explicit
historical identity fields.

Identity fields:
  economic_security_id = ISIN when valid, else SYMBOL fallback
  listing_id           = ISIN|SERIES|SYMBOL when ISIN is valid
                         SYMBOL|SERIES otherwise
  row_identity_key     = listing_id|DATE

The raw row is never silently discarded because of symbol/date collisions.
Exact duplicate raw identity rows are retained in the audit and only exact
byte-equivalent business rows are removed if encountered.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(
    "/content/drive/MyDrive/Indian_Market_Research/"
    "production_data/nse_official"
)

RAW_LEGACY = ROOT / "cash_history_raw" / "legacy"
RAW_UDIFF = ROOT / "cash_history_raw" / "udiff"

OUT_ROOT = ROOT / "cash_history_identity_preserved"
OUT_LEGACY = OUT_ROOT / "legacy"
OUT_UDIFF = OUT_ROOT / "udiff"

MASTER = ROOT / "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet"
META = ROOT / "ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.meta.json"
AUDIT = ROOT / "cash_identity_preserved_rebuild_audit_2016_2026.csv"
CLASS_AUDIT = ROOT / "cash_identity_preserved_classification_2016_2026.csv"

LEGACY_CUTOFF = pd.Timestamp("2024-07-05")
UDIFF_START = pd.Timestamp("2024-07-08")


def read_zip_csv(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        members = [
            n for n in zf.namelist()
            if not n.endswith("/") and n.lower().endswith(".csv")
        ]
        if not members:
            raise RuntimeError(f"No CSV member in {path}")
        raw = zf.read(members[0])

    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                encoding=enc,
                low_memory=False,
            )
        except Exception as exc:
            last = exc
    raise RuntimeError(f"Unable to parse {path}: {last}")


def clean_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.upper()


def valid_isin(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip().str.upper()
    return (
        t.notna()
        & t.ne("")
        & ~t.isin(["NAN", "NONE", "<NA>"])
    )


def classify_series(series: pd.Series) -> pd.Series:
    out = pd.Series("OTHER", index=series.index, dtype="string")
    s = series.astype("string").str.strip().str.upper()

    out.loc[s.isin(["EQ", "BE", "BZ"])] = "FULLY_PAID_EQUITY"
    out.loc[s.isin(["SM", "ST", "SZ"])] = "SME_EQUITY"

    # Partly-paid: E@ / X@ where @ is 0-9 or A-Z.
    # Exact EQ is deliberately excluded.
    pp = s.str.match(r"^[EX][0-9A-Z]$", na=False)
    out.loc[pp & ~s.eq("EQ")] = "PARTLY_PAID_EQUITY"

    mapping = {
        "MF": "MUTUAL_FUND",
        "ME": "MUTUAL_FUND",
        "IV": "INVIT",
        "ID": "INVIT",
        "RR": "REIT",
        "RT": "REIT",
        "GB": "GOLD_BOND",
        "GS": "GOVERNMENT_SECURITY",
        "TB": "TREASURY_BILL",
        "SG": "STATE_DEVELOPMENT_LOAN",
    }
    for code, cls in mapping.items():
        out.loc[s.eq(code)] = cls

    return out


def normalize_legacy(df: pd.DataFrame, archive: Path) -> pd.DataFrame:
    mapping = {
        "SYMBOL": "symbol",
        "SERIES": "series",
        "ISIN": "isin",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "LAST": "last",
        "PREVCLOSE": "prev_close",
        "TOTTRDQTY": "volume",
        "TOTTRDVAL": "traded_value",
        "TIMESTAMP": "date",
        "TOTALTRADES": "num_trades",
    }

    x = df.rename(
        columns={k: v for k, v in mapping.items() if k in df.columns}
    ).copy()

    required = {"symbol", "series", "isin", "date", "close"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(
            f"Legacy missing columns {sorted(missing)} in {archive.name}"
        )

    x["symbol"] = clean_text(x["symbol"])
    x["series"] = clean_text(x["series"])
    x["isin"] = clean_text(x["isin"])
    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
        dayfirst=True,
    ).dt.normalize()

    for c in (
        "open", "high", "low", "close", "last",
        "prev_close", "volume", "traded_value", "num_trades",
    ):
        if c in x.columns:
            x[c] = pd.to_numeric(
                x[c].astype("string").str.replace(",", "", regex=False),
                errors="coerce",
            )

    x["source_format"] = "NSE_LEGACY_CM_BHAVCOPY"
    x["instrument_type"] = "LEGACY_CM"

    return x


def normalize_udiff(df: pd.DataFrame, archive: Path) -> pd.DataFrame:
    mapping = {
        "TradDt": "date",
        "BizDt": "business_date",
        "Sgmt": "segment",
        "Src": "source",
        "FinInstrmTp": "instrument_type",
        "FinInstrmId": "instrument_id",
        "ISIN": "isin",
        "TckrSymb": "symbol",
        "SctySrs": "series",
        "FininstrmActlXpryDt": "actual_expiry",
        "FininstrmNm": "instrument_name",
        "OpnPric": "open",
        "HghPric": "high",
        "LwPric": "low",
        "ClsPric": "close",
        "LastPric": "last",
        "PrvsClsgPric": "prev_close",
        "TtlTradgVol": "volume",
        "TtlTrfVal": "traded_value",
        "TtlNbOfTxsExctd": "num_trades",
    }

    x = df.rename(
        columns={k: v for k, v in mapping.items() if k in df.columns}
    ).copy()

    required = {
        "symbol", "series", "isin",
        "instrument_type", "date", "close",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(
            f"UDiFF missing columns {sorted(missing)} in {archive.name}"
        )

    # Preserve only stock instruments in the equity-capable CM UDiFF source.
    x = x[
        clean_text(x["instrument_type"]).eq("STK")
    ].copy()

    x["symbol"] = clean_text(x["symbol"])
    x["series"] = clean_text(x["series"])
    x["isin"] = clean_text(x["isin"])
    x["instrument_type"] = clean_text(x["instrument_type"])
    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    ).dt.normalize()

    if "business_date" in x.columns:
        x["business_date"] = pd.to_datetime(
            x["business_date"],
            errors="coerce",
        ).dt.normalize()

    for c in (
        "open", "high", "low", "close", "last",
        "prev_close", "volume", "traded_value", "num_trades",
    ):
        if c in x.columns:
            x[c] = pd.to_numeric(
                x[c].astype("string").str.replace(",", "", regex=False),
                errors="coerce",
            )

    x["source_format"] = "NSE_CM_UDIFF"

    return x


def add_identity(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()

    x["security_class"] = classify_series(x["series"])

    isin_ok = valid_isin(x["isin"])

    x["economic_security_id"] = x["symbol"].astype("string")
    x.loc[isin_ok, "economic_security_id"] = (
        "ISIN:" + x.loc[isin_ok, "isin"].astype("string")
    )

    x["listing_id"] = (
        "SYMBOL:"
        + x["symbol"].astype("string")
        + "|SERIES:"
        + x["series"].astype("string")
    )

    x.loc[isin_ok, "listing_id"] = (
        "ISIN:"
        + x.loc[isin_ok, "isin"].astype("string")
        + "|SERIES:"
        + x.loc[isin_ok, "series"].astype("string")
        + "|SYMBOL:"
        + x.loc[isin_ok, "symbol"].astype("string")
    )

    x["row_identity_key"] = (
        x["listing_id"].astype("string")
        + "|DATE:"
        + x["date"].dt.strftime("%Y-%m-%d").astype("string")
    )

    # LAST=0 is retained but treated as unavailable for feature calculations.
    x["last_price_zero"] = (
        x["last"].notna() & (pd.to_numeric(x["last"], errors="coerce") <= 0)
    )
    x["last_price_for_research"] = pd.to_numeric(
        x["last"], errors="coerce"
    ).where(
        pd.to_numeric(x["last"], errors="coerce") > 0,
        pd.NA,
    )
    x["research_price"] = (
        x["last_price_for_research"]
        .fillna(pd.to_numeric(x["close"], errors="coerce"))
    )

    x["raw_archive"] = pd.NA
    return x


def final_columns(x: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "symbol", "date", "series", "isin",
        "economic_security_id", "listing_id", "row_identity_key",
        "security_class",
        "instrument_type", "source_format",
        "instrument_id", "instrument_name",
        "business_date", "segment", "source",
        "actual_expiry",
        "open", "high", "low", "close", "last", "prev_close",
        "last_price_zero", "last_price_for_research", "research_price",
        "volume", "traded_value", "num_trades",
        "raw_archive",
    ]

    for c in preferred:
        if c not in x.columns:
            x[c] = pd.NA

    return x[preferred].copy()


def write_partition(x: pd.DataFrame) -> None:
    if x.empty:
        return

    year = int(pd.Timestamp(x["date"].iloc[0]).year)
    month = int(pd.Timestamp(x["date"].iloc[0]).month)

    source_dir = (
        OUT_LEGACY if x["source_format"].iloc[0] == "NSE_LEGACY_CM_BHAVCOPY"
        else OUT_UDIFF
    )

    path = (
        source_dir
        / f"{year:04d}"
        / f"{month:02d}"
        / f"{year:04d}{month:02d}.parquet"
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    # Multiple archive files can contribute to the same month.
    if path.exists():
        old = pd.read_parquet(path)
        x = pd.concat([old, x], ignore_index=True, sort=False)

    # Never dedupe on symbol/date. Use full historical listing identity.
    x = (
        x.drop_duplicates(
            subset=["row_identity_key"],
            keep="last",
        )
        .sort_values(["date", "listing_id"])
        .reset_index(drop=True)
    )

    tmp = path.with_suffix(".tmp.parquet")
    x.to_parquet(tmp, index=False)
    tmp.replace(path)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    archives = []
    archives.extend(
        ("LEGACY", p)
        for p in RAW_LEGACY.rglob("*.zip")
    )
    archives.extend(
        ("UDIFF", p)
        for p in RAW_UDIFF.glob("*.zip")
    )
    archives.sort(key=lambda item: str(item[1]))

    print("=" * 100)
    print("NSE CASH — IDENTITY-PRESERVING REBUILD")
    print("=" * 100)
    print("Legacy archives :", sum(s == "LEGACY" for s, _ in archives))
    print("UDiFF archives  :", sum(s == "UDIFF" for s, _ in archives))
    print("Total archives  :", len(archives))
    print()

    audit_rows = []
    class_rows = []

    for i, (source, archive) in enumerate(archives, 1):
        try:
            raw = read_zip_csv(archive)

            if source == "LEGACY":
                x = normalize_legacy(raw, archive)
            else:
                x = normalize_udiff(raw, archive)

            x = add_identity(x)
            x["raw_archive"] = str(archive)

            # Preserve raw identity rows. Only remove an exact repeat of the
            # complete identity key inside the same archive.
            exact_identity_duplicates = int(
                x.duplicated(
                    subset=["row_identity_key"],
                    keep=False,
                ).sum()
            )

            exact_identity_keys = int(
                x["row_identity_key"].duplicated().sum()
            )

            # Write by calendar month so the rebuild is resumable and memory
            # usage stays bounded.
            for month_value, part in x.groupby(
                x["date"].dt.to_period("M"),
                sort=True
            ):
                write_partition(
                    final_columns(part)
                )

            counts = (
                x.groupby("security_class")
                 .agg(
                     rows=("symbol", "size"),
                     symbols=("symbol", "nunique"),
                     securities=("economic_security_id", "nunique"),
                 )
                 .reset_index()
            )
            counts["source"] = source
            counts["archive"] = str(archive)
            class_rows.append(counts)

            audit_rows.append({
                "source": source,
                "archive": str(archive),
                "date_min": str(x["date"].min().date())
                    if x["date"].notna().any() else "",
                "date_max": str(x["date"].max().date())
                    if x["date"].notna().any() else "",
                "raw_rows": int(len(raw)),
                "normalized_rows": int(len(x)),
                "unique_row_identity_keys": int(
                    x["row_identity_key"].nunique()
                ),
                "identity_duplicate_rows": exact_identity_duplicates,
                "identity_duplicate_keys": exact_identity_keys,
                "symbol_count": int(x["symbol"].nunique()),
                "isin_count": int(
                    x.loc[valid_isin(x["isin"]), "isin"].nunique()
                ),
                "be_rows": int(x["series"].eq("BE").sum()),
                "zero_last_rows": int(x["last_price_zero"].sum()),
                "status": "OK",
                "error": "",
            })

        except Exception as exc:
            audit_rows.append({
                "source": source,
                "archive": str(archive),
                "date_min": "",
                "date_max": "",
                "raw_rows": 0,
                "normalized_rows": 0,
                "unique_row_identity_keys": 0,
                "identity_duplicate_rows": 0,
                "identity_duplicate_keys": 0,
                "symbol_count": 0,
                "isin_count": 0,
                "be_rows": 0,
                "zero_last_rows": 0,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            })

        if (
            i % 100 == 0
            or i == len(archives)
        ):
            print(f"Processed {i:,}/{len(archives):,}")

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(AUDIT, index=False)

    if class_rows:
        class_df = (
            pd.concat(class_rows, ignore_index=True)
              .groupby("security_class", as_index=False)
              .agg(
                  rows=("rows", "sum"),
                  symbols=("symbols", "sum"),
                  securities=("securities", "sum"),
              )
              .sort_values("rows", ascending=False)
        )
    else:
        class_df = pd.DataFrame(
            columns=[
                "security_class",
                "rows",
                "symbols",
                "securities",
            ]
        )

    class_df.to_csv(CLASS_AUDIT, index=False)

    # Build a compact master from the monthly identity-preserved partitions.
    monthly = sorted(
        list(OUT_LEGACY.glob("*/*/*.parquet"))
        + list(OUT_UDIFF.glob("*/*/*.parquet"))
    )

    if not monthly:
        raise RuntimeError("No rebuilt monthly parquet partitions were created.")

    pieces = [
        pd.read_parquet(p)
        for p in monthly
    ]

    master = (
        pd.concat(
            pieces,
            ignore_index=True,
            sort=False,
        )
        .drop_duplicates(
            subset=["row_identity_key"],
            keep="last",
        )
        .sort_values(["date", "listing_id"])
        .reset_index(drop=True)
    )

    tmp_master = MASTER.with_suffix(".tmp.parquet")
    master.to_parquet(tmp_master, index=False)
    tmp_master.replace(MASTER)

    duplicate_master = int(
        master["row_identity_key"].duplicated().sum()
    )

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NSE_OFFICIAL_CM",
        "legacy_window": "2016-01-01 through 2024-07-05",
        "udiff_window": "2024-07-08 through 2026-10-01 expected",
        "raw_archives_scanned": int(len(archives)),
        "successful_archives": int(
            (audit_df["status"] == "OK").sum()
        ),
        "error_archives": int(
            (audit_df["status"] == "ERROR").sum()
        ),
        "rows": int(len(master)),
        "symbols": int(master["symbol"].nunique()),
        "economic_securities": int(
            master["economic_security_id"].nunique()
        ),
        "listing_ids": int(
            master["listing_id"].nunique()
        ),
        "dates": int(master["date"].nunique()),
        "min_date": str(master["date"].min()),
        "max_date": str(master["date"].max()),
        "duplicate_row_identity_keys": duplicate_master,
        "master": str(MASTER),
        "audit": str(AUDIT),
        "classification_audit": str(CLASS_AUDIT),
        "note": (
            "Identity-preserved historical cash layer. "
            "No symbol/date deduplication is used."
        ),
    }

    META.write_text(
        json.dumps(
            meta,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("REBUILD COMPLETE")
    print("=" * 100)
    print("Rows                  :", f"{len(master):,}")
    print("Symbols               :", f"{master['symbol'].nunique():,}")
    print("Economic securities   :", f"{master['economic_security_id'].nunique():,}")
    print("Listing identities    :", f"{master['listing_id'].nunique():,}")
    print("Dates                 :", f"{master['date'].nunique():,}")
    print("Min date              :", master["date"].min())
    print("Max date              :", master["date"].max())
    print("Duplicate row keys    :", f"{duplicate_master:,}")
    print("Archive errors        :", f"{(audit_df['status'] == 'ERROR').sum():,}")
    print()
    print("Master:", MASTER)
    print("Meta  :", META)
    print("Audit :", AUDIT)
    print("Class :", CLASS_AUDIT)


if __name__ == "__main__":
    main()
