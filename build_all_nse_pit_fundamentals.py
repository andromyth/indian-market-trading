#!/usr/bin/env python3
"""Build a broad ALL-NSE PIT fundamentals layer from NSE Integrated Filing XBRL.

The source is NSE's Integrated Filing catalog plus the direct XBRL URLs hosted on
nsearchives.nseindia.com. Existing Drive PIT files are retained as a seed, but
the new layer is not restricted to the legacy three-stock pilot.

PIT rule: fundamental_available_at is the filing/broadcast timestamp. The
candidate builder may use a fundamental row only when that timestamp is at or
before the decision_timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from delivery_training_universe import training_roster
from multifactor_data_layer_v2 import normalize_symbol

BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
DEFAULT_BULK = BASE / "production_data/nse_official/bulk_eod"
DEFAULT_OUT = BASE / "corporate/pit_fundamentals_all_nse_2025_2026.parquet"
CACHE_ROOT = BASE / "corporate/xbrl_cache/all_nse_2025_2026"

NSE_HOME = "https://www.nseindia.com/"
NSE_API = "https://www.nseindia.com/api/integrated-filing-results"

FIELD_ALIASES = {
    "fundamental_revenue": (
        "RevenueFromOperations",
        "RevenueFromOperationsNetOfExciseDuty",
        "InterestEarned",
        "GrossPremiumIncome",
        "GrossPremiumsWritten",
    ),
    "fundamental_pat": (
        "ProfitLossForPeriod",
        "ProfitLossForThePeriod",
        "ProfitLossAfterTax",
        "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
        "ProfitLossFromOrdinaryActivitiesAfterTax",
        "ProfitLossAfterTaxAndExtraordinaryItems",
        "ProfitLossAfterTaxBeforeExtraordinaryItems",
    ),
    "fundamental_eps": (
        "BasicAndDilutedEPSAfterExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized",
        "BasicEarningsPerShare",
        "DilutedEarningsPerShare",
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
    ),
    "fundamental_finance_cost": ("FinanceCosts", "FinanceCost"),
    "fundamental_total_equity": (
        "TotalEquity",
        "Equity",
        "EquityAttributableToOwnersOfParent",
        "OtherEquity",
    ),
    "fundamental_total_debt": (
        "TotalDebt",
        "Debt",
        "Borrowings",
        "BorrowingsCurrent",
        "BorrowingsNoncurrent",
    ),
    "fundamental_cash": (
        "CashAndCashEquivalents",
        "CashAndBankBalances",
        "CashAndBank",
    ),
}

_thread_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is not None:
        return s
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/134.0 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Referer": "https://www.nseindia.com/companies-listing/corporate-integrated-filing",
            "X-Requested-With": "XMLHttpRequest",
            "Connection": "keep-alive",
        }
    )
    try:
        s.get(NSE_HOME, timeout=15)
    except Exception:
        pass
    _thread_local.session = s
    return s


def _parse_dt(value) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        return pd.NaT
    return pd.to_datetime(str(value).strip(), errors="coerce", utc=True, dayfirst=True)


def _num(value) -> float:
    try:
        text = str(value).strip().replace(",", "")
        if not text:
            return np.nan
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]
        return float(text)
    except Exception:
        return np.nan


def _local(tag: str) -> str:
    return str(tag).split("}", 1)[-1].split(":", 1)[-1]


def _catalog_page(session: requests.Session, page: int, size: int, from_date: str, to_date: str):
    params = {
        "type": "Integrated Filing- Financials",
        "index": "equities",
        "from_date": from_date,
        "to_date": to_date,
        "page": page,
        "size": size,
    }
    last = None
    for attempt in range(3):
        try:
            r = session.get(NSE_API, params=params, timeout=30)
            if r.status_code in (403, 429, 500, 502, 503, 504):
                last = RuntimeError(f"NSE catalog HTTP {r.status_code}")
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            data = payload.get("data", []) if isinstance(payload, dict) else payload
            return data if isinstance(data, list) else []
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last if last else RuntimeError("NSE catalog request failed")


def catalog_all(from_date: str, to_date: str, size: int) -> list[dict]:
    session = _session()
    rows: list[dict] = []
    for page in range(1, 2001):
        batch = _catalog_page(session, page, size, from_date, to_date)
        if not batch:
            break
        rows.extend(batch)
        if page == 1 or page % 25 == 0:
            print(f"NSE integrated catalog: page={page:,} rows={len(rows):,}")
        if len(batch) < size:
            break
    print(f"NSE integrated catalog total rows={len(rows):,}")
    return rows


def choose_filings(rows: list[dict], symbols: set[str], max_filings: int) -> list[dict]:
    clean = []
    for r in rows:
        if str(r.get("type", "")).strip().lower() != "integrated filing- financials":
            continue
        sym = normalize_symbol(r.get("symbol", ""))
        url = str(r.get("xbrl") or "").strip()
        period_end = _parse_dt(r.get("qe_Date"))
        available = _parse_dt(r.get("broadcast_Date"))
        consolidated = str(r.get("consolidated", "")).strip().lower() == "consolidated"
        if (
            not sym
            or sym not in symbols
            or not url
            or pd.isna(period_end)
            or pd.isna(available)
        ):
            continue
        clean.append(
            {
                "symbol": sym,
                "period_end": period_end.normalize(),
                "fundamental_available_at": available,
                "xbrl_url": url,
                "consolidated": consolidated,
                "audited": str(r.get("audited", "")).strip().lower() == "audited",
                "company": r.get("cmName", ""),
            }
        )
    if not clean:
        return []

    d = pd.DataFrame(clean)
    chosen = []
    for (sym, period_end), g in d.groupby(["symbol", "period_end"], sort=False):
        g = g.sort_values(
            ["consolidated", "fundamental_available_at"],
            ascending=[False, False],
        )
        chosen.append(g.iloc[0].to_dict())

    d = pd.DataFrame(chosen).sort_values(
        ["symbol", "period_end", "fundamental_available_at"]
    )
    # Keep the latest N filed quarters for each symbol. Four quarters plus
    # a previous quarter are enough for the 2026 growth features; the default
    # of six gives additional PIT history without pulling the whole archive.
    d = d.groupby("symbol", sort=False, group_keys=False).tail(max_filings)
    return d.to_dict("records")


def _cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return CACHE_ROOT / f"{key}.xbrl"


def _download_xbrl(url: str) -> bytes:
    cp = _cache_path(url)
    if cp.exists() and cp.stat().st_size > 0:
        return cp.read_bytes()

    session = _session()
    last = None
    for attempt in range(3):
        try:
            r = session.get(
                url,
                headers={"Accept": "*/*", "Referer": "https://www.nseindia.com/"},
                timeout=30,
            )
            if r.status_code in (403, 429, 500, 502, 503, 504):
                last = RuntimeError(f"XBRL HTTP {r.status_code}")
                time.sleep(1.0 * (attempt + 1))
                continue
            r.raise_for_status()
            content = r.content
            if not content:
                raise RuntimeError("empty XBRL response")
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_suffix(".tmp")
            tmp.write_bytes(content)
            tmp.replace(cp)
            return content
        except Exception as exc:
            last = exc
            time.sleep(1.0 * (attempt + 1))
    raise last if last else RuntimeError("XBRL download failed")


def _contexts(root: ET.Element):
    meta = {}
    xbrli = "http://www.xbrl.org/2003/instance"
    for ctx in root.findall(f"{{{xbrli}}}context"):
        cid = ctx.get("id")
        if not cid:
            continue
        start = end = instant = pd.NaT
        period = ctx.find(f"{{{xbrli}}}period")
        if period is not None:
            for e in period:
                name = _local(e.tag).lower()
                if name == "startdate":
                    start = _parse_dt(e.text)
                elif name == "enddate":
                    end = _parse_dt(e.text)
                elif name == "instant":
                    instant = _parse_dt(e.text)
        has_dim = any(
            _local(e.tag) in {"explicitMember", "typedMember"} for e in ctx.iter()
        )
        meta[cid] = (start, end, instant, has_dim)
    return meta


def _fact_map(root: ET.Element, meta: dict):
    facts = {}
    for e in root.iter():
        ctx = e.get("contextRef")
        if not ctx or e.text is None:
            continue
        m = meta.get(ctx)
        if m and m[3]:
            continue
        value = _num(e.text)
        if not np.isfinite(value):
            continue
        facts.setdefault(ctx, {})[_local(e.tag)] = value
    return facts


def _pick(facts: dict, aliases) -> float:
    for name in aliases:
        value = facts.get(name)
        if value is not None and np.isfinite(value):
            return float(value)
    return np.nan


def _parse_filing(f: dict):
    try:
        root = ET.fromstring(_download_xbrl(f["xbrl_url"]))
        meta = _contexts(root)
        facts = _fact_map(root, meta)
        q = facts.get("OneD", {}) or {}
        bs = facts.get("OneI", {}) or {}

        revenue = _pick(q, FIELD_ALIASES["fundamental_revenue"])
        pat = _pick(q, FIELD_ALIASES["fundamental_pat"])
        eps = _pick(q, FIELD_ALIASES["fundamental_eps"])
        finance_cost = _pick(q, FIELD_ALIASES["fundamental_finance_cost"])
        equity = _pick(bs, FIELD_ALIASES["fundamental_total_equity"])
        cash = _pick(bs, FIELD_ALIASES["fundamental_cash"])

        current_debt = _pick(bs, ("BorrowingsCurrent",))
        noncurrent_debt = _pick(bs, ("BorrowingsNoncurrent",))
        total_debt = _pick(bs, ("TotalDebt", "Debt", "Borrowings"))
        if np.isfinite(current_debt) or np.isfinite(noncurrent_debt):
            total_debt = sum(
                x for x in (current_debt, noncurrent_debt) if np.isfinite(x)
            )

        pbt = _pick(q, ("ProfitBeforeTax", "ProfitLossFromOrdinaryActivitiesBeforeTax"))
        dep = _pick(
            q,
            (
                "DepreciationDepletionAndAmortisationExpense",
                "DepreciationAndAmortisationExpense",
                "Depreciation",
            ),
        )
        ebitda = np.nan
        if np.isfinite(pbt):
            ebit = pbt + (finance_cost if np.isfinite(finance_cost) else 0.0)
            ebitda = ebit + (dep if np.isfinite(dep) else 0.0)

        vals = {
            "fundamental_revenue": revenue,
            "fundamental_pat": pat,
            "fundamental_eps": eps,
            "fundamental_ebitda": ebitda,
            "fundamental_finance_cost": finance_cost,
            "fundamental_total_equity": equity,
            "fundamental_total_debt": total_debt,
            "fundamental_cash": cash,
        }
        if not any(np.isfinite(v) for v in vals.values()):
            return None

        return {
            **f,
            **vals,
            "fundamental_source": "NSE_XBRL_PIT",
            "fundamental_basis": "Integrated_Filing_OneD_OneI",
        }
    except Exception:
        return None


def _load_seed() -> pd.DataFrame:
    paths = [
        BASE / "processed/features/PIT_FUNDAMENTAL_FEATURES_2026.parquet",
        BASE / "processed/features/MASTER_PIT_WITH_FUNDAMENTALS_2026.parquet",
        BASE / "corporate/fundamentals_v2/PIT_FUNDAMENTALS_STRICT_2025_2026.parquet",
        BASE / "corporate/pit_fundamentals_filings_2025_2026.parquet",
    ]
    wanted = [
        "symbol",
        "period_end",
        "fundamental_available_at",
        "fundamental_source",
        "fundamental_basis",
        *FIELD_ALIASES.keys(),
    ]
    out = []
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if "symbol" not in df.columns or "fundamental_available_at" not in df.columns:
            continue
        df["symbol"] = df["symbol"].map(normalize_symbol)
        df["fundamental_available_at"] = pd.to_datetime(
            df["fundamental_available_at"], errors="coerce", utc=True
        )
        if "period_end" not in df.columns:
            if "trade_date" in df.columns:
                df["period_end"] = pd.to_datetime(
                    df["trade_date"], errors="coerce", utc=True
                ).dt.normalize()
            else:
                continue
        for c in wanted:
            if c not in df.columns:
                df[c] = np.nan
        out.append(df[wanted])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=wanted)


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    x["symbol"] = x["symbol"].map(normalize_symbol)
    x["period_end"] = pd.to_datetime(x["period_end"], errors="coerce", utc=True).dt.normalize()
    x["fundamental_available_at"] = pd.to_datetime(
        x["fundamental_available_at"], errors="coerce", utc=True
    )
    for c in FIELD_ALIASES:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["symbol", "period_end", "fundamental_available_at"])
    x = x.sort_values(["symbol", "period_end", "fundamental_available_at"])
    x = x.drop_duplicates(["symbol", "period_end"], keep="last")

    pieces = []
    for symbol, g in x.groupby("symbol", sort=False):
        g = g.sort_values("period_end").copy()
        for base, growth in (
            ("fundamental_revenue", "fund_revenue_growth"),
            ("fundamental_pat", "fund_pat_growth"),
            ("fundamental_eps", "fund_eps_growth"),
            ("fundamental_ebitda", "fund_ebitda_growth"),
        ):
            g[growth] = g[base].pct_change(fill_method=None)
        g["fund_pat_margin"] = g["fundamental_pat"] / g["fundamental_revenue"].replace(0, np.nan)
        g["fund_debt_to_equity"] = (
            g["fundamental_total_debt"] / g["fundamental_total_equity"].replace(0, np.nan)
        )
        g["fund_cash_to_debt"] = (
            g["fundamental_cash"] / g["fundamental_total_debt"].replace(0, np.nan)
        )
        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)
    return out.replace([np.inf, -np.inf], np.nan).sort_values(
        ["symbol", "period_end", "fundamental_available_at"]
    ).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bulk-root", default=str(DEFAULT_BULK))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--from-date", default="01-01-2025")
    ap.add_argument("--to-date", default="30-09-2026")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-filings-per-symbol", type=int, default=6)
    args = ap.parse_args()

    from colab_build_all_nse_delivery_multifactor_candidate import load_bulk

    bulk = load_bulk(Path(args.bulk_root))
    roster = training_roster(bulk, min_history=60)
    symbols = set(roster["trainable_symbols"])
    print(f"TARGET ALL-NSE TRAINABLE SYMBOLS: {len(symbols):,}")

    seed = _derive(_load_seed())
    print(
        "SEED FUNDAMENTALS:",
        f"rows={len(seed):,}",
        f"symbols={seed['symbol'].nunique() if not seed.empty else 0:,}",
    )

    catalog = catalog_all(args.from_date, args.to_date, args.page_size)
    filings = choose_filings(catalog, symbols, args.max_filings_per_symbol)
    print(
        "SELECTED FILINGS:",
        f"{len(filings):,}",
        f"symbols={len({f['symbol'] for f in filings}):,}",
    )

    parsed = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_parse_filing, f) for f in filings]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            if row:
                parsed.append(row)
            if i % 500 == 0:
                print(f"XBRL progress: {i:,}/{len(filings):,}; parsed={len(parsed):,}")

    fresh = _derive(pd.DataFrame(parsed)) if parsed else pd.DataFrame()
    combined = (
        _derive(pd.concat([seed, fresh], ignore_index=True))
        if not fresh.empty
        else seed
    )
    combined = combined[combined["symbol"].isin(symbols)].copy()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)

    report = {
        "output": str(out),
        "rows": int(len(combined)),
        "symbols": int(combined["symbol"].nunique()) if not combined.empty else 0,
        "target_trainable_symbols": len(symbols),
        "symbol_coverage_pct": (
            round(100.0 * combined["symbol"].nunique() / len(symbols), 3)
            if symbols else 0.0
        ),
        "catalog_rows": len(catalog),
        "filings_selected": len(filings),
        "filings_parsed": len(fresh),
        "feature_coverage_pct": {
            c: round(100.0 * combined[c].notna().mean(), 3) if len(combined) else 0.0
            for c in FIELD_ALIASES
        },
        "source": "NSE Integrated Filing catalog + NSE nsearchives XBRL; Drive PIT seed retained",
        "pit_policy": "Only fundamental_available_at <= decision_timestamp is eligible at merge time",
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
