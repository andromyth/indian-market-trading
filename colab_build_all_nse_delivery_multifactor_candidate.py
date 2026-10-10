#!/usr/bin/env python3
"""Build ALL-NSE 59-feature multi-factor Delivery candidates in Colab."""
from __future__ import annotations
import argparse, hashlib, json, os, threading, time, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zoneinfo import ZoneInfo
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from delivery_training_universe import benchmark_for_symbol, official_nse_symbols, training_roster
from nse_data_layer import fetch_index_history, load_canonical_cash_research_history
from nse_delivery_data_layer import DEFAULT_DELIVERY_ROOT, load_delivery_history
from multifactor_data_layer_v2 import FEATURES_59, assert_training_feature_coverage, attach_readiness_flags, feature_coverage_report, load_optional_layer, merge_pit_layer, normalize_symbol, readiness_report, session_close_timestamp_utc

DEFAULT_BASE = "/content/drive/MyDrive/Indian_Market_Research"
BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", DEFAULT_BASE))
DEFAULT_BULK=str(BASE/"production_data/nse_official/bulk_eod")
DEFAULT_CASH_MASTER=str(BASE/"production_data/nse_official/ALL_NSE_CASH_HISTORY_2016_2026_IDENTITY_PRESERVED.parquet")
DEFAULT_OUT=str(BASE/"models/all_nse_delivery_multifactor_replacement_2026")

FUND_PARSER_VERSION="2026-10-10-context-coverage-v4"
FUND_CATALOG_CACHE=BASE/"corporate/nse_integrated_filing_catalog_2025_2026.json"
FUND_BROAD=BASE/"corporate/pit_fundamentals_all_nse_2025_2026.parquet"
FUND_BROAD_META=BASE/"corporate/pit_fundamentals_all_nse_2025_2026.meta.json"
FUND_AVAILABILITY=BASE/"corporate/pit_fundamental_availability_2025_2026.parquet"
FUND_AVAILABILITY_META=BASE/"corporate/pit_fundamental_availability_2025_2026.meta.json"
FUND_FILING_CACHE=BASE/"corporate/pit_fundamentals_filing_cache_2025_2026_v4.parquet"
FUND_CACHE=BASE/"corporate/xbrl_cache/all_nse_2025_2026"
CANDIDATE_META_NAME="CANDIDATE_BUILD_META_2026.json"
NSE_HOME="https://www.nseindia.com/"
NSE_IF_API="https://www.nseindia.com/api/integrated-filing-results"
FUND_FIELDS={
 "fundamental_revenue":("RevenueFromOperations","RevenueFromOperationsNetOfExciseDuty","InterestEarned","GrossPremiumIncome","GrossPremiumsWritten"),
 "fundamental_pat":("ProfitLossForPeriod","ProfitLossForThePeriod","ProfitLossAfterTax","ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates","ProfitLossFromOrdinaryActivitiesAfterTax","ProfitLossAfterTaxAndExtraordinaryItems","ProfitLossAfterTaxBeforeExtraordinaryItems"),
 "fundamental_eps":("BasicAndDilutedEPSAfterExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized","BasicEarningsPerShare","DilutedEarningsPerShare","EarningsPerShareBasic","EarningsPerShareDiluted"),
 "fundamental_finance_cost":("FinanceCosts","FinanceCost"),
 "fundamental_total_equity":("TotalEquity","Equity","EquityAttributableToOwnersOfParent","OtherEquity","Capital","ReservesAndSurplus"),
 "fundamental_total_debt":("TotalDebt","Debt","Borrowings"),
 "fundamental_cash":("CashAndCashEquivalents","CashAndBankBalances","CashAndBank"),
}
_fund_local=threading.local()
def _fund_session():
    s=getattr(_fund_local,"s",None)
    if s is None:
        s=requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/134 Safari/537.36","Accept":"application/json,text/plain,*/*","Accept-Language":"en-GB,en-US;q=0.9,en;q=0.8","Referer":"https://www.nseindia.com/companies-listing/corporate-integrated-filing","X-Requested-With":"XMLHttpRequest"})
        try: s.get(NSE_HOME,timeout=15)
        except Exception: pass
        _fund_local.s=s
    return s
def _fd(x):
    if x is None:
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")
    s=str(x).strip()
    if not s:
        return pd.NaT
    for fmt in ("%Y-%m-%d %H:%M:%S","%d-%m-%Y %H:%M:%S","%Y-%m-%d","%d-%m-%Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s,format=fmt,utc=True))
        except Exception:
            pass
    try:
        return pd.to_datetime(s,errors="coerce",utc=True,dayfirst=True)
    except Exception:
        return pd.NaT
def _fn(x):
    try:
        t=str(x).strip().replace(",","")
        if t.startswith("(") and t.endswith(")"): t="-"+t[1:-1]
        return float(t)
    except Exception: return np.nan
def _fl(tag): return str(tag).split("}",1)[-1].split(":",1)[-1]
def _catalog_all(force_refresh=False):
    from datetime import datetime, timezone
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    from_date = "01-01-2025"
    to_date = today.strftime("%d-%m-%Y")
    if not force_refresh and FUND_CATALOG_CACHE.exists() and FUND_CATALOG_CACHE.stat().st_size>0:
        try:
            cached=json.loads(FUND_CATALOG_CACHE.read_text(encoding="utf-8"))
            meta=cached.get("meta",{})
            same_window=meta.get("from_date")==from_date and meta.get("to_date")==to_date
            fetched=datetime.fromisoformat(meta["fetched_at"]).replace(tzinfo=timezone.utc) if meta.get("fetched_at") else None
            fresh=fetched is not None and (datetime.now(timezone.utc)-fetched).total_seconds() < 24*3600
            if same_window and fresh and meta.get("coverage_complete") is True and isinstance(cached.get("data"),list):
                print(f"fundamental catalog cache reused rows={len(cached['data']):,}", flush=True)
                return cached["data"]
        except Exception:
            pass

    # Stage B can be quiet for several minutes if NSE times out, so report the
    # source window, cache status, every page attempt, and bounded retry failures.
    FUND_CATALOG_CACHE.parent.mkdir(parents=True,exist_ok=True)
    partial_path = FUND_CATALOG_CACHE.with_name(
        FUND_CATALOG_CACHE.stem + ".partial.json"
    )
    size=500
    out=[]
    first_page=1
    if partial_path.exists() and partial_path.stat().st_size>0:
        try:
            partial=json.loads(partial_path.read_text(encoding="utf-8"))
            meta=partial.get("meta",{})
            if (
                meta.get("from_date")==from_date
                and meta.get("to_date")==to_date
                and meta.get("coverage_complete") is not True
                and isinstance(partial.get("data"),list)
            ):
                out=partial["data"]
                first_page=max(1,int(meta.get("next_page",1)))
                print(
                    f"resuming partial NSE filing catalog: page={first_page}; "
                    f"checkpoint_rows={len(out):,}; window={from_date}..{to_date}",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"partial NSE filing catalog ignored: {type(exc).__name__}: {exc}",
                flush=True,
            )
    if first_page==1 and not out:
        print(
            f"starting NSE integrated-filing catalog: window={from_date}..{to_date}; "
            f"page_size={size}; source={NSE_IF_API}",
            flush=True,
        )

    def save_partial(next_page, status="PARTIAL"):
        payload={
            "meta":{
                "cache_version":"3",
                "from_date":from_date,
                "to_date":to_date,
                "coverage_start":"2025-01-01",
                "coverage_end":today.isoformat(),
                "coverage_complete":False,
                "pagination_complete":False,
                "status":status,
                "next_page":int(next_page),
                "checkpointed_at":datetime.now(timezone.utc).isoformat(),
                "rows":len(out),
            },
            "data":out,
        }
        tmp=partial_path.with_name(partial_path.name+".tmp")
        tmp.write_text(json.dumps(payload,default=str),encoding="utf-8")
        os.replace(tmp,partial_path)

    s=_fund_session()
    pagination_complete = False
    pages_done=first_page-1
    for page in range(first_page,401):
        params={"type":"Integrated Filing- Financials","index":"equities","from_date":from_date,"to_date":to_date,"page":page,"size":size}
        print(
            f"NSE filing catalog: requesting page={page}; rows_collected={len(out):,}",
            flush=True,
        )
        last=None
        request_started=time.monotonic()
        for attempt in range(3):
            try:
                rr=s.get(NSE_IF_API,params=params,timeout=30)
                rr.raise_for_status()
                payload=rr.json()
                batch=payload.get("data",[]) if isinstance(payload,dict) else payload
                last=None
                break
            except Exception as exc:
                last=exc
                print(
                    f"NSE filing catalog page={page} attempt={attempt+1}/3 failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                time.sleep(1.0*(attempt+1))
        if last:
            save_partial(page,status="REQUEST_FAILED")
            raise RuntimeError(
                f"NSE filing catalog stopped at page {page}; partial checkpoint saved "
                f"to {partial_path}. Original error: {type(last).__name__}: {last}"
            ) from last
        elapsed=time.monotonic()-request_started
        if not batch:
            pagination_complete = True
            print(
                f"NSE filing catalog terminal page={page}; elapsed={elapsed:.1f}s; "
                f"total_rows={len(out):,}",
                flush=True,
            )
            break
        out.extend(batch)
        pages_done=page
        print(
            f"NSE filing catalog page={page} complete; returned={len(batch):,}; "
            f"total_rows={len(out):,}; elapsed={elapsed:.1f}s",
            flush=True,
        )
        # If a Colab session closes mid-ingestion, resume from the next verified
        # page instead of discarding the entire partial catalogue.
        save_partial(page+1)
        if len(batch)<size:
            pagination_complete = True
            break
    if not pagination_complete:
        save_partial(pages_done+1,status="PAGINATION_LIMIT")
        raise RuntimeError("NSE integrated-filing catalog did not reach a verified terminal page; refusing to mark coverage complete.")

    from datetime import datetime, timezone
    complete_payload={
        "meta":{
            "cache_version":"3",
            "from_date":from_date,
            "to_date":to_date,
            "coverage_start":"2025-01-01",
            "coverage_end":today.isoformat(),
            "coverage_complete":True,
            "pagination_complete":True,
            "fetched_at":datetime.now(timezone.utc).isoformat(),
            "rows":len(out),
        },
        "data":out,
    }
    complete_tmp=FUND_CATALOG_CACHE.with_name(FUND_CATALOG_CACHE.name+".tmp")
    complete_tmp.write_text(json.dumps(complete_payload,default=str),encoding="utf-8")
    os.replace(complete_tmp,FUND_CATALOG_CACHE)
    try:
        partial_path.unlink()
    except FileNotFoundError:
        pass
    print(f"fundamental catalog rows={len(out):,}; pages={pages_done-first_page+1:,}", flush=True)
    return out

def _fund_filings(catalog,symbols,max_each=8):
    rows=[]
    for r in catalog:
        if str(r.get("type","")).strip().lower()!="integrated filing- financials": continue
        sym=normalize_symbol(r.get("symbol","")); url=str(r.get("xbrl") or "").strip(); pe=_fd(r.get("qe_Date")); av=_fd(r.get("broadcast_Date")); con=str(r.get("consolidated","")).strip().lower()=="consolidated"
        if sym in symbols and url and pd.notna(pe) and pd.notna(av): rows.append({"symbol":sym,"period_end":pe.normalize(),"fundamental_available_at":av,"xbrl_url":url,"consolidated":con})
    if not rows: return []
    d=pd.DataFrame(rows); chosen=[]
    for (sym,pe),g in d.groupby(["symbol","period_end"],sort=False):
        g=g.sort_values(["consolidated","fundamental_available_at"],ascending=[False,False]); chosen.append(g.iloc[0].to_dict())
    d=pd.DataFrame(chosen).sort_values(["symbol","period_end","fundamental_available_at"])
    return d.groupby("symbol",sort=False,group_keys=False).tail(max_each).to_dict("records")

def _build_fundamental_availability(catalog, symbols):
    """Create the full eligible-filing availability index, independent of XBRL parse success."""
    rows = []
    wanted = set(map(normalize_symbol, symbols))
    for item in catalog:
        if str(item.get("type", "")).strip().lower() != "integrated filing- financials":
            continue
        symbol = normalize_symbol(item.get("symbol", ""))
        url = str(item.get("xbrl") or "").strip()
        period_end = _fd(item.get("qe_Date"))
        available_at = _fd(item.get("broadcast_Date"))
        consolidated = str(item.get("consolidated", "")).strip().lower() == "consolidated"
        if symbol not in wanted or not url or pd.isna(period_end) or pd.isna(available_at):
            continue
        rows.append({
            "symbol": symbol,
            "period_end": period_end.normalize(),
            "fundamental_available_at": available_at,
            "consolidated": consolidated,
        })
    columns = ["symbol", "period_end", "fundamental_available_at", "consolidated"]
    index = pd.DataFrame(rows, columns=columns)
    if not index.empty:
        index = index.sort_values(
            ["symbol", "period_end", "consolidated", "fundamental_available_at"],
            ascending=[True, True, False, False],
        ).drop_duplicates(["symbol", "period_end"], keep="first")
        index = index.sort_values(["symbol", "fundamental_available_at"]).reset_index(drop=True)
    FUND_AVAILABILITY.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(FUND_AVAILABILITY, index=False)
    catalog_manifest = {}
    try:
        cached = json.loads(FUND_CATALOG_CACHE.read_text(encoding="utf-8"))
        catalog_manifest = cached.get("meta", {})
    except Exception:
        pass
    report = {
        "path": str(FUND_AVAILABILITY),
        "rows": int(len(index)),
        "symbols_with_eligible_filings": int(index["symbol"].nunique()) if not index.empty else 0,
        "target_symbol_count": len(wanted),
        "coverage_start": catalog_manifest.get("coverage_start"),
        "coverage_end": catalog_manifest.get("coverage_end"),
        "coverage_complete": catalog_manifest.get("coverage_complete") is True,
        "source": "NSE_OFFICIAL_INTEGRATED_FILING_CATALOG",
        "definition": "All eligible as-of financial filing timestamps from a completely paginated catalogue, independent of XBRL fact parse success.",
    }
    FUND_AVAILABILITY_META.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not report["coverage_complete"]:
        raise RuntimeError("Fundamental availability index cannot be trusted: filing catalogue coverage is incomplete.")
    return report
def _fund_cache(url):
    import hashlib
    return FUND_CACHE/(hashlib.sha256(url.encode()).hexdigest()[:24]+".xbrl")
def _fund_xml(f):
    cp=_fund_cache(f["xbrl_url"]); cp.parent.mkdir(parents=True,exist_ok=True)
    if cp.exists() and cp.stat().st_size: return cp.read_bytes()
    s=_fund_session()
    for attempt in range(3):
        try:
            rr=s.get(f["xbrl_url"],headers={"Accept":"*/*","Referer":NSE_HOME},timeout=30); rr.raise_for_status(); cp.write_bytes(rr.content); return rr.content
        except Exception:
            time.sleep(1.0*(attempt+1))
    return b""
def _fund_parse(f):
    """Parse the primary dimension-free PIT financial contexts from an NSE XBRL filing.

    Older versions assumed context IDs named OneD/OneI. NSE filings are not
    contractually required to use those IDs, so select duration/instant
    contexts by their periods and the filing period end instead.
    """
    try:
        raw = _fund_xml(f)
        if not raw:
            return None
        root=ET.fromstring(raw)
        xbrli="http://www.xbrl.org/2003/instance"
        meta={}
        contexts=[]
        target_end=pd.Timestamp(f["period_end"]).tz_convert("UTC") if pd.Timestamp(f["period_end"]).tzinfo else pd.Timestamp(f["period_end"],tz="UTC")

        for ctx in root.findall(f"{{{xbrli}}}context"):
            cid=ctx.get("id")
            if not cid:
                continue
            per=ctx.find(f"{{{xbrli}}}period")
            start_ts=end_ts=inst_ts=pd.NaT
            if per is not None:
                for e in per:
                    n=_fl(e.tag).lower()
                    if n=="startdate":
                        start_ts=_fd(e.text)
                    elif n=="enddate":
                        end_ts=_fd(e.text)
                    elif n=="instant":
                        inst_ts=_fd(e.text)
            dim=any(_fl(e.tag) in ("explicitmember","typedmember") for e in ctx.iter())
            duration_days=np.nan
            if pd.notna(start_ts) and pd.notna(end_ts):
                duration_days=(end_ts-start_ts).total_seconds()/86400.0
            meta[cid]={"start":start_ts,"end":end_ts,"instant":inst_ts,"dim":dim,"duration_days":duration_days}
            contexts.append(cid)

        facts={}
        for e in root.iter():
            cid=e.get("contextRef")
            m=meta.get(cid)
            if not cid or e.text is None or m is None or m["dim"]:
                continue
            v=_fn(e.text)
            if np.isfinite(v):
                facts.setdefault(cid,{})[_fl(e.tag)]=v

        duration_ids=[]
        instant_ids=[]
        for cid,m in meta.items():
            end=m["end"]
            if pd.notna(end) and pd.notna(target_end):
                diff_days=abs((end-target_end).total_seconds()/86400.0)
            else:
                diff_days=1e9
            if pd.notna(m["start"]) and pd.notna(m["end"]) and m["duration_days"] > 0:
                duration_ids.append((diff_days,m["duration_days"],cid))
            elif pd.notna(m["instant"]):
                inst_diff=abs((m["instant"]-target_end).total_seconds()/86400.0) if pd.notna(target_end) else 1e9
                instant_ids.append((inst_diff,cid))

        flow_groups = [
            FUND_FIELDS["fundamental_revenue"],
            FUND_FIELDS["fundamental_pat"],
            FUND_FIELDS["fundamental_eps"],
            FUND_FIELDS["fundamental_finance_cost"],
            ("ProfitBeforeTax", "ProfitLossFromOrdinaryActivitiesBeforeTax"),
            ("DepreciationDepletionAndAmortisationExpense", "DepreciationAndAmortisationExpense", "Depreciation"),
        ]
        balance_groups = [
            ("TotalEquity", "Equity", "EquityAttributableToOwnersOfParent", "OtherEquity", "Capital", "EquityShareCapital", "ReservesAndSurplus"),
            ("TotalDebt", "Debt", "Borrowings", "BorrowingsCurrent", "BorrowingsNoncurrent"),
            FUND_FIELDS["fundamental_cash"],
        ]

        def context_score(cid, concept_groups):
            values = facts.get(cid, {})
            return sum(
                any(name in values and np.isfinite(values[name]) for name in aliases)
                for aliases in concept_groups
            )

        # Prefer the reporting period closest to the filing's period_end and
        # score contexts by how many required concepts they actually contain.
        # The old code selected the first context with ANY numeric fact, which
        # often chose a context unrelated to Revenue/PAT/EPS and left the
        # locked financial features empty despite parseable facts in other
        # dimension-free contexts in the same filing.
        duration_ids.sort(
            key=lambda item: (item[0], abs(item[1] - 90.0), -context_score(item[2], flow_groups))
        )
        best_diff = min((item[0] for item in duration_ids), default=1e9)
        near_period = [item for item in duration_ids if item[0] <= best_diff + 3.0]
        quarterly = [item for item in near_period if 45.0 <= item[1] <= 130.0]
        flow_candidates = quarterly or near_period
        flow_candidates.sort(
            key=lambda item: (-context_score(item[2], flow_groups), abs(item[1] - 90.0), item[0])
        )
        q = {}
        for _, _, cid in flow_candidates:
            for name, value in facts.get(cid, {}).items():
                q.setdefault(name, value)

        instant_ids.sort(
            key=lambda item: (item[0], -context_score(item[1], balance_groups))
        )
        best_instant_diff = min((item[0] for item in instant_ids), default=1e9)
        near_instant = [item for item in instant_ids if item[0] <= best_instant_diff + 3.0]
        near_instant.sort(key=lambda item: (-context_score(item[1], balance_groups), item[0]))
        bs = {}
        for _, cid in near_instant:
            for name, value in facts.get(cid, {}).items():
                bs.setdefault(name, value)

        def pick(d,names):
            for n in names:
                if n in d and np.isfinite(d[n]):
                    return float(d[n])
            return np.nan

        rev=pick(q,FUND_FIELDS["fundamental_revenue"])
        pat=pick(q,FUND_FIELDS["fundamental_pat"])
        eps=pick(q,FUND_FIELDS["fundamental_eps"])
        fin=pick(q,FUND_FIELDS["fundamental_finance_cost"])

        eq=pick(bs,("TotalEquity","Equity","EquityAttributableToOwnersOfParent","OtherEquity"))
        if not np.isfinite(eq):
            cap=pick(bs,("Capital","EquityShareCapital"))
            res=pick(bs,("ReservesAndSurplus","OtherEquity"))
            eq=(cap+res) if np.isfinite(cap) and np.isfinite(res) else np.nan

        cash=pick(bs,FUND_FIELDS["fundamental_cash"])
        debt=pick(bs,("TotalDebt","Debt","Borrowings"))
        cur=pick(bs,("BorrowingsCurrent",))
        non=pick(bs,("BorrowingsNoncurrent",))
        if np.isfinite(cur) or np.isfinite(non):
            debt=sum(v for v in (cur,non) if np.isfinite(v))

        pbt=pick(q,("ProfitBeforeTax","ProfitLossFromOrdinaryActivitiesBeforeTax"))
        dep=pick(q,("DepreciationDepletionAndAmortisationExpense","DepreciationAndAmortisationExpense","Depreciation"))
        ebitda=(pbt+(fin if np.isfinite(fin) else 0)+(dep if np.isfinite(dep) else 0)) if np.isfinite(pbt) else np.nan

        vals={
            "fundamental_revenue":rev,
            "fundamental_pat":pat,
            "fundamental_eps":eps,
            "fundamental_ebitda":ebitda,
            "fundamental_finance_cost":fin,
            "fundamental_total_equity":eq,
            "fundamental_total_debt":debt,
            "fundamental_cash":cash,
        }
        if not any(np.isfinite(v) for v in vals.values()):
            return None
        return {**f,**vals,"fundamental_source":"NSE_XBRL_PIT","fundamental_basis":"Integrated_Filing_period_aware_OneD_OneI_fallback"}
    except Exception as exc:
        print(f"fundamental parse failed for {f.get('symbol', '')}: {type(exc).__name__}: {exc}", flush=True)
        return None

def _funding_key(f):
    raw="|".join([
        str(f.get("symbol","")),
        str(pd.Timestamp(f.get("period_end")).date()) if pd.notna(f.get("period_end")) else "",
        str(pd.Timestamp(f.get("fundamental_available_at")).isoformat()) if pd.notna(f.get("fundamental_available_at")) else "",
        str(f.get("xbrl_url","")),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _hash_strings(values):
    h=hashlib.sha256()
    for v in sorted(str(x) for x in values):
        h.update(v.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()

def _fund_target_hash(symbols):
    return _hash_strings(symbols)

def _load_filing_cache():
    if not FUND_FILING_CACHE.exists() or FUND_FILING_CACHE.stat().st_size==0:
        return pd.DataFrame()
    try:
        x=pd.read_parquet(FUND_FILING_CACHE)
        if "parser_version" not in x.columns:
            return pd.DataFrame()
        x=x[x["parser_version"].eq(FUND_PARSER_VERSION)].copy()
        return x
    except Exception:
        return pd.DataFrame()

def _save_filing_cache(cache):
    # Atomic checkpoint: an interrupted Colab session must never leave a
    # half-written filing cache that destroys the progress from prior batches.
    _atomic_parquet(cache, FUND_FILING_CACHE)

def _parse_one_cached(f):
    key=_funding_key(f)
    r=_fund_parse(f)
    if r is None:
        return {**f,"filing_key":key,"parser_version":FUND_PARSER_VERSION,"parse_status":"failed"}
    return {**r,"filing_key":key,"parser_version":FUND_PARSER_VERSION,"parse_status":"parsed"}

def _fundamental_build_signature(symbols,filings):
    return {
        "parser_version":FUND_PARSER_VERSION,
        "target_symbols_hash":_fund_target_hash(symbols),
        "target_symbol_count":len(symbols),
        "selected_filing_keys_hash":_hash_strings([_funding_key(f) for f in filings]),
        "selected_filing_count":len(filings),
    }

def _broad_meta_matches(meta,sig):
    return all(meta.get(k)==v for k,v in sig.items())

def _bootstrap_cache_from_broad(filing_cache):
    if not FUND_BROAD.exists() or FUND_BROAD.stat().st_size==0:
        return filing_cache
    try:
        old=pd.read_parquet(FUND_BROAD)
        if old.empty or "symbol" not in old.columns:
            return filing_cache
        basis=old.get("fundamental_basis")
        if basis is None or not basis.astype(str).str.contains("period_aware",case=False,na=False).any():
            return filing_cache
        if "xbrl_url" not in old.columns or "period_end" not in old.columns or "fundamental_available_at" not in old.columns:
            return filing_cache
        rows=[]
        for r in old.to_dict("records"):
            f={k:r.get(k) for k in ("symbol","period_end","fundamental_available_at","xbrl_url","consolidated")}
            rows.append({**r,"filing_key":_funding_key(f),"parser_version":FUND_PARSER_VERSION,"parse_status":"parsed"})
        boot=pd.DataFrame(rows)
        if filing_cache.empty:
            return boot
        return pd.concat([filing_cache,boot],ignore_index=True).drop_duplicates("filing_key",keep="last")
    except Exception:
        return filing_cache

def _ensure_broad_fundamentals(symbols,force_refresh=False,retry_failed=False,max_workers=2):
    catalog=_catalog_all(force_refresh=force_refresh)
    availability_report = _build_fundamental_availability(catalog, symbols)
    print(
        f"fundamental availability index: rows={availability_report['rows']:,} "
        f"symbols={availability_report['symbols_with_eligible_filings']:,}/"
        f"{availability_report['target_symbol_count']:,} "
        f"coverage={availability_report['coverage_start']}..{availability_report['coverage_end']}",
        flush=True,
    )
    filings=_fund_filings(catalog,set(symbols),max_each=8)
    if not filings:
        raise RuntimeError("ALL-NSE PIT fundamentals: no eligible NSE Integrated Filing records found")

    sig=_fundamental_build_signature(symbols,filings)
    if not force_refresh and FUND_BROAD.exists() and FUND_BROAD.stat().st_size>0:
        try:
            meta=json.loads(FUND_BROAD_META.read_text(encoding="utf-8")) if FUND_BROAD_META.exists() else {}
            if meta and _broad_meta_matches(meta,sig):
                print(f"fundamentals build reused from manifest; rows={int(meta.get('rows',0)):,}",flush=True)
                return {"path":str(FUND_BROAD),"availability_path":str(FUND_AVAILABILITY),"availability_report":availability_report,"rebuilt":False,"cached":True,"symbols":int(meta.get("symbols",0)),"rows":int(meta.get("rows",0))}
            # One-time bootstrap for a broad file produced by the period-aware
            # parser before the persistent build metadata existed. This prevents
            # an immediate second full NSE/XBRL pass after a completed run.
            if not meta:
                old=pd.read_parquet(FUND_BROAD)
                basis=old.get("fundamental_basis")
                compatible = basis is not None and basis.astype(str).str.contains("period_aware",case=False,na=False).any()
                if compatible and "symbol" in old.columns:
                    boot={
                        **sig,
                        "rows":int(len(old)),
                        "symbols":int(old["symbol"].map(normalize_symbol).nunique()),
                        "bootstrapped_from_existing_broad":True,
                    }
                    FUND_BROAD_META.write_text(json.dumps(boot,indent=2,default=str),encoding="utf-8")
                    print(
                        f"fundamentals build metadata bootstrapped from existing broad file; "
                        f"rows={len(old):,} symbols={old['symbol'].map(normalize_symbol).nunique():,}",
                        flush=True,
                    )
                    return {"path":str(FUND_BROAD),"availability_path":str(FUND_AVAILABILITY),
                            "availability_report":availability_report,"rebuilt":False,"cached":True,
                            "symbols":int(boot["symbols"]),"rows":int(boot["rows"])}
        except Exception:
            pass

    # The parser version was advanced because the old context selector chose
    # the first numeric XBRL context rather than the contexts containing the
    # actual revenue/PAT/EPS/balance-sheet concepts. Reparse cached source XML
    # using the v4 parser; do not relabel old derived values as current parses.
    filing_cache=_load_filing_cache()
    cached_keys=set(filing_cache["filing_key"].astype(str)) if not filing_cache.empty else set()
    selected_by_key={_funding_key(f):f for f in filings}
    missing=[f for k,f in selected_by_key.items() if k not in cached_keys]
    failed_cached=set()
    if not filing_cache.empty and "parse_status" in filing_cache.columns:
        failed_cached=set(filing_cache.loc[filing_cache["parse_status"].eq("failed"),"filing_key"].astype(str))
    if retry_failed:
        missing.extend(selected_by_key[k] for k in failed_cached if k in selected_by_key and k not in {_funding_key(x) for x in missing})

    print(
        f"fundamentals cache: selected={len(filings):,} "
        f"cached={len(filings)-len(missing):,} "
        f"to_process={len(missing):,} "
        f"failed_cached={len(failed_cached):,}",
        flush=True,
    )

    new_rows=[]
    processed=0
    parsed_checkpointed=0

    def flush_filing_checkpoint():
        nonlocal filing_cache, new_rows, parsed_checkpointed
        if not new_rows:
            return
        chunk=pd.DataFrame(new_rows)
        parsed_checkpointed += int(chunk.get("parse_status",pd.Series(dtype=str)).eq("parsed").sum())
        if filing_cache.empty:
            filing_cache=chunk
        else:
            filing_cache=pd.concat([filing_cache,chunk],ignore_index=True)
        filing_cache=filing_cache.drop_duplicates("filing_key",keep="last")
        _save_filing_cache(filing_cache)
        new_rows=[]
        print(
            f"fundamentals checkpoint saved: processed={processed:,}/{len(missing):,}; "
            f"new_parsed={parsed_checkpointed:,}; cache_rows={len(filing_cache):,}",
            flush=True,
        )

    if missing:
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
            futs=[ex.submit(_parse_one_cached,f) for f in missing]
            for i,fut in enumerate(as_completed(futs),1):
                new_rows.append(fut.result())
                processed=i
                # Persist every 25 filings. If Colab disconnects, a later run
                # can reuse parsed/failed cache entries instead of losing the
                # entire current refresh.
                if len(new_rows)>=25:
                    flush_filing_checkpoint()
        flush_filing_checkpoint()

    if not filing_cache.empty:
        filing_cache=filing_cache.drop_duplicates("filing_key",keep="last")
        _save_filing_cache(filing_cache)

    selected_keys=set(selected_by_key)
    usable=filing_cache[
        filing_cache["filing_key"].astype(str).isin(selected_keys) &
        filing_cache["parse_status"].eq("parsed")
    ].copy()
    if usable.empty:
        raise RuntimeError("ALL-NSE PIT fundamentals: selected filings have no parsed records")

    d=pd.DataFrame(usable).drop(columns=["filing_key","parser_version","parse_status"],errors="ignore")
    d["symbol"]=d["symbol"].map(normalize_symbol)
    d["period_end"]=pd.to_datetime(d["period_end"],utc=True).dt.normalize()
    d["fundamental_available_at"]=pd.to_datetime(d["fundamental_available_at"],utc=True)
    for c in FUND_FIELDS:
        if c in d.columns:
            d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.sort_values(["symbol","period_end","fundamental_available_at"]).drop_duplicates(["symbol","period_end"],keep="last")
    pieces=[]
    for sym,g in d.groupby("symbol",sort=False):
        g=g.sort_values("period_end").copy()
        for base,growth in (("fundamental_revenue","fund_revenue_growth"),("fundamental_pat","fund_pat_growth"),("fundamental_eps","fund_eps_growth"),("fundamental_ebitda","fund_ebitda_growth")):
            g[growth]=g[base].pct_change(fill_method=None)
        g["fund_pat_margin"]=g["fundamental_pat"]/g["fundamental_revenue"].replace(0,np.nan)
        g["fund_debt_to_equity"]=g["fundamental_total_debt"]/g["fundamental_total_equity"].replace(0,np.nan)
        g["fund_cash_to_debt"]=g["fundamental_cash"]/g["fundamental_total_debt"].replace(0,np.nan)
        pieces.append(g)
    d=pd.concat(pieces,ignore_index=True).replace([np.inf,-np.inf],np.nan)
    FUND_BROAD.parent.mkdir(parents=True,exist_ok=True)
    d.to_parquet(FUND_BROAD,index=False)
    report={
        "path":str(FUND_BROAD),
        "rows":len(d),
        "symbols":int(d["symbol"].nunique()),
        "target_symbols":len(symbols),
        "symbol_coverage_pct":round(100*d["symbol"].nunique()/len(symbols),3) if symbols else 0.0,
        "filings_selected":len(filings),
        "filings_parsed":int((filing_cache["filing_key"].astype(str).isin(selected_keys) & filing_cache["parse_status"].eq("parsed")).sum()),
        "filings_failed_cached":int((filing_cache["filing_key"].astype(str).isin(selected_keys) & filing_cache["parse_status"].eq("failed")).sum()),
        "feature_coverage_pct":{c:round(100*d[c].notna().mean(),3) for c in FUND_FIELDS if c in d.columns},
        **sig,
    }
    FUND_BROAD_META.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    report["availability_index"] = availability_report
    print(json.dumps(report,indent=2,default=str),flush=True)
    return report

if len(FEATURES_59) != 59:
    raise RuntimeError(f"Feature contract must contain exactly 59 features; found {len(FEATURES_59)}")

def sha256(path: Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def load_bulk(root: Path)->pd.DataFrame:
    files=sorted(root.rglob("*.parquet"))+sorted(root.rglob("*.csv"))
    if not files: raise FileNotFoundError(f"No EOD bulk files under {root}")
    parts=[]
    for p in files:
        try: x=pd.read_parquet(p) if p.suffix.lower()==".parquet" else pd.read_csv(p)
        except Exception: continue
        if x.empty: continue
        ren={}
        for c in x.columns:
            lc=str(c).strip().lower()
            mp={"ticker":"symbol","tckrsymb":"symbol","trade_date":"date","trad_dt":"date",
                "open_price":"open","opnpric":"open","high_price":"high","hghpric":"high",
                "low_price":"low","lwpric":"low","close_price":"close","clspric":"close",
                "traded_volume":"volume","ttltradgvol":"volume","dly_pct":"delivery_pct",
                "dly_qty":"delivery_qty"}
            if lc in ("symbol","date","open","high","low","close","volume","delivery_pct","delivery_qty"): ren[c]=lc
            elif lc in mp: ren[c]=mp[lc]
        x=x.rename(columns=ren)
        req={"symbol","date","open","high","low","close","volume"}
        if not req.issubset(x.columns): continue
        x["symbol"]=x["symbol"].map(normalize_symbol)
        x["date"]=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
        for c in ("open","high","low","close","volume","delivery_pct","delivery_qty"):
            if c in x.columns: x[c]=pd.to_numeric(x[c],errors="coerce")
        keep=list(req|set(["delivery_pct","delivery_qty"])&set(x.columns))
        parts.append(x[keep])
    if not parts: raise RuntimeError("No readable EOD bulk dataset")
    return pd.concat(parts,ignore_index=True).dropna(subset=["symbol","date","open","high","low","close"]).sort_values(["symbol","date"]).drop_duplicates(["symbol","date"],keep="last").reset_index(drop=True)

def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).rolling(n,min_periods=n).mean(); l=(-d.clip(upper=0)).rolling(n,min_periods=n).mean()
    return (100-100/(1+g/l.replace(0,np.nan))).where(l.ne(0),100.0)

def add_features(x):
    z=x.sort_values("date").copy(); c=z["close"]; v=z["volume"]
    for n in (1,3,6,12,24,48): z[f"ret_{n}"]=c.pct_change(n)
    z["rsi14"]=rsi(c,14)
    prev=c.shift(1); tr=pd.concat([(z["high"]-z["low"]),(z["high"]-prev).abs(),(z["low"]-prev).abs()],axis=1).max(axis=1)
    z["atr14"]=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    ma20=c.rolling(20).mean(); ma50=c.rolling(50).mean(); vm=v.rolling(20).mean(); vs=v.rolling(20).std()
    z["ma20_ratio"]=c/ma20-1; z["ma50_ratio"]=c/ma50-1; z["volume_ratio"]=v/vm; z["volume_z20"]=(v-vm)/vs
    z["range_pct"]=(z["high"]-z["low"])/c; z["body_pct"]=(c-z["open"])/z["open"]; z["up_bar"]=(c>z["open"]).astype(float)
    for c0 in ("delivery_pct","delivery_qty"):
        if c0 not in z.columns: z[c0]=np.nan
    dp=z["delivery_pct"]; dq=z["delivery_qty"]
    z["delivery_pct_change_1"]=dp.pct_change(fill_method=None); z["delivery_pct_ratio20"]=dp/dp.rolling(20).mean()
    z["delivery_pct_z20"]=(dp-dp.rolling(20).mean())/dp.rolling(20).std()
    z["delivery_qty_ratio20"]=dq/dq.rolling(20).mean(); z["delivery_qty_z20"]=(dq-dq.rolling(20).mean())/dq.rolling(20).std()
    return z

def _stored_nifty50_daily(start_date, end_date) -> pd.DataFrame:
    """Build daily NIFTY 50 benchmark closes from the project's persisted NSE archive.

    The archive already contains NSE NIFTY 50 one-minute data in Drive. Reusing
    that persisted official archive avoids an unnecessary historical-index API
    call while preserving the same benchmark semantics.
    """
    paths = [
        BASE / "raw/2018_2025/NIFTY50-INDEX.parquet",
        BASE / "raw/2026/NIFTY50-INDEX.parquet",
    ]
    parts = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            x = pd.read_parquet(path)
        except Exception:
            continue
        cols = {str(c).strip().lower(): c for c in x.columns}
        if "date" not in cols or "close" not in cols:
            continue
        x = x.rename(columns={cols["date"]: "date", cols["close"]: "close"})
        x["date"] = pd.to_datetime(x["date"], errors="coerce")
        x["close"] = pd.to_numeric(x["close"], errors="coerce")
        x = x.dropna(subset=["date", "close"]).copy()
        x["date"] = x["date"].dt.normalize()
        parts.append(x[["date", "close"]])

    if not parts:
        return pd.DataFrame()

    x = pd.concat(parts, ignore_index=True).sort_values("date", kind="stable")
    x = x.groupby("date", as_index=False, sort=True)["close"].last()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    x = x[(x["date"] >= start_ts) & (x["date"] <= end_ts)].reset_index(drop=True)

    if x.empty:
        return x

    first = x["date"].min().date()
    last = x["date"].max().date()
    span_days = max(1, (end_ts - start_ts).days)
    expected_weekdays = max(1, int(span_days * 5 / 7))
    if (
        first > (start_ts + pd.Timedelta(days=10)).date()
        or last < (end_ts - pd.Timedelta(days=10)).date()
        or len(x) < max(30, int(expected_weekdays * 0.50))
    ):
        return pd.DataFrame()

    return x


def _official_market_context(index_symbol: str, start_date, end_date) -> pd.DataFrame:
    """Load authoritative official NSE benchmark history without stale-cache bypasses."""
    root = str(index_symbol).strip().upper()
    frame = fetch_index_history(
        root,
        pd.Timestamp(start_date).date(),
        pd.Timestamp(end_date).date(),
        sync_drive=True,
    )

    if frame is None or frame.empty:
        raise RuntimeError(
            f"Official NSE benchmark history unavailable for {index_symbol}: "
            f"{start_date}..{end_date}"
        )

    x = frame.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = (
        x.dropna(subset=["date", "close"])
         .sort_values("date")
         .drop_duplicates("date", keep="last")
         .reset_index(drop=True)
    )

    c0 = x["close"]
    d = c0.diff()
    gain = d.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    rsi14 = (
        100.0 - 100.0 / (1.0 + gain / loss.replace(0, np.nan))
    ).where(loss.ne(0), 100.0)

    out = pd.DataFrame({"date": x["date"]})
    for n in (1, 3, 6, 12):
        out[f"market_ret_{n}"] = c0.pct_change(n)
    out["market_rsi14"] = rsi14
    out["market_mood_proxy"] = out["market_ret_1"]
    out["market_stress_proxy"] = c0.pct_change().rolling(20, min_periods=20).std()
    return out


def market_context_tables(bulk) -> dict[str, pd.DataFrame]:
    start_date = pd.Timestamp(bulk["date"].min()).normalize()
    end_date = pd.Timestamp(bulk["date"].max()).normalize()
    return {
        "^NSEI": _official_market_context("^NSEI", start_date, end_date),
        "^NSEBANK": _official_market_context("^NSEBANK", start_date, end_date),
    }


def make_panel(bulk, symbols, sort_output=True, delivery=None):
    # Normalize once, then restrict the canonical research frame to the
    # trainable universe before any per-symbol feature work. The previous
    # implementation re-scanned the full multi-million-row frame for every
    # symbol, which made ALL-NSE panel construction unnecessarily expensive.
    # sort_output=False is intended for read-only audits that do not need the
    # final global ordering and avoids an expensive multi-million-row sort.
    work = bulk.copy()
    work["_symbol_norm"] = work["symbol"].map(normalize_symbol)

    wanted = {normalize_symbol(sym) for sym in symbols}
    work = work[work["_symbol_norm"].isin(wanted)].copy()
    if work.empty:
        return pd.DataFrame()

    if delivery is None:
        delivery = load_delivery_history(
            DEFAULT_DELIVERY_ROOT,
            symbols=wanted,
            start=work["date"].min(),
            end=work["date"].max(),
        )

    if delivery is not None and not delivery.empty:
        d = delivery.copy()
        d["symbol"] = d["symbol"].map(normalize_symbol)
        d["date"] = pd.to_datetime(
            d["date"],
            errors="coerce",
        ).dt.normalize()
        d = d.dropna(subset=["symbol", "date"]).copy()

        value_cols = [
            "delivery_qty",
            "delivery_pct",
        ]

        for c in value_cols:
            if c not in d.columns:
                d[c] = np.nan

        d = (
            d[["symbol", "date"] + value_cols]
            .sort_values(["symbol", "date"])
            .drop_duplicates(["symbol", "date"], keep="last")
        )

        work = work.merge(
            d,
            on=["symbol", "date"],
            how="left",
            suffixes=("", "_delivery"),
        )

        for c in value_cols:
            dc = f"{c}_delivery"
            if dc in work.columns:
                work[c] = work[dc]
                work = work.drop(columns=[dc])

    ctxs = market_context_tables(work)
    parts = []
    total = len(wanted)
    total_rows = 0
    for i, (sym, group) in enumerate(work.groupby("_symbol_norm", sort=False), 1):
        z = add_features(group.drop(columns=["_symbol_norm"]).copy())
        if z.empty:
            continue
        benchmark = benchmark_for_symbol(sym)
        ctx = ctxs.get(benchmark)
        if ctx is None or ctx.empty:
            raise RuntimeError(f"Missing official NSE benchmark context for {sym}: {benchmark}")
        z = z.merge(ctx, on="date", how="left")
        for n in (1, 6, 12):
            z[f"relative_ret_{n}"] = z[f"ret_{n}"] - z[f"market_ret_{n}"]
        z["decision_timestamp"] = session_close_timestamp_utc(z["date"])
        total_rows += len(z)
        parts.append(z)
        if i == 1 or i % 100 == 0 or i == total:
            print(
                f"base panel progress: {i:,}/{total:,} symbols; "
                f"rows={total_rows:,}",
                flush=True,
            )

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    if sort_output:
        out = out.sort_values(
            ["symbol", "decision_timestamp"]
        ).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out

def _merge_broad_fno_pit(panel: pd.DataFrame, source_frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Exact next-session join for official NSE derivatives data.

    A source-day F&O row is available from the following exchange session at
    09:15 IST. It must not be forward-filled to later dates. Missing symbols are
    marked fno_available=0 only on target sessions for which a source-day
    archive is actually present; other dates remain unknown (NaN).
    """
    path = BASE / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
    fno_columns = [
        "fno_pcr", "fno_oi_change", "fno_iv_change", "fno_available",
        "fno_fut_close", "fno_fut_basis", "fno_fut_return", "fno_fut_oi",
    ]
    if not path.exists() or path.stat().st_size == 0:
        out = panel.copy()
        for column in fno_columns:
            if column not in out.columns:
                out[column] = np.nan
        return out, {
            "exists": False, "rows": 0, "symbols": 0,
            "source_dates": 0, "covered_target_sessions": 0,
            "path": str(path), "pit_rule": "source_trade_date_available_next_session_09:15_IST",
        }

    f = source_frame.copy() if source_frame is not None else pd.read_parquet(path)
    if f.empty or not {"symbol", "date"}.issubset(f.columns):
        out = panel.copy()
        for column in fno_columns:
            if column not in out.columns:
                out[column] = np.nan
        return out, {
            "exists": False, "rows": 0, "symbols": 0,
            "source_dates": 0, "covered_target_sessions": 0,
            "path": str(path), "pit_rule": "source_trade_date_available_next_session_09:15_IST",
        }

    f = f.copy()
    f["symbol"] = f["symbol"].map(normalize_symbol)
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
    f = f.dropna(subset=["symbol", "date"]).copy()
    # Keep a factual coverage map separately from per-symbol rows. If a date
    # appears in the parsed official source archive, the archive is known for
    # that session even for listed names with no derivative observations.
    source_dates = sorted(pd.Timestamp(d).normalize() for d in f["date"].dropna().unique())

    panel_dates = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    session_values = sorted(pd.Timestamp(d).normalize() for d in panel_dates.dropna().unique())
    next_session = {session_values[i]: session_values[i + 1] for i in range(len(session_values) - 1)}
    f["next_session_date"] = f["date"].map(next_session)
    f = f.dropna(subset=["next_session_date"]).copy()
    f["next_session_date"] = pd.to_datetime(f["next_session_date"], errors="coerce").dt.normalize()

    covered_target_dates = {
        next_session[d] for d in source_dates if d in next_session
    }
    if f.empty:
        out = panel.copy()
        for column in fno_columns:
            if column not in out.columns:
                out[column] = np.nan
        return out, {
            "exists": True, "rows": 0, "symbols": 0,
            "source_dates": len(source_dates), "covered_target_sessions": len(covered_target_dates),
            "path": str(path), "pit_rule": "source_trade_date_available_next_session_09:15_IST",
        }

    value_map = {column: column for column in fno_columns if column in f.columns}
    f = f.rename(columns=value_map)
    for column in fno_columns:
        if column not in f.columns:
            f[column] = np.nan
    # An observed underlying in the official derivatives archive is applicable
    # even when some options/futures metrics cannot be calculated.
    f["fno_available"] = pd.to_numeric(f["fno_available"], errors="coerce")
    f["fno_available"] = f["fno_available"].fillna(1.0)
    f = f[["symbol", "next_session_date"] + fno_columns]
    f = f.sort_values(["symbol", "next_session_date"]).drop_duplicates(
        ["symbol", "next_session_date"], keep="last"
    )

    # The base panel should not contain the legacy three-stock pilot values.
    out = panel.drop(columns=fno_columns, errors="ignore").copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.merge(
        f,
        left_on=["symbol", "date"],
        right_on=["symbol", "next_session_date"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_fno"),
    )
    out = out.drop(columns=["next_session_date"], errors="ignore")
    # Zero means a verified archive exists for this session but this symbol did
    # not appear in the F&O underlyings. Uncovered dates stay unknown.
    covered = out["date"].isin(covered_target_dates)
    out.loc[covered, "fno_available"] = pd.to_numeric(
        out.loc[covered, "fno_available"], errors="coerce"
    ).fillna(0.0)

    return out, {
        "exists": True,
        "path": str(path),
        "rows": int(len(f)),
        "symbols": int(f["symbol"].nunique()),
        "source_dates": len(source_dates),
        "first_source_date": str(min(source_dates).date()) if source_dates else None,
        "last_source_date": str(max(source_dates).date()) if source_dates else None,
        "covered_target_sessions": len(covered_target_dates),
        "covered_target_date_start": str(min(covered_target_dates).date()) if covered_target_dates else None,
        "covered_target_date_end": str(max(covered_target_dates).date()) if covered_target_dates else None,
        "pit_rule": "exact_symbol_and_next_session_date; source_trade_date_available_next_session_09:15_IST; no forward-fill",
    }

def merge_optional(panel, source_cache=None):
    out = panel.copy()
    audit = {}
    fno_columns = [
        "fno_pcr", "fno_oi_change", "fno_iv_change", "fno_available",
        "fno_fut_close", "fno_fut_basis", "fno_fut_return", "fno_fut_oi",
    ]

    # The derivatives join is exact by next exchange session. It never carries
    # yesterday's F&O figures into later sessions without a source observation.
    out, fno_audit = _merge_broad_fno_pit(
        out, source_cache.get("fno") if source_cache is not None else None
    )
    audit["fno"] = fno_audit

    # A complete source manifest is required before missing per-symbol events
    # can truthfully mean zero. File existence alone is not source coverage.
    source_manifest_path = (
        BASE / "processed/pit_replay/nse_all_universe_raw_2026/"
        "ALL_NSE_NEWS_EVENTS_MANIFEST_2026.json"
    )
    source_manifest = {}
    if source_cache is not None:
        source_manifest = source_cache.get("news_events_manifest", {}) or {}
    else:
        try:
            if source_manifest_path.exists() and source_manifest_path.stat().st_size > 0:
                source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            audit["news_events_manifest_error"] = f"{type(exc).__name__}: {exc}"

    coverage_complete = source_manifest.get("coverage_complete") is True
    coverage_start = pd.to_datetime(source_manifest.get("coverage_start"), errors="coerce")
    coverage_end = pd.to_datetime(source_manifest.get("coverage_end"), errors="coerce")
    if pd.isna(coverage_start) or pd.isna(coverage_end):
        coverage_complete = False
    audit["news_events_coverage"] = {
        "manifest_path": str(source_manifest_path),
        "coverage_complete": bool(coverage_complete),
        "coverage_start": str(coverage_start.date()) if pd.notna(coverage_start) else None,
        "coverage_end": str(coverage_end.date()) if pd.notna(coverage_end) else None,
    }

    news_path = BASE / "processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet"
    events_path = BASE / "processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet"
    news = (
        source_cache.get("news", pd.DataFrame())
        if source_cache is not None else load_optional_layer(news_path)
    )
    events = (
        source_cache.get("events", pd.DataFrame())
        if source_cache is not None else load_optional_layer(events_path)
    )

    # These flags mean "at least one announcement/event in the last 7 days",
    # not "this symbol ever had a row in the archive". Old news must expire.
    freshness = pd.Timedelta(days=7)
    if not news.empty and "news_available" in news.columns:
        out = merge_pit_layer(out, news, ["news_available"], "news", max_age=freshness)
        audit["news"] = {
            "exists": True, "rows": int(len(news)),
            "symbols": int(news["symbol"].nunique()),
            "freshness_days": 7,
        }
    else:
        if "news_available" not in out.columns:
            out["news_available"] = np.nan
        audit["news"] = {"exists": False, "rows": 0, "freshness_days": 7}

    # The base panel does not necessarily contain this optional flag.
    # Create an explicitly unknown column before combining the separate
    # high-impact event stream; absence must remain NaN, not a fabricated zero.
    if "event_high_impact" not in out.columns:
        out["event_high_impact"] = np.nan

    if not events.empty and "event_available" in events.columns:
        out = merge_pit_layer(out, events, ["event_available"], "events", max_age=freshness)
        # High-impact status is independent of the latest low-impact event:
        # join only high-impact observations, so any such event stays active for
        # the freshness window even if a later low-impact event was published.
        if "event_high_impact" in events.columns:
            high = events[
                pd.to_numeric(events["event_high_impact"], errors="coerce").eq(1)
            ][["symbol", "decision_timestamp", "event_high_impact"]].copy()
            if not high.empty:
                high = high.rename(columns={"event_high_impact": "recent_high_impact"})
                out = merge_pit_layer(
                    out, high, ["recent_high_impact"], "high_impact_events",
                    max_age=freshness,
                )
                recent_high = pd.to_numeric(out["recent_high_impact"], errors="coerce")
                current_high = pd.to_numeric(out["event_high_impact"], errors="coerce")
                # If either independent stream has a recent high-impact event,
                # preserve it. If both are unknown, remain unknown until the
                # source coverage window authorizes a known-zero flag.
                out["event_high_impact"] = current_high.where(
                    recent_high.isna(),
                    np.maximum(current_high.fillna(0.0), recent_high),
                )
                out = out.drop(columns=["recent_high_impact"], errors="ignore")
        audit["events"] = {
            "exists": True, "rows": int(len(events)),
            "symbols": int(events["symbol"].nunique()),
            "high_impact_rows": int(
                pd.to_numeric(events.get("event_high_impact"), errors="coerce").eq(1).sum()
            ) if "event_high_impact" in events else 0,
            "freshness_days": 7,
        }
    else:
        for column in ("event_available", "event_high_impact"):
            if column not in out.columns:
                out[column] = np.nan
        audit["events"] = {"exists": False, "rows": 0, "freshness_days": 7}

    # Ensure the contract flags exist even when a complete source returned
    # zero rows for the requested window. Values remain NaN until verified
    # source coverage authorizes a known-zero state.
    for column in ("news_available", "event_available", "event_high_impact"):
        if column not in out.columns:
            out[column] = np.nan

    # Fill "no recent item" with zero only inside a successfully fetched,
    # explicitly recorded coverage window and only after a seven-day warm-up.
    # Outside that window, or after a failed/partial source build, retain NaN.
    if coverage_complete and not out.empty:
        local_dates = pd.to_datetime(out["decision_timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata").dt.normalize().dt.tz_localize(None)
        first_fully_covered_day = coverage_start.normalize() + pd.Timedelta(days=7)
        known_window = (
            local_dates.ge(first_fully_covered_day)
            & local_dates.le(coverage_end.normalize())
        )
        for column in ("news_available", "event_available", "event_high_impact"):
            if column in out.columns:
                out.loc[known_window, column] = pd.to_numeric(
                    out.loc[known_window, column], errors="coerce"
                ).fillna(0.0)
        audit["news_events_coverage"]["known_zero_rows"] = int(known_window.sum())
    else:
        audit["news_events_coverage"]["known_zero_rows"] = 0

    # ONLY the broad ALL-NSE PIT fundamentals layer is accepted here.
    # Never silently fall back to the legacy three-stock pilot.
    fp = BASE / "corporate/pit_fundamentals_all_nse_2025_2026.parquet"
    fp = fp if fp.exists() and fp.stat().st_size > 0 else None
    cached_fundamentals = (
        source_cache.get("fundamentals") if source_cache is not None else None
    )
    if cached_fundamentals is not None and not cached_fundamentals.empty:
        f = cached_fundamentals
        fp = fp or Path(source_cache.get("fundamentals_path", str(FUND_BROAD)))
    elif fp is not None:
        f = pd.read_parquet(fp)
        f["symbol"] = f["symbol"].map(normalize_symbol)
        f["fundamental_available_at"] = pd.to_datetime(
            f["fundamental_available_at"], errors="coerce", utc=True
        )
        f = f.dropna(subset=["symbol", "fundamental_available_at"]).sort_values(
            ["symbol", "fundamental_available_at"]
        )
        vals = [
            "fundamental_revenue", "fundamental_pat", "fundamental_eps",
            "fundamental_ebitda", "fundamental_finance_cost",
            "fundamental_total_equity", "fundamental_total_debt", "fundamental_cash",
            "fund_revenue_growth", "fund_pat_growth", "fund_eps_growth",
            "fund_ebitda_growth", "fund_pat_margin", "fund_debt_to_equity",
            "fund_cash_to_debt",
        ]
        pieces = []
        for sym, lp in out.groupby("symbol", sort=False):
            rp = f[f["symbol"].eq(sym)]
            if rp.empty:
                pieces.append(lp)
                continue
            available_values = [x for x in vals if x in rp.columns]
            cols = ["fundamental_available_at"] + available_values
            rp = rp[cols].drop_duplicates(["fundamental_available_at"], keep="last")
            m = pd.merge_asof(
                lp.sort_values("decision_timestamp"),
                rp.sort_values("fundamental_available_at"),
                left_on="decision_timestamp",
                right_on="fundamental_available_at",
                direction="backward",
                allow_exact_matches=True,
            )
            pieces.append(m)
        if pieces:
            out = pd.concat(pieces, ignore_index=True)
        fa = pd.to_datetime(out.get("fundamental_available_at"), errors="coerce", utc=True)
        parsed_valid = fa.notna() & fa.le(out["decision_timestamp"])
        out["fundamental_age_days"] = np.nan
        out.loc[parsed_valid, "fundamental_age_days"] = (
            out.loc[parsed_valid, "decision_timestamp"] - fa.loc[parsed_valid]
        ).dt.total_seconds() / 86400.0
        out["fundamental_stale_90d"] = pd.Series(
            np.where(parsed_valid, (out["fundamental_age_days"] > 90).astype(float), np.nan),
            index=out.index, dtype="float64",
        )
        out["fundamental_stale_180d"] = pd.Series(
            np.where(parsed_valid, (out["fundamental_age_days"] > 180).astype(float), np.nan),
            index=out.index, dtype="float64",
        )

        # Join a complete index of eligible NSE filing publication timestamps.
        # This separates a genuinely absent-as-of filing (known 0) from an
        # available filing that did not parse (unknown NaN, fail closed).
        availability_meta = {}
        if source_cache is not None:
            availability_meta = source_cache.get("fundamental_availability_meta", {}) or {}
        else:
            try:
                if FUND_AVAILABILITY_META.exists() and FUND_AVAILABILITY_META.stat().st_size:
                    availability_meta = json.loads(FUND_AVAILABILITY_META.read_text(encoding="utf-8"))
            except Exception as exc:
                audit["fundamental_availability_meta_error"] = f"{type(exc).__name__}: {exc}"
        availability_path = FUND_AVAILABILITY
        if (
            availability_meta.get("coverage_complete") is True
            and availability_path.exists()
            and availability_path.stat().st_size > 0
        ):
            availability = (
                source_cache.get("fundamental_availability")
                if source_cache is not None
                else pd.read_parquet(availability_path)
            )
            availability = availability.copy()
            availability["symbol"] = availability["symbol"].map(normalize_symbol)
            availability["fundamental_available_at"] = pd.to_datetime(
                availability["fundamental_available_at"], errors="coerce", utc=True
            )
            availability = availability.dropna(
                subset=["symbol", "fundamental_available_at"]
            ).copy()
            availability["decision_timestamp"] = availability["fundamental_available_at"]
            availability = availability.rename(
                columns={"fundamental_available_at": "catalog_filing_available_at"}
            )
            if availability.empty:
                out["catalog_filing_available_at"] = pd.NaT
            else:
                out = merge_pit_layer(
                    out,
                    availability[["symbol", "decision_timestamp", "catalog_filing_available_at"]],
                    ["catalog_filing_available_at"],
                    "fundamental_catalog",
                )
            # merge_pit_layer groups by symbol and concatenates rows, so re-read
            # parsed availability from the merged frame to maintain row alignment.
            fa = pd.to_datetime(out.get("fundamental_available_at"), errors="coerce", utc=True)
            parsed_valid = fa.notna() & fa.le(out["decision_timestamp"])
            local_dates = (
                pd.to_datetime(out["decision_timestamp"], errors="coerce", utc=True)
                .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
            )
            cov_start = pd.to_datetime(availability_meta.get("coverage_start"), errors="coerce")
            cov_end = pd.to_datetime(availability_meta.get("coverage_end"), errors="coerce")
            in_catalog_window = pd.Series(False, index=out.index)
            if pd.notna(cov_start) and pd.notna(cov_end):
                in_catalog_window = (
                    local_dates.ge(cov_start.normalize())
                    & local_dates.le(cov_end.normalize())
                )
            latest_catalog = pd.to_datetime(
                out.get("catalog_filing_available_at"), errors="coerce", utc=True
            )
            known_no_filing = in_catalog_window & latest_catalog.isna()
            latest_filing_parsed = (
                in_catalog_window
                & latest_catalog.notna()
                & parsed_valid
                & fa.ge(latest_catalog)
            )
            out["fundamentals_available"] = pd.Series(
                np.where(known_no_filing, 0.0, np.where(latest_filing_parsed, 1.0, np.nan)),
                index=out.index, dtype="float64",
            )
            out = out.drop(columns=["catalog_filing_available_at"], errors="ignore")
            audit["fundamental_availability_index"] = {
                **availability_meta,
                "exists": True,
                "rows": int(len(availability)),
                "symbols": int(availability["symbol"].nunique()),
                "known_no_filing_rows": int(known_no_filing.sum()),
                "parsed_asof_filing_rows": int(latest_filing_parsed.sum()),
                "unknown_or_parse_gap_rows": int(
                    (~known_no_filing & ~latest_filing_parsed).sum()
                ),
            }
        else:
            out["fundamentals_available"] = pd.Series(
                np.where(parsed_valid, 1.0, np.nan), index=out.index, dtype="float64"
            )
            audit["fundamental_availability_index"] = {
                "exists": False,
                "coverage_complete": False,
                "path": str(availability_path),
                "reason": "No verified complete NSE filing-availability index; absence remains unknown.",
            }
        # Ensure every fundamental contract column exists, but do not fabricate
        # numeric accounting values when NSE/XBRL did not publish/parse them.
        for column in [
            "fundamental_revenue", "fundamental_pat", "fundamental_eps",
            "fundamental_ebitda", "fundamental_finance_cost", "fundamental_total_equity",
            "fundamental_total_debt", "fundamental_cash", "fund_revenue_growth",
            "fund_pat_growth", "fund_eps_growth", "fund_ebitda_growth",
            "fund_pat_margin", "fund_debt_to_equity", "fund_cash_to_debt",
        ]:
            if column not in out.columns:
                out[column] = np.nan
        audit["fundamentals"] = {
            "exists": True, "path": str(fp), "rows": int(len(f)),
            "symbols": int(f["symbol"].nunique()),
            "feature_coverage": feature_coverage_report(out, [
                "fundamental_revenue", "fundamental_pat", "fundamental_eps",
                "fundamental_ebitda", "fundamental_finance_cost",
                "fundamental_total_equity", "fundamental_total_debt", "fundamental_cash",
                "fund_revenue_growth", "fund_pat_growth", "fund_eps_growth",
                "fund_ebitda_growth", "fund_pat_margin", "fund_debt_to_equity",
                "fund_cash_to_debt",
            ]),
        }
    else:
        audit["fundamentals"] = {
            "exists": False, "rows": 0,
            "error": "Broad PIT fundamentals source file is missing; readiness must remain unknown.",
        }
        out["fundamentals_available"] = pd.Series(np.nan, index=out.index, dtype="float64")
        out["fundamental_age_days"] = np.nan
        out["fundamental_stale_90d"] = np.nan
        out["fundamental_stale_180d"] = np.nan
        for column in [
            "fundamental_revenue", "fundamental_pat", "fundamental_eps",
            "fundamental_ebitda", "fundamental_finance_cost", "fundamental_total_equity",
            "fundamental_total_debt", "fundamental_cash", "fund_revenue_growth",
            "fund_pat_growth", "fund_eps_growth", "fund_ebitda_growth",
            "fund_pat_margin", "fund_debt_to_equity", "fund_cash_to_debt",
        ]:
            if column not in out.columns:
                out[column] = np.nan

    out = attach_readiness_flags(out)
    audit["feature_coverage"] = feature_coverage_report(out)
    return out, audit

def add_targets(panel,steps,name):
    x=panel.sort_values(["symbol","decision_timestamp"]).copy()
    with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
        x[f"target_return_{name}"]=x.groupby("symbol")["close"].shift(-steps)/x["close"]-1.0
    x[f"target_return_{name}"]=x[f"target_return_{name}"].replace([np.inf,-np.inf],np.nan)
    x[f"target_up_{name}"]=(x[f"target_return_{name}"]>0).astype(int)
    return x.dropna(subset=[f"target_return_{name}"])

def fit_models(frame, horizon):
    cutoff = frame["decision_timestamp"].max() - pd.Timedelta(days=60)
    tr = frame[frame["decision_timestamp"] < cutoff]
    te = frame[frame["decision_timestamp"] >= cutoff]
    if tr.empty or te.empty or tr[f"target_up_{horizon}"].nunique() < 2:
        raise RuntimeError(f"{horizon}: insufficient chronological train/holdout data")

    Xtr = tr[FEATURES_59].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    Xte = te[FEATURES_59].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
    yreg = tr[f"target_return_{horizon}"].replace([np.inf, -np.inf], np.nan)
    good = yreg.notna()
    Xtr = Xtr.loc[good]
    yreg = yreg.loc[good]
    yclf = tr.loc[good, f"target_up_{horizon}"].astype(int)
    if Xtr.empty or yclf.nunique() < 2:
        raise RuntimeError(f"{horizon}: finite training set is insufficient after non-finite-value cleanup")

    # Prevent a nominal 59-column model where one or more inputs were never
    # observed in training and therefore cannot contribute a learned signal.
    coverage = assert_training_feature_coverage(
        tr.loc[good], FEATURES_59, min_observations=100, min_fraction=0.005
    )
    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(C=.5, class_weight="balanced", solver="lbfgs", max_iter=2500, random_state=42)),
    ])
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("model", HistGradientBoostingRegressor(max_iter=180, learning_rate=.04, max_leaf_nodes=19, l2_regularization=2.0, random_state=42)),
    ])
    # Fit both targets on the exact same clean-label rows.
    clf.fit(Xtr, yclf)
    reg.fit(Xtr, yreg.astype(float))
    for fitted in (clf, reg):
        imp = fitted.named_steps.get("imputer")
        learned = int(getattr(imp, "n_features_in_", 0)) if imp is not None else 0
        if not learned and imp is not None and getattr(imp, "statistics_", None) is not None:
            learned = int(len(imp.statistics_))
        if learned != len(FEATURES_59):
            raise RuntimeError(f"{horizon}: fitted model contract has {learned} original features; expected {len(FEATURES_59)}")

    p = clf.predict_proba(Xte)[:, 1]
    pr = reg.predict(Xte)
    y = te[f"target_up_{horizon}"].to_numpy()
    return clf, reg, {
        "train_rows": int(len(Xtr)),
        "holdout_rows": int(len(te)),
        "holdout_accuracy": float(np.mean((p >= .5) == y)),
        "mean_abs_return_error": float(np.mean(np.abs(pr - te[f"target_return_{horizon}"].to_numpy()))),
        "holdout_mean_readiness_pct": float(te["data_readiness_score"].mean() * 100),
        "training_feature_coverage_mean_pct": coverage["mean_non_missing_pct"],
        "training_feature_coverage_gate": "PASS",
        "training_feature_coverage_minimum_rows_floor": coverage["minimum_observations_required_floor"],
    }

def _candidate_input_signature(cash_master,bulk,trainable,fund_report):
    optional=[
        BASE/"processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet",
        BASE/"processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet",
        BASE/"production_data/nse_official/fno/fno_daily_features_2026.parquet",
        BASE/"processed/pit_replay/nse_all_universe_raw_2026/ALL_NSE_NEWS_EVENTS_MANIFEST_2026.json",
        FUND_AVAILABILITY,
        FUND_AVAILABILITY_META,
        BASE/"corporate/pit_fundamentals_all_nse_2025_2026.parquet",
    ]
    sig={
        "engine_version":"2026-all-nse-delivery-multifactor-59-replacement-v1",
        "feature_contract_hash":hashlib.sha256(json.dumps(FEATURES_59,separators=(",",":")).encode()).hexdigest(),
        "feature_count":len(FEATURES_59),
        "training_feature_coverage_gate":"v1-min-100-and-0.5pct",
        "cash_master":str(cash_master),
        "cash_master_size":int(cash_master.stat().st_size) if cash_master.exists() else 0,
        "cash_master_mtime_ns":int(cash_master.stat().st_mtime_ns) if cash_master.exists() else 0,
        "bulk_rows":int(len(bulk)),
        "bulk_symbols":int(bulk["symbol"].nunique()),
        "bulk_min_date":str(pd.Timestamp(bulk["date"].min()).date()),
        "bulk_max_date":str(pd.Timestamp(bulk["date"].max()).date()),
        "trainable_hash":_fund_target_hash(trainable),
        "fundamental_meta_hash":sha256(FUND_BROAD_META) if FUND_BROAD_META.exists() else "",
    }
    for p in optional[:-1]:
        sig[str(p)]=({"exists":True,"size":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns} if p.exists() else {"exists":False})
    return sig

def _candidate_artifacts_complete(out):
    mp=out/"ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json"
    if not mp.exists() or mp.stat().st_size==0:
        return False
    try:
        m=json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return False
    if m.get("promotion_status")!="BRANCH_REPLACEMENT_ONLY" or m.get("features")!=FEATURES_59:
        return False
    for item in m.get("artifacts",{}).values():
        if not isinstance(item,dict) or item.get("status")!="CANDIDATE_READY":
            return False
        for k in ("classifier","regressor"):
            if not Path(item[k]).exists():
                return False
    return True



def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.parquet")
    frame.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def _file_signature(path: Path) -> dict:
    try:
        if not path.exists():
            return {"exists": False, "path": str(path)}
        st = path.stat()
        return {
            "exists": True, "path": str(path), "is_dir": path.is_dir(),
            "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns),
        }
    except OSError as exc:
        return {"exists": False, "path": str(path), "error": str(exc)}


def _tree_signature(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    if path.is_file():
        return _file_signature(path)
    count = 0
    total_size = 0
    latest_mtime = 0
    for root, _, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() not in {".parquet", ".csv", ".json"}:
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            count += 1
            total_size += int(st.st_size)
            latest_mtime = max(latest_mtime, int(st.st_mtime_ns))
    return {
        "exists": True, "path": str(path), "file_count": count,
        "total_size": total_size, "latest_mtime_ns": latest_mtime,
    }


def _build_roster_from_canonical_master(
    cash_master: Path, min_history: int = 60, force_refresh: bool = False
) -> dict:
    """Compute the trainable roster in DuckDB without materializing the full master in pandas."""
    import duckdb
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if not cash_master.exists() or cash_master.stat().st_size == 0:
        raise FileNotFoundError(f"Canonical NSE cash master missing or empty: {cash_master}")
    universe = set(official_nse_symbols(force_refresh=force_refresh))
    if not universe:
        raise RuntimeError("Official NSE universe is empty; refusing to create a training roster.")

    path_sql = str(cash_master).replace("'", "''")
    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=2")
        con.execute("PRAGMA memory_limit='2GB'")
        counts = con.execute(f"""
            SELECT
                UPPER(TRIM(CAST(symbol AS VARCHAR))) AS symbol,
                COUNT(DISTINCT TRY_CAST(date AS DATE)) AS history_sessions
            FROM read_parquet('{path_sql}')
            WHERE UPPER(TRIM(CAST(series AS VARCHAR))) IN ('EQ','BE','BZ','SM','ST','SZ')
              AND TRY_CAST(date AS DATE) IS NOT NULL
              AND TRY_CAST(open AS DOUBLE) IS NOT NULL
              AND TRY_CAST(high AS DOUBLE) IS NOT NULL
              AND TRY_CAST(low AS DOUBLE) IS NOT NULL
              AND TRY_CAST(close AS DOUBLE) IS NOT NULL
              AND TRY_CAST(close AS DOUBLE) > 0
            GROUP BY 1
        """).fetchdf()
    finally:
        con.close()

    counts["symbol"] = counts["symbol"].map(normalize_symbol)
    history_by_symbol = {
        str(row.symbol): int(row.history_sessions)
        for row in counts.itertuples(index=False)
        if str(row.symbol).strip()
    }
    available = sorted(universe.intersection(history_by_symbol))
    trainable = sorted(
        symbol for symbol in available
        if history_by_symbol.get(symbol, 0) >= int(min_history)
    )
    excluded = sorted(universe.difference(trainable))
    return {
        "universe_source": "NSE_OFFICIAL_EQUITY_L",
        "universe_as_of": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "universe_symbol_count": len(universe),
        "symbols_with_bulk_history": len(available),
        "trainable_symbol_count": len(trainable),
        "trainable_symbols": trainable,
        "excluded_symbol_count": len(excluded),
        "excluded_symbols": excluded,
        "minimum_history_sessions": int(min_history),
        "history_sessions_by_symbol": {
            symbol: history_by_symbol[symbol] for symbol in trainable
        },
        "cash_master_signature": _file_signature(cash_master),
        "roster_method": "DuckDB distinct valid cash sessions; no full pandas cash-master load",
    }


def _read_checkpoint(path: Path, stage: str) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing {stage} checkpoint: {path}. Run the previous stage first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise RuntimeError(f"{stage} checkpoint is not marked COMPLETE: {path}")
    return payload


def _load_roster_checkpoint(out: Path, cash_master: Path) -> dict:
    payload = _read_checkpoint(out / "TRAINING_UNIVERSE_2026.json", "universe")
    expected = _file_signature(cash_master)
    actual = payload.get("cash_master_signature", {})
    if expected != actual:
        raise RuntimeError(
            "Canonical cash master changed after the universe stage. "
            "Run --stage universe again before proceeding."
        )
    roster = payload.get("roster")
    if not isinstance(roster, dict) or not roster.get("trainable_symbols"):
        raise RuntimeError("Universe checkpoint has no trainable symbols.")
    return roster


def _load_panel_source_cache() -> dict:
    """Load the smaller optional PIT layers once per panel process, not once per symbol batch."""
    fno_path = BASE / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
    news_path = BASE / "processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet"
    events_path = BASE / "processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet"
    manifest_path = (
        BASE / "processed/pit_replay/nse_all_universe_raw_2026/"
        "ALL_NSE_NEWS_EVENTS_MANIFEST_2026.json"
    )
    fundamentals_path = BASE / "corporate/pit_fundamentals_all_nse_2025_2026.parquet"

    cache = {
        "fno": pd.read_parquet(fno_path) if fno_path.exists() and fno_path.stat().st_size else pd.DataFrame(),
        "news": load_optional_layer(news_path),
        "events": load_optional_layer(events_path),
        "fundamentals": pd.read_parquet(fundamentals_path) if fundamentals_path.exists() and fundamentals_path.stat().st_size else pd.DataFrame(),
        "fundamentals_path": str(fundamentals_path),
        "fundamental_availability": pd.read_parquet(FUND_AVAILABILITY) if FUND_AVAILABILITY.exists() and FUND_AVAILABILITY.stat().st_size else pd.DataFrame(),
        "fundamental_availability_meta": json.loads(FUND_AVAILABILITY_META.read_text(encoding="utf-8")) if FUND_AVAILABILITY_META.exists() and FUND_AVAILABILITY_META.stat().st_size else {},
        "news_events_manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() and manifest_path.stat().st_size else {},
    }
    print(
        "Optional PIT source cache: "
        f"F&O rows={len(cache['fno']):,}; news rows={len(cache['news']):,}; "
        f"event rows={len(cache['events']):,}; fundamental rows={len(cache['fundamentals']):,}; "
        f"filing-index rows={len(cache['fundamental_availability']):,}",
        flush=True,
    )
    return cache


def _panel_source_signature(
    cash_master: Path, delivery_root: Path, roster: dict, fund_report: dict
) -> dict:
    source_paths = {
        "fno": BASE / "production_data/nse_official/fno/fno_daily_features_2026.parquet",
        "news": BASE / "processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet",
        "events": BASE / "processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet",
        "news_events_manifest": BASE / "processed/pit_replay/nse_all_universe_raw_2026/ALL_NSE_NEWS_EVENTS_MANIFEST_2026.json",
        "fundamentals": FUND_BROAD,
        "fundamentals_meta": FUND_BROAD_META,
        "fundamental_availability": FUND_AVAILABILITY,
        "fundamental_availability_meta": FUND_AVAILABILITY_META,
    }
    return {
        "stage_version": 2,
        "feature_count": len(FEATURES_59),
        "feature_contract_hash": hashlib.sha256(
            json.dumps(FEATURES_59, separators=(",", ":")).encode()
        ).hexdigest(),
        "cash_master": _file_signature(cash_master),
        "trainable_symbols_hash": _hash_strings(roster["trainable_symbols"]),
        "delivery_tree": _tree_signature(delivery_root),
        "fundamentals_report_path": fund_report.get("path"),
        "fundamentals_rows": fund_report.get("rows"),
        "sources": {name: _file_signature(path) for name, path in source_paths.items()},
    }


def _feature_coverage_report_from_parquet(path: Path) -> dict:
    """Calculate feature coverage without keeping 59 distinct-value hash sets in RAM."""
    import duckdb
    import tempfile

    p = str(path).replace("'", "''")
    temp_dir = Path(tempfile.gettempdir()) / "indian_market_feature_coverage_duckdb"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_sql = "'" + str(temp_dir).replace("'", "''") + "'"

    # Keep the working set below Colab's common ~1.8 GiB process limit and let
    # DuckDB spill temporary state to local /tmp if required. The previous
    # single aggregate contained 59 COUNT(DISTINCT ...) expressions, which
    # built many large hash sets concurrently and failed after all panel
    # checkpoints had already been written.
    expressions = ["COUNT(*) AS rows", "COUNT(DISTINCT symbol) AS symbols"]
    for i, name in enumerate(FEATURES_59):
        q = '"' + name.replace('"', '""') + '"'
        v = f"TRY_CAST({q} AS DOUBLE)"
        expressions.append(
            f"SUM(CASE WHEN {v} IS NOT NULL AND isfinite({v}) THEN 1 ELSE 0 END) AS obs_{i}"
        )

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("PRAGMA threads=1")
        con.execute("PRAGMA memory_limit='1GB'")
        con.execute("PRAGMA preserve_insertion_order=false")
        con.execute(f"SET temp_directory={temp_sql}")

        # One lightweight pass computes row/symbol totals and per-feature
        # observation counts. Exact distinct counts are then calculated one
        # feature at a time so their hash tables do not accumulate together.
        row = con.execute(
            f"SELECT {', '.join(expressions)} FROM read_parquet('{p}')"
        ).fetchone()
        n_rows = int(row[0] or 0)
        n_symbols = int(row[1] or 0)
        features = {}

        for i, name in enumerate(FEATURES_59):
            observed = int(row[2 + i] or 0)
            q = '"' + name.replace('"', '""') + '"'
            v = f"TRY_CAST({q} AS DOUBLE)"
            unique = int(con.execute(
                f"SELECT COUNT(DISTINCT CASE "
                f"WHEN {v} IS NOT NULL AND isfinite({v}) THEN {v} "
                f"ELSE NULL END) FROM read_parquet('{p}')"
            ).fetchone()[0] or 0)
            features[name] = {
                "column_missing_from_frame": False,
                "non_missing_rows": observed,
                "missing_rows": n_rows - observed,
                "non_missing_pct": round(100.0 * observed / n_rows, 3) if n_rows else 0.0,
                "unique_non_missing_values": unique,
                "zero_coverage": observed == 0,
            }
    finally:
        con.close()

    return {
        "rows": n_rows,
        "feature_count": len(FEATURES_59),
        "mean_non_missing_pct": round(
            sum(v["non_missing_pct"] for v in features.values()) / len(FEATURES_59), 3
        ),
        "features_missing_as_columns": [],
        "features_with_zero_observations": [
            c for c, v in features.items() if v["zero_coverage"]
        ],
        "features": features,
        "symbols": n_symbols,
        "method": "memory-bounded DuckDB aggregation over persisted panel parquet; distinct counts computed one feature at a time",
    }

def _sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _prepare_training_sample(
    panel_path: Path, sample_path: Path, horizon: str, steps: int, max_training_rows: int
) -> dict:
    """Use DuckDB to label and deterministically sample without loading the full panel into pandas."""
    import duckdb
    import math
    from shutil import rmtree

    if max_training_rows < 1000:
        raise ValueError("--max-training-rows must be at least 1000.")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    temp_dir = Path(tempfile.gettempdir()) / f"indian_market_duckdb_tmp_{horizon}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    panel_sql = _sql_string(panel_path)
    temp_sql = _sql_string(temp_dir)
    cols = [
        "symbol", "decision_timestamp", "close", "data_readiness_score",
        "ready_market_price", "ready_market_context", "ready_news",
        "ready_events", "ready_fno", "ready_fundamentals",
    ] + list(FEATURES_59)
    cols = list(dict.fromkeys(cols))
    projection = ", ".join('"' + c.replace('"', '""') + '"' for c in cols)
    horizon_ret = f"target_return_{horizon}"
    horizon_up = f"target_up_{horizon}"
    con = duckdb.connect(database=":memory:")
    sample_tmp = sample_path.with_name(sample_path.stem + ".tmp.parquet")
    try:
        con.execute("PRAGMA threads=2")
        con.execute("PRAGMA memory_limit='3GB'")
        con.execute(f"SET temp_directory={temp_sql}")
        con.execute(f"""
            CREATE TEMP TABLE labelled AS
            WITH ordered AS (
                SELECT {projection},
                    LEAD(TRY_CAST(close AS DOUBLE), {int(steps)}) OVER (
                        PARTITION BY symbol ORDER BY decision_timestamp
                    ) AS _future_close
                FROM read_parquet({panel_sql})
            )
            SELECT * EXCLUDE (_future_close),
                _future_close / NULLIF(TRY_CAST(close AS DOUBLE), 0) - 1.0 AS "{horizon_ret}",
                CASE WHEN _future_close / NULLIF(TRY_CAST(close AS DOUBLE), 0) - 1.0 > 0
                     THEN 1 ELSE 0 END AS "{horizon_up}"
            FROM ordered
            WHERE _future_close IS NOT NULL
              AND TRY_CAST(close AS DOUBLE) > 0
        """)
        max_labelled_ts = con.execute(
            "SELECT MAX(decision_timestamp) FROM labelled"
        ).fetchone()[0]
        if max_labelled_ts is None:
            raise RuntimeError(f"{horizon}: no rows have a valid {steps}-session future close.")
        cutoff = pd.Timestamp(max_labelled_ts) - pd.Timedelta(days=60)
        cutoff_literal = _sql_string(cutoff.to_pydatetime())
        train_symbols = int(con.execute(
            f"SELECT COUNT(DISTINCT symbol) FROM labelled WHERE decision_timestamp < {cutoff_literal}"
        ).fetchone()[0] or 0)
        if train_symbols <= 0:
            raise RuntimeError(f"{horizon}: no rows available before the chronological holdout.")
        quota = max(1, int(math.ceil(max_training_rows / train_symbols)))

        rare_signal = """
            COALESCE(TRY_CAST(fno_available AS DOUBLE), 0) = 1
            OR fno_pcr IS NOT NULL OR fno_oi_change IS NOT NULL
            OR fno_iv_change IS NOT NULL OR fno_fut_close IS NOT NULL
            OR fno_fut_basis IS NOT NULL OR fno_fut_return IS NOT NULL OR fno_fut_oi IS NOT NULL
            OR COALESCE(TRY_CAST(news_available AS DOUBLE), 0) = 1
            OR COALESCE(TRY_CAST(event_available AS DOUBLE), 0) = 1
            OR COALESCE(TRY_CAST(event_high_impact AS DOUBLE), 0) = 1
        """
        con.execute(f"""
            CREATE TEMP TABLE sampled AS
            SELECT *, 'full_60_day_holdout' AS _sample_role
            FROM labelled WHERE decision_timestamp >= {cutoff_literal}
            UNION ALL
            SELECT *, 'preserved_rare_source_row' AS _sample_role
            FROM labelled
            WHERE decision_timestamp < {cutoff_literal} AND ({rare_signal})
            UNION ALL
            SELECT * EXCLUDE (_sample_rank), 'deterministic_history_sample' AS _sample_role
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY symbol
                    ORDER BY HASH(symbol, CAST(decision_timestamp AS VARCHAR))
                ) AS _sample_rank
                FROM labelled
                WHERE decision_timestamp < {cutoff_literal}
                  AND NOT ({rare_signal})
            ) ordinary
            WHERE _sample_rank <= {quota}
        """)
        counts = con.execute("""
            SELECT _sample_role, COUNT(*) AS rows
            FROM sampled GROUP BY _sample_role
        """).fetchdf()
        sample_stats = {
            str(item["_sample_role"]): int(item["rows"])
            for item in counts.to_dict("records")
        }
        sample_stats.update({
            "horizon": horizon,
            "steps": int(steps),
            "chronological_cutoff": str(cutoff),
            "max_regular_training_rows": int(max_training_rows),
            "trainable_symbols_before_cutoff": train_symbols,
            "per_symbol_regular_row_quota": quota,
            "labelled_population_rows": int(con.execute("SELECT COUNT(*) FROM labelled").fetchone()[0]),
            "sample_rows_total": int(con.execute("SELECT COUNT(*) FROM sampled").fetchone()[0]),
            "label_policy": "labels are calculated with full symbol histories before sampling; no sampled-row shift is used",
            "sampling_policy": "all 60-day holdout rows and all rows with F&O observations or positive news/events are preserved; other pre-cutoff rows are deterministically sampled per symbol",
        })
        con.execute(f"""
            COPY (SELECT * EXCLUDE (_sample_role) FROM sampled)
            TO {_sql_string(sample_tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        con.close()
        # The DuckDB spill directory is transient; the durable training sample
        # has already been written to Google Drive.
        try:
            rmtree(temp_dir)
        except Exception:
            pass
    os.replace(sample_tmp, sample_path)
    return sample_stats

def main():
    ap=argparse.ArgumentParser(
        description="Build the ALL-NSE 59-feature Delivery candidate as resumable, memory-controlled stages."
    )
    ap.add_argument("--cash-master",default=DEFAULT_CASH_MASTER)
    ap.add_argument("--bulk-root",default=DEFAULT_BULK)
    ap.add_argument("--allow-bulk-fallback",action="store_true")
    ap.add_argument("--delivery-root",default=str(DEFAULT_DELIVERY_ROOT))
    ap.add_argument("--out",default=DEFAULT_OUT)
    ap.add_argument("--min-history",type=int,default=60)
    ap.add_argument("--refresh-fundamentals",action="store_true")
    ap.add_argument("--retry-failed-fundamentals",action="store_true")
    ap.add_argument("--fundamental-workers",type=int,default=1)
    ap.add_argument("--refresh-universe",action="store_true")
    ap.add_argument("--force-rebuild-candidate",action="store_true")
    ap.add_argument("--panel-batch-size",type=int,default=75)
    ap.add_argument("--max-training-rows",type=int,default=750000)
    ap.add_argument(
        "--stage",
        choices=("universe","fundamentals","panel","train-next-session","train-3d","train-5d","train-7d"),
        default="universe",
        help="Run exactly one stage. Default is universe preparation only.",
    )
    args=ap.parse_args()
    if args.fundamental_workers < 1:
        raise ValueError("--fundamental-workers must be >= 1")
    if args.panel_batch_size < 1:
        raise ValueError("--panel-batch-size must be >= 1")
    if args.max_training_rows < 1000:
        raise ValueError("--max-training-rows must be at least 1000.")

    import gc
    import shutil

    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    cash_master=Path(args.cash_master)
    delivery_root=Path(args.delivery_root)
    roster_path=out/"TRAINING_UNIVERSE_2026.json"
    fund_stage_path=out/"FUNDAMENTALS_STAGE_REPORT_2026.json"
    panel_state_path=out/"PANEL_STAGE_STATE_2026.json"
    panel_meta_path=out/"PANEL_STAGE_META_2026.json"
    panel_path=out/"ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"
    readiness_path=out/"DATA_READINESS_REPORT_2026.json"
    shard_dir=out/"panel_shards"
    manifest_path=out/"ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json"
    candidate_meta=out/CANDIDATE_META_NAME

    # Stage 1: compute trainable symbols in DuckDB. The millions-row cash master
    # is scanned on disk and only the compact per-symbol counts are returned.
    if args.stage=="universe":
        roster=_build_roster_from_canonical_master(
            cash_master,args.min_history,force_refresh=args.refresh_universe
        )
        checkpoint={
            "stage":"universe",
            "status":"COMPLETE",
            "created_at":pd.Timestamp.now(tz="UTC").isoformat(),
            "cash_master_signature":_file_signature(cash_master),
            "roster":roster,
        }
        _atomic_json(roster_path,checkpoint)
        print(json.dumps({
            "stage":"universe","status":"COMPLETE",
            "universe_symbols":roster["universe_symbol_count"],
            "symbols_with_history":roster["symbols_with_bulk_history"],
            "trainable_symbols":roster["trainable_symbol_count"],
            "excluded_symbols":roster["excluded_symbol_count"],
            "checkpoint":str(roster_path),
            "next_command":"--stage fundamentals --refresh-fundamentals --retry-failed-fundamentals --fundamental-workers 1",
        },indent=2),flush=True)
        return

    roster=_load_roster_checkpoint(out,cash_master)
    trainable=list(roster["trainable_symbols"])
    if not trainable:
        raise RuntimeError("Universe checkpoint contains no trainable symbols.")

    # Stage 2: refresh fundamentals without loading the full cash master or
    # security-wise delivery dataset into pandas.
    if args.stage=="fundamentals":
        print(
            f"FUNDAMENTALS stage: trainable_symbols={len(trainable):,}; "
            f"workers={args.fundamental_workers}; root={BASE}",
            flush=True,
        )
        fund_report=_ensure_broad_fundamentals(
            trainable,
            force_refresh=args.refresh_fundamentals,
            retry_failed=args.retry_failed_fundamentals,
            max_workers=args.fundamental_workers,
        )
        if not FUND_BROAD.exists() or FUND_BROAD.stat().st_size==0:
            raise RuntimeError("Fundamentals function returned without a persisted broad PIT layer.")
        if not FUND_AVAILABILITY.exists() or not FUND_AVAILABILITY_META.exists():
            raise RuntimeError("Fundamentals function returned without the PIT filing availability index.")
        availability_meta=json.loads(FUND_AVAILABILITY_META.read_text(encoding="utf-8"))
        if availability_meta.get("coverage_complete") is not True:
            raise RuntimeError("Fundamental filing availability index is not coverage-complete.")
        _atomic_json(fund_stage_path,{
            "stage":"fundamentals","status":"COMPLETE",
            "created_at":pd.Timestamp.now(tz="UTC").isoformat(),
            "roster_hash":_hash_strings(trainable),
            "fundamentals_report":fund_report,
            "fundamentals_signature":_file_signature(FUND_BROAD),
            "fundamentals_meta_signature":_file_signature(FUND_BROAD_META),
            "availability_signature":_file_signature(FUND_AVAILABILITY),
            "availability_meta":availability_meta,
        })
        print(json.dumps({
            "stage":"fundamentals","status":"COMPLETE",
            "report":fund_report,
            "checkpoint":str(fund_stage_path),
            "next_command":"--stage panel --panel-batch-size 75",
        },indent=2,default=str),flush=True)
        return

    # Stage 3: chunked, checkpointed PIT panel generation. Each batch reads only
    # its security subset via DuckDB, writes a durable shard, and releases RAM.
    if args.stage=="panel":
        fund_stage=_read_checkpoint(fund_stage_path,"fundamentals")
        if fund_stage.get("roster_hash") != _hash_strings(trainable):
            raise RuntimeError("Fundamentals checkpoint belongs to a different universe; rerun fundamentals.")
        for required in (FUND_BROAD,FUND_BROAD_META,FUND_AVAILABILITY,FUND_AVAILABILITY_META):
            if not required.exists() or required.stat().st_size==0:
                raise FileNotFoundError(f"Required fundamentals artifact is missing: {required}")

        source_cache=_load_panel_source_cache()
        input_signature=_panel_source_signature(cash_master,delivery_root,roster,fund_stage["fundamentals_report"])
        panel_state={}
        if args.force_rebuild_candidate:
            if shard_dir.exists():
                shutil.rmtree(shard_dir)
            if panel_state_path.exists():
                panel_state_path.unlink()
            if panel_meta_path.exists():
                panel_meta_path.unlink()
            # A new panel invalidates the candidate index, but old model files
            # are not deleted; nothing will reference them until retrained.
            _atomic_json(manifest_path,{
                "engine_version":"2026-all-nse-delivery-multifactor-59-replacement-v2",
                "promotion_status":"BRANCH_REPLACEMENT_ONLY",
                "feature_count":len(FEATURES_59),"features":FEATURES_59,
                "artifacts":{},
            })
        if panel_state_path.exists() and panel_state_path.stat().st_size:
            panel_state=json.loads(panel_state_path.read_text(encoding="utf-8"))
            if panel_state.get("input_signature") != input_signature:
                raise RuntimeError(
                    "Panel sources changed since the last checkpoint. To intentionally rebuild all panel shards, rerun with --force-rebuild-candidate."
                )
        elif shard_dir.exists() and any(shard_dir.glob("*.parquet")):
            raise RuntimeError(
                f"Untracked panel shards exist in {shard_dir}. Use --force-rebuild-candidate to rebuild safely."
            )
        shard_dir.mkdir(parents=True,exist_ok=True)
        completed=panel_state.get("completed_batches",{})
        batch_size=int(args.panel_batch_size)
        batches=[trainable[i:i+batch_size] for i in range(0,len(trainable),batch_size)]
        total_batches=len(batches)
        for batch_index,symbol_batch in enumerate(batches,1):
            batch_id=f"{batch_index:05d}"
            shard_path=shard_dir/f"panel_part_{batch_id}.parquet"
            existing=completed.get(batch_id)
            if existing and shard_path.exists() and shard_path.stat().st_size>0:
                try:
                    import pyarrow.parquet as pq
                    rows=int(pq.ParquetFile(shard_path).metadata.num_rows)
                    if rows==int(existing.get("rows",-1)):
                        print(
                            f"PANEL batch {batch_index}/{total_batches}: checkpoint reused "
                            f"({len(symbol_batch)} symbols, {rows:,} rows)",
                            flush=True,
                        )
                        continue
                except Exception:
                    pass
                print(f"PANEL batch {batch_index}/{total_batches}: invalid shard, rebuilding",flush=True)

            print(
                f"PANEL batch {batch_index}/{total_batches}: loading {len(symbol_batch)} symbols",
                flush=True,
            )
            bulk=load_canonical_cash_research_history(cash_master,symbols=symbol_batch)
            if bulk.empty:
                raise RuntimeError(f"Panel batch {batch_id} returned empty canonical cash history.")
            delivery=load_delivery_history(
                delivery_root,
                symbols=symbol_batch,
                start=bulk["date"].min(),
                end=bulk["date"].max(),
            )
            panel,audit=merge_optional(
                make_panel(bulk,symbol_batch,delivery=delivery),
                source_cache=source_cache,
            )
            # Preserve the exact 59-column contract. Missing source values stay
            # NaN, and the coverage report/gate—not fabricated zeros—decides readiness.
            _atomic_parquet(panel,shard_path)
            import pyarrow.parquet as pq
            shard_rows=int(pq.ParquetFile(shard_path).metadata.num_rows)
            completed[batch_id]={
                "path":str(shard_path),
                "rows":shard_rows,
                "symbols":len(symbol_batch),
                "first_symbol":symbol_batch[0],
                "last_symbol":symbol_batch[-1],
                "size":int(shard_path.stat().st_size),
            }
            panel_state={
                "stage":"panel","status":"IN_PROGRESS",
                "input_signature":input_signature,
                "batch_size":batch_size,"total_batches":total_batches,
                "completed_batches":completed,
                "updated_at":pd.Timestamp.now(tz="UTC").isoformat(),
            }
            _atomic_json(panel_state_path,panel_state)
            print(
                f"PANEL batch {batch_index}/{total_batches} saved: rows={shard_rows:,}; "
                f"file_size_mb={shard_path.stat().st_size/1024/1024:.1f}",
                flush=True,
            )
            del bulk,delivery,panel,audit
            gc.collect()

        if len(completed)!=total_batches:
            raise RuntimeError(
                f"Panel checkpoint incomplete: {len(completed)}/{total_batches} batches saved."
            )
        shard_glob=str(shard_dir/"panel_part_*.parquet").replace("'","''")
        tmp_panel=panel_path.with_name(panel_path.stem+".tmp.parquet")
        import duckdb
        con=duckdb.connect(database=":memory:")
        temp_dir=out/"duckdb_panel_temp"
        temp_dir.mkdir(parents=True,exist_ok=True)
        try:
            con.execute("PRAGMA threads=2")
            con.execute("PRAGMA memory_limit='3GB'")
            con.execute(f"SET temp_directory={_sql_string(temp_dir)}")
            con.execute(
                f"COPY (SELECT * FROM read_parquet('{shard_glob}', union_by_name=true)) "
                f"TO {_sql_string(tmp_panel)} (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            final_rows=int(con.execute(
                f"SELECT COUNT(*) FROM read_parquet({_sql_string(tmp_panel)})"
            ).fetchone()[0])
            final_symbols=int(con.execute(
                f"SELECT COUNT(DISTINCT symbol) FROM read_parquet({_sql_string(tmp_panel)})"
            ).fetchone()[0])
        finally:
            con.close()
        os.replace(tmp_panel,panel_path)
        coverage=_feature_coverage_report_from_parquet(panel_path)
        readiness={
            "rows":coverage["rows"],"symbols":coverage["symbols"],
            "mean_feature_non_null_pct":coverage["mean_non_missing_pct"],
            "feature_columns_missing_before_readiness":[],
            "features_with_zero_observations":coverage["features_with_zero_observations"],
            "feature_coverage":coverage,
            "panel_build_mode":"symbol-batched-resumable",
            "training_gate":{
                "minimum_observations":100,
                "minimum_fraction_of_applicable_rows":0.005,
                "description":"Each feature must have real observations in the applicable training cohort; missingness is preserved.",
            },
        }
        _atomic_json(readiness_path,readiness)
        panel_meta={
            "stage":"panel","status":"COMPLETE",
            "created_at":pd.Timestamp.now(tz="UTC").isoformat(),
            "panel_path":str(panel_path),
            "panel_rows":final_rows,
            "panel_symbols":final_symbols,
            "feature_count":len(FEATURES_59),
            "feature_contract_hash":input_signature["feature_contract_hash"],
            "input_signature":input_signature,
            "trainable_symbols":trainable,
            "universe":{
                k:roster.get(k) for k in (
                    "universe_source","universe_as_of","universe_symbol_count",
                    "symbols_with_bulk_history","trainable_symbol_count",
                    "excluded_symbol_count","minimum_history_sessions",
                )
            },
            "roster":{
                "universe_symbol_count":roster["universe_symbol_count"],
                "trainable_symbol_count":roster["trainable_symbol_count"],
                "excluded_symbol_count":roster["excluded_symbol_count"],
                "minimum_history_sessions":roster["minimum_history_sessions"],
            },
            "cash_master":str(cash_master),
            "cash_source":"CANONICAL_IDENTITY_PRESERVED_RESEARCH_VIEW",
            "delivery_root":str(delivery_root),
            "delivery_signature":input_signature["delivery_tree"],
            "fundamentals_report":fund_stage["fundamentals_report"],
            "readiness_path":str(readiness_path),
            "coverage":coverage,
        }
        _atomic_json(panel_meta_path,panel_meta)
        _atomic_json(panel_state_path,{
            **panel_state,"status":"COMPLETE",
            "completed_at":pd.Timestamp.now(tz="UTC").isoformat(),
            "panel_path":str(panel_path),"panel_rows":final_rows,
        })
        print(json.dumps({
            "stage":"panel","status":"COMPLETE",
            "rows":final_rows,"symbols":final_symbols,
            "feature_coverage_mean_pct":coverage["mean_non_missing_pct"],
            "zero_coverage_features":coverage["features_with_zero_observations"],
            "panel_path":str(panel_path),
            "next_commands":[
                "--stage train-next-session",
                "--stage train-3d",
                "--stage train-5d",
                "--stage train-7d",
            ],
        },indent=2,default=str),flush=True)
        return

    # Each train command handles one horizon only. Label construction and
    # deterministic sampling are performed by DuckDB over the persisted panel;
    # pandas only loads the bounded training sample.
    stage_to_horizon={
        "train-next-session":("next_session",1),
        "train-3d":("3d",3),
        "train-5d":("5d",5),
        "train-7d":("7d",7),
    }
    if args.stage not in stage_to_horizon:
        raise RuntimeError(f"Unexpected stage: {args.stage}")
    panel_meta=_read_checkpoint(panel_meta_path,"panel")
    if panel_meta.get("feature_count")!=len(FEATURES_59) or not panel_path.exists():
        raise RuntimeError("Panel checkpoint feature contract/path is invalid; rebuild panel.")
    if panel_meta.get("panel_rows")<=0:
        raise RuntimeError("Persisted panel has zero rows; refusing to train.")
    horizon,steps=stage_to_horizon[args.stage]
    sample_path=out/"training_samples"/f"TRAINING_SAMPLE_{horizon}_2026.parquet"
    sample_meta_path=out/"training_samples"/f"TRAINING_SAMPLE_{horizon}_2026.meta.json"
    print(
        f"TRAIN stage {horizon}: building bounded sample from {panel_meta['panel_rows']:,} persisted rows",
        flush=True,
    )
    sample_stats=_prepare_training_sample(
        panel_path,sample_path,horizon,steps,args.max_training_rows
    )
    _atomic_json(sample_meta_path,{
        "stage":"training_sample","status":"COMPLETE",
        "panel_input_signature":panel_meta["input_signature"],
        "sample_path":str(sample_path),
        **sample_stats,
    })
    frame=pd.read_parquet(sample_path)
    missing=[c for c in FEATURES_59 if c not in frame.columns]
    if missing:
        raise RuntimeError(f"{horizon}: training sample missing features: {missing}")
    clf,reg,metrics=fit_models(frame,horizon)
    metrics["sampling"]={
        **sample_stats,
        "sample_meta_path":str(sample_meta_path),
    }

    cp=out/f"deployment_all_nse_{horizon}_multifactor59_classifier_2026.joblib"
    rp=out/f"deployment_all_nse_{horizon}_multifactor59_regressor_2026.joblib"
    joblib.dump(clf,cp)
    joblib.dump(reg,rp)
    artifact={
        "status":"CANDIDATE_READY",
        "classifier":str(cp),"regressor":str(rp),
        "classifier_sha256":sha256(cp),"regressor_sha256":sha256(rp),
        "metrics":metrics,
        "training_sample":str(sample_path),
    }

    existing={}
    if manifest_path.exists() and manifest_path.stat().st_size:
        try:
            existing=json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing={}
    same_inputs=existing.get("panel_input_signature")==panel_meta.get("input_signature")
    artifacts=existing.get("artifacts",{}) if same_inputs else {}
    artifacts[horizon]=artifact
    manifest={
        "engine_version":"2026-all-nse-delivery-multifactor-59-replacement-v2",
        "promotion_status":"BRANCH_REPLACEMENT_ONLY",
        "universe":panel_meta["universe"],
        **panel_meta["roster"],
        "cash_data_source":panel_meta["cash_source"],
        "delivery_data_source":"NSE_OFFICIAL_SECURITY_WISE_DELIVERY_MTO",
        "delivery_root":str(delivery_root),
        "canonical_cash_master":str(cash_master),
        "trainable_symbols":panel_meta["trainable_symbols"],
        "feature_count":len(FEATURES_59),"features":FEATURES_59,
        "feature_contract_hash":panel_meta["feature_contract_hash"],
        "artifacts":artifacts,
        "panel_input_signature":panel_meta["input_signature"],
        "data_readiness_path":str(readiness_path),
        "pit_policy":"Only observations available by the 15:30 IST session decision time are eligible; missing information is never fabricated.",
        "sampling_policy":"DuckDB labels on full per-symbol chronology; cap ordinary pre-cutoff rows per symbol, preserve all 60-day holdout rows and positive news/events/F&O-observation rows.",
        "no_auto_orders":True,
        "branch_replacement_target":"production_delivery_model",
    }
    _atomic_json(manifest_path,manifest)
    _atomic_json(candidate_meta,{
        "panel_input_signature":panel_meta["input_signature"],
        "manifest_path":str(manifest_path),
        "last_trained_horizon":horizon,
        "training_sample_meta_path":str(sample_meta_path),
        "promotion_status":"BRANCH_REPLACEMENT_ONLY",
    })
    del frame,clf,reg
    gc.collect()
    print(json.dumps({
        "stage":args.stage,"status":"COMPLETE",
        "horizon":horizon,"metrics":metrics,
        "artifacts":artifacts[horizon],
        "manifest_path":str(manifest_path),
        "next_stages":[x for x in stage_to_horizon if x != args.stage],
        "promotion_status":"BRANCH_REPLACEMENT_ONLY",
    },indent=2,default=str),flush=True)

if __name__=="__main__":
    main()
