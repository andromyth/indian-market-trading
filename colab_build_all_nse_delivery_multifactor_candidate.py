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
from delivery_training_universe import benchmark_for_symbol, training_roster
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
            if same_window and fresh and isinstance(cached.get("data"),list):
                print(f"fundamental catalog cache reused rows={len(cached['data']):,}", flush=True)
                return cached["data"]
        except Exception:
            pass

    s=_fund_session(); out=[]; size=500
    for page in range(1,401):
        params={"type":"Integrated Filing- Financials","index":"equities","from_date":from_date,"to_date":to_date,"page":page,"size":size}
        last=None
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
                time.sleep(1.0*(attempt+1))
        if last:
            raise last
        if not batch:
            break
        out.extend(batch)
        if page%20==0:
            print(f"fundamental catalog pages={page} rows={len(out):,}", flush=True)
        if len(batch)<size:
            break

    FUND_CATALOG_CACHE.parent.mkdir(parents=True,exist_ok=True)
    from datetime import datetime, timezone
    FUND_CATALOG_CACHE.write_text(json.dumps({
        "meta":{
            "cache_version":"1",
            "from_date":from_date,
            "to_date":to_date,
            "fetched_at":datetime.now(timezone.utc).isoformat(),
            "rows":len(out),
        },
        "data":out,
    },default=str),encoding="utf-8")
    print(f"fundamental catalog rows={len(out):,}", flush=True)
    return out

def _fund_filings(catalog,symbols,max_each=4):
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
    FUND_FILING_CACHE.parent.mkdir(parents=True,exist_ok=True)
    cache.to_parquet(FUND_FILING_CACHE,index=False)

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

def _ensure_broad_fundamentals(symbols,force_refresh=False,retry_failed=False):
    catalog=_catalog_all(force_refresh=force_refresh)
    filings=_fund_filings(catalog,set(symbols),max_each=4)
    if not filings:
        raise RuntimeError("ALL-NSE PIT fundamentals: no eligible NSE Integrated Filing records found")

    sig=_fundamental_build_signature(symbols,filings)
    if not force_refresh and FUND_BROAD.exists() and FUND_BROAD.stat().st_size>0:
        try:
            meta=json.loads(FUND_BROAD_META.read_text(encoding="utf-8")) if FUND_BROAD_META.exists() else {}
            if meta and _broad_meta_matches(meta,sig):
                print(f"fundamentals build reused from manifest; rows={int(meta.get('rows',0)):,}",flush=True)
                return {"path":str(FUND_BROAD),"rebuilt":False,"cached":True,"symbols":int(meta.get("symbols",0)),"rows":int(meta.get("rows",0))}
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
                    return {"path":str(FUND_BROAD),"rebuilt":False,"cached":True,
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
    if missing:
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs=[ex.submit(_parse_one_cached,f) for f in missing]
            for i,fut in enumerate(as_completed(futs),1):
                new_rows.append(fut.result())
                if i%500==0:
                    parsed_now=sum(r.get("parse_status")=="parsed" for r in new_rows)
                    print(f"fundamentals parse={i:,}/{len(missing):,} new_parsed={parsed_now:,}",flush=True)

    if new_rows:
        filing_cache=pd.concat([filing_cache,pd.DataFrame(new_rows)],ignore_index=True)
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

def _merge_broad_fno_pit(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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

    f = pd.read_parquet(path)
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
        "source_dates": source_date_rows,
        "first_source_date": str(min(source_dates).date()) if source_dates else None,
        "last_source_date": str(max(source_dates).date()) if source_dates else None,
        "covered_target_sessions": len(covered_target_dates),
        "covered_target_date_start": str(min(covered_target_dates).date()) if covered_target_dates else None,
        "covered_target_date_end": str(max(covered_target_dates).date()) if covered_target_dates else None,
        "pit_rule": "exact_symbol_and_next_session_date; source_trade_date_available_next_session_09:15_IST; no forward-fill",
    }

def merge_optional(panel):
    out = panel.copy()
    audit = {}
    fno_columns = [
        "fno_pcr", "fno_oi_change", "fno_iv_change", "fno_available",
        "fno_fut_close", "fno_fut_basis", "fno_fut_return", "fno_fut_oi",
    ]

    # The derivatives join is exact by next exchange session. It never carries
    # yesterday's F&O figures into later sessions without a source observation.
    out, fno_audit = _merge_broad_fno_pit(out)
    audit["fno"] = fno_audit

    # A complete source manifest is required before missing per-symbol events
    # can truthfully mean zero. File existence alone is not source coverage.
    source_manifest_path = (
        BASE / "processed/pit_replay/nse_all_universe_raw_2026/"
        "ALL_NSE_NEWS_EVENTS_MANIFEST_2026.json"
    )
    source_manifest = {}
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
    news = load_optional_layer(news_path)
    events = load_optional_layer(events_path)

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

    if not events.empty and "event_available" in events.columns:
        out = merge_pit_layer(out, events, ["event_available"], "events", max_age=freshness)
        # High-impact status is independent of the latest low-impact event:
        # join only high-impact observations, so any such event stays active for
        # the freshness window even if a later low-impact event was published.
        if "event_high_impact" in events.columns:
            high = events[
                pd.to_numeric(events["event_high_impact"], errors="coerce").eq(1)
            ][["symbol", "decision_timestamp", "event_high_impact"]].copy()
            out = merge_pit_layer(
                out, high, ["event_high_impact"], "high_impact_events",
                max_age=freshness,
            )
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
    if fp is not None:
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
        valid = fa.notna() & fa.le(out["decision_timestamp"])
        out["fundamentals_available"] = valid.astype("Int8")
        out["fundamental_age_days"] = np.nan
        out.loc[valid, "fundamental_age_days"] = (
            out.loc[valid, "decision_timestamp"] - fa.loc[valid]
        ).dt.total_seconds() / 86400.0
        out["fundamental_stale_90d"] = pd.Series(
            np.where(valid, (out["fundamental_age_days"] > 90).astype(float), np.nan),
            index=out.index, dtype="float64",
        )
        out["fundamental_stale_180d"] = pd.Series(
            np.where(valid, (out["fundamental_age_days"] > 180).astype(float), np.nan),
            index=out.index, dtype="float64",
        )
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

    Xtr = tr[FEATURES_59].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xte = te[FEATURES_59].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
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
        Xtr, FEATURES_59, min_observations=100, min_fraction=0.005
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
        "training_feature_coverage_minimum_rows": coverage["minimum_observations_required"],
    }

def _candidate_input_signature(cash_master,bulk,trainable,fund_report):
    optional=[
        BASE/"processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet",
        BASE/"processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet",
        BASE/"production_data/nse_official/fno/fno_daily_features_2026.parquet",
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--cash-master",
        default=DEFAULT_CASH_MASTER,
        help="Canonical identity-preserved NSE cash master. This is the authoritative training input.",
    )
    ap.add_argument("--bulk-root",default=DEFAULT_BULK, help="Legacy official bulk-EOD fallback; unused unless --allow-bulk-fallback is set.")
    ap.add_argument(
        "--allow-bulk-fallback",
        action="store_true",
        help="Explicitly allow the older official bulk-EOD view when the canonical cash master is unavailable.",
    )
    ap.add_argument("--delivery-root",default=str(DEFAULT_DELIVERY_ROOT),
                    help="Persisted official NSE Security-wise Delivery Position files.")
    ap.add_argument("--out",default=DEFAULT_OUT)
    ap.add_argument("--min-history",type=int,default=60)
    ap.add_argument("--refresh-fundamentals",action="store_true")
    ap.add_argument("--retry-failed-fundamentals",action="store_true")
    ap.add_argument("--refresh-universe",action="store_true")
    ap.add_argument("--force-rebuild-candidate",action="store_true")
    args=ap.parse_args()

    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    cash_master=Path(args.cash_master)
    if cash_master.exists() and cash_master.stat().st_size>0:
        print(f"Using canonical identity-preserved NSE cash master: {cash_master}",flush=True)
        bulk=load_canonical_cash_research_history(cash_master)
        bulk_source="CANONICAL_IDENTITY_PRESERVED_RESEARCH_VIEW"
    elif args.allow_bulk_fallback:
        bulk_root=Path(args.bulk_root)
        print(f"WARNING: canonical cash master unavailable; using explicit official bulk-EOD fallback: {bulk_root}",flush=True)
        bulk=load_bulk(bulk_root)
        bulk_source="OFFICIAL_BULK_EOD_FALLBACK"
    else:
        raise FileNotFoundError(
            f"Canonical NSE cash master missing or empty: {cash_master}. "
            "Use --allow-bulk-fallback only for a deliberate non-canonical run."
        )
    if bulk.empty:
        raise RuntimeError("Canonical NSE cash research view is empty")
    roster=training_roster(bulk,min_history=args.min_history,force_refresh=args.refresh_universe)
    trainable=list(roster["trainable_symbols"])

    delivery_root = Path(args.delivery_root)
    delivery = load_delivery_history(
        delivery_root,
        symbols=trainable,
        start=bulk["date"].min(),
        end=bulk["date"].max(),
    )
    print(
        f"Using persisted NSE delivery layer: {delivery_root}; "
        f"rows={len(delivery):,}; symbols={delivery['symbol'].nunique() if not delivery.empty else 0:,}",
        flush=True,
    )

    fund_report=_ensure_broad_fundamentals(
        trainable,
        force_refresh=args.refresh_fundamentals,
        retry_failed=args.retry_failed_fundamentals,
    )

    universe={
        "universe_source":roster.get("universe_source"),
        "universe_as_of":roster.get("universe_as_of"),
        "universe_symbol_count":int(roster["universe_symbol_count"]),
        "symbols_with_bulk_history":int(roster["symbols_with_bulk_history"]),
        "trainable_symbol_count":int(roster["trainable_symbol_count"]),
        "excluded_symbol_count":int(roster["excluded_symbol_count"]),
        "minimum_history_sessions":int(roster["minimum_history_sessions"]),
    }

    input_sig=_candidate_input_signature(cash_master,bulk,trainable,fund_report)
    input_sig["cash_source"]=bulk_source
    input_sig["delivery_root"]=str(delivery_root)
    input_sig["delivery_rows"]=int(len(delivery))
    input_sig["delivery_symbols"]=int(delivery["symbol"].nunique()) if not delivery.empty else 0
    candidate_meta=out/CANDIDATE_META_NAME
    if not args.force_rebuild_candidate and not args.refresh_fundamentals and candidate_meta.exists() and _candidate_artifacts_complete(out):
        try:
            old=json.loads(candidate_meta.read_text(encoding="utf-8"))
            if old.get("input_signature")==input_sig:
                print("CANDIDATE BUILD REUSED; inputs and feature contract unchanged.",flush=True)
                return
        except Exception:
            pass

    panel,audit=merge_optional(make_panel(bulk,trainable,delivery=delivery))
    source_coverage = feature_coverage_report(panel)
    print(
        "59-feature data coverage: "
        f"mean_non_missing={source_coverage['mean_non_missing_pct']:.3f}% "
        f"zero_coverage={source_coverage['features_with_zero_observations']} "
        f"missing_columns={source_coverage['features_missing_as_columns']}",
        flush=True,
    )
    (out/"DATA_READINESS_REPORT_2026.json").write_text(
        json.dumps({
            "universe": universe,
            "readiness": readiness_report(panel),
            "feature_coverage": source_coverage,
            "optional_sources": audit,
            "feature_training_gate": {
                "minimum_observations": 100,
                "minimum_fraction_of_training_rows": 0.005,
                "description": "Each of the 59 features must have enough real observations in the training fold; sparse per-security values may remain missing and be imputed, but never fabricate values.",
            },
        }, indent=2, default=str),
        encoding="utf-8",
    )
    panel.to_parquet(out/"ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet",index=False)

    artifacts={}
    for horizon,steps in (("next_session",1),("3d",3),("5d",5),("7d",7)):
        cp=out/f"deployment_all_nse_{horizon}_multifactor59_classifier_2026.joblib"
        rp=out/f"deployment_all_nse_{horizon}_multifactor59_regressor_2026.joblib"
        try:
            d=add_targets(panel,steps,horizon)
            clf,reg,metrics=fit_models(d,horizon)
            # Save only after the 59-input contract has been checked.
            joblib.dump(clf,cp)
            joblib.dump(reg,rp)
            artifacts[horizon]={
                "status":"CANDIDATE_READY",
                "classifier":str(cp),
                "regressor":str(rp),
                "classifier_sha256":sha256(cp),
                "regressor_sha256":sha256(rp),
                "metrics":metrics,
            }
        except Exception as exc:
            if cp.exists():
                cp.unlink()
            if rp.exists():
                rp.unlink()
            artifacts[horizon]={
                "status":"BLOCKED",
                "reason":f"{type(exc).__name__}: {exc}",
            }

    manifest={
        "engine_version":"2026-all-nse-delivery-multifactor-59-replacement-v1",
        "promotion_status":"BRANCH_REPLACEMENT_ONLY",
        "universe":universe,
        "universe_symbol_count":int(roster["universe_symbol_count"]),
        "trainable_symbol_count":int(roster["trainable_symbol_count"]),
        "excluded_symbol_count":int(roster["excluded_symbol_count"]),
        "minimum_history_sessions":int(roster["minimum_history_sessions"]),
        "cash_data_source":bulk_source,
        "delivery_data_source":"NSE_OFFICIAL_SECURITY_WISE_DELIVERY_MTO",
        "delivery_root":str(delivery_root),
        "delivery_row_count":int(len(delivery)),
        "delivery_symbol_count":int(delivery["symbol"].nunique()) if not delivery.empty else 0,
        "canonical_cash_master":str(cash_master),
        "research_series_policy":["EQ","BE","BZ","SM","ST","SZ"],
        "t0_excluded_from_delivery_research":True,
        "trainable_symbols":trainable,
        "feature_count":len(FEATURES_59),
        "features":FEATURES_59,
        "feature_contract_hash":input_sig["feature_contract_hash"],
        "artifacts":artifacts,
        "data_readiness_path":str(out/"DATA_READINESS_REPORT_2026.json"),
        "pit_policy":"Only observations at or before decision_timestamp are eligible. Missing information is not fabricated.",
        "no_auto_orders":True,
        "branch_replacement_target":"production_delivery_model",
    }
    (out/"ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json").write_text(
        json.dumps(manifest,indent=2,default=str),
        encoding="utf-8",
    )
    candidate_meta.write_text(json.dumps({
        "input_signature":input_sig,
        "manifest_path":str(out/"ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json"),
        "fundamentals_report":fund_report,
    },indent=2,default=str),encoding="utf-8")
    print(json.dumps(manifest,indent=2,default=str),flush=True)

if __name__=="__main__":
    main()
