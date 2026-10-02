#!/usr/bin/env python3
"""Build ALL-NSE 59-feature multi-factor Delivery candidates in Colab."""
from __future__ import annotations
import argparse, hashlib, json, threading, time, xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
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
from nse_data_layer import fetch_index_history
from multifactor_data_layer_v2 import FEATURES_59, attach_readiness_flags, load_optional_layer, merge_pit_layer, normalize_symbol, readiness_report

DEFAULT_BULK="/content/drive/MyDrive/Indian_Market_Research/production_data/nse_official/bulk_eod"
DEFAULT_OUT="/content/drive/MyDrive/Indian_Market_Research/models/all_nse_delivery_multifactor_candidate_2026"
BASE=Path("/content/drive/MyDrive/Indian_Market_Research")

FUND_BROAD=BASE/"corporate/pit_fundamentals_all_nse_2025_2026.parquet"
FUND_CACHE=BASE/"corporate/xbrl_cache/all_nse_2025_2026"
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
    return pd.to_datetime(x,errors="coerce",utc=True,dayfirst=True) if x is not None else pd.NaT
def _fn(x):
    try:
        t=str(x).strip().replace(",","")
        if t.startswith("(") and t.endswith(")"): t="-"+t[1:-1]
        return float(t)
    except Exception: return np.nan
def _fl(tag): return str(tag).split("}",1)[-1].split(":",1)[-1]
def _catalog_all():
    s=_fund_session(); out=[]; size=500
    for page in range(1,401):
        params={"type":"Integrated Filing- Financials","index":"equities","from_date":"01-01-2025","to_date":"30-09-2026","page":page,"size":size}
        last=None
        for attempt in range(3):
            try:
                rr=s.get(NSE_IF_API,params=params,timeout=30); rr.raise_for_status(); payload=rr.json(); batch=payload.get("data",[]) if isinstance(payload,dict) else payload; last=None; break
            except Exception as exc: last=exc; time.sleep(1.0*(attempt+1))
        if last: raise last
        if not batch: break
        out.extend(batch)
        if page%20==0: print(f"fundamental catalog pages={page} rows={len(out):,}")
        if len(batch)<size: break
    print(f"fundamental catalog rows={len(out):,}")
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

        # For income/flow fields prefer the shortest clean duration ending on the
        # filing period (normally the quarter) with non-empty facts. If there is
        # no exact-period context, use the closest period-end duration.
        duration_ids.sort(key=lambda x:(x[0],x[1]))
        q={}
        for diff_days,duration,cid in duration_ids:
            if facts.get(cid):
                q=facts[cid]
                if diff_days <= 3:
                    break
                # keep the closest available context when no exact period exists
        bs={}
        for diff_days,cid in sorted(instant_ids):
            if facts.get(cid):
                bs=facts[cid]
                if diff_days <= 3:
                    break

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
def _ensure_broad_fundamentals(symbols):
    if FUND_BROAD.exists() and FUND_BROAD.stat().st_size>0:
        try:
            q=pd.read_parquet(FUND_BROAD); cov=q["symbol"].map(normalize_symbol).nunique() if "symbol" in q else 0
            if cov>=max(100,int(len(symbols)*0.8)): return {"path":str(FUND_BROAD),"rows":len(q),"symbols":cov,"rebuilt":False}
        except Exception: pass
    print(f"Building broad PIT fundamentals for {len(symbols):,} trainable symbols...")
    cat=_catalog_all(); filings=_fund_filings(cat,set(symbols),max_each=4)
    parsed=[]
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs=[ex.submit(_fund_parse,f) for f in filings]
        for i,fut in enumerate(as_completed(futs),1):
            r=fut.result()
            if r: parsed.append(r)
            if i%500==0: print(f"fundamentals XBRL={i:,}/{len(filings):,} parsed={len(parsed):,}")
    if not parsed: raise RuntimeError("ALL-NSE PIT fundamentals: no XBRL filings could be parsed")
    d=pd.DataFrame(parsed); d["symbol"]=d["symbol"].map(normalize_symbol); d["period_end"]=pd.to_datetime(d["period_end"],utc=True).dt.normalize(); d["fundamental_available_at"]=pd.to_datetime(d["fundamental_available_at"],utc=True)
    for c in FUND_FIELDS: d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.sort_values(["symbol","period_end","fundamental_available_at"]).drop_duplicates(["symbol","period_end"],keep="last")
    pieces=[]
    for sym,g in d.groupby("symbol",sort=False):
        g=g.sort_values("period_end").copy()
        for base,growth in (("fundamental_revenue","fund_revenue_growth"),("fundamental_pat","fund_pat_growth"),("fundamental_eps","fund_eps_growth"),("fundamental_ebitda","fund_ebitda_growth")): g[growth]=g[base].pct_change(fill_method=None)
        g["fund_pat_margin"]=g["fundamental_pat"]/g["fundamental_revenue"].replace(0,np.nan); g["fund_debt_to_equity"]=g["fundamental_total_debt"]/g["fundamental_total_equity"].replace(0,np.nan); g["fund_cash_to_debt"]=g["fundamental_cash"]/g["fundamental_total_debt"].replace(0,np.nan); pieces.append(g)
    d=pd.concat(pieces,ignore_index=True).replace([np.inf,-np.inf],np.nan); FUND_BROAD.parent.mkdir(parents=True,exist_ok=True); d.to_parquet(FUND_BROAD,index=False)
    report={"path":str(FUND_BROAD),"rows":len(d),"symbols":int(d["symbol"].nunique()),"target_symbols":len(symbols),"symbol_coverage_pct":round(100*d["symbol"].nunique()/len(symbols),3) if symbols else 0.0,"filings_selected":len(filings),"filings_parsed":len(parsed),"feature_coverage_pct":{c:round(100*d[c].notna().mean(),3) for c in FUND_FIELDS}}
    FUND_BROAD.with_suffix(".json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); return report

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

def _official_market_context(index_symbol: str, start_date, end_date) -> pd.DataFrame:
    """Load official NSE benchmark history from Drive first, API second."""
    frame = fetch_index_history(
        index_symbol,
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


def make_panel(bulk, symbols):
    # Normalize the bulk symbol column before filtering. The training roster
    # is built on normalized symbols; filtering against raw symbol text can
    # silently collapse the panel when archive symbols carry series/suffix
    # variants such as '-EQ'.
    work = bulk.copy()
    work["_symbol_norm"] = work["symbol"].map(normalize_symbol)
    ctxs = market_context_tables(work)
    parts = []
    for sym in symbols:
        z = add_features(work[work["_symbol_norm"].eq(sym)].drop(columns=["_symbol_norm"]).copy())
        if z.empty:
            continue
        benchmark = benchmark_for_symbol(sym)
        ctx = ctxs.get(benchmark)
        if ctx is None or ctx.empty:
            raise RuntimeError(f"Missing official NSE benchmark context for {sym}: {benchmark}")
        z = z.merge(ctx, on="date", how="left")
        for n in (1, 6, 12):
            z[f"relative_ret_{n}"] = z[f"ret_{n}"] - z[f"market_ret_{n}"]
        z["decision_timestamp"] = z["date"] + pd.Timedelta(hours=15, minutes=30)
        parts.append(z)
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "decision_timestamp"]).reset_index(drop=True)

def merge_optional(panel):
    out=panel.copy(); audit={}
    sources=[
      ("news",BASE/"processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet",["news_available"]),
      ("events",BASE/"processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet",["event_available","event_high_impact"]),
      ("fno",BASE/"processed/pit_replay/FNO_PIT_REPLAY_2026.parquet",["fno_pcr","fno_oi_change","fno_iv_change","fno_available","fno_fut_close","fno_fut_return","fno_fut_oi"]),
    ]
    for kind,path,cols in sources:
        layer=load_optional_layer(path)
        if layer.empty: audit[kind]={"exists":False,"rows":0}; continue
        out=merge_pit_layer(out,layer,cols,kind)
        audit[kind]={"exists":True,"rows":int(len(layer)),"symbols":int(layer["symbol"].nunique())}
    # Derived F&O basis is PIT-safe because fno_fut_close is already merged
    # with the same backward-as-of F&O timestamp contract.
    if "fno_fut_basis" not in out.columns:
        fut=pd.to_numeric(out.get("fno_fut_close"),errors="coerce")
        spot=pd.to_numeric(out.get("close"),errors="coerce")
        out["fno_fut_basis"]=(fut/spot.replace(0,np.nan))-1.0

    # ONLY the broad ALL-NSE PIT fundamentals layer is accepted here.
    # Never silently fall back to the legacy three-stock pilot.
    fp=BASE/"corporate/pit_fundamentals_all_nse_2025_2026.parquet"
    fp=fp if fp.exists() and fp.stat().st_size>0 else None
    if fp is not None:
        f=pd.read_parquet(fp)
        f["symbol"]=f["symbol"].map(normalize_symbol)
        f["fundamental_available_at"]=pd.to_datetime(
            f["fundamental_available_at"],errors="coerce",utc=True
        )
        f["period_end"]=pd.to_datetime(
            f.get("period_end"),errors="coerce",utc=True
        ).dt.normalize() if "period_end" in f.columns else pd.NaT
        f=f.dropna(subset=["symbol","fundamental_available_at"]).sort_values(
            ["symbol","fundamental_available_at"]
        )
        vals=[
            "fundamental_revenue","fundamental_pat","fundamental_eps",
            "fundamental_ebitda","fundamental_finance_cost",
            "fundamental_total_equity","fundamental_total_debt","fundamental_cash",
            "fund_revenue_growth","fund_pat_growth","fund_eps_growth",
            "fund_ebitda_growth","fund_pat_margin","fund_debt_to_equity",
            "fund_cash_to_debt",
        ]
        pieces=[]
        for sym,lp in out.groupby("symbol",sort=False):
            rp=f[f["symbol"].eq(sym)]
            if rp.empty:
                pieces.append(lp)
                continue
            cols=["fundamental_available_at"]+[x for x in vals if x in rp.columns]
            rp=rp[cols].drop_duplicates(["fundamental_available_at"],keep="last")
            m=pd.merge_asof(
                lp.sort_values("decision_timestamp"),
                rp.sort_values("fundamental_available_at"),
                left_on="decision_timestamp",
                right_on="fundamental_available_at",
                direction="backward",
                allow_exact_matches=True,
            )
            pieces.append(m)
        if pieces:
            out=pd.concat(pieces,ignore_index=True)
        # Fundamental readiness is based on the existence of an as-of filing,
        # not on every optional accounting field being present.
        fa=pd.to_datetime(out.get("fundamental_available_at"),errors="coerce",utc=True)
        valid=fa.notna() & fa.le(out["decision_timestamp"])
        out["fundamentals_available"]=valid.astype("Int8")
        out["fundamental_age_days"]=np.nan
        out.loc[valid,"fundamental_age_days"]=(
            out.loc[valid,"decision_timestamp"]-fa.loc[valid]
        ).dt.total_seconds()/86400.0
        out["fundamental_stale_90d"]=pd.Series(
            np.where(valid,(out["fundamental_age_days"]>90).astype(float),np.nan),
            index=out.index,dtype="float64"
        )
        out["fundamental_stale_180d"]=pd.Series(
            np.where(valid,(out["fundamental_age_days"]>180).astype(float),np.nan),
            index=out.index,dtype="float64"
        )
        audit["fundamentals"]={
            "exists":True,
            "path":str(fp),
            "rows":int(len(f)),
            "symbols":int(f["symbol"].nunique()),
        }
    else:
        audit["fundamentals"]={"exists":False,"rows":0}
        out["fundamentals_available"]=pd.Series(np.zeros(len(out),dtype="Int8"),index=out.index)
        out["fundamental_age_days"]=np.nan
        out["fundamental_stale_90d"]=np.nan
        out["fundamental_stale_180d"]=np.nan
    return attach_readiness_flags(out),audit

def add_targets(panel,steps,name):
    x=panel.sort_values(["symbol","decision_timestamp"]).copy()
    with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
        x[f"target_return_{name}"]=x.groupby("symbol")["close"].shift(-steps)/x["close"]-1.0
    x[f"target_return_{name}"]=x[f"target_return_{name}"].replace([np.inf,-np.inf],np.nan)
    x[f"target_up_{name}"]=(x[f"target_return_{name}"]>0).astype(int)
    return x.dropna(subset=[f"target_return_{name}"])

def fit_models(frame,horizon):
    cutoff=frame["decision_timestamp"].max()-pd.Timedelta(days=60); tr=frame[frame["decision_timestamp"]<cutoff]; te=frame[frame["decision_timestamp"]>=cutoff]
    if tr.empty or te.empty or tr[f"target_up_{horizon}"].nunique()<2: raise RuntimeError(f"{horizon}: insufficient chronological train/holdout data")
    Xtr=tr[FEATURES_59].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    Xte=te[FEATURES_59].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    yreg=tr[f"target_return_{horizon}"].replace([np.inf,-np.inf],np.nan)
    good=yreg.notna()
    Xtr=Xtr.loc[good]
    yreg=yreg.loc[good]
    yclf=tr.loc[good,f"target_up_{horizon}"].astype(int)
    if Xtr.empty or yclf.nunique()<2:
        raise RuntimeError(f"{horizon}: finite training set is insufficient after non-finite-value cleanup")
    clf=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("scale",StandardScaler(with_mean=False)),("model",LogisticRegression(C=.5,class_weight="balanced",solver="lbfgs",max_iter=2500,random_state=42))])
    reg=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),("model",HistGradientBoostingRegressor(max_iter=180,learning_rate=.04,max_leaf_nodes=19,l2_regularization=2.0,random_state=42))])
    clf.fit(Xtr,tr[f"target_up_{horizon}"].astype(int)); reg.fit(Xtr,tr[f"target_return_{horizon}"].astype(float))
    p=clf.predict_proba(Xte)[:,1]; pr=reg.predict(Xte); y=te[f"target_up_{horizon}"].to_numpy()
    return clf,reg,{"train_rows":int(len(tr)),"holdout_rows":int(len(te)),"holdout_accuracy":float(np.mean((p>=.5)==y)),"mean_abs_return_error":float(np.mean(np.abs(pr-te[f"target_return_{horizon}"].to_numpy()))),"holdout_mean_readiness_pct":float(te["data_readiness_score"].mean()*100)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bulk-root",default=DEFAULT_BULK); ap.add_argument("--out",default=DEFAULT_OUT); ap.add_argument("--min-history",type=int,default=60); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); bulk=load_bulk(Path(args.bulk_root))
    roster=training_roster(bulk,min_history=args.min_history)
    trainable=list(roster["trainable_symbols"])
    _ensure_broad_fundamentals(trainable)
    universe={"universe_source":roster.get("universe_source"),"universe_as_of":roster.get("universe_as_of"),"universe_symbol_count":int(roster["universe_symbol_count"]),"symbols_with_bulk_history":int(roster["symbols_with_bulk_history"]),"trainable_symbol_count":int(roster["trainable_symbol_count"]),"excluded_symbol_count":int(roster["excluded_symbol_count"]),"minimum_history_sessions":int(roster["minimum_history_sessions"])}
    panel,audit=merge_optional(make_panel(bulk,trainable))
    (out/"DATA_READINESS_REPORT_2026.json").write_text(json.dumps({"universe":universe,"readiness":readiness_report(panel),"optional_sources":audit},indent=2,default=str),encoding="utf-8")
    panel.to_parquet(out/"ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet",index=False)
    artifacts={}
    for horizon,steps in (("next_session",1),("3d",3),("5d",5),("7d",7)):
        try:
            d=add_targets(panel,steps,horizon); clf,reg,metrics=fit_models(d,horizon)
            cp=out/f"deployment_all_nse_{horizon}_multifactor59_classifier_2026.joblib"; rp=out/f"deployment_all_nse_{horizon}_multifactor59_regressor_2026.joblib"
            joblib.dump(clf,cp); joblib.dump(reg,rp)
            artifacts[horizon]={"status":"CANDIDATE_READY","classifier":str(cp),"regressor":str(rp),"classifier_sha256":sha256(cp),"regressor_sha256":sha256(rp),"metrics":metrics}
        except Exception as exc: artifacts[horizon]={"status":"BLOCKED","reason":f"{type(exc).__name__}: {exc}"}
    manifest={"engine_version":"2026-all-nse-delivery-multifactor-59-candidate","promotion_status":"CANDIDATE_ONLY","universe":universe,"feature_count":59,"features":FEATURES_59,"artifacts":artifacts,"data_readiness_path":str(out/"DATA_READINESS_REPORT_2026.json"),"pit_policy":"Only observations at or before decision_timestamp are eligible. Missing information is not fabricated.","no_auto_orders":True}
    (out/"ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")
    print(json.dumps(manifest,indent=2,default=str))

if __name__=="__main__": main()
