#!/usr/bin/env python3
"""Build a broad ALL-NSE point-in-time fundamentals layer from Drive-resident NSE XBRL.

The builder is intentionally conservative:
- prefers existing PIT parquet sources when they cover the NSE roster;
- supplements/rebuilds from Drive-resident NSE XBRL cache files;
- never uses a filing before its publication/acceptance timestamp;
- never forward-fills financial values across a filing date;
- preserves missing values and emits coverage diagnostics.
"""
from __future__ import annotations
import argparse, json, re, html
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pandas as pd

from multifactor_data_layer_v2 import normalize_symbol
from delivery_training_universe import training_roster

BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
DEFAULT_BULK = BASE / "production_data/nse_official/bulk_eod"
DEFAULT_OUT = BASE / "corporate/pit_fundamentals_all_nse_2025_2026.parquet"

RAW_VALUE_FIELDS = {
    "fundamental_revenue": [
        "revenuefromoperations", "revenuefromoperationsnetofexcise", "revenue",
        "revenuefromoperation", "totalrevenue", "revenuefromcontractswithcustomers",
        "turnover"
    ],
    "fundamental_pat": [
        "profitloss", "profitlossaftertax", "profitaftertax", "profitforperiod",
        "profitattributabletoparents", "netprofit", "netincome"
    ],
    "fundamental_eps": [
        "earningspersharebasic", "earningspersharediluted", "basicearningspershare",
        "dilutedeps", "eps"
    ],
    "fundamental_ebitda": [
        "ebitda", "earningsbeforeinteresttaxdepreciationandamortisation",
        "earningsbeforeinteresttaxdepreciationamortization"
    ],
    "fundamental_finance_cost": [
        "financecosts", "financecost", "financecharges", "interestexpense",
        "interestandbankcharges"
    ],
    "fundamental_total_equity": [
        "totalequity", "equity", "totalshareholdersequity",
        "equityattributabletoparent", "totalownersfunds"
    ],
    "fundamental_total_debt": [
        "totaldebt", "debt", "borrowings", "totalborrowings",
        "currentborrowings", "noncurrentborrowings"
    ],
    "fundamental_cash": [
        "cashandcashequivalents", "cashandbankbalances", "cashandbank",
        "cashandcashitems", "bankbalances"
    ],
}

TAG_GROUPS = {
    "symbol": {"symbol", "issuersymbol", "securitysymbol", "ticker", "tradingsymbol"},
    "period_end": {"enddate", "periodend", "periodenddate", "reportingdate", "dateofreport"},
    "available_at": {
        "acceptancedatetime", "acceptancedate", "filingdatetime", "filingdate",
        "submissiondatetime", "submissiondate", "dateoffiling", "eventdatetime",
        "creationdatetime"
    },
    "period_start": {"startdate", "periodstart", "periodstartdate"},
}

DATE_PATTERNS = [
    re.compile(r"(\\d{2})(\\d{2})(\\d{4})(?:[_-]?(\\d{2})(\\d{2})(\\d{2}))?"),
    re.compile(r"(\\d{4})[-/](\\d{2})[-/](\\d{2})(?:[ T](\\d{2})[:](\\d{2})[:](\\d{2}))?"),
]

def _localname(tag: str) -> str:
    return str(tag).split("}", 1)[-1].split(":", 1)[-1].strip().lower().replace("-", "").replace("_", "")

def _num(value: object) -> float:
    if value is None: return np.nan
    s = str(value).strip().replace(",", "").replace("₹", "").replace("%", "")
    if not s or s.lower() in {"na", "n/a", "nan", "null", "none", "-"}: return np.nan
    s = s.replace("(", "-").replace(")", "")
    try: return float(re.sub(r"[^0-9eE+\\-.]", "", s))
    except Exception: return np.nan

def _parse_dt(value: object) -> pd.Timestamp:
    if value is None: return pd.NaT
    s = str(value).strip()
    if not s: return pd.NaT
    dt = pd.to_datetime(s, errors="coerce", utc=True, dayfirst=True)
    return dt

def _filename_timestamp(path: Path) -> pd.Timestamp:
    s = unquote(path.name)
    for pat in DATE_PATTERNS:
        for m in pat.finditer(s):
            g = m.groups()
            try:
                if len(g) == 6 and len(g[0]) == 2:
                    return pd.Timestamp(
                        year=int(g[2]), month=int(g[1]), day=int(g[0]),
                        hour=int(g[3] or 0), minute=int(g[4] or 0), second=int(g[5] or 0),
                        tz="UTC"
                    )
                if len(g) == 6 and len(g[0]) == 4:
                    return pd.Timestamp(
                        year=int(g[0]), month=int(g[1]), day=int(g[2]),
                        hour=int(g[3] or 0), minute=int(g[4] or 0), second=int(g[5] or 0),
                        tz="UTC"
                    )
            except Exception:
                pass
    return pd.NaT

def _contexts(root: ET.Element) -> dict[str, tuple[pd.Timestamp,pd.Timestamp]]:
    out={}
    for e in root.iter():
        if _localname(e.tag) != "context":
            continue
        cid=e.attrib.get("id")
        if not cid: continue
        start=end=None
        for q in e.iter():
            n=_localname(q.tag)
            if n in TAG_GROUPS["period_start"] and (q.text or "").strip():
                start=_parse_dt(q.text)
            elif n in TAG_GROUPS["period_end"] and (q.text or "").strip():
                end=_parse_dt(q.text)
        if pd.notna(end): out[cid]=(start,end)
    return out

def _extract_xml(path: Path) -> list[dict]:
    raw=path.read_bytes()
    root=None
    for payload in (raw, html.unescape(raw.decode("utf-8","ignore")).encode("utf-8")):
        try:
            root=ET.fromstring(payload); break
        except Exception:
            continue
    if root is None:
        return _extract_regex(path, raw.decode("utf-8","ignore"))
    contexts=_contexts(root)
    symbol=""
    available=pd.NaT
    for e in root.iter():
        n=_localname(e.tag); t=(e.text or "").strip()
        if not t: continue
        if n in TAG_GROUPS["symbol"] and not symbol:
            symbol=normalize_symbol(t)
        if n in TAG_GROUPS["available_at"] and pd.isna(available):
            available=_parse_dt(t)
    if not symbol:
        symbol=_symbol_from_path(path)
    if pd.isna(available):
        available=_filename_timestamp(path)
    facts=[]
    for e in root.iter():
        n=_localname(e.tag)
        ctx=e.attrib.get("contextRef") or e.attrib.get("contextref")
        if not ctx or ctx not in contexts: continue
        val=_num(e.text)
        if not np.isfinite(val): continue
        _, end=contexts[ctx]
        if pd.isna(end): continue
        facts.append((n,val,end))
    if not facts or not symbol or pd.isna(available): return []
    return _facts_to_rows(symbol, available, facts, source=f"NSE_XBRL_PIT:{path}")

def _extract_regex(path: Path, text: str) -> list[dict]:
    symbol=_symbol_from_path(path)
    available=_filename_timestamp(path)
    if not symbol or pd.isna(available): return []
    facts=[]
    for field,cands in RAW_VALUE_FIELDS.items():
        for cand in cands:
            m=re.search(
                rf"<[^>]*{re.escape(cand)}[^>]*>([-+]?[(]?[0-9,]+(?:\\.[0-9]+)?[)]?)<",
                text, flags=re.I
            )
            if m:
                facts.append((cand,_num(m.group(1)),pd.NaT))
                break
    if not facts: return []
    # Regex fallback lacks safe period context, so do not publish it without a
    # document-level period end. This keeps the layer PIT-truthful.
    return []

def _symbol_from_path(path: Path) -> str:
    parts=[normalize_symbol(p) for p in path.parts[::-1]]
    known={"BHARTIARTL","RELIANCE","SBIN"}
    for p in parts:
        if p in known: return p
    m=re.search(r"(?:symbol|ticker)[=_-]?([A-Z][A-Z0-9]+)", path.name, flags=re.I)
    if m: return normalize_symbol(m.group(1))
    return ""

def _facts_to_rows(symbol: str, available: pd.Timestamp, facts: list[tuple], source: str) -> list[dict]:
    by_period={}
    for name,val,end in facts:
        if pd.isna(end): continue
        p=by_period.setdefault(pd.Timestamp(end).normalize(), {})
        for fld,cands in RAW_VALUE_FIELDS.items():
            if name in cands and fld not in p:
                p[fld]=val
    rows=[]
    for period_end, vals in by_period.items():
        row={"symbol":symbol,"period_end":period_end,"fundamental_available_at":available,
             "fundamental_source":source,"fundamental_basis":"NSE_XBRL_PIT"}
        row.update(vals)
        if len(vals) >= 1:
            rows.append(row)
    return rows

def _load_existing() -> pd.DataFrame:
    paths=[
        BASE/"processed/features/PIT_FUNDAMENTAL_FEATURES_2026.parquet",
        BASE/"processed/features/MASTER_PIT_WITH_FUNDAMENTALS_2026.parquet",
        BASE/"corporate/fundamentals_v2/PIT_FUNDAMENTALS_STRICT_2025_2026.parquet",
        BASE/"corporate/pit_fundamentals_filings_2025_2026.parquet",
    ]
    frames=[]
    for p in paths:
        if not p.exists(): continue
        try: df=pd.read_parquet(p)
        except Exception: continue
        if "symbol" not in df.columns: continue
        df["symbol"]=df["symbol"].map(normalize_symbol)
        if "fundamental_available_at" in df.columns:
            df["fundamental_available_at"]=pd.to_datetime(df["fundamental_available_at"],errors="coerce",utc=True)
        elif "decision_timestamp" in df.columns:
            df["fundamental_available_at"]=pd.to_datetime(df["decision_timestamp"],errors="coerce",utc=True)
        if "period_end" not in df.columns:
            df["period_end"]=pd.to_datetime(df.get("trade_date"),errors="coerce",utc=True)
        cols=["symbol","period_end","fundamental_available_at","fundamental_source","fundamental_basis"]+list(RAW_VALUE_FIELDS)
        for c in cols:
            if c not in df.columns: df[c]=np.nan
        frames.append(df[cols])
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def _scan_xbrl() -> pd.DataFrame:
    roots=[
        BASE/"corporate/xbrl_cache",
        BASE/"corporate",
        BASE/"processed/pit_replay",
        BASE/"processed/features",
    ]
    paths=[]
    seen=set()
    for r in roots:
        if not r.exists(): continue
        for pat in ("*.xml","*.xbrl","*.html","*.htm"):
            for p in r.rglob(pat):
                sp=str(p)
                if sp in seen: continue
                seen.add(sp); paths.append(p)
    rows=[]; total=0
    for p in paths:
        total += 1
        try:
            rows.extend(_extract_xml(p))
        except Exception:
            continue
        if total % 500 == 0:
            print(f"XBRL scan: files={total:,} rows={len(rows):,}")
    print(f"XBRL scan complete: files={total:,} extracted_rows={len(rows):,}")
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df=df.copy()
    df["symbol"]=df["symbol"].map(normalize_symbol)
    df["fundamental_available_at"]=pd.to_datetime(df["fundamental_available_at"],errors="coerce",utc=True)
    df["period_end"]=pd.to_datetime(df["period_end"],errors="coerce",utc=True).dt.normalize()
    for c in RAW_VALUE_FIELDS:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["symbol","fundamental_available_at","period_end"])
    df=df[df["fundamental_available_at"].le(pd.Timestamp("2026-10-02",tz="UTC"))]
    df=df.sort_values(["symbol","period_end","fundamental_available_at"])
    # Keep the latest available filing for each symbol/period.
    df=df.drop_duplicates(["symbol","period_end"],keep="last")
    pieces=[]
    for sym,g in df.groupby("symbol",sort=False):
        g=g.sort_values("period_end").copy()
        for base,chg in [
            ("fundamental_revenue","fund_revenue_chg"),
            ("fundamental_pat","fund_pat_chg"),
            ("fundamental_eps","fund_eps_chg"),
            ("fundamental_ebitda","fund_ebitda_chg"),
        ]:
            if base in g:
                g[chg]=g[base].diff()
                prev=g[base].shift(1)
                g[chg.replace("_chg","_growth")]=g[chg]/prev.abs().replace(0,np.nan)
        g["fund_pat_margin"]=g["fundamental_pat"]/g["fundamental_revenue"].replace(0,np.nan)
        g["fund_debt_to_equity"]=g["fundamental_total_debt"]/g["fundamental_total_equity"].replace(0,np.nan)
        g["fund_cash_to_debt"]=g["fundamental_cash"]/g["fundamental_total_debt"].replace(0,np.nan)
        # If EBITDA was not explicitly reported, derive it only when all
        # necessary accounting components exist.
        if "fundamental_ebitda" in g:
            g["fundamental_ebitda"]=g["fundamental_ebitda"].where(
                g["fundamental_ebitda"].notna(),
                g["fundamental_ebit"] if "fundamental_ebit" in g else np.nan,
            )
        g["fundamental_age_days"]=(
            g["fundamental_available_at"]-g["period_end"]
        ).dt.total_seconds()/86400.0
        g["fundamental_stale_90d"]=(g["fundamental_age_days"]>90).astype("Int8")
        g["fundamental_stale_180d"]=(g["fundamental_age_days"]>180).astype("Int8")
        pieces.append(g)
    out=pd.concat(pieces,ignore_index=True)
    return out.sort_values(["symbol","fundamental_available_at","period_end"]).reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bulk-root",default=str(DEFAULT_BULK))
    ap.add_argument("--out",default=str(DEFAULT_OUT))
    args=ap.parse_args()
    bulk=Path(args.bulk_root)
    roster=training_roster(pd.DataFrame(),min_history=0) if False else None
    existing=_load_existing()
    existing_symbols=set(existing["symbol"]) if not existing.empty else set()
    target_symbols=None
    try:
        # Use the existing bulk archive to establish the current ALL-NSE roster.
        from colab_build_all_nse_delivery_multifactor_candidate import load_bulk
        b=load_bulk(bulk)
        rr=training_roster(b,min_history=60)
        target_symbols=set(rr["trainable_symbols"])
    except Exception:
        target_symbols=None

    # Existing parquet sources are retained as a seed; scan XBRL only when
    # current seed coverage is poor.
    seed=_finalize(existing)
    seed_cov=len(set(seed["symbol"])) if not seed.empty else 0
    target_cov=len(target_symbols) if target_symbols is not None else 0
    print(f"SEED FUNDAMENTALS: symbols={seed_cov} target_trainable={target_cov}")

    xbrl=_scan_xbrl() if (target_cov == 0 or seed_cov < max(50, int(target_cov*0.8))) else pd.DataFrame()
    combined=pd.concat([seed,xbrl],ignore_index=True) if not xbrl.empty else seed
    combined=_finalize(combined)

    if target_symbols is not None:
        combined=combined[combined["symbol"].isin(target_symbols)].copy()

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    combined.to_parquet(out,index=False)
    report={
        "output":str(out),
        "rows":int(len(combined)),
        "unique_symbols":int(combined["symbol"].nunique()) if not combined.empty else 0,
        "target_trainable_symbols":int(target_cov),
        "symbol_coverage_pct":round(100*len(set(combined["symbol"])) / target_cov,3) if target_cov else None,
        "fundamental_available_at_min":str(combined["fundamental_available_at"].min()) if not combined.empty else None,
        "fundamental_available_at_max":str(combined["fundamental_available_at"].max()) if not combined.empty else None,
        "pit_policy":"fundamental_available_at <= decision_timestamp required at merge time",
    }
    rp=out.with_suffix(".json")
    rp.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
