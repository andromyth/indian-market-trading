#!/usr/bin/env python3
"""Authoritative read-only ALL-NSE 59-feature data-readiness audit.

This replaces the earlier checklist-style diagnostic. It validates the actual
repository contract, existing replacement artifacts, PIT replay schemas,
derived F&O timing semantics, fundamentals coverage, and row/feature/group
coverage without retraining or mutating models/data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REPO = Path("/content/indian-market-trading")
DRIVE = Path("/content/drive/MyDrive/Indian_Market_Research")
OUT = DRIVE / "diagnostics" / "final_readiness_v2"
MODEL_DIR = DRIVE / "models" / "all_nse_delivery_multifactor_replacement_2026"
PANEL = MODEL_DIR / "ALL_NSE_MULTIFACTOR_59_FEATURE_PIT_PANEL_2026.parquet"
MANIFEST = MODEL_DIR / "ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json"
NEWS = DRIVE / "processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet"
EVENTS = DRIVE / "processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet"
FNO = DRIVE / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
FNO_RAW = DRIVE / "production_data/nse_official/fno/normalized_daily/2026"
FUND_BROAD = DRIVE / "corporate/pit_fundamentals_all_nse_2025_2026.parquet"
FUND_CACHE = DRIVE / "corporate/pit_fundamentals_filing_cache_2025_2026_v3.parquet"

PIT_ALIASES = [
    "decision_timestamp", "fundamental_available_at", "fno_observed_at",
    "available_at", "publication_timestamp", "published_at",
    "announcement_timestamp", "observed_at", "timestamp", "datetime",
    "date_time", "trade_timestamp"
]
SYMBOL_ALIASES = ["symbol", "ticker", "tckrsymb", "underlying", "undrlyg"]
DATE_ALIASES = ["trade_date", "date", "tradedate", "trading_date"]

FALLBACK_FEATURES_59 = [
    "ret_1","ret_3","ret_6","ret_12","ret_24","ret_48","rsi14","atr14",
    "ma20_ratio","ma50_ratio","volume_ratio","volume_z20","delivery_pct",
    "delivery_pct_change_1","delivery_pct_ratio20","delivery_pct_z20",
    "delivery_qty_ratio20","delivery_qty_z20","range_pct","body_pct","up_bar",
    "market_ret_1","market_ret_3","market_ret_6","market_rsi14",
    "relative_ret_1","relative_ret_6","relative_ret_12","market_mood_proxy",
    "market_stress_proxy","news_available","event_available","event_high_impact",
    "fno_pcr","fno_oi_change","fno_iv_change","fno_available","fno_fut_close",
    "fno_fut_basis","fno_fut_return","fno_fut_oi","fundamental_revenue",
    "fundamental_pat","fundamental_eps","fundamental_ebitda",
    "fundamental_finance_cost","fundamental_total_equity",
    "fundamental_total_debt","fundamental_cash","fund_revenue_growth",
    "fund_pat_growth","fund_eps_growth","fund_ebitda_growth","fund_pat_margin",
    "fund_debt_to_equity","fund_cash_to_debt","fundamental_age_days",
    "fundamental_stale_90d","fundamental_stale_180d"
]
FALLBACK_GROUPS = {
    "market_price": FALLBACK_FEATURES_59[:21],
    "market_context": FALLBACK_FEATURES_59[21:30],
    "news": ["news_available"],
    "events": ["event_available","event_high_impact"],
    "fno": FALLBACK_FEATURES_59[33:41],
    "fundamentals": FALLBACK_FEATURES_59[41:],
}
FLAG = {
    "market_price":"ready_market_price",
    "market_context":"ready_market_context",
    "news":"ready_news",
    "events":"ready_events",
    "fno":"ready_fno",
    "fundamentals":"ready_fundamentals",
}

def norm_symbol(x: Any) -> str:
    s = str(x or "").strip().upper()
    for suffix in (".NS",".BO","-EQ",".EQ"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    return s

def hash_features(features: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(features, separators=(",",":"), ensure_ascii=False).encode()
    ).hexdigest()

def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        x = json.loads(path.read_text(encoding="utf-8"))
        return x if isinstance(x, dict) else {}
    except Exception:
        return {}

def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return pd.read_parquet(path)

def col_match(columns, aliases) -> str | None:
    exact = {str(c).strip().lower(): c for c in columns}
    compact = {re.sub(r"[^a-z0-9]","",str(c).lower()): c for c in columns}
    for a in aliases:
        if a.lower() in exact:
            return exact[a.lower()]
        k = re.sub(r"[^a-z0-9]","",a.lower())
        if k in compact:
            return compact[k]
    return None

def pit_column(columns) -> str | None:
    # decision_timestamp is intentionally preferred over legacy metadata fields.
    return col_match(columns, PIT_ALIASES)

def git_info() -> dict[str, Any]:
    out = {"repo_exists": REPO.exists()}
    if not REPO.exists():
        return out
    for key, cmd in {
        "branch":["git","-C",str(REPO),"branch","--show-current"],
        "head":["git","-C",str(REPO),"rev-parse","HEAD"],
        "last_commit":["git","-C",str(REPO),"log","-1","--pretty=%s"],
    }.items():
        try:
            out[key] = subprocess.check_output(cmd, text=True).strip()
        except Exception as e:
            out[key] = f"{type(e).__name__}: {e}"
    return out

def feature_contract() -> tuple[list[str],dict[str,list[str]],str]:
    if REPO.exists():
        sys.path.insert(0,str(REPO))
        try:
            from multifactor_data_layer_v2 import FEATURES_59, GROUPS
            return list(FEATURES_59), {k:list(v) for k,v in GROUPS.items()}, str(REPO / "multifactor_data_layer_v2.py")
        except Exception:
            pass
    return list(FALLBACK_FEATURES_59), {k:list(v) for k,v in FALLBACK_GROUPS.items()}, "defensive-fallback"

def source_audit(path: Path, name: str, expected_symbols: set[str], require_pit: bool) -> tuple[dict[str,Any],pd.DataFrame]:
    r = {
        "name":name,"path":str(path),"exists":path.exists() and path.stat().st_size>0,
        "rows":0,"symbols":0,"pit_column":None,"pit_valid_pct":None,
        "expected_symbol_coverage_pct":None,"duplicate_symbol_pit_rows":None,
        "status":"MISSING","hard_failures":[]
    }
    if not r["exists"]:
        r["hard_failures"]=["SOURCE_MISSING"]
        return r,pd.DataFrame()
    df = read_parquet(path)
    r["rows"] = int(len(df))
    sc = col_match(df.columns,SYMBOL_ALIASES)
    dc = col_match(df.columns,DATE_ALIASES)
    pc = pit_column(df.columns)
    r["symbol_column"],r["trade_date_column"],r["pit_column"] = sc,dc,pc
    if sc:
        df["__symbol"] = df[sc].map(norm_symbol)
        r["symbols"] = int(df["__symbol"].nunique())
        if expected_symbols:
            r["expected_symbol_coverage_pct"] = round(100*len(set(df["__symbol"]) & expected_symbols)/len(expected_symbols),3)
    else:
        r["hard_failures"].append("NO_SYMBOL_COLUMN")
    if pc:
        df["__pit"] = pd.to_datetime(df[pc],errors="coerce",utc=True)
        r["pit_valid_pct"] = round(100*float(df["__pit"].notna().mean()),3) if len(df) else 0.0
        if sc:
            r["duplicate_symbol_pit_rows"] = int(df.duplicated(["__symbol","__pit"],keep=False).sum())
        if r["pit_valid_pct"] < 100:
            r["hard_failures"].append("INVALID_PIT_TIMESTAMPS")
    elif require_pit:
        r["hard_failures"].append("PIT_TIMESTAMP_COLUMN_UNRESOLVED")
    # latest_event_available_at is metadata to inspect, not a replacement for decision_timestamp.
    if pc:
        internal = [c for c in df.columns if str(c) != pc and ("available_at" in str(c).lower() or "published_at" in str(c).lower() or "announcement_timestamp" in str(c).lower())]
        checks = {}
        for c in internal:
            ts = pd.to_datetime(df[c],errors="coerce",utc=True)
            mask = ts.notna() & df["__pit"].notna()
            if mask.any():
                future = ts[mask] > df.loc[mask,"__pit"]
                checks[str(c)] = {"checked_rows":int(mask.sum()),"future_rows":int(future.sum()),"future_pct":round(100*float(future.mean()),3)}
        r["internal_timestamp_checks"] = checks
        for c,v in checks.items():
            if v["future_rows"]:
                r["hard_failures"].append(f"FUTURE_{c}")
    # Duplicate source rows can be valid for news/events; report only.
    r["status"] = "FAIL" if r["hard_failures"] else "PASS"
    return r,df

def model_contract(manifest_path: Path, features: list[str]) -> tuple[dict[str,Any],pd.DataFrame]:
    r = {
        "manifest_exists":manifest_path.exists(),"feature_count_expected":len(features),
        "manifest_feature_count":None,"manifest_features_match":False,
        "manifest_feature_contract_hash":None,"expected_feature_contract_hash":hash_features(features),
        "feature_contract_hash_match":False,"promotion_status":None,"engine_version":None,
        "artifacts":{},"status":"MISSING","hard_failures":[]
    }
    m = read_json(manifest_path)
    if not m:
        r["hard_failures"]=["MANIFEST_MISSING_OR_INVALID"]
        return r,pd.DataFrame()
    r["manifest_feature_count"] = m.get("feature_count")
    r["manifest_features_match"] = m.get("features") == features
    r["manifest_feature_contract_hash"] = m.get("feature_contract_hash")
    r["feature_contract_hash_match"] = m.get("feature_contract_hash") == r["expected_feature_contract_hash"]
    r["promotion_status"],r["engine_version"] = m.get("promotion_status"),m.get("engine_version")
    if r["promotion_status"] != "BRANCH_REPLACEMENT_ONLY":
        r["hard_failures"].append("UNEXPECTED_PROMOTION_STATUS")
    if r["manifest_feature_count"] != len(features):
        r["hard_failures"].append("MANIFEST_FEATURE_COUNT_MISMATCH")
    if not r["manifest_features_match"]:
        r["hard_failures"].append("MANIFEST_FEATURE_LIST_MISMATCH")
    if not r["feature_contract_hash_match"]:
        r["hard_failures"].append("MANIFEST_FEATURE_CONTRACT_HASH_MISMATCH")
    rows=[]
    artifacts=m.get("artifacts",{})
    if not isinstance(artifacts,dict) or not artifacts:
        r["hard_failures"].append("NO_HORIZON_ARTIFACTS")
    for horizon,item in artifacts.items():
        item=item if isinstance(item,dict) else {}
        cp = Path(item.get("classifier","")) if item.get("classifier") else MODEL_DIR/f"deployment_all_nse_{horizon}_multifactor59_classifier_2026.joblib"
        rp = Path(item.get("regressor","")) if item.get("regressor") else MODEL_DIR/f"deployment_all_nse_{horizon}_multifactor59_regressor_2026.joblib"
        if not cp.exists(): cp=MODEL_DIR/f"deployment_all_nse_{horizon}_multifactor59_classifier_2026.joblib"
        if not rp.exists(): rp=MODEL_DIR/f"deployment_all_nse_{horizon}_multifactor59_regressor_2026.joblib"
        row={"horizon":horizon,"manifest_status":item.get("status"),"classifier":str(cp),"regressor":str(rp),"classifier_loaded":False,"regressor_loaded":False,"classifier_original_features":None,"regressor_original_features":None,"status":"FAIL"}
        for label,path in (("classifier",cp),("regressor",rp)):
            if not path.exists() or path.stat().st_size==0:
                r["hard_failures"].append(f"{horizon.upper()}_{label.upper()}_MISSING"); continue
            try:
                obj=joblib.load(path)
                row[f"{label}_loaded"]=True
                imp=obj.named_steps.get("imputer") if hasattr(obj,"named_steps") else None
                n=getattr(imp,"n_features_in_",None) if imp is not None else getattr(obj,"n_features_in_",None)
                if n is None and imp is not None and getattr(imp,"statistics_",None) is not None:
                    n=len(imp.statistics_)
                row[f"{label}_original_features"]=int(n) if n is not None else None
                if n != len(features):
                    r["hard_failures"].append(f"{horizon.upper()}_{label.upper()}_FEATURE_COUNT_MISMATCH")
            except Exception as e:
                r["hard_failures"].append(f"{horizon.upper()}_{label.upper()}_LOAD_ERROR:{type(e).__name__}")
        row["status"]="PASS" if row["manifest_status"]=="CANDIDATE_READY" and row["classifier_loaded"] and row["regressor_loaded"] and row["classifier_original_features"]==len(features) and row["regressor_original_features"]==len(features) else "FAIL"
        rows.append(row); r["artifacts"][horizon]=row
    r["status"]="PASS" if not r["hard_failures"] else "FAIL"
    return r,pd.DataFrame(rows)

def fno_derived_audit(path: Path, panel_dates: set[pd.Timestamp], trainable: set[str]) -> tuple[dict[str,Any],pd.DataFrame]:
    r={"path":str(path),"exists":path.exists() and path.stat().st_size>0,"rows":0,"symbols":0,"observed_trainable_symbols":0,"unobserved_trainable_symbols":0,"duplicate_symbol_date_rows":0,"pit_rule":"trade_date -> next panel trading session 09:15 IST","status":"MISSING","hard_failures":[]}
    if not r["exists"]:
        r["hard_failures"]=["FNO_DERIVED_MISSING"]; return r,pd.DataFrame()
    f=read_parquet(path)
    r["rows"]=int(len(f))
    sc=col_match(f.columns,SYMBOL_ALIASES); dc=col_match(f.columns,DATE_ALIASES)
    if not sc or not dc:
        r["hard_failures"]=["FNO_DERIVED_MISSING_SYMBOL_OR_DATE"]; r["status"]="FAIL"; return r,f
    f=f.rename(columns={sc:"symbol",dc:"date"}).copy()
    f["symbol"]=f["symbol"].map(norm_symbol); f["date"]=pd.to_datetime(f["date"],errors="coerce").dt.normalize()
    f=f.dropna(subset=["symbol","date"])
    r["symbols"]=int(f["symbol"].nunique())
    r["date_min"]=f["date"].min().date().isoformat() if len(f) else None
    r["date_max"]=f["date"].max().date().isoformat() if len(f) else None
    r["duplicate_symbol_date_rows"]=int(f.duplicated(["symbol","date"],keep=False).sum())
    av=pd.to_numeric(f["fno_available"],errors="coerce") if "fno_available" in f.columns else pd.Series(1,index=f.index,dtype=float)
    if "fno_available" in f.columns and int((~(av.isin([0,1])|av.isna())).sum()):
        r["hard_failures"].append("INVALID_FNO_AVAILABLE_DOMAIN")
    obs=set(f.loc[av.eq(1),"symbol"])
    r["observed_trainable_symbols"]=len(obs & trainable)
    r["unobserved_trainable_symbols"]=len(trainable-obs)
    r["observed_symbol_coverage_pct"]=round(100*len(obs&trainable)/len(trainable),3) if trainable else 0.0
    ordered=sorted(panel_dates); next_map={ordered[i]:ordered[i+1] for i in range(len(ordered)-1)}
    f["expected_next_session"]=f["date"].map(next_map)
    terminal=max(panel_dates) if panel_dates else None
    nonterminal=f[f["date"]<terminal] if terminal is not None else f.iloc[0:0]
    bad=int(nonterminal["expected_next_session"].isna().sum())
    r["nonterminal_rows_without_next_session"]=bad
    if bad: r["hard_failures"].append("FNO_TRADE_DATE_WITHOUT_NEXT_PANEL_SESSION")
    r["status"]="FAIL" if r["hard_failures"] else "PASS"
    return r,f

def fno_raw_audit(root: Path) -> dict[str,Any]:
    r={"root":str(root),"exists":root.exists(),"files":0,"rows":0,"symbols":0,"dates":0,"duplicate_symbol_date_rows":0,"read_errors":[],"status":"MISSING","duplicate_semantics":"EXPECTED_CONTRACT_LEVEL_DUPLICATES"}
    if not root.exists(): return r
    symbols=set(); dates=set(); dup=0
    files=sorted(root.glob("*.parquet")); r["files"]=len(files)
    for path in files:
        try:
            x=pd.read_parquet(path); r["rows"]+=len(x)
            sc=col_match(x.columns,SYMBOL_ALIASES); dc=col_match(x.columns,DATE_ALIASES)
            if sc:
                sv=x[sc].map(norm_symbol); symbols.update(sv.dropna())
            if dc:
                dv=pd.to_datetime(x[dc],errors="coerce").dt.normalize(); dates.update(pd.Timestamp(v) for v in dv.dropna().unique())
                if sc: dup += int(pd.DataFrame({"symbol":sv,"date":dv}).duplicated(["symbol","date"],keep=False).sum())
        except Exception as e:
            r["read_errors"].append({"file":str(path),"error":f"{type(e).__name__}: {e}"})
    r["symbols"],r["dates"],r["duplicate_symbol_date_rows"]=len(symbols),len(dates),dup
    if dates: r["min_date"],r["max_date"]=min(dates).date().isoformat(),max(dates).date().isoformat()
    # Duplicate symbol/date rows are normal because this archive is contract-level.
    r["status"]="FAIL" if r["read_errors"] else "PASS"
    return r

def fundamentals_audit(broad_path: Path, cache_path: Path, replay_symbols: set[str], trainable: set[str]) -> tuple[dict[str,Any],pd.DataFrame]:
    r={"broad":{},"cache":{},"comparison":{},"status":"MISSING","hard_failures":[]}
    frames={}
    for name,path in (("broad",broad_path),("cache",cache_path)):
        df=read_parquet(path); frames[name]=df
        sc=col_match(df.columns,SYMBOL_ALIASES) if not df.empty else None
        pc=col_match(df.columns,["fundamental_available_at","decision_timestamp","available_at"]) if not df.empty else None
        sym=set(df[sc].map(norm_symbol)) if sc else set()
        r[name]={"path":str(path),"exists":not df.empty,"rows":int(len(df)),"symbols":len(sym),"pit_column":pc,"pit_valid_pct":None}
        if pc:
            ts=pd.to_datetime(df[pc],errors="coerce",utc=True); r[name]["pit_valid_pct"]=round(100*float(ts.notna().mean()),3)
        frames[name+"__symbols"]=sym
    broad,cache=frames["broad__symbols"],frames["cache__symbols"]
    cache_only=(cache-broad)&trainable
    replay_only=(replay_symbols-broad)&trainable
    no_source=trainable-broad-cache-replay_symbols
    r["comparison"]={
        "trainable_symbols":len(trainable),"broad_symbols":len(broad),"cache_symbols":len(cache),
        "replay_symbols":len(replay_symbols),"cache_only_trainable":len(cache_only),
        "replay_only_trainable":len(replay_only),"no_fundamental_source_observed":len(no_source),
        "cache_only_upper_bound_pct":round(100*len(cache_only)/len(trainable),3) if trainable else 0.0
    }
    rows=[]
    for s in sorted(trainable):
        rows.append({"symbol":s,"in_broad":s in broad,"in_filing_cache":s in cache,"in_events_replay":s in replay_symbols,
                     "classification":"BROAD_READY" if s in broad else "CACHE_ONLY" if s in cache else "REPLAY_ONLY" if s in replay_symbols else "NO_FUND_SOURCE_OBSERVED"})
    gap=pd.DataFrame(rows)
    if not broad_path.exists() or broad_path.stat().st_size==0: r["hard_failures"].append("BROAD_FUNDAMENTALS_MISSING")
    elif not r["broad"]["pit_column"]: r["hard_failures"].append("BROAD_FUNDAMENTAL_PIT_COLUMN_UNRESOLVED")
    elif r["broad"]["pit_valid_pct"]<100: r["hard_failures"].append("BROAD_FUNDAMENTAL_PIT_TIMESTAMPS_INCOMPLETE")
    r["status"]="FAIL" if r["hard_failures"] else "PASS"
    return r,gap

def panel_audit(path: Path, features: list[str], groups: dict[str,list[str]], trainable: set[str], fno_symbols: set[str]) -> tuple[dict[str,Any],pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    r={"path":str(path),"exists":path.exists() and path.stat().st_size>0,"rows":0,"symbols":0,"feature_count_present":0,"missing_features":[],"duplicate_symbol_timestamp_rows":0,"group_readiness":{},"strict_full_ready_pct":None,"state_known_metrics":{},"status":"MISSING","hard_failures":[]}
    empty=pd.DataFrame()
    if not r["exists"]:
        r["hard_failures"]=["PANEL_MISSING"]; return r,empty,empty,empty,empty,empty
    x=pd.read_parquet(path); r["rows"]=len(x); r["symbols"]=int(x["symbol"].map(norm_symbol).nunique()) if "symbol" in x.columns else 0
    if "symbol" not in x.columns: r["hard_failures"]=["PANEL_MISSING_SYMBOL"]; r["status"]="FAIL"; return r,x,empty,empty,empty,empty
    x["symbol"]=x["symbol"].map(norm_symbol)
    if "decision_timestamp" in x.columns:
        x["decision_timestamp"]=pd.to_datetime(x["decision_timestamp"],errors="coerce",utc=True)
        r["duplicate_symbol_timestamp_rows"]=int(x.duplicated(["symbol","decision_timestamp"],keep=False).sum())
    if "date" in x.columns: x["date"]=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
    else: x["date"]=x["decision_timestamp"].dt.tz_convert(None).dt.normalize()
    missing=[f for f in features if f not in x.columns]; r["missing_features"]=missing; r["feature_count_present"]=len(features)-len(missing)
    if missing: r["hard_failures"].append("PANEL_MISSING_FEATURE_COLUMNS")
    if r["duplicate_symbol_timestamp_rows"]: r["hard_failures"].append("PANEL_DUPLICATE_SYMBOL_TIMESTAMP")
    # Feature table
    fr=[]
    for f in features:
        s=x[f] if f in x.columns else pd.Series(index=x.index,dtype=float); n=s.notna(); num=pd.to_numeric(s,errors="coerce"); finite=np.isfinite(num.to_numpy(dtype=float))
        fr.append({"feature":f,"group":next((g for g,c in groups.items() if f in c),"unknown"),"non_null_rows":int(n.sum()),"coverage_pct":round(100*float(n.mean()),3) if len(x) else 0.0,
                   "finite_rows":int(finite.sum()),"finite_pct":round(100*float(finite.mean()),3) if len(x) else 0.0,
                   "unique_non_null":int(s.dropna().nunique()),"zero_rows":int(num.eq(0).sum()),"constant_non_null":bool(s.dropna().nunique()<=1)})
    feature_df=pd.DataFrame(fr).sort_values(["coverage_pct","group","feature"],ignore_index=True)
    # Mirror production readiness semantics for comparison.
    flags={}
    def any_present(cols):
        p=[c for c in cols if c in x.columns]
        if not p: return pd.Series(np.nan,index=x.index)
        return x[p].apply(pd.to_numeric,errors="coerce").notna().any(axis=1).astype("Int8")
    flags["market_price"]=any_present(groups.get("market_price",[]))
    flags["market_context"]=any_present(groups.get("market_context",[]))
    for g,src in (("news","news_available"),("events","event_available"),("fno","fno_available")):
        flags[g]=pd.to_numeric(x[src],errors="coerce").astype("Int8") if src in x.columns else any_present(groups.get(g,[]))
    flags["fundamentals"]=pd.to_numeric(x["fundamentals_available"],errors="coerce").astype("Int8") if "fundamentals_available" in x.columns else any_present(groups.get("fundamentals",[]))
    for g,s in flags.items(): x[FLAG[g]]=s
    flag_cols=list(FLAG.values()); x["strict_full_ready"]=x[flag_cols].eq(1).all(axis=1).astype("Int8")
    for g,s in flags.items(): r["group_readiness"][g]={"known_rows":int(s.notna().sum()),"ready_rows":int(s.eq(1).sum()),"known_pct":round(100*float(s.notna().mean()),3),"ready_pct":round(100*float(s.eq(1).mean()),3),"ready_symbols":int(x.loc[s.eq(1),"symbol"].nunique())}
    r["strict_full_ready_pct"]=round(100*float(x["strict_full_ready"].mean()),3)
    # State-known semantics: 0 is a known state. For F&O, absence is NOT declared ineligible.
    state=pd.DataFrame(index=x.index)
    state["news"]=pd.to_numeric(x["news_available"],errors="coerce").isin([0,1]) if "news_available" in x.columns else flags["news"].notna()
    state["events"]=pd.to_numeric(x["event_available"],errors="coerce").isin([0,1]) if "event_available" in x.columns else flags["events"].notna()
    state["fno"]=pd.to_numeric(x["fno_available"],errors="coerce").isin([0,1]) if "fno_available" in x.columns else flags["fno"].notna()
    state["fundamentals"]=pd.to_numeric(x["fundamentals_available"],errors="coerce").isin([0,1]) if "fundamentals_available" in x.columns else flags["fundamentals"].notna()
    state["market_price"]=flags["market_price"].eq(1); state["market_context"]=flags["market_context"].eq(1)
    x["core_market_ready"]=state["market_price"]&state["market_context"]
    x["non_fno_modalities_state_known"]=x["core_market_ready"]&state["news"]&state["events"]&state["fundamentals"]
    x["fno_observed_symbol"]=x["symbol"].isin(fno_symbols)
    x["fno_relaxed_upper_bound"]=x["non_fno_modalities_state_known"]&(state["fno"]|~x["fno_observed_symbol"])
    r["state_known_metrics"]={
        "core_market_ready_pct":round(100*float(x["core_market_ready"].mean()),3),
        "news_state_known_pct":round(100*float(state["news"].mean()),3),
        "events_state_known_pct":round(100*float(state["events"].mean()),3),
        "fundamentals_state_known_pct":round(100*float(state["fundamentals"].mean()),3),
        "fno_state_known_pct":round(100*float(state["fno"].mean()),3),
        "non_fno_modalities_state_known_pct":round(100*float(x["non_fno_modalities_state_known"].mean()),3),
        "fno_relaxed_upper_bound_pct":round(100*float(x["fno_relaxed_upper_bound"].mean()),3),
        "fno_observed_symbols":int(x.loc[x["fno_observed_symbol"],"symbol"].nunique()),
        "fno_unobserved_symbols":int(x.loc[~x["fno_observed_symbol"],"symbol"].nunique())
    }
    # Bottleneck combinations
    counts=Counter()
    bottleneck_cols=[(g, col) for g, col in FLAG.items()]
    for row in x[flag_cols].itertuples(index=False,name=None):
        bad=[]
        for (g, col), v in zip(bottleneck_cols, row):
            if pd.isna(v) or v != 1:
                bad.append(str(g))
        counts["+".join(bad) if bad else "ALL_READY"]+=1
    bdf=pd.DataFrame([{"missing_groups_combination":k,"rows":v,"pct_rows":round(100*v/len(x),3)} for k,v in counts.most_common()])
    sdf=x.groupby("symbol",as_index=False).agg(rows=("symbol","size"),first_date=("date","min"),last_date=("date","max"),strict_full_ready_pct=("strict_full_ready","mean"),feature_non_null_mean=(features[0],"count"))
    if not x.empty:
        sdf["strict_full_ready_pct"]*=100
    fdf=pd.DataFrame({"symbol":sorted(trainable),"observed_in_derived_fno":[s in fno_symbols for s in sorted(trainable)],"classification":["FNO_OBSERVED" if s in fno_symbols else "FNO_NOT_OBSERVED_APPLICABILITY_UNKNOWN" for s in sorted(trainable)]})
    r["status"]="FAIL" if r["hard_failures"] else "PASS"
    return r,x,feature_df,bdf,sdf,fdf

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",default=str(OUT))
    ap.add_argument("--skip-fno-raw",action="store_true")
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    git=git_info(); features,groups,feature_source=feature_contract()
    if len(features)!=59 or len(set(features))!=59: raise SystemExit(f"Invalid 59-feature contract: {len(features)}")
    manifest=read_json(MANIFEST); trainable={norm_symbol(s) for s in manifest.get("trainable_symbols",[]) if str(s).strip()}
    model_r,model_df=model_contract(MANIFEST,features)
    if not PANEL.exists(): raise SystemExit(f"Replacement panel not found: {PANEL}")
    dates_df=read_parquet(PANEL,["date"]) if "date" in (read_parquet(PANEL,["date"]).columns if PANEL.exists() else []) else read_parquet(PANEL,["decision_timestamp"])
    if "date" in dates_df.columns:
        panel_dates=set(pd.to_datetime(dates_df["date"],errors="coerce").dt.normalize().dropna())
    else:
        panel_dates=set(pd.to_datetime(dates_df["decision_timestamp"],errors="coerce",utc=True).dt.tz_convert(None).dt.normalize().dropna())
    fno_r,fno_df=fno_derived_audit(FNO,panel_dates,trainable)
    fno_syms=set(fno_df.loc[pd.to_numeric(fno_df["fno_available"],errors="coerce").eq(1),"symbol"]) if not fno_df.empty and "fno_available" in fno_df.columns else set(fno_df["symbol"]) if not fno_df.empty else set()
    fno_raw={} if args.skip_fno_raw else fno_raw_audit(FNO_RAW)
    news_r,news_df=source_audit(NEWS,"NEWS_ENRICHED_PIT_REPLAY_2026",trainable,True)
    events_r,events_df=source_audit(EVENTS,"EVENTS_FUNDAMENTALS_PIT_REPLAY_2026",trainable,True)
    replay_syms=set(events_df["__symbol"]) if "__symbol" in events_df.columns else set()
    fund_r,fund_gap=fundamentals_audit(FUND_BROAD,FUND_CACHE,replay_syms,trainable)
    panel_r,panel_df,feature_df,bottle_df,symbol_df,fno_symbol_df=panel_audit(PANEL,features,groups,trainable,fno_syms)
    panel_syms=set(panel_df["symbol"]) if not panel_df.empty else set()
    universe_cross={"manifest_trainable":len(trainable),"panel_symbols":len(panel_syms),"manifest_trainable_observed_in_panel":len(trainable&panel_syms),"trainable_missing_from_panel":len(trainable-panel_syms),"panel_not_in_trainable":len(panel_syms-trainable),"exact_symbol_match":trainable==panel_syms}
    structural=[]
    if fno_r.get("unobserved_trainable_symbols"): structural.append("FNO_ABSENCE_IS_NOT_AUTOMATIC_INELIGIBILITY")
    if fund_r.get("comparison",{}).get("cache_only_trainable"): structural.append("BROAD_FUNDAMENTALS_NARROWER_THAN_FILING_CACHE")
    blocking=[]
    for rr in (model_r,panel_r,news_r,events_r,fno_r,fund_r):
        blocking.extend(rr.get("hard_failures",[]))
    report={
        "report_version":"final-algorithm-data-readiness-v2",
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "read_only":True,
        "git":git,
        "feature_contract":{"count":len(features),"hash":hash_features(features),"source":feature_source,"groups":groups},
        "replacement_model_contract":model_r,
        "universe_crosscheck":universe_cross,
        "panel":panel_r,
        "fno_derived":fno_r,
        "fno_raw_normalized_archive":fno_raw,
        "news_pit_replay":news_r,
        "events_fundamentals_pit_replay":events_r,
        "fundamentals":fund_r,
        "overall":{
            "status":"BLOCKED" if blocking else "PASS_WITH_STRUCTURAL_FINDINGS",
            "blocking_findings":sorted(set(blocking)),
            "structural_findings":sorted(set(structural)),
            "no_model_training":True,
            "no_model_promotion":True,
            "no_dataset_mutation":True,
        },
        "interpretation":{
            "strict_full_ready":"Literal simultaneous readiness of all six production groups.",
            "state_known":"Explicit 0/1 availability values are known states, not missing values.",
            "fno_applicability":"Symbols absent from derived F&O are classified as applicability unknown, not ineligible.",
            "fno_relaxed_upper_bound":"Diagnostic upper bound only; not a production readiness score.",
            "pit":"decision_timestamp is the canonical replay PIT field; legacy latest_event_available_at is not substituted for it.",
            "fno_raw_duplicates":"Expected because normalized F&O archive is contract-level; symbol/date uniqueness is enforced in the derived feature layer."
        }
    }
    write=lambda df,name: df.to_csv(out/name,index=False)
    write(feature_df,"FEATURE_COVERAGE_59_V2_20261003.csv")
    write(pd.DataFrame([{"group":g,**v} for g,v in panel_r["group_readiness"].items()]),"GROUP_READINESS_V2_20261003.csv")
    write(bottle_df,"DECISION_BOTTLENECKS_V2_20261003.csv")
    write(symbol_df,"SYMBOL_COVERAGE_V2_20261003.csv")
    write(fno_symbol_df,"FNO_SYMBOL_COVERAGE_V2_20261003.csv")
    write(fund_gap,"FUNDAMENTAL_SYMBOL_GAPS_V2_20261003.csv")
    write(model_df,"MODEL_CONTRACT_AUDIT_V2_20261003.csv")
    json_path=out/"FINAL_ALGORITHM_DATA_READINESS_V2_20261003.json"
    json_path.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    print("="*100)
    print("FINAL ALGORITHM DATA READINESS V2")
    print("="*100)
    print(f"BRANCH: {git.get('branch')}")
    print(f"HEAD: {git.get('head','')[:12]}")
    print(f"59-FEATURE CONTRACT: {model_r.get('status')}")
    print(f"PANEL: {panel_r.get('rows',0):,} rows / {panel_r.get('symbols',0):,} symbols")
    print(f"STRICT FULL READINESS: {panel_r.get('strict_full_ready_pct')}")
    print(f"NON-F&O STATE-KNOWN: {panel_r.get('state_known_metrics',{}).get('non_fno_modalities_state_known_pct')}")
    print(f"F&O OBSERVED SYMBOLS: {fno_r.get('observed_trainable_symbols')}")
    print(f"F&O UNOBSERVED SYMBOLS: {fno_r.get('unobserved_trainable_symbols')}")
    print(f"BROAD FUNDAMENTALS: {fund_r.get('broad',{}).get('symbols')}")
    print(f"FILING CACHE: {fund_r.get('cache',{}).get('symbols')}")
    print("-"*100)
    if blocking:
        print("BLOCKING FINDINGS:")
        for x in sorted(set(blocking)): print("  BLOCK:",x)
    else:
        print("BLOCKING FINDINGS: NONE")
    print("STRUCTURAL FINDINGS:")
    for x in sorted(set(structural)): print("  INFO:",x)
    print("-"*100)
    print("OUTPUT:",json_path)
    print("="*100)
    return 2 if blocking else 0

if __name__ == "__main__":
    raise SystemExit(main())
