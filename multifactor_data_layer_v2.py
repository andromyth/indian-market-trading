"""Truthful point-in-time multi-factor readiness utilities.

Standardizes availability flags for news, corporate events, F&O and
fundamentals for the existing 59-feature ALL-NSE Delivery contract.
Missing historical information is never fabricated.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

FEATURES_59 = [
    "ret_1","ret_3","ret_6","ret_12","ret_24","ret_48","rsi14","atr14",
    "ma20_ratio","ma50_ratio","volume_ratio","volume_z20","delivery_pct",
    "delivery_pct_change_1","delivery_pct_ratio20","delivery_pct_z20",
    "delivery_qty_ratio20","delivery_qty_z20","range_pct","body_pct","up_bar",
    "market_ret_1","market_ret_3","market_ret_6","market_rsi14",
    "relative_ret_1","relative_ret_6","relative_ret_12","market_mood_proxy",
    "market_stress_proxy","news_available","event_available","event_high_impact",
    "fno_pcr","fno_oi_change","fno_iv_change","fno_available","fno_fut_close",
    "fno_fut_basis","fno_fut_return","fno_fut_oi","fundamental_revenue","fundamental_pat",
    "fundamental_eps","fundamental_ebitda","fundamental_finance_cost",
    "fundamental_total_equity","fundamental_total_debt","fundamental_cash",
    "fund_revenue_growth","fund_pat_growth","fund_eps_growth",
    "fund_ebitda_growth","fund_pat_margin","fund_debt_to_equity",
    "fund_cash_to_debt","fundamental_age_days","fundamental_stale_90d",
    "fundamental_stale_180d",
]

GROUPS = {
    "market_price": FEATURES_59[:21],
    "market_context": FEATURES_59[21:30],
    "news": ["news_available"],
    "events": ["event_available","event_high_impact"],
    "fno": FEATURES_59[33:41],
    "fundamentals": FEATURES_59[41:],
}
if len(FEATURES_59) != 59:
    raise RuntimeError(f"Feature contract must contain exactly 59 features; found {len(FEATURES_59)}")

READINESS_COLUMNS = {
    "market_price":"ready_market_price",
    "market_context":"ready_market_context",
    "news":"ready_news",
    "events":"ready_events",
    "fno":"ready_fno",
    "fundamentals":"ready_fundamentals",
}

def normalize_symbol(value: object) -> str:
    s = str(value or "").strip().upper()
    for suffix in (".NS",".BO","-EQ",".EQ"):
        if s.endswith(suffix):
            s=s[: -len(suffix)]
    return s

def normalize_timestamp(frame: pd.DataFrame, preferred: Iterable[str] = ()) -> pd.DataFrame:
    x=frame.copy()
    if "decision_timestamp" not in x.columns:
        aliases=list(preferred)+["timestamp","datetime","date_time","bar_timestamp","trade_timestamp","available_at","publication_timestamp","observed_at","fundamental_available_at","fno_observed_at"]
        present=[c for c in aliases if c in x.columns]
        if len(present)!=1:
            raise ValueError(f"Unable to resolve PIT timestamp; present={present}")
        x=x.rename(columns={present[0]:"decision_timestamp"})
    x["decision_timestamp"]=pd.to_datetime(x["decision_timestamp"],errors="coerce",utc=True)
    if "symbol" not in x.columns:
        raise ValueError("PIT frame requires symbol")
    x["symbol"]=x["symbol"].map(normalize_symbol)
    return x.dropna(subset=["symbol","decision_timestamp"]).copy()

def _available_from_values(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    present=[c for c in cols if c in frame.columns]
    if not present:
        return pd.Series(np.nan,index=frame.index,dtype="float64")
    numeric=frame[present].apply(pd.to_numeric,errors="coerce")
    return numeric.notna().any(axis=1).astype("Int8")

def attach_readiness_flags(frame: pd.DataFrame) -> pd.DataFrame:
    x=frame.copy()
    for c in FEATURES_59:
        if c not in x.columns:
            x[c]=np.nan
    x["ready_market_price"]=_available_from_values(x,GROUPS["market_price"])
    x["ready_market_context"]=_available_from_values(x,GROUPS["market_context"])

    for group,source in (("news","news_available"),("events","event_available"),("fno","fno_available")):
        if source in x.columns:
            s=pd.to_numeric(x[source],errors="coerce")
            x[READINESS_COLUMNS[group]]=s.astype("Int8")
        else:
            x[READINESS_COLUMNS[group]]=_available_from_values(x,GROUPS[group])

    if "fundamentals_available" in x.columns:
        x["ready_fundamentals"]=pd.to_numeric(x["fundamentals_available"],errors="coerce").astype("Int8")
    else:
        x["ready_fundamentals"]=_available_from_values(x,GROUPS["fundamentals"])

    flags=list(READINESS_COLUMNS.values())
    x["information_groups_available"]=x[flags].notna().sum(axis=1).astype("Int8")
    x["information_groups_ready"]=x[flags].fillna(0).astype(float).sum(axis=1).astype("Int8")
    x["multi_factor_full_ready"]=x[flags].eq(1).all(axis=1).astype("Int8")
    x["core_market_ready"]=(x["ready_market_price"].eq(1)&x["ready_market_context"].eq(1)).astype("Int8")
    x["data_readiness_score"]=x[flags].fillna(0).astype(float).sum(axis=1)/len(flags)
    x["feature_non_null_fraction"]=x[FEATURES_59].notna().mean(axis=1)
    return x

def readiness_report(frame: pd.DataFrame) -> dict:
    x=attach_readiness_flags(frame)
    n=len(x)
    out={
        "rows":int(n),
        "symbols":int(x["symbol"].nunique()) if "symbol" in x else 0,
        "mean_group_readiness_pct":round(float(x["data_readiness_score"].mean()*100),3) if n else 0.0,
        "full_multifactor_ready_pct":round(float(x["multi_factor_full_ready"].mean()*100),3) if n else 0.0,
        "mean_feature_non_null_pct":round(float(x["feature_non_null_fraction"].mean()*100),3) if n else 0.0,
        "groups":{},
    }
    for group,flag in READINESS_COLUMNS.items():
        s=x[flag]
        out["groups"][group]={
            "known_rows":int(s.notna().sum()),
            "ready_rows":int(s.eq(1).sum()),
            "known_coverage_pct":round(float(s.notna().mean()*100),3) if n else 0.0,
            "ready_coverage_pct":round(float(s.eq(1).mean()*100),3) if n else 0.0,
            "ready_symbols":int(x.loc[s.eq(1),"symbol"].nunique()) if n else 0,
        }
    return out

def merge_pit_layer(base: pd.DataFrame, layer: pd.DataFrame, value_columns: list[str], source_name: str) -> pd.DataFrame:
    if layer is None or layer.empty:
        return base
    left=normalize_timestamp(base).sort_values(["symbol","decision_timestamp"])
    right=normalize_timestamp(layer).sort_values(["symbol","decision_timestamp"])
    cols=["symbol","decision_timestamp"]+[c for c in value_columns if c in right.columns]
    right=right[cols].drop_duplicates(["symbol","decision_timestamp"],keep="last")
    pieces=[]
    for symbol,lp in left.groupby("symbol",sort=False):
        rp=right[right["symbol"].eq(symbol)]
        if rp.empty:
            pieces.append(lp); continue
        m=pd.merge_asof(
            lp.sort_values("decision_timestamp"),
            rp.drop(columns=["symbol"]).sort_values("decision_timestamp"),
            on="decision_timestamp",direction="backward",allow_exact_matches=True,
            suffixes=("",f"_{source_name}")
        )
        pieces.append(m)
    out=pd.concat(pieces,ignore_index=True)
    for c in ("fundamental_available_at","fno_observed_at","available_at"):
        if c in out.columns:
            ts=pd.to_datetime(out[c],errors="coerce",utc=True)
            bad=ts.notna()&ts.gt(out["decision_timestamp"])
            if bool(bad.any()):
                raise RuntimeError(f"{source_name}: {int(bad.sum())} future PIT observations")
    return out

def load_optional_layer(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size==0:
        return pd.DataFrame()
    x=pd.read_parquet(path)
    return normalize_timestamp(x) if not x.empty else pd.DataFrame()
