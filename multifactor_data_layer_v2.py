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

def build_feature_matrix(frame: pd.DataFrame, features: Iterable[str] = FEATURES_59) -> pd.DataFrame:
    """Return the exact ordered numeric model input matrix without inventing values.

    Columns must exist in the assembled feature panel. Genuine missing values
    remain NaN for the fitted model's training-time imputer; the caller can no
    longer silently drop/reorder inputs or create a partial-width matrix.
    """
    feature_order = list(features)
    missing = [c for c in feature_order if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Model input is missing {len(missing)} required feature columns: {missing}"
        )
    matrix = frame.loc[:, feature_order].copy()
    for column in feature_order:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def feature_coverage_report(
    frame: pd.DataFrame,
    features: Iterable[str] = FEATURES_59,
) -> dict:
    """Report raw, pre-imputation coverage for every feature in the locked contract."""
    feature_order = list(features)
    rows = int(len(frame))
    by_feature = {}
    for column in feature_order:
        if column not in frame.columns:
            observed = 0
            unique = 0
            missing_column = True
        else:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            observed = int(values.notna().sum())
            unique = int(values.nunique(dropna=True))
            missing_column = False
        by_feature[column] = {
            "column_missing_from_frame": missing_column,
            "non_missing_rows": observed,
            "missing_rows": int(rows - observed),
            "non_missing_pct": round(100.0 * observed / rows, 3) if rows else 0.0,
            "unique_non_missing_values": unique,
            "zero_coverage": observed == 0,
        }
    zero_coverage = [c for c, detail in by_feature.items() if detail["zero_coverage"]]
    missing_columns = [c for c, detail in by_feature.items() if detail["column_missing_from_frame"]]
    return {
        "rows": rows,
        "feature_count": len(feature_order),
        "mean_non_missing_pct": round(
            sum(v["non_missing_pct"] for v in by_feature.values()) / len(feature_order), 3
        ) if feature_order else 0.0,
        "features_missing_as_columns": missing_columns,
        "features_with_zero_observations": zero_coverage,
        "features": by_feature,
    }


def session_close_timestamp_utc(values: pd.Series) -> pd.Series:
    """Convert session dates to the actual 15:30 Asia/Kolkata decision clock in UTC."""
    ts = pd.to_datetime(values, errors="coerce")
    if ts.dt.tz is None:
        local_dates = ts.dt.normalize().dt.tz_localize("Asia/Kolkata")
    else:
        local_dates = ts.dt.tz_convert("Asia/Kolkata").dt.normalize()
    return (local_dates + pd.Timedelta(hours=15, minutes=30)).dt.tz_convert("UTC")


def assert_training_feature_coverage(
    frame: pd.DataFrame,
    features: Iterable[str] = FEATURES_59,
    *,
    min_observations: int = 100,
    min_fraction: float = 0.005,
) -> dict:
    """Fail candidate training if any locked feature is effectively unobserved.

    This is a data-availability gate, not a demand that every security have every
    factor. Per-security non-applicability remains NaN and is left for the fitted
    imputer. Across the training cohort, however, every feature must have enough
    real observations to be estimable rather than silently becoming a constant.
    """
    feature_order = list(features)
    report = feature_coverage_report(frame, feature_order)
    base_population = int(len(frame))
    scoped = {}
    blocked = {}

    # Numeric derivatives features are applicable only to rows for which a
    # derivatives underlying was actually available. Fundamental values are
    # measured among rows with an as-of filing. This prevents genuine
    # non-applicability for a stock from hiding whether the feature source works.
    for name, detail in report["features"].items():
        mask = pd.Series(True, index=frame.index)
        scope = "all_training_rows"
        if name in GROUPS["fno"] and name != "fno_available" and "fno_available" in frame.columns:
            availability = pd.to_numeric(frame["fno_available"], errors="coerce")
            mask = availability.eq(1)
            scope = "fno_applicable_rows"
        elif name in GROUPS["fundamentals"] and "ready_fundamentals" in frame.columns:
            mask = pd.to_numeric(frame["ready_fundamentals"], errors="coerce").eq(1)
            scope = "rows_with_asof_fundamental"
        elif name in GROUPS["market_price"] and "ready_market_price" in frame.columns:
            mask = pd.to_numeric(frame["ready_market_price"], errors="coerce").eq(1)
            scope = "market_price_ready_rows"
        elif name in GROUPS["market_context"] and "ready_market_context" in frame.columns:
            mask = pd.to_numeric(frame["ready_market_context"], errors="coerce").eq(1)
            scope = "market_context_ready_rows"

        scope_count = int(mask.sum())
        if name not in frame.columns:
            observed_in_scope = 0
        else:
            values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
            observed_in_scope = int(values.loc[mask].notna().sum())
        minimum_required = max(
            int(min_observations),
            int(np.ceil(scope_count * float(min_fraction))),
        )
        scoped[name] = {
            "scope": scope,
            "scope_rows": scope_count,
            "scope_pct_of_training_rows": round(100.0 * scope_count / base_population, 3) if base_population else 0.0,
            "observed_rows_in_scope": observed_in_scope,
            "observed_pct_of_scope": round(100.0 * observed_in_scope / scope_count, 3) if scope_count else 0.0,
            "minimum_observations_required": minimum_required,
        }
        if observed_in_scope < minimum_required:
            blocked[name] = {
                **detail,
                **scoped[name],
            }

    report["coverage_scope_by_feature"] = scoped
    report["minimum_observations_required_floor"] = int(min_observations)
    report["minimum_fraction_of_applicable_rows"] = float(min_fraction)
    report["blocked_features"] = blocked
    report["passed"] = not blocked
    if blocked:
        details = ", ".join(
            f"{name}={info['observed_rows_in_scope']}/{info['scope_rows']} "
            f"({info['observed_pct_of_scope']:.3f}% of {info['scope']})"
            for name, info in blocked.items()
        )
        raise RuntimeError(
            "59-feature candidate blocked: features lack enough real training observations "
            f"(floor={int(min_observations)}, threshold={float(min_fraction)*100:.2f}% of applicable rows): {details}"
        )
    return report


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
    # A row is fully multifactor-ready only when all six groups are explicitly ready.
    # DataFrame.all(skipna=True) would otherwise treat unavailable groups as satisfied.
    x["multi_factor_full_ready"]=x[flags].eq(1).fillna(False).all(axis=1).astype("Int8")
    x["core_market_ready"]=(x["ready_market_price"].eq(1)&x["ready_market_context"].eq(1)).astype("Int8")
    x["data_readiness_score"]=x[flags].fillna(0).astype(float).sum(axis=1)/len(flags)
    x["feature_non_null_fraction"]=x[FEATURES_59].notna().mean(axis=1)
    return x

def readiness_report(frame: pd.DataFrame) -> dict:
    # Capture columns missing before attach_readiness_flags adds placeholders.
    # A placeholder preserves the 59-column contract, but must not be mistaken
    # for a feature that was actually collected or calculated.
    missing_columns = [c for c in FEATURES_59 if c not in frame.columns]
    x=attach_readiness_flags(frame)
    n=len(x)
    out={
        "rows":int(n),
        "symbols":int(x["symbol"].nunique()) if "symbol" in x else 0,
        "mean_group_readiness_pct":round(float(x["data_readiness_score"].mean()*100),3) if n else 0.0,
        "full_multifactor_ready_pct":round(float(x["multi_factor_full_ready"].mean()*100),3) if n else 0.0,
        "mean_feature_non_null_pct":round(float(x["feature_non_null_fraction"].mean()*100),3) if n else 0.0,
        "feature_columns_missing_before_readiness":missing_columns,
        "features_with_zero_observations": [c for c in FEATURES_59 if c not in frame.columns or pd.to_numeric(frame[c],errors="coerce").replace([np.inf,-np.inf],np.nan).notna().sum()==0],
        "feature_coverage":feature_coverage_report(frame),
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

def merge_pit_layer(
    base: pd.DataFrame,
    layer: pd.DataFrame,
    value_columns: list[str],
    source_name: str,
    max_age: pd.Timedelta | str | None = None,
) -> pd.DataFrame:
    """Backward as-of join an optional PIT layer, optionally expiring stale observations.

    Sparse news/events are observations, not permanent state. Callers should
    provide a max_age for event-like layers so one old announcement is not
    carried forward indefinitely. Values outside that window remain unknown
    here; the caller may set a known-zero only when source coverage is verified.
    """
    if layer is None or layer.empty:
        return base
    left=normalize_timestamp(base).sort_values(["symbol","decision_timestamp"])
    right=normalize_timestamp(layer).sort_values(["symbol","decision_timestamp"])
    cols=["symbol","decision_timestamp"]+[c for c in value_columns if c in right.columns]
    right=right[cols].drop_duplicates(["symbol","decision_timestamp"],keep="last")
    source_ts = f"_pit_source_timestamp_{source_name}"
    right = right.rename(columns={"decision_timestamp": source_ts})
    age_limit = pd.Timedelta(max_age) if max_age is not None else None
    pieces=[]
    for symbol,lp in left.groupby("symbol",sort=False):
        rp=right[right["symbol"].eq(symbol)]
        if rp.empty:
            pieces.append(lp); continue
        m=pd.merge_asof(
            lp.sort_values("decision_timestamp"),
            rp.drop(columns=["symbol"]).sort_values(source_ts),
            left_on="decision_timestamp",
            right_on=source_ts,
            direction="backward",
            allow_exact_matches=True,
            suffixes=("",f"_{source_name}")
        )
        if age_limit is not None:
            age = m["decision_timestamp"] - m[source_ts]
            stale = m[source_ts].isna() | age.gt(age_limit)
            for column in value_columns:
                if column in m.columns:
                    m.loc[stale, column] = np.nan
        m = m.drop(columns=[source_ts], errors="ignore")
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
