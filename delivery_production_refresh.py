#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, os, re, socket, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

DEFAULT_BASE = "/content/drive/MyDrive/Indian_Market_Research"
BASE = Path(os.getenv("INDIAN_MARKET_RESEARCH_ROOT", DEFAULT_BASE))
FEATURE_DIR = BASE / "features" / "daily"
DELIVERY_LOCAL = BASE / "production_data" / "nse_official" / "delivery_full"
DELIVERY_REMOTE = "production_data/nse_official/delivery_full"
EOD_REMOTE_ROOT = "production_data/nse_official/bulk_eod"
EOD_LOCAL_ROOT = BASE / "production_data" / "nse_official" / "bulk_eod"
REMOTE_SOURCES = {
    "fno": "production_data/nse_official/fno/fno_daily_features_2026.parquet",
    "news": "processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet",
    "web_news": "processed/live_news/website_news_pit_2026.parquet",
    "events": "processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet",
    "news_events_manifest": "processed/pit_replay/nse_all_universe_raw_2026/ALL_NSE_NEWS_EVENTS_MANIFEST_2026.json",
    "fundamentals": "corporate/pit_fundamentals_all_nse_2025_2026.parquet",
    "fundamental_availability": "corporate/pit_fundamental_availability_2025_2026.parquet",
    "fundamental_availability_meta": "corporate/pit_fundamental_availability_2025_2026.meta.json",
    "fundamental_filing_catalog": "corporate/nse_integrated_filing_catalog_2025_2026.json",
}
MIN_HISTORY = 60


def _stage(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def readiness_summary(values: pd.Series) -> dict[str, float | int]:
    """Summarize a readiness column safely, including pandas nullable dtypes."""
    series = pd.to_numeric(values, errors="coerce")
    row_count = int(len(series))
    denominator = max(row_count, 1)
    known_count = int(series.notna().sum())
    # Comparisons on nullable Float64/boolean Series can retain pd.NA. Count
    # only explicit ones as ready; unknowns remain unknown, never ready.
    ready_count = int(series.eq(1).fillna(False).sum())
    return {
        "known_pct": round(known_count * 100.0 / denominator, 3) if row_count else 0.0,
        "ready_pct": round(ready_count * 100.0 / denominator, 3) if row_count else 0.0,
        "unknown_rows": int(series.isna().sum()),
    }

def _store():
    from persistent_store import GoogleDriveStore
    return GoogleDriveStore()


def _refresh_fundamental_availability(
    store,
    symbols: list[str],
    *,
    as_of_date: date,
    allow_stale_fallback: bool = False,
) -> dict:
    """Refresh filing availability as of the target session.

    An explicit fallback may reuse older metadata, but never extends its
    coverage. Rows beyond cached coverage remain unknown and blocked.
    """
    from colab_build_all_nse_delivery_multifactor_candidate import (
        FUND_AVAILABILITY,
        FUND_AVAILABILITY_META,
        FUND_CATALOG_CACHE,
        _build_fundamental_availability,
        _catalog_all,
    )

    target = as_of_date
    try:
        catalog = _catalog_all(force_refresh=False, as_of_date=target)
        report = _build_fundamental_availability(catalog, symbols)
        coverage_end = pd.to_datetime(report.get("coverage_end"), errors="coerce")
        if report.get("coverage_complete") is not True or pd.isna(coverage_end) or coverage_end.date() < target:
            raise RuntimeError(
                "NSE filing availability does not cover target session "
                f"{target}: {report}"
            )
    except Exception as exc:
        if not allow_stale_fallback:
            raise
        cached_meta = {}
        try:
            if FUND_AVAILABILITY_META.exists() and FUND_AVAILABILITY_META.stat().st_size:
                cached_meta = json.loads(FUND_AVAILABILITY_META.read_text(encoding="utf-8"))
        except Exception:
            cached_meta = {}
        coverage_start = pd.to_datetime(cached_meta.get("coverage_start"), errors="coerce")
        coverage_end = pd.to_datetime(cached_meta.get("coverage_end"), errors="coerce")
        index_available = FUND_AVAILABILITY.exists() and FUND_AVAILABILITY.stat().st_size > 0
        current_covered = bool(
            cached_meta.get("coverage_complete") is True
            and index_available
            and pd.notna(coverage_start)
            and pd.notna(coverage_end)
            and coverage_start.date() <= target <= coverage_end.date()
        )
        return {
            **cached_meta,
            "target_date": target.isoformat(),
            "refresh_status": "FALLBACK_STALE_OR_MISSING_CATALOG",
            "fallback_used": True,
            "current_target_covered": current_covered,
            "availability_index_available": index_available,
            "catalog_available": FUND_CATALOG_CACHE.exists() and FUND_CATALOG_CACHE.stat().st_size > 0,
            "refresh_error": f"{type(exc).__name__}: {exc}",
            "coverage_end_required": target.isoformat(),
            "safety_rule": "Coverage dates were not extended. Rows outside cached coverage remain unknown and blocked; missing filings were not converted to zero.",
            "catalog_persisted": False,
            "availability_index_persisted": False,
        }

    store.upload_or_update(FUND_CATALOG_CACHE, "corporate/nse_integrated_filing_catalog_2025_2026.json", mime_type="application/json")
    store.upload_or_update(FUND_AVAILABILITY, "corporate/pit_fundamental_availability_2025_2026.parquet", mime_type="application/octet-stream")
    store.upload_or_update(FUND_AVAILABILITY_META, "corporate/pit_fundamental_availability_2025_2026.meta.json", mime_type="application/json")
    return {
        **report,
        "catalog_rows": len(catalog),
        "catalog_persisted": True,
        "availability_index_persisted": True,
        "target_date": target.isoformat(),
        "coverage_end_required": target.isoformat(),
        "refresh_status": "REFRESHED",
        "fallback_used": False,
        "current_target_covered": True,
    }

def _is_retryable_download_error(exc: Exception) -> bool:
    """Return true only for transient Drive/network failures worth retrying."""
    response = getattr(exc, "resp", None)
    status = getattr(response, "status", None) or getattr(exc, "status_code", None)
    if status in {408, 429, 500, 502, 503, 504}:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, socket.timeout)):
        return True
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return any(token in name or token in message for token in (
        "timeout", "timed out", "connection reset", "connection aborted",
        "temporarily unavailable", "incompleteread", "server disconnected",
        "internal error",
    ))


def _retry_drive_operation(operation, *, context: str, retries: int = 4):
    """Retry a Drive API request on transient server, rate-limit, or network errors."""
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= attempts or not _is_retryable_download_error(exc):
                raise
            delay = min(2 ** (attempt - 1), 8)
            response = getattr(exc, "resp", None)
            status = getattr(response, "status", None) or getattr(exc, "status_code", None)
            _stage(
                f"DRIVE API RETRY: context={context}, "
                f"status={status if status is not None else 'network'}, "
                f"attempt={attempt}/{attempts}, "
                f"error={type(exc).__name__}: {exc}, sleep_seconds={delay}"
            )
            time.sleep(delay)
    raise RuntimeError(f"Drive retry loop ended unexpectedly: {context}")


def _find_child_with_retry(
    store,
    parent_id: str,
    name: str,
    mime_type: str | None = None,
    *,
    retries: int = 4,
):
    """Find a Drive child; confirmed missing items are not retried."""
    return _retry_drive_operation(
        lambda: store._find_child(parent_id, name, mime_type),
        context=f"find Drive child {name!r}",
        retries=retries,
    )


def _download_file(store, remote: str, local: Path, retries: int = 4) -> bool:
    """Download a Drive file atomically, retrying transient read failures."""
    parts = [p for p in remote.strip("/").split("/") if p]
    if not parts:
        return False
    from googleapiclient.http import MediaIoBaseDownload

    local.parent.mkdir(parents=True, exist_ok=True)
    temp = local.with_suffix(local.suffix + ".tmp")
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            parent = store.root_id
            for part in parts[:-1]:
                hit = store._find_child(parent, part)
                if not hit:
                    return False
                parent = hit["id"]
            hit = store._find_child(parent, parts[-1])
            if not hit:
                return False

            temp.unlink(missing_ok=True)
            with temp.open("wb") as fh:
                downloader = MediaIoBaseDownload(
                    fh,
                    store.service.files().get_media(fileId=hit["id"]),
                )
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            if temp.stat().st_size == 0:
                raise RuntimeError(f"Drive download returned an empty file: {remote}")
            temp.replace(local)
            return True
        except Exception as exc:
            temp.unlink(missing_ok=True)
            if attempt >= attempts or not _is_retryable_download_error(exc):
                raise
            delay = min(2 ** (attempt - 1), 8)
            _stage(
                f"DRIVE DOWNLOAD RETRY: path={remote}, "
                f"attempt={attempt}/{attempts}, "
                f"error={type(exc).__name__}: {exc}, sleep_seconds={delay}"
            )
            time.sleep(delay)
    return False

def _walk(store, folder_id: str, prefix: str, depth: int, out: list[dict]) -> None:
    if depth < 0: return
    token = None
    while True:
        kwargs = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,mimeType,size,modifiedTime)",
            "pageSize": 1000,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if token: kwargs["pageToken"] = token
        resp = _retry_drive_operation(
            lambda: store.service.files().list(**kwargs).execute(),
            context=f"list Drive folder {prefix or folder_id!r}",
        )
        for item in resp.get("files", []):
            name = str(item.get("name",""))
            rel = f"{prefix}/{name}".strip("/")
            if item.get("mimeType") == "application/vnd.google-apps.folder":
                _walk(store, item["id"], rel, depth-1, out)
            else:
                out.append({**item, "relative": rel})
        token = resp.get("nextPageToken")
        if not token: break

def _decorate_snapshot_rows(frame, features, readiness_cols):
    """Attach the shared production gate and coverage diagnostics to a PIT slice."""
    from delivery_quality_gate import quality_gate_reasons

    snapshot = frame.copy()
    missing = [column for column in features if column not in snapshot.columns]
    if missing:
        raise RuntimeError(f"59-feature production snapshot missing columns: {missing}")
    snapshot["production_data_quality_score"] = (
        snapshot[readiness_cols].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
        / len(readiness_cols)
    )
    numeric_features = snapshot[features].apply(pd.to_numeric, errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )
    snapshot["production_feature_coverage"] = numeric_features.notna().mean(axis=1)
    snapshot["production_feature_non_missing_count"] = numeric_features.notna().sum(axis=1)
    snapshot["production_core_ready"] = (
        pd.to_numeric(snapshot["ready_market_price"], errors="coerce").eq(1)
        & pd.to_numeric(snapshot["ready_market_context"], errors="coerce").eq(1)
    )
    snapshot["production_optional_state_known"] = (
        snapshot[["ready_news", "ready_events", "ready_fno", "ready_fundamentals"]]
        .apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    )
    snapshot["production_data_quality_gate_reasons"] = snapshot.apply(
        lambda row: json.dumps(quality_gate_reasons(row)), axis=1
    )
    snapshot["production_data_quality_gate"] = snapshot["production_data_quality_gate_reasons"].eq("[]")
    return snapshot


def _download_recent_eod(store, lookback_days: int):
    p = store.root_id
    for part in ("production_data","nse_official","bulk_eod"):
        hit = _find_child_with_retry(store, p, part, "application/vnd.google-apps.folder")
        if not hit: raise RuntimeError(f"Drive folder missing: {part}")
        p = hit["id"]
    files=[]
    _walk(store,p,EOD_REMOTE_ROOT,4,files)
    rx=re.compile(r"BhavCopy_NSE_CM_0_0_0_(\d{8})_F_0000\.parquet$",re.I)
    cutoff=datetime.now(timezone.utc).date()-timedelta(days=lookback_days)
    chosen=[]
    for item in files:
        m=rx.match(item["name"])
        if not m: continue
        d=datetime.strptime(m.group(1),"%Y%m%d").date()
        if d>=cutoff: chosen.append((d,item))
    chosen.sort()
    for d,item in chosen:
        rel=Path(item["relative"].split(EOD_REMOTE_ROOT+"/",1)[1])
        local=EOD_LOCAL_ROOT/rel
        if not local.exists() or local.stat().st_size==0:
            _download_file(store,f"{EOD_REMOTE_ROOT}/{rel.as_posix()}",local)
    return {"files_found":len(chosen),"first_date":str(chosen[0][0]) if chosen else None,"last_date":str(chosen[-1][0]) if chosen else None}

def _download_recent_delivery(store, lookback_days:int):
    p=store.root_id
    for part in DELIVERY_REMOTE.split("/"):
        hit=_find_child_with_retry(store,p,part,"application/vnd.google-apps.folder")
        if not hit: return {"files_found":0,"files_downloaded":0}
        p=hit["id"]
    items=[]
    page_token=None
    while True:
        kwargs={
            "q": f"'{p}' in parents and trashed = false",
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name)",
            "pageSize": 1000,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if page_token: kwargs["pageToken"]=page_token
        resp=_retry_drive_operation(
            lambda: store.service.files().list(**kwargs).execute(),
            context=f"list Delivery folder {DELIVERY_REMOTE}",
        )
        items.extend(resp.get("files",[]))
        page_token=resp.get("nextPageToken")
        if not page_token: break
    rx=re.compile(r"delivery_(\d{8})\.parquet$",re.I)
    cutoff=datetime.now(timezone.utc).date()-timedelta(days=lookback_days)
    chosen=[]
    for item in items:
        m=rx.match(str(item.get("name","")))
        if not m: continue
        d=datetime.strptime(m.group(1),"%Y%m%d").date()
        if d>=cutoff: chosen.append((d,item))
    chosen.sort()
    downloaded=0
    for d,item in chosen:
        local=DELIVERY_LOCAL/item["name"]
        if local.exists() and local.stat().st_size>0: continue
        if _download_file(
            store,
            f"{DELIVERY_REMOTE}/{item['name']}",
            local,
        ):
            downloaded+=1
    return {"files_found":len(chosen),"files_downloaded":downloaded,"first_date":str(chosen[0][0]) if chosen else None,"last_date":str(chosen[-1][0]) if chosen else None}

def _load_eod(root:Path,start:date,end:date)->pd.DataFrame:
    files=sorted(root.rglob("BhavCopy_NSE_CM_0_0_0_*_F_0000.parquet"))
    parts=[]
    for path in files:
        m=re.search(r"_(\d{8})_F_0000\.parquet$",path.name)
        if not m: continue
        d=datetime.strptime(m.group(1),"%Y%m%d").date()
        if not (start<=d<=end): continue
        try: x=pd.read_parquet(path)
        except Exception: continue
        if x.empty: continue
        cols={str(c).strip().lower():c for c in x.columns}
        def pick(names):
            for n in names:
                if n in cols: return cols[n]
            return None
        ren={}
        for t,names in {
            "symbol":["symbol","tckrsymb","ticker"],
            "date":["date","trade_date","trad_dt"],
            "open":["open","open_price","opnpric"],
            "high":["high","high_price","hghpric"],
            "low":["low","low_price","lwpric"],
            "close":["close","close_price","clspric"],
            "volume":["volume","traded_volume","ttltradgvol"],
        }.items():
            c=pick(names)
            if c is not None: ren[c]=t
        x=x.rename(columns=ren)
        req={"symbol","date","open","high","low","close","volume"}
        if not req.issubset(x.columns): continue
        x["symbol"]=x["symbol"].astype(str).str.upper().str.strip()
        x["date"]=pd.to_datetime(x["date"],errors="coerce").dt.normalize()
        for c in req-{"symbol","date"}: x[c]=pd.to_numeric(x[c],errors="coerce")
        parts.append(x.dropna(subset=list(req)))
    if not parts: raise RuntimeError("No recent NSE EOD data in Drive")
    return pd.concat(parts,ignore_index=True).sort_values(["symbol","date"]).drop_duplicates(["symbol","date"],keep="last").reset_index(drop=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lookback-days",type=int,default=150)
    ap.add_argument("--delivery-lookback-days",type=int,default=150)
    ap.add_argument("--source-wait-minutes",type=int,default=10)
    ap.add_argument(
        "--expected-date",
        default="",
        help="Expected NSE production session date (YYYY-MM-DD).",
    )
    args=ap.parse_args()
    expected_date = (
        date.fromisoformat(args.expected_date.strip())
        if args.expected_date.strip()
        else None
    )
    _stage("START: creating Google Drive store")
    store=_store()
    _stage("Drive store ready")

    # Upstream F&O/news workflows can finish a little later than the Delivery
    # archive job. Poll missing upstream files briefly rather than producing a
    # falsely data-limited production snapshot during normal workflow overlap.
    deadline=datetime.now(timezone.utc)+timedelta(minutes=max(0,args.source_wait_minutes))
    _stage("SOURCE PHASE: checking F&O/news/events/fundamentals")
    source_status={}
    while True:
        source_status={}
        for k,remote in REMOTE_SOURCES.items():
            local=BASE/remote
            if local.exists() and local.stat().st_size>0: source_status[k]="LOCAL"
            elif _download_file(store,remote,local): source_status[k]="DOWNLOADED"
            else: source_status[k]="MISSING"
        # The filing catalogue is refreshable from NSE directly; its absence
        # in Drive must not consume the entire upstream wait window.
        missing=[
            k for k,v in source_status.items()
            if v=="MISSING" and k not in ("fundamental_filing_catalog", "web_news")
        ]
        if not missing or datetime.now(timezone.utc)>=deadline:
            break
        print(f"Waiting for upstream source files: {', '.join(missing)}",flush=True)
        import time as _time
        _time.sleep(60)
    _stage("EOD PHASE: scanning Drive archive and downloading missing recent files")
    eod_status=_download_recent_eod(store,args.lookback_days)
    _stage(f"EOD PHASE DONE: {eod_status}")
    _stage("DELIVERY PHASE: scanning Drive archive and downloading missing recent files")
    delivery_status=_download_recent_delivery(store,args.delivery_lookback_days)
    _stage(f"DELIVERY PHASE DONE: {delivery_status}")
    _stage("IMPORT PHASE: loading panel/model builder modules")
    from colab_build_all_nse_delivery_multifactor_candidate import make_panel,merge_optional
    from delivery_training_universe import training_roster,load_official_nse_universe
    from multifactor_data_layer_v2 import FEATURES_59, attach_readiness_flags, feature_coverage_report
    from nse_delivery_data_layer import load_delivery_history
    reference_date = (
        expected_date
        if expected_date is not None
        else pd.Timestamp.now(tz="Asia/Kolkata").date()
    )
    today=reference_date

    _stage(f"TARGET PHASE: production reference date={reference_date}")
    # Persist the current RSS headline snapshot so automated and interactive
    # Delivery predictions use the same latest-news source rather than relying
    # on a user's browser session to fetch it first.
    live_news_path=BASE/"processed/live_news/news_latest.parquet"
    try:
        from selfdriving.sd_news_collect import build_alias_map,collect
        universe_df=load_official_nse_universe(force_refresh=False)
        alias_map=build_alias_map(universe_df)
        live_news_result=collect(alias_map,live_news_path)
        source_status["live_rss_news"]="UPDATED"
    except Exception as exc:
        live_news_result={"status":"ERROR","error":f"{type(exc).__name__}: {exc}"}
        source_status["live_rss_news"]="ERROR"
    _stage(f"NEWS PHASE DONE: {live_news_result.get('status', 'unknown')}")

    _stage("BULK EOD PHASE: loading recent NSE history from local cache")
    bulk=_load_eod(
        EOD_LOCAL_ROOT,
        reference_date-timedelta(days=args.lookback_days),
        reference_date,
    )
    _stage(f"BULK EOD LOADED: rows={len(bulk):,}, symbols={bulk['symbol'].nunique():,}")
    _stage("ROSTER PHASE: computing current 60-session clean-history universe")
    roster=training_roster(bulk,min_history=MIN_HISTORY,force_refresh=True)
    trainable=list(roster["trainable_symbols"])
    _stage(f"ROSTER DONE: universe={roster['universe_symbol_count']}, trainable={len(trainable):,}")
    if not trainable: raise RuntimeError("No current NSE symbols meet the 60-session clean-history gate")
    _stage("DELIVERY HISTORY PHASE: loading local Delivery parquet files")
    delivery=load_delivery_history(DELIVERY_LOCAL,symbols=trainable,start=bulk["date"].min(),end=bulk["date"].max())
    _stage(f"DELIVERY HISTORY LOADED: rows={len(delivery):,}")
    _stage("PANEL PHASE: building multifactor production panel")
    panel=make_panel(bulk,trainable,sort_output=False,delivery=delivery)
    _stage(f"PANEL BUILT: rows={len(panel):,}, cols={len(panel.columns):,}")
    _stage("FUNDAMENTAL AVAILABILITY PHASE: refreshing official NSE filing coverage")
    fundamental_availability_refresh = _refresh_fundamental_availability(
        store, trainable
    )
    _stage(
        "FUNDAMENTAL AVAILABILITY REFRESH DONE: "
        + json.dumps(fundamental_availability_refresh, default=str)
    )
    _stage("OPTIONAL SOURCES PHASE: merging F&O/news/events/fundamentals")
    panel,source_audit=merge_optional(panel)
    source_audit["fundamental_availability_daily_refresh"] = fundamental_availability_refresh
    _stage(f"OPTIONAL SOURCES MERGED: rows={len(panel):,}, cols={len(panel.columns):,}")

    # Distinguish "source is present and this symbol has no applicable item"
    # from "the source pipeline itself is missing". The former is known state
    # (0); the latter remains NaN and is blocked by the production gate.
    _stage("READINESS PHASE: attaching source-aware production data-quality flags")
    # A source file existing somewhere in Drive does not prove every date and
    # every symbol has known coverage. Optional-source readiness zeros are set
    # only by merge_optional() for verified covered dates; preserve NaN elsewhere
    # so the production gate can block signals instead of claiming false readiness.
    panel=attach_readiness_flags(panel)
    _stage("READINESS PHASE DONE")

    panel["decision_timestamp"]=pd.to_datetime(panel["decision_timestamp"],errors="coerce",utc=True)
    panel=panel.dropna(subset=["symbol","decision_timestamp"]).sort_values(["symbol","decision_timestamp"]).drop_duplicates("symbol",keep="last").reset_index(drop=True)
    missing=[c for c in FEATURES_59 if c not in panel.columns]
    if missing: raise RuntimeError(f"59-feature production panel missing columns: {missing}")
    readiness_cols=["ready_market_price","ready_market_context","ready_news","ready_events","ready_fno","ready_fundamentals"]
    panel["production_data_quality_score"]=panel[readiness_cols].apply(pd.to_numeric,errors="coerce").notna().sum(axis=1)/len(readiness_cols)
    panel["production_feature_coverage"] = panel[FEATURES_59].apply(
        pd.to_numeric, errors="coerce"
    ).replace([float("inf"), float("-inf")], float("nan")).notna().mean(axis=1)
    panel["production_feature_non_missing_count"] = panel[FEATURES_59].apply(
        pd.to_numeric, errors="coerce"
    ).replace([float("inf"), float("-inf")], float("nan")).notna().sum(axis=1)
    panel["production_core_ready"]=(
        pd.to_numeric(panel["ready_market_price"],errors="coerce").eq(1)&
        pd.to_numeric(panel["ready_market_context"],errors="coerce").eq(1)
    )
    panel["production_optional_state_known"]=panel[["ready_news","ready_events","ready_fno","ready_fundamentals"]].apply(pd.to_numeric,errors="coerce").notna().all(axis=1)
    # Keep the strict gate; make every block explainable instead of silently
    # converting unknown source state into a generic HOLD downstream.
    from delivery_quality_gate import quality_gate_reasons
    panel["production_data_quality_gate_reasons"] = panel.apply(
        lambda row: json.dumps(quality_gate_reasons(row)), axis=1
    )
    panel["production_data_quality_gate"] = panel["production_data_quality_gate_reasons"].eq("[]")
    latest_date=pd.to_datetime(panel["decision_timestamp"],utc=True).dt.tz_convert("Asia/Kolkata").dt.date.max()
    if expected_date is not None and latest_date != expected_date:
        raise RuntimeError(
            f"Production refresh resolved latest session {latest_date}, "
            f"but expected target session is {expected_date}. "
            "Refusing to publish an older PIT snapshot."
        )
    snap=FEATURE_DIR/f"DELIVERY_PIT_FEATURES_{latest_date}.parquet"
    snap.parent.mkdir(parents=True,exist_ok=True)
    _stage(f"SNAPSHOT WRITE: {snap.name}")
    panel.to_parquet(snap,index=False)
    _stage("SNAPSHOT WRITE DONE")
    group_ready={}
    for c in readiness_cols:
        group_ready[c]=readiness_summary(panel[c])
    report={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "snapshot":snap.name,"snapshot_date":str(latest_date),
        "feature_count":len(FEATURES_59),"symbols":int(panel["symbol"].nunique()),"rows":int(len(panel)),
        "universe_symbol_count":int(roster["universe_symbol_count"]),
        "trainable_symbol_count":int(roster["trainable_symbol_count"]),
        "excluded_symbol_count":int(roster["excluded_symbol_count"]),
        "minimum_history_sessions":MIN_HISTORY,
        "source_status":source_status,"eod_status":eod_status,"delivery_status":delivery_status,"live_rss_news":live_news_result,
        "source_audit":source_audit,"group_readiness":group_ready,
        "feature_coverage":feature_coverage_report(panel),
        "production_feature_coverage_pct":round(float(panel["production_feature_coverage"].mean()*100),3),
        "production_rows_all_59_features":int(panel["production_feature_non_missing_count"].eq(len(FEATURES_59)).sum()),
        "production_data_quality_gate_pct":readiness_summary(panel["production_data_quality_gate"])["ready_pct"],
        "production_data_quality_gate_block_reason_counts": {
            reason: int(sum(reason in json.loads(raw) for raw in panel["production_data_quality_gate_reasons"]))
            for reason in sorted({
                reason
                for raw in panel["production_data_quality_gate_reasons"]
                for reason in json.loads(raw)
            })
        },
        "production_data_quality_gate_pass_rows": int(panel["production_data_quality_gate"].sum()),
        "production_data_quality_gate_blocked_rows": int((~panel["production_data_quality_gate"]).sum()),
        "ipo_policy":"Any current official NSE equity becomes eligible automatically after 60 clean sessions.",
        "model_weights_changed":False,"model_promotion_performed":False
    }
    report_path=FEATURE_DIR/f"DELIVERY_PIT_READINESS_{latest_date}.json"
    report_path.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    _stage("DRIVE UPLOAD: snapshot + readiness report")
    store.upload_or_update(snap,f"features/daily/{snap.name}",mime_type="application/octet-stream")
    store.upload_or_update(report_path,f"features/daily/{report_path.name}",mime_type="application/json")
    if live_news_path.exists():
        store.upload_or_update(live_news_path,"processed/live_news/news_latest.parquet",mime_type="application/octet-stream")
    _stage("DONE: Delivery PIT production refresh completed")
    print(json.dumps(report,indent=2,default=str))
if __name__=="__main__":
    main()
