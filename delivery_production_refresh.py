#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

BASE = Path("/content/drive/MyDrive/Indian_Market_Research")
FEATURE_DIR = BASE / "features" / "daily"
DELIVERY_LOCAL = BASE / "production_data" / "nse_official" / "delivery_full"
DELIVERY_REMOTE = "production_data/nse_official/delivery_full"
EOD_REMOTE_ROOT = "production_data/nse_official/bulk_eod"
EOD_LOCAL_ROOT = BASE / "production_data" / "nse_official" / "bulk_eod"
REMOTE_SOURCES = {
    "fno": "production_data/nse_official/fno/fno_daily_features_2026.parquet",
    "news": "processed/pit_replay/NEWS_ENRICHED_PIT_REPLAY_2026.parquet",
    "events": "processed/pit_replay/EVENTS_FUNDAMENTALS_PIT_REPLAY_2026.parquet",
    "fundamentals": "corporate/pit_fundamentals_all_nse_2025_2026.parquet",
}
MIN_HISTORY = 60

def _store():
    from persistent_store import GoogleDriveStore
    return GoogleDriveStore()

def _download_file(store, remote: str, local: Path) -> bool:
    parts = [p for p in remote.strip("/").split("/") if p]
    parent = store.root_id
    for part in parts[:-1]:
        hit = store._find_child(parent, part)
        if not hit:
            return False
        parent = hit["id"]
    hit = store._find_child(parent, parts[-1])
    if not hit:
        return False
    from googleapiclient.http import MediaIoBaseDownload
    local.parent.mkdir(parents=True, exist_ok=True)
    temp = local.with_suffix(local.suffix + ".tmp")
    with temp.open("wb") as fh:
        downloader = MediaIoBaseDownload(
            fh,
            store.service.files().get_media(fileId=hit["id"]),
        )
        done = False
        while not done:
            _, done = downloader.next_chunk()
    temp.replace(local)
    return True

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
        resp = store.service.files().list(**kwargs).execute()
        for item in resp.get("files", []):
            name = str(item.get("name",""))
            rel = f"{prefix}/{name}".strip("/")
            if item.get("mimeType") == "application/vnd.google-apps.folder":
                _walk(store, item["id"], rel, depth-1, out)
            else:
                out.append({**item, "relative": rel})
        token = resp.get("nextPageToken")
        if not token: break

def _download_recent_eod(store, lookback_days: int):
    p = store.root_id
    for part in ("production_data","nse_official","bulk_eod"):
        hit = store._find_child(p, part, "application/vnd.google-apps.folder")
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
        hit=store._find_child(p,part,"application/vnd.google-apps.folder")
        if not hit: return {"files_found":0,"files_downloaded":0}
        p=hit["id"]
    resp=store.service.files().list(q=f"'{p}' in parents and trashed = false",spaces="drive",
        fields="files(id,name)",pageSize=1000,includeItemsFromAllDrives=True,supportsAllDrives=True).execute()
    rx=re.compile(r"delivery_(\d{8})\.parquet$",re.I)
    cutoff=datetime.now(timezone.utc).date()-timedelta(days=lookback_days)
    chosen=[]
    for item in resp.get("files",[]):
        m=rx.match(str(item.get("name","")))
        if not m: continue
        d=datetime.strptime(m.group(1),"%Y%m%d").date()
        if d>=cutoff: chosen.append((d,item))
    chosen.sort()
    downloaded=0
    from googleapiclient.http import MediaIoBaseDownload
    for d,item in chosen:
        local=DELIVERY_LOCAL/item["name"]
        if local.exists() and local.stat().st_size>0: continue
        local.parent.mkdir(parents=True,exist_ok=True)
        temp=local.with_suffix(".tmp")
        with temp.open("wb") as fh:
            dl=MediaIoBaseDownload(fh,store.service.files().get_media(fileId=item["id"]))
            done=False
            while not done: _,done=dl.next_chunk()
        temp.replace(local); downloaded+=1
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
    args=ap.parse_args()
    store=_store()

    # Upstream F&O/news workflows can finish a little later than the Delivery
    # archive job. Poll missing upstream files briefly rather than producing a
    # falsely data-limited production snapshot during normal workflow overlap.
    deadline=datetime.now(timezone.utc)+timedelta(minutes=max(0,args.source_wait_minutes))
    source_status={}
    while True:
        source_status={}
        for k,remote in REMOTE_SOURCES.items():
            local=BASE/remote
            if local.exists() and local.stat().st_size>0: source_status[k]="LOCAL"
            elif _download_file(store,remote,local): source_status[k]="DOWNLOADED"
            else: source_status[k]="MISSING"
        missing=[k for k,v in source_status.items() if v=="MISSING"]
        if not missing or datetime.now(timezone.utc)>=deadline:
            break
        print(f"Waiting for upstream source files: {', '.join(missing)}",flush=True)
        import time as _time
        _time.sleep(60)
    eod_status=_download_recent_eod(store,args.lookback_days)
    delivery_status=_download_recent_delivery(store,args.delivery_lookback_days)
    from colab_build_all_nse_delivery_multifactor_candidate import make_panel,merge_optional
    from delivery_training_universe import training_roster,load_official_nse_universe
    from multifactor_data_layer_v2 import FEATURES_59,attach_readiness_flags
    from nse_delivery_data_layer import load_delivery_history
    today=datetime.now(timezone.utc).date()

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

    bulk=_load_eod(EOD_LOCAL_ROOT,today-timedelta(days=args.lookback_days),today)
    roster=training_roster(bulk,min_history=MIN_HISTORY,force_refresh=True)
    trainable=list(roster["trainable_symbols"])
    if not trainable: raise RuntimeError("No current NSE symbols meet the 60-session clean-history gate")
    delivery=load_delivery_history(DELIVERY_LOCAL,symbols=trainable,start=bulk["date"].min(),end=bulk["date"].max())
    panel=make_panel(bulk,trainable,sort_output=False,delivery=delivery)
    panel,source_audit=merge_optional(panel)

    # Distinguish "source is present and this symbol has no applicable item"
    # from "the source pipeline itself is missing". The former is known state
    # (0); the latter remains NaN and is blocked by the production gate.
    source_to_flag = {
        "news": "ready_news",
        "events": "ready_events",
        "fno": "ready_fno",
        "fundamentals": "ready_fundamentals",
    }
    for source, flag in source_to_flag.items():
        if source_status.get(source) in {"LOCAL", "DOWNLOADED"} and flag in panel.columns:
            panel[flag] = pd.to_numeric(panel[flag], errors="coerce").fillna(0).astype("Int8")

    panel=attach_readiness_flags(panel)
    panel["decision_timestamp"]=pd.to_datetime(panel["decision_timestamp"],errors="coerce",utc=True)
    panel=panel.dropna(subset=["symbol","decision_timestamp"]).sort_values(["symbol","decision_timestamp"]).drop_duplicates("symbol",keep="last").reset_index(drop=True)
    missing=[c for c in FEATURES_59 if c not in panel.columns]
    if missing: raise RuntimeError(f"59-feature production panel missing columns: {missing}")
    readiness_cols=["ready_market_price","ready_market_context","ready_news","ready_events","ready_fno","ready_fundamentals"]
    panel["production_data_quality_score"]=panel[readiness_cols].apply(pd.to_numeric,errors="coerce").notna().sum(axis=1)/len(readiness_cols)
    panel["production_core_ready"]=(
        pd.to_numeric(panel["ready_market_price"],errors="coerce").eq(1)&
        pd.to_numeric(panel["ready_market_context"],errors="coerce").eq(1)
    )
    panel["production_optional_state_known"]=panel[["ready_news","ready_events","ready_fno","ready_fundamentals"]].apply(pd.to_numeric,errors="coerce").notna().all(axis=1)
    panel["production_data_quality_gate"]=panel["production_core_ready"]&panel["production_optional_state_known"]
    latest_date=pd.to_datetime(panel["decision_timestamp"],utc=True).dt.tz_convert("Asia/Kolkata").dt.date.max()
    snap=FEATURE_DIR/f"DELIVERY_PIT_FEATURES_{latest_date}.parquet"
    snap.parent.mkdir(parents=True,exist_ok=True)
    panel.to_parquet(snap,index=False)
    group_ready={}
    for c in readiness_cols:
        s=pd.to_numeric(panel[c],errors="coerce")
        group_ready[c]={"known_pct":round(float(s.notna().mean()*100),3),"ready_pct":round(float(s.eq(1).mean()*100),3),"unknown_rows":int(s.isna().sum())}
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
        "production_data_quality_gate_pct":round(float(panel["production_data_quality_gate"].mean()*100),3),
        "ipo_policy":"Any current official NSE equity becomes eligible automatically after 60 clean sessions.",
        "model_weights_changed":False,"model_promotion_performed":False
    }
    report_path=FEATURE_DIR/f"DELIVERY_PIT_READINESS_{latest_date}.json"
    report_path.write_text(json.dumps(report,indent=2,default=str),encoding="utf-8")
    store.upload_or_update(snap,f"features/daily/{snap.name}",mime_type="application/octet-stream")
    store.upload_or_update(report_path,f"features/daily/{report_path.name}",mime_type="application/json")
    if live_news_path.exists():
        store.upload_or_update(live_news_path,"processed/live_news/news_latest.parquet",mime_type="application/octet-stream")
    print(json.dumps(report,indent=2,default=str))
if __name__=="__main__":
    main()
