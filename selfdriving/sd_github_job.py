"""Thin GitHub Actions runner: sync Drive <-> /tmp, run daily stages, write status.json.

Retraining is NOT run here (CPU-limited). It is triggered from Colab (see COLAB_SD_RETRAIN.py)
weekly, or immediately when validate_stage sets retrain=True in status_validate.json.
Pulling/pushing files uses the repo's existing GoogleDriveStore (persistent_store.py).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path("/tmp/Indian_Market_Research/selfdriving")
FILES = ["global_cues.parquet", "raw_news.parquet", "ledger/predictions.parquet", "status_validate.json"]


def _drive():
    from persistent_store import GoogleDriveStore
    return GoogleDriveStore()


def pull(g) -> None:
    from googleapiclient.http import MediaIoBaseDownload
    for rel in FILES + ["models/registry.json"]:
        remote = f"selfdriving/{rel}"
        parts = remote.split("/")
        parent = g.ensure_path("/".join(parts[:-1]))
        q = f"name = '{parts[-1]}' and '{parent}' in parents and trashed = false"
        hits = g.service.files().list(q=q, spaces="drive", fields="files(id)", pageSize=2).execute().get("files", [])
        if not hits:
            continue
        local = ROOT / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        with local.open("wb") as fh:
            d = MediaIoBaseDownload(fh, g.service.files().get_media(fileId=hits[0]["id"]))
            done = False
            while not done:
                _, done = d.next_chunk()


def push(g) -> None:
    for rel in FILES:
        p = ROOT / rel
        if p.exists():
            g.upload_or_update(p, f"selfdriving/{rel}", mime_type="application/octet-stream")


def main() -> None:
    status = {"started": datetime.now(timezone.utc).isoformat(), "stages": {}}
    g = _drive()
    pull(g)

    # 1. global cues, incremental (re-fetch a 10-day overlap so revisions are picked up)
    from selfdriving.sd_global_cues import fetch_global_daily
    gp = ROOT / "global_cues.parquet"
    old = pd.read_parquet(gp) if gp.exists() else pd.DataFrame()
    start = (pd.to_datetime(old["date"]).max() - timedelta(days=10)).strftime("%Y-%m-%d") if len(old) else "2018-01-01"
    new = fetch_global_daily(start, (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d"))
    failed = new.attrs.get("failed", {})
    if len(new):
        both = pd.concat([old, new]).drop_duplicates(["key", "date"], keep="last")
        both.to_parquet(gp, index=False)
    status["stages"]["global_cues"] = {"rows": int(len(new)), "failed_series": failed}

    # 2. headline archive (forward only)
    from selfdriving.sd_news_collect import build_alias_map, collect
    try:
        from nse_data_layer import fetch_nse_equity_universe  # repo function, downloads NSE EQUITY_L.csv
        universe = fetch_nse_equity_universe(force_refresh=True)
        status["stages"]["news"] = collect(build_alias_map(universe), ROOT / "raw_news.parquet")
    except Exception as e:
        status["stages"]["news"] = {"error": f"{type(e).__name__}: {e}"}

    # 3/4. predictions and live validation need the inference panel and realised returns that the
    # existing builders produce; wire them in per INTEGRATION.md, then call predict_stage/validate_stage.
    status["stages"]["predict_validate"] = {"skipped": "inference panel path not configured yet"}

    status["finished"] = datetime.now(timezone.utc).isoformat()
    (ROOT / "status_cycle.json").write_text(json.dumps(status, indent=2, default=str))
    push(g)
    g.upload_or_update(ROOT / "status_cycle.json", "selfdriving/status_cycle.json", mime_type="application/json")
    print(json.dumps(status, indent=2, default=str))
    if len(failed) > len(__import__("selfdriving.sd_global_cues", fromlist=["GLOBAL_SERIES"]).GLOBAL_SERIES) // 2:
        raise SystemExit("More than half of the global series failed; failing the job so it is visible")


if __name__ == "__main__":
    main()
