"""Autonomous Beta V2 daily predict/validate cycle.

This job is isolated from the current production models. It reads the newest
official NSE Delivery PIT snapshot, durable Beta V2 training/realised artefacts,
and point-in-time global/news/F&O layers, then writes only under the
selfdriving/beta_version2 Drive prefix.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path("/tmp/Indian_Market_Research/selfdriving/beta_version2")
ROOT.mkdir(parents=True, exist_ok=True)
REMOTE_BASE = "selfdriving/beta_version2"


def drive():
    from persistent_store import GoogleDriveStore
    return GoogleDriveStore()


def download_file(store, remote: str, local: Path) -> bool:
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
    tmp = local.with_suffix(local.suffix + ".tmp")
    with tmp.open("wb") as fh:
        dl = MediaIoBaseDownload(
            fh,
            store.service.files().get_media(fileId=hit["id"]),
        )
        done = False
        while not done:
            _, done = dl.next_chunk()
    tmp.replace(local)
    return True


def upload(store, local: Path, remote: str,
           mime: str = "application/octet-stream") -> None:
    store.upload_or_update(local, remote, mime_type=mime)


def list_pit_snapshot_items(store) -> list[dict]:
    """List PIT snapshots and sort by the date encoded in the canonical filename."""
    from selfdriving.sd_data_quality import parse_pit_date

    parent = store.root_id
    for part in ("features", "daily"):
        hit = store._find_child(parent, part, "application/vnd.google-apps.folder")
        if not hit:
            return []
        parent = hit["id"]

    items = []
    token = None
    while True:
        kwargs = {
            "q": (
                f"'{parent}' in parents and "
                "name contains 'DELIVERY_PIT_FEATURES_' and trashed = false"
            ),
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,modifiedTime,size)",
            "pageSize": 1000,
            "orderBy": "name desc",
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if token:
            kwargs["pageToken"] = token
        response = store.service.files().list(**kwargs).execute()
        for item in response.get("files", []):
            name = str(item.get("name", ""))
            day = parse_pit_date(name)
            if day is not None and name.endswith(".parquet"):
                items.append({**item, "snapshot_date": day})
        token = response.get("nextPageToken")
        if not token:
            break

    unique = {item["name"]: item for item in items}
    return sorted(unique.values(), key=lambda item: item["snapshot_date"])


def pull_recent_pit_snapshots(store, limit: int = 20) -> list[tuple[date, Path]]:
    """Download a bounded recent sequence of PIT snapshots for labels and audits."""
    items = list_pit_snapshot_items(store)[-max(1, int(limit)):]
    snapshots: list[tuple[date, Path]] = []
    for item in items:
        local = ROOT / "pit_history" / str(item["name"])
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists() or local.stat().st_size == 0:
            from googleapiclient.http import MediaIoBaseDownload

            temp = local.with_suffix(local.suffix + ".tmp")
            try:
                with temp.open("wb") as fh:
                    downloader = MediaIoBaseDownload(
                        fh, store.service.files().get_media(fileId=item["id"])
                    )
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                if temp.stat().st_size == 0:
                    raise RuntimeError(f"Empty Drive PIT download: {item['name']}")
                temp.replace(local)
            finally:
                if temp.exists():
                    temp.unlink()
        snapshots.append((item["snapshot_date"], local))
    return snapshots


def pull_latest_pit_snapshot(store) -> Path | None:
    snapshots = pull_recent_pit_snapshots(store, limit=1)
    return snapshots[-1][1] if snapshots else None

def pull_sources(store) -> dict:
    sources = {
        "global_cues": (
            f"{REMOTE_BASE}/global_cues.parquet",
            ROOT / "global_cues.parquet",
        ),
        "raw_news": (
            f"{REMOTE_BASE}/raw_news.parquet",
            ROOT / "raw_news.parquet",
        ),
        "fno_daily": (
            "production_data/nse_official/fno/fno_daily_features_2026.parquet",
            ROOT / "fno_daily_features_2026.parquet",
        ),
        "realised": (
            f"{REMOTE_BASE}/BETA_V2_DELIVERY_REALISED_RETURNS.parquet",
            ROOT / "BETA_V2_DELIVERY_REALISED_RETURNS.parquet",
        ),
        "registry": (
            f"{REMOTE_BASE}/models/registry.json",
            ROOT / "models/registry.json",
        ),
        "ledger": (
            f"{REMOTE_BASE}/ledger/predictions.parquet",
            ROOT / "ledger/predictions.parquet",
        ),
    }

    result = {}
    for key, (remote, local) in sources.items():
        result[key] = (
            "OK" if download_file(store, remote, local) else "MISSING"
        )

    # The registry points to the actual champion bundle filename. Pull that
    # bundle too; Registry.champion_bundle() will otherwise fail at inference
    # even though registry.json itself is present.
    registry_path = ROOT / "models" / "registry.json"
    if registry_path.exists():
        try:
            state = json.loads(registry_path.read_text(encoding="utf-8"))
            champion = state.get("champion") or {}
            bundle_name = str(champion.get("file", "")).strip()
            if bundle_name:
                result["champion_bundle"] = (
                    "OK"
                    if download_file(
                        store,
                        f"{REMOTE_BASE}/models/{bundle_name}",
                        ROOT / "models" / bundle_name,
                    )
                    else "MISSING"
                )
            else:
                result["champion_bundle"] = "NO_CHAMPION"
        except Exception as exc:
            result["champion_bundle"] = f"ERROR:{type(exc).__name__}"

    snapshots = pull_recent_pit_snapshots(store, limit=20)
    latest_pit = snapshots[-1][1] if snapshots else None
    result["current_pit"] = "OK" if latest_pit else "MISSING"
    result["pit_snapshots_downloaded"] = len(snapshots)
    return result, latest_pit, snapshots


def refresh_global(root: Path) -> dict:
    from selfdriving.sd_global_cues import fetch_global_daily

    path = root / "global_cues.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    start = (
        (
            pd.to_datetime(old["date"]).max()
            - pd.Timedelta(days=10)
        ).strftime("%Y-%m-%d")
        if len(old)
        else "2018-01-01"
    )

    fresh = fetch_global_daily(
        start,
        (datetime.now(timezone.utc) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if len(fresh):
        merged = (
            pd.concat([old, fresh], ignore_index=True)
            .drop_duplicates(["key", "date"], keep="last")
            .sort_values(["key", "date"])
        )
        merged.to_parquet(path, index=False)

    return {
        "rows": int(len(fresh)),
        "failed_series": fresh.attrs.get("failed", {}),
    }


def refresh_news(root: Path) -> dict:
    from selfdriving.sd_news_collect import build_alias_map, collect
    from nse_data_layer import fetch_nse_equity_universe

    path = root / "raw_news.parquet"
    universe = fetch_nse_equity_universe(force_refresh=False)
    return collect(build_alias_map(universe), path)


def refresh_realised_outcomes(store, snapshots: list[tuple[date, Path]]) -> dict:
    """Update next-session realised returns from adjacent official PIT snapshots."""
    from selfdriving.sd_data_quality import derive_realised_returns, merge_realised_returns

    realised_path = ROOT / "BETA_V2_DELIVERY_REALISED_RETURNS.parquet"
    old = pd.read_parquet(realised_path) if realised_path.exists() and realised_path.stat().st_size else pd.DataFrame(
        columns=["symbol", "date", "y_return"]
    )
    frames = []
    for day, path in snapshots:
        frame = pd.read_parquet(path)
        if "date" in frame.columns:
            observed = pd.to_datetime(frame["date"], errors="coerce").dt.date.dropna()
            if len(observed) and observed.nunique() != 1:
                raise RuntimeError(
                    f"PIT snapshot {path.name} contains multiple source dates: {sorted(map(str, observed.unique()))[:5]}"
                )
            if len(observed) and observed.iloc[0] != day:
                raise RuntimeError(
                    f"PIT snapshot date mismatch: filename={day}, parquet={observed.iloc[0]}"
                )
        frames.append((day, frame))

    fresh, gaps = derive_realised_returns(frames)
    merged, merge_stats = merge_realised_returns(old, fresh)
    if len(merged) or not realised_path.exists():
        realised_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(realised_path, index=False)

    merge_stats.update({
        "snapshot_count": len(frames),
        "snapshot_dates": [day.isoformat() for day, _ in frames],
        "session_gaps_skipped": gaps,
        "outcome_source": "consecutive_official_NSE_Delivery_PIT_snapshots",
    })
    return merge_stats


def _current_expected_reference_date() -> date | None:
    from production_data_freshness_gate import expected_session_date
    from selfdriving.sd_data_quality import is_nse_session_day

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    # The challenger is a pre-open prediction job. On a weekend or NSE holiday
    # there is no session to predict, so do not interpret the prior close as a
    # current-session prediction simply because it is the newest saved PIT.
    if not is_nse_session_day(now_ist.date()):
        return None
    return expected_session_date(now_ist)


def _load_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def main() -> int:
    status = {
        "version": "2026-10-10-beta-v2-daily",
        "started": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "production_promotion": False,
        "production_artifacts_changed": False,
    }

    store = drive()
    pull_status, latest_pit, pit_snapshots = pull_sources(store)
    status["stages"]["pull"] = pull_status

    # Actual outcomes are updated from official, adjacent NSE PIT sessions before
    # the validation stage. A missing session is skipped; multi-session returns
    # are never mislabeled as next-session outcomes.
    try:
        outcome_status = refresh_realised_outcomes(store, pit_snapshots)
        status["stages"]["realised_outcomes"] = {"status": "PASS", **outcome_status}
    except Exception as exc:
        status["stages"]["realised_outcomes"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        status["stages"]["global_cues"] = refresh_global(ROOT)
    except Exception as exc:
        status["stages"]["global_cues"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    try:
        status["stages"]["news"] = refresh_news(ROOT)
    except Exception as exc:
        status["stages"]["news"] = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    from selfdriving.sd_data_quality import build_input_audit, parse_pit_date

    expected_date = _current_expected_reference_date()
    observed_date = parse_pit_date(latest_pit.name) if latest_pit else None
    freshness_ok = expected_date is not None and observed_date == expected_date
    if expected_date is None:
        status["stages"]["snapshot_freshness"] = {
            "status": "SKIPPED",
            "expected_session_date": None,
            "observed_snapshot_date": observed_date.isoformat() if observed_date else None,
            "reason": "No NSE reference session expected at the current clock time.",
        }
    else:
        status["stages"]["snapshot_freshness"] = {
            "status": "PASS" if freshness_ok else "ERROR",
            "expected_session_date": expected_date.isoformat(),
            "observed_snapshot_date": observed_date.isoformat() if observed_date else None,
            "reason": None if freshness_ok else "Refusing to score a stale or missing PIT snapshot.",
        }

    pit = _load_frame(latest_pit) if latest_pit else pd.DataFrame()
    global_cues = _load_frame(ROOT / "global_cues.parquet")
    news = _load_frame(ROOT / "raw_news.parquet")
    fno = _load_frame(ROOT / "fno_daily_features_2026.parquet")
    realised_path = ROOT / "BETA_V2_DELIVERY_REALISED_RETURNS.parquet"
    realised = _load_frame(realised_path)
    registry_path = ROOT / "models"
    ledger_path = ROOT / "ledger" / "predictions.parquet"

    panel = pd.DataFrame()
    panel_audit = {}
    if not pit.empty:
        try:
            from selfdriving.sd_v2_panel import build_beta_delivery_panel, split_training_inference

            panel, panel_audit = build_beta_delivery_panel(
                pit,
                global_cues=global_cues,
                raw_news=news,
                fno_daily=fno,
            )
        except Exception as exc:
            status["stages"]["panel_build"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        status["stages"]["panel_build"] = {
            "status": "ERROR",
            "error": "No PIT snapshot was available to audit or score.",
        }

    active_champion_features = []
    active_champion_status = "NO_CHAMPION"
    active_champion_file = None
    registry_file = registry_path / "registry.json"
    if registry_file.exists() and registry_file.stat().st_size:
        try:
            registry_state = json.loads(registry_file.read_text(encoding="utf-8"))
            champion = registry_state.get("champion") or {}
            active_champion_file = str(champion.get("file", "")).strip() or None
            if active_champion_file:
                bundle_path = registry_path / active_champion_file
                if not bundle_path.exists() or not bundle_path.stat().st_size:
                    active_champion_status = "MISSING_BUNDLE"
                else:
                    import joblib
                    bundle = joblib.load(bundle_path)
                    active_champion_features = list(getattr(bundle, "features", []) or [])
                    active_champion_status = "READY" if active_champion_features else "EMPTY_FEATURE_CONTRACT"
            else:
                active_champion_status = "NO_CHAMPION"
        except Exception as exc:
            active_champion_status = f"LOAD_ERROR:{type(exc).__name__}"
            status["stages"]["active_model_contract"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }

    try:
        audit_report = build_input_audit(
            pit,
            panel,
            global_cues,
            news,
            fno,
            expected_date=expected_date,
            observed_date=observed_date,
            realised_update=status["stages"].get("realised_outcomes", {}),
            source_status={
                "drive_sources": pull_status,
                "global_refresh": status["stages"].get("global_cues", {}),
                "news_refresh": status["stages"].get("news", {}),
            },
            active_champion_features=active_champion_features,
            active_champion_status=active_champion_status,
            active_champion_file=active_champion_file,
        )
        audit_status = "ERROR" if audit_report["hard_failures"] else (
            "WARN" if audit_report["warnings"] else "PASS"
        )
        status["stages"]["data_quality_audit"] = {
            "status": audit_status,
            "report": audit_report,
        }
    except Exception as exc:
        audit_report = {
            "overall_status": "FAIL",
            "hard_failures": ["AUDIT_EXECUTION_FAILED"],
            "error": f"{type(exc).__name__}: {exc}",
        }
        status["stages"]["data_quality_audit"] = {
            "status": "ERROR",
            "report": audit_report,
        }

    inference = pd.DataFrame()
    if latest_pit is None:
        status["stages"]["inference"] = {
            "status": "BLOCKED",
            "reason": "No official NSE Delivery PIT snapshot found in Drive.",
        }
    elif expected_date is None:
        status["stages"]["inference"] = {
            "status": "BLOCKED",
            "reason": "No NSE reference session is expected now; no signal was generated.",
        }
    elif not freshness_ok:
        status["stages"]["inference"] = {
            "status": "BLOCKED",
            "reason": (
                f"Stale PIT snapshot: expected {expected_date}, "
                f"observed {observed_date}. No signal was generated."
            ),
        }
    elif status["stages"].get("data_quality_audit", {}).get("status") == "ERROR":
        status["stages"]["inference"] = {
            "status": "BLOCKED",
            "reason": "PIT failed the hard data-quality checks; no signal was generated.",
        }
    elif not panel.empty:
        try:
            from selfdriving.sd_cycle import predict_stage

            _, inference, _ = split_training_inference(panel)
            inference_path = ROOT / "BETA_V2_DELIVERY_INFERENCE_PANEL.parquet"
            inference_path.parent.mkdir(parents=True, exist_ok=True)
            inference.to_parquet(inference_path, index=False)

            reg_file = registry_path / "registry.json"
            registry_state = {}
            if reg_file.exists():
                try:
                    registry_state = json.loads(reg_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise RuntimeError(
                        f"Invalid Beta V2 registry: {type(exc).__name__}: {exc}"
                    ) from exc

            if not reg_file.exists() or not registry_state.get("champion"):
                status["stages"]["inference"] = {
                    "status": "NO_CHAMPION",
                    "reason": (
                        "No validated Beta V2 champion is registered. "
                        "Data collection and audit completed; prediction remains abstained."
                    ),
                    "audit": panel_audit,
                }
            else:
                predictions = predict_stage(inference, registry_path)

                # The first prediction for a symbol/date is immutable. A later
                # rerun may enrich the audit, but cannot rewrite the issued signal.
                ledger_path.parent.mkdir(parents=True, exist_ok=True)
                if ledger_path.exists() and ledger_path.stat().st_size:
                    old_ledger = pd.read_parquet(ledger_path)
                    for frame in (old_ledger, predictions):
                        if "date" in frame.columns:
                            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
                    predictions = (
                        pd.concat([old_ledger, predictions], ignore_index=True)
                        .drop_duplicates(["symbol", "date"], keep="first")
                        .sort_values(["date", "symbol"])
                    )
                predictions.to_parquet(ledger_path, index=False)

                status["stages"]["inference"] = {
                    "status": "PASS",
                    "audit": panel_audit,
                    "rows_latest": int(
                        predictions[predictions["date"].eq(predictions["date"].max())].shape[0]
                    ) if len(predictions) else 0,
                    "rows_ledger": int(len(predictions)),
                    "signals_latest": (
                        predictions[predictions["date"].eq(predictions["date"].max())]["signal"].value_counts().to_dict()
                        if len(predictions) else {}
                    ),
                    "prediction_reference_date": str(predictions["date"].max()) if len(predictions) else None,
                    "signals_are_decision_support_only": True,
                }

        except Exception as exc:
            status["stages"]["inference"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }

    # Validate the stored prediction ledger against the updated official
    # realised-return table even when today's inference is blocked or no
    # champion is registered. This is monitoring only; it never promotes a model.
    if ledger_path.exists() and ledger_path.stat().st_size and not realised.empty:
        try:
            from selfdriving.sd_cycle import validate_stage

            status["stages"]["validation"] = validate_stage(
                pd.read_parquet(ledger_path),
                realised,
                registry_path,
            )
            status["stages"]["validation"]["realised_rows"] = int(len(realised))
        except Exception as exc:
            status["stages"]["validation"] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
    else:
        status["stages"]["validation"] = {
            "status": "WAITING_FOR_PREDICTIONS_OR_REALISED_RETURNS",
            "ledger_exists": ledger_path.exists(),
            "realised_rows": int(len(realised)),
        }

    persistence_errors = []

    def safe_upload(path: Path, remote: str, mime: str = "application/octet-stream") -> None:
        if not path.exists() or path.stat().st_size == 0:
            return
        try:
            upload(store, path, remote, mime)
        except Exception as exc:
            persistence_errors.append(
                f"{remote}: {type(exc).__name__}: {exc}"
            )

    safe_upload(ROOT / "global_cues.parquet", f"{REMOTE_BASE}/global_cues.parquet")
    safe_upload(ROOT / "raw_news.parquet", f"{REMOTE_BASE}/raw_news.parquet")
    safe_upload(realised_path, f"{REMOTE_BASE}/BETA_V2_DELIVERY_REALISED_RETURNS.parquet")
    safe_upload(ledger_path, f"{REMOTE_BASE}/ledger/predictions.parquet")
    safe_upload(
        ROOT / "BETA_V2_DELIVERY_INFERENCE_PANEL.parquet",
        f"{REMOTE_BASE}/BETA_V2_DELIVERY_INFERENCE_PANEL.parquet",
    )

    audit_path = ROOT / "status_data_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_report, indent=2, default=str), encoding="utf-8")
    safe_upload(audit_path, f"{REMOTE_BASE}/status_data_audit.json", "application/json")

    runtime_errors = []
    for stage_name, stage in status["stages"].items():
        if isinstance(stage, dict) and stage.get("status") == "ERROR":
            runtime_errors.append(f"{stage_name}: {stage.get('error', stage.get('reason', 'failed'))}")
    runtime_errors.extend(persistence_errors)
    status["runtime_errors"] = runtime_errors
    status["persistence_errors"] = persistence_errors
    status["finished"] = datetime.now(timezone.utc).isoformat()

    status_path = ROOT / "status_cycle.json"
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    try:
        upload(store, status_path, f"{REMOTE_BASE}/status_cycle.json", "application/json")
    except Exception as exc:
        persistence_errors.append(
            f"{REMOTE_BASE}/status_cycle.json: {type(exc).__name__}: {exc}"
        )
        status["persistence_errors"] = persistence_errors
        status["runtime_errors"] = runtime_errors + [persistence_errors[-1]]

    # Retain the final local report even if uploading status_cycle.json itself failed.
    # The GitHub Actions workflow can attach this local copy as a run artifact.
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    print(json.dumps(status, indent=2, default=str))
    return 1 if status["runtime_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
