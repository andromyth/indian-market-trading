#!/usr/bin/env python3
"""Promote a validated to-close retrain candidate to the live champion.

``retrain_to_close_job.py`` only publishes a *candidate* under
``models/to_close/runs/<run_id>/`` plus a pointer ``models/to_close/latest.json``.
This script performs the separate promotion step:

  1. read the candidate pointer (or an explicit ``--run-id``),
  2. verify the bundle SHA-256 matches the published status,
  3. apply the gate and the three-night hysteresis (a FAIL never moves the pointer),
  4. write ``versions/<v>/bundle.joblib``, read it back and hash-check it,
  5. write ``champion.json`` LAST so serving swaps only to a verified bundle.

The Drive layout is exactly the one ``daily_cycle`` serves, so
``champion_model_api`` picks the new champion up unchanged. ``--dry-run`` prints
the decision and writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone

import joblib

CANDIDATE_PREFIX = "models/to_close"
CHAMPION_PREFIX = "production_data/all_nse_champion"
ACTIVATE_AFTER = 3


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_store(prefix: str):
    import daily_cycle

    return daily_cycle.make_drive_store(prefix)


def _read_json(store, rel: str) -> dict | None:
    raw = store.read(rel)
    return json.loads(raw) if raw else None


def _load_candidate(store, run_id: str | None):
    latest = _read_json(store, "latest.json")
    if latest is None:
        raise RuntimeError(f"{CANDIDATE_PREFIX}/latest.json not found")
    rid = run_id or latest.get("run_id")
    if not rid:
        raise RuntimeError("candidate status has no run_id")
    base = f"runs/{rid}"
    bundle_raw = store.read(f"{base}/candidate_bundle.joblib")
    report_raw = store.read(f"{base}/training_report.json")
    if not bundle_raw or not report_raw:
        raise RuntimeError(f"candidate artifacts missing for run {rid}")
    sha = _sha256(bundle_raw)
    expected = latest.get("candidate_bundle_sha256") if rid == latest.get("run_id") else None
    if expected and expected != sha:
        raise RuntimeError(
            f"candidate bundle sha256 mismatch (status {expected[:12]}, bundle {sha[:12]})"
        )
    report = json.loads(report_raw)
    return rid, latest, bundle_raw, sha, report


def _verify_bundle(bundle_raw: bytes) -> dict:
    bundle = joblib.load(io.BytesIO(bundle_raw))
    if "close" not in bundle:
        raise RuntimeError("candidate bundle has no 'close' model")
    deploy = bundle["close"]
    if not deploy.get("features"):
        raise RuntimeError("candidate bundle has no feature list")
    if deploy.get("thresholds") is None:
        raise RuntimeError("candidate bundle has no thresholds")
    return deploy


def _streak(previous: dict | None, *, trained_through: str, passed: bool, activate_after: int):
    hz_prev = ((previous or {}).get("horizons") or {}).get("close") or {}
    base = hz_prev.get("streak_base") or {} if previous and previous.get("trained_through") == trained_through else hz_prev
    old = {
        "pass_streak": int(base.get("pass_streak", 0) or 0),
        "fail_streak": int(base.get("fail_streak", 0) or 0),
        "active": bool(base.get("active", False)),
    }
    pass_streak = old["pass_streak"] + 1 if passed else 0
    fail_streak = 0 if passed else old["fail_streak"] + 1
    active = bool(passed and (old["active"] or pass_streak >= activate_after))
    return old, pass_streak, fail_streak, active


def _version(champion_store, trained_through: str, sha: str) -> str:
    version = f"v{trained_through}"
    rel = f"versions/{version}/bundle.joblib"
    existing = champion_store.read(rel)
    if existing is not None and _sha256(existing) != sha:
        version = f"{version}-{sha[:8]}"
    return version


def promote(
    candidate_store,
    champion_store,
    *,
    run_id: str | None = None,
    dry_run: bool = False,
    force: bool = False,
    activate_after: int = ACTIVATE_AFTER,
) -> dict:
    rid, latest, bundle_raw, sha, report = _load_candidate(candidate_store, run_id)
    deploy = _verify_bundle(bundle_raw)
    result = report.get("result") or {}
    verdict = result.get("verdict") or latest.get("training_verdict") or "UNKNOWN"
    trained_through = latest.get("trained_through") or latest.get("covered_end")
    if not trained_through:
        raise RuntimeError("candidate status has no trained_through/covered_end; re-run the retrain")
    trained_through = str(trained_through)[:10]

    previous = _read_json(champion_store, "champion.json")
    passed = verdict == "PASS"
    old, pass_streak, fail_streak, active = _streak(
        previous, trained_through=trained_through, passed=passed, activate_after=activate_after
    )
    thresholds = deploy.get("thresholds") or {}
    horizon = {
        "verdict": verdict,
        "pass_streak": pass_streak,
        "fail_streak": fail_streak,
        "active": active,
        "streak_base": old,
        "thresholds": thresholds,
        "holdout_auc": result.get("holdout_auc"),
        "holdout_auc_by_time_left": result.get("holdout_auc_by_time_left"),
        "holdout_trading": result.get("holdout_trading"),
        "gate": result.get("gate") or deploy.get("gate"),
    }

    decision = "DRY_RUN" if dry_run else ("PROMOTE" if passed or force else "KEEP_PREVIOUS")
    if previous and previous.get("bundle_sha256") == sha:
        decision = "ALREADY_PROMOTED"

    version = _version(champion_store, trained_through, sha)
    champion = {
        "version": version,
        "trained_through": trained_through,
        "bundle_path": f"versions/{version}/bundle.joblib",
        "bundle_sha256": sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_module": deploy.get("feature_module", "features_5m"),
        "target": "target_session_close",
        "feature_count": len(deploy.get("features") or []),
        "feature_warmup": deploy.get("feature_warmup"),
        "signal_rule": {
            "use_edge": thresholds.get("min_edge") is not None,
            "min_edge": thresholds.get("min_edge"),
        },
        "sessions_used": latest.get("training_sessions"),
        "symbols_trained": deploy.get("training_symbols") or [],
        "coverage": report.get("coverage"),
        "horizons": {"close": horizon},
        "source_run_id": rid,
    }
    out = {
        "decision": decision,
        "run_id": rid,
        "verdict": verdict,
        "gate": horizon["gate"],
        "trained_through": trained_through,
        "version": version,
        "pass_streak": pass_streak,
        "active": active,
        "bundle_sha256": sha,
        "previous_version": (previous or {}).get("version"),
    }

    if decision in {"DRY_RUN", "KEEP_PREVIOUS", "ALREADY_PROMOTED"}:
        return out

    champion_store.write(champion["bundle_path"], bundle_raw)
    back = champion_store.read(champion["bundle_path"])
    if back is None or _sha256(back) != sha:
        raise RuntimeError("bundle read-back hash mismatch; champion pointer NOT changed")
    joblib.load(io.BytesIO(back))  # must deserialize before the pointer moves
    champion_store.write("champion.json", json.dumps(champion, indent=2, default=str).encode())
    out["champion_path"] = champion["bundle_path"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", help="candidate run id (default: models/to_close/latest.json)")
    ap.add_argument("--dry-run", action="store_true", help="print the decision, write nothing")
    ap.add_argument("--force", action="store_true", help="publish even when the gate verdict is FAIL")
    ap.add_argument("--activate-after", type=int, default=ACTIVATE_AFTER,
                    help="consecutive PASS nights before serving leaves HOLD")
    ap.add_argument("--candidate-prefix", default=CANDIDATE_PREFIX)
    ap.add_argument("--champion-prefix", default=CHAMPION_PREFIX)
    a = ap.parse_args(argv)

    candidate_store = _make_store(a.candidate_prefix)
    champion_store = _make_store(a.champion_prefix)
    try:
        out = promote(candidate_store, champion_store, run_id=a.run_id, dry_run=a.dry_run,
                      force=a.force, activate_after=a.activate_after)
    except Exception as exc:  # noqa: BLE001 - the workflow must see a clean failure
        print("PROMOTE_ERROR=" + json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print("PROMOTE_RESULT=" + json.dumps(out, sort_keys=True, default=str))
    return 0 if out["decision"] != "KEEP_PREVIOUS" else 3


if __name__ == "__main__":
    sys.exit(main())
