#!/usr/bin/env python3
"""One-command Google Colab runner for the multi-factor candidate."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="/content/indian-market-trading")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument("--bulk-root", default=None, help="Canonical NSE bulk EOD archive.")
    ap.add_argument("--refresh-fundamentals", action="store_true", help="Force a fresh NSE fundamentals catalog/XBRL pass.")
    ap.add_argument("--retry-failed-fundamentals", action="store_true", help="Retry only filings previously cached as failed.")
    ap.add_argument("--force-rebuild-candidate", action="store_true", help="Force panel/model regeneration even when inputs are unchanged.")
    ap.add_argument("--preflight", action="store_true", help="Check code contract and Drive caches without running the build.")
    a = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    log("COLAB MULTIFACTOR UPGRADE STARTING",)
    log(f"Project: {a.project}")
    log(f"Drive root: {a.drive_root}")
    if a.preflight:
        from multifactor_data_layer_v2 import FEATURES_59
        if len(FEATURES_59) != 59:
            raise SystemExit(f"PRECHECK FAILED: feature count={len(FEATURES_59)}")
        builder_path = Path(a.project) / "colab_build_all_nse_delivery_multifactor_candidate.py"
        builder_text = builder_path.read_text(encoding="utf-8")
        required = [
            "2026-all-nse-delivery-multifactor-59-candidate-v2",
            "pit_fundamentals_filing_cache_2025_2026_v3",
            "CANDIDATE_BUILD_META_2026.json",
        ]
        missing = [x for x in required if x not in builder_text]
        if missing:
            raise SystemExit("PRECHECK FAILED: stale builder; missing=" + ",".join(missing))
        log("PRECHECK OK: feature contract=59; incremental-cache builder present.")
        return 0

    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            log("Mounting Google Drive...")
            drive.mount("/content/drive")
    except Exception as exc:
        log(f"Drive mount check: {type(exc).__name__}: {exc}")

    p = Path(a.project).resolve()
    d = Path(a.drive_root).resolve()
    if not p.exists():
        raise SystemExit(f"Project not found: {p}")

    if a.bulk_root:
        bulk = Path(a.bulk_root).resolve()
    else:
        candidates = [
            d / "production_data/nse_official/bulk_eod",
            d / "processed/universe_expansion_2026_v4/bulk_eod",
        ]
        bulk = next((x for x in candidates if x.exists()), candidates[0])

    if not bulk.exists():
        raise SystemExit(
            "NSE bulk EOD archive not found. Checked:\n  "
            + "\n  ".join(str(x) for x in (
                d / "production_data/nse_official/bulk_eod",
                d / "processed/universe_expansion_2026_v4/bulk_eod",
            ))
        )

    out = d / "models/all_nse_delivery_multifactor_candidate_2026"
    log(f"NSE bulk archive: {bulk}")
    log(f"Candidate output: {out}")
    log("Step 1/2: building ALL-NSE 59-feature candidate...")
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    build_cmd = [
        sys.executable, "-u", "colab_build_all_nse_delivery_multifactor_candidate.py",
        "--bulk-root", str(bulk), "--out", str(out),
    ]
    if a.refresh_fundamentals:
        build_cmd.append("--refresh-fundamentals")
    if a.retry_failed_fundamentals:
        build_cmd.append("--retry-failed-fundamentals")
    if a.force_rebuild_candidate:
        build_cmd.append("--force-rebuild-candidate")
    subprocess.run(build_cmd, cwd=p, check=True, env=env)

    log("Step 2/2: validating candidate artifacts...")
    validate_cmd = [
        sys.executable, "-u", "validate_multifactor_candidate.py",
        "--model-dir", str(out),
    ]
    subprocess.run(validate_cmd, cwd=p, check=True, env=env)

    log("COLAB MULTIFACTOR CANDIDATE COMPLETE")
    log(f"NSE bulk EOD: {bulk}")
    log(f"Candidate: {out}")
    log("Production model: UNCHANGED")
    return 0

if __name__ == "__main__":
    main()
