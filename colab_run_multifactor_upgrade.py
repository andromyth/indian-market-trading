#!/usr/bin/env python3
"""One-command Google Colab runner for the multi-factor candidate."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",default="/content/indian-market-trading")
    ap.add_argument("--drive-root",default="/content/drive/MyDrive/Indian_Market_Research")
    ap.add_argument("--bulk-root",default=None,help="Canonical NSE bulk EOD archive. Defaults to the production_data archive, with the older processed path as fallback.")
    a=ap.parse_args()
    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
    except Exception:
        pass
    p=Path(a.project).resolve()
    d=Path(a.drive_root).resolve()
    if not p.exists():
        raise SystemExit(f"Project not found: {p}")

    if a.bulk_root:
        bulk=Path(a.bulk_root).resolve()
    else:
        candidates=[
            d/"production_data/nse_official/bulk_eod",
            d/"processed/universe_expansion_2026_v4/bulk_eod",
        ]
        bulk=next((x for x in candidates if x.exists()),candidates[0])
    if not bulk.exists():
        raise SystemExit(
            "NSE bulk EOD archive not found. Checked:\n  "
            + "\n  ".join(str(x) for x in (
                d/"production_data/nse_official/bulk_eod",
                d/"processed/universe_expansion_2026_v4/bulk_eod",
            ))
        )

    out=d/"models/all_nse_delivery_multifactor_candidate_2026"
    subprocess.run(
        [sys.executable,"colab_build_all_nse_delivery_multifactor_candidate.py",
         "--bulk-root",str(bulk),"--out",str(out)],
        cwd=p,check=True
    )
    subprocess.run(
        [sys.executable,"validate_multifactor_candidate.py","--model-dir",str(out)],
        cwd=p,check=True
    )
    print("COLAB MULTIFACTOR CANDIDATE COMPLETE")
    print("NSE bulk EOD:",bulk)
    print("Candidate:",out)
    print("Production model: UNCHANGED")
    return 0

if __name__=="__main__":
    main()
