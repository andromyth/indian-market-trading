#!/usr/bin/env python3
"""Colab launcher for the ALL-NSE F&O repair and 2026 derived-layer rebuild."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from google.colab import drive

REPO = Path("/content/indian-market-trading")
BRANCH = "beta_version"
DRIVE_ROOT = Path("/content/drive/MyDrive/Indian_Market_Research")


def run(cmd: list[str], env=None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


if not Path("/content/drive/MyDrive").exists():
    drive.mount("/content/drive")

if not DRIVE_ROOT.exists():
    raise SystemExit(f"Google Drive research root not found: {DRIVE_ROOT}")

if not REPO.exists():
    run([
        "git", "clone", "--branch", BRANCH, "--single-branch",
        "https://github.com/andromyth/indian-market-trading.git", str(REPO)
    ])
else:
    run(["git", "-C", str(REPO), "fetch", "origin", BRANCH])
    run(["git", "-C", str(REPO), "checkout", BRANCH])
    run(["git", "-C", str(REPO), "pull", "--ff-only", "origin", BRANCH])

env = os.environ.copy()
env["INDIAN_MARKET_RESEARCH_ROOT"] = str(DRIVE_ROOT)

run([sys.executable, "-m", "py_compile", str(REPO / "nse_fno_udiff_ingest.py")], env=env)
run([sys.executable, "-m", "py_compile", str(REPO / "nse_fno_applicability_snapshot.py")], env=env)

run([sys.executable, str(REPO / "nse_fno_applicability_snapshot.py")], env=env)

print("\nREBUILDING 2026 F&O DERIVED FEATURES FROM PERSISTED RAW ARCHIVES", flush=True)
run([
    sys.executable, "-u", str(REPO / "nse_fno_udiff_ingest.py"),
    "--rebuild-derived-2026",
], env=env)

print("\nF&O REPAIR COMPLETE", flush=True)
print("Historical F&O derived layer rebuilt from NSE UDiFF raw archives.", flush=True)
print("Current F&O applicability snapshot persisted separately.", flush=True)
print("No model training or promotion performed.", flush=True)
