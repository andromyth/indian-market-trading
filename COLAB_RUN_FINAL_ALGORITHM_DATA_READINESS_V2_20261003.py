#!/usr/bin/env python3
"""Google Colab entrypoint for the repository-native readiness V2 audit.

Mounts Google Drive, restores the consolidated beta_version after a runtime
reset, verifies the 59-feature source, and runs the read-only diagnostic.
No model builder is called.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from google.colab import drive

REPO = Path("/content/indian-market-trading")
BRANCH = "beta_version"
DRIVE_ROOT = Path("/content/drive/MyDrive/Indian_Market_Research")
SCRIPT = REPO / "final_algorithm_data_readiness_v2.py"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


print("=" * 90)
print("COLAB — FINAL ALL-NSE 59-FEATURE READINESS V2")
print("=" * 90)

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

if not SCRIPT.exists():
    raise SystemExit(f"Repository diagnostic missing: {SCRIPT}")

run([sys.executable, "-m", "py_compile", str(SCRIPT)])

import os
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from multifactor_data_layer_v2 import FEATURES_59, GROUPS

print("
SOURCE")
print("branch:", subprocess.check_output(
    ["git", "branch", "--show-current"], text=True
).strip())
print("head:", subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"], text=True
).strip())
print("feature_count:", len(FEATURES_59))
print("groups:", list(GROUPS.keys()))

if len(FEATURES_59) != 59:
    raise SystemExit(f"59-feature contract check failed: {len(FEATURES_59)}")

print("
RUNNING READ-ONLY DIAGNOSTIC")
run([
    sys.executable, "-u", str(SCRIPT),
    "--output-dir", str(DRIVE_ROOT / "diagnostics" / "final_readiness_v2")
])

print("
COMPLETE")
print("Output:", DRIVE_ROOT / "diagnostics" / "final_readiness_v2")
print("NO MODEL TRAINING")
print("NO MODEL PROMOTION")
print("NO DATASET MUTATION")
