"""Grouped command launcher for rebuilds utilities.

Usage: python -m tools.rebuilds <command> [original arguments...]

This first-stage consolidation centralizes discovery and invocation while
keeping the original scripts as the executable source of truth. That preserves
workflow, test, and operator entry-point compatibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_COMMANDS = {
    "rebuild_all_nse_intraday_5m_model_from_drive": "rebuild_all_nse_intraday_5m_model_from_drive.py",
    "rebuild_nse_cash_history_identity_aware": "rebuild_nse_cash_history_identity_aware.py",
    "rebuild_nse_cash_history_identity_aware_parallel": "rebuild_nse_cash_history_identity_aware_parallel.py",
    "refresh_canonical_cash_master_from_bulk_eod": "refresh_canonical_cash_master_from_bulk_eod.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rebuilds command group")
    parser.add_argument("command", choices=sorted(_COMMANDS))
    args, forwarded = parser.parse_known_args(argv)
    script = _REPO_ROOT / _COMMANDS[args.command]
    if not script.is_file():
        parser.error(f"script not found: {script}")
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
