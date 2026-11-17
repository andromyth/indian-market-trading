"""Grouped command launcher for backfills utilities.

Usage: python -m tools.backfills <command> [original arguments...]

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
    "backfill_missing_delivery_pit_snapshots": "backfill_missing_delivery_pit_snapshots.py",
    "backfill_missing_worker_actuals": "backfill_missing_worker_actuals.py",
    "dhan_backfill_all_nse_5m_history": "dhan_backfill_all_nse_5m_history.py",
    "indstocks_backfill_all_nse_5m_history": "indstocks_backfill_all_nse_5m_history.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="backfills command group")
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
