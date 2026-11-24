"""Grouped command launcher for validators utilities.

Usage: python -m tools.validators <command> [original arguments...]

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
    "validate_canonical_cash_research_view": "validate_canonical_cash_research_view.py",
    "validate_delivery_prediction_ledger": "validate_delivery_prediction_ledger.py",
    "validate_identity_preserved_cash_master": "validate_identity_preserved_cash_master.py",
    "validate_intraday_session_close_predictions": "validate_intraday_session_close_predictions.py",
    "validate_multifactor_candidate": "validate_multifactor_candidate.py",
    "validate_prediction_ledger": "validate_prediction_ledger.py",
    "verify_minute_prediction_persistence": "verify_minute_prediction_persistence.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validators command group")
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
