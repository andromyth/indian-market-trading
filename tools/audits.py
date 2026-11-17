"""Grouped command launcher for audits utilities.

Usage: python -m tools.audits <command> [original arguments...]

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
    "audit": "audit.py",
    "audit_5m_prediction_dataset": "audit_5m_prediction_dataset.py",
    "audit_all_universe_minute_snapshots": "audit_all_universe_minute_snapshots.py",
    "audit_autonomous_intraday_outcomes": "audit_autonomous_intraday_outcomes.py",
    "audit_beta_v2_intraday_preflight": "audit_beta_v2_intraday_preflight.py",
    "audit_beta_v2_intraday_session_validation": "audit_beta_v2_intraday_session_validation.py",
    "audit_full_multimodel_engine": "audit_full_multimodel_engine.py",
    "audit_identity_preserved_duplicates": "audit_identity_preserved_duplicates.py",
    "audit_live_prediction_health": "audit_live_prediction_health.py",
    "cross_sectional_daily_ohlc_audit": "cross_sectional_daily_ohlc_audit.py",
    "delivery_decision_deep_audit": "delivery_decision_deep_audit.py",
    "fno_iv_input_audit": "fno_iv_input_audit.py",
    "hf_staging_quality_audit": "hf_staging_quality_audit.py",
    "historical_audit": "historical_audit.py",
    "nse_eod_archive_audit": "nse_eod_archive_audit.py",
    "rolling_daily_ohlc_audit": "rolling_daily_ohlc_audit.py",
    "strict_delivery_walkforward_audit": "strict_delivery_walkforward_audit.py",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="audits command group")
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
