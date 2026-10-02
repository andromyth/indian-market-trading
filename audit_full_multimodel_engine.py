#!/usr/bin/env python3
"""Audit the complete Drive-backed multi-model registry.

Run this from the repository root in Colab or the Streamlit runtime.
It performs discovery and contract inspection only; it never changes model
artifacts or the live prediction engine.
"""
from __future__ import annotations

import json

from multi_model_engine import FullMultiModelRegistry


def main() -> int:
    registry = FullMultiModelRegistry()
    report = registry_audit(registry)

    print("=" * 80)
    print("FULL MULTI-MODEL ENGINE — ARTIFACT / CONTRACT AUDIT")
    print("=" * 80)

    print(
        "Registered auxiliary artifacts :",
        report["registered_auxiliary_models"],
    )

    print("\nBY HORIZON")
    for horizon, count in sorted(
        report["by_horizon"].items()
    ):
        print(
            f"  {horizon:15s}: {count}"
        )

    print("\nBY SCOPE")
    for scope, count in sorted(
        report["by_scope"].items()
    ):
        print(
            f"  {scope:25s}: {count}"
        )

    print("\nARTIFACTS")
    for artifact in report["artifacts"]:
        print(
            json.dumps(
                {
                    "name": artifact["name"],
                    "relative_path": artifact["relative_path"],
                    "role": artifact["role"],
                    "scope": artifact["scope"],
                    "horizon": artifact["horizon"],
                    "feature_count": artifact["feature_count"],
                    "metadata_status": artifact["metadata_status"],
                    "metadata_error": artifact["metadata_error"],
                    "sha256": artifact["sha256"],
                },
                sort_keys=True,
            )
        )

    print("\nSTATUS: AUDIT_ONLY")
    print(
        "No model, production artifact, prediction or UI was changed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
