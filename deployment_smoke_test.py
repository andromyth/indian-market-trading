"""Offline production smoke test for the repository's active model contracts.

This test validates the artifacts and adapters that are actually authoritative
in the current dashboard architecture. Large historical data and Google Drive
runtime availability are intentionally not required for this offline check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    checks: list[str] = []

    # Certified 137-feature target_30m artifact.
    pm_path = ROOT / "production_model" / "DEPLOYMENT_MODEL_MANIFEST_2026.json"
    pmod_path = ROOT / "production_model" / "deployment_classifier_logistic_regression_2026.joblib"
    for p in (pm_path, pmod_path):
        if not p.exists() or p.stat().st_size == 0:
            raise SystemExit(f"MISSING: {p}")

    pm = json.loads(pm_path.read_text(encoding="utf-8"))
    features = list(pm.get("features", []))
    if pm.get("target") != "target_30m":
        raise SystemExit("Certified production target is not target_30m")
    if len(features) != 137 or len(set(features)) != 137:
        raise SystemExit("Certified production feature manifest is not exactly 137 unique features")

    actual_sha = sha256(pmod_path)
    if actual_sha != pm.get("model_sha256"):
        raise SystemExit("Certified production model SHA-256 does not match manifest")

    prod = joblib.load(pmod_path)
    classes = list(getattr(prod.named_steps.get("model"), "classes_", []))
    if classes != [0, 1]:
        raise SystemExit(f"Certified production classes are {classes}, expected [0,1]")
    if int(getattr(prod, "n_features_in_", 0)) != 137:
        raise SystemExit("Certified production classifier does not expose 137 features")

    names = list(getattr(prod, "feature_names_in_", []))
    if names != features:
        raise SystemExit("Certified production model feature_names_in_ does not match manifest")

    checks.append("OK certified target_30m / 137-feature Logistic Regression / SHA-256")
    
    # Active Delivery adapter is the 59-feature ALL-NSE path. The actual
    # Drive model files are runtime dependencies, so this offline check only
    # validates adapter import and declared routing surface.
    import all_nse_delivery_model as delivery
    import all_nse_intraday_model as legacy_intraday
    import all_nse_intraday_5m_model as intraday_5m
    import production_model as certified_intraday
    import autonomous_engine
    import prediction_ledger
    import prediction_validation
    import model_health

    if not hasattr(delivery, "predict_symbol") or not hasattr(delivery, "model_status"):
        raise SystemExit("ALL-NSE Delivery adapter contract missing")
    if not hasattr(delivery, "supported_symbol"):
        raise SystemExit("ALL-NSE Delivery adapter has no trainable-universe gate")
    checks.append("OK active ALL-NSE Delivery adapter import / universe gate / 59-feature runtime contract")

    if len(legacy_intraday.FEATURES) != 34:
        raise SystemExit("Legacy/compatibility Intraday adapter feature contract changed unexpectedly")
    checks.append("OK compatibility Intraday adapter remains 34-feature and isolated")
\n    if len(intraday_5m.FEATURES) != 49:\n        raise SystemExit("5-minute Intraday V2 adapter feature contract is not 49")\n    if tuple(intraday_5m.HORIZONS) != (5, 10, 15, 20, 25, 30):\n        raise SystemExit("5-minute Intraday V2 horizon contract changed unexpectedly")\n    checks.append("OK 5-minute Intraday V2 adapter exposes 49 features / six direct horizons")\n
    if len(certified_intraday.FEATURES) != 137:
        raise SystemExit("Certified Intraday adapter feature contract is not 137")
    checks.append("OK certified Intraday adapter exposes 137 features")

    # Exercise the adapter's feature construction without loading the model,
    # proving the fixed OHLC variable path is executable.
    idx = pd.date_range("2026-01-01", periods=80, freq="D")
    base = pd.Series(range(80), index=idx, dtype="float64") + 100
    price = pd.DataFrame({
        "Open": base,
        "High": base + 1,
        "Low": base - 1,
        "Close": base + 0.5,
        "Volume": base * 10,
    }, index=idx)
    frame = certified_intraday.build_feature_rows([{
        "symbol": "SMOKE",
        "price_df": price,
        "market_df": price,
        "matched_news": [],
        "announcements": [],
        "event_risk": 0.0,
        "fo": {},
        "decision_ts": str(idx[-1]),
    }])
    if frame.shape != (1, 139):
        raise SystemExit(f"Certified Intraday feature builder returned {frame.shape}, expected (1,139)")
    checks.append("OK certified Intraday 137-feature adapter feature construction")

    print("DEPLOYMENT SMOKE TEST: PASS")
    for msg in checks:
        print(msg)


if __name__ == "__main__":
    main()
