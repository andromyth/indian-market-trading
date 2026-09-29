"""Offline deployment smoke test for model artifacts and application imports."""
from __future__ import annotations

import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent


def main() -> None:
    checks = []

    prod_manifest = ROOT / "production_model" / "DEPLOYMENT_MODEL_MANIFEST_2026.json"
    prod_model = ROOT / "production_model" / "deployment_classifier_logistic_regression_2026.joblib"
    d_manifest = ROOT / "production_delivery_model" / "PREDICTION_ENGINE_MANIFEST_V2_2026.json"
    d_classifier = ROOT / "production_delivery_model" / "deployment_next_session_logistic_classifier_2026.joblib"
    d_regressor = ROOT / "production_delivery_model" / "deployment_next_session_hgb_return_regressor_2026.joblib"

    for p in [prod_manifest, prod_model, d_manifest, d_classifier, d_regressor]:
        if not p.exists():
            raise SystemExit(f"MISSING: {p}")
        checks.append(f"OK {p.relative_to(ROOT)}")

    pm = json.loads(prod_manifest.read_text(encoding="utf-8"))
    if pm.get("target") != "target_30m" or int(pm.get("feature_count", 0)) != 137:
        raise SystemExit("Production 30m manifest contract mismatch")
    checks.append("OK production target_30m / 137 features")

    dm = json.loads(d_manifest.read_text(encoding="utf-8"))
    expected = {"BHARTIARTL", "RELIANCE", "SBIN"}
    if not expected.issubset({str(x).upper() for x in dm.get("trained_symbols", [])}):
        raise SystemExit("Delivery trained-universe contract mismatch")

    c = joblib.load(d_classifier)
    r = joblib.load(d_regressor)
    if list(c.named_steps["model"].classes_) != [0, 1]:
        raise SystemExit("Delivery classifier classes are not [0, 1]")
    if int(c.n_features_in_) != 53 or int(r.n_features_in_) != 53:
        raise SystemExit("Delivery model feature count is not 53")
    checks.append("OK delivery classifier/regressor / 53 features / classes [0,1]")

    import delivery_model  # noqa: F401
    import prediction_ledger  # noqa: F401
    import prediction_validation  # noqa: F401
    import model_health  # noqa: F401
    import autonomous_engine  # noqa: F401
    checks.append("OK new runtime modules import")

    print("DEPLOYMENT SMOKE TEST: PASS")
    for c in checks:
        print(c)


if __name__ == "__main__":
    main()
