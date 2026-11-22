from __future__ import annotations

import io
import json

import joblib
import pytest

import promote_to_close_candidate as promo


class MemoryStore:
    def __init__(self):
        self.data: dict[str, bytes] = {}

    def read(self, rel):
        return self.data.get(rel)

    def write(self, rel, data):
        self.data[rel] = data

    def stamp(self, rel):
        blob = self.data.get(rel)
        return None if blob is None else f"{len(blob)}"


def _bundle(tag="x"):
    deploy = {
        "tag": tag,
        "features": ["f1", "f2"],
        "thresholds": {"buy_p": 0.6, "sell_p": 0.4, "min_edge": 0.0011},
        "feature_warmup": 3,
        "feature_module": "features_5m",
        "training_symbols": ["AAA", "BBB"],
    }
    buf = io.BytesIO()
    joblib.dump({"close": deploy}, buf)
    return buf.getvalue()


def _seed(candidate, *, rid="20261122T000000Z", verdict="PASS", trained_through="2026-11-18", bundle=None):
    bundle = bundle or _bundle(f"{rid}-{trained_through}")
    candidate.write(f"runs/{rid}/candidate_bundle.joblib", bundle)
    candidate.write(
        f"runs/{rid}/training_report.json",
        json.dumps(
            {
                "coverage": {"fno_available_share": 0.9},
                "result": {
                    "verdict": verdict,
                    "gate": {"auc_lower_gt_0.5": True},
                    "holdout_auc": {"mean": 0.55, "lower": 0.52},
                    "holdout_auc_by_time_left": {},
                    "holdout_trading": {"mean_net_bps": 3.0, "se_bps": 1.0},
                },
            }
        ).encode(),
    )
    candidate.write(
        "latest.json",
        json.dumps(
            {
                "run_id": rid,
                "training_verdict": verdict,
                "trained_through": trained_through,
                "training_sessions": 500,
                "candidate_bundle_sha256": promo._sha256(bundle),
            }
        ).encode(),
    )
    return bundle


def test_fail_keeps_previous_pointer():
    candidate, champion = MemoryStore(), MemoryStore()
    _seed(candidate, verdict="FAIL")
    out = promo.promote(candidate, champion)
    assert out["decision"] == "KEEP_PREVIOUS"
    assert champion.read("champion.json") is None
    assert not any(k.startswith("versions/") for k in champion.data)


def test_pass_publishes_verified_bundle():
    candidate, champion = MemoryStore(), MemoryStore()
    bundle = _seed(candidate)
    out = promo.promote(candidate, champion)
    assert out["decision"] == "PROMOTE"
    assert out["active"] is False and out["pass_streak"] == 1
    champ = json.loads(champion.read("champion.json"))
    assert champ["bundle_sha256"] == promo._sha256(bundle)
    assert promo._sha256(champion.read(champ["bundle_path"])) == champ["bundle_sha256"]
    assert champ["horizons"]["close"]["active"] is False


def test_three_passes_activate_and_keep_streak():
    candidate, champion = MemoryStore(), MemoryStore()
    _seed(candidate, trained_through="2026-11-16")
    promo.promote(candidate, champion)
    _seed(candidate, rid="20261117T000000Z", trained_through="2026-11-17")
    promo.promote(candidate, champion)
    _seed(candidate, rid="20261118T000000Z", trained_through="2026-11-18")
    out = promo.promote(candidate, champion)
    assert out["pass_streak"] == 3 and out["active"] is True
    assert json.loads(champion.read("champion.json"))["horizons"]["close"]["active"] is True


def test_same_session_rerun_does_not_double_count():
    candidate, champion = MemoryStore(), MemoryStore()
    _seed(candidate, rid="a", trained_through="2026-11-18")
    promo.promote(candidate, champion)
    _seed(candidate, rid="b", trained_through="2026-11-18")
    out = promo.promote(candidate, champion)
    assert out["pass_streak"] == 1


def test_bundle_hash_mismatch_raises():
    candidate, champion = MemoryStore(), MemoryStore()
    _seed(candidate)
    latest = json.loads(candidate.read("latest.json"))
    latest["candidate_bundle_sha256"] = "0" * 64
    candidate.write("latest.json", json.dumps(latest).encode())
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        promo.promote(candidate, champion)


def test_already_promoted_is_idempotent():
    candidate, champion = MemoryStore(), MemoryStore()
    _seed(candidate)
    promo.promote(candidate, champion)
    before = champion.read("champion.json")
    out = promo.promote(candidate, champion)
    assert out["decision"] == "ALREADY_PROMOTED"
    assert champion.read("champion.json") == before


def test_dry_run_writes_nothing():
    candidate, champion = MemoryStore(), MemoryStore()
    _seed(candidate)
    out = promo.promote(candidate, champion, dry_run=True)
    assert out["decision"] == "DRY_RUN"
    assert champion.data == {}


def test_force_publishes_fail_but_stays_inactive():
    candidate, champion = MemoryStore(), MemoryStore()
    _seed(candidate, verdict="FAIL")
    out = promo.promote(candidate, champion, force=True)
    assert out["decision"] == "PROMOTE"
    assert out["active"] is False
    assert json.loads(champion.read("champion.json"))["horizons"]["close"]["verdict"] == "FAIL"
