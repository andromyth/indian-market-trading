from __future__ import annotations

import json

from selfdriving import sd_v2_runtime as runtime


def test_intraday_horizon_model_bundle_is_cached_and_loaded_once(tmp_path, monkeypatch):
    registry_root = tmp_path / "intraday" / "models" / "h5"
    registry_root.mkdir(parents=True)
    (registry_root / "registry.json").write_text(
        json.dumps({"champion": {"file": "champion_h5.joblib"}}),
        encoding="utf-8",
    )

    bundle = object()
    loads = []
    monkeypatch.setattr(runtime, "CACHE", tmp_path)
    monkeypatch.setattr(runtime, "_ensure", lambda remote, local: local)
    monkeypatch.setattr(
        runtime.joblib,
        "load",
        lambda path: loads.append(str(path)) or bundle,
    )
    monkeypatch.setattr(runtime, "_INTRADAY_BUNDLE_CACHE", {})
    monkeypatch.setenv("BETA_V2_MODEL_CACHE_TTL_SECONDS", "180")

    state1, loaded1, elapsed1, cache_hit1 = runtime._load_intraday_champion(5)
    state2, loaded2, elapsed2, cache_hit2 = runtime._load_intraday_champion(5)

    assert state1["champion"]["file"] == "champion_h5.joblib"
    assert state2 is state1
    assert loaded1 is bundle
    assert loaded2 is bundle
    assert cache_hit1 is False
    assert cache_hit2 is True
    assert len(loads) == 1
    assert elapsed1 >= 0
    assert elapsed2 == 0.0
