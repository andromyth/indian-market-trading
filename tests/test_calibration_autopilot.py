from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import calibration_autopilot as ca
import calibration_overlay as co
import prediction_learning_engine as engine

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def synth(n_days=30, per_day=300, seed=1, mode="overconfident", edge=0.4, start="2026-08-03 04:00", version="v1"):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(start, tz="UTC")
    frames = []
    for d in range(n_days):
        ts = base + pd.Timedelta(days=d) + pd.to_timedelta(rng.integers(0, 330, per_day), unit="m")
        z = rng.normal(size=per_day)
        true_p = 1 / (1 + np.exp(-edge * z))
        up = rng.random(per_day) < true_p
        true_er = (true_p - 0.5) * 0.004
        ret = np.where(up, 1, -1) * np.abs(rng.normal(0.001, 0.0005, per_day))
        p = {
            "overconfident": 1 / (1 + np.exp(-(1.6 * z + 0.3))),
            "calibrated": true_p,
            "noise": 1 / (1 + np.exp(-(1.6 * rng.normal(size=per_day)))),
        }[mode]
        frames.append(pd.DataFrame({
            "target_timestamp_utc": ts, "probability_up": p, "actual_return": ret,
            "expected_return": 3 * true_er, "validation_status": "VALIDATED", "horizon_min": 30,
            "symbol": "SBIN", "model_version": version, "input_age_seconds": 30.0,
        }))
    return pd.concat(frames, ignore_index=True)


def fit(frame, key="h30__v1", rules=ca.Rules()):
    df, _ = ca.prepare_fit_frame(frame, rules)
    return ca.evaluate_probability_candidates(df, key, rules)


def test_key_and_paths_match_the_engine():
    assert ca.profile_key(30, "a/b") == "h30__a_b"
    assert ca.profile_key(None, None) == "hunknown__UNKNOWN"
    assert ca.LEARNING_ROOT == engine.LEARNING_ROOT
    assert ca.REMOTE_DIR == engine.CALIBRATION_REMOTE_DIR


def test_engine_scores_the_raw_forecast_even_when_a_profile_was_applied():
    record = {"probability_up": 0.61, "probability_up_raw": 0.80, "signal": "BUY", "signal_raw": "HOLD",
              "expected_return": 0.001, "expected_return_raw": 0.003, "predicted_price": 101.0,
              "predicted_price_raw": 102.0}
    view = engine.raw_forecast_view(record)
    assert view["probability_up"] == 0.80 and view["signal"] == "HOLD"
    assert view["expected_return"] == 0.003 and view["predicted_price"] == 102.0
    plain = {"probability_up": 0.7, "signal": "BUY"}
    assert engine.raw_forecast_view(plain) == plain


def test_apply_calibrator_is_bounded_and_nan_safe():
    out = ca.apply_calibrator({"type": "platt", "a": 0.0, "b": 1.0}, [0.0, 1.0, 0.5, np.nan, 1.5, -0.1])
    assert out[0] == pytest.approx(0.02) and out[1] == pytest.approx(0.98) and out[2] == pytest.approx(0.5)
    assert np.isnan(out[3]) and np.isnan(out[4]) and np.isnan(out[5])
    assert np.allclose(ca.apply_calibrator({"type": "mystery"}, [0.3]), [0.3])


def test_platt_is_monotone_and_slope_is_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.random(5000) < 0.5).astype(float)
    cal = ca.fit_calibrator("platt", p, y, ca.Rules())
    assert cal["b"] >= ca.Rules().slope_bounds[0] - 1e-9
    grid = ca.apply_calibrator(cal, np.linspace(0.05, 0.95, 50))
    assert np.all(np.diff(grid) >= -1e-12)


def _session_dates(n):
    return [(pd.Timestamp("2026-06-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]


def test_folds_are_chronological_with_embargo_and_use_more_history_as_it_grows():
    rules = ca.Rules()
    tested_sessions = {}
    for n in (20, 40, 80):
        dates = _session_dates(n)
        folds = ca.rolling_origin_folds(dates, rules)
        assert folds
        tested = set()
        for train, test in folds:
            assert max(train) < min(test)
            gap = dates.index(min(test)) - dates.index(max(train)) - 1
            assert gap >= rules.embargo_dates
            assert len(train) >= rules.min_train_dates
            tested |= set(test)
        tested_sessions[n] = len(tested)
    assert tested_sessions[20] < tested_sessions[40] < tested_sessions[80]


def test_insufficient_history_does_not_fit():
    result, art = fit(synth(n_days=10))
    assert result["status"] == "INSUFFICIENT_HISTORY" and art is None


def test_stale_input_rows_are_excluded_from_fitting():
    frame = synth(n_days=20)
    frame.loc[frame.index[:500], "input_age_seconds"] = 999.0
    df, report = ca.prepare_fit_frame(frame, ca.Rules())
    assert report["stale_input_rows_excluded"] == 500 and len(df) == len(frame) - 500


def test_overconfident_model_gets_a_shrinking_logistic_correction():
    result, art = fit(synth(n_days=30))
    assert result["status"] == "ELIGIBLE" and result["chosen"] == "platt"
    assert 0.2 < art["calibrator"]["b"] < 1.0
    assert result["candidates"]["platt"]["logloss_gain"]["lo"] > 0
    assert result["large_correction_flag"] is True


def test_already_calibrated_model_is_left_alone():
    result, art = fit(synth(n_days=30, mode="calibrated"))
    assert result["status"] == "REJECTED_BY_GATE" and art is None


def test_model_without_detectable_skill_is_not_dressed_up_as_validated():
    result, art = fit(synth(n_days=30, mode="noise"))
    assert result["status"] == "REJECTED_NO_DISCRIMINATION" and art is None
    assert result["retrain_recommended"] is True


def test_overconfidence_beyond_the_slope_floor_asks_for_retraining():
    result, art = fit(synth(n_days=30, edge=0.15))
    assert result["status"] == "REJECTED_UNINFORMATIVE_MODEL" and art is None
    assert result["retrain_recommended"] is True


def test_return_head_that_is_three_times_too_large_is_scaled_down():
    df, _ = ca.prepare_fit_frame(synth(n_days=30), ca.Rules())
    out = ca.evaluate_return_candidates(df, "h30__v1", ca.Rules())
    assert out["status"] == "ELIGIBLE" and out["chosen"] in {"scale", "affine"}
    assert 0.10 < out["correction"]["b"] < 0.25
    assert out["price_forecast_reliable"] is True


def _band_frame(informative: bool, sessions=24, per=400, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(sessions):
        z = rng.normal(size=per)
        p = 1 / (1 + np.exp(-3 * z))
        drift = np.sign(z) * 0.0025 if informative else 0.0
        ret = drift + rng.normal(0, 0.002, per)
        rows.append(pd.DataFrame({"session_date": f"2026-09-{d + 1:02d}", "p_x": p, "ret": ret}))
    return pd.concat(rows, ignore_index=True)


def test_action_band_requires_net_of_cost_edge_and_otherwise_abstains():
    good = ca.fit_action_band(_band_frame(True), "p_x", "k", ca.Rules())
    assert good["abstain_all"] is False and good["buy_p"] > 0.5 > good["sell_p"]
    assert good["net_bps_lb"] > 0
    bad = ca.fit_action_band(_band_frame(False), "p_x", "k", ca.Rules())
    assert bad["abstain_all"] is True
    assert ca.signal_from_thresholds(0.9, bad) == "ABSTAIN"
    assert ca.signal_from_thresholds(0.9, good) == "BUY" and ca.signal_from_thresholds(0.1, good) == "SELL"


def test_drift_helpers():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    assert ca.psi(a, rng.normal(0, 1, 2000)) < 0.05
    assert ca.psi(a, rng.normal(1.5, 1, 2000)) > 0.25
    flat = np.full(30, 0.25)
    shifted = np.concatenate([np.full(15, 0.25), np.full(15, 0.32)])
    assert ca.page_hinkley(flat, 0.002, 0.05)["alarm"] is False
    assert ca.page_hinkley(shifted, 0.002, 0.05)["alarm"] is True


def test_state_loading_never_resets_on_bad_input():
    assert ca.load_state(None)["entries"] == {}
    for bad in (b"", b"{not json", b"[]", json.dumps({"schema_version": 99, "entries": {}}).encode(),
                json.dumps({"schema_version": 1}).encode()):
        with pytest.raises(ca.AutopilotStateError):
            ca.load_state(bad)


def _profile(stage=ca.STAGE_SHADOW, created=NOW, a=-0.3, b=0.5, valid_days=35, pid="p1"):
    return {"profile_id": pid, "key": "k", "stage": stage, "created_utc": created.isoformat(),
            "valid_until_utc": (created + timedelta(days=valid_days)).isoformat(),
            "calibrator": {"type": "platt", "a": a, "b": b}, "return_correction": None, "thresholds": {"abstain_all": True}}


def _live(dates, lo, hi=None, mean=None, brier_ok=True):
    return {"dates": dates, "brier_ok": brier_ok, "logloss_gain": {"mean": mean if mean is not None else lo, "lo": lo, "hi": hi if hi is not None else (lo or 0) + 0.01}}


def test_lifecycle_shadow_then_promotion_then_live_demotion_and_cooldown():
    rules = ca.Rules()
    entry = {"champion": None, "history": [], "events": []}
    ca.advance_entry(entry, candidate={"profile": _profile()}, live={"dates": 0}, now=NOW, rules=rules)
    assert entry["champion"]["stage"] == ca.STAGE_SHADOW
    ca.advance_entry(entry, candidate=None, live=_live(rules.shadow_min_dates - 1, 0.01), now=NOW, rules=rules)
    assert entry["champion"]["stage"] == ca.STAGE_SHADOW
    ca.advance_entry(entry, candidate=None, live=_live(rules.shadow_min_dates, -0.001, 0.01), now=NOW, rules=rules)
    assert entry["champion"]["stage"] == ca.STAGE_SHADOW
    ca.advance_entry(entry, candidate=None, live=_live(rules.shadow_min_dates, 0.002), now=NOW, rules=rules)
    assert entry["champion"]["stage"] == ca.STAGE_ACTIVE
    ev = ca.advance_entry(entry, candidate=None, live=_live(rules.demote_min_dates, -0.02, -0.005, mean=-0.01, brier_ok=False), now=NOW, rules=rules)
    assert entry["champion"] is None and any(e["event"].startswith("DEMOTED") for e in ev)
    held = ca.advance_entry(entry, candidate={"profile": _profile(pid="p2")}, live={"dates": 0}, now=NOW + timedelta(days=1), rules=rules)
    assert entry["champion"] is None and held[0]["event"] == "CANDIDATE_HELD_BY_COOLDOWN"
    ca.advance_entry(entry, candidate={"profile": _profile(pid="p2")}, live={"dates": 0}, now=NOW + timedelta(days=rules.cooldown_days + 1), rules=rules)
    assert entry["champion"]["stage"] == ca.STAGE_SHADOW


def test_shadow_that_is_clearly_worse_or_times_out_is_rejected_and_expiry_retires_profiles():
    rules = ca.Rules()
    worse = {"champion": _profile(), "history": [], "events": []}
    ca.advance_entry(worse, candidate=None, live=_live(rules.shadow_min_dates, -0.02, -0.004), now=NOW, rules=rules)
    assert worse["champion"] is None and worse["history"][-1]["stage"] == "REJECTED"
    timeout = {"champion": _profile(), "history": [], "events": []}
    ca.advance_entry(timeout, candidate=None, live=_live(rules.shadow_max_dates, -0.001, 0.003), now=NOW, rules=rules)
    assert timeout["champion"] is None
    old = {"champion": _profile(created=NOW - timedelta(days=40)), "history": [], "events": []}
    ev = ca.advance_entry(old, candidate=None, live={"dates": 0}, now=NOW, rules=rules)
    assert old["champion"] is None and ev[0]["event"] == "EXPIRED"


def test_refit_is_refreshed_in_place_when_close_and_only_noted_when_far():
    rules = ca.Rules()
    created = NOW - timedelta(days=rules.refit_after_days + 1)
    entry = {"champion": _profile(stage=ca.STAGE_ACTIVE, created=created, pid="old"), "history": [], "events": []}
    ca.advance_entry(entry, candidate={"profile": _profile(created=NOW, a=-0.29, b=0.51, pid="new")}, live={"dates": 0}, now=NOW, rules=rules)
    assert entry["champion"]["profile_id"] == "new" and entry["champion"]["stage"] == ca.STAGE_ACTIVE
    entry2 = {"champion": _profile(stage=ca.STAGE_ACTIVE, created=created, pid="old"), "history": [], "events": []}
    ca.advance_entry(entry2, candidate={"profile": _profile(created=NOW, a=0.5, b=1.5, pid="far")}, live={"dates": 0}, now=NOW, rules=rules)
    assert entry2["champion"]["profile_id"] == "old" and entry2["challenger_note"]["profile_id"] == "far"


def test_shadow_profile_is_not_reset_at_seven_calendar_days_before_min_sessions():
    rules = ca.Rules()
    history = synth(n_days=20, start="2026-08-03 04:00", seed=1)
    state, doc, _ = ca.run_autopilot(history, ca.new_state(), now=NOW, rules=rules)
    shadow_profile = doc["profiles"][0]
    original_id = shadow_profile["profile_id"]
    week1_live = _live_rows(shadow_profile, 5, "2026-10-18 04:00", seed=21)
    state, doc, _ = ca.run_autopilot(
        pd.concat([history, week1_live], ignore_index=True), state,
        now=NOW + timedelta(days=7), rules=rules,
    )
    assert doc["profiles"][0]["stage"] == ca.STAGE_SHADOW
    assert doc["profiles"][0]["profile_id"] == original_id
    week2_live = _live_rows(shadow_profile, 5, "2026-10-23 04:00", seed=22)
    state, doc, report = ca.run_autopilot(
        pd.concat([history, week1_live, week2_live], ignore_index=True), state,
        now=NOW + timedelta(days=14), rules=rules,
    )
    assert doc["profiles"][0]["stage"] == ca.STAGE_ACTIVE
    assert any(e["event"] == "PROMOTED_TO_ACTIVE" for e in report["events"])


def test_return_head_prefers_scale_over_zero_even_when_raw_error_is_huge():
    frame = synth(n_days=30, seed=5)
    frame["expected_return"] = frame["expected_return"] * 10.0
    df, _ = ca.prepare_fit_frame(frame, ca.Rules())
    out = ca.evaluate_return_candidates(df, "h30__v1", ca.Rules())
    assert out["status"] == "ELIGIBLE"
    assert out["chosen"] in {"scale", "affine"}
    assert out["price_forecast_reliable"] is True


def _live_rows(profile, n_days, start, seed, worse=False):
    rows = synth(n_days=n_days, seed=seed, start=start)
    rows["calibration_profile_id"] = profile["profile_id"]
    raw = rows["probability_up"].to_numpy()
    if worse:
        cal = np.clip(1 / (1 + np.exp(-3.0 * np.log(raw / (1 - raw)))), 0.02, 0.98)
    else:
        cal = ca.apply_calibrator(profile["calibrator"], raw)
    rows["probability_up_calibrated"] = cal
    rows["expected_return_calibrated"] = rows["expected_return"]
    return rows


def test_two_runs_shadow_started_then_promoted_by_live_evidence():
    history = synth(n_days=30)
    state, doc, report = ca.run_autopilot(history, ca.new_state(), now=NOW)
    profile = doc["profiles"][0]
    assert profile["stage"] == ca.STAGE_SHADOW and report["keys"]["h30__v1"]["champion_stage"] == ca.STAGE_SHADOW
    live = _live_rows(profile, ca.Rules().shadow_min_dates + 2, "2026-10-18 04:00", seed=11)
    state2, doc2, report2 = ca.run_autopilot(
        pd.concat([history, live], ignore_index=True), state, now=NOW + timedelta(days=14)
    )
    assert doc2["profiles"][0]["stage"] == ca.STAGE_ACTIVE
    assert any(e["event"] == "PROMOTED_TO_ACTIVE" for e in report2["events"])
    assert json.dumps(state2) and json.dumps(report2)


def test_active_profile_that_turns_harmful_is_demoted_automatically():
    history = synth(n_days=30)
    state, doc, _ = ca.run_autopilot(history, ca.new_state(), now=NOW)
    state["entries"]["h30__v1"]["champion"]["stage"] = ca.STAGE_ACTIVE
    profile = state["entries"]["h30__v1"]["champion"]
    bad = _live_rows(profile, ca.Rules().demote_min_dates + 1, "2026-10-18 04:00", seed=12, worse=True)
    state2, doc2, report2 = ca.run_autopilot(pd.concat([history, bad], ignore_index=True), state, now=NOW + timedelta(days=14))
    assert doc2["profiles"] == []
    assert any(e["event"].startswith("DEMOTED") for e in report2["events"])


def test_no_outcomes_still_expires_stale_profiles():
    state = ca.new_state()
    state["entries"]["k"] = {"champion": _profile(created=NOW - timedelta(days=60)), "history": [], "events": []}
    new_state, doc, report = ca.run_autopilot(pd.DataFrame(), state, now=NOW)
    assert report["status"] == "NO_MATURED_OUTCOMES" and doc["profiles"] == []
    assert new_state["entries"]["k"]["champion"] is None


def test_run_autopilot_does_not_mutate_its_input_state():
    state = ca.new_state()
    snapshot = json.dumps(state, sort_keys=True)
    ca.run_autopilot(synth(n_days=30), state, now=NOW)
    assert json.dumps(state, sort_keys=True) == snapshot


def _doc(stage=ca.STAGE_ACTIVE, valid_until=NOW + timedelta(days=5), thresholds=None, correction=None, key="h30__v1"):
    return {"schema_version": 1, "profiles": [{
        "profile_id": "pid", "key": key, "stage": stage, "valid_until_utc": valid_until.isoformat(),
        "calibrator": {"type": "platt", "a": 0.0, "b": 0.5}, "return_correction": correction,
        "price_forecast_reliable": True,
        "thresholds": thresholds or {"abstain_all": False, "buy_p": 0.55, "sell_p": 0.45}}]}


def _row(p=0.9):
    return {"probability_up": p, "horizon_min": 30, "model_version": "v1", "signal": "HOLD",
            "expected_return": 0.004, "reference_close": 100.0, "predicted_price": 100.4}


def test_overlay_shadow_mode_adds_fields_but_never_changes_primary_fields():
    overlay = co.CalibrationOverlay(lambda: _doc(), mode="shadow", now_fn=lambda: NOW)
    out = overlay.annotate(_row())
    assert out["probability_up"] == 0.9 and out["signal"] == "HOLD"
    assert out["probability_up_calibrated"] < 0.9 and out["calibration_applied"] is False
    assert out["calibration_profile_id"] == "pid"


def test_overlay_active_mode_applies_active_profiles_and_keeps_raw_values():
    cor = {"type": "scale", "a": 0.0, "b": 0.5}
    overlay = co.CalibrationOverlay(lambda: _doc(correction=cor), mode="active", now_fn=lambda: NOW)
    out = overlay.annotate(_row())
    assert out["probability_up"] == pytest.approx(out["probability_up_calibrated"]) and out["probability_up_raw"] == 0.9
    assert out["expected_return"] == pytest.approx(0.002) and out["expected_return_raw"] == 0.004
    assert out["predicted_price"] == pytest.approx(100.2) and out["predicted_price_raw"] == 100.4
    assert out["signal"] == "BUY" and out["signal_raw"] == "HOLD" and out["calibration_applied"] is True


def test_overlay_never_applies_shadow_stage_even_in_active_mode_and_respects_abstain():
    shadow = co.CalibrationOverlay(lambda: _doc(stage=ca.STAGE_SHADOW), mode="active", now_fn=lambda: NOW).annotate(_row())
    assert shadow["probability_up"] == 0.9 and shadow["calibration_applied"] is False
    abstain = co.CalibrationOverlay(lambda: _doc(thresholds={"abstain_all": True}), mode="active", now_fn=lambda: NOW).annotate(_row())
    assert abstain["signal"] == "HOLD" and abstain["probability_up"] < 0.9 and abstain["signal_calibrated"] == "ABSTAIN"


def test_overlay_fails_open_for_every_kind_of_problem():
    row = _row()
    assert co.CalibrationOverlay(lambda: _doc(), mode="off").annotate(row) is row
    expired = co.CalibrationOverlay(lambda: _doc(valid_until=NOW - timedelta(seconds=1)), mode="active", now_fn=lambda: NOW)
    assert expired.annotate(row) == row
    assert co.CalibrationOverlay(lambda: _doc(key="h5__v1"), mode="active", now_fn=lambda: NOW).annotate(row) == row
    def boom():
        raise RuntimeError("drive down")
    failing = co.CalibrationOverlay(boom, mode="active", now_fn=lambda: NOW)
    assert failing.annotate(row) == row and "drive down" in failing.last_error
    ok = co.CalibrationOverlay(lambda: _doc(), mode="active", now_fn=lambda: NOW)
    for garbage in ({"probability_up": None}, {"probability_up": "x"}, {"probability_up": 1.7}, {}, None, "text"):
        assert ok.annotate(garbage) == garbage
    assert co.CalibrationOverlay(lambda: {"schema_version": 9, "profiles": []}, mode="active").annotate(row) == row
    assert co.CalibrationOverlay(lambda: None, mode="active").annotate(row) == row


def test_overlay_keeps_serving_last_good_profiles_when_a_refresh_fails_and_retries_soon():
    clock = {"t": 0.0}
    calls = {"n": 0}
    def loader():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("transient")
        return _doc()
    overlay = co.CalibrationOverlay(loader, mode="shadow", refresh_seconds=900, clock=lambda: clock["t"], now_fn=lambda: NOW)
    assert "probability_up_calibrated" in overlay.annotate(_row())
    clock["t"] = 901
    assert "probability_up_calibrated" in overlay.annotate(_row())
    assert overlay.last_error and calls["n"] == 2
    clock["t"] = 901 + co.RETRY_AFTER_FAILURE_SECONDS + 1
    overlay.annotate(_row())
    assert calls["n"] == 3 and overlay.last_error is None


def test_mode_resolution_falls_back_to_observe_only():
    assert co.resolve_mode("ACTIVE ") == "active" and co.resolve_mode("off") == "off"
    assert co.resolve_mode("garbage") == "shadow" and co.resolve_mode("") == "shadow"


class _FakeStore:
    def __init__(self, upload_status="OK"):
        self.files: dict[str, bytes] = {}
        self.upload_status = upload_status
    def upload_or_update(self, path, remote, mime_type=None):
        if self.upload_status == "OK":
            self.files[remote] = open(path, "rb").read()
        return {"status": self.upload_status}


def test_build_outcomes_scores_the_raw_forecast_when_a_profile_was_applied(monkeypatch):
    import io
    bars = pd.DataFrame({"timestamp": pd.date_range("2026-10-14 04:30", "2026-10-14 06:00", freq="1min", tz="UTC")})
    bars["Close"] = 100.0
    bars.loc[bars["timestamp"] >= pd.Timestamp("2026-10-14 05:30", tz="UTC"), "Close"] = 101.0
    buffer = io.BytesIO()
    bars.to_parquet(buffer, index=False)
    monkeypatch.setattr(engine, "_download_drive_bytes", lambda store, remote: buffer.getvalue())
    record = {
        "symbol": "SBIN.NS", "horizon_min": 30, "model_version": "v1", "decision_timestamp": "2026-10-14T05:00:00+00:00",
        "reference_close": 100.0, "forecast_key": "SBIN|2026-10-14T05:00:00+00:00|30|v1",
        "probability_up": 0.55, "probability_up_raw": 0.90, "signal": "BUY", "signal_raw": "HOLD",
        "expected_return": 0.001, "expected_return_raw": 0.004, "predicted_price": 100.1, "predicted_price_raw": 100.4,
        "probability_up_calibrated": 0.55, "calibration_profile_id": "pid", "calibration_stage": "ACTIVE",
        "calibration_applied": True,
    }
    frame, _ = engine._build_outcomes(object(), [("2026-10-14", record)])
    row = frame.iloc[0]
    assert row["validation_status"] == "VALIDATED"
    assert row["probability_up"] == pytest.approx(0.90) and row["signal"] == "HOLD"
    assert row["expected_return"] == pytest.approx(0.004) and row["predicted_price"] == pytest.approx(100.4)
    assert row["brier_score"] == pytest.approx(0.01)
    assert row["probability_up_calibrated"] == pytest.approx(0.55) and row["calibration_profile_id"] == "pid"
    assert bool(row["calibration_applied"]) is True


def test_engine_autopilot_persists_state_and_never_overwrites_an_unreadable_registry(tmp_path, monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(engine, "_download_drive_bytes", lambda st, remote: st.files.get(remote))
    registry = f"{engine.CALIBRATION_REMOTE_DIR}/{engine.REGISTRY_NAME}"
    active = f"{engine.CALIBRATION_REMOTE_DIR}/{engine.ACTIVE_PROFILE_NAME}"
    frame = synth(n_days=30)
    first = engine._run_calibration_autopilot(store, frame, tmp_path)
    assert first["status"] == "COMPLETED" and first["registry_overwritten"] is True
    assert json.loads(store.files[active])["profiles"][0]["stage"] == "SHADOW"
    saved = store.files[registry]
    second = engine._run_calibration_autopilot(store, frame, tmp_path)
    assert second["status"] == "COMPLETED"
    store.files[registry] = b"{corrupt"
    third = engine._run_calibration_autopilot(store, frame, tmp_path)
    assert third["status"] == "STATE_LOAD_FAILED" and third["registry_overwritten"] is False
    assert store.files[registry] == b"{corrupt" and saved != b"{corrupt"
    failing = _FakeStore(upload_status="ERROR")
    failed = engine._run_calibration_autopilot(failing, frame, tmp_path)
    assert failed["status"] == "UPLOAD_FAILED"
    assert "UPLOAD_FAILED" in engine.AUTOPILOT_FAILURE_STATUSES


def test_engine_autopilot_local_mode_and_empty_frame(tmp_path):
    result = engine._run_calibration_autopilot(None, pd.DataFrame(), tmp_path)
    assert result["status"] == "NO_MATURED_OUTCOMES" and result["registry_overwritten"] is True
    assert (tmp_path / "calibration" / engine.REGISTRY_NAME).exists()
    assert json.loads((tmp_path / "calibration" / engine.ACTIVE_PROFILE_NAME).read_text())["profiles"] == []


def test_worker_hook_fails_open_if_the_overlay_cannot_be_built(monkeypatch):
    from autonomous_intraday_worker import AutonomousIntradayWorker
    def broken(*args, **kwargs):
        raise ImportError("simulated missing module")
    monkeypatch.setattr(co, "CalibrationOverlay", broken)
    worker = AutonomousIntradayWorker(segment="once")
    rows = [_row()]
    assert worker._apply_calibration(rows) is rows
    assert worker._apply_calibration(rows) is rows


def test_worker_hook_annotates_through_the_real_overlay(monkeypatch):
    from autonomous_intraday_worker import AutonomousIntradayWorker
    worker = AutonomousIntradayWorker(segment="once")
    worker._calibration = co.CalibrationOverlay(lambda: _doc(), mode="shadow", now_fn=lambda: NOW)
    out = worker._apply_calibration([_row()])
    assert out[0]["probability_up"] == 0.9 and "probability_up_calibrated" in out[0]
