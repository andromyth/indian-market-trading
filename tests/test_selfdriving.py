import numpy as np
import pandas as pd
import pytest

from selfdriving.sd_walkforward import (FoldSpec, ModelConfig, add_targets, choose_thresholds, evaluate,
                                        purged_folds, run_walkforward, to_signal, wilson_lower)
from selfdriving.sd_registry import GateRules, Registry, gate, psi, live_check
from selfdriving.sd_global_cues import GLOBAL_SERIES, attach_global_features, availability
from selfdriving.sd_news_features import build_news_features, score_news_frame
from selfdriving.sd_intraday_features import build_intraday_panel
from selfdriving.sd_cycle import train_stage, predict_stage, select_groups, ensure_delivery_targets

FAST = ModelConfig(hgb_params=dict(max_depth=3, learning_rate=0.1, max_iter=60, min_samples_leaf=100,
                                   l2_regularization=5.0, max_bins=31, early_stopping=False, random_state=7))
SPEC = FoldSpec(n_folds=5, test_days=20, min_train_days=150)


def make_panel(signal: float, n_sym=80, n_days=330, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for d_i, d in enumerate(days):
        mkt = rng.normal(0, 0.008)
        f = rng.normal(size=(n_sym, 4))
        ret = mkt + signal * (f[:, 0] - 0.5 * f[:, 1]) + rng.normal(0, 0.02, n_sym)
        df = pd.DataFrame(f, columns=["ret_1", "ret_3", "rsi14", "atr14"])
        df["symbol"] = [f"S{i}" for i in range(n_sym)]
        df["date"] = d
        df["decision_timestamp"] = (d + pd.Timedelta(hours=10)).tz_localize("UTC")
        df["y_return"] = ret
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_trusted_delivery_labels_do_not_require_close():
    panel = pd.DataFrame({
        "symbol": ["AAA", "AAA"],
        "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
        "y_return": [0.02, -0.01],
        "ret_1": [0.1, -0.2],
    })
    out = ensure_delivery_targets(panel)
    assert out["y_return"].tolist() == [0.02, -0.01]
    assert "close" not in out.columns


def test_delivery_labels_are_derived_only_when_close_exists():
    panel = pd.DataFrame({
        "symbol": ["AAA", "AAA"],
        "date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
        "close": [100.0, 105.0],
    })
    out = ensure_delivery_targets(panel)
    assert np.isclose(out["y_return"].iloc[0], 0.05)
    assert np.isnan(out["y_return"].iloc[-1])


def test_purge_and_embargo():
    days = pd.bdate_range("2024-01-01", periods=400)
    spec = FoldSpec(n_folds=4, test_days=20, min_train_days=200, horizon_days=5, embargo_days=2)
    for tr, te in purged_folds(days, spec):
        gap = np.searchsorted(days.values, te[0]) - np.searchsorted(days.values, tr[-1])
        assert gap >= spec.horizon_days + spec.embargo_days + 1 - 1  # >= 7 sessions between last train and first test
        assert tr.max() < te.min()


def test_signal_is_found_and_promoted(tmp_path):
    p = make_panel(signal=0.006)
    pred = run_walkforward(add_targets(p), ["ret_1", "ret_3", "rsi14", "atr14"], cfg=FAST, spec=SPEC)
    m = evaluate(pred)
    assert m["auc"] > 0.53 and m["ic_t"] > 2
    assert m["accuracy"] > m["base_rate_majority"]


def test_pure_noise_is_rejected_and_abstains(tmp_path):
    """The most important test: with no real signal the system must not claim edge."""
    p = make_panel(signal=0.0, seed=3)
    out = train_stage(p, tmp_path / "m", spec=SPEC, cfg=FAST, rules=GateRules(min_dates=60))
    assert out["status"] == "REJECTED"
    assert out["thresholds"].get("abstain_all", True) is True
    assert Registry(tmp_path / "m").state["champion"] is None


def test_end_to_end_promote_predict_rollback(tmp_path):
    p = make_panel(signal=0.007, seed=1)
    out = train_stage(p, tmp_path / "m", spec=SPEC, cfg=FAST)
    assert out["status"] == "PROMOTED", out["gate"]
    last = p[p["date"] == p["date"].max()].drop(columns=["y_return"])
    sig = predict_stage(last, tmp_path / "m")
    assert set(sig["signal"]).issubset({"BUY", "SELL", "HOLD", "ABSTAIN"})
    reg = Registry(tmp_path / "m")
    assert reg.champion_bundle() is not None


def test_global_cue_point_in_time():
    # one US bar for Mon 2024-03-04 closes 16:00 NY (= 21:00 UTC); never visible before that
    d = pd.Timestamp("2024-03-04")
    av = availability(d, GLOBAL_SERIES["spx"])
    long = pd.DataFrame({"key": "spx", "date": pd.bdate_range("2024-01-01", periods=60),
                         "close": np.linspace(4000, 4200, 60)})
    long["available_at"] = [availability(x, GLOBAL_SERIES["spx"]) for x in long["date"]]
    panel = pd.DataFrame({"symbol": "A", "decision_timestamp": [av - pd.Timedelta(minutes=1), av + pd.Timedelta(minutes=30)]})
    out = attach_global_features(panel, long)
    ret_before, ret_after = out["g_spx_r1"].iloc[0], out["g_spx_r1"].iloc[1]
    exp_mon = np.log(long.loc[long["date"] == d, "close"].iloc[0] / long.loc[long["date"] == d - pd.Timedelta(days=3), "close"].iloc[0])
    assert np.isclose(ret_after, exp_mon)
    assert not np.isclose(ret_before, exp_mon)   # Monday's bar not yet known one minute before close+lag


def test_global_stale_bars_become_nan():
    long = pd.DataFrame({"key": "spx", "date": pd.bdate_range("2024-01-01", periods=30), "close": np.arange(30) + 100.0})
    long["available_at"] = [availability(x, GLOBAL_SERIES["spx"]) for x in long["date"]]
    panel = pd.DataFrame({"symbol": "A", "decision_timestamp": [pd.Timestamp("2024-06-01", tz="UTC")]})
    assert np.isnan(attach_global_features(panel, long)["g_spx_r1"].iloc[0])


def test_news_no_lookahead_and_archive_mask():
    news = pd.DataFrame({"symbol": ["A", "A"], "published_at": ["2024-05-02 10:00", "2024-05-09 10:00"],
                         "text": ["Company A bags order worth 500 crore", "Company A faces SEBI penalty and fraud probe"]})
    sc = score_news_frame(news)
    assert sc["score"].iloc[0] > 0 > sc["score"].iloc[1]
    dec = pd.DataFrame({"symbol": "A", "decision_timestamp": pd.to_datetime(
        ["2024-05-01 10:30", "2024-05-03 10:30", "2024-05-08 10:30", "2024-05-10 10:30"], utc=True)})
    out = build_news_features(sc, dec, archive_start="2024-05-02")
    assert np.isnan(out["news_sent_ewm"].iloc[0])                 # before the archive existed: unknown
    assert out["news_cnt_1"].iloc[1] == 1 and out["news_sent_ewm"].iloc[1] > 0
    assert out["news_cnt_1"].iloc[2] == 0                          # 05-09 item not visible on 05-08
    assert out["news_sent_ewm"].iloc[3] < out["news_sent_ewm"].iloc[1]  # bad news pulls it down


def test_intraday_label_never_crosses_session():
    idx = pd.date_range("2024-05-06 09:15", "2024-05-06 15:30", freq="1min", tz="Asia/Kolkata")
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, len(idx))))
    bars = pd.DataFrame({"symbol": "A", "ts": idx.tz_convert("UTC"), "open": px, "high": px * 1.0005, "low": px * 0.9995, "close": px})
    bars2 = bars.assign(symbol="B", close=px[::-1].copy(), open=px[::-1].copy())
    p = build_intraday_panel(pd.concat([bars, bars2]), step_min=5, horizon_min=30)
    last = p["ts"].dt.tz_convert("Asia/Kolkata")
    assert (last.dt.hour * 60 + last.dt.minute).max() <= 15 * 60
    r = p[p["symbol"] == "A"].iloc[3]
    t = r["ts"]
    i = bars.index[bars["ts"] == t][0]
    assert np.isclose(r["fwd_ret"], np.log(px[i + 30] / px[i]))


def test_thresholds_use_wilson_and_abstain_on_noise():
    rng = np.random.default_rng(0)
    n = 20000
    pred = pd.DataFrame({"p": rng.uniform(0.3, 0.7, n), "y": rng.integers(0, 2, n)})
    assert choose_thresholds(pred)["abstain_all"] is True
    p = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < p).astype(int)           # perfectly calibrated => higher confidence is more accurate
    thr = choose_thresholds(pd.DataFrame({"p": p, "y": y}), target_precision=0.60)
    assert thr["abstain_all"] is False and thr["wilson_lb"] >= 0.60
    assert to_signal(0.2, {"abstain_all": True}) == "ABSTAIN"
    assert to_signal(thr["buy_p"] + 0.01, thr) == "BUY"
    assert to_signal(thr["buy_p"] + 0.01, thr, market_p=0.30) == "HOLD"   # market gate vetoes


def test_psi_detects_shift():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(0, 1, 5000)); b = pd.Series(rng.normal(0, 1, 1000)); c = pd.Series(rng.normal(1.5, 1, 1000))
    assert psi(a, b) < 0.1 and psi(a, c) > 0.25


def test_group_gating_ignores_thin_news():
    p = make_panel(0.0, n_sym=20, n_days=300)
    p["news_sent_ewm"] = np.nan
    p.loc[p["date"] >= p["date"].max() - pd.Timedelta(days=10), "news_sent_ewm"] = 0.1   # only 10 days of archive
    sel = select_groups(p)
    assert sel["groups"]["news"]["used"] is False


def test_alias_map_handles_nse_header_spacing():
    from selfdriving.sd_news_collect import build_alias_map, match_symbols
    eq = pd.DataFrame({"SYMBOL": ["HDFCBANK", "TCS"], " NAME OF COMPANY": ["HDFC Bank Limited", "Tata Consultancy Services Limited"]})
    m = build_alias_map(eq)
    assert match_symbols("HDFC Bank posts record profit", m) == ["HDFCBANK"]
    assert match_symbols("Tata Consultancy Services wins order", m) == ["TCS"]
