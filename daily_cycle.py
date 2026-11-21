#!/usr/bin/env python3
"""Nightly cycle: ingest the day, score yesterday's predictions against what happened,
retrain on everything captured, gate, publish a versioned model, and tell serving what
it is allowed to do tomorrow.

Target: ONE prediction per 5-minute bar, "will the 15:30 close be above the current price" (fr_close).
The ledger and champion use the single key "close" where the fixed-horizon version used +5..+30.

Order inside run_nightly (run it after 15:30 IST once the day's bars are final):
  1. ingest_day          validate + store the session's 5-minute bars (idempotent)
  2. evaluate_predictions  join logged predictions to realised prices -> live ledger
  3. retrain             frozen recipe, ALL completed sessions, walk-forward gate per horizon
  4. publish             bundle reload + smoke test, then atomic swap of champion.json
  5. serving_policy      LIVE / PAPER / HOLD per horizon (staleness, gate, live circuit breaker)

Design rules
  * Realised prices are the training labels. Logged predictions are never training targets
    (that would make the model learn from its own output). They drive evaluation, the
    circuit breaker and the train/serve parity audit.
  * The model is refit every night, but a refit is not automatically trusted: a horizon
    serves signals only while its gate passes and its live record is not degrading.
  * If the nightly job fails, the model ages. Serving must then fall to HOLD, not keep
    trading an old model (see serving_policy, max_lag_sessions).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import brier_score_loss, roc_auc_score

import pretrain_pipeline as P
from features_5m import FEATURES_5M, build_training_table_5m, make_features_5m, market_features_5m, prepare_fno

ALIASES = {
    "decision_timestamp": "decision_ts", "timestamp": "decision_ts", "slot": "decision_ts",
    "p_up": "probability_up", "prob_up": "probability_up",
    "probability_up_session_close": "probability_up", "expected_return_session_close": "expected_return",
    "expected_return_pct": "expected_return",
}
LEDGER_COLS = ["session", "horizon", "model_version", "n_ok", "n_no_actual", "n_pending", "auc", "brier",
               "brier_baseline", "n_trades", "sum_net", "gross_hits"]


# ----------------------------------------------------------------------------
# 1. ingest
# ----------------------------------------------------------------------------
def ingest_day(store: str | Path, day_bars: pd.DataFrame) -> dict:
    """Validate one day of bars with the same rules as training, then write its partition."""
    clean, rep = P.unify_bars(None, day_bars)
    if clean.empty:
        raise ValueError("Day produced no valid bars; store left untouched")
    for day, g in clean.groupby(clean["session"].dt.strftime("%Y-%m-%d")):
        P.append_day(store, day, g.drop(columns="session"))
    rep["days_written"] = int(clean["session"].nunique())
    return rep


# ----------------------------------------------------------------------------
# 2. evaluate logged predictions against realised prices
# ----------------------------------------------------------------------------
def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Required: symbol, decision_ts, probability_up. Optional: expected_return, signal, model_version,
    and the served FEATURES_5M values (for the parity audit). One row per (symbol, decision_ts)."""
    p = df.rename(columns={c: ALIASES.get(str(c).lower(), c) for c in df.columns}).copy()
    need = {"symbol", "decision_ts", "probability_up"}
    if need - set(p.columns):
        raise ValueError(f"Prediction log is missing columns: {sorted(need - set(p.columns))}")
    p["symbol"] = p["symbol"].astype(str).str.strip().str.upper()
    p["decision_ts"] = P._ist(p["decision_ts"])
    for c, d in (("expected_return", np.nan), ("signal", "HOLD"), ("model_version", "unknown")):
        if c not in p:
            p[c] = d
    return p.drop_duplicates(["symbol", "decision_ts"], keep="last").reset_index(drop=True)


def score_predictions(pred: pd.DataFrame, bars: pd.DataFrame, cost: float) -> pd.DataFrame:
    """One row per prediction. Realised outcome = Close[15:30 of the same session] / Close[decision bar] - 1.
    Status: OK, PENDING (15:30 bar not stored yet), NO_ACTUAL (a needed bar is missing)."""
    p = normalize_predictions(pred)
    px = bars.drop_duplicates(["symbol", "Timestamp"], keep="last").set_index(["symbol", "Timestamp"])["Close"]
    end_ts = p["decision_ts"].dt.floor("D") + pd.Timedelta(hours=15, minutes=30)
    ref = px.reindex(pd.MultiIndex.from_arrays([p["symbol"], p["decision_ts"]])).to_numpy()
    fut = px.reindex(pd.MultiIndex.from_arrays([p["symbol"], end_ts])).to_numpy()
    status = np.where(end_ts > bars["Timestamp"].max(), "PENDING",
             np.where(np.isnan(ref) | np.isnan(fut), "NO_ACTUAL", "OK"))
    p["status"] = status
    p["session"] = p["decision_ts"].dt.floor("D")
    p["horizon"] = "close"
    fr = np.where(status == "OK", fut / ref - 1.0, np.nan)
    p["fr"] = fr
    sig = p["signal"].astype(str).str.upper().to_numpy()
    ok = status == "OK"
    net = np.full(len(p), np.nan)
    net[(sig == "BUY") & ok] = fr[(sig == "BUY") & ok] - cost
    net[(sig == "SELL") & ok] = -fr[(sig == "SELL") & ok] - cost
    p["net"] = net
    return p


def ledger_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sess, g in scored.groupby("session"):
        ok = g[g["status"] == "OK"]
        y = (ok["fr"] > 0).astype(int).to_numpy()
        pr = ok["probability_up"].to_numpy(float)
        two = len(ok) > 200 and 0 < y.mean() < 1
        taken = ok["net"].dropna()
        direction = np.where(ok.loc[taken.index, "signal"].astype(str).str.upper() == "BUY", 1.0, -1.0)
        rows.append({
            "session": pd.Timestamp(sess).strftime("%Y-%m-%d"), "horizon": "close",
            "model_version": g["model_version"].mode().iat[0], "n_ok": len(ok),
            "n_no_actual": int((g["status"] == "NO_ACTUAL").sum()), "n_pending": int((g["status"] == "PENDING").sum()),
            "auc": float(roc_auc_score(y, pr)) if two else np.nan,
            "brier": float(brier_score_loss(y, pr)) if two else np.nan,
            "brier_baseline": float(brier_score_loss(y, np.full(len(y), 0.5))) if two else np.nan,     # no-skill 0.5; same-day base rate would be an oracle
            "n_trades": int(len(taken)), "sum_net": float(taken.sum()),
            "gross_hits": int((direction * ok.loc[taken.index, "fr"].to_numpy() > 0).sum()),
        })
    return pd.DataFrame(rows, columns=LEDGER_COLS)


LEDGER_REL = "ledger/live_eval.csv"


def read_ledger(store) -> pd.DataFrame:
    raw = store.read(LEDGER_REL)
    return pd.read_csv(io.BytesIO(raw)) if raw else pd.DataFrame(columns=LEDGER_COLS)


def update_ledger(store, new: pd.DataFrame) -> pd.DataFrame:
    old = read_ledger(store)
    if len(new):
        keys = set(zip(new["session"], new["horizon"]))
        old = old[[k not in keys for k in zip(old["session"], old["horizon"])]]
    out = pd.concat([old, new], ignore_index=True).sort_values(["horizon", "session"])
    store.write(LEDGER_REL, out.to_csv(index=False).encode())
    return out


def feature_parity(pred: pd.DataFrame, bars: pd.DataFrame, market: pd.DataFrame | None, fno: pd.DataFrame | None = None) -> dict:
    """If the prediction log carries the feature values used at serve time, recompute them from the
    stored bars with the shared module. Differences mean training and serving disagree."""
    p = normalize_predictions(pred)
    cols = [c for c in FEATURES_5M if c in p.columns]
    if not cols:
        return {"status": "NO_FEATURES_LOGGED"}
    mkt = market_features_5m(market) if market is not None and len(market) else None
    diffs = {c: [] for c in cols}
    for sym, g in p.drop_duplicates(["symbol", "decision_ts"]).groupby("symbol"):
        x = bars[bars["symbol"] == sym].set_index("Timestamp").sort_index()
        f = make_features_5m(x[["Open", "High", "Low", "Close", "Volume"]], mkt, prepare_fno(fno, sym))
        g = g.set_index("decision_ts")
        common = g.index.intersection(f.index)
        for c in cols:
            a, b = g.loc[common, c].astype(float), f.loc[common, c].astype(float)
            diffs[c].extend((np.abs(a - b) / (np.abs(b) + 1e-9)).dropna().tolist())
    bad = {c: float(np.mean(np.array(v) > 0.01)) for c, v in diffs.items() if v}
    return {"status": "OK", "features_compared": len(bad), "worst": sorted(bad.items(), key=lambda kv: -kv[1])[:5]}


# ----------------------------------------------------------------------------
# live circuit breaker and serving policy
# ----------------------------------------------------------------------------
def live_status(ledger: pd.DataFrame, horizon: str, window: int = 20, min_sessions: int = 10) -> dict:
    g = ledger[(ledger["horizon"] == horizon) & (ledger["n_trades"] > 0)].sort_values("session").tail(window)
    if len(g) < min_sessions:
        return {"state": "PAPER", "reason": f"only {len(g)} live sessions with trades (need {min_sessions})", "sessions": len(g)}
    per = (g["sum_net"] / g["n_trades"]).to_numpy()
    mean, se = per.mean(), per.std(ddof=1) / np.sqrt(len(per))
    if mean < 0:
        return {"state": "HALT", "reason": "live net expectancy below zero", "mean_net_bps": mean * 1e4, "sessions": len(g)}
    if len(g) >= window and mean - float(student_t.ppf(0.95, len(per) - 1)) * se > 0:
        return {"state": "LIVE_OK", "reason": "live net expectancy positive (one-sided 95%, t-quantile)", "mean_net_bps": mean * 1e4, "sessions": len(g)}
    return {"state": "PAPER", "reason": "live record positive but not yet conclusive", "mean_net_bps": mean * 1e4, "sessions": len(g)}


def serving_policy(champion: dict, session_dates: list[str], ledger: pd.DataFrame, max_lag_sessions: int = 1) -> dict:
    """What serving may do today. `session_dates` = every stored session date (YYYY-MM-DD)."""
    lag = sum(1 for d in session_dates if d > champion["trained_through"])
    out = {"model_version": champion["version"], "trained_through": champion["trained_through"],
           "lag_sessions": lag, "horizons": {}}
    for h, st in champion["horizons"].items():
        if lag > max_lag_sessions:
            mode, why = "HOLD", f"STALE: model is {lag} sessions behind the data"
        elif not st["active"]:
            mode, why = "HOLD", f"gate not active (pass_streak={st['pass_streak']}, last verdict {st['verdict']})"
        else:
            ls = live_status(ledger, h)
            mode = {"LIVE_OK": "LIVE", "PAPER": "PAPER", "HALT": "HOLD"}[ls["state"]]
            why = ls["reason"]
        out["horizons"][h] = {"mode": mode, "reason": why, "thresholds": st.get("thresholds")}
    return out


# ----------------------------------------------------------------------------
# champion storage (local or Drive-backed) and fail-closed loading
# ----------------------------------------------------------------------------
class LocalStore:
    """Directory-backed store. Writes go to a temp file then os.replace (atomic on one filesystem)."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self, rel: str) -> bytes | None:
        p = self.root / rel
        return p.read_bytes() if p.exists() else None

    def write(self, rel: str, data: bytes) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)

    def stamp(self, rel: str) -> str | None:
        p = self.root / rel
        return f"{p.stat().st_mtime_ns}-{p.stat().st_size}" if p.exists() else None


class DriveStore:
    """Adapter over YOUR existing Drive helpers; nothing here guesses your auth.
    download(rel) -> bytes | None ; upload(rel, data) -> None (create or overwrite) ;
    stat(rel) -> str | None (Drive md5Checksum or modifiedTime; None if missing).
    Drive overwrites are not atomic, so correctness does not depend on atomicity: the bundle is written
    first, read back and hash-checked, and champion.json is written LAST. Serving re-verifies the hash."""
    def __init__(self, download, upload, stat):
        self._dl, self._up, self._st = download, upload, stat

    def read(self, rel):
        return self._dl(rel)

    def write(self, rel, data):
        self._up(rel, data)

    def stamp(self, rel):
        return self._st(rel)


def make_drive_store(prefix: str = "production_data/all_nse_champion") -> "DriveStore":
    """DriveStore over the repo's own helpers (prediction_ledger._drive_file_bytes, persistent_store.GoogleDriveStore).
    Reads raise on transport errors and return None only for a missing file, so a Drive outage cannot look like an
    empty champion. The change stamp is modifiedTime + size (already returned by list_folder); integrity comes from the
    SHA-256 check, so no md5Checksum field change is needed. NOT exercised against live Drive (no credentials here)."""
    import tempfile
    from persistent_store import GoogleDriveStore
    from prediction_ledger import _drive_file_bytes
    cache: dict = {}

    def _store():
        if "g" not in cache:
            cache["g"] = GoogleDriveStore()
        return cache["g"]

    def download(rel):
        return _drive_file_bytes(f"{prefix}/{rel}")

    def upload(rel, data):
        mime = "application/json" if rel.endswith(".json") else "text/csv" if rel.endswith(".csv") else "application/octet-stream"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / Path(rel).name
            p.write_bytes(data)
            _store().upload_or_update(p, f"{prefix}/{rel}", mime)

    def stat(rel):
        parts = f"{prefix}/{rel}".split("/")
        g = _store()
        parent = g.ensure_path("/".join(parts[:-1]))
        for f in g.list_folder(parent):
            if f.get("name") == parts[-1]:
                return f"{f.get('modifiedTime')}-{f.get('size')}"
        return None

    return DriveStore(download, upload, stat)


def _hold_policy(reason: str) -> dict:
    return {"model_version": None, "trained_through": None, "lag_sessions": None,
            "horizons": {"close": {"mode": "HOLD", "reason": reason, "thresholds": None}}}


def load_champion_bundle(store):
    """Read and verify the champion. Returns (champion_dict, bundle, None) or (None, None, reason). Never raises."""
    try:
        raw = store.read("champion.json")
        if not raw:
            return None, None, "champion.json missing"
        champ = json.loads(raw)
        blob = store.read(champ["bundle_path"])
        if not blob:
            return None, None, "bundle missing"
        if hashlib.sha256(blob).hexdigest() != champ.get("bundle_sha256"):
            return None, None, "bundle hash mismatch"
        return champ, joblib.load(io.BytesIO(blob)), None
    except Exception as exc:                                         # noqa: BLE001 - serving must never raise
        return None, None, f"champion unreadable: {type(exc).__name__}"


def load_champion_for_serving(store, session_dates: list[str], ledger: pd.DataFrame, max_lag_sessions: int = 1):
    """Fail closed: any problem returns (None, HOLD policy, stamp). Serving calls store.stamp('champion.json')
    cheaply each cycle to learn whether a reload is needed."""
    stamp = store.stamp("champion.json")
    champ, bundle, err = load_champion_bundle(store)
    if err:
        return None, _hold_policy(err), stamp
    return bundle, serving_policy(champ, session_dates, ledger, max_lag_sessions), stamp


# ----------------------------------------------------------------------------
# 3-4. retrain, gate, publish
# ----------------------------------------------------------------------------
def _complete_enough(bars: pd.DataFrame, lookback: int = 10, ratio: float = 0.9) -> tuple[bool, str]:
    n = bars.groupby("session")["symbol"].nunique().sort_index()
    if len(n) < 2:
        return True, "first sessions"
    last, ref = n.iloc[-1], n.iloc[-1 - lookback:-1].median()
    return bool(last >= ratio * ref), f"latest session has {int(last)} symbols vs median {int(ref)}"


def retrain_and_publish(bars: pd.DataFrame, market: pd.DataFrame, fno: pd.DataFrame | None, store,
                        cfg: dict, previous: dict | None) -> dict:
    a = argparse.Namespace(**cfg)
    tbl = build_training_table_5m(bars, market, fno, stride=a.stride, warmup=a.warmup_bars)
    trained_through = bars["session"].max().strftime("%Y-%m-%d")
    version = f"v{trained_through}"
    z = P.prepare_target(tbl, FEATURES_5M)
    res, deploy = P.run_target(z, list(FEATURES_5M), a)

    base = (previous or {}).get("horizons", {}).get("close", {})
    if previous and previous.get("trained_through") == trained_through:
        base = base.get("streak_base", {})                    # same-day rerun: do not count the night twice
    old = {"pass_streak": base.get("pass_streak", 0), "fail_streak": base.get("fail_streak", 0),
           "active": base.get("active", False)}
    passed = res["verdict"] == "PASS"
    pass_streak = old["pass_streak"] + 1 if passed else 0
    # Hysteresis: one failed night turns the target off at once; it needs `activate_after` straight passes to return.
    active = bool(passed and (old["active"] or pass_streak >= a.activate_after))
    hz = {"close": {"verdict": res["verdict"], "pass_streak": pass_streak,
                    "fail_streak": 0 if passed else old["fail_streak"] + 1, "active": active,
                    "streak_base": old, "thresholds": deploy["thresholds"], "holdout_auc": res["holdout_auc"],
                    "holdout_auc_by_time_left": res["holdout_auc_by_time_left"],
                    "holdout_trading": res["holdout_trading"], "gate": res["gate"]}}

    buf = io.BytesIO()
    joblib.dump({"close": deploy}, buf)
    data = buf.getvalue()
    sha = hashlib.sha256(data).hexdigest()
    rel = f"versions/{version}/bundle.joblib"
    store.write(rel, data)
    back = store.read(rel)
    if back is None or hashlib.sha256(back).hexdigest() != sha:
        raise RuntimeError("Bundle read-back hash mismatch; champion pointer not changed")
    chk = joblib.load(io.BytesIO(back))["close"]
    p, er = P.predict(chk, tbl[FEATURES_5M].head(200).astype("float32"))
    if not (np.isfinite(p).all() and np.isfinite(er).all()):
        raise RuntimeError("Smoke test failed; champion pointer not changed")

    champion = {"version": version, "trained_through": trained_through, "bundle_path": rel, "bundle_sha256": sha,
                "created_at": datetime.now(timezone.utc).isoformat(), "feature_module": "features_5m",
                "target": "target_session_close", "feature_count": len(FEATURES_5M), "feature_warmup": a.warmup_bars,
                "signal_rule": {"use_edge": a.use_edge, "min_edge": a.cost if a.use_edge else None},
                "sessions_used": int(bars["session"].nunique()), "coverage": P.coverage_report(tbl), "horizons": hz}
    store.write("champion.json", json.dumps(champion, indent=2, default=str).encode())     # pointer written LAST
    return champion


def run_nightly(bars_store: str, market: pd.DataFrame, store, cfg: dict, predictions: pd.DataFrame | None = None,
                backfill: pd.DataFrame | None = None, fno: pd.DataFrame | None = None) -> dict:
    report: dict = {}
    bars, rep = P.unify_bars(backfill, P.load_store(bars_store), stamp=cfg.get("stamp", "close"))
    report["data"] = rep

    if predictions is not None and len(predictions):
        scored = score_predictions(predictions, bars, cfg["cost"])
        ledger = update_ledger(store, ledger_rows(scored))
        ok = scored[scored["status"] == "OK"]
        report["live_eval"] = {"predictions": int(len(scored)), "scored": int(len(ok)),
                               "no_actual_share": float((scored["status"] == "NO_ACTUAL").mean()),
                               "pending": int((scored["status"] == "PENDING").sum())}
        report["parity"] = feature_parity(predictions, bars, market, fno)
    else:
        ledger = read_ledger(store)

    raw = store.read("champion.json")
    previous = json.loads(raw) if raw else None
    ok, why = _complete_enough(bars)
    report["completeness"] = why
    if not ok:
        report["retrain"] = "SKIPPED: latest session incomplete; previous champion kept, staleness policy applies"
        champion = previous
    else:
        champion = retrain_and_publish(bars, market, fno, store, cfg, previous)
        report["retrain"] = f"published {champion['version']}"
    if champion:
        dates = sorted(bars["session"].dt.strftime("%Y-%m-%d").unique())
        report["policy"] = serving_policy(champion, dates, ledger, cfg.get("max_lag_sessions", 1))
    store.write("nightly_report.json", json.dumps(report, indent=2, default=str).encode())
    return report


def default_cfg(**kw) -> dict:
    cfg = dict(stride=3, max_train_rows=2_500_000, n_splits=5, min_train_sessions=100, holdout_sessions=40,
               half_life=None, cost=0.0005, min_trades_cv=1500, min_trades_holdout=500,
               min_trade_sessions_cv=20, min_trade_sessions_holdout=24, activate_after=3, max_lag_sessions=1,
               stamp="close", warmup_bars=12, use_edge=True)
    cfg.update(kw)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True, help="local directory of accumulated 5m bars")
    ap.add_argument("--models", required=True, help="local champion directory (use DriveStore in code for Drive)")
    ap.add_argument("--market-parquet", required=True)
    ap.add_argument("--fno-parquet")
    ap.add_argument("--backfill-parquet")
    ap.add_argument("--predictions", help="parquet/csv of logged predictions")
    ap.add_argument("--cost", type=float, default=0.0005, help="ASSUMED round-trip cost; replace with your real figure")
    a = ap.parse_args()
    preds = None
    if a.predictions:
        preds = pd.read_parquet(a.predictions) if a.predictions.endswith(".parquet") else pd.read_csv(a.predictions)
    bf = pd.read_parquet(a.backfill_parquet) if a.backfill_parquet else None
    fno = pd.read_parquet(a.fno_parquet) if a.fno_parquet else None
    rep = run_nightly(a.store, P.load_market(a.market_parquet), LocalStore(a.models), default_cfg(cost=a.cost), preds, bf, fno)
    print(json.dumps(rep.get("policy", rep), indent=2, default=str))


if __name__ == "__main__":
    main()
