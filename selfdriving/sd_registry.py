"""Champion/challenger registry, promotion gate, drift and live-accuracy monitors.

A new model never replaces the live one on a single good number. It must beat the
champion on identical walk-forward folds with a confidence interval that excludes zero,
and keep calibration and IC sane. The previous champion is always retained for rollback.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .sd_walkforward import Bundle, paired_logloss_gain, wilson_lower


@dataclass
class GateRules:
    min_dates: int = 60
    min_auc: float = 0.505            # challenger AUC must be above chance by this much
    min_ic_t: float = 2.0
    min_pos_fold_frac: float = 0.6    # share of folds with AUC > 0.5
    max_brier_worsen: float = 0.0005
    require_ci_excludes_zero: bool = True


def gate(champion_metrics: Optional[Dict], challenger_metrics: Dict, paired: Optional[Dict],
         rules: GateRules = GateRules()) -> Dict:
    reasons: List[str] = []
    m = challenger_metrics
    if m["dates"] < rules.min_dates:
        reasons.append(f"only {m['dates']} test dates (< {rules.min_dates})")
    if not (m["auc"] >= rules.min_auc):
        reasons.append(f"AUC {m['auc']:.4f} < {rules.min_auc}")
    if not (m["ic_t"] >= rules.min_ic_t):
        reasons.append(f"IC t-stat {m['ic_t']:.2f} < {rules.min_ic_t}")
    folds = m.get("fold_auc", [])
    if folds and np.mean(np.array(folds) > 0.5) < rules.min_pos_fold_frac:
        reasons.append("too few folds above AUC 0.5")
    if champion_metrics is not None:
        if m["brier"] > champion_metrics["brier"] + rules.max_brier_worsen:
            reasons.append("Brier worse than champion")
        if paired is not None:
            lo = paired["ci95"][0]
            if rules.require_ci_excludes_zero and not lo > 0:
                reasons.append(f"paired log-loss gain CI [{paired['ci95'][0]:.5f}, {paired['ci95'][1]:.5f}] includes 0")
        else:
            reasons.append("no paired comparison against champion")
    return {"promote": len(reasons) == 0, "reasons": reasons or ["all gates passed"]}


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {"champion": None, "history": []}

    def _save(self):
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def champion_bundle(self) -> Optional[Bundle]:
        c = self.state["champion"]
        return joblib.load(self.root / c["file"]) if c else None

    def register(self, name: str, bundle: Bundle, metrics: Dict, decision: Dict, thresholds: Dict) -> Dict:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        fn = f"{name}_{stamp}.joblib"
        joblib.dump(bundle, self.root / fn)
        slim = {k: v for k, v in metrics.items() if k != "per_date_hit"}
        entry = {"file": fn, "name": name, "created": stamp, "metrics": slim, "thresholds": thresholds,
                 "gate": decision, "meta": bundle.meta}
        self.state["history"].append({**entry, "status": "PROMOTED" if decision["promote"] else "REJECTED"})
        if decision["promote"]:
            self.state["champion"] = entry
        self._save()
        for old in self.state["history"][:-6]:       # keep the last 6 artifacts for rollback
            p = self.root / old["file"]
            if p.exists() and self.state["champion"] and old["file"] != self.state["champion"]["file"]:
                p.unlink()
        return entry

    def rollback(self) -> Optional[Dict]:
        prev = [h for h in self.state["history"] if h["status"] == "PROMOTED" and (self.root / h["file"]).exists()]
        cur = self.state["champion"]
        prev = [h for h in prev if not cur or h["file"] != cur["file"]]
        if not prev:
            return None
        self.state["champion"] = prev[-1]
        self._save()
        return prev[-1]


# ------------------------------------------------------------------ monitors
def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e, a = expected.dropna().to_numpy(float), actual.dropna().to_numpy(float)
    if len(e) < 200 or len(a) < 50:
        return np.nan
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, 1e-4, None), np.clip(pa, 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_report(train: pd.DataFrame, recent: pd.DataFrame, features: List[str], warn: float = 0.1, alarm: float = 0.25) -> Dict:
    vals = {f: psi(train[f], recent[f]) for f in features if f in train and f in recent}
    vals = {k: v for k, v in vals.items() if not np.isnan(v)}
    return {"n_alarm": int(sum(v >= alarm for v in vals.values())), "n_warn": int(sum(alarm > v >= warn for v in vals.values())),
            "worst": sorted(vals.items(), key=lambda kv: -kv[1])[:8]}


def live_check(ledger: pd.DataFrame, thresholds: Dict, window: int = 60, min_signals: int = 40) -> Dict:
    """ledger: rows with date, p (at prediction time), y (realised 0/1, NaN until known).

    Compares realised precision of acted-on signals with the validated expectation.
    Flags retraining when the recent Wilson UPPER-ish bound falls clearly below expectation.
    """
    if thresholds.get("abstain_all", True) or ledger.empty:
        return {"status": "NO_ACTIVE_BAND", "retrain": False}
    L = ledger.dropna(subset=["y"]).sort_values("date")
    dates = L["date"].drop_duplicates().tail(window)
    L = L[L["date"].isin(dates)]
    act = L[(L["p"] >= thresholds["buy_p"]) | (L["p"] <= thresholds["sell_p"])]
    if len(act) < min_signals:
        return {"status": "INSUFFICIENT_SIGNALS", "n": int(len(act)), "retrain": False}
    hit = ((act["p"] >= 0.5).astype(int) == act["y"].astype(int))
    k, n = int(hit.sum()), int(len(hit))
    lb = wilson_lower(k, n)
    expected = thresholds["oof_accuracy"]
    degraded = (k / n) < expected - 0.06 and lb < thresholds["wilson_lb"] - 0.03
    return {"status": "DEGRADED" if degraded else "OK", "n": n, "live_accuracy": k / n, "live_wilson_lb": lb,
            "expected_accuracy": expected, "retrain": bool(degraded)}
