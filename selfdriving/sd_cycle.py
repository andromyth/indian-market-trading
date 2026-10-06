"""Daily/weekly self-driving cycle.

Stages (each is a plain function so Colab, GitHub Actions and tests share them):
  train_stage    build challenger, compare with champion on identical folds, gate, register
  predict_stage  champion + validated thresholds -> BUY/SELL/HOLD/ABSTAIN rows
  validate_stage fill realised outcomes, live accuracy vs expectation, drift, retrain flag

Panel contract (parquet written by the existing Colab builder, see INTEGRATION.md):
  symbol, date, decision_timestamp (UTC), y_return (next-session close/close - 1), + features.
Inference panel: same columns without labels, one row per symbol for the latest session.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .sd_registry import GateRules, Registry, drift_report, gate, live_check
from .sd_walkforward import (FoldSpec, ModelConfig, add_cross_sectional_ranks, add_targets,
                             choose_thresholds, evaluate, paired_logloss_gain, run_walkforward, to_signal)

PRICE_BASE = ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_48", "rsi14", "atr14", "ma20_ratio", "ma50_ratio",
              "volume_ratio", "volume_z20", "delivery_pct", "delivery_pct_change_1", "delivery_pct_ratio20",
              "delivery_pct_z20", "delivery_qty_ratio20", "delivery_qty_z20", "range_pct", "body_pct", "up_bar",
              "market_ret_1", "market_ret_3", "market_ret_6", "market_rsi14", "relative_ret_1", "relative_ret_6",
              "relative_ret_12"]
RANK_SOURCE = ["ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "volume_ratio", "delivery_pct_z20", "rsi14", "atr14"]


def ensure_delivery_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Create causal next-session return labels from symbol/date/close when absent."""
    x = panel.copy()
    if "symbol" not in x.columns or "date" not in x.columns or "close" not in x.columns:
        raise ValueError("Delivery panel requires symbol, date and close")

    x["symbol"] = x["symbol"].astype(str).str.upper().str.strip()
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    x["close"] = pd.to_numeric(x["close"], errors="coerce")
    x = (
        x.dropna(subset=["symbol", "date", "close"])
         .sort_values(["symbol", "date"])
         .drop_duplicates(["symbol", "date"], keep="last")
         .reset_index(drop=True)
    )

    # If a trusted upstream y_return exists, retain it; otherwise derive it
    # strictly from the next observed trading session for the same symbol.
    if "y_return" not in x.columns:
        x["y_return"] = (
            x.groupby("symbol", sort=False)["close"].shift(-1) / x["close"] - 1.0
        )
    x["y_return"] = pd.to_numeric(x["y_return"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )

    # The final session per symbol has no realised next-session label yet and
    # must never enter training.
    return x


def feature_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = list(df.columns)
    return {
        "price": [c for c in PRICE_BASE if c in cols] + [c for c in cols if c.endswith("_csr")],
        "global": [c for c in cols if c.startswith("g_") and not c.endswith("_age_h")],
        "news": [c for c in cols if c.startswith(("news_", "mkt_news_"))],
        "fno": [c for c in cols if c.startswith("fno_")],
        "fundamental": [c for c in cols if c.startswith(("fundamental_", "fund_"))],
    }


def select_groups(df: pd.DataFrame, min_cov: float = 0.6, min_dates: Dict[str, int] | None = None,
                  recent_days: int = 252) -> Dict:
    """Include a group only when it is actually populated in the recent window.

    This is what stops an empty or thin source (e.g. a news archive that started last week)
    from adding noise, and lets it switch itself on automatically once coverage is real.
    """
    min_dates = min_dates or {"news": 120, "global": 120, "fno": 60}
    d = pd.to_datetime(df["date"]).dt.normalize()
    recent = df[d >= d.max() - pd.Timedelta(days=recent_days * 1.5)]
    chosen, report = [], {}
    for g, cols in feature_groups(df).items():
        if not cols:
            report[g] = {"used": False, "why": "no columns"}
            continue
        cov = float(recent[cols].notna().mean().mean())
        known_dates = int(recent.loc[recent[cols].notna().any(axis=1), "date"].nunique())
        need = min_dates.get(g, 30)
        ok = cov >= min_cov and known_dates >= need
        report[g] = {"used": ok, "coverage": round(cov, 3), "dates_with_data": known_dates, "need_dates": need}
        if ok:
            chosen += cols
    return {"features": chosen, "groups": report}


def train_stage(panel: pd.DataFrame, registry_dir: str | Path, spec: Optional[FoldSpec] = None,
                rules: GateRules = GateRules(), cost_bps: float = 30.0, target_precision: float = 0.56,
                cfg: Optional[ModelConfig] = None) -> Dict:
    spec = spec or FoldSpec()
    p = ensure_delivery_targets(panel)
    p = add_targets(p)
    p = add_cross_sectional_ranks(p, RANK_SOURCE)
    sel = select_groups(p)
    if not sel["features"]:
        return {"status": "NO_FEATURES", "groups": sel["groups"]}
    reg = Registry(registry_dir)

    pred_c = run_walkforward(p, sel["features"], "y_up_excess", "y_return", cfg, spec)
    m_c = evaluate(pred_c, cost_bps)
    thr = choose_thresholds(pred_c, target_precision)

    champ = reg.champion_bundle()
    champ_pred, champ_m, paired = None, None, None
    if champ is not None:
        cc = ModelConfig(**{**{"name": champ.meta["cfg"]["name"]}, "hgb_params": champ.meta["cfg"]["hgb_params"],
                            "logit_C": champ.meta["cfg"]["logit_C"], "val_frac": champ.meta["cfg"]["val_frac"]})
        try:
            champ_pred = run_walkforward(p, champ.features, "y_up_excess", "y_return", cc, spec)
            champ_m = evaluate(champ_pred, cost_bps)
            paired = paired_logloss_gain(champ_pred, pred_c)
        except Exception as e:  # champion features no longer in panel, etc.
            champ_m = None
            paired = {
                "status": "UNAVAILABLE",
                "reason": f"{type(e).__name__}: {e}",
            }

    decision = gate(champ_m, m_c, paired if champ_m is not None else None, rules)
    if decision["promote"]:
        from .sd_walkforward import fit_bundle
        final = fit_bundle(p.dropna(subset=["y_up_excess"]), sel["features"], "y_up_excess", cfg or ModelConfig())
    else:
        final = None
    entry = None
    if final is not None:
        entry = reg.register("delivery_excess", final, m_c, decision, thr)
    else:
        # record the rejected attempt (needs a bundle object to register metadata; store a lightweight record)
        reg.state["history"].append({"name": "delivery_excess", "status": "REJECTED", "gate": decision,
                                     "metrics": {k: v for k, v in m_c.items() if k != "per_date_hit"}})
        reg._save()
    return {"status": "PROMOTED" if decision["promote"] else "REJECTED", "gate": decision, "groups": sel["groups"],
            "thresholds": thr, "challenger": {k: v for k, v in m_c.items() if k not in ("per_date_hit",)},
            "paired_vs_champion": paired}


def predict_stage(inference_rows: pd.DataFrame, registry_dir: str | Path, market_p: Optional[float] = None) -> pd.DataFrame:
    reg = Registry(registry_dir)
    b = reg.champion_bundle()
    if b is None:
        raise RuntimeError("No champion model registered; run train_stage first")
    thr = reg.state["champion"]["thresholds"]
    x = add_cross_sectional_ranks(inference_rows, RANK_SOURCE)
    miss = [f for f in b.features if f not in x.columns]
    if miss:
        raise RuntimeError(f"Inference panel missing champion features: {miss[:6]}")
    out = x[["symbol", "date"]].copy()
    out["p"] = b.predict_proba_up(x)
    out["signal"] = [to_signal(v, thr, market_p) for v in out["p"]]
    out["model_file"] = reg.state["champion"]["file"]
    out["feature_coverage"] = x[b.features].notna().mean(axis=1).to_numpy()
    out["signal_reason"] = "VALIDATED_THRESHOLD"
    low_cov = out["feature_coverage"] < 0.6
    out.loc[low_cov, "signal"] = "ABSTAIN"
    out.loc[low_cov, "signal_reason"] = "INSUFFICIENT_FEATURE_COVERAGE"
    if thr.get("abstain_all", True):
        out["signal"] = "ABSTAIN"
        out["signal_reason"] = "NO_VALIDATED_ACTION_BAND"
    return out


def validate_stage(ledger: pd.DataFrame, realised: pd.DataFrame, registry_dir: str | Path,
                   train_ref: Optional[pd.DataFrame] = None, recent: Optional[pd.DataFrame] = None) -> Dict:
    """ledger: symbol,date,p. realised: symbol,date,y_return (same-session next close/close)."""
    reg = Registry(registry_dir)
    thr = reg.state["champion"]["thresholds"] if reg.state["champion"] else {"abstain_all": True}
    L = ledger.merge(realised[["symbol", "date", "y_return"]], on=["symbol", "date"], how="left")
    med = L.groupby("date")["y_return"].transform("median")
    L["y"] = np.where(L["y_return"].notna(), (L["y_return"] > med).astype(float), np.nan)
    res = {"live": live_check(L, thr)}
    if train_ref is not None and recent is not None and reg.state["champion"]:
        res["drift"] = drift_report(train_ref, recent, reg.champion_bundle().features)
        res["retrain"] = bool(res["live"].get("retrain") or res["drift"]["n_alarm"] >= 3)
    else:
        res["retrain"] = bool(res["live"].get("retrain"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "predict", "validate"], required=True)
    ap.add_argument("--root", default="/content/drive/MyDrive/Indian_Market_Research/selfdriving")
    ap.add_argument("--panel", help="parquet panel (train) or inference rows (predict)")
    ap.add_argument("--ledger"); ap.add_argument("--realised")
    a = ap.parse_args()
    root = Path(a.root)
    if a.stage == "train":
        out = train_stage(pd.read_parquet(a.panel), root / "models")
        (root / "status_train.json").write_text(json.dumps(out, indent=2, default=str))
        print(json.dumps({k: out[k] for k in ("status", "gate", "thresholds") if k in out}, indent=2, default=str))
    elif a.stage == "predict":
        out = predict_stage(pd.read_parquet(a.panel), root / "models")
        p = root / "ledger" / "predictions.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            out = pd.concat([pd.read_parquet(p), out]).drop_duplicates(["symbol", "date"], keep="last")
        out.to_parquet(p, index=False)
        print(out["signal"].value_counts().to_string())
    else:
        out = validate_stage(pd.read_parquet(a.ledger), pd.read_parquet(a.realised), root / "models")
        (root / "status_validate.json").write_text(json.dumps(out, indent=2, default=str))
        print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
