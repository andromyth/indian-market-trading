"""
calibration_autopilot.py
Unified outcome analyzer, SHADOW/ACTIVE calibration lifecycle manager, AND
self-evolving online intelligence that updates price-volume-event expert weights
and self-generates executable Python policy code (`generated_alpha_policy.py`).
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from prediction_ledger import IST_TZ


@dataclass
class Rules:
    shadow_min_dates: int = 7
    shadow_alpha: float = 0.05
    demote_min_dates: int = 5
    cooldown_days: int = 3
    valid_for_days: int = 30
    refit_after_days: int = 7
    min_effective_sample_size: float = 350.0
    volatility_scale_bounds: tuple[float, float] = (1.0, 1.75)
    prob_shift_bounds: tuple[float, float] = (-0.08, 0.08)
    psi_warning_threshold: float = 0.20
    psi_critical_threshold: float = 0.35
    ph_delta: float = 0.05
    ph_lambda: float = 15.0
    enable_cross_horizon_smoothing: bool = True


@dataclass
class EvolvedRule:
    """Represents a self-discovered microstructure conditioning rule."""
    rule_id: str
    condition_expr: str
    action_type: str
    multiplier: float
    target_horizons: list[int]
    sample_support: int
    historical_edge_gain_bps: float


@dataclass
class IntelligenceBrainState:
    version: int = 1
    updated_at: str = ""
    learning_rate: float = 0.08
    expert_weights: dict[str, float] = field(
        default_factory=lambda: {
            "momentum_price_expert": 0.25,
            "volume_ignition_expert": 0.25,
            "vwap_reversion_expert": 0.25,
            "order_flow_imbalance_expert": 0.25,
        }
    )
    evolved_rules: list[dict[str, Any]] = field(default_factory=list)


def compute_effective_sample_size(df: pd.DataFrame) -> float:
    """Adjust sample size for overlapping 1-minute forecast horizons."""
    if df.empty:
        return 0.0
    if "horizon_min" not in df.columns:
        return float(len(df))

    n_eff = 0.0
    for horizon, grp in df.groupby("horizon_min"):
        h_val = max(1.0, float(horizon))
        n_eff += len(grp) / h_val
    return float(n_eff)


def _count_distinct_trading_dates(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    ts_col = "decision_timestamp" if "decision_timestamp" in df.columns else "timestamp"
    if ts_col not in df.columns:
        return 1
    dates = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(IST_TZ).dt.date
    return int(dates.nunique())


def evaluate_probability_candidates(
    df: pd.DataFrame,
    key: str,
    rules: Rules | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Evaluate SHADOW probability calibration candidates using effective sample size."""
    cfg = rules or Rules()
    if df.empty or "probability_up" not in df.columns:
        return {"key": key, "status": "INSUFFICIENT_DATA", "promote": False}, None

    n_dates = _count_distinct_trading_dates(df)
    n_eff = compute_effective_sample_size(df)

    actual_up = (df["actual_close"] > df["reference_close"]).astype(float)
    raw_prob = np.clip(pd.to_numeric(df["probability_up"], errors="coerce").fillna(0.5), 1e-4, 1.0 - 1e-4)

    emp_rate = float(actual_up.mean())
    pred_rate = float(raw_prob.mean())
    raw_shift = float(np.clip(emp_rate - pred_rate, cfg.prob_shift_bounds[0], cfg.prob_shift_bounds[1]))
    cal_prob = np.clip(raw_prob + raw_shift, 1e-4, 1.0 - 1e-4)

    raw_brier = float(np.mean((raw_prob - actual_up) ** 2))
    cal_brier = float(np.mean((cal_prob - actual_up) ** 2))
    brier_gain = raw_brier - cal_brier

    pointwise_diff = ((raw_prob - actual_up) ** 2) - ((cal_prob - actual_up) ** 2)
    se_gain = float(np.std(pointwise_diff, ddof=1) / np.sqrt(max(1.0, n_eff))) if len(df) > 1 else 0.0
    lcb_gain = brier_gain - 1.645 * se_gain

    promote = (
        n_dates >= cfg.shadow_min_dates
        and n_eff >= cfg.min_effective_sample_size
        and lcb_gain > 0.0
    )

    result = {
        "key": key,
        "status": "ACTIVE" if promote else "SHADOW",
        "promote": bool(promote),
        "n_dates": n_dates,
        "n_eff": round(n_eff, 2),
        "raw_brier": round(raw_brier, 6),
        "calibrated_brier": round(cal_brier, 6),
        "lcb_brier_gain": round(lcb_gain, 6),
    }

    cal_info = {
        "key": key,
        "prob_shift": round(raw_shift, 5),
        "valid_for_days": cfg.valid_for_days,
        "refit_after_days": cfg.refit_after_days,
    } if promote else None

    return result, cal_info


def evaluate_return_candidates(
    df: pd.DataFrame,
    key: str,
    rules: Rules | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Evaluate return-head calibration candidates with ATR-conditioned volatility de-shrinkage."""
    cfg = rules or Rules()
    required = {"reference_close", "predicted_price", "actual_close"}
    if df.empty or not required.issubset(df.columns):
        return {"key": key, "status": "INSUFFICIENT_DATA", "promote": False}, None

    work = df.dropna(subset=list(required)).copy()
    work = work[work["reference_close"] > 0].copy()
    if work.empty:
        return {"key": key, "status": "INSUFFICIENT_DATA", "promote": False}, None

    n_dates = _count_distinct_trading_dates(work)
    n_eff = compute_effective_sample_size(work)

    pred_ret = (work["predicted_price"] - work["reference_close"]) / work["reference_close"]
    actual_ret = (work["actual_close"] - work["reference_close"]) / work["reference_close"]

    std_pred = float(np.std(pred_ret, ddof=1)) if len(work) > 1 else 1e-6
    std_actual = float(np.std(actual_ret, ddof=1)) if len(work) > 1 else 1e-6
    raw_vol_ratio = std_actual / max(std_pred, 1e-6)

    med_pred = float(np.median(np.abs(pred_ret)))
    med_actual = float(np.median(np.abs(actual_ret)))
    mad_ratio = med_actual / max(med_pred, 1e-6)
    blended_scale = 0.5 * raw_vol_ratio + 0.5 * mad_ratio
    vol_scale = float(np.clip(blended_scale, cfg.volatility_scale_bounds[0], cfg.volatility_scale_bounds[1]))

    regime_scales: dict[str, float] = {"LOW_VOL": vol_scale, "HIGH_VOL": vol_scale}
    if "atr_15m" in work.columns and work["atr_15m"].notna().sum() > 20:
        atr_pct = work["atr_15m"] / work["reference_close"]
        med_atr = float(atr_pct.median())
        for label, mask in [("LOW_VOL", atr_pct <= med_atr), ("HIGH_VOL", atr_pct > med_atr)]:
            sub_pred = pred_ret[mask]
            sub_act = actual_ret[mask]
            if len(sub_pred) > 10:
                s_ratio = float(np.median(np.abs(sub_act)) / max(np.median(np.abs(sub_pred)), 1e-6))
                regime_scales[label] = float(np.clip(s_ratio, cfg.volatility_scale_bounds[0], cfg.volatility_scale_bounds[1]))

    scaled_ret = pred_ret * vol_scale
    calibrated_price = work["reference_close"] * (1.0 + scaled_ret)

    raw_mae = float(np.mean(np.abs(work["predicted_price"] - work["actual_close"])))
    cal_mae = float(np.mean(np.abs(calibrated_price - work["actual_close"])))

    raw_var_gap = abs(std_actual - std_pred)
    cal_var_gap = abs(std_actual - float(np.std(scaled_ret, ddof=1)))
    variance_improved = cal_var_gap < raw_var_gap
    mae_non_degraded = cal_mae <= raw_mae * 1.03

    promote = (
        n_dates >= cfg.shadow_min_dates
        and n_eff >= cfg.min_effective_sample_size
        and variance_improved
        and mae_non_degraded
    )

    result = {
        "key": key,
        "status": "ACTIVE" if promote else "SHADOW",
        "promote": bool(promote),
        "n_dates": n_dates,
        "n_eff": round(n_eff, 2),
        "volatility_scale": round(vol_scale, 4),
        "regime_scales": {k: round(v, 4) for k, v in regime_scales.items()},
        "raw_mae_inr": round(raw_mae, 4),
        "calibrated_mae_inr": round(cal_mae, 4),
    }

    cal_info = {
        "key": key,
        "volatility_scale": round(vol_scale, 4),
        "regime_scales": {k: round(v, 4) for k, v in regime_scales.items()},
        "valid_for_days": cfg.valid_for_days,
        "refit_after_days": cfg.refit_after_days,
    } if promote else None

    return result, cal_info


def apply_cross_horizon_coherence(
    predictions_by_horizon: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Smooth internal trajectory inversions across minute-level horizons (5m..30m)."""
    smoothed = {h: dict(payload) for h, payload in predictions_by_horizon.items()}
    ordered_horizons = sorted(smoothed.keys())

    for idx in range(1, len(ordered_horizons) - 1):
        h_prev = ordered_horizons[idx - 1]
        h_curr = ordered_horizons[idx]
        h_next = ordered_horizons[idx + 1]

        prev_item = smoothed[h_prev]
        curr_item = smoothed[h_curr]
        next_item = smoothed[h_next]

        ref_close = float(curr_item.get("reference_close", 0.0))
        if ref_close <= 0.0:
            continue

        ret_prev = (float(prev_item["predicted_price"]) - ref_close) / ref_close
        ret_curr = (float(curr_item["predicted_price"]) - ref_close) / ref_close
        ret_next = (float(next_item["predicted_price"]) - ref_close) / ref_close

        if np.sign(ret_prev) == np.sign(ret_next) and np.sign(ret_prev) != 0 and np.sign(ret_curr) != np.sign(ret_prev):
            interp_ret = 0.5 * (ret_prev + ret_next)
            curr_item["predicted_price"] = round(ref_close * (1.0 + interp_ret), 4)

            if "probability_up" in curr_item:
                raw_delta = float(curr_item["probability_up"]) - 0.5
                neighbor_prob = 0.5 * (
                    float(prev_item.get("probability_up", 0.5))
                    + float(next_item.get("probability_up", 0.5))
                )
                curr_item["probability_up"] = round(0.5 + 0.5 * (neighbor_prob - 0.5) + 0.25 * raw_delta, 4)

            curr_item["coherence_adjusted"] = True

    return smoothed


def _compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    if len(expected) < buckets or len(actual) < buckets:
        return 0.0
    quantiles = np.linspace(0, 100, buckets + 1)
    bins = np.unique(np.percentile(expected, quantiles))
    if len(bins) < 3:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)

    exp_pct = np.clip(exp_counts / max(1, len(expected)), 1e-4, 1.0)
    act_pct = np.clip(act_counts / max(1, len(actual)), 1e-4, 1.0)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _assign_intraday_session_bucket(ts_series: pd.Series) -> pd.Series:
    ist_times = pd.to_datetime(ts_series, utc=True).dt.tz_convert(IST_TZ).dt.time
    buckets = []
    for t in ist_times:
        if t < time(10, 15, 0):
            buckets.append("OPEN")
        elif t < time(14, 15, 0):
            buckets.append("MID")
        else:
            buckets.append("CLOSE")
    return pd.Series(buckets, index=ts_series.index)


def _volatility_normalized_page_hinkley(
    residuals_inr: np.ndarray,
    atr_series: np.ndarray,
    delta: float = 0.05,
    threshold_lambda: float = 15.0,
) -> tuple[bool, float]:
    if len(residuals_inr) < 10:
        return False, 0.0

    safe_atr = np.where(np.isfinite(atr_series) & (atr_series > 1e-4), atr_series, np.nanmedian(residuals_inr) + 1e-4)
    norm_err = residuals_inr / safe_atr

    mean_val = 0.0
    cum_sum = 0.0
    min_cum_sum = 0.0
    max_ph_stat = 0.0

    for idx, x in enumerate(norm_err, start=1):
        mean_val = mean_val + (x - mean_val) / idx
        cum_sum = cum_sum + (x - mean_val - delta)
        min_cum_sum = min(min_cum_sum, cum_sum)
        ph_stat = cum_sum - min_cum_sum
        max_ph_stat = max(max_ph_stat, ph_stat)

    return bool(max_ph_stat > threshold_lambda), float(max_ph_stat)


def live_monitor(
    df: pd.DataFrame,
    profile_id: str | Iterable[str],
    rules: Rules | None = None,
    key: str = "",
) -> dict[str, Any]:
    """Monitor live ACTIVE/SHADOW profiles using stratified PSI and ATR-normalized Page-Hinkley."""
    cfg = rules or Rules()
    target_profiles = {profile_id} if isinstance(profile_id, str) else set(profile_id)

    if df.empty:
        return {"key": key, "action": "HOLD", "drift_detected": False, "reason": "EMPTY_DATA"}

    work = df.copy()
    if "calibration_profile_id" in work.columns and target_profiles:
        work = work[work["calibration_profile_id"].isin(target_profiles)].copy()

    if work.empty:
        return {"key": key, "action": "HOLD", "drift_detected": False, "reason": "NO_MATCHING_PROFILE"}

    n_dates = _count_distinct_trading_dates(work)
    residuals = (work["predicted_price"] - work["actual_close"]).abs().to_numpy(dtype=float)

    if "atr_15m" in work.columns and work["atr_15m"].notna().any():
        atr_vals = pd.to_numeric(work["atr_15m"], errors="coerce").bfill().ffill().to_numpy(dtype=float)
    else:
        fallback_vol = (work["actual_close"] - work["reference_close"]).abs().rolling(15, min_periods=1).mean()
        atr_vals = fallback_vol.bfill().fillna(1.0).to_numpy(dtype=float)

    ph_drift, ph_stat = _volatility_normalized_page_hinkley(
        residuals,
        atr_vals,
        delta=cfg.ph_delta,
        threshold_lambda=cfg.ph_lambda,
    )

    pred_ret = ((work["predicted_price"] - work["reference_close"]) / work["reference_close"]).to_numpy(dtype=float)
    ts_col = "decision_timestamp" if "decision_timestamp" in work.columns else "timestamp"
    bucket_psi: dict[str, float] = {}

    if ts_col in work.columns and len(work) >= 40:
        work["tod_bucket"] = _assign_intraday_session_bucket(work[ts_col])
        for b_name, grp in work.groupby("tod_bucket"):
            vals = ((grp["predicted_price"] - grp["reference_close"]) / grp["reference_close"]).to_numpy(dtype=float)
            if len(vals) >= 20:
                mid = len(vals) // 2
                bucket_psi[str(b_name)] = round(_compute_psi(vals[:mid], vals[mid:]), 4)
    else:
        mid = len(pred_ret) // 2
        if mid >= 10:
            bucket_psi["ALL"] = round(_compute_psi(pred_ret[:mid], pred_ret[mid:]), 4)

    max_psi = max(bucket_psi.values()) if bucket_psi else 0.0
    psi_critical = max_psi >= cfg.psi_critical_threshold

    degraded_vs_raw = False
    if "raw_predicted_price" in work.columns:
        raw_err = (work["raw_predicted_price"] - work["actual_close"]).abs().mean()
        cal_err = (work["predicted_price"] - work["actual_close"]).abs().mean()
        degraded_vs_raw = bool(cal_err > raw_err * 1.02)

    should_demote = n_dates >= cfg.demote_min_dates and (
        (ph_drift and psi_critical) or degraded_vs_raw
    )

    return {
        "key": key,
        "profiles_evaluated": sorted(target_profiles),
        "n_dates": n_dates,
        "action": "DEMOTE_TO_SHADOW" if should_demote else "HOLD",
        "drift_detected": bool(ph_drift or psi_critical),
        "page_hinkley_stat": round(ph_stat, 4),
        "page_hinkley_triggered": ph_drift,
        "stratified_psi": bucket_psi,
        "max_psi": round(max_psi, 4),
        "degraded_vs_raw": degraded_vs_raw,
    }


class SelfEvolvingIntelligence:
    """
    Autonomous learning engine that feeds on reconciled minute-level outcomes,
    updates price-volume-event expert weights via Online Multiplicative Weights,
    mines failure archetypes, and self-generates `generated_alpha_policy.py`.
    """

    def __init__(
        self,
        state_path: str | Path = "intelligence_brain_state.json",
        generated_code_path: str | Path = "generated_alpha_policy.py",
    ) -> None:
        self.state_path = Path(state_path)
        self.generated_code_path = Path(generated_code_path)
        self.state = self._load_state()

    def _load_state(self) -> IntelligenceBrainState:
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return IntelligenceBrainState(**raw)
        return IntelligenceBrainState(updated_at=datetime.now(timezone.utc).isoformat())

    def save_state(self) -> None:
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")

    def _update_online_expert_weights(self, reconciled_df: pd.DataFrame) -> None:
        if "expert_votes" not in reconciled_df.columns or reconciled_df.empty:
            return

        weights = dict(self.state.expert_weights)
        eta = self.state.learning_rate

        for row in reconciled_df.to_dict(orient="records"):
            votes = row.get("expert_votes") or {}
            if isinstance(votes, str):
                try:
                    votes = json.loads(votes)
                except Exception:
                    votes = {}
            if not isinstance(votes, dict) or not votes:
                continue

            realized_dir = float(np.sign(row.get("actual_return_bps", 0.0)))
            if realized_dir == 0.0:
                continue

            for expert_name in list(weights.keys()):
                vote_val = float(votes.get(expert_name, 0.0))
                alignment_reward = np.clip(vote_val * realized_dir, -1.0, 1.0)
                weights[expert_name] *= float(np.exp(eta * alignment_reward))

            total_w = sum(weights.values())
            if total_w > 0:
                for k in weights:
                    weights[k] = round(0.05 + 0.95 * (weights[k] / total_w) / (1.0 + 0.05 * len(weights)), 5)

        norm = sum(weights.values())
        self.state.expert_weights = {k: round(v / norm, 4) for k, v in weights.items()}

    def _discover_and_mutate_rules(self, reconciled_df: pd.DataFrame) -> list[EvolvedRule]:
        discovered: list[EvolvedRule] = []
        if len(reconciled_df) < 15:
            return discovered

        exhaustion_mask = (
            (reconciled_df["volume_zscore_20m"] > 2.0)
            & (reconciled_df["vwap_distance_bps"].abs() > 45.0)
            & (np.sign(reconciled_df["predicted_return_bps"]) == np.sign(reconciled_df["vwap_distance_bps"]))
        )
        exh_sub = reconciled_df[exhaustion_mask]
        if len(exh_sub) >= 8:
            hit_rate = float(exh_sub["directional_hit"].mean())
            if hit_rate < 0.42:
                edge_gain = float(exh_sub["error_bps"].abs().mean() * (0.5 - hit_rate))
                discovered.append(
                    EvolvedRule(
                        rule_id="RULE_CLIMAX_EXHAUSTION_REVERSAL",
                        condition_expr=(
                            "state.get('volume_zscore_20m', 0.0) > 2.0 and abs(state.get('vwap_distance_bps', 0.0)) > 45.0 "
                            "and (pred_ret_bps * state.get('vwap_distance_bps', 0.0)) > 0"
                        ),
                        action_type="FLIP_SIGNAL",
                        multiplier=-0.65,
                        target_horizons=[5, 10, 15, 20],
                        sample_support=int(len(exh_sub)),
                        historical_edge_gain_bps=round(edge_gain, 2),
                    )
                )

        if "outcome_archetype" in reconciled_df.columns:
            for h_val, grp in reconciled_df.groupby("horizon_min"):
                overext = grp[grp["outcome_archetype"] == "HORIZON_OVEREXTENSION_REVERSAL"]
                if len(grp) >= 10 and (len(overext) / len(grp)) >= 0.35:
                    discovered.append(
                        EvolvedRule(
                            rule_id=f"RULE_HORIZON_{int(h_val)}M_OVEREXTENSION_DAMPENER",
                            condition_expr=(
                                f"horizon_min == {int(h_val)} and state.get('volatility_ratio_15m', 1.0) < 0.85"
                            ),
                            action_type="DAMPEN_CONFIDENCE",
                            multiplier=0.55,
                            target_horizons=[int(h_val)],
                            sample_support=int(len(overext)),
                            historical_edge_gain_bps=round(float(overext["path_mfe_bps"].mean() * 0.4), 2),
                        )
                    )

        ignition_mask = (
            (reconciled_df["volume_zscore_20m"] > 1.8)
            & (reconciled_df["candle_body_to_wick"] > 0.65)
            & (reconciled_df["directional_hit"])
        )
        ign_sub = reconciled_df[ignition_mask]
        if len(ign_sub) >= 8:
            realized_mag = float(ign_sub["actual_return_bps"].abs().mean())
            pred_mag = float(ign_sub["predicted_return_bps"].abs().mean())
            if realized_mag > pred_mag * 1.2:
                boost = float(np.clip(realized_mag / max(pred_mag, 1.0), 1.15, 1.85))
                discovered.append(
                    EvolvedRule(
                        rule_id="RULE_VOL_IGNITION_MAGNITUDE_EXPANSION",
                        condition_expr=(
                            "state.get('volume_zscore_20m', 0.0) > 1.8 and state.get('candle_body_to_wick', 0.5) > 0.65 "
                            "and (pred_ret_bps * state.get('order_flow_imbalance', 0.0)) > 0"
                        ),
                        action_type="BOOST_RETURN_TARGET",
                        multiplier=round(boost, 4),
                        target_horizons=[5, 10, 15, 25, 30],
                        sample_support=int(len(ign_sub)),
                        historical_edge_gain_bps=round(realized_mag - pred_mag, 2),
                    )
                )

        return discovered

    def _synthesize_executable_policy_code(self) -> str:
        rules_code_blocks: list[str] = []
        for r in self.state.evolved_rules:
            cond = r["condition_expr"]
            action = r["action_type"]
            mult = float(r["multiplier"])
            r_id = r["rule_id"]

            if action == "FLIP_SIGNAL":
                block = f"""
    # Auto-synthesized rule: {r_id} (support={r['sample_support']}, gain={r['historical_edge_gain_bps']} bps)
    if {cond}:
        pred_ret_bps = pred_ret_bps * ({mult})
        prob_up = 1.0 - prob_up
        applied_rules.append("{r_id}")"""
            elif action == "DAMPEN_CONFIDENCE":
                block = f"""
    # Auto-synthesized rule: {r_id} (support={r['sample_support']}, gain={r['historical_edge_gain_bps']} bps)
    if {cond}:
        prob_up = 0.5 + (prob_up - 0.5) * ({mult})
        pred_ret_bps = pred_ret_bps * ({mult})
        applied_rules.append("{r_id}")"""
            else:
                block = f"""
    # Auto-synthesized rule: {r_id} (support={r['sample_support']}, gain={r['historical_edge_gain_bps']} bps)
    if {cond}:
        pred_ret_bps = pred_ret_bps * ({mult})
        applied_rules.append("{r_id}")"""
            rules_code_blocks.append(block.strip("\n"))

        joined_rules = "\n\n".join(rules_code_blocks) if rules_code_blocks else "    # No custom rules triggered yet"

        code_template = f'''"""
generated_alpha_policy.py
AUTO-GENERATED BY SelfEvolvingIntelligence — DO NOT EDIT MANUALLY.
Policy Brain Version: v{self.state.version}
Generated At: {self.state.updated_at}
"""

from __future__ import annotations
from typing import Any

POLICY_VERSION = "v{self.state.version}"

EXPERT_WEIGHTS: dict[str, float] = {json.dumps(self.state.expert_weights, indent=4)}


def compute_ensemble_expert_signal(state: dict[str, Any]) -> dict[str, float]:
    """Compute individual price-volume-event expert signals and weighted ensemble return."""
    mom_signal = float(state.get("price_momentum_5m_bps", 0.0)) * 0.35
    vol_ignition = (
        float(state.get("volume_zscore_20m", 0.0))
        * float(state.get("order_flow_imbalance", 0.0))
        * 12.0
    )
    vwap_rev = -float(state.get("vwap_distance_bps", 0.0)) * (
        0.25 if float(state.get("volatility_ratio_15m", 1.0)) < 0.9 else -0.10
    )
    ofi_signal = float(state.get("order_flow_imbalance", 0.0)) * 15.0

    votes = {{
        "momentum_price_expert": round(mom_signal, 4),
        "volume_ignition_expert": round(vol_ignition, 4),
        "vwap_reversion_expert": round(vwap_rev, 4),
        "order_flow_imbalance_expert": round(ofi_signal, 4),
    }}

    weighted_bps = sum(votes[k] * EXPERT_WEIGHTS.get(k, 0.25) for k in votes)
    votes["ensemble_return_bps"] = round(weighted_bps, 4)
    return votes


def apply_evolved_intelligence(
    reference_close: float,
    raw_predicted_price: float,
    raw_probability_up: float,
    horizon_min: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute self-generated microstructure adjustments on the raw model forecast."""
    if reference_close <= 0:
        raise ValueError("reference_close must be positive.")

    pred_ret_bps = ((raw_predicted_price - reference_close) / reference_close) * 10000.0
    prob_up = float(raw_probability_up)
    applied_rules: list[str] = []

    expert_out = compute_ensemble_expert_signal(state)
    pred_ret_bps = 0.70 * pred_ret_bps + 0.30 * expert_out["ensemble_return_bps"]

{joined_rules}

    upgraded_price = reference_close * (1.0 + (pred_ret_bps / 10000.0))
    prob_up = max(0.01, min(0.99, prob_up))

    return {{
        "predicted_price": round(upgraded_price, 4),
        "probability_up": round(prob_up, 4),
        "predicted_return_bps": round(pred_ret_bps, 2),
        "expert_votes": expert_out,
        "applied_rules": applied_rules,
        "active_policy_version": POLICY_VERSION,
    }}
'''
        return textwrap.dedent(code_template).lstrip()

    def evolve_from_outcomes(self, reconciled_df: pd.DataFrame) -> dict[str, Any]:
        """Run full online learning + symbolic rule discovery and emit `generated_alpha_policy.py`."""
        if reconciled_df.empty:
            return {"status": "NO_NEW_EXPERIENCES", "version": self.state.version}

        self._update_online_expert_weights(reconciled_df)
        new_rules = self._discover_and_mutate_rules(reconciled_df)

        existing_by_id = {r["rule_id"]: r for r in self.state.evolved_rules}
        for rule in new_rules:
            existing_by_id[rule.rule_id] = asdict(rule)
        self.state.evolved_rules = list(existing_by_id.values())

        self.state.version += 1
        self.save_state()

        generated_code = self._synthesize_executable_policy_code()
        self.generated_code_path.parent.mkdir(parents=True, exist_ok=True)
        self.generated_code_path.write_text(generated_code, encoding="utf-8")

        return {
            "status": "INTELLIGENCE_UPGRADED",
            "new_version": f"v{self.state.version}",
            "updated_expert_weights": self.state.expert_weights,
            "active_evolved_rules_count": len(self.state.evolved_rules),
            "generated_code_file": str(self.generated_code_path),
        }