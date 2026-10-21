"""
validate_prediction_ledger.py
Unified point-in-time reconciliation of minute-level predictions against live 1-minute NSE bars,
realized volume-shock detection, root-cause failure attribution, and automatic triggering of
SelfEvolvingIntelligence to upgrade `generated_alpha_policy.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from calibration_autopilot import SelfEvolvingIntelligence
from prediction_ledger import IST_TZ, is_within_nse_session


def _prepare_market_df(market_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize 1-minute NSE OHLCV market data for merge_asof alignment."""
    if market_df.empty:
        raise ValueError("market_df cannot be empty for live market reconciliation.")

    mkt = market_df.copy()
    ts_col = "timestamp" if "timestamp" in mkt.columns else "datetime"
    mkt["market_timestamp"] = pd.to_datetime(mkt[ts_col], utc=True).dt.tz_convert(IST_TZ)
    mkt["symbol"] = mkt["symbol"].astype(str).str.strip().str.upper()

    for col in ("open", "high", "low", "close", "volume"):
        if col in mkt.columns:
            mkt[col] = pd.to_numeric(mkt[col], errors="coerce")

    if "high" in mkt.columns and "low" in mkt.columns:
        mkt["typical_price"] = (mkt["high"] + mkt["low"] + mkt["close"]) / 3.0
    else:
        mkt["typical_price"] = mkt["close"]

    return mkt.sort_values("market_timestamp").reset_index(drop=True)


def _compute_path_excursions_and_shocks(
    reconciled_df: pd.DataFrame,
    mkt_df: pd.DataFrame,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """
    Compute path-wise MFE/MAE (both in INR and bps) and realized mid-horizon volume shock
    ratios over (decision_timestamp, target_timestamp] for each prediction.
    """
    mkt_by_sym = {
        sym: group.sort_values("market_timestamp")
        for sym, group in mkt_df.groupby("symbol", sort=False)
    }

    mfe_inr_list: list[float] = []
    mae_inr_list: list[float] = []
    mfe_bps_list: list[float] = []
    mae_bps_list: list[float] = []
    vol_shock_list: list[float] = []

    for row in reconciled_df.itertuples(index=False):
        sym_mkt = mkt_by_sym.get(row.symbol)
        ref_close = getattr(row, "reference_close", np.nan)
        pred_price = getattr(row, "predicted_price", np.nan)

        if sym_mkt is None or pd.isna(ref_close) or pd.isna(pred_price) or ref_close <= 0:
            mfe_inr_list.append(np.nan)
            mae_inr_list.append(np.nan)
            mfe_bps_list.append(np.nan)
            mae_bps_list.append(np.nan)
            vol_shock_list.append(1.0)
            continue

        window = sym_mkt[
            (sym_mkt["market_timestamp"] > row.decision_timestamp)
            & (sym_mkt["market_timestamp"] <= row.target_timestamp)
        ]
        if window.empty:
            mfe_inr_list.append(np.nan)
            mae_inr_list.append(np.nan)
            mfe_bps_list.append(np.nan)
            mae_bps_list.append(np.nan)
            vol_shock_list.append(1.0)
            continue

        max_high = float(window["high"].max()) if "high" in window.columns else float(window["close"].max())
        min_low = float(window["low"].min()) if "low" in window.columns else float(window["close"].min())

        direction = 1.0 if pred_price >= ref_close else -1.0
        if direction > 0:
            mfe_inr = max(0.0, max_high - float(ref_close))
            mae_inr = max(0.0, float(ref_close) - min_low)
        else:
            mfe_inr = max(0.0, float(ref_close) - min_low)
            mae_inr = max(0.0, max_high - float(ref_close))

        mfe_inr_list.append(mfe_inr)
        mae_inr_list.append(mae_inr)
        mfe_bps_list.append((mfe_inr / float(ref_close)) * 10000.0)
        mae_bps_list.append((mae_inr / float(ref_close)) * 10000.0)

        if "volume" in sym_mkt.columns and "volume" in window.columns:
            pre_win = sym_mkt[
                (sym_mkt["market_timestamp"] <= row.decision_timestamp)
                & (sym_mkt["market_timestamp"] > row.decision_timestamp - pd.Timedelta(minutes=20))
            ]
            base_vol = float(pre_win["volume"].mean()) if not pre_win.empty else float(window["volume"].mean())
            peak_vol = float(window["volume"].max())
            vol_shock_list.append(round(peak_vol / max(base_vol, 1.0), 3))
        else:
            vol_shock_list.append(1.0)

    return mfe_inr_list, mae_inr_list, mfe_bps_list, mae_bps_list, vol_shock_list


def _classify_outcome_archetypes(df: pd.DataFrame) -> list[str]:
    """Classify each reconciled prediction into a causal root-cause archetype for rule synthesis."""
    archetypes: list[str] = []
    for row in df.itertuples(index=False):
        dir_hit = bool(getattr(row, "reg_dir_hit", False))
        pred_bps = abs(float(getattr(row, "predicted_return_bps", 0.0)))
        mfe_bps = float(getattr(row, "path_mfe_bps", 0.0) or 0.0)
        vol_shock = float(getattr(row, "realized_vol_shock_ratio", 1.0) or 1.0)
        vol_z = float(getattr(row, "volume_zscore_20m", 0.0) or 0.0)

        if dir_hit:
            archetypes.append("VALIDATED_EDGE")
        elif mfe_bps > max(pred_bps * 0.8, 8.0) and not dir_hit:
            archetypes.append("HORIZON_OVEREXTENSION_REVERSAL")
        elif vol_shock >= 2.5 and not dir_hit:
            archetypes.append("UNMODELED_VOLUME_EVENT_SHOCK")
        elif vol_z >= 1.5 and not dir_hit:
            archetypes.append("EXHAUSTION_TRAP_FAKEOUT")
        else:
            archetypes.append("REGIME_CHOP_WHIPSAW")
    return archetypes


def compare_with_live_market(
    ledger_df: pd.DataFrame,
    market_df: pd.DataFrame,
    *,
    max_tolerance: str = "2min",
    high_conf_delta: float = 0.06,
    compute_excursions: bool = True,
    auto_evolve_intelligence: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Align minute-level predictions with realized 1-minute NSE bars using a strict
    backward merge_asof, backfill missing reference_close values, compute
    horizon-adaptive accuracy and causal failure archetypes, and automatically
    trigger SelfEvolvingIntelligence to upgrade `generated_alpha_policy.py`.
    """
    if ledger_df.empty:
        return ledger_df.copy(), {"matched_count": 0, "match_rate": 0.0}

    pred = ledger_df.copy()
    pred["symbol"] = pred["symbol"].astype(str).str.strip().str.upper()
    pred["horizon_min"] = pd.to_numeric(pred["horizon_min"], errors="coerce").fillna(5).astype(int)
    pred["decision_timestamp"] = pd.to_datetime(pred["decision_timestamp"], utc=True).dt.tz_convert(IST_TZ)
    pred["target_timestamp"] = pred["decision_timestamp"] + pd.to_timedelta(pred["horizon_min"], unit="m")

    in_session_mask = [
        is_within_nse_session(d_ts, h)
        for d_ts, h in zip(pred["decision_timestamp"], pred["horizon_min"])
    ]
    pred["in_session"] = in_session_mask
    pred = pred[pred["in_session"]].copy()
    total_in_session = len(pred)

    mkt = _prepare_market_df(market_df)
    tol = pd.Timedelta(max_tolerance)

    # 1. Backfill missing reference_close at decision_timestamp
    if "reference_close" not in pred.columns:
        pred["reference_close"] = np.nan
    pred["reference_close"] = pd.to_numeric(pred["reference_close"], errors="coerce")

    if pred["reference_close"].isna().any():
        pred_sorted_dec = pred.sort_values("decision_timestamp")
        ref_merged = pd.merge_asof(
            pred_sorted_dec,
            mkt[["symbol", "market_timestamp", "close"]].rename(
                columns={"market_timestamp": "ref_market_ts", "close": "backfilled_ref_close"}
            ),
            left_on="decision_timestamp",
            right_on="ref_market_ts",
            by="symbol",
            direction="backward",
            tolerance=tol,
            allow_exact_matches=True,
        )
        ref_merged["reference_close"] = ref_merged["reference_close"].fillna(ref_merged["backfilled_ref_close"])
        pred = ref_merged.drop(columns=["ref_market_ts", "backfilled_ref_close"])

    # 2. Match realized target_timestamp against 1-minute NSE bars (strictly backward)
    pred = pred.sort_values("target_timestamp").reset_index(drop=True)
    reconciled = pd.merge_asof(
        pred,
        mkt[["symbol", "market_timestamp", "close", "typical_price"]].rename(
            columns={
                "close": "actual_close",
                "typical_price": "actual_typical_price",
            }
        ),
        left_on="target_timestamp",
        right_on="market_timestamp",
        by="symbol",
        direction="backward",
        tolerance=tol,
        allow_exact_matches=True,
    )

    reconciled["tick_lag_seconds"] = (
        reconciled["target_timestamp"] - reconciled["market_timestamp"]
    ).dt.total_seconds()

    valid_mask = reconciled["actual_close"].notna() & reconciled["reference_close"].notna() & (reconciled["reference_close"] > 0)
    matched = reconciled[valid_mask].copy()

    if matched.empty:
        return reconciled, {
            "total_in_session": total_in_session,
            "matched_count": 0,
            "match_rate": 0.0,
        }

    # 3. Compute directional and regression error metrics (both ratios and bps)
    matched["predicted_return"] = (matched["predicted_price"] - matched["reference_close"]) / matched["reference_close"]
    matched["actual_return"] = (matched["actual_close"] - matched["reference_close"]) / matched["reference_close"]
    matched["typical_return"] = (matched["actual_typical_price"] - matched["reference_close"]) / matched["reference_close"]

    matched["predicted_return_bps"] = matched["predicted_return"] * 10000.0
    matched["actual_return_bps"] = matched["actual_return"] * 10000.0
    matched["error_bps"] = matched["actual_return_bps"] - matched["predicted_return_bps"]

    matched["abs_error_inr"] = (matched["predicted_price"] - matched["actual_close"]).abs()
    matched["zero_change_error_inr"] = (matched["actual_close"] - matched["reference_close"]).abs()
    matched["beats_zero_change"] = matched["abs_error_inr"] < matched["zero_change_error_inr"]

    matched["reg_dir_hit"] = np.sign(matched["predicted_return"]) == np.sign(matched["actual_return"])
    matched["directional_hit"] = matched["reg_dir_hit"]
    matched["typical_dir_hit"] = np.sign(matched["predicted_return"]) == np.sign(matched["typical_return"])

    if "probability_up" in matched.columns:
        matched["prob_delta"] = (matched["probability_up"] - 0.5).abs()
        matched["clf_dir"] = np.where(matched["probability_up"] >= 0.5, 1.0, -1.0)
        matched["clf_dir_hit"] = matched["clf_dir"] == np.sign(matched["actual_return"])
        matched["is_high_confidence"] = matched["prob_delta"] > high_conf_delta
    else:
        matched["prob_delta"] = np.nan
        matched["clf_dir_hit"] = matched["reg_dir_hit"]
        matched["is_high_confidence"] = False

    if compute_excursions:
        mfe_inr, mae_inr, mfe_bps, mae_bps, vol_shocks = _compute_path_excursions_and_shocks(matched, mkt)
        matched["path_mfe_inr"] = mfe_inr
        matched["path_mae_inr"] = mae_inr
        matched["path_mfe_bps"] = mfe_bps
        matched["path_mae_bps"] = mae_bps
        matched["realized_vol_shock_ratio"] = vol_shocks
    else:
        matched["path_mfe_bps"] = 0.0
        matched["path_mae_bps"] = 0.0
        matched["realized_vol_shock_ratio"] = 1.0

    for col, d_val in [
        ("volume_zscore_20m", 0.0),
        ("vwap_distance_bps", 0.0),
        ("volatility_ratio_15m", 1.0),
        ("candle_body_to_wick", 0.5),
        ("order_flow_imbalance", 0.0),
    ]:
        if col not in matched.columns:
            matched[col] = d_val

    matched["outcome_archetype"] = _classify_outcome_archetypes(matched)

    # 4. Build summary diagnostics dictionary
    active_mask = (
        matched["signal"].astype(str).str.upper().isin(["BUY", "SELL", "LONG", "SHORT"])
        if "signal" in matched.columns
        else pd.Series(True, index=matched.index)
    )
    active_df = matched[active_mask]
    high_conf_df = active_df[active_df["is_high_confidence"]]

    horizon_breakdown: dict[int, dict[str, float]] = {}
    for h, grp in matched.groupby("horizon_min"):
        grp_active = grp[active_mask.loc[grp.index]]
        grp_hc = grp_active[grp_active["is_high_confidence"]]
        horizon_breakdown[int(h)] = {
            "count": int(len(grp)),
            "active_dir_hit_rate": float(grp_active["reg_dir_hit"].mean()) if not grp_active.empty else float(grp["reg_dir_hit"].mean()),
            "typical_dir_hit_rate": float(grp_active["typical_dir_hit"].mean()) if not grp_active.empty else float(grp["typical_dir_hit"].mean()),
            "high_conf_hit_rate": float(grp_hc["reg_dir_hit"].mean()) if not grp_hc.empty else float("nan"),
            "mae_inr": float(grp["abs_error_inr"].mean()),
        }

    archetype_counts = matched["outcome_archetype"].value_counts().to_dict()

    summary: dict[str, Any] = {
        "total_in_session": int(total_in_session),
        "matched_count": int(len(matched)),
        "match_rate": float(len(matched) / max(1, total_in_session)),
        "overall_reg_hit_rate": float(matched["reg_dir_hit"].mean()),
        "overall_clf_hit_rate": float(matched["clf_dir_hit"].mean()),
        "active_dir_hit_rate": float(active_df["reg_dir_hit"].mean()) if not active_df.empty else float(matched["reg_dir_hit"].mean()),
        "high_conf_dir_hit_rate": float(high_conf_df["reg_dir_hit"].mean()) if not high_conf_df.empty else float("nan"),
        "overall_mae_inr": float(matched["abs_error_inr"].mean()),
        "beats_zero_change_rate": float(matched["beats_zero_change"].mean()),
        "outcome_archetypes": {str(k): int(v) for k, v in archetype_counts.items()},
        "horizons": horizon_breakdown,
    }

    # 5. Zero-edit autonomous upgrade: learn from reconciled outcomes & rewrite generated_alpha_policy.py
    if auto_evolve_intelligence:
        try:
            brain = SelfEvolvingIntelligence(
                state_path="intelligence_brain_state.json",
                generated_code_path="generated_alpha_policy.py",
            )
            summary["intelligence_evolution"] = brain.evolve_from_outcomes(matched)
        except Exception as exc:
            summary["intelligence_evolution"] = {"status": "SKIPPED", "error": str(exc)}

    return matched, summary