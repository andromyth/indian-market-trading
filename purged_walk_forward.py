"""Leakage-resistant chronological splits for per-observation future-return labels."""
from __future__ import annotations

from typing import Any

import pandas as pd


def purged_chronological_split(
    frame: pd.DataFrame,
    *,
    decision_column: str = "decision_timestamp",
    target_column: str = "_target_timestamp",
    cutoff_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split by decision time and purge training labels that reach the holdout.

    A training row is eligible only if both its decision and the timestamp of
    its realised target precede the holdout cutoff. This prevents a 1/3/5/7
    session label from using prices observed inside the validation window.
    Holdout rows retain their future target timestamp so they can be scored
    after their outcomes become available.
    """
    if cutoff_days < 1:
        raise ValueError("cutoff_days must be >= 1")
    missing = [c for c in (decision_column, target_column) if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required timestamp columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot split an empty frame")

    decisions = pd.to_datetime(frame[decision_column], errors="coerce", utc=True)
    targets = pd.to_datetime(frame[target_column], errors="coerce", utc=True)
    if decisions.notna().sum() == 0:
        raise ValueError(f"No valid timestamps in {decision_column}")

    cutoff = decisions.max() - pd.Timedelta(days=cutoff_days)
    pre_cutoff = decisions.lt(cutoff)
    holdout = decisions.ge(cutoff) & targets.notna()
    crossed_cutoff = pre_cutoff & targets.ge(cutoff)
    invalid_target = pre_cutoff & targets.isna()
    train = pre_cutoff & targets.lt(cutoff)

    train_frame = frame.loc[train].copy()
    holdout_frame = frame.loc[holdout].copy()
    if train_frame.empty or holdout_frame.empty:
        raise ValueError(
            "Chronological split has an empty train or holdout set: "
            f"train={len(train_frame)}, holdout={len(holdout_frame)}, "
            f"cutoff={cutoff.isoformat()}"
        )

    latest_train_target = pd.to_datetime(
        train_frame[target_column], errors="coerce", utc=True
    ).max()
    first_holdout_decision = pd.to_datetime(
        holdout_frame[decision_column], errors="coerce", utc=True
    ).min()
    if not latest_train_target < cutoff:
        raise AssertionError(
            "Purged split invalid: a training target is not strictly before cutoff"
        )
    if first_holdout_decision < cutoff:
        raise AssertionError(
            "Purged split invalid: a holdout decision precedes cutoff"
        )

    audit = {
        "cutoff_timestamp": cutoff.isoformat(),
        "cutoff_days": int(cutoff_days),
        "input_rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "holdout_rows": int(len(holdout_frame)),
        "pre_cutoff_rows": int(pre_cutoff.sum()),
        "purged_boundary_rows": int(crossed_cutoff.sum()),
        "pre_cutoff_rows_without_target_timestamp": int(invalid_target.sum()),
        "train_latest_target_timestamp": latest_train_target.isoformat(),
        "holdout_first_decision_timestamp": first_holdout_decision.isoformat(),
        "target_boundary_check": "PASS",
        "split_policy": "train target timestamp strictly before cutoff; holdout decision timestamp at or after cutoff",
    }
    return train_frame, holdout_frame, audit
