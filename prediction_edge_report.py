"""Measure whether live predictions have directional edge before trusting them.

This is deliberately separate from the dashboard scorecard. The scorecard
answers "how did the last seven rows look"; this tool answers "is the edge
distinguishable from chance for this horizon/model/version", using independent
samples, base rates, Brier/log-loss and confidence intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from probability_calibration import probability_report


def _bootstrap_auc_interval(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = 500,
    seed: int = 20261003,
) -> tuple[float | None, float | None]:
    if probabilities.size < 30 or np.unique(labels).size < 2:
        return None, None
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(iterations)):
        idx = rng.integers(0, probabilities.size, probabilities.size)
        sampled_labels = labels[idx]
        if np.unique(sampled_labels).size < 2:
            continue
        values.append(float(roc_auc_score(sampled_labels, probabilities[idx])))
    if not values:
        return None, None
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _verdict(report: dict[str, Any], auc_low: float | None) -> str:
    n = int(report.get("n") or 0)
    if n == 0:
        return "NO_ROWS"
    if n < 100:
        return "INSUFFICIENT_SAMPLE"
    action = report.get("action") or {}
    action_low = (action.get("precision_ci95") or [None, None])[0]
    if action_low is not None and action_low > 0.50:
        return "ACTIONABLE_EDGE"
    if auc_low is not None and auc_low > 0.50:
        return "WEAK_EDGE"
    return "NO_EDGE_EVIDENCE"


def _slice_report(frame: pd.DataFrame, probability_column: str, label_column: str) -> dict[str, Any]:
    probabilities = pd.to_numeric(frame[probability_column], errors="coerce").to_numpy(dtype=float)
    labels = pd.to_numeric(frame[label_column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(probabilities) & np.isfinite(labels)
    probabilities, labels = probabilities[mask], labels[mask]
    report = probability_report(probabilities, labels)
    report["roc_auc_ci95"] = list(_bootstrap_auc_interval(probabilities, labels))
    report["verdict"] = _verdict(report, report["roc_auc_ci95"][0])
    return report


def build_edge_report(
    frame: pd.DataFrame,
    *,
    probability_column: str = "probability_up_session_close",
    label_column: str = "target_up",
    group_column: str | None = "horizon_minutes",
    min_samples: int = 100,
) -> dict[str, Any]:
    """Build an overall + per-group edge report from a matured ledger frame."""
    if probability_column not in frame.columns:
        raise ValueError(f"missing probability column: {probability_column}")
    if label_column not in frame.columns:
        raise ValueError(f"missing label column: {label_column}")
    work = frame.copy()
    work[probability_column] = pd.to_numeric(work[probability_column], errors="coerce")
    work[label_column] = pd.to_numeric(work[label_column], errors="coerce")
    work = work.loc[work[probability_column].notna() & work[label_column].notna()]
    overall = _slice_report(work, probability_column, label_column)
    overall["minimum_samples"] = int(min_samples)
    if overall["n"] < int(min_samples):
        overall["verdict"] = "INSUFFICIENT_SAMPLE"
    groups: dict[str, Any] = {}
    if group_column and group_column in work.columns and not work.empty:
        grouped = work.groupby(work[group_column].astype(str), dropna=False)
        for key, part in grouped:
            if len(part) < 20:
                continue
            item = _slice_report(part, probability_column, label_column)
            item["verdict"] = (
                "INSUFFICIENT_SAMPLE" if item["n"] < int(min_samples) else item["verdict"]
            )
            groups[str(key)] = item
    return {
        "probability_column": probability_column,
        "label_column": label_column,
        "group_column": group_column,
        "overall": overall,
        "groups": groups,
    }


def _read_ledger(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Matured prediction ledger (.jsonl/.csv/.parquet)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--probability-column", default="probability_up_session_close")
    parser.add_argument("--label-column", default="target_up")
    parser.add_argument("--group-column", default="horizon_minutes")
    parser.add_argument("--min-samples", type=int, default=100)
    args = parser.parse_args(argv)
    report = build_edge_report(
        _read_ledger(args.input),
        probability_column=args.probability_column,
        label_column=args.label_column,
        group_column=args.group_column,
        min_samples=args.min_samples,
    )
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
