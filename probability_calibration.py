"""Probability calibration, threshold selection and edge metrics."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

_EPS = 1e-6


def _clean(probabilities: Iterable[Any], labels: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(list(probabilities), dtype=float).reshape(-1)
    y = np.asarray(list(labels), dtype=float).reshape(-1)
    if p.size != y.size:
        raise ValueError("probabilities and labels must have the same length")
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    return np.clip(p, _EPS, 1.0 - _EPS), y


def logit(probabilities: Any) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def sigmoid(values: Any) -> np.ndarray:
    z = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def brier_score(probabilities: Any, labels: Any) -> float | None:
    p, y = _clean(probabilities, labels)
    if not p.size:
        return None
    return float(np.mean((p - y) ** 2))


def log_loss(probabilities: Any, labels: Any) -> float | None:
    p, y = _clean(probabilities, labels)
    if not p.size:
        return None
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    n = int(total)
    if n <= 0:
        return None, None
    p = float(successes) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    margin = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def fit_platt(probabilities: Any, labels: Any, *, min_rows: int = 200) -> dict[str, Any] | None:
    p, y = _clean(probabilities, labels)
    if p.size < max(10, int(min_rows)) or np.unique(y).size < 2:
        return None
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit(p).reshape(-1, 1), y.astype(int))
    spec = {
        "method": "platt",
        "coef": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "fit_rows": int(p.size),
        "brier_before": brier_score(p, y),
    }
    spec["brier_after"] = brier_score(apply_probability_calibration(p, spec), y)
    before, after = spec["brier_before"], spec["brier_after"]
    if spec["coef"] <= 0:
        return None
    if before is not None and after is not None and after > before:
        return None
    return spec


def apply_probability_calibration(probabilities: Any, spec: dict[str, Any] | None) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), _EPS, 1.0 - _EPS)
    if not spec:
        return p
    method = str(spec.get("method") or "").strip().lower()
    if method == "platt":
        return sigmoid(float(spec["coef"]) * logit(p) + float(spec["intercept"]))
    if method == "isotonic":
        x = np.asarray(spec.get("x") or [], dtype=float)
        y = np.asarray(spec.get("y") or [], dtype=float)
        if x.size and x.size == y.size:
            return np.interp(p, x, y)
    return p


def probability_report(probabilities: Any, labels: Any, *, thresholds: tuple[float, float] = (0.60, 0.40)) -> dict[str, Any]:
    p, y = _clean(probabilities, labels)
    if not p.size:
        return {"n": 0, "base_rate_up": None, "brier": None, "log_loss": None, "accuracy_at_0_5": None, "balanced_accuracy": None, "roc_auc": None}
    buy_threshold, sell_threshold = float(thresholds[0]), float(thresholds[1])
    predicted_up = p >= 0.5
    actual_up = y >= 0.5
    accuracy = float(np.mean(predicted_up == actual_up))
    up_recall = float(np.mean(predicted_up[actual_up])) if actual_up.any() else None
    down_recall = float(np.mean(~predicted_up[~actual_up])) if (~actual_up).any() else None
    balanced = float(np.mean([up_recall, down_recall])) if up_recall is not None and down_recall is not None else None
    auc = None
    if np.unique(y).size == 2:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(y, p))
    buys = p >= buy_threshold
    sells = p <= sell_threshold
    actions = buys | sells
    action_correct = (buys & actual_up) | (sells & ~actual_up)
    buy_low, buy_high = wilson_interval(float((buys & actual_up).sum()), int(buys.sum()))
    sell_low, sell_high = wilson_interval(float((sells & ~actual_up).sum()), int(sells.sum()))
    action_low, action_high = wilson_interval(float(action_correct[actions].sum()), int(actions.sum()))
    overall_low, overall_high = wilson_interval(float((predicted_up == actual_up).sum()), int(p.size))
    return {
        "n": int(p.size),
        "base_rate_up": float(np.mean(actual_up)),
        "brier": brier_score(p, y),
        "log_loss": log_loss(p, y),
        "accuracy_at_0_5": accuracy,
        "accuracy_at_0_5_ci95": [overall_low, overall_high],
        "balanced_accuracy": balanced,
        "roc_auc": auc,
        "buy": {"threshold": buy_threshold, "count": int(buys.sum()), "precision": float((buys & actual_up).sum() / buys.sum()) if buys.any() else None, "precision_ci95": [buy_low, buy_high], "coverage": float(buys.mean())},
        "sell": {"threshold": sell_threshold, "count": int(sells.sum()), "precision": float((sells & ~actual_up).sum() / sells.sum()) if sells.any() else None, "precision_ci95": [sell_low, sell_high], "coverage": float(sells.mean())},
        "action": {"count": int(actions.sum()), "precision": float(action_correct[actions].mean()) if actions.any() else None, "precision_ci95": [action_low, action_high], "coverage": float(actions.mean())},
    }


def _side_thresholds(side: str, default: float) -> tuple[float, float, float]:
    if side == "buy":
        return 0.50, max(0.50, default) + 0.30, 0.01
    return min(0.50, default) - 0.30, 0.50, 0.01


def choose_threshold(probabilities: Any, labels: Any, side: str, *, default: float, min_coverage: float = 0.02, min_precision: float = 0.55, min_action_samples: int = 100, z: float = 1.96) -> float:
    side = str(side).strip().lower()
    p, y = _clean(probabilities, labels)
    if p.size < int(min_action_samples) or np.unique(y).size < 2:
        return float(default)
    actual_up = y >= 0.5
    best = float(default)
    best_coverage = -1.0
    start, stop, step = _side_thresholds(side, float(default))
    for threshold in np.arange(start, stop + step / 2.0, step):
        if side == "buy":
            selected = p >= threshold
            correct = selected & actual_up
        else:
            selected = p <= threshold
            correct = selected & ~actual_up
        count = int(selected.sum())
        if count < max(1, int(min_action_samples * 0.5)):
            continue
        coverage = count / p.size
        if coverage < float(min_coverage):
            continue
        precision = float(correct.sum()) / count
        low, _ = wilson_interval(float(correct.sum()), count, z=z)
        if precision < float(min_precision) or low is None or low <= 0.50:
            continue
        if coverage > best_coverage:
            best_coverage = coverage
            best = float(threshold)
    if side == "sell":
        best = min(0.50, float(best))
    return float(best)
