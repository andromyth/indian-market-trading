"""Step 52 research gate and current-news perception labels for the dashboard.

The Step 52 walk-forward result is research evidence only. This module deliberately
does not load its predictions into the live signal engine.
"""
from __future__ import annotations

STEP52_AUDIT = [
    {"fold": 1, "horizon": "30m", "test_rows": 5670, "auc": 0.493869, "balanced_accuracy": 0.504888},
    {"fold": 1, "horizon": "60m", "test_rows": 5184, "auc": 0.511964, "balanced_accuracy": 0.503583},
    {"fold": 2, "horizon": "30m", "test_rows": 5571, "auc": 0.505482, "balanced_accuracy": 0.504880},
    {"fold": 2, "horizon": "60m", "test_rows": 5085, "auc": 0.485962, "balanced_accuracy": 0.488069},
    {"fold": 3, "horizon": "30m", "test_rows": 5628, "auc": 0.513004, "balanced_accuracy": 0.514754},
    {"fold": 3, "horizon": "60m", "test_rows": 5124, "auc": 0.521763, "balanced_accuracy": 0.529142},
]

STEP52_SYMBOLS = ("BHARTIARTL", "RELIANCE", "SBIN")
STEP52_JOINED_ROWS = 41145
STEP52_PREDICTORS = 87
STEP52_LAST_TEST_DATE = "2026-09-25"


def news_perception(metrics: dict) -> tuple[str, str]:
    """Return a readable sentiment and freshness label without implying probability."""
    count = int(metrics.get("count") or 0)
    if count <= 0:
        return "No coverage", "No matching RSS headlines"

    score = float(metrics.get("score") or 0.0)
    if score >= 0.15:
        polarity = "Positive"
    elif score <= -0.15:
        polarity = "Negative"
    else:
        polarity = "Neutral/mixed"

    age = metrics.get("fresh_hours")
    if age is None:
        freshness = "Publication time unavailable"
    elif float(age) <= 6:
        freshness = "Freshest item <6h"
    elif float(age) <= 24:
        freshness = "Freshest item 6–24h"
    else:
        freshness = "No matching item in the last 24h"
    return f"{polarity} {score:+.2f}", freshness
