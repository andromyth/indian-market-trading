"""Single readiness contract for stored ALL-NSE intraday predictions.

The Live desk and Paper Trading must agree on what "a usable persisted
prediction" means. They previously diverged: the desk re-implemented a weaker
manifest check while Paper Trading used this stricter row-level contract. This
module is the one definition both paths import.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MATCHED_LINK_STATUS = "MATCHED_TO_SNAPSHOT_QUOTE"


def stored_prediction_is_ready(row: Any) -> bool:
    """True only for a persisted, snapshot-linked, successfully predicted row.

    Accepts any mapping-like object (``dict`` or a pandas ``Series``) so the
    Live desk and Paper Trading share one readiness contract. A row is ready
    only when it is stored, linked to the stored snapshot quote, marked
    ``PREDICTED*``, and carries a finite session-close probability.
    """
    if row is None or not hasattr(row, "get"):
        return False
    try:
        if len(row) == 0:
            return False
    except TypeError:
        return False
    if not bool(row.get("_prediction_row_stored")):
        return False
    if str(row.get("prediction_snapshot_link_status") or "") != MATCHED_LINK_STATUS:
        return False
    if not str(row.get("prediction_status") or "").upper().startswith("PREDICTED"):
        return False
    probability = pd.to_numeric(
        pd.Series([row.get("probability_up_session_close")]), errors="coerce"
    ).iloc[0]
    return bool(pd.notna(probability) and np.isfinite(float(probability)))
