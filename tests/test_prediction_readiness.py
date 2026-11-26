"""One readiness contract for the Live desk and Paper Trading."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from prediction_readiness import stored_prediction_is_ready

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ready_row(**overrides):
    row = {
        "_prediction_row_stored": True,
        "prediction_snapshot_link_status": "MATCHED_TO_SNAPSHOT_QUOTE",
        "prediction_status": "PREDICTED",
        "probability_up_session_close": 0.61,
    }
    row.update(overrides)
    return row


def test_ready_dict_row():
    assert stored_prediction_is_ready(_ready_row()) is True


def test_ready_pandas_series_row():
    assert stored_prediction_is_ready(pd.Series(_ready_row())) is True


def test_missing_or_empty_rows_are_not_ready():
    assert stored_prediction_is_ready(None) is False
    assert stored_prediction_is_ready({}) is False
    assert stored_prediction_is_ready(pd.Series(dtype=object)) is False


def test_unstored_row_is_not_ready():
    assert stored_prediction_is_ready(_ready_row(_prediction_row_stored=False)) is False


def test_unlinked_row_is_not_ready():
    assert stored_prediction_is_ready(
        _ready_row(prediction_snapshot_link_status="MISMATCH_OR_UNVERIFIED")
    ) is False


def test_non_predicted_status_is_not_ready():
    assert stored_prediction_is_ready(_ready_row(prediction_status="WARMING")) is False


def test_missing_or_non_finite_probability_is_not_ready():
    assert stored_prediction_is_ready(_ready_row(probability_up_session_close=None)) is False
    assert stored_prediction_is_ready(
        _ready_row(probability_up_session_close=float("nan"))
    ) is False


def test_desk_uses_the_shared_readiness_contract():
    source = (REPO_ROOT / "app.py").read_text(encoding="utf-8")
    assert (
        "from prediction_readiness import stored_prediction_is_ready as _stored_prediction_is_ready"
        in source
    )
    # The Live desk must base readiness on the shared contract, not the old
    # inline manifest/link re-implementation.
    assert "prediction_ready = _stored_prediction_is_ready(row)" in source
    assert "This GitHub prediction cycle is stale" not in source
