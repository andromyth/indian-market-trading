import numpy as np
import pandas as pd

from delivery_quality_gate import apply_quality_gate, quality_gate_reasons


def test_unknown_optional_sources_are_reported_and_not_mislabelled_hold():
    row = {
        "ready_market_price": 1,
        "ready_market_context": 1,
        "ready_news": 0,
        "ready_events": 0,
        "ready_fno": np.nan,
        "ready_fundamentals": 1,
    }
    assert quality_gate_reasons(row) == ["FNO_STATE_UNKNOWN"]
    assert apply_quality_gate(["BUY", "SELL", "HOLD"], [False, np.nan, True]) == [
        "DATA INSUFFICIENT", "DATA INSUFFICIENT", "HOLD"
    ]


def test_core_market_not_ready_blocks_action_and_explains_reason():
    row = {
        "ready_market_price": 0,
        "ready_market_context": np.nan,
        "ready_news": 0,
        "ready_events": 1,
        "ready_fno": 0,
        "ready_fundamentals": 0,
    }
    assert quality_gate_reasons(row) == [
        "MARKET_PRICE_NOT_READY",
        "MARKET_CONTEXT_READINESS_UNKNOWN",
    ]


def test_explicit_optional_zero_is_known_and_does_not_block_by_itself():
    row = {
        "ready_market_price": 1,
        "ready_market_context": 1,
        "ready_news": 0,
        "ready_events": 0,
        "ready_fno": 0,
        "ready_fundamentals": 0,
    }
    assert quality_gate_reasons(row) == []
    assert apply_quality_gate(["BUY"], [True]) == ["BUY"]
