import numpy as np
import pandas as pd
from datetime import date

from multifactor_data_layer_v2 import FEATURES_59
from strict_delivery_walkforward_audit import (
    policy_signals,
    prepare_labeled_panel,
    quality_gate_from_readiness_flags,
    session_successors,
    summarize_production_inventory,
)


def test_successor_respects_weekend_special_session_and_missing_weekday():
    sessions = pd.to_datetime(["2026-01-02", "2026-01-04", "2026-01-05", "2026-01-07"]).tolist()
    successors, gaps = session_successors(sessions, set())
    assert successors[pd.Timestamp("2026-01-02")] == pd.Timestamp("2026-01-04")
    assert successors[pd.Timestamp("2026-01-04")] == pd.Timestamp("2026-01-05")
    assert successors[pd.Timestamp("2026-01-05")] is None
    assert any(g["missing_expected_session"] == "2026-01-06" for g in gaps)


def test_panel_targets_exact_next_session_and_checks_decision_clock():
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"])
    rows = []
    for symbol in ("AAA", "BBB"):
        for idx, day in enumerate(dates):
            row = {
                "symbol": symbol, "date": day,
                "decision_timestamp": (day + pd.Timedelta(hours=15, minutes=30)).tz_localize("Asia/Kolkata").tz_convert("UTC"),
                "close": 100.0 + idx, "production_data_quality_gate": True,
            }
            row.update({feature: float(idx + 1) for feature in FEATURES_59})
            rows.append(row)
    labeled, report = prepare_labeled_panel(pd.DataFrame(rows), {pd.Timestamp("2026-01-12").date()})
    assert len(labeled) == 10
    assert report["duplicate_symbol_session_rows"] == 0
    assert labeled["target_date"].gt(labeled["reference_date"]).all()
    assert labeled["feature_coverage"].eq(1.0).all()


def test_signals_abstain_when_quality_or_feature_gate_fails():
    predictions = pd.DataFrame({
        "probability_up": [0.8, 0.8, 0.2],
        "predicted_return": [0.02, 0.02, -0.02],
        "eligible_for_action": [True, False, True],
    })
    result = policy_signals(predictions, 0.6, 0.4, 0.0015)
    assert result.tolist() == ["BUY", "ABSTAIN", "SELL"]
    assert np.all(policy_signals(predictions, 0.6, 0.4, 0.0015, abstain_all=True) == "ABSTAIN")



def test_production_inventory_checks_current_dated_snapshot_not_research_panel():
    target = date(2026, 10, 12)
    inventory = {
        "eod": {date(2026, 10, 9), target},
        "delivery": {date(2026, 10, 9), target},
        "pit": {date(2026, 10, 9)},
    }
    report = summarize_production_inventory(inventory, target)
    assert report["status"] == "CURRENT_SESSION_PIT_MISSING_SOURCES_READY"
    assert report["target_eod_archive_present"] is True
    assert report["target_delivery_archive_present"] is True
    assert report["target_pit_snapshot_present"] is False
    assert report["latest_production_pit_snapshot_date"] == "2026-10-09"
    assert report["read_only"] is True


def test_production_inventory_reports_ready_only_when_all_three_files_exist():
    target = date(2026, 10, 12)
    inventory = {
        "eod": {target},
        "delivery": {target},
        "pit": {target},
    }
    report = summarize_production_inventory(inventory, target)
    assert report["status"] == "CURRENT_SESSION_PIT_AND_SOURCES_READY"
    assert report["target_eod_archive_present"] is True
    assert report["target_delivery_archive_present"] is True
    assert report["target_pit_snapshot_present"] is True


def test_production_inventory_handles_no_target_session():
    report = summarize_production_inventory({"eod": set(), "delivery": set(), "pit": set()}, None)
    assert report["status"] == "NO_NSE_SESSION_EXPECTED"
    assert report["target_session"] is None



def test_readiness_flags_reconstruct_existing_quality_gate_without_relaxing_unknowns():
    ready = pd.Series({
        "ready_market_price": 1,
        "ready_market_context": 1,
        "ready_news": 0,
        "ready_events": 0,
        "ready_fno": 0,
        "ready_fundamentals": 0,
    })
    assert quality_gate_from_readiness_flags(ready) is True
    ready["ready_news"] = np.nan
    assert quality_gate_from_readiness_flags(ready) is False


def test_research_cohort_excludes_pre_availability_dates_and_keeps_exact_next_session_label():
    dates = pd.to_datetime([
        "2025-05-30", "2025-06-02", "2025-06-03", "2025-06-04",
        "2025-06-05", "2025-06-06", "2025-06-09",
    ])
    rows = []
    for idx, day in enumerate(dates):
        row = {
            "symbol": "AAA",
            "date": day,
            "decision_timestamp": (
                day + pd.Timedelta(hours=15, minutes=30)
            ).tz_localize("Asia/Kolkata").tz_convert("UTC"),
            "close": 100.0 + idx,
            "production_data_quality_gate": True,
        }
        row.update({feature: float(idx + 1) for feature in FEATURES_59})
        rows.append(row)
    labeled, report = prepare_labeled_panel(
        pd.DataFrame(rows),
        set(),
        research_start_date=date(2025, 6, 1),
    )
    assert labeled["reference_date"].min() >= pd.Timestamp("2025-06-01")
    assert report["research_cohort_start_date"] == "2025-06-01"
    assert report["rows_excluded_before_research_cohort"] == 1
    assert pd.Timestamp("2025-06-02") in set(labeled["reference_date"])
    row = labeled.loc[labeled["reference_date"].eq(pd.Timestamp("2025-06-02"))].iloc[0]
    assert row["target_date"] == pd.Timestamp("2025-06-03")
