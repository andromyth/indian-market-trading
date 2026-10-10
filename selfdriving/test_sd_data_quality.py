"""Unit tests for autonomous point-in-time data auditing."""
from datetime import date
import unittest

import pandas as pd

from selfdriving.sd_data_quality import (
    build_input_audit, derive_realised_returns, merge_realised_returns,
    is_nse_session_day, next_nse_session, parse_pit_date,
)


def snap(rows):
    return pd.DataFrame(rows, columns=["symbol", "close"])


class RealisedReturnTests(unittest.TestCase):
    def test_adjacent_sessions_create_one_session_returns(self):
        snapshots = [
            (date(2026, 10, 7), snap([("AAA", 100), ("BBB.NS", 50)])),
            (date(2026, 10, 8), snap([("AAA", 110), ("BBB", 55)])),
            (date(2026, 10, 9), snap([("AAA", 99), ("BBB", 60.5)])),
        ]
        result, gaps = derive_realised_returns(snapshots)
        self.assertFalse(result.empty)
        self.assertEqual(len(gaps), 0)
        aaa = result[result["symbol"].eq("AAA")].sort_values("date")
        self.assertAlmostEqual(float(aaa.iloc[0]["y_return"]), 0.10, places=8)
        self.assertAlmostEqual(float(aaa.iloc[1]["y_return"]), -0.10, places=8)

    def test_missing_nse_session_is_not_mislabeled_as_next_session(self):
        snapshots = [
            (date(2026, 10, 7), snap([("AAA", 100)])),
            (date(2026, 10, 9), snap([("AAA", 120)])),
        ]
        result, gaps = derive_realised_returns(snapshots)
        self.assertTrue(result.empty)
        self.assertEqual(gaps[0]["status"], "SKIPPED_SESSION_GAP")
        self.assertEqual(gaps[0]["expected_next_session"], "2026-10-08")

    def test_merge_keeps_history_and_refreshes_matching_label(self):
        old = pd.DataFrame([("AAA", "2026-10-07", 0.01)], columns=["symbol", "date", "y_return"])
        fresh = pd.DataFrame([("AAA", "2026-10-07", 0.02), ("BBB", "2026-10-08", -0.01)],
                             columns=["symbol", "date", "y_return"])
        merged, stats = merge_realised_returns(old, fresh)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(float(merged.iloc[0]["y_return"]), 0.02, places=8)
        self.assertEqual(stats["new_keys"], 1)
        self.assertEqual(stats["refreshed_keys"], 1)


    def test_pit_snapshot_date_parser_rejects_noncanonical_names(self):
        self.assertEqual(parse_pit_date("DELIVERY_PIT_FEATURES_2026-10-09.parquet"), date(2026, 10, 9))
        self.assertIsNone(parse_pit_date("DELIVERY_PIT_FEATURES_latest.parquet"))

    def test_audit_reports_symbolwise_usdinr_and_volume_coverage(self):
        from multifactor_data_layer_v2 import FEATURES_59

        rows = []
        for symbol in ("AAA", "BBB"):
            row = {"symbol": symbol, "close": 100.0, "volume": 1000.0, "production_data_quality_gate": True}
            row.update({feature: 0.1 for feature in FEATURES_59})
            rows.append(row)
        pit = pd.DataFrame(rows)
        panel = pd.DataFrame([
            {"symbol": "AAA", "g_inr_z": 0.2, "news_available": 0.0, "news_sent_ewm": 0.1, "fno_buildup": 0.2},
            {"symbol": "BBB", "g_inr_z": float("nan"), "news_available": float("nan"), "news_sent_ewm": float("nan"), "fno_buildup": float("nan")},
        ])
        report = build_input_audit(
            pit, panel, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            expected_date=date(2026, 10, 9),
            observed_date=date(2026, 10, 9),
            now=pd.Timestamp("2026-10-10T00:00:00Z"),
        )
        self.assertEqual(report["hard_failures"], [])
        self.assertEqual(len(report["stockwise_input_readiness"]), 2)
        self.assertEqual(report["stockwise_source_coverage"]["symbols_with_usdinr_feature"], 1)
        self.assertEqual(report["stockwise_source_coverage"]["symbols_with_daily_volume_ratio"], 2)

    def test_nse_session_guard_skips_weekends_and_known_holidays(self):
        self.assertTrue(is_nse_session_day(date(2026, 10, 12)))
        self.assertFalse(is_nse_session_day(date(2026, 10, 10)))
        self.assertFalse(is_nse_session_day(date(2026, 10, 20)))

    def test_known_trading_session_calendar_skips_weekend(self):
        self.assertEqual(next_nse_session(date(2026, 10, 9)), date(2026, 10, 12))


if __name__ == "__main__":
    unittest.main()
