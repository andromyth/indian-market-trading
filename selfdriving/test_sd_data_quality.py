"""Unit tests for autonomous point-in-time data auditing."""
from datetime import date
import unittest

import pandas as pd

from selfdriving.sd_data_quality import derive_realised_returns, merge_realised_returns, next_nse_session


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

    def test_known_trading_session_calendar_skips_weekend(self):
        self.assertEqual(next_nse_session(date(2026, 10, 9)), date(2026, 10, 12))


if __name__ == "__main__":
    unittest.main()
