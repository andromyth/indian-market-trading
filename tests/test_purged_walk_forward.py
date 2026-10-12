from __future__ import annotations

import unittest

import pandas as pd

from purged_walk_forward import purged_chronological_split


class PurgedChronologicalSplitTests(unittest.TestCase):
    def test_rows_whose_target_crosses_cutoff_are_purged(self):
        dates = pd.date_range("2026-01-01", periods=11, freq="D", tz="UTC")
        frame = pd.DataFrame({
            "decision_timestamp": dates[:-1],
            "_target_timestamp": dates[1:],
            "symbol": ["AAA"] * 10,
            "target": list(range(10)),
        })
        train, holdout, audit = purged_chronological_split(
            frame, cutoff_days=5
        )
        cutoff = pd.Timestamp(audit["cutoff_timestamp"])
        self.assertTrue(
            (pd.to_datetime(train["_target_timestamp"], utc=True) < cutoff).all()
        )
        self.assertTrue(
            (pd.to_datetime(holdout["decision_timestamp"], utc=True) >= cutoff).all()
        )
        self.assertEqual(audit["purged_boundary_rows"], 1)
        self.assertEqual(audit["target_boundary_check"], "PASS")
        self.assertEqual(len(train), 4)
        self.assertEqual(len(holdout), 5)

    def test_missing_target_timestamps_are_not_used_for_training(self):
        dates = pd.date_range("2026-01-01", periods=11, freq="D", tz="UTC")
        frame = pd.DataFrame({
            "decision_timestamp": dates[:-1],
            "_target_timestamp": dates[1:],
        })
        frame.loc[1, "_target_timestamp"] = pd.NaT
        train, holdout, audit = purged_chronological_split(frame, cutoff_days=5)
        self.assertNotIn(1, train.index)
        self.assertGreaterEqual(len(holdout), 1)
        self.assertEqual(audit["pre_cutoff_rows_without_target_timestamp"], 1)

    def test_missing_timestamp_contract_fails_closed(self):
        frame = pd.DataFrame({"decision_timestamp": pd.date_range("2026-01-01", periods=3)})
        with self.assertRaisesRegex(ValueError, "Missing required timestamp columns"):
            purged_chronological_split(frame, cutoff_days=1)

    def test_split_never_silently_accepts_empty_train_or_holdout(self):
        frame = pd.DataFrame({
            "decision_timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
            "_target_timestamp": pd.date_range("2026-01-02", periods=3, tz="UTC"),
        })
        with self.assertRaisesRegex(ValueError, "empty train or holdout"):
            purged_chronological_split(frame, cutoff_days=60)


if __name__ == "__main__":
    unittest.main()
