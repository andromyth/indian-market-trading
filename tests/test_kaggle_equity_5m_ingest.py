import unittest
import numpy as np
import pandas as pd
from ingest_kaggle_equity_5m import prepare_symbol_bars

class TestKaggleEquity5m(unittest.TestCase):
    def test_minute_start_labels_become_75_completed_bars(self):
        ts = pd.date_range("2026-01-05 09:15", periods=375, freq="min")
        f = pd.DataFrame({
            "Timestamp":ts,"Open":100.0,"High":101.0,"Low":99.0,
            "Close":100.5,"Volume":1
        })
        out, meta = prepare_symbol_bars(f)
        self.assertEqual(len(out),75)
        self.assertEqual(out.Timestamp.iloc[0],pd.Timestamp("2026-01-05 09:20",tz="Asia/Kolkata"))
        self.assertEqual(out.Timestamp.iloc[-1],pd.Timestamp("2026-01-05 15:30",tz="Asia/Kolkata"))
        self.assertEqual(int(out.Volume.sum()),375)
        self.assertEqual(meta["source_interval"],"1m")

    def test_five_minute_start_labels_shift_once(self):
        ts = pd.date_range("2026-01-05 09:15", periods=75, freq="5min")
        f = pd.DataFrame({
            "Timestamp":ts,"Open":100.0,"High":101.0,"Low":99.0,
            "Close":100.5,"Volume":50
        })
        out, meta = prepare_symbol_bars(f)
        self.assertEqual(len(out),75)
        self.assertEqual(out.Timestamp.iloc[0],pd.Timestamp("2026-01-05 09:20",tz="Asia/Kolkata"))
        self.assertEqual(out.Timestamp.iloc[-1],pd.Timestamp("2026-01-05 15:30",tz="Asia/Kolkata"))
        self.assertEqual(meta["source_interval"],"5m")

    def test_conflicting_duplicate_is_removed(self):
        ts = pd.date_range("2026-01-05 09:15", periods=75, freq="5min")
        f = pd.DataFrame({
            "Timestamp":ts,"Open":100.0,"High":101.0,"Low":99.0,
            "Close":100.5,"Volume":50
        })
        extra = f.iloc[[5]].copy()
        extra["Close"] = 100.8
        out, meta = prepare_symbol_bars(pd.concat([f,extra],ignore_index=True),"5m","start")
        self.assertEqual(len(out),74)
        self.assertEqual(meta["conflicting_duplicate_timestamps_excluded"],1)

    def test_short_session_rejected(self):
        ts = pd.date_range("2026-01-05 09:15", periods=45, freq="5min")
        f = pd.DataFrame({
            "Timestamp":ts,"Open":100.0,"High":101.0,"Low":99.0,
            "Close":100.5,"Volume":50
        })
        with self.assertRaisesRegex(ValueError,"50-bar minimum"):
            prepare_symbol_bars(f,"5m","start")

if __name__ == "__main__":
    unittest.main()
