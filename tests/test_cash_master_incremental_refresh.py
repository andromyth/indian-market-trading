from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from refresh_canonical_cash_master_from_bulk_eod import _merge_master, _master_stats


class IncrementalCashMasterMergeTests(unittest.TestCase):
    def test_new_official_rows_append_and_overlapping_identity_is_refreshed(self):
        with tempfile.TemporaryDirectory(prefix="cash-master-test-") as directory:
            root = Path(directory)
            master = root / "master.parquet"
            incoming = root / "incoming.parquet"
            output = root / "output.parquet"

            pd.DataFrame([
                {"row_identity_key": "ISIN:AAA|SERIES:EQ|SYMBOL:AAA|DATE:2026-10-08", "symbol": "AAA", "date": "2026-10-08", "listing_id": "ISIN:AAA|SERIES:EQ|SYMBOL:AAA", "close": 100.0},
                {"row_identity_key": "ISIN:BBB|SERIES:EQ|SYMBOL:BBB|DATE:2026-10-08", "symbol": "BBB", "date": "2026-10-08", "listing_id": "ISIN:BBB|SERIES:EQ|SYMBOL:BBB", "close": 200.0},
            ]).to_parquet(master, index=False)
            pd.DataFrame([
                {"row_identity_key": "ISIN:AAA|SERIES:EQ|SYMBOL:AAA|DATE:2026-10-08", "symbol": "AAA", "date": "2026-10-08", "listing_id": "ISIN:AAA|SERIES:EQ|SYMBOL:AAA", "close": 101.0},
                {"row_identity_key": "ISIN:CCC|SERIES:EQ|SYMBOL:CCC|DATE:2026-10-09", "symbol": "CCC", "date": "2026-10-09", "listing_id": "ISIN:CCC|SERIES:EQ|SYMBOL:CCC", "close": 300.0},
            ]).to_parquet(incoming, index=False)

            result = _merge_master(master, incoming, output)
            merged = pd.read_parquet(output).set_index("row_identity_key")

            self.assertEqual(result["incoming_rows"], 2)
            self.assertEqual(result["new_identity_keys"], 1)
            self.assertEqual(result["overlapping_identity_keys_refreshed_from_official_archive"], 1)
            self.assertEqual(len(merged), 3)
            self.assertEqual(merged.loc["ISIN:AAA|SERIES:EQ|SYMBOL:AAA|DATE:2026-10-08", "close"], 101.0)
            self.assertIn("ISIN:CCC|SERIES:EQ|SYMBOL:CCC|DATE:2026-10-09", merged.index)
            stats = _master_stats(output)
            self.assertEqual(stats["duplicate_identity_key_rows"], 0)
            self.assertEqual(stats["null_identity_key_rows"], 0)
            self.assertEqual(stats["max_date"], "2026-10-09")

    def test_refresh_rejects_master_without_identity_key(self):
        with tempfile.TemporaryDirectory(prefix="cash-master-test-") as directory:
            root = Path(directory)
            master = root / "master.parquet"
            incoming = root / "incoming.parquet"
            output = root / "output.parquet"
            pd.DataFrame([{"symbol": "AAA", "date": "2026-10-08", "close": 100.0}]).to_parquet(master, index=False)
            pd.DataFrame([{"symbol": "AAA", "date": "2026-10-09", "close": 101.0}]).to_parquet(incoming, index=False)
            with self.assertRaisesRegex(RuntimeError, "lacks row_identity_key"):
                _merge_master(master, incoming, output)


if __name__ == "__main__":
    unittest.main()
