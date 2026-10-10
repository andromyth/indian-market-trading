"""Unit tests for autonomous point-in-time data auditing."""
from datetime import date, datetime
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

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

    def test_audit_distinguishes_available_inputs_from_active_champion_features(self):
        from multifactor_data_layer_v2 import FEATURES_59

        rows = []
        for symbol in ("AAA", "BBB"):
            row = {"symbol": symbol, "close": 100.0, "volume": 1000.0, "production_data_quality_gate": True}
            row.update({feature: 0.1 for feature in FEATURES_59})
            rows.append(row)
        pit = pd.DataFrame(rows)
        panel = pd.DataFrame([
            {"symbol": "AAA", "g_inr_z": 0.2, "g_usdinr_z1": 0.2, "news_sent_ewm": 0.3, "volume_ratio": 1.2},
            {"symbol": "BBB", "g_inr_z": float("nan"), "g_usdinr_z1": float("nan"), "news_sent_ewm": float("nan"), "volume_ratio": 1.1},
        ])
        report = build_input_audit(
            pit, panel, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            expected_date=date(2026, 10, 9),
            observed_date=date(2026, 10, 9),
            active_champion_features=["g_usdinr_z1", "news_sent_ewm", "volume_ratio"],
            active_champion_status="READY",
            active_champion_file="delivery_excess_test.joblib",
            now=pd.Timestamp("2026-10-10T00:00:00Z"),
        )
        model = report["active_model_contract"]
        self.assertEqual(model["status"], "READY")
        self.assertEqual(model["feature_count"], 3)
        self.assertTrue(model["usd_inr"]["selected_by_active_champion"])
        self.assertEqual(model["usd_inr"]["selected_feature_names"], ["g_usdinr_z1"])
        self.assertEqual(model["usd_inr"]["selected_feature_mean_non_null_pct"], 50.0)
        self.assertEqual(report["stockwise_source_coverage"]["symbols_with_active_champion_coverage_at_least_60pct"], 1)
        stock = {row["symbol"]: row for row in report["stockwise_input_readiness"]}
        self.assertEqual(stock["AAA"]["active_champion_feature_non_null_pct"], 100.0)
        self.assertAlmostEqual(stock["BBB"]["active_champion_feature_non_null_pct"], 33.333, places=2)

    def test_nse_session_guard_skips_weekends_and_known_holidays(self):
        self.assertTrue(is_nse_session_day(date(2026, 10, 12)))
        self.assertFalse(is_nse_session_day(date(2026, 10, 10)))
        self.assertFalse(is_nse_session_day(date(2026, 10, 20)))

    def test_known_trading_session_calendar_skips_weekend(self):
        self.assertEqual(next_nse_session(date(2026, 10, 9)), date(2026, 10, 12))



    def test_weekend_run_uses_last_completed_session_and_next_session_target(self):
        from selfdriving.sd_github_job_v2 import _current_expected_reference_date

        saturday = datetime(2026, 10, 10, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
        sunday = datetime(2026, 10, 11, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        self.assertEqual(_current_expected_reference_date(saturday), date(2026, 10, 9))
        self.assertEqual(_current_expected_reference_date(sunday), date(2026, 10, 9))
        self.assertEqual(next_nse_session(_current_expected_reference_date(saturday)), date(2026, 10, 12))


class EntrypointImportTests(unittest.TestCase):
    def test_beta_v2_script_can_import_root_modules_from_outside_repo(self):
        script = Path(__file__).resolve().with_name("sd_github_job_v2.py")
        code = (
            "import runpy; "
            f"runpy.run_path({str(script)!r}, run_name='beta_v2_import_check'); "
            "import persistent_store; print('ROOT_MODULE_IMPORT_OK')"
        )
        with tempfile.TemporaryDirectory() as working_dir:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=working_dir,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("ROOT_MODULE_IMPORT_OK", result.stdout)




class BetaV2SourceEdgeCaseTests(unittest.TestCase):
    def test_news_published_after_last_decision_remains_unknown(self):
        from selfdriving.sd_news_features import build_news_features, score_news_frame

        raw = pd.DataFrame([{
            "symbol": "AAA",
            "published_at": "2026-10-10T05:00:00Z",
            "text": "Company reports quarterly operational update",
        }])
        scored = score_news_frame(raw)
        decisions = pd.DataFrame({
            "symbol": ["AAA"],
            "decision_timestamp": pd.to_datetime(["2026-10-09T10:00:00Z"], utc=True),
        })
        result = build_news_features(scored, decisions)
        self.assertEqual(len(result), 1)
        for column in (
            "news_cnt_1",
            "news_sent_ewm",
            "news_cnt_ewm",
            "news_available",
            "mkt_news_sent_ewm",
            "mkt_news_cnt_ewm",
        ):
            self.assertTrue(pd.isna(result.loc[0, column]), msg=column)

    def test_realised_returns_exclude_rows_from_older_source_dates(self):
        from selfdriving import sd_github_job_v2 as beta_v2

        first_day = date(2026, 10, 8)
        second_day = date(2026, 10, 9)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_root = beta_v2.ROOT
            beta_v2.ROOT = root
            try:
                first = root / "DELIVERY_PIT_FEATURES_2026-10-08.parquet"
                second = root / "DELIVERY_PIT_FEATURES_2026-10-09.parquet"
                pd.DataFrame([
                    {"symbol": "AAA", "date": "2026-10-08", "close": 100.0},
                    {"symbol": "BBB", "date": "2026-10-07", "close": 50.0},
                ]).to_parquet(first, index=False)
                pd.DataFrame([
                    {"symbol": "AAA", "date": "2026-10-09", "close": 110.0},
                    {"symbol": "BBB", "date": "2026-10-09", "close": 55.0},
                ]).to_parquet(second, index=False)

                stats = beta_v2.refresh_realised_outcomes(
                    None, [(first_day, first), (second_day, second)]
                )
                realised = pd.read_parquet(root / "BETA_V2_DELIVERY_REALISED_RETURNS.parquet")
                self.assertEqual(stats["status"], "WARN")
                self.assertEqual(stats["rows_excluded_nonmatching_filename_date"], 1)
                self.assertEqual(stats["derived_rows"], 1)
                self.assertEqual(len(realised), 1)
                self.assertEqual(realised.iloc[0]["symbol"], "AAA")
                self.assertAlmostEqual(float(realised.iloc[0]["y_return"]), 0.10, places=8)
            finally:
                beta_v2.ROOT = previous_root



if __name__ == "__main__":
    unittest.main()
