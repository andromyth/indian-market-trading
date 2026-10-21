from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backfill_missing_delivery_pit_snapshots import _write_report, plan_snapshot_backfill
from nse_bulk_eod_ingest import business_days


class PITSnapshotBackfillPlanTests(unittest.TestCase):
    def test_october_holiday_is_not_expected_and_missing_sessions_are_backfilled(self):
        start = date(2026, 9, 28)
        target = date(2026, 10, 12)
        sessions = set(business_days(start, target))
        self.assertNotIn(date(2026, 10, 2), sessions)
        self.assertIn(date(2026, 10, 1), sessions)
        self.assertIn(date(2026, 10, 5), sessions)

        eod = set(sessions)
        delivery = set(sessions)
        pit = {
            date(2026, 10, 1),
            date(2026, 10, 5),
            date(2026, 10, 9),
        }
        plan = plan_snapshot_backfill(
            target_date=target,
            lookback_calendar_days=14,
            eod_dates=eod,
            delivery_dates=delivery,
            pit_dates=pit,
            max_backfills_per_run=4,
        )
        self.assertNotIn("2026-10-02", plan["expected_nse_sessions"])
        self.assertNotIn("2026-10-12", plan["expected_nse_sessions"])
        self.assertEqual(
            plan["missing_pit_dates_with_sources_ready"],
            [
                "2026-09-28",
                "2026-09-29",
                "2026-09-30",
                "2026-10-06",
                "2026-10-07",
                "2026-10-08",
            ],
        )
        self.assertEqual(
            plan["backfill_dates_this_run"],
            ["2026-09-28", "2026-09-29", "2026-09-30", "2026-10-06"],
        )
        self.assertEqual(plan["deferred_backfill_dates"], ["2026-10-07", "2026-10-08"])

    def test_audit_includes_target_session_and_detects_missing_sources(self):
        target = date(2026, 10, 12)
        sessions = set(business_days(date(2026, 9, 28), target))
        eod = set(sessions)
        delivery = set(sessions)
        pit = set(sessions)
        eod.remove(date(2026, 10, 7))
        delivery.remove(date(2026, 10, 8))
        pit.remove(date(2026, 10, 9))

        plan = plan_snapshot_backfill(
            target_date=target,
            lookback_calendar_days=14,
            eod_dates=eod,
            delivery_dates=delivery,
            pit_dates=pit,
            max_backfills_per_run=4,
            include_target=True,
        )
        self.assertIn("2026-10-12", plan["expected_nse_sessions"])
        self.assertIn("2026-10-07", plan["missing_eod_dates"])
        self.assertIn("2026-10-08", plan["missing_delivery_dates"])
        self.assertIn("2026-10-09", plan["missing_pit_dates_with_sources_ready"])

    def test_empty_or_invalid_bounds_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "lookback_calendar_days"):
            plan_snapshot_backfill(
                target_date=date(2026, 10, 12),
                lookback_calendar_days=0,
                eod_dates=set(),
                delivery_dates=set(),
                pit_dates=set(),
                max_backfills_per_run=4,
            )
        with self.assertRaisesRegex(ValueError, "max_backfills_per_run"):
            plan_snapshot_backfill(
                target_date=date(2026, 10, 12),
                lookback_calendar_days=14,
                eod_dates=set(),
                delivery_dates=set(),
                pit_dates=set(),
                max_backfills_per_run=0,
            )


class PITSnapshotReportRetryTests(unittest.TestCase):
    def test_report_write_retries_a_transient_drive_timeout(self):
        class FlakyStore:
            def __init__(self):
                self.calls = 0
                self.remote_paths = []

            def upload_or_update(self, local_path, remote_path, mime_type=None):
                self.calls += 1
                self.remote_paths.append(remote_path)
                if self.calls == 1:
                    raise TimeoutError("The read operation timed out")
                return {"status": "OK"}

        store = FlakyStore()
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "continuity.json"
            with (
                patch("backfill_missing_delivery_pit_snapshots.REPORT_PATH", path),
                patch("backfill_missing_delivery_pit_snapshots.time.sleep") as sleep,
            ):
                _write_report(store, {"status": "PASS"})
                self.assertTrue(path.exists())
                self.assertEqual(store.calls, 2)
                self.assertEqual(store.remote_paths, [
                    "diagnostics/production/pit_snapshot_continuity_latest.json",
                    "diagnostics/production/pit_snapshot_continuity_latest.json",
                ])
                sleep.assert_called_once_with(1)

    def test_report_write_does_not_retry_nontransient_errors(self):
        class FailingStore:
            def __init__(self):
                self.calls = 0

            def upload_or_update(self, local_path, remote_path, mime_type=None):
                self.calls += 1
                raise ValueError("invalid report")

        store = FailingStore()
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "continuity.json"
            with (
                patch("backfill_missing_delivery_pit_snapshots.REPORT_PATH", path),
                patch("backfill_missing_delivery_pit_snapshots.time.sleep") as sleep,
            ):
                with self.assertRaisesRegex(ValueError, "invalid report"):
                    _write_report(store, {"status": "PASS"})
                self.assertEqual(store.calls, 1)
                sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
