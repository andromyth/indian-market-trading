from __future__ import annotations

import unittest
from datetime import date

from backfill_missing_delivery_pit_snapshots import plan_snapshot_backfill
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


if __name__ == "__main__":
    unittest.main()
