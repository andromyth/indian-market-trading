from __future__ import annotations

from datetime import datetime, timedelta, timezone

import indstocks_backfill_all_nse_5m_history as backfill
import indstocks_client


def test_market_guard_is_rechecked_before_each_retry(monkeypatch):
    request_calls = []
    guard_calls = []
    sleeps = []

    def flaky_request(*_args, **_kwargs):
        request_calls.append(1)
        raise RuntimeError("HTTP 503 temporary provider failure")

    def market_guard():
        guard_calls.append(1)
        if len(guard_calls) == 2:
            raise RuntimeError("market window has started")

    monkeypatch.setattr(indstocks_client, "_request", flaky_request)
    monkeypatch.setattr(backfill.time, "sleep", lambda seconds: sleeps.append(seconds))

    start = datetime(2026, 1, 5, tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    try:
        backfill.fetch_indstocks_5m_batch(
            [("TEST", "123")],
            start,
            end,
            pause_seconds=0.2,
            request_guard=market_guard,
        )
    except RuntimeError as exc:
        assert str(exc) == "market window has started"
    else:
        raise AssertionError("market guard should stop the retry")

    assert len(request_calls) == 1
    assert len(guard_calls) == 2
    assert sleeps == [2]
