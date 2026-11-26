"""Per-slot tick lease used when Cloud Run runs more than one instance."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from worker_tick_lease import (
    InMemoryLeaseIO,
    acquire_lease,
    can_acquire,
    is_expired,
    lease_payload,
    lease_remote_path,
)

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=timezone.utc)


def test_no_existing_lease_is_acquired():
    io = InMemoryLeaseIO()
    decision = acquire_lease(io, "20261126_1000", "worker-a", NOW, ttl_seconds=120)
    assert decision.acquired is True
    stored = io.read(lease_remote_path("20261126_1000"))
    assert stored["owner"] == "worker-a"
    assert stored["slot_key"] == "20261126_1000"


def test_second_owner_is_blocked_while_lease_is_live():
    io = InMemoryLeaseIO()
    acquire_lease(io, "20261126_1000", "worker-a", NOW, ttl_seconds=120)
    decision = acquire_lease(io, "20261126_1000", "worker-b", NOW + timedelta(seconds=30))
    assert decision.acquired is False
    assert decision.existing_owner == "worker-a"
    assert "another instance" in decision.reason


def test_reentrant_owner_is_allowed():
    io = InMemoryLeaseIO()
    acquire_lease(io, "20261126_1000", "worker-a", NOW, ttl_seconds=120)
    decision = acquire_lease(io, "20261126_1000", "worker-a", NOW + timedelta(seconds=10))
    assert decision.acquired is True


def test_expired_lease_can_be_reacquired():
    io = InMemoryLeaseIO()
    acquire_lease(io, "20261126_1000", "worker-a", NOW, ttl_seconds=120)
    decision = acquire_lease(
        io, "20261126_1000", "worker-b", NOW + timedelta(seconds=121), ttl_seconds=120
    )
    assert decision.acquired is True
    assert decision.existing_owner == "worker-a"


def test_is_expired_handles_missing_and_bad_payloads():
    assert is_expired(None, NOW) is True
    assert is_expired({}, NOW) is True
    assert is_expired({"expires_at_utc": "not-a-date"}, NOW) is True
    live = lease_payload("worker-a", "slot", NOW, ttl_seconds=120)
    assert is_expired(live, NOW + timedelta(seconds=30)) is False
    assert is_expired(live, NOW + timedelta(seconds=200)) is True


def test_can_acquire_reports_owner():
    held = lease_payload("worker-a", "slot", NOW, ttl_seconds=120)
    decision = can_acquire(held, "worker-b", NOW + timedelta(seconds=5))
    assert decision.acquired is False
    assert decision.existing_owner == "worker-a"
