"""Best-effort per-slot tick lease for the Cloud Run intraday worker.

Cloud Run runs with ``--concurrency 1``. When ``--max-instances`` is greater
than one, a request that arrives while the first instance is still busy can be
routed to a second instance instead of returning 429. That protects the
five-minute cadence, but two instances must not run the *same* slot twice.

The lease is a tiny JSON document under ``production_data/autonomous_worker``
keyed by the slot minute (``YYYYMMDD_HHMM``). It is best-effort: the caller
fails open if the lease store is unavailable, because losing a slot is worse
than a rare duplicate. A crashed holder's lease simply expires after
``ttl_seconds`` and the next tick (or the backlog drain) recovers the slot.

The store interaction is injected as a small ``LeaseIO`` adapter so the lease
logic is pure and unit-testable without Google Drive.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

LEASE_REMOTE_DIR = "production_data/autonomous_worker/tick_lease"
DEFAULT_LEASE_TTL_SECONDS = 120


class LeaseIO(Protocol):
    """Minimal read/write adapter over a remote lease document."""

    def read(self, remote_path: str) -> dict[str, Any] | None: ...

    def write(self, remote_path: str, payload: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class LeaseDecision:
    acquired: bool
    reason: str
    existing_owner: str | None = None


def lease_remote_path(slot_key: str) -> str:
    return f"{LEASE_REMOTE_DIR}/lease_{slot_key}.json"


def lease_payload(
    owner: str,
    slot_key: str,
    now: datetime,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
) -> dict[str, Any]:
    expires = now + timedelta(seconds=max(1, int(ttl_seconds)))
    return {
        "owner": str(owner),
        "slot_key": str(slot_key),
        "acquired_at_utc": now.astimezone(timezone.utc).isoformat(),
        "expires_at_utc": expires.astimezone(timezone.utc).isoformat(),
    }


def _parse_utc(value: Any) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def is_expired(payload: dict[str, Any] | None, now: datetime) -> bool:
    if not payload:
        return True
    expires = _parse_utc(payload.get("expires_at_utc"))
    if expires is None:
        return True
    return now.astimezone(timezone.utc) >= expires


def can_acquire(
    payload: dict[str, Any] | None,
    owner: str,
    now: datetime,
) -> LeaseDecision:
    if not payload:
        return LeaseDecision(True, "no existing lease")
    existing_owner = str(payload.get("owner") or "") or None
    if existing_owner == str(owner):
        return LeaseDecision(True, "re-entrant lease", existing_owner)
    if is_expired(payload, now):
        return LeaseDecision(True, "previous lease expired", existing_owner)
    return LeaseDecision(False, "lease held by another instance", existing_owner)


def acquire_lease(
    io: LeaseIO,
    slot_key: str,
    owner: str,
    now: datetime,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
) -> LeaseDecision:
    """Try to claim ``slot_key`` for ``owner``.

    Raises nothing for a held lease; the caller inspects ``decision.acquired``.
    Store errors propagate so the caller can fail open explicitly.
    """
    remote = lease_remote_path(slot_key)
    existing = io.read(remote)
    decision = can_acquire(existing, owner, now)
    if not decision.acquired:
        return decision
    io.write(remote, lease_payload(owner, slot_key, now, ttl_seconds))
    return LeaseDecision(True, decision.reason, decision.existing_owner)


class InMemoryLeaseIO:
    """Trivial adapter used by tests and as a last-resort fallback."""

    def __init__(self, initial: dict[str, dict[str, Any]] | None = None) -> None:
        self.documents: dict[str, dict[str, Any]] = dict(initial or {})

    def read(self, remote_path: str) -> dict[str, Any] | None:
        document = self.documents.get(remote_path)
        return dict(document) if document is not None else None

    def write(self, remote_path: str, payload: dict[str, Any]) -> None:
        self.documents[remote_path] = json.loads(json.dumps(payload))
