from __future__ import annotations

import io
import json
import threading

import pytest

import autonomous_intraday_worker_service as service
import indstocks_client


@pytest.fixture(autouse=True)
def reset_backfill_state(monkeypatch):
    monkeypatch.setattr(service, "BACKFILL_START_LOCK", threading.Lock())
    monkeypatch.setattr(service, "BACKFILL_THREAD", None)
    with service.STATE_LOCK:
        service.STATE["five_minute_backfill"] = {"status": "IDLE"}
    yield


def _post_backfill(body: bytes):
    handler = service.WorkerHandler.__new__(service.WorkerHandler)
    handler.path = "/backfill"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    response = {}
    handler.send_response = lambda code: response.update(http_status=code)
    handler.send_header = lambda *_args: None
    handler.end_headers = lambda: None

    handler.do_POST()

    response["body"] = json.loads(handler.wfile.getvalue().decode("utf-8"))
    return response


def _confirmed_body() -> bytes:
    return json.dumps({"confirm": service.BACKFILL_CONFIRMATION}).encode("utf-8")


def test_backfill_requires_exact_explicit_confirmation(monkeypatch):
    monkeypatch.setattr(
        service,
        "_preflight",
        lambda: pytest.fail("preflight must not run without explicit confirmation"),
    )

    for body in (
        b"",
        b"{",
        b'{"confirm":"YES"}',
        b'{"confirm":"start_native_5m_backfill"}',
    ):
        response = _post_backfill(body)
        assert response["http_status"] == 400
        assert response["body"]["status"] == "BACKFILL_CONFIRMATION_REQUIRED"
        assert response["body"]["required_confirmation"] == service.BACKFILL_CONFIRMATION

    response = _post_backfill(b"x" * 1025)
    assert response["http_status"] == 400
    assert response["body"]["status"] == "BACKFILL_CONFIRMATION_REQUIRED"


def test_backfill_requires_strict_drive_write_preflight(monkeypatch):
    monkeypatch.setattr(
        service,
        "_preflight",
        lambda: {
            "ready": True,
            "configured": True,
            "reachable": True,
            "write_ready": None,
            "reason": None,
        },
    )
    monkeypatch.setattr(
        indstocks_client,
        "configured",
        lambda: pytest.fail("INDstocks check must not run after failed Drive gate"),
    )

    response = _post_backfill(_confirmed_body())

    assert response["http_status"] == 503
    assert response["body"]["status"] == "DRIVE_PREFLIGHT_FAILED"
    assert response["body"]["checks"]["write_ready"] is None
    assert service.BACKFILL_THREAD is None


def test_backfill_rejects_when_an_existing_run_is_active(monkeypatch):
    class RunningThread:
        @staticmethod
        def is_alive():
            return True

    service.BACKFILL_THREAD = RunningThread()
    with service.STATE_LOCK:
        service.STATE["five_minute_backfill"] = {"status": "RUNNING"}
    monkeypatch.setattr(
        service,
        "_preflight",
        lambda: pytest.fail("active run must be rejected before preflight"),
    )

    response = _post_backfill(_confirmed_body())

    assert response["http_status"] == 409
    assert response["body"]["status"] == "BACKFILL_ALREADY_RUNNING"


def test_backfill_requires_configured_indstocks_credentials(monkeypatch):
    monkeypatch.setattr(
        service,
        "_preflight",
        lambda: {
            "ready": True,
            "configured": True,
            "reachable": True,
            "write_ready": True,
        },
    )
    monkeypatch.setattr(indstocks_client, "configured", lambda: False)

    response = _post_backfill(_confirmed_body())

    assert response["http_status"] == 503
    assert response["body"]["status"] == "INDSTOCKS_CREDENTIALS_NOT_CONFIGURED"
    assert service.BACKFILL_THREAD is None


def test_backfill_starts_one_in_process_thread_after_all_gates_pass(monkeypatch):
    monkeypatch.setattr(
        service,
        "_preflight",
        lambda: {
            "ready": True,
            "configured": True,
            "reachable": True,
            "write_ready": True,
        },
    )
    monkeypatch.setattr(indstocks_client, "configured", lambda: True)

    created = []

    class StubThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon
            self.started = False
            created.append(self)

        def start(self):
            self.started = True

        def is_alive(self):
            return self.started

    monkeypatch.setattr(service.threading, "Thread", StubThread)

    response = _post_backfill(_confirmed_body())

    assert response["http_status"] == 202
    assert response["body"]["status"] == "BACKFILL_STARTED"
    assert response["body"]["status_endpoint"] == "/backfillz"
    assert len(created) == 1
    assert created[0].target is service._run_five_minute_backfill
    assert created[0].name == "indstocks-native-5m-backfill"
    assert created[0].daemon is True
    assert created[0].started is True
    assert service.BACKFILL_THREAD is created[0]
