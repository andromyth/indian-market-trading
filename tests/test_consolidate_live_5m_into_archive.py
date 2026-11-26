from __future__ import annotations

import pandas as pd

import consolidate_live_5m_into_archive as mod


def _frame(day: str) -> pd.DataFrame:
    index = pd.date_range(f"{day} 09:20", periods=60, freq="5min", tz=mod.IST)
    return pd.DataFrame(
        {
            "Open": [100.0] * 60,
            "High": [101.0] * 60,
            "Low": [99.0] * 60,
            "Close": [100.5] * 60,
            "Volume": [1000.0] * 60,
        },
        index=index,
    )


class _Store:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, int]] = []

    def upload_or_update(self, local, remote, mime_type=None):
        rows = len(pd.read_parquet(local))
        self.uploads.append((remote, rows))
        return {"status": "OK"}


def test_session_coverage_flags_missing_five_minute_slot():
    index = pd.date_range("2026-11-23 09:20", periods=74, freq="5min", tz=mod.IST)
    index = index[index != pd.Timestamp("2026-11-23 10:30", tz=mod.IST)]
    frame = pd.DataFrame({"Close": [100.0] * len(index)}, index=index)

    report = mod.session_coverage_report({"RELIANCE": frame}, ["RELIANCE", "HDFCBANK"])

    assert report["complete"] is False
    assert report["session"] == "2026-11-23"
    assert "10:30" in report["missing_slots"]


def test_session_coverage_accepts_full_grid():
    index = pd.date_range("2026-11-23 09:20", periods=74, freq="5min", tz=mod.IST)
    frame = pd.DataFrame({"Close": [100.0] * len(index)}, index=index)

    report = mod.session_coverage_report({"RELIANCE": frame, "HDFCBANK": frame},
                                         ["RELIANCE", "HDFCBANK"])

    assert report["complete"] is True
    assert report["missing_slots"] == []
    assert report["min_slot_coverage"] == 2


def test_consolidate_appends_live_sessions(monkeypatch, tmp_path):
    archive = _frame("2026-11-18")
    live = _frame("2026-11-19")
    monkeypatch.setattr(mod, "_day_snapshots", lambda *a, **k: {"RELIANCE": live})
    monkeypatch.setattr(mod, "_download_existing", lambda store, remote, local: archive.copy())
    monkeypatch.setattr(mod, "is_nse_trading_day", lambda d: True)
    store = _Store()

    report = mod.consolidate(
        store=store,
        symbols=["RELIANCE", "HDFCBANK"],
        days=1,
        end_date=pd.Timestamp("2026-11-19", tz=mod.IST),
        dry_run=False,
        work_root=tmp_path,
    )

    assert report["status"] == "OK"
    assert report["symbols_updated"] == 1
    assert report["rows_added"] == 60
    assert store.uploads[0][0].endswith("symbol=RELIANCE/RELIANCE.parquet")


def test_consolidate_dry_run_does_not_upload(monkeypatch, tmp_path):
    live = _frame("2026-11-19")
    monkeypatch.setattr(mod, "_day_snapshots", lambda *a, **k: {"RELIANCE": live})
    monkeypatch.setattr(mod, "_download_existing", lambda store, remote, local: live.copy())
    monkeypatch.setattr(mod, "is_nse_trading_day", lambda d: True)
    store = _Store()

    report = mod.consolidate(
        store=store,
        symbols=["RELIANCE"],
        days=1,
        end_date=pd.Timestamp("2026-11-19", tz=mod.IST),
        dry_run=True,
        work_root=tmp_path,
    )

    assert report["symbols_unchanged"] == 1
    assert store.uploads == []


def test_consolidate_coverage_counts_unchanged_and_live_less_symbols(monkeypatch, tmp_path):
    archives = {
        "RELIANCE": _frame("2026-11-18"),
        "HDFCBANK": _frame("2026-11-19"),
    }
    monkeypatch.setattr(
        mod,
        "_day_snapshots",
        lambda *a, **k: {"RELIANCE": _frame("2026-11-19")},
    )
    monkeypatch.setattr(
        mod,
        "_download_existing",
        lambda store, remote, local: archives[remote.split("symbol=")[1].split("/")[0]].copy(),
    )
    monkeypatch.setattr(mod, "is_nse_trading_day", lambda d: True)
    store = _Store()

    report = mod.consolidate(
        store=store,
        symbols=["RELIANCE", "HDFCBANK"],
        days=1,
        end_date=pd.Timestamp("2026-11-19", tz=mod.IST),
        dry_run=True,
        work_root=tmp_path,
    )

    coverage = report["latest_session_coverage"]
    assert report["symbols_updated"] == 1
    assert report["symbols_unchanged"] == 1
    assert coverage["session"] == "2026-11-19"
    assert coverage["complete"] is True
    assert coverage["min_slot_coverage"] == 2


def test_consolidate_coverage_fails_when_a_symbol_is_stale(monkeypatch, tmp_path):
    archives = {
        "RELIANCE": _frame("2026-11-18"),
        "HDFCBANK": _frame("2026-11-18"),
    }
    monkeypatch.setattr(
        mod,
        "_day_snapshots",
        lambda *a, **k: {"RELIANCE": _frame("2026-11-19")},
    )
    monkeypatch.setattr(
        mod,
        "_download_existing",
        lambda store, remote, local: archives[remote.split("symbol=")[1].split("/")[0]].copy(),
    )
    monkeypatch.setattr(mod, "is_nse_trading_day", lambda d: True)
    store = _Store()

    report = mod.consolidate(
        store=store,
        symbols=["RELIANCE", "HDFCBANK"],
        days=1,
        end_date=pd.Timestamp("2026-11-19", tz=mod.IST),
        dry_run=True,
        work_root=tmp_path,
    )

    coverage = report["latest_session_coverage"]
    assert coverage["session"] == "2026-11-19"
    assert coverage["complete"] is False
    assert coverage["min_slot_coverage"] == 1


def test_consolidate_dry_run_reports_new_rows_without_upload(monkeypatch, tmp_path):
    archive = _frame("2026-11-18")
    live = _frame("2026-11-19")
    monkeypatch.setattr(mod, "_day_snapshots", lambda *a, **k: {"RELIANCE": live})
    monkeypatch.setattr(mod, "_download_existing", lambda store, remote, local: archive.copy())
    monkeypatch.setattr(mod, "is_nse_trading_day", lambda d: True)
    store = _Store()

    report = mod.consolidate(
        store=store,
        symbols=["RELIANCE"],
        days=1,
        end_date=pd.Timestamp("2026-11-19", tz=mod.IST),
        dry_run=True,
        work_root=tmp_path,
    )

    assert report["symbols_updated"] == 1
    assert report["rows_added"] == 60
    assert store.uploads == []
