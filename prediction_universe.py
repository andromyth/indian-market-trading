#!/usr/bin/env python3
"""Maintain the owner-pinned 500-symbol roster and affordable tradable picks.

The live five-minute capture and prediction pipeline uses only
backfill_universe.json. Dynamic selection statistics may be reported for
research, but they cannot replace or expand that live roster.

Pure selection functions remain independently testable; active_symbols
always returns the pinned roster.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PLAN_PATH = REPO_ROOT / "capital_plan.json"
PINNED_ROSTER_PATH = REPO_ROOT / "backfill_universe.json"

DELIVERY_LEDGER = "production_data/predictions/delivery_prediction_actual_ledger.csv"
FNO_FEATURES = "production_data/nse_official/fno/fno_daily_features_2026.parquet"
OUTCOMES_ROOT = "production_data/intraday_predictions/outcomes_audit"
PREDICTIONS_ROOT = "production_data/all_nse_minute_predictions"
SNAPSHOT_ROOT = "production_data/all_universe_minute_snapshots"
UNIVERSE_ROOT = "production_data/prediction_universe"
UNIVERSE_CURRENT = f"{UNIVERSE_ROOT}/current.json"
UNIVERSE_HISTORY = f"{UNIVERSE_ROOT}/history"
MARKET_CONTEXT_SYMBOL = "__NIFTY50__"
SCHEMA_VERSION = 1

DEFAULTS: dict[str, Any] = {
    "monthly_capital": 7000,
    "max_positions": 2,
    "reserve_amount": 150,
    "price_cap_multiplier": 0.95,
    "hard_price_cap": 2500,
    "growth": "reinvest_realised_pnl",
    "ladder": [
        {"capital": 7000, "positions": 2},
        {"capital": 10000, "positions": 3},
        {"capital": 15000, "positions": 4},
        {"capital": 25000, "positions": 5},
    ],
    "universe_size": 500,
    "delivery_lookback_sessions": 20,
    "intraday_lookback_days": 10,
    "min_records_profitable": 3,
    "activity_min_delivery_sessions": 8,
    "activity_min_intraday_days": 3,
    "top_active_fallback_count": 250,
    "min_move_delivery": 0.008,
    "min_move_intraday": 0.015,
    "manifest_max_age_days": 14,
}


# --------------------------------------------------------------------------- #
# Capital plan
# --------------------------------------------------------------------------- #
def load_capital_plan(path: str | Path | None = None) -> dict[str, Any]:
    """Load the repo capital plan, falling back to defaults for missing keys."""
    plan = dict(DEFAULTS)
    if path is None:
        path = DEFAULT_PLAN_PATH
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            plan.update({k: v for k, v in raw.items() if v is not None})
    except (OSError, ValueError):
        pass
    return plan


def authoritative_roster(path: str | Path | None = None) -> list[str]:
    """Owner-supplied authoritative symbol roster; never re-selected."""
    roster_path = Path(path) if path else PINNED_ROSTER_PATH
    try:
        data = json.loads(roster_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    values = data.get("symbols") if isinstance(data, dict) else data
    if not isinstance(values, list):
        return []
    return sorted({
        str(value).strip().upper().replace(".NS", "").replace(".BO", "")
        for value in values if str(value).strip()
    })


def position_budget(plan: dict[str, Any], realised_pnl: float = 0.0) -> dict[str, Any]:
    """Capital, per-position budget and price cap for the current equity."""
    capital = float(plan["monthly_capital"])
    if plan.get("growth") == "reinvest_realised_pnl":
        capital += max(0.0, float(realised_pnl or 0.0))
    positions = int(plan["max_positions"])
    for rung in sorted(plan.get("ladder") or [], key=lambda r: float(r["capital"])):
        if capital >= float(rung["capital"]):
            positions = int(rung["positions"])
    positions = max(1, positions)
    investable = max(0.0, capital - float(plan["reserve_amount"]))
    per_trade = investable / positions
    price_cap = min(
        float(plan["hard_price_cap"]),
        float(plan["price_cap_multiplier"]) * per_trade,
    )
    return {
        "capital": round(capital, 2),
        "positions": positions,
        "investable": round(investable, 2),
        "per_trade": round(per_trade, 2),
        "price_cap": round(price_cap, 2),
    }


# --------------------------------------------------------------------------- #
# Profitability statistics (pure)
# --------------------------------------------------------------------------- #
def _signed_return(frame: pd.DataFrame) -> pd.Series:
    """Signed realised return per ledger row; NaN where the trade is open."""
    if {"reference_close", "actual_close"} <= set(frame.columns):
        base = pd.to_numeric(frame["reference_close"], errors="coerce")
        realised = pd.to_numeric(frame["actual_close"], errors="coerce")
        direction = frame.get("signal", pd.Series(index=frame.index, dtype=object))
        sign = direction.map(lambda s: 1.0 if str(s).upper() == "BUY" else -1.0 if str(s).upper() == "SELL" else 0.0)
        return ((realised - base) / base.replace(0, pd.NA)) * sign
    if "signed_return" in frame.columns:
        return pd.to_numeric(frame["signed_return"], errors="coerce")
    if "actual_return" in frame.columns:
        return pd.to_numeric(frame["actual_return"], errors="coerce")
    return pd.Series(index=frame.index, dtype=float)


def _aggregate(frame: pd.DataFrame, returns: pd.Series, activity: pd.Series) -> dict[str, dict[str, Any]]:
    rows = pd.DataFrame({
        "symbol": frame["symbol"].astype(str).str.strip().str.upper(),
        "ret": returns,
        "activity": activity,
    }).dropna(subset=["ret"])
    rows = rows[rows["symbol"] != ""]
    out: dict[str, dict[str, Any]] = {}
    if rows.empty:
        return out
    for symbol, group in rows.groupby("symbol"):
        records = int(len(group))
        hits = float((group["ret"] > 0).sum()) / records
        out[symbol] = {
            "records": records,
            "hit_rate": round(hits, 4),
            "mean_return": round(float(group["ret"].mean()), 6),
            "activity": int(group["activity"].sum()),
        }
    return out


def delivery_stats(ledger: pd.DataFrame, lookback_sessions: int = 20,
                   min_records: int = 3, as_of: str | None = None) -> dict[str, dict[str, Any]]:
    """Realised delivery profitability over the trailing session window."""
    if ledger is None or ledger.empty or "symbol" not in ledger.columns:
        return {}
    frame = ledger.copy()
    frame["_ret"] = _signed_return(frame)
    dates = pd.to_datetime(frame.get("prediction_date"), errors="coerce")
    frame = frame[dates.notna()]
    if frame.empty:
        return {}
    dates = pd.to_datetime(frame["prediction_date"], errors="coerce")
    cutoff = pd.Timestamp(as_of) if as_of else dates.max()
    window_start = cutoff - pd.Timedelta(days=max(1, int(lookback_sessions)) * 2)
    frame = frame[dates >= window_start]
    if frame.empty:
        return {}
    sessions = dates[frame.index].dt.date
    latest = sorted(set(sessions))[-int(lookback_sessions):]
    frame = frame[sessions.isin(latest)]
    stats = _aggregate(frame, frame["_ret"], pd.Series(1, index=frame.index))
    for row in stats.values():
        row["profitable"] = bool(
            row["records"] >= int(min_records)
            and (row["mean_return"] > 0 or row["hit_rate"] >= 0.5)
        )
    return stats


def intraday_stats(frames: Iterable[pd.DataFrame], min_records: int = 3) -> dict[str, dict[str, Any]]:
    """Realised intraday profitability across the supplied outcome files."""
    usable = [f for f in frames if f is not None and not f.empty and "symbol" in f.columns]
    if not usable:
        return {}
    frame = pd.concat(usable, ignore_index=True, sort=False)
    if "validation_status" in frame.columns:
        frame = frame[frame["validation_status"].astype(str).str.upper().isin({"MATURED", "OK", "VALIDATED"}) | frame["validation_status"].isna()]
    returns = _signed_return(frame)
    day = pd.to_datetime(frame.get("ledger_date"), errors="coerce").dt.date
    if day.isna().all():
        day = pd.Series(date.today(), index=frame.index)
    stats = _aggregate(frame, returns, pd.Series(1, index=frame.index))
    for row in stats.values():
        row["profitable"] = bool(row["records"] >= int(min_records) and row["mean_return"] > 0)
    return stats


def fno_universe(frame: pd.DataFrame, max_sessions: int = 2) -> set[str]:
    """Symbols present in the newest F&O sessions."""
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return set()
    frame = frame.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.notna().any():
            keep = sorted(set(dates.dropna()))[-int(max_sessions):]
            frame = frame[dates.isin(keep)]
    return set(frame["symbol"]) - {""}


def latest_closes(ledger: pd.DataFrame) -> dict[str, float]:
    """Most recent reference close per symbol from the delivery ledger."""
    if ledger is None or ledger.empty or "symbol" not in ledger.columns or "reference_close" not in ledger.columns:
        return {}
    frame = ledger.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame["_close"] = pd.to_numeric(frame["reference_close"], errors="coerce")
    frame["_date"] = pd.to_datetime(frame.get("prediction_date"), errors="coerce")
    frame = frame.dropna(subset=["_close"]).sort_values("_date")
    return {str(s): float(v) for s, v in frame.groupby("symbol")["_close"].last().items()}


# --------------------------------------------------------------------------- #
# Universe selection (pure)
# --------------------------------------------------------------------------- #
def select_universe(closes: dict[str, float], delivery: dict[str, dict[str, Any]],
                    intraday: dict[str, dict[str, Any]], fno: set[str],
                    plan: dict[str, Any]) -> dict[str, Any]:
    """Pick the ranked symbol set plus a per-symbol reason trail."""
    budget = position_budget(plan)
    price_cap = float(budget["price_cap"])
    affordable = {s for s, c in closes.items() if c and c <= price_cap}
    act_del = int(plan["activity_min_delivery_sessions"])
    act_int = int(plan["activity_min_intraday_days"])

    def row(symbol: str) -> dict[str, Any]:
        d = delivery.get(symbol, {})
        i = intraday.get(symbol, {})
        return {
            "symbol": symbol,
            "close": round(float(closes.get(symbol, 0.0)), 2),
            "delivery_profitable": bool(d.get("profitable")),
            "delivery_records": int(d.get("records", 0)),
            "delivery_mean_return": float(d.get("mean_return", 0.0)),
            "intraday_profitable": bool(i.get("profitable")),
            "intraday_records": int(i.get("records", 0)),
            "intraday_mean_return": float(i.get("mean_return", 0.0)),
            "fno": symbol in fno,
            "activity": int(max(d.get("activity", 0), i.get("activity", 0))),
            "liquid": bool(
                symbol in fno
                or int(d.get("activity", 0)) >= act_del
                or int(i.get("activity", 0)) >= act_int
            ),
        }

    candidates = {s: row(s) for s in affordable}
    eligible = [
        r for r in candidates.values()
        if r["delivery_profitable"] or r["intraday_profitable"] or r["liquid"]
    ]

    max_activity = max((r["activity"] for r in candidates.values()), default=1) or 1

    def score(r: dict[str, Any]) -> float:
        return (
            4.0 * r["delivery_profitable"]
            + 3.0 * r["intraday_profitable"]
            + 2.0 * r["fno"]
            + (r["activity"] / max_activity)
        )

    eligible.sort(key=lambda r: (-score(r), -r["activity"], r["symbol"]))
    limit = int(plan["universe_size"])
    chosen = eligible[:limit]

    if len(chosen) < limit:
        taken = {r["symbol"] for r in chosen}
        fallback = sorted(
            (r for s, r in candidates.items() if s not in taken),
            key=lambda r: (-r["activity"], r["symbol"]),
        )[: int(plan["top_active_fallback_count"])]
        chosen.extend(fallback[: max(0, limit - len(chosen))])

    if MARKET_CONTEXT_SYMBOL in closes and all(r["symbol"] != MARKET_CONTEXT_SYMBOL for r in chosen):
        chosen.insert(0, row(MARKET_CONTEXT_SYMBOL))

    chosen.sort(key=lambda r: (-score(r), r["symbol"]))
    symbols = sorted(r["symbol"] for r in chosen)
    version = hashlib.sha1("|".join(symbols).encode("utf-8")).hexdigest()[:12]
    return {
        "schema_version": SCHEMA_VERSION,
        "universe_version": version,
        "symbols": symbols,
        "details": chosen,
        "stats": {
            "affordable": len(affordable),
            "eligible": len(eligible),
            "selected": len(symbols),
            "delivery_profitable": sum(1 for r in chosen if r["delivery_profitable"]),
            "intraday_profitable": sum(1 for r in chosen if r["intraday_profitable"]),
            "fno": sum(1 for r in chosen if r["fno"]),
        },
        "budget": budget,
    }


# --------------------------------------------------------------------------- #
# Drive IO
# --------------------------------------------------------------------------- #
def _download_bytes(store, remote: str) -> bytes | None:
    from drive_all_nse_prediction_compiler import _find_remote
    from persistent_store import download_drive_file_with_retry

    item = _find_remote(store, remote)
    if not item:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / (Path(remote).name or "download.bin")
        download_drive_file_with_retry(store.service, item["id"], target, f"download:{remote}")
        return target.read_bytes()


def _latest_file(store, folder: str, suffix: str) -> str | None:
    from drive_all_nse_prediction_compiler import _find_remote, _list_children

    item = _find_remote(store, folder)
    if not item:
        return None
    names = [c.get("name", "") for c in _list_children(store, item["id"]) if c.get("name", "").endswith(suffix)]
    return f"{folder}/{max(names)}" if names else None


def _read_csv_bytes(payload: bytes | None) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    import io
    try:
        return pd.read_csv(io.BytesIO(payload))
    except Exception:
        return pd.DataFrame()


def _outcomes_frames(store, now: datetime, days: int) -> tuple[list[pd.DataFrame], list[str]]:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []
    for offset in range(max(1, days) + 4):
        day = (now.date() - timedelta(days=offset))
        folder = f"{OUTCOMES_ROOT}/{day:%Y/%m/%d}"
        payload = None
        try:
            latest = _latest_file(store, folder, ".csv")
            if latest:
                payload = _download_bytes(store, latest)
                sources.append(latest)
        except Exception:
            payload = None
        if payload:
            frame = _read_csv_bytes(payload)
            if not frame.empty:
                frames.append(frame)
    return frames, sources


def build_manifest(store=None, plan: dict[str, Any] | None = None,
                   realised_pnl: float = 0.0, now: datetime | None = None,
                   sync: bool = True) -> dict[str, Any]:
    """Assemble and (optionally) upload the universe manifest."""
    plan = plan or load_capital_plan()
    now = now or datetime.now(IST)

    # --- FIX: apply the authoritative 500-symbol roster first ---
    # It must take precedence over the data-driven selection so the manifest
    # never falls below the required universe size, even when downstream
    # data sources (delivery ledger, FNO features, intraday frames) are
    # incomplete or unavailable.
    pinned = sorted(set(authoritative_roster()))

    now = now or datetime.now(IST)
    local_store = False
    if store is None and sync:
        from persistent_store import GoogleDriveStore
        store = GoogleDriveStore()
        local_store = True

    status: dict[str, str] = {}
    try:
        ledger = _read_csv_bytes(_download_bytes(store, DELIVERY_LEDGER)) if store else pd.DataFrame()
        status["delivery_ledger"] = "OK" if not ledger.empty else "EMPTY"
    except Exception as exc:
        ledger = pd.DataFrame()
        status["delivery_ledger"] = f"ERROR:{type(exc).__name__}"

    try:
        frames, sources = _outcomes_frames(store, now, int(plan["intraday_lookback_days"])) if store else ([], [])
        status["intraday_outcomes"] = f"OK:{len(sources)}" if frames else "EMPTY"
    except Exception as exc:
        frames = []
        status["intraday_outcomes"] = f"ERROR:{type(exc).__name__}"

    fno: set[str] = set()
    try:
        payload = _download_bytes(store, FNO_FEATURES) if store else None
        if payload:
            import io
            fno = fno_universe(pd.read_parquet(io.BytesIO(payload)))
        status["fno"] = f"OK:{len(fno)}"
    except Exception as exc:
        status["fno"] = f"ERROR:{type(exc).__name__}"

    closes = latest_closes(ledger)
    status["closes"] = f"OK:{len(closes)}"

    # Build the data-driven selection, then override with the pinned roster.
    delivery = delivery_stats(ledger, int(plan["delivery_lookback_sessions"]), int(plan["min_records_profitable"]))
    intraday = intraday_stats(frames, int(plan["min_records_profitable"]))
    selection = select_universe(closes, delivery, intraday, fno, plan)
    # Always favour the pinned roster; if the data-driven selection is too small
    # it still gets replaced by the full 500-symbol roster.
    if pinned:
        selection = dict(selection)
        selection["symbols"] = pinned
        selection["universe_version"] = hashlib.sha1(
            "|".join(pinned).encode("utf-8")
        ).hexdigest()[:12]
    elif len(selection.get("symbols", [])) < 150:
        return {
            "status": "ABORTED",
            "reason": f"universe too small ({len(selection.get('symbols', []))})",
            "generated_at": now.isoformat(),
            "sources": status,
            "stats": selection.get("stats") if isinstance(selection, dict) else {},
        }

    pinned = sorted(set(authoritative_roster()))
    if pinned:
        selection = dict(selection)
        selection["symbols"] = pinned
        selection["universe_version"] = hashlib.sha1("|".join(pinned).encode("utf-8")).hexdigest()[:12]

    manifest = {
        "status": "OK",
        "generated_at": now.isoformat(),
        "sources": status,
        "plan": {k: plan[k] for k in ("monthly_capital", "max_positions", "hard_price_cap", "universe_size")},
        **selection,
    }
    local_copy = REPO_ROOT / "production_data" / "prediction_universe" / "current.json"
    try:
        local_copy.parent.mkdir(parents=True, exist_ok=True)
        local_copy.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except OSError:
        pass
    if sync and store is not None:
        stamp = now.strftime("%Y%m%dT%H%M%S")
        for remote in (UNIVERSE_CURRENT, f"{UNIVERSE_HISTORY}/universe_{stamp}.json"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "manifest.json"
                path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                store.upload_or_update(path, remote, "application/json")
    return manifest


def active_symbols(store=None, plan: dict[str, Any] | None = None,
                   now: datetime | None = None) -> tuple[list[str], dict[str, Any]]:
    """Return the fixed owner-selected roster; Drive manifests never reselect it."""
    symbols = authoritative_roster()
    if len(symbols) != 500:
        return [], {"status": "BLOCKED", "reason": f"pinned roster must contain exactly 500 symbols; found {len(symbols)}"}
    version = hashlib.sha1("|".join(symbols).encode("utf-8")).hexdigest()[:12]
    return symbols, {
        "status": "PINNED_500_ROSTER",
        "universe_version": version,
        "selected_symbol_count": len(symbols),
    }


_SHARED: dict[str, Any] = {}


def active_symbols_cached(plan: dict[str, Any] | None = None, ttl_seconds: float = 900.0,
                          now: datetime | None = None) -> tuple[list[str] | None, dict[str, Any]]:
    """Process-cached ``active_symbols`` for the per-tick capture path."""
    now = now or datetime.now(IST)
    fetched = _SHARED.get("fetched_at")
    if fetched is not None and (now - fetched).total_seconds() < ttl_seconds:
        return _SHARED.get("symbols"), _SHARED.get("meta") or {}
    symbols, meta = active_symbols()
    _SHARED.update({"symbols": symbols, "meta": meta, "fetched_at": now})
    return symbols, meta


def restrict_universe(symbols: Iterable[str], allowed: Iterable[str] | None) -> list[str]:
    """Keep only allowed symbols; an empty allow-list denies all symbols."""
    if allowed is None:
        return list(symbols)
    keep = {str(s).strip().upper().replace(".NS", "") for s in allowed}
    return [s for s in symbols if str(s).strip().upper().replace(".NS", "") in keep]


# --------------------------------------------------------------------------- #
# Tradable picks (pure)
# --------------------------------------------------------------------------- #
PRICE_COLUMNS = ("last_price", "ltp", "close", "Close", "price", "reference_close")


def _price_column(frame: pd.DataFrame) -> str | None:
    for name in PRICE_COLUMNS:
        if name in frame.columns:
            return name
    return None


def tradable_picks(predictions: pd.DataFrame, plan: dict[str, Any] | None = None,
                   realised_pnl: float = 0.0, prices: dict[str, float] | None = None) -> dict[str, Any]:
    """Long-only, affordable, ranked picks from one prediction frame."""
    plan = plan or load_capital_plan()
    budget = position_budget(plan, realised_pnl)
    empty = {"picks": [], "budget": budget, "skipped": 0, "reason": ""}
    if predictions is None or predictions.empty or "symbol" not in predictions.columns:
        return {**empty, "reason": "no predictions"}

    frame = predictions.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if "prediction_status" in frame.columns:
        good = frame["prediction_status"].astype(str).str.upper().str.startswith("PREDICTED")
        frame = frame[good]
    if "signal" in frame.columns:
        frame = frame[frame["signal"].astype(str).str.upper() == "BUY"]
    # Session-close predictions are the live contract. Legacy 30-minute
    # snapshots remain readable so archived days can still be inspected.
    probability_column = (
        "probability_up_session_close"
        if "probability_up_session_close" in frame.columns
        else "probability_up_30m"
    )
    expected_return_column = (
        "expected_return_session_close"
        if "expected_return_session_close" in frame.columns
        else "expected_return_30m"
    )
    for column in (probability_column, expected_return_column):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if expected_return_column not in frame.columns:
        return {**empty, "reason": f"no {expected_return_column}"}
    min_move = float(plan["min_move_intraday"])
    frame = frame[frame[expected_return_column] >= min_move]
    if probability_column in frame.columns:
        frame = frame[frame[probability_column].fillna(0) >= 0.6]
    if frame.empty:
        return {**empty, "reason": "no qualifying signals"}

    price_col = _price_column(frame)
    candidates = []
    skipped = 0
    for _, row in frame.iterrows():
        price = (prices or {}).get(row["symbol"])
        if price is None and price_col:
            price = row.get(price_col)
        try:
            price = float(price)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if price <= 0 or price > float(budget["price_cap"]):
            skipped += 1
            continue
        quantity = int(float(budget["per_trade"]) // price)
        if quantity < 1:
            skipped += 1
            continue
        prob = float(row.get(probability_column, 0.0) or 0.0)
        ret = float(row[expected_return_column])
        candidates.append({
            "symbol": row["symbol"],
            "price": round(price, 2),
            "quantity": quantity,
            "cost": round(price * quantity, 2),
            "expected_return": round(ret, 4),
            "probability_up": round(prob, 4),
            "score": round(ret * max(prob, 0.5), 6),
            "sleeve": "intraday",
        })
    candidates.sort(key=lambda p: (-p["score"], p["symbol"]))
    picks = candidates[: int(budget["positions"])]
    return {
        "picks": picks,
        "budget": budget,
        "skipped": skipped,
        "reason": "" if picks else "no affordable candidates",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the prediction universe manifest")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN_PATH))
    parser.add_argument("--realised-pnl", type=float, default=0.0)
    parser.add_argument("--no-sync", action="store_true", help="build only, do not upload")
    args = parser.parse_args(argv)
    manifest = build_manifest(plan=load_capital_plan(args.plan), realised_pnl=args.realised_pnl, sync=not args.no_sync)
    print(json.dumps({k: v for k, v in manifest.items() if k != "details"}, indent=2))
    return 0 if manifest.get("status") == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
