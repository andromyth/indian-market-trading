        parts.append(z)
    return pd.concat(parts, ignore_index=True).sort_values(["symbol", "decision_timestamp"]).reset_index(drop=True)

def _merge_broad_fno_pit(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Merge the broad NSE F&O UDiFF layer under an explicit next-trading-day PIT clock.

    The source date is the F&O trade date. Its information becomes eligible at
    the next trading session's 09:15, so a delivery decision at T 15:30 can only
    consume F&O information from the prior trading session. This avoids same-day
    EOD F&O leakage into the delivery model.
    """
    path = BASE / "production_data/nse_official/fno/fno_daily_features_2026.parquet"
    if not path.exists() or path.stat().st_size == 0:
        return panel, {"exists": False, "rows": 0, "symbols": 0, "pit_rule": "next_trading_day_09:15"}

    f = pd.read_parquet(path)
    if f.empty or not {"symbol", "date"}.issubset(f.columns):
        return panel, {"exists": False, "rows": 0, "symbols": 0, "pit_rule": "next_trading_day_09:15"}

    f["symbol"] = f["symbol"].map(normalize_symbol)
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
    f = f.dropna(subset=["symbol", "date"]).copy()

    # Recover missing futures basis from the PIT-safe equity close on the
    # same F&O trade date. The resulting F&O row is only exposed to the model
    # at the next trading session's 09:15, so this does not introduce
    # same-session EOD leakage.
    if "fno_fut_basis" in f.columns and "close" in panel.columns:
        equity_ref = panel[["symbol", "date", "close"]].copy()
        equity_ref["symbol"] = equity_ref["symbol"].map(normalize_symbol)
        equity_ref["date"] = pd.to_datetime(equity_ref["date"], errors="coerce").dt.normalize()
        equity_ref["close"] = pd.to_numeric(equity_ref["close"], errors="coerce")
        equity_ref = equity_ref.dropna(subset=["symbol", "date", "close"]).drop_duplicates(
            ["symbol", "date"], keep="last"
        )
        f = f.merge(
            equity_ref.rename(columns={"close": "_equity_close_trade_date"}),
            on=["symbol", "date"],
            how="left",
        )
        basis = pd.to_numeric(f["fno_fut_basis"], errors="coerce")
        fut_close = pd.to_numeric(f.get("fno_fut_close"), errors="coerce")
        spot_close = pd.to_numeric(f["_equity_close_trade_date"], errors="coerce")
        fallback_basis = fut_close.div(spot_close).sub(1.0)
        f["fno_fut_basis"] = basis.where(
            basis.notna(),
            fallback_basis.where(np.isfinite(fallback_basis), np.nan),
        )
        f = f.drop(columns=["_equity_close_trade_date"], errors="ignore")

    value_map = {
        "fno_pcr": "fno_pcr",
        "fno_oi_change": "fno_oi_change",
        "fno_iv_change": "fno_iv_change",
        "fno_available": "fno_available",
        "fno_fut_close": "fno_fut_close",
        "fno_fut_basis": "fno_basis",
        "fno_fut_return": "fno_fut_return",
        "fno_fut_oi": "fno_fut_oi",
    }
    present = {src: dst for src, dst in value_map.items() if src in f.columns}
    if not present:
        return panel, {"exists": False, "rows": 0, "symbols": 0, "pit_rule": "next_trading_day_09:15"}

    f = f.rename(columns=present)
    keep = ["symbol", "date"] + list(present.values())
    f = f[keep].drop_duplicates(["symbol", "date"], keep="last").copy()

    # Use the candidate panel's actual trading sessions to determine the next
    # session. This handles weekends and exchange holidays without assumptions.
    panel_dates = (
        pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
        if "date" in panel.columns
        else pd.to_datetime(panel["decision_timestamp"], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    )
    sessions = pd.Series(sorted(panel_dates.dropna().unique()))
    if sessions.empty:
        return panel, {"exists": True, "rows": int(len(f)), "symbols": int(f["symbol"].nunique()), "pit_rule": "next_trading_day_09:15"}

    next_session = pd.Series(sessions.to_numpy(), index=sessions.to_numpy()).shift(-1)
    session_map = next_session.to_dict()
    f["next_session_date"] = f["date"].map(session_map)
    f = f.dropna(subset=["next_session_date"]).copy()
    f["decision_timestamp"] = (
        pd.to_datetime(f["next_session_date"]).dt.tz_localize("Asia/Kolkata")
        .dt.tz_convert("UTC")
        + pd.Timedelta(hours=0)
    )
    # Panel timestamps are represented as UTC in the PIT merger.  The source's
    # next-session availability clock is 09:15 IST = 03:45 UTC.
    f["decision_timestamp"] = (
        pd.to_datetime(f["next_session_date"]).dt.tz_localize("Asia/Kolkata")
        + pd.Timedelta(hours=9, minutes=15)
    ).dt.tz_convert("UTC")

    f = f.drop(columns=["date", "next_session_date"])
    f = f.sort_values(["symbol", "decision_timestamp"])
    f = f.drop_duplicates(["symbol", "decision_timestamp"], keep="last")

    # The candidate panel may already contain the legacy three-stock F&O pilot
    # columns. Remove those target columns before merging so the authoritative
    # broad NSE layer lands on the canonical 59-feature names rather than being
    # silently suffixed (for example, fno_pcr_broad_fno).
    target_columns = [x for x in present.values() if x in panel.columns]
    panel_for_merge = panel.drop(columns=target_columns, errors="ignore")

    merged = merge_pit_layer(
        panel_for_merge,
        f,
        [x for x in present.values()],
        "broad_fno",
    )

    # Prefer the authoritative NSE UDiFF basis already calculated in the source.
    if "fno_fut_basis" in merged.columns:
        merged["fno_fut_basis"] = pd.to_numeric(merged["fno_fut_basis"], errors="coerce")

    return merged, {
        "exists": True,
        "path": str(path),
        "rows": int(len(f)),
        "symbols": int(f["symbol"].nunique()),
        "source": "NSE_OFFICIAL_FNO_UDIFF",
        "pit_rule": "trade_date_fno_available_next_trading_day_09:15_IST",
    }


def merge_optional(panel):
    out=panel.copy(); audit={}

    # Broad NSE F&O takes precedence over the legacy three-stock PIT pilot.
    # The pilot is retained on Drive for historical reproducibility but is not
    # merged into the ALL-NSE candidate because doing so would create a mixed
    # provenance layer.
    out, fno_audit = _merge_broad_fno_pit(out)