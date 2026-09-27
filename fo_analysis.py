"""Options-chain-based F&O read: Put-Call Ratio, Max Pain, and OI buildup at
the at-the-money strike.

These are widely watched heuristics in Indian F&O commentary, not validated
predictive signals, so this module reports the numbers rather than folding
them into one buy/sell figure the way the equity technical score does.

PCR specifically is read two different ways depending on who you ask: a
high PCR (more open put interest than call interest) gets read as bearish
by some traders (heavy hedging or bets on a fall) and as a contrarian
bullish signal by others (heavy put-buying often precedes a bounce). Both
readings are common in practice; neither is reliably right, so this module
deliberately doesn't pick one.
"""
import pandas as pd


def chain_to_dataframe(raw_chain: dict, expiry: str | None = None) -> pd.DataFrame:
    """Flatten NSE's option-chain JSON into a per-strike DataFrame."""
    records = (raw_chain or {}).get("records", {})
    data = records.get("data", [])
    rows = []
    for row in data:
        if expiry and row.get("expiryDate") != expiry:
            continue
        ce = row.get("CE", {}) or {}
        pe = row.get("PE", {}) or {}
        rows.append(
            {
                "strike": row.get("strikePrice"),
                "expiry": row.get("expiryDate"),
                "CE_OI": ce.get("openInterest") or 0,
                "CE_chgOI": ce.get("changeinOpenInterest") or 0,
                "CE_chg": ce.get("change", ce.get("pChange")),
                "CE_IV": ce.get("impliedVolatility"),
                "CE_LTP": ce.get("lastPrice"),
                "PE_OI": pe.get("openInterest") or 0,
                "PE_chgOI": pe.get("changeinOpenInterest") or 0,
                "PE_chg": pe.get("change", pe.get("pChange")),
                "PE_IV": pe.get("impliedVolatility"),
                "PE_LTP": pe.get("lastPrice"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("strike").reset_index(drop=True)


def nearest_expiry(raw_chain: dict):
    if not raw_chain:
        return None
    dates = raw_chain.get("records", {}).get("expiryDates") or []
    return dates[0] if dates else None


def underlying_value(raw_chain: dict):
    if not raw_chain:
        return None
    return raw_chain.get("records", {}).get("underlyingValue")


def put_call_ratio(df: pd.DataFrame):
    if df.empty:
        return None
    total_ce = df["CE_OI"].sum()
    total_pe = df["PE_OI"].sum()
    if not total_ce:
        return None
    return round(total_pe / total_ce, 2)


def max_pain(df: pd.DataFrame):
    """Strike where option WRITERS collectively lose the least at expiry.
    A commonly cited 'price magnet' theory, debated among practitioners,
    not a guarantee of where price will land.
    """
    if df.empty:
        return None
    strikes = df["strike"].tolist()
    best_strike, min_loss = None, None
    for candidate in strikes:
        total_loss = 0.0
        for _, r in df.iterrows():
            k = r["strike"]
            if candidate > k:
                total_loss += (candidate - k) * r["CE_OI"]
            if candidate < k:
                total_loss += (k - candidate) * r["PE_OI"]
        if min_loss is None or total_loss < min_loss:
            min_loss, best_strike = total_loss, candidate
    return best_strike


def classify_buildup(chg_oi, price_change):
    """Classic four-way OI + price read. Returns None if either input is
    missing, NSE doesn't always populate the per-strike price-change field,
    and that should show up as 'data unavailable', not a guessed answer.
    """
    if chg_oi is None or price_change is None:
        return None
    if chg_oi > 0 and price_change > 0:
        return "Long buildup (bullish)"
    if chg_oi > 0 and price_change < 0:
        return "Short buildup (bearish)"
    if chg_oi < 0 and price_change > 0:
        return "Short covering (bullish, short-term)"
    if chg_oi < 0 and price_change < 0:
        return "Long unwinding (bearish, short-term)"
    return "Flat"


def summarize(raw_chain: dict, expiry: str | None = None, atm_window: int = 5) -> dict:
    """One-shot F&O snapshot: PCR, max pain, ATM OI reads, and the near-ATM
    chain table. Returns {} if there's nothing usable (no derivatives on
    this symbol, or NSE didn't return data)."""
    if not raw_chain:
        return {}
    expiry = expiry or nearest_expiry(raw_chain)
    df = chain_to_dataframe(raw_chain, expiry)
    if df.empty:
        return {}

    spot = underlying_value(raw_chain)
    result = {
        "expiry": expiry,
        "spot": spot,
        "pcr": put_call_ratio(df),
        "max_pain": max_pain(df),
        "call_read": None,
        "put_read": None,
        "near_atm_chain": df,
    }

    if spot is not None:
        df = df.copy()
        df["dist"] = (df["strike"] - spot).abs()
        atm_idx = df["dist"].idxmin()
        atm_row = df.loc[atm_idx]
        result["call_read"] = classify_buildup(atm_row["CE_chgOI"], atm_row["CE_chg"])
        result["put_read"] = classify_buildup(atm_row["PE_chgOI"], atm_row["PE_chg"])
        lo, hi = max(0, atm_idx - atm_window), min(len(df), atm_idx + atm_window + 1)
        result["near_atm_chain"] = df.iloc[lo:hi].drop(columns="dist")

    return result
