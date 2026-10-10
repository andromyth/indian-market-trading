"""Safe INDstocks connection smoke test. Never prints credentials or profile PII."""
from __future__ import annotations

import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from indstocks_client import configured, latest_quote, historical_1m, profile

IST = ZoneInfo("Asia/Kolkata")


def main() -> int:
    symbol = sys.argv[1].strip().upper() if len(sys.argv) > 1 else "RELIANCE.NS"
    if not configured():
        print("CONFIGURATION: FAIL — provide INDSTOCKS_ACCESS_TOKEN or complete TOTP settings in environment variables.")
        return 2

    try:
        profile()
        print("AUTHENTICATION: PASS")
        quote = latest_quote(symbol)
        print(f"QUOTE: PASS — {symbol} live_price={quote.get('live_price')} volume={quote.get('volume')}")
        candles = historical_1m(symbol, days=3)
        if candles.empty:
            print("HISTORICAL_1M: FAIL — no candles returned for the selected recent window")
            return 1
        print(
            "HISTORICAL_1M: PASS — "
            f"rows={len(candles)} first={candles.index.min()} last={candles.index.max()}"
        )
        print(f"CHECKED_AT_IST: {datetime.now(IST).isoformat(timespec='seconds')}")
        return 0
    except Exception as exc:
        # Adapter exceptions deliberately exclude Authorization headers and secrets.
        print(f"INDSTOCKS CHECK: FAIL — {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
