from __future__ import annotations

import pandas as pd

from selfdriving.sd_global_cues import Series, attach_global_features


def test_attach_global_features_normalizes_microsecond_availability_key():
    available = pd.to_datetime(
        [
            "2026-10-07 20:15:00+00:00",
            "2026-10-08 20:15:00+00:00",
            "2026-10-09 20:15:00+00:00",
        ],
        utc=True,
    ).astype("datetime64[us, UTC]")
    long = pd.DataFrame(
        {
            "key": ["spx", "spx", "spx"],
            "date": pd.to_datetime(["2026-10-07", "2026-10-08", "2026-10-09"]),
            "close": [6700.0, 6720.0, 6690.0],
            "available_at": available,
        }
    )
    panel = pd.DataFrame(
        {
            "symbol": ["RELIANCE", "SBIN"],
            "decision_timestamp": pd.to_datetime(
                ["2026-10-08 21:00:00+00:00", "2026-10-09 21:00:00+00:00"],
                utc=True,
            ).astype("datetime64[ns, UTC]"),
        }
    )

    result = attach_global_features(
        panel,
        long,
        max_age_days=5,
        series={"spx": Series("^GSPC", "America/New_York", close=__import__("datetime").time(16, 0))},
    )

    assert len(result) == len(panel)
    assert "g_spx_age_h" in result.columns
    assert result["decision_timestamp"].dtype == "datetime64[ns, UTC]"
    assert result["g_spx_age_h"].notna().all()
