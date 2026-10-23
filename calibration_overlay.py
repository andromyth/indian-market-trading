#!/usr/bin/env python3
"""Fail-open worker overlay for validated calibration profiles."""
from __future__ import annotations

import copy
import time
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

import calibration_autopilot as ca

RETRY_AFTER_FAILURE_SECONDS = 60


def resolve_mode(value: Any) -> str:
    mode = str(value or "shadow").strip().lower()
    return mode if mode in {"off", "shadow", "active"} else "shadow"


class CalibrationOverlay:
    def __init__(
        self,
        loader: Callable[[], dict[str, Any] | None],
        *,
        mode: str = "shadow",
        refresh_seconds: float = 900,
        clock: Callable[[], float] = time.monotonic,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.loader = loader
        self.mode = resolve_mode(mode)
        self.refresh_seconds = max(1.0, float(refresh_seconds))
        self.clock = clock
        self.now_fn = now_fn
        self._last_refresh = float("-inf")
        self._retry_after = self.refresh_seconds
        self._document: dict[str, Any] | None = None
        self.last_error: str | None = None

    def _refresh_if_due(self) -> None:
        now = self.clock()
        if now - self._last_refresh < self._retry_after:
            return
        self._last_refresh = now
        try:
            doc = self.loader()
            if not isinstance(doc, dict) or doc.get("schema_version") != ca.SCHEMA_VERSION or not isinstance(doc.get("profiles"), list):
                raise ValueError("invalid calibration profile document")
            self._document = doc
            self.last_error = None
            self._retry_after = self.refresh_seconds
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._retry_after = RETRY_AFTER_FAILURE_SECONDS
            # Keep the last valid document if a transient refresh fails.

    def annotate(self, row: Any) -> Any:
        if self.mode == "off" or not isinstance(row, dict):
            return row
        try:
            self._refresh_if_due()
            doc = self._document
            if not doc:
                return row
            horizon = row.get("horizon_min")
            version = row.get("model_version")
            key = ca.profile_key(horizon, version)
            profile = next((p for p in doc["profiles"] if isinstance(p, dict) and p.get("key") == key), None)
            if not profile or profile.get("stage") not in ca.LIVE_STAGES:
                return row
            valid_until = ca._parse_iso(profile.get("valid_until_utc"))
            now = self.now_fn()
            if valid_until is None or now >= valid_until:
                return row
            raw_p = pd.to_numeric(pd.Series([row.get("probability_up")]), errors="coerce").iloc[0]
            if not np.isfinite(raw_p) or not 0 <= raw_p <= 1:
                return row
            p_cal = float(ca.apply_calibrator(profile.get("calibrator"), [raw_p])[0])
            out = copy.deepcopy(row)
            out["probability_up_calibrated"] = p_cal
            out["calibration_profile_id"] = profile.get("profile_id")
            out["calibration_stage"] = profile.get("stage")
            out["calibration_applied"] = False
            out["signal_calibrated"] = ca.signal_from_thresholds(p_cal, profile.get("thresholds"))
            raw_er = pd.to_numeric(pd.Series([row.get("expected_return")]), errors="coerce").iloc[0]
            er_cal = float(raw_er) if np.isfinite(raw_er) else None
            correction = profile.get("return_correction")
            if er_cal is not None and correction and profile.get("price_forecast_reliable"):
                er_cal = float(ca.apply_return_correction(correction, [er_cal])[0])
            out["expected_return_calibrated"] = er_cal
            if er_cal is not None:
                ref_close = pd.to_numeric(pd.Series([row.get("reference_close")]), errors="coerce").iloc[0]
                if np.isfinite(ref_close) and ref_close > 0:
                    out["predicted_price_calibrated"] = float(ref_close * (1 + er_cal))
                else:
                    out["predicted_price_calibrated"] = row.get("predicted_price")
            else:
                out["predicted_price_calibrated"] = row.get("predicted_price")
            if self.mode == "active" and profile.get("stage") == ca.STAGE_ACTIVE:
                signal = out["signal_calibrated"]
                # Abstention does not masquerade as a tradable signal; the production
                # display keeps HOLD while the calibrated advisory field carries ABSTAIN.
                if signal == "ABSTAIN":
                    signal = "HOLD"
                out["probability_up_raw"] = row.get("probability_up")
                out["expected_return_raw"] = row.get("expected_return")
                out["predicted_price_raw"] = row.get("predicted_price")
                out["signal_raw"] = row.get("signal")
                out["probability_up"] = p_cal
                if er_cal is not None:
                    out["expected_return"] = er_cal
                out["predicted_price"] = out["predicted_price_calibrated"]
                thresholds = profile.get("thresholds") or {"abstain_all": True}
                # An active profile that cannot safely emit BUY/SELL must force
                # the production-facing signal to HOLD rather than preserve the
                # raw pre-calibration action by accident.
                out["signal"] = signal if not thresholds.get("abstain_all", True) else "HOLD"
                out["calibration_applied"] = True
            else:
                # Record raw columns for scoring lineage even in shadow mode, but
                # never alter primary prediction values.
                out.setdefault("probability_up_raw", row.get("probability_up"))
                out.setdefault("expected_return_raw", row.get("expected_return"))
                out.setdefault("predicted_price_raw", row.get("predicted_price"))
                out.setdefault("signal_raw", row.get("signal"))
            return out
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return row
