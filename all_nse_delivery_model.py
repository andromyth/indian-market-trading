"""Live ALL-NSE Delivery model adapter.

This module promotes the already-built ALL-NSE 59-feature Delivery candidate to
the live Delivery path without retraining it.

Design:
- Candidate artifacts remain in Google Drive, not GitHub.
- The branch replacement manifest hashes are verified before use.
- The current official NSE security-wise historical interface supplies the
  point-in-time OHLCV + delivery inputs for the selected symbol.
- Official NSE NIFTY 50 / NIFTY BANK history supplies market context.
- Optional information remains explicit and point-in-time; current/future values are not silently substituted.
- Any current NSE equity with the required clean history can be scored; the build-time trainable list remains provenance.
- No order-placement code exists here.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import date, timedelta, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nse_data_layer import (
    fetch_equity_history,
    fetch_index_history,
    fetch_nse_equity_universe,
)
from delivery_training_universe import benchmark_for_symbol


DRIVE_MODEL_ROOT = (
    "models/all_nse_delivery_multifactor_replacement_2026"
)

REMOTE_MANIFEST = (
    f"{DRIVE_MODEL_ROOT}/ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json"
)

REMOTE_CLASSIFIER = (
    f"{DRIVE_MODEL_ROOT}/deployment_all_nse_next_session_multifactor59_classifier_2026.joblib"
)

REMOTE_REGRESSOR = (
    f"{DRIVE_MODEL_ROOT}/deployment_all_nse_next_session_multifactor59_regressor_2026.joblib"
)

CACHE_ROOT = (
    Path(__file__).resolve().parent
    / "data"
    / "all_nse_delivery_replacement"
)

LOCAL_MANIFEST = CACHE_ROOT / "ALL_NSE_MULTIFACTOR_59_MODEL_MANIFEST_2026.json"
LOCAL_CLASSIFIER = CACHE_ROOT / "deployment_all_nse_next_session_multifactor59_classifier_2026.joblib"
LOCAL_REGRESSOR = CACHE_ROOT / "deployment_all_nse_next_session_multifactor59_regressor_2026.joblib"

EXPECTED_ENGINE_VERSIONS = {
    "2026-all-nse-delivery-multifactor-59-replacement-v1",
    # Compatibility with the replacement artifact set already built before
    # the branch version string was normalized. No model data is changed.
    "2026-all-nse-delivery-multifactor-59-candidate-v2",
}

BANK_SYMBOLS = {
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
    "AXISBANK",
    "KOTAKBANK",
    "INDUSINDBK",
    "BANKBARODA",
    "PNB",
}

# Signal policy already used in the candidate's research/deployment diagnostics.
BUY_P = 0.60
SELL_P = 0.40
MIN_EXPECTED_RETURN = 0.0015
POLICY_REMOTE = f"{DRIVE_MODEL_ROOT}/ACTIVE_SIGNAL_POLICY.json"
LOCAL_POLICY = CACHE_ROOT / "ACTIVE_SIGNAL_POLICY.json"

_manifest_cache: dict[str, Any] | None = None
_classifier = None
_regressor = None
_signal_policy_cache: dict[str, Any] | None = None
_current_nse_symbol_cache: set[str] | None = None
_current_nse_symbol_cache_at: float = 0.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s.rsplit(".", 1)[0]
    return s


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _drive_download(remote_relative_path: str, local_path: Path) -> None:
    """Download one small release artifact from the configured Google Drive."""
    from persistent_store import GoogleDriveStore
    from googleapiclient.http import MediaIoBaseDownload

    store = GoogleDriveStore()

    parts = remote_relative_path.strip("/").split("/")
    parent = store.root_id

    for part in parts[:-1]:
        found = store._find_child(parent, part)
        if not found:
            raise FileNotFoundError(
                f"Google Drive path component not found: {part}"
            )
        parent = found["id"]

    name = parts[-1]
    found = store._find_child(parent, name)

    if not found:
        raise FileNotFoundError(
            f"Google Drive artifact not found: {remote_relative_path}"
        )

    request = store.service.files().get_media(fileId=found["id"])
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    local_path.write_bytes(buf.getvalue())


def _ensure_artifacts() -> None:
    CACHE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    needed = [
        (REMOTE_MANIFEST, LOCAL_MANIFEST),
        (REMOTE_CLASSIFIER, LOCAL_CLASSIFIER),
        (REMOTE_REGRESSOR, LOCAL_REGRESSOR),
    ]

    for remote, local in needed:
        if not local.exists() or local.stat().st_size == 0:
            _drive_download(remote, local)


def _signal_policy() -> dict[str, Any]:
    global _signal_policy_cache
    if _signal_policy_cache is not None:
        return _signal_policy_cache
    policy = {
        "status": "DEFAULT",
        "samples": 0,
        "buy_probability_threshold": BUY_P,
        "sell_probability_threshold": SELL_P,
        "min_expected_return": MIN_EXPECTED_RETURN,
    }
    try:
        if not LOCAL_POLICY.exists() or LOCAL_POLICY.stat().st_size == 0:
            _drive_download(POLICY_REMOTE, LOCAL_POLICY)
        if LOCAL_POLICY.exists() and LOCAL_POLICY.stat().st_size > 0:
            candidate = json.loads(LOCAL_POLICY.read_text(encoding="utf-8"))
            samples = int(candidate.get("samples", 0))
            buy = float(candidate.get("buy_probability_threshold", BUY_P))
            sell = float(candidate.get("sell_probability_threshold", SELL_P))
            mer = float(candidate.get("min_expected_return", MIN_EXPECTED_RETURN))
            if samples >= 80 and 0.10 <= sell < 0.50 < buy <= 0.90 and -0.05 <= mer <= 0.05:
                policy.update({
                    "status": candidate.get("status", "ADAPTIVE"),
                    "samples": samples,
                    "buy_probability_threshold": buy,
                    "sell_probability_threshold": sell,
                    "min_expected_return": mer,
                    "updated_at": candidate.get("updated_at"),
                    "method": candidate.get("method"),
                })
    except Exception:
        pass
    _signal_policy_cache = policy
    return policy


def _current_nse_symbol_set() -> set[str]:
    global _current_nse_symbol_cache, _current_nse_symbol_cache_at
    now = datetime.now().timestamp()
    if _current_nse_symbol_cache is not None and (now - _current_nse_symbol_cache_at) < 900:
        return _current_nse_symbol_cache
    try:
        frame = fetch_nse_equity_universe(force_refresh=False)
        if frame is None or frame.empty:
            return set()
        col = next((c for c in frame.columns if str(c).strip().upper() in {"SYMBOL","CH_SYMBOL","SYMBOL_NAME"}), None)
        if col is None:
            return set()
        values = {_normalize_symbol(v) for v in frame[col].dropna().astype(str) if str(v).strip()}
        _current_nse_symbol_cache = {x for x in values if x}
        _current_nse_symbol_cache_at = now
        return _current_nse_symbol_cache
    except Exception:
        return set()


def _dynamic_history_eligibility(symbol: str, as_of_date: date, minimum_history: int) -> bool:
    root = _normalize_symbol(symbol)
    try:
        history = fetch_equity_history(
            root,
            as_of_date - timedelta(days=370),
            as_of_date,
            series="ALL",
            sync_drive=True,
        )
    except Exception:
        return False
    if history is None or history.empty:
        return False
    x = history.copy()
    required = ("date","open","high","low","close")
    if any(c not in x.columns for c in required):
        return False
    x["date"] = pd.to_datetime(x["date"], errors="coerce").dt.normalize()
    for c in ("open","high","low","close"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=list(required)).sort_values("date").drop_duplicates("date", keep="last")
    return int(x["date"].nunique()) >= int(minimum_history)


def _load_manifest() -> dict[str, Any]:
    global _manifest_cache

    if _manifest_cache is not None:
        return _manifest_cache

    _ensure_artifacts()

    with LOCAL_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as fh:
        manifest = json.load(fh)

    if manifest.get("engine_version") not in EXPECTED_ENGINE_VERSIONS:
        raise RuntimeError(
            "Unexpected ALL-NSE Delivery replacement engine version: "
            f"{manifest.get('engine_version')!r}"
        )

    if int(manifest.get("feature_count", 0)) != 59:
        raise RuntimeError(
            "ALL-NSE Delivery candidate must expose exactly 59 features."
        )

    universe = manifest.get("universe", {})
    universe_symbol_count = int(manifest.get("universe_symbol_count", universe.get("universe_symbol_count", 0)))
    trainable_symbol_count = int(manifest.get("trainable_symbol_count", universe.get("trainable_symbol_count", 0)))
    if universe_symbol_count < 1:
        raise RuntimeError("ALL-NSE replacement manifest has an invalid universe size.")
    if trainable_symbol_count < 1 or trainable_symbol_count > universe_symbol_count:
        raise RuntimeError("ALL-NSE replacement manifest has an invalid trainable-symbol count.")

    _manifest_cache = manifest
    return manifest


def _patch_sklearn_compat(model):
    try:
        for step_name in ("imputer",):
            step = getattr(
                model,
                "named_steps",
                {},
            ).get(step_name)

            if (
                step is not None
                and not hasattr(step, "_fill_dtype")
                and hasattr(step, "_fit_dtype")
            ):
                step._fill_dtype = step._fit_dtype
    except Exception:
        pass

    return model


def _load_models():
    global _classifier, _regressor

    manifest = _load_manifest()
    _ensure_artifacts()

    next_session = manifest.get("artifacts", {}).get("next_session", {})
    expected_classifier_sha = next_session.get("classifier_sha256")
    expected_regressor_sha = next_session.get("regressor_sha256")
    if expected_classifier_sha and _sha256(LOCAL_CLASSIFIER) != expected_classifier_sha:
        raise RuntimeError("ALL-NSE replacement classifier SHA-256 does not match its branch manifest.")
    if expected_regressor_sha and _sha256(LOCAL_REGRESSOR) != expected_regressor_sha:
        raise RuntimeError("ALL-NSE replacement regressor SHA-256 does not match its branch manifest.")

    if _classifier is None:
        _classifier = _patch_sklearn_compat(
            joblib.load(LOCAL_CLASSIFIER)
        )

    if _regressor is None:
        _regressor = _patch_sklearn_compat(
            joblib.load(LOCAL_REGRESSOR)
        )

    for label, model in (
        ("classifier", _classifier),
        ("regressor", _regressor),
    ):
        count = int(
            getattr(
                model,
                "n_features_in_",
                0,
            )
        )

        if count != 59:
            raise RuntimeError(
                f"ALL-NSE Delivery {label} exposes "
                f"{count} features, expected 59."
            )

    classifier_features = list(
        getattr(
            _classifier,
            "feature_names_in_",
            [],
        )
    )

    manifest_features = list(
        manifest.get("features", [])
    )

    if (
        len(classifier_features) != 59
        or classifier_features != manifest_features
    ):
        raise RuntimeError(
            "ALL-NSE Delivery classifier feature contract "
            "does not exactly match its locked manifest."
        )

    return _classifier, _regressor, manifest


def model_status() -> dict[str, Any]:
    clf, reg, manifest = _load_models()

    return {
        "status": "READY",
        "engine_version": manifest.get("engine_version"),
        "feature_count": int(
            manifest.get("feature_count", 0)
        ),
        "universe_symbol_count": int(
            manifest.get("universe_symbol_count", 0)
        ),
        "trainable_symbol_count": int(
            manifest.get("trainable_symbol_count", 0)
        ),
        "excluded_symbol_count": int(
            manifest.get("excluded_symbol_count", 0)
        ),
        "minimum_history_sessions": int(
            manifest.get("minimum_history_sessions", 60)
        ),
        "classifier_sha256": _sha256(LOCAL_CLASSIFIER),
        "regressor_sha256": _sha256(LOCAL_REGRESSOR),
        "manifest_sha256": _sha256(LOCAL_MANIFEST),
        "classifier_n_features": int(
            getattr(clf, "n_features_in_", 0)
        ),
        "regressor_n_features": int(
            getattr(reg, "n_features_in_", 0)
        ),
        "promotion_status_at_build": manifest.get(
            "promotion_status",
            "UNKNOWN",
        ),
        "live_status": "ACTIVE_ALL_NSE_DELIVERY_REPLACEMENT_BRANCH",
        "production_artifacts_originally_changed": bool(
            manifest.get(
                "production_artifacts_changed",
                False,
            )
        ),
    }


def supported_symbol(symbol: str) -> bool:
    root = _normalize_symbol(symbol)
    manifest = _load_manifest()
    trainable = {_normalize_symbol(x) for x in manifest.get("trainable_symbols", [])}
    if root in trainable:
        return True

    today = datetime.now().astimezone().date()
    minimum_history = int(manifest.get("minimum_history_sessions", 60))

    # The daily PIT builder is authoritative for newly eligible IPOs. This
    # avoids waiting for the next model rebuild after session 60 is reached.
    try:
        pit, _ = _latest_drive_pit_features(today)
        if pit is not None and not pit.empty and "symbol" in pit.columns:
            hit = pit[pit["symbol"].astype(str).str.upper().eq(root)]
            if not hit.empty:
                return True
    except Exception:
        pass

    if root not in _current_nse_symbol_set():
        return False
    return _dynamic_history_eligibility(root, today, minimum_history)


def universe_status() -> dict[str, Any]:
    manifest = _load_manifest()

    return {
        "universe_symbol_count": int(
            manifest.get("universe_symbol_count", 0)
        ),
        "trainable_symbol_count": int(
            manifest.get("trainable_symbol_count", 0)
        ),
        "excluded_symbol_count": int(
            manifest.get("excluded_symbol_count", 0)
        ),
        "minimum_history_sessions": int(
            manifest.get("minimum_history_sessions", 60)
        ),
        "engine_version": manifest.get(
            "engine_version"
        ),
    }


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(
            index=frame.index,
            dtype="float64",
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def _ret(series: pd.Series, n: int) -> float:
    x = series.dropna()

    if len(x) <= n:
        return np.nan

    return _safe_float(
        x.iloc[-1] / x.iloc[-1 - n] - 1.0
    )


def _rsi(series: pd.Series, n: int = 14) -> float:
    x = series.dropna()

    if len(x) < n + 1:
        return np.nan

    d = x.diff()

    gain = (
        d.clip(lower=0)
        .rolling(n)
        .mean()
        .iloc[-1]
    )

    loss = (
        (-d.clip(upper=0))
        .rolling(n)
        .mean()
        .iloc[-1]
    )

    if not np.isfinite(gain) or not np.isfinite(loss):
        return np.nan

    if loss == 0:
        return 100.0

    rs = gain / loss

    return float(
        100.0
        - 100.0 / (1.0 + rs)
    )


def _atr(frame: pd.DataFrame, n: int = 14) -> float:
    c = _series(frame, "close")
    h = _series(frame, "high")
    l = _series(frame, "low")

    prev = c.shift(1)

    tr = pd.concat(
        [
            h - l,
            (h - prev).abs(),
            (l - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)

    value = (
        tr.ewm(
            alpha=1 / n,
            adjust=False,
            min_periods=n,
        )
        .mean()
        .iloc[-1]
    )

    return _safe_float(value)


def _market_features(market: pd.DataFrame) -> pd.DataFrame:
    if market is None or market.empty:
        return pd.DataFrame()

    x = market.copy()

    x.columns = [
        str(c).strip().lower()
        for c in x.columns
    ]

    if "date" not in x.columns or "close" not in x.columns:
        raise RuntimeError(
            "NSE benchmark history lacks date/close columns."
        )

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    ).dt.normalize()

    x["close"] = pd.to_numeric(
        x["close"],
        errors="coerce",
    )

    x = (
        x.dropna(
            subset=[
                "date",
                "close",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    c = x["close"]

    d = c.diff()

    gain = (
        d.clip(lower=0)
        .rolling(14)
        .mean()
    )

    loss = (
        (-d.clip(upper=0))
        .rolling(14)
        .mean()
    )

    market_rsi = (
        100.0
        - 100.0 / (
            1.0 + gain / loss
        )
    )

    market_rsi = market_rsi.where(
        loss != 0,
        100.0,
    )

    ret = c.pct_change()

    out = pd.DataFrame({
        "date": x["date"],
        "market_ret_1": c.pct_change(1),
        "market_ret_3": c.pct_change(3),
        "market_ret_6": c.pct_change(6),
        "market_ret_12": c.pct_change(12),
        "market_rsi14": market_rsi,
    })

    out["market_mood_proxy"] = (
        out["market_ret_1"]
    )

    out["market_stress_proxy"] = (
        ret.rolling(20).std()
    )

    return out


def _dedupe_dataframe_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Make duplicate column names deterministic by keeping the first."""
    if frame is None or frame.empty:
        return pd.DataFrame()

    x = frame.copy()

    # Normalize labels first.
    x.columns = [
        str(col).strip().lower()
        for col in x.columns
    ]

    if x.columns.duplicated().any():
        x = x.loc[
            :,
            ~x.columns.duplicated(
                keep="first"
            ),
        ].copy()

    return x


def _build_latest_features(
    history: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:

    x = _dedupe_dataframe_columns(
        history
    )

    if x.empty:
        raise RuntimeError(
            "Historical NSE frame is empty after column normalization."
        )

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in required:
        if col not in x.columns:
            x[col] = np.nan

    x["date"] = pd.to_datetime(
        x["date"],
        errors="coerce",
    ).dt.normalize()

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "delivery_qty",
        "delivery_pct",
    ]:
        if col not in x.columns:
            # Historical compact files may omit delivery fields. Preserve that
            # as explicit missing information so the trained model's imputation
            # policy can handle it; never fabricate delivery values.
            x[col] = np.nan

        x[col] = pd.to_numeric(
            x[col],
            errors="coerce",
        )

    x = (
        x.dropna(
            subset=[
                "date",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    if len(x) < 60:
        raise RuntimeError(
            f"Only {len(x)} clean NSE EOD sessions are available; "
            "60 are required by the locked candidate."
        )

    c = x["close"]
    v = x["volume"]
    dp = x["delivery_pct"]
    dq = x["delivery_qty"]

    features = {}

    for n in [
        1,
        3,
        6,
        12,
        24,
        48,
    ]:
        features[f"ret_{n}"] = _ret(
            c,
            n,
        )

    ma20 = (
        c.rolling(
            20,
            min_periods=20,
        )
        .mean()
        .iloc[-1]
    )

    ma50 = (
        c.rolling(
            50,
            min_periods=50,
        )
        .mean()
        .iloc[-1]
    )

    vma20 = (
        v.rolling(
            20,
            min_periods=20,
        )
        .mean()
        .iloc[-1]
    )

    vstd20 = (
        v.rolling(
            20,
            min_periods=20,
        )
        .std()
        .iloc[-1]
    )

    last = _safe_float(
        c.iloc[-1]
    )

    last_open = _safe_float(
        x["open"].iloc[-1]
    )

    last_high = _safe_float(
        x["high"].iloc[-1]
    )

    last_low = _safe_float(
        x["low"].iloc[-1]
    )

    last_volume = _safe_float(
        v.iloc[-1]
    )

    features["rsi14"] = _rsi(
        c,
        14,
    )

    features["atr14"] = _atr(
        x,
        14,
    )

    features["ma20_ratio"] = (
        last / ma20 - 1.0
        if np.isfinite(ma20)
        and ma20 != 0
        else np.nan
    )

    features["ma50_ratio"] = (
        last / ma50 - 1.0
        if np.isfinite(ma50)
        and ma50 != 0
        else np.nan
    )

    features["volume_ratio"] = (
        last_volume / vma20
        if np.isfinite(last_volume)
        and np.isfinite(vma20)
        and vma20 != 0
        else np.nan
    )

    features["volume_z20"] = (
        (last_volume - vma20)
        / vstd20
        if np.isfinite(last_volume)
        and np.isfinite(vma20)
        and np.isfinite(vstd20)
        and vstd20 != 0
        else np.nan
    )

    # -------------------------------------------------------------------------
    # Point-in-time NSE delivery features
    # -------------------------------------------------------------------------

    delivery_pct = _safe_float(
        dp.iloc[-1]
        if len(dp)
        else np.nan
    )

    delivery_pct_prev = _safe_float(
        dp.iloc[-2]
        if len(dp) >= 2
        else np.nan
    )

    delivery_pct_ma20 = (
        dp.rolling(
            20,
            min_periods=10,
        )
        .mean()
        .iloc[-1]
        if len(dp) >= 10
        else np.nan
    )

    delivery_pct_std20 = (
        dp.rolling(
            20,
            min_periods=10,
        )
        .std()
        .iloc[-1]
        if len(dp) >= 10
        else np.nan
    )

    delivery_qty_ma20 = (
        dq.rolling(
            20,
            min_periods=10,
        )
        .mean()
        .iloc[-1]
        if len(dq) >= 10
        else np.nan
    )

    delivery_qty_std20 = (
        dq.rolling(
            20,
            min_periods=10,
        )
        .std()
        .iloc[-1]
        if len(dq) >= 10
        else np.nan
    )

    delivery_qty = _safe_float(
        dq.iloc[-1]
        if len(dq)
        else np.nan
    )

    features["delivery_pct"] = (
        delivery_pct
    )

    features["delivery_pct_change_1"] = (
        delivery_pct - delivery_pct_prev
        if np.isfinite(delivery_pct)
        and np.isfinite(delivery_pct_prev)
        else np.nan
    )

    features["delivery_pct_ratio20"] = (
        delivery_pct / delivery_pct_ma20
        if np.isfinite(delivery_pct)
        and np.isfinite(delivery_pct_ma20)
        and delivery_pct_ma20 != 0
        else np.nan
    )

    features["delivery_pct_z20"] = (
        (delivery_pct - delivery_pct_ma20)
        / delivery_pct_std20
        if np.isfinite(delivery_pct)
        and np.isfinite(delivery_pct_ma20)
        and np.isfinite(delivery_pct_std20)
        and delivery_pct_std20 != 0
        else np.nan
    )

    features["delivery_qty_ratio20"] = (
        delivery_qty / delivery_qty_ma20
        if np.isfinite(delivery_qty)
        and np.isfinite(delivery_qty_ma20)
        and delivery_qty_ma20 != 0
        else np.nan
    )

    features["delivery_qty_z20"] = (
        (delivery_qty - delivery_qty_ma20)
        / delivery_qty_std20
        if np.isfinite(delivery_qty)
        and np.isfinite(delivery_qty_ma20)
        and np.isfinite(delivery_qty_std20)
        and delivery_qty_std20 != 0
        else np.nan
    )

    # -------------------------------------------------------------------------
    # Candle features
    # -------------------------------------------------------------------------

    features["range_pct"] = (
        (last_high - last_low) / last
        if np.isfinite(last_high)
        and np.isfinite(last_low)
        and np.isfinite(last)
        and last != 0
        else np.nan
    )

    features["body_pct"] = (
        (last - last_open) / last_open
        if np.isfinite(last)
        and np.isfinite(last_open)
        and last_open != 0
        else np.nan
    )

    features["up_bar"] = (
        1.0
        if np.isfinite(last)
        and np.isfinite(last_open)
        and last > last_open
        else 0.0
        if np.isfinite(last)
        and np.isfinite(last_open)
        else np.nan
    )

    # -------------------------------------------------------------------------
    # Official NSE market-context features
    # -------------------------------------------------------------------------

    market_features = _market_features(
        market
    )

    latest_market = (
        market_features[
            market_features["date"]
            <= x["date"].iloc[-1]
        ]
        .tail(1)
    )

    for col in [
        "market_ret_1",
        "market_ret_3",
        "market_ret_6",
        "market_ret_12",
        "market_rsi14",
        "market_mood_proxy",
        "market_stress_proxy",
    ]:

        if latest_market.empty:
            value = np.nan
        else:
            value = latest_market[
                col
            ].iloc[0]

        features[col] = _safe_float(
            value
        )

    for n in [
        1,
        6,
        12,
    ]:

        stock_value = features[
            f"ret_{n}"
        ]

        market_value = features[
            f"market_ret_{n}"
        ]

        features[
            f"relative_ret_{n}"
        ] = (
            stock_value - market_value
            if np.isfinite(stock_value)
            and np.isfinite(market_value)
            else np.nan
        )

    # -------------------------------------------------------------------------
    # Optional information layers.
    #
    # The locked candidate was trained with these fields explicitly missing in
    # the current research run. Leave them NaN rather than injecting current
    # information into a trained distribution that did not contain it.
    # -------------------------------------------------------------------------

    optional_features = [
        "news_available",
        "event_available",
        "event_high_impact",
        "fno_pcr",
        "fno_oi_change",
        "fno_iv_change",
        "fno_available",
        "fno_fut_close",
        "fno_fut_return",
        "fno_fut_oi",
        "fundamental_revenue",
        "fundamental_pat",
        "fundamental_eps",
        "fundamental_ebitda",
        "fundamental_finance_cost",
        "fundamental_total_equity",
        "fundamental_total_debt",
        "fundamental_cash",
        "fund_revenue_growth",
        "fund_pat_growth",
        "fund_eps_growth",
        "fund_ebitda_growth",
        "fund_pat_margin",
        "fund_debt_to_equity",
        "fund_cash_to_debt",
        "fundamental_age_days",
        "fundamental_stale_90d",
        "fundamental_stale_180d",
    ]

    for col in optional_features:
        features[col] = np.nan

    manifest = _load_manifest()
    feature_list = list(
        manifest["features"]
    )

    frame = pd.DataFrame(
        [
            features
        ]
    )

    for col in feature_list:
        if col not in frame.columns:
            frame[col] = np.nan

    frame = frame[
        feature_list
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return frame


@lru_cache(maxsize=128)
def _latest_drive_pit_features(as_of_date: date) -> tuple[pd.DataFrame, str | None]:
    """Load the latest persisted ALL-NSE Delivery PIT snapshot on/before date."""
    cache_dir = CACHE_ROOT / "pit"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        from persistent_store import GoogleDriveStore
        from googleapiclient.http import MediaIoBaseDownload

        store = GoogleDriveStore()

        features_folder = store._find_child(
            store.root_id,
            "features",
            "application/vnd.google-apps.folder",
        )
        if not features_folder:
            return pd.DataFrame(), None

        daily_folder = store._find_child(
            features_folder["id"],
            "daily",
            "application/vnd.google-apps.folder",
        )
        if not daily_folder:
            return pd.DataFrame(), None

        resp = store.service.files().list(
            q=(
                f"'{daily_folder['id']}' in parents "
                "and name contains 'DELIVERY_PIT_FEATURES_' "
                "and trashed = false"
            ),
            spaces="drive",
            fields="files(id,name)",
            pageSize=100,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()

        candidates = []

        for item in resp.get("files", []):
            name = str(item.get("name", ""))
            m = re.search(
                r"DELIVERY_PIT_FEATURES_(20\d{2}-\d{2}-\d{2})\.parquet$",
                name,
            )
            if not m:
                continue

            try:
                d = pd.Timestamp(m.group(1)).date()
            except Exception:
                continue

            if d <= as_of_date:
                candidates.append((d, item["id"], name))

        if not candidates:
            return pd.DataFrame(), None

        target_date, file_id, name = max(
            candidates,
            key=lambda x: x[0],
        )

        local = cache_dir / name

        if not local.exists() or local.stat().st_size == 0:
            request = store.service.files().get_media(fileId=file_id)
            temp = local.with_suffix(".tmp")

            with temp.open("wb") as fh:
                downloader = MediaIoBaseDownload(
                    fh,
                    request,
                )
                done = False
                while not done:
                    _, done = downloader.next_chunk()

            temp.replace(local)

        frame = pd.read_parquet(local)

        if frame.empty:
            return pd.DataFrame(), str(target_date)

        return frame, str(target_date)

    except Exception:
        return pd.DataFrame(), None


def _pit_row_for_symbol(
    symbol: str,
    as_of_date: date,
) -> tuple[pd.DataFrame, str | None]:

    frame, snapshot_date = _latest_drive_pit_features(
        as_of_date
    )

    if frame.empty or "symbol" not in frame.columns:
        return pd.DataFrame(), snapshot_date

    x = frame.copy()

    x["symbol"] = (
        x["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    hit = x[
        x["symbol"] == _normalize_symbol(symbol)
    ].copy()

    if hit.empty:
        return pd.DataFrame(), snapshot_date

    for date_col in (
        "date",
        "prediction_date",
        "as_of_date",
        "feature_date",
    ):
        if date_col in hit.columns:
            parsed = pd.to_datetime(
                hit[date_col],
                errors="coerce",
            ).dt.normalize()

            hit["_pit_date"] = parsed

            hit = hit[
                hit["_pit_date"] <= pd.Timestamp(as_of_date)
            ]

            if not hit.empty:
                hit = hit.sort_values(
                    "_pit_date"
                ).tail(1)

            break

    return hit, snapshot_date


def _official_history(
    symbol: str,
    end_iso: str,
) -> pd.DataFrame:

    end = pd.Timestamp(
        end_iso
    ).date()

    start = (
        end
        - timedelta(days=365)
    )

    history = fetch_equity_history(
        symbol,
        start,
        end,
        series="EQ",
        sync_drive=True,
    )

    if history is None or history.empty:
        raise RuntimeError(
            f"NSE official EOD history unavailable for {symbol}."
        )

    return history


@lru_cache(maxsize=32)
def _official_market_history(
    benchmark: str,
    end_iso: str,
) -> pd.DataFrame:

    end = pd.Timestamp(
        end_iso
    ).date()

    start = (
        end
        - timedelta(days=365)
    )

    market = fetch_index_history(
        benchmark,
        start,
        end,
        sync_drive=True,
    )

    if market is None or market.empty:
        raise RuntimeError(
            f"NSE official benchmark history unavailable for {benchmark}."
        )

    return market


def predict_symbol(
    symbol: str,
    decision_timestamp: Any = None,
    market_override: pd.DataFrame | None = None,
) -> dict[str, Any]:

    root = _normalize_symbol(
        symbol
    )

    if not supported_symbol(root):
        universe = universe_status()

        return {
            "symbol": root,
            "prediction_status": "BLOCKED",
            "reason": "symbol_not_in_current_all_nse_trainable_roster",
            "universe_symbol_count": universe[
                "universe_symbol_count"
            ],
            "trainable_symbol_count": universe[
                "trainable_symbol_count"
            ],
            "excluded_symbol_count": universe[
                "excluded_symbol_count"
            ],
        }

    # Prefer the exact persisted PIT feature snapshot from the candidate
    # builder. This avoids feature-definition drift between Colab and Streamlit.
    now_ist = (
        pd.Timestamp.now(
            tz="Asia/Kolkata"
        )
    )

    if decision_timestamp is not None:
        try:
            ts0 = pd.Timestamp(
                decision_timestamp
            )
            if ts0.tzinfo is None:
                ts0 = ts0.tz_localize(
                    "Asia/Kolkata"
                )
            now_ist = ts0.tz_convert(
                "Asia/Kolkata"
            )
        except Exception:
            pass

    pit_row, pit_date = _pit_row_for_symbol(
        root,
        now_ist.date(),
    )

    clf, reg, manifest = _load_models()
    expected_features = list(
        manifest["features"]
    )

    if not pit_row.empty:

        feature_frame = pd.DataFrame([
            {
                col: (
                    pit_row.iloc[0][col]
                    if col in pit_row.columns
                    else np.nan
                )
                for col in expected_features
            }
        ]).apply(
            pd.to_numeric,
            errors="coerce",
        )

        if list(feature_frame.columns) != expected_features:
            raise RuntimeError(
                "Persisted PIT feature order does not match the "
                "locked 59-feature model contract."
            )

        p_up = float(
            np.clip(
                clf.predict_proba(
                    feature_frame
                )[0, 1],
                0.0,
                1.0,
            )
        )

        expected_return = float(
            reg.predict(
                feature_frame
            )[0]
        )

        # Numeric price outputs are derived directly from the locked model's
        # predicted next-session return and the causal PIT reference close.
        # The production model itself remains unchanged.
        reference_close = _safe_float(
            pit_row.iloc[0].get("close", np.nan)
        )
        reference_open = _safe_float(
            pit_row.iloc[0].get("open", np.nan)
        )
        reference_high = _safe_float(
            pit_row.iloc[0].get("high", np.nan)
        )
        reference_low = _safe_float(
            pit_row.iloc[0].get("low", np.nan)
        )
        predicted_price_next_session = (
            reference_close * (1.0 + expected_return)
            if np.isfinite(reference_close)
            and np.isfinite(expected_return)
            else np.nan
        )
        predicted_price_change_next_session = (
            predicted_price_next_session - reference_close
            if np.isfinite(predicted_price_next_session)
            and np.isfinite(reference_close)
            else np.nan
        )

        policy = _signal_policy()
        buy_p = float(policy["buy_probability_threshold"])
        sell_p = float(policy["sell_probability_threshold"])
        min_expected_return = float(policy["min_expected_return"])

        if p_up >= buy_p and expected_return > min_expected_return:
            signal = "BUY"
        elif p_up <= sell_p and expected_return < -min_expected_return:
            signal = "SELL"
        else:
            signal = "HOLD"

        raw_gate = pit_row.iloc[0].get("production_data_quality_gate", np.nan)
        data_gate = bool(raw_gate) if pd.notna(raw_gate) else False
        if not data_gate and signal in {"BUY", "SELL"}:
            signal = "HOLD"

        non_missing = int(
            feature_frame.notna()
            .sum(axis=1)
            .iloc[0]
        )

        # The locked ALL-NSE model is the production prediction path.
        # Auxiliary legacy-model ensemble scoring is intentionally excluded from
        # the live path so unrelated joblib artifacts and additional sklearn
        # pipelines are not loaded during production prediction.
        # This does not alter the 59-feature production classifier/regressor,
        # signal policy, data-quality gate, or prediction itself.
        auxiliary_models = {
            "status": "NOT_RUN",
            "reason": "production_path_primary_model_only",
        }
        auxiliary_horizons = {
            "status": "NOT_RUN",
            "reason": "production_path_primary_model_only",
        }
        return {
            "symbol": root,
            "prediction_status": "PASS",
            "mode": "Delivery",
            "prediction_horizon": "next_session",
            "model_version": manifest.get("engine_version"),
            "model_family": manifest.get(
                "model_family",
                "logistic_regression + hgb_return_regressor",
            ),
            "engine_version": manifest.get(
                "engine_version"
            ),
            "decision_timestamp": (
                f"{pit_date}T15:30:00+05:30"
                if pit_date
                else now_ist.isoformat()
            ),
            "reference_session": pit_date,
            "probability_up_next_session": p_up,
            "probability_down_next_session": 1.0 - p_up,
            "predicted_return_next_session": expected_return,
            "reference_open": reference_open,
            "reference_high": reference_high,
            "reference_low": reference_low,
            "reference_close": reference_close,
            "predicted_price_next_session": predicted_price_next_session,
            "predicted_price_change_next_session": predicted_price_change_next_session,
            "signal": signal,
            "confidence": (
                "HIGH"
                if abs(p_up - 0.5) >= 0.25
                else "MEDIUM"
                if abs(p_up - 0.5) >= 0.12
                else "LOW"
            ),
            "feature_count": len(expected_features),
            "non_missing_feature_count": non_missing,
            "feature_non_missing_fraction": (
                non_missing / float(len(expected_features))
            ),
            "data_readiness_score": (
                non_missing / float(len(expected_features))
            ),
            "feature_source": "PERSISTED_PIT_SNAPSHOT",
            "pit_snapshot_date": pit_date,
            "optional_information_status": {
                "news": "KNOWN" if pd.notna(feature_frame.iloc[0].get("news_available")) else "UNKNOWN",
                "fno": "KNOWN" if pd.notna(feature_frame.iloc[0].get("fno_available")) else "UNKNOWN",
                "fundamentals": "KNOWN" if pd.notna(feature_frame.iloc[0].get("fundamental_age_days")) else "UNKNOWN",
            },
            "production_data_quality_gate": data_gate,
            "production_data_quality_score": _safe_float(pit_row.iloc[0].get("production_data_quality_score", np.nan)),
            "signal_policy": policy,
            "auxiliary_model_ensemble": auxiliary_models,
            "auxiliary_multi_horizon_evidence": auxiliary_horizons,
            "auxiliary_model_consensus_p_up": (
                auxiliary_models.get("consensus_p_up")
                if isinstance(auxiliary_models, dict)
                else None
            ),
            "auxiliary_model_consensus_direction": (
                auxiliary_models.get("consensus_direction", "UNKNOWN")
                if isinstance(auxiliary_models, dict)
                else "UNKNOWN"
            ),
            "universe_symbol_count": int(
                manifest.get("universe_symbol_count", 2593)
            ),
            "trainable_symbol_count": int(
                manifest.get("trainable_symbol_count", 2372)
            ),
            "excluded_symbol_count": int(
                manifest.get("excluded_symbol_count", 221)
            ),
            "minimum_history_sessions": int(
                manifest.get("minimum_history_sessions", 60)
            ),
        }

    # Fallback: build the current feature row from official NSE history when
    # no persisted PIT snapshot is available.
    # Use the latest completed official NSE session available.
    now_ist = (
        pd.Timestamp.now(
            tz="Asia/Kolkata"
        )
    )

    if decision_timestamp is not None:
        try:
            ts = pd.Timestamp(
                decision_timestamp
            )
            if ts.tzinfo is None:
                ts = ts.tz_localize(
                    "Asia/Kolkata"
                )
            now_ist = ts.tz_convert(
                "Asia/Kolkata"
            )
        except Exception:
            pass

    request_end = now_ist.date()

    history = _official_history(
        root,
        str(request_end),
    )

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce",
    ).dt.normalize()

    history = (
        history[
            history["date"]
            <= pd.Timestamp(request_end)
        ]
        .sort_values("date")
        .drop_duplicates(
            "date",
            keep="last",
        )
        .reset_index(drop=True)
    )

    if history.empty:
        raise RuntimeError(
            f"No completed official NSE EOD history available for {root}."
        )

    latest_date = pd.Timestamp(
        history["date"].iloc[-1]
    ).normalize()

    benchmark = benchmark_for_symbol(
        root
    )

    if (
        market_override is not None
        and not market_override.empty
    ):
        market = market_override.copy()
        market_source = "INJECTED_REPLAY_BENCHMARK"
    else:
        market = _official_market_history(
            benchmark,
            str(latest_date.date()),
        )
        market_source = "OFFICIAL_NSE"

    feature_frame = _build_latest_features(
        history,
        market,
    )

    clf, reg, manifest = _load_models()

    expected_features = list(
        manifest["features"]
    )

    if list(feature_frame.columns) != expected_features:
        raise RuntimeError(
            "Live ALL-NSE Delivery feature order/contract "
            "does not exactly match the locked 59-feature model."
        )

    p_up = float(
        np.clip(
            clf.predict_proba(
                feature_frame
            )[0, 1],
            0.0,
            1.0,
        )
    )

    expected_return = float(
        reg.predict(
            feature_frame
        )[0]
    )

    policy = _signal_policy()
    buy_p = float(policy["buy_probability_threshold"])
    sell_p = float(policy["sell_probability_threshold"])
    min_expected_return = float(policy["min_expected_return"])

    if p_up >= buy_p and expected_return > min_expected_return:
        signal = "BUY"
    elif p_up <= sell_p and expected_return < -min_expected_return:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Direct live fallback does not contain the persisted production readiness
    # state for news, events, F&O, and fundamentals; fail closed to HOLD.
    signal = "HOLD"

    non_missing = int(
        feature_frame.notna()
        .sum(axis=1)
        .iloc[0]
    )

    try:
        from directional_ensemble import (
            model_ensemble_direction,
            multi_horizon_model_evidence,
        )

        auxiliary_models = model_ensemble_direction(
            root,
            feature_frame,
            horizon="next_session",
            fno_available=False,
        )

        auxiliary_horizons = multi_horizon_model_evidence(
            root,
            feature_frame,
            fno_available=False,
        )

    except Exception as exc:
        auxiliary_models = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }
        auxiliary_horizons = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }

    feature_count = int(
        len(expected_features)
    )

    # Core readiness is intentionally based on the price/technical,
    # NSE-delivery and market-context features actually available to this
    # candidate; optional news/F&O/fundamentals remain explicitly absent.
    core_features = [
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "ret_48",
        "rsi14",
        "atr14",
        "ma20_ratio",
        "ma50_ratio",
        "volume_ratio",
        "volume_z20",
        "delivery_pct",
        "delivery_pct_change_1",
        "delivery_pct_ratio20",
        "delivery_pct_z20",
        "delivery_qty_ratio20",
        "delivery_qty_z20",
        "range_pct",
        "body_pct",
        "up_bar",
        "market_ret_1",
        "market_ret_3",
        "market_ret_6",
        "market_ret_12",
        "market_rsi14",
        "relative_ret_1",
        "relative_ret_6",
        "relative_ret_12",
        "market_mood_proxy",
        "market_stress_proxy",
    ]

    core_available = int(
        feature_frame[
            core_features
        ]
        .notna()
        .sum(axis=1)
        .iloc[0]
    )

    core_readiness = (
        core_available
        / float(len(core_features))
    )

    decision_ts = pd.Timestamp(
        latest_date.date()
    ).tz_localize(
        "Asia/Kolkata"
    ) + pd.Timedelta(
        hours=15,
        minutes=30,
    )

    return {
        "symbol": root,
        "prediction_status": "PASS",
        "mode": "Delivery",
        "prediction_horizon": "next_session",
        "model_version": EXPECTED_ENGINE_VERSION,
        "model_family": manifest.get(
            "model_family",
            "logistic_regression + hgb_return_regressor",
        ),
        "engine_version": manifest.get(
            "engine_version"
        ),
        "decision_timestamp": decision_ts.isoformat(),
        "reference_session": str(
            latest_date.date()
        ),
        "market_context_source": market_source,
        "probability_up_next_session": p_up,
        "probability_down_next_session": 1.0 - p_up,
        "predicted_return_next_session": expected_return,
        "signal": signal,
        "confidence": (
            "HIGH"
            if abs(p_up - 0.5) >= 0.25
            else "MEDIUM"
            if abs(p_up - 0.5) >= 0.12
            else "LOW"
        ),
        "feature_count": feature_count,
        "non_missing_feature_count": non_missing,
        "feature_non_missing_fraction": (
            non_missing / float(feature_count)
        ),
        "data_readiness_score": float(
            core_readiness
        ),
        "auxiliary_model_ensemble": auxiliary_models,
        "auxiliary_multi_horizon_evidence": auxiliary_horizons,
        "auxiliary_model_consensus_p_up": (
            auxiliary_models.get("consensus_p_up")
            if isinstance(auxiliary_models, dict)
            else None
        ),
        "auxiliary_model_consensus_direction": (
            auxiliary_models.get("consensus_direction", "UNKNOWN")
            if isinstance(auxiliary_models, dict)
            else "UNKNOWN"
        ),
        "optional_information_status": {
            "news": "UNKNOWN_IN_FALLBACK",
            "fno": "UNKNOWN_IN_FALLBACK",
            "fundamentals": "UNKNOWN_IN_FALLBACK",
        },
        "production_data_quality_gate": False,
        "production_data_quality_score": float(core_readiness),
        "signal_policy": policy,
        "universe_symbol_count": int(
            manifest.get(
                "universe_symbol_count",
                2593,
            )
        ),
        "trainable_symbol_count": int(
            manifest.get(
                "trainable_symbol_count",
                2372,
            )
        ),
        "excluded_symbol_count": int(
            manifest.get(
                "excluded_symbol_count",
                221,
            )
        ),
        "minimum_history_sessions": int(
            manifest.get(
                "minimum_history_sessions",
                60,
            )
        ),
    }