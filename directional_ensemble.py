"""Gated multi-model directional evidence for the NSE Delivery engine.

This module does not retrain or mutate existing artifacts. It discovers the
legacy research joblib artifacts already stored in Google Drive, checks their
feature contract/horizon/scope before scoring, and combines only compatible
outputs. The locked ALL-NSE 59-feature model remains the primary model.

External information is analysed separately:
- stock-specific news
- NSE corporate announcements
- stock technical state
- broad/benchmark market regime
- optional F&O context

No current/future information is substituted for missing historical inputs.
The returned ensemble/context fields are advisory evidence until independently
validated OOS.
"""
from __future__ import annotations

import io
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


LEGACY_VALIDATED_SYMBOLS = {
    "BHARTIARTL",
    "RELIANCE",
    "SBIN",
}

NEXT_SESSION_HORIZON_NAMES = {
    "next_session",
    "1d",
}

MODELS_REMOTE_ROOT = "models"
LOCAL_CACHE = (
    Path(__file__).resolve().parent
    / "data"
    / "directional_model_cache"
)


def _root(symbol: str) -> str:
    return (
        str(symbol or "")
        .strip()
        .upper()
        .replace(".NS", "")
        .replace(".BO", "")
    )


def _safe(value: Any) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def _model_horizon(name: str) -> str:
    lower = name.lower()
    if "next_session" in lower:
        return "next_session"
    if "_30m_" in lower or lower.startswith("deployment_30m_"):
        return "30m"
    if "_60m_" in lower or lower.startswith("deployment_60m_"):
        return "60m"
    if "_1d_" in lower:
        return "1d"
    if "_7d_" in lower:
        return "7d"
    return "unknown"


def _model_scope(name: str) -> str:
    upper = name.upper()

    for symbol in LEGACY_VALIDATED_SYMBOLS:
        if upper.startswith(symbol + "_"):
            return symbol

    if upper.startswith("GLOBAL_"):
        return "GLOBAL_LEGACY"

    if upper.startswith("DEPLOYMENT_"):
        return "DEPLOYMENT_LEGACY"

    if upper.startswith("FNO_ONLY_"):
        return "FNO_ONLY"

    return "UNKNOWN"


def _is_classifier(name: str) -> bool:
    return "classifier" in name.lower()


def _is_regressor(name: str) -> bool:
    return "regressor" in name.lower()


def _list_drive_model_files():
    from persistent_store import GoogleDriveStore

    store = GoogleDriveStore()

    root = store._find_child(
        store.root_id,
        MODELS_REMOTE_ROOT,
        "application/vnd.google-apps.folder",
    )

    if not root:
        return []

    found = []

    def walk(folder_id: str, rel: str, depth: int = 0):
        if depth > 3:
            return

        token = None

        while True:

            kwargs = {
                "q": (
                    f"'{folder_id}' in parents "
                    "and trashed = false"
                ),
                "spaces": "drive",
                "fields": "nextPageToken,files(id,name,mimeType)",
                "pageSize": 1000,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }

            if token:
                kwargs["pageToken"] = token

            response = store.service.files().list(
                **kwargs
            ).execute()

            for item in response.get("files", []):

                name = str(
                    item.get("name", "")
                ).strip()

                mime = item.get(
                    "mimeType",
                    "",
                )

                if mime == "application/vnd.google-apps.folder":
                    walk(
                        item["id"],
                        f"{rel}/{name}".strip("/"),
                        depth + 1,
                    )

                elif name.lower().endswith(".joblib"):
                    found.append({
                        "id": item["id"],
                        "name": name,
                        "relative_path": (
                            f"{rel}/{name}".strip("/")
                        ),
                    })

            token = response.get(
                "nextPageToken"
            )

            if not token:
                break

    walk(
        root["id"],
        "",
        0,
    )

    return found


@lru_cache(maxsize=1)
def discover_legacy_artifacts() -> tuple[dict[str, Any], ...]:
    """Discover existing research artifacts in Drive once per runtime."""
    files = _list_drive_model_files()

    output = []

    for item in files:

        name = item["name"]

        # Current ALL-NSE pair is managed by all_nse_delivery_model.py.
        if "all_nse_delivery_" in name.lower():
            continue

        # The certified 137-feature intraday model is managed by
        # production_model.py and must not be mixed into Delivery.
        if (
            name == "deployment_classifier_logistic_regression_2026.joblib"
        ):
            continue

        horizon = _model_horizon(name)
        scope = _model_scope(name)

        if horizon == "unknown":
            continue

        output.append({
            **item,
            "horizon": horizon,
            "scope": scope,
            "classifier": _is_classifier(name),
            "regressor": _is_regressor(name),
        })

    return tuple(
        sorted(
            output,
            key=lambda x: (
                x["horizon"],
                x["scope"],
                x["name"],
            ),
        )
    )


def _download_artifact(
    item: dict[str, Any],
) -> Path:

    from persistent_store import GoogleDriveStore
    from googleapiclient.http import MediaIoBaseDownload

    LOCAL_CACHE.mkdir(
        parents=True,
        exist_ok=True,
    )

    local = (
        LOCAL_CACHE
        / item["name"]
    )

    if (
        not local.exists()
        or local.stat().st_size == 0
    ):

        store = GoogleDriveStore()

        request = (
            store.service
            .files()
            .get_media(
                fileId=item["id"]
            )
        )

        temp = local.with_suffix(
            ".tmp"
        )

        with temp.open("wb") as fh:

            downloader = MediaIoBaseDownload(
                fh,
                request,
            )

            done = False

            while not done:
                _, done = downloader.next_chunk()

        temp.replace(local)

    return local


def _patch_sklearn_compat(model):
    try:
        imp = getattr(
            model,
            "named_steps",
            {},
        ).get("imputer")

        if (
            imp is not None
            and not hasattr(
                imp,
                "_fill_dtype",
            )
            and hasattr(
                imp,
                "_fit_dtype",
            )
        ):
            imp._fill_dtype = imp._fit_dtype

    except Exception:
        pass

    return model


@lru_cache(maxsize=128)
def _load_artifact(
    file_id: str,
    name: str,
):
    item = {
        "id": file_id,
        "name": name,
    }

    path = _download_artifact(
        item
    )

    return _patch_sklearn_compat(
        joblib.load(path)
    )


def _feature_names(model) -> list[str]:
    names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if names is None:
        names = getattr(
            getattr(
                model,
                "named_steps",
                {},
            ).get(
                "imputer"
            ),
            "feature_names_in_",
            None,
        )

    if names is None:
        return []

    return [
        str(x)
        for x in names
    ]


def _scope_allowed(
    scope: str,
    symbol: str,
) -> bool:

    root = _root(symbol)

    if scope == "GLOBAL_LEGACY":
        return root in LEGACY_VALIDATED_SYMBOLS

    if scope == "DEPLOYMENT_LEGACY":
        return root in LEGACY_VALIDATED_SYMBOLS

    if scope in LEGACY_VALIDATED_SYMBOLS:
        return root == scope

    if scope == "FNO_ONLY":
        return True

    return False


def _score_model_pair(
    classifier_item: dict[str, Any],
    regressor_item: dict[str, Any] | None,
    feature_row: pd.DataFrame,
    symbol: str,
) -> dict[str, Any]:

    root = _root(symbol)

    clf = _load_artifact(
        classifier_item["id"],
        classifier_item["name"],
    )

    required = _feature_names(
        clf
    )

    if not required:
        return {
            "status": "SKIP",
            "model": classifier_item["name"],
            "reason": "classifier_feature_contract_unavailable",
        }

    missing_contract = [
        col
        for col in required
        if col not in feature_row.columns
    ]

    if missing_contract:
        return {
            "status": "SKIP",
            "model": classifier_item["name"],
            "reason": "feature_contract_not_covered",
            "missing_features": missing_contract,
        }

    X = feature_row[
        required
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().all(axis=None):
        return {
            "status": "SKIP",
            "model": classifier_item["name"],
            "reason": "all_required_inputs_missing",
        }

    try:
        p_up = _safe(
            clf.predict_proba(X)[0, 1]
        )
    except Exception as exc:
        return {
            "status": "SKIP",
            "model": classifier_item["name"],
            "reason": (
                f"classifier_error:{type(exc).__name__}"
            ),
        }

    expected = np.nan

    if regressor_item is not None:
        try:
            reg = _load_artifact(
                regressor_item["id"],
                regressor_item["name"],
            )

            reg_features = _feature_names(
                reg
            )

            if reg_features and all(
                col in feature_row.columns
                for col in reg_features
            ):
                expected = _safe(
                    reg.predict(
                        feature_row[
                            reg_features
                        ].apply(
                            pd.to_numeric,
                            errors="coerce",
                        )
                    )[0]
                )

        except Exception:
            expected = np.nan

    direction = (
        "UP"
        if np.isfinite(p_up) and p_up >= 0.5
        else "DOWN"
        if np.isfinite(p_up)
        else "UNKNOWN"
    )

    return {
        "status": "PASS",
        "model": classifier_item["name"],
        "regressor": (
            regressor_item["name"]
            if regressor_item is not None
            else None
        ),
        "scope": classifier_item["scope"],
        "horizon": classifier_item["horizon"],
        "p_up": p_up,
        "expected_return": expected,
        "direction": direction,
        "feature_count": len(required),
        "missing_inputs": int(
            X.isna().sum(axis=1).iloc[0]
        ),
    }


def model_ensemble_direction(
    symbol: str,
    feature_row: pd.DataFrame,
    horizon: str = "next_session",
    fno_available: bool = False,
) -> dict[str, Any]:

    artifacts = discover_legacy_artifacts()

    root = _root(symbol)

    pairs = {}

    for item in artifacts:

        model_horizon = item["horizon"]

        compatible_horizon = (
            (
                horizon == "next_session"
                and model_horizon
                in NEXT_SESSION_HORIZON_NAMES
            )
            or model_horizon == horizon
        )

        if not compatible_horizon:
            continue

        if item["scope"] == "FNO_ONLY" and not fno_available:
            continue

        if not _scope_allowed(
            item["scope"],
            root,
        ):
            continue

        key = (
            item["scope"],
            model_horizon,
        )

        entry = pairs.setdefault(
            key,
            {
                "classifier": None,
                "regressor": None,
            },
        )

        if item["classifier"]:
            entry["classifier"] = item

        if item["regressor"]:
            entry["regressor"] = item

    votes = []

    for key, pair in pairs.items():

        if pair["classifier"] is None:
            continue

        votes.append(
            _score_model_pair(
                pair["classifier"],
                pair["regressor"],
                feature_row,
                root,
            )
        )

    passed = [
        vote
        for vote in votes
        if vote.get("status") == "PASS"
        and np.isfinite(
            _safe(vote.get("p_up"))
        )
    ]

    if not passed:
        return {
            "status": "NO_COMPATIBLE_LEGACY_MODEL",
            "registered_artifacts": len(artifacts),
            "eligible_models": 0,
            "votes": votes,
        }

    weighted = []

    for vote in passed:

        scope = vote["scope"]

        weight = (
            2.0
            if scope == root
            else 1.0
        )

        # F&O-only 1D models are specialist evidence, not the base model.
        if scope == "FNO_ONLY":
            weight = 0.75

        weighted.append(
            (
                weight,
                float(vote["p_up"]),
            )
        )

    denom = sum(
        weight
        for weight, _ in weighted
    )

    consensus_p_up = (
        sum(
            weight * p
            for weight, p in weighted
        )
        / denom
        if denom
        else np.nan
    )

    consensus_direction = (
        "UP"
        if consensus_p_up >= 0.5
        else "DOWN"
        if np.isfinite(consensus_p_up)
        else "UNKNOWN"
    )

    return {
        "status": "PASS",
        "registered_artifacts": len(
            artifacts
        ),
        "eligible_models": len(passed),
        "consensus_p_up": consensus_p_up,
        "consensus_direction": consensus_direction,
        "votes": votes,
    }


def multi_horizon_model_evidence(
    symbol: str,
    feature_row: pd.DataFrame,
    fno_available: bool = False,
) -> dict[str, Any]:
    """Score compatible auxiliary artifacts separately by prediction horizon.

    Horizons are never averaged together: a 30m vote is not treated as a
    next-session vote. The result is evidence for the directional engine and
    retains each horizon's provenance.
    """
    horizons = [
        "30m",
        "60m",
        "next_session",
        "7d",
    ]

    out: dict[str, Any] = {}

    for horizon in horizons:
        try:
            out[horizon] = model_ensemble_direction(
                symbol,
                feature_row,
                horizon=horizon,
                fno_available=fno_available,
            )
        except Exception as exc:
            out[horizon] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }

    return out


def directional_context(
    symbol: str,
    as_of: Any,
    stock_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    news_items: list[dict[str, Any]] | None = None,
    announcements: list[dict[str, Any]] | None = None,
    fno: dict[str, Any] | None = None,
) -> dict[str, Any]:

    from signals import (
        combined_signal,
        market_regime_score,
        news_metrics,
        technical_score,
    )
    from triggers import (
        announcement_to_news,
        event_risk_score,
    )

    root = _root(symbol)

    stock = (
        stock_history.copy()
        if stock_history is not None
        else pd.DataFrame()
    )

    market = (
        benchmark_history.copy()
        if benchmark_history is not None
        else pd.DataFrame()
    )

    ts = pd.Timestamp(as_of)

    if not stock.empty:

        if "date" in stock.columns:
            stock["date"] = pd.to_datetime(
                stock["date"],
                errors="coerce",
            ).dt.normalize()

            stock = stock[
                stock["date"]
                <= ts.normalize()
            ].copy()

        if "Close" not in stock.columns:
            stock = stock.rename(
                columns={
                    "close": "Close",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "volume": "Volume",
                }
            )

    historical_news_items = (
        list(news_items)
        if news_items is not None
        else []
    )

    historical_announcements = (
        list(announcements)
        if announcements is not None
        else []
    )

    event_news = announcement_to_news(
        historical_announcements
    )

    combined_news = (
        historical_news_items
        + event_news
    )

    news = news_metrics(
        combined_news,
        now=ts.to_pydatetime()
        if ts.tzinfo is not None
        else ts.to_pydatetime(),
    )

    event_risk = event_risk_score(
        historical_news_items,
        historical_announcements,
    )

    tech = 0.0

    if not stock.empty:

        try:
            tech = _safe(
                technical_score(
                    stock,
                    fast=20,
                    slow=50,
                )
            )
        except Exception:
            tech = 0.0

    market_score = 0.0

    if not market.empty:

        try:
            if "Close" not in market.columns:
                market = market.rename(
                    columns={
                        "close": "Close",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "volume": "Volume",
                    }
                )

            market_score = _safe(
                market_regime_score(
                    market,
                    fast=20,
                    slow=50,
                )
            )
        except Exception:
            market_score = 0.0

    context_direction, context_score = combined_signal(
        tech_score_value=(
            0.0 if not np.isfinite(tech)
            else tech
        ),
        news_score=(
            0.0
            if not np.isfinite(
                news.get("score", np.nan)
            )
            else float(news["score"])
        ),
        market_score=(
            0.0
            if not np.isfinite(market_score)
            else market_score
        ),
        event_risk=(
            0.0
            if not np.isfinite(event_risk)
            else event_risk
        ),
        tech_weight=0.45,
        news_weight=0.30,
        market_weight=0.25,
    )

    return {
        "symbol": root,
        "as_of": ts.isoformat(),
        "technical_score": _safe(tech),
        "news_score": _safe(
            news.get("score")
        ),
        "news_count": int(
            news.get("count", 0) or 0
        ),
        "high_impact_news_count": int(
            news.get("high_impact", 0) or 0
        ),
        "announcement_count": len(
            historical_announcements
        ),
        "event_risk": _safe(
            event_risk
        ),
        "market_regime_score": _safe(
            market_score
        ),
        "context_score": _safe(
            context_score
        ),
        "context_direction": context_direction,
        "fno_available": bool(
            fno
        ),
    }


def enrich_delivery_prediction(
    result: dict[str, Any],
    symbol: str,
    feature_row: pd.DataFrame,
    stock_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    as_of: Any,
    news_items: list[dict[str, Any]] | None = None,
    announcements: list[dict[str, Any]] | None = None,
    fno: dict[str, Any] | None = None,
) -> dict[str, Any]:

    out = dict(result)

    fno_available = bool(
        fno
    )

    ensemble = model_ensemble_direction(
        symbol,
        feature_row,
        horizon="next_session",
        fno_available=fno_available,
    )

    multi_horizon = multi_horizon_model_evidence(
        symbol,
        feature_row,
        fno_available=fno_available,
    )

    context = directional_context(
        symbol,
        as_of,
        stock_history,
        benchmark_history,
        news_items=news_items,
        announcements=announcements,
        fno=fno,
    )

    base_p = _safe(
        result.get(
            "probability_up_next_session"
        )
    )

    ensemble_p = _safe(
        ensemble.get(
            "consensus_p_up"
        )
    )

    # The current locked ALL-NSE output remains authoritative. The new evidence
    # is deliberately exposed as a separate consensus until prospective OOS
    # testing establishes a safe fusion weight.
    out.update({
        "legacy_model_ensemble": ensemble,
        "multi_horizon_model_evidence": multi_horizon,
        "directional_information_context": context,
        "base_model_direction": (
            "UP"
            if np.isfinite(base_p) and base_p >= 0.5
            else "DOWN"
            if np.isfinite(base_p)
            else "UNKNOWN"
        ),
        "ensemble_context_direction": (
            context.get(
                "context_direction",
                "UNKNOWN",
            )
        ),
        "ensemble_consensus_direction": ensemble.get(
            "consensus_direction",
            "UNKNOWN",
        ),
        "ensemble_consensus_p_up": ensemble_p,
        "ensemble_context_status": (
            "AVAILABLE"
            if (
                ensemble.get("status") == "PASS"
                or context.get("news_count", 0) > 0
                or context.get("announcement_count", 0) > 0
                or np.isfinite(
                    _safe(
                        context.get(
                            "market_regime_score"
                        )
                    )
                )
            )
            else "LIMITED"
        ),
    })

    return out
