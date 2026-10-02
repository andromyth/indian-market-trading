"""Full multi-model engine foundation for Indian NSE.

The module is UI-independent and deliberately contract-gated.

Responsibilities:
- discover real .joblib artifacts from Google Drive
- classify scope and forecast horizon
- inspect feature contracts
- lazily load artifacts
- determine eligibility from symbol/horizon/data contract
- score compatible models
- keep horizons separate
- produce an audit explaining PASS/SKIP decisions

The current ALL-NSE 59-feature Delivery model remains the primary model in the
existing engine. This module manages auxiliary/specialist evidence until those
artifacts have full-universe prospective OOS validation.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


SUPPORTED_HORIZONS = {
    "30m",
    "60m",
    "next_session",
    "1d",
    "3d",
    "5d",
    "7d",
}

PRIMARY_MARKERS = (
    "all_nse_delivery_classifier",
    "all_nse_delivery_regressor",
    "ALL_NSE_DELIVERY_MODEL_MANIFEST",
)

LEGACY_CERTIFIED_SYMBOLS = {
    "BHARTIARTL",
    "RELIANCE",
    "SBIN",
}

LOCAL_CACHE_ROOT = (
    Path(__file__).resolve().parent
    / "data"
    / "full_multimodel_cache"
)


@dataclass(frozen=True)
class ModelRecord:
    name: str
    drive_id: str
    relative_path: str
    role: str
    scope: str
    horizon: str
    is_classifier: bool
    is_regressor: bool
    feature_count: int | None = None
    feature_names: tuple[str, ...] = ()
    sha256: str | None = None
    metadata_status: str = "UNINSPECTED"
    metadata_error: str | None = None


def normalize_symbol(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        s = s.rsplit(".", 1)[0]
    return s


def classify_horizon(name: str) -> str:
    lower = str(name).lower()

    if "30m" in lower:
        return "30m"
    if "60m" in lower:
        return "60m"
    if "next_session" in lower:
        return "next_session"

    match = re.search(
        r"_(1|3|5|7)d(?:_|\.|$)",
        lower,
    )
    if match:
        return f"{match.group(1)}d"

    return "unknown"


def classify_scope(name: str) -> str:
    upper = str(name).upper()

    if "FNO_ONLY" in upper:
        return "FNO_ONLY"

    if upper.startswith("GLOBAL_"):
        return "GLOBAL"

    if upper.startswith("DEPLOYMENT_"):
        return "DEPLOYMENT_LEGACY"

    for symbol in sorted(
        LEGACY_CERTIFIED_SYMBOLS,
        key=len,
        reverse=True,
    ):
        if upper.startswith(symbol + "_"):
            return f"SYMBOL:{symbol}"

    return "UNKNOWN"


def classify_role(name: str) -> str:
    lower = str(name).lower()

    if "classifier" in lower:
        return "classifier"
    if "regressor" in lower:
        return "regressor"

    return "unknown"


def _is_primary(name: str) -> bool:
    lower = str(name).lower()
    return any(
        marker.lower() in lower
        for marker in PRIMARY_MARKERS
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as fh:
        for block in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def _feature_names(model: Any) -> list[str]:
    names = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if names is None:
        named_steps = getattr(
            model,
            "named_steps",
            {},
        )

        for name in (
            "imputer",
            "scaler",
            "classifier",
            "regressor",
        ):
            step = named_steps.get(name)

            if step is None:
                continue

            names = getattr(
                step,
                "feature_names_in_",
                None,
            )

            if names is not None:
                break

    if names is None:
        return []

    return [
        str(x)
        for x in names
    ]


def _feature_count(model: Any) -> int | None:
    value = getattr(
        model,
        "n_features_in_",
        None,
    )

    if value is not None:
        try:
            return int(value)
        except Exception:
            pass

    names = _feature_names(model)
    return len(names) if names else None


def _has_imputer(model: Any) -> bool:
    step = getattr(
        model,
        "named_steps",
        {},
    ).get("imputer")

    if step is None:
        return False

    return (
        hasattr(step, "statistics_")
        or hasattr(step, "feature_names_in_")
    )


def _patch_sklearn_compat(model: Any) -> Any:
    try:
        step = getattr(
            model,
            "named_steps",
            {},
        ).get("imputer")

        if (
            step is not None
            and not hasattr(step, "_fill_dtype")
            and hasattr(step, "_fit_dtype")
        ):
            step._fill_dtype = step._fit_dtype

    except Exception:
        pass

    return model


class FullMultiModelRegistry:
    def __init__(
        self,
        drive_store: Any | None = None,
    ):
        self._store = drive_store
        self._records: list[ModelRecord] | None = None
        self._loaded: dict[
            tuple[str, str],
            Any,
        ] = {}

    def _get_store(self):
        if self._store is None:
            from persistent_store import GoogleDriveStore

            self._store = GoogleDriveStore()

        return self._store

    def _models_folder_id(self) -> str:
        store = self._get_store()

        found = store._find_child(
            store.root_id,
            "models",
            "application/vnd.google-apps.folder",
        )

        if not found:
            raise RuntimeError(
                "Google Drive models folder not found."
            )

        return found["id"]

    def _walk(
        self,
        folder_id: str,
        relative_path: str = "",
        depth: int = 0,
    ) -> list[dict[str, str]]:
        if depth > 5:
            return []

        store = self._get_store()
        results = []
        page_token = None

        while True:
            kwargs = {
                "q": (
                    f"'{folder_id}' in parents "
                    "and trashed = false"
                ),
                "spaces": "drive",
                "fields": (
                    "nextPageToken,"
                    "files(id,name,mimeType)"
                ),
                "pageSize": 1000,
                "includeItemsFromAllDrives": True,
                "supportsAllDrives": True,
            }

            if page_token:
                kwargs["pageToken"] = page_token

            response = (
                store.service
                .files()
                .list(**kwargs)
                .execute()
            )

            for item in response.get("files", []):
                name = str(
                    item.get(
                        "name",
                        "",
                    )
                ).strip()

                mime = item.get(
                    "mimeType",
                    "",
                )

                child_path = (
                    f"{relative_path}/{name}".strip("/")
                )

                if mime == (
                    "application/vnd.google-apps.folder"
                ):
                    results.extend(
                        self._walk(
                            item["id"],
                            child_path,
                            depth + 1,
                        )
                    )

                elif name.lower().endswith(".joblib"):
                    results.append(
                        {
                            "id": str(item["id"]),
                            "name": name,
                            "relative_path": child_path,
                        }
                    )

            page_token = response.get(
                "nextPageToken"
            )

            if not page_token:
                break

        return results

    def discover(
        self,
        inspect_contracts: bool = True,
    ) -> list[ModelRecord]:
        if self._records is not None:
            return list(self._records)

        artifacts = self._walk(
            self._models_folder_id()
        )

        records = []

        for item in artifacts:
            name = item["name"]

            if _is_primary(name):
                continue

            role = classify_role(name)
            horizon = classify_horizon(name)
            scope = classify_scope(name)

            if (
                role == "unknown"
                or horizon not in SUPPORTED_HORIZONS
                or scope == "UNKNOWN"
            ):
                continue

            record = ModelRecord(
                name=name,
                drive_id=item["id"],
                relative_path=item[
                    "relative_path"
                ],
                role=role,
                scope=scope,
                horizon=horizon,
                is_classifier=(
                    role == "classifier"
                ),
                is_regressor=(
                    role == "regressor"
                ),
            )

            if inspect_contracts:
                record = self._inspect(
                    record
                )

            records.append(record)

        self._records = sorted(
            records,
            key=lambda r: (
                r.horizon,
                r.scope,
                r.role,
                r.name,
            ),
        )

        return list(self._records)

    def _local_path(
        self,
        record: ModelRecord,
    ) -> Path:
        safe = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            record.name,
        )

        return (
            LOCAL_CACHE_ROOT
            / safe
        )

    def _download(
        self,
        record: ModelRecord,
    ) -> Path:
        from googleapiclient.http import (
            MediaIoBaseDownload,
        )

        store = self._get_store()
        target = self._local_path(
            record
        )

        if (
            target.exists()
            and target.stat().st_size > 0
        ):
            return target

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        request = (
            store.service
            .files()
            .get_media(
                fileId=record.drive_id
            )
        )

        temp = target.with_suffix(
            ".tmp"
        )

        with temp.open("wb") as fh:
            downloader = MediaIoBaseDownload(
                fh,
                request,
            )

            done = False

            while not done:
                _, done = (
                    downloader.next_chunk()
                )

        temp.replace(target)
        return target

    def _inspect(
        self,
        record: ModelRecord,
    ) -> ModelRecord:
        try:
            path = self._download(
                record
            )

            model = _patch_sklearn_compat(
                joblib.load(path)
            )

            return ModelRecord(
                **{
                    **asdict(record),
                    "feature_count": _feature_count(model),
                    "feature_names": tuple(
                        _feature_names(model)
                    ),
                    "sha256": _sha256(path),
                    "metadata_status": "PASS",
                    "metadata_error": None,
                }
            )

        except Exception as exc:
            return ModelRecord(
                **{
                    **asdict(record),
                    "metadata_status": "ERROR",
                    "metadata_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

    def load(
        self,
        record: ModelRecord,
    ):
        key = (
            record.drive_id,
            record.name,
        )

        if key in self._loaded:
            return self._loaded[key]

        model = _patch_sklearn_compat(
            joblib.load(
                self._download(record)
            )
        )

        self._loaded[key] = model
        return model

    @staticmethod
    def eligible(
        record: ModelRecord,
        symbol: str,
        horizon: str,
        feature_frame: pd.DataFrame,
        required_data: dict[str, bool] | None = None,
    ) -> tuple[bool, str]:
        if record.horizon != horizon:
            return False, "horizon_mismatch"

        root = normalize_symbol(symbol)
        required_data = (
            required_data or {}
        )

        if record.scope.startswith(
            "SYMBOL:"
        ):
            trained_symbol = record.scope.split(
                ":",
                1,
            )[1]

            if root != trained_symbol:
                return False, "symbol_scope_mismatch"

        elif record.scope == "FNO_ONLY":
            if not required_data.get(
                "fno",
                False,
            ):
                return False, "fno_data_unavailable"

        elif record.scope not in {
            "GLOBAL",
            "DEPLOYMENT_LEGACY",
        }:
            return False, "unsupported_scope"

        if record.metadata_status != "PASS":
            return False, "artifact_metadata_unavailable"

        if not record.feature_names:
            return False, "feature_contract_unavailable"

        return True, "eligible_for_input_validation"

    def score(
        self,
        symbol: str,
        horizon: str,
        feature_frame: pd.DataFrame,
        required_data: dict[str, bool] | None = None,
        include_regressor: bool = True,
    ) -> dict[str, Any]:
        if horizon not in SUPPORTED_HORIZONS:
            return {
                "status": "ERROR",
                "horizon": horizon,
                "reason": "unsupported_horizon",
                "votes": [],
            }

        records = self.discover(
            inspect_contracts=True
        )

        pairs: dict[
            tuple[str, str],
            dict[str, ModelRecord | None],
        ] = {}

        for record in records:
            key = (
                record.scope,
                record.horizon,
            )

            entry = pairs.setdefault(
                key,
                {
                    "classifier": None,
                    "regressor": None,
                },
            )

            if record.is_classifier:
                entry["classifier"] = record
            elif record.is_regressor:
                entry["regressor"] = record

        votes = []

        for pair in pairs.values():
            classifier = pair[
                "classifier"
            ]

            if classifier is None:
                continue

            eligible, reason = self.eligible(
                classifier,
                symbol,
                horizon,
                feature_frame,
                required_data,
            )

            if not eligible:
                votes.append(
                    {
                        "status": "SKIP",
                        "model": classifier.name,
                        "scope": classifier.scope,
                        "horizon": classifier.horizon,
                        "reason": reason,
                    }
                )
                continue

            model = self.load(
                classifier
            )

            required = list(
                classifier.feature_names
            )

            X = feature_frame.copy()

            for col in required:
                if col not in X.columns:
                    X[col] = np.nan

            X = X[
                required
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            missing_count = int(
                X.isna()
                .sum(axis=1)
                .iloc[0]
            )

            if X.isna().all(
                axis=None
            ):
                votes.append(
                    {
                        "status": "SKIP",
                        "model": classifier.name,
                        "scope": classifier.scope,
                        "horizon": classifier.horizon,
                        "reason": "all_inputs_missing",
                        "missing_inputs": missing_count,
                    }
                )
                continue

            if (
                missing_count > 0
                and not _has_imputer(
                    model
                )
            ):
                votes.append(
                    {
                        "status": "SKIP",
                        "model": classifier.name,
                        "scope": classifier.scope,
                        "horizon": classifier.horizon,
                        "reason": "missing_inputs_without_imputer",
                        "missing_inputs": missing_count,
                    }
                )
                continue

            try:
                p_up = float(
                    model.predict_proba(
                        X
                    )[0, 1]
                )

            except Exception as exc:
                votes.append(
                    {
                        "status": "SKIP",
                        "model": classifier.name,
                        "scope": classifier.scope,
                        "horizon": classifier.horizon,
                        "reason": (
                            "classifier_runtime_error:"
                            f"{type(exc).__name__}"
                        ),
                        "missing_inputs": missing_count,
                    }
                )
                continue

            vote = {
                "status": "PASS",
                "model": classifier.name,
                "scope": classifier.scope,
                "horizon": classifier.horizon,
                "p_up": p_up,
                "direction": (
                    "UP"
                    if p_up >= 0.5
                    else "DOWN"
                ),
                "feature_count": classifier.feature_count,
                "missing_inputs": missing_count,
                "model_sha256": classifier.sha256,
            }

            regressor = pair[
                "regressor"
            ]

            if (
                include_regressor
                and regressor is not None
            ):
                try:
                    reg = self.load(
                        regressor
                    )

                    required_reg = list(
                        regressor.feature_names
                    )

                    RX = feature_frame.copy()

                    for col in required_reg:
                        if col not in RX.columns:
                            RX[col] = np.nan

                    RX = RX[
                        required_reg
                    ].apply(
                        pd.to_numeric,
                        errors="coerce",
                    )

                    reg_missing = int(
                        RX.isna()
                        .sum(axis=1)
                        .iloc[0]
                    )

                    if (
                        reg_missing
                        and not _has_imputer(reg)
                    ):
                        raise RuntimeError(
                            "missing_inputs_without_imputer"
                        )

                    if not RX.isna().all(
                        axis=None
                    ):
                        vote[
                            "expected_return"
                        ] = float(
                            reg.predict(RX)[0]
                        )
                    else:
                        vote[
                            "expected_return"
                        ] = np.nan

                    vote[
                        "regressor_missing_inputs"
                    ] = reg_missing

                except Exception as exc:
                    vote[
                        "regressor_status"
                    ] = (
                        "ERROR:"
                        f"{type(exc).__name__}"
                    )

            votes.append(vote)

        passed = [
            x for x in votes
            if x.get("status") == "PASS"
        ]

        if not passed:
            return {
                "status": "NO_ELIGIBLE_MODELS",
                "horizon": horizon,
                "registered_models": len(records),
                "eligible_models": 0,
                "votes": votes,
            }

        weighted = []

        for vote in passed:
            scope = vote["scope"]

            weight = {
                "GLOBAL": 1.0,
                "FNO_ONLY": 0.75,
                "DEPLOYMENT_LEGACY": 0.75,
            }.get(
                scope,
                1.0,
            )

            if scope.startswith("SYMBOL:"):
                weight = 1.0

            weighted.append(
                (
                    weight,
                    float(vote["p_up"]),
                )
            )

        total = sum(
            weight
            for weight, _
            in weighted
        )

        consensus = (
            sum(
                weight * p
                for weight, p in weighted
            )
            / total
            if total
            else np.nan
        )

        return {
            "status": "PASS",
            "horizon": horizon,
            "registered_models": len(records),
            "eligible_models": len(passed),
            "consensus_p_up": consensus,
            "consensus_direction": (
                "UP"
                if consensus >= 0.5
                else "DOWN"
            ),
            "votes": votes,
        }


def registry_audit(
    registry: FullMultiModelRegistry,
) -> dict[str, Any]:
    records = registry.discover(
        inspect_contracts=True
    )

    by_scope: dict[str, int] = {}
    by_horizon: dict[str, int] = {}

    for record in records:
        by_scope[record.scope] = (
            by_scope.get(record.scope, 0)
            + 1
        )

        by_horizon[record.horizon] = (
            by_horizon.get(record.horizon, 0)
            + 1
        )

    return {
        "status": "PASS",
        "registered_auxiliary_models": len(records),
        "by_scope": by_scope,
        "by_horizon": by_horizon,
        "artifacts": [
            asdict(x)
            for x in records
        ],
    }
