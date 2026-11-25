"""Streamlit page: to-close champion predictions for the authoritative 500 symbols.

Open with ``streamlit run app.py`` (auto-discovered under pages/) or
``streamlit run pages/1_Champion_Predictions.py``.
"""
from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd
import streamlit as st

from persistent_store import GoogleDriveStore
from prediction_universe import authoritative_roster

PREDICTION_ROOT = ["production_data", "all_nse_minute_predictions"]
CHAMPION_ROOT = "production_data/all_nse_champion"
FOLDER_MIME = "application/vnd.google-apps.folder"


@st.cache_data(ttl=300, show_spinner=False)
def _drive_store() -> GoogleDriveStore:
    return GoogleDriveStore()


def _store_or_none() -> GoogleDriveStore | None:
    try:
        return _drive_store()
    except Exception:
        return None


def _folder_id(store: GoogleDriveStore, parts: list[str]) -> str | None:
    parent = store.root_id
    for part in parts:
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            return None
        parent = hit["id"]
    return parent


def _children(store: GoogleDriveStore, parent: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_token = None
    while True:
        params = {
            "q": f"'{parent}' in parents and trashed = false",
            "spaces": "drive",
            "fields": "nextPageToken,files(id,name,mimeType,modifiedTime)",
            "pageSize": 1000,
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
        }
        if page_token:
            params["pageToken"] = page_token
        response = store.service.files().list(**params).execute()
        items.extend(response.get("files") or [])
        page_token = response.get("nextPageToken")
        if not page_token:
            return items


def _read_bytes(store: GoogleDriveStore, file_id: str) -> bytes:
    raw = store.service.files().get_media(fileId=file_id).execute()
    return raw.read() if hasattr(raw, "read") else bytes(raw)


def _latest_session_folder(store: GoogleDriveStore) -> str | None:
    root = _folder_id(store, PREDICTION_ROOT)
    if not root:
        return None
    for depth in range(3):
        folders = [f for f in _children(store, root) if f.get("mimeType") == FOLDER_MIME]
        folders = [f for f in folders if str(f.get("name") or "").isdigit()]
        if not folders:
            return None
        parent = sorted(folders, key=lambda f: str(f.get("name")))[-1]["id"]
        root = parent
    return root


@st.cache_data(ttl=300, show_spinner=False)
def _load_predictions(slots: int, refresh_revision: int = 0) -> pd.DataFrame:
    del refresh_revision
    store = _store_or_none()
    if store is None:
        return pd.DataFrame()
    session_folder = _latest_session_folder(store)
    if not session_folder:
        return pd.DataFrame()
    roster = set(authoritative_roster())
    files = [
        f for f in _children(store, session_folder)
        if str(f.get("name") or "").startswith("predictions_")
        and str(f.get("name") or "").endswith(".parquet")
    ]
    files = sorted(files, key=lambda f: str(f.get("name")), reverse=True)[: max(1, int(slots))]
    frames: list[pd.DataFrame] = []
    for item in files:
        name = str(item.get("name") or "")
        try:
            frame = pd.read_parquet(io.BytesIO(_read_bytes(store, item["id"])))
        except Exception:
            continue
        if frame.empty or "symbol" not in frame.columns:
            continue
        frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper().str.replace(".NS", "", regex=False)
        frame = frame.loc[frame["symbol"].isin(roster)].copy()
        frame["slot"] = name[len("predictions_"):-len(".parquet")]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(["slot", "symbol"], keep="last")
    for column in ("expected_return_session_close", "probability_up_session_close", "data_readiness_score"):
        if column in combined:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


@st.cache_data(ttl=300, show_spinner=False)
def _load_champion(refresh_revision: int = 0) -> dict:
    del refresh_revision
    store = _store_or_none()
    if store is None:
        return {}
    parts = CHAMPION_ROOT.strip("/").split("/")
    parent = store.root_id
    for part in parts:
        hit = store._find_child(parent, part, FOLDER_MIME)
        if not hit:
            return {}
        parent = hit["id"]
    pointer = next((f for f in _children(store, parent) if f.get("name") == "champion.json"), None)
    if not pointer:
        return {}
    try:
        return json.loads(_read_bytes(store, pointer["id"]).decode("utf-8"))
    except Exception:
        return {}


st.set_page_config(page_title="Champion predictions", layout="wide")
st.title("To-close champion · predictions")
st.caption(
    "Nightly-trained to-close model applied to the authoritative 500-symbol roster. "
    "Signals are emitted only when expected return clears the configured cost edge."
)

refresh_col, slots_col = st.columns([1, 3])
with refresh_col:
    if st.button("Refresh", key="champion_prediction_refresh"):
        _load_predictions.clear()
        _load_champion.clear()
        st.rerun()
with slots_col:
    slots = st.slider("Five-minute slots to load", 1, 24, 6)

champion = _load_champion()
if champion:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Version", str(champion.get("version") or "—"))
    c2.metric("Trained through", str(champion.get("trained_through") or "—"))
    c3.metric("Pass streak", str(champion.get("pass_streak") or "—"))
    c4.metric("Serving", "ACTIVE" if champion.get("active") else "HOLD")
    with st.expander("Champion metadata", expanded=False):
        st.json(champion)
else:
    st.info("No champion pointer found in Drive yet.")

with st.spinner("Reading the latest predictions from Drive…"):
    predictions = _load_predictions(slots)

if predictions.empty:
    st.info("No to-close prediction rows are available in Drive yet.")
    st.stop()

latest_slot = sorted(predictions["slot"].astype(str).unique())[-1]
latest = predictions.loc[predictions["slot"] == latest_slot].copy()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest slot", latest_slot)
m2.metric("Symbols predicted", f"{len(latest):,}")
m3.metric("BUY signals", int((latest.get("signal", pd.Series(dtype=str)) == "BUY").sum()))
m4.metric("SELL signals", int((latest.get("signal", pd.Series(dtype=str)) == "SELL").sum()))

roster_count = len(authoritative_roster())
if roster_count and len(latest) < roster_count:
    st.warning(f"{roster_count - len(latest):,} of {roster_count:,} roster symbols have no row in this slot.")

search = st.text_input("Find symbol", key="champion_pred_search").strip().upper()
signal_filter = st.selectbox(
    "Signal", ["All", "BUY", "SELL", "HOLD"], key="champion_signal_filter"
)
serving_filter = st.selectbox(
    "Serving mode", ["All"] + sorted(predictions["model_serving_mode"].dropna().astype(str).unique().tolist())
    if "model_serving_mode" in predictions else ["All"],
    key="champion_serving_filter",
)
view = latest.copy()
if search:
    view = view.loc[view["symbol"].astype(str).str.contains(search, regex=False)]
if signal_filter != "All" and "signal" in view:
    view = view.loc[view["signal"] == signal_filter]
if serving_filter != "All" and "model_serving_mode" in view:
    view = view.loc[view["model_serving_mode"] == serving_filter]

columns = [
    "symbol", "signal", "probability_up_session_close", "expected_return_session_close",
    "confidence", "prediction_status", "model_serving_mode", "model_target",
    "model_coverage_symbols", "data_readiness_score", "signal_min_edge",
    "forecast_target_timestamp_utc", "model_version",
]
# Filter to columns that exist in the frame and de-duplicate while preserving
# order. A duplicated name here used to produce a duplicate-column frame that
# pyarrow rejected with ``Duplicate column names found`` when rendering the
# dataframe.
columns = list(dict.fromkeys(c for c in columns if c in view.columns))
if "expected_return_session_close" in columns:
    view = view.sort_values("expected_return_session_close", ascending=False)
st.dataframe(view[columns], width="stretch", hide_index=True, height=640)

if "expected_return_session_close" in predictions and "symbol" in predictions:
    st.subheader("Expected return by slot")
    chart_symbols = sorted(predictions["symbol"].unique().tolist())
    default_symbols = [s for s in ["RELIANCE", "INFY", "TATASTEEL", "SBIN"] if s in chart_symbols]
    selected = st.multiselect("Symbols", chart_symbols, default=default_symbols, key="champion_chart_symbols")
    if selected:
        chart = (
            predictions.loc[predictions["symbol"].isin(selected)]
            .sort_values("slot")
            .pivot_table(index="slot", columns="symbol", values="expected_return_session_close")
        )
        st.line_chart(chart * 100)
        st.caption("Expected to-close return in percent, by five-minute decision slot.")

with st.expander("Prediction slot metadata", expanded=False):
    st.json({
        "slot": latest_slot,
        "rows": int(len(latest)),
        "status_counts": latest.get("prediction_status", pd.Series(dtype=str)).value_counts().to_dict(),
        "signal_counts": latest.get("signal", pd.Series(dtype=str)).value_counts().to_dict(),
        "serving_modes": latest.get("model_serving_mode", pd.Series(dtype=str)).value_counts().to_dict(),
    })
