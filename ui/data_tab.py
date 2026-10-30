"""Data tab: upload/load a file, then review and edit column roles."""
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from engine.config import get_settings
from engine.loader import file_hash, list_sheets, load_table
from engine.profile import profile_frame
from engine.roles import ROLE_HELP, ROLES, check_role, detect_roles, entity_attributes

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECK_ICON = {"ok": "✅ ok", "warn": "⚠️ warn", "fail": "❌ fail"}


@st.cache_data(max_entries=4, show_spinner=False)
def _sheets(data: bytes, name: str) -> list[str]:
    return list_sheets(data, name)


@st.cache_data(max_entries=4, show_spinner=False)
def _load(data: bytes, name: str, sheet, header_row):
    t = time.perf_counter()
    df, info = load_table(data, name, sheet, header_row)
    return df, info, time.perf_counter() - t


def _pick_source() -> tuple[bytes, str] | None:
    samples = sorted(p.name for p in DATA_DIR.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xls"))
    options = ["Upload a file"] + (["Use a file from the data folder"] if samples else [])
    src = st.radio("Data source", options, horizontal=True, label_visibility="collapsed")
    if src == "Upload a file":
        up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xlsm", "xls"])
        return (up.getvalue(), up.name) if up else None
    name = st.selectbox("File in data folder", samples)
    return (DATA_DIR / name).read_bytes(), name


def _reset_for(dataset_id: str, df: pd.DataFrame):
    st.session_state.dataset_id = dataset_id
    st.session_state.profile = profile_frame(df)
    st.session_state.roles = None
    st.session_state.roles_version = 0
    st.session_state.fixes = []
    st.session_state.qa_history = []


def _apply_edits(edited: pd.DataFrame):
    roles = st.session_state.roles
    changed = False
    for i, new in edited["role"].items():
        if new != roles.at[i, "role"]:
            col = roles.at[i, "column"]
            status, msg = check_role(new, st.session_state.profile[col], col)
            roles.loc[i, ["role", "source", "check", "note"]] = [new, "user", status, msg or "set by user"]
            changed = True
    if changed:
        st.session_state.roles_version += 1
        st.rerun()


def render():
    picked = _pick_source()
    if not picked:
        st.info("Upload a CSV or Excel file to begin.")
        return
    data, name = picked

    c1, c2 = st.columns(2)
    sheets = _sheets(data, name)
    sheet = c1.selectbox("Sheet", sheets) if len(sheets) > 1 else (sheets[0] if sheets else None)
    with st.spinner("Loading file..."):
        df_auto, info_auto, _ = _load(data, name, sheet, None)
    header = c2.number_input(f"Header row (auto-detected: {info_auto.header_row_auto + 1})",
                             min_value=1, value=info_auto.header_row_auto + 1, step=1)
    with st.spinner("Loading file..."):
        df, info, secs = _load(data, name, sheet, int(header) - 1)

    dataset_id = f"{file_hash(data)}:{sheet}:{info.header_row}"
    if st.session_state.get("dataset_id") != dataset_id:
        _reset_for(dataset_id, df)
    st.session_state.df = df
    st.session_state.load_info = info

    m = st.columns(4)
    m[0].metric("Rows", f"{info.n_rows:,}")
    m[1].metric("Columns", info.n_cols)
    m[2].metric("Sheet", info.sheet or "CSV")
    m[3].metric("Load time", f"{secs:.1f}s")
    for note in info.notes:
        st.caption(note)
    if info.dropped_empty_rows or info.dropped_empty_cols:
        st.caption(f"Removed {info.dropped_empty_rows} fully empty row(s) and "
                   f"{info.dropped_empty_cols} fully empty column(s).")
    with st.expander(f"Preview (first {min(500, len(df))} of {len(df):,} rows)", expanded=False):
        head = df.head(500)
        mixed = [c for c in head.columns if head[c].dtype == object]  # e.g. numbers mixed with text
        st.dataframe(head.astype({c: "string" for c in mixed}), width="stretch")

    _roles_section(df)


def _roles_section(df: pd.DataFrame):
    st.subheader("Column roles")
    ai_on = get_settings().ai_enabled
    rerun = st.button("Re-detect roles")
    if st.session_state.roles is None or rerun:
        with st.spinner("Detecting column roles..." + (" (asking AI)" if ai_on else "")):
            roles, msg = detect_roles(df, st.session_state.dataset_id, use_ai=ai_on,
                                      prof=st.session_state.profile)
        st.session_state.roles, st.session_state.roles_msg = roles, msg
        st.session_state.roles_version += 1
    roles = st.session_state.roles
    if not ai_on:
        st.warning("AI is off, so roles come from rules only. You can still edit them below.")
    elif st.session_state.get("roles_msg"):
        st.warning(st.session_state.roles_msg)

    st.caption("Change any **role** in the table and the app uses your choice. "
               "**Check** shows whether the data supports the role. "
               "**Source**: AI = model label confirmed by the data; rules = code-only guess; user = your override.")
    view = roles.assign(check=roles["check"].map(CHECK_ICON))
    edited = st.data_editor(
        view, key=f"roles_{st.session_state.dataset_id}_{st.session_state.roles_version}",
        hide_index=True, width="stretch", disabled=[c for c in view.columns if c != "role"],
        column_config={"role": st.column_config.SelectboxColumn(
            "role", options=ROLES, required=True,
            help="; ".join(f"{r} = {h}" for r, h in ROLE_HELP.items()))},
    )
    _apply_edits(edited)

    bad = roles[roles["check"] == "fail"]
    if len(bad):
        st.error(f"{len(bad)} role(s) are not supported by the data: {', '.join(bad['column'])}. "
                 "Calculations will skip these until fixed.")
    counts = roles["role"].value_counts()
    st.caption("Roles found: " + ", ".join(f"{r} × {n}" for r, n in counts.items()))

    st.subheader("Entity attributes")
    attrs = entity_attributes(df, roles)
    st.session_state.entity_attrs = attrs
    if attrs.empty:
        st.caption("No columns found that stay fixed per group (e.g. a capacity repeated for each site).")
    else:
        st.caption("These columns repeat one value per group, so they describe the group, not each row. "
                   "Later phases will not sum them across rows.")
        st.dataframe(attrs, hide_index=True, width="stretch")
