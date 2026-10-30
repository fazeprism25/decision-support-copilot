"""Working-data pipeline shared by all tabs: raw upload → quality fixes → confirmed value mapping (→ scenario)."""
import json

import pandas as pd
import streamlit as st

from engine.quality import apply_fixes, find_issues
from engine.valuemap import apply_value_map


def has_data() -> bool:
    return st.session_state.get("df") is not None and st.session_state.get("roles") is not None


def entity_cols() -> set[str]:
    attrs = st.session_state.get("entity_attrs")
    return set(attrs["attribute"]) if attrs is not None and len(attrs) else set()


def roles_sig() -> tuple:
    r = st.session_state.roles
    return tuple(zip(r["column"], r["role"], r["check"]))


def fixes_sig() -> tuple:
    return tuple((f.key, f.action) for f in st.session_state.get("fixes", []))


def vmap() -> dict:
    """The confirmed C01 value mapping (empty until the user confirms it on the Data tab)."""
    return st.session_state.get("vmap") or {}


def vmap_sig() -> str:
    return json.dumps(vmap(), sort_keys=True)


def data_sig() -> tuple:
    return (st.session_state.dataset_id, fixes_sig(), vmap_sig())


def fixed_df() -> pd.DataFrame:
    """Raw data with the applied quality fixes only (used to propose the value mapping)."""
    key = (st.session_state.dataset_id, fixes_sig())
    memo = st.session_state.get("_fixed_memo")
    if not memo or memo[0] != key:
        st.session_state._fixed_memo = memo = (key, apply_fixes(st.session_state.df, st.session_state.get("fixes", [])))
    return memo[1]


def clean_df() -> pd.DataFrame:
    """Raw data → applied quality fixes → confirmed value mapping, memoised. Every tab uses this frame."""
    key = data_sig()
    memo = st.session_state.get("_clean_memo")
    if not memo or memo[0] != key:
        st.session_state._clean_memo = memo = (key, apply_value_map(fixed_df(), vmap()))
    return memo[1]


def data_hash() -> str:
    """Content hash of the working data (for answer caches)."""
    key = data_sig()
    memo = st.session_state.get("_hash_memo")
    if not memo or memo[0] != key:
        h = str(int(pd.util.hash_pandas_object(clean_df().astype("string"), index=False).sum()))
        st.session_state._hash_memo = memo = (key, h)
    return memo[1]


def issues() -> list:
    key = (data_sig(), roles_sig())
    memo = st.session_state.get("_issues_memo")
    if not memo or memo[0] != key:
        st.session_state._issues_memo = memo = (key, find_issues(clean_df(), st.session_state.roles, entity_cols()))
    return memo[1]


def role_map(roles: tuple = ("*",)) -> dict[str, str]:
    """Column → role for columns whose role passed the data check."""
    r = st.session_state.roles
    return {c: ro for c, ro, chk in zip(r["column"], r["role"], r["check"])
            if chk != "fail" and ro != "ignore" and ("*" in roles or ro in roles)}


def meanings() -> dict[str, str]:
    r = st.session_state.roles
    return {c: m for c, m in zip(r["column"], r["meaning"]) if m}


def kpi_ctx():
    from engine.kpis import Ctx

    attrs = st.session_state.get("entity_attrs")
    ent = dict(zip(attrs["attribute"], attrs["entity"])) if attrs is not None and len(attrs) else {}
    return Ctx(roles=role_map(), entity_of=ent)


def pack():
    """Delivery-ops domain pack for the current data (on only when the needed roles exist)."""
    from engine.domains import delivery as D
    from ui import whatif_tab

    ctx = kpi_ctx()
    key = (data_sig(), roles_sig())
    memo = st.session_state.get("_pack_memo")
    if not memo or memo[0] != key:
        pk = D.detect(clean_df(), ctx.roles, ctx.entity_of, whatif_tab.setup())
        st.session_state._pack_memo = memo = (key, pk)
    return memo[1]


def pack_metrics(df: pd.DataFrame | None = None):
    from engine.domains import delivery as D

    pk = pack()
    if not pk.active:
        return None
    ctx = kpi_ctx()
    return D.metrics(clean_df() if df is None else df, pk, ctx.entity_of, ctx.roles)


def cleaning_summary() -> str:
    raw, clean = st.session_state.df, clean_df()
    fixes = st.session_state.get("fixes", [])
    return (f"Running on cleaned data: {len(clean):,} rows, {len(raw) - len(clean):,} excluded "
            f"({len(fixes)} quality fix(es) applied) · the same data the KPIs use")
