"""Working-data pipeline shared by all tabs: raw upload → quality fixes (→ scenario, Phase 6)."""
import pandas as pd
import streamlit as st

from engine.quality import apply_fixes, find_issues


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


def clean_df() -> pd.DataFrame:
    """Raw data with all applied quality fixes, memoised per (dataset, fixes)."""
    key = (st.session_state.dataset_id, fixes_sig())
    memo = st.session_state.get("_clean_memo")
    if not memo or memo[0] != key:
        df = apply_fixes(st.session_state.df, st.session_state.get("fixes", []))
        st.session_state._clean_memo = memo = (key, df)
    return memo[1]


def issues() -> list:
    key = (st.session_state.dataset_id, fixes_sig(), roles_sig())
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


def cleaning_summary() -> str:
    raw, clean = st.session_state.df, clean_df()
    fixes = st.session_state.get("fixes", [])
    return (f"Running on cleaned data: {len(clean):,} rows, {len(raw) - len(clean):,} excluded "
            f"({len(fixes)} quality fix(es) applied) · the same data the KPIs use")
