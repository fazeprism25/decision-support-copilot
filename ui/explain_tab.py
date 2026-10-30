"""Explain tab: every formula, assumption, exclusion and limitation from what the app ran on this dataset."""
import streamlit as st

from engine import explain, llm
from engine.quality import quality_score
from engine.recommend import date_span
from ui import state, whatif_tab


def render():
    df, ctx = state.clean_df(), state.kpi_ctx()
    params = whatif_tab.current_params()
    res = whatif_tab.run_scenario(params)
    issues = state.issues()
    q, _ = quality_score(df, issues)
    ok, msg = llm.status()
    roles_msg = st.session_state.get("roles_msg") or ""
    secs = explain.build(
        name=st.session_state.get("file_name", "uploaded file"), rows_raw=len(st.session_state.df), rows_clean=len(df),
        roles=st.session_state.roles, entity_of=ctx.entity_of, fixes=st.session_state.get("fixes", []),
        vmap=(st.session_state.get("vmap") or {}) or _proposed(), vmap_confirmed=bool(state.vmap()),
        kpis=st.session_state.get("kpi_results", []), pack=state.pack(), pack_metrics=state.pack_metrics(),
        params=params, notes=res["notes"], cascade=res["cascade"], recs=st.session_state.get("last_recs", []),
        issues=issues, quality_score=q, days=date_span(df, ctx.roles),
        ai_status=("AI on" if ok else msg) + (f"; roles: {roles_msg}" if roles_msg else ""), rows_scenario=res["rows"][1],
        route=res["route"])
    st.caption("Generated from what the app actually ran on the current data, fixes, mapping and scenario.")
    for s in secs:
        with st.expander(s.title, expanded=s.title.startswith(("1.", "8."))):
            for line in s.lines:
                st.markdown(line)
            if s.table is not None and len(s.table):
                st.dataframe(s.table.astype(str), hide_index=True, width="stretch")


def _proposed() -> dict:
    from engine.valuemap import from_table

    t = st.session_state.get("vmap_prop")
    return from_table(t) if t is not None and len(t) else {}
