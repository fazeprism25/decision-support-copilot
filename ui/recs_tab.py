"""Recommendations tab: rule-generated, ranked by impact, evidence before vs after the active scenario."""
import hashlib

import streamlit as st

from engine import llm
from engine import recommend as R
from engine.quality import quality_score
from engine.scenario import apply_rows
from ui import state, whatif_tab

CONF_ICON = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}
STATUS_ICON = {"new in scenario": "🆕 new in scenario", "resolved in scenario": "✅ resolved in scenario",
               "changed": "↕️ changed by scenario", "unchanged": "unchanged by scenario"}


def compute() -> tuple[list[R.Rec], list[R.Rec], list[dict], object, dict]:
    df, ctx = state.clean_df(), state.kpi_ctx()
    params = whatif_tab.current_params()
    res = whatif_tab.run_scenario(params)
    scen_df, _ = apply_rows(df, params)
    issues = state.issues()
    q, _ = quality_score(df, issues)
    pk = state.pack()
    days = R.date_span(df, ctx.roles)
    common = dict(ctx=ctx, issues=issues, quality_score=q, pack=pk, days=days)
    base = R.generate(R.Inputs(df, kpis=res["base"], pack_metrics=state.pack_metrics(), **common))
    scen = R.generate(R.Inputs(scen_df, kpis=res["scen"], pack_metrics=state.pack_metrics(scen_df),
                               cascade=res["cascade"], is_scenario=not params.is_empty(), **common))
    return base, scen, R.compare(base, scen), params, res


def render():
    base, scen, rows, params, res = compute()
    st.session_state.last_recs = scen
    n_c = len(st.session_state.get("constraints", []))
    st.caption("Each recommendation comes from a rule over computed numbers (see Explain tab). Ranked by impact. "
               + (f"Comparing baseline with the current scenario ({n_c} constraint(s) + sliders)."
                  if not params.is_empty() else "No scenario active: before = after. Add a constraint on What-if."))
    ok, why = llm.status()
    if ok and scen:
        key = hashlib.sha1((state.data_hash() + params.model_dump_json()).encode()).hexdigest()[:16]
        with st.spinner("AI is rewording recommendations in plain language..."):
            msg = R.rewrite(scen, key)
        st.caption(msg)
    elif scen:
        st.caption(f"Showing rule wording (AI-down mode: {why}).")
    if not rows:
        st.success("No rule fired: nothing stands out in the computed numbers.")
    for i, c in enumerate(rows, 1):
        r = c["rec"]
        with st.container(border=True):
            st.markdown(f"**{i}. {r.title}**")
            st.markdown(r.plain or r.action)
            if r.plain:
                st.caption(f"Rule text: {r.action}")
            st.caption(f"Confidence {CONF_ICON[r.confidence]} · impact score {r.impact:.2f} ({r.impact_formula}) · "
                       f"{STATUS_ICON[c['status']]}" + (f" · {r.rewrite_note}" if r.rewrite_note else ""))
            with st.expander("Evidence: numbers, formulas, before vs after scenario"):
                st.dataframe(c["table"], hide_index=True, width="stretch")
                if r.assumptions:
                    st.markdown("**Assumptions**\n" + "\n".join(f"- {a}" for a in r.assumptions))
                if r.limitations:
                    st.markdown("**Limitations**\n" + "\n".join(f"- {x}" for x in r.limitations))
