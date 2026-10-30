"""Ask tab: plain-English questions answered by executed pandas code."""
import pandas as pd
import streamlit as st

from engine.config import get_settings
from engine.kpis import fmt_value
from engine.qa import ask, auto_chart, schema_text
from ui import state
from ui.charts import figure

EXAMPLES = "e.g. 'Which group has the highest average of X?', 'How many rows are missing Y?', 'Show the total per week'"


def render():
    st.session_state.setdefault("qa_history", [])
    if not get_settings().ai_enabled:
        st.warning("Q&A needs AI, which is off (no LLM_API_KEY in .env). All other tabs still work.")
        return
    st.caption("The AI writes pandas code, the app runs it on your data in a locked-down sandbox, and every answer "
               "comes from that executed code. The code is shown under each answer. " + EXAMPLES)
    with st.form("ask", clear_on_submit=True):
        q = st.text_input("Your question")
        go = st.form_submit_button("Ask")
    if go and q.strip():
        df = state.clean_df()
        notes = (f"Note: {len(st.session_state.fixes)} data-quality fix(es) have already been applied to df.\n"
                 if st.session_state.get("fixes") else "")
        open_issues = [i.detail for i in state.issues() if i.severity != "low"]
        if open_issues:  # so answers use the same validity rules as the KPIs
            notes += ("Known data-quality issues still in df (exclude invalid rows, e.g. repeated IDs, zero durations, "
                      "out-of-scale values, and say so in `method`):\n- " + "\n- ".join(open_issues[:12]))
        schema = schema_text(df, state.role_map(), state.meanings(), st.session_state.profile)
        defs = [f"{k.title}: {k.formula}; filter {k.filter}" for k in st.session_state.get("kpi_results", [])
                if k.calc in ("on_time_rate", "margin", "share")]
        if defs:  # same definitions as the KPI cards
            notes += "\nKPI definitions used on the KPI tab (use the same when relevant):\n- " + "\n- ".join(defs)
        with st.spinner("Writing and running code..."):
            a = ask(q.strip(), df, schema, notes)
            a.data_note = state.cleaning_summary()
            st.session_state.qa_history.insert(0, a)

    for n, a in enumerate(st.session_state.qa_history):
        with st.container(border=True):
            st.markdown(f"**Q: {a.question}**")
            st.caption(getattr(a, "data_note", ""))
            if a.error:
                st.error(f"Could not answer after {a.attempts} attempt(s): {a.error}")
            else:
                st.markdown(a.narration)
                for w in a.warnings:
                    st.warning(w)
                _show_result(a, n)
                if a.method:
                    st.caption(f"Method: {a.method} · calculated by executed code")
            if a.code:
                with st.expander("Code that produced this answer" + (" (retried once)" if a.attempts > 1 else "")):
                    st.code(a.code, language="python")


def _show_result(a, n: int):
    r = a.result
    if r is None:
        return
    if isinstance(r, (pd.DataFrame, pd.Series)):
        table = r if isinstance(r, pd.DataFrame) else r.rename(r.name or "value").reset_index()
        chart = a.chart
        if not chart:
            chart, table2 = auto_chart(r)
            table = table2 if table2 is not None else table
        if chart:
            st.plotly_chart(figure(chart, table), width="stretch", key=f"qa_{n}")
        st.dataframe(table, hide_index=True, width="stretch")
    elif isinstance(r, (int, float)):
        st.metric("Result", fmt_value(r, "number"))
    else:
        st.code(str(r)[:2000])
