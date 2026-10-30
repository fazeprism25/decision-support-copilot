"""Ask tab: plain-English questions answered by executed pandas code. Answers are cached per (question, data);
six example questions are answered in the background as soon as a file loads, so clicking one is instant."""
import time

import pandas as pd
import streamlit as st

from engine import llm
from engine.kpis import fmt_value
from engine.qa import cached, cached_ask, example_questions, pending, prefetch, schema_text
from ui import state
from ui.charts import figure


def _context() -> tuple[pd.DataFrame, str, str]:
    df = state.clean_df()
    notes = (f"Note: {len(st.session_state.fixes)} data-quality fix(es) have already been applied to df.\n"
             if st.session_state.get("fixes") else "")
    if state.vmap():
        notes += "Category labels in df are already standardised (value mapping applied); use the labels listed.\n"
    open_issues = [i.detail for i in state.issues() if i.severity != "low"]
    if open_issues:  # so answers use the same validity rules as the KPIs
        notes += ("Known data-quality issues still in df (exclude invalid rows, e.g. repeated IDs, zero durations, "
                  "out-of-scale values, and say so in `method`):\n- " + "\n- ".join(open_issues[:12]))
    defs = [f"{k.title}: {k.formula}; filter {k.filter}" for k in st.session_state.get("kpi_results", [])
            if k.calc in ("on_time_rate", "margin", "share")]
    if defs:  # same definitions as the KPI cards
        notes += "\nKPI definitions used on the KPI tab (use the same when relevant):\n- " + "\n- ".join(defs)
    return df, schema_text(df, state.role_map(), state.meanings(), st.session_state.profile), notes


def _ask(q: str):
    df, schema, notes = _context()
    t = time.perf_counter()
    with st.spinner("Writing and running code..."):
        a, hit = cached_ask(q, df, schema, notes, state.data_hash())
    a.data_note = state.cleaning_summary()
    a.timing = "⚡ from cache (same question, same data)" if hit else f"answered in {time.perf_counter() - t:.1f}s"
    st.session_state.qa_history.insert(0, a)


def render():
    st.session_state.setdefault("qa_history", [])
    ok, why = llm.status()
    ctx = state.kpi_ctx()
    h = state.data_hash()
    examples = example_questions(state.clean_df(), ctx.roles, ctx.entity_of, state.pack())
    if ok and st.session_state.get("prefetched_for") != h:
        df, schema, notes = _context()
        prefetch(examples, df, schema, notes, h)
        st.session_state.prefetched_for = h
    if not ok:
        st.warning(f"Q&A needs AI for new questions. {why} Cached answers below still work; all other tabs work.")
    st.caption("The AI writes pandas code, the app runs it on your data in a locked-down sandbox, and every answer "
               "comes from that executed code. The code is shown under each answer.")

    ready = [cached(q, h) is not None for q in examples]
    busy = pending(examples, h)
    st.markdown("**Example questions** " + (f"(preparing {busy} in the background…)" if busy else ""))
    cols = st.columns(3)
    for i, q in enumerate(examples):
        if cols[i % 3].button(("⚡ " if ready[i] else "") + q, key=f"ex_{i}", disabled=not (ok or ready[i]),
                              width="stretch"):
            _ask(q)
    with st.form("ask", clear_on_submit=True):
        q = st.text_input("Your question")
        go = st.form_submit_button("Ask", disabled=not ok)
    if go and q.strip():
        _ask(q.strip())

    for n, a in enumerate(st.session_state.qa_history):
        with st.container(border=True):
            st.markdown(f"**Q: {a.question}**")
            st.caption(getattr(a, "data_note", "") + (f" · {a.timing}" if getattr(a, "timing", "") else ""))
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
            from engine.qa import auto_chart

            chart, table2 = auto_chart(r)
            table = table2 if table2 is not None else table
        if chart:
            st.plotly_chart(figure(chart, table), width="stretch", key=f"qa_{n}")
        st.dataframe(table, hide_index=True, width="stretch")
    elif isinstance(r, (int, float)):
        st.metric("Result", fmt_value(r, "number"))
    else:
        st.code(str(r)[:2000])
