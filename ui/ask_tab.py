"""Ask tab: plain-English questions answered by executed pandas code. Answers are cached per (question, data);
six example questions are answered in the background as soon as a file loads, so clicking one is instant."""
import time

import pandas as pd
import streamlit as st

from engine import llm
from engine.kpis import fmt_value
from engine.qa import cached, cached_ask, example_questions, pending, prefetch, schema_text
from ui import state, style
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
        notes += "\nKPI definitions used on the KPI page (use the same when relevant):\n- " + "\n- ".join(defs)
    return df, schema_text(df, state.role_map(), state.meanings(), st.session_state.profile), notes


def _ask(q: str):
    df, schema, notes = _context()
    t = time.perf_counter()
    with st.spinner("Writing and running code..."):
        a, hit = cached_ask(q, df, schema, notes, state.data_hash())
    a.data_note = state.cleaning_summary()
    a.timing = "from cache (same question, same data)" if hit else f"answered in {time.perf_counter() - t:.1f}s"
    st.session_state.qa_history.insert(0, a)


def ensure_prefetch():
    """Start answering the example questions in the background once per dataset, whichever page is open."""
    ctx, h = state.kpi_ctx(), state.data_hash()
    if llm.status()[0] and st.session_state.get("prefetched_for") != h:
        examples = example_questions(state.clean_df(), ctx.roles, ctx.entity_of, state.pack())
        df, schema, notes = _context()
        prefetch(examples, df, schema, notes, h)
        st.session_state.prefetched_for = h


def render():
    st.session_state.setdefault("qa_history", [])
    ok, why = llm.status()
    ctx = state.kpi_ctx()
    h = state.data_hash()
    examples = example_questions(state.clean_df(), ctx.roles, ctx.entity_of, state.pack())
    ensure_prefetch()
    if not ok:
        st.warning(f"Q&A needs AI for new questions. {why} Cached answers below still work; all other pages work.")

    with st.form("ask", clear_on_submit=True, border=True):
        c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
        q = c1.text_input("Your question", placeholder="e.g. Which hub has the lowest on-time rate?")
        go = c2.form_submit_button("Ask", disabled=not ok, type="primary", width="stretch")
    if go and q.strip():
        _ask(q.strip())

    ready = [cached(q, h) is not None for q in examples]
    busy = pending(examples, h)
    style.label("Example questions" + (f" (preparing {busy} in the background…)" if busy else ""),
                "Built from this file's own columns and answered in the background as soon as the file loads. "
                "Questions marked *ready* are already answered, so clicking one is instant.")
    cols = st.columns(3)
    for i, q in enumerate(examples):
        if cols[i % 3].button(q + (" (ready)" if ready[i] else ""), key=f"ex_{i}", disabled=not (ok or ready[i]),
                              width="stretch"):
            _ask(q)
    if st.session_state.qa_history:
        st.divider()
    for n, a in enumerate(st.session_state.qa_history):
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"#### {a.question}", width="content")
                style.info((getattr(a, "data_note", "") or "") + (f"\n\n{a.timing}" if getattr(a, "timing", "") else ""))
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
