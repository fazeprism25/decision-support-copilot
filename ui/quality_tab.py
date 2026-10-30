"""Quality tab: score, issue list, one-click fixes with before/after counts."""
import pandas as pd
import streamlit as st

from engine.quality import FIX_LABELS, Fix, quality_score
from ui import state

RULE_NAMES = {
    "duplicate_id": "Duplicate ID", "duplicate_record": "Duplicate record", "missing": "Missing values", "not_numeric": "Not a number",
    "negative": "Negative value", "zero_duration": "Zero duration", "out_of_scale": "Rating outside scale",
    "outlier": "Extreme value", "bad_date": "Unreadable date", "inconsistent_text": "Inconsistent spelling",
}
SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}


def render():
    st.session_state.setdefault("fixes", [])
    df = state.clean_df()
    found = state.issues()
    score, clean = quality_score(df, found)

    m = st.columns(4)
    m[0].metric("Quality score", f"{score}%")
    m[1].metric("Rows with no issue", f"{clean:,} / {len(df):,}")
    m[2].metric("Issues found", len(found))
    m[3].metric("Fixes applied", len(st.session_state.fixes))
    with st.popover("How is the score calculated?"):
        st.markdown(
            f"**Quality score = rows with no detected issue ÷ all rows** = {clean:,} ÷ {len(df):,} = **{score}%** "
            "(calculated).\n\nExtreme values (outliers) are listed but not counted against the score, "
            "because unusual is not necessarily wrong. Checks depend on each column's role (Data tab).")

    if not found:
        st.success("No issues detected with the current roles and fixes.")
    for i in found:
        with st.container(border=True):
            c1, c2, c3 = st.columns([5, 3, 1.2])
            c1.markdown(f"{SEV_ICON[i.severity]} **{RULE_NAMES[i.rule]}** · `{i.column}` · "
                        f"**{i.count:,}** of {i.denominator:,} rows ({i.pct}%)  \n{i.detail}")
            action = c2.selectbox("Fix", i.fixes, format_func=FIX_LABELS.get, key=f"fx_{i.key}",
                                  label_visibility="collapsed")
            if c3.button("Apply", key=f"ap_{i.key}"):
                st.session_state.fixes.append(Fix(i.key, i.rule, i.column, action, dict(i.params)))
                st.rerun()

    if st.session_state.fixes:
        st.subheader("Applied fixes")
        st.caption("Fixes are replayed in order on the original upload, so they can be undone. "
                   "All other tabs use the fixed data.")
        st.dataframe(pd.DataFrame([{
            "step": n + 1, "column": f.column, "issue": RULE_NAMES[f.rule], "action": FIX_LABELS[f.action],
            "rows before": f.rows_before, "rows after": f.rows_after,
            "rows removed": f.rows_before - f.rows_after, "values changed": f.cells_changed,
        } for n, f in enumerate(st.session_state.fixes)]), hide_index=True, width="stretch")
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("Undo last fix"):
            st.session_state.fixes.pop()
            st.rerun()
        if b2.button("Reset all fixes"):
            st.session_state.fixes = []
            st.rerun()
