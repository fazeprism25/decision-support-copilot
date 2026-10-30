"""Decision-support app entry point. Run with: streamlit run app.py"""
import streamlit as st

from engine.config import get_settings
from engine.llm import LLMError, chat
from ui import ask_tab, data_tab, kpi_tab, quality_tab, state

st.set_page_config(page_title="Decision Support", layout="wide")

settings = get_settings()

if "change_log" not in st.session_state:
    st.session_state.change_log = []

with st.sidebar:
    st.title("Decision Support")
    if settings.ai_enabled:
        st.success(f"AI on ({settings.provider})  \nsmart: {settings.model_smart}  \nfast: {settings.model_fast}")
        if st.button("Test AI connection"):
            try:
                with st.spinner("Calling model..."):
                    reply = chat("Reply with exactly: OK", tier="fast", max_tokens=20)
                st.info(f"Model replied: {reply.strip()[:100]}")
            except LLMError as e:
                st.error(f"AI call failed: {e}")
    else:
        st.warning("AI off: no LLM_API_KEY in .env. Profiling, KPIs, charts and sliders still work.")
    st.caption("Scenario sliders will appear here (Phase 6).")

TABS = ["Data", "Quality", "KPIs", "Ask", "What-if", "Recommendations", "Explain", "Change Log"]
PLANNED = {
    "Data": "Upload a CSV/Excel file and review detected column roles (Phases 1-2).",
    "Quality": "Data-quality checks, score and one-click fixes (Phase 3).",
    "KPIs": "Role-driven KPIs and interactive charts with explanations (Phase 4).",
    "Ask": "Ask questions in plain English; answers come from executed code (Phase 5).",
    "What-if": "Scenario simulation and constraint cards: baseline vs scenario (Phase 6).",
    "Recommendations": "Evidence-based recommendations with confidence and limitations (Phase 7).",
    "Explain": "Every formula, assumption, exclusion and limitation (Phase 7).",
    "Change Log": "Every constraint applied and how each KPI moved (Phase 6).",
}

BUILT = {"Quality": quality_tab, "KPIs": kpi_tab, "Ask": ask_tab}

for tab, name in zip(st.tabs(TABS), TABS):
    with tab:
        st.header(name)
        if name == "Data":
            data_tab.render()
            continue
        if name in BUILT:
            if state.has_data():
                BUILT[name].render()
            else:
                st.info("Load a dataset in the Data tab first.")
            continue
        st.info(PLANNED[name])
        if name == "Change Log" and not st.session_state.change_log:
            st.caption("No changes recorded yet.")
