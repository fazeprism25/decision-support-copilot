"""Decision-support app entry point. Run with: streamlit run app.py"""
import streamlit as st

from engine.config import get_settings
from engine.llm import LLMError, chat
from ui import data_tab

st.set_page_config(page_title="Decision Support", layout="wide")

settings = get_settings()

if "change_log" not in st.session_state:
    st.session_state.change_log = []

with st.sidebar:
    st.title("Decision Support")
    if settings.ai_enabled:
        st.success(f"AI on · smart: {settings.model_smart} · fast: {settings.model_fast}")
    else:
        st.warning("AI off: no ANTHROPIC_API_KEY in .env. Profiling, KPIs, charts and sliders still work.")
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

for tab, name in zip(st.tabs(TABS), TABS):
    with tab:
        st.header(name)
        if name == "Data":
            data_tab.render()
            continue
        if "df" not in st.session_state:
            st.caption("Load a dataset in the Data tab first.")
        st.info(PLANNED[name])
        if name == "Change Log" and not st.session_state.change_log:
            st.caption("No changes recorded yet.")
