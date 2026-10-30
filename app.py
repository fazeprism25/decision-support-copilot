"""Decision-support app entry point. Run with: streamlit run app.py"""
import streamlit as st

from engine import llm
from engine.config import get_settings
from engine.llm import LLMError, chat
from ui import ask_tab, data_tab, demo, explain_tab, kpi_tab, quality_tab, recs_tab, route_tab, state, whatif_tab

st.set_page_config(page_title="Decision Support", layout="wide")

settings = get_settings()

with st.sidebar:
    st.title("Decision Support")
    c1, c2 = st.columns(2)
    c1.button("🔄 Reset everything", on_click=demo.reset, help="Clears data, fixes, mapping, constraints and answers.")
    c2.button("▶️ Load demo scenario", on_click=demo.request, disabled=not demo.available(),
              help=f"Loads {demo.DEMO_FILE}, applies recommended fixes and the value mapping, and adds the "
                   f"constraint “{demo.DEMO_CONSTRAINT}”.")
    if st.session_state.get("demo_msg"):
        st.success(st.session_state.demo_msg)
    ok, msg = llm.status()
    if ok:
        st.success(f"AI on ({settings.provider})  \nsmart: {settings.model_smart}  \nfast: {settings.model_fast}")
        if st.button("Test AI connection"):
            try:
                with st.spinner("Calling model..."):
                    reply = chat("Reply with exactly: OK", tier="fast", max_tokens=20)
                st.info(f"Model replied: {reply.strip()[:100]}")
            except LLMError as e:
                st.error(f"AI call failed: {e}")
    elif settings.ai_enabled:
        st.error(f"⚠️ AI-down mode. {msg}  \nProfiling, quality, KPIs, What-if, sliders and rule-based "
                 "recommendations keep working.")
        if st.button("Retry AI now"):
            llm.reset()
            st.rerun()
    else:
        st.warning("AI off: no LLM_API_KEY in .env. Profiling, KPIs, charts, sliders and rule-based "
                   "recommendations still work.")
    if state.has_data():
        whatif_tab.sidebar()
    else:
        st.caption("What-if sliders appear here once data is loaded.")

TABS = ["Data", "Quality", "KPIs", "Ask", "What-if", "Route cost", "Recommendations", "Explain", "Change Log"]
BUILT = {"Quality": quality_tab.render, "KPIs": kpi_tab.render, "Ask": ask_tab.render,
         "What-if": whatif_tab.render, "Route cost": route_tab.render,
         "Recommendations": recs_tab.render, "Explain": explain_tab.render,
         "Change Log": whatif_tab.render_changelog}

for tab, name in zip(st.tabs(TABS), TABS):
    with tab:
        st.header(name)
        if name == "Data":
            data_tab.render()
            continue
        if state.has_data():
            BUILT[name]()
        else:
            st.info("Load a dataset in the Data tab first.")
