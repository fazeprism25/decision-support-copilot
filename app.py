"""Decision-support app entry point. Run with: streamlit run app.py"""
import streamlit as st

from engine import llm
from engine.config import get_settings
from engine.llm import LLMError, chat
from ui import ask_tab, data_tab, demo, explain_tab, kpi_tab, quality_tab, recs_tab, route_tab, state, style, whatif_tab

st.set_page_config(page_title="Decision Support", page_icon=":material/insights:", layout="wide")
style.apply()

settings = get_settings()

PAGES = ["Data", "Quality", "KPIs", "Ask", "What-if", "Route cost", "Recommendations", "Explain", "Change Log"]
# Only the open page runs, and Streamlit forgets a widget's value while its page is closed. Re-saving these keys
# every run keeps the What-if levers, the chosen file and the open page across page switches.
KEEP = ("sl_", "src", "sheet_", "header_row_", "wi_metric", "page")
for k in list(st.session_state):
    if str(k).startswith(KEEP):
        st.session_state[k] = st.session_state[k]

with st.sidebar:
    st.markdown("### Decision Support")
    with st.container(key="nav"):
        page = st.radio("Section", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    if state.has_data():
        whatif_tab.sidebar_summary()
        st.divider()
    ok, msg = llm.status()
    if ok:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f":green-badge[AI on] {settings.provider}", width="content")
            style.info(f"**Smart model** (questions, constraints): `{settings.model_smart}`\n\n"
                       f"**Fast model** (roles, KPIs, wording): `{settings.model_fast}`\n\n"
                       "The AI never calculates numbers: it only labels, chooses and words things. "
                       "Every number comes from code run on your data.")
        if st.button("Test AI connection", type="tertiary"):
            try:
                with st.spinner("Calling model..."):
                    reply = chat("Reply with exactly: OK", tier="fast", max_tokens=20)
                st.info(f"Model replied: {reply.strip()[:100]}")
            except LLMError as e:
                st.error(f"AI call failed: {e}")
    elif settings.ai_enabled:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(":orange-badge[AI-down mode]", width="content")
            style.info(f"{msg}\n\nProfiling, quality, KPIs, What-if, levers and rule-based recommendations keep "
                       "working without AI. Only new questions on the Ask page need it.")
        if st.button("Retry AI now", type="tertiary"):
            llm.reset()
            st.rerun()
    else:
        st.warning("AI off: no LLM_API_KEY in .env.")
        style.info("Profiling, KPIs, charts, levers and rule-based recommendations still work without AI.")
    st.divider()
    with st.container(key="session", horizontal=True, gap="small"):
        st.button("Load demo data", on_click=demo.request, disabled=not demo.available(), type="tertiary",
                  help=f"Loads {demo.DEMO_FILE}, applies recommended fixes and the value mapping, and adds the "
                       f"constraint “{demo.DEMO_CONSTRAINT}”.")
        st.button("Reset everything", on_click=demo.reset, type="tertiary",
                  help="Clears data, fixes, mapping, constraints and answers.")
    if st.session_state.get("demo_msg"):
        st.caption(st.session_state.demo_msg)

INTRO = {
    "Data": ("Load a file and tell the app what each column means.",
             "Pick a CSV or Excel file. The app finds the header row, then gives every column a **role** "
             "(date, money, category, …). Roles decide which checks, KPIs and charts apply, so fix any that look "
             "wrong. You can also map messy category labels (P1/P2, rto, Y/Yes) to standard ones."),
    "Quality": ("Find and fix problems in the data before anything is calculated.",
                "The app checks every column against its role: repeated IDs, missing values, negative amounts, "
                "zero durations, ratings outside their scale, unreadable dates, spelling variants and extreme values. "
                "Fixes are applied in order and can be undone; every other page uses the fixed data."),
    "KPIs": ("Key numbers and charts, chosen for this dataset.",
             "The AI (or rules, if AI is off) picks which calculations fit your columns; pandas computes every "
             "value. Click **Explain** on any card for its formula, filter and the rows it used. "
             "Add your own with **Add a KPI**, remove any with **Remove**."),
    "Ask": ("Ask a question about your data in plain English.",
            "The AI writes pandas code, the app runs it on your data in a locked-down sandbox, and every answer "
            "comes from that executed code. The code is shown under each answer. Example questions are prepared "
            "in the background; *(ready)* means the answer is already there."),
    "What-if": ("Test a change and see what it does downstream.",
                "Move the levers at the top of the page or type a constraint (e.g. *Hub H02 is down for 3 days*). "
                "The impact panel updates live and shows what each lever did on its own. Typed constraints show "
                "the change as parameters, you confirm, and it recomputes the cascade: capacity → allocation → "
                "utilisation → time → on-time → cost & margin → feasibility. Constraints stack and are logged."),
    "Route cost": ("Is the cheapest route really the cheapest?",
                   "Management ranks routes on base cost. This page adds the costs that estimate may leave out "
                   "(failed attempts, returns, waiting, parking, tolls, your own entries), each with a status "
                   "(confirmed / provisional / unverified), and checks whether the cheapest route still wins."),
    "Recommendations": ("What to do next, ranked by impact.",
                        "Each recommendation comes from a rule over computed numbers, never from the AI's opinion. "
                        "The AI only rewrites the wording. Open **Evidence** to see the numbers before and after "
                        "the active scenario, plus assumptions and limitations."),
    "Explain": ("Every formula, assumption and exclusion behind the numbers.",
                "Generated from what the app actually ran on the current data, fixes, mapping and scenario. "
                "Use this to answer *how did you get that number?*"),
    "Change Log": ("A record of every constraint you applied and what it moved.",
                   "Each entry shows the constraint, how it was interpreted, the parameters and every metric "
                   "that changed. Export it as CSV."),
}
BUILT = {"Quality": quality_tab.render, "KPIs": kpi_tab.render, "Ask": ask_tab.render,
         "What-if": whatif_tab.render, "Route cost": route_tab.render,
         "Recommendations": recs_tab.render, "Explain": explain_tab.render,
         "Change Log": whatif_tab.render_changelog}


lead, more = INTRO[page]
with st.container(horizontal=True, vertical_alignment="center", gap="small"):
    st.markdown(f"## {page}", width="content")
    style.info(more)
st.caption(lead)
if page == "Data":
    data_tab.render()
elif state.has_data():
    kpi_tab.ensure_plan()
    ask_tab.ensure_prefetch()
    BUILT[page]()
else:
    st.info("Load a dataset on the Data page first.")
