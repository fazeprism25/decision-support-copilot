"""Demo safety: 'Reset everything' and 'Load demo scenario' (BUDS + recommended fixes + confirmed value mapping +
one constraint). The demo constraint is parsed by the rule parser, so it works with the AI down."""
from pathlib import Path

import streamlit as st

from engine import llm
from engine.domains import delivery as D
from engine.quality import recommended_fixes
from engine.valuemap import from_table, to_table
from ui import state

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEMO_FILE = "Round_1_BUDS.xlsx"
DEMO_CONSTRAINT = "Central hub is down for 3 days"


def reset():
    """on_click: forget everything (data, fixes, mapping, constraints, answers) and clear the AI-down flag."""
    st.session_state.clear()
    llm.reset()


def request():
    """on_click: start from a clean session and point the Data tab at the demo file."""
    reset()
    st.session_state["src"] = "Use a file from the data folder"
    st.session_state["src_file"] = DEMO_FILE
    st.session_state["demo_pending"] = True


def available() -> bool:
    return (DATA_DIR / DEMO_FILE).exists()


def finish():
    """Called by the Data tab once the demo file is loaded and roles exist."""
    if not st.session_state.pop("demo_pending", False):
        return
    from ui import data_tab, kpi_tab, whatif_tab

    steps = []
    st.session_state.fixes = recommended_fixes(state.issues())
    steps.append(f"{len(st.session_state.fixes)} recommended quality fixes applied")
    df, ctx = state.fixed_df(), state.kpi_ctx()
    vm, src = D.propose_mapping_rules(df, ctx.roles)
    st.session_state.vmap_prop = to_table(vm, df, src)
    st.session_state.vmap_prop_for = (st.session_state.dataset_id, state.fixes_sig(), state.roles_sig())
    st.session_state.vmap_msg = "Proposed by rules (demo)."
    data_tab.confirm_vmap(from_table(st.session_state.vmap_prop))
    steps.append("value mapping confirmed")
    kpi_tab._make_plan(state.clean_df(), state.kpi_ctx(), use_ai=llm.status()[0])
    st.session_state.kpi_plan_for = st.session_state.dataset_id
    whatif_tab.setup()
    p = whatif_tab.interpret(DEMO_CONSTRAINT)
    if p.ok:
        whatif_tab.apply_constraint(p)
        steps.append(f"constraint “{DEMO_CONSTRAINT}” applied ({p.interpretation})")
    else:
        steps.append(f"constraint not applied: {p.reason}")
    st.session_state.demo_msg = "Demo loaded: " + "; ".join(steps) + "."
    st.rerun()
