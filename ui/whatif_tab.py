"""What-if tab + sidebar sliders + Change Log: stacked constraints, baseline vs scenario, C10 cascade view."""
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from engine import kpis as K
from engine.config import get_settings
from engine.constraints import parse
from engine.scenario import (CapacityChange, MeasureChange, Params, Setup, apply_rows, cascade, change_rows, combine,
                             guess_setup)
from ui import state

VERDICT_STYLE = {"Feasible": "success", "Feasible with risk": "warning", "Not feasible": "error"}


# ---------- setup ----------

def setup() -> Setup:
    new_data = st.session_state.get("setup_for") != st.session_state.dataset_id
    plan = st.session_state.get("kpi_plan") if st.session_state.get("kpi_plan_for") == st.session_state.dataset_id else None
    if new_data or (plan and not st.session_state.get("setup_used_plan")):  # re-guess once the KPI plan exists
        ctx = state.kpi_ctx()
        st.session_state.setup = guess_setup(state.clean_df(), ctx.roles, ctx.entity_of, plan or [],
                                             st.session_state.profile)
        st.session_state.setup_used_plan = bool(plan)
        if new_data:
            st.session_state.setup_for = st.session_state.dataset_id
            st.session_state.constraints = []
            st.session_state.change_log = []
    return st.session_state.setup


def _setup_editor(s: Setup, df: pd.DataFrame, ctx: K.Ctx):
    by = lambda *rs: [None] + [c for c, r in ctx.roles.items() if r in rs]
    pick = lambda label, opts, cur, key: st.selectbox(label, opts, index=opts.index(cur) if cur in opts else 0, key=key)
    with st.expander("Model setup: which columns drive the cascade (auto-filled, editable)",
                     expanded=bool(s.cascade_ready())):
        c1, c2, c3 = st.columns(3)
        s.entity = pick("Entity (e.g. site/hub)", by("location", "category", "id"), s.entity, "su_ent")
        attrs = [None] + [a for a, e in ctx.entity_of.items() if e == s.entity]
        s.capacity = c1.selectbox("Capacity per day", attrs, index=attrs.index(s.capacity) if s.capacity in attrs else 0,
                                  key="su_cap")
        s.reported_util = c2.selectbox("Reported utilisation (optional)", attrs,
                                       index=attrs.index(s.reported_util) if s.reported_util in attrs else 0, key="su_ru")
        bases = ["observed", "reported"] if s.reported_util else ["observed"]
        s.load_basis = c3.radio("Load basis", bases, index=bases.index(s.load_basis) if s.load_basis in bases else 0,
                                horizontal=True, key="su_lb",
                                help="observed = rows per day in the file; reported = reported utilisation × capacity "
                                     "(use when the file is a sample of real volume)")
        c1, c2, c3 = st.columns(3)
        s.date = pick("Date", by("date"), s.date, "su_date")
        s.actual = c1.selectbox("Actual time", by("duration"), index=by("duration").index(s.actual)
                                if s.actual in by("duration") else 0, key="su_act")
        s.promised = c2.selectbox("Promised time", by("duration"), index=by("duration").index(s.promised)
                                  if s.promised in by("duration") else 0, key="su_pro")
        s.status = c3.selectbox("Status column", by("category", "flag"), index=by("category", "flag").index(s.status)
                                if s.status in by("category", "flag") else 0, key="su_st")
        if s.status:
            vals = sorted(df[s.status].dropna().astype("string").unique().tolist())
            s.completed_values = st.multiselect("Completed status value(s)", vals,
                                                [v for v in s.completed_values if v in vals], key="su_cv")
        c1, c2 = st.columns(2)
        s.fee = c1.selectbox("Fee / charge (revenue of the activity)", by("money"),
                             index=by("money").index(s.fee) if s.fee in by("money") else 0, key="su_fee")
        s.cost = c2.selectbox("Cost of the same activity", by("money"),
                              index=by("money").index(s.cost) if s.cost in by("money") else 0, key="su_cost")


# ---------- sidebar sliders ----------

def sidebar():
    s, ctx = setup(), state.kpi_ctx()
    st.subheader("What-if sliders")
    nums = [c for r in ("money", "duration", "quantity", "measure") for c, rr in ctx.roles.items()
            if rr == r and c not in ctx.entity_of][:6]
    changes = []
    for c in nums:
        v = st.slider(f"{c} % change", -50, 50, 0, 1, key=f"sl_{c}")
        if v:
            changes.append(MeasureChange(column=c, pct=float(v)))
    cap = []
    if s.entity and s.capacity:
        ents = sorted(state.clean_df()[s.entity].dropna().astype("string").unique().tolist())
        e = st.selectbox(f"Capacity change for {s.entity}", ents, key="sl_ent")
        pct = st.slider("Capacity % change", -100, 50, 0, 5, key="sl_cap")
        days = st.number_input("For how many days (0 = whole period)", 0, 365, 0, key="sl_days")
        if pct:
            cap.append(CapacityChange(entity_value=e, pct=float(pct), days=float(days) if days else None))
    util = st.slider("Utilisation cap for receiving entities %", 50, 100, 85, 1, key="sl_util")
    xc = st.number_input("Assumed extra cost per moved order", 0.0, value=0.0, key="sl_xc",
                         help="0 = default of 10% of the average cost per order")
    xm = st.number_input("Assumed extra minutes per moved order", 0.0, value=10.0, key="sl_xm")
    st.session_state.slider_params = Params(measure_changes=changes, capacity_changes=cap, util_cap=util / 100,
                                            moved_extra_cost=xc or None, moved_extra_minutes=xm)


def current_params(extra: Params | None = None) -> Params:
    parts = [st.session_state.get("slider_params", Params())] + [c["params"] for c in st.session_state.get("constraints", [])]
    return combine(parts + ([extra] if extra else []))


def run_scenario(params: Params) -> dict:
    df, ctx, s = state.clean_df(), state.kpi_ctx(), setup()
    scen, notes = apply_rows(df, params)
    plan = [it for it in st.session_state.get("kpi_plan", []) if not K.validate(it, df, ctx)]
    base_k, scen_k = K.run_plan(plan, df, ctx), K.run_plan(plan, scen, ctx)
    casc = cascade(df, scen, s, params) if not s.cascade_ready() else None
    return {"notes": notes, "base": base_k, "scen": scen_k, "cascade": casc, "rows": (len(df), len(scen))}


def snapshot(res: dict) -> dict[str, float | str]:
    out = {f"KPI · {k.title}": k.value for k in res["scen"] if k.chart is None or k.calc in ("top_n",)}
    if res["cascade"]:
        for r in res["cascade"]["steps"].itertuples():
            out[f"Cascade · {r.metric}"] = r.scenario
        out["Cascade · feasibility"] = res["cascade"]["verdict"]
    return out


def log_change(text: str, interpretation: str, params: Params, before: dict, after: dict):
    log = st.session_state.setdefault("change_log", [])
    n = len({e["#"] for e in log}) + 1
    log.extend(change_rows(n, datetime.now().strftime("%H:%M:%S"), text, interpretation, params, before, after))


# ---------- tab ----------

def render():
    df, ctx, s = state.clean_df(), state.kpi_ctx(), setup()
    _setup_editor(s, df, ctx)
    st.session_state.setdefault("constraints", [])

    st.subheader("New constraint")
    with st.form("constraint", clear_on_submit=True):
        text = st.text_input("Type a constraint card, e.g. 'Hub H02 is down for 3 days', 'fee rise capped at 5%', "
                             "'ignore cancelled orders'")
        go = st.form_submit_button("Interpret", disabled=not get_settings().ai_enabled)
    if not get_settings().ai_enabled:
        st.caption("Typed constraints need AI (off). The sidebar sliders work without it.")
    if go and text.strip():
        with st.spinner("Translating constraint into parameters..."):
            st.session_state.pending = parse(text.strip(), df, ctx.roles, s, st.session_state.profile,
                                             f"{st.session_state.dataset_id}:{text.strip().lower()}")
    pend = st.session_state.get("pending")
    if pend:
        with st.container(border=True):
            if pend.ok:
                st.markdown(f"**Proposed:** {pend.interpretation}")
                st.json(pend.params.model_dump(exclude_defaults=True))
                c1, c2, _ = st.columns([1, 1, 4])
                if c1.button("✅ Apply", type="primary"):
                    before = snapshot(run_scenario(current_params()))
                    after = snapshot(run_scenario(current_params(pend.params)))
                    st.session_state.constraints.append({"text": pend.text, "params": pend.params,
                                                         "interpretation": pend.interpretation})
                    log_change(pend.text, pend.interpretation, pend.params, before, after)
                    st.session_state.pending = None
                    st.rerun()
                if c2.button("Discard"):
                    st.session_state.pending = None
                    st.rerun()
            else:
                st.error(f"Cannot apply “{pend.text}”: {pend.reason}")
                for e in pend.errors:
                    st.caption(f"• {e}")
                if pend.suggestion:
                    st.info(f"Try: {pend.suggestion}")
                if st.button("Dismiss"):
                    st.session_state.pending = None
                    st.rerun()

    if st.session_state.constraints:
        st.markdown("**Active constraints (stacked in order)**")
        for i, c in enumerate(st.session_state.constraints):
            c1, c2 = st.columns([8, 1])
            c1.markdown(f"{i + 1}. {c['text']} — *{c['interpretation']}*")
            if c2.button("Remove", key=f"rmc_{i}"):
                st.session_state.constraints.pop(i)
                st.rerun()

    params = current_params()
    res = run_scenario(params)
    if st.button("Record current slider settings in the Change Log"):
        sp = st.session_state.get("slider_params", Params())
        before = snapshot(run_scenario(combine([c["params"] for c in st.session_state.constraints])))
        log_change("Sidebar sliders", "slider settings at time of recording", sp, before, snapshot(res))
        st.success("Recorded.")
    for n in res["notes"]:
        st.caption(n)
    st.caption(f"Baseline: {res['rows'][0]:,} rows (cleaned data) · scenario: {res['rows'][1]:,} rows")

    st.subheader("Cascade: capacity → allocation → utilisation → time → on-time → cost & margin → feasibility")
    casc = res["cascade"]
    if casc is None:
        st.info(f"Cascade needs: {', '.join(s.cascade_ready())}. Set them in Model setup above.")
    else:
        getattr(st, VERDICT_STYLE[casc["verdict"]])(f"**{casc['verdict']}** — " + "; ".join(casc["reasons"]))
        steps = casc["steps"].copy()
        for c in ("baseline", "scenario", "change"):
            steps[c] = steps[c].map(lambda v: f"{v:,.2f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else "")
        st.dataframe(steps, hide_index=True, width="stretch")
        st.caption(f"Period {casc['days']} days" + (f", capacity change on {casc['affected_days']:g} of them"
                                                     if casc["affected_days"] else "") +
                   f" · load basis: {casc['basis']} · {casc['calibration']}. "
                   "All scenario values are estimates based on the assumptions listed here.")
        with st.expander("Per-entity view (calculated vs reported utilisation shown side by side)"):
            st.dataframe(casc["entities"], hide_index=True, width="stretch")
        with st.expander("Assumptions"):
            for k, v in casc["assumptions"].items():
                st.markdown(f"- **{k}:** {v}")

    st.subheader("KPIs: baseline vs scenario")
    rows = [{"KPI": b.title, "baseline": K.fmt_value(b.value, b.fmt), "scenario": K.fmt_value(sc.value, sc.fmt),
             "change": (K.fmt_value(sc.value - b.value, b.fmt) if b.value is not None and sc.value is not None else "n/a"),
             "basis": b.basis} for b, sc in zip(res["base"], res["scen"]) if b.chart is None or b.calc == "top_n"]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("KPIs reflect row-level changes (exclusions, % changes, price changes). Capacity effects appear "
                   "in the cascade above.")


def render_changelog():
    log = st.session_state.get("change_log", [])
    if not log:
        st.caption("No changes recorded yet. Apply a constraint on the What-if tab.")
        return
    df = pd.DataFrame(log)
    for n, g in df.groupby("#", sort=False):
        with st.container(border=True):
            st.markdown(f"**#{n} · {g['time'].iloc[0]} · {g['constraint'].iloc[0]}**  \n*{g['interpretation'].iloc[0]}*")
            st.code(g["parameters"].iloc[0], language="json")
            moved = g[g["change"].astype(str).ne("") & g["change"].astype(str).ne("0") & g["change"].astype(str).ne("0.0")]
            show = moved if len(moved) else g
            st.dataframe(show[["metric", "before", "after", "change"]].astype(str), hide_index=True, width="stretch")
    st.download_button("Export Change Log (CSV)", df.to_csv(index=False).encode(), "change_log.csv")
