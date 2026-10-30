"""What-if page (levers + live impact panel) + Change Log: stacked constraints, baseline vs scenario, C10 cascade view."""
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import kpis as K
from engine import llm
from engine.constraints import parse
from engine import routecost as RC
from engine.domains import delivery as D
from engine.scenario import (CapacityChange, MeasureChange, Params, Setup, apply_rows, cascade, change_rows, combine,
                             guess_setup)
from ui import state, style
from ui.charts import dark

VERDICT_STYLE = {"Feasible": "success", "Feasible with risk": "warning", "Not feasible": "error"}


# ---------- setup ----------

def setup() -> Setup:
    new_data = st.session_state.get("setup_for") != st.session_state.dataset_id
    plan = st.session_state.get("kpi_plan") if st.session_state.get("kpi_plan_for") == st.session_state.dataset_id else None
    if new_data or (plan and not st.session_state.get("setup_used_plan")):  # re-guess once the KPI plan exists
        ctx = state.kpi_ctx()
        su = guess_setup(state.clean_df(), ctx.roles, ctx.entity_of, plan or [], st.session_state.profile)
        st.session_state.setup = D.fill_setup(su, D.detect(state.clean_df(), ctx.roles, ctx.entity_of, su))
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
        st.caption("Filled in automatically from the column roles. Change a choice only if it looks wrong.")
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


# ---------- levers (sliders) ----------

UP_FIRST = ("on-time", "on time", "margin", "revenue", "profit", "rating", "sales")
LOWER_BETTER = ("unserved", "moved", "utilisation", "cycle time", "cost", "time", "duration", "delay", "late", "wait",
                "cancel", "return", "fail", "missing")
HIGHER_BETTER = ("served", "capacity", "fee")
HEADLINE = ("on-time %", "margin per order", "unserved orders per day", "highest utilisation %")
SHORT = {"on-time %": "On-time", "margin per order": "Margin / order", "unserved orders per day": "Unserved / day",
         "highest utilisation %": "Peak utilisation"}


def _lever_columns(ctx: K.Ctx) -> list[str]:
    return [c for r in ("money", "duration", "quantity", "measure") for c, rr in ctx.roles.items()
            if rr == r and c not in ctx.entity_of][:6]


def _reset_levers():
    for k in [k for k in st.session_state if str(k).startswith("sl_")]:
        del st.session_state[k]


def sidebar_summary():
    """Short scenario status in the sidebar; the levers themselves live on the What-if page."""
    moved = [k for k in st.session_state if str(k).startswith("sl_") and k not in ("sl_util", "sl_xc", "sl_xm", "sl_ent",
                                                                                   "sl_days") and st.session_state[k]]
    n_c = len(st.session_state.get("constraints", []))
    style.eyebrow("Scenario")
    st.markdown(f"**{len(moved)}** lever(s) moved · **{n_c}** constraint(s)" if moved or n_c else
                "Baseline (nothing changed)")
    st.caption("Adjust levers and constraints on the What-if page.")
    if moved:
        st.button("Reset levers", on_click=_reset_levers, width="stretch")


IDLE = '<div class="effect idle">Not moved</div>'


def levers(s: Setup, ctx: K.Ctx) -> tuple[list[tuple[str, Params, object]], Params]:
    """Draw the sliders. Returns the moved levers as (name, params, placeholder for its effect line) + assumptions."""
    out = []
    style.eyebrow("Change a number")
    for c in _lever_columns(ctx):
        v = st.slider(f"{c}", -50, 50, 0, 1, key=f"sl_{c}", format="%+d%%",
                      help=f"Raise or lower every value of '{c}' by this percentage.")
        slot = st.empty()
        if v:
            out.append((f"{c} {v:+d}%", Params(measure_changes=[MeasureChange(column=c, pct=float(v))]), slot))
        else:
            slot.markdown(IDLE, unsafe_allow_html=True)
    if s.entity and s.capacity:
        style.eyebrow(f"Change one {s.entity}'s capacity")
        ents = sorted(state.clean_df()[s.entity].dropna().astype("string").unique().tolist())
        c1, c2 = st.columns([3, 2])
        e = c1.selectbox(f"{s.entity}", ents, key="sl_ent")
        days = c2.number_input("Days (0 = all)", 0, 365, 0, key="sl_days",
                               help="How many days the capacity change lasts. 0 = the whole period.")
        pct = st.slider("Capacity change", -100, 50, 0, 5, key="sl_cap", format="%+d%%")
        slot = st.empty()
        if pct:
            name = f"{e} capacity {pct:+d}%" + (f" for {days} days" if days else "")
            out.append((name, Params(capacity_changes=[CapacityChange(entity_value=e, pct=float(pct),
                                                                      days=float(days) if days else None)]), slot))
        else:
            slot.markdown(IDLE, unsafe_allow_html=True)
    with st.expander("Assumptions used when orders move between sites"):
        util = st.slider("Utilisation cap for receiving sites", 50, 100, 85, 1, key="sl_util", format="%d%%",
                         help="When a site loses capacity, its extra orders move to other sites, but no site is "
                              "filled above this level. Orders that fit nowhere are counted as unserved.")
        xc = st.number_input("Extra cost per moved order", 0.0, value=0.0, key="sl_xc",
                             help="0 = default of 10% of the average cost per order")
        xm = st.number_input("Extra minutes per moved order", 0.0, value=10.0, key="sl_xm",
                             help="Extra travel/handling time for an order moved to another site.")
    assume = Params(util_cap=util / 100, moved_extra_cost=xc or None, moved_extra_minutes=xm)
    st.button("Reset levers", on_click=_reset_levers, key="reset_levers_tab", disabled=not out)
    st.session_state.slider_params = combine([p for _, p, _ in out] + [assume])
    return out, assume


# ---------- impact panel ----------

def _better(name: str) -> str | None:
    """Which direction is good for a metric, from its name; None = no clear direction (shown grey)."""
    n = name.lower()
    for words, way in ((UP_FIRST, "up"), (LOWER_BETTER, "down"), (HIGHER_BETTER, "up")):
        if any(w in n for w in words):
            return way
    return None


def _metric_rows(base: dict, scen: dict) -> list[dict]:
    """Metric name -> baseline / scenario / format / which direction is good."""
    rows = []
    for name, (b, fmt) in base.items():
        sc = scen.get(name, (None, fmt))[0]
        rows.append({"name": name, "base": b, "scen": sc, "fmt": fmt, "better": _better(name)})
    return rows


def _values(res: dict, base: bool) -> dict:
    """{metric: (value, fmt)} from the cascade steps and the scalar KPIs of one scenario run."""
    out = {}
    if res.get("cascade"):
        for r in res["cascade"]["steps"].itertuples():
            out[r.metric] = (r.baseline if base else r.scenario, "pct" if "%" in r.metric else "number")
    for k in res["base" if base else "scen"]:
        if k.chart is None and k.title not in out:
            out[k.title] = (k.value, k.fmt)
    return out


def _num(v) -> float | None:
    return float(v) if v is not None and pd.notna(v) else None


def _delta(r: dict) -> float | None:
    b, sc = _num(r["base"]), _num(r["scen"])
    return None if b is None or sc is None else sc - b


def _fmt_delta(d: float, fmt: str) -> str:
    return f"{d:+.1f} pts" if fmt == "pct" else f"{d:+,.2f}"


def _tone(r: dict, d: float) -> str:
    if abs(d) < 1e-9 or r["better"] is None:
        return ""
    return "up" if (d > 0) == (r["better"] == "up") else "down"


def _effect_line(rows: list[dict]) -> str:
    """HTML: the two biggest relative moves, e.g. 'on-time % 20.7% -> 18.1% (-2.6 pts)'."""
    moved = [(r, d) for r in rows if (d := _delta(r)) is not None and abs(d) > 1e-9]
    if not moved:
        return '<div class="effect idle">On its own: no change to the numbers below.</div>'
    moved.sort(key=lambda t: -abs(t[1]) / (abs(_num(t[0]["base"]) or 0) or 1))
    parts = [f'{r["name"]} {K.fmt_value(r["base"], r["fmt"])} &rarr; {K.fmt_value(r["scen"], r["fmt"])} '
             f'<span class="{_tone(r, d)}">({_fmt_delta(d, r["fmt"])})</span>' for r, d in moved[:2]]
    return '<div class="effect">On its own: ' + " · ".join(parts) + "</div>"


def _quick(params: Params) -> dict:
    """Scenario run without the route analysis, for per-lever effects."""
    df, ctx, s = state.clean_df(), state.kpi_ctx(), setup()
    scen, _ = apply_rows(df, params)
    plan = [it for it in st.session_state.get("kpi_plan", []) if not K.validate(it, df, ctx)]
    return {"base": [], "scen": K.run_plan(plan, scen, ctx),
            "cascade": cascade(df, scen, s, params) if not s.cascade_ready() else None}


def _tiles(rows: list[dict], casc: dict | None):
    pick = [r for h in HEADLINE for r in rows if r["name"] == h] or rows
    pick = pick[:4]
    if casc:
        tone = {"Feasible": "green", "Feasible with risk": "orange", "Not feasible": "red"}[casc["verdict"]]
        st.markdown(f":{tone}-badge[{casc['verdict']}] " + "; ".join(casc["reasons"]))
    cols = st.columns(max(len(pick), 1))
    for col, r in zip(cols, pick):
        d = _delta(r)
        col.metric(SHORT.get(r["name"], r["name"]), K.fmt_value(r["scen"], r["fmt"]),
                   _fmt_delta(d, r["fmt"]) if d is not None and abs(d) > 1e-9 else None,
                   delta_color={"up": "normal", "down": "inverse"}.get(r["better"], "off"),
                   help=f"{r['name']}: baseline {K.fmt_value(r['base'], r['fmt'])}")


def _waterfall(metric: str, rows: list[dict], solo: list[tuple[str, list[dict]]]):
    r = next(x for x in rows if x["name"] == metric)
    b, total = _num(r["base"]), _num(r["scen"])
    if b is None or total is None:
        st.caption("This number is not available for both baseline and scenario.")
        return
    parts = []
    for name, srows in solo:
        sr = next((x for x in srows if x["name"] == metric), None)
        d = _delta(sr) if sr else None
        parts.append((name, d or 0.0))
    overlap = (total - b) - sum(d for _, d in parts)
    if abs(overlap) > 1e-6 * max(1.0, abs(b)):
        parts.append(("Levers combined (overlap)", overlap))
    good, bad = style.GREEN, style.RED
    if r["better"] == "down":
        good, bad = bad, good
    elif r["better"] is None:
        good = bad = "#8a8580"
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=["absolute"] + ["relative"] * len(parts) + ["total"],
        x=["Baseline"] + [n for n, _ in parts] + ["Scenario"], y=[b] + [d for _, d in parts] + [total],
        text=[K.fmt_value(b, r["fmt"])] + [_fmt_delta(d, r["fmt"]) for _, d in parts] + [K.fmt_value(total, r["fmt"])],
        textposition="outside", cliponaxis=False,
        increasing=dict(marker_color=good), decreasing=dict(marker_color=bad),
        totals=dict(marker_color="#57534e"), connector=dict(line=dict(color=style.LINE, width=1))))
    fig.update_layout(showlegend=False, title=f"Where the change in {metric} comes from", waterfallgap=0.35)
    lo, hi = min(b, total, *(b + d for _, d in parts)), max(b, total, *(b + d for _, d in parts))
    pad = (hi - lo) * 0.25 or abs(b) * 0.1 or 1
    fig.update_yaxes(range=[lo - pad if lo - pad > 0 or lo < 0 else 0, hi + pad])
    st.plotly_chart(dark(fig, 320), width="stretch", key="wi_waterfall")


def _moved_bars(rows: list[dict]):
    moved = [(r, d) for r in rows if (d := _delta(r)) is not None and abs(d) > 1e-9 and _num(r["base"])]
    if not moved:
        return
    moved.sort(key=lambda t: abs(t[1] / _num(t[0]["base"])))
    pct = [100 * d / abs(_num(r["base"])) for r, d in moved]
    color = [{"up": style.GREEN, "down": style.RED}.get(_tone(r, d), "#8a8580") for r, d in moved]
    names = [f"{r['name']}<br><span style='font-size:11px;color:{style.INK_FAINT}'>"
             f"{K.fmt_value(r['base'], r['fmt'])} → {K.fmt_value(r['scen'], r['fmt'])}</span>" for r, _ in moved]
    fig = go.Figure(go.Bar(
        x=pct, y=names, orientation="h", marker_color=color, text=[f"{v:+.1f}%" for v in pct],
        textposition="outside", cliponaxis=False, customdata=[r["name"] for r, _ in moved],
        hovertemplate="%{customdata}: %{x:+.1f}% vs baseline<extra></extra>"))
    fig.update_layout(title="Every number that moved (% change vs baseline)", bargap=0.35)
    reach = max(abs(v) for v in pct) * 1.3 or 1
    fig.update_xaxes(zeroline=True, zerolinecolor="#57534e", ticksuffix="%", showgrid=True, gridcolor=style.LINE,
                     range=[-reach if min(pct) < 0 else -reach * 0.08, reach if max(pct) > 0 else reach * 0.08])
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(dark(fig, max(220, 46 * len(moved) + 90)), width="stretch", key="wi_moved")
    st.caption("Green = better, red = worse, grey = no better/worse direction. Under each name: baseline → scenario.")


def _chain(casc: dict):
    """The cascade as a strip of chips, one per step, so the ripple reads left to right."""
    chips = []
    for r in casc["steps"].itertuples():
        row = {"name": r.metric, "base": r.baseline, "scen": r.scenario, "fmt": "pct" if "%" in r.metric else "number",
               "better": _better(r.metric)}
        d = _delta(row)
        tone = _tone(row, d) if d is not None else ""
        dtxt = (f"was {K.fmt_value(r.baseline, row['fmt'])} ({_fmt_delta(d, row['fmt'])})" if d is not None and abs(d) > 1e-9
                else "unchanged")
        chips.append(f'<div class="chip {tone}"><div class="k">{r.metric}</div>'
                     f'<div class="v">{K.fmt_value(r.scenario, row["fmt"])}</div><div class="d">{dtxt}</div></div>')
    st.markdown('<div class="chain">' + '<span class="arrow">&rarr;</span>'.join(chips) + "</div>",
                unsafe_allow_html=True)


def current_params(extra: Params | None = None) -> Params:
    parts = [st.session_state.get("slider_params", Params())] + [c["params"] for c in st.session_state.get("constraints", [])]
    return combine(parts + ([extra] if extra else []))


def run_scenario(params: Params) -> dict:
    df, ctx, s = state.clean_df(), state.kpi_ctx(), setup()
    scen, notes = apply_rows(df, params)
    plan = [it for it in st.session_state.get("kpi_plan", []) if not K.validate(it, df, ctx)]
    base_k, scen_k = K.run_plan(plan, df, ctx), K.run_plan(plan, scen, ctx)
    casc = cascade(df, scen, s, params) if not s.cascade_ready() else None
    from ui import route_tab

    return {"notes": notes, "base": base_k, "scen": scen_k, "cascade": casc, "rows": (len(df), len(scen)),
            "route": route_tab.analyse(scen, params), "route_base": route_tab.analyse(df, Params())}


def snapshot(res: dict) -> dict[str, float | str]:
    out = {f"KPI · {k.title}": k.value for k in res["scen"] if k.chart is None or k.calc in ("top_n",)}
    out.update(RC.snapshot(res.get("route")))
    if res["cascade"]:
        for r in res["cascade"]["steps"].itertuples():
            out[f"Cascade · {r.metric}"] = r.scenario
        out["Cascade · feasibility"] = res["cascade"]["verdict"]
    return out


def log_change(text: str, interpretation: str, params: Params, before: dict, after: dict):
    log = st.session_state.setdefault("change_log", [])
    n = len({e["#"] for e in log}) + 1
    log.extend(change_rows(n, datetime.now().strftime("%H:%M:%S"), text, interpretation, params, before, after))


def interpret(text: str):
    df, ctx, s = state.clean_df(), state.kpi_ctx(), setup()
    return parse(text, df, ctx.roles, s, st.session_state.profile, f"{state.data_hash()}:{text.lower()}",
                 ctx.entity_of, use_ai=llm.status()[0])


def apply_constraint(pend):
    """Stack a confirmed constraint and log what it moved."""
    st.session_state.setdefault("constraints", [])
    before = snapshot(run_scenario(current_params()))
    after = snapshot(run_scenario(current_params(pend.params)))
    st.session_state.constraints.append({"text": pend.text, "params": pend.params,
                                         "interpretation": pend.interpretation, "source": pend.source})
    log_change(pend.text, pend.interpretation, pend.params, before, after)


def constraint_box(prefix: str, hint: str = ""):
    """Typed constraint → confirm → stack. Shown on What-if and Route cost; both share one constraint list."""
    st.session_state.setdefault("constraints", [])
    with st.form(f"constraint_{prefix}", clear_on_submit=True):
        c1, c2 = st.columns([6, 1], vertical_alignment="bottom")
        text = c1.text_input(hint or "Type a constraint card, e.g. 'Hub H02 is down for 3 days', 'fee rise capped "
                             "at 5%', 'ignore cancelled orders', 'tolls of Rs 40 per order on East routes, unverified'")
        go = c2.form_submit_button("Interpret", type="primary", width="stretch")
    ai_ok, ai_msg = llm.status()
    if not ai_ok:
        st.caption(f"AI-down mode: common wordings are parsed by rules; others need AI. ({ai_msg})")
    if go and text.strip():
        with st.spinner("Translating constraint into parameters..."):
            st.session_state.pending = interpret(text.strip())
    pend = st.session_state.get("pending")
    if pend:
        with st.container(border=True):
            if pend.ok:
                st.markdown(f"**Is this what you meant?** {pend.interpretation}")
                with st.expander(f"Parameters ({'rule-based parser' if pend.source == 'rules' else 'AI parser'})"):
                    st.json(pend.params.model_dump(exclude_defaults=True))
                row = st.container(horizontal=True)
                if row.button("Apply", type="primary", key=f"{prefix}_apply"):
                    apply_constraint(pend)
                    st.session_state.pending = None
                    st.rerun()
                if row.button("Discard", key=f"{prefix}_discard"):
                    st.session_state.pending = None
                    st.rerun()
            else:
                st.error(f"Cannot apply “{pend.text}”: {pend.reason}")
                for e in pend.errors:
                    st.caption(f"• {e}")
                if pend.suggestion:
                    st.info(f"Try: {pend.suggestion}")
                if st.button("Dismiss", key=f"{prefix}_dismiss"):
                    st.session_state.pending = None
                    st.rerun()

    if st.session_state.constraints:
        style.label(f"Active constraints ({len(st.session_state.constraints)})",
                    "Constraints stack in order: each one is applied on top of the ones before it. "
                    "Remove one to recompute without it.")
        for i, c in enumerate(st.session_state.constraints):
            c1, c2 = st.columns([8, 1], vertical_alignment="center")
            c1.markdown(f"{i + 1}. {c['text']} — *{c['interpretation']}*")
            if c2.button("Remove", key=f"{prefix}_rmc_{i}", type="tertiary"):
                st.session_state.constraints.pop(i)
                st.rerun()


# ---------- tab ----------

def render():
    df, ctx, s = state.clean_df(), state.kpi_ctx(), setup()
    _setup_editor(s, df, ctx)
    style.heading("New constraint",
                  "Type a change in plain English. The app turns it into parameters and shows them to you; nothing "
                  "changes until you click **Apply**. Common wordings are understood without AI.\n\n"
                  "Examples:\n- Hub H02 is down for 3 days\n- Fee rise capped at 5%\n- Ignore cancelled orders\n"
                  "- Tolls of Rs 40 per order on East routes, unverified")
    constraint_box("wi")

    st.divider()
    style.heading("Levers and impact",
                  "Move a lever on the left; the right side updates straight away.\n\n"
                  "- Under each moved lever: **what that lever does on its own** (its two biggest moves).\n"
                  "- **Tiles**: the headline numbers now, with the change from baseline (green = better, red = worse).\n"
                  "- **Waterfall**: how much of the total change in one number comes from each lever and each typed "
                  "constraint. *Overlap* is the part that only appears when changes act together "
                  "(e.g. two cuts that both push sites over capacity).\n\n"
                  "Levers stack with typed constraints. The Route cost and Recommendations pages use the same scenario.")
    left, right = st.columns([5, 8], gap="large")
    with left, st.container(border=True):
        moved, assume = levers(s, ctx)

    params = current_params()
    res = run_scenario(params)
    st.session_state.scenario_res = (params, res)
    rows = _metric_rows(_values(res, base=True), _values(res, base=False))

    # every lever and typed constraint on its own (plus the shared assumptions), for effect lines + the waterfall
    solo = []
    for name, p, slot in moved:
        srows = _metric_rows(_values(res, base=True), _values(_quick(combine([assume, p])), base=False))
        slot.markdown(_effect_line(srows), unsafe_allow_html=True)
        solo.append((name, srows))
    for i, c in enumerate(st.session_state.get("constraints", []), 1):
        srows = _metric_rows(_values(res, base=True), _values(_quick(combine([assume, c["params"]])), base=False))
        solo.append((f"Constraint {i}", srows))

    with right:
        style.eyebrow("Impact")
        _tiles(rows, res["cascade"])
        if not solo:
            st.info("Nothing is changed yet. Move a lever on the left or apply a constraint above, and this panel "
                    "shows what moves and why.")
        else:
            movable = [r["name"] for r in rows if (d := _delta(r)) is not None and abs(d) > 1e-9]
            if movable:
                default = next((h for h in HEADLINE if h in movable), movable[0])
                metric = st.selectbox("Explain the change in", movable, index=movable.index(default), key="wi_metric")
                _waterfall(metric, rows, solo)
            else:
                st.caption("The current changes do not move any of the numbers tracked here.")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"Baseline: {res['rows'][0]:,} rows (cleaned data) · scenario: {res['rows'][1]:,} rows",
                       width="content")
            if res["notes"]:
                style.info("\n".join(f"- {n}" for n in res["notes"]))

    if solo:
        _moved_bars(rows)
    if st.button("Record current lever settings in the Change Log"):
        sp = st.session_state.get("slider_params", Params())
        before = snapshot(run_scenario(combine([c["params"] for c in st.session_state.constraints])))
        log_change("What-if levers", "lever settings at time of recording", sp, before, snapshot(res))
        st.success("Recorded.")

    st.divider()
    style.heading("Cascade",
                  "One change ripples through: **capacity → allocation → utilisation → time → on-time → cost & "
                  "margin → feasibility**. Each chip is one step: the scenario value, and what it was. "
                  "Any change recomputes every later step. The table below has the formula for each.")
    casc = res["cascade"]
    if casc is None:
        st.info(f"Cascade needs: {', '.join(s.cascade_ready())}. Set them in Model setup above.")
    else:
        getattr(st, VERDICT_STYLE[casc["verdict"]])(f"**{casc['verdict']}** — " + "; ".join(casc["reasons"]))
        _chain(casc)
        steps = casc["steps"].copy()
        for c in ("baseline", "scenario", "change"):
            steps[c] = steps[c].map(lambda v: f"{v:,.2f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else "")
        with st.expander("Cascade table with formulas"):
            st.dataframe(steps, hide_index=True, width="stretch")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption("All scenario values are estimates based on assumptions.", width="content")
            style.info(f"Period {casc['days']} days" + (f", capacity change on {casc['affected_days']:g} of them"
                                                        if casc["affected_days"] else "") +
                       f" · load basis: **{casc['basis']}** · {casc['calibration']}. "
                       "See **Assumptions** below for every value assumed.")
        if len(casc.get("targets", [])):
            st.markdown("**Targets**")
            st.dataframe(casc["targets"], hide_index=True, width="stretch")
        with st.expander("Per-entity view (calculated vs reported utilisation shown side by side)"):
            st.dataframe(casc["entities"], hide_index=True, width="stretch")
        with st.expander("Assumptions"):
            for k, v in casc["assumptions"].items():
                st.markdown(f"- **{k}:** {v}")

    st.divider()
    style.heading("KPIs: baseline vs scenario",
                  "KPIs reflect row-level changes (exclusions, % changes, price changes). Capacity effects appear "
                  "in the cascade above.")
    rows = [{"KPI": b.title, "baseline": K.fmt_value(b.value, b.fmt), "scenario": K.fmt_value(sc.value, sc.fmt),
             "change": (K.fmt_value(sc.value - b.value, b.fmt) if b.value is not None and sc.value is not None else "n/a"),
             "basis": b.basis} for b, sc in zip(res["base"], res["scen"]) if b.chart is None or b.calc == "top_n"]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_changelog():
    log = st.session_state.get("change_log", [])
    if not log:
        st.info("No changes recorded yet. Apply a constraint on the What-if page.")
        return
    df = pd.DataFrame(log)
    st.download_button("Export Change Log (CSV)", df.to_csv(index=False).encode(), "change_log.csv")
    for n, g in df.groupby("#", sort=False):
        with st.container(border=True):
            st.markdown(f"**#{n} · {g['time'].iloc[0]} · {g['constraint'].iloc[0]}**  \n*{g['interpretation'].iloc[0]}*")
            with st.expander("Parameters"):
                st.code(g["parameters"].iloc[0], language="json")
            moved = g[g["change"].astype(str).ne("") & g["change"].astype(str).ne("0") & g["change"].astype(str).ne("0.0")]
            show = moved if len(moved) else g
            st.dataframe(show[["metric", "before", "after", "change"]].astype(str), hide_index=True, width="stretch")
