"""KPIs tab: AI- or rule-chosen plan, computed KPI cards with explain popovers, charts, KPI builder."""
import hashlib

import streamlit as st

from engine import kpis as K
from engine import llm
from engine.domains import delivery as D
from engine.llm import LLMError, chat_json
from engine.profile import to_datetime_clean
from ui import state
from ui.charts import figure


def explain_md(k: K.KPI) -> str:
    lines = [f"**{k.title}** = {K.fmt_value(k.value, k.fmt)} · *{k.basis}*",
             f"- **Formula:** {k.formula}", f"- **Filter:** {k.filter}", f"- **Denominator:** {k.denominator}"]
    if k.extra:
        lines.append(f"- **Note:** {k.extra}")
    lines += [f"- **Excluded:** {e}" for e in k.exclusions]
    lines += [f"- ⚠️ {w}" for w in k.warnings]
    if st.session_state.get("fixes"):
        lines.append(f"- Computed after {len(st.session_state.fixes)} quality fix(es) (see Quality tab).")
    return "\n".join(lines)


def _columns_for_prompt(ctx: K.Ctx) -> list[dict]:
    prof, mean = st.session_state.profile, state.meanings()
    out = []
    for c, r in ctx.roles.items():
        p = prof[c]
        d = {"column": c, "role": r, "meaning": mean.get(c, ""), "distinct": p["n_unique"]}
        if "top_values" in p and p["n_unique"] <= 20:  # labels as the data now has them (after value mapping)
            d["labels"] = list(dict.fromkeys(str(state.vmap().get(c, {}).get(v, v)) for v in p["top_values"]))
        if c in ctx.entity_of:
            d["entity_attribute_of"] = ctx.entity_of[c]
        out.append(d)
    return out


def _ai_plan(df, ctx) -> tuple[list[dict], str]:
    sig = hashlib.sha1(str(state.roles_sig()).encode()).hexdigest()[:10]
    raw = chat_json(K.plan_prompt(_columns_for_prompt(ctx)), tier="fast",
                    cache_key=f"kpis:{st.session_state.dataset_id}:{sig}",
                    system="You select analytics calculations from a fixed catalog. Output strict JSON only.")
    items = raw.get("kpis", []) if isinstance(raw, dict) else []
    plan, rejected, fixed = [], [], 0
    pk = state.pack()
    for it in items:
        if not isinstance(it, dict):
            continue
        p = it.get("params") or {}
        if pk.active and it.get("calc") == "margin" and (p.get("revenue"), p.get("cost")) != (pk.cols["fee"], pk.cols["cost"]):
            it = {**it, "params": {"revenue": pk.cols["fee"], "cost": pk.cols["cost"]},
                  "title": f"Delivery margin ({pk.cols['fee']} − {pk.cols['cost']})"}
            fixed += 1
        errs = K.validate(it, df, ctx)
        (rejected.append(f"{it.get('calc')}: {'; '.join(errs)}") if errs else plan.append(it))
    if not plan:
        raise LLMError("AI returned no valid KPI choices.")
    msg = f"AI chose {len(plan)} KPIs." + (f" Rejected {len(rejected)} invalid choice(s): " + " | ".join(rejected)
                                           if rejected else "") +         (f" Corrected {fixed} margin KPI(s) to fee − cost (order value is not revenue)." if fixed else "")
    return plan, msg


def _pack_items() -> list[dict]:
    pk = state.pack()
    if not pk.active:
        return []
    c = pk.cols
    ot = {"actual": c["actual"], "promised": c["promised"]}
    if c["status"] and pk.completed:
        ot |= {"status": c["status"], "completed_values": pk.completed}
    return [{"calc": "on_time_rate", "params": ot, "title": "On-time rate"},
            {"calc": "margin", "params": {"revenue": c["fee"], "cost": c["cost"]},
             "title": f"Delivery margin ({c['fee']} − {c['cost']})"}]


def _make_plan(df, ctx, use_ai: bool):
    _make_plan_inner(df, ctx, use_ai)
    have = {it.get("calc") for it in st.session_state.kpi_plan}
    st.session_state.kpi_plan += [it for it in _pack_items() if it["calc"] not in have]


def _make_plan_inner(df, ctx, use_ai: bool):
    if use_ai and llm.status()[0]:
        try:
            with st.spinner("AI is choosing KPIs..."):
                plan, msg = _ai_plan(df, ctx)
            st.session_state.kpi_plan, st.session_state.kpi_msg = plan, ("info", msg)
            return
        except LLMError as e:
            st.session_state.kpi_msg = ("warning", f"AI KPI choice unavailable, using default KPIs. ({str(e)[:150]})")
    else:
        st.session_state.kpi_msg = ("info", "Default KPIs chosen by rules from the column roles.")
    st.session_state.kpi_plan = K.default_plan(df, ctx)


def _date_warning(df, ctx):
    for c in [c for c, r in ctx.roles.items() if r == "date"]:
        d = to_datetime_clean(df[c]).dropna()
        if len(d):
            span = (d.max().normalize() - d.min().normalize()).days + 1
            if span < K.MIN_TREND_DAYS:
                st.warning(f"'{c}' covers only {span} days ({d.min().date()} → {d.max().date()}). "
                           f"Trends and growth over fewer than {K.MIN_TREND_DAYS} days are not reliable.")
            else:
                st.caption(f"'{c}' covers {span} days ({d.min().date()} → {d.max().date()}).")
            return


def render():
    df, ctx = state.clean_df(), state.kpi_ctx()
    if st.session_state.get("kpi_plan_for") != st.session_state.dataset_id:
        st.session_state.kpi_plan_for = st.session_state.dataset_id
        _make_plan(df, ctx, use_ai=True)

    b1, b2, _ = st.columns([1.3, 1.3, 4])
    if b1.button("Let AI choose KPIs", disabled=not llm.status()[0]):
        _make_plan(df, ctx, use_ai=True)
    if b2.button("Use default KPIs"):
        _make_plan(df, ctx, use_ai=False)
    kind, msg = st.session_state.get("kpi_msg", ("info", ""))
    if msg:
        getattr(st, kind)(msg)
    _date_warning(df, ctx)
    _pack_section()

    plan = st.session_state.kpi_plan
    valid = [(i, it) for i, it in enumerate(plan) if not K.validate(it, df, ctx)]
    for i, it in enumerate(plan):
        if (errs := K.validate(it, df, ctx)):
            st.caption(f"Skipped '{it.get('title') or it['calc']}': {'; '.join(errs)}")
    results = [(i, k) for (i, _), k in zip(valid, K.run_plan([it for _, it in valid], df, ctx))]
    st.session_state.kpi_results = [k for _, k in results]

    cards = [(i, k) for i, k in results if k.chart is None]
    for row in range(0, len(cards), 4):
        cols = st.columns(4)
        for col, (i, k) in zip(cols, cards[row:row + 4]):
            with col.container(border=True):
                _card(i, k)

    charts = [(i, k) for i, k in results if k.chart is not None]
    for row in range(0, len(charts), 2):
        cols = st.columns(2)
        for col, (i, k) in zip(cols, charts[row:row + 2]):
            with col.container(border=True):
                st.plotly_chart(figure(k.chart, k.table, k.title), width="stretch", key=f"ch_{i}")
                _card(i, k, compact=True)

    _builder(df, ctx)


def _pack_section():
    pk = state.pack()
    if not pk.active:
        st.caption(f"Delivery-ops pack {pk.reason}.")
        return
    m = state.pack_metrics()
    with st.expander("🚚 Delivery-ops pack: on-time, promise gap, margin per order, hub load", expanded=True):
        for note in m["notes"]:
            st.caption(note)
        heads = m["heads"]
        for row in range(0, len(heads), 4):
            cols = st.columns(4)
            for col, h in zip(cols, heads[row:row + 4]):
                with col.container(border=True):
                    st.metric(h.label, D.fmt(h))
                    st.caption(f"{h.basis} · n = {h.n:,}")
                    with st.popover("Explain"):
                        st.markdown(f"- **Formula:** {h.formula}\n- **Filter:** {h.filter}\n- **Rows:** {h.n:,}\n"
                                    f"- **Basis:** {h.basis}")
        seg = m["segments"]
        tabs = st.tabs([f"On-time by {pk.cols[k]}" for k in seg])
        for t, (k, tbl) in zip(tabs, seg.items()):
            with t:
                st.plotly_chart(figure({"type": "bar", "x": pk.cols[k], "y": "on-time %"}, tbl,
                                       f"On-time % by {pk.cols[k]} (calculated)"), width="stretch", key=f"pk_{k}")
                st.dataframe(tbl.round(2), hide_index=True, width="stretch")
        hub = m["hub"]
        st.markdown("**Hub load model** — reported and calculated figures side by side")
        st.dataframe(hub["table"], hide_index=True, width="stretch")
        st.caption(f"Model: {hub['calibration']}. Wait multiplier uses {hub.get('util_basis', '')} utilisation. "
                   "[reported] = value in the file · [calculated] = computed here · [model] = estimate based on "
                   "assumptions.")
        if hub.get("mismatch"):
            st.warning(hub["mismatch"])


def _card(i: int, k: K.KPI, compact: bool = False):
    if not compact:
        st.metric(k.title, K.fmt_value(k.value, k.fmt))
    st.caption(f"{k.basis}" + (f" · {k.extra}" if k.extra else "") + (" · ⚠️ see explain" if k.warnings else ""))
    c1, c2 = st.columns([3, 1])
    with c1.popover("Explain"):
        st.markdown(explain_md(k))
        if k.table is not None and k.chart and k.chart["type"] != "hist":
            st.dataframe(k.table, hide_index=True)
    if c2.button("✕", key=f"rm_{i}", help="Remove this KPI"):
        st.session_state.kpi_plan.pop(i)
        st.rerun()


def _builder(df, ctx: K.Ctx):
    with st.expander("➕ Add a KPI"):
        calc = st.selectbox("Calculation", list(K.CATALOG), format_func=lambda c: f"{c} — {K.CATALOG[c]['desc']}")
        params = {}
        for name, rule in K.CATALOG[calc]["params"].items():
            opt = rule.get("optional")
            key = f"b_{calc}_{name}"
            if "roles" in rule:
                cols = [c for c, r in ctx.roles.items() if r in rule["roles"]]
                params[name] = st.selectbox(name, ([None] if opt else []) + cols, key=key)
            elif "choices" in rule:
                params[name] = st.selectbox(name, rule["choices"], key=key)
            elif rule.get("int"):
                params[name] = int(st.number_input(name, 1, 50, 5, key=key))
            elif rule.get("bool"):
                params[name] = st.checkbox(f"{name} (bottom instead of top)", key=key)
            else:
                src = params.get(rule.get("value_of") or rule.get("values_of"))
                vals = sorted(df[src].dropna().astype("string").unique().tolist())[:200] if src else []
                params[name] = (st.multiselect(name, vals, key=key) if "values_of" in rule
                                else st.selectbox(name, vals, key=key))
        title = st.text_input("Title (optional)", key=f"b_{calc}_title")
        item = {"calc": calc, "params": params, "title": title}
        if st.button("Add KPI"):
            errs = K.validate(item, df, ctx)
            if errs:
                st.error("; ".join(errs))
            else:
                st.session_state.kpi_plan.append(item)
                st.rerun()
