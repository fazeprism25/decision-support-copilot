"""Route cost page (cost-analytics card): base vs full cost per route, three rankings side by side, hidden cost,
decision robustness verdict with break-even. Typed constraints share the What-if constraint list and Change Log."""
import pandas as pd
import plotly.express as px
import streamlit as st

from engine import routecost as RC
from engine.scenario import CostComponent, Params, apply_rows
from ui import state, style

HIGHLIGHT = "background-color: rgba(216, 162, 79, 0.18)"
CHEAPEST = "background-color: rgba(92, 179, 56, 0.24); font-weight: 600"
ENTRY_COLS = ["name", "value", "per", "status"]


def route_setup() -> RC.RouteSetup:
    if st.session_state.get("route_setup_for") != st.session_state.dataset_id:
        from ui import whatif_tab

        ctx = state.kpi_ctx()
        st.session_state.route_setup = RC.guess(state.clean_df(), ctx.roles, ctx.entity_of, whatif_tab.setup())
        st.session_state.route_setup_for = st.session_state.dataset_id
    return st.session_state.route_setup


def analyse(df: pd.DataFrame, params: Params) -> dict | None:
    """Route cost result for a (scenario) frame; used by What-if snapshots, Recommendations and Explain."""
    return RC.compute(df, route_setup(), params)


def _editor(rs: RC.RouteSetup, df: pd.DataFrame, roles: dict[str, str], entity_of: dict[str, str]):
    k = lambda n: f"rt_{n}_{st.session_state.dataset_id}"
    of = lambda *rr: [c for c, r in roles.items() if r in rr and c not in entity_of]

    def pick(where, label, opts, cur, key, none="(none)"):
        opts = [None] + opts
        return where.selectbox(label, opts, index=opts.index(cur) if cur in opts else 0, key=k(key),
                               format_func=lambda x: none if x is None else x)

    with st.expander("Route setup: grouping and cost columns (auto-filled, editable)", expanded=bool(rs.missing())):
        st.caption("Filled in automatically from the column roles. Hover a field's label for help.")
        labels = of("category", "location", "id")
        rs.route_cols = st.multiselect("Route = combination of (one route column, or e.g. hub × zone)", labels,
                                       [c for c in rs.route_cols if c in labels], key=k("cols"))
        c1, c2, c3 = st.columns(3)
        rs.base_cost = pick(c1, "Base cost per order (the number management ranks on)", of("money", "measure"),
                            rs.base_cost, "base")
        rs.status = pick(c2, "Delivery status column", of("category", "flag"), rs.status, "status")
        rs.min_orders = int(c3.number_input("Min orders to rank a route", 1, 10_000, rs.min_orders, key=k("min")))
        if rs.status:
            vals = sorted(df[rs.status].dropna().astype("string").unique().tolist())
            c1, c2 = st.columns(2)
            rs.failed_values = c1.multiselect("Failed-attempt value(s)", vals,
                                              [v for v in rs.failed_values if v in vals], key=k("fv"))
            rs.returned_values = c2.multiselect("Returned value(s)", vals,
                                                [v for v in rs.returned_values if v in vals], key=k("rv"))
        c1, c2 = st.columns(2)
        rs.reattempt_cost = c1.number_input("Re-attempt cost per failed order (0 = the order's own base cost)", 0.0,
                                            value=float(rs.reattempt_cost or 0), key=k("rc")) or None
        rs.return_cost = c2.number_input("Return-journey cost per returned order (0 = base cost again)", 0.0,
                                         value=float(rs.return_cost or 0), key=k("retc")) or None
        c1, c2, c3 = st.columns(3)
        rs.wait_col = pick(c1, "Waiting minutes column", of("duration", "measure", "quantity"), rs.wait_col, "wait",
                           none="(none: proxy = actual − expected)")
        rs.wait_rate = c2.number_input("Cost per waiting minute (0 = not costed)", 0.0, value=float(rs.wait_rate),
                                       key=k("wr"))
        rs.wait_status = c3.selectbox("Waiting cost status", RC.STATUSES, index=RC.STATUSES.index(rs.wait_status),
                                      key=k("ws"))
        c1, c2, c3 = st.columns(3)
        rs.cost_cols = c1.multiselect("Hidden-cost columns in the data (tolls, parking…)", of("money", "measure"),
                                      [c for c in rs.cost_cols if c in of("money", "measure")], key=k("cc"))
        rs.cost_status_col = pick(c2, "Cost verification column (confirmed/provisional/unverified)",
                                  of("category", "flag"), rs.cost_status_col, "csc")
        rs.distance = pick(c3, "Distance column (for per-km entries)", of("measure", "quantity"), rs.distance, "dist")

        style.label("Your cost entries",
                    "Add parking, tolls or anything else here. Each entry applies to **every route**; its default "
                    "status is *unverified*. For one place only, type it in the cost constraint box instead, e.g. "
                    "'tolls of Rs 40 per order on East routes, unverified'.")
        cur = pd.DataFrame([e.model_dump(include=set(ENTRY_COLS)) for e in rs.entries], columns=ENTRY_COLS)
        ed = st.data_editor(cur, num_rows="dynamic", key=k("entries"), hide_index=True, width="stretch",
                            column_config={"per": st.column_config.SelectboxColumn(options=["order", "km", "minute"]),
                                           "status": st.column_config.SelectboxColumn(options=list(RC.STATUSES)),
                                           "value": st.column_config.NumberColumn(min_value=0.0)})
        rs.entries = [CostComponent(name=str(r["name"]), value=float(r["value"]), per=r.get("per") or "order",
                                    status=r.get("status") or "unverified")
                      for r in ed.to_dict("records") if str(r.get("name") or "").strip() and pd.notna(r.get("value"))]
        keys = ["failed attempts", "return journey"] + [RC.comp_key(c) for c in rs.cost_cols]
        style.label("Status of data components",
                    "Costs read from the data count as *confirmed* by default. Set a status here once you know "
                    "better, e.g. 'confirmed' after you have verified a figure.")
        cols = st.columns(max(1, len(keys)))
        for col, ck in zip(cols, keys):
            opts = ["from data"] + list(RC.STATUSES)
            v = col.selectbox(ck, opts, index=opts.index(rs.status_override.get(ck, "from data")), key=k(f"so_{ck}"))
            if v == "from data":
                rs.status_override.pop(ck, None)
            else:
                rs.status_override[ck] = v


def _style(tab: pd.DataFrame, views: dict, winners: dict):
    show = tab.copy()
    cols = ["route", "orders"] + [c for v in views for c in (f"{v} per order", f"{v} rank")] + \
        ["hidden % of base", "rank changes"]
    show = show[cols]

    def row_style(r):
        out = [HIGHLIGHT if r["rank changes"] else ""] * len(r)
        for v in views:
            if r["route"] == winners[v]:
                out[cols.index(f"{v} per order")] = CHEAPEST
        return out

    fmt = {c: "{:,.2f}" for c in cols if c.endswith("per order")} | {c: "{:.0f}" for c in cols if c.endswith("rank")}
    fmt["hidden % of base"] = "{:.1f}%"
    return show.style.apply(row_style, axis=1).format(fmt, na_rep="not ranked")


def render():
    from ui import whatif_tab

    df, ctx, rs = state.clean_df(), state.kpi_ctx(), route_setup()
    _editor(rs, df, ctx.roles, ctx.entity_of)
    style.heading("Cost constraint",
                  "Type a cost change in plain English; you confirm before it applies. It shares one list with the "
                  "What-if page and is logged in the Change Log.\n\nExamples:\n"
                  "- Tolls of Rs 40 per order on East routes, unverified\n- Include waiting cost at Rs 2 per minute\n"
                  "- Parking is confirmed at Rs 15 per order\n- Exclude unverified costs")
    whatif_tab.constraint_box("rt", "Type a cost card, e.g. 'Tolls of Rs 40 per order on East routes, unverified', "
                                    "'Include waiting cost at Rs 2 per minute', 'Parking is confirmed at Rs 15 per "
                                    "order', 'Exclude unverified costs'")
    params = whatif_tab.current_params()
    scen, _ = apply_rows(df, params)
    r = RC.compute(scen, rs, params)
    if r is None:
        st.info(f"Route cost needs: {', '.join(rs.missing()) or 'valid columns'}. Set them in Route setup above.")
        return

    (st.success if r["verdict"] == "Robust" else st.warning)(f"**{r['verdict_text']}**")
    views, w, t = r["views"], r["winners"], r["totals"]
    cols = st.columns(len(views))
    for col, v in zip(cols, views):
        col.metric(f"Cheapest · {views[v]}", w[v])
    c1, c2, c3 = st.columns(3)
    c1.metric("Under-estimated by base cost (all statuses)", f"{t['hidden_all']:,.2f}",
              help="Σ full cost − Σ base cost over every order")
    c2.metric("…of which confirmed", f"{t['hidden_confirmed']:,.2f}",
              help="Only the hidden costs with status 'confirmed'.")
    c3.metric("Hidden cost % of base", f"{t['hidden_pct'] or 0:.1f}%",
              help="How much the base cost under-states the full cost, as a % of base cost.")

    st.divider()
    style.heading("Rankings side by side",
                  "Cost per order for each route under three views (**base** only, **confirmed** costs, **all** "
                  "costs); rank 1 = cheapest.\n\nAmber rows: the rank changes between views.\n\n"
                  "Green cell: the cheapest route in that view.\n\n" + " ".join(r["notes"]))
    st.dataframe(_style(r["routes"], views, w), hide_index=True, width="stretch")
    if r["flips"]:
        style.label("What flips the decision",
                    "Each row is a cost component that, if it changes by the amount shown (break-even), would "
                    "make a different route the cheapest.")
        st.dataframe(pd.DataFrame(r["flips"]), hide_index=True, width="stretch")

    st.divider()
    style.heading("Cost components", "Every cost that makes up the full cost, with its total, per-order value "
                  "and status (confirmed / provisional / unverified).")
    st.dataframe(r["components"].style.format({"total": "{:,.2f}", "per order": "{:,.2f}"}), hide_index=True,
                 width="stretch")
    ranked = r["routes"].dropna(subset=[f"{'decision' if r['excluded'] else 'all'} rank"])["route"].head(10).tolist()
    by = r["by_route"]
    top = by[by["route"].isin(ranked)].groupby(["route", "component"], sort=False)["per order"].sum().reset_index()
    if len(top):
        fig = px.bar(top, x="route", y="per order", color="component", category_orders={"route": ranked},
                     title="Cost per order by component (10 cheapest ranked routes, all statuses)")
        fig.update_layout(height=380, margin=dict(t=50, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    with st.expander("One route: every component and status"):
        pick = st.selectbox("Route", r["routes"]["route"].tolist(), key="rt_pick")
        one = by[by["route"] == pick][["component", "status", "per order", "amount"]]
        st.dataframe(one.style.format({"per order": "{:,.2f}", "amount": "{:,.2f}"}), hide_index=True,
                     width="stretch")
