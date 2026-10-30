"""Route cost card: base vs full cost per route, component statuses, three rankings, robustness verdict,
typed cost constraints, Explain + Recommendations, and a second file (data/routes_test.csv) with no code changes."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from engine import recommend as R
from engine import routecost as RC
from engine.constraints import parse
from engine.explain import build
from engine.pipeline import prepare
from engine.scenario import CostComponent, Params, combine

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ROUTES = DATA / "routes_test.csv"


@pytest.fixture(scope="module")
def buds():
    run = prepare(DATA / "Round_1_BUDS.xlsx")
    return run, RC.guess(run.df, run.role_map, run.entity_of, run.setup)


@pytest.fixture(scope="module")
def routes():
    if not ROUTES.exists():
        pytest.skip("run scripts/make_synthetic.py")
    run = prepare(ROUTES)
    return run, RC.guess(run.df, run.role_map, run.entity_of, run.setup)


def _parse(run, text):
    p = parse(text, run.df, run.role_map, run.setup, run.profile, f"t:{text}", run.entity_of, use_ai=False)
    assert p.ok, (text, p.reason, p.errors)
    return p.params


def test_buds_hub_by_zone_routes_with_failed_and_return_costs_from_statuses(buds):
    run, rs = buds
    assert rs.route_cols[0] == run.setup.entity and len(rs.route_cols) == 2  # entity × a zone-like location
    assert rs.status == run.setup.status and rs.failed_values and rs.returned_values
    r = RC.compute(run.df, rs)
    comp = r["components"].set_index("component")
    d = run.df.dropna(subset=rs.route_cols)
    base = pd.to_numeric(d[rs.base_cost])
    st = d[rs.status].astype("string")
    assert comp.at["failed attempts", "total"] == pytest.approx(base[st.isin(rs.failed_values)].sum())
    assert comp.at["return journeys", "total"] == pytest.approx(base[st.isin(rs.returned_values)].sum())
    assert comp.at["failed attempts", "status"] == "confirmed"
    assert "PROXY" in comp.at["waiting", "source"]  # no wait column in BUDS: labelled proxy
    assert r["totals"]["hidden_all"] == pytest.approx(comp["total"].sum() - comp.at["base cost", "total"])


def test_unverified_toll_changes_rank_and_exclusion_restores_confirmed_ranking(buds):
    run, rs = buds
    r0 = RC.compute(run.df, rs)
    w = r0["winners"]["all"]
    zone_col = rs.route_cols[1]
    zone = w.split(RC.SEP)[1]
    toll = Params(cost_components=[CostComponent(name="toll", value=40, segment_column=zone_col, segment_value=zone)])
    r1 = RC.compute(run.df, rs, toll)
    t1 = r1["routes"].set_index("route")
    assert t1.at[w, "all rank"] > 1 and t1.at[w, "confirmed rank"] == 1  # the toll moved the winner down
    assert r1["verdict"] == "Fragile" and "toll" in r1["verdict_text"] and "break-even" in r1["verdict_text"]
    # break-even: the toll value at which the two routes tie under the full-cost view
    be = float(r1["flips"][-1]["break-even"].split(" of ")[1].split(" per")[0])
    alt = r1["winners"]["all"]
    t0 = r0["routes"].set_index("route")
    gap = t0.at[alt, "confirmed per order"] - t0.at[w, "confirmed per order"]
    assert be == pytest.approx(gap, abs=0.01)

    r2 = RC.compute(run.df, rs, combine([toll, _parse(run, "Exclude unverified costs")]))
    t2 = r2["routes"].set_index("route")
    assert r2["decision"] == r2["winners"]["confirmed"] == w
    ranked = t2["decision rank"].notna()
    assert (t2.loc[ranked, "decision rank"] == t2.loc[ranked, "confirmed rank"]).all()


def test_typed_cost_constraints(buds):
    run, _ = buds
    p = _parse(run, "Tolls of Rs 40 per order on East routes, unverified").cost_components[0]
    assert (p.name, p.value, p.per, p.status, p.segment_value) == ("toll", 40, "order", "unverified", "East")
    p = _parse(run, "Include waiting cost at Rs 2 per minute").cost_components[0]
    assert (p.name, p.value, p.per, p.status) == ("waiting", 2, "minute", "unverified")
    p = _parse(run, "Parking is confirmed at Rs 15 per order").cost_components[0]
    assert (p.name, p.value, p.status, p.segment_column) == ("parking", 15, "confirmed", None)
    assert _parse(run, "Exclude unverified costs").exclude_cost_status == ["unverified"]
    assert _parse(run, "Ignore cancelled orders").exclusions  # row exclusions still work
    bad = parse("Exclude confirmed costs", run.df, run.role_map, run.setup, run.profile, "b", run.entity_of,
                use_ai=False)
    assert not bad.ok and bad.suggestion


def test_routes_file_works_with_no_code_changes(routes):
    run, rs = routes
    assert rs.route_cols == ["route_id"] and rs.base_cost == "base_cost" and rs.wait_col == "wait_min"
    assert set(rs.cost_cols) == {"toll", "parking"} and rs.cost_status_col == "cost_status"
    r = RC.compute(run.df, rs)
    assert r["winners"]["base"] == r["winners"]["confirmed"] == "R03" and r["winners"]["all"] == "R05"
    assert r["verdict"] == "Fragile" and "toll" in r["flips"][-1]["component"]
    comp = r["components"].set_index("component")
    assert "unverified" in comp.at["toll", "status"] and "provisional" in comp.at["toll", "status"]
    assert comp.at["waiting", "total"] == 0  # minutes exist but no rate yet
    p = combine([_parse(run, t) for t in ("Include waiting cost at Rs 2 per minute",
                                          "Parking is confirmed at Rs 15 per order", "Exclude unverified costs")])
    r2 = RC.compute(run.df, rs, p)
    comp2 = r2["components"].set_index("component")
    assert comp2.at["waiting", "total"] == pytest.approx(2 * pd.to_numeric(run.df["wait_min"]).sum())
    assert comp2.at["parking", "status"] == "confirmed" and comp2.at["parking", "per order"] == pytest.approx(15)
    assert "decision" in r2["views"]

    recs = R.generate(R.Inputs(run.df, run.ctx, route=r))
    assert any(x.rule == "route_fragile" for x in recs)
    secs = build(name="routes", rows_raw=len(run.raw), rows_clean=len(run.df), roles=run.roles, entity_of={},
                 fixes=[], vmap={}, vmap_confirmed=False, kpis=[], pack=None, pack_metrics=None, route=r)
    sec = next(s for s in secs if s.title.startswith("9."))
    assert {"base cost", "toll", "waiting"} <= set(sec.table["component"]) and "source" in sec.table


def test_route_tab_in_app_logs_constraint(routes):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    at.radio(key="src").set_value("Use a file from the data folder").run()
    at.selectbox(key="src_file").set_value(ROUTES.name).run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("Fragile" in w.value for w in at.warning)
    box = next(t for t in at.text_input if t.label.startswith("Type a cost card"))
    box.input("Exclude unverified costs")
    next(b for b in at.button if b.label == "Interpret" and b.form_id == "constraint_rt").click().run()
    at.button(key="rt_apply").click().run()
    assert not at.exception, [e.message for e in at.exception]
    log = pd.DataFrame(at.session_state["change_log"])
    assert "Route · cheapest (decision view)" in set(log["metric"])
    assert "9. Route cost (cost analytics)" in [e.label for e in at.expander]
