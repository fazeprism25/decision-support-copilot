"""Phase 7: C01 value mapping, delivery-ops pack, rule-based recommendations, Explain content."""
from pathlib import Path

import pandas as pd
import pytest

from engine import explain
from engine import kpis as K
from engine import recommend as R
from engine.domains import delivery as D
from engine.pipeline import prepare
from engine.quality import find_issues, quality_score
from engine.scenario import CapacityChange, Params, Target, apply_rows, cascade
from engine.valuemap import apply_value_map, from_table, remap_plan, to_table

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def unknown():
    if not (DATA / "BUDS_unknown_schema_test.csv").exists():
        pytest.skip("acceptance file missing")
    return prepare(DATA / "BUDS_unknown_schema_test.csv")


def _inputs(run, df, p=Params(), scenario=False):
    iss = find_issues(df, run.roles, set(run.entity_of))
    q, _ = quality_score(df, iss)
    pm = D.metrics(df, run.pack, run.entity_of, run.role_map) if run.pack.active else None
    cz = cascade(run.df, df, run.setup, p) if not run.setup.cascade_ready() else None
    return R.Inputs(df, run.ctx, iss, q, K.run_plan(run.plan, df, run.ctx), run.pack, pm, cz, is_scenario=scenario)


def test_value_mapping_proposal_unknown_schema(unknown):
    vm = unknown.vmap
    pri = next(c for c, m in vm.items() if "P1" in m)
    stat = next(c for c, m in vm.items() if "rto" in m)
    peak = next(c for c, m in vm.items() if "Y" in m)
    assert vm[pri] == {"P1": "Critical", "P2": "Express", "P3": "Standard"}
    assert vm[stat]["completed"] == "Delivered" and vm[stat]["rto"] == "Returned"
    assert vm[peak]["Y"] == "Peak"
    assert set(unknown.df[pri].dropna().unique()) == {"Critical", "Express", "Standard"}  # feeds the shared frame


def test_value_map_table_roundtrip_and_plan_remap():
    df = pd.DataFrame({"s": ["done", "rto", "done", None]})
    vm = {"s": {"done": "Delivered", "rto": "Returned"}}
    out = apply_value_map(df, vm)
    assert out["s"].tolist()[:3] == ["Delivered", "Returned", "Delivered"] and pd.isna(out["s"].iloc[3])
    assert from_table(to_table(vm, df)) == vm
    plan = [{"calc": "on_time_rate", "params": {"status": "s", "completed_values": ["done"]}}]
    assert remap_plan(plan, vm)[0]["params"]["completed_values"] == ["Delivered"]


def test_weak_yes_no_needs_peak_name():
    df = pd.DataFrame({"is_vip": ["Y", "N"] * 20, "rush_hour": ["Y", "N"] * 20})
    slots = D.find_slots(df, {"is_vip": "flag", "rush_hour": "flag"})
    assert slots == {"peak": "rush_hour"}


def test_pack_on_for_delivery_with_acceptance_numbers(unknown):
    pk = unknown.pack
    assert pk.active, pk.reason
    assert (pk.cols["fee"], pk.cols["cost"], pk.cols["order_value"]) == ("shipping_charge", "last_mile_cost", "basket_rs")
    m = D.metrics(unknown.df, pk, unknown.entity_of, unknown.role_map)
    h = {x.label: x for x in m["heads"]}
    assert round(h["On-time % (all eligible orders)"].value, 1) == 20.7
    assert h["On-time % (all eligible orders)"].n == 1817
    assert round(h["Margin per order"].value, 2) == -5.50  # fee − cost, never order value − cost
    assert "basket_rs" not in h["Margin per order"].formula
    assert set(m["segments"]) == {"priority", "peak", "entity"}
    assert m["segments"]["priority"].set_index("urgency").at["Critical", "on-time %"] == 0.0


def test_hub_model_shows_reported_and_calculated(unknown):
    hub = D.metrics(unknown.df, unknown.pack, unknown.entity_of, unknown.role_map)["hub"]
    cols = hub["table"].columns
    assert any("[reported]" in c for c in cols) and any("[calculated" in c for c in cols)
    assert any("[model]" in c for c in cols)
    assert "sample" in hub["mismatch"]
    w = hub["table"]["wait multiplier 1/(1−u) [model]"]
    assert w.max() == pytest.approx(20.0)  # 96% reported, capped at 95%


def test_pack_off_without_delivery_roles():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=50), "store": ["A", "B"] * 25,
                       "price": range(50), "cost": range(50)})
    pk = D.detect(df, {"date": "date", "store": "location", "price": "money", "cost": "money"}, {})
    assert not pk.active and "actual" in pk.missing


def test_recommendations_have_evidence_and_rank(unknown):
    recs = R.generate(_inputs(unknown, unknown.df))
    assert recs and [r.impact for r in recs] == sorted([r.impact for r in recs], reverse=True)
    ids = {r.id for r in recs}
    assert {"on_time", "margin", "priority_inversion", "promise_gap", "hub_overload"} <= ids
    for r in recs:
        assert r.evidence and all(e.formula for e in r.evidence) and r.confidence in R.LEVELS and r.impact_formula
    margin = next(r for r in recs if r.id == "margin")
    be = next(e for e in margin.evidence if e.label.startswith("Fee rise"))
    assert be.value == pytest.approx(5.50 / 67.26 * 100, abs=0.1)
    assert "infeasible" not in ids  # scenario-only rule stays off on the baseline


def test_recommendations_refresh_after_constraint(unknown):
    df, s = unknown.df, unknown.setup
    busiest = df[s.entity].value_counts().index[0]
    p = Params(capacity_changes=[CapacityChange(entity_value=busiest, pct=-100, days=3)],
               targets=[Target(metric="on_time_pct", op=">=", value=80, segment_column="urgency",
                               segment_value="Critical")])
    base = R.generate(_inputs(unknown, df))
    scen = R.generate(_inputs(unknown, apply_rows(df, p)[0], p, scenario=True))
    rows = {c["id"]: c for c in R.compare(base, scen)}
    assert rows["infeasible"]["status"] == "new in scenario"
    assert rows["on_time"]["status"] == "changed"
    t = rows["on_time"]["table"]
    assert {"before (baseline)", "after (scenario)", "formula"} <= set(t.columns)


def test_segment_target_in_cascade(unknown):
    df, s = unknown.df, unknown.setup
    p = Params(targets=[Target(metric="on_time_pct", op=">=", value=80, segment_column="urgency",
                               segment_value="Critical")])
    cz = cascade(df, df, s, p)
    assert cz["verdict"] == "Not feasible" and not cz["targets"]["met"].iloc[0]
    assert cz["targets"]["scenario estimate"].iloc[0] < 5


def test_ai_rewrite_rejects_new_numbers(unknown, monkeypatch):
    recs = R.generate(_inputs(unknown, unknown.df))[:2]
    good, bad = recs
    monkeypatch.setattr("engine.recommend.chat_json", lambda *a, **k: {
        good.id: "Plain words: " + good.evidence[0].text() + " matters.", bad.id: "This costs 12,345 per day."})
    R.rewrite(recs, "t")
    assert good.plain and "checked" in good.rewrite_note
    assert not bad.plain and "12,345" in bad.rewrite_note


def test_ai_rewrite_falls_back_when_ai_down(unknown):
    recs = R.generate(_inputs(unknown, unknown.df))
    msg = R.rewrite(recs, "down")  # conftest makes every live call fail
    assert "unavailable" in msg and all(not r.plain for r in recs)


def test_explain_is_built_from_what_ran(unknown):
    run = unknown
    kpis = K.run_plan(run.plan, run.df, run.ctx)
    p = Params(capacity_changes=[CapacityChange(entity_value=run.df[run.setup.entity].iloc[0], pct=-30)])
    cz = cascade(run.df, apply_rows(run.df, p)[0], run.setup, p)
    recs = R.generate(_inputs(run, run.df))
    secs = explain.build(name=run.name, rows_raw=len(run.raw), rows_clean=len(run.df), roles=run.roles,
                         entity_of=run.entity_of, fixes=run.fixes, vmap=run.vmap, vmap_confirmed=True, kpis=kpis,
                         pack=run.pack, pack_metrics=D.metrics(run.df, run.pack, run.entity_of, run.role_map),
                         params=p, notes=[], cascade=cz, recs=recs, issues=[], quality_score=100.0, days=14)
    text = "\n".join(line for s in secs for line in s.lines)
    assert f"{len(run.raw):,} rows uploaded" in text and f"{len(run.df):,} rows after quality fixes" in text
    assert "shipping_charge" in text and "last_mile_cost" in text  # pack formulas name the real columns
    assert "calibrated" in text or "assumption" in text
    kpi_tbl = next(s.table for s in secs if s.title.startswith("4."))
    assert set(kpi_tbl["KPI"]) == {k.title for k in kpis}
    assert next(s.table for s in secs if s.title.startswith("3.")) is not None  # mapping table
    rules = next(s.table for s in secs if s.title.startswith("7."))
    assert set(rules.loc[rules["fired now"] == "yes", "rule"]) == {r.rule for r in recs}
