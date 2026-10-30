from pathlib import Path

import pandas as pd
import pytest

from engine.constraints import parse
from engine.loader import load_table
from engine.profile import profile_frame
from engine.quality import apply_fixes, find_issues, recommended_fixes
from engine.roles import detect_roles, entity_attributes
from engine.scenario import (CapacityChange, Exclusion, MeasureChange, Params, PriceChange, Setup, Target, _allocate,
                             apply_rows, cascade, change_rows, combine, guess_setup, validate_params)

DATA = Path(__file__).resolve().parent.parent / "data" / "BUDS_unknown_schema_test.csv"
PLAN = [{"calc": "on_time_rate", "params": {"actual": "tat_mins", "promised": "sla_mins", "status": "outcome",
                                            "completed_values": ["completed"]}},
        {"calc": "margin", "params": {"revenue": "shipping_charge", "cost": "last_mile_cost"}}]


@pytest.fixture(scope="module")
def buds():
    if not DATA.exists():
        pytest.skip("acceptance file missing")
    df, _ = load_table(DATA.read_bytes(), DATA.name)
    prof = profile_frame(df)
    roles, _ = detect_roles(df, "scen", use_ai=False, prof=prof)
    ent = entity_attributes(df, roles)
    df = apply_fixes(df, recommended_fixes(find_issues(df, roles, set(ent["attribute"]))))
    rmap = {c: r for c, r, k in zip(roles["column"], roles["role"], roles["check"]) if k != "fail"}
    setup = guess_setup(df, rmap, dict(zip(ent["attribute"], ent["entity"])), PLAN, prof)
    return df, rmap, setup, prof


def test_guess_setup_finds_cascade_columns(buds):
    _, _, s, _ = buds
    assert (s.entity, s.capacity, s.reported_util, s.load_basis) == ("dc_code", "dc_capacity_per_day", "dc_util", "reported")
    assert (s.actual, s.promised, s.fee, s.cost) == ("tat_mins", "sla_mins", "shipping_charge", "last_mile_cost")
    assert not s.cascade_ready()


def test_c10_busiest_hub_cut_30pct_cascades(buds):
    """C10 acceptance: one capacity change moves allocation, utilisation, on-time, cost and feasibility together."""
    df, _, s, _ = buds
    busiest = df[s.entity].value_counts().index[0]
    base = cascade(df, df, s, Params())
    p = Params(capacity_changes=[CapacityChange(entity_value=busiest, pct=-30)])
    scen = cascade(df, apply_rows(df, p)[0], s, p)
    b, c = base["steps"].set_index("metric"), scen["steps"].set_index("metric")
    assert c.at["orders moved per day", "scenario"] > 0
    assert c.at["highest utilisation %", "scenario"] != b.at["highest utilisation %", "scenario"]
    assert c.at["on-time %", "scenario"] < b.at["on-time %", "scenario"]
    assert c.at["cost per day", "scenario"] != b.at["cost per day", "scenario"]
    assert base["verdict"] != scen["verdict"] == "Not feasible"
    before = {r.metric: r.scenario for r in base["steps"].itertuples()} | {"feasibility": base["verdict"]}
    after = {r.metric: r.scenario for r in scen["steps"].itertuples()} | {"feasibility": scen["verdict"]}
    log = change_rows(1, "12:00", "busiest hub -30%", "capacity cut", p, before, after)
    moved = {r["metric"] for r in log if r["change"] not in ("", 0, 0.0)}
    assert {"orders moved per day", "highest utilisation %", "on-time %", "cost per day", "feasibility"} <= moved


def test_baseline_cascade_matches_kpi_on_time(buds):
    df, _, s, _ = buds
    on = cascade(df, df, s, Params())["steps"].set_index("metric").at["on-time %", "baseline"]
    assert abs(on - 20.7) < 0.3  # rows without an entity are left out of the cascade


def test_outage_days_and_receivers_capped(buds):
    df, _, s, _ = buds
    p = Params(capacity_changes=[CapacityChange(entity_value="DC-01", pct=-100, days=3)], util_cap=0.9)
    r = cascade(df, apply_rows(df, p)[0], s, p)
    assert r["affected_days"] == 3
    assert (r["entities"]["utilisation % scenario"].dropna() <= 100).all()


def test_allocation_lowest_utilisation_first_and_unserved():
    kept, flow, unserved, final = _allocate({"A": 100, "B": 50, "C": 10}, {"A": 60, "B": 100, "C": 100}, 0.85)
    assert flow["A"] == {"C": 40}  # C (10%) fills before B (50%)
    _, _, unserved, final = _allocate({"A": 100, "B": 80}, {"A": 10, "B": 100}, 0.85)
    assert unserved["A"] == pytest.approx(85) and final["B"] == pytest.approx(85)


def test_row_level_changes():
    df = pd.DataFrame({"s": ["ok", "cancel", "ok"], "fee": [10.0, 20.0, 30.0], "t": [1.0, 2.0, 3.0]})
    p = Params(exclusions=[Exclusion(column="s", op="==", value="cancel")],
               measure_changes=[MeasureChange(column="t", pct=10)],
               price_changes=[PriceChange(column="fee", pct=8, cap_pct=5)])
    out, notes = apply_rows(df, p)
    assert len(out) == 2 and out["fee"].tolist() == pytest.approx([10.5, 31.5]) and out["t"].tolist() == pytest.approx([1.1, 3.3])
    assert any("capped at 5%" in n for n in notes)


def test_combine_stacks_constraints():
    a = Params(exclusions=[Exclusion(column="s", op="==", value="x")], util_cap=0.8)
    b = Params(measure_changes=[MeasureChange(column="t", pct=5)], util_cap=0.9)
    c = combine([a, b])
    assert len(c.exclusions) == 1 and len(c.measure_changes) == 1 and c.util_cap == 0.9


def test_targets_drive_feasibility(buds):
    df, _, s, _ = buds
    p = Params(targets=[Target(metric="on_time_pct", op=">=", value=90)])
    r = cascade(df, df, s, p)
    assert r["verdict"] == "Not feasible" and any("on_time_pct" in x for x in r["reasons"])


def test_validate_params_rejects_bad_values(buds):
    df, roles, s, _ = buds
    assert validate_params(Params(capacity_changes=[CapacityChange(entity_value="DC-99", pct=-10)]), df, roles, s)
    assert validate_params(Params(exclusions=[Exclusion(column="outcome", op="==", value="lost")]), df, roles, s)
    assert validate_params(Params(price_changes=[PriceChange(column="km", pct=5)]), df, roles, s)
    assert not validate_params(Params(exclusions=[Exclusion(column="outcome", op="==", value="cancelled")]), df, roles, s)


@pytest.mark.parametrize("reply,ok", [
    ({"mappable": True, "params": {"capacity_changes": [{"entity_value": "DC-02", "pct": -100, "days": 3}]},
      "interpretation": "DC-02 down 3 days"}, True),
    ({"mappable": True, "params": {"exclusions": [{"column": "outcome", "op": "==", "value": "cancelled"}]}}, True),
    ({"mappable": True, "params": {"capacity_changes": [{"entity_value": "Hub X", "pct": -100}]}}, False),
    ({"mappable": True, "params": {"weather": "rain"}}, False),
    ({"mappable": False, "reason": "no weather column", "suggestion": "use a capacity change"}, False),
])
def test_constraint_parse_validation(buds, monkeypatch, reply, ok):
    df, roles, s, prof = buds
    monkeypatch.setattr("engine.constraints.chat_json", lambda *a, **k: reply)
    r = parse("x", df, roles, s, prof, "k")
    assert r.ok is ok
    if not ok:
        assert r.reason
