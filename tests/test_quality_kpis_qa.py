import pandas as pd
import pytest

from engine import kpis as K
from engine.qa import auto_chart, extract_code, unverified_numbers
from engine.quality import Fix, apply_fix, find_issues, quality_score, rating_scale, spelling_groups
from engine.roles import detect_roles, entity_attributes
from engine.sandbox import SandboxError, check_code, run_code
from tests.fixtures import make_delivery


@pytest.fixture(scope="module")
def setup():
    df = make_delivery()
    roles, _ = detect_roles(df, "q", use_ai=False)
    ent = entity_attributes(df, roles)
    return df, roles, set(ent["attribute"]), K.Ctx(
        roles={c: r for c, r, k in zip(roles["column"], roles["role"], roles["check"]) if k != "fail" and r != "ignore"},
        entity_of=dict(zip(ent["attribute"], ent["entity"])))


# ---------- quality ----------

def test_expected_issues(setup):
    df, roles, ents, _ = setup
    got = {i.key: i.count for i in find_issues(df, roles, ents)}
    assert got["duplicate_id:Order_ID"] == 12
    assert got["missing:Zone"] == 15
    assert got["missing:Hub"] == 10
    assert got["negative:Distance_km"] == 1
    assert got["out_of_scale:Customer_Rating"] == 4
    assert got["zero_duration:Actual_Time_min"] == int((df["Actual_Time_min"] == 0).sum()) >= 3


def test_every_default_fix_clears_its_issue(setup):
    df, roles, ents, _ = setup
    for i in find_issues(df, roles, ents):
        fixed = apply_fix(df, Fix(i.key, i.rule, i.column, i.fixes[0], dict(i.params)))
        assert i.key not in {j.key for j in find_issues(fixed, roles, ents)}, i.key


def test_fix_counts_before_after(setup):
    df, roles, ents, _ = setup
    f = Fix("duplicate_id:Order_ID", "duplicate_id", "Order_ID", "drop_duplicates", {})
    out = apply_fix(df, f)
    assert (f.rows_before, f.rows_after) == (len(df), len(df) - 12) and len(out) == len(df) - 12


def test_score_is_share_of_clean_rows():
    df = pd.DataFrame({"id": ["a", "a", "b", "c"], "v": [1.0, 2.0, None, 4.0]})
    roles = pd.DataFrame({"column": ["id", "v"], "role": ["id", "money"], "check": ["ok", "ok"]})
    score, clean = quality_score(df, find_issues(df, roles))
    assert (score, clean) == (50.0, 2)


def test_rating_scale_and_spelling():
    assert rating_scale("Rating_1to5", pd.Series([1, 6]))[:2] == (1, 5)
    assert rating_scale("stars", pd.Series([1, 2, 3, 4, 5] * 40 + [6]))[:2] == (1, 5)
    m = spelling_groups(pd.Series(["North"] * 10 + ["north "] * 2 + ["NORTH-EAST"] * 5 + ["HUB-EAST"] * 30 + ["HUB-E"]))
    assert m["north "] == "North" and m["HUB-E"] == "HUB-EAST" and "NORTH-EAST" not in m


# ---------- KPIs ----------

def test_total_and_average(setup):
    df, _, _, ctx = setup
    k = K.run({"calc": "total", "params": {"col": "Delivery_Fee"}}, df, ctx)
    assert k.value == pytest.approx(df["Delivery_Fee"].sum())
    a = K.run({"calc": "average", "params": {"col": "Customer_Rating", "stat": "median"}}, df, ctx)
    assert a.value == df["Customer_Rating"].median()


def test_entity_attribute_not_summed_per_row(setup):
    df, _, _, ctx = setup
    k = K.run({"calc": "total", "params": {"col": "Hub_Capacity"}}, df, ctx)
    assert k.value == 120 + 90 + 150 + 80 + 100 + 110 and k.basis == "reported"


def test_margin_and_on_time(setup):
    df, _, _, ctx = setup
    m = K.run({"calc": "margin", "params": {"revenue": "Delivery_Fee", "cost": "Delivery_Cost"}}, df, ctx)
    assert m.value == pytest.approx((df["Delivery_Fee"] - df["Delivery_Cost"]).sum())
    o = K.run({"calc": "on_time_rate", "params": {"actual": "Actual_Time_min", "promised": "Promised_Time_min",
                                                  "status": "Order_Status", "completed_values": ["Delivered"]}}, df, ctx)
    d = df[df["Order_Status"] == "Delivered"]
    assert o.value == pytest.approx(100 * (d["Actual_Time_min"] <= d["Promised_Time_min"]).mean())
    assert "Delivered" in o.filter


def test_share_top_growth_trend(setup):
    df, _, _, ctx = setup
    s = K.run({"calc": "share", "params": {"col": "Peak_Hour", "value": "Yes"}}, df, ctx)
    assert s.value == pytest.approx(100 * (df["Peak_Hour"] == "Yes").mean())
    t = K.run({"calc": "top_n", "params": {"group": "Hub", "agg": "count", "n": 3}}, df, ctx)
    assert len(t.table) == 3 and t.value == df["Hub"].value_counts().iloc[0]
    tr = K.run({"calc": "trend", "params": {"date": "Order_Date", "agg": "count"}}, df, ctx)
    assert tr.table["rows"].sum() == len(df)
    g = K.run({"calc": "growth", "params": {"date": "Order_Date", "agg": "count"}}, df, ctx)
    assert g.value is not None and not any("too short" in w for w in g.warnings)  # 30 days


def test_short_range_warns():
    df = pd.DataFrame({"d": pd.date_range("2025-01-01", periods=10), "v": range(10)})
    k = K.trend(df, K.Ctx({"d": "date", "v": "money"}), "d")
    assert any("too short" in w for w in k.warnings)


def test_validation_rejects_bad_plans(setup):
    df, _, _, ctx = setup
    assert K.validate({"calc": "total", "params": {"col": "Hub"}}, df, ctx)
    assert K.validate({"calc": "nope"}, df, ctx)
    assert K.validate({"calc": "share", "params": {"col": "Peak_Hour", "value": "Maybe"}}, df, ctx)
    assert K.validate({"calc": "margin", "params": {"revenue": "Delivery_Fee", "cost": "Delivery_Fee"}}, df, ctx)
    assert not K.validate({"calc": "total", "params": {"col": "Delivery_Fee"}}, df, ctx)


def test_default_plan_runs_clean(setup):
    df, _, _, ctx = setup
    plan = K.default_plan(df, ctx)
    assert all(not K.validate(p, df, ctx) for p in plan)
    assert all(k.value is not None for k in K.run_plan(plan, df, ctx))


# ---------- sandbox & Q&A ----------

@pytest.mark.parametrize("code", ["import os", "open('x')", "df.to_csv('x')", "().__class__", "pd.read_csv('x')",
                                  "class A: pass", "getattr(df, 'x')", "from pandas import read_csv",
                                  "pd.set_option('x', 1)", "x = '__import__'"])
def test_sandbox_blocks(code):
    assert check_code(code)[1]


def test_sandbox_runs_and_times_out():
    df = pd.DataFrame({"a": [1, 2, 3], "g": ["x", "y", "x"]})
    r = run_code("import pandas as pd\nresult = df.groupby('g')['a'].sum()", df)
    assert r["ok"] and r["result"].to_dict() == {"x": 4, "y": 2}
    assert not run_code("result = 1/0", df)["ok"]
    with pytest.raises(SandboxError):
        run_code("while True:\n    pass", df, timeout=3)


def test_narration_number_check():
    res = pd.DataFrame({"hub": ["A", "B"], "rate": [0.8734, 0.5]})
    assert unverified_numbers("Hub A leads at 87.3%, B at 0.5.", res, "") == []
    assert unverified_numbers("Hub A leads at 91%.", res, "") == ["91"]


def test_extract_code_and_auto_chart():
    assert extract_code("text\n```python\nresult = 1\n```\nmore") == "result = 1"
    chart, tbl = auto_chart(pd.Series([3, 1], index=pd.Index(["a", "b"], name="g"), name="n"))
    assert chart == {"type": "bar", "x": "g", "y": "n"} and len(tbl) == 2


# ---------- unknown-schema acceptance test (renamed BUDS columns, messy formats) ----------

def test_buds_unknown_schema_targets():
    from pathlib import Path

    from engine.loader import load_table

    path = Path(__file__).resolve().parent.parent / "data" / "BUDS_unknown_schema_test.csv"
    if not path.exists():
        pytest.skip("data/BUDS_unknown_schema_test.csv not present")
    df, _ = load_table(path.read_bytes(), path.name)
    roles, _ = detect_roles(df, "unknown-schema", use_ai=False)
    ent = entity_attributes(df, roles)
    role = dict(zip(roles["column"], roles["role"]))
    ids = [c for c, r in role.items() if r == "id"]
    assert len(ids) == 1
    from engine.quality import apply_fixes, recommended_fixes

    df = apply_fixes(df, recommended_fixes(find_issues(df, roles, set(ent["attribute"]))))
    assert len(df) == 2000
    ctx = K.Ctx({c: r for c, r, k in zip(roles["column"], roles["role"], roles["check"]) if k != "fail"},
                dict(zip(ent["attribute"], ent["entity"])))
    durs = [c for c, r in role.items() if r == "duration"]
    assert len(durs) == 2  # promised + actual, one stored as text like "32 min"
    # column choice is the AI's job; here we pass it explicitly like a confirmed plan would
    promised = next(c for c in durs if df[c].astype("string").str.contains("min").any())
    actual = next(c for c in durs if c != promised)
    status = next(c for c, r in role.items() if r == "category" and "completed" in set(df[c].dropna()))
    o = K.run({"calc": "on_time_rate", "params": {"actual": actual, "promised": promised, "status": status,
                                                  "completed_values": ["completed"]}}, df, ctx)
    assert "1,817 eligible" in o.denominator and round(o.value, 1) == 20.7
    money = [c for c, r in role.items() if r == "money" and c not in ctx.entity_of]
    fee = next(c for c in money if "charge" in c)
    cost = next(c for c in money if "cost" in c)
    m = K.run({"calc": "margin", "params": {"revenue": fee, "cost": cost}}, df, ctx)
    assert round(m.value / 2000, 2) == -5.50 and "per record = -5.50" in m.extra
