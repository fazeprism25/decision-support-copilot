"""Role-driven KPI library. Every result carries formula, filter, denominator and exclusions.
The LLM may only choose calculations and parameters from CATALOG; all numbers are computed here."""
import json
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from engine.profile import to_datetime_clean, to_numeric_clean

MIN_TREND_DAYS = 14
NUMERIC = {"money", "quantity", "duration", "rating", "measure"}
GROUPS = {"category", "location", "flag", "id"}


@dataclass
class Ctx:
    roles: dict[str, str]  # column -> role (only columns that passed the check)
    entity_of: dict[str, str] = field(default_factory=dict)  # attribute column -> entity column


@dataclass
class KPI:
    title: str
    value: float | None
    fmt: str = "number"  # number | pct | int
    basis: str = "calculated"  # calculated | reported
    formula: str = ""
    filter: str = "none"
    denominator: str = ""
    exclusions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extra: str = ""
    table: pd.DataFrame | None = None
    chart: dict | None = None  # {"type": bar|line|hist, "x": col, "y": col}
    calc: str = ""
    params: dict = field(default_factory=dict)


def fmt_value(v, fmt: str) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    if fmt == "pct":
        return f"{v:.1f}%"
    if fmt == "int" or float(v).is_integer() and abs(v) >= 1:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def _num(df, col) -> tuple[pd.Series, list[str]]:
    x = to_numeric_clean(df[col])
    miss = int(x.isna().sum())
    return x.dropna(), [f"{miss:,} rows with no numeric value in '{col}' excluded"] if miss else []


def _entity_view(df, col, ctx) -> tuple[pd.DataFrame, str | None]:
    ent = ctx.entity_of.get(col)
    if not ent:
        return df, None
    return df.dropna(subset=[ent]).drop_duplicates(ent), ent


def _date_span(df, col) -> int:
    d = to_datetime_clean(df[col]).dropna()
    return int((d.max().normalize() - d.min().normalize()).days) + 1 if len(d) else 0


def _agg_label(agg, measure):
    return "number of rows" if agg == "count" or not measure else f"{agg} of '{measure}'"


def _group_agg(df, group, measure, agg):
    if agg == "count" or not measure:
        g = df.groupby(group, dropna=True).size().rename("rows")
        return g, "rows", []
    x = to_numeric_clean(df[measure])
    excl = [f"{int(x.isna().sum()):,} rows with no value in '{measure}' excluded"] if x.isna().any() else []
    g = x.groupby(df[group], dropna=True).agg(agg).rename(f"{agg}_{measure}")
    return g, g.name, excl


# ---------- calculations ----------

def count_rows(df, ctx, **_):
    return KPI("Records", len(df), "int", formula="count of rows", denominator=f"{len(df):,} rows")


def distinct(df, ctx, col):
    k = df[col].nunique()
    return KPI(f"Distinct {col}", k, "int", formula=f"count of distinct values of '{col}'",
               denominator=f"{df[col].notna().sum():,} rows with a value")


def total(df, ctx, col):
    view, ent = _entity_view(df, col, ctx)
    x, excl = _num(view, col)
    k = KPI(f"Total {col}", float(x.sum()), formula=f"sum of '{col}'", denominator=f"{len(x):,} values",
            exclusions=excl)
    if ent:
        k.basis = "reported"
        k.formula = f"sum of '{col}' taking one value per '{ent}' (it repeats on every row of the group)"
        k.denominator = f"{len(x)} distinct '{ent}' values"
    return k


def average(df, ctx, col, stat="mean"):
    view, ent = _entity_view(df, col, ctx)
    x, excl = _num(view, col)
    v = float(x.median() if stat == "median" else x.mean()) if len(x) else None
    k = KPI(f"{'Median' if stat == 'median' else 'Average'} {col}", v,
            formula=f"{stat} of '{col}'", denominator=f"{len(x):,} values", exclusions=excl)
    if ent:
        k.basis = "reported"
        k.formula += f", one value per '{ent}'"
        k.denominator = f"{len(x)} distinct '{ent}' values"
    return k


def share(df, ctx, col, value):
    s = df[col].dropna().astype("string")
    hit = int((s == str(value)).sum())
    return KPI(f"Share {col} = {value}", 100 * hit / len(s) if len(s) else None, "pct",
               formula=f"rows where '{col}' = '{value}' ÷ rows with a value in '{col}'",
               denominator=f"{len(s):,} rows ({hit:,} matching)",
               exclusions=[f"{len(df) - len(s):,} rows with no value excluded"] if len(df) > len(s) else [])


def top_n(df, ctx, group, measure=None, agg="sum", n=5, ascending=False):
    g, ycol, excl = _group_agg(df, group, measure, agg)
    g = g.sort_values(ascending=ascending).head(int(n))
    tbl = g.reset_index()
    word = "Bottom" if ascending else "Top"
    lead = f"{tbl.iloc[0][group]}" if len(tbl) else None
    return KPI(f"{word} {n} {group} by {_agg_label(agg, measure)}", float(g.iloc[0]) if len(g) else None,
               formula=f"{_agg_label(agg, measure)} per '{group}', sorted, first {n}",
               denominator=f"{df[group].nunique()} groups", exclusions=excl, extra=f"leader: {lead}",
               table=tbl, chart={"type": "bar", "x": group, "y": ycol})


def segment_breakdown(df, ctx, group, measure=None, agg="sum"):
    g, ycol, excl = _group_agg(df, group, measure, agg)
    g = g.sort_values(ascending=False)
    tbl = g.reset_index()
    if agg in ("sum", "count") or not measure:
        tbl["share_%"] = (100 * tbl[ycol] / tbl[ycol].sum()).round(1)
    if len(tbl) > 20:
        excl.append(f"chart shows the largest 20 of {len(tbl)} groups")
    k = KPI(f"{_agg_label(agg, measure).capitalize()} by {group}", float(g.max()) if len(g) else None,
            formula=f"{_agg_label(agg, measure)} per '{group}'", denominator=f"{len(g)} groups",
            exclusions=excl, extra=f"largest: {g.index[0]}" if len(g) else "",
            table=tbl, chart={"type": "bar", "x": group, "y": ycol})
    miss = int(df[group].isna().sum())
    if miss:
        k.exclusions.append(f"{miss:,} rows with no '{group}' excluded")
    return k


def trend(df, ctx, date, measure=None, agg="sum", freq="D"):
    d = to_datetime_clean(df[date])
    y = pd.Series(1, index=df.index) if agg == "count" or not measure else to_numeric_clean(df[measure])
    frame = pd.DataFrame({"period": d.dt.to_period(freq).dt.start_time, "y": y}).dropna()
    s = frame.groupby("period")["y"].agg("sum" if agg == "count" or not measure else agg)
    ycol = "rows" if agg == "count" or not measure else f"{agg}_{measure}"
    tbl = s.rename(ycol).reset_index()
    span = _date_span(df, date)
    k = KPI(f"Trend of {_agg_label(agg, measure)} per {dict(D='day', W='week', M='month').get(freq, freq)}",
            float(s.iloc[-1]) if len(s) else None,
            formula=f"{_agg_label(agg, measure)} grouped by '{date}' ({freq})",
            denominator=f"{len(s)} periods over {span} days", extra="value shown = latest period",
            exclusions=[f"{len(df) - len(frame):,} rows without a date or value excluded"] if len(df) > len(frame) else [],
            table=tbl, chart={"type": "line", "x": "period", "y": ycol})
    if span < MIN_TREND_DAYS:
        k.warnings.append(f"Only {span} days of data: too short (< {MIN_TREND_DAYS} days) for trend claims.")
    return k


def growth(df, ctx, date, measure=None, agg="sum"):
    d = to_datetime_clean(df[date])
    y = pd.Series(1.0, index=df.index) if agg == "count" or not measure else to_numeric_clean(df[measure])
    frame = pd.DataFrame({"d": d.dt.normalize(), "y": y}).dropna()
    if frame.empty:
        return KPI("Growth", None, "pct", warnings=["no dated values"])
    start, end = frame["d"].min(), frame["d"].max()
    mid = start + (end - start) / 2
    first, second = frame[frame["d"] <= mid]["y"], frame[frame["d"] > mid]["y"]
    a = first.sum() if agg in ("sum", "count") or not measure else first.agg(agg)
    b = second.sum() if agg in ("sum", "count") or not measure else second.agg(agg)
    v = 100 * (b - a) / a if a else None
    span = int((end - start).days) + 1
    k = KPI(f"Growth in {_agg_label(agg, measure)}", v, "pct",
            formula=f"(second half − first half) ÷ first half; halves split at {mid.date()}",
            denominator=f"first half {a:,.2f}, second half {b:,.2f}",
            extra=f"{start.date()} → {end.date()}")
    if span < MIN_TREND_DAYS:
        k.warnings.append(f"Only {span} days of data: too short (< {MIN_TREND_DAYS} days) for a growth claim.")
    if span % 2:
        k.warnings.append("Halves have unequal numbers of days.")
    return k


def margin(df, ctx, revenue, cost):
    r, c = to_numeric_clean(df[revenue]), to_numeric_clean(df[cost])
    ok = r.notna() & c.notna()
    m = float((r[ok] - c[ok]).sum())
    rev = float(r[ok].sum())
    return KPI(f"Margin ({revenue} − {cost})", m, formula=f"sum('{revenue}' − '{cost}') over rows with both",
               denominator=f"{int(ok.sum()):,} rows",
               extra=(f"per record = {m / ok.sum():,.2f}" if ok.any() else "") +
                     (f" · margin % = {100 * m / rev:.1f}% of {revenue}" if rev else ""),
               exclusions=[f"{int((~ok).sum()):,} rows missing either value excluded"] if (~ok).any() else [])


def on_time_rate(df, ctx, actual, promised, status=None, completed_values=None):
    a, p = to_numeric_clean(df[actual]), to_numeric_clean(df[promised])
    mask = pd.Series(True, index=df.index)
    filt = "none"
    if status and completed_values:
        mask = df[status].astype("string").isin([str(v) for v in completed_values])
        filt = f"'{status}' in {list(completed_values)}"
    base = mask & a.notna() & p.notna()
    hit = int((a[base] <= p[base]).sum())
    n = int(base.sum())
    k = KPI("On-time rate", 100 * hit / n if n else None, "pct",
            formula=f"rows where '{actual}' ≤ '{promised}' ÷ eligible rows", filter=filt,
            denominator=f"{n:,} eligible rows ({hit:,} on time)",
            exclusions=[f"{int((~mask).sum()):,} rows not in the completed status excluded",
                        f"{int((mask & ~base).sum()):,} completed rows missing a time excluded"])
    bad = int((base & (a <= 0)).sum())
    if bad:  # no hidden rule here: invalid values are fixed once, in the Quality tab, for every tab
        k.warnings.append(f"{bad:,} eligible rows have '{actual}' ≤ 0 and count as on time. "
                          "Apply the zero-duration fix in the Quality tab to exclude them everywhere.")
    return k


def distribution(df, ctx, col):
    x, excl = _num(df, col)
    q = x.quantile([0.1, 0.5, 0.9]) if len(x) else pd.Series([None] * 3, index=[0.1, 0.5, 0.9])
    return KPI(f"Distribution of {col}", float(q[0.5]) if len(x) else None,
               formula=f"median of '{col}' (histogram shows all values)", denominator=f"{len(x):,} values",
               exclusions=excl, extra=f"10th–90th percentile: {q[0.1]:,.2f} – {q[0.9]:,.2f}" if len(x) else "",
               table=pd.DataFrame({col: x}), chart={"type": "hist", "x": col})


# ---------- catalog & validation ----------

CATALOG: dict[str, dict] = {
    "count_rows": {"fn": count_rows, "params": {}, "desc": "number of records"},
    "distinct": {"fn": distinct, "params": {"col": {"roles": GROUPS}}, "desc": "distinct count of an id/label column"},
    "total": {"fn": total, "params": {"col": {"roles": NUMERIC - {"rating"}}}, "desc": "sum of a numeric column"},
    "average": {"fn": average, "params": {"col": {"roles": NUMERIC}, "stat": {"choices": ["mean", "median"], "optional": True}},
                "desc": "mean or median of a numeric column"},
    "share": {"fn": share, "params": {"col": {"roles": GROUPS - {"id"}}, "value": {"value_of": "col"}},
              "desc": "% of rows where a label column equals a value"},
    "top_n": {"fn": top_n, "params": {"group": {"roles": GROUPS}, "measure": {"roles": NUMERIC, "optional": True},
                                      "agg": {"choices": ["sum", "mean", "count"], "optional": True},
                                      "n": {"int": True, "optional": True}, "ascending": {"bool": True, "optional": True}},
              "desc": "top or bottom N groups by an aggregate"},
    "segment_breakdown": {"fn": segment_breakdown, "params": {"group": {"roles": GROUPS - {"id"}},
                                                              "measure": {"roles": NUMERIC, "optional": True},
                                                              "agg": {"choices": ["sum", "mean", "count"], "optional": True}},
                          "desc": "aggregate per group with shares"},
    "trend": {"fn": trend, "params": {"date": {"roles": {"date"}}, "measure": {"roles": NUMERIC, "optional": True},
                                      "agg": {"choices": ["sum", "mean", "count"], "optional": True},
                                      "freq": {"choices": ["D", "W", "M"], "optional": True}},
              "desc": "aggregate over time"},
    "growth": {"fn": growth, "params": {"date": {"roles": {"date"}}, "measure": {"roles": NUMERIC, "optional": True},
                                        "agg": {"choices": ["sum", "mean", "count"], "optional": True}},
               "desc": "change between first and second half of the date range"},
    "margin": {"fn": margin, "params": {"revenue": {"roles": {"money"}}, "cost": {"roles": {"money"}}},
               "desc": "revenue column minus cost column"},
    "on_time_rate": {"fn": on_time_rate, "params": {"actual": {"roles": {"duration"}}, "promised": {"roles": {"duration"}},
                                                    "status": {"roles": {"category", "flag"}, "optional": True},
                                                    "completed_values": {"values_of": "status", "optional": True}},
                     "desc": "% of rows with actual time <= promised time, among completed rows"},
    "distribution": {"fn": distribution, "params": {"col": {"roles": NUMERIC}}, "desc": "histogram and percentiles"},
}


def normalize(item: dict) -> dict:
    """Drop harmless contradictions, e.g. a measure given with agg='count' (counting ignores the measure)."""
    params = dict(item.get("params") or {})
    if params.get("agg") == "count":
        params.pop("measure", None)
    return {**item, "params": params}


def validate(item: dict, df: pd.DataFrame, ctx: Ctx) -> list[str]:
    """Return a list of problems; empty means the plan item is safe to run."""
    item = normalize(item)
    calc, params = item.get("calc"), item.get("params") or {}
    if calc not in CATALOG:
        return [f"unknown calculation '{calc}'"]
    spec = CATALOG[calc]["params"]
    errs = [f"unexpected parameter '{k}'" for k in params if k not in spec]
    for name, rule in spec.items():
        v = params.get(name)
        if v in (None, "", []):
            if not rule.get("optional"):
                errs.append(f"missing '{name}'")
            continue
        if "roles" in rule:
            if v not in ctx.roles:
                errs.append(f"'{v}' is not a usable column")
            elif ctx.roles[v] not in rule["roles"]:
                errs.append(f"'{v}' has role {ctx.roles[v]}, needs one of {sorted(rule['roles'])}")
        elif "choices" in rule and v not in rule["choices"]:
            errs.append(f"'{name}' must be one of {rule['choices']}")
        elif rule.get("int") and not isinstance(v, int):
            errs.append(f"'{name}' must be a whole number")
        elif rule.get("bool") and not isinstance(v, bool):
            errs.append(f"'{name}' must be true or false")
        elif "value_of" in rule or "values_of" in rule:
            src = params.get(rule.get("value_of") or rule.get("values_of"))
            if src in df:
                present = set(df[src].dropna().astype("string"))
                missing = [x for x in (v if isinstance(v, list) else [v]) if str(x) not in present]
                if missing:
                    errs.append(f"value(s) {missing} not found in '{src}'")
    pair = {"margin": ("revenue", "cost"), "on_time_rate": ("actual", "promised")}.get(calc)
    if pair and params.get(pair[0]) == params.get(pair[1]):
        errs.append("the two columns must differ")
    return errs


def run(item: dict, df: pd.DataFrame, ctx: Ctx) -> KPI:
    item = normalize(item)
    fn: Callable = CATALOG[item["calc"]]["fn"]
    params = {k: v for k, v in (item.get("params") or {}).items() if v not in (None, "", [])}
    k = fn(df, ctx, **params)
    if item.get("title"):
        k.title = item["title"]
    k.calc, k.params = item["calc"], params
    return k


def run_plan(plan: list[dict], df: pd.DataFrame, ctx: Ctx) -> list[KPI]:
    out = []
    for item in plan:
        try:
            out.append(run(item, df, ctx))
        except Exception as e:  # a bad calc must not take the tab down
            out.append(KPI(item.get("title") or item["calc"], None, warnings=[f"could not compute: {e}"],
                           calc=item["calc"], params=item.get("params", {})))
    return out


def default_plan(df: pd.DataFrame, ctx: Ctx) -> list[dict]:
    """Rule-based plan used when AI is off or fails."""
    by = lambda *rs: [c for c, r in ctx.roles.items() if r in rs]
    plan = [{"calc": "count_rows"}]
    plan += [{"calc": "distinct", "params": {"col": c}} for c in by("id")[:1]]
    plan += [{"calc": "total", "params": {"col": c}} for c in by("money") if c not in ctx.entity_of][:3]
    plan += [{"calc": "average", "params": {"col": c}} for c in by("duration", "rating") if c not in ctx.entity_of][:3]
    dates = sorted(by("date"), key=lambda c: df[c].nunique())
    if dates:
        plan.append({"calc": "trend", "params": {"date": dates[0], "agg": "count"}})
    groups = sorted([c for c in by("category", "location", "flag") if 2 <= df[c].nunique() <= 20],
                    key=lambda c: df[c].nunique())
    plan += [{"calc": "segment_breakdown", "params": {"group": c, "agg": "count"}} for c in groups[:2]]
    nums = [c for c in by("duration", "money") if c not in ctx.entity_of]
    if nums:
        plan.append({"calc": "distribution", "params": {"col": nums[0]}})
    return plan


def catalog_for_prompt() -> str:
    lines = []
    for name, c in CATALOG.items():
        ps = []
        for p, r in c["params"].items():
            t = (f"column with role in {sorted(r['roles'])}" if "roles" in r else f"one of {r['choices']}" if "choices" in r
                 else "integer" if r.get("int") else "true/false" if r.get("bool")
                 else f"value(s) present in column '{r.get('value_of') or r.get('values_of')}'")
            ps.append(f"{p}{'?' if r.get('optional') else ''}: {t}")
        lines.append(f"- {name}({', '.join(ps)}): {c['desc']}")
    return "\n".join(lines)


def plan_prompt(columns: list[dict]) -> str:
    return (
        "Choose the 8-14 most decision-relevant KPIs and charts for this dataset from the calculation catalog.\n"
        f"Catalog:\n{catalog_for_prompt()}\n\nColumns (role, meaning, distinct values, top labels):\n"
        f"{json.dumps(columns, default=str)}\n\n"
        "Rules: use only catalog calculations and exact column names; never invent numbers. MARGIN means the fee or "
        "charge for an activity minus the cost of that SAME activity (e.g. delivery/shipping fee − delivery cost). "
        "NEVER use order/basket/goods value as revenue for margin; prefer on_time_rate when actual and promised durations exist (use the "
        "status column and its completed label(s) if present). Give each a short business title.\n"
        'Return ONLY JSON: {"kpis": [{"calc": "...", "params": {...}, "title": "..."}]}'
    )
