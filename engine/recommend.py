"""Rule-generated recommendations. Every recommendation comes from a rule over numbers pandas already computed and
carries its evidence (value + formula), assumptions, limitations, a confidence level and an impact score used for
ranking. The AI may only rewrite the wording, and a rewrite that adds any number not in the evidence is rejected."""
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import kpis as K
from engine.domains import delivery as D
from engine.llm import LLMError, chat_json
from engine.profile import to_datetime_clean, to_numeric_clean, yes_no

ON_TIME_GOAL = 80.0  # assumption, shown on every recommendation that uses it
UTIL_ALERT = 0.85
LEVELS = ["Low", "Medium", "High"]

RULES = {
    "quality": "open data-quality issues (medium/high severity) remain",
    "on_time": f"on-time % below the {ON_TIME_GOAL:g}% goal (assumption)",
    "priority_inversion": "highest-priority orders are on time less often than the lowest priority",
    "promise_gap": "median delivery is later than promised",
    "margin": "average fee − cost per order is negative",
    "hub_overload": f"an entity's utilisation is at or above {UTIL_ALERT:.0%}",
    "peak": "peak on-time % is at least 5 points below off-peak",
    "util_mismatch": "reported and calculated utilisation disagree by more than 50%",
    "segment_gap": "a segment's average is at least 25% below the overall average (non-delivery data)",
    "flag_concentration": "a yes/no flag is at least 1.5× more common in one segment",
    "ratio_gap": "a part-of rate (e.g. returned ÷ sold) is at least 1.5× higher in one segment",
    "decline": "the measure fell more than 10% between the first and second half of the period",
    "kpi_margin": "a margin KPI is negative",
    "infeasible": "the scenario leaves orders unserved or misses a target",
    "route_fragile": "the cheapest route differs between base, confirmed and full cost views",
    "route_cost": "hidden route costs are at least 2% of base cost or change a route's rank",
}


@dataclass
class Evidence:
    label: str
    value: float | str | None
    fmt: str = "number"  # number | pct | money | min | int | text
    formula: str = ""
    basis: str = "calculated"

    def text(self) -> str:
        v = self.value
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "n/a"
        if isinstance(v, str) or self.fmt == "text":
            return str(v)
        return {"pct": f"{v:.1f}%", "money": f"{v:,.2f}", "min": f"{v:,.1f} min", "int": f"{v:,.0f}"}.get(self.fmt, f"{v:,.2f}")


@dataclass
class Rec:
    id: str
    rule: str
    title: str
    action: str
    evidence: list[Evidence]
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    confidence: str = "High"
    impact: float = 0.0  # 0..1, ranking only
    impact_formula: str = ""
    plain: str = ""  # AI rewrite (checked); empty = use title + action
    rewrite_note: str = ""

    def facts(self) -> str:
        return "\n".join([self.title, self.action] + [f"{e.label}: {e.text()}" for e in self.evidence])


def _down(level: str, steps: int = 1) -> str:
    return LEVELS[max(0, LEVELS.index(level) - steps)]


@dataclass
class Inputs:
    df: pd.DataFrame
    ctx: K.Ctx
    issues: list = field(default_factory=list)
    quality_score: float | None = None
    kpis: list = field(default_factory=list)
    pack: D.Pack | None = None
    pack_metrics: dict | None = None
    cascade: dict | None = None
    days: int | None = None
    is_scenario: bool = False
    route: dict | None = None  # engine.routecost.compute result


def date_span(df: pd.DataFrame, roles: dict[str, str]) -> int | None:
    spans = []
    for c in [c for c, r in roles.items() if r == "date"]:
        d = to_datetime_clean(df[c]).dropna()
        if len(d):
            spans.append((d.max().normalize() - d.min().normalize()).days + 1)
    return min(spans) if spans else None


# ---------- rules ----------

def _quality(inp: Inputs) -> list[Rec]:
    open_ = [i for i in inp.issues if i.severity != "low"]
    if not open_:
        return []
    top = sorted(open_, key=lambda i: -i.count)[:3]
    ev = [Evidence("Quality score", inp.quality_score, "pct", "rows with no medium/high issue ÷ all rows")]
    ev += [Evidence(f"{i.rule.replace('_', ' ')} in '{i.column}'", i.count, "int", f"rows affected of {i.denominator:,}")
           for i in top]
    return [Rec("quality", "quality", "Fix the remaining data-quality issues before acting on these numbers",
                "Open the Quality tab and click 'Apply recommended fixes' (or choose a fix per issue).", ev,
                limitations=["Numbers on every tab include these rows until they are fixed."],
                impact=round(1 - (inp.quality_score or 100) / 100, 4), impact_formula="share of rows with an issue")]


def _delivery(inp: Inputs) -> list[Rec]:
    pk, m = inp.pack, inp.pack_metrics
    if not pk or not pk.active or not m:
        return []
    out, c = [], pk.cols
    heads = {h.label: h for h in m["heads"]}
    on = heads["On-time % (all eligible orders)"]
    seg = m["segments"]
    casc_on = None
    if inp.cascade and inp.is_scenario:
        st = inp.cascade["steps"].set_index("metric")
        casc_on = float(st.at["on-time %", "scenario"]) if "on-time %" in st.index else None

    def worst(key, col="on-time %", min_n=20):
        t = seg.get(key)
        if t is None or not len(t):
            return None
        t = t[t["eligible orders"] >= min_n].dropna(subset=[col])
        return t.sort_values(col).iloc[0] if len(t) else None

    # on-time below goal
    level = casc_on if casc_on is not None else on.value
    if level is not None and level < ON_TIME_GOAL:
        ev = [Evidence("On-time % (calculated from rows)", on.value, "pct", on.formula + f"; {on.filter}")]
        if casc_on is not None:
            ev.append(Evidence("On-time % after capacity effects", casc_on, "pct",
                               "cascade estimate: time + modelled queueing delay ≤ promised", "estimate"))
        acts = []
        for key in ("entity", "priority"):
            w = worst(key)
            if w is not None:
                ev.append(Evidence(f"Lowest on-time {c[key]}: {w[c[key]]}", float(w["on-time %"]), "pct",
                                   f"on time ÷ eligible orders where '{c[key]}' = {w[c[key]]} "
                                   f"({int(w['eligible orders'])} orders)"))
                acts.append(f"{c[key]} {w[c[key]]}")
        conf = "Medium" if casc_on is not None else "High"
        out.append(Rec("on_time", "on_time", f"On-time delivery is {level:.1f}%, below the {ON_TIME_GOAL:g}% goal",
                       "Start with " + " and ".join(acts) + ", where on-time is lowest." if acts else
                       "Review promised times and cycle times.", ev,
                       assumptions=[f"Goal of {ON_TIME_GOAL:g}% on time is an assumption, not from the data."],
                       confidence=conf, impact=round((ON_TIME_GOAL - level) / 100, 4),
                       impact_formula=f"({ON_TIME_GOAL:g} − on-time %) ÷ 100"))

    # priority inversion
    t = seg.get("priority")
    if t is not None and len(t):
        idx = t.set_index(c["priority"])
        if {"Critical", "Standard"} <= set(idx.index) and idx.at["Critical", "eligible orders"] > 0:
            cr, sd = idx.at["Critical", "on-time %"], idx.at["Standard", "on-time %"]
            if pd.notna(cr) and pd.notna(sd) and cr < sd:
                n_cr = int(idx.at["Critical", "eligible orders"])
                share = n_cr / max(int(t["eligible orders"].sum()), 1)
                out.append(Rec(
                    "priority_inversion", "priority_inversion",
                    f"Critical orders are on time {cr:.1f}% vs {sd:.1f}% for Standard",
                    "Give Critical orders first dispatch or a separate promise; today their promise is the hardest "
                    "to meet.",
                    [Evidence("On-time % Critical", float(cr), "pct", f"'{c['priority']}' = Critical, {n_cr} orders"),
                     Evidence("On-time % Standard", float(sd), "pct",
                              f"'{c['priority']}' = Standard, {int(idx.at['Standard', 'eligible orders'])} orders"),
                     Evidence("Avg promise gap Critical", float(idx.at["Critical", "avg promise gap (min)"]), "min",
                              f"mean('{c['actual']}' − '{c['promised']}')")],
                    assumptions=["Labels come from the confirmed value mapping (C01)."],
                    confidence="High" if n_cr >= 100 else "Medium",
                    impact=round(share * (sd - cr) / 100 + 0.15, 4),
                    impact_formula="Critical share of orders × on-time gap ÷ 100 + 0.15 (priority weight)"))

    # promise gap
    med, avg, p90 = heads["Median promise gap"], heads["Average promise gap"], heads["Late orders: 90th-percentile gap"]
    if med.value is not None and med.value > 0:
        ev = [Evidence("Median promise gap", med.value, "min", med.formula), Evidence("Average promise gap", avg.value,
                                                                                      "min", avg.formula),
              Evidence("90th-percentile gap", p90.value, "min", p90.formula)]
        tp = seg.get("priority")
        if tp is not None and len(tp):
            w = tp.sort_values("avg promise gap (min)", ascending=False).iloc[0]
            ev.append(Evidence(f"Largest gap {c['priority']}: {w[c['priority']]}", float(w["avg promise gap (min)"]),
                               "min", "mean gap in that segment"))
        out.append(Rec("promise_gap", "promise_gap", f"Half of deliveries arrive {med.value:.1f}+ minutes after the promise",
                       "Either set promised times from actual performance (by priority) or cut cycle time; "
                       "promises below what operations deliver guarantee late orders.", ev,
                       limitations=["Gap uses delivered orders with both times only."],
                       impact=round(min(1.0, (100 - (on.value or 0)) / 100 * 0.8), 4),
                       impact_formula="share of eligible orders late × 0.8"))

    # margin
    mg, fee = heads["Margin per order"], heads["Average fee"]
    loss = heads["Share of orders losing money"]
    if mg.value is not None and mg.value < 0 and fee.value:
        be_fee = -mg.value / fee.value * 100
        ev = [Evidence("Margin per order", mg.value, "money", mg.formula + f" ({mg.n:,} orders)"),
              Evidence("Orders losing money", loss.value, "pct", loss.formula),
              Evidence("Average fee", fee.value, "money", fee.formula),
              Evidence("Fee rise needed to break even", be_fee, "pct", "−margin per order ÷ average fee × 100",
                       "calculated")]
        ws = None
        for key in ("priority", "entity"):
            tt = seg.get(key)
            if tt is not None and len(tt):
                r = tt.sort_values("margin per order").iloc[0]
                ev.append(Evidence(f"Lowest margin {c[key]}: {r[c[key]]}", float(r["margin per order"]), "money",
                                   "mean (fee − cost) in that segment"))
                ws = ws or f"{c[key]} {r[c[key]]}"
        out.append(Rec("margin", "margin", f"Each order loses {abs(mg.value):,.2f} on average (fee − cost)",
                       f"A fee rise of about {be_fee:.1f}% (or the same cut in cost per order) breaks even"
                       + (f"; look first at {ws}." if ws else "."), ev,
                       assumptions=["Order volume does not fall when the fee rises (no price elasticity modelled)."],
                       limitations=[f"Margin = '{c['fee']}' − '{c['cost']}'; fixed entity costs are not included."]
                       + pk.notes,
                       impact=round(min(1.0, (loss.value or 0) / 100), 4), impact_formula="share of orders losing money"))

    # hub overload
    hub = m.get("hub") or {}
    ht = hub.get("table")
    if ht is not None and len(ht):
        ucol = "utilisation % [reported]" if ht["utilisation % [reported]"].notna().all() else \
            "utilisation % [calculated: orders ÷ capacity]"
        h = ht.dropna(subset=[ucol]).sort_values(ucol)
        if len(h) >= 2 and h[ucol].iloc[-1] >= 100 * UTIL_ALERT:
            hi, lo = h.iloc[-1], h.iloc[0]
            e = c["entity"]
            out.append(Rec("hub_overload", "hub_overload",
                           f"{e} {hi[e]} runs at {hi[ucol]:.1f}% utilisation; {lo[e]} at {lo[ucol]:.1f}%",
                           f"Shift load from {hi[e]} to {lo[e]}; queueing delay rises steeply above {UTIL_ALERT:.0%}.",
                           [Evidence(f"Utilisation {hi[e]}", float(hi[ucol]), "pct", ucol),
                            Evidence(f"Utilisation {lo[e]}", float(lo[ucol]), "pct", ucol),
                            Evidence(f"Wait multiplier {hi[e]}", float(hi["wait multiplier 1/(1−u) [model]"]), "number",
                                     "1 ÷ (1 − utilisation), capped at 95%", "estimate"),
                            Evidence(f"Avg cycle time {hi[e]}", hi["avg cycle time min [calculated]"], "min",
                                     f"mean '{c['actual']}'")],
                           assumptions=["Wait ∝ 1/(1 − utilisation) (queueing model), " + hub.get("calibration", "")],
                           limitations=[hub["mismatch"]] if hub.get("mismatch") else [],
                           confidence="Medium", impact=round(hi[ucol] / 100 - UTIL_ALERT + 0.05, 4),
                           impact_formula="highest utilisation − 0.85 + 0.05"))
        if hub.get("mismatch"):
            out.append(Rec("util_mismatch", "util_mismatch", "Reported utilisation does not match the orders in the file",
                           "Confirm real daily volumes per entity before using capacity results; the model uses the "
                           "reported figure.",
                           [Evidence("Average utilisation [reported]", float(ht["utilisation % [reported]"].mean()),
                                     "pct", f"mean of '{c['reported_util']}' per entity", "reported"),
                            Evidence("Average utilisation [calculated]",
                                     float(ht["utilisation % [calculated: orders ÷ capacity]"].mean()), "pct",
                                     f"orders in file per day ÷ '{c['capacity']}', mean per entity")],
                           limitations=[hub["mismatch"]],
                           confidence="High", impact=0.05, impact_formula="fixed 0.05 (data caveat)"))

    # peak
    tp = seg.get("peak")
    if tp is not None and len(tp):
        idx = tp.set_index(c["peak"])
        if {"Peak", "Off-Peak"} <= set(idx.index):
            pv, ov = idx.at["Peak", "on-time %"], idx.at["Off-Peak", "on-time %"]
            if pd.notna(pv) and pd.notna(ov) and ov - pv >= 5:
                share = idx.at["Peak", "eligible orders"] / max(int(tp["eligible orders"].sum()), 1)
                out.append(Rec("peak", "peak", f"Peak-time on-time is {pv:.1f}% vs {ov:.1f}% off-peak",
                               "Add riders/slots in peak windows or widen peak promises.",
                               [Evidence("On-time % Peak", float(pv), "pct", f"'{c['peak']}' = Peak"),
                                Evidence("On-time % Off-Peak", float(ov), "pct", f"'{c['peak']}' = Off-Peak"),
                                Evidence("Peak share of orders", 100 * float(share), "pct", "peak eligible ÷ all eligible")],
                               impact=round(float(share) * (ov - pv) / 100, 4),
                               impact_formula="peak share × on-time gap ÷ 100"))
    return out


def _generic(inp: Inputs) -> list[Rec]:
    df, roles, ent = inp.df, inp.ctx.roles, inp.ctx.entity_of
    out = []
    groups = [c for c, r in roles.items() if r in ("category", "location") and 2 <= df[c].nunique() <= 20]
    nums = [c for c, r in roles.items() if r in ("money", "quantity", "rating", "duration") and c not in ent]
    best = None
    for g in groups:
        for x in nums:
            v = to_numeric_clean(df[x])
            overall = v.mean()
            if not overall or not np.isfinite(overall) or overall <= 0:
                continue
            key = df[g].astype("string")
            t = v.groupby(key).agg(["mean", "count"])
            t = t[t["count"] >= 20]
            if len(t) < 2:
                continue
            ok = v.notna() & key.notna()
            between = (v[ok].groupby(key[ok]).transform("mean") - v[ok].mean()).pow(2).sum()
            if between / max(((v[ok] - v[ok].mean()) ** 2).sum(), 1e-9) > 0.8:
                continue  # the group defines the value (e.g. a product's price): not a performance gap
            lo = t.sort_values("mean").iloc[0]
            gap = 1 - lo["mean"] / overall
            score = gap * lo["count"] / len(df)
            if gap >= 0.25 and (best is None or score > best[0]):
                best = (score, g, x, t["mean"].idxmin(), float(lo["mean"]), float(overall), int(lo["count"]), gap)
    if best:
        score, g, x, val, lo_m, all_m, n, gap = best
        out.append(Rec("segment_gap", "segment_gap", f"{g} '{val}' averages {lo_m:,.2f} {x} vs {all_m:,.2f} overall",
                       f"Investigate what is different about {g} '{val}' ({gap:.0%} below average).",
                       [Evidence(f"Average {x} where {g} = {val}", lo_m, "number", f"mean '{x}', {n:,} rows"),
                        Evidence(f"Average {x} overall", all_m, "number", f"mean '{x}', all rows")],
                       limitations=["A difference in averages is not a cause; check mix effects."],
                       confidence="High" if n >= 100 else "Medium", impact=round(min(1.0, score * 2), 4),
                       impact_formula="(1 − segment mean ÷ overall mean) × segment share × 2"))
    out += _ratio_gaps(df, roles, ent, groups)
    for f in [c for c, r in roles.items() if r == "flag"]:
        yn = df[f].map(yes_no)
        if yn.notna().mean() < 0.8:
            continue
        yes = (yn == "yes").astype(float).where(yn.notna())
        rate = yes.mean()
        if not rate or rate >= 0.9:
            continue
        for g in groups:
            t = yes.groupby(df[g].astype("string")).agg(["mean", "count"])
            t = t[t["count"] >= 20]
            if len(t) < 2:
                continue
            top = t.sort_values("mean").iloc[-1]
            if top["mean"] >= 1.5 * rate:
                name = t["mean"].idxmax()
                out.append(Rec(f"flag_concentration:{f}:{g}", "flag_concentration",
                               f"'{f}' is {100 * top['mean']:.1f}% in {g} '{name}' vs {100 * rate:.1f}% overall",
                               f"Look into {g} '{name}' first.",
                               [Evidence(f"'{f}' = yes in {g} {name}", 100 * float(top["mean"]), "pct",
                                         f"yes ÷ rows with a value, {int(top['count'])} rows"),
                                Evidence(f"'{f}' = yes overall", 100 * float(rate), "pct", "yes ÷ rows with a value")],
                               limitations=["Yes/no read from the values (yes, y, true, 1)."],
                               confidence="High" if top["count"] >= 100 else "Medium",
                               impact=round(min(1.0, float(top["mean"] - rate) * top["count"] / len(df) * 3), 4),
                               impact_formula="(segment rate − overall rate) × segment share × 3"))
                break
    for k in inp.kpis:
        if k.calc == "growth" and k.value is not None and k.value < -10:
            out.append(Rec(f"decline:{k.title}", "decline", f"{k.title}: {k.value:.1f}%",
                           "Find which segments drove the drop before planning on the second-half level.",
                           [Evidence(k.title, k.value, "pct", k.formula)], limitations=list(k.warnings),
                           impact=round(min(1.0, -k.value / 100), 4), impact_formula="−growth % ÷ 100"))
        if k.calc == "margin" and k.value is not None and k.value < 0:
            out.append(Rec(f"kpi_margin:{k.title}", "kpi_margin", f"{k.title} is negative ({k.value:,.2f})",
                           "Check pricing against cost for the rows that lose money.",
                           [Evidence(k.title, k.value, "number", k.formula + f" ({k.denominator})")],
                           impact=0.3, impact_formula="fixed 0.3 (negative margin)"))
    return out


def _ratio_gaps(df, roles, ent, groups) -> list[Rec]:
    """Part-of pairs (a ≤ b on ≥99% of rows, e.g. returned units vs units): a segment whose rate is ≥1.5× overall."""
    nums = [c for c, r in roles.items() if r in ("quantity", "measure", "money") and c not in ent]
    vals = {c: to_numeric_clean(df[c]) for c in nums}
    best = None
    for a in nums:
        for b in nums:
            kind = lambda c: "money" if roles[c] == "money" else "count"
            if a == b or kind(a) != kind(b):  # like for like: money with money, counts with counts
                continue
            x, y = vals[a], vals[b]
            ok = x.notna() & y.notna() & (y > 0)
            if ok.sum() < 50 or (x[ok] < 0).any() or (x[ok] <= y[ok]).mean() < 0.99:
                continue
            rate = x[ok].sum() / y[ok].sum()
            if not 0 < rate < 0.5:
                continue
            for g in groups:
                t = pd.DataFrame({"a": x[ok], "b": y[ok], "g": df.loc[ok, g].astype("string")}).groupby("g").agg(
                    a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
                t = t[t["n"] >= 20]
                if len(t) < 2:
                    continue
                t["rate"] = t["a"] / t["b"]
                top = t.sort_values("rate").iloc[-1]
                lift = top["rate"] / rate
                if lift >= 1.5 and (best is None or lift > best[0]):
                    best = (lift, a, b, g, t["rate"].idxmax(), float(top["rate"]), float(rate), int(top["n"]))
    if not best:
        return []
    lift, a, b, g, name, r_top, r_all, n = best
    return [Rec(f"ratio_gap:{a}:{b}:{g}", "ratio_gap", f"{a} ÷ {b} is {100 * r_top:.1f}% in {g} '{name}' vs "
                f"{100 * r_all:.1f}% overall", f"Investigate {g} '{name}': its rate is {lift:.1f}× the average.",
                [Evidence(f"{a} ÷ {b} in {g} {name}", 100 * r_top, "pct", f"sum('{a}') ÷ sum('{b}'), {n:,} rows"),
                 Evidence(f"{a} ÷ {b} overall", 100 * r_all, "pct", f"sum('{a}') ÷ sum('{b}'), all rows")],
                limitations=[f"'{a}' is treated as part of '{b}' because it is ≤ it on 99%+ of rows."],
                confidence="High" if n >= 100 else "Medium",
                impact=round(min(1.0, (r_top - r_all) * n / len(df) * 5), 4),
                impact_formula="(segment rate − overall rate) × segment share × 5")]


def _scenario(inp: Inputs) -> list[Rec]:
    cz = inp.cascade
    if not inp.is_scenario or not cz or cz["verdict"] == "Feasible":
        return []
    st = cz["steps"].set_index("metric")
    uns = float(st.at["unserved orders per day", "scenario"])
    load = float(st.at["capacity per day (all entities)", "scenario"]) or 1
    ev = [Evidence("Unserved orders per day", uns, "number", st.at["unserved orders per day", "formula"], "estimate"),
          Evidence("Highest utilisation %", float(st.at["highest utilisation %", "scenario"]), "pct",
                   st.at["highest utilisation %", "formula"], "estimate"),
          Evidence("On-time % estimate", float(st.at["on-time %", "scenario"]), "pct",
                   st.at["on-time %", "formula"], "estimate")]
    tg = cz.get("targets")
    if tg is not None and len(tg):
        for r in tg.to_dict("records"):
            ev.append(Evidence(f"Target {r['target']}", r["scenario estimate"], "number",
                               "scenario estimate; " + ("met" if r["met"] else "NOT met"), "estimate"))
    act = (f"Add about {uns:,.0f} orders/day of capacity, raise the utilisation cap, or spread the change over more "
           "days." if uns > 0 else "Targets are not reachable under this scenario; relax the target or add capacity.")
    return [Rec("infeasible", "infeasible", f"Scenario is {cz['verdict'].lower()}: " + "; ".join(cz["reasons"])[:160],
                act, ev, assumptions=[f"{k}: {v}" for k, v in cz["assumptions"].items()],
                limitations=["All scenario values are model estimates."], confidence="Medium",
                impact=round(min(1.0, 0.5 + uns / load), 4), impact_formula="0.5 + unserved ÷ capacity")]


def _route(inp: Inputs) -> list[Rec]:
    r = inp.route
    if not r:
        return []
    t, w = r["totals"], r["winners"]
    moved = int(r["routes"]["rank changes"].sum())
    ev = [Evidence("Hidden cost, all statuses", t["hidden_all"], "money", "Σ full cost − Σ base cost over all orders"),
          Evidence("Hidden cost, confirmed only", t["hidden_confirmed"], "money",
                   "Σ (base + confirmed components) − Σ base cost"),
          Evidence("Hidden cost % of base", t["hidden_pct"], "pct", "hidden cost (all statuses) ÷ base cost × 100"),
          Evidence("Routes whose rank changes", moved, "int", "rank differs between the three views"),
          Evidence("Cheapest route, base cost only", w["base"], "text", "lowest base cost per order"),
          Evidence("Cheapest route, confirmed full cost", w["confirmed"], "text", "lowest (base + confirmed) per order"),
          Evidence("Cheapest route, full cost (all statuses)", w["all"], "text", "lowest full cost per order"),
          Evidence("Decision verdict", r["verdict"], "text", r["verdict_text"])]
    lim = [f"Routes with fewer than {r['min_orders']} orders are not ranked."] + list(r["notes"])
    assum = [f"{c['component']}: {c['formula']} ({c['status']})" for c in r["components"].to_dict("records")
             if c["component"] != "base cost"]
    if r["verdict"] == "Fragile":
        f = r["flips"][-1]
        return [Rec("route", "route_fragile",
                    f"Route choice is fragile: verify {f['component']} before choosing route {f['cheapest before']}",
                    f"{r['verdict_text']} Confirm that figure first; until then the decision view picks route "
                    f"{r['decision']}.", ev, assum, lim, confidence="Medium", impact=0.6,
                    impact_formula="fixed 0.6: the decision changes with unverified inputs")]
    if (t["hidden_pct"] or 0) >= 2 or moved:
        return [Rec("route", "route_cost",
                    f"Rank routes on full cost, not base cost: hidden costs add {t['hidden_pct']:.1f}% to base",
                    f"Route {w['all']} stays cheapest under every view, but {moved} route(s) change rank once hidden "
                    "costs are included. Use full cost per order for route decisions.", ev, assum, lim,
                    impact=round(min(1.0, (t["hidden_pct"] or 0) / 20), 4), impact_formula="hidden % of base ÷ 20")]
    return []


def generate(inp: Inputs) -> list[Rec]:
    recs = _quality(inp) + _scenario(inp) + _route(inp) + _delivery(inp)
    if not (inp.pack and inp.pack.active):
        recs += _generic(inp)
    days = inp.days if inp.days is not None else date_span(inp.df, inp.ctx.roles)
    for r in recs:
        if days is not None and days < K.MIN_TREND_DAYS and r.rule not in ("quality", "util_mismatch"):
            r.limitations.append(f"Only {days} days of data; patterns may not hold over a longer period.")
            r.confidence = _down(r.confidence)
        if inp.quality_score is not None and inp.quality_score < 80 and r.rule != "quality":
            r.limitations.append(f"Quality score is {inp.quality_score}%.")
            r.confidence = _down(r.confidence)
    return sorted(recs, key=lambda r: -r.impact)


def compare(base: list[Rec], scen: list[Rec]) -> list[dict]:
    """Before vs after: one entry per recommendation id, ranked by the scenario impact."""
    b = {r.id: r for r in base}
    s = {r.id: r for r in scen}
    out = []
    for rid in [r.id for r in scen] + [r.id for r in base if r.id not in s]:
        rb, rs = b.get(rid), s.get(rid)
        cur = rs or rb
        status = "unchanged"
        if rb and not rs:
            status = "resolved in scenario"
        elif rs and not rb:
            status = "new in scenario"
        elif [e.text() for e in rb.evidence] != [e.text() for e in rs.evidence]:
            status = "changed"
        bev = {e.label: e.text() for e in rb.evidence} if rb else {}
        sev = {e.label: e.text() for e in rs.evidence} if rs else {}
        rows = [{"evidence": lab, "before (baseline)": bev.get(lab, "—"), "after (scenario)": sev.get(lab, "—"),
                 "formula": next((e.formula for e in (rs or rb).evidence if e.label == lab), ""),
                 "basis": next((e.basis for e in (rs or rb).evidence if e.label == lab), "")}
                for lab in dict.fromkeys(list(sev) + list(bev))]
        out.append({"id": rid, "rec": cur, "before": rb, "after": rs, "status": status, "table": pd.DataFrame(rows)})
    return out


# ---------- AI wording ----------

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def unsupported_numbers(text: str, facts: str) -> list[str]:
    have = [float(x.replace(",", "")) for x in _NUM.findall(facts) if x.replace(",", "").replace(".", "").strip("-")]
    bad = []
    for raw in _NUM.findall(text):
        s = raw.replace(",", "").rstrip(".")
        if not s.strip("-").replace(".", ""):
            continue
        v = float(s)
        dec = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** -dec + 1e-9
        if not any(abs(h - v) <= tol or abs(abs(h) - abs(v)) <= tol for h in have):
            bad.append(raw)
    return bad


def rewrite(recs: list[Rec], cache_key: str) -> str:
    """Ask the AI for plain-language wording. Returns a status message; recs keep rule text on any failure."""
    if not recs:
        return ""
    items = [{"id": r.id, "facts": r.facts()} for r in recs]
    prompt = ("Rewrite each recommendation for a non-technical manager in 1-2 short sentences. Use ONLY the numbers "
              "in its facts (you may round them as written); add no new numbers, no new causes, no new claims.\n"
              f"{json.dumps(items, ensure_ascii=False)}\n"
              'Return ONLY JSON: {"<id>": "<plain text>", ...}')
    try:
        raw = chat_json(prompt, tier="fast", cache_key=f"recs:{cache_key}",
                        system="You rewrite analysis findings in plain English. You never compute or add numbers.")
    except LLMError as e:
        return f"AI wording unavailable; showing rule text. ({str(e)[:120]})"
    raw = raw if isinstance(raw, dict) else {}
    rejected = 0
    for r in recs:
        t = raw.get(r.id)
        if not isinstance(t, str) or not t.strip():
            continue
        bad = unsupported_numbers(t, r.facts())
        if bad:
            r.plain, r.rewrite_note = "", f"AI wording rejected: it added number(s) {', '.join(bad)} not in the evidence."
            rejected += 1
        else:
            r.plain, r.rewrite_note = t.strip(), "Plain-language wording by AI, checked: every number is from the evidence."
    return f"AI rewrote {sum(bool(r.plain) for r in recs)} recommendation(s)" + \
        (f"; {rejected} rejected for adding numbers." if rejected else ".")
