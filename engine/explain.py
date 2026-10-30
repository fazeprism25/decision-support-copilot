"""Explain tab content, generated from the objects the app actually ran on the current dataset (roles, fixes,
value mapping, KPI results, domain pack, scenario, recommendations). Nothing here is fixed prose about a dataset."""
from dataclasses import dataclass, field

import pandas as pd

from engine import kpis as K
from engine import recommend as R
from engine.domains import delivery as D
from engine.quality import FIX_LABELS


@dataclass
class Section:
    title: str
    lines: list[str] = field(default_factory=list)
    table: pd.DataFrame | None = None


def build(*, name: str, rows_raw: int, rows_clean: int, roles: pd.DataFrame, entity_of: dict[str, str],
          fixes: list, vmap: dict, vmap_confirmed: bool, kpis: list, pack: D.Pack | None, pack_metrics: dict | None,
          params=None, notes: list[str] | None = None, cascade: dict | None = None, recs: list | None = None,
          issues: list | None = None, quality_score: float | None = None, days: int | None = None,
          ai_status: str = "", rows_scenario: int | None = None, route: dict | None = None) -> list[Section]:
    out = []

    s = Section("1. Data used")
    s.lines.append(f"File **{name}**: {rows_raw:,} rows uploaded → **{rows_clean:,} rows after quality fixes** "
                   f"({rows_raw - rows_clean:,} removed). Every tab (KPIs, Ask, What-if, Recommendations) uses these "
                   "cleaned rows.")
    if fixes:
        s.table = pd.DataFrame([{"step": i + 1, "column": f.column, "rule": f.rule, "action": FIX_LABELS.get(f.action, f.action),
                                 "rows removed": f.rows_before - f.rows_after, "values changed": f.cells_changed}
                                for i, f in enumerate(fixes)])
    else:
        s.lines.append("No quality fixes applied: numbers include every uploaded row.")
    if days is not None:
        s.lines.append(f"Date range covers **{days} days**.")
    s.lines.append(f"AI status: {ai_status or 'n/a'}. The AI never computes numbers; pandas does.")
    out.append(s)

    s = Section("2. Column roles (how each column was read)")
    s.lines.append("Roles drive every calculation. Source: AI = model label confirmed by data statistics; rules = "
                   "code-only guess; user = your override. A role whose check failed is not used.")
    s.table = roles[["column", "role", "source", "check", "note"]].copy()
    if entity_of:
        s.lines.append("Entity attributes (one value per group, never summed across rows): " +
                       ", ".join(f"'{a}' per '{e}'" for a, e in entity_of.items()) + ".")
    out.append(s)

    s = Section("3. Category value mapping (C01)")
    n = sum(1 for m in vmap.values() for raw, lab in m.items() if lab and lab != raw)
    if vmap and vmap_confirmed:
        s.lines.append(f"Confirmed mapping applied after quality fixes: {n} value(s) relabelled in "
                       f"{len(vmap)} column(s).")
        s.table = pd.DataFrame([{"column": c, "original": raw, "standard label": lab}
                                for c, m in vmap.items() for raw, lab in m.items() if lab])
    elif vmap:
        s.lines.append("A mapping is proposed but **not confirmed**, so original labels are used everywhere.")
    else:
        s.lines.append("No value mapping applies to this data.")
    out.append(s)

    s = Section("4. KPI formulas")
    if kpis:
        s.table = pd.DataFrame([{"KPI": k.title, "value": K.fmt_value(k.value, k.fmt), "basis": k.basis,
                                 "formula": k.formula, "filter": k.filter, "denominator": k.denominator,
                                 "excluded": "; ".join(k.exclusions), "warnings": "; ".join(k.warnings)}
                                for k in kpis])
    else:
        s.lines.append("Open the KPIs tab once to compute KPIs.")
    out.append(s)

    s = Section("5. Delivery-operations pack")
    if pack is None:
        s.lines.append("Not evaluated.")
    elif not pack.active:
        s.lines.append(f"**Off** for this data ({pack.reason}). Generic recommendation rules are used instead.")
    else:
        s.lines.append(f"**On** ({pack.reason}).")
        s.lines += [f"- {n}" for n in pack.notes]
        s.table = pd.DataFrame(D.formulas(pack), columns=["metric", "formula", "filter / note"])
        if pack_metrics:
            s.lines += [f"- {h.label} = **{D.fmt(h)}** [{h.basis}] · {h.formula} · filter: {h.filter} · n = {h.n:,}"
                        for h in pack_metrics["heads"]]
            hub = pack_metrics.get("hub") or {}
            if hub.get("calibration"):
                s.lines.append(f"- Hub load model: {hub['calibration']} (utilisation basis: {hub.get('util_basis', '')}).")
            if hub.get("mismatch"):
                s.lines.append(f"- ⚠️ {hub['mismatch']}")
    out.append(s)

    s = Section("6. Scenario (What-if)")
    if params is None or params.is_empty():
        s.lines.append("No constraint or slider change active: scenario = baseline.")
    else:
        s.lines.append("Active parameters (all constraints and sliders stacked):")
        s.lines.append(f"`{params.model_dump_json(exclude_defaults=True)}`")
        s.lines += [f"- {n}" for n in notes or []]
        if rows_scenario is not None:
            s.lines.append(f"- Scenario rows: {rows_scenario:,} (baseline {rows_clean:,}).")
    if cascade:
        s.lines.append(f"Cascade over {cascade['days']} days; load basis: {cascade['basis']}; {cascade['calibration']}.")
        s.lines += [f"- Assumption · {k}: {v}" for k, v in cascade["assumptions"].items()]
        s.lines.append(f"- Feasibility: **{cascade['verdict']}** — " + "; ".join(cascade["reasons"]))
        s.table = cascade["steps"][["step", "metric", "formula", "basis"]]
    else:
        s.lines.append("Capacity cascade not run (needs an entity, a capacity and a date column).")
    out.append(s)

    s = Section("7. Recommendation rules")
    s.lines.append("Each recommendation is produced by a rule over computed numbers; the AI may only reword it and is "
                   "rejected if it adds a number. Ranked by impact score (0-1, formula per rule).")
    s.lines.append("Confidence starts High for calculated evidence and Medium for model estimates; it drops one level "
                   f"if the data covers fewer than {K.MIN_TREND_DAYS} days or the quality score is below 80%, and "
                   "segment rules use Medium below 100 rows.")
    fired = {r.rule for r in recs or []}
    s.table = pd.DataFrame([{"rule": k, "fires when": v, "fired now": "yes" if k in fired else "no"}
                            for k, v in R.RULES.items()])
    for r in recs or []:
        s.lines.append(f"- **{r.title}** · impact {r.impact:.2f} = {r.impact_formula} · confidence {r.confidence}")
    out.append(s)

    s = Section("8. Exclusions and limitations")
    for k in kpis:
        for e in k.exclusions:
            s.lines.append(f"- {k.title}: {e}")
    if days is not None and days < K.MIN_TREND_DAYS:
        s.lines.append(f"- Only {days} days of data: trends, growth and weekly patterns are not reliable.")
    open_ = [i for i in issues or [] if i.severity != "low"]
    if open_:
        s.lines.append(f"- Quality score {quality_score}%: {len(open_)} open medium/high issue(s) still in the data "
                       "(e.g. " + "; ".join(i.detail for i in open_[:3]) + ").")
    unmapped = roles[(roles["role"] == "ignore") | (roles["check"] == "fail")]["column"].tolist()
    if unmapped:
        s.lines.append(f"- Not used in any calculation (ignored or failed role check): {', '.join(unmapped)}.")
    if pack is not None and pack.active and pack_metrics and (pack_metrics.get("hub") or {}).get("mismatch"):
        s.lines.append("- Capacity results depend on reported utilisation, which the file's own volumes do not confirm.")
    if cascade:
        s.lines.append("- Every scenario value is an estimate based on the assumptions in section 6.")
    s.lines.append("- Correlation is not causation: segment differences show where to look, not why.")
    out.append(s)

    s = Section("9. Route cost (cost analytics)")
    if route:
        t = route["totals"]
        s.lines.append(f"Route = {' × '.join(route['route_cols'])}. Cost per order is ranked three ways: base cost only, "
                       "base + confirmed components, and base + every component (confirmed, provisional, unverified)."
                       + (f" Decision view leaves out {' & '.join(route['excluded'])} components." if route["excluded"]
                          else ""))
        s.lines.append(f"**{route['verdict_text']}**")
        s.lines.append(f"Hidden cost: {t['hidden_all']:,.2f} in total ({t['hidden_pct'] or 0:.1f}% of base), of which "
                       f"{t['hidden_confirmed']:,.2f} is confirmed. Status: calculated from data = confirmed; your "
                       "entries = unverified unless you state otherwise; a row-level verification column is read when "
                       "present.")
        for n in route["notes"]:
            s.lines.append(f"- {n}")
        s.table = route["components"][["component", "status", "source", "formula"]]
    else:
        s.lines.append("Not computed: set a route grouping and a base cost column on the Route cost tab.")
    out.append(s)
    return out
