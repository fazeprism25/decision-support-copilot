"""Scenario engine: one Params object, row-level changes, and the C10 capacity cascade
(capacity → allocation → utilisation → wait/cycle time → on-time % → cost & margin → feasibility).
Column choices come from roles/setup; nothing here names a dataset's columns."""
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from engine.profile import to_datetime_clean, to_numeric_clean

UTIL_CEILING = 0.95  # wait model blows up at 100%; capped, labelled as an assumption
SCALAR_FIELDS = ("util_cap", "moved_extra_cost", "moved_extra_minutes")


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeasureChange(_M):
    column: str
    pct: float | None = None
    abs: float | None = None


class Exclusion(_M):
    column: str
    op: Literal["==", "!=", "in", "not_in", ">", "<", ">=", "<="]
    value: Any


class CapacityChange(_M):
    entity_value: str
    pct: float | None = Field(None, description="% change to capacity, -100 = fully down")
    abs: float | None = None
    days: float | None = Field(None, description="number of days affected; None = whole period")


class PriceChange(_M):
    column: str
    pct: float
    cap_pct: float | None = None


class Target(_M):
    metric: Literal["on_time_pct", "max_utilisation_pct", "margin_per_order", "unserved_per_day", "avg_cycle_time"]
    op: Literal[">=", "<="]
    value: float
    segment_column: str | None = Field(None, description="optional: only rows where this column == segment_value")
    segment_value: str | None = None

    def label(self) -> str:
        seg = f" for {self.segment_column} = {self.segment_value}" if self.segment_column else ""
        return f"{self.metric}{seg} {self.op} {self.value:g}"


class Params(_M):
    measure_changes: list[MeasureChange] = []
    exclusions: list[Exclusion] = []
    capacity_changes: list[CapacityChange] = []
    price_changes: list[PriceChange] = []
    targets: list[Target] = []
    util_cap: float | None = Field(None, description="max utilisation receiving entities may reach, 0-1")
    moved_extra_cost: float | None = Field(None, description="assumed extra cost per re-allocated order")
    moved_extra_minutes: float | None = Field(None, description="assumed extra minutes per re-allocated order")

    def is_empty(self) -> bool:
        return self == Params()


def combine(parts: list[Params]) -> Params:
    """Constraints stack: lists concatenate, scalar assumptions take the latest value."""
    out = Params()
    for p in parts:
        for f in ("measure_changes", "exclusions", "capacity_changes", "price_changes", "targets"):
            getattr(out, f).extend(getattr(p, f))
        for f in SCALAR_FIELDS:
            if getattr(p, f) is not None:
                setattr(out, f, getattr(p, f))
    return out


@dataclass
class Setup:
    """Which columns play which part in the cascade. Filled from roles/KPI plan, editable by the user."""
    entity: str | None = None
    capacity: str | None = None
    reported_util: str | None = None
    load_basis: str = "observed"  # observed | reported
    date: str | None = None
    actual: str | None = None
    promised: str | None = None
    status: str | None = None
    completed_values: list[str] = field(default_factory=list)
    fee: str | None = None
    cost: str | None = None

    def cascade_ready(self) -> list[str]:
        need = {"entity": self.entity, "capacity": self.capacity, "date": self.date}
        return [k for k, v in need.items() if not v]


def guess_setup(df: pd.DataFrame, roles: dict[str, str], entity_of: dict[str, str], plan: list[dict],
                profile: dict) -> Setup:
    s = Setup()
    for it in plan:
        p = it.get("params") or {}
        if it.get("calc") == "on_time_rate":
            s.actual, s.promised = p.get("actual"), p.get("promised")
            s.status, s.completed_values = p.get("status"), list(p.get("completed_values") or [])
        if it.get("calc") == "margin":
            s.fee, s.cost = p.get("revenue"), p.get("cost")
    num_attrs = [a for a, e in entity_of.items() if roles.get(a) in ("quantity", "measure")]
    cap = [a for a in num_attrs if re.search(r"capac|limit|max|throughput", a, re.I)] or \
          [a for a in num_attrs if roles.get(a) == "quantity"]
    if cap:
        s.capacity, s.entity = cap[0], entity_of[cap[0]]
        util = [a for a, e in entity_of.items() if e == s.entity and a != s.capacity and
                (profile.get(a, {}).get("percent_sign") or re.search(r"util|occup", a, re.I))]
        if util:
            s.reported_util, s.load_basis = util[0], "reported"
    dates = sorted([c for c, r in roles.items() if r == "date"], key=lambda c: df[c].nunique())
    s.date = dates[0] if dates else None
    return s


# ---------- validation ----------

def validate_params(p: Params, df: pd.DataFrame, roles: dict[str, str], setup: Setup) -> list[str]:
    errs = []
    numeric = {"money", "quantity", "duration", "measure", "rating"}
    for m in p.measure_changes:
        if roles.get(m.column) not in numeric:
            errs.append(f"'{m.column}' is not a numeric column")
        if m.pct is None and m.abs is None:
            errs.append(f"change to '{m.column}' needs pct or abs")
        if m.pct is not None and not -100 <= m.pct <= 500:
            errs.append(f"{m.pct}% change to '{m.column}' is outside -100..500")
    for x in p.exclusions:
        if x.column not in df:
            errs.append(f"'{x.column}' is not a column")
            continue
        vals = x.value if isinstance(x.value, list) else [x.value]
        if x.op in ("==", "!=", "in", "not_in"):
            present = set(df[x.column].dropna().astype("string"))
            missing = [v for v in vals if str(v) not in present]
            if missing:
                errs.append(f"value(s) {missing} not found in '{x.column}'")
        elif to_numeric_clean(pd.Series(vals)).isna().any():
            errs.append(f"'{x.op}' on '{x.column}' needs a number")
    for c in p.capacity_changes:
        if not setup.entity:
            errs.append("capacity change needs an entity column (What-if setup)")
            break
        if c.entity_value not in set(df[setup.entity].dropna().astype("string")):
            errs.append(f"'{c.entity_value}' is not a value of '{setup.entity}'")
        if c.pct is None and c.abs is None:
            errs.append("capacity change needs pct or abs")
        if c.pct is not None and not -100 <= c.pct <= 500:
            errs.append(f"capacity change {c.pct}% is outside -100..500")
        if c.days is not None and c.days <= 0:
            errs.append("days must be positive")
    for pc in p.price_changes:
        if roles.get(pc.column) != "money":
            errs.append(f"'{pc.column}' is not a money column")
    for t in p.targets:
        if t.segment_column:
            if t.segment_column not in df:
                errs.append(f"'{t.segment_column}' is not a column")
            elif str(t.segment_value) not in set(df[t.segment_column].dropna().astype("string")):
                errs.append(f"'{t.segment_value}' not found in '{t.segment_column}'")
            if t.metric != "on_time_pct":
                errs.append("segment targets are supported for on_time_pct only")
    if p.util_cap is not None and not 0.1 <= p.util_cap <= 1.0:
        errs.append("util_cap must be between 0.1 and 1.0")
    return errs


# ---------- row-level scenario ----------

def apply_rows(df: pd.DataFrame, p: Params) -> tuple[pd.DataFrame, list[str]]:
    """Exclusions, % / absolute changes to measures, price changes (with cap). Returns new df + notes."""
    out, notes = df.copy(), []
    for x in p.exclusions:
        s = out[x.column]
        vals = x.value if isinstance(x.value, list) else [x.value]
        if x.op in ("==", "in"):
            hit = s.astype("string").isin([str(v) for v in vals])
        elif x.op in ("!=", "not_in"):
            hit = ~s.astype("string").isin([str(v) for v in vals])
        else:
            num, v = to_numeric_clean(s), float(to_numeric_clean(pd.Series([vals[0]]))[0])
            hit = {">": num > v, "<": num < v, ">=": num >= v, "<=": num <= v}[x.op]
        hit = hit.fillna(False).astype(bool)
        notes.append(f"Excluded {int(hit.sum()):,} rows where '{x.column}' {x.op} {x.value}")
        out = out[~hit]
    for m in p.measure_changes:
        v = to_numeric_clean(out[m.column])
        out[m.column] = v * (1 + (m.pct or 0) / 100) + (m.abs or 0)
        notes.append(f"'{m.column}' changed by {m.pct or 0:+g}%" + (f" and {m.abs:+g}" if m.abs else ""))
    for pc in p.price_changes:
        eff = min(pc.pct, pc.cap_pct) if pc.cap_pct is not None else pc.pct
        out[pc.column] = to_numeric_clean(out[pc.column]) * (1 + eff / 100)
        notes.append(f"'{pc.column}' changed by {eff:+g}%" + (f" (requested {pc.pct:+g}%, capped at {pc.cap_pct:g}%)"
                                                              if pc.cap_pct is not None and pc.pct > pc.cap_pct else ""))
    return out.reset_index(drop=True), notes


# ---------- cascade ----------

def _f(u):
    return 1 / (1 - np.minimum(u, UTIL_CEILING))


def _as_fraction(x: float) -> float:
    return x / 100 if x > 1.5 else x


@dataclass
class EntityStats:
    name: str
    rows: int
    capacity: float
    reported_util: float | None
    mean_time: float | None
    fee: float
    cost: float
    actual: np.ndarray
    promised: np.ndarray


def entity_stats(df: pd.DataFrame, s: Setup) -> tuple[list[EntityStats], int]:
    d = to_datetime_clean(df[s.date]).dropna()
    days = max(int((d.max().normalize() - d.min().normalize()).days) + 1, 1) if len(d) else 1
    eligible = pd.Series(True, index=df.index)
    if s.status and s.completed_values:
        eligible = df[s.status].astype("string").isin(s.completed_values)
    a = to_numeric_clean(df[s.actual]) if s.actual else pd.Series(np.nan, index=df.index)
    pr = to_numeric_clean(df[s.promised]) if s.promised else pd.Series(np.nan, index=df.index)
    fee = to_numeric_clean(df[s.fee]) if s.fee else pd.Series(0.0, index=df.index)
    cost = to_numeric_clean(df[s.cost]) if s.cost else pd.Series(0.0, index=df.index)
    cap = to_numeric_clean(df[s.capacity])
    rep = to_numeric_clean(df[s.reported_util]) if s.reported_util else None
    out = []
    for name, idx in df.groupby(df[s.entity].astype("string")).groups.items():
        ok = eligible[idx] & a[idx].notna() & pr[idx].notna()
        ru = rep[idx].dropna() if rep is not None else pd.Series(dtype=float)
        out.append(EntityStats(
            str(name), len(idx), float(cap[idx].dropna().iloc[0]) if cap[idx].notna().any() else np.nan,
            _as_fraction(float(ru.mean())) if len(ru) else None,
            float(a[idx][ok].mean()) if ok.any() else None,
            float(fee[idx].mean()) if fee[idx].notna().any() else 0.0,
            float(cost[idx].mean()) if cost[idx].notna().any() else 0.0,
            a[idx][ok].to_numpy(float), pr[idx][ok].to_numpy(float)))
    return out, days


def _allocate(load: dict, cap: dict, util_cap: float):
    kept = {e: min(load[e], cap[e]) for e in load}
    excess = {e: load[e] - kept[e] for e in load}
    flow = {e: {} for e in load}
    room = {e: max(0.0, util_cap * cap[e] - kept[e]) if cap[e] > 0 else 0.0 for e in load}
    order = sorted([e for e in load if excess[e] <= 0 and cap[e] > 0], key=lambda e: kept[e] / cap[e])
    for src in sorted(excess, key=lambda e: -excess[e]):
        for r in order:
            if excess[src] <= 1e-9:
                break
            take = min(room[r], excess[src])
            if take > 0:
                flow[src][r] = flow[src].get(r, 0) + take
                room[r] -= take
                excess[src] -= take
    received = {r: sum(flow[e].get(r, 0) for e in load) for r in load}
    final = {e: kept[e] + received[e] for e in load}
    return kept, flow, excess, final  # excess left over = unserved


def calibrate(stats: list[EntityStats], util: dict, wait_share: float = 0.25) -> tuple[float, str]:
    """Slope b in cycle_time ≈ a + b / (1 - utilisation). Fit across entities; fall back to an assumption."""
    pts = [(float(_f(util[e.name])), e.mean_time) for e in stats if e.mean_time is not None and util.get(e.name) is not None]
    if len(pts) >= 3:
        x, y = np.array(pts).T
        if np.ptp(x) > 0:
            b, a = np.polyfit(x, y, 1)
            if b > 0:
                r2 = 1 - ((y - (a + b * x)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
                return float(b), (f"calibrated from {len(pts)} entities: cycle time ≈ {a:.1f} + {b:.2f} / (1 − utilisation) "
                                  f"(R² = {r2:.2f}); utilisation capped at {UTIL_CEILING:.0%}")
    if not pts:
        return 0.0, "no cycle-time data: wait effect not modelled"
    x, y = np.array(pts).T
    b = wait_share * y.mean() / x.mean()
    return float(b), (f"assumption: {wait_share:.0%} of cycle time is queueing that scales with 1 / (1 − utilisation) "
                      "(data too thin to calibrate)")


def _regime(stats, load, cap, util_cap, b, util_b, extra_min):
    kept, flow, unserved, final = _allocate(load, cap, util_cap)
    util_s = {e: (final[e] / cap[e]) if cap[e] > 0 else (np.inf if final[e] > 0 else 0.0) for e in load}
    dT = {e: b * (float(_f(util_s[e])) - float(_f(util_b[e]))) if np.isfinite(util_s[e]) and final[e] > 0 else 0.0
          for e in load}
    on_b = on_s = served_w = time_b = time_s = 0.0
    total_rows = sum(st.rows for st in stats) or 1
    for st in stats:
        e, w = st.name, st.rows / total_rows
        if load[e] <= 0 or not len(st.actual):
            continue
        stay, moved = kept[e] / load[e], {r: v / load[e] for r, v in flow[e].items()}
        served = stay + sum(moved.values())
        on_b += w * float(np.mean(st.actual <= st.promised))
        time_b += w * st.mean_time
        on_s += w * (stay * float(np.mean(st.actual + dT[e] <= st.promised)) +
                     sum(m * float(np.mean(st.actual + dT[r] + extra_min <= st.promised)) for r, m in moved.items()))
        time_s += w * (stay * (st.mean_time + dT[e]) + sum(m * (st.mean_time + dT[r] + extra_min) for r, m in moved.items()))
        served_w += w * served
    return dict(kept=kept, flow=flow, unserved=unserved, final=final, util=util_s, dT=dT,
                on_b=on_b, on_s=on_s / served_w if served_w else np.nan, time_b=time_b,
                time_s=time_s / served_w if served_w else np.nan)


def _on_time_subset(stats, r, extra_min) -> float:
    """On-time share for a subset of rows (e.g. one priority) under a regime's allocation and delays."""
    on = w_tot = 0.0
    for st in stats:
        e = st.name
        if e not in r["kept"] or not len(st.actual):
            continue
        load = r["kept"][e] + sum(r["flow"][e].values()) + r["unserved"][e]
        if load <= 0:
            continue
        stay, moved = r["kept"][e] / load, {x: v / load for x, v in r["flow"][e].items()}
        served = stay + sum(moved.values())
        on += len(st.actual) * (stay * float(np.mean(st.actual + r["dT"][e] <= st.promised)) +
                                sum(m * float(np.mean(st.actual + r["dT"][x] + extra_min <= st.promised))
                                    for x, m in moved.items()))
        w_tot += len(st.actual) * served
    return 100 * on / w_tot if w_tot else np.nan


def cascade(df_base: pd.DataFrame, df_scen: pd.DataFrame, s: Setup, p: Params,
            default_extra_minutes: float = 10.0) -> dict:
    """Run the dependency chain for baseline and scenario. Every scenario number is an estimate."""
    util_cap = p.util_cap if p.util_cap is not None else 0.85
    base, days = entity_stats(df_base, s)
    scen, _ = entity_stats(df_scen, s)
    scen = {e.name: e for e in scen}
    names = [e.name for e in base]
    cap_b = {e.name: e.capacity for e in base}
    obs_load = {e.name: e.rows / days for e in base}
    if s.load_basis == "reported" and all(e.reported_util is not None for e in base):
        load_b = {e.name: e.reported_util * e.capacity for e in base}
        basis = f"reported utilisation × capacity ('{s.reported_util}' × '{s.capacity}')"
    else:
        load_b = obs_load
        basis = f"observed rows per day (rows ÷ {days} days)"
    # exclusions shrink load proportionally to the rows removed per entity
    load_s = {e: load_b[e] * (scen[e].rows / max(base_e.rows, 1) if e in scen else 0.0)
              for e, base_e in zip(names, base)}
    util_b = {e: load_b[e] / cap_b[e] if cap_b[e] else np.nan for e in names}
    b, calib = calibrate(base, util_b)
    avg_cost = np.mean([e.cost for e in base]) if base else 0.0
    extra_cost = p.moved_extra_cost if p.moved_extra_cost is not None else round(0.1 * avg_cost, 2)
    extra_min = p.moved_extra_minutes if p.moved_extra_minutes is not None else default_extra_minutes

    whole = [c for c in p.capacity_changes if c.days is None]
    limited = [c for c in p.capacity_changes if c.days is not None]
    d_aff = min(max([c.days for c in limited], default=0), days)

    def caps(changes):
        c = dict(cap_b)
        for ch in changes:
            if ch.entity_value in c:
                c[ch.entity_value] = max(0.0, c[ch.entity_value] * (1 + (ch.pct or 0) / 100) + (ch.abs or 0))
        return c

    reg_b = _regime(base, load_b, cap_b, util_cap, b, util_b, 0.0)  # baseline: nothing moves
    cap_whole = caps(whole)
    regs = [(days - d_aff, _regime(base, load_s, cap_whole, util_cap, b, util_b, extra_min), cap_whole)]
    if d_aff:
        cap_lim = caps(whole + limited)
        regs.append((d_aff, _regime(base, load_s, cap_lim, util_cap, b, util_b, extra_min), cap_lim))
    regs = [(w, r, c) for w, r, c in regs if w > 0]
    W = sum(w for w, _, _ in regs) or 1

    def avg(fn):
        return sum(w * fn(r, c) for w, r, c in regs) / W

    fee_s = {e: (scen[e].fee if e in scen else 0.0) for e in names}
    cost_s = {e: (scen[e].cost if e in scen else 0.0) for e in names}
    fee_b = {e.name: e.fee for e in base}
    cost_b = {e.name: e.cost for e in base}
    moved_day = avg(lambda r, c: sum(sum(f.values()) for f in r["flow"].values()))
    unserved_day = avg(lambda r, c: sum(r["unserved"].values()))
    served_day = avg(lambda r, c: sum(load_s[e] - r["unserved"][e] for e in names))
    rev_b_day = sum(load_b[e] * fee_b[e] for e in names)
    cost_b_day = sum(load_b[e] * cost_b[e] for e in names)
    rev_s_day = avg(lambda r, c: sum((load_s[e] - r["unserved"][e]) * fee_s[e] for e in names))
    cost_s_day = avg(lambda r, c: sum((load_s[e] - r["unserved"][e]) * cost_s[e] for e in names)) + moved_day * extra_cost
    max_u_b = max(util_b.values()) if util_b else np.nan
    max_u_s = avg(lambda r, c: max(v for v in r["util"].values() if np.isfinite(v)) if r["util"] else 0)
    total_load_b = sum(load_b.values())
    on_s = avg(lambda r, c: r["on_s"]) if regs else np.nan
    time_s = avg(lambda r, c: r["time_s"]) if regs else np.nan
    cap_tot_s = avg(lambda r, c: sum(c.values()))

    metrics = {"on_time_pct": 100 * on_s, "max_utilisation_pct": 100 * max_u_s,
               "margin_per_order": (rev_s_day - cost_s_day) / served_day if served_day else np.nan,
               "unserved_per_day": unserved_day, "avg_cycle_time": time_s}
    margin_b_order = (rev_b_day - cost_b_day) / total_load_b if total_load_b else np.nan
    steps = [
        ("1. Capacity", "capacity per day (all entities)", sum(cap_b.values()), cap_tot_s,
         f"sum of '{s.capacity}', one value per '{s.entity}'; changes applied for the days affected", "reported → scenario"),
        ("2. Allocation", "orders moved per day", 0.0, moved_day,
         f"excess load above capacity moved to the least-utilised entities, none above {util_cap:.0%}", "estimate"),
        ("2. Allocation", "unserved orders per day", 0.0, unserved_day, "excess that fits nowhere under the cap", "estimate"),
        ("3. Utilisation", "highest utilisation %", 100 * max_u_b, 100 * max_u_s,
         f"load ÷ capacity; load = {basis}; row exclusions reduce each entity's load in proportion to rows removed",
         "calculated / estimate"),
        ("4. Cycle time", "average cycle time (min)", reg_b["time_b"], time_s,
         f"{calib}; moved orders +{extra_min:g} min (assumption)", "estimate"),
        ("5. On-time", "on-time %", 100 * reg_b["on_b"], 100 * on_s,
         "share of eligible orders (with a known entity) whose time + modelled delay ≤ promised", "estimate"),
        ("6. Cost", "cost per day", cost_b_day, cost_s_day,
         f"served orders × average cost per entity + moved orders × {extra_cost:g} (assumption)", "estimate"),
        ("6. Margin", "margin per day", rev_b_day - cost_b_day, rev_s_day - cost_s_day, "served orders × (fee − cost)",
         "estimate"),
        ("6. Margin", "margin per order", margin_b_order, metrics["margin_per_order"], "margin per day ÷ served orders per day",
         "estimate"),
    ]
    rows = [{"step": a, "metric": m, "baseline": bv, "scenario": sv,
             "change": (sv - bv) if pd.notna(bv) and pd.notna(sv) else np.nan, "formula": f, "basis": k}
            for a, m, bv, sv, f, k in steps]

    verdict, reasons = "Feasible", []
    if unserved_day > 1e-6:
        verdict = "Not feasible"
        reasons.append(f"{unserved_day:,.1f} orders/day ({unserved_day * W:,.0f} over the period) cannot be served "
                       f"without pushing an entity above the {util_cap:.0%} cap")
    over = [e for w, r, c in regs for e, u in r["util"].items() if np.isfinite(u) and u > util_cap + 1e-9]
    if over and verdict == "Feasible":
        verdict = "Feasible with risk"
        reasons.append(f"{', '.join(sorted(set(over)))} above the {util_cap:.0%} utilisation cap")
    target_rows = []
    for t in p.targets:
        if t.segment_column:
            sub = df_base[df_base[t.segment_column].astype("string") == str(t.segment_value)]
            sub_stats, _ = entity_stats(sub, s) if len(sub) else ([], days)
            v = avg(lambda r, c: _on_time_subset(sub_stats, r, extra_min)) if regs else np.nan
            v_b = _on_time_subset(sub_stats, reg_b, 0.0)
        else:
            v, v_b = metrics[t.metric], None
        ok = bool(pd.notna(v) and (v >= t.value if t.op == ">=" else v <= t.value))
        target_rows.append({"target": t.label(), "baseline": v_b, "scenario estimate": v, "met": ok})
        if not ok:
            verdict = "Not feasible"
            reasons.append(f"target {t.label()} not met (estimate {v:,.2f})")
    if not reasons:
        reasons.append(f"all load served with every entity at or below {util_cap:.0%}")

    last = regs[-1][1] if regs else reg_b
    last_cap = regs[-1][2] if regs else cap_b
    ent = pd.DataFrame([{
        s.entity: e, "capacity (baseline)": cap_b[e], "capacity (scenario)": last_cap[e],
        "load/day (baseline)": round(load_b[e], 1), "orders observed/day": round(obs_load[e], 1),
        "utilisation % calculated (observed ÷ capacity)": round(100 * obs_load[e] / cap_b[e], 1) if cap_b[e] else None,
        "utilisation % reported": round(100 * next(x.reported_util for x in base if x.name == e), 1)
        if s.reported_util and next(x.reported_util for x in base if x.name == e) is not None else None,
        "utilisation % scenario": round(100 * last["util"][e], 1) if np.isfinite(last["util"][e]) else None,
        "moved out/day": round(sum(last["flow"][e].values()), 1),
        "received/day": round(sum(f.get(e, 0) for f in last["flow"].values()), 1),
        "unserved/day": round(last["unserved"][e], 1),
        "est. extra minutes": round(last["dT"][e], 1),
    } for e in names])
    return {"steps": pd.DataFrame(rows), "entities": ent, "verdict": verdict, "reasons": reasons,
            "metrics": metrics, "targets": pd.DataFrame(target_rows), "days": days, "affected_days": d_aff, "basis": basis, "calibration": calib,
            "assumptions": {"utilisation cap": f"{util_cap:.0%}", "extra cost per moved order": extra_cost,
                            "extra minutes per moved order": extra_min,
                            "wait model": f"wait ∝ 1/(1 − utilisation), utilisation capped at {UTIL_CEILING:.0%}"}}


def change_rows(n: int, when: str, text: str, interpretation: str, params: Params,
                before: dict, after: dict) -> list[dict]:
    """One Change Log row per metric: what a constraint moved, before → after."""
    import json

    pj = json.dumps(params.model_dump(exclude_defaults=True), default=str)
    rows = []
    for m in after:
        b, a = before.get(m), after.get(m)
        num = all(isinstance(v, (int, float, np.floating)) and pd.notna(v) for v in (a, b))
        rows.append({"#": n, "time": when, "constraint": text, "interpretation": interpretation, "parameters": pj,
                     "metric": m, "before": b, "after": a,
                     "change": (a - b) if num else ("" if a == b else "changed")})
    return rows
