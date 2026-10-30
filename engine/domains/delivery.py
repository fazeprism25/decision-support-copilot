"""Delivery-operations domain pack. Active only when the data has an actual time, a promised time, a fee, a cost
and an entity (hub/site) with a capacity. Provides the standard labels for C01 value mapping and the delivery
metrics: on-time %, promise gap, margin per order (fee − cost, never order value − cost) and the hub load model.
Column choices come from roles + generic English name hints (rule fallback), never fixed column names."""
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.llm import chat_json
from engine.profile import to_numeric_clean
from engine.scenario import UTIL_CEILING, Setup, calibrate, entity_stats

NAME = "Delivery operations"
VOCAB = {"priority": ["Critical", "Express", "Standard"],
         "status": ["Delivered", "Cancelled", "Returned", "Failed"],
         "peak": ["Peak", "Off-Peak"]}
_SYN = {
    "priority": {"Critical": {"critical", "p1", "urgent", "priority1", "high", "highest", "p0"},
                 "Express": {"express", "p2", "priority2", "medium", "fast", "expedited", "mid"},
                 "Standard": {"standard", "p3", "priority3", "low", "normal", "regular", "economy", "std"}},
    "status": {"Delivered": {"delivered", "completed", "complete", "done", "success", "successful", "fulfilled"},
               "Cancelled": {"cancelled", "canceled", "cancel", "void", "voided"},
               "Returned": {"returned", "rto", "return", "returntoorigin", "rejected", "refused"},
               "Failed": {"failed", "fail", "undelivered", "notdelivered", "lost", "attemptfailed"}},
    "peak": {"Peak": {"peak", "y", "yes", "true", "1", "rush", "rushhour", "peakhour", "busy"},
             "Off-Peak": {"offpeak", "n", "no", "false", "0", "nonpeak", "normal", "quiet"}},
}
# Weak values (yes/no, high/low) only count when the column name also points at the slot.
_WEAK = {"priority": {"high", "highest", "medium", "mid", "low", "normal", "fast"},
         "peak": {"y", "yes", "true", "1", "n", "no", "false", "0", "normal", "busy", "quiet"}}
_SLOT_NAME = {"priority": r"prior|urgen|sla_?class|service_?level|tier", "status": r"status|outcome|state|result",
              "peak": r"peak|rush|busy"}
_COL_HINTS = {"promised": r"promis|sla|target|expected|eta|commit|quoted",
              "actual": r"actual|tat|cycle|taken|elapsed|turnaround|lead|delivery_?time|duration",
              "fee": r"fee|charge|shipping|freight|delivery_?(price|rev)",
              "cost": r"cost|expense|payout|spend",
              "order_value": r"order_?val|basket|gmv|goods|cart|invoice|(^|_)value"}


def norm(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _snake(name: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower().replace(" ", "_")


def standard_label(slot: str, v, name_hint: bool) -> str | None:
    n = norm(v)
    for lab, syn in _SYN[slot].items():
        if n in syn and (name_hint or n not in _WEAK.get(slot, set())):
            return lab
        if n == norm(lab):
            return lab
    return None


def find_slots(df: pd.DataFrame, roles: dict[str, str], min_cover: float = 0.6) -> dict[str, str]:
    """Which category/flag column plays priority / status / peak, by share of rows whose value is a known label."""
    best: dict[str, tuple[float, str]] = {}
    for c, r in roles.items():
        if r not in ("category", "flag") or c not in df:
            continue
        s = df[c].dropna().astype("string")
        if not len(s) or not 2 <= s.nunique() <= 15:
            continue
        counts = s.value_counts()
        for slot in VOCAB:
            hint = re.search(_SLOT_NAME[slot], _snake(c)) is not None
            labs = {v: standard_label(slot, v, hint) for v in counts.index}
            cover = counts[[v for v, lab in labs.items() if lab]].sum() / counts.sum()
            if cover >= min_cover and len({lab for lab in labs.values() if lab}) >= 2:
                if cover + 0.01 * hint > best.get(slot, (0, ""))[0]:
                    best[slot] = (cover + 0.01 * hint, c)
    out, used = {}, set()
    for slot, (_, c) in sorted(best.items(), key=lambda kv: -kv[1][0]):
        if c not in used:
            out[slot] = c
            used.add(c)
    return out


def propose_mapping_rules(df: pd.DataFrame, roles: dict[str, str]) -> tuple[dict, dict]:
    """Rule proposal for C01: {column: {raw: standard label or ''}}, plus {(column, raw): source}."""
    vmap, src = {}, {}
    for slot, c in find_slots(df, roles).items():
        hint = re.search(_SLOT_NAME[slot], _snake(c)) is not None
        vmap[c] = {}
        for v in df[c].dropna().astype("string").unique():
            lab = standard_label(slot, v, hint)
            vmap[c][str(v)] = lab or ""
            src[(c, str(v))] = f"rules ({slot})" if lab else f"rules ({slot}): no match, please set"
    return vmap, src


def propose_mapping_ai(df: pd.DataFrame, roles: dict[str, str], cache_key: str) -> tuple[dict, dict]:
    """AI proposal; every label is checked against the pack vocabulary and every raw value against the data."""
    cols = {c: sorted(df[c].dropna().astype("string").unique().tolist())
            for c, r in roles.items() if r in ("category", "flag") and c in df and 2 <= df[c].nunique() <= 15}
    prompt = ("Map raw category values to standard delivery-operations labels.\n"
              f"Standard vocabularies: {json.dumps(VOCAB)}\nColumns and their values: {json.dumps(cols)}\n"
              "Only map columns that clearly represent order priority, order status or peak-time flag; skip the "
              "rest. Use null for a value you cannot map with confidence.\n"
              'Return ONLY JSON: {"<column>": {"slot": "priority|status|peak", "map": {"<raw value>": "<label or null>"}}}')
    raw = chat_json(prompt, tier="fast", cache_key=f"vmap:{cache_key}",
                    system="You map category values to a fixed vocabulary. Output strict JSON only.")
    vmap, src = {}, {}
    for c, spec in (raw or {}).items() if isinstance(raw, dict) else []:
        if c not in cols or not isinstance(spec, dict) or spec.get("slot") not in VOCAB:
            continue
        allowed, present = set(VOCAB[spec["slot"]]), set(cols[c])
        for v, lab in (spec.get("map") or {}).items():
            if str(v) in present:
                ok = lab in allowed
                vmap.setdefault(c, {})[str(v)] = lab if ok else ""
                src[(c, str(v))] = f"AI ({spec['slot']})" if ok else "AI: no confident label"
    return vmap, src


def merge_proposals(rules: tuple[dict, dict], ai: tuple[dict, dict]) -> tuple[dict, dict]:
    """Rules win where they found a label; AI fills the gaps; AI and rules agreeing is noted."""
    (rv, rs), (av, as_) = rules, ai
    vmap = {c: dict(m) for c, m in rv.items()}
    src = dict(rs)
    for c, m in av.items():
        for v, lab in m.items():
            cur = vmap.setdefault(c, {}).get(v, "")
            if not cur and lab:
                vmap[c][v], src[(c, v)] = lab, as_[(c, v)]
            elif cur and cur == lab:
                src[(c, v)] = src.get((c, v), "rules") + " + AI agree"
            elif cur not in ("", None) and lab and cur != lab:
                src[(c, v)] = src.get((c, v), "rules") + f" (AI said {lab})"
            elif v not in vmap[c]:
                vmap[c][v], src[(c, v)] = "", as_[(c, v)]
    return vmap, src


# ---------- activation ----------

@dataclass
class Pack:
    active: bool
    missing: list[str] = field(default_factory=list)
    cols: dict[str, str | None] = field(default_factory=dict)
    completed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def reason(self) -> str:
        if self.active:
            return "on: " + ", ".join(f"{k} = '{v}'" for k, v in self.cols.items() if v)
        return "off: no column for " + ", ".join(self.missing)


def _pick(cands: list[str], key: str, avoid: set[str]) -> str | None:
    hits = [c for c in cands if c not in avoid and re.search(_COL_HINTS[key], _snake(c))]
    return hits[0] if hits else None


def detect(df: pd.DataFrame, roles: dict[str, str], entity_of: dict[str, str], setup: Setup | None = None) -> Pack:
    setup = setup or Setup()
    durs = [c for c, r in roles.items() if r == "duration" and c not in entity_of]
    money = [c for c, r in roles.items() if r == "money" and c not in entity_of]
    cols: dict[str, str | None] = {}
    cols["promised"] = _pick(durs, "promised", set()) or setup.promised
    cols["actual"] = _pick(durs, "actual", {cols["promised"]}) or (setup.actual if setup.actual != cols["promised"] else None)
    cols["order_value"] = _pick(money, "order_value", set())
    cols["fee"] = _pick(money, "fee", {cols["order_value"]}) or (setup.fee if setup.fee != cols["order_value"] else None)
    cols["cost"] = _pick([c for c in money if not re.search(r"fixed", _snake(c))], "cost",
                         {cols["fee"], cols["order_value"]}) or setup.cost
    cols["entity"], cols["capacity"], cols["reported_util"] = setup.entity, setup.capacity, setup.reported_util
    cols["date"] = setup.date
    slots = find_slots(df, roles)
    cols["status"] = slots.get("status") or setup.status
    cols["priority"], cols["peak"] = slots.get("priority"), slots.get("peak")
    need = ["actual", "promised", "fee", "cost", "entity", "capacity"]
    missing = [k for k in need if not cols.get(k)]
    completed = []
    if cols["status"]:
        hint = re.search(_SLOT_NAME["status"], _snake(cols["status"])) is not None
        completed = sorted(str(v) for v in df[cols["status"]].dropna().astype("string").unique()
                           if standard_label("status", v, hint) == "Delivered")
        if not completed and setup.status == cols["status"]:
            completed = list(setup.completed_values)
    pk = Pack(not missing, missing, cols, completed)
    if cols["order_value"]:
        pk.notes.append(f"'{cols['order_value']}' looks like the value of the goods, so it is NOT used as revenue: "
                        f"margin uses the fee '{cols['fee']}' minus the cost '{cols['cost']}' of the same delivery.")
    return pk


def fill_setup(s: Setup, pk: Pack) -> Setup:
    """Fill cascade columns the KPI plan did not provide (and replace an order-value 'fee')."""
    if not pk.active:
        return s
    c = pk.cols
    s.actual, s.promised = s.actual or c["actual"], s.promised or c["promised"]
    if not s.fee or s.fee == c["order_value"]:
        s.fee = c["fee"]
    s.cost = s.cost or c["cost"]
    if c["status"] and (not s.status or not s.completed_values):
        s.status, s.completed_values = c["status"], list(pk.completed)
    return s


# ---------- metrics ----------

@dataclass
class Figure:
    label: str
    value: float | None
    fmt: str  # pct | min | money | number
    basis: str  # calculated | reported | estimate (model)
    formula: str
    filter: str = "none"
    n: int = 0


def _fmt(v, fmt):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return {"pct": f"{v:.1f}%", "min": f"{v:,.1f} min", "money": f"{v:,.2f}", "number": f"{v:,.1f}"}[fmt]


def fmt(f: Figure) -> str:
    return _fmt(f.value, f.fmt)


def _eligible(df, pk):
    c = pk.cols
    a, p = to_numeric_clean(df[c["actual"]]), to_numeric_clean(df[c["promised"]])
    mask = a.notna() & p.notna()
    filt = "rows with both times"
    if c["status"] and pk.completed:
        mask &= df[c["status"]].astype("string").isin(pk.completed)
        filt = f"'{c['status']}' in {pk.completed} and both times present"
    return a, p, mask, filt


def setup_of(pk: Pack) -> Setup:
    c = pk.cols
    return Setup(entity=c["entity"], capacity=c["capacity"], reported_util=c["reported_util"],
                 load_basis="reported" if c["reported_util"] else "observed", date=c["date"], actual=c["actual"],
                 promised=c["promised"], status=c["status"], completed_values=list(pk.completed), fee=c["fee"],
                 cost=c["cost"])


def metrics(df: pd.DataFrame, pk: Pack, entity_of: dict[str, str] | None = None,
            roles: dict[str, str] | None = None) -> dict:
    """All pack numbers, computed by pandas. Every figure carries its formula, filter and basis."""
    c = pk.cols
    a, p, ok, filt = _eligible(df, pk)
    late = a - p
    heads: list[Figure] = []
    n = int(ok.sum())
    heads.append(Figure("On-time % (all eligible orders)", 100 * float((a[ok] <= p[ok]).mean()) if n else None, "pct",
                        "calculated", f"orders with '{c['actual']}' ≤ '{c['promised']}' ÷ eligible orders", filt, n))
    heads.append(Figure("Average promise gap", float(late[ok].mean()) if n else None, "min", "calculated",
                        f"mean of ('{c['actual']}' − '{c['promised']}'); positive = later than promised", filt, n))
    heads.append(Figure("Median promise gap", float(late[ok].median()) if n else None, "min", "calculated",
                        f"median of ('{c['actual']}' − '{c['promised']}')", filt, n))
    heads.append(Figure("Late orders: 90th-percentile gap", float(late[ok].quantile(0.9)) if n else None, "min",
                        "calculated", "90% of eligible orders are at most this many minutes late", filt, n))
    fee, cost = to_numeric_clean(df[c["fee"]]), to_numeric_clean(df[c["cost"]])
    both = fee.notna() & cost.notna()
    m = fee - cost
    heads.append(Figure("Margin per order", float(m[both].mean()) if both.any() else None, "money", "calculated",
                        f"mean of ('{c['fee']}' − '{c['cost']}') per order", "orders with both values",
                        int(both.sum())))
    dl = both & ok
    heads.append(Figure("Margin per delivered order", float(m[dl].mean()) if dl.any() else None, "money", "calculated",
                        f"mean of ('{c['fee']}' − '{c['cost']}')", filt + " and both money values", int(dl.sum())))
    heads.append(Figure("Share of orders losing money", 100 * float((m[both] < 0).mean()) if both.any() else None,
                        "pct", "calculated", f"orders with '{c['fee']}' < '{c['cost']}' ÷ orders with both values",
                        "orders with both values", int(both.sum())))
    heads.append(Figure("Average fee", float(fee[both].mean()) if both.any() else None, "money", "calculated",
                        f"mean of '{c['fee']}'", "orders with both values", int(both.sum())))

    def by(col):
        if not col or col not in df:
            return pd.DataFrame()
        g = pd.DataFrame({"segment": df[col].astype("string"), "ok": ok, "on": ok & (a <= p), "gap": late.where(ok),
                          "m": m.where(both), "loss": (m < 0).where(both)})
        t = g.groupby("segment", dropna=True).agg(orders=("segment", "size"), eligible=("ok", "sum"),
                                                  on_time=("on", "sum"), gap=("gap", "mean"), margin=("m", "mean"),
                                                  loss=("loss", "mean")).reset_index()
        t["on-time %"] = (100 * t["on_time"] / t["eligible"]).where(t["eligible"] > 0)
        t["loss"] = 100 * t["loss"]
        return t.rename(columns={"segment": col, "eligible": "eligible orders", "on_time": "on time",
                                 "gap": "avg promise gap (min)", "margin": "margin per order",
                                 "loss": "% orders losing money"})

    seg = {k: by(c[k]) for k in ("priority", "peak", "entity") if c.get(k)}
    hub = hub_load(df, pk, entity_of or {}, roles or {})
    return {"heads": heads, "segments": seg, "hub": hub, "eligible_filter": filt, "notes": list(pk.notes)}


def hub_load(df: pd.DataFrame, pk: Pack, entity_of: dict[str, str], roles: dict[str, str]) -> dict:
    """Hub load model: wait ∝ 1/(1 − utilisation), utilisation capped at 95%, slope calibrated across hubs.
    Reported and calculated utilisation are shown side by side; they can disagree when the file is a sample."""
    s = setup_of(pk)
    if not s.date:
        return {"table": pd.DataFrame(), "calibration": "no date column: orders per day cannot be computed",
                "days": None}
    stats, days = entity_stats(df, s)
    obs = {e.name: e.rows / days for e in stats}
    calc_u = {e.name: obs[e.name] / e.capacity if e.capacity else np.nan for e in stats}
    rep_u = {e.name: e.reported_util for e in stats}
    use_rep = all(v is not None for v in rep_u.values()) and bool(s.reported_util)
    u = rep_u if use_rep else calc_u
    b, calib = calibrate(stats, u)
    rep_dur = [a for a, e in entity_of.items() if e == s.entity and roles.get(a) == "duration"]
    rows = []
    for e in stats:
        ue = u[e.name]
        row = {s.entity: e.name, "capacity/day [reported]": e.capacity, "orders/day in file [calculated]": round(obs[e.name], 1),
               "utilisation % [calculated: orders ÷ capacity]": round(100 * calc_u[e.name], 1) if np.isfinite(calc_u[e.name]) else None,
               "utilisation % [reported]": round(100 * rep_u[e.name], 1) if rep_u[e.name] is not None else None,
               "wait multiplier 1/(1−u) [model]": round(float(1 / (1 - min(ue, UTIL_CEILING))), 2)
               if ue is not None and np.isfinite(ue) else None,
               "avg cycle time min [calculated]": round(e.mean_time, 1) if e.mean_time is not None else None}
        for a in rep_dur:
            v = to_numeric_clean(df.loc[df[s.entity].astype("string") == e.name, a]).dropna()
            row[f"{a} [reported]"] = round(float(v.iloc[0]), 1) if len(v) else None
        rows.append(row)
    return {"table": pd.DataFrame(rows), "calibration": calib, "days": days,
            "util_basis": f"reported ('{s.reported_util}')" if use_rep else "calculated (orders in file ÷ capacity)",
            "mismatch": _mismatch(calc_u, rep_u)}


def _mismatch(calc_u, rep_u) -> str:
    pairs = [(calc_u[e], rep_u[e]) for e in calc_u if rep_u.get(e) is not None and np.isfinite(calc_u[e])]
    if not pairs:
        return ""
    c, r = np.mean([x for x, _ in pairs]), np.mean([y for _, y in pairs])
    if r > 0 and abs(c - r) / r > 0.5:
        return (f"Reported utilisation averages {100 * r:.1f}% but orders in the file ÷ capacity give {100 * c:.1f}%. "
                "The file is probably a sample of real volume; the load model uses the reported figure.")
    return ""


def formulas(pk: Pack) -> list[tuple[str, str, str]]:
    c = pk.cols
    return [
        ("On-time %", f"orders with '{c['actual']}' ≤ '{c['promised']}' ÷ eligible orders",
         "eligible = completed status (" + ", ".join(pk.completed or ["any"]) + ") with both times"),
        ("Promise gap (min)", f"'{c['actual']}' − '{c['promised']}' per eligible order; mean, median, 90th percentile",
         "positive = later than promised"),
        ("Margin per order", f"mean of ('{c['fee']}' − '{c['cost']}')",
         "fee and cost of the same delivery; order value is never used as revenue"),
        ("Hub utilisation [calculated]", f"orders in file per day ÷ '{c['capacity']}'", "per entity"),
        ("Hub utilisation [reported]", f"'{c['reported_util']}'" if c["reported_util"] else "not in data", "per entity"),
        ("Wait model", f"cycle time ≈ a + b / (1 − utilisation), utilisation capped at {UTIL_CEILING:.0%}",
         "b calibrated across entities when possible; otherwise an assumption (labelled)"),
    ]
