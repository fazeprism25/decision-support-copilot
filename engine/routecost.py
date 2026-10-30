"""Route cost model (cost-analytics card): base cost vs full cost per route.
A route is any user-chosen grouping (a route column, or e.g. entity × zone). Every hidden cost is its own component
with a formula, a source (data / assumption / proxy) and a status (confirmed / provisional / unverified).
Columns come from roles + generic English name hints; nothing here names a dataset's columns."""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.profile import to_numeric_clean
from engine.scenario import CostComponent, Params, Setup

STATUSES = ("confirmed", "provisional", "unverified")
VIEWS = {"base": "Base cost only", "confirmed": "Full cost (confirmed only)", "all": "Full cost (all statuses)"}
SEP = " × "
_ROUTE_HINT = r"route|lane|corridor|beat|trip"
_HIDDEN_HINT = (r"toll|park|wait|detention|idle|fuel|loading|surcharge|congestion|penalt|re_?attempt|return|"
                r"handling|dwell")
_COSTY = r"cost|charge|fee|inr|rs|usd|amount|amt|toll|park"
_NOT_BASE = r"fee|price|revenue|value|basket|gmv|fixed"
_FAIL = r"fail|undeliver|notdeliver|attempt|lost|missed"
_RETURN = r"return|rto|refus|reject"
_SUCCESS = r"^(deliver|success|complet|done)"
_WAIT_HINT = r"wait|idle|dwell|detention|queue|halt"
_DIST_HINT = r"(^|_)kms?($|_)|dist|mile|kilomet"
_STATUS_COL_HINT = r"status|outcome|attempt|result|state"
_DROP_WORDS = {"cost", "costs", "charge", "charges", "fee", "fees", "inr", "rs", "usd", "amount", "amt", "per", "order",
               "rupees", "total", "est", "estimated", "the", "of"}


def _s(c) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", str(c)).lower()


def _n(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def comp_key(name: str) -> str:
    """Canonical component name: 'Toll_INR' / 'tolls' → 'toll', 'wait_cost' / 'waiting' → 'waiting'."""
    out = []
    for t in re.findall(r"[a-z]+", _s(name)):
        if t in _DROP_WORDS:
            continue
        if t.startswith("wait"):
            t = "waiting"
        elif t.startswith("park"):
            t = "parking"
        elif t.startswith("toll"):
            t = "toll"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return " ".join(out) or str(name).strip().lower()


def status_of(v) -> str | None:
    """Read a verification label: confirmed / provisional / unverified, or None if it is not one."""
    n = _n(v)
    if not n:
        return None
    if re.search(r"unverif|unconfirm|notverif|notconfirm|unknown|assum|unvalid", n):
        return "unverified"
    if re.search(r"provision|estimat|tentativ|pending|draft|interim|partial", n):
        return "provisional"
    if re.search(r"confirm|verif|actual|final|audit|valid", n):
        return "confirmed"
    return None


def _one_to_one(df: pd.DataFrame, a: str, b: str | None) -> bool:
    if b is None or a == b:
        return a == b
    pair = df[[a, b]].dropna().drop_duplicates()
    return len(pair) == pair[a].nunique() == pair[b].nunique()


@dataclass
class RouteSetup:
    route_cols: list[str] = field(default_factory=list)
    base_cost: str | None = None
    status: str | None = None
    failed_values: list[str] = field(default_factory=list)
    returned_values: list[str] = field(default_factory=list)
    reattempt_cost: float | None = None  # None = the order's own base cost
    return_cost: float | None = None  # None = the order's own base cost (the journey again)
    wait_col: str | None = None  # waiting minutes per order
    actual: str | None = None  # proxy when no wait column: (actual − expected) minutes
    promised: str | None = None
    wait_rate: float = 0.0  # money per waiting minute; 0 = waiting not costed
    wait_status: str = "unverified"
    distance: str | None = None
    cost_cols: list[str] = field(default_factory=list)  # hidden-cost money columns in the data
    cost_status_col: str | None = None  # row-level confirmed/provisional/unverified label
    status_override: dict[str, str] = field(default_factory=dict)  # component key → status the user set
    entries: list[CostComponent] = field(default_factory=list)  # entered on the Route cost tab
    min_orders: int = 15

    def missing(self) -> list[str]:
        return [k for k, v in {"route grouping": self.route_cols, "base cost column": self.base_cost}.items() if not v]


def guess(df: pd.DataFrame, roles: dict[str, str], entity_of: dict[str, str], setup: Setup) -> RouteSetup:
    rs = RouteSetup()
    n = len(df)
    labels = [c for c, r in roles.items() if r in ("category", "location", "id") and c in df and c not in entity_of]
    route = [c for c in labels if re.search(_ROUTE_HINT, _s(c)) and 2 <= df[c].nunique() <= max(2, n // 3)]
    if route:
        rs.route_cols = route[:1]
    else:
        ent = setup.entity if setup.entity in df else None
        locs = [c for c in labels if roles[c] == "location" and c != ent and 2 <= df[c].nunique() <= 200
                and not _one_to_one(df, c, ent)]
        rs.route_cols = [x for x in (ent, locs[0] if locs else None) if x]

    nums = [c for c, r in roles.items() if r in ("money", "measure", "quantity", "duration") and c not in entity_of
            and c in df]
    for c in nums:
        name = _s(c)
        if re.search(_WAIT_HINT, name) and not re.search(r"cost|charge|inr|rs|amount", name):
            if roles[c] in ("duration", "measure", "quantity") and rs.wait_col is None:
                rs.wait_col = c
        elif re.search(_HIDDEN_HINT, name) and (roles[c] == "money" or re.search(_COSTY, name)):
            rs.cost_cols.append(c)
        elif roles[c] in ("measure", "quantity") and re.search(_DIST_HINT, name) and rs.distance is None:
            rs.distance = c
    money = [c for c in nums if roles[c] == "money" and c not in rs.cost_cols and c != setup.fee
             and not re.search(_NOT_BASE, _s(c))]
    if setup.cost in money:
        rs.base_cost = setup.cost
    else:
        rs.base_cost = next((c for c in money if re.search(r"base|variable", _s(c))), None) or \
            next((c for c in money if "cost" in _s(c)), None)

    cand = ([setup.status] if setup.status in df else []) + \
        [c for c in labels + [c for c, r in roles.items() if r == "flag"] if re.search(_STATUS_COL_HINT, _s(c))]
    for c in dict.fromkeys(cand):
        vals = df[c].dropna().astype("string").unique().tolist()
        fail = [v for v in vals if re.search(_FAIL, _n(v)) and not re.search(_SUCCESS, _n(v))]
        ret = [v for v in vals if re.search(_RETURN, _n(v)) and not re.search(_SUCCESS, _n(v))]
        if fail or ret:
            rs.status, rs.failed_values, rs.returned_values = c, fail, ret
            break
    rs.actual, rs.promised = setup.actual, setup.promised

    for c, r in roles.items():
        if r not in ("category", "flag") or c not in df or c == rs.status:
            continue
        v = df[c].dropna().astype("string")
        if len(v) and v.nunique() <= 6 and v.map(status_of).notna().mean() >= 0.9:
            rs.cost_status_col = c
            break
    return rs


# ---------- computation ----------

@dataclass
class _Comp:
    label: str
    amount: pd.Series
    status: pd.Series
    source: str
    formula: str
    entry: CostComponent | None = None  # set when one typed entry defines the whole component


def flip_k(b: pd.Series, c: pd.Series) -> tuple[float | None, str | None]:
    """Smallest k ≥ 0 at which the cheapest route under b + k·c is no longer the cheapest under b."""
    w = b.idxmin()
    best, who = None, None
    for r in b.index:
        if r == w or c[r] >= c[w]:
            continue
        k = (b[r] - b[w]) / (c[w] - c[r])
        if k >= 0 and (best is None or k < best):
            best, who = k, r
    return best, who


def compute(df: pd.DataFrame, rs: RouteSetup, params: Params | None = None) -> dict | None:
    if rs.missing() or any(c not in df for c in rs.route_cols + [rs.base_cost]):
        return None
    params = params or Params()
    notes = []
    key = df[rs.route_cols].astype("string")
    base = to_numeric_clean(df[rs.base_cost])
    ok = key.notna().all(axis=1) & base.notna()
    if (~ok).sum():
        notes.append(f"{int((~ok).sum()):,} rows without a route value or a base cost are left out.")
    d, base = df[ok], base[ok]
    if not len(d):
        return None
    route = key[ok].agg(SEP.join, axis=1)
    conf = pd.Series("confirmed", index=d.index)
    comps: dict[str, _Comp] = {"base": _Comp("base cost", base, conf, f"data: '{rs.base_cost}'",
                                             f"average '{rs.base_cost}' per order")}

    def event(label, values, unit_cost, what):
        hit = d[rs.status].astype("string").isin(values).fillna(False).astype(bool)
        per = base if unit_cost is None else pd.Series(float(unit_cost), index=d.index)
        cost_txt = "the order's own base cost (assumption, editable)" if unit_cost is None else f"{unit_cost:,.2f}"
        comps[comp_key(label)] = _Comp(
            label, per.where(hit, 0.0), conf.copy(), f"data: '{rs.status}' statuses",
            f"{what} cost × {label.split()[0]} share; {label.split()[0]} = rows where '{rs.status}' is "
            f"{' or '.join(values)}; {what} cost per event = {cost_txt}")

    if rs.status and rs.status in d:
        if rs.failed_values:
            event("failed attempts", rs.failed_values, rs.reattempt_cost, "re-attempt")
        if rs.returned_values:
            event("return journeys", rs.returned_values, rs.return_cost, "return")

    minutes, m_src = None, ""
    if rs.wait_col and rs.wait_col in d:
        minutes, m_src = to_numeric_clean(d[rs.wait_col]).fillna(0).clip(lower=0), f"data: '{rs.wait_col}' minutes"
    elif rs.actual in d and rs.promised in d:
        minutes = (to_numeric_clean(d[rs.actual]) - to_numeric_clean(d[rs.promised])).clip(lower=0).fillna(0)
        m_src = f"proxy: max(0, '{rs.actual}' − '{rs.promised}') minutes"
    if minutes is not None:
        comps["waiting"] = _Comp(
            "waiting", minutes * rs.wait_rate, pd.Series(rs.wait_status, index=d.index),
            m_src + ("" if rs.wait_col else " (PROXY, not measured waiting)"),
            f"waiting minutes × {rs.wait_rate:,.2f} per minute" + (" (rate not set: waiting is not costed)"
                                                                   if not rs.wait_rate else " (rate is your entry)"))

    row_status = None
    if rs.cost_status_col and rs.cost_status_col in d:
        row_status = d[rs.cost_status_col].map(status_of).fillna("unverified")
    for c in rs.cost_cols:
        if c not in d:
            continue
        k = comp_key(c)
        st = row_status.copy() if row_status is not None else conf.copy()
        comps[k] = _Comp(k, to_numeric_clean(d[c]).fillna(0), st, f"data: '{c}'",
                         f"average '{c}' per order; status " +
                         (f"per row from '{rs.cost_status_col}'" if row_status is not None else "confirmed (from data)"))

    for cc in list(rs.entries) + list(params.cost_components):
        if cc.per == "order":
            basis = pd.Series(1.0, index=d.index)
        elif cc.per == "km":
            if not rs.distance or rs.distance not in d:
                notes.append(f"'{cc.name}' is per km but no distance column is set: left out.")
                continue
            basis = to_numeric_clean(d[rs.distance]).fillna(0)
        else:
            if minutes is None:
                notes.append(f"'{cc.name}' is per minute but there are no waiting minutes: left out.")
                continue
            basis = minutes
        mask = pd.Series(True, index=d.index)
        seg = ""
        if cc.segment_column and cc.segment_column in d:
            mask = d[cc.segment_column].astype("string").eq(str(cc.segment_value)).fillna(False).astype(bool)
            seg = f" on rows where '{cc.segment_column}' = {cc.segment_value}"
        new = basis * cc.value
        k = comp_key(cc.name)
        txt = f"{cc.value:,.2f} per {cc.per}{seg} (your entry, {cc.status})"
        if k in comps:
            old = comps[k]
            old.amount = new.where(mask, old.amount)
            old.status = pd.Series(cc.status, index=d.index).where(mask, old.status)
            old.source += " + your entry"
            old.formula += f"; replaced by {txt}"
            old.entry = cc if mask.all() else None
        else:
            comps[k] = _Comp(k, new.where(mask, 0.0), pd.Series(cc.status, index=d.index), "assumption (your entry)",
                             txt, cc)
    for k, s in rs.status_override.items():
        if k in comps and k != "base" and s in STATUSES:
            comps[k].status = pd.Series(s, index=d.index)
            comps[k].source += " · status set by you"

    long = pd.concat([pd.DataFrame({"route": route.values, "key": k, "component": c.label, "status": c.status.values,
                                    "amount": c.amount.values}) for k, c in comps.items()], ignore_index=True)
    orders = route.value_counts()
    excluded = [s for s in dict.fromkeys(params.exclude_cost_status) if s != "confirmed"]
    views = dict(VIEWS)
    if excluded:
        views["decision"] = f"Decision view (excluding {' & '.join(excluded)})"
    masks = {"base": long["key"].eq("base"), "confirmed": long["status"].eq("confirmed"),
             "all": pd.Series(True, index=long.index), "decision": ~long["status"].isin(excluded)}
    cost = {v: long[masks[v]].groupby("route")["amount"].sum().reindex(orders.index, fill_value=0) / orders
            for v in views}
    eligible = orders >= rs.min_orders
    if eligible.sum() < 2:
        notes.append(f"Fewer than 2 routes have at least {rs.min_orders} orders, so all routes are ranked "
                     "(thin data: treat the ranking with caution).")
        eligible = orders > 0
    else:
        thin = int((~eligible).sum())
        if thin:
            notes.append(f"{thin} route(s) with fewer than {rs.min_orders} orders are shown but not ranked (thin data).")
    el = eligible[eligible].index

    tab = pd.DataFrame({"route": orders.index, "orders": orders.values})
    for v in views:
        tab[f"{v} per order"] = cost[v].values
        tab[f"{v} rank"] = cost[v][el].rank(method="min").reindex(orders.index).values
    tab["hidden % of base"] = ((cost["all"] - cost["base"]) / cost["base"] * 100).values
    tab["confirmed hidden % of base"] = ((cost["confirmed"] - cost["base"]) / cost["base"] * 100).values
    rank_cols = [f"{v} rank" for v in views]
    tab["rank changes"] = tab[rank_cols].nunique(axis=1).gt(1) & tab[rank_cols].notna().all(axis=1)
    tab = tab.sort_values([f"{'decision' if excluded else 'all'} rank", "orders"], ascending=[True, False],
                          na_position="last").reset_index(drop=True)
    win = {v: cost[v][el].idxmin() for v in views}

    # per-route component table (per order, split by status)
    by = long.groupby(["route", "component", "status"], sort=False)["amount"].sum().reset_index()
    by["per order"] = by["amount"] / by["route"].map(orders)

    def contrib(kind: str, k: str) -> pd.Series:
        m = long["key"].eq(k) & (long["status"].ne("confirmed") if kind == "unconfirmed" else long["status"].eq("confirmed"))
        return long[m].groupby("route")["amount"].sum().reindex(el, fill_value=0) / orders[el]

    def explain_flip(a: str, kind: str):
        """Which single component, added on top of view a, flips the cheapest route, and its break-even size."""
        b = cost[a][el]
        best = None
        for k, c in comps.items():
            if k == "base":
                continue
            cvec = contrib(kind, k)
            if not cvec.abs().sum():
                continue
            kk, who = flip_k(b, cvec)
            if kk is not None and (best is None or kk < best[0]):
                best = (kk, who, k, cvec)
        return best

    def break_even(k: str, kk: float, w: str, cvec: pd.Series) -> str:
        c = comps[k]
        if c.entry is not None:
            return f"{c.label} of {kk * c.entry.value:,.2f} per {c.entry.per} (entered: {c.entry.value:,.2f})"
        pct = f"{kk:.0%}" if kk >= 0.1 else f"{kk:.1%}"
        return (f"{c.label} at {pct} of its estimated size (≈ {kk * cvec[w]:,.2f} per order on route {w}, "
                f"estimated {cvec[w]:,.2f})")

    flips = []
    for a, b_, kind in (("base", "confirmed", "confirmed"), ("confirmed", "all", "unconfirmed")):
        if win[a] == win[b_]:
            continue
        f = explain_flip(a, kind)
        if f and f[0] <= 1 + 1e-9:
            kk, who, k, cvec = f
            flips.append({"from view": VIEWS[a], "to view": VIEWS[b_], "component": comps[k].label,
                          "status": "confirmed" if kind == "confirmed" else "provisional/unverified",
                          "cheapest before": win[a], "cheapest after": win[b_],
                          "break-even": break_even(k, kk, win[a], cvec)})
        else:
            flips.append({"from view": VIEWS[a], "to view": VIEWS[b_], "component": "several components together",
                          "status": "confirmed" if kind == "confirmed" else "provisional/unverified",
                          "cheapest before": win[a], "cheapest after": win[b_],
                          "break-even": "no single component flips it alone"})
    if not flips:
        verdict = "Robust"
        text = f"Robust: route {win['all']} is cheapest under all three views."
        near = None
        for k, c in comps.items():
            if k == "base":
                continue
            cvec = long[long["key"].eq(k)].groupby("route")["amount"].sum().reindex(el, fill_value=0) / orders[el]
            if not cvec.abs().sum():
                continue
            kk, who = flip_k(cost["all"][el], cvec)
            if kk is not None and (near is None or kk < near[0]):
                near = (kk, who, k, cvec)
        if near:
            kk, who, k, cvec = near
            text += (f" Nearest flip: route {who} becomes cheapest only if {comps[k].label} is "
                     f"{1 + kk:.1f}× its current size.")
    else:
        verdict = "Fragile"
        f = flips[-1]
        text = (f"Fragile: the choice flips from route {f['cheapest before']} to {f['cheapest after']} when "
                f"{f['status']} {f['component']} is included (break-even: {f['break-even']}).")
        if len(flips) > 1:
            text += (f" It also flips between base and confirmed cost ({flips[0]['component']}: "
                     f"{flips[0]['cheapest before']} → {flips[0]['cheapest after']}).")

    total = {v: float(long[masks[v]]["amount"].sum()) for v in views}
    totals = {"orders": int(len(d)), "base": total["base"], "confirmed": total["confirmed"], "all": total["all"],
              "hidden_confirmed": total["confirmed"] - total["base"], "hidden_all": total["all"] - total["base"],
              "hidden_pct": (total["all"] - total["base"]) / total["base"] * 100 if total["base"] else None}
    comp_rows = []
    for k, c in comps.items():
        amt = float(c.amount.sum())
        share = c.status[c.amount.ne(0)].value_counts(normalize=True) if c.amount.ne(0).any() else c.status.value_counts(normalize=True)
        st_txt = share.index[0] if len(share) == 1 else ", ".join(f"{s} {p:.0%}" for s, p in share.items())
        comp_rows.append({"component": c.label, "status": st_txt, "source": c.source, "formula": c.formula,
                          "total": amt, "per order": amt / len(d)})
    return {"routes": tab, "views": views, "components": pd.DataFrame(comp_rows), "by_route": by, "winners": win,
            "decision": win.get("decision", win["all"]), "verdict": verdict, "verdict_text": text, "flips": flips,
            "totals": totals, "notes": notes, "excluded": excluded, "route_cols": list(rs.route_cols),
            "min_orders": rs.min_orders}


def snapshot(r: dict | None) -> dict:
    """Change Log metrics for the route cost card."""
    if not r:
        return {}
    out = {f"Route · cheapest ({VIEWS[v]})": r["winners"][v] for v in VIEWS}
    out["Route · cheapest (decision view)"] = r["decision"]
    out["Route · verdict"] = r["verdict"]
    out["Route · hidden cost total"] = r["totals"]["hidden_all"]
    out["Route · hidden cost % of base"] = r["totals"]["hidden_pct"]
    return out
