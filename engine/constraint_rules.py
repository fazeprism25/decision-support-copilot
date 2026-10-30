"""Rule-based constraint parser: common card wordings → Params with no AI (instant, works when the AI is down).
It matches words in the sentence to columns and values found in the data (via roles and the What-if setup),
never to fixed column names. Anything it cannot fully place returns None so the AI parser (or a refusal) takes over."""
import re
from dataclasses import dataclass

import pandas as pd

from engine.scenario import CapacityChange, CostComponent, Exclusion, MeasureChange, Params, PriceChange, Setup, Target

_NUM = r"(\d+(?:\.\d+)?)"
_DOWN = r"cut|reduced?|drops?|falls?|lowered|decreased?|down|shrinks?|lose|loses"
_UP = r"rises?|increases?|goes up|grows?|up|higher|raised?|jumps?"
_ENTITY_WORDS = r"hub|hubs|dc|dcs|site|sites|centre|center|warehouse|warehouses|store|stores|depot|depots|branch|branches"
_ROW_WORDS = r"orders?|rows?|records?|shipments?|deliveries|transactions?|sales|items?"


@dataclass
class RuleResult:
    ok: bool
    params: Params | None = None
    interpretation: str = ""
    reason: str = ""
    suggestion: str = ""


def _n(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _stem(w: str) -> str:
    w = w.lower()
    for suf in ("ies", "es", "s", "ed", "ing"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def entity_aliases(df: pd.DataFrame, entity: str, roles: dict[str, str]) -> dict[str, str]:
    """Every name that points at exactly one entity value: the value itself plus values of other label columns
    that map one-to-one to it (e.g. a hub's region or name), compared case-insensitively."""
    out = {_n(v): str(v) for v in df[entity].dropna().astype("string").unique()}
    for c, r in roles.items():
        if c == entity or r not in ("category", "location", "id") or c not in df:
            continue
        sub = df[[entity, c]].dropna().astype("string")
        sub = sub.assign(_k=sub[c].map(_n))
        if not len(sub) or sub[entity].nunique() < 2:
            continue
        fwd = sub.groupby(entity)["_k"].nunique()
        back = sub.groupby("_k")[entity].nunique()
        if (fwd == 1).all() and (back == 1).all():
            for k, e in sub.groupby("_k")[entity].first().items():
                out.setdefault(k, str(e))
    return out


def _resolve_entity(text: str, aliases: dict[str, str]) -> str | None:
    t = _n(re.sub(rf"\b(the|{_ENTITY_WORDS})\b", " ", text, flags=re.I))
    if not t:
        return None
    if t in aliases:
        return aliases[t]
    hits = {e for k, e in aliases.items() if t in k or (len(k) >= 3 and k in t)}
    return hits.pop() if len(hits) == 1 else None


def _find_values(words: list[str], df: pd.DataFrame, roles: dict[str, str], prefer: list[str]) -> tuple[str, list[str]] | None:
    """Find one label column containing all the given words as values (case/plural-insensitive)."""
    cols = [c for c in prefer if c] + [c for c, r in roles.items() if r in ("category", "flag", "location")]
    for c in dict.fromkeys(cols):
        if c not in df:
            continue
        vals = df[c].dropna().astype("string").unique()
        lookup = {}
        for v in vals:
            lookup.setdefault(_n(v), str(v))
            lookup.setdefault(_stem(_n(v)), str(v))
        found = []
        for w in words:
            k = _n(w)
            v = lookup.get(k) or lookup.get(_stem(k))
            if v is None:
                break
            found.append(v)
        else:
            if found:
                return c, found
    return None


def _measure_column(phrase: str, roles: dict[str, str], setup: Setup, entity_of: dict[str, str]) -> str | None:
    """Numeric column named by a phrase like 'rider costs' or 'delivery fee'."""
    nums = [c for c, r in roles.items() if r in ("money", "duration", "quantity", "measure") and c not in entity_of]
    words = [_stem(w) for w in re.findall(r"[a-z]+", phrase.lower()) if len(w) > 2]
    special = {"cost": setup.cost, "costs": setup.cost, "fee": setup.fee, "charge": setup.fee}
    scored = []
    for c in nums:
        toks = {_stem(t) for t in re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", c).lower())}
        score = sum(w in toks for w in words)
        if score:
            scored.append((score, c == special.get(words[-1] if words else ""), c))
    if scored:
        scored.sort(reverse=True)
        top = [x for x in scored if x[:2] == scored[0][:2]]
        if len(top) == 1:
            return top[0][2]
    for w in reversed(words):  # e.g. "rider costs" → the per-order cost column used by the cascade
        if special.get(w):
            return special[w]
    return None


_STATUS_WORDS = r"unverified|provisional|confirmed"
_CUR = r"(?:rs\.?|inr|₹|\$|usd|eur|€|£|aed)?"
_PER = r"(?:per|/|an?|each)\s*(?P<unit>orders?|deliver(?:y|ies)|shipments?|trips?|kms?|kilomet\w*|minutes?|mins?)\b"


def _cost_status(low: str) -> str:
    m = re.search(r"\b(unverified|not verified|unconfirmed|provisional|tentative|estimated|confirmed|verified)\b", low)
    word = m.group(1) if m else "unverified"
    return {"not verified": "unverified", "unconfirmed": "unverified", "tentative": "provisional",
            "estimated": "provisional", "verified": "confirmed"}.get(word, word)


def _route_cost_rules(low: str, df: pd.DataFrame, roles: dict[str, str], setup: Setup) -> RuleResult | None:
    """Route cost card: 'exclude unverified costs', '<name> of Rs 40 per order on <label> routes, unverified',
    'include waiting cost at Rs 2 per minute', 'parking is confirmed at Rs 15 per order'."""
    from engine.routecost import _DIST_HINT, _ROUTE_HINT, comp_key

    m = re.match(rf"^(?:exclude|ignore|drop|remove|leave out|without)\s+(?:all\s+|the\s+|any\s+)?(?P<st>(?:{_STATUS_WORDS})"
                 rf"(?:\s*(?:,|and|or|&|/)\s*(?:{_STATUS_WORDS}))*)\s+(?:costs?|components?|estimates?|items?|charges?)$",
                 low)
    only = re.match(rf"^(?:use\s+|include\s+|count\s+)?only\s+confirmed\s+(?:costs?|components?)$", low)
    if m or only:
        sts = ["provisional", "unverified"] if only else list(dict.fromkeys(re.findall(_STATUS_WORDS, m.group("st"))))
        if "confirmed" in sts:
            return RuleResult(False, reason="Confirmed costs cannot be excluded: they are the verified baseline.",
                              suggestion="Use 'exclude unverified costs' or 'exclude provisional and unverified costs'.")
        return RuleResult(True, Params(exclude_cost_status=sts),
                          f"route decision view leaves out {' and '.join(sts)} cost components "
                          "(all three rankings stay visible).")

    m = re.search(rf"{_CUR}\s*{_NUM}\s*(?:rs\.?|inr|rupees?)?\s*{_PER}", low)
    if not m:
        return None
    head = re.sub(r"^(?:include|add|assume|apply|set|with|charge|there (?:is|are))\s+", "", low[: m.start()])
    name_m = re.match(r"^(?:an?\s+|the\s+)?(?P<name>[a-z][a-z \-]*?)(?:\s+(?:costs?|charges?|fees?))?\s*"
                      r"(?:of|at|is|are|=|:|,|costs?|will be|would be|\s)*(?:(?:" + _STATUS_WORDS + r")\s*)?(?:at\s*)?$", head)
    if not name_m or not name_m.group("name").strip():
        return None
    name = comp_key(name_m.group("name"))
    if not name or name in ("cost", "price"):
        return None
    value = float(m.group(1))
    unit = m.group("unit")
    per = "km" if unit.startswith(("km", "kilomet")) else "minute" if unit.startswith("min") else "order"
    if per == "km" and not any(re.search(_DIST_HINT, c.lower()) and roles.get(c) in ("measure", "quantity")
                               for c in df.columns):
        return RuleResult(False, reason="A per-km cost needs a distance column, and none was found.",
                          suggestion=f"Give it per order instead, e.g. '{name} of Rs {value:g} per order'.")
    status = _cost_status(low)
    seg_col = seg_val = None
    tail = low[m.end():]
    sm = re.search(r"\b(?:on|for|in|at|across|via)\s+(?:the\s+)?(?:route\s+)?(?P<seg>[a-z0-9][a-z0-9 _\-]*?)\s*"
                   r"(?:routes?|lanes?|zones?|hubs?|orders?|deliveries|regions?|areas?)?\s*(?:,|$|\bunverified|\bprovisional|"
                   r"\bconfirmed)", tail)
    if sm and sm.group("seg").strip() not in ("all", "every", "each", "the", ""):
        seg = sm.group("seg").strip()
        prefer = [c for c in df.columns if re.search(_ROUTE_HINT, c.lower())] + [setup.entity]
        hit = _find_values([seg], df, {**roles, **{c: "category" for c in prefer if c}}, prefer)
        if hit is None:
            return RuleResult(False, reason=f"'{seg}' is not a value in any label column, so the routes it applies to "
                                            "are unknown.",
                              suggestion=f"Use a label from the data, e.g. '{name} of Rs {value:g} per order on <label> "
                                         "routes, unverified', or leave out the place to apply it to every route.")
        seg_col, (seg_val,) = hit
    cc = CostComponent(name=name, value=value, per=per, status=status, segment_column=seg_col, segment_value=seg_val)
    where = f" on rows where '{seg_col}' = {seg_val}" if seg_col else " on every route"
    return RuleResult(True, Params(cost_components=[cc]),
                      f"route cost component '{name}' = {value:g} per {per}{where}, status {status}"
                      + (" (status not stated: defaults to unverified)" if not re.search(
                          r"unverified|not verified|unconfirmed|provisional|tentative|estimated|confirmed|verified",
                          low) else "") + ".")


def parse_rules(text: str, df: pd.DataFrame, roles: dict[str, str], setup: Setup,
                entity_of: dict[str, str] | None = None) -> RuleResult | None:
    entity_of = entity_of or {}
    t = text.strip().rstrip(".").strip()
    low = t.lower()

    # view change, not a scenario
    m = re.search(r"\b(?:show|group|break\s*down|split|view|report|display)\b.*\bby\s+([a-z_ ]+?)(?:\s+instead.*)?$", low)
    if m:
        target = m.group(1).strip()
        cand = [c for c in df.columns if _stem(target.split()[0]) in _n(c)]
        has_cap = [c for c in cand if any(e == c and roles.get(a) == "quantity" for a, e in entity_of.items())]
        where = f"'{cand[0]}'" if cand else f"a '{target}' column"
        sugg = (f"This is a view, not a scenario. On the KPIs page use 'Add a KPI' → segment_breakdown with group {where}"
                + (f", or in What-if → Model setup pick {where} as the entity" if has_cap else
                   f". The cascade needs a capacity per {target}, which this data does not have, so it stays by "
                   f"'{setup.entity}'" if setup.entity else ""))
        return RuleResult(False, reason=f"Changing the grouping to '{target}' does not change any number, so it "
                                        "cannot be applied as a constraint.", suggestion=sugg + ".")

    cost = _route_cost_rules(low, df, roles, setup)
    if cost is not None:
        return cost

    aliases = entity_aliases(df, setup.entity, roles) if setup.entity and setup.entity in df else {}
    days_m = re.search(rf"\bfor\s+{_NUM}\s*days?\b", low)
    days = float(days_m.group(1)) if days_m else None
    ents_all = sorted(df[setup.entity].dropna().astype("string").unique()) if aliases else []

    # outage: "<entity> is down / closed for N days"
    m = re.match(rf"^(?P<ent>.+?)\s+(?:is\s+|goes\s+|will\s+be\s+)?(?:down|closed|offline|shut(?:\s+down)?|out of service|unavailable)\b", low)
    if m and aliases:
        e = _resolve_entity(m.group("ent"), aliases)
        if e is None:
            return RuleResult(False, reason=f"'{m.group('ent')}' does not match one value of '{setup.entity}'.",
                              suggestion=f"Use one of: {', '.join(ents_all)}, e.g. '{ents_all[0]} is down for 3 days'.")
        p = Params(capacity_changes=[CapacityChange(entity_value=e, pct=-100, days=days)])
        return RuleResult(True, p, f"{setup.entity} {e} has zero capacity " +
                          (f"for {days:g} days" if days else "for the whole period") + "; its orders are re-allocated.")

    # capacity change: "capacity cut by 20% (at X | across all hubs) (for N days)"
    m = re.search(rf"capacity\b.*?\b(?P<dir>{_DOWN}|{_UP})\b.*?(?:by\s+)?{_NUM}\s*%", low) or \
        re.search(rf"\b(?P<dir>{_DOWN}|{_UP})\b.*?capacity\b.*?(?:by\s+)?{_NUM}\s*%", low)
    if m and "capacity" in low:
        if not aliases:
            return RuleResult(False, reason="No entity with a capacity was found in this data.",
                              suggestion="Set the entity and capacity columns in What-if → Model setup first.")
        pct = float(m.group(2)) * (-1 if re.match(_DOWN, m.group("dir")) else 1)
        every = re.search(rf"\b(all|every|each)\b|across\s+(?:the\s+)?(?:{_ENTITY_WORDS})", low)
        ent_txt = re.search(rf"\b(?:at|for|in)\s+(?!all\b|every\b|each\b|\d)(.+?)(?:\s+for\s+\d|\s+by\s+\d|$)", low)
        head = re.split(r"capacity", low)[0]
        e = None
        if not every:
            e = (_resolve_entity(ent_txt.group(1), aliases) if ent_txt else None) or _resolve_entity(head, aliases)
            if e is None:
                return RuleResult(False, reason="It is not clear which entity's capacity changes.",
                                  suggestion=f"Name one of {', '.join(ents_all)}, or say 'across all {setup.entity}'.")
        targets = ents_all if every else [e]
        p = Params(capacity_changes=[CapacityChange(entity_value=x, pct=pct, days=days) for x in targets])
        return RuleResult(True, p, f"capacity {pct:+g}% at " + ("every " + setup.entity if every else f"{setup.entity} {e}") +
                          (f" for {days:g} days" if days else " for the whole period") + ".")

    # price change with cap: "(delivery) fee increase capped at 5%" / "raise fee by 8%, capped at 5%"
    m = re.search(rf"capped\s+at\s+{_NUM}\s*%|(?:max(?:imum)?|limit(?:ed)?(?:\s+to)?|at most|no more than)\s+{_NUM}\s*%", low)
    if m and re.search(rf"\b(fee|charge|price|tariff|rate|cost)s?\b", low):
        cap = float(m.group(1) or m.group(2))
        req = re.search(rf"(?:by|of)\s+{_NUM}\s*%", low[: m.start()])
        pct = float(req.group(1)) if req else cap
        phrase = re.split(rf"\b(?:{_UP}|capped|by)\b", low)[0]
        col = _measure_column(phrase or "fee", roles, setup, entity_of) or setup.fee
        if not col or roles.get(col) != "money":
            return RuleResult(False, reason="No fee/price money column was found for this change.",
                              suggestion="Name the money column, e.g. '<fee column> increase capped at 5%'.")
        p = Params(price_changes=[PriceChange(column=col, pct=pct, cap_pct=cap)])
        return RuleResult(True, p, f"'{col}' rises {min(pct, cap):g}%" + (f" (asked {pct:g}%, capped at {cap:g}%)"
                                                                          if pct > cap else f", the cap of {cap:g}%") + ".")

    # exclusions: "ignore cancelled and returned orders"
    m = re.match(rf"^(?:ignore|exclude|remove|drop|leave out|filter out|without)\s+(?:all\s+|the\s+)?(?P<vals>.+?)(?:\s+(?:{_ROW_WORDS}))?$", low)
    if m:
        words = [w.strip() for w in re.split(r",|\band\b|\bor\b|&|/", m.group("vals")) if w.strip()]
        words = [re.sub(rf"\s+(?:{_ROW_WORDS})$", "", w) for w in words]
        hit = _find_values(words, df, roles, [setup.status])
        if hit is None:
            return RuleResult(False, reason=f"Could not find {words} as values of one column.",
                              suggestion="Use the exact labels shown in the data, e.g. 'ignore <label> orders'.")
        col, vals = hit
        p = Params(exclusions=[Exclusion(column=col, op="in", value=vals)])
        return RuleResult(True, p, f"rows where '{col}' is {' or '.join(vals)} are removed before every calculation.")

    # target: "Critical orders must reach 80% on time"
    m = re.match(rf"^(?P<seg>.*?)\s*(?:{_ROW_WORDS})?\s*(?:must|should|need to|needs to|has to|have to|to)\s+"
                 rf"(?:reach|be|hit|achieve|get to|stay)\s+(?:at least\s+|above\s+|over\s+)?{_NUM}\s*%\s+on[- ]?time", low)
    if m:
        v = float(m.group(2))
        seg = re.sub(rf"\b(?:{_ROW_WORDS}|all|overall|the)\b", " ", m.group("seg")).strip()
        if not seg:
            return RuleResult(True, Params(targets=[Target(metric="on_time_pct", op=">=", value=v)]),
                              f"feasibility requires on-time ≥ {v:g}% overall.")
        hit = _find_values([seg], df, roles, [])
        if hit is None:
            return RuleResult(False, reason=f"'{seg}' is not a value in any label column.",
                              suggestion="Use a label from the data, e.g. '<priority label> orders must reach 80% on time'.")
        col, (val,) = hit
        return RuleResult(True, Params(targets=[Target(metric="on_time_pct", op=">=", value=v, segment_column=col,
                                                       segment_value=val)]),
                          f"feasibility requires on-time ≥ {v:g}% for orders with '{col}' = {val}.")

    # measure change: "rider costs rise 10%"
    m = re.match(rf"^(?P<what>.+?)\s+(?:will\s+|to\s+)?(?P<dir>{_UP}|{_DOWN})\s+(?:by\s+)?{_NUM}\s*%", low)
    if m:
        col = _measure_column(m.group("what"), roles, setup, entity_of)
        if not col:
            return RuleResult(False, reason=f"No numeric column matches '{m.group('what')}'.",
                              suggestion="Name the column, e.g. '<cost column> rises 10%'.")
        pct = float(m.group(3)) * (-1 if re.match(_DOWN, m.group("dir")) else 1)
        note = "" if _stem(m.group("what").split()[-1]) in _n(col) and len(m.group("what").split()) == 1 else \
            f" (assumption: '{m.group('what')}' = column '{col}')"
        return RuleResult(True, Params(measure_changes=[MeasureChange(column=col, pct=pct)]),
                          f"'{col}' changes by {pct:+g}% on every row{note}.")
    return None
