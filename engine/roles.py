"""Column-role detection: rule-based guess, optional LLM labels, and a stats check that can veto either.
Name hints below are generic English words, not names from any particular dataset."""
import json
import re

import pandas as pd
from pydantic import BaseModel, ValidationError

from engine.llm import LLMError, chat_json
from engine.profile import profile_frame, sample_rows

ROLES = ["date", "id", "category", "money", "quantity", "duration", "rating",
         "flag", "location", "measure", "text", "ignore"]
ROLE_HELP = {
    "date": "date or timestamp", "id": "identifier of a record or entity",
    "category": "small set of labels used for grouping", "money": "currency amount (price, fee, cost, revenue)",
    "quantity": "count of things (units, items, orders)", "duration": "length of time (minutes, hours, days)",
    "rating": "score on a fixed scale (e.g. 1-5)", "flag": "yes/no or two-valued indicator",
    "location": "place (zone, city, region, hub, address)", "measure": "any other numeric measurement (distance, %, ratio)",
    "text": "free text / comments", "ignore": "not useful for analysis",
}
NUMERIC_ROLES = {"money", "quantity", "duration", "rating", "measure"}

_HINTS = {
    "id": r"(^|_|\b)(id|code|no|num|number|key|ref|sku)($|_|\b)",
    "money": r"price|cost|fee|revenue|amount|sales|salary|profit|margin|value|charge|pay|income|spend|discount|tip|wage|bonus|(^|_)(rs|inr|usd|eur|gbp|amt)($|_)",
    "duration": r"time|min|mins|minute|hours|hrs|duration|days|sec|lead|eta|delay|sla|tat|tenure|wait",
    "rating": r"rating|score|stars|csat|nps|satisfaction|review",
    "quantity": r"qty|quantity|count|units|items|orders|pieces|volume|headcount|capacity",
    "location": r"city|region|zone|state|country|area|location|address|lat|lon|postcode|zip|hub|branch|store|warehouse|site|district",
}


def _hint(name: str, role: str) -> bool:
    n = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower().replace(" ", "_")
    return re.search(_HINTS[role], n) is not None


def rule_role(name: str, p: dict) -> tuple[str, str]:
    """Best-guess role from stats + generic name hints. Returns (role, reason)."""
    n_valid = p["n"] - p["missing"]
    if n_valid == 0:
        return "ignore", "column is empty"
    if p["n_unique"] == 1:
        return "ignore", "single constant value"
    if p["date_rate"] >= 0.8:
        return "date", f"{p['date_rate']:.0%} of values parse as dates/times"
    if p.get("yes_no_like"):
        return "flag", "values are yes/no style"
    if p["n_unique"] <= 2 and p.get("avg_len", 0) <= 12:
        return "flag", f"only {p['n_unique']} distinct values"
    if p["numeric_rate"] >= 0.9:
        if _hint(name, "id") and p["integer_like"] and p["unique_ratio"] >= 0.8:
            return "id", "integer values, mostly unique, name suggests an identifier"
        if p.get("currency_symbol"):
            return "money", "values carry a currency symbol"
        if p.get("unit") in ("min", "mins", "m", "h", "hr", "hrs", "hour", "hours", "sec", "s", "d", "day", "days"):
            return "duration", f"values carry the time unit '{p['unit']}'"
        for role in ("rating", "money", "duration", "quantity"):
            if _hint(name, role):
                if role == "rating" and not (p["n_unique"] <= 21 and p["min"] >= 0):
                    continue
                return role, f"numeric; name suggests {ROLE_HELP[role]}"
        return "measure", "numeric with no more specific hint"
    if p["unique_ratio"] >= 0.8 and n_valid > 20:
        if p.get("avg_len", 0) > 30:
            return "text", "long, mostly unique text"
        return "id", f"{p['unique_ratio']:.0%} of values are unique"
    if _hint(name, "location"):
        return "location", "text labels; name suggests a place"
    if p.get("avg_len", 0) > 30:
        return "text", "long text values"
    return "category", f"{p['n_unique']} distinct labels"


def check_role(role: str, p: dict, name: str = "") -> tuple[str, str]:
    """Verify a proposed role against the data. Returns (status, message); status in ok / warn / fail."""
    n_valid = p["n"] - p["missing"]
    if role == "ignore":
        return "ok", ""
    if role == "text":
        if p["n_unique"] <= 50 and p.get("avg_len", 0) <= 30:
            return "fail", f"only {p['n_unique']} short distinct values; looks like labels, not free text"
        return "ok", ""
    if n_valid == 0:
        return "fail", "column is empty"
    if role == "date":
        return ("ok", "") if p["date_rate"] >= 0.8 else ("fail", f"only {p['date_rate']:.0%} parse as dates")
    if role in NUMERIC_ROLES:
        if p["numeric_rate"] < 0.9:
            return "fail", f"only {p['numeric_rate']:.0%} of values are numbers"
        if role == "rating" and p["n_unique"] > 21:
            return "fail", f"{p['n_unique']} distinct values is too many for a rating scale"
        if role == "rating" and p["max"] > 10 and not re.search(r"\d+\s*(?:to|-|_)\s*\d+|out_?of", name or ""):
            return "fail", f"values up to {p['max']:g} are unusual for a rating scale"
        if role in ("duration", "quantity") and p["share_negative"] > 0:
            return "warn", f"{p['share_negative']:.1%} negative values"
        return "ok", ""
    if role == "flag":
        k = 2 if p.get("yes_no_like") else p.get("n_unique_normalized", p["n_unique"])
        if k > 3:
            return "fail", f"{k} distinct values; a flag should have 2"
        return ("warn", f"{k} distinct values (inconsistent spelling?)") if k == 3 else ("ok", "")
    if role == "id":
        return ("ok", "") if p["unique_ratio"] >= 0.8 else ("fail", f"only {p['unique_ratio']:.0%} unique")
    if role in ("category", "location"):
        if p["unique_ratio"] > 0.9 and n_valid > 50:
            return "fail", f"{p['unique_ratio']:.0%} unique; too many labels to group by"
        return "ok", ""
    return "ok", ""


class _Label(BaseModel):
    role: str
    meaning: str = ""


def _llm_prompt(df: pd.DataFrame, prof: dict) -> str:
    keep = ("dtype", "missing_pct", "n_unique", "unique_ratio", "numeric_rate", "date_rate",
            "min", "max", "mean", "integer_like", "avg_len", "top_values", "examples", "unit",
            "currency_symbol", "percent_sign", "yes_no_like", "date_order")
    stats = {c: {k: v for k, v in p.items() if k in keep} for c, p in prof.items()}
    roles = "\n".join(f"- {r}: {h}" for r, h in ROLE_HELP.items())
    return (
        "Label the role of every column in this dataset.\n"
        f"Allowed roles:\n{roles}\n\n"
        f"Column statistics (computed by code):\n{json.dumps(stats, default=str)}\n\n"
        f"Five sample rows:\n{json.dumps(sample_rows(df), default=str)}\n\n"
        'Return ONLY JSON: {"<column name>": {"role": "<allowed role>", "meaning": "<max 12 words: what it '
        'represents, include units if visible>"}, ...} with every column exactly as named.'
    )


def detect_roles(df: pd.DataFrame, dataset_hash: str, use_ai: bool = True,
                 prof: dict | None = None) -> tuple[pd.DataFrame, str | None]:
    """Returns (roles table, AI status message or None). The table is the single source of truth for roles."""
    prof = prof or profile_frame(df)
    ai, ai_msg = {}, None
    if use_ai:
        try:
            raw = chat_json(_llm_prompt(df, prof), tier="fast", cache_key=f"roles:{dataset_hash}",
                            system="You label spreadsheet columns. Output strict JSON only.")
            lookup = {str(k).strip().lower(): v for k, v in raw.items()} if isinstance(raw, dict) else {}
            for c in df.columns:
                try:
                    lab = _Label.model_validate(lookup.get(c.lower(), {}))
                    if lab.role.lower() in ROLES:
                        ai[c] = (lab.role.lower(), lab.meaning)
                except ValidationError:
                    pass
            if len(ai) < len(df.columns):
                ai_msg = f"AI labelled {len(ai)} of {len(df.columns)} columns; the rest use rules."
        except LLMError as e:
            ai_msg = f"AI labelling unavailable, using rules only. ({str(e)[:150]})"

    rows = []
    for c in df.columns:
        p = prof[c]
        rule, reason = rule_role(c, p)
        meaning = ""
        if c in ai:
            role, meaning = ai[c]
            status, msg = check_role(role, p, c)
            if status == "fail":
                source, note = "rules (AI rejected)", f"AI said '{role}' but {msg}. Rules: {reason}"
                role = rule
                status, msg = check_role(role, p, c)
            else:
                source = "AI" if role == rule else "AI (rules said " + rule + ")"
                note = msg
        else:
            role, source, note = rule, "rules", reason
            status, msg = check_role(role, p, c)
            note = f"{reason}; {msg}" if msg else reason
        rows.append({
            "column": c, "role": role, "source": source, "check": status,
            "note": note, "meaning": meaning, "missing_%": p["missing_pct"],
            "unique": p["n_unique"], "example": ", ".join(p["examples"][:3]),
        })
    return pd.DataFrame(rows), ai_msg


def entity_attributes(df: pd.DataFrame, roles: pd.DataFrame, min_consistency: float = 0.98) -> pd.DataFrame:
    """Find columns whose value is (almost) fixed per entity, e.g. a capacity repeated on every row of a group.
    Candidate entities: category/location/id columns with 2..500 groups and at most half the rows unique."""
    role_of = dict(zip(roles["column"], roles["role"]))
    n = len(df)
    entities = []
    for c, r in role_of.items():
        if r in ("category", "location", "id"):
            k = df[c].nunique()
            if 2 <= k <= 500 and k <= 0.5 * n:
                entities.append((k, c))
    entities.sort()
    found = []
    for col, r in role_of.items():
        if r in ("ignore", "text", "date") or col in [e for _, e in entities]:
            continue
        vals = df[col]
        if vals.nunique() < 2:
            continue
        for k, ent in entities:
            if vals.nunique() > k:
                continue
            sub = df[[ent, col]].dropna()
            if len(sub) < 0.5 * n:
                continue
            mode_hits = sub.groupby([ent, col]).size().groupby(level=0).max().sum()
            consistency = mode_hits / len(sub)
            if consistency >= min_consistency:
                found.append({"attribute": col, "entity": ent, "groups": k,
                              "consistency_%": round(100 * consistency, 1),
                              "meaning": f"'{col}' is a property of each '{ent}', not of each row"})
                break
    return pd.DataFrame(found, columns=["attribute", "entity", "groups", "consistency_%", "meaning"])
