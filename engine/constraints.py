"""Constraint card text → validated Params. The LLM only translates words into the fixed schema;
pydantic + validate_params decide whether it is usable. Nothing is applied without user confirmation."""
import json
from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError

from engine.llm import LLMError, chat_json
from engine.scenario import Params, Setup, validate_params

SCHEMA_HELP = """Params JSON schema (all keys optional):
- measure_changes: [{"column": <numeric column>, "pct": <% change>, "abs": <absolute change>}]
- exclusions: [{"column": <column>, "op": "==|!=|in|not_in|>|<|>=|<=", "value": <value or list>}]  (matching rows are REMOVED)
- capacity_changes: [{"entity_value": <one value of the entity column>, "pct": <% change, -100 = fully down>,
                      "abs": <absolute change>, "days": <days affected, omit for the whole period>}]
- price_changes: [{"column": <money column>, "pct": <% change>, "cap_pct": <maximum % allowed>}]
- targets: [{"metric": "on_time_pct|max_utilisation_pct|margin_per_order|unserved_per_day|avg_cycle_time",
             "op": ">=|<=", "value": <number>}]
- util_cap: <0-1, max utilisation entities receiving moved orders may reach>
- moved_extra_cost: <extra cost per re-allocated order>, moved_extra_minutes: <extra minutes per re-allocated order>"""


@dataclass
class Parsed:
    text: str
    ok: bool
    params: Params | None = None
    interpretation: str = ""
    reason: str = ""
    suggestion: str = ""
    errors: list[str] = field(default_factory=list)


def _context(df: pd.DataFrame, roles: dict[str, str], setup: Setup, profile: dict) -> str:
    cols = []
    for c, r in roles.items():
        d = {"column": c, "role": r}
        top = profile.get(c, {}).get("top_values")
        if top and profile[c]["n_unique"] <= 25:
            d["values"] = sorted(df[c].dropna().astype("string").unique().tolist())[:25]
        cols.append(d)
    ent = None
    if setup.entity:
        ent = {"entity_column": setup.entity, "values": sorted(df[setup.entity].dropna().astype("string").unique())[:50],
               "capacity_column": setup.capacity}
    return json.dumps({"columns": cols, "entity": ent}, default=str)


def parse(text: str, df: pd.DataFrame, roles: dict[str, str], setup: Setup, profile: dict, cache_key: str) -> Parsed:
    prompt = (
        f"Convert this business constraint into scenario parameters.\nConstraint: \"{text}\"\n\n{SCHEMA_HELP}\n\n"
        f"Dataset context (use exact column names and values):\n{_context(df, roles, setup, profile)}\n\n"
        "Rules: map names in the constraint to the closest exact value (e.g. 'Hub 3' → the matching entity value). "
        "A rise 'capped at X%' with no other number means pct = X and cap_pct = X. 'Ignore/exclude X' means an "
        "exclusion. If it cannot be expressed with this schema or the data, set mappable=false.\n"
        'Return ONLY JSON: {"mappable": true|false, "params": {...}, "interpretation": "<one sentence in plain '
        'English of what will change>", "reason": "<if not mappable>", "suggestion": "<a rephrasing that would work>"}'
    )
    try:
        raw = chat_json(prompt, tier="smart", cache_key=f"constraint:{cache_key}",
                        system="You translate business constraints into a fixed JSON schema. Never compute results.")
    except LLMError as e:
        return Parsed(text, False, reason=f"AI unavailable: {e}",
                      suggestion="Use the sliders in the sidebar, which work without AI.")
    if not isinstance(raw, dict) or not raw.get("mappable"):
        return Parsed(text, False, reason=(raw or {}).get("reason") or "The constraint could not be mapped.",
                      suggestion=(raw or {}).get("suggestion", ""))
    try:
        params = Params.model_validate(raw.get("params") or {})
    except ValidationError as e:
        return Parsed(text, False, reason="The AI's parameters did not fit the allowed schema.",
                      errors=[f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()],
                      suggestion="Rephrase with a column name and a number, e.g. 'increase <fee column> by 5%'.")
    errs = validate_params(params, df, roles, setup)
    if params.is_empty():
        errs.append("no parameters were produced")
    if errs:
        return Parsed(text, False, params, raw.get("interpretation", ""), "The parameters do not match the data.",
                      raw.get("suggestion", ""), errs)
    return Parsed(text, True, params, raw.get("interpretation", ""))
