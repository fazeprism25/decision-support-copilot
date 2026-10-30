"""Headless version of the app flow (load → roles → recommended fixes → value mapping → What-if setup → domain pack).
Used by tests, the 'Load demo scenario' button and dataset checks; the UI runs the same functions step by step."""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from engine import kpis as K
from engine.domains import delivery as D
from engine.loader import file_hash, load_table
from engine.profile import profile_frame
from engine.quality import Fix, apply_fixes, find_issues, recommended_fixes
from engine.roles import detect_roles, entity_attributes
from engine.scenario import Setup, guess_setup
from engine.valuemap import apply_value_map


@dataclass
class Run:
    name: str
    raw: pd.DataFrame
    profile: dict
    roles: pd.DataFrame
    entity_of: dict[str, str]
    fixes: list[Fix]
    vmap: dict
    df: pd.DataFrame  # cleaned + mapped: what every tab uses
    setup: Setup
    pack: D.Pack
    plan: list[dict] = field(default_factory=list)

    @property
    def role_map(self) -> dict[str, str]:
        return {c: r for c, r, k in zip(self.roles["column"], self.roles["role"], self.roles["check"])
                if k != "fail" and r != "ignore"}

    @property
    def ctx(self) -> K.Ctx:
        return K.Ctx(self.role_map, self.entity_of)


def prepare(path: str | Path, use_ai: bool = False, fixes: bool = True, mapping: bool = True) -> Run:
    path = Path(path)
    data = path.read_bytes()
    raw, info = load_table(data, path.name)
    prof = profile_frame(raw)
    roles, _ = detect_roles(raw, f"{file_hash(data)}:{info.sheet}:{info.header_row}", use_ai=use_ai, prof=prof)
    ent = entity_attributes(raw, roles)
    entity_of = dict(zip(ent["attribute"], ent["entity"]))
    rm = {c: r for c, r, k in zip(roles["column"], roles["role"], roles["check"]) if k != "fail" and r != "ignore"}
    fx = recommended_fixes(find_issues(raw, roles, set(entity_of))) if fixes else []
    clean = apply_fixes(raw, fx)
    vmap = {c: {k: v for k, v in m.items() if v} for c, m in D.propose_mapping_rules(clean, rm)[0].items()} if mapping else {}
    df = apply_value_map(clean, vmap)
    ctx = K.Ctx(rm, entity_of)
    plan = K.default_plan(df, ctx)
    setup = guess_setup(df, rm, entity_of, plan, prof)
    pack = D.detect(df, rm, entity_of, setup)
    D.fill_setup(setup, pack)
    return Run(path.name, raw, prof, roles, entity_of, fx, vmap, df, setup, pack, plan)
