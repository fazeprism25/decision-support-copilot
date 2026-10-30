"""Live-AI checks. Slow and need a key, so they run only with RUN_AI_TESTS=1."""
import os
import re
from pathlib import Path

import pandas as pd
import pytest

from engine import kpis as K
from engine.loader import load_table
from engine.profile import profile_frame
from engine.qa import ask, schema_text
from engine.quality import apply_fixes, find_issues, recommended_fixes
from engine.roles import detect_roles, entity_attributes

pytestmark = pytest.mark.skipif(os.getenv("RUN_AI_TESTS") != "1", reason="set RUN_AI_TESTS=1 to run live-AI tests")
DATA = Path(__file__).resolve().parent.parent / "data" / "BUDS_unknown_schema_test.csv"


def _as_pct(result) -> float:
    if isinstance(result, pd.DataFrame):
        result = result.select_dtypes("number").iloc[0, 0]
    elif isinstance(result, pd.Series):
        result = result.iloc[0]
    v = float(result)
    return 100 * v if v <= 1 else v


def test_qa_on_time_matches_kpi():
    df, _ = load_table(DATA.read_bytes(), DATA.name)
    prof = profile_frame(df)
    roles, _ = detect_roles(df, "ai-test", use_ai=False, prof=prof)
    ent = entity_attributes(df, roles)
    clean = apply_fixes(df, recommended_fixes(find_issues(df, roles, set(ent["attribute"]))))
    ctx = K.Ctx({c: r for c, r, k in zip(roles["column"], roles["role"], roles["check"]) if k != "fail"})
    item = {"calc": "on_time_rate", "params": {"actual": "tat_mins", "promised": "sla_mins", "status": "outcome",
                                               "completed_values": ["completed"]}}
    kpi = K.run(item, clean, ctx)
    notes = f"KPI definitions used on the KPI tab (use the same when relevant):\n- {kpi.title}: {kpi.formula}; filter {kpi.filter}"
    a = ask("What percentage of completed shipments were on time?", clean,
            schema_text(clean, ctx.roles, {}, prof), notes)
    assert not a.error, a.error
    assert abs(_as_pct(a.result) - kpi.value) < 0.05, (a.code, a.result, kpi.value)
