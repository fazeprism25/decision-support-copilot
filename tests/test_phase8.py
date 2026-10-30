"""Phase 8: AI-down mode, Q&A cache + examples, unrelated datasets, typed-constraint set, demo safety.
The whole app is driven headlessly with Streamlit's AppTest; the AI is always mocked (see conftest.py)."""
import sys
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from engine import llm, qa
from engine.pipeline import prepare
from engine.qa import Answer
from engine.quality import find_issues

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))
from constraint_check import CARDS, check  # noqa: E402
from make_synthetic import hr, retail  # noqa: E402


@pytest.fixture
def ai_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")  # AI "on" in settings; every call still fails (conftest)


# ---------- AI-down mode ----------

def test_slow_ai_is_abandoned_and_marked_down(monkeypatch, ai_key):
    monkeypatch.setenv("AI_TIMEOUT_S", "1.5")
    calls = []

    def slow(*a, **k):
        calls.append(1)
        time.sleep(5)
        return "too late"

    monkeypatch.setattr(llm, "_call_once", slow)
    t = time.perf_counter()
    with pytest.raises(llm.LLMError, match="no reply within"):
        llm.chat("hi")
    assert time.perf_counter() - t < 2.5
    ok, msg = llm.status()
    assert not ok and "unavailable" in msg.lower()
    t = time.perf_counter()
    with pytest.raises(llm.LLMError):  # cool-down: fails instantly, no new request
        llm.chat_json("hi", cache_key="x")
    assert time.perf_counter() - t < 0.2 and len(calls) == 1
    llm.reset()
    assert llm.status()[0]


def test_dead_api_is_not_retried_by_chat_json(monkeypatch, ai_key):
    calls = []

    def dead(*a, **k):
        calls.append(1)
        raise ConnectionError("down")

    monkeypatch.setattr(llm, "_call_once", dead)
    monkeypatch.setenv("MODEL_FALLBACK", "")
    with pytest.raises(llm.LLMError):
        llm.chat_json("hi", cache_key="y")
    assert len(calls) == 1


def _app() -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    return at


def _load(at: AppTest, name: str) -> AppTest:
    at.radio(key="src").set_value("Use a file from the data folder").run()
    at.selectbox(key="src_file").set_value(name).run()
    return at


PAGES = ["Data", "Quality", "KPIs", "Ask", "What-if", "Route cost", "Recommendations", "Explain", "Change Log"]


def _visit(at: AppTest, page: str) -> AppTest:
    """Open one page from the sidebar menu (only the open page runs)."""
    at.radio(key="page").set_value(page).run()
    assert not at.exception, (page, [e.message for e in at.exception])
    return at


def _all_pages(at: AppTest) -> tuple[str, list[str], int]:
    """Visit every page; return all text, all expander labels and the number of sliders seen."""
    txt, exp, sliders = [], [], 0
    for page in PAGES:
        _visit(at, page)
        txt.append(_text(at))
        exp += [e.label for e in at.expander]
        sliders += len(at.slider)
    return "\n".join(txt), exp, sliders


def _text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [m.value for m in at.caption]
    for kind in (at.error, at.warning, at.info, at.success):
        parts += [m.value for m in kind]
    return "\n".join(parts)


def test_app_keeps_working_with_ai_down(ai_key):
    at = _load(_app(), "BUDS_unknown_schema_test.csv")
    assert not at.exception, [e.message for e in at.exception]
    txt, exp, sliders = _all_pages(at)
    assert "AI-down mode" in txt  # clear message in the sidebar
    assert "AI labelling unavailable, using rules only" in txt  # roles still detected by rules
    assert "Quality score" in txt  # quality
    assert "Margin per order" in txt and "Hub load model" in txt  # KPIs + delivery pack
    assert "Cascade" in txt or "Feasible" in txt  # What-if
    assert "Showing rule wording" in txt and "below the 80% goal" in txt  # rule-based recommendations
    assert "1. Data used" in exp  # Explain page
    assert sliders >= 3  # What-if levers


@pytest.mark.parametrize("name", ["retail_sales.csv", "hr_employees.csv"])
def test_unrelated_datasets_run_with_pack_off(name):
    if not (DATA / name).exists():
        pytest.skip("run scripts/make_synthetic.py")
    at = _load(_app(), name)
    assert not at.exception, [e.message for e in at.exception]
    txt = _all_pages(at)[0]
    assert "Delivery-ops pack off" in txt
    assert "Hub load model" not in txt
    assert "Duplicate record" in txt  # the planted duplicates are found


def test_demo_and_reset_buttons():
    from ui import demo

    if not demo.available():
        pytest.skip(f"demo file data/{demo.DEMO_FILE} not present")
    at = _app()
    next(b for b in at.sidebar.button if b.label == "Load demo data").click().run()
    assert not at.exception, [e.message for e in at.exception]
    msg = " ".join(s.value for s in at.sidebar.caption)
    assert "Demo loaded" in msg and "Central hub is down for 3 days" in msg and "applied" in msg
    assert at.session_state["constraints"] and at.session_state["fixes"]
    assert "#1" in _text(_visit(at, "Change Log"))  # logged in the Change Log
    next(b for b in at.sidebar.button if b.label == "Reset everything").click().run()
    assert not at.exception
    assert "constraints" not in at.session_state
    assert "Load a dataset on the Data page first." in _text(_visit(at, "KPIs"))


def test_kpi_page_rebuilds_plan_lost_mid_run():
    """A run cut short while the AI picks KPIs (page switch) leaves kpi_plan_for set but no kpi_plan."""
    at = _visit(_load(_app(), "Round_1_BUDS.xlsx"), "KPIs")
    del at.session_state["kpi_plan"]
    _visit(at, "Data")
    assert "Key numbers" in _text(_visit(at, "KPIs"))


# ---------- Q&A cache + examples ----------

def test_qa_cache_by_question_and_data(monkeypatch):
    calls = []

    def fake_ask(q, df, schema, notes=""):
        calls.append(q)
        return Answer(q, code="result = 1", result=1, narration="One.")

    monkeypatch.setattr(qa, "ask", fake_ask)
    a1, hit1 = qa.cached_ask("How many rows?", None, "", "", "h1")
    a2, hit2 = qa.cached_ask("  how many rows ", None, "", "", "h1")
    a3, hit3 = qa.cached_ask("How many rows?", None, "", "", "h2")  # different data → new answer
    assert (hit1, hit2, hit3) == (False, True, False) and len(calls) == 2 and a2.narration == "One."


def test_failed_answers_are_not_cached(monkeypatch):
    monkeypatch.setattr(qa, "ask", lambda q, *a, **k: Answer(q, error="AI unavailable"))
    qa.cached_ask("q", None, "", "", "h")
    assert qa.cached("q", "h") is None


def test_prefetch_fills_cache(monkeypatch):
    monkeypatch.setattr(qa, "ask", lambda q, *a, **k: Answer(q, result=2, narration="Two."))
    qs = [f"question {i}" for i in range(6)]
    assert qa.prefetch(qs, None, "", "", "hp") == 6
    for _ in range(100):
        if all(qa.cached(q, "hp") for q in qs):
            break
        time.sleep(0.05)
    assert all(qa.cached(q, "hp") for q in qs)
    assert qa.prefetch(qs, None, "", "", "hp") == 0  # nothing left to do


@pytest.mark.parametrize("name", ["Round_1_BUDS.xlsx", "BUDS_unknown_schema_test.csv", "retail_sales.csv",
                                  "hr_employees.csv"])
def test_six_example_questions(name):
    if not (DATA / name).exists():
        pytest.skip("missing")
    run = prepare(DATA / name)
    qs = qa.example_questions(run.df, run.role_map, run.entity_of, run.pack)
    assert len(qs) == 6 and len(set(qs)) == 6
    cols = set(run.df.columns)
    assert all(any(c in q for c in cols) for q in qs[:5])  # built from this data's own columns


# ---------- synthetic datasets ----------

@pytest.mark.parametrize("make,messy", [(retail, "price"), (hr, "dept")])
def test_synthetic_datasets_are_messy_and_pack_off(tmp_path, make, messy):
    path = tmp_path / "d.csv"
    make().to_csv(path, index=False, encoding="utf-8")
    run = prepare(path, fixes=False)
    assert not run.pack.active
    issues = find_issues(run.raw, run.roles, set(run.entity_of))
    assert any(i.rule in ("duplicate_record", "duplicate_id") for i in issues)
    assert any(i.column == messy for i in issues)
    if messy == "price":  # "₹1,234.50" and "Rs. 320" parse; only the "n/a" are left missing
        assert run.roles.set_index("column").at["price", "role"] == "money"
        assert not any(i.rule == "not_numeric" for i in issues)


def test_retail_finds_planted_returns_signal(tmp_path):
    from engine import kpis as K
    from engine import recommend as R

    path = tmp_path / "r.csv"
    retail().to_csv(path, index=False, encoding="utf-8")
    run = prepare(path)
    recs = R.generate(R.Inputs(run.df, run.ctx, kpis=K.run_plan(run.plan, run.df, run.ctx), pack=run.pack))
    assert any(r.rule == "ratio_gap" and "Whitefield" in r.title for r in recs)
    assert not any(r.rule == "segment_gap" and "product" in r.title for r in recs)  # price defined by product


# ---------- typed-constraint test set ----------

@pytest.mark.parametrize("name", ["Round_1_BUDS.xlsx", "BUDS_unknown_schema_test.csv"])
def test_constraint_cards(name):
    rows = check(DATA / name)
    assert len(rows) == len(CARDS)
    failed = [(t, d) for t, _, res, _, d in rows if res != "PASS"]
    assert not failed, failed
