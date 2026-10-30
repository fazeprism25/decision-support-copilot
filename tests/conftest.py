"""Automated tests never spend API requests: the real model call is replaced by a stub that fails like a dead API,
and the disk cache is pointed at an empty temp folder. Only RUN_AI_TESTS=1 (tests/test_ai_integration.py) calls
the real model."""
import ast
import os
from pathlib import Path

import pytest

from engine import llm, qa

# Competition files are not in the public repo; tests that need them skip when they are absent.
DATA = Path(__file__).resolve().parent.parent / "data"
PRIVATE = ["Round_1_BUDS.xlsx", "BUDS_unknown_schema_test.csv"]


def pytest_collection_modifyitems(items):
    missing = [n for n in PRIVATE if not (DATA / n).exists()]
    if not missing:
        return
    sources = {}  # module path -> {function name: source}
    for item in items:
        path = Path(item.fspath)
        if path not in sources:
            src = path.read_text(encoding="utf-8")
            defs = {n.name: ast.get_source_segment(src, n) for n in ast.walk(ast.parse(src))
                    if isinstance(n, ast.FunctionDef)}
            # a module-level `DATA = .../<missing file>` makes every use of DATA count as a use of that file
            data_line = next((l for l in src.splitlines() if l.startswith("DATA =")), "")
            sources[path] = defs, any(n in data_line for n in missing)
        defs, data_missing = sources[path]
        # the test body, the module fixtures it uses and its parameters
        text = " ".join([defs.get(item.originalname, ""), str(getattr(item, "callspec", None) and item.callspec.params)]
                        + [defs.get(f, "") for f in item.fixturenames])
        if any(n in text for n in missing) or (data_missing and "DATA" in text):
            item.add_marker(pytest.mark.skip(reason="competition data file not present in data/"))


@pytest.fixture(autouse=True)
def _no_live_ai(monkeypatch, tmp_path):
    llm.reset()
    if os.getenv("RUN_AI_TESTS") == "1":
        yield
        return

    def _blocked(*a, **k):
        raise ConnectionError("live AI blocked in automated tests")

    monkeypatch.setattr(llm, "_call_once", _blocked)
    monkeypatch.setattr(llm, "_CACHE_DIR", tmp_path / "llm_cache")
    monkeypatch.setattr(qa, "_QA_DIR", tmp_path / "qa_cache")
    yield
    llm.reset()
