"""Automated tests never spend API requests: the real model call is replaced by a stub that fails like a dead API,
and the disk cache is pointed at an empty temp folder. Only RUN_AI_TESTS=1 (tests/test_ai_integration.py) calls
the real model."""
import os

import pytest

from engine import llm, qa


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
