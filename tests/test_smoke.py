import pytest

from engine.config import get_settings
from engine.llm import LLMError, chat


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")


def test_settings_load_without_key(no_key):
    s = get_settings()
    assert s.ai_enabled is False
    assert s.model_smart and s.model_fast


def test_chat_fails_cleanly_without_key(no_key):
    with pytest.raises(LLMError):
        chat("hi")
