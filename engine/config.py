"""Settings loaded from .env / environment. Nothing here may raise if the key is missing."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str
    base_url: str
    model_smart: str
    model_fast: str
    model_fallback: str
    reasoning_fast: str = "off"
    reasoning_smart: str = "low"
    timeout_s: float = 20.0  # any AI call slower than this is abandoned and the app falls back to rules
    cooldown_s: float = 60.0  # after a failure, skip AI for this long so every click doesn't wait again

    @property
    def ai_enabled(self) -> bool:
        return bool(self.api_key)


def get_settings() -> Settings:
    env = lambda k, d="": os.getenv(k, d).strip()
    return Settings(
        provider=env("LLM_PROVIDER", "openrouter").lower(),
        api_key=env("LLM_API_KEY") or env("ANTHROPIC_API_KEY"),
        base_url=env("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        model_smart=env("MODEL_SMART", "openrouter/free"),
        model_fast=env("MODEL_FAST", "openrouter/free"),
        model_fallback=env("MODEL_FALLBACK"),
        reasoning_fast=env("REASONING_FAST", "off").lower(),
        reasoning_smart=env("REASONING_SMART", "low").lower(),
        timeout_s=float(env("AI_TIMEOUT_S", "20") or 20),
        cooldown_s=float(env("AI_COOLDOWN_S", "60") or 60),
    )
