"""Single entry point for LLM calls. Provider is chosen in .env; callers never see SDK details.
Failures raise LLMError only; callers catch it and show a message, so the app keeps working."""
import hashlib
import json
import re
from pathlib import Path

from engine.config import get_settings


class LLMError(Exception):
    pass


def _call_once(model: str, system: str, prompt: str, max_tokens: int, reasoning: str = "off") -> str:
    s = get_settings()
    if s.provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=s.api_key, timeout=60)
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    from openai import OpenAI

    client = OpenAI(api_key=s.api_key, base_url=s.base_url, timeout=60)
    # "off" disables thinking (much faster); otherwise think at that effort but hide the thinking text.
    # Thinking still uses tokens, so keep a floor.
    rb = {"enabled": False} if reasoning == "off" else {"effort": reasoning, "exclude": True}
    resp = client.chat.completions.create(
        model=model, max_tokens=max(max_tokens, 1024 if reasoning == "off" else 4096),
        extra_body={"reasoning": rb},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


def chat(prompt: str, system: str = "You are a precise assistant.", tier: str = "fast",
         max_tokens: int = 1024) -> str:
    """tier: 'fast' (labeling) or 'smart' (code-writing, reasoning). Raises LLMError on failure."""
    s = get_settings()
    if not s.ai_enabled:
        raise LLMError("AI is off: no LLM_API_KEY in .env.")
    models = [s.model_smart if tier == "smart" else s.model_fast]
    reasoning = s.reasoning_smart if tier == "smart" else s.reasoning_fast
    if s.model_fallback and s.model_fallback not in models:
        models.append(s.model_fallback)
    errors = []
    for m in models:
        try:
            text = _call_once(m, system, prompt, max_tokens, reasoning)
            if text.strip():
                return text
            errors.append(f"{m}: empty reply")
        except Exception as e:  # network, rate limit, retired model, bad key
            errors.append(f"{m}: {type(e).__name__}: {str(e)[:200]}")
    raise LLMError(" | ".join(errors))


_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "llm"


def _extract_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, flags=re.S)
        if not m:
            raise LLMError("Model reply contained no JSON.")
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise LLMError(f"Model reply was not valid JSON: {e}") from e


def chat_json(prompt: str, system: str = "Output strict JSON only.", tier: str = "fast",
              cache_key: str = "", max_tokens: int = 4096):
    """chat() that parses JSON. Successful results are cached on disk per (cache_key, model, prompt);
    cache_key should include the dataset hash so a new upload never reuses old answers."""
    s = get_settings()
    model = s.model_smart if tier == "smart" else s.model_fast
    h = hashlib.sha256(f"{cache_key}|{model}|{system}|{prompt}".encode()).hexdigest()[:24]
    path = _CACHE_DIR / f"{h}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        result = _extract_json(chat(prompt, system=system, tier=tier, max_tokens=max_tokens))
    except LLMError as first:
        result = _extract_json(chat(prompt + f"\n\nYour previous reply failed: {first}. Return valid JSON only.",
                                    system=system, tier=tier, max_tokens=max_tokens))
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return result
