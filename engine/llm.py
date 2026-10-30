"""Single entry point for LLM calls. Provider is chosen in .env; callers never see SDK details.
Every function returns None (never raises) when AI is unavailable, so the app keeps working."""
from engine.config import get_settings


class LLMError(Exception):
    pass


def _call_once(model: str, system: str, prompt: str, max_tokens: int) -> str:
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
    # Hide "thinking" text from reasoning models; reasoning still uses tokens, so keep a floor.
    resp = client.chat.completions.create(
        model=model, max_tokens=max(max_tokens, 1024),
        extra_body={"reasoning": {"exclude": True}},
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
    if s.model_fallback and s.model_fallback not in models:
        models.append(s.model_fallback)
    errors = []
    for m in models:
        try:
            text = _call_once(m, system, prompt, max_tokens)
            if text.strip():
                return text
            errors.append(f"{m}: empty reply")
        except Exception as e:  # network, rate limit, retired model, bad key
            errors.append(f"{m}: {type(e).__name__}: {str(e)[:200]}")
    raise LLMError(" | ".join(errors))
