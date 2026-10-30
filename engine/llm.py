"""Single entry point for LLM calls. Provider is chosen in .env; callers never see SDK details.
Failures raise LLMError only; callers catch it and show a message, so the app keeps working.
AI-down mode: every call has a hard wall-clock limit (AI_TIMEOUT_S, default 20s). After a failure the AI is
marked down for AI_COOLDOWN_S so later calls fail instantly instead of each waiting again."""
import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from engine.config import get_settings


class LLMError(Exception):
    pass


_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="llm")
_lock = threading.Lock()
_state = {"down_until": 0.0, "reason": "", "last_ok": None, "calls": []}  # calls: (when, tier, seconds, ok)


def status() -> tuple[bool, str]:
    """(usable, message). Usable is False while no key is set or during the cool-down after a failure."""
    s = get_settings()
    if not s.ai_enabled:
        return False, "AI is off: no LLM_API_KEY in .env."
    left = _state["down_until"] - time.time()
    if left > 0:
        return False, f"AI unavailable ({_state['reason'][:160]}). Retrying automatically in {left:.0f}s."
    return True, "AI on"


def mark_down(reason: str):
    with _lock:
        _state["down_until"] = time.time() + get_settings().cooldown_s
        _state["reason"] = reason


def reset():
    """Clear the AI-down flag (the 'Retry AI' button)."""
    with _lock:
        _state["down_until"], _state["reason"] = 0.0, ""


def recent_calls() -> list[tuple]:
    return list(_state["calls"][-50:])


def _call_once(model: str, system: str, prompt: str, max_tokens: int, reasoning: str = "off",
               timeout: float = 20.0) -> str:
    s = get_settings()
    if s.provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=s.api_key, timeout=timeout, max_retries=0)
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    from openai import OpenAI

    client = OpenAI(api_key=s.api_key, base_url=s.base_url, timeout=timeout, max_retries=0)
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
    """tier: 'fast' (labeling) or 'smart' (code-writing, reasoning). Raises LLMError on failure or after
    AI_TIMEOUT_S seconds in total (fallback model included)."""
    ok, msg = status()
    if not ok:
        raise LLMError(msg)
    s = get_settings()
    models = [s.model_smart if tier == "smart" else s.model_fast]
    reasoning = s.reasoning_smart if tier == "smart" else s.reasoning_fast
    if s.model_fallback and s.model_fallback not in models:
        models.append(s.model_fallback)
    errors, t0 = [], time.perf_counter()
    deadline = t0 + s.timeout_s
    for m in models:
        left = deadline - time.perf_counter()
        if left < 1:
            break
        fut = _POOL.submit(_call_once, m, system, prompt, max_tokens, reasoning, left)
        try:
            text = fut.result(timeout=left)
            if text.strip():
                _state["calls"].append((time.time(), tier, time.perf_counter() - t0, True))
                _state["last_ok"] = time.time()
                return text
            errors.append(f"{m}: empty reply")
        except FutureTimeout:
            errors.append(f"{m}: no reply within {s.timeout_s:g}s")
            break  # the time budget is spent
        except Exception as e:  # network, rate limit, retired model, bad key
            errors.append(f"{m}: {type(e).__name__}: {str(e)[:200]}")
    if time.perf_counter() >= deadline - 1 and not any("within" in e for e in errors):
        errors.append(f"stopped after {s.timeout_s:g}s")
    reason = " | ".join(errors) or "no model available"
    _state["calls"].append((time.time(), tier, time.perf_counter() - t0, False))
    mark_down(reason)
    raise LLMError(reason)


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
    text = chat(prompt, system=system, tier=tier, max_tokens=max_tokens)
    try:
        result = _extract_json(text)
    except LLMError as first:  # bad JSON (not a dead API): one repair attempt
        result = _extract_json(chat(prompt + f"\n\nYour previous reply failed: {first}. Return valid JSON only.",
                                    system=system, tier=tier, max_tokens=max_tokens))
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result), encoding="utf-8")
    return result
