"""Natural-language Q&A: LLM writes pandas code → sandbox executes → LLM narrates only the computed result."""
import hashlib
import json
import pickle
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from engine.llm import LLMError, chat
from engine.profile import sample_rows, to_datetime_clean
from engine.sandbox import SandboxError, run_code

SYSTEM = "You write short, correct pandas code to answer business questions about a DataFrame named df."


@dataclass
class Answer:
    question: str
    code: str = ""
    result: object = None
    method: str = ""
    narration: str = ""
    chart: dict | None = None
    error: str = ""
    attempts: int = 0
    warnings: list[str] = field(default_factory=list)


def schema_text(df: pd.DataFrame, roles: dict[str, str], meanings: dict[str, str], profile: dict) -> str:
    cols = []
    for c in df.columns:
        p = profile.get(c, {})
        d = {"column": c, "dtype": str(df[c].dtype), "role": roles.get(c, "ignore")}
        if meanings.get(c):
            d["meaning"] = meanings[c]
        if "top_values" in p and p.get("n_unique", 99) <= 25:  # labels as they are now (after value mapping)
            top = df[c].dropna().astype("string").value_counts()
            d["labels"] = top.index[:8].tolist() + (["..."] if len(top) > 8 else [])
        elif "min" in p:
            d["range"] = [p["min"], p["max"]]
        cols.append(d)
    return (f"df has {len(df):,} rows. Columns:\n{json.dumps(cols, default=str)}\n"
            f"Sample rows:\n{json.dumps(sample_rows(df), default=str)}")


def _prompt(question: str, schema: str, notes: str) -> str:
    return (
        f"{schema}\n{notes}\n\nQuestion: {question}\n\n"
        "Write Python code that computes the answer from df.\n"
        "Rules:\n- pd and np are already available; do not import anything else; no files, no printing needed.\n"
        "- Use exact column names. Numeric-looking text columns may need pd.to_numeric(..., errors='coerce').\n"
        "- Assign the final answer to `result` (a number, a Series, or a DataFrame with at most 50 rows).\n"
        "- Assign `method` = one sentence stating the filter(s) and the denominator used.\n"
        "- Optionally assign `chart = {'type': 'bar' or 'line', 'x': <column of result>, 'y': <column of result>}` "
        "when a chart helps (result must then be a DataFrame; use reset_index()).\n"
        "- If the columns cannot answer the question, set `result = None` and `cannot_answer` = the reason.\n"
        "Reply with ONE ```python code block and nothing else."
    )


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip()


def result_text(result, limit: int = 3000) -> str:
    if isinstance(result, (pd.DataFrame, pd.Series)):
        return result.head(50).to_string()[:limit]
    return str(result)[:limit]


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def unverified_numbers(narration: str, result, question: str) -> list[str]:
    """Numbers in the narration that do not appear (allowing rounding or ×100 for %) in the computed result."""
    def nums(t):
        return [float(x.replace(",", "")) for x in _NUM.findall(t) if x.replace(",", "").replace(".", "").strip("-")]

    have = nums(result_text(result, 20000)) + nums(question)
    bad = []
    for raw in _NUM.findall(narration):
        s = raw.replace(",", "").rstrip(".")
        if not s.strip("-").replace(".", ""):
            continue
        v = float(s)
        dec = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** -dec + 1e-9
        if not any(abs(h - v) <= tol or abs(100 * h - v) <= tol or abs(h / 100 - v) <= tol for h in have):
            bad.append(raw)
    return bad


def _chart_ok(chart, result) -> dict | None:
    if not isinstance(chart, dict) or not isinstance(result, pd.DataFrame):
        return None
    if chart.get("type") in ("bar", "line") and chart.get("x") in result and chart.get("y") in result:
        return {"type": chart["type"], "x": chart["x"], "y": chart["y"]}
    return None


def auto_chart(result) -> tuple[dict | None, pd.DataFrame | None]:
    """Chart a small grouped result if the model did not ask for one."""
    if isinstance(result, pd.Series) and 2 <= len(result) <= 50 and pd.api.types.is_numeric_dtype(result):
        result = result.rename(result.name or "value").reset_index()
    if not isinstance(result, pd.DataFrame) or not 2 <= len(result) <= 50 or result.shape[1] != 2:
        return None, None
    x, y = result.columns
    if not pd.api.types.is_numeric_dtype(result[y]):
        return None, None
    kind = "line" if pd.api.types.is_datetime64_any_dtype(result[x]) else "bar"
    return {"type": kind, "x": x, "y": y}, result


def ask(question: str, df: pd.DataFrame, schema: str, notes: str = "") -> Answer:
    ans = Answer(question)
    prompt = _prompt(question, schema, notes)
    for attempt in (1, 2):
        ans.attempts = attempt
        try:
            ans.code = extract_code(chat(prompt, system=SYSTEM, tier="smart", max_tokens=2048))
        except LLMError as e:
            ans.error = f"AI unavailable: {e}"
            return ans
        try:
            reply = run_code(ans.code, df)
            if reply["ok"] and reply.get("result") is None and not reply.get("cannot_answer"):
                reply = {"ok": False, "error": "code ran but did not assign `result`"}
        except SandboxError as e:
            reply = {"ok": False, "error": str(e)}
        if reply["ok"]:
            break
        ans.error = reply["error"]
        prompt = (f"{_prompt(question, schema, notes)}\n\nYour previous code:\n```python\n{ans.code}\n```\n"
                  f"failed with: {reply['error']}\nFix it and reply with one corrected ```python block.")
    else:
        return ans

    ans.error = ""
    if reply.get("cannot_answer"):
        ans.narration = f"This dataset cannot answer that: {reply['cannot_answer']}"
        return ans
    ans.result, ans.method = reply["result"], str(reply.get("method") or "")
    ans.chart = _chart_ok(reply.get("chart"), ans.result)
    try:
        ans.narration = chat(
            f"Question: {question}\nMethod: {ans.method}\nComputed result:\n{result_text(ans.result)}\n\n"
            "Answer the question in 1-3 plain-English sentences for a business user. Use ONLY numbers that appear "
            "in the computed result (you may round). Do not calculate anything new. If the result looks "
            "insufficient to answer, say so.", system="You explain computed results. You never do arithmetic.",
            tier="fast", max_tokens=1024).strip()
        bad = unverified_numbers(ans.narration, ans.result, question)
        if bad:
            ans.warnings.append(f"The summary mentions number(s) not found in the computed result: {', '.join(bad)}. "
                                "Trust the table below.")
    except LLMError:
        ans.narration = "(Summary unavailable; see the computed result below.)"
    return ans


# ---------- answer cache + example questions ----------

_QA_DIR = Path(__file__).resolve().parent.parent / ".cache" / "qa"
_PREFETCH = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qa-prefetch")
_inflight: set[str] = set()
_lock = threading.Lock()


def _key(question: str, data_hash: str) -> str:
    q = re.sub(r"\s+", " ", question.strip().lower()).rstrip("?. ")
    return hashlib.sha256(f"{q}|{data_hash}".encode()).hexdigest()[:24]


def cached(question: str, data_hash: str) -> Answer | None:
    path = _QA_DIR / f"{_key(question, data_hash)}.pkl"
    if path.exists():
        try:
            return pickle.loads(path.read_bytes())
        except Exception:  # stale/corrupt cache entry: recompute
            return None
    return None


def cached_ask(question: str, df: pd.DataFrame, schema: str, notes: str, data_hash: str) -> tuple[Answer, bool]:
    """Answer from cache when the same question was asked on the same data; otherwise ask and cache successes."""
    hit = cached(question, data_hash)
    if hit is not None:
        return hit, True
    ans = ask(question, df, schema, notes)
    if not ans.error:
        _QA_DIR.mkdir(parents=True, exist_ok=True)
        (_QA_DIR / f"{_key(question, data_hash)}.pkl").write_bytes(pickle.dumps(ans))
    return ans, False


def prefetch(questions: list[str], df: pd.DataFrame, schema: str, notes: str, data_hash: str) -> int:
    """Answer example questions in the background so a click is instant. Returns how many were queued."""
    n = 0
    for q in questions:
        k = _key(q, data_hash)
        with _lock:
            if k in _inflight or cached(q, data_hash) is not None:
                continue
            _inflight.add(k)

        def job(q=q, k=k):
            try:
                cached_ask(q, df, schema, notes, data_hash)
            finally:
                with _lock:
                    _inflight.discard(k)

        _PREFETCH.submit(job)
        n += 1
    return n


def pending(questions: list[str], data_hash: str) -> int:
    with _lock:
        return sum(_key(q, data_hash) in _inflight for q in questions)


def example_questions(df: pd.DataFrame, roles: dict[str, str], entity_of: dict[str, str], pack=None) -> list[str]:
    """Six questions that this dataset can answer, built from column roles (and the domain pack when on)."""
    qs = []
    if pack is not None and pack.active:
        c = pack.cols
        done = " or ".join(pack.completed) or "completed"
        if c.get("priority"):
            qs.append(f"What percentage of {done} orders were on time ({c['actual']} <= {c['promised']}) by {c['priority']}?")
        qs.append(f"Which {c['entity']} has the largest average ({c['actual']} - {c['promised']}) for {done} orders?")
        qs.append(f"What is the average ({c['fee']} - {c['cost']}) per order by {c['entity']}?")
        if c.get("peak"):
            qs.append(f"What is the on-time percentage for {done} orders by {c['peak']}?")
        if c.get("status"):
            qs.append(f"How many orders are there per {c['status']}?")
    by = lambda *rs: [c for c, r in roles.items() if r in rs and c not in entity_of]
    groups = sorted([c for c in by("category", "location") if 2 <= df[c].nunique() <= 20], key=lambda c: df[c].nunique())
    money, nums = by("money"), by("money", "quantity", "duration", "rating", "measure")
    dates = sorted(by("date"), key=lambda c: df[c].nunique())
    flags = by("flag")
    if groups:
        qs.append(f"How many rows are there per {groups[0]}?")
    if money and groups:
        qs.append(f"What is the total {money[0]} by {groups[0]}?")
    if nums and len(groups) > 1:
        qs.append(f"Which {groups[1]} has the highest average {nums[-1]}?")
    if dates:
        d = to_datetime_clean(df[dates[0]]).dropna()
        span = (d.max() - d.min()).days if len(d) else 0
        per = "day" if span <= 31 else "week" if span <= 180 else "month" if span <= 730 else "year"
        qs.append(f"How many rows are there per {per} of {dates[0]}?" if not money or span > 730 else
                  f"What is the total {money[0]} per {per} of {dates[0]}?")
    if flags:
        top = df[flags[0]].dropna().astype("string").value_counts()
        if len(top):
            qs.append(f"What share of rows have {flags[0]} = {top.index[-1]}?")
    if money and len(groups) > 1:
        qs.append(f"What are the top 5 {groups[-1]} by total {money[0]}?")
    if nums:
        qs.append(f"What is the average and median {nums[0]}?")
    for g in groups:  # fallbacks so there are always enough
        for x in nums:
            qs.append(f"What is the average {x} by {g}?")
    if flags and groups:
        qs.append(f"What share of rows have each {flags[0]} value, by {groups[0]}?")
    qs.append("How many rows and columns does the data have, and which columns have missing values?")
    return list(dict.fromkeys(qs))[:6]
