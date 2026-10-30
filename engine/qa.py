"""Natural-language Q&A: LLM writes pandas code → sandbox executes → LLM narrates only the computed result."""
import json
import re
from dataclasses import dataclass, field

import pandas as pd

from engine.llm import LLMError, chat
from engine.profile import sample_rows
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
        if "top_values" in p and p.get("n_unique", 99) <= 25:
            d["labels"] = list(p["top_values"]) + (["..."] if p["n_unique"] > 5 else [])
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
