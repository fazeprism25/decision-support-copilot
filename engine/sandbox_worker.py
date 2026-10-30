"""Runs one already-validated code snippet in a separate, isolated Python process.
Reads a pickle {code, df} on stdin, writes a pickle {ok, result, error, printed} on stdout.
Must not import anything from the project."""
import builtins
import io
import pickle
import sys
import traceback

import numpy as np
import pandas as pd

SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "int",
    "isinstance", "len", "list", "map", "max", "min", "pow", "print", "range", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip", "ValueError", "KeyError", "TypeError", "ZeroDivisionError",
    "Exception", "True", "False", "None",
) if hasattr(builtins, n)}


def main():
    job = pickle.load(sys.stdin.buffer)
    out = io.StringIO()
    env = {"__builtins__": {**SAFE_BUILTINS, "print": lambda *a, **k: print(*a, file=out)},
           "pd": pd, "np": np, "df": job["df"]}
    try:
        exec(compile(job["code"], "<question>", "exec"), env)
        res = env.get("result")
        if isinstance(res, pd.DataFrame) and len(res) > 1000:
            res = res.head(1000)
        elif isinstance(res, pd.Series) and len(res) > 1000:
            res = res.head(1000)
        elif isinstance(res, np.generic):
            res = res.item()
        reply = {"ok": True, "result": res, "chart": env.get("chart"), "method": env.get("method"),
                 "cannot_answer": env.get("cannot_answer"), "printed": out.getvalue()[-2000:]}
    except Exception as e:
        tb = traceback.format_exception_only(type(e), e)[-1].strip()
        reply = {"ok": False, "error": tb, "printed": out.getvalue()[-2000:]}
    try:
        data = pickle.dumps(reply)
    except Exception as e:
        data = pickle.dumps({"ok": False, "error": f"result could not be returned: {e}"})
    sys.stdout.buffer.write(data)


if __name__ == "__main__":
    main()
