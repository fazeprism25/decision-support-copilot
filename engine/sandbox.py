"""Static checks + isolated execution for LLM-written pandas code.
Layers: (1) AST allow/deny rules, (2) separate process with no imports and a tiny builtins set, (3) timeout."""
import ast
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

WORKER = Path(__file__).with_name("sandbox_worker.py")
ALLOWED_IMPORTS = {"pandas", "numpy"}
# the worker gets the environment minus anything secret-looking (API keys reach the app as environment variables)
_SECRET_VAR = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|^LLM_|^MODEL_", re.I)
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit", "memoryview", "type",
                "object", "super", "classmethod", "staticmethod", "property", "dir"}
BANNED_ATTRS = {"to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_hdf", "to_feather", "to_clipboard",
                "to_json", "to_html", "to_latex", "to_xml", "to_stata", "to_orc", "to_markdown", "load", "save",
                "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap",
                "DataSource", "io", "f2py", "ctypeslib", "lib", "testing", "os", "sys", "subprocess", "system",
                "popen", "plotting", "api", "options", "set_option"}


class SandboxError(Exception):
    pass


def check_code(code: str) -> tuple[str, list[str]]:
    """Return (code with pandas/numpy imports removed, list of violations)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return code, [f"syntax error: {e.msg} (line {e.lineno})"]
    errs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [m for m in mods if m.split(".")[0] not in ALLOWED_IMPORTS or "." in m]
            if bad or isinstance(node, ast.ImportFrom):
                errs.append(f"import not allowed: {', '.join(mods)} (pd and np are already available)")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errs.append(f"'{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                errs.append(f"private attribute '.{node.attr}' is not allowed")
            elif node.attr in BANNED_ATTRS or node.attr.startswith("read_"):
                errs.append(f"'.{node.attr}' is not allowed (no file, system or settings access)")
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("_"):
            errs.append(f"function name '{node.name}' is not allowed")
        # No classes: the worker's reply is unpickled here, so user code must not define custom pickling.
        elif isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            errs.append(f"{type(node).__name__} is not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            errs.append("strings containing '__' are not allowed")
    tree.body = [n for n in tree.body if not isinstance(n, ast.Import)]
    return ast.unparse(tree), sorted(set(errs))


def run_code(code: str, df: pd.DataFrame, timeout: float = 30) -> dict:
    """Validate and execute. Returns the worker reply dict; raises SandboxError on rule violations/timeouts."""
    clean, errs = check_code(code)
    if errs:
        raise SandboxError("Code rejected by safety check: " + "; ".join(errs))
    payload = pickle.dumps({"code": clean, "df": df})
    try:
        proc = subprocess.run([sys.executable, "-I", str(WORKER)], input=payload, capture_output=True,
                              timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                              env={k: v for k, v in os.environ.items() if not _SECRET_VAR.search(k)})
    except subprocess.TimeoutExpired:
        raise SandboxError(f"Code took longer than {timeout:.0f}s and was stopped.")
    if proc.returncode != 0 or not proc.stdout:
        raise SandboxError("Sandbox process failed: " + proc.stderr.decode(errors="replace")[-500:])
    return pickle.loads(proc.stdout)
