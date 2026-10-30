import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("path", [ROOT / "app.py", *sorted((ROOT / "ui").glob("*.py")), *sorted((ROOT / "engine").rglob("*.py")),
                                  *sorted((ROOT / "scripts").glob("*.py"))],
                         ids=lambda p: p.name)
def test_compiles(path):
    py_compile.compile(str(path), doraise=True)
