#!/bin/bash
# Mac/Linux launcher (the run.bat equivalent). First run creates .venv and installs packages.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  PY=$(command -v python3.11 || command -v python3.12 || command -v python3.13 || command -v python3)
  "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "Python 3.11+ needed: brew install python@3.11"; exit 1; }
  "$PY" -m venv .venv && .venv/bin/python -m pip install -r requirements.txt || exit 1
fi
[ -f .env ] || { cp .env.example .env; echo "Created .env - paste your API key into it, then run this again."; open -e .env 2>/dev/null; exit 1; }
.venv/bin/streamlit run app.py
