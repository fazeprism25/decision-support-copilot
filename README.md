# Decision Support App

Upload any CSV/Excel file → quality checks, KPIs, charts, Q&A, what-if scenarios, recommendations.

## Setup and run: Windows
Open PowerShell in this folder and run once (needs Python 3.11+):
```
py -V:Astral/CPython3.11.15 -m venv .venv      # or: py -3.11 -m venv .venv (python.org install)
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```
Open `.env` in Notepad and paste your API key after `LLM_API_KEY=` (DeepSeek by default; other providers
are described inside the file). Then double-click `run.bat`, or run `.venv\Scripts\streamlit run app.py`.

## Setup and run: Mac
Needs Python 3.11+ (`python3 --version`; if older, `brew install python@3.11`). In Terminal, in this folder:
```
chmod +x run.sh
./run.sh
```
The first run installs everything and creates `.env`, then stops so you can paste your API key into it.
Run `./run.sh` again to start the app.

The browser opens at http://localhost:8501. Stop with Ctrl+C in the terminal.
Without a key the app still works: AI features switch off and rules take over.

## Run the tests
```
.venv\Scripts\python -m pytest -q        # Mac: .venv/bin/python -m pytest -q
```

## Demo
See `DEMO.md` for the 5-minute script and what to do if something breaks.
Sidebar buttons: **Reset everything** and **Load demo scenario**.

## Checks (no AI needed)
```
.venv\Scripts\python scripts\constraint_check.py      # typed constraint cards: pass/fail
.venv\Scripts\python scripts\make_synthetic.py        # rebuild data/retail_sales.csv, data/hr_employees.csv
.venv\Scripts\python scripts\measure_ai.py            # live AI timing table (uses ~9 API requests)
```

## Layout
- `app.py` – the screen (tabs, sidebar)
- `engine/` – generic calculations (no dataset-specific names allowed)
- `ui/` – tab rendering code
- `tests/` – automated checks
- `data/` – sample files. The competition files (`Round_1_BUDS.xlsx`, `BUDS_unknown_schema_test.csv`) are not in this repo; put them in `data/` to enable **Load demo data** and the tests that use them (those tests skip otherwise).
