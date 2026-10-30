# Decision Support App

Upload any CSV/Excel file → quality checks, KPIs, charts, Q&A, what-if scenarios, recommendations.

## First-time setup (once)
Open PowerShell in this folder and run:
```
py -V:Astral/CPython3.11.15 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```
Then open `.env` in Notepad and paste your OpenRouter key after `LLM_API_KEY=`.
To switch provider/model later, edit only `.env` (instructions inside it) and restart the app.

## Run the app
Double-click `run.bat`, or in PowerShell:
```
.venv\Scripts\streamlit run app.py
```
The browser opens at http://localhost:8501. Stop with Ctrl+C in the terminal.

## Run the tests
```
.venv\Scripts\python -m pytest -q
```

## Layout
- `app.py` – the screen (tabs, sidebar)
- `engine/` – generic calculations (no dataset-specific names allowed)
- `ui/` – tab rendering code
- `tests/` – automated checks
- `data/` – put sample files here (e.g. `Round_1_BUDS.xlsx`)
