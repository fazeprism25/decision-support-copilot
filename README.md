# Decision Support App

Upload any CSV/Excel file → quality checks, KPIs, charts, Q&A, what-if scenarios, recommendations.

## Install and open the app (Windows or Mac)
1. Download this project: green **Code** button at the top of the GitHub page → **Download ZIP**, then unzip it
   (Windows: right-click → **Extract All**; Mac: double-click the ZIP).
2. Open the unzipped folder and double-click:
   - **Windows:** `START HERE - Windows.bat`
     If a blue "Windows protected your PC" box appears: click **More info** → **Run anyway**.
   - **Mac:** `START HERE - Mac.command`
     The first time, Mac may say it "cannot be opened": instead **right-click** the file → **Open** → **Open**.
3. The first time, it asks for an AI key: paste a DeepSeek key
   ([get one here](https://platform.deepseek.com/api_keys)) and press Enter. It checks the key with DeepSeek and
   asks again if it's wrong. No key? Just press Enter: the app works without AI.
4. Then it downloads Python and everything the app needs by itself (a few minutes, needs internet).
5. The app opens in your browser. Keep the black window open while you use it; close it to stop the app.

Next time, double-click the same file: it skips everything already done and opens the app in seconds.

**Add or change the AI key later:** double-click `Change AI Key - Windows.bat` (Mac: `Change AI Key - Mac.command`),
paste the key, then restart the app.

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
