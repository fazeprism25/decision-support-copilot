# Project plan & status (read this first when resuming)

## Who / how
- User has NO coding background: plain English, exact commands, say what to click to verify. Keep replies short, save tokens (no re-reading known files, no full rewrites for small edits).
- Workflow: build phase(s) → tell user how to verify → wait for OK → `git commit`. User chose to do Phases 3-5 without stopping and test everything at the END (after Phase 8); Phases 3-5 were committed as a checkpoint, still untested by the user.
- Business-analytics competition final: Streamlit app presented live. Judges upload unseen data and give "constraint cards" (e.g. "Hub X down 3 days", "fee rise capped at 5%", "ignore cancelled orders").

## Non-negotiable rules
1. No hardcoded column names/KPIs/business terms in the generic engine (`engine/`). Everything from column ROLES inferred per upload. (Generic English name hints in `engine/roles.py` are allowed as rule fallbacks.)
2. LLM never does arithmetic. pandas computes every number. LLM only: labels roles, picks calculations, writes pandas code for questions, converts constraint text to validated params, narrates computed results.
3. Every number traceable: formula, filter, denominator.
4. Never guess: thin data / ambiguous column / unmappable constraint → say so and ask. Label metrics "calculated" or "reported"; model outputs "estimate based on assumptions".
5. Whole dataset stays in Python; LLM sees only column names, 5 sample rows, computed stats.
6. App works (profiling, KPIs, charts, sliders) if the API fails; AI features show a clear message.

## Stack / environment
- Python 3.11 venv at `.venv` (created with `py -V:Astral/CPython3.11.15 -m venv .venv`). pandas is 3.x (string dtype is "str", not object; use `pd.api.types` checks).
- Run: `run.bat` or `.venv\Scripts\streamlit run app.py`. Tests: `.venv\Scripts\python -m pytest -q`.
- LLM: provider-agnostic `engine/llm.py` (`chat`, `chat_json` with disk cache in `.cache/llm`, key includes dataset hash). `.env`: LLM_PROVIDER=openrouter, LLM_API_KEY, LLM_BASE_URL, MODEL_SMART=nvidia/nemotron-3-ultra-550b-a55b:free, MODEL_FAST=nvidia/nemotron-3.5-lightning:free, MODEL_FALLBACK=openrouter/free. OpenRouter calls send `reasoning.exclude=true` and max_tokens floor 1024. REASONING_FAST=off / REASONING_SMART=low in .env (thinking off cut role labelling from ~177s to ~10s). Timings on free models: roles ~10s, KPI plan 35-80s, Q&A 15-60s. Rate-limited; user may switch to Anthropic later via `.env` only.
- Test data: `data/Inventory_Stock.xlsx` (321 rows, 14 cols, 13 days; hidden issues: 'HUB-E' abbreviation, 1 duplicate record with a new ID, 'unknown' in a numeric column). `data/Round_1_BUDS.xlsx` (1 sheet, 2012 rows, 38 cols, header row 1, 14 days of dates). Synthetic look-alike in `tests/fixtures.py`.

## Architecture
- `app.py` – tabs + sidebar. `ui/*_tab.py` – one module per tab. `ui/state.py` – working-data pipeline (raw → quality fixes → scenario).
- `engine/loader.py` load CSV/Excel, header detection · `engine/profile.py` per-column stats, `to_numeric_clean` · `engine/roles.py` ROLES list, rule_role, check_role (stats veto AI labels), detect_roles, entity_attributes.
- Roles: date, id, category, money, quantity, duration, rating, flag, location, measure, text, ignore. Role table columns: column, role, source (AI / rules / rules (AI rejected) / user), check (ok/warn/fail), note, meaning, ...
- Entity attributes = columns constant per group (e.g. Hub_Daily_Capacity per Hub_ID); never sum them across rows — aggregate per entity.

## Phases
- [x] 0 Skeleton, .env, empty tabs (Data, Quality, KPIs, Ask, What-if, Recommendations, Explain, Change Log)
- [x] 1 Upload/loading (CSV/XLSX/XLS, sheet selector, header-row detection, 100k rows ≈ 7s)
- [x] 2 Column roles (AI label + stats check, editable table, entity attributes)
- [x] 3 Data quality: duplicate IDs, missing, negatives in money/distance, zero durations, ratings outside scale, unparseable dates, inconsistent text, outliers. Score, per-issue counts, one-click fixes with before/after. Expected on BUDS: 12 duplicate Order_IDs, 15 missing Zone, 10 missing Hub, a distance of -1, ratings up to 6, 0-minute cycle times.
- [x] 4 KPI + chart engine: tested role-driven calc library (totals, averages, growth, share, top/bottom N, trend, margin = revenue - cost, on-time rate = actual <= promised among completed, distribution, segment breakdown). LLM picks which apply; code computes; plotly renders. Every KPI card has explain popover (formula, denominator, exclusions). Warn if date range < 14 days.
- [x] 5 NL Q&A: LLM writes pandas code → restricted sandbox (no imports beyond pandas/numpy, no file/network, timeout) → retry once with error → show answer, table/chart AND code. Never answer numerically without executing code.
- [ ] 6 Scenario engine: one Params object (pct/abs change to any measure, filters/exclusions, capacity change/outage for an entity, price/fee change with optional cap, target thresholds). Sidebar sliders auto-generated from roles. "New constraint" text box → LLM → JSON validated by pydantic against allowed schema → show parsed params for confirmation → apply; unmappable → refusal + suggestion. Baseline vs scenario side by side. Change Log tab records each constraint, params, KPI movement; CSV export.
- [ ] 7 Recommendations (rule-generated from computed numbers: evidence, assumptions, limitations, confidence; LLM only rewrites in plain language) + Explain tab (all formulas, assumptions, exclusions, limitations) + delivery-ops domain pack (activates only if promised time, actual time, fee, cost, entity/hub roles exist): on-time % overall and by priority/peak/hub, promise gap, margin per order, hub load model wait ~ 1/(1-utilisation) capped at 0.95, calibrated from data, labelled assumption. Reported utilisation may disagree with orders/capacity: show both, label them.
- [ ] 8 Hardening: pytest per calculation; run on BUDS + two synthetic unrelated datasets (retail sales, HR) and fix; one-click export report (HTML/PDF); saved-session reload for demo safety.

## Definition of done
Upload an unseen file → correct roles, quality issues, KPIs, charts, Q&A with no code changes; apply 3 different constraints via text box → baseline vs scenario + change log; every figure explainable.

## Status log
- Phase 0-2 committed.
- Phases 3-5 done (checkpoint commit). Modules: `engine/quality.py` (Issue/Fix, find_issues, apply_fix, quality_score = % rows with no medium/high issue), `engine/kpis.py` (CATALOG of calcs, validate/normalize/run_plan/default_plan, plan_prompt; entity attributes aggregated once per entity and marked 'reported'), `engine/sandbox.py` + `engine/sandbox_worker.py` (AST checks + separate `python -I` process + timeout; classes banned because reply is unpickled), `engine/qa.py` (code → sandbox → retry once → narration; `unverified_numbers` flags narrated numbers absent from the result). UI: `ui/state.py` (clean_df pipeline, role_map, kpi_ctx), `ui/quality_tab.py`, `ui/kpi_tab.py` (explain popover via explain_md, KPI builder), `ui/ask_tab.py`, `ui/charts.py` (plotly, series colour #2a78d6, no dual axes). 34 pytest tests pass.
- Known: free model sometimes picks margin = Order_Value − cost (mixes bases) despite prompt rule; user can remove the card; Phase 7 domain pack should compute fee − cost explicitly. AI sometimes labels hour-of-day as duration.
- Gotcha: when editing files via Python heredocs, always use encoding='utf-8' (cp1252 default silently broke a replace once). Known nit: rule fallback labels "Order_Hour" as duration (hour-of-day); AI usually corrects.

## Decisions (2026-10-31)
- Provider: stay on OpenRouter free models until results demand otherwise (user decision). Tiers: MODEL_FAST = role labelling, KPI picking, value mapping, narration; MODEL_SMART = Q&A code and constraint parsing only. Cache every AI result by dataset hash.
- Consistency is structural: KPIs, Q&A, What-if all run on ONE shared cleaned dataframe (`ui/state.clean_df()`: raw → user's quality fixes). No hidden validity rules inside individual KPIs; invalid values are handled by quality fixes ("Apply recommended fixes" one click). Q&A shows "running on cleaned data: N rows, X excluded" above every answer. Test: Q&A on-time % == KPI on-time %.
- Acceptance file `data/BUDS_unknown_schema_test.csv` (= user's "new data"): after recommended fixes → 2,000 orders, 1,817 valid delivered, 20.7% on time, margin -5.50 per order.

## Constraint cards
- **C01 – The Unknown Schema Attack (Data engineering):** BUDS receives a new operational dataset from a department that uses different column names, formats, and naming conventions. The app was developed on a different dataset.
  Build: (done) role inference, unit/currency/day-first/yes-no parsing. Add: "Schema mapping report" (original column → role → how parsed → confidence, plus unmapped columns) and an editable category value mapping table (e.g. P1/P2/P3 → Critical/Express/Standard, completed → Delivered, rto → Returned) that the user confirms. Canonical labels come from the domain pack (Phase 7), never from the generic engine.
- **C10 – The Live What-If Cascade (Scenario modelling):** One operational change may affect several downstream decisions. Reduced hub capacity could affect allocation, estimated cost, utilisation and plan feasibility.
  Build in Phase 6: (1) dependency chain capacity → allocation across entities → utilisation → estimated wait/cycle time → on-time % → cost and margin → feasibility; any change recomputes every later step. (2) Allocation: when an entity's capacity drops below its load, move the excess to entities with spare capacity (lowest utilisation first, none pushed above a user-set cap, default 85%); orders that fit nowhere are "unserved". (3) Moved orders get extra cost and time per order from an editable, labelled assumption. (4) Cascade view: one row per step with baseline, scenario, change and formula, plus a feasibility verdict with reason. (5) Constraints stack; each logs to the Change Log with downstream KPI changes.
  Test: cutting the busiest hub's capacity by 30% changes allocation, utilisation, on-time %, cost and feasibility together, and is logged.
  Design note: BUDS rows are a sample (≈44 orders/day vs capacity 720-1000/day → calculated utilisation ≈5%) while reported utilisation is 71-96%. Cascade load basis is selectable: "reported utilisation × capacity" (default when a reported-utilisation column exists) or "observed rows per day"; both shown and labelled.
- Gotcha: Git Bash heredocs turn \n inside Python strings into real newlines. Write edit scripts with the Write tool (scratchpad), never heredocs, when they contain escapes.
