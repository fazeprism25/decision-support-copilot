# 5-minute demo script

Before you start (once, 10 minutes before):
1. Double-click `run.bat`. The app opens at http://localhost:8501.
2. Sidebar: click **Test AI connection**. "Model replied: OK" = AI on. If it fails, carry on: everything below
   except new typed questions works without AI (the sidebar says **AI-down mode**).
3. Click **▶️ Load demo scenario** once, open each tab so answers are warmed up, then click **🔄 Reset everything**.

## The script

| Time | Do | Say |
|---|---|---|
| 0:00 | Sidebar → **▶️ Load demo scenario** | "One click loads the BUDS file, cleans it, standardises labels and applies a constraint card." |
| 0:20 | **Data** tab: scroll to *Column roles* and *Schema mapping report* | "Nothing is hard-coded. Each column's role is inferred and checked against the data. This works on files the app has never seen." |
| 0:50 | **Data** tab: *Category value mapping (C01)* | "Other departments write P1/P2/P3 or completed/rto. We map them to standard labels and you confirm before any tab uses them." |
| 1:10 | **Quality** tab | "A quality score with counts. Duplicates, negative distances and impossible ratings were fixed in one click, and each fix can be undone." |
| 1:40 | **KPIs** tab → *Delivery-ops pack* | "On-time %, promise gap and margin per order = fee − cost, never order value − cost. Hub load shows reported and calculated utilisation side by side; they disagree, so the file is a sample." |
| 2:20 | **What-if** tab | "The constraint 'Central hub is down for 3 days' runs the cascade: capacity → allocation → utilisation → time → on-time → cost → feasibility. Verdict: not feasible, with the reason." |
| 2:50 | Type a judge's card in **New constraint** (e.g. `Critical orders must reach 80% on time`) → **Interpret** → **✅ Apply** | "The card turns into checked parameters. I confirm before anything changes. Constraints stack." |
| 3:30 | **Recommendations** tab | "Rules over the computed numbers, ranked by impact. Each one shows its evidence and formula, before vs after the scenario, its assumptions, limitations and a confidence level. The AI only rewords and cannot add a number." |
| 4:10 | **Ask** tab → click an ⚡ example question | "The AI writes pandas code, we run it in a sandbox, and the code is shown. Example answers are ready in advance." |
| 4:40 | **Explain** tab, then **Change Log** → *Export CSV* | "Every formula, exclusion and assumption is generated from what actually ran. Every change is logged." |

Judges' own file: **Data** tab → *Upload a file* → choose it. Then follow the same tabs.

## If something breaks

| Problem | Fix |
|---|---|
| Sidebar says **AI-down mode** | Keep going. Everything except new typed questions still works. Typed constraint cards use the rule parser. After 60 s the app retries by itself, or click **Retry AI now**. |
| An AI step spins | It stops by itself after 20 s and the app falls back to rules. |
| A constraint is refused | Read the reason and the "Try:" suggestion, then retype it that way. Examples that always work: `<hub> is down for 3 days`, `capacity cut by 20% across all hubs`, `fee increase capped at 5%`, `ignore cancelled orders`, `<label> orders must reach 80% on time`, `<cost column> rises 10%`. |
| Wrong numbers after many changes | Sidebar → **🔄 Reset everything**, then **▶️ Load demo scenario**. |
| Screen frozen or error page | Press **R** in the browser (rerun). Still stuck: close the terminal, double-click `run.bat` again, then **Load demo scenario**. |
| A wrong column role | Data tab → change the **role** in the table. Every tab updates. |
| Uploaded file shows a title row as the header | Data tab → change **Header row**. |
