r"""Live AI timing for every AI step (cache bypassed). Uses ~9 real API requests.
Usage: .venv\Scripts\python scripts\measure_ai.py [data file]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import kpis as K  # noqa: E402
from engine import llm  # noqa: E402
from engine import recommend as R  # noqa: E402
from engine.config import get_settings  # noqa: E402
from engine.constraints import parse  # noqa: E402
from engine.domains import delivery as D  # noqa: E402
from engine.pipeline import prepare  # noqa: E402
from engine.qa import ask, schema_text  # noqa: E402
from engine.roles import detect_roles  # noqa: E402


def timed(name, fn):
    llm.reset()
    t, u0 = time.perf_counter(), llm.usage()
    try:
        out, status = fn(), "ok"
    except Exception as e:  # report, keep measuring the rest
        out, status = None, f"FAILED: {str(e)[:140]}"
    u1 = llm.usage()
    return name, time.perf_counter() - t, status, out, {k: u1[k] - u0[k] for k in u1}


def main(path):
    s = get_settings()
    print(f"provider={s.provider} fast={s.model_fast} smart={s.model_smart} timeout={s.timeout_s:g}s")
    stamp = str(time.time())  # unique cache key: forces real calls
    run = prepare(path)
    rows = []
    rows.append(timed("Connection test (fast)", lambda: llm.chat("Reply with exactly: OK", max_tokens=20)))
    rows.append(timed("Column roles (fast)", lambda: detect_roles(run.raw, "t" + stamp, use_ai=True, prof=run.profile)[1]
                      or "all columns labelled"))
    rows.append(timed("Value mapping proposal (fast)",
                      lambda: D.propose_mapping_ai(run.df, run.role_map, "t" + stamp)[0] and "proposed"))

    def kpi_pick():
        cols = [{"column": c, "role": r} for c, r in run.role_map.items()]
        return len(llm.chat_json(K.plan_prompt(cols), cache_key="t" + stamp).get("kpis", []))
    rows.append(timed("KPI picking (fast)", kpi_pick))

    def constraint():
        p = parse("Suppose the busiest site loses a third of its throughput for the next two days", run.df,
                  run.role_map, run.setup, run.profile, "t" + stamp, run.entity_of)
        if p.source != "AI" or not p.ok:
            raise RuntimeError(f"{p.source}: {p.reason} {p.errors}")
        return p.interpretation
    rows.append(timed("Constraint parse, AI path (smart)", constraint))
    rows.append(timed("Constraint parse, rule path (no AI)", lambda: parse(
        "Central hub is down for 3 days", run.df, run.role_map, run.setup, run.profile, "r", run.entity_of).interpretation))

    def qa():
        a = ask(f"What is the average {run.setup.cost or list(run.role_map)[0]} per {run.setup.entity or 'row'}? #{stamp}",
                run.df, schema_text(run.df, run.role_map, {}, run.profile))
        if a.error:
            raise RuntimeError(a.error)
        return a.narration
    rows.append(timed("Q&A: code + run + narration (smart+fast)", qa))

    def rewrite():
        recs = R.generate(R.Inputs(run.df, run.ctx, kpis=K.run_plan(run.plan, run.df, run.ctx), pack=run.pack,
                                   pack_metrics=D.metrics(run.df, run.pack, run.entity_of, run.role_map)
                                   if run.pack.active else None))
        msg = R.rewrite(recs, "t" + stamp)
        if "unavailable" in msg:
            raise RuntimeError(msg)
        return msg
    rows.append(timed("Recommendation wording (fast)", rewrite))
    print("\n| AI step | seconds | requests | input tok | (cached) | output tok | result |"
          "\n|---|---:|---:|---:|---:|---:|---|")
    for name, secs, status, _, u in rows:
        print(f"| {name} | {secs:.1f} | {u['requests']} | {u['input']} | {u['cached_input']} | {u['output']} | {status} |")
    tot = llm.usage()
    print(f"\nTotal: {tot['requests']} requests, {tot['input']} input tokens ({tot['cached_input']} cached), "
          f"{tot['output']} output tokens")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/Round_1_BUDS.xlsx")
