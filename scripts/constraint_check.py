r"""Typed-constraint test set: runs the demo constraint cards through the parser and prints pass/fail.
Usage: .venv\Scripts\python scripts\constraint_check.py [--ai] [data file ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.constraints import parse  # noqa: E402
from engine.pipeline import prepare  # noqa: E402

# (card, expected: "apply" = must map to valid parameters, "refuse" = must refuse with reason + suggestion)
CARDS = [
    ("Central hub is down for 3 days", "apply"),
    ("Hub capacity cut by 20% across all hubs", "apply"),
    ("Delivery fee increase capped at 5%", "apply"),
    ("Ignore cancelled and returned orders", "apply"),
    ("Critical orders must reach 80% on time", "apply"),
    ("Rider costs rise 10%", "apply"),
    ("Show results by zone instead of hub", "refuse"),
    # route cost card
    ("Tolls of Rs 40 per order on East routes, unverified", "apply"),
    ("Include waiting cost at Rs 2 per minute", "apply"),
    ("Parking is confirmed at Rs 15 per order", "apply"),
    ("Exclude unverified costs", "apply"),
    ("Tolls of Rs 40 per order on Mars routes", "refuse"),
]


def check(path, use_ai=False):
    run = prepare(path)
    rows = []
    for text, expect in CARDS:
        p = parse(text, run.df, run.role_map, run.setup, run.profile, f"check:{run.name}:{text}",
                  run.entity_of, use_ai=use_ai)
        if expect == "apply":
            ok = p.ok
            detail = (p.interpretation + "  " + p.params.model_dump_json(exclude_defaults=True)) if p.ok else \
                f"{p.reason} {p.errors} -> try: {p.suggestion}"
        else:
            ok = (not p.ok) and bool(p.reason) and bool(p.suggestion)
            detail = f"refused: {p.reason} -> try: {p.suggestion}"
        rows.append((text, expect, "PASS" if ok else "FAIL", p.source, detail))
    return rows


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--ai"]
    files = args or ["data/Round_1_BUDS.xlsx", "data/BUDS_unknown_schema_test.csv"]
    for f in files:
        print(f"\n== {f}")
        for text, expect, res, src, detail in check(f, "--ai" in sys.argv):
            print(f"{res}  [{src}] {text!r} (expect {expect})\n      {detail}")
