"""C01 category value mapping: {column: {raw value: standard label}}. Generic mechanics only; the standard labels
themselves come from a domain pack (engine/domains) or from the user. Applied after quality fixes, so every tab
(KPIs, Ask, What-if, Recommendations) sees the same labels once the user confirms the table."""
import pandas as pd

VMap = dict[str, dict[str, str]]


def apply_value_map(df: pd.DataFrame, vmap: VMap) -> pd.DataFrame:
    if not vmap:
        return df
    out = df.copy()
    for col, m in vmap.items():
        m = {k: v for k, v in m.items() if v not in (None, "") and k != v}
        if col in out and m:
            s = out[col]
            out[col] = s.astype("string").replace(m).where(s.notna(), s)
    return out


def to_table(vmap: VMap, df: pd.DataFrame, source: dict[tuple[str, str], str] | None = None) -> pd.DataFrame:
    rows = []
    for col, m in vmap.items():
        counts = df[col].astype("string").value_counts() if col in df else pd.Series(dtype=int)
        for raw, lab in m.items():
            rows.append({"column": col, "original value": raw, "standard label": lab,
                         "rows": int(counts.get(raw, 0)), "source": (source or {}).get((col, raw), "")})
    return pd.DataFrame(rows, columns=["column", "original value", "standard label", "rows", "source"])


def from_table(t: pd.DataFrame) -> VMap:
    out: VMap = {}
    for r in t.itertuples(index=False):
        lab = r[2]
        if isinstance(lab, str) and lab.strip():
            out.setdefault(str(r[0]), {})[str(r[1])] = lab.strip()
    return out


def map_value(col: str | None, v, vmap: VMap):
    return vmap.get(col or "", {}).get(str(v), v) if v is not None else v


def remap_plan(plan: list[dict], vmap: VMap) -> list[dict]:
    """Translate label parameters in a KPI plan (share value, completed values) to the confirmed labels."""
    out = []
    for it in plan:
        p = dict(it.get("params") or {})
        if "value" in p and p.get("col"):
            p["value"] = map_value(p["col"], p["value"], vmap)
        if p.get("completed_values") and p.get("status"):
            p["completed_values"] = list(dict.fromkeys(map_value(p["status"], v, vmap) for v in p["completed_values"]))
        out.append({**it, "params": p})
    return out


def remap_values(col: str | None, values, vmap: VMap):
    if isinstance(values, list):
        return list(dict.fromkeys(map_value(col, v, vmap) for v in values))
    return map_value(col, values, vmap)
