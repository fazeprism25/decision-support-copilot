"""Per-column statistics computed by code. These are the only facts about the data the LLM ever sees."""
import datetime as dt
import re

import numpy as np
import pandas as pd

SAMPLE = 5000
CURRENCY = "₹$€£¥"
_NUM_JUNK = re.compile(rf"[,\s{CURRENCY}%]|(?:rs|inr|usd|eur|gbp)\.?", flags=re.I)
# a number optionally followed by a short unit word, e.g. "32 min", "4.6km", "3 hrs"
_NUM_UNIT = re.compile(r"^([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)([a-z]{0,5})\.?$", flags=re.I)
_DMY = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})")
YES = {"y", "yes", "true", "t", "1"}
NO = {"n", "no", "false", "f", "0"}


def to_numeric_clean(s: pd.Series) -> pd.Series:
    """Parse numbers, tolerating thousands separators, currency symbols/codes, % signs and unit suffixes."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype("string").str.replace(_NUM_JUNK, "", regex=True)
    return pd.to_numeric(t.str.extract(_NUM_UNIT, expand=False)[0], errors="coerce")


def unit_suffix(s: pd.Series) -> str:
    """Most common unit word after numbers ('min', 'km'), or ''."""
    t = s.dropna().astype("string").str.replace(_NUM_JUNK, "", regex=True).str.extract(_NUM_UNIT)[1].dropna()
    t = t[t != ""]
    return str(t.str.lower().mode().iloc[0]) if len(t) > 0.5 * max(s.notna().sum(), 1) else ""


def day_first(s: pd.Series) -> bool | None:
    """For dd/mm vs mm/dd strings: decide from values > 12, else pick the reading with the tighter date range.
    None when the values are not in that form."""
    t = s.dropna().astype("string")
    m = t.str.extract(_DMY).dropna().astype(int)
    if len(m) < 0.5 * max(len(t), 1) or not len(m):
        return None
    if m[0].max() > 12:
        return True
    if m[1].max() > 12:
        return False
    samp = t.sample(min(2000, len(t)), random_state=0)
    spans = {}
    for df_ in (True, False):
        d = pd.to_datetime(samp, errors="coerce", dayfirst=df_, format="mixed")
        spans[df_] = (d.max() - d.min()) if d.notna().any() else pd.Timedelta.max
    return spans[True] <= spans[False]


def to_datetime_clean(s: pd.Series, dayfirst: bool | None = None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.Series(pd.NaT, index=s.index)
    if dayfirst is None:
        dayfirst = bool(day_first(s))
    return pd.to_datetime(s.astype("string"), errors="coerce", format="mixed", dayfirst=dayfirst)


def yes_no(v) -> str | None:
    k = str(v).strip().lower()
    return "yes" if k in YES else "no" if k in NO else None


def _rate(parsed: pd.Series, nonnull: pd.Series) -> float:
    return float(parsed.notna().sum() / len(nonnull)) if len(nonnull) else 0.0


def profile_column(s: pd.Series) -> dict:
    n = len(s)
    nonnull = s.dropna()
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        nonnull = nonnull[nonnull.astype("string").str.strip() != ""]
    samp = nonnull.sample(min(SAMPLE, len(nonnull)), random_state=0) if len(nonnull) else nonnull
    p = {
        "dtype": str(s.dtype),
        "n": n,
        "missing": int(n - len(nonnull)),
        "missing_pct": round(100 * (n - len(nonnull)) / n, 1) if n else 0.0,
        "n_unique": int(nonnull.nunique()),
        "unique_ratio": round(nonnull.nunique() / len(nonnull), 3) if len(nonnull) else 0.0,
        "examples": [str(v)[:40] for v in nonnull.drop_duplicates().head(5).tolist()],
    }
    is_time = len(samp) and samp.map(lambda v: isinstance(v, dt.time)).mean() > 0.8
    num = to_numeric_clean(samp)
    p["numeric_rate"] = 0.0 if is_time else round(_rate(num, samp), 3)
    if pd.api.types.is_bool_dtype(s) or p["numeric_rate"] >= 0.9:
        p["date_rate"] = 0.0
    elif is_time:
        p["date_rate"] = 1.0
        p["time_of_day"] = True
    else:
        p["date_rate"] = round(_rate(to_datetime_clean(samp), samp), 3)

    if p["numeric_rate"] >= 0.9:
        full = to_numeric_clean(nonnull).dropna()
        p.update({
            "min": float(full.min()), "max": float(full.max()),
            "mean": round(float(full.mean()), 4), "median": float(full.median()),
            "integer_like": bool(np.all(np.isclose(full % 1, 0))),
            "share_negative": round(float((full < 0).mean()), 4),
            "share_zero": round(float((full == 0).mean()), 4),
        })
        if not pd.api.types.is_numeric_dtype(s):
            txt = samp.astype("string")
            p["unit"] = unit_suffix(samp)
            p["currency_symbol"] = bool(txt.str.contains(rf"[{CURRENCY}]|(?:rs|inr|usd|eur|gbp)", case=False).mean() > 0.5)
            p["percent_sign"] = bool(txt.str.contains("%").mean() > 0.5)
    elif p["date_rate"] >= 0.8 and not is_time:
        dayf = day_first(nonnull) if not pd.api.types.is_datetime64_any_dtype(s) else None
        d = to_datetime_clean(nonnull, dayf).dropna()
        p.update({"min": str(d.min()), "max": str(d.max()),
                  "span_days": int((d.max() - d.min()).days)})
        if dayf is not None:
            p["date_order"] = "day/month" if dayf else "month/day"
    else:
        txt = nonnull.astype("string")
        p["avg_len"] = round(float(txt.str.len().mean()), 1) if len(txt) else 0.0
        vals = set(txt.str.strip().str.lower().unique()) if p["n_unique"] <= 6 else set()
        p["yes_no_like"] = bool(vals) and all(yes_no(v) for v in vals)
        p["n_unique_normalized"] = int(txt.str.strip().str.lower().nunique())
        p["top_values"] = {str(k)[:40]: int(v) for k, v in txt.value_counts().head(5).items()}
    return p


def profile_frame(df: pd.DataFrame) -> dict[str, dict]:
    return {c: profile_column(df[c]) for c in df.columns}


def sample_rows(df: pd.DataFrame, k: int = 5) -> list[dict]:
    rows = df.sample(min(k, len(df)), random_state=0)
    return [{c: (None if pd.isna(v) else str(v)[:60]) for c, v in r.items()} for _, r in rows.iterrows()]
