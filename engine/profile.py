"""Per-column statistics computed by code. These are the only facts about the data the LLM ever sees."""
import datetime as dt
import re

import numpy as np
import pandas as pd

SAMPLE = 5000
_NUM_JUNK = re.compile(r"[,\s₹$€£%]")


def to_numeric_clean(s: pd.Series) -> pd.Series:
    """Parse numbers, tolerating thousands separators, currency symbols and % signs."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype("string").str.replace(_NUM_JUNK, "", regex=True), errors="coerce")


def to_datetime_clean(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        return pd.Series(pd.NaT, index=s.index)
    return pd.to_datetime(s.astype("string"), errors="coerce", format="mixed")


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
    elif p["date_rate"] >= 0.8 and not is_time:
        d = to_datetime_clean(nonnull).dropna()
        p.update({"min": str(d.min()), "max": str(d.max()),
                  "span_days": int((d.max() - d.min()).days)})
    else:
        txt = nonnull.astype("string")
        p["avg_len"] = round(float(txt.str.len().mean()), 1) if len(txt) else 0.0
        p["n_unique_normalized"] = int(txt.str.strip().str.lower().nunique())
        p["top_values"] = {str(k)[:40]: int(v) for k, v in txt.value_counts().head(5).items()}
    return p


def profile_frame(df: pd.DataFrame) -> dict[str, dict]:
    return {c: profile_column(df[c]) for c in df.columns}


def sample_rows(df: pd.DataFrame, k: int = 5) -> list[dict]:
    rows = df.sample(min(k, len(df)), random_state=0)
    return [{c: (None if pd.isna(v) else str(v)[:60]) for c, v in r.items()} for _, r in rows.iterrows()]
