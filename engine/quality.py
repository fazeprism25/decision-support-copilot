"""Role-driven data-quality checks and replayable fixes. Every issue carries its rule, count and denominator."""
import difflib
import re
from dataclasses import asdict, dataclass, field

import pandas as pd

from engine.profile import to_datetime_clean, to_numeric_clean, yes_no

MEASURE_ROLES = {"money", "quantity", "duration", "measure"}
GROUP_ROLES = {"category", "location", "flag"}
STANDARD_SCALES = [(1, 5), (0, 5), (1, 7), (1, 10), (0, 10), (0, 100)]


@dataclass
class Issue:
    key: str  # stable id: rule:column
    rule: str
    column: str
    count: int  # affected rows
    denominator: int  # rows checked
    detail: str
    fixes: list[str]  # allowed actions
    params: dict = field(default_factory=dict)
    severity: str = "medium"

    @property
    def pct(self) -> float:
        return round(100 * self.count / self.denominator, 2) if self.denominator else 0.0


@dataclass
class Fix:
    key: str
    rule: str
    column: str
    action: str
    params: dict
    rows_before: int = 0
    rows_after: int = 0
    cells_changed: int = 0

    def to_dict(self):
        return asdict(self)


FIX_LABELS = {
    "drop_duplicates": "Drop repeated rows (keep first)",
    "drop_rows": "Exclude affected rows",
    "set_missing": "Set affected values to missing",
    "fill_unknown": "Fill missing with 'Unknown'",
    "fill_median": "Fill missing with median",
    "cap": "Cap at the limit",
    "standardize": "Merge spelling variants into the most common form",
}


def rating_scale(name: str, values: pd.Series) -> tuple[float, float, str]:
    """Scale from the column name ('1to5', '1-10', 'out of 5') or the smallest standard scale holding 99% of values."""
    m = re.search(r"(\d+)\s*(?:to|-|_)\s*(\d+)", name)
    if m and float(m.group(1)) < float(m.group(2)) <= 100:
        return float(m.group(1)), float(m.group(2)), f"scale {m.group(1)}-{m.group(2)} taken from the column name"
    m = re.search(r"(?:out_?of|of)\s*_?(\d+)", name, flags=re.I)
    if m:
        return 1.0, float(m.group(1)), f"scale 1-{m.group(1)} taken from the column name"
    v = values.dropna()
    for lo, hi in STANDARD_SCALES:
        if len(v) and ((v >= lo) & (v <= hi)).mean() >= 0.99:
            return float(lo), float(hi), f"scale {lo}-{hi}: smallest standard scale holding at least 99% of values"
    return float(v.min()), float(v.max()), "no standard scale fits; observed range used"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def spelling_groups(values: pd.Series, yes_no_flag: bool = False) -> dict[str, str]:
    """Map each variant to the most common spelling of its group (same after normalising, or >= 0.9 similar).
    Values whose digits differ are never merged ('16 min' vs '116 min'). For yes/no flags, synonyms merge."""
    counts = values.dropna().astype("string").value_counts()
    if len(counts) > 300:
        return {}
    canon: dict[str, str] = {}
    reps: list[str] = []  # most common first
    if yes_no_flag:
        for v in counts.index:
            first = next((r for r in reps if yes_no(r) == yes_no(v)), None)
            if first is None:
                reps.append(v)
            else:
                canon[v] = first
        return canon
    digits = lambda x: re.sub(r"\D", "", str(x))
    for v in counts.index:
        n = _norm(v)
        if digits(v):
            match = next((r for r in reps if _norm(r) == n), None)
            if match is None:
                reps.append(v)
            elif v != match:
                canon[v] = match
            continue
        match = next((r for r in reps if _norm(r) == n or
                      (len(n) > 3 and difflib.SequenceMatcher(None, _norm(r), n).ratio() >= 0.9) or
                      # rare abbreviation of a common label, e.g. 'ABC-E' next to 'ABC-EAST'
                      (len(n) >= 3 and _norm(r).startswith(n) and counts[v] < 0.05 * counts[r])), None)
        if match is None:
            reps.append(v)
        elif v != match:
            canon[v] = match
    return canon


def _iqr_fences(x: pd.Series, k: float = 3.0) -> tuple[float, float]:
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def find_issues(df: pd.DataFrame, roles: pd.DataFrame, entity_cols: set[str] = frozenset()) -> list[Issue]:
    n = len(df)
    role_of = {c: r for c, r, chk in zip(roles["column"], roles["role"], roles["check"]) if chk != "fail" and c in df}
    issues: list[Issue] = []

    for c, r in role_of.items():
        s = df[c]
        if r == "id" and c not in entity_cols:
            dup = int((s.duplicated(keep="first") & s.notna()).sum())
            if dup > 0:
                issues.append(Issue(f"duplicate_id:{c}", "duplicate_id", c, dup, n,
                                    f"{dup} rows repeat an ID already seen earlier in '{c}'",
                                    ["drop_duplicates"], severity="high"))

        if r != "ignore":
            miss = int(s.isna().sum() + (s.astype("string").str.strip() == "").sum())
            if miss:
                # least destructive option first: labels get 'Unknown', numbers are excluded rather than invented
                fixes = (["fill_unknown"] if r in GROUP_ROLES | {"text"} else []) + ["drop_rows"] + \
                        (["fill_median"] if r in MEASURE_ROLES | {"rating"} else [])
                sev = "high" if r in ("id", "date") else "low" if r == "text" else "medium"
                issues.append(Issue(f"missing:{c}", "missing", c, miss, n, f"{miss} rows have no value in '{c}'",
                                    fixes, severity=sev))

        if r in MEASURE_ROLES | {"rating"}:
            x = to_numeric_clean(s)
            bad = int((x.isna() & s.notna()).sum())
            if bad:
                issues.append(Issue(f"not_numeric:{c}", "not_numeric", c, bad, int(s.notna().sum()),
                                    f"{bad} values in '{c}' are not numbers", ["set_missing", "drop_rows"]))
            valid = int(x.notna().sum())
            neg = int((x < 0).sum())
            if r in MEASURE_ROLES and neg and neg / max(valid, 1) < 0.05:
                issues.append(Issue(f"negative:{c}", "negative", c, neg, valid,
                                    f"{neg} negative values in '{c}' (min {x.min():g}); "
                                    "negatives are rare here so they look like errors", ["set_missing", "drop_rows"],
                                    severity="high"))
            if r == "duration":
                zero = int((x == 0).sum())
                if zero:
                    issues.append(Issue(f"zero_duration:{c}", "zero_duration", c, zero, valid,
                                        f"{zero} rows have a duration of 0 in '{c}'", ["set_missing", "drop_rows"]))
            if r == "rating":
                lo, hi, why = rating_scale(c, x)
                out = int(((x < lo) | (x > hi)).sum())
                if out:
                    issues.append(Issue(f"out_of_scale:{c}", "out_of_scale", c, out, valid,
                                        f"{out} ratings outside {lo:g}-{hi:g} ({why}); observed max {x.max():g}",
                                        ["set_missing", "cap", "drop_rows"], {"lo": lo, "hi": hi}, "high"))
            elif r in MEASURE_ROLES and c not in entity_cols and x.nunique() > 10:
                lo, hi = _iqr_fences(x.dropna())
                out = int(((x < lo) | (x > hi)).sum())
                if out:
                    issues.append(Issue(f"outlier:{c}", "outlier", c, out, valid,
                                        f"{out} extreme values in '{c}' outside [{lo:g}, {hi:g}] "
                                        "(3 × IQR beyond the quartiles)", ["cap", "set_missing", "drop_rows"],
                                        {"lo": lo, "hi": hi}, "low"))

        if r == "date":
            d = to_datetime_clean(s)
            bad = int((d.isna() & s.notna()).sum())
            if bad:
                issues.append(Issue(f"bad_date:{c}", "bad_date", c, bad, int(s.notna().sum()),
                                    f"{bad} values in '{c}' cannot be read as dates", ["set_missing", "drop_rows"],
                                    severity="high"))

        if r in GROUP_ROLES:
            canon = spelling_groups(s, yes_no_flag=r == "flag" and s.dropna().map(yes_no).notna().all())
            if canon:
                rows = int(s.isin(list(canon)).sum())
                ex = "; ".join(f"'{a}' → '{b}'" for a, b in list(canon.items())[:4])
                issues.append(Issue(f"inconsistent_text:{c}", "inconsistent_text", c, rows, int(s.notna().sum()),
                                    f"{len(canon)} spelling/casing variants in '{c}' ({ex})", ["standardize"],
                                    {"mapping": canon}))
    ids = [c for c, r in role_of.items() if r == "id"]
    rest = [c for c in df.columns if c not in ids]
    if ids and rest:
        dup = int(df.duplicated(subset=rest, keep="first").sum())
        if dup:
            issues.append(Issue("duplicate_record:*", "duplicate_record", ", ".join(ids), dup, n,
                                f"{dup} rows are identical to an earlier row in every column except the ID",
                                ["drop_duplicates"], {"subset": rest}))
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(issues, key=lambda i: (order[i.severity], -i.count))


def _affected_mask(df: pd.DataFrame, rule: str, c: str, params: dict) -> pd.Series:
    if rule == "duplicate_record":
        return df.duplicated(subset=[x for x in params["subset"] if x in df], keep="first")
    s = df[c]
    if rule == "missing":
        return s.isna() | (s.astype("string").str.strip() == "")
    if rule == "duplicate_id":
        return s.duplicated(keep="first") & s.notna()
    if rule == "bad_date":
        return to_datetime_clean(s).isna() & s.notna()
    x = to_numeric_clean(s)
    if rule == "not_numeric":
        return x.isna() & s.notna()
    if rule == "negative":
        return x < 0
    if rule == "zero_duration":
        return x == 0
    if rule in ("out_of_scale", "outlier"):
        return (x < params["lo"]) | (x > params["hi"])
    if rule == "inconsistent_text":
        return s.isin(list(params["mapping"]))
    raise ValueError(rule)


def apply_fix(df: pd.DataFrame, fix: Fix) -> pd.DataFrame:
    """Apply one fix and fill in its before/after counts. Returns a new frame."""
    df = df.copy()
    c, a = fix.column, fix.action
    mask = _affected_mask(df, fix.rule, c, fix.params).fillna(False).astype(bool)
    fix.rows_before = len(df)
    if a in ("drop_rows", "drop_duplicates"):
        df = df[~mask].reset_index(drop=True)
    elif a == "set_missing":
        if fix.rule != "bad_date":
            df[c] = to_numeric_clean(df[c])
        df.loc[mask, c] = None
    elif a == "fill_unknown":
        df.loc[mask, c] = "Unknown"
    elif a == "fill_median":
        x = to_numeric_clean(df[c])
        fix.params["median"] = float(x.median())
        df[c] = x.fillna(fix.params["median"])
    elif a == "cap":
        df[c] = to_numeric_clean(df[c]).clip(fix.params["lo"], fix.params["hi"])
    elif a == "standardize":
        df[c] = df[c].replace(fix.params["mapping"])
    else:
        raise ValueError(a)
    fix.rows_after = len(df)
    fix.cells_changed = 0 if a in ("drop_rows", "drop_duplicates") else int(mask.sum())
    return df


def apply_fixes(df: pd.DataFrame, fixes: list[Fix]) -> pd.DataFrame:
    for f in fixes:
        df = apply_fix(df, f)
    return df


def quality_score(df: pd.DataFrame, issues: list[Issue]) -> tuple[float, int]:
    """Score = % of rows with no detected issue. Low-severity items (outliers, missing free text) are excluded:
    unusual or blank-comment is not necessarily wrong."""
    if not len(df):
        return 0.0, 0
    bad = pd.Series(False, index=df.index)
    for i in issues:
        if i.severity != "low":
            bad |= _affected_mask(df, i.rule, i.column, i.params).fillna(False).astype(bool)
    clean = int((~bad).sum())
    return round(100 * clean / len(df), 1), clean
