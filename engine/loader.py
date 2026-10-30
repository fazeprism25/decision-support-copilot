"""Load CSV/Excel bytes into a DataFrame: sheet listing, header-row detection, name cleaning."""
import csv
import hashlib
import io
from dataclasses import dataclass, field

import pandas as pd

EXCEL_EXT = (".xlsx", ".xlsm", ".xls")
SCAN_ROWS = 30


@dataclass
class LoadInfo:
    file_name: str
    sheet: str | None
    header_row: int  # 0-based row index in the raw file
    header_row_auto: int
    n_rows: int
    n_cols: int
    dropped_empty_rows: int
    dropped_empty_cols: int
    notes: list[str] = field(default_factory=list)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def is_excel(name: str) -> bool:
    return name.lower().endswith(EXCEL_EXT)


def _excel_engine() -> str:
    try:
        import python_calamine  # noqa: F401

        return "calamine"
    except ImportError:
        return "openpyxl"


def list_sheets(data: bytes, name: str) -> list[str]:
    if not is_excel(name):
        return []
    return pd.ExcelFile(io.BytesIO(data), engine=_excel_engine()).sheet_names


def _looks_numeric(v) -> bool:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True
    try:
        float(str(v).replace(",", "").strip())
        return True
    except ValueError:
        return False


def detect_header_row(raw: pd.DataFrame) -> int:
    """Pick the earliest row that looks most like a header: wide, mostly non-numeric text, distinct values."""
    head = raw.head(SCAN_ROWS)
    counts = head.notna().sum(axis=1)
    max_nn = counts.max() if len(counts) else 0
    if not max_nn:
        return 0
    scores = []
    for i, row in head.iterrows():
        vals = [v for v in row.tolist() if pd.notna(v) and str(v).strip() != ""]
        nn = len(vals)
        if nn < 0.5 * max_nn:
            scores.append(0.0)
            continue
        text_share = sum(isinstance(v, str) and not _looks_numeric(v) for v in vals) / nn
        distinct = len({str(v).strip().lower() for v in vals}) / nn
        scores.append(text_share * distinct * nn / max_nn)
    best = max(scores)
    if best == 0:
        return 0
    return next(i for i, s in enumerate(scores) if s >= 0.9 * best)


def clean_columns(cols) -> list[str]:
    out, seen = [], {}
    for i, c in enumerate(cols):
        name = "" if c is None or (isinstance(c, float) and pd.isna(c)) else " ".join(str(c).split())
        if not name or name.lower().startswith("unnamed:"):
            name = f"Column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def _csv_dialect(data: bytes) -> tuple[str, str]:
    """Return (encoding, delimiter)."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            sample = data[:65536].decode(enc)
            break
        except UnicodeDecodeError:
            continue
    # Pick the delimiter that splits the most lines into the same (>1) number of fields.
    lines = [ln for ln in sample.splitlines()[:200] if ln.strip()]
    best, sep = (0, 0), ","
    for d in (",", ";", "	", "|"):
        counts = [len(r) for r in csv.reader(lines, delimiter=d)]
        width = max(set(counts), key=counts.count) if counts else 1
        score = (counts.count(width) if width > 1 else 0, width)
        if score > best:
            best, sep = score, d
    return enc, sep


def _csv_head(data: bytes, enc: str, sep: str) -> pd.DataFrame:
    """First rows as a ragged-safe grid (title lines can have fewer fields than the table)."""
    lines = data[:262144].decode(enc, errors="replace").splitlines()[:SCAN_ROWS]
    rows = list(csv.reader(lines, delimiter=sep))
    width = max((len(r) for r in rows), default=0)
    grid = [[(v if v.strip() else None) for v in r] + [None] * (width - len(r)) for r in rows]
    return pd.DataFrame(grid)


def load_table(data: bytes, name: str, sheet: str | None = None,
               header_row: int | None = None) -> tuple[pd.DataFrame, LoadInfo]:
    notes = []
    if is_excel(name):
        sheet = sheet or list_sheets(data, name)[0]
        raw = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=None, engine=_excel_engine())
        auto = detect_header_row(raw)
        h = auto if header_row is None else header_row
        df = raw.iloc[h + 1:].copy()
        df.columns = clean_columns(raw.iloc[h].tolist())
        df = df.infer_objects()
    else:
        sheet = None
        enc, sep = _csv_dialect(data)
        raw = _csv_head(data, enc, sep)
        auto = detect_header_row(raw)
        h = auto if header_row is None else header_row
        names = clean_columns(raw.iloc[h].tolist()) if h < len(raw) else None
        df = pd.read_csv(io.BytesIO(data), sep=sep, encoding=enc, low_memory=False,
                         header=None, skiprows=h + 1, names=names, skip_blank_lines=False)

    rows0, cols0 = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if h:
        notes.append(f"Header found on row {h + 1}; {h} row(s) above it were skipped.")
    info = LoadInfo(name, sheet, h, auto, df.shape[0], df.shape[1],
                    rows0 - df.shape[0], cols0 - df.shape[1], notes)
    return df, info
