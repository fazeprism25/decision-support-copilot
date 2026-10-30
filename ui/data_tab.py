"""Data page: upload/load a file, then review and edit column roles."""
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from engine import llm
from engine.config import get_settings
from engine.domains import delivery as D
from engine.loader import file_hash, list_sheets, load_table
from engine.profile import profile_frame
from engine.roles import ROLE_HELP, ROLES, check_role, detect_roles, entity_attributes, schema_report
from engine.valuemap import from_table, map_value, remap_plan, to_table
from ui import state, style

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHECK_ICON = {"ok": "ok", "warn": "warn", "fail": "fail"}


@st.cache_data(max_entries=4, show_spinner=False)
def _sheets(data: bytes, name: str) -> list[str]:
    return list_sheets(data, name)


@st.cache_data(max_entries=4, show_spinner=False)
def _load(data: bytes, name: str, sheet, header_row):
    t = time.perf_counter()
    df, info = load_table(data, name, sheet, header_row)
    return df, info, time.perf_counter() - t


def _pick_source() -> tuple[bytes, str] | None:
    samples = sorted(p.name for p in DATA_DIR.glob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xlsm", ".xls"))
    options = ["Upload a file"] + (["Use a file from the data folder"] if samples else [])
    src = st.radio("Data source", options, horizontal=True, label_visibility="collapsed", key="src")
    if src == "Upload a file":
        up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xlsm", "xls"],
                              help="CSV, XLSX, XLSM or XLS. Empty rows and columns are removed automatically.")
        if up:
            st.session_state.upload = (up.getvalue(), up.name)
        elif st.session_state.get("upload"):  # the uploader empties when another page is opened; keep the file
            with st.container(horizontal=True, vertical_alignment="center"):
                st.caption(f"Using `{st.session_state.upload[1]}` (uploaded earlier).", width="content")
                if st.button("Remove", type="tertiary", key="rm_upload"):
                    del st.session_state.upload
                    st.rerun()
        return st.session_state.get("upload")
    name = st.selectbox("File in data folder", samples, key="src_file")
    return (DATA_DIR / name).read_bytes(), name


def _reset_for(dataset_id: str, df: pd.DataFrame):
    st.session_state.dataset_id = dataset_id
    st.session_state.profile = profile_frame(df)
    st.session_state.roles = None
    st.session_state.roles_version = 0
    st.session_state.fixes = []
    st.session_state.qa_history = []
    for k in ("vmap", "vmap_prop", "vmap_msg", "pending", "scenario_res", "rec_cache"):
        st.session_state.pop(k, None)


def _apply_edits(edited: pd.DataFrame):
    roles = st.session_state.roles
    changed = False
    for i, new in edited["role"].items():
        if new != roles.at[i, "role"]:
            col = roles.at[i, "column"]
            status, msg = check_role(new, st.session_state.profile[col], col)
            roles.loc[i, ["role", "source", "check", "note"]] = [new, "user", status, msg or "set by user"]
            changed = True
    if changed:
        st.session_state.roles_version += 1
        st.rerun()


def render():
    picked = _pick_source()
    if not picked:
        _landing()
        return
    data, name = picked

    c1, c2 = st.columns(2)
    fh = file_hash(data)
    sheets = _sheets(data, name)
    sheet = c1.selectbox("Sheet", sheets, key=f"sheet_{fh}") if len(sheets) > 1 else (sheets[0] if sheets else None)
    with st.spinner("Loading file..."):
        df_auto, info_auto, _ = _load(data, name, sheet, None)
    header = c2.number_input("Header row", min_value=1, value=info_auto.header_row_auto + 1, step=1,
                             key=f"header_row_{fh}_{sheet}",
                             help=f"The row that holds the column names. Auto-detected: row "
                                  f"{info_auto.header_row_auto + 1}. Change it if the column names look wrong.")
    with st.spinner("Loading file..."):
        df, info, secs = _load(data, name, sheet, int(header) - 1)

    dataset_id = f"{fh}:{sheet}:{info.header_row}"
    if st.session_state.get("dataset_id") != dataset_id:
        _reset_for(dataset_id, df)
    st.session_state.df = df
    st.session_state.load_info = info
    st.session_state.file_name = name

    m = st.columns(4)
    m[0].metric("Rows", f"{info.n_rows:,}")
    m[1].metric("Columns", info.n_cols)
    m[2].metric("Sheet", info.sheet or "CSV")
    m[3].metric("Load time", f"{secs:.1f}s")
    notes = list(info.notes)
    if info.dropped_empty_rows or info.dropped_empty_cols:
        notes.append(f"Removed {info.dropped_empty_rows} fully empty row(s) and "
                     f"{info.dropped_empty_cols} fully empty column(s).")
    if notes:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"Loaded `{name}` · {len(notes)} loading note(s)", width="content")
            style.info("\n".join(f"- {n}" for n in notes))
    with st.expander(f"Preview (first {min(500, len(df))} of {len(df):,} rows)", expanded=False):
        head = df.head(500)
        mixed = [c for c in head.columns if head[c].dtype == object]  # e.g. numbers mixed with text
        st.dataframe(head.astype({c: "string" for c in mixed}), width="stretch")

    _roles_section(df)
    from ui import demo

    demo.finish()


def _landing():
    st.markdown('<div class="hero"><h2>Turn a spreadsheet into decisions</h2>'
                "<p>Upload a file above to begin.</p>"
                "</div>", unsafe_allow_html=True)
    style.steps([("Load data", "Upload CSV/Excel and check what each column means"),
                 ("Clean it", "One click fixes the common problems"),
                 ("Read KPIs", "Cards and charts, each with its formula"),
                 ("Ask & test", "Plain-English questions and what-if scenarios"),
                 ("Decide", "Recommendations backed by evidence")])


def _roles_section(df: pd.DataFrame):
    st.divider()
    style.heading("Column roles",
                  "Each column gets a **role**, which decides how the app treats it.\n\n"
                  + "\n".join(f"- **{r}**: {h}" for r, h in ROLE_HELP.items()) +
                  "\n\n**Check** shows whether the data supports the role (ok · warn · fail).\n\n"
                  "**Source**: *AI* = model label confirmed by the data · *rules* = code-only guess · "
                  "*user* = your override.")
    ai_on = get_settings().ai_enabled
    rerun = st.button("Re-detect roles", help="Run role detection again from scratch (drops your edits).")
    if st.session_state.roles is None or rerun:
        with st.spinner("Detecting column roles..." + (" (asking AI)" if ai_on else "")):
            roles, msg = detect_roles(df, st.session_state.dataset_id, use_ai=ai_on,
                                      prof=st.session_state.profile)
        st.session_state.roles, st.session_state.roles_msg = roles, msg
        st.session_state.roles_version += 1
    roles = st.session_state.roles
    if not ai_on:
        st.warning("AI is off, so roles come from rules only. You can still edit them below.")
    elif st.session_state.get("roles_msg"):
        st.warning(st.session_state.roles_msg)

    st.caption("Click a cell in the **role** column to change it.")
    view = roles.assign(check=roles["check"].map(CHECK_ICON))
    edited = st.data_editor(
        view, key=f"roles_{st.session_state.dataset_id}_{st.session_state.roles_version}",
        hide_index=True, width="stretch", disabled=[c for c in view.columns if c != "role"],
        column_config={"role": st.column_config.SelectboxColumn(
            "role", options=ROLES, required=True,
            help="; ".join(f"{r} = {h}" for r, h in ROLE_HELP.items()))},
    )
    _apply_edits(edited)

    bad = roles[roles["check"] == "fail"]
    if len(bad):
        st.error(f"{len(bad)} role(s) are not supported by the data: {', '.join(bad['column'])}. "
                 "Calculations will skip these until fixed.")
    counts = roles["role"].value_counts()
    st.caption("Roles found: " + " · ".join(f"{r} × {n}" for r, n in counts.items()))

    rep = schema_report(roles, st.session_state.profile)
    unmapped = rep[rep["role"] == "UNMAPPED"]
    with st.expander(f"Schema mapping report ({len(rep) - len(unmapped)} mapped, {len(unmapped)} unmapped)"):
        st.caption("Original column → role → how it was parsed → confidence. Unmapped columns are not used.")
        st.dataframe(rep, hide_index=True, width="stretch")
        st.download_button("Download report (CSV)", rep.to_csv(index=False).encode(), "schema_mapping_report.csv")

    attrs = entity_attributes(df, roles)
    st.session_state.entity_attrs = attrs
    with st.expander(f"Entity attributes ({len(attrs)} found)"):
        if attrs.empty:
            st.caption("No columns found that stay fixed per group (e.g. a capacity repeated for each site).")
        else:
            st.caption("These columns repeat one value per group (e.g. a hub's daily capacity), so they describe "
                       "the group, not each row. They are never summed across rows.")
            st.dataframe(attrs, hide_index=True, width="stretch")

    _value_mapping_section()


def _propose():
    df, ctx = state.fixed_df(), state.kpi_ctx()
    rules = D.propose_mapping_rules(df, ctx.roles)
    msg = "Proposed by rules (pack vocabulary)."
    ok, why = llm.status()
    if ok:
        try:
            with st.spinner("AI is proposing standard labels..."):
                rules = D.merge_proposals(rules, D.propose_mapping_ai(df, ctx.roles, state.data_hash()))
            msg = "Proposed by AI + rules; every label is checked against the standard vocabulary."
        except llm.LLMError as e:
            msg = f"AI proposal unavailable, rules only. ({str(e)[:120]})"
    else:
        msg += f" AI not used: {why}"
    st.session_state.vmap_prop = to_table(rules[0], df, rules[1])
    st.session_state.vmap_msg = msg


def _remap(col, v, old: dict, new: dict):
    inv = {lab: raw for raw, lab in old.get(col or "", {}).items()}
    return map_value(col, inv.get(v, v), new)


def confirm_vmap(new: dict):
    """Apply a confirmed mapping everywhere: KPI plan labels, What-if setup and stacked constraints follow."""
    old = state.vmap()
    if st.session_state.get("kpi_plan"):
        inv_plan = remap_plan(st.session_state.kpi_plan, {c: {lab: raw for raw, lab in m.items()} for c, m in old.items()})
        st.session_state.kpi_plan = remap_plan(inv_plan, new)
    for c in st.session_state.get("constraints", []):
        for x in c["params"].exclusions:
            x.value = [_remap(x.column, v, old, new) for v in x.value] if isinstance(x.value, list) else                 _remap(x.column, x.value, old, new)
        for t in c["params"].targets:
            if t.segment_column:
                t.segment_value = _remap(t.segment_column, t.segment_value, old, new)
    st.session_state.vmap = new
    st.session_state.setup_used_plan = False  # re-derive What-if setup (completed labels) from the mapped data


def _value_mapping_section():
    st.divider()
    style.heading("Category value mapping",
                  "Different departments use different labels for the same thing (P1/P2/P3, completed/rto, Y/Yes). "
                  "Map them to standard labels once; after you **confirm**, every page uses the standard labels.\n\n"
                  "Edit the **standard label** column freely; leave it blank to keep the original value.\n\n"
                  "Standard vocabulary:\n" + "\n".join(f"- **{k}**: {', '.join(v)}" for k, v in D.VOCAB.items()))
    key = (st.session_state.dataset_id, state.fixes_sig(), state.roles_sig())
    if st.session_state.get("vmap_prop") is None or st.session_state.get("vmap_prop_for") != key:
        _propose()
        st.session_state.vmap_prop_for = key
    prop = st.session_state.vmap_prop
    if not len(prop):
        st.caption("No category column matches a standard vocabulary (priority, order status, peak flag), so no "
                   "mapping is needed.")
    else:
        st.caption(st.session_state.get("vmap_msg", ""))
        edited = st.data_editor(prop, hide_index=True, width="stretch", key=f"vm_{st.session_state.dataset_id}",
                                disabled=["column", "original value", "rows", "source"],
                                column_config={"standard label": st.column_config.TextColumn(
                                    "standard label", help="Edit freely; leave blank to keep the original value. "
                                    "Standard: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in D.VOCAB.items()))})
        row = st.container(horizontal=True)
        if row.button("Confirm mapping", type="primary"):
            confirm_vmap(from_table(edited))
            st.rerun()
        if row.button("Clear mapping", disabled=not state.vmap()):
            confirm_vmap({})
            st.rerun()
        if row.button("Propose again"):
            _propose()
            st.rerun()
        if state.vmap():
            n = sum(1 for m in state.vmap().values() for k, v in m.items() if v != k)
            st.success(f"Mapping confirmed: {n} value(s) relabelled. KPIs, Ask, What-if and Recommendations use the "
                       "standard labels." if n else "Mapping confirmed: the labels are already standard.")
        else:
            st.warning("Mapping not confirmed yet: all pages still use the original labels.")
    pk = state.pack()
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown(f"{':green-badge[ON]' if pk.active else ':gray-badge[OFF]'} Delivery-ops pack {pk.reason}.", width="content")
        style.info("The **delivery-ops pack** adds delivery-specific metrics (on-time %, promise gap, margin per "
                   "order, hub load). It switches on only when the data has promised time, actual time, fee, cost, "
                   "a hub/site and its capacity.")
