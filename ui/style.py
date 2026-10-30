"""Shared look and feel: dark stone + green page CSS, small circular "i" tips that open a pop-up with the details."""
import streamlit as st

# Palette (also used by ui/charts.py). Green is the only accent; red/green carry meaning (worse/better).
GREEN, GREEN_SOFT, RED, AMBER = "#5cb338", "rgba(92, 179, 56, 0.14)", "#e5675e", "#d8a24f"
INK, INK_MUTED, INK_FAINT, LINE, CARD = "#e7e5e4", "#a8a29e", "#78716c", "#2f2d2a", "#1c1b19"

CSS = f"""
<style>
.block-container {{padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}}
h1, h2, h3, h4 {{letter-spacing: -0.015em;}}
h2 {{font-size: 1.45rem !important;}}
h3 {{font-size: 1.02rem !important; margin-top: 0.4rem;}}
h4 {{font-size: 0.95rem !important;}}
hr {{border-color: {LINE} !important; margin: 1.2rem 0 !important;}}
[data-testid="stCaptionContainer"] {{color: {INK_MUTED};}}

/* sidebar section menu: a radio drawn as a list of links, the open one tinted green */
.st-key-nav [role="radiogroup"] {{gap: 2px; width: 100%;}}
.st-key-nav [role="radiogroup"] label {{width: 100%; margin: 0; padding: 0.42rem 0.7rem; border-radius: 0.4rem;}}
.st-key-nav [data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]) {{display: none;}}
.st-key-nav [role="radiogroup"] label p {{font-size: 0.92rem; color: {INK_MUTED};}}
.st-key-nav [role="radiogroup"] label:hover {{background: #24221f;}}
.st-key-nav [role="radiogroup"] label:hover p {{color: {INK};}}
.st-key-nav [data-testid="stRadioOption"][data-selected="true"] {{background: {GREEN_SOFT};}}
.st-key-nav [data-testid="stRadioOption"][data-selected="true"] p {{color: {GREEN}; font-weight: 600;}}
.st-key-session button p {{font-size: 0.82rem; color: {INK_FAINT};}}
.st-key-session button:hover p {{color: {INK};}}

/* metric tiles: card with a big tight number */
[data-testid="stMetric"] {{background: {CARD}; border: 1px solid {LINE}; border-radius: 0.6rem; padding: 0.8rem 1rem;}}
[data-testid="stMetricValue"] {{font-size: 1.85rem; font-weight: 600; letter-spacing: -0.02em;}}
[data-testid="stMetricLabel"] p {{font-size: 0.82rem; color: {INK_MUTED};}}
[data-testid="stVerticalBlockBorderWrapper"] {{border-radius: 0.6rem;}}

/* table headers: small, uppercase, muted */
[data-testid="stDataFrame"] {{border: 1px solid {LINE}; border-radius: 0.5rem;}}
[data-testid="stExpander"] details {{border-color: {LINE}; border-radius: 0.5rem;}}
[data-testid="stExpander"] summary p {{font-size: 0.9rem;}}

/* "i" tip: a 17px ringed circle with the letter dead centre (Streamlit's dropdown-arrow box is removed) */
[class*="st-key-tip_"] button {{min-height: 0 !important; height: 17px; width: 17px; padding: 0 !important;
  display: flex; align-items: center; justify-content: center;
  border-radius: 50% !important; border: 1px solid {INK_FAINT} !important; background: transparent !important;}}
[class*="st-key-tip_"] button > div {{gap: 0 !important; margin: 0 !important; width: 100%; height: 100%; display: flex;
  align-items: center; justify-content: center;}}
[class*="st-key-tip_"] button div[aria-hidden="true"] {{display: none !important;}}
[class*="st-key-tip_"] button [data-testid="stMarkdownContainer"] {{display: flex; align-items: center; justify-content: center;}}
[class*="st-key-tip_"] button p {{margin: 0 !important; padding: 0; font-size: 10px !important; font-weight: 600;
  line-height: 1 !important; color: {INK_MUTED};}}
[class*="st-key-tip_"] button:hover {{border-color: {GREEN} !important;}}
[class*="st-key-tip_"] button:hover p {{color: {GREEN};}}

/* landing hero + numbered steps */
.hero {{padding: 1.2rem 1.4rem; border-radius: 0.7rem; margin-bottom: 1rem; background: {CARD};
  border: 1px solid {LINE}; border-left: 3px solid {GREEN};}}
.hero h2 {{margin: 0 0 0.3rem 0; padding: 0; font-size: 1.25rem !important;}}
.hero p {{margin: 0; color: {INK_MUTED};}}
.steps {{display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.7rem; margin: 0.4rem 0 1rem;}}
.step {{border: 1px solid {LINE}; background: {CARD}; border-radius: 0.6rem; padding: 0.85rem 0.95rem;}}
.step b {{display: block; margin-bottom: 0.2rem; font-weight: 600;}}
.step span {{font-size: 0.85rem; color: {INK_MUTED};}}
.step .n {{display: inline-block; font-size: 0.72rem; font-weight: 600; color: {GREEN}; letter-spacing: 0.06em;
  margin-bottom: 0.35rem;}}

/* small uppercase section label */
.eyebrow {{font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase;
  color: {INK_FAINT}; margin: 0.2rem 0 0.4rem;}}

/* What-if cascade strip: one chip per step, arrows between */
.chain {{display: flex; flex-wrap: wrap; align-items: stretch; gap: 0.35rem; margin: 0.3rem 0 0.6rem;}}
.chain .arrow {{align-self: center; color: {INK_FAINT}; font-size: 0.9rem;}}
.chain .chip {{flex: 1 1 120px; min-width: 120px; border: 1px solid {LINE}; background: {CARD}; border-radius: 0.5rem;
  padding: 0.5rem 0.6rem;}}
.chain .chip .k {{font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: {INK_FAINT};}}
.chain .chip .v {{font-size: 1.02rem; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 0.1rem;}}
.chain .chip .d {{font-size: 0.75rem; font-variant-numeric: tabular-nums; color: {INK_MUTED};}}
.chain .chip.up {{border-color: rgba(92, 179, 56, 0.55);}}
.chain .chip.down {{border-color: rgba(229, 103, 94, 0.55);}}
.chain .chip.up .d {{color: {GREEN};}}
.chain .chip.down .d {{color: {RED};}}

/* per-lever effect line under a slider */
.effect {{font-size: 0.8rem; color: {INK_MUTED}; margin: -0.6rem 0 0.7rem; font-variant-numeric: tabular-nums;}}
.effect .up {{color: {GREEN};}}
.effect .down {{color: {RED};}}
.effect.idle {{color: {INK_FAINT};}}

section[data-testid="stSidebar"] h3 {{font-size: 0.95rem !important;}}
</style>
"""


def apply():
    st.session_state._tip_n = 0
    st.markdown(CSS, unsafe_allow_html=True)


def info(md: str):
    """A small "i" button; clicking it opens a pop-up with `md`."""
    st.session_state._tip_n = st.session_state.get("_tip_n", 0) + 1
    with st.container(key=f"tip_{st.session_state._tip_n}", width="content"):
        with st.popover("i", type="tertiary", help="What is this?"):
            st.markdown(md)


def label(text: str, md: str | None = None):
    """Bold inline label with an optional "i" pop-up next to it."""
    heading(f"**{text}**", md, level=0)


def heading(text: str, md: str | None = None, level: int = 3):
    """Heading with an optional "i" pop-up next to it."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"{'#' * level} {text}" if level else text, width="content")
        if md:
            info(md)


def eyebrow(text: str):
    st.markdown(f'<div class="eyebrow">{text}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]):
    """Numbered step cards (title, one-line hint)."""
    cards = "".join(f'<div class="step"><div class="n">STEP {i:02d}</div><b>{t}</b><span>{h}</span></div>'
                    for i, (t, h) in enumerate(items, 1))
    st.markdown(f'<div class="steps">{cards}</div>', unsafe_allow_html=True)
