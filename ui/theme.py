"""Visual tokens and CSS: warm near-black surfaces with antique-gold accents.

Headings use a serif display face (Cormorant Garamond) and body text Inter,
both loaded through ``.streamlit/config.toml``. Gold is the interface accent
only - it never encodes data. Chart colours are the dark steps of a validated
categorical palette (lightness band, chroma, CVD separation and contrast,
checked against ``SURFACE``). Status colours are reserved for state and
always ship with a text label.
"""

from __future__ import annotations

import streamlit as st

PAGE = "#0e0c09"
SURFACE = "#15120d"  # cards and chart plot surface
SURFACE_2 = "#1a1611"
BORDER = "#2e281e"
GRID = "#262119"
AXIS = "#3b3427"
INK = "#f1e9d8"
INK_2 = "#cfc6b3"
MUTED = "#958c7a"

# Interface accent (never a data colour)
GOLD = "#c8b16a"
GOLD_HI = "#dcc888"
GOLD_LO = "#a9934f"
ON_GOLD = "#1a150c"

# Categorical (validated order on SURFACE): blue, orange, aqua, yellow, violet, red
BLUE = "#3987e5"
ORANGE = "#d95926"
AQUA = "#199e70"
YELLOW = "#c98500"
VIOLET = "#9085e9"
RED = "#e66767"

# Status (reserved; always paired with a label)
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

SEVERITY_COLOR = {"info": MUTED, "low": BLUE, "medium": WARNING, "high": SERIOUS, "critical": CRITICAL}

# Line icons (24px grid, stroke = currentColor) for chips, notices and cards.
_ICON_PATHS = {
    "shield": '<path d="M12 3l7 3v6c0 4.5-3 7.7-7 9-4-1.3-7-4.5-7-9V6l7-3z"/><path d="M9 12l2 2 4-4"/>',
    "cloud_off": '<path d="M3 3l18 18"/><path d="M8.5 8.6A5 5 0 0 0 7 18h10.5"/>'
    '<path d="M20.4 16.6A4 4 0 0 0 17 10h-.8A6 6 0 0 0 10 6.1"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.8"/>',
    "trend": '<path d="M3 17l6-6 4 4 8-8"/><path d="M15 7h6v6"/>',
    "clock": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
    "bell": '<path d="M6 16V11a6 6 0 1 1 12 0v5l1.5 2h-15z"/><path d="M10 20.5a2 2 0 0 0 4 0"/>',
    "chart": '<path d="M4 20V4"/><path d="M4 20h16"/><path d="M8 16v-5M12 16V8M16 16v-3"/>',
    "wallet": '<rect x="3" y="6" width="18" height="13" rx="2.5"/><path d="M16 12.5h2"/><path d="M3 9.5h18"/>',
    "pulse": '<path d="M3 12h4l2.5-6 4 12 2.5-6H21"/>',
    "arrow_right": '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/>',
}


def icon(name: str, size: int = 16) -> str:
    """Inline SVG line icon in the current text colour."""
    return (
        f'<svg class="brc-icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" '
        f'aria-hidden="true">{_ICON_PATHS[name]}</svg>'
    )


CSS = f"""
<style>
:root {{
  --brc-page: {PAGE}; --brc-surface: {SURFACE}; --brc-surface-2: {SURFACE_2}; --brc-border: {BORDER};
  --brc-ink: {INK}; --brc-ink-2: {INK_2}; --brc-muted: {MUTED};
  --brc-gold: {GOLD}; --brc-gold-hi: {GOLD_HI}; --brc-gold-lo: {GOLD_LO}; --brc-on-gold: {ON_GOLD};
  --brc-serif: "Cormorant Garamond", Georgia, "Times New Roman", serif;
}}
.stApp {{
  background:
    radial-gradient(1100px 480px at 50% -140px, rgba(200,177,106,0.11), transparent 70%),
    radial-gradient(900px 600px at 100% 0%, rgba(200,177,106,0.035), transparent 60%),
    {PAGE};
}}
header[data-testid="stHeader"] {{
  background: rgba(14,12,9,0.82); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(46,40,30,0.7);
}}
.block-container {{ padding-top: 4.2rem; padding-bottom: 3rem; max-width: 1360px; }}

/* Headings: serif display face, lining figures so prices read cleanly */
h1, h2, h3, h4 {{ font-family: var(--brc-serif) !important; letter-spacing: 0; color: var(--brc-ink); }}
h1, h2, h3, h4, .brc-title, .brc-card-title, .brc-score {{
  font-variant-numeric: lining-nums tabular-nums; font-feature-settings: "lnum" 1, "tnum" 1;
}}
h1 {{ font-size: 2.4rem !important; }}
h2 {{ font-size: 1.75rem !important; }}
h3 {{ font-size: 1.4rem !important; margin-top: 0.4rem; }}
h4 {{ font-size: 1.15rem !important; margin-top: 0.6rem; }}

/* Page header: eyebrow, title with one gold word, plain-English purpose */
.brc-eyebrow {{
  display: flex; align-items: center; gap: 14px; color: var(--brc-gold); font-size: 0.7rem;
  font-weight: 500; letter-spacing: 0.32em; text-transform: uppercase; margin-bottom: 0.35rem;
}}
.brc-eyebrow::after {{
  content: ""; height: 1px; width: 56px; background: linear-gradient(90deg, var(--brc-gold-lo), transparent);
}}
.brc-eyebrow.center {{ justify-content: center; }}
.brc-eyebrow.center::before {{
  content: ""; height: 1px; width: 56px; background: linear-gradient(270deg, var(--brc-gold-lo), transparent);
}}
.brc-title {{
  font-family: var(--brc-serif); font-weight: 500; font-size: 2.7rem; line-height: 1.08;
  color: var(--brc-ink); margin: 0 0 0.45rem 0;
}}
.brc-title .gold {{ color: var(--brc-gold); }}
.brc-sub {{ color: var(--brc-ink-2); font-size: 1rem; line-height: 1.6; max-width: 760px; margin-bottom: 1.1rem; }}
.brc-hero {{ text-align: center; padding: 1.6rem 0 0.4rem; }}
.brc-hero .brc-title {{ font-size: 3.6rem; }}
.brc-hero .brc-sub {{ margin-left: auto; margin-right: auto; font-size: 1.08rem; }}

/* Chips (outlined, gold icon, small caps) */
.brc-chips {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 0.4rem 0 1.4rem; }}
.brc-hero .brc-chips {{ justify-content: center; }}
.brc-chip {{
  display: inline-flex; align-items: center; gap: 10px; padding: 10px 16px; min-width: 0;
  border: 1px solid var(--brc-border); border-radius: 12px; background: rgba(26,22,17,0.55);
  color: var(--brc-ink-2); font-size: 0.76rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase;
}}
.brc-chip .brc-icon {{ color: var(--brc-gold); flex: none; }}
.brc-chip b {{ color: var(--brc-ink); font-weight: 600; }}

/* Notices: slim, gold-outlined */
.brc-banner, .brc-notice {{
  display: flex; gap: 10px; align-items: flex-start; border: 1px solid rgba(200,177,106,0.22);
  background: rgba(200,177,106,0.045); padding: 10px 14px; border-radius: 12px; font-size: 0.85rem;
  color: var(--brc-ink-2); margin-bottom: 1rem; line-height: 1.5;
}}
.brc-banner .brc-icon, .brc-notice .brc-icon {{ color: var(--brc-gold); flex: none; margin-top: 2px; }}
.brc-banner b, .brc-notice b {{ color: var(--brc-ink); }}

/* Cards */
.brc-card {{
  background: linear-gradient(180deg, rgba(26,22,17,0.92), rgba(21,18,13,0.92));
  border: 1px solid var(--brc-border); border-radius: 16px; padding: 16px 18px; margin-bottom: 12px;
}}
.brc-card-title {{ font-family: var(--brc-serif); font-size: 1.3rem; font-weight: 600; color: var(--brc-ink); }}
.brc-card-meta {{ color: var(--brc-muted); font-size: 0.8rem; margin-top: 2px; }}
.brc-kv {{ display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; font-size: 0.88rem; }}
.brc-kv .k {{ color: var(--brc-muted); }}
.brc-kv .v {{ font-variant-numeric: tabular-nums; color: var(--brc-ink); }}
.brc-muted {{ color: var(--brc-muted); font-size: 0.84rem; }}
.brc-bar {{ height: 6px; border-radius: 6px; background: #2a241b; overflow: hidden; margin: 6px 0 2px; }}
.brc-bar > span {{
  display: block; height: 100%; border-radius: 6px;
  background: linear-gradient(90deg, var(--brc-gold-lo), var(--brc-gold-hi));
}}
.brc-levels {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
.brc-levels .k {{ color: var(--brc-muted); font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase; }}
.brc-levels .v {{ font-size: 1.02rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--brc-ink); }}
.brc-pill {{
  display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.14em; text-transform: uppercase; border: 1px solid var(--brc-border); color: var(--brc-ink);
}}
.brc-row {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap; }}
.brc-score {{ font-family: var(--brc-serif); font-size: 1.6rem; color: var(--brc-gold); line-height: 1; }}
.brc-score small {{ font-family: Inter, sans-serif; font-size: 0.72rem; color: var(--brc-muted); }}
.brc-empty {{
  border: 1px dashed var(--brc-border); border-radius: 16px; padding: 22px 18px; text-align: center;
  color: var(--brc-muted); font-size: 0.9rem; margin-bottom: 12px;
}}

/* Status badges: coloured dot + ink text */
.brc-badge {{
  display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
  font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em; border: 1px solid var(--brc-border);
  margin: 0 6px 4px 0; white-space: nowrap; background: rgba(26,22,17,0.6);
}}

/* Top navigation: small-caps links, active page as a gold pill */
a[data-testid="stTopNavLink"], button[data-testid="stTopNavSection"] {{ border-radius: 999px !important; }}
a[data-testid="stTopNavLink"] > span:first-child {{ display: none; }}
a[data-testid="stTopNavLink"] p, button[data-testid="stTopNavSection"] p {{
  font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 500;
}}
a[data-testid="stTopNavLink"][aria-current="page"] {{
  background: rgba(200,177,106,0.12) !important; box-shadow: inset 0 0 0 1px rgba(200,177,106,0.4);
}}
a[data-testid="stTopNavLink"][aria-current="page"] p {{ color: var(--brc-gold) !important; }}

/* Metrics: small-caps label; values never truncated */
div[data-testid="stMetric"] {{
  background: linear-gradient(180deg, rgba(26,22,17,0.92), rgba(21,18,13,0.92));
  border: 1px solid var(--brc-border); border-radius: 16px; padding: 14px 16px; height: 100%;
}}
label[data-testid="stMetricLabel"] p {{
  font-size: 0.68rem; color: var(--brc-muted); letter-spacing: 0.12em; text-transform: uppercase;
}}
[data-testid="stMetricLabel"] .stMarkdownEllipsis, [data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"] .stMarkdownEllipsis, [data-testid="stMetricValue"] p {{
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}}
div[data-testid="stMetricValue"] {{ font-size: 1.45rem; font-variant-numeric: tabular-nums; line-height: 1.25; }}

/* Status line (data source, contract, clock) */
.brc-status {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 4px 10px; font-size: 0.82rem;
  color: var(--brc-ink-2); margin: 0 0 1rem;
}}
.brc-status .brc-badge {{ margin: 0; }}
.brc-status .sep {{ color: var(--brc-border); }}

/* Buttons: gold gradient pill (primary), quiet outline (secondary) */
button[data-testid^="stBaseButton-primary"] {{
  background: linear-gradient(135deg, {GOLD_HI}, {GOLD_LO}) !important; color: {ON_GOLD} !important;
  border: none !important; font-weight: 600; letter-spacing: 0.04em;
  box-shadow: 0 8px 28px rgba(200,177,106,0.22);
}}
button[data-testid^="stBaseButton-primary"]:hover {{ filter: brightness(1.06); }}
button[data-testid^="stBaseButton-primary"] p {{ color: {ON_GOLD} !important; }}
button[data-testid^="stBaseButton-secondary"] {{ background: rgba(26,22,17,0.6); }}
button[data-testid^="stBaseButton-secondary"]:hover {{ border-color: var(--brc-gold) !important; color: var(--brc-gold); }}

/* Page links: small-caps gold text links */
a[data-testid="stPageLink-NavLink"] p {{
  color: var(--brc-gold); font-size: 0.74rem; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 500;
}}
a[data-testid="stPageLink-NavLink"] span[data-testid="stIconMaterial"] {{ color: var(--brc-gold); }}
a[data-testid="stPageLink-NavLink"] .stMarkdownEllipsis, a[data-testid="stPageLink-NavLink"] p {{
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}}

/* Tabs */
[data-testid="stTab"] p {{ font-size: 0.74rem; letter-spacing: 0.14em; text-transform: uppercase; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: var(--brc-gold); }}

/* Expanders */
details[data-testid="stExpander"] {{ border-radius: 14px; background: rgba(21,18,13,0.6); }}

/* Sidebar */
section[data-testid="stSidebar"] h3 {{ font-size: 1.35rem !important; }}

@media (max-width: 640px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 3.6rem; }}
  .brc-title {{ font-size: 2rem; }}
  .brc-hero .brc-title {{ font-size: 2.4rem; }}
  .brc-chip {{ font-size: 0.68rem; padding: 8px 12px; }}
  .brc-levels {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  div[data-testid="stMetricValue"] {{ font-size: 1.3rem; }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    """Status pill: coloured dot + text in ink (text never wears the data colour)."""
    return (
        f'<span class="brc-badge"><span style="color:{color}">&#9679;</span> '
        f'<span style="color:{INK}">{text}</span></span>'
    )


FEED_BADGE = {
    "live": ("LIVE", GOOD),
    "delayed": ("DELAYED", WARNING),
    "historical": ("HISTORICAL", BLUE),
    "simulated": ("SIMULATED", VIOLET),
}

FRESHNESS_BADGE = {
    "fresh": ("FRESH", GOOD),
    "simulated": ("SIMULATED CLOCK", VIOLET),
    "historical": ("HISTORICAL", BLUE),
    "market_closed": ("MARKET CLOSED", MUTED),
    "stale": ("STALE", CRITICAL),
    "no_data": ("NO DATA", CRITICAL),
}

RULE_BADGE = {
    "VERIFIED": ("VERIFIED", GOOD),
    "USER_CONFIRMED": ("USER CONFIRMED", GOOD),
    "UNVERIFIED": ("UNVERIFIED", WARNING),
    "CONFLICTING": ("CONFLICTING", CRITICAL),
}

STATE_COLOR = {
    "qualified": GOOD,
    "rejected": MUTED,
    "expired": MUTED,
    "invalidated": CRITICAL,
    "awaiting_retest": BLUE,
    "awaiting_confirmation": YELLOW,
    "retest_detected": YELLOW,
    "breakout_detected": BLUE,
    "level_identified": MUTED,
}
