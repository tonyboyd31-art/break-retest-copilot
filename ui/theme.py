"""Visual tokens and CSS for the dark trading-workstation theme.

Chart colours are the dark steps of a validated categorical palette
(checked against the chart surface for lightness band, chroma, CVD
separation and contrast). Status colours are reserved for state and always
ship with a text label.
"""

from __future__ import annotations

import streamlit as st

SURFACE = "#0f141a"  # chart plot surface
PAGE = "#0d1117"
GRID = "#232a33"
AXIS = "#38404b"
INK = "#e6edf3"
INK_2 = "#c3c2b7"
MUTED = "#8b949e"

# Categorical (validated order): blue, orange, aqua, yellow, violet, red
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

CSS = f"""
<style>
:root {{ --brc-surface: {SURFACE}; --brc-grid: {GRID}; --brc-muted: {MUTED}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
h1 {{ font-size: 1.55rem !important; }}
h2 {{ font-size: 1.2rem !important; }}
h3 {{ font-size: 1.02rem !important; }}
h4 {{ font-size: 0.98rem !important; margin-top: 0.6rem; }}
div[data-testid="stMetric"] {{
  background: #161b22; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 12px;
}}
div[data-testid="stMetricLabel"] p {{ font-size: 0.78rem; color: {MUTED}; }}
div[data-testid="stMetricValue"] {{ font-size: 1.25rem; }}
.brc-banner {{
  border: 1px solid rgba(255,255,255,0.10); border-left: 3px solid {BLUE}; background: #121820;
  padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; color: {INK_2}; margin-bottom: 0.8rem;
}}
.brc-badge {{
  display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.02em; border: 1px solid rgba(255,255,255,0.14); margin-right: 4px; white-space: nowrap;
}}
.brc-card {{
  background: #161b22; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 14px;
  margin-bottom: 10px;
}}
.brc-kv {{ display: grid; grid-template-columns: max-content 1fr; gap: 2px 14px; font-size: 0.86rem; }}
.brc-kv .k {{ color: {MUTED}; }}
.brc-kv .v {{ font-variant-numeric: tabular-nums; }}
.brc-muted {{ color: {MUTED}; font-size: 0.82rem; }}
@media (max-width: 640px) {{
  .block-container {{ padding-left: 0.8rem; padding-right: 0.8rem; }}
  div[data-testid="stMetricValue"] {{ font-size: 1.05rem; }}
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
