"""Break & Retest Copilot - Streamlit entry point.

Run with:  python -m streamlit run app.py

Analysis and decision support only: the application never submits,
modifies or cancels orders.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Break & Retest Copilot",
    page_icon=":material/candlestick_chart:",
    layout="wide",
    initial_sidebar_state="auto",
)

from data.database import storage_is_ephemeral
from ui.components import notice
from ui.safe_markdown import install as install_safe_markdown
from ui.state import app, render_sidebar
from ui.theme import inject_css

install_safe_markdown()  # currency "$" must never render as LaTeX

inject_css()
st.logo(str(ROOT / "ui/assets/logo.svg"), size="large", icon_image=str(ROOT / "ui/assets/mark.svg"))
app()  # migrate DB, sync instruments and firm rules

# The everyday pages sit directly in the top bar; research and system pages under "More".
PAGES = {
    "": [
        st.Page("app_pages/overview.py", title="Home", icon=":material/home:", default=True),
        st.Page("app_pages/live_signals.py", title="Signals", icon=":material/bolt:"),
        st.Page("app_pages/levels.py", title="Levels", icon=":material/stacked_line_chart:"),
        st.Page("app_pages/trade_manager.py", title="My Trades", icon=":material/rule:"),
        st.Page("app_pages/firm_accounts.py", title="Accounts", icon=":material/account_balance:"),
    ],
    "More": [
        st.Page("app_pages/journal.py", title="Journal", icon=":material/menu_book:"),
        st.Page("app_pages/backtester.py", title="Backtest", icon=":material/science:"),
        st.Page("app_pages/analytics.py", title="Stats", icon=":material/monitoring:"),
        st.Page("app_pages/alerts.py", title="Alerts", icon=":material/notifications:"),
        st.Page("app_pages/settings.py", title="Settings", icon=":material/settings:"),
    ],
}

nav = st.navigation(PAGES, position="top")
render_sidebar()
if storage_is_ephemeral():
    notice(
        "cloud_off",
        "<b>Temporary storage</b> - this hosted copy can erase trades, journal entries, accounts and settings "
        "whenever it restarts, sleeps or updates. For a permanent journal, run the app on your own computer.",
    )
nav.run()
