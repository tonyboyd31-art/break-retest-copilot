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
from ui.safe_markdown import install as install_safe_markdown
from ui.state import app, render_sidebar
from ui.theme import inject_css

install_safe_markdown()  # currency "$" must never render as LaTeX

inject_css()
app()  # migrate DB, sync instruments and firm rules

PAGES = {
    "Trading": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/live_signals.py", title="Live Market Signals", icon=":material/bolt:"),
        st.Page("app_pages/levels.py", title="Support & Resistance", icon=":material/stacked_line_chart:"),
        st.Page("app_pages/trade_manager.py", title="Trade Manager", icon=":material/rule:"),
    ],
    "Accounts": [
        st.Page(
            "app_pages/firm_accounts.py", title="Firm & Account Manager", icon=":material/account_balance:"
        ),
    ],
    "Research": [
        st.Page("app_pages/backtester.py", title="Backtesting Lab", icon=":material/science:"),
        st.Page("app_pages/analytics.py", title="Analytics", icon=":material/monitoring:"),
        st.Page("app_pages/journal.py", title="Trade Journal", icon=":material/menu_book:"),
    ],
    "System": [
        st.Page("app_pages/alerts.py", title="Alerts Center", icon=":material/notifications:"),
        st.Page("app_pages/settings.py", title="Settings & Data Sources", icon=":material/settings:"),
    ],
}

nav = st.navigation(PAGES)
render_sidebar()
if storage_is_ephemeral():
    st.warning(
        "**Temporary storage.** This copy runs on a hosted service that erases its database whenever the "
        "app restarts, sleeps or is updated. Trades, journal entries, accounts and settings entered here "
        "can be lost without warning. For a permanent journal, run the app on your own computer.",
        icon=":material/cloud_off:",
    )
nav.run()
