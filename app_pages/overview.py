"""Overview: data status, signals, open trades, accounts and recent alerts at a glance."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from core.retest_engine import STATE_LABELS, SetupState
from core.timeutils import to_display
from data import repositories as repo
from services.account_service import build_dashboard
from services.scan_service import latest_qualified
from services.trade_service import monitor_open_trades
from ui import components as ui
from ui import state

s = state.settings()
tz = state.tz()
ui.page_header("Overview", "What the copilot sees right now - signals, open trades, account risk and alerts.")
ui.safety_banner()

ctx = state.market_context()
md, scan = ctx.md, ctx.scan
if not md.ok:
    st.error(md.error or "No market data available.")
    st.stop()
ui.data_status(md, tz, compact=True)
spec = md.spec

c1, c2, c3, c4 = st.columns(4)
counts = Counter(x.state for x in scan.setups) if scan else Counter()
active = sum(
    v
    for k, v in counts.items()
    if k not in (SetupState.QUALIFIED, SetupState.REJECTED, SetupState.EXPIRED, SetupState.INVALIDATED)
)
c1.metric("Pending setups", active, help="Breakouts awaiting retest or confirmation at the current clock.")
c2.metric(
    "Qualified (this contract)",
    counts.get(SetupState.QUALIFIED, 0),
    help=f"Met every rule and scored at least {s.scoring.min_score:g}/100.",
)
c3.metric(
    "Rejected / expired / invalidated",
    counts.get(SetupState.REJECTED, 0)
    + counts.get(SetupState.EXPIRED, 0)
    + counts.get(SetupState.INVALIDATED, 0),
)
unack = repo.count_unacknowledged(state.db())
c4.metric(
    "Unacknowledged alerts",
    sum(unack.values()),
    help=", ".join(f"{k}: {v}" for k, v in unack.items()) or "None",
)

left, right = st.columns([3, 2])
with left:
    st.markdown("### Latest qualified setups")
    q = latest_qualified(scan, 6) if scan else []
    if not q:
        st.info(
            "No qualified setups in the analysed window. The engine is selective by design - see Live Market "
            "Signals for pending and rejected setups and why."
        )
    else:
        rows = []
        for x in q:
            rows.append(
                {
                    "confirmed": to_display(x.confirm_ts, tz, "%m-%d %H:%M"),
                    "dir": x.direction,
                    "score": round(x.score.total, 1),
                    "entry": x.plan.entry_price,
                    "stop": x.plan.stop,
                    "T1": x.plan.target1,
                    "R:R": x.plan.rr,
                    "zone": f"{spec.fmt(x.zone['lower'])}-{spec.fmt(x.zone['upper'])}",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Score = rule-match ranking, not a probability. Proposed levels are advisory.")
    st.page_link("app_pages/live_signals.py", label="Open Live Market Signals", icon=":material/bolt:")

    st.markdown("### Open manual trades")
    open_trades = repo.list_trades(state.db(), status="open")
    if not open_trades:
        st.caption(
            "No open trades registered. Register a trade you placed at your broker in the Trade Manager."
        )
    else:
        mons = monitor_open_trades(state.db(), s, md, scan, None)
        rows = []
        for tr in open_trades:
            m = mons.get(tr["id"])
            be = next((b for b in (m.be_status if m else []) if b["reached"]), None)
            rows.append(
                {
                    "#": tr["id"],
                    "instrument": tr["instrument"],
                    "dir": tr["direction"],
                    "qty": tr["quantity"],
                    "entry": tr["entry_price"],
                    "current R": round(m.current_r, 2) if m and m.current_r is not None else None,
                    "stop (user-reported)": m.registered_stop if m else tr["initial_stop"],
                    "BE review": ("reached " + to_display(be["trigger_ts"], tz, "%H:%M")) if be else "-",
                    "status": (
                        "monitoring"
                        if m and m.data_ok
                        else (m.data_message[:60] if m else f"Load {tr['instrument']} data to monitor")
                    ),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.page_link("app_pages/trade_manager.py", label="Open Trade Manager", icon=":material/rule:")

with right:
    st.markdown("### Accounts")
    accts = repo.list_accounts(state.db(), include_closed=False)
    if not accts:
        st.caption("No accounts yet - add your Tradeify / Topstep accounts in the Firm & Account Manager.")
    for a in accts:
        try:
            d = build_dashboard(state.db(), a, s)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{a['nickname']}: {exc}")
            continue
        p = d.progress
        dd = p.drawdown
        items = [
            ("Firm / type", f"{d.firm.name} - {a['account_type_code']}"),
            ("Stage", a["stage"]),
            ("Est. balance", ui.money(p.current_balance)),
            ("Net profit (cycle)", ui.money(p.net_profit, signed=True)),
        ]
        if p.profit_target:
            items.append(
                (
                    "Target progress",
                    f"{max(p.net_profit, 0) / p.profit_target:.0%} of {ui.money(p.profit_target)}",
                )
            )
        items.append(
            (
                "Drawdown allowance",
                ui.money(dd.remaining_allowance) + " (estimate)" if dd else p.drawdown_message,
            )
        )
        items.append(("Status", p.status))
        ui.kv_card(a["nickname"], items)
        if d.warnings:
            st.caption(
                f"{len(d.warnings)} warning(s) - unconfirmed rules or estimates. See Firm & Account Manager."
            )
    st.page_link(
        "app_pages/firm_accounts.py", label="Open Firm & Account Manager", icon=":material/account_balance:"
    )

    st.markdown("### Recent alerts")
    alerts = repo.list_alerts(state.db(), limit=6)
    if not alerts:
        st.caption("No alerts yet.")
    for a in alerts:
        mark = "" if a["acknowledged"] else "  -  **new**"
        st.markdown(
            f"**{a['title']}**{mark}  \n<span class='brc-muted'>{to_display(a['trigger_ts'], tz)} - "
            f"{a['severity']}</span>",
            unsafe_allow_html=True,
        )
    st.page_link("app_pages/alerts.py", label="Open Alerts Center", icon=":material/notifications:")

if scan is not None:
    with st.expander("Setup state counts (current contract window)"):
        st.dataframe(
            pd.DataFrame([{"state": STATE_LABELS[k], "count": v} for k, v in counts.items()]), hide_index=True
        )
