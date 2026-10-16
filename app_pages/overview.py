"""Home: market status, ready setups, open trades, accounts and recent alerts at a glance."""

from __future__ import annotations

from collections import Counter
from html import escape

import streamlit as st

from core.retest_engine import SetupState
from core.timeutils import to_display
from data import repositories as repo
from services.account_service import build_dashboard
from services.scan_service import latest_qualified
from services.trade_service import monitor_open_trades
from ui import components as ui
from ui import state
from ui import theme as t

s = state.settings()
tz = state.tz()
db = state.db()

ctx = state.market_context()
md, scan = ctx.md, ctx.scan
unack = repo.count_unacknowledged(db)
n_alerts = sum(unack.values())

status_chips = []
if md.ok:
    feed_lbl = t.FEED_BADGE.get(md.feed_type, (md.feed_type.upper(), t.MUTED))[0]
    status_chips = [
        ("pulse", f"<b>{escape(md.spec.symbol)}</b>&nbsp;{escape(md.contract or '')}"),
        ("chart", escape(feed_lbl.title()) + " data"),
        ("clock", "Last bar " + escape(to_display(md.as_of, tz, "%a %H:%M %Z"))),
        ("bell", f"{n_alerts} new alert{'s' if n_alerts != 1 else ''}"),
    ]
ui.hero(
    "Trade With Discipline.",
    "Break-and-retest setups, your open trades and your prop-firm limits in one calm view. "
    "Analysis only - every decision is yours.",
    eyebrow="Your trading desk",
    accent="Discipline.",
    chips=ui.chips(status_chips),
)

_, b1, b2, _ = st.columns([1.2, 1, 1, 1.2])
if b1.button("See signals", type="primary", icon=":material/bolt:", width="stretch"):
    st.switch_page("app_pages/live_signals.py")
if b2.button("Log a trade", icon=":material/add:", width="stretch"):
    st.switch_page("app_pages/trade_manager.py")

if not md.ok:
    st.error(md.error or "No market data available.")
    st.stop()
ui.data_alerts(md)
spec = md.spec

counts = Counter(x.state for x in scan.setups) if scan else Counter()
watching = sum(
    v
    for k, v in counts.items()
    if k not in (SetupState.QUALIFIED, SetupState.REJECTED, SetupState.EXPIRED, SetupState.INVALIDATED)
)
open_trades = repo.list_trades(db, status="open")

st.write("")
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Ready setups",
    counts.get(SetupState.QUALIFIED, 0),
    help=f"Setups that met every rule and scored at least {s.scoring.min_score:g}/100 in the current window.",
)
m2.metric("Watching", watching, help="Breakouts waiting for a retest or confirmation right now.")
m3.metric("Open trades", len(open_trades), help="Trades you logged that are not closed yet.")
m4.metric(
    "New alerts",
    n_alerts,
    help=", ".join(f"{k}: {v}" for k, v in unack.items()) or "Nothing waiting for you.",
)

st.write("")
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown("### Ready setups")
    q = latest_qualified(scan, 3) if scan else []
    if not q:
        st.markdown(
            '<div class="brc-empty">No setups are ready right now. The engine is selective on purpose - '
            "open Signals to see what it is watching and why others were rejected.</div>",
            unsafe_allow_html=True,
        )
    for x in q:
        p = x.plan
        side = t.badge(x.direction.upper(), t.AQUA if x.direction == "long" else t.RED)
        levels = [
            ("Entry", spec.fmt(p.entry_price)),
            ("Stop", spec.fmt(p.stop)),
            ("Target 1", spec.fmt(p.target1)),
            ("Reward : risk", f"{p.rr:.1f}"),
        ]
        st.markdown(
            f'<div class="brc-card"><div class="brc-row"><div>{side}'
            f'<div class="brc-card-title">{escape(spec.symbol)} at '
            f"{escape(spec.fmt(x.zone['lower']))} - {escape(spec.fmt(x.zone['upper']))}</div>"
            f'<div class="brc-card-meta">Confirmed {escape(to_display(x.confirm_ts, tz, "%a %b %d, %H:%M"))}'
            f" &middot; {escape(x.contract or '')}</div></div>"
            f'<div class="brc-score">{x.score.total:.0f}<small> /100</small></div></div>'
            '<div class="brc-levels">'
            + "".join(
                f'<div><div class="k">{k}</div><div class="v">{escape(v)}</div></div>' for k, v in levels
            )
            + "</div></div>",
            unsafe_allow_html=True,
        )
    if q:
        st.caption(
            "The score ranks how well a setup matches the rules - it is not a probability. Levels are suggestions."
        )
    st.page_link("app_pages/live_signals.py", label="See all signals", icon=":material/arrow_forward:")

    st.markdown("### Open trades")
    if not open_trades:
        st.markdown(
            '<div class="brc-empty">No open trades. When you place a trade at your broker, log it under '
            "My Trades to get breakeven, trailing and invalidation prompts.</div>",
            unsafe_allow_html=True,
        )
    else:
        mons = monitor_open_trades(db, s, md, scan, None)
        for tr in open_trades:
            tspec = state.app().registry.get(tr["instrument"])
            m = mons.get(tr["id"])
            r_now = f"{m.current_r:+.2f}R" if m and m.current_r is not None else "-"
            stop = tspec.fmt(m.registered_stop if m else tr["initial_stop"])
            status = (
                "Monitoring"
                if m and m.data_ok
                else (m.data_message if m else f"Load {tr['instrument']} data to monitor this trade")
            )
            side = t.badge(tr["direction"].upper(), t.AQUA if tr["direction"] == "long" else t.RED)
            st.markdown(
                f'<div class="brc-card"><div class="brc-row"><div>{side}'
                f'<div class="brc-card-title">#{tr["id"]} {escape(tr["instrument"])} &times; {tr["quantity"]} '
                f"at {escape(tspec.fmt(tr['entry_price']))}</div>"
                f'<div class="brc-card-meta">Your stop {escape(stop)} (as you reported it) &middot; '
                f"{escape(status)}</div></div>"
                f'<div class="brc-score" style="color:var(--brc-ink)">{escape(r_now)}<small> now</small></div>'
                "</div></div>",
                unsafe_allow_html=True,
            )
    st.page_link("app_pages/trade_manager.py", label="Manage my trades", icon=":material/arrow_forward:")

with right:
    st.markdown("### Accounts")
    accts = repo.list_accounts(db, include_closed=False)
    if not accts:
        st.markdown(
            '<div class="brc-empty">No accounts yet. Add your Topstep or Tradeify accounts to track targets '
            "and drawdown room.</div>",
            unsafe_allow_html=True,
        )
    for a in accts:
        try:
            d = build_dashboard(db, a, s)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"{a['nickname']}: {exc}")
            continue
        pr = d.progress
        dd = pr.drawdown
        try:
            type_label = d.firm.account_type(a["account_type_code"]).label
        except KeyError:
            type_label = a["account_type_code"]
        body = (
            f'<div class="brc-card-title">{escape(a["nickname"])}</div>'
            f'<div class="brc-card-meta">{escape(d.firm.name)} &middot; {escape(type_label)} &middot; '
            f"{escape(a['stage'])}</div>"
            '<div class="brc-kv" style="margin-top:10px">'
            f'<div class="k">Balance (est.)</div><div class="v">{escape(ui.money(pr.current_balance))}</div>'
            f'<div class="k">Profit this cycle</div><div class="v">{escape(ui.money(pr.net_profit, signed=True))}</div>'
            f'<div class="k">Drawdown room</div><div class="v">'
            + (
                escape(ui.money(dd.remaining_allowance)) + " (estimate)"
                if dd
                else escape(pr.drawdown_message)
            )
            + "</div></div>"
        )
        if pr.profit_target:
            frac = max(pr.net_profit, 0) / pr.profit_target
            body += (
                f'<div class="brc-bar"><span style="width:{min(frac, 1) * 100:.0f}%"></span></div>'
                f'<div class="brc-muted">{frac:.0%} of the {escape(ui.money(pr.profit_target))} target'
                f" &middot; {escape(pr.status)}</div>"
            )
        if d.warnings:
            body += (
                f'<div class="brc-muted" style="margin-top:6px">{len(d.warnings)} item(s) to check - '
                "unconfirmed rules or estimates.</div>"
            )
        st.markdown(f'<div class="brc-card">{body}</div>', unsafe_allow_html=True)
    st.page_link("app_pages/firm_accounts.py", label="Open accounts", icon=":material/arrow_forward:")

    st.markdown("### Recent alerts")
    alerts = repo.list_alerts(db, limit=5)
    if not alerts:
        st.markdown('<div class="brc-empty">No alerts yet.</div>', unsafe_allow_html=True)
    else:
        rows = []
        for a in alerts:
            dot = t.SEVERITY_COLOR.get(a["severity"], t.MUTED)
            new = ' <span class="brc-pill">new</span>' if not a["acknowledged"] else ""
            rows.append(
                f'<div style="padding:8px 0;border-bottom:1px solid var(--brc-border)">'
                f'<span style="color:{dot}">&#9679;</span> <b>{escape(a["title"])}</b>{new}'
                f'<div class="brc-muted">{escape(to_display(a["trigger_ts"], tz, "%a %b %d, %H:%M"))} &middot; '
                f"{escape(a['severity'])}</div></div>"
            )
        st.markdown(f'<div class="brc-card">{"".join(rows)}</div>', unsafe_allow_html=True)
    st.page_link("app_pages/alerts.py", label="All alerts", icon=":material/arrow_forward:")

st.write("")
ui.safety_banner()
