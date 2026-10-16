"""Trade Manager: manual trade registration, advisory guidance, recorded actions and the risk calculator."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.risk import RiskInputError, size_position
from core.timeutils import from_iso, to_display, utcnow
from core.trade_management import r_progress_series, trade_view_from_rows
from data import repositories as repo
from data.models import ACTION_TYPES, ManualTradeInput, TradeActionInput
from services.account_service import account_contract_limits, build_dashboard
from services.trade_service import TradeValidationError, monitor_open_trades, record_action, register_trade
from ui import charts, state
from ui import components as ui

s = state.settings()
tz = state.tz()
db = state.db()
reg = state.app().registry
ui.page_header(
    "My Trades",
    "Log the trades you placed at your broker, get calm management prompts, and record what you actually did.",
    eyebrow="Your positions",
    accent="Trades",
)
ui.safety_banner(
    "Stops shown here are USER-REPORTED - the app cannot see your broker and never changes orders."
)

ctx = state.market_context()
md, scan = ctx.md, ctx.scan


def _ts_input(label: str, default: pd.Timestamp, key: str) -> pd.Timestamp:
    local = default.tz_convert(tz)
    c1, c2 = st.columns(2)
    d = c1.date_input(f"{label} date ({tz})", local.date(), key=f"{key}_d")
    t_ = c2.time_input(
        f"{label} time ({tz})", local.time().replace(second=0, microsecond=0), key=f"{key}_t", step=60
    )
    return (
        pd.Timestamp(datetime.combine(d, t_))
        .tz_localize(tz, ambiguous="raise", nonexistent="raise")
        .tz_convert("UTC")
    )


def _clock() -> pd.Timestamp:
    return md.now if (md.ok and md.is_replay and md.now is not None) else utcnow()


tab_open, tab_reg, tab_risk, tab_closed = st.tabs(
    ["Open trades & guidance", "Register trade", "Risk calculator", "Closed trades"]
)

# ---------------------------------------------------------------------------
with tab_reg:
    pre = st.session_state.get("prefill_trade") or {}
    if pre:
        st.info(
            f"Prefilled from setup {pre.get('setup_id')}. Replace the values with your ACTUAL fill and stop."
        )
    accounts = repo.list_accounts(db, include_closed=False)
    acct_opts = {"(no account)": None} | {f"{a['nickname']} ({a['firm_code']})": a["id"] for a in accounts}
    symbols = reg.symbols()
    with st.form("register_trade", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        acct_lbl = c1.selectbox("Account", list(acct_opts))
        default_sym = pre.get("instrument") or (md.spec.symbol if md.ok else symbols[0])
        sym = c2.selectbox(
            "Instrument", symbols, index=symbols.index(default_sym) if default_sym in symbols else 0
        )
        contract = c3.text_input(
            "Contract month",
            value=pre.get("contract") or (md.contract or ""),
            help="e.g. MNQZ26 - must match the contract you traded.",
        )
        c4, c5, c6 = st.columns(3)
        direction = c4.selectbox(
            "Direction", ["long", "short"], index=0 if pre.get("direction", "long") == "long" else 1
        )
        qty = c5.number_input("Quantity (contracts)", 1, 1000, 1, step=1)
        spec_r = reg.get(sym)
        tick = spec_r.tick_size
        entry_price = c6.number_input(
            "Actual entry price",
            min_value=0.0,
            value=float(pre.get("entry_price") or 0.0),
            step=tick,
            format=f"%.{spec_r.price_decimals}f",
        )
        c7, c8, c9 = st.columns(3)
        stop = c7.number_input(
            "Initial stop (as placed at broker)",
            min_value=0.0,
            value=float(pre.get("initial_stop") or 0.0),
            step=tick,
            format=f"%.{spec_r.price_decimals}f",
        )
        target = c8.number_input(
            "Planned target (0 = none)",
            min_value=0.0,
            value=float(pre.get("planned_target") or 0.0),
            step=tick,
            format=f"%.{spec_r.price_decimals}f",
        )
        setup_rows = repo.list_setups(db, instrument=sym, states=["qualified"], limit=50)
        setup_opts = {"(none)": None}
        if not setup_rows.empty:
            for r in setup_rows.to_dict("records"):
                setup_opts[
                    f"{r['direction']} {to_display(r['confirm_ts'], tz, '%m-%d %H:%M')} score "
                    f"{r['score']:.0f} ({r['id'][:8]})"
                ] = r["id"]
        pre_setup = next((k for k, v in setup_opts.items() if v == pre.get("setup_id")), "(none)")
        setup_lbl = c9.selectbox(
            "Linked setup (optional)", list(setup_opts), index=list(setup_opts).index(pre_setup)
        )
        entry_ts = _ts_input("Entry", pre.get("entry_ts") or _clock(), "reg_entry")
        c10, c11 = st.columns(2)
        comm = c10.number_input(
            "Commission per contract (round trip, USD)",
            0.0,
            100.0,
            float(s.risk.commission_for(sym)),
            step=0.1,
        )
        slip = c11.number_input(
            "Estimated slippage (ticks per side)", 0.0, 50.0, float(s.risk.slippage_ticks_per_side), step=0.5
        )
        notes = st.text_area("Notes", "")
        tags = st.text_input("Tags (comma separated)", "")
        submitted = st.form_submit_button("Register trade", type="primary")
    if submitted:
        try:
            inp = ManualTradeInput(
                account_id=acct_opts[acct_lbl],
                instrument=sym,
                contract=contract.strip().upper() or None,
                direction=direction,
                entry_ts=entry_ts,
                entry_price=entry_price,
                initial_stop=stop,
                quantity=int(qty),
                planned_target=target or None,
                setup_id=setup_opts[setup_lbl],
                commission_per_contract_rt=comm,
                slippage_ticks=slip,
                notes=notes,
                tags=[x.strip() for x in tags.split(",") if x.strip()],
            )
            tid = register_trade(db, inp)
            st.session_state.pop("prefill_trade", None)
            state.bump_trade_rev()
            st.success(f"Trade #{tid} registered. Guidance is in the 'Open trades & guidance' tab.")
        except TradeValidationError as exc:
            st.error(str(exc))
        except Exception as exc:  # pydantic validation
            st.error(f"Invalid input: {exc}")

# ---------------------------------------------------------------------------
with tab_open:
    trades = repo.list_trades(db, status="open")
    if not trades:
        st.info("No open trades. Register one in the 'Register trade' tab after you place it at your broker.")
    else:
        opts = {
            f"#{t_['id']} {t_['instrument']} {t_['direction']} {t_['quantity']} @ {t_['entry_price']} "
            f"({to_display(t_['entry_ts'], tz, '%m-%d %H:%M')})": t_["id"]
            for t_ in trades
        }
        tid = opts[st.selectbox("Trade", list(opts), key="tm_trade")]
        trade = repo.get_trade(db, tid)
        spec_t = reg.get(trade["instrument"])
        actions = repo.list_actions(db, tid)
        setup = repo.get_setup(db, trade["setup_id"]) if trade.get("setup_id") else None
        tv = trade_view_from_rows(trade, actions, setup)
        mon = None
        if md.ok and md.spec.symbol == trade["instrument"]:
            mon = monitor_open_trades(db, s, md, scan, None).get(tid)
        if mon is None:
            st.warning(
                f"Load {trade['instrument']} market data (sidebar) to evaluate this trade. Without data the "
                "application cannot assess it."
            )
        else:
            if not mon.data_ok:
                st.error(mon.data_message)
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Current price", spec_t.fmt(mon.current_price))
            m2.metric("Current R", ui.rfmt(mon.current_r))
            m3.metric("Best / worst R", f"{mon.mfe_r:+.2f} / {mon.mae_r:+.2f}")
            m4.metric(
                "Your stop",
                spec_t.fmt(mon.registered_stop),
                f"{mon.registered_stop_r:+.2f}R",
                delta_color="off",
                delta_arrow="off",
                help="USER-REPORTED: the last stop you recorded. The app cannot verify your broker stop.",
            )
            m5.metric("Open contracts", mon.open_qty)
            m6.metric(
                "Open P&L (est.)",
                ui.money(mon.unrealized_pnl, signed=True),
                help="From the last completed bar close; excludes commissions.",
            )
            for w in mon.warnings:
                st.caption(w)
            g1, g2 = st.columns([3, 2])
            with g1:
                trail = mon.current_trailing
                lines = {
                    "entry_ts": tv.entry_ts,
                    "entry": tv.entry_price,
                    "stop": mon.registered_stop,
                    "target": tv.planned_target,
                    "trail": trail.suggested_stop if trail else None,
                }
                entry_pos = int(md.visible.index.searchsorted(tv.entry_ts))
                view = md.visible.iloc[max(0, entry_pos - 60) :]
                fig = charts.candle_chart(
                    view if len(view) > 20 else md.visible.iloc[-150:],
                    spec_t,
                    tz,
                    bars=500,
                    zones=scan.zone_history if scan else None,
                    swings=scan.swings if scan else None,
                    trade_lines=lines,
                    height=440,
                    show_vwap=False,
                    title=f"Trade #{tid}",
                )
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
                rp = r_progress_series(tv, md.visible)
                if not rp.empty:
                    st.plotly_chart(
                        charts.line_chart(
                            rp["close_ts"].dt.tz_convert(tz).dt.tz_localize(None),
                            rp["r"],
                            "R",
                            height=200,
                            title="Close-based R since entry",
                            zero_line=True,
                        ),
                        width="stretch",
                        config={"displayModeBar": False},
                    )
            with g2:
                st.markdown("**Breakeven review**")
                st.caption(
                    f"Breakeven incl. estimated costs: {spec_t.fmt(mon.breakeven_price)}. These are review "
                    "prompts at configured milestones - not recommendations to move your stop."
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "threshold": f"+{b['threshold_r']:g}R",
                                "price reached": "yes" if b["reached"] else "no",
                                "trigger time": to_display(b["trigger_ts"], tz, "%m-%d %H:%M")
                                if b["trigger_ts"]
                                else "",
                                "note": "your stop was already at/through BE" if b["already_at_be"] else "",
                            }
                            for b in mon.be_status
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.markdown("**Trailing-stop suggestion**")
                if trail:
                    ui.kv_card(
                        "Latest confirmed-swing suggestion",
                        [
                            ("Suggested level", spec_t.fmt(trail.suggested_stop)),
                            ("Your stop (user-reported)", spec_t.fmt(trail.registered_stop)),
                            (
                                "Swing",
                                f"{trail.swing_type} {spec_t.fmt(trail.swing_price)} confirmed "
                                f"{to_display(trail.confirm_ts, tz, '%H:%M')}",
                            ),
                            ("Buffer", trail.buffer_desc),
                            (
                                "Distance from price",
                                f"{trail.distance_ticks:.0f} ticks / {trail.distance_points:g} pts",
                            ),
                            (
                                "If moved manually",
                                ("locks " if trail.locked_pnl >= 0 else "risks ")
                                + ui.money(abs(trail.locked_pnl)),
                            ),
                        ],
                    )
                    st.caption(trail.reason)
                else:
                    st.caption("No confirmed swing has produced a tighter level yet.")
                if mon.provisional_trailing:
                    st.caption("Provisional: " + mon.provisional_trailing.reason)
                if mon.nearest_opposing_zone:
                    z = mon.nearest_opposing_zone
                    st.markdown(
                        f"**Next opposing zone:** {spec_t.fmt(z['lower'])}-{spec_t.fmt(z['upper'])} "
                        f"(strength {z['strength']:.2f})"
                    )
                if mon.target_distance_ticks is not None:
                    st.markdown(f"**Distance to planned target:** {mon.target_distance_ticks:.0f} ticks")

            st.markdown("#### Alerts for this trade")
            st.caption(
                "Lifecycle: price reached trigger (trigger time) -> alert issued -> you review -> you record "
                "what you actually did. The app never assumes an action was taken."
            )
            talerts = repo.list_alerts(db, trade_id=tid, limit=200)
            if not talerts:
                st.caption("No alerts yet for this trade.")
            show_all = len(talerts) <= 6 or st.toggle(f"Show all {len(talerts)} alerts", key=f"all_al_{tid}")
            for a in talerts if show_all else talerts[:6]:
                cols = st.columns([6, 1, 1])
                status = (
                    "unreviewed"
                    if not a["acknowledged"]
                    else f"{a['user_response']} {to_display(a['acknowledged_at'], tz, '%m-%d %H:%M')}"
                )
                cols[0].markdown(
                    f"**{a['title']}** - trigger {to_display(a['trigger_ts'], tz, '%m-%d %H:%M')} - "
                    f"_{status}_  \n{a['message']}"
                )
                if not a["acknowledged"]:
                    if cols[1].button("Reviewed", key=f"ackr_{a['id']}"):
                        repo.acknowledge_alert(db, a["id"], "reviewed")
                        st.rerun()
                    if cols[2].button("Dismiss", key=f"ackd_{a['id']}"):
                        repo.acknowledge_alert(db, a["id"], "dismissed")
                        st.rerun()

        st.markdown("#### Record a management action you took")
        open_alerts = {"(none)": None} | {
            f"#{a['id']} {a['title'][:60]}": a["id"]
            for a in repo.list_alerts(db, trade_id=tid, acknowledged=False)
        }
        with st.form(f"action_{tid}"):
            a1, a2 = st.columns(2)
            atype = a1.selectbox(
                "What did you do?", list(ACTION_TYPES), format_func=lambda k: ACTION_TYPES[k]
            )
            link = a2.selectbox("In response to alert (optional)", list(open_alerts))
            ats = _ts_input("Action", max(_clock(), tv.entry_ts), f"act_{tid}")
            b1, b2, b3 = st.columns(3)
            aprice = b1.number_input(
                "Fill / reference price (0 = n/a)",
                min_value=0.0,
                value=0.0,
                step=spec_t.tick_size,
                format=f"%.{spec_t.price_decimals}f",
            )
            aqty = b2.number_input("Quantity (partial exit)", 0, 1000, 0, step=1)
            anew = b3.number_input(
                "New stop at broker (stop actions)",
                min_value=0.0,
                value=0.0,
                step=spec_t.tick_size,
                format=f"%.{spec_t.price_decimals}f",
            )
            anotes = st.text_input("Notes")
            sub = st.form_submit_button("Record action", type="primary")
        if sub:
            try:
                aid = record_action(
                    db,
                    TradeActionInput(
                        trade_id=tid,
                        action_type=atype,
                        action_ts=ats,
                        price=aprice or None,
                        quantity=aqty or None,
                        new_stop=anew or None,
                        notes=anotes,
                        alert_id=open_alerts[link],
                    ),
                )
                state.bump_trade_rev()
                st.success(f"Recorded action #{aid}.")
                st.rerun()
            except TradeValidationError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Invalid input: {exc}")

        st.markdown("#### Action history (audit trail)")
        if actions:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "id": a["id"],
                            "action": ACTION_TYPES.get(a["action_type"], a["action_type"]),
                            "time": to_display(a["action_ts"], tz, "%m-%d %H:%M"),
                            "price": a["price"],
                            "qty": a["quantity"],
                            "new stop": a["new_stop"],
                            "notes": a["notes"],
                            "alert": a["alert_id"],
                            "corrects": a["corrects_action_id"],
                            "VOID": "void: " + (a["void_reason"] or "") if a["is_void"] else "",
                            "recorded": to_display(a["recorded_at"], tz, "%m-%d %H:%M"),
                        }
                        for a in actions
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            live_actions = {
                f"#{a['id']} {ACTION_TYPES.get(a['action_type'])} {a['action_ts'][:16]}": a
                for a in actions
                if not a["is_void"]
            }
            if live_actions:
                with st.expander("Correct an erroneous entry"):
                    st.caption(
                        "The original stays in the audit trail marked VOID; the replacement references it."
                    )
                    with st.form(f"correct_{tid}"):
                        which = st.selectbox("Entry to correct", list(live_actions))
                        reason = st.text_input("Reason for correction (required)")
                        replace = st.checkbox("Record a replacement entry")
                        r1, r2, r3 = st.columns(3)
                        rprice = r1.number_input(
                            "Replacement price",
                            min_value=0.0,
                            value=0.0,
                            step=spec_t.tick_size,
                            format=f"%.{spec_t.price_decimals}f",
                        )
                        rqty = r2.number_input("Replacement qty", 0, 1000, 0)
                        rstop = r3.number_input(
                            "Replacement new stop",
                            min_value=0.0,
                            value=0.0,
                            step=spec_t.tick_size,
                            format=f"%.{spec_t.price_decimals}f",
                        )
                        csub = st.form_submit_button("Apply correction")
                    if csub:
                        orig = live_actions[which]
                        try:
                            rep = None
                            if replace:
                                rep = TradeActionInput(
                                    trade_id=tid,
                                    action_type=orig["action_type"],
                                    action_ts=from_iso(orig["action_ts"]),
                                    price=rprice or None,
                                    quantity=rqty or None,
                                    new_stop=rstop or None,
                                    notes=f"Correction of #{orig['id']}: {reason}",
                                )
                            repo.correct_action(db, orig["id"], rep, reason)
                            state.bump_trade_rev()
                            st.success("Correction recorded.")
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            st.error(str(exc))
        else:
            st.caption("No actions recorded yet.")
        with st.expander("Void this trade record (data-entry mistake)"):
            reason = st.text_input("Reason", key=f"void_{tid}")
            if st.button("Void trade", key=f"voidb_{tid}"):
                try:
                    repo.void_trade(db, tid, reason)
                    state.bump_trade_rev()
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

# ---------------------------------------------------------------------------
with tab_risk:
    st.caption(
        "Sizing uses the most restrictive of your risk budget, available drawdown, firm limit and personal "
        "limit. It never increases size after losses or near a target."
    )
    accounts = repo.list_accounts(db, include_closed=False)
    acct_opts = {"(no account)": None} | {f"{a['nickname']} ({a['firm_code']})": a for a in accounts}
    r1, r2, r3 = st.columns(3)
    a_lbl = r1.selectbox("Account (for firm limits)", list(acct_opts), key="rc_acct")
    account = acct_opts[a_lbl]
    syms = reg.symbols()
    rsym = r2.selectbox("Instrument", syms, index=syms.index(md.spec.symbol) if md.ok else 0, key="rc_sym")
    rdir = r3.selectbox("Direction", ["long", "short"], key="rc_dir")
    rs = reg.get(rsym)
    last = float(md.visible["close"].iloc[-1]) if md.ok and md.spec.symbol == rsym else 0.0
    q1, q2, q3 = st.columns(3)
    r_entry = q1.number_input(
        "Entry",
        min_value=0.0,
        value=last,
        step=rs.tick_size,
        format=f"%.{rs.price_decimals}f",
        key="rc_entry",
    )
    default_stop = (last - 20 * rs.tick_size) if rdir == "long" else (last + 20 * rs.tick_size)
    r_stop = q2.number_input(
        "Proposed stop",
        min_value=0.0,
        value=max(default_stop, 0.0),
        step=rs.tick_size,
        format=f"%.{rs.price_decimals}f",
        key="rc_stop",
    )
    r_tgt = q3.number_input(
        "Target (0 = none)",
        min_value=0.0,
        value=0.0,
        step=rs.tick_size,
        format=f"%.{rs.price_decimals}f",
        key="rc_tgt",
    )
    firm_n, personal_n, limit_detail = account_contract_limits(db, account, rs, s)
    dd_default = None
    if account is not None:
        try:
            dash = build_dashboard(db, account, s)
            if dash.progress.drawdown is not None:
                dd_default = float(dash.progress.drawdown.remaining_allowance)
        except Exception:  # noqa: BLE001
            dd_default = None
    w1, w2, w3 = st.columns(3)
    max_risk = w1.number_input(
        "Maximum dollar risk for this trade",
        1.0,
        1e6,
        float(s.risk.personal_max_dollar_risk_per_trade),
        step=25.0,
        key="rc_risk",
    )
    use_dd = w2.checkbox("Apply available drawdown", value=dd_default is not None, key="rc_usedd")
    avail_dd = w2.number_input(
        "Available drawdown (USD)",
        0.0,
        1e7,
        float(dd_default or 0.0),
        step=50.0,
        key="rc_dd",
        help="Defaults to the selected account's ESTIMATED remaining allowance.",
    )
    dd_buf = w3.number_input(
        "Drawdown safety buffer (USD)",
        0.0,
        1e6,
        float(s.risk.drawdown_safety_buffer),
        step=50.0,
        key="rc_buf",
    )
    v1, v2, v3, v4 = st.columns(4)
    firm_in = v1.number_input(
        "Firm max contracts (0 = n/a)", 0, 1000, int(firm_n or 0), key="rc_firm", help=limit_detail
    )
    pers_in = v2.number_input(
        "Personal max contracts (0 = n/a)", 0, 1000, int(personal_n or 0), key="rc_pers"
    )
    comm_in = v3.number_input(
        "Commission / contract RT", 0.0, 100.0, float(s.risk.commission_for(rsym)), step=0.1, key="rc_comm"
    )
    slip_in = v4.number_input(
        "Slippage ticks / side", 0.0, 50.0, float(s.risk.slippage_ticks_per_side), step=0.5, key="rc_slip"
    )
    try:
        res = size_position(
            rs,
            rdir,
            r_entry,
            r_stop,
            max_risk,
            avail_dd if use_dd else None,
            firm_in or None,
            pers_in or None,
            comm_in,
            slip_in,
            r_tgt or None,
            dd_buf,
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Contracts (most restrictive)", res.final_contracts)
        k2.metric(
            "Stop distance",
            f"{res.stop_ticks:.0f} ticks",
            f"{res.stop_points:g} pts",
            delta_color="off",
            delta_arrow="off",
        )
        k3.metric(
            "All-in risk / contract",
            ui.money(res.all_in_risk_per_contract),
            f"stop {ui.money(res.risk_per_contract)} + costs {ui.money(res.cost_per_contract)}",
            delta_color="off",
            delta_arrow="off",
        )
        k4.metric("Total all-in risk", ui.money(res.total_all_in_risk))
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "limit": lc.name,
                        "max contracts": str(lc.contracts) if lc.contracts is not None else "not applied",
                        "basis": lc.detail,
                        "binding": "<- binding" if lc.name in res.binding_constraints else "",
                    }
                    for lc in res.limits
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        extra = []
        if res.remaining_drawdown_after_loss is not None:
            extra.append(
                f"Remaining drawdown after a full stop-out: **{ui.money(res.remaining_drawdown_after_loss)}**"
            )
        if res.potential_profit is not None:
            extra.append(
                f"Potential profit at target (net of costs): **{ui.money(res.potential_profit)}**, "
                f"R:R **{res.rr:.2f}**"
            )
        for e in extra:
            st.markdown(e)
        for w in res.warnings:
            st.warning(w)
        if account is not None:
            st.caption(f"Firm limit source: {limit_detail}. Drawdown is an estimate from your records.")
    except RiskInputError as exc:
        st.error(str(exc))

# ---------------------------------------------------------------------------
with tab_closed:
    closed = repo.list_trades(db, status="closed")
    if not closed:
        st.caption("No closed trades yet. A trade closes when you record exits for the full quantity.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": t_["id"],
                        "instrument": t_["instrument"],
                        "dir": t_["direction"],
                        "qty": t_["quantity"],
                        "entry": t_["entry_price"],
                        "entry time": to_display(t_["entry_ts"], tz, "%m-%d %H:%M"),
                        "closed": to_display(t_["closed_ts"], tz, "%m-%d %H:%M"),
                        "realized P&L": t_["realized_pnl"],
                        "P&L source": t_["pnl_source"],
                        "account": t_["account_id"],
                    }
                    for t_ in closed
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Realized P&L is computed from the exits YOU recorded (less commissions) unless you entered the "
            "broker-statement figure in the Trade Journal."
        )
