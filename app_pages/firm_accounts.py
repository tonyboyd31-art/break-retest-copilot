"""Firm & Account Manager: Tradeify / Topstep accounts, rule verification, drawdown and payout estimates."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import streamlit as st

from data import repositories as repo
from data.models import AccountInput, DailyPnlInput, PayoutInput
from firms.base import STANDARD_RULE_LABELS
from services.account_service import build_dashboard, drawdown_frame, import_journal_into_daily
from ui import charts, state
from ui import components as ui
from ui import theme as t

s = state.settings()
tz = state.tz()
db = state.db()
firms = state.app().firms
ui.page_header(
    "Firm & Account Manager",
    "Separate, configurable rules per firm and account type. Figures are "
    "ESTIMATES from your own records - never official balances or approvals.",
)
st.markdown(
    '<div class="brc-banner"><b>No firm rule shipped with this app is VERIFIED.</b> The official help '
    "centers could not be fetched from the build environment, so values came from web-search summaries of "
    "the official pages (2026-10-16). Confirm each rule on the official page and mark it USER CONFIRMED. "
    "This app does not guarantee compliance with any firm's rules.</div>",
    unsafe_allow_html=True,
)

tab_acc, tab_dash, tab_rules, tab_proc = st.tabs(
    ["Accounts", "Account dashboard", "Firm rules", "Verification process"]
)

# ---------------------------------------------------------------------------
with tab_acc:
    accounts = repo.list_accounts(db)
    if accounts:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "id": a["id"],
                        "nickname": a["nickname"],
                        "firm": a["firm_code"],
                        "type": a["account_type_code"],
                        "stage": a["stage"],
                        "size": a["account_size"],
                        "start balance": a["starting_balance"],
                        "entered balance": a["current_balance"],
                        "status": a["status"],
                        "drawdown model confirmed": bool(a["drawdown_model_confirmed"]),
                        "rules verified on": a["rules_verified_on"],
                    }
                    for a in accounts
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No accounts yet. Add one below.")

    st.markdown("#### Add account")
    f1, f2 = st.columns(2)
    firm_code = f1.selectbox("Firm", list(firms), format_func=lambda c: firms[c].name, key="na_firm")
    firm = firms[firm_code]
    types = {a.code: a for a in firm.account_types()}
    at_code = f2.selectbox("Account type", list(types), format_func=lambda c: types[c].label, key="na_type")
    at = types[at_code]
    rules = firm.rules_for(at_code, repo.list_rule_rows(db, firm_code, at_code))
    sb_rule = rules.get("starting_balance")
    with st.form("new_account"):
        g1, g2, g3 = st.columns(3)
        nickname = g1.text_input("Nickname", f"{firm.name} {at.label}")
        stage = g2.selectbox(
            "Stage",
            ["evaluation", "funded", "live", "practice"],
            index=["evaluation", "funded", "live", "practice"].index(at.stage)
            if at.stage in ("evaluation", "funded", "live", "practice")
            else 0,
        )
        sb_default = float(sb_rule.value) if sb_rule and sb_rule.value is not None else float(at.account_size)
        start_bal = g3.number_input(
            "Starting balance",
            0.0,
            1e8,
            sb_default,
            step=500.0,
            help="Enter the starting balance shown in YOUR firm dashboard."
            + (" (Rule value not configured - verify!)" if not sb_rule or sb_rule.value is None else ""),
        )
        h1, h2, h3 = st.columns(3)
        cur_bal = h1.number_input("Current balance (user-entered, optional)", 0.0, 1e8, 0.0, step=100.0)
        pdl = h2.number_input("Personal daily loss limit (0 = use global setting)", 0.0, 1e7, 0.0, step=50.0)
        pmc = h3.number_input(
            "Personal max contracts, mini-equivalents (0 = global)", 0.0, 1000.0, 0.0, step=0.5
        )
        notes = st.text_input("Notes")
        if st.form_submit_button("Create account", type="primary"):
            try:
                aid = repo.create_account(
                    db,
                    AccountInput(
                        firm_code=firm_code,
                        account_type_code=at_code,
                        nickname=nickname.strip(),
                        stage=stage,
                        account_size=at.account_size,
                        starting_balance=start_bal,
                        current_balance=cur_bal or None,
                        personal_daily_loss_limit=pdl or None,
                        personal_max_contracts=pmc or None,
                        notes=notes,
                    ),
                )
                st.success(f"Account #{aid} created. Review its rules in the 'Firm rules' tab.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not create account: {exc}")

    if accounts:
        st.markdown("#### Edit account")
        amap = {f"#{a['id']} {a['nickname']}": a for a in accounts}
        a = amap[st.selectbox("Account", list(amap), key="ea_pick")]
        with st.form(f"edit_{a['id']}"):
            e1, e2, e3 = st.columns(3)
            nick = e1.text_input("Nickname", a["nickname"])
            stg = e2.selectbox(
                "Stage",
                ["evaluation", "funded", "live", "practice"],
                index=["evaluation", "funded", "live", "practice"].index(a["stage"]),
            )
            status = e3.selectbox(
                "Status",
                ["active", "passed", "failed", "closed"],
                index=["active", "passed", "failed", "closed"].index(a["status"]),
            )
            e4, e5, e6 = st.columns(3)
            start_b = e4.number_input("Starting balance", 0.0, 1e8, float(a["starting_balance"]), step=100.0)
            cur_b = e5.number_input(
                "Current balance (user-entered)",
                0.0,
                1e8,
                float(a["current_balance"] or 0.0),
                step=100.0,
                help="Not an official balance unless verified from your firm.",
            )
            cyc = e6.date_input(
                "Cycle start / last reset date",
                date.fromisoformat(a["cycle_start_date"]) if a["cycle_start_date"] else None,
            )
            e7, e8, e9 = st.columns(3)
            pdl = e7.number_input(
                "Personal daily loss limit", 0.0, 1e7, float(a["personal_daily_loss_limit"] or 0.0), step=50.0
            )
            pmc = e8.number_input(
                "Personal max contracts (mini-eq.)",
                0.0,
                1000.0,
                float(a["personal_max_contracts"] or 0.0),
                step=0.5,
            )
            rv = e9.date_input(
                "Rules last verified on (by you)",
                date.fromisoformat(a["rules_verified_on"]) if a["rules_verified_on"] else None,
            )
            ddc = st.checkbox(
                "I confirmed that the configured drawdown model (type, amount, lock behaviour) matches "
                "this account's official rules",
                value=bool(a["drawdown_model_confirmed"]),
            )
            notes = st.text_input("Notes", a["notes"] or "")
            if st.form_submit_button("Save changes"):
                repo.update_account(
                    db,
                    a["id"],
                    nickname=nick,
                    stage=stg,
                    status=status,
                    starting_balance=start_b,
                    current_balance=cur_b or None,
                    cycle_start_date=cyc,
                    personal_daily_loss_limit=pdl or None,
                    personal_max_contracts=pmc or None,
                    rules_verified_on=rv,
                    drawdown_model_confirmed=int(ddc),
                    drawdown_confirmed_on=date.today()
                    if ddc and not a["drawdown_model_confirmed"]
                    else a["drawdown_confirmed_on"],
                    notes=notes,
                )
                st.success("Saved.")
                st.rerun()
        with st.expander("Delete account"):
            st.caption(
                "Journal trades linked to it are kept and unlinked; its daily P&L and payout records are "
                "deleted."
            )
            if st.checkbox("I understand", key=f"del_ok_{a['id']}") and st.button(
                "Delete", key=f"del_{a['id']}"
            ):
                repo.delete_account(db, a["id"])
                st.rerun()

# ---------------------------------------------------------------------------
with tab_dash:
    accounts = repo.list_accounts(db)
    if not accounts:
        st.info("Add an account first.")
    else:
        amap = {f"#{a['id']} {a['nickname']}": a for a in accounts}
        a = amap[st.selectbox("Account", list(amap), key="ad_pick")]
        dash = build_dashboard(db, a, s)
        p = dash.progress
        for w in dash.warnings:
            st.warning(w)
        st.caption(p.label)
        m = st.columns(5)
        m[0].metric("Net profit (cycle)", ui.money(p.net_profit, signed=True))
        m[1].metric(
            "Profit target",
            ui.money(p.profit_target),
            help=f"Rule status: {dash.rules.status('profit_target').value if dash.rules.get('profit_target') else 'n/a'}",
        )
        m[2].metric("Remaining to target", ui.money(p.remaining_target))
        m[3].metric("Today's P&L", ui.money(p.today_pnl, signed=True))
        m[4].metric("Largest winning day", ui.money(p.largest_winning_day))
        m2 = st.columns(5)
        c = p.consistency
        m2[0].metric(
            "Consistency ratio",
            ui.pct(c.ratio) if c else "n/a",
            f"limit {c.threshold:.0%}" if c else None,
            delta_color="off",
            delta_arrow="off",
            help=(c.explanation if c else "No consistency rule configured for this account type."),
        )
        m2[1].metric("Winning days", p.winning_days)
        m2[2].metric(
            "Trading days", f"{p.trading_days}" + (f" / {p.min_trading_days}" if p.min_trading_days else "")
        )
        dd = p.drawdown
        m2[3].metric(
            "Drawdown allowance (est.)",
            ui.money(dd.remaining_allowance) if dd else "n/a",
            help=p.drawdown_message if not dd else "; ".join(dd.assumptions),
        )
        m2[4].metric("Est. balance", ui.money(p.current_balance))
        st.markdown(f"**Status:** {p.status}  \n**Next milestone:** {p.next_milestone}")
        for n in p.notes:
            st.caption(n)
        if dd and dd.days:
            fr = drawdown_frame(p)
            st.plotly_chart(charts.balance_floor_chart(fr), width="stretch", config={"displayModeBar": False})
            with st.expander("Drawdown model detail"):
                st.markdown(
                    f"Model **{dd.model}**, max loss {ui.money(dd.max_loss)}, floor now "
                    f"{ui.money(dd.current_floor)}{' (locked)' if dd.locked else ''}."
                )
                for asm in dd.assumptions:
                    st.caption(asm)
                st.dataframe(fr, hide_index=True, width="stretch")
        if c and c.passes is False:
            st.info(c.explanation)

        st.markdown("#### Payout checklist (estimate)")
        if not dash.payout_paths:
            st.caption(
                "No payout rules are configured for this account type (evaluation accounts usually have "
                "none). Add them under Firm rules if they apply."
            )
        for path in dash.payout_paths:
            col = {"appears_eligible": t.GOOD, "not_eligible": t.CRITICAL, "unknown": t.WARNING}[path.overall]
            st.markdown(
                t.badge(path.overall.replace("_", " ").upper(), col) + f" **{path.path}** - est. max payout "
                f"{ui.money(path.estimated_max_payout)} ({path.cap_detail})",
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(
                    [{"requirement": i.name, "status": i.status, "detail": i.detail} for i in path.items]
                ),
                hide_index=True,
                width="stretch",
            )
        if dash.payout_paths:
            for n in dash.payout_notes:
                st.caption(n)
        if dash.interpretation:
            with st.expander("How the rules are interpreted for this account"):
                for n in dash.interpretation:
                    st.markdown(f"- {n}")

        st.markdown("#### Daily P&L records")
        st.caption(
            "Enter each trading day's NET P&L from your firm dashboard. Optional intraday min/max equity lets "
            "the app detect intraday drawdown breaches."
        )
        with st.form(f"pnl_{a['id']}"):
            d1, d2, d3, d4 = st.columns(4)
            pd_date = d1.date_input("Trade date", date.today())
            pd_pnl = d2.number_input("Net P&L (USD)", -1e7, 1e7, 0.0, step=10.0)
            pd_min = d3.number_input("Intraday low equity (0 = unknown)", 0.0, 1e8, 0.0, step=10.0)
            pd_max = d4.number_input("Intraday high equity (0 = unknown)", 0.0, 1e8, 0.0, step=10.0)
            pd_notes = st.text_input("Notes", key=f"pdn_{a['id']}")
            if st.form_submit_button("Save day"):
                try:
                    repo.upsert_daily_pnl(
                        db,
                        DailyPnlInput(
                            account_id=a["id"],
                            trade_date=pd_date,
                            net_pnl=pd_pnl,
                            min_equity=pd_min or None,
                            max_equity=pd_max or None,
                            notes=pd_notes,
                        ),
                    )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        rows = repo.list_daily_pnl(db, a["id"])
        if rows:
            st.dataframe(
                pd.DataFrame(rows)[["trade_date", "net_pnl", "min_equity", "max_equity", "source", "notes"]],
                hide_index=True,
                width="stretch",
            )
            x1, x2 = st.columns(2)
            del_day = x1.selectbox("Delete a day", [r["trade_date"] for r in rows], key=f"dd_{a['id']}")
            if x1.button("Delete selected day", key=f"ddb_{a['id']}"):
                repo.delete_daily_pnl(db, a["id"], date.fromisoformat(del_day))
                st.rerun()
        if st.button("Import realized P&L from closed journal trades", key=f"imp_{a['id']}"):
            n = import_journal_into_daily(db, a["id"])
            st.success(f"Imported {n} day(s) from journal trades (existing days were not overwritten).")
            st.rerun()

        st.markdown("#### Payouts / withdrawals")
        with st.form(f"payout_{a['id']}"):
            p1, p2, p3 = st.columns(3)
            po_date = p1.date_input("Requested on", date.today())
            po_amt = p2.number_input("Amount", 0.0, 1e7, 0.0, step=50.0)
            po_status = p3.selectbox("Status", ["requested", "paid", "denied", "cancelled"])
            if st.form_submit_button("Record payout"):
                if po_amt <= 0:
                    st.error("Amount must be positive.")
                else:
                    repo.add_payout(
                        db,
                        PayoutInput(
                            account_id=a["id"], requested_on=po_date, amount=po_amt, status=po_status
                        ),
                    )
                    st.rerun()
        pos = repo.list_payouts(db, a["id"])
        if pos:
            st.dataframe(pd.DataFrame(pos)[["requested_on", "amount", "status", "notes"]], hide_index=True)
            st.caption(
                "Recording a payout starts a new payout cycle and, where the rules say so, locks the drawdown "
                "floor. Denied/cancelled payouts are ignored."
            )

# ---------------------------------------------------------------------------
with tab_rules:
    r1, r2 = st.columns(2)
    fc = r1.selectbox("Firm", list(firms), format_func=lambda c: firms[c].name, key="fr_firm")
    firm = firms[fc]
    types = {a.code: a for a in firm.account_types()}
    tc = r2.selectbox("Account type", list(types), format_func=lambda c: types[c].label, key="fr_type")
    st.caption(firm.verification_summary)
    rows_db = repo.list_rule_rows(db, fc, tc)
    rs = firm.rules_for(tc, rows_db)
    table = []
    for key, rv in rs.rules.items():
        val = rv.value
        val_s = (
            json.dumps(val)
            if isinstance(val, (dict, list))
            else ("not configured" if val is None else str(val))
        )
        table.append(
            {
                "rule": rv.label,
                "key": key,
                "value": val_s,
                "status": rv.status.value,
                "origin": "your confirmation" if rv.origin == "user" else "shipped config",
                "verified on": rv.verified_on.isoformat() if rv.verified_on else "",
                "source": rv.source_url or "",
                "notes": rv.notes,
            }
        )
    st.dataframe(
        pd.DataFrame(table)[["rule", "status", "value", "origin", "verified on", "source", "notes", "key"]],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "source": st.column_config.LinkColumn("source", width="medium"),
            "value": st.column_config.TextColumn("value", width="medium"),
            "notes": st.column_config.TextColumn("notes", width="large"),
        },
    )

    st.markdown("#### Confirm or override a rule")
    st.caption(
        "Open the source link, check the value for YOUR account, then record it here. Your entry takes "
        "precedence over the shipped configuration and is kept in the audit log."
    )
    keys = list(rs.rules) + [k for k in STANDARD_RULE_LABELS if k not in rs.rules]
    with st.form("rule_override"):
        o1, o2 = st.columns(2)
        key = o1.selectbox("Rule", keys, format_func=lambda k: f"{STANDARD_RULE_LABELS.get(k, k)} ({k})")
        scope = o2.selectbox(
            "Applies to",
            [tc, "*"],
            format_func=lambda x: "this account type" if x == tc else "all account types of this firm",
        )
        cur = rs.get(key)
        default_val = json.dumps(cur.value) if cur and cur.value is not None else ""
        val_raw = st.text_area(
            "Value (number, text, true/false, or JSON for structured rules)",
            default_val,
            help='Examples: 3000 | {"type": "eod_trailing", "lock_floor_offset": 100, '
            '"lock_on_payout": true} | {"formula": "largest_day_over_total_profit", '
            '"threshold": 0.4}',
        )
        o3, o4, o5 = st.columns(3)
        status = o3.selectbox("Status", ["USER_CONFIRMED", "UNVERIFIED", "CONFLICTING"])
        src = o4.text_input("Official source URL", cur.source_url if cur and cur.source_url else "")
        ver = o5.date_input("Verified on", date.today())
        notes = st.text_input("Notes / interpretation")
        if st.form_submit_button("Save rule", type="primary"):
            try:
                value = json.loads(val_raw) if val_raw.strip() else None
            except json.JSONDecodeError:
                value = val_raw.strip()
            if status == "USER_CONFIRMED" and not src.strip():
                st.error("A USER_CONFIRMED rule needs the official source URL you checked.")
            else:
                repo.set_rule_override(
                    db,
                    fc,
                    scope,
                    key,
                    value,
                    status,
                    src.strip() or None,
                    ver if status == "USER_CONFIRMED" else None,
                    notes,
                )
                st.success("Rule saved.")
                st.rerun()
    user_rows = [r for r in rows_db if r["origin"] == "user"]
    if user_rows:
        with st.expander("Your confirmations / overrides"):
            st.dataframe(
                pd.DataFrame(user_rows)[
                    ["account_type_code", "rule_key", "value", "status", "verified_on", "source_url", "notes"]
                ],
                hide_index=True,
                width="stretch",
            )
            ck = st.selectbox(
                "Clear an override", [f"{r['account_type_code']}|{r['rule_key']}" for r in user_rows]
            )
            if st.button("Clear override"):
                atc, rk = ck.split("|")
                repo.clear_rule_override(db, fc, atc, rk)
                st.rerun()

# ---------------------------------------------------------------------------
with tab_proc:
    st.markdown("""
**How to verify firm rules (recommended monthly, and before every new account):**

1. Open the official source link for each rule (Firm rules tab). Use the firm's own help center or rules page -
   not third-party comparison sites.
2. Check the value **for your exact account type and purchase date** (firms keep legacy parameters for older accounts).
3. Record it in *Confirm or override a rule* with status **USER_CONFIRMED**, the URL and today's date.
4. Confirm the drawdown model for each account (type, amount, lock behaviour) in *Accounts -> Edit account*.
5. Unconfirmed rules stay labelled **UNVERIFIED** or **CONFLICTING**, and every figure that depends on them is
   labelled an estimate. Rules older than the staleness window (Settings) raise an "outdated rules" alert.

**What the app does with firm rules:** risk limits, drawdown/consistency/payout *estimates* and warnings.
**What it never does:** change signal generation, guarantee compliance, or label a payout approved.

**Adding another firm:** copy `firms/configurations/topstep.json` to `<firm>.json`, edit the account types and
rules, and restart. A Python subclass is only needed for custom interpretation notes.
""")
