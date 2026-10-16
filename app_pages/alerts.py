"""Alerts Center: de-duplicated alerts, acknowledgement, delivery status and channel configuration."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import CategorySetting
from core.events import ALERT_CATEGORIES, SEVERITIES
from core.timeutils import to_display
from data import repositories as repo
from notifications.telegram import TelegramNotifier
from ui import components as ui
from ui import state
from ui import theme as t

s = state.settings()
tz = state.tz()
db = state.db()
ui.page_header(
    "Alerts Center",
    "Every alert is stored once, with when it fired, where it was sent and how you responded.",
    eyebrow="Notifications",
    accent="Alerts",
)

state.market_context()  # make sure the latest scan has been processed

tab_list, tab_cfg = st.tabs(["Alerts", "Channels & categories"])

with tab_list:
    f1, f2, f3 = st.columns(3)
    status = f1.selectbox("Status", ["Unacknowledged", "All", "Acknowledged"], key="al_status")
    cats = f2.multiselect(
        "Categories", list(ALERT_CATEGORIES), format_func=lambda c: ALERT_CATEGORIES[c][0], key="al_cats"
    )
    sevs = f3.multiselect("Severity", list(SEVERITIES), key="al_sev")
    ack = {"Unacknowledged": False, "All": None, "Acknowledged": True}[status]
    alerts = repo.list_alerts(
        db, acknowledged=ack, categories=cats or None, severities=sevs or None, limit=300
    )
    unacked = [a for a in alerts if not a["acknowledged"]]
    if unacked and st.button(f"Mark all {len(unacked)} shown as reviewed"):
        for a in unacked:
            repo.acknowledge_alert(db, a["id"], "reviewed", "bulk")
        st.rerun()
    if not alerts:
        st.info(
            "No alerts match. New alerts appear when a scan sees a new event at the current clock (historical "
            "events are journaled, not alerted)."
        )
    for a in alerts:
        col = t.SEVERITY_COLOR.get(a["severity"], t.MUTED)
        with st.container(border=True):
            c1, c2 = st.columns([5, 2])
            c1.markdown(
                t.badge(a["severity"].upper(), col)
                + t.badge(ALERT_CATEGORIES.get(a["category"], (a["category"],))[0], t.MUTED)
                + f" **{a['title']}**",
                unsafe_allow_html=True,
            )
            c1.markdown(a["message"])
            meta = [
                f"triggered {to_display(a['trigger_ts'], tz)}",
                f"issued {to_display(a['created_at'], tz)}",
            ]
            if a["instrument"]:
                meta.append(a["instrument"])
            if a["price"] is not None:
                meta.append(f"price {a['price']}")
            if a["setup_id"]:
                meta.append(f"setup {a['setup_id'][:8]}")
            if a["trade_id"]:
                meta.append(f"trade #{a['trade_id']}")
            deliveries = ", ".join(f"{d['channel']}: {d['status']}" for d in a["deliveries"]) or "none"
            c1.caption(" | ".join(meta) + f" | delivery - {deliveries}")
            if a["acknowledged"]:
                c2.caption(
                    f"{a['user_response']} at {to_display(a['acknowledged_at'], tz)}"
                    + (f" (action #{a['linked_action_id']})" if a["linked_action_id"] else "")
                )
            else:
                if c2.button("Reviewed", key=f"rv_{a['id']}", width="stretch"):
                    repo.acknowledge_alert(db, a["id"], "reviewed")
                    st.rerun()
                if c2.button("Dismiss", key=f"ds_{a['id']}", width="stretch"):
                    repo.acknowledge_alert(db, a["id"], "dismissed")
                    st.rerun()

with tab_cfg:
    tg = TelegramNotifier()
    st.markdown("#### Channels")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        t.badge("IN-DASHBOARD: ON", t.GOOD)
        + "<br><span class='brc-muted'>Always stored and shown here, plus "
        "toasts while the dashboard is open.</span>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        t.badge(
            "TELEGRAM: " + ("CONFIGURED" if tg.configured else "NOT CONFIGURED"),
            t.GOOD if tg.configured else t.MUTED,
        )
        + f"<br><span class='brc-muted'>{tg.describe()} Untested "
        "end-to-end in development.</span>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        t.badge(
            "BROWSER: " + ("ON" if s.notifications.browser_enabled else "OFF"),
            t.GOOD if s.notifications.browser_enabled else t.MUTED,
        )
        + "<br><span class='brc-muted'>Experimental - browsers may block notifications from embedded frames.</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Secrets are read from environment variables (.env) and are never shown or stored in the database."
    )
    if tg.configured and st.button("Send Telegram test message"):
        res = tg.send("Break & Retest Copilot test message - analysis only; no orders are placed.")
        (st.success if res.status == "sent" else st.error)(f"{res.status}: {res.detail}")

    with st.form("notif_cfg"):
        n1, n2 = st.columns(2)
        tg_on = n1.toggle("Send Telegram for enabled categories", value=s.notifications.telegram_enabled)
        br_on = n2.toggle("Browser notifications (experimental)", value=s.notifications.browser_enabled)
        st.markdown("**Categories** - enable, severity, and whether to send to Telegram")
        rows = [
            {
                "category": k,
                "label": ALERT_CATEGORIES[k][0],
                "enabled": v.enabled,
                "severity": v.severity,
                "telegram": v.telegram,
            }
            for k, v in s.notifications.categories.items()
        ]
        edited = st.data_editor(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            disabled=["category", "label"],
            column_config={"severity": st.column_config.SelectboxColumn(options=list(SEVERITIES))},
        )
        stale_days = st.number_input(
            "Warn when confirmed firm rules are older than (days)",
            1,
            3650,
            s.notifications.rule_staleness_days,
        )
        if st.form_submit_button("Save notification settings", type="primary"):
            cats_new = {
                r["category"]: CategorySetting(
                    enabled=bool(r["enabled"]), severity=r["severity"], telegram=bool(r["telegram"])
                )
                for r in edited.to_dict("records")
            }
            try:
                new = s.model_copy(deep=True)
                new = new.updated(
                    "notifications",
                    telegram_enabled=tg_on,
                    browser_enabled=br_on,
                    rule_staleness_days=int(stale_days),
                    categories={k: v.model_dump() for k, v in cats_new.items()},
                )
                state.save_settings(new)
                st.success("Saved.")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
