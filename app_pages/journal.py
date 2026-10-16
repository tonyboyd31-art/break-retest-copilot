"""Trade Journal: the signal journal (every detected setup) and the executed-trade journal (your trades)."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from core.retest_engine import STATE_LABELS, SetupState
from core.timeutils import to_display
from data import repositories as repo
from data.models import ACTION_TYPES
from ui import components as ui
from ui import state

s = state.settings()
tz = state.tz()
db = state.db()
ui.page_header(
    "Trade Journal",
    "Signals and executed trades are separate records. A hypothetical signal outcome is "
    "never account profit.",
)

tab_sig, tab_exec, tab_audit = st.tabs(["Signal journal", "Executed trades", "Audit log"])

with tab_sig:
    contexts = repo.list_setup_contexts(db)
    f1, f2, f3, f4 = st.columns(4)
    ctx = f1.selectbox("Data context", ["(all)"] + contexts, key="sj_ctx")
    states = f2.multiselect(
        "States",
        [st_.value for st_ in SetupState if st_ != SetupState.LEVEL_IDENTIFIED],
        default=["qualified", "rejected"],
        format_func=lambda v: STATE_LABELS[SetupState(v)],
        key="sj_states",
    )
    min_sc = f3.number_input("Min score", 0.0, 100.0, 0.0, step=5.0, key="sj_min")
    direction = f4.selectbox("Direction", ["(all)", "long", "short"], key="sj_dir")
    df = repo.list_setups(
        db, None if ctx == "(all)" else ctx, states=states or None, min_score=min_sc if min_sc > 0 else None
    )
    if not df.empty and direction != "(all)":
        df = df[df["direction"] == direction]
    if df.empty:
        st.info(
            "No setups recorded with these filters. Setups are recorded automatically when the dashboard scans "
            "market data (including rejected ones)."
        )
    else:
        st.caption(
            f"{len(df)} setup(s). Context shows the data source - 'synthetic' setups come from SIMULATED data."
        )
        view = df.copy()
        for c in ("breakout_ts", "confirm_ts"):
            view[c] = view[c].dt.tz_convert(tz).dt.strftime("%m-%d %H:%M")
        view["reason"] = view["rejection_reasons"].map(lambda r: r[0] if r else "")
        st.dataframe(
            view[
                [
                    "id",
                    "context",
                    "instrument",
                    "direction",
                    "state",
                    "breakout_ts",
                    "confirm_ts",
                    "score",
                    "entry",
                    "stop",
                    "target1",
                    "rr",
                    "reason",
                ]
            ],
            hide_index=True,
            width="stretch",
            height=360,
        )
        sid = st.selectbox("Open setup", df["id"].tolist(), key="sj_open")
        rec = repo.get_setup(db, sid)
        if rec:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(
                    f"**{rec['instrument']} {rec['direction'].upper()} - "
                    f"{STATE_LABELS[SetupState(rec['state'])]}** (data: {rec['data_source']}, "
                    f"{rec['feed_type']}; settings {rec['settings_hash']})"
                )
                st.markdown(
                    "Swing level / zone: "
                    + (
                        f"{rec['zone']['lower']} - {rec['zone']['upper']} "
                        f"({rec['zone'].get('reactions')} reactions)"
                        if rec.get("zone")
                        else "-"
                    )
                )
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "state": STATE_LABELS[SetupState(h["state"])],
                                "time": to_display(h["ts"], tz, "%m-%d %H:%M"),
                                "note": h["note"],
                            }
                            for h in rec.get("history") or []
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                if rec.get("plan"):
                    p = rec["plan"]
                    st.markdown(
                        f"Proposed: entry **{p['entry_price']}**, stop **{p['stop']}**, T1 **{p['target1']}**, "
                        f"T2 **{p['target2']}**, R:R **{p['rr']}** - advisory only."
                    )
                if rec.get("rejection_reasons"):
                    st.markdown(
                        "**Rejection / end reasons:** " + " ".join(dict.fromkeys(rec["rejection_reasons"]))
                    )
                if rec.get("breakout_eval"):
                    with st.expander("Breakout details"):
                        st.dataframe(
                            ui.check_rows(rec["breakout_eval"]["checks"]), hide_index=True, width="stretch"
                        )
            with c2:
                if rec["components"]:
                    st.markdown(f"**Score {rec['score']}** (stored as computed at signal time)")
                    st.dataframe(
                        pd.DataFrame(rec["components"])[
                            ["factor", "points", "weight", "available", "explanation"]
                        ],
                        hide_index=True,
                        width="stretch",
                    )
                o = rec.get("outcome")
                st.markdown("**Subsequent hypothetical outcome**")
                if o:
                    st.markdown(
                        f"{o['outcome']} - R {o['r_result']} (MFE {o['mfe_r']}, MAE {o['mae_r']}), evaluated "
                        f"through {to_display(o['evaluated_through_ts'], tz, '%m-%d %H:%M')}. "
                        "_HYPOTHETICAL - not an executed trade._"
                    )
                else:
                    st.caption("Not evaluated. Use Analytics -> Signal outcomes.")

with tab_exec:
    trades = repo.list_trades(db, include_void=True)
    if not trades:
        st.info("No trades registered yet (Trade Manager -> Register trade).")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": t_["id"],
                        "instrument": t_["instrument"],
                        "contract": t_["contract"],
                        "dir": t_["direction"],
                        "qty": t_["quantity"],
                        "entry": t_["entry_price"],
                        "stop": t_["initial_stop"],
                        "entry time": to_display(t_["entry_ts"], tz, "%m-%d %H:%M"),
                        "status": "VOID" if t_["is_void"] else t_["status"],
                        "realized P&L": t_["realized_pnl"],
                        "P&L source": t_["pnl_source"],
                        "tags": ", ".join(t_["tags"]),
                        "setup": (t_["setup_id"] or "")[:8],
                    }
                    for t_ in trades
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        tmap = {
            f"#{t_['id']} {t_['instrument']} {t_['direction']} {t_['entry_ts'][:16]}": t_ for t_ in trades
        }
        tr = tmap[st.selectbox("Edit journal entry", list(tmap), key="tj_pick")]
        with st.form(f"journal_{tr['id']}"):
            j1, j2 = st.columns(2)
            notes = j1.text_area("Notes", tr["notes"] or "")
            mistakes = j2.text_area("Mistakes", tr["mistakes"] or "")
            j3, j4 = st.columns(2)
            adherence = j3.selectbox(
                "Rule adherence",
                ["", "followed plan", "partially followed", "broke rules"],
                index=["", "followed plan", "partially followed", "broke rules"].index(
                    tr["rule_adherence"] or ""
                )
                if (tr["rule_adherence"] or "") in ["", "followed plan", "partially followed", "broke rules"]
                else 0,
            )
            emotions = j4.text_input("Emotional / process observations (optional)", tr["emotions"] or "")
            j5, j6 = st.columns(2)
            tags = j5.text_input("Tags (comma separated)", ", ".join(tr["tags"]))
            shots = j6.text_area(
                "Screenshot references (one path/URL per line)", "\n".join(tr["screenshot_refs"])
            )
            j7, j8 = st.columns(2)
            pnl_override = j7.number_input(
                "Actual realized P&L from broker statement (0 = keep computed)",
                -1e7,
                1e7,
                float(tr["realized_pnl"] or 0.0) if tr["pnl_source"] in ("user_entered", "verified") else 0.0,
                step=1.0,
            )
            reason = j8.text_input("Reason for edit (kept in audit log)")
            if st.form_submit_button("Save journal entry"):
                fields = {
                    "notes": notes,
                    "mistakes": mistakes,
                    "rule_adherence": adherence or None,
                    "emotions": emotions or None,
                    "tags_json": json.dumps([x.strip() for x in tags.split(",") if x.strip()]),
                    "screenshot_refs_json": json.dumps([x.strip() for x in shots.splitlines() if x.strip()]),
                }
                if pnl_override:
                    fields.update(realized_pnl=float(pnl_override), pnl_source="user_entered")
                repo.update_trade_fields(db, tr["id"], reason=reason, **fields)
                st.success("Saved.")
                st.rerun()
        acts = repo.list_actions(db, tr["id"])
        st.markdown("**Recorded management actions**")
        if acts:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "id": a["id"],
                            "action": ACTION_TYPES.get(a["action_type"]),
                            "time": to_display(a["action_ts"], tz, "%m-%d %H:%M"),
                            "price": a["price"],
                            "qty": a["quantity"],
                            "new stop": a["new_stop"],
                            "notes": a["notes"],
                            "void": a["void_reason"] if a["is_void"] else "",
                        }
                        for a in acts
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("None.")

with tab_audit:
    ents = st.multiselect(
        "Entities",
        [
            "trade",
            "trade_action",
            "account",
            "account_daily_pnl",
            "account_payout",
            "rule",
            "alert",
            "settings",
            "dataset",
            "backtest",
            "instrument",
        ],
        default=[],
    )
    rows = []
    for e in ents or [None]:
        rows += repo.list_audit(db, entity=e, limit=300)
    if rows:
        st.dataframe(
            pd.DataFrame(rows).sort_values("id", ascending=False),
            hide_index=True,
            width="stretch",
            height=420,
        )
    else:
        st.caption("No audit events.")
