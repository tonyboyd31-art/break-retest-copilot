"""Live Market Signals: detected setups, qualified signals and proposed (advisory) trade plans."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.explain import explain_setup
from core.retest_engine import STATE_LABELS, SetupState
from core.timeutils import to_display
from services import explainer
from ui import charts, state
from ui import components as ui

s = state.settings()
tz = state.tz()
ui.page_header(
    "Live Market Signals",
    "Break-and-retest setups detected on completed 5-minute candles, with transparent scoring.",
)
ui.safety_banner("A qualified signal is a candidate for YOUR review - not a trade, and not an instruction.")

ctx = state.market_context()
md, scan = ctx.md, ctx.scan
if not md.ok:
    st.error(md.error or "No market data.")
    st.stop()
ui.data_status(md, tz)
if scan is None or not scan.scanned:
    st.error(scan.message if scan else "Scan unavailable.")
    st.stop()
spec = md.spec

# --- auto refresh / auto-advance ------------------------------------------------
with st.expander("Auto-refresh", expanded=False):
    if md.is_replay:
        auto = st.toggle(
            "Auto-advance the replay clock",
            key="auto_adv",
            help="Steps the replay forward one bar at a time so you can watch setups evolve.",
        )
        speed = st.select_slider("Seconds per bar", [2, 3, 5, 10], value=3, key="auto_speed")
        if auto:

            @st.fragment(run_every=speed)
            def _advance() -> None:
                n = len(md.segment)
                cur = st.session_state.get("replay_bar")
                pos = (n - 1) if cur is None else int(cur)
                if pos < n - 1:
                    nxt = pos + 1
                    st.session_state["replay_bar"] = None if nxt >= n - 1 else nxt
                    st.rerun(scope="app")
                else:
                    st.caption("Replay reached the latest bar.")

            if st.session_state.get("replay_bar") is None:
                st.info("The replay is at the latest bar. Move the replay clock back in the sidebar first.")
            else:
                _advance()
    else:
        live = st.toggle("Refresh every 60 seconds", key="auto_live")
        if live:

            @st.fragment(run_every=60)
            def _refresh() -> None:
                st.rerun(scope="app")

            _refresh()

qualified = sorted(scan.by_state(SetupState.QUALIFIED), key=lambda x: x.confirm_ts, reverse=True)
pending = [x for x in scan.setups if not x.is_terminal]
others = sorted(
    [x for x in scan.setups if x.state in (SetupState.REJECTED, SetupState.EXPIRED, SetupState.INVALIDATED)],
    key=lambda x: x.last_update_ts,
    reverse=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Qualified", len(qualified))
c2.metric("Pending (awaiting retest/confirmation)", len(pending))
c3.metric("Rejected / expired / invalidated", len(others))
c4.metric(
    "Minimum score",
    f"{s.scoring.min_score:g}/100",
    help="Change in Settings. Score is a ranking, not a probability.",
)

# --- chart ------------------------------------------------------------------------
options = {"(none)": None}
for x in qualified + pending + others[:60]:
    options[
        f"{STATE_LABELS[x.state]} | {x.direction} | {to_display(x.breakout_ts, tz, '%m-%d %H:%M')} | "
        f"{spec.fmt(x.zone['lower'])}-{spec.fmt(x.zone['upper'])}"
        + (f" | score {x.score.total:.0f}" if x.score else "")
    ] = x.setup_id
default_label = next(iter(list(options)[1:2]), "(none)")
cc1, cc2, cc3, cc4 = st.columns([3, 1, 1, 1])
pick = cc1.selectbox(
    "Highlight setup on chart", list(options), index=list(options).index(default_label), key="ls_pick"
)
bars = cc2.select_slider(
    "Bars shown", [200, 400, 600, 900, 1200], value=min(s.data.chart_bars, 1200), key="ls_bars"
)
show_swings = cc3.toggle("Swings", value=True, key="ls_sw")
show_dead = cc4.toggle("Dead zones", value=False, key="ls_dead")
selected = scan.setup(options[pick]) if options[pick] else None
fig = charts.candle_chart(
    md.visible,
    spec,
    tz,
    bars=bars,
    zones=scan.zone_history,
    swings=scan.swings,
    setups=qualified + pending + others,
    selected_setup=selected,
    indicators=scan.indicators,
    show_swings=show_swings,
    show_dead_zones=show_dead,
    title=f"{spec.symbol} {md.contract or ''} 5-minute",
)
st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
st.caption(
    "Markers are drawn on the bar that produced each event; an event is known only at that bar's close. "
    "Swings appear at their pivot but are only known at their confirmation time (see hover)."
)

tab_q, tab_p, tab_r, tab_how = st.tabs(
    [
        f"Qualified ({len(qualified)})",
        f"Pending ({len(pending)})",
        f"Rejected / expired ({len(others)})",
        "How qualification works",
    ]
)


def _setup_detail(x) -> None:
    left, right = st.columns([3, 2])
    with left:
        facts = explain_setup(x, spec, tz)
        st.markdown(facts)
        ok, why = explainer.available()
        if ok and st.button(
            "Rephrase in plain English (Claude)",
            key=f"nar_{x.setup_id}",
            help="Optional. Claude only rephrases the facts above; it never computes levels or "
            "decides whether a setup qualifies.",
        ):
            res = explainer.narrate(facts)
            (st.info if res.ok else st.warning)((res.text + "\n\n" if res.ok else "") + f"_{res.detail}_")
        if x.plan is not None:
            st.markdown("**Proposed trade plan (advisory - requires manual review)**")
            st.dataframe(ui.plan_rows(x, spec), hide_index=True, width="stretch")
    with right:
        if x.score is not None:
            st.markdown(
                f"**Quality score {x.score.total:.1f}/100** - policy: {x.score.policy.replace('_', ' ')}, "
                f"{x.score.available_weight:g}/100 weight with data"
            )
            st.plotly_chart(
                charts.score_bars(x.score.components),
                width="stretch",
                config={"displayModeBar": False},
                key=f"sb_{x.setup_id}",
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "factor": c.label,
                            "points": c.points,
                            "weight": c.weight,
                            "raw": c.raw,
                            "explanation": c.explanation,
                        }
                        for c in x.score.components
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        if x.breakout_eval is not None:
            st.markdown("**Breakout checks**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "check": c.name,
                            "mode": c.mode,
                            "result": "n/a" if c.passed is None else ("pass" if c.passed else "FAIL"),
                            "value": str(c.value),
                            "threshold": str(c.threshold),
                            "detail": c.message,
                        }
                        for c in x.breakout_eval.checks
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        st.markdown("**State history**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "state": STATE_LABELS[h.state],
                        "time": to_display(h.ts, tz, "%m-%d %H:%M"),
                        "note": h.note,
                    }
                    for h in x.history
                ]
            ),
            hide_index=True,
            width="stretch",
        )


with tab_q:
    if not qualified:
        st.info(
            "No qualified setups in this window. Lowering the minimum score in Settings will show more "
            "candidates - but fewer, better setups is the intent."
        )
    stale = md.freshness is not None and md.freshness.is_stale
    for x in qualified:
        p = x.plan
        head = (
            f"{x.direction.upper()} {spec.symbol} | score {x.score.total:.0f} | entry {spec.fmt(p.entry_price)} "
            f"stop {spec.fmt(p.stop)} T1 {spec.fmt(p.target1)} | R:R {p.rr:.2f} | confirmed "
            f"{to_display(x.confirm_ts, tz, '%m-%d %H:%M')}"
        )
        with st.expander(head, expanded=(x is qualified[0])):
            if stale:
                st.warning("Data is stale - this signal cannot be considered current.")
            _setup_detail(x)
            b1, b2 = st.columns(2)
            if b1.button(
                "Register a trade I placed from this setup", key=f"reg_{x.setup_id}", type="primary"
            ):
                st.session_state["prefill_trade"] = {
                    "instrument": spec.symbol,
                    "contract": md.contract,
                    "direction": x.direction,
                    "entry_price": p.entry_price,
                    "initial_stop": p.stop,
                    "planned_target": p.target1,
                    "setup_id": x.setup_id,
                    "entry_ts": x.confirm_ts,
                }
                st.switch_page("app_pages/trade_manager.py")
            b2.caption("Registering records a trade YOU executed at your broker. The app places nothing.")

with tab_p:
    if not pending:
        st.caption("No setups are pending at the current clock.")
    else:
        st.dataframe(
            pd.DataFrame([ui.setup_summary_row(x, spec, tz) for x in pending]),
            hide_index=True,
            width="stretch",
        )
        for x in pending:
            with st.expander(
                f"{STATE_LABELS[x.state]} - {x.direction} {spec.fmt(x.zone['lower'])}-"
                f"{spec.fmt(x.zone['upper'])}"
            ):
                _setup_detail(x)

with tab_r:
    if others:
        df = pd.DataFrame([ui.setup_summary_row(x, spec, tz) for x in others])
        st.dataframe(df, hide_index=True, width="stretch", height=380)
        ids = {f"{r['state']} {r['dir']} {r['breakout']} {r['zone']}": r["id"] for r in df.to_dict("records")}
        lbl = st.selectbox("Inspect", list(ids), key="ls_inspect")
        if lbl:
            _setup_detail(scan.setup(ids[lbl]))
    else:
        st.caption("Nothing rejected yet.")

with tab_how:
    st.markdown(f"""
**States:** level identified -> breakout detected -> awaiting retest -> retest detected -> awaiting confirmation ->
**qualified** or **rejected**; any pending setup can become **expired** or **invalidated**.

1. **Breakout** - a completed candle closes beyond a relevant zone (>= {s.zones.min_reactions} reactions) by
   the buffer, with a directional body >= {s.breakout.min_body_ratio:.0%} of range, size {s.breakout.min_range_atr}-
   {s.breakout.max_range_atr} ATR, extension <= {s.breakout.max_extension_atr} ATR and >= {s.breakout.min_room_atr}
   ATR of room to the next established opposing zone.
2. **Retest** - within {s.retest.max_bars_to_retest} bars, price returns to within the retest tolerance of the zone.
3. **Confirmation** - within {s.retest.max_bars_to_confirm} bars of the retest, a candle closes back beyond the zone
   with a directional body{", after a rejection wick >= " + format(s.retest.min_wick_ratio, ".0%") if s.retest.require_rejection_wick else ""}.
4. **Plan & score** - a structurally valid plan with R:R >= {s.planning.min_rr:g}, score >= {s.scoring.min_score:g},
   inside the session windows {s.sessions.signal_windows if s.sessions.signal_windows_enabled else "(disabled)"}.

**Invalidation:** closing back through the zone, running more than {s.retest.max_run_atr} ATR away before
confirmation, missing candles, or the session ending. Each setup produces each alert at most once.
""")
