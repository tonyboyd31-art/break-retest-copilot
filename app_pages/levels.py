"""Support & Resistance: automatically detected swing points and zones, with explainable strength."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.swing_detection import params_from_settings, provisional_swings
from core.timeutils import to_display
from ui import charts, state
from ui import components as ui

s = state.settings()
tz = state.tz()
ui.page_header(
    "Support & Resistance",
    "Price zones where the market has turned before - built only from swings that were confirmed at the time.",
    eyebrow="Price map",
    accent="Resistance",
)

ctx = state.market_context()
md, scan = ctx.md, ctx.scan
if not md.ok:
    st.error(md.error or "No market data.")
    st.stop()
ui.data_status(md, tz, compact=True)
if scan is None or not scan.scanned:
    st.error(scan.message if scan else "Scan unavailable.")
    st.stop()
spec = md.spec
last_close = float(md.visible["close"].iloc[-1])

c1, c2, c3, c4 = st.columns(4)
alive = scan.zones
c1.metric("Live zones", len(alive))
c2.metric("Relevant for breakouts", sum(1 for z in alive if z["relevant"]))
c3.metric(
    "Confirmed swings", int((scan.swings["status"] == "confirmed").sum()) if not scan.swings.empty else 0
)
c4.metric("Last close", spec.fmt(last_close))

o1, o2, o3 = st.columns(3)
show_dead = o1.toggle("Show invalidated / expired zones", value=False, key="lv_dead")
show_sw = o2.toggle("Show swing points", value=True, key="lv_sw")
bars = o3.select_slider("Bars shown", [200, 400, 600, 900, 1200, 2000], value=600, key="lv_bars")
fig = charts.candle_chart(
    md.visible,
    spec,
    tz,
    bars=bars,
    zones=scan.zone_history,
    swings=scan.swings,
    indicators=scan.indicators,
    show_vwap=False,
    show_swings=show_sw,
    show_dead_zones=show_dead,
    title=f"{spec.symbol} zones as of {to_display(md.as_of, tz)}",
)
st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})
st.caption(
    "Styles: solid aqua = support, solid red = resistance, dashed = broken level (possible role flip), "
    "amber = broken level that was retested and held, dotted gray = invalidated/expired."
)

tab_z, tab_detail, tab_sw, tab_cfg = st.tabs(["Zones", "Zone detail", "Swings", "Parameters & method"])
with tab_z:
    rows = []
    for z in sorted(alive, key=lambda z: -z["upper"]):
        rows.append(
            {
                "zone": z["zone_id"],
                "kind": z["kind"].replace("_", " "),
                "lower": z["lower"],
                "upper": z["upper"],
                "strength": round(z["strength"], 3),
                "reactions": z["reactions"],
                "tests": z["tests"],
                "holds": z["holds"],
                "failures": z["failures"],
                "broken": z["break_direction"] or "",
                "retested": z["retested"],
                "dist (ticks)": round(z["distance"] / spec.tick_size, 1),
                "dist (ATR)": None if z["distance_atr"] is None else round(z["distance_atr"], 2),
                "relevant": z["relevant"],
                "why": z["relevance_reason"],
                "last reaction": to_display(z["last_reaction_ts"], tz, "%m-%d %H:%M"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=420)
    st.caption(
        "Distance is from the last close to the nearest zone edge. 'Relevant' zones can generate breakout "
        "setups; established zones also act as obstacles for targets."
    )

with tab_detail:
    pool = scan.zone_history if show_dead else alive
    if not pool:
        st.caption("No zones.")
    else:
        lab = {
            f"#{z['zone_id']} {z['kind']} {spec.fmt(z['lower'])}-{spec.fmt(z['upper'])} (strength "
            f"{z['strength']:.2f})": z
            for z in sorted(pool, key=lambda z: -z["strength"])
        }
        z = lab[st.selectbox("Zone", list(lab), key="lv_zone")]
        w = s.zones
        parts = z["strength_parts"]
        weights = {
            "reactions": w.w_reactions,
            "recency": w.w_recency,
            "rejection": w.w_rejection,
            "holds": w.w_holds,
        }
        tw = sum(weights.values())
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "component": k,
                        "sub-score (0-1)": parts[k],
                        "weight": round(weights[k] / tw, 3),
                        "contribution": round(parts[k] * weights[k] / tw, 4),
                        "definition": {
                            "reactions": f"min(distinct reactions / {w.reactions_saturation}, 1)",
                            "recency": f"0.5 ^ (bars since last reaction / {w.recency_half_life_bars})",
                            "rejection": f"min(mean member prominence in ATR / {w.rejection_saturation_atr}, 1)",
                            "holds": "(holds + 1) / (holds + failures + 2)",
                        }[k],
                    }
                    for k in parts
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown(
            f"**Strength {z['strength']:.3f}** = weighted sum above. Member swing prices: "
            + ", ".join(spec.fmt(p) for p in z["member_prices"])
        )
        st.markdown(
            f"Status **{z['status']}**"
            + (
                f", broken {z['break_direction']} at {to_display(z['broken_ts'], tz)}"
                if z["broken_ts"]
                else ""
            )
            + (f", ended: {z['end_reason']}" if z.get("end_reason") else "")
        )

with tab_sw:
    sw = scan.swings
    if sw.empty:
        st.caption("No swings detected.")
    else:
        recent = sw.sort_values("confirm_ts", ascending=False).head(150)
        st.dataframe(
            pd.DataFrame(
                {
                    "type": recent["swing_type"],
                    "price": recent["price"],
                    "pivot bar": [to_display(t_, tz, "%m-%d %H:%M") for t_ in recent["pivot_ts"]],
                    "confirmed at": [to_display(t_, tz, "%m-%d %H:%M") for t_ in recent["confirm_ts"]],
                    "prominence": recent["prominence"].round(2),
                    "threshold": recent["threshold"].round(2),
                    "status": recent["status"],
                }
            ),
            hide_index=True,
            width="stretch",
            height=380,
        )
        prov = provisional_swings(md.visible, params_from_settings(s.swings))
        if not prov.empty:
            st.markdown(
                "**Provisional (unconfirmed) pivots at the right edge** - display only, never alerted:"
            )
            st.dataframe(
                prov.assign(pivot_ts=[to_display(t_, tz, "%m-%d %H:%M") for t_ in prov["pivot_ts"]]),
                hide_index=True,
            )

with tab_cfg:
    st.markdown(f"""
- **Swing rule:** pivot high = strictly above the previous {s.swings.left_bars} highs and at least the next
  {s.swings.right_bars} highs (lows mirrored). Confirmed only after the {s.swings.right_bars} later bars complete.
- **Minimum prominence:** max({s.swings.min_prominence_ticks:g} ticks{", " + str(s.swings.atr_prominence_mult) + " x ATR(" + str(s.swings.atr_period) + ")" if s.swings.use_atr_prominence else ""}).
  Same-type swings within {s.swings.duplicate_window_bars} bars and {s.swings.duplicate_tolerance_ticks:g} ticks are suppressed.
- **Clustering tolerance:** max({s.zones.tolerance_ticks:g} ticks, {s.zones.tolerance_atr_mult} x ATR); merged zones
  stay narrower than {s.zones.max_zone_width_atr} ATR. Swings closer than {s.zones.min_reaction_gap_bars} bars count as one reaction.
- **Break:** close beyond the far edge by max({s.zones.break_buffer_ticks:g} ticks, {s.zones.break_buffer_atr_mult} x ATR).
- **Relevance:** >= {s.zones.min_reactions} reactions, last reaction within {s.zones.max_age_bars} bars, within
  {s.zones.relevance_max_distance_atr} ATR of price. Obstacles for targets need >= {s.zones.opposing_min_reactions} reactions.

All parameters are editable under **Settings -> Strategy**.
""")
