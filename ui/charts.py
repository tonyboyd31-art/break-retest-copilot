"""Plotly figures. Presentation only - every number comes from the engine.

Conventions: one y-axis per chart (never dual axes), 2px lines, >= 8px
markers, hairline solid grid, legend for >= 2 series, text in ink colours.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.instruments import InstrumentSpec
from core.timeutils import timeframe_delta
from ui import theme as t

ZONE_STYLE = {
    # kind: (fill colour, fill opacity, line colour, dash, legend label)
    "support": (t.AQUA, 0.16, t.AQUA, "solid", "Support zone"),
    "resistance": (t.RED, 0.16, t.RED, "solid", "Resistance zone"),
    "broken_resistance": (t.RED, 0.06, t.RED, "dash", "Broken resistance (now support?)"),
    "broken_support": (t.AQUA, 0.06, t.AQUA, "dash", "Broken support (now resistance?)"),
    "retest": (t.YELLOW, 0.18, t.YELLOW, "solid", "Retested flip zone"),
    "invalidated": (t.MUTED, 0.07, t.MUTED, "dot", "Invalidated / expired zone"),
}


def _local(ts: pd.Timestamp | pd.DatetimeIndex, tz: str):
    return ts.tz_convert(tz).tz_localize(None)


def base_layout(
    fig: go.Figure, height: int = 420, title: str | None = None, legend: bool = True
) -> go.Figure:
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=t.SURFACE,
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=t.INK_2),
        margin=dict(l=10, r=10, t=40 if title else 16, b=10),
        showlegend=legend,
        # With a (left-aligned) title, the legend sits top-right so the two never collide.
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right" if title else "left",
            x=1 if title else 0,
            font=dict(size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(
            text=title,
            font=dict(family="Cormorant Garamond, Georgia, serif", size=17, color=t.INK),
            x=0,
            xanchor="left",
        )
        if title
        else None,
        hoverlabel=dict(bgcolor=t.SURFACE_2, bordercolor=t.AXIS, font=dict(color=t.INK)),
    )
    fig.update_xaxes(
        gridcolor=t.GRID, gridwidth=1, linecolor=t.AXIS, zeroline=False, tickfont=dict(color=t.MUTED)
    )
    fig.update_yaxes(
        gridcolor=t.GRID, gridwidth=1, linecolor=t.AXIS, zeroline=False, tickfont=dict(color=t.MUTED)
    )
    return fig


def _rangebreaks(idx: pd.DatetimeIndex, tf: pd.Timedelta, tz: str) -> list[dict]:
    """Hide missing 5-minute slots (halts, weekends) so candles are contiguous."""
    if len(idx) < 2:
        return []
    values = []
    diffs = idx[1:] - idx[:-1]
    for pos in np.where(diffs > tf)[0]:
        gap = pd.date_range(idx[pos] + tf, idx[pos + 1] - tf, freq=tf)
        values.extend(_local(gap, tz).strftime("%Y-%m-%d %H:%M:%S").tolist())
    if not values:
        return []
    return [dict(values=values, dvalue=int(tf.total_seconds() * 1000))]


def candle_chart(
    df: pd.DataFrame,
    spec: InstrumentSpec,
    tz: str,
    timeframe: str = "5min",
    bars: int = 600,
    zones: list[dict] | None = None,
    swings: pd.DataFrame | None = None,
    setups: list | None = None,
    selected_setup=None,
    indicators: pd.DataFrame | None = None,
    show_vwap: bool = True,
    show_swings: bool = True,
    show_dead_zones: bool = False,
    trade_lines: dict | None = None,
    height: int = 560,
    title: str | None = None,
) -> go.Figure:
    tf = timeframe_delta(timeframe)
    focus_ts = None
    if selected_setup is not None:
        focus_ts = selected_setup.confirm_ts or selected_setup.breakout_ts
    if focus_ts is not None and len(df) and focus_ts - tf >= df.index[0]:
        pos = int(df.index.searchsorted(focus_ts - tf))
        end = min(len(df), pos + max(int(bars * 0.35), 30))
        view = df.iloc[max(0, end - bars) : end]
    else:
        view = df.iloc[-bars:]
    fig = go.Figure()
    if view.empty:
        return base_layout(fig, height, title or "No data")
    x = _local(view.index, tz)
    w_start, w_end = view.index[0], view.index[-1] + tf
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name="Price",
            increasing=dict(line=dict(color=t.AQUA, width=1), fillcolor=t.AQUA),
            decreasing=dict(line=dict(color=t.RED, width=1), fillcolor=t.RED),
            showlegend=False,
            hoverinfo="x+y",
        )
    )

    # zones: draw each zone's life inside the window with the role it had at the time; the style switches at
    # the break. A zone that died later is still drawn for the part of its life inside the window.
    used_kinds: set[str] = set()
    close_end = float(view["close"].iloc[-1])
    for z in zones or []:
        created = z.get("created_ts")
        ended = z.get("ended_ts")
        if created is None or created > w_end or (ended is not None and ended < w_start):
            continue
        dead_in_window = ended is not None and ended <= w_end
        if dead_in_window and not show_dead_zones:
            continue
        start = max(created, w_start)
        stop = min(ended, w_end) if ended is not None else w_end
        if start >= stop:
            continue
        broken = z.get("broken_ts")
        up = z.get("break_direction") == "up"
        retest_ts = z.get("retest_ts")
        if broken is not None:
            pre_kind = "resistance" if up else "support"
            post_kind = (
                "retest"
                if z.get("retested") and retest_ts is not None and retest_ts <= stop
                else ("broken_resistance" if up else "broken_support")
            )
        else:
            if z.get("kind") in ("support", "resistance"):
                pre_kind = z["kind"]
            elif z.get("origin") in ("high", "low"):
                pre_kind = "resistance" if z["origin"] == "high" else "support"
            else:
                pre_kind = "support" if z["mid"] < close_end else "resistance"
            post_kind = pre_kind
        segs = []
        if broken is not None and start < broken < stop:
            segs = [(start, broken, pre_kind), (broken, stop, post_kind)]
        elif broken is not None and broken <= start:
            segs = [(start, stop, post_kind)]
        else:
            segs = [(start, stop, pre_kind)]
        for s0, s1, kind in segs:
            if dead_in_window:
                kind = "invalidated"
            fill, op, line, dash, _lbl = ZONE_STYLE.get(kind, ZONE_STYLE["support"])
            used_kinds.add(kind)
            fig.add_shape(
                type="rect",
                x0=_local(s0, tz),
                x1=_local(s1, tz),
                y0=z["lower"],
                y1=z["upper"],
                fillcolor=fill,
                opacity=op if dash == "solid" else min(op * 2, 0.25),
                line=dict(color=line, width=1, dash=dash),
                layer="below",
            )
    for kind in ZONE_STYLE:
        if kind in used_kinds:
            fill, _op, line, dash, label = ZONE_STYLE[kind]
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=label,
                    marker=dict(symbol="square", size=10, color=fill, line=dict(color=line, width=1)),
                )
            )

    if show_vwap and indicators is not None and "vwap" in indicators and indicators["vwap"].notna().any():
        v = indicators["vwap"].reindex(view.index)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=v,
                mode="lines",
                name="VWAP",
                line=dict(color=t.VIOLET, width=2),
                hovertemplate="VWAP %{y:.2f}<extra></extra>",
            )
        )

    if show_swings and swings is not None and not swings.empty:
        sw = swings[
            (swings["status"] == "confirmed")
            & (swings["pivot_ts"] >= w_start)
            & (swings["confirm_ts"] <= w_end)
        ]
        for kind, sym, col in (("high", "triangle-down", t.ORANGE), ("low", "triangle-up", t.BLUE)):
            s = sw[sw["swing_type"] == kind]
            if s.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=_local(pd.DatetimeIndex(s["pivot_ts"]), tz),
                    y=s["price"],
                    mode="markers",
                    name=f"Swing {kind} (confirmed)",
                    marker=dict(symbol=sym, size=9, color=col, line=dict(color=t.SURFACE, width=2)),
                    customdata=np.stack(
                        [
                            _local(pd.DatetimeIndex(s["confirm_ts"]), tz).strftime("%m-%d %H:%M"),
                            s["prominence"].round(2),
                        ],
                        axis=1,
                    ),
                    hovertemplate=f"Swing {kind} %{{y}}<br>pivot %{{x|%m-%d %H:%M}}<br>confirmed %{{customdata[0]}}"
                    "<br>prominence %{customdata[1]}<extra></extra>",
                )
            )

    if setups:
        marks = {
            "Breakout": ([], [], [], "diamond", t.BLUE),
            "Retest": ([], [], [], "circle", t.YELLOW),
            "Confirmation": ([], [], [], "star", t.ORANGE),
        }
        for s in setups:
            if s.breakout_ts - tf < w_start or s.breakout_ts > w_end:
                continue
            lbl = f"{s.direction} {s.state.value}" + (f" score {s.score.total:.0f}" if s.score else "")
            marks["Breakout"][0].append(_local(s.breakout_ts - tf, tz))
            marks["Breakout"][1].append(s.breakout_bar["c"])
            marks["Breakout"][2].append(lbl)
            if s.retest_ts is not None and s.retest_ts <= w_end:
                marks["Retest"][0].append(_local(s.retest_ts - tf, tz))
                marks["Retest"][1].append(s.retest_price)
                marks["Retest"][2].append(lbl)
            if s.confirm_ts is not None and s.confirm_ts <= w_end:
                marks["Confirmation"][0].append(_local(s.confirm_ts - tf, tz))
                marks["Confirmation"][1].append(s.confirm_price)
                marks["Confirmation"][2].append(lbl)
        for name, (xs, ys, txt, sym, col) in marks.items():
            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        name=name,
                        text=txt,
                        marker=dict(symbol=sym, size=11, color=col, line=dict(color=t.SURFACE, width=2)),
                        hovertemplate=f"{name} %{{y}}<br>%{{text}}<extra></extra>",
                    )
                )

    def hline(
        y: float | None,
        start: pd.Timestamp,
        color: str,
        label: str,
        dash: str = "solid",
        end: pd.Timestamp | None = None,
        label_left: bool = False,
    ) -> None:
        if y is None or (isinstance(y, float) and math.isnan(y)):
            return
        x0 = _local(max(start, w_start), tz)
        x1 = _local(min(end, w_end) if end is not None else w_end, tz)
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(color=color, width=2, dash=dash))
        fig.add_annotation(
            x=x0 if label_left else x1,
            y=y,
            text=f"{label} {spec.fmt(y)}",
            showarrow=False,
            xanchor="right" if label_left else "right",
            yanchor="middle" if label_left else "bottom",
            font=dict(size=11, color=t.INK),
            bgcolor="rgba(13,17,23,0.75)",
        )

    if (
        selected_setup is not None
        and selected_setup.plan is not None
        and selected_setup.confirm_ts is not None
    ):
        p = selected_setup.plan
        st0 = selected_setup.confirm_ts - tf
        until = selected_setup.confirm_ts + 100 * tf
        hline(p.entry_price, st0, t.BLUE, "Entry", end=until)
        hline(p.stop, st0, t.CRITICAL, "SL", end=until)
        hline(p.structural_invalidation, st0, t.MUTED, "Inv", "dot", end=until, label_left=True)
        hline(p.target1, st0, t.GOOD, "T1", end=until)
        hline(p.target2, st0, t.GOOD, "T2", "dot", end=until)
        plan_levels = [
            v for v in (p.entry_price, p.stop, p.structural_invalidation, p.target1, p.target2) if v
        ]
    else:
        plan_levels = []
    if trade_lines:
        plan_levels += [
            v
            for v in (
                trade_lines.get("entry"),
                trade_lines.get("stop"),
                trade_lines.get("target"),
                trade_lines.get("trail"),
            )
            if v
        ]
        st0 = trade_lines["entry_ts"]
        hline(trade_lines.get("entry"), st0, t.BLUE, "Your entry")
        hline(trade_lines.get("stop"), st0, t.CRITICAL, "Your stop (user-reported)")
        hline(trade_lines.get("target"), st0, t.GOOD, "Your target")
        if trade_lines.get("trail"):
            hline(trade_lines["trail"], st0, t.YELLOW, "Suggested trail", "dash")

    lo_y = min([float(view["low"].min())] + plan_levels)
    hi_y = max([float(view["high"].max())] + plan_levels)
    pad = (hi_y - lo_y) * 0.06 or spec.tick_size * 10
    base_layout(fig, height, None)
    if title:
        fig.add_annotation(
            text=title,
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=12, color=t.INK_2),
            bgcolor="rgba(15,20,26,0.7)",
        )
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        hovermode="x",
        margin=dict(l=10, r=10, t=70, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis=dict(
            rangebreaks=_rangebreaks(view.index, tf, tz),
            showspikes=True,
            spikemode="across",
            spikethickness=1,
            spikecolor=t.AXIS,
        ),
        yaxis=dict(
            showspikes=True, spikethickness=1, spikecolor=t.AXIS, side="right", range=[lo_y - pad, hi_y + pad]
        ),
    )
    return fig


def line_chart(
    xs,
    ys,
    name: str,
    color: str = t.BLUE,
    height: int = 260,
    title: str | None = None,
    y_title: str | None = None,
    fill: bool = False,
    zero_line: bool = False,
) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            fill="tozeroy" if fill else None,
            fillcolor="rgba(57,135,229,0.10)" if fill else None,
        )
    )
    if zero_line:
        fig.add_hline(y=0, line=dict(color=t.AXIS, width=1))
    base_layout(fig, height, title, legend=False)
    fig.update_layout(hovermode="x unified", yaxis_title=y_title)
    return fig


def balance_floor_chart(frame: pd.DataFrame, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["balance"],
            name="Balance (from your records)",
            mode="lines+markers",
            line=dict(color=t.BLUE, width=2),
            marker=dict(size=8, line=dict(color=t.SURFACE, width=2)),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frame["date"],
            y=frame["floor"],
            name="Drawdown floor (estimate)",
            mode="lines",
            line=dict(color=t.CRITICAL, width=2, shape="hv"),
        )
    )
    base_layout(fig, height, "Balance vs drawdown floor")
    fig.update_layout(hovermode="x unified", yaxis_title="USD")
    return fig


def histogram(values: pd.Series, title: str, x_title: str, bins: int = 20, height: int = 260) -> go.Figure:
    v = pd.Series(values).dropna().astype(float)
    fig = go.Figure()
    if not v.empty:
        counts, edges = np.histogram(v, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        colors = [t.BLUE if c >= 0 else t.RED for c in centers]
        fig.add_trace(
            go.Bar(
                x=centers,
                y=counts,
                marker=dict(color=colors, line=dict(width=0)),
                width=(edges[1] - edges[0]) * 0.85,
                name=x_title,
                hovertemplate=f"{x_title} %{{x:.2f}}<br>trades %{{y}}<extra></extra>",
            )
        )
    base_layout(fig, height, title, legend=False)
    fig.update_layout(xaxis_title=x_title, yaxis_title="Trades", bargap=0.1)
    return fig


def bar_breakdown(df: pd.DataFrame, by: str, metric: str, title: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    if not df.empty and metric in df:
        vals = df[metric].fillna(0).astype(float)
        fig.add_trace(
            go.Bar(
                y=df[by].astype(str),
                x=vals,
                orientation="h",
                marker=dict(color=[t.BLUE if v >= 0 else t.RED for v in vals]),
                width=0.6,
                text=[f"{v:.2f} ({n})" for v, n in zip(vals, df["trades"], strict=False)],
                textposition="outside",
                textfont=dict(color=t.INK_2),
                hovertemplate=f"%{{y}}: %{{x:.3f}}<extra>{metric}</extra>",
            )
        )
    base_layout(fig, height, title, legend=False)
    fig.add_vline(x=0, line=dict(color=t.AXIS, width=1))
    return fig


def score_bars(components: list, height: int = 260) -> go.Figure:
    """Bullet-style: weight as a track, points earned as the fill (unavailable factors flagged)."""
    labels = [c.label if hasattr(c, "label") else c["factor"] for c in components]
    weights = [c.weight if hasattr(c, "weight") else c["weight"] for c in components]
    pts = [c.points if hasattr(c, "points") else c["points"] for c in components]
    avail = [c.available if hasattr(c, "available") else bool(c["available"]) for c in components]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=weights,
            orientation="h",
            name="Weight",
            marker=dict(color="#1d2733"),
            width=0.6,
            hovertemplate="%{y}: weight %{x}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=pts,
            orientation="h",
            name="Points earned",
            marker=dict(color=[t.BLUE if a else t.MUTED for a in avail]),
            width=0.36,
            text=[
                f"{p:.1f}/{w:g}" if a else "n/a (missing data)"
                for p, w, a in zip(pts, weights, avail, strict=True)
            ],
            textposition="outside",
            textfont=dict(color=t.INK_2),
            hovertemplate="%{y}: %{x:.1f} points<extra></extra>",
        )
    )
    base_layout(fig, height, None)
    fig.update_layout(
        barmode="overlay", yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(weights + [1]) * 1.35])
    )
    return fig


def heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    z: str,
    title: str,
    height: int = 300,
    diverging: bool = True,
    text_col: str | None = None,
) -> go.Figure:
    pv = df.pivot_table(index=y, columns=x, values=z, aggfunc="first")
    txt = None
    if text_col:
        tv = df.pivot_table(index=y, columns=x, values=text_col, aggfunc="first")
        txt = [
            [
                (f"{pv.iloc[i, j]:.2f}" if pd.notna(pv.iloc[i, j]) else "-") + f"<br>n={int(tv.iloc[i, j])}"
                for j in range(pv.shape[1])
            ]
            for i in range(pv.shape[0])
        ]
    if diverging:
        scale = [[0, "#b13b3b"], [0.5, "#383835"], [1, "#1c5cab"]]
        zmax = float(np.nanmax(np.abs(pv.to_numpy()))) if np.isfinite(pv.to_numpy()).any() else 1.0
        kw = dict(zmid=0, zmin=-zmax, zmax=zmax)
    else:
        scale = [[0, "#184f95"], [1, "#9ec5f4"]]
        kw = {}
    fig = go.Figure(
        go.Heatmap(
            z=pv.to_numpy(),
            x=[str(c) for c in pv.columns],
            y=[str(i) for i in pv.index],
            colorscale=scale,
            text=txt,
            texttemplate="%{text}" if txt else None,
            textfont=dict(color=t.INK, size=11),
            hovertemplate=f"{x} %{{x}}<br>{y} %{{y}}<br>{z} %{{z:.3f}}<extra></extra>",
            xgap=2,
            ygap=2,
            **kw,
        )
    )
    base_layout(fig, height, title, legend=False)
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    return fig
