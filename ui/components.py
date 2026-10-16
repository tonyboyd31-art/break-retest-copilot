"""Reusable Streamlit display components."""

from __future__ import annotations

import math
from html import escape

import pandas as pd
import streamlit as st

from core.instruments import InstrumentSpec
from core.retest_engine import STATE_LABELS, Setup
from core.timeutils import to_display
from services.market_service import MarketData
from ui import theme as t


def page_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<div class="brc-muted">{escape(subtitle)}</div>', unsafe_allow_html=True)


def safety_banner(extra: str = "") -> None:
    st.markdown(
        '<div class="brc-banner"><b>Analysis only.</b> This application never submits, modifies or cancels orders, '
        "never moves stops and never takes profits. Every proposed level is advisory; every trade decision is yours."
        + (f" {escape(extra)}" if extra else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def money(x: float | None, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{'+' if signed and x > 0 else ''}{'-' if x < 0 else ''}${abs(x):,.2f}"


def pct(x: float | None) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"


def rfmt(x: float | None) -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2f}R"


def data_status(md: MarketData, tz: str, compact: bool = False) -> None:
    feed_lbl, feed_col = t.FEED_BADGE.get(md.feed_type, (md.feed_type.upper(), t.MUTED))
    badges = [t.badge(feed_lbl, feed_col)]
    if md.freshness:
        f_lbl, f_col = t.FRESHNESS_BADGE.get(md.freshness.status, (md.freshness.status, t.MUTED))
        badges.append(t.badge(f_lbl, f_col))
    if md.quality is not None:
        if md.quality.errors:
            badges.append(t.badge("DATA ERRORS", t.CRITICAL))
        elif md.quality.missing_bars:
            badges.append(t.badge(f"{md.quality.missing_bars} MISSING BARS", t.WARNING))
        if not md.quality.volume_reliable:
            badges.append(t.badge("NO RELIABLE VOLUME", t.WARNING))
    clock = "Replay clock" if md.is_replay else "Wall clock"
    info = (
        f"<b>{escape(md.spec.symbol)}</b> {escape(md.contract or 'contract unknown')} &middot; "
        f"{escape(md.source_label)} &middot; last bar closed {escape(to_display(md.as_of, tz))} &middot; "
        f"{clock} {escape(to_display(md.now, tz))}"
    )
    st.markdown(
        f'<div class="brc-card">{"".join(badges)}<div style="margin-top:6px;font-size:0.86rem">{info}</div>'
        + (
            f'<div class="brc-muted" style="margin-top:4px">{escape(md.freshness.message)}</div>'
            if md.freshness and not compact
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    if md.freshness and md.freshness.is_stale:
        st.error(
            "Market data is STALE - new signal qualification is disabled and trade guidance cannot be relied on."
        )
    if md.quality is not None and md.quality.errors:
        st.error(
            "Data failed validation - scanning disabled: " + "; ".join(e.message for e in md.quality.errors)
        )
    if not compact:
        for note in md.notes:
            st.caption(note)


def quality_table(report) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "severity": i.severity,
                "check": i.code,
                "message": i.message,
                "count": i.count,
                "examples": ", ".join(i.examples),
            }
            for i in report.issues
        ]
    )


def plan_rows(s: Setup, spec: InstrumentSpec) -> pd.DataFrame:
    p = s.plan
    if p is None:
        return pd.DataFrame()
    rows = [
        ("Direction", s.direction.upper(), ""),
        ("Zone", f"{spec.fmt(p.zone_lower)} - {spec.fmt(p.zone_upper)}", "Broken level that was retested"),
        ("Breakout close", spec.fmt(p.breakout_price), ""),
        ("Retest extreme", spec.fmt(p.retest_price), ""),
        ("Confirmation close", spec.fmt(p.confirmation_price), ""),
        (
            "Proposed entry zone",
            f"{spec.fmt(p.entry_zone_low)} - {spec.fmt(p.entry_zone_high)}",
            p.explanations.get("entry", ""),
        ),
        ("Reference entry", spec.fmt(p.entry_price), p.entry_method.replace("_", " ")),
        (
            "Structural invalidation",
            spec.fmt(p.structural_invalidation),
            p.explanations.get("invalidation", ""),
        ),
        ("Proposed protective stop", spec.fmt(p.stop), p.explanations.get("stop", "")),
        ("First target", spec.fmt(p.target1), p.explanations.get("target1", "")),
        ("Extended target", spec.fmt(p.target2), p.explanations.get("target2", "")),
        (
            "Risk",
            f"{p.risk_ticks:.0f} ticks / {p.risk_points:g} pts",
            f"${p.risk_per_contract:,.2f} per contract",
        ),
        ("Reward to T1", f"{p.reward_ticks:.0f} ticks" if p.reward_ticks is not None else "-", ""),
        (
            "Planned R:R",
            f"{p.rr:.2f}" if p.rr is not None else "-",
            f"extended {p.rr_extended:.2f}" if p.rr_extended else "",
        ),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value", "Reasoning"])


def setup_summary_row(s: Setup, spec: InstrumentSpec, tz: str) -> dict:
    return {
        "id": s.setup_id,
        "state": STATE_LABELS[s.state],
        "dir": s.direction,
        "zone": f"{spec.fmt(s.zone['lower'])}-{spec.fmt(s.zone['upper'])}",
        "breakout": to_display(s.breakout_ts, tz, "%m-%d %H:%M"),
        "retest": to_display(s.retest_ts, tz, "%m-%d %H:%M") if s.retest_ts is not None else "",
        "confirm": to_display(s.confirm_ts, tz, "%m-%d %H:%M") if s.confirm_ts is not None else "",
        "score": round(s.score.total, 1) if s.score else None,
        "entry": s.plan.entry_price if s.plan else None,
        "stop": s.plan.stop if s.plan else None,
        "T1": s.plan.target1 if s.plan else None,
        "R:R": s.plan.rr if s.plan else None,
        "reason": (s.rejection_reasons[0] if s.rejection_reasons else ""),
    }


def rule_badge(status: str) -> str:
    lbl, col = t.RULE_BADGE.get(status, (status, t.MUTED))
    return t.badge(lbl, col)


def kv_card(title: str, items: list[tuple[str, str]]) -> None:
    body = "".join(f'<div class="k">{escape(k)}</div><div class="v">{escape(v)}</div>' for k, v in items)
    st.markdown(
        f'<div class="brc-card"><b>{escape(title)}</b><div class="brc-kv" style="margin-top:6px">{body}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
