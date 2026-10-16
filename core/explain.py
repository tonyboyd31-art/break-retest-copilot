"""Deterministic plain-English explanations of setups.

Built from stored engine values with templates - no language model is
involved in any number. (An optional Claude-based narrative lives in
``services.explainer`` and only rephrases these facts.)
"""

from __future__ import annotations

from core.instruments import InstrumentSpec
from core.retest_engine import STATE_LABELS, Setup, SetupState
from core.timeutils import to_display


def explain_setup(s: Setup, spec: InstrumentSpec, tz: str = "UTC") -> str:
    long = s.is_long

    def when(ts, fmt: str = "%H:%M %Z") -> str:
        return to_display(ts, tz, fmt)

    side = "resistance" if long else "support"
    z = s.zone
    lines = [
        f"**{STATE_LABELS[s.state]}** - {spec.symbol} {s.direction.upper()} break-and-retest of the "
        f"{side} zone {spec.fmt(z['lower'])}-{spec.fmt(z['upper'])} ({z.get('reactions', '?')} reaction(s), "
        f"zone strength {z.get('strength', 0):.2f} when broken).",
    ]
    bb = s.breakout_bar
    lines.append(
        f"1. **Breakout** at {when(s.breakout_ts, '%Y-%m-%d %H:%M %Z')}: a completed candle closed at "
        f"{spec.fmt(bb['c'])}, {'above' if long else 'below'} the zone."
    )
    if s.breakout_eval:
        failed = [c.message for c in s.breakout_eval.checks if c.mode == "require" and c.passed is False]
        unavailable = [c.name for c in s.breakout_eval.checks if c.mode == "unavailable"]
        if failed:
            lines.append("   - Failed: " + " ".join(failed))
        else:
            lines.append("   - Candle-quality checks passed (body, size, extension, room to the next zone).")
        if unavailable:
            lines.append(f"   - Not evaluated (data unavailable): {', '.join(unavailable)}.")
    if s.retest_ts is not None:
        lines.append(
            f"2. **Retest** at {when(s.retest_ts)}: price came back to {spec.fmt(s.retest_price)}; best "
            f"rejection wick {s.best_wick_ratio:.0%} of the candle range."
        )
    if s.confirm_ts is not None:
        lines.append(
            f"3. **Confirmation** at {when(s.confirm_ts)}: close {spec.fmt(s.confirm_price)} back "
            f"{'above' if long else 'below'} the zone with a {'bullish' if long else 'bearish'} body."
        )
    if s.plan:
        p = s.plan
        if p.structural_valid:
            lines.append(
                f"4. **Proposed plan (advisory)**: entry ~{spec.fmt(p.entry_price)}, stop {spec.fmt(p.stop)} "
                f"({p.risk_ticks:.0f} ticks, ${p.risk_per_contract:,.2f}/contract), T1 {spec.fmt(p.target1)} "
                f"(R:R {p.rr:.2f})" + (f", T2 {spec.fmt(p.target2)}" if p.target2 else "") + "."
            )
        else:
            lines.append("4. **No valid trade plan**: " + " ".join(p.invalid_reasons))
    if s.score:
        top = sorted((c for c in s.score.components if c.available), key=lambda c: c.points, reverse=True)
        weak = [c for c in s.score.components if c.available and (c.raw or 0) < 0.5]
        lines.append(
            f"5. **Quality score {s.score.total:.0f}/100** (a ranking, not a win probability). Strongest: "
            + ", ".join(f"{c.label} {c.points:.0f}/{c.weight:.0f}" for c in top[:3])
            + "."
        )
        if weak:
            lines.append("   - Weak factors: " + ", ".join(f"{c.label} ({c.explanation})" for c in weak[:3]))
        if s.score.missing:
            lines.append("   - " + " ".join(s.score.notes))
    if s.state in (SetupState.REJECTED, SetupState.EXPIRED, SetupState.INVALIDATED) and s.rejection_reasons:
        lines.append("**Why it did not qualify:** " + " ".join(dict.fromkeys(s.rejection_reasons)))
    lines.append("_All levels are suggestions for manual review. No order has been or will be placed._")
    return "\n".join(lines)
