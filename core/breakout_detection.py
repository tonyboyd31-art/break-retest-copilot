"""Breakout-candle evaluation.

A zone break is first detected by :class:`core.support_resistance.ZoneTracker`
(a completed close beyond the zone edge plus the break buffer). This module
decides whether that break is a *quality* breakout worth tracking for a
retest. Every check is recorded with its measured value and threshold so the
UI can show exactly why a breakout passed or failed.

Check modes:
* ``require`` - failing rejects the breakout.
* ``score``   - informational; feeds the quality score but does not reject.
* ``unavailable`` - the input does not exist (e.g. no reliable volume). An
  unavailable check never passes silently.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from core.config import BreakoutSettings


@dataclass
class Check:
    name: str
    passed: bool | None  # None = unavailable
    mode: str  # require | score | unavailable
    value: float | str | None
    threshold: float | str | None
    message: str

    def __post_init__(self) -> None:
        # numpy.bool_ is not ``True``/``False`` by identity; normalise.
        if self.passed is not None:
            self.passed = bool(self.passed)
        if hasattr(self.value, "item"):
            self.value = self.value.item()  # type: ignore[union-attr]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class BreakoutEvaluation:
    direction: str
    passed: bool
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, float | None] = field(default_factory=dict)

    @property
    def failed_required(self) -> list[Check]:
        return [c for c in self.checks if c.mode == "require" and c.passed is False]


def _nan(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def evaluate_breakout(
    direction: str,
    zone_lower: float,
    zone_upper: float,
    o: float,
    h: float,
    low: float,
    c: float,
    atr: float,
    tick_size: float,
    cfg: BreakoutSettings,
    vwap: float | None = None,
    rvol: float | None = None,
    htf_alignment: float | None = None,
    next_opposing_edge: float | None = None,
) -> BreakoutEvaluation:
    """Evaluate a breakout candle (direction 'long' = break up through resistance)."""
    long = direction == "long"
    checks: list[Check] = []
    metrics: dict[str, float | None] = {}

    if _nan(atr) or atr <= 0:
        checks.append(
            Check(
                "atr_available",
                False,
                "require",
                None,
                None,
                "ATR unavailable (indicator warm-up) - candle quality cannot be normalised.",
            )
        )
        return BreakoutEvaluation(direction, False, checks, metrics)

    rng = h - low
    body = abs(c - o)
    edge = zone_upper if long else zone_lower
    close_dist = (c - edge) if long else (edge - c)
    buffer = max(cfg.close_buffer_ticks * tick_size, cfg.close_buffer_atr_mult * atr)
    body_ratio = body / rng if rng > 0 else 0.0
    range_atr = rng / atr
    extension_atr = close_dist / atr
    close_pos = ((c - low) / rng if long else (h - c) / rng) if rng > 0 else 0.0
    metrics.update(
        close_distance_ticks=close_dist / tick_size,
        close_distance_atr=extension_atr,
        body_ratio=body_ratio,
        range_atr=range_atr,
        close_position=close_pos,
        atr=atr,
    )

    checks.append(
        Check(
            "close_beyond_zone",
            close_dist >= buffer,
            "require",
            round(close_dist / tick_size, 2),
            round(buffer / tick_size, 2),
            f"Close {'above' if long else 'below'} zone edge by {close_dist / tick_size:.1f} ticks "
            f"(buffer {buffer / tick_size:.1f}).",
        )
    )
    direction_ok = c > o if long else c < o
    checks.append(
        Check(
            "candle_direction",
            direction_ok,
            "require",
            "bullish" if c > o else "bearish" if c < o else "doji",
            "bullish" if long else "bearish",
            "Breakout candle body points in the breakout direction."
            if direction_ok
            else "Breakout candle closed against the breakout direction.",
        )
    )
    checks.append(
        Check(
            "body_ratio",
            body_ratio >= cfg.min_body_ratio,
            "require",
            round(body_ratio, 3),
            cfg.min_body_ratio,
            f"Body is {body_ratio:.0%} of the candle range.",
        )
    )
    size_ok = cfg.min_range_atr <= range_atr <= cfg.max_range_atr
    checks.append(
        Check(
            "range_atr",
            size_ok,
            "require",
            round(range_atr, 3),
            f"{cfg.min_range_atr}-{cfg.max_range_atr}",
            f"Candle range is {range_atr:.2f} x ATR.",
        )
    )
    checks.append(
        Check(
            "extension",
            extension_atr <= cfg.max_extension_atr,
            "require",
            round(extension_atr, 3),
            cfg.max_extension_atr,
            f"Close is {extension_atr:.2f} ATR beyond the zone (max {cfg.max_extension_atr}).",
        )
    )

    if next_opposing_edge is None:
        metrics["room_atr"] = None
        checks.append(
            Check(
                "room_to_opposing_zone",
                True,
                "require",
                "none found",
                cfg.min_room_atr,
                "No opposing zone identified within the relevance distance (open space).",
            )
        )
    else:
        room = (next_opposing_edge - c) if long else (c - next_opposing_edge)
        room_atr = room / atr
        metrics["room_atr"] = room_atr
        checks.append(
            Check(
                "room_to_opposing_zone",
                room_atr >= cfg.min_room_atr,
                "require",
                round(room_atr, 2),
                cfg.min_room_atr,
                f"{room_atr:.2f} ATR of room before the next opposing zone.",
            )
        )

    # Context filters (optional requirement)
    if htf_alignment is None:
        checks.append(
            Check(
                "htf_alignment",
                None,
                "unavailable",
                None,
                None,
                "Higher-timeframe trend unavailable (disabled or insufficient history).",
            )
        )
    else:
        ok = htf_alignment >= 0.5
        mode = "require" if cfg.require_htf_alignment else "score"
        checks.append(
            Check(
                "htf_alignment",
                ok,
                mode,
                round(htf_alignment, 3),
                0.5,
                f"Higher-timeframe alignment {htf_alignment:.0%}.",
            )
        )
    metrics["htf_alignment"] = htf_alignment

    if _nan(vwap):
        checks.append(
            Check(
                "vwap_side",
                None,
                "unavailable",
                None,
                None,
                "VWAP unavailable (no reliable volume) - filter not applied.",
            )
        )
        metrics["vwap_distance_atr"] = None
    else:
        vw = float(vwap)  # type: ignore[arg-type]  # not None: checked by _nan above
        ok = c > vw if long else c < vw
        mode = "require" if cfg.require_vwap_alignment else "score"
        dist = ((c - vw) if long else (vw - c)) / atr
        metrics["vwap_distance_atr"] = dist
        checks.append(
            Check(
                "vwap_side",
                ok,
                mode,
                round(dist, 2),
                0.0,
                f"Close is {'above' if c > vw else 'below'} VWAP ({dist:+.2f} ATR).",
            )
        )

    if _nan(rvol):
        checks.append(
            Check(
                "relative_volume",
                None,
                "unavailable",
                None,
                None,
                "Relative volume unavailable (no reliable volume) - filter not applied.",
            )
        )
        metrics["rvol"] = None
    else:
        rv = float(rvol)  # type: ignore[arg-type]  # not None: checked by _nan above
        ok = rv >= cfg.min_relative_volume
        mode = "require" if cfg.require_relative_volume else "score"
        metrics["rvol"] = rv
        checks.append(
            Check(
                "relative_volume",
                ok,
                mode,
                round(rv, 2),
                cfg.min_relative_volume,
                f"Breakout volume is {rv:.2f} x the recent average.",
            )
        )

    passed = bool(all(ch.passed for ch in checks if ch.mode == "require"))
    # A *required* context filter that is unavailable must fail, not pass silently.
    if cfg.require_htf_alignment and htf_alignment is None:
        passed = False
        checks.append(
            Check(
                "htf_required_missing",
                False,
                "require",
                None,
                None,
                "HTF alignment is required but unavailable.",
            )
        )
    if cfg.require_vwap_alignment and _nan(vwap):
        passed = False
        checks.append(
            Check(
                "vwap_required_missing",
                False,
                "require",
                None,
                None,
                "VWAP alignment is required but VWAP is unavailable.",
            )
        )
    if cfg.require_relative_volume and _nan(rvol):
        passed = False
        checks.append(
            Check(
                "rvol_required_missing",
                False,
                "require",
                None,
                None,
                "Relative volume is required but unavailable.",
            )
        )
    return BreakoutEvaluation(direction, passed, checks, metrics)
