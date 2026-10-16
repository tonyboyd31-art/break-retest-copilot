"""Incremental support/resistance zone engine.

Zones are built causally, bar by bar:

1. When a swing is *confirmed* (at its ``confirm_ts``) it either merges into
   an existing zone (if its price is within tolerance and the merged zone
   stays narrower than the maximum width) or creates a new zone.
2. Every completed bar updates each zone's tests, holds, failures, breaks
   and retests.

Because the tracker only ever sees bars and swings that are known at the
current bar, a zone's statistics at time *t* can never include reactions
that happened later. The scanner snapshots zone strength at the moment a
setup is created and stores it with the signal.

Definitions
-----------
* **reaction**: a confirmed swing belonging to the zone. Swings closer than
  ``min_reaction_gap_bars`` to the previous member count as the same
  reaction (avoids counting every nearby candle as independent).
* **test**: a run of consecutive bars overlapping the zone.
* **hold**: a test that ends with price leaving on the side it came from.
* **failure / break**: a close beyond the far edge by the break buffer.
* **retest**: after a break, a test from the new side that holds.
* A zone broken once and then broken back the other way is *invalidated*.
* Zones without a reaction for ``max_age_bars`` *expire*.

Strength (0..1) = weighted mean of reactions, recency, rejection (member
prominence in ATR) and hold ratio - see :meth:`ZoneTracker.strength`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import ZoneSettings
from core.instruments import round_to_tick


@dataclass
class ZoneMember:
    swing_id: str
    swing_type: str
    price: float
    pivot_idx: int
    pivot_ts: pd.Timestamp
    confirm_idx: int
    confirm_ts: pd.Timestamp
    prominence_atr: float | None


@dataclass
class Zone:
    zone_id: int
    lower: float
    upper: float
    origin: str
    created_idx: int
    created_ts: pd.Timestamp
    members: list[ZoneMember] = field(default_factory=list)
    reactions: int = 1
    last_reaction_idx: int = 0
    last_reaction_ts: pd.Timestamp | None = None
    tests: int = 0
    holds: int = 0
    failures: int = 0
    status: str = "active"  # active | broken | invalidated | expired
    break_direction: str | None = None  # up | down
    broken_idx: int | None = None
    broken_ts: pd.Timestamp | None = None
    retested: bool = False
    retest_ts: pd.Timestamp | None = None
    ended_ts: pd.Timestamp | None = None
    end_reason: str | None = None
    ref_side: str = "below"  # last side (above/below) price was on, ignoring "inside"
    in_test: bool = False
    test_from: str | None = None
    history: list[tuple[pd.Timestamp, str]] = field(default_factory=list)

    @property
    def mid(self) -> float:
        return (self.lower + self.upper) / 2.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def alive(self) -> bool:
        return self.status in ("active", "broken")


@dataclass
class ZoneBreak:
    zone: Zone
    direction: str  # "up" (resistance broken) or "down" (support broken)
    bar_idx: int
    ts: pd.Timestamp


class ZoneTracker:
    def __init__(self, tick_size: float, settings: ZoneSettings, price_decimals: int = 2):
        self.tick = tick_size
        self.cfg = settings
        self.decimals = price_decimals
        self.zones: list[Zone] = []  # every zone ever created (history)
        self._alive: list[Zone] = []
        self._next_id = 1

    # -- helpers -----------------------------------------------------------
    def _buffer(self, atr: float) -> float:
        b = self.cfg.break_buffer_ticks * self.tick
        if not math.isnan(atr):
            b = max(b, self.cfg.break_buffer_atr_mult * atr)
        return b

    def _tolerance(self, atr: float) -> float:
        t = self.cfg.tolerance_ticks * self.tick
        if not math.isnan(atr):
            t = max(t, self.cfg.tolerance_atr_mult * atr)
        return t

    def _max_width(self, atr: float) -> float:
        min_w = self.cfg.min_zone_width_ticks * self.tick
        if math.isnan(atr):
            return max(min_w, 4 * self.cfg.tolerance_ticks * self.tick)
        return max(min_w, self.cfg.max_zone_width_atr * atr)

    # -- swings ------------------------------------------------------------
    def add_swing(self, swing: dict, bar_idx: int, ts: pd.Timestamp, close: float, atr: float) -> Zone:
        """Merge a newly confirmed swing into a zone or create one."""
        price = float(swing["price"])
        prom_atr = None
        if not math.isnan(atr) and atr > 0:
            prom_atr = float(swing["prominence"]) / atr
        member = ZoneMember(
            swing_id=str(swing["swing_id"]),
            swing_type=str(swing["swing_type"]),
            price=price,
            pivot_idx=int(swing["pivot_idx"]),
            pivot_ts=swing["pivot_ts"],
            confirm_idx=int(swing["confirm_idx"]),
            confirm_ts=swing["confirm_ts"],
            prominence_atr=prom_atr,
        )
        tol = self._tolerance(atr)
        max_w = self._max_width(atr)
        best: Zone | None = None
        best_dist = float("inf")
        for z in self._alive:
            if z.lower - tol <= price <= z.upper + tol:
                new_w = max(z.upper, price) - min(z.lower, price)
                if new_w <= max_w + 1e-9:
                    d = abs(price - z.mid)
                    if d < best_dist:
                        best, best_dist = z, d
        if best is not None:
            last_pivot = max(m.pivot_idx for m in best.members)
            best.members.append(member)
            best.lower = min(best.lower, price)
            best.upper = max(best.upper, price)
            if member.pivot_idx - last_pivot >= self.cfg.min_reaction_gap_bars:
                best.reactions += 1
                best.last_reaction_idx = bar_idx
                best.last_reaction_ts = ts
                best.history.append((ts, f"reaction #{best.reactions} ({member.swing_type} {price})"))
            if best.origin != member.swing_type:
                best.origin = "mixed"
            return best

        min_w = self.cfg.min_zone_width_ticks * self.tick
        single_cap = max(min_w, 0.5 * max_w)
        if member.swing_type == "high":
            upper = price
            body_edge = float(swing.get("body_top", price))
            lower = max(body_edge, price - single_cap)
            if upper - lower < min_w:
                lower = upper - min_w
        else:
            lower = price
            body_edge = float(swing.get("body_bottom", price))
            upper = min(body_edge, price + single_cap)
            if upper - lower < min_w:
                upper = lower + min_w
        lower = round_to_tick(lower, self.tick, "down", self.decimals)
        upper = round_to_tick(upper, self.tick, "up", self.decimals)
        if close > upper:
            ref = "above"
        elif close < lower:
            ref = "below"
        else:
            ref = "below" if member.swing_type == "high" else "above"
        z = Zone(
            zone_id=self._next_id,
            lower=lower,
            upper=upper,
            origin=member.swing_type,
            created_idx=bar_idx,
            created_ts=ts,
            members=[member],
            reactions=1,
            last_reaction_idx=bar_idx,
            last_reaction_ts=ts,
            ref_side=ref,
        )
        z.history.append((ts, f"created from swing {member.swing_type} {price}"))
        self._next_id += 1
        self.zones.append(z)
        self._alive.append(z)
        return z

    # -- bars --------------------------------------------------------------
    def update_bar(
        self, bar_idx: int, ts: pd.Timestamp, o: float, h: float, low: float, c: float, atr: float
    ) -> list[ZoneBreak]:
        """Process one completed bar; return first-time zone breaks."""
        if not self._alive:
            return []
        buf = self._buffer(atr)
        n = len(self._alive)
        lower = np.fromiter((z.lower for z in self._alive), float, n)
        upper = np.fromiter((z.upper for z in self._alive), float, n)
        in_test = np.fromiter((z.in_test for z in self._alive), bool, n)
        ref_above = np.fromiter((z.ref_side == "above" for z in self._alive), bool, n)
        overlaps = (low <= upper) & (h >= lower)
        beyond_up = c > upper + buf
        beyond_down = c < lower - buf
        side_flip = (c > upper) & ~ref_above | (c < lower) & ref_above
        touched = overlaps | in_test | side_flip | beyond_up | beyond_down
        breaks: list[ZoneBreak] = []
        ended: list[Zone] = []
        for k in np.nonzero(touched)[0]:
            z = self._alive[int(k)]
            ov = bool(overlaps[k])
            if ov and not z.in_test:
                z.in_test = True
                z.tests += 1
                z.test_from = z.ref_side

            broke = None
            if z.ref_side == "below" and c > z.upper + buf:
                broke = "up"
            elif z.ref_side == "above" and c < z.lower - buf:
                broke = "down"

            if broke:
                z.failures += 1
                z.in_test = False
                z.test_from = None
                z.ref_side = "above" if broke == "up" else "below"
                if z.status == "active":
                    z.status = "broken"
                    z.break_direction = broke
                    z.broken_idx = bar_idx
                    z.broken_ts = ts
                    z.history.append((ts, f"broken {broke} (close {c})"))
                    breaks.append(ZoneBreak(z, broke, bar_idx, ts))
                elif z.status == "broken":
                    z.status = "invalidated"
                    z.ended_ts = ts
                    z.end_reason = f"broken back {broke} after a {z.break_direction} break"
                    z.history.append((ts, z.end_reason))
                    ended.append(z)
                continue

            if z.in_test and not ov:
                # Test finished: which side did price leave on?
                left_side = "above" if low > z.upper else ("below" if h < z.lower else None)
                if left_side is not None and left_side == z.test_from:
                    z.holds += 1
                    z.history.append((ts, f"held from {left_side}"))
                    if z.status == "broken" and not z.retested:
                        new_side = "above" if z.break_direction == "up" else "below"
                        if left_side == new_side:
                            z.retested = True
                            z.retest_ts = ts
                            z.history.append((ts, "retest held"))
                if left_side is not None:
                    z.ref_side = left_side
                z.in_test = False
                z.test_from = None
            elif not z.in_test:
                if c > z.upper:
                    z.ref_side = "above"
                elif c < z.lower:
                    z.ref_side = "below"

        # Expiry by age
        for z in self._alive:
            if z.status in ("active", "broken") and bar_idx - z.last_reaction_idx > self.cfg.max_age_bars:
                ref_idx = z.broken_idx if z.broken_idx is not None else z.last_reaction_idx
                if bar_idx - ref_idx > self.cfg.max_age_bars:
                    z.status = "expired"
                    z.ended_ts = ts
                    z.end_reason = f"no reaction for {self.cfg.max_age_bars} bars"
                    ended.append(z)
        if ended:
            ended_ids = {id(z) for z in ended}
            self._alive = [z for z in self._alive if id(z) not in ended_ids]
        return breaks

    # -- evaluation ----------------------------------------------------------
    def alive_zones(self) -> list[Zone]:
        return list(self._alive)

    def strength(self, z: Zone, bar_idx: int) -> tuple[float, dict[str, float]]:
        cfg = self.cfg
        age = max(bar_idx - z.last_reaction_idx, 0)
        reactions = min(z.reactions / cfg.reactions_saturation, 1.0)
        recency = 0.5 ** (age / cfg.recency_half_life_bars)
        proms = [m.prominence_atr for m in z.members if m.prominence_atr is not None]
        rejection = min(float(np.mean(proms)) / cfg.rejection_saturation_atr, 1.0) if proms else 0.5
        holds = (z.holds + 1) / (z.holds + z.failures + 2)
        weights = {
            "reactions": cfg.w_reactions,
            "recency": cfg.w_recency,
            "rejection": cfg.w_rejection,
            "holds": cfg.w_holds,
        }
        parts = {"reactions": reactions, "recency": recency, "rejection": rejection, "holds": holds}
        total_w = sum(weights.values())
        score = sum(weights[k] * parts[k] for k in parts) / total_w
        return float(score), {k: round(float(v), 4) for k, v in parts.items()}

    def relevance(self, z: Zone, bar_idx: int, close: float, atr: float) -> tuple[bool, str]:
        if z.status not in ("active", "broken"):
            return False, f"zone {z.status}"
        if z.reactions < self.cfg.min_reactions:
            return False, f"{z.reactions} reaction(s) < minimum {self.cfg.min_reactions}"
        age = bar_idx - z.last_reaction_idx
        if age > self.cfg.max_age_bars:
            return False, f"last reaction {age} bars ago"
        if not math.isnan(atr) and atr > 0:
            dist = distance_to_zone(close, z.lower, z.upper)
            if dist / atr > self.cfg.relevance_max_distance_atr:
                return False, f"{dist / atr:.1f} ATR from price"
        return True, "relevant"

    def snapshot(self, bar_idx: int, close: float, atr: float, include_dead: bool = False) -> list[dict]:
        zones = self.zones if include_dead else self._alive
        return [zone_to_dict(z, self, bar_idx, close, atr) for z in zones]


def distance_to_zone(price: float, lower: float, upper: float) -> float:
    if price > upper:
        return price - upper
    if price < lower:
        return lower - price
    return 0.0


def display_kind(z: Zone, close: float) -> str:
    if z.status in ("invalidated", "expired"):
        return "invalidated"
    if z.status == "broken":
        if z.retested:
            return "retest"
        return "broken_resistance" if z.break_direction == "up" else "broken_support"
    if close > z.upper:
        return "support"
    if close < z.lower:
        return "resistance"
    return "support" if z.ref_side == "above" else "resistance"


def zone_to_dict(z: Zone, tracker: ZoneTracker, bar_idx: int, close: float, atr: float) -> dict:
    strength, parts = tracker.strength(z, bar_idx)
    relevant, reason = tracker.relevance(z, bar_idx, close, atr)
    dist = distance_to_zone(close, z.lower, z.upper)
    return {
        "zone_id": z.zone_id,
        "lower": z.lower,
        "upper": z.upper,
        "mid": z.mid,
        "width": z.width,
        "origin": z.origin,
        "kind": display_kind(z, close),
        "status": z.status,
        "break_direction": z.break_direction,
        "broken_ts": z.broken_ts,
        "retested": z.retested,
        "retest_ts": z.retest_ts,
        "reactions": z.reactions,
        "tests": z.tests,
        "holds": z.holds,
        "failures": z.failures,
        "created_ts": z.created_ts,
        "last_reaction_ts": z.last_reaction_ts,
        "ended_ts": z.ended_ts,
        "end_reason": z.end_reason,
        "age_bars": bar_idx - z.last_reaction_idx,
        "distance": dist,
        "distance_atr": (dist / atr) if atr and not math.isnan(atr) else None,
        "relevant": relevant,
        "relevance_reason": reason,
        "strength": strength,
        "strength_parts": parts,
        "members": len(z.members),
        "member_prices": [m.price for m in z.members],
    }
