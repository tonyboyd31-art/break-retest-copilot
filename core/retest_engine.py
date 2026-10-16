"""Break-and-retest setup state machine.

States::

    LEVEL_IDENTIFIED -> BREAKOUT_DETECTED -> AWAITING_RETEST -> RETEST_DETECTED
        -> AWAITING_CONFIRMATION -> QUALIFIED | REJECTED
    (any non-terminal) -> EXPIRED | INVALIDATED
    BREAKOUT_DETECTED -> REJECTED   (breakout failed quality checks)

Objective, backtestable rules (long shown; short is mirrored):

* **Retest**: a completed bar whose low comes within ``retest tolerance`` of
  the broken zone's top (``low <= upper + tol``) within
  ``max_bars_to_retest`` bars of the breakout.
* **Confirmation** (on the retest bar or up to ``max_bars_to_confirm`` bars
  after it): close above ``upper + confirm buffer``, bullish body, close in
  the top ``min_close_position`` of the range and - if required - a
  rejection wick of at least ``min_wick_ratio`` on a retest bar.
* **Invalidation**: a close below ``lower - invalidation buffer``; price
  running more than ``max_run_atr`` ATR away before confirmation; a data gap.
* **Expiry**: no retest in time, no confirmation in time, total age beyond
  ``setup_expiry_bars``, or (optionally) the trading session ended.

Every transition is stored with the close time of the bar that made it
observable, so a historical replay shows exactly what was known when.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

import pandas as pd

from core.breakout_detection import BreakoutEvaluation
from core.config import RetestSettings
from core.scoring import ScoreResult
from core.trade_planning import TradePlan


class SetupState(StrEnum):
    LEVEL_IDENTIFIED = "level_identified"
    BREAKOUT_DETECTED = "breakout_detected"
    AWAITING_RETEST = "awaiting_retest"
    RETEST_DETECTED = "retest_detected"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    QUALIFIED = "qualified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    INVALIDATED = "invalidated"


TERMINAL_STATES = {SetupState.QUALIFIED, SetupState.REJECTED, SetupState.EXPIRED, SetupState.INVALIDATED}

STATE_LABELS = {
    SetupState.LEVEL_IDENTIFIED: "Level identified",
    SetupState.BREAKOUT_DETECTED: "Breakout detected",
    SetupState.AWAITING_RETEST: "Awaiting retest",
    SetupState.RETEST_DETECTED: "Retest detected",
    SetupState.AWAITING_CONFIRMATION: "Awaiting confirmation",
    SetupState.QUALIFIED: "Qualified",
    SetupState.REJECTED: "Rejected",
    SetupState.EXPIRED: "Expired",
    SetupState.INVALIDATED: "Invalidated",
}


def make_setup_id(
    instrument: str, contract: str | None, timeframe: str, direction: str, breakout_ts: pd.Timestamp
) -> str:
    """Stable ID: one setup per instrument/contract/direction/breakout bar."""
    key = (
        f"{instrument}|{contract or '-'}|{timeframe}|{direction}|{breakout_ts.tz_convert('UTC').isoformat()}"
    )
    return hashlib.sha1(key.encode()).hexdigest()[:16]


@dataclass
class Transition:
    state: SetupState
    ts: pd.Timestamp
    bar_idx: int
    note: str


@dataclass
class Setup:
    setup_id: str
    instrument: str
    contract: str | None
    timeframe: str
    direction: str  # long | short
    zone: dict[str, Any]
    breakout_idx: int
    breakout_ts: pd.Timestamp
    breakout_bar: dict[str, float]
    atr_at_breakout: float
    trade_date: date
    state: SetupState = SetupState.LEVEL_IDENTIFIED
    history: list[Transition] = field(default_factory=list)
    breakout_eval: BreakoutEvaluation | None = None
    retest_idx: int | None = None
    retest_ts: pd.Timestamp | None = None
    retest_price: float | None = None  # extreme reached into the zone
    best_wick_ratio: float = 0.0
    precision: float | None = None
    confirm_idx: int | None = None
    confirm_ts: pd.Timestamp | None = None
    confirm_price: float | None = None
    confirm_close_pos: float | None = None
    context: dict[str, Any] = field(default_factory=dict)
    score: ScoreResult | None = None
    plan: TradePlan | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    terminal_ts: pd.Timestamp | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_long(self) -> bool:
        return self.direction == "long"

    @property
    def last_update_ts(self) -> pd.Timestamp:
        return self.history[-1].ts if self.history else self.breakout_ts

    def transition(self, state: SetupState, ts: pd.Timestamp, bar_idx: int, note: str = "") -> None:
        self.state = state
        self.history.append(Transition(state, ts, bar_idx, note))
        if state in TERMINAL_STATES:
            self.terminal_ts = ts


@dataclass
class BarCtx:
    idx: int
    ts: pd.Timestamp  # bar close time
    o: float
    h: float
    low: float
    c: float
    atr: float
    trade_date: date
    gap_before: bool


@dataclass
class AdvanceResult:
    needs_finalize: bool = False
    changed: list[SetupState] = field(default_factory=list)


def _wick_ratio(direction: str, b: BarCtx) -> float:
    rng = b.h - b.low
    if rng <= 0:
        return 0.0
    if direction == "long":
        return (min(b.o, b.c) - b.low) / rng
    return (b.h - max(b.o, b.c)) / rng


def _close_pos(direction: str, b: BarCtx) -> float:
    rng = b.h - b.low
    if rng <= 0:
        return 0.0
    return (b.c - b.low) / rng if direction == "long" else (b.h - b.c) / rng


def _precision(direction: str, extreme: float, lower: float, upper: float, atr: float) -> float:
    """1.0 inside the zone, 0.8 front-run within tolerance, penalised for deep pierces."""
    half_atr = 0.5 * atr if atr > 0 else max(upper - lower, 1e-9)
    if direction == "long":
        if extreme < lower:
            return max(0.0, 1.0 - (lower - extreme) / half_atr)
        return 1.0 if extreme <= upper else 0.8
    if extreme > upper:
        return max(0.0, 1.0 - (extreme - upper) / half_atr)
    return 1.0 if extreme >= lower else 0.8


def advance_setup(s: Setup, b: BarCtx, cfg: RetestSettings, tick: float) -> AdvanceResult:
    """Advance a non-terminal setup with one completed bar."""
    res = AdvanceResult()
    if s.is_terminal:
        return res
    long = s.is_long
    lower, upper = float(s.zone["lower"]), float(s.zone["upper"])
    atr_b = s.atr_at_breakout
    tol = max(cfg.retest_tolerance_ticks * tick, cfg.retest_tolerance_atr_mult * atr_b)
    inval_buf = max(cfg.invalidation_buffer_ticks * tick, cfg.invalidation_buffer_atr_mult * atr_b)
    confirm_buf = cfg.confirm_buffer_ticks * tick

    def end(state: SetupState, note: str) -> AdvanceResult:
        s.transition(state, b.ts, b.idx, note)
        s.rejection_reasons.append(note)
        res.changed.append(state)
        return res

    if b.gap_before:
        return end(SetupState.INVALIDATED, "Data unreliable: missing candles during the setup.")
    if cfg.expire_at_session_end and b.trade_date != s.trade_date:
        return end(SetupState.EXPIRED, "Trading session ended before the setup completed.")
    if b.idx - s.breakout_idx > cfg.setup_expiry_bars:
        return end(SetupState.EXPIRED, f"Setup older than {cfg.setup_expiry_bars} bars without qualifying.")

    closed_back = (b.c < lower - inval_buf) if long else (b.c > upper + inval_buf)
    touched = (b.low <= upper + tol) if long else (b.h >= lower - tol)
    run = ((b.h - upper) if long else (lower - b.low)) / atr_b if atr_b > 0 else 0.0

    if s.state == SetupState.AWAITING_RETEST:
        if closed_back:
            return end(
                SetupState.INVALIDATED,
                "Closed back through the broken zone before a retest (failed breakout).",
            )
        if touched:
            s.retest_idx, s.retest_ts = b.idx, b.ts
            s.retest_price = b.low if long else b.h
            s.best_wick_ratio = _wick_ratio(s.direction, b)
            s.transition(
                SetupState.RETEST_DETECTED,
                b.ts,
                b.idx,
                f"Price revisited the zone ({'low' if long else 'high'} {s.retest_price}).",
            )
            s.transition(SetupState.AWAITING_CONFIRMATION, b.ts, b.idx, "Waiting for a confirmation close.")
            res.changed += [SetupState.RETEST_DETECTED, SetupState.AWAITING_CONFIRMATION]
            # The retest bar itself may confirm.
            return _check_confirmation(s, b, cfg, lower, upper, confirm_buf, res)
        if run > cfg.max_run_atr:
            return end(
                SetupState.INVALIDATED,
                f"Price moved {run:.1f} ATR away from the level before retesting (max {cfg.max_run_atr:g}).",
            )
        if b.idx - s.breakout_idx >= cfg.max_bars_to_retest:
            return end(SetupState.EXPIRED, f"No retest within {cfg.max_bars_to_retest} bars of the breakout.")
        return res

    if s.state == SetupState.AWAITING_CONFIRMATION:
        if closed_back:
            return end(SetupState.INVALIDATED, "Retest failed: closed back through the zone.")
        if touched:
            ext = b.low if long else b.h
            s.retest_price = min(s.retest_price, ext) if long else max(s.retest_price, ext)  # type: ignore[type-var]
            s.best_wick_ratio = max(s.best_wick_ratio, _wick_ratio(s.direction, b))
        _check_confirmation(s, b, cfg, lower, upper, confirm_buf, res)
        if res.needs_finalize:
            return res
        if run > cfg.max_run_atr:
            return end(
                SetupState.INVALIDATED,
                f"Price moved {run:.1f} ATR away without a valid confirmation (max {cfg.max_run_atr:g}).",
            )
        if b.idx - (s.retest_idx or b.idx) >= cfg.max_bars_to_confirm:
            return end(
                SetupState.EXPIRED, f"No confirmation within {cfg.max_bars_to_confirm} bars of the retest."
            )
        return res
    return res


def _check_confirmation(
    s: Setup,
    b: BarCtx,
    cfg: RetestSettings,
    lower: float,
    upper: float,
    confirm_buf: float,
    res: AdvanceResult,
) -> AdvanceResult:
    long = s.is_long
    close_ok = (b.c > upper + confirm_buf) if long else (b.c < lower - confirm_buf)
    body_ok = (b.c > b.o) if long else (b.c < b.o)
    pos = _close_pos(s.direction, b)
    pos_ok = pos >= cfg.min_close_position
    wick_ok = (not cfg.require_rejection_wick) or s.best_wick_ratio >= cfg.min_wick_ratio
    if close_ok and body_ok and pos_ok and wick_ok:
        s.confirm_idx, s.confirm_ts, s.confirm_price = b.idx, b.ts, b.c
        s.confirm_close_pos = pos
        s.precision = _precision(s.direction, float(s.retest_price), lower, upper, s.atr_at_breakout)  # type: ignore[arg-type]
        res.needs_finalize = True
    return res
