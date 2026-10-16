"""Breakout quality checks and the retest state machine."""

from __future__ import annotations

from datetime import date

import pandas as pd

from core.breakout_detection import evaluate_breakout
from core.config import BreakoutSettings, RetestSettings
from core.retest_engine import BarCtx, Setup, SetupState, advance_setup, make_setup_id

CFG = BreakoutSettings()


def good_long(**kw):
    args = dict(
        direction="long",
        zone_lower=99.0,
        zone_upper=100.0,
        o=99.5,
        h=101.6,
        low=99.4,
        c=101.5,
        atr=2.0,
        tick_size=0.25,
        cfg=CFG,
        vwap=99.0,
        rvol=1.4,
        htf_alignment=1.0,
        next_opposing_edge=110.0,
    )
    args.update(kw)
    return evaluate_breakout(**args)


class TestBreakout:
    def test_quality_breakout_passes(self):
        ev = good_long()
        assert ev.passed, [c.message for c in ev.failed_required]

    def test_failures(self):
        assert not good_long(c=100.1).passed  # barely through the level
        assert not good_long(o=101.4).passed  # tiny body
        assert not good_long(c=99.6, o=101.0).passed  # bearish candle
        assert not good_long(h=106, c=105.9, o=99.5).passed  # over-extended / oversized
        assert not good_long(next_opposing_edge=102.0).passed  # no room
        assert not good_long(atr=float("nan")).passed  # warm-up

    def test_short_mirror(self):
        ev = evaluate_breakout(
            "short",
            100.0,
            101.0,
            100.5,
            100.6,
            98.4,
            98.5,
            2.0,
            0.25,
            CFG,
            vwap=101.0,
            rvol=1.2,
            htf_alignment=1.0,
            next_opposing_edge=90.0,
        )
        assert ev.passed

    def test_missing_volume_is_unavailable_not_passed(self):
        ev = good_long(vwap=None, rvol=None)
        assert ev.passed  # optional filters
        modes = {c.name: (c.mode, c.passed) for c in ev.checks}
        assert modes["relative_volume"] == ("unavailable", None)
        assert modes["vwap_side"] == ("unavailable", None)

    def test_required_but_unavailable_filter_fails(self):
        cfg = BreakoutSettings(require_relative_volume=True)
        ev = evaluate_breakout(
            "long",
            99,
            100,
            99.5,
            101.6,
            99.4,
            101.5,
            2.0,
            0.25,
            cfg,
            rvol=None,
            htf_alignment=1.0,
            next_opposing_edge=110,
        )
        assert not ev.passed

    def test_checks_are_plain_bools(self):
        assert all(
            c.passed in (True, False, None) and (c.passed is None or type(c.passed) is bool)
            for c in good_long().checks
        )


T = pd.Timestamp("2026-03-10 14:00", tz="UTC")
D = date(2026, 3, 10)


def bar(i, o, h, lo, c, gap=False, tdate=D):
    return BarCtx(i, T + pd.Timedelta(minutes=5 * (i + 1)), o, h, lo, c, 2.0, tdate, gap)


def new_setup(direction="long"):
    s = Setup(
        setup_id=make_setup_id("NQ", "NQM26", "5min", direction, T),
        instrument="NQ",
        contract="NQM26",
        timeframe="5min",
        direction=direction,
        zone={"zone_id": 1, "lower": 99.0, "upper": 100.0},
        breakout_idx=0,
        breakout_ts=T,
        breakout_bar={"o": 99.5, "h": 101.6, "l": 99.4, "c": 101.5},
        atr_at_breakout=2.0,
        trade_date=D,
    )
    s.transition(SetupState.AWAITING_RETEST, T, 0, "test")
    return s


RC = RetestSettings(max_bars_to_retest=6, max_bars_to_confirm=2, setup_expiry_bars=20, max_run_atr=3.0)


class TestStateMachine:
    def test_retest_then_confirmation(self):
        s = new_setup()
        assert not advance_setup(s, bar(1, 101.5, 102.0, 101.2, 101.8), RC, 0.25).changed
        r = advance_setup(s, bar(2, 101.0, 101.2, 100.1, 100.4), RC, 0.25)  # retest, weak close
        assert SetupState.RETEST_DETECTED in r.changed and s.state == SetupState.AWAITING_CONFIRMATION
        assert not r.needs_finalize
        r = advance_setup(s, bar(3, 100.6, 101.6, 100.0, 101.5), RC, 0.25)  # rejection wick + bullish close
        assert r.needs_finalize and s.confirm_idx == 3
        assert s.retest_price == 100.0 and s.precision == 1.0
        states = [h.state for h in s.history]
        assert states[:3] == [
            SetupState.AWAITING_RETEST,
            SetupState.RETEST_DETECTED,
            SetupState.AWAITING_CONFIRMATION,
        ]

    def test_same_bar_retest_and_confirmation(self):
        s = new_setup()
        r = advance_setup(s, bar(1, 100.6, 101.8, 99.9, 101.7), RC, 0.25)  # dips into zone, closes strong
        assert r.needs_finalize and s.retest_idx == s.confirm_idx == 1

    def test_failed_breakout_invalidated(self):
        s = new_setup()
        advance_setup(s, bar(1, 101.0, 101.0, 98.0, 98.2), RC, 0.25)
        assert s.state == SetupState.INVALIDATED and s.is_terminal

    def test_no_retest_expires(self):
        s = new_setup()
        for i in range(1, 7):
            advance_setup(s, bar(i, 101.5, 102.0, 101.3, 101.6), RC, 0.25)  # never within tolerance
        assert s.state == SetupState.EXPIRED and "No retest" in s.rejection_reasons[-1]

    def test_runs_away_invalidated(self):
        s = new_setup()
        advance_setup(s, bar(1, 101.5, 107.5, 101.4, 107.0), RC, 0.25)
        assert s.state == SetupState.INVALIDATED and "ATR" in s.rejection_reasons[-1]

    def test_no_confirmation_expires(self):
        s = new_setup()
        advance_setup(s, bar(1, 101.0, 101.2, 100.1, 100.4), RC, 0.25)
        for i in (2, 3):
            advance_setup(s, bar(i, 100.6, 100.8, 100.2, 100.5), RC, 0.25)  # bearish bodies never confirm
        assert s.state == SetupState.EXPIRED

    def test_data_gap_invalidates(self):
        s = new_setup()
        advance_setup(s, bar(1, 101.5, 102.0, 101.2, 101.8, gap=True), RC, 0.25)
        assert s.state == SetupState.INVALIDATED and "missing candles" in s.rejection_reasons[-1]

    def test_session_end_expires(self):
        s = new_setup()
        advance_setup(s, bar(1, 101.5, 102.0, 101.2, 101.8, tdate=date(2026, 3, 11)), RC, 0.25)
        assert s.state == SetupState.EXPIRED

    def test_rejection_wick_required(self):
        s = new_setup()
        # retest bar with no lower wick (opens at low), then strong close: wick requirement not met
        advance_setup(s, bar(1, 100.0, 100.6, 100.0, 100.5), RC, 0.25)
        r = advance_setup(s, bar(2, 100.5, 101.6, 100.4, 101.5), RC, 0.25)
        assert not r.needs_finalize

    def test_terminal_setups_do_not_change(self):
        s = new_setup()
        advance_setup(s, bar(1, 101.0, 101.0, 98.0, 98.2), RC, 0.25)
        n = len(s.history)
        advance_setup(s, bar(2, 100.4, 101.6, 100.0, 101.5), RC, 0.25)
        assert len(s.history) == n

    def test_stable_ids(self):
        assert make_setup_id("NQ", "NQM26", "5min", "long", T) == make_setup_id(
            "NQ", "NQM26", "5min", "long", T
        )
        assert make_setup_id("NQ", "NQM26", "5min", "long", T) != make_setup_id(
            "NQ", "NQM26", "5min", "short", T
        )
