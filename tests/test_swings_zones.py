"""Swing confirmation timing, no look-ahead, and support/resistance zone behaviour."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from core.config import ZoneSettings
from core.indicators import atr
from core.support_resistance import ZoneTracker, display_kind
from core.swing_detection import SwingParams, detect_swings, provisional_swings, swings_known_at
from tests.conftest import make_bars

P = SwingParams(left=2, right=2, min_prominence_ticks=0, atr_mult=None, duplicate_window_bars=0)


def peak_rows():
    # bar 3 is a clear swing high (105), bar 7 a swing low (95)
    highs = [100, 101, 102, 105, 103, 102, 100, 98, 99, 101, 102]
    lows = [99, 100, 101, 103, 101, 100, 97, 95, 97, 99, 100]
    return [(lo + 0.25, h, lo, h - 0.25) for h, lo in zip(highs, lows, strict=True)]


class TestSwings:
    def test_confirmation_timestamp_is_close_of_right_bar(self):
        df = make_bars(peak_rows())
        sw = detect_swings(df, 0.25, P)
        hi = sw[sw["swing_type"] == "high"].iloc[0]
        assert hi["pivot_idx"] == 3 and hi["price"] == 105
        assert hi["confirm_idx"] == 5
        assert hi["confirm_ts"] == df.index[5] + pd.Timedelta(minutes=5)
        lo = sw[sw["swing_type"] == "low"].iloc[0]
        assert lo["pivot_idx"] == 7 and lo["confirm_idx"] == 9

    def test_swing_not_known_before_confirmation(self):
        df = make_bars(peak_rows())
        sw = detect_swings(df, 0.25, P)
        hi = sw[sw["swing_type"] == "high"].iloc[0]
        assert (
            swings_known_at(sw, hi["confirm_ts"] - pd.Timedelta(seconds=1))
            .query("swing_type == 'high'")
            .empty
        )
        assert not swings_known_at(sw, hi["confirm_ts"]).query("swing_type == 'high'").empty

    def test_unconfirmed_swing_absent_until_right_bars_exist(self):
        df = make_bars(peak_rows())
        truncated = df.iloc[:5]  # only 1 bar after the pivot
        assert detect_swings(truncated, 0.25, P).query("swing_type == 'high'").empty
        prov = provisional_swings(truncated, P)
        assert (prov["swing_type"] == "high").any() and (prov["status"] == "provisional").all()

    def test_truncation_invariance_no_lookahead(self, synthetic_nq):
        df = synthetic_nq.iloc[:1500]
        params = SwingParams(left=3, right=3, min_prominence_ticks=8, atr_mult=0.5, duplicate_window_bars=6)
        a = atr(df, 14)
        full = detect_swings(df, 0.25, params, a)
        for cut in (400, 777, 1200):
            part = detect_swings(df.iloc[:cut], 0.25, params, a.iloc[:cut])
            expect = full[full["confirm_idx"] < cut].reset_index(drop=True)
            pd.testing.assert_frame_equal(part.reset_index(drop=True), expect)

    def test_equal_highs_resolve_to_first(self):
        highs = [100, 101, 104, 104, 102, 101, 100]
        rows = [(h - 1, h, h - 2, h - 0.5) for h in highs]
        sw = detect_swings(make_bars(rows), 0.25, P)
        highs_found = sw[sw["swing_type"] == "high"]
        assert list(highs_found["pivot_idx"]) == [2]

    def test_prominence_filter(self):
        df = make_bars(peak_rows())
        strict = SwingParams(left=2, right=2, min_prominence_ticks=40, atr_mult=None, duplicate_window_bars=0)
        assert detect_swings(df, 0.25, strict).empty  # 10 points < required prominence of 40 ticks

    def test_duplicate_suppression(self):
        highs = [100, 101, 105, 103, 102, 103, 105, 103, 101, 100]
        rows = [(h - 1, h, h - 2, h - 0.5) for h in highs]
        params = SwingParams(
            left=2,
            right=2,
            min_prominence_ticks=0,
            atr_mult=None,
            duplicate_window_bars=6,
            duplicate_tolerance_ticks=4,
        )
        sw = detect_swings(make_bars(rows), 0.25, params)
        h = sw[sw["swing_type"] == "high"]
        assert list(h["status"]) == ["confirmed", "suppressed"]


def zcfg(**kw) -> ZoneSettings:
    base = dict(
        tolerance_ticks=4,
        tolerance_atr_mult=0.0,
        max_zone_width_atr=10,
        min_zone_width_ticks=2,
        min_reaction_gap_bars=3,
        break_buffer_ticks=1,
        break_buffer_atr_mult=0.0,
        min_reactions=1,
        max_age_bars=500,
    )
    base.update(kw)
    return ZoneSettings(**base)


def swing(price, kind="high", pivot=0, conf=2):
    return {
        "swing_id": f"{kind}:{pivot}",
        "swing_type": kind,
        "price": price,
        "pivot_idx": pivot,
        "pivot_ts": pd.Timestamp("2026-03-10", tz="UTC") + pd.Timedelta(minutes=5 * pivot),
        "confirm_idx": conf,
        "confirm_ts": pd.Timestamp("2026-03-10", tz="UTC") + pd.Timedelta(minutes=5 * (conf + 1)),
        "prominence": 2.0,
        "body_top": price - 0.5,
        "body_bottom": price + 0.5,
    }


class TestZones:
    def ts(self, i):
        return pd.Timestamp("2026-03-10", tz="UTC") + pd.Timedelta(minutes=5 * i)

    def test_clustering_merges_nearby_swings_and_counts_distinct_reactions(self):
        tr = ZoneTracker(0.25, zcfg())
        z1 = tr.add_swing(swing(100.0, pivot=0), 2, self.ts(2), 99.0, 1.0)
        z2 = tr.add_swing(
            swing(100.5, pivot=1), 3, self.ts(3), 99.0, 1.0
        )  # too close in time -> same reaction
        z3 = tr.add_swing(swing(100.25, pivot=10), 12, self.ts(12), 99.0, 1.0)
        assert z1 is z2 is z3
        assert z1.reactions == 2
        assert z1.upper == 100.5
        far = tr.add_swing(swing(110.0, pivot=20), 22, self.ts(22), 99.0, 1.0)
        assert far is not z1 and len(tr.alive_zones()) == 2

    def test_break_retest_and_invalidation(self):
        tr = ZoneTracker(0.25, zcfg())
        z = tr.add_swing(swing(100.0), 2, self.ts(2), 99.0, 1.0)  # resistance [99.5, 100]
        assert display_kind(z, 99.0) == "resistance"
        assert tr.update_bar(3, self.ts(3), 99, 99.8, 98.9, 99.5, 1.0) == []  # touch from below
        breaks = tr.update_bar(4, self.ts(4), 99.5, 101.0, 99.4, 100.75, 1.0)  # close above + buffer
        assert len(breaks) == 1 and breaks[0].direction == "up"
        assert z.status == "broken" and z.failures == 1
        tr.update_bar(5, self.ts(5), 100.8, 101.5, 100.6, 101.2, 1.0)  # moves away above
        tr.update_bar(6, self.ts(6), 101.0, 101.1, 99.9, 100.6, 1.0)  # retest from above, holds
        tr.update_bar(7, self.ts(7), 100.6, 102.0, 100.5, 101.8, 1.0)  # leaves upward
        assert z.retested and z.holds == 1
        assert display_kind(z, 101.8) == "retest"
        tr.update_bar(8, self.ts(8), 101.0, 101.0, 98.0, 98.5, 1.0)  # closes back below -> invalidated
        assert z.status == "invalidated"
        assert z not in tr.alive_zones()

    def test_holds_counted_once_per_test(self):
        tr = ZoneTracker(0.25, zcfg())
        z = tr.add_swing(swing(100.0), 2, self.ts(2), 99.0, 1.0)
        for i in range(3, 6):  # three consecutive bars poking the zone = one test
            tr.update_bar(i, self.ts(i), 99.2, 99.9, 99.0, 99.3, 1.0)
        tr.update_bar(6, self.ts(6), 99.0, 99.2, 98.0, 98.5, 1.0)
        assert z.tests == 1 and z.holds == 1

    def test_expiry(self):
        tr = ZoneTracker(0.25, zcfg(max_age_bars=50))
        z = tr.add_swing(swing(100.0), 2, self.ts(2), 90.0, 1.0)
        for i in range(3, 60):
            tr.update_bar(i, self.ts(i), 90, 90.5, 89.5, 90, 1.0)
        assert z.status == "expired"

    def test_strength_components_explainable(self):
        tr = ZoneTracker(0.25, zcfg())
        z = tr.add_swing(swing(100.0), 2, self.ts(2), 99.0, 1.0)
        score, parts = tr.strength(z, 2)
        assert set(parts) == {"reactions", "recency", "rejection", "holds"}
        assert 0 <= score <= 1
        assert parts["holds"] == pytest.approx(0.5)  # Laplace prior with no tests
        older, _ = tr.strength(z, 2 + 576)
        assert older < score  # recency decays

    def test_gap_through_zone_counts_as_break(self):
        tr = ZoneTracker(0.25, zcfg())
        z = tr.add_swing(swing(100.0), 2, self.ts(2), 99.0, 1.0)
        breaks = tr.update_bar(3, self.ts(3), 102.0, 103.0, 101.5, 102.5, 1.0)
        assert breaks and z.break_direction == "up"
        assert not math.isnan(z.lower)
