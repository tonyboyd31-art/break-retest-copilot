"""Scoring policy, trade-plan levels and settings validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import DEFAULT_WEIGHTS, AppSettings, PlanningSettings, ScoringSettings, diff_strategy
from core.scoring import aggregate_score, factor_htf, factor_rr, factor_vwap
from core.trade_planning import build_trade_plan

ALL = {k: (1.0, "ok") for k in DEFAULT_WEIGHTS}


class TestScoring:
    def test_perfect_score_is_100(self):
        r = aggregate_score(ALL, ScoringSettings())
        assert r.total == 100 and r.passed

    def test_points_are_weight_times_raw(self):
        f = dict(ALL)
        f["retest_quality"] = (0.5, "half")
        r = aggregate_score(f, ScoringSettings())
        assert r.component("retest_quality").points == 10.0
        assert r.total == 90.0

    def test_missing_factor_renormalised_not_awarded(self):
        f = dict(ALL)
        f["relative_volume"] = (None, "no volume")
        f["vwap_alignment"] = (None, "no volume")
        f["htf_alignment"] = (0.0, "opposed")
        r = aggregate_score(f, ScoringSettings())
        assert r.component("relative_volume").points == 0 and not r.component("relative_volume").available
        # (100 - 15 missing weight - 20 htf points lost) / 85 available weight
        assert r.total == pytest.approx(round(65 / 85 * 100, 1))
        assert r.missing == ["vwap_alignment", "relative_volume"]
        assert any("Missing data" in n for n in r.notes)

    def test_zero_policy_is_more_conservative(self):
        f = dict(ALL)
        f["relative_volume"] = (None, "no volume")
        r = aggregate_score(f, ScoringSettings(missing_factor_policy="zero"))
        assert r.total == 95.0

    def test_essential_factor_missing_rejects(self):
        f = dict(ALL)
        f["reward_risk"] = (None, "no plan")
        r = aggregate_score(f, ScoringSettings())
        assert not r.passed and any("Essential" in x for x in r.reasons)

    def test_insufficient_available_weight_rejects(self):
        f = {k: (None, "x") for k in DEFAULT_WEIGHTS}
        f["zone_quality"] = (1.0, "")
        f["breakout_quality"] = (1.0, "")
        f["retest_quality"] = (1.0, "")
        f["reward_risk"] = (1.0, "")
        r = aggregate_score(f, ScoringSettings(min_available_weight=75))
        assert r.available_weight == 65 and not r.passed

    def test_threshold(self):
        f = dict(ALL)
        f["htf_alignment"] = (0.0, "")
        assert not aggregate_score(f, ScoringSettings(min_score=85)).passed
        assert aggregate_score(f, ScoringSettings(min_score=80)).passed

    def test_factor_helpers(self):
        raw, _ = factor_htf({"15min": "up", "1h": "neutral", "4h": "down"}, "long")
        assert raw == pytest.approx(0.5)
        assert factor_htf({"1h": None}, "long")[0] is None
        assert factor_rr(1.9, 2.0)[0] == 0.0
        assert factor_rr(2.0, 2.0)[0] == 0.5
        assert factor_rr(4.0, 2.0)[0] == 1.0
        assert factor_vwap(None)[0] is None
        assert factor_vwap(-0.1)[0] == 0.0
        assert factor_vwap(5.0, 4.0)[0] == 0.6


class TestSettingsValidation:
    def test_weights_must_sum_to_100(self):
        w = dict(DEFAULT_WEIGHTS)
        w["htf_alignment"] = 30
        with pytest.raises(ValidationError, match="sum to 100"):
            ScoringSettings(weights=w)

    def test_invalid_settings_rejected(self):
        s = AppSettings()
        with pytest.raises(ValidationError):
            s.updated("timeframes", trend_ema_fast=60, trend_ema_slow=50)
        with pytest.raises(ValidationError):
            s.updated("planning", min_stop_ticks=500, max_stop_ticks=100)
        with pytest.raises(ValidationError):
            s.updated("general", display_timezone="Mars/Olympus")
        with pytest.raises(ValidationError):
            s.updated("sessions", signal_windows=[("09:30", "09:30")])
        with pytest.raises(ValidationError):
            s.updated("management", breakeven_thresholds_r=[-1.0])

    def test_strategy_hash_tracks_only_strategy_params(self):
        s = AppSettings()
        assert s.updated("risk", personal_max_dollar_risk_per_trade=999).strategy_hash() == s.strategy_hash()
        changed = s.updated("swings", right_bars=5)
        assert changed.strategy_hash() != s.strategy_hash()
        assert diff_strategy(s, changed) == ["swings.right_bars: 3 -> 5"]

    def test_json_round_trip(self):
        s = AppSettings().updated("scoring", min_score=85)
        assert AppSettings.from_json(s.to_json()) == s


ZONE = (99.0, 100.0)


def plan(nq, direction="long", conf=101.0, retest_ext=99.5, opposing=None, **cfg):
    zl, zu = ZONE if direction == "long" else (101.0, 102.0)
    return build_trade_plan(
        direction,
        nq,
        zl,
        zu,
        101.5 if direction == "long" else 99.5,
        retest_ext,
        conf,
        retest_ext,
        2.0,
        opposing or [],
        PlanningSettings(**cfg),
    )


class TestTradePlanning:
    def test_long_levels_and_rounding(self, nq):
        p = plan(nq, opposing=[{"zone_id": 9, "lower": 110.0, "upper": 111.0, "strength": 0.7}])
        assert p.entry_price == 101.0
        assert p.structural_invalidation == 99.0  # min(zone lower, retest low)
        # buffer = max(2 ticks, 0.1 * ATR 2.0 = 0.2) = 0.5 -> 98.5
        assert p.stop == 98.5
        assert p.risk_ticks == 10 and p.risk_per_contract == 50.0
        assert p.target1 == 109.5  # 2 ticks in front of the opposing zone
        assert p.target1_basis == "structure"
        assert p.rr == pytest.approx(34 / 10)
        assert p.valid

    def test_short_mirror(self, nq):
        p = build_trade_plan(
            "short",
            nq,
            101.0,
            102.0,
            99.5,
            102.5,
            100.0,
            102.5,
            2.0,
            [{"zone_id": 3, "lower": 88.0, "upper": 89.0, "strength": 0.6}],
            PlanningSettings(),
        )
        assert p.structural_invalidation == 102.5
        assert p.stop == 103.0 and p.target1 == 89.5
        assert p.valid and p.rr == pytest.approx(42 / 12)

    def test_target_blocked_by_opposing_zone(self, nq):
        p = plan(
            nq,
            opposing=[{"zone_id": 9, "lower": 103.0, "upper": 104.0, "strength": 0.7}],
            target_method="fixed_r",
        )
        assert not p.valid and any("blocked" in r for r in p.invalid_reasons)

    def test_structure_target_too_close_rejected_not_assumed(self, nq):
        p = plan(nq, opposing=[{"zone_id": 9, "lower": 103.0, "upper": 104.0, "strength": 0.7}])
        assert p.rr < 2 and not p.valid and p.structural_valid and not p.rr_ok

    def test_no_structure_fixed_projection_is_labelled(self, nq):
        p = plan(nq)
        assert p.target1_basis == "fixed_r_no_structure"
        assert "NOT a structural level" in p.explanations["target1"]
        assert p.rr == pytest.approx(2.0) and p.valid

    def test_rr_exactly_minimum_passes_despite_float_noise(self, gc):
        p = build_trade_plan(
            "long",
            gc,
            2410.0,
            2412.0,
            2413.0,
            2411.5,
            2412.3,
            2411.5,
            1.0,
            [],
            PlanningSettings(stop_buffer_ticks=2, stop_buffer_atr_mult=0.0),
        )
        assert p.rr == pytest.approx(2.0) and p.rr_ok

    def test_entry_inside_opposing_zone_invalid(self, nq):
        p = plan(nq, opposing=[{"zone_id": 4, "lower": 100.5, "upper": 101.5, "strength": 0.5}])
        assert not p.structural_valid

    def test_stop_too_wide_rejected(self, nq):
        p = plan(nq, retest_ext=90.0)
        assert not p.structural_valid and any("ATR" in r or "ticks" in r for r in p.invalid_reasons)

    def test_zero_risk_rejected(self, nq):
        p = build_trade_plan("long", nq, 99.0, 100.0, 101.0, 99.5, 98.0, 99.5, 2.0, [], PlanningSettings())
        assert not p.valid and any("negative risk" in r for r in p.invalid_reasons)

    def test_zone_edge_entry_method(self, nq):
        p = plan(nq, entry_method="zone_edge")
        assert p.entry_price == 100.0 and "pullback" in p.explanations["entry"]
