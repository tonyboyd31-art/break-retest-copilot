"""Risk sizing, R-multiples, breakeven, trailing suggestions and trade warnings."""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import AppSettings
from core.data_quality import FreshnessStatus
from core.risk import (
    RiskInputError,
    breakeven_price,
    cost_per_contract,
    firm_limit_units,
    personal_limit_units,
    r_multiple,
    size_position,
)
from core.trade_management import TradeView, monitor_trade
from tests.conftest import T0, make_bars


class TestRMultiple:
    def test_long_and_short(self):
        assert r_multiple("long", 100.0, 98.0, 103.0) == pytest.approx(1.5)
        assert r_multiple("long", 100.0, 98.0, 99.0) == pytest.approx(-0.5)
        assert r_multiple("short", 100.0, 102.0, 97.0) == pytest.approx(1.5)
        assert r_multiple("short", 100.0, 102.0, 101.0) == pytest.approx(-0.5)

    def test_zero_or_negative_risk_rejected(self):
        with pytest.raises(RiskInputError):
            r_multiple("long", 100.0, 100.0, 101.0)
        with pytest.raises(RiskInputError):
            r_multiple("long", 100.0, 101.0, 101.0)
        with pytest.raises(RiskInputError):
            r_multiple("short", 100.0, 99.0, 98.0)
        with pytest.raises(RiskInputError):
            r_multiple("long", -1.0, -2.0, 1.0)


class TestSizing:
    def test_risk_budget_binding(self, nq):
        r = size_position(nq, "long", 20000.0, 19990.0, 1000.0)  # 40 ticks = $200/contract
        assert r.stop_ticks == 40 and r.risk_per_contract == 200.0
        assert r.final_contracts == 5 and r.binding_constraints == ["Risk budget"]

    def test_costs_included(self, nq):
        r = size_position(nq, "long", 20000.0, 19990.0, 1000.0, commission_rt=4.0, slippage_ticks_per_side=1)
        assert r.cost_per_contract == 14.0 and r.all_in_risk_per_contract == 214.0
        assert r.final_contracts == 4

    def test_most_restrictive_limit_wins(self, mnq):
        r = size_position(
            mnq,
            "long",
            20000.0,
            19990.0,
            10_000.0,
            available_drawdown=500.0,
            firm_max_contracts=50,
            personal_max_contracts=30,
            drawdown_buffer=100.0,
        )
        # $20/contract risk; drawdown (500-100)/20 = 20 is the tightest
        assert r.final_contracts == 20 and r.binding_constraints == ["Available drawdown"]
        assert r.remaining_drawdown_after_loss == pytest.approx(100.0)

    def test_never_exceeds_firm_or_personal(self, nq):
        r = size_position(nq, "long", 20000.0, 19999.0, 1e6, firm_max_contracts=5, personal_max_contracts=3)
        assert r.final_contracts == 3 and "Personal contract limit" in r.binding_constraints
        r2 = size_position(nq, "long", 20000.0, 19999.0, 1e6, firm_max_contracts=2, personal_max_contracts=3)
        assert r2.final_contracts == 2 and r2.binding_constraints == ["Firm contract limit"]

    def test_zero_contracts_warns(self, nq):
        r = size_position(nq, "long", 20000.0, 19900.0, 100.0)
        assert r.final_contracts == 0 and r.warnings

    def test_target_profit_and_rr(self, nq):
        r = size_position(nq, "short", 20000.0, 20010.0, 1000.0, target=19980.0)
        assert r.target_ticks == 80 and r.rr == 2.0
        assert r.potential_profit == pytest.approx(80 * 5.0 * r.final_contracts)

    def test_input_validation(self, nq):
        with pytest.raises(RiskInputError):
            size_position(nq, "long", 20000.1, 19990.0, 1000.0)  # off tick
        with pytest.raises(RiskInputError):
            size_position(nq, "long", 20000.0, 20010.0, 1000.0)  # stop on wrong side
        with pytest.raises(RiskInputError):
            size_position(nq, "long", 20000.0, 19990.0, 0.0)
        with pytest.raises(RiskInputError):
            size_position(nq, "long", 20000.0, 19990.0, 100.0, available_drawdown=-5)

    def test_firm_units_micro_vs_mini(self, nq, mnq):
        assert firm_limit_units(nq, 5, 50) == 5
        assert firm_limit_units(mnq, 5, 50) == 50
        assert firm_limit_units(mnq, 4, None) == 40
        assert firm_limit_units(nq, None, 40) == 4
        assert firm_limit_units(nq, None, None) is None
        assert personal_limit_units(mnq, 2) == 20 and personal_limit_units(nq, 2) == 2

    def test_breakeven_with_costs(self, nq, mnq):
        assert cost_per_contract(nq, 4.0, 1) == 14.0
        assert breakeven_price(nq, "long", 20000.0, 4.0, 1) == 20000.75  # 0.7 pts rounded up to tick
        assert breakeven_price(nq, "short", 20000.0, 4.0, 1) == 19999.25
        assert breakeven_price(mnq, "long", 20000.0, 1.5, 0, include_costs=False) == 20000.0


def view(direction="long", entry=100.0, stop=99.0, target=None, actions=None, qty=1):
    return TradeView(
        trade_id=7,
        instrument="NQ",
        contract="NQM26",
        direction=direction,
        entry_ts=T0,
        entry_price=entry,
        initial_stop=stop,
        quantity=qty,
        planned_target=target,
        commission_rt=0.0,
        slippage_ticks=0.0,
        actions=actions or [],
    )


def trend_bars(n=40, start=99.0, step=0.25, pull_every=None):
    rows = []
    p = start
    for i in range(n):
        o = p
        c = p + step
        if pull_every and i % pull_every == pull_every - 1:
            c = p - 3 * step
        rows.append((o, max(o, c) + 0.1, min(o, c) - 0.1, c))
        p = c
    return make_bars(rows, start=T0 - pd.Timedelta(minutes=10))


class TestTradeMonitor:
    def test_breakeven_review_once_at_threshold(self, nq):
        s = AppSettings().updated(
            "management", trailing_enabled=False, momentum_enabled=False, time_stop_bars=0
        )
        df = trend_bars()
        m = monitor_trade(view(), df, nq, s)
        be = [e for e in m.events if e.category == "breakeven_review"]
        assert len(be) == 1
        assert m.be_status[0]["reached"] and m.be_status[0]["price"] == 101.0
        assert "has not moved any stop" in be[0].message or "not changed any order" in be[0].message
        assert be[0].dedup_key == "trade:7:be:1.0"
        # Evaluating again produces the same key -> the alert layer stores it once.
        assert [e.dedup_key for e in monitor_trade(view(), df, nq, s).events] == [
            e.dedup_key for e in m.events
        ]

    def test_no_breakeven_prompt_when_user_already_moved_stop(self, nq):
        s = AppSettings().updated(
            "management", trailing_enabled=False, momentum_enabled=False, time_stop_bars=0
        )
        acts = [
            {
                "id": 1,
                "action_type": "stop_to_breakeven",
                "action_ts": (T0 + pd.Timedelta(minutes=1)).isoformat(),
                "new_stop": 100.0,
                "is_void": 0,
                "quantity": None,
            }
        ]
        m = monitor_trade(view(actions=acts), trend_bars(start=100.0), nq, s)
        assert not [e for e in m.events if e.category == "breakeven_review"]
        assert m.be_status[0]["already_at_be"] and m.registered_stop == 100.0

    def test_entry_bar_excluded(self, nq):
        # A spike inside the bar that contains the entry must not count as favourable excursion.
        rows = [(99.0, 110.0, 98.9, 99.0), (99.0, 99.2, 98.9, 99.1)]
        df = make_bars(rows, start=T0 - pd.Timedelta(minutes=2))  # first bar opened before entry
        m = monitor_trade(view(), df, nq, AppSettings())
        assert m.mfe_r < 1

    def test_trailing_suggestions_only_from_confirmed_swings_and_only_tighter(self, nq):
        s = AppSettings().updated(
            "swings",
            left_bars=2,
            right_bars=2,
            min_prominence_ticks=0,
            use_atr_prominence=False,
            duplicate_window_bars=0,
        )
        s = s.updated("management", momentum_enabled=False, time_stop_bars=0)
        m = monitor_trade(view(), trend_bars(60, pull_every=6), nq, s)
        assert m.trailing, "expected suggestions from pullback swing lows"
        for t_ in m.trailing:
            assert t_.confirm_ts > t_.pivot_ts  # confirmed strictly after the pivot
            assert not t_.provisional
            if t_.valid and t_.tighter:
                assert t_.suggested_stop > t_.registered_stop
        trail_events = [e for e in m.events if e.category == "trailing_stop"]
        assert len(trail_events) == len({e.dedup_key for e in trail_events})
        assert all(e.details["suggested_stop"] > 99.0 for e in trail_events)

    def test_looser_level_never_suggested(self, nq):
        s = AppSettings().updated(
            "swings",
            left_bars=2,
            right_bars=2,
            min_prominence_ticks=0,
            use_atr_prominence=False,
            duplicate_window_bars=0,
        )
        acts = [
            {
                "id": 1,
                "action_type": "stop_tightened",
                "action_ts": (T0 + pd.Timedelta(minutes=1)).isoformat(),
                "new_stop": 150.0 - 50,
                "is_void": 0,
                "quantity": None,
            }
        ]  # user stop at 100 (breakeven)
        m = monitor_trade(view(actions=acts), trend_bars(30, pull_every=6), nq, s)
        for e in m.events:
            if e.category == "trailing_stop":
                assert e.details["suggested_stop"] > 100.0

    def test_target_approach_and_reached(self, nq):
        s = AppSettings().updated(
            "management", trailing_enabled=False, momentum_enabled=False, time_stop_bars=0
        )
        m = monitor_trade(view(target=104.0), trend_bars(40), nq, s)
        kinds = [e.dedup_key.split(":")[-1] for e in m.events if e.category == "target_approaching"]
        assert kinds == ["target_approach", "target_reached"]

    def test_stop_touch_warning_and_short(self, nq):
        rows = [(100.0, 100.2, 99.8, 100.1), (100.1, 101.6, 100.0, 101.5)]
        df = make_bars(rows, start=T0)
        m = monitor_trade(view("short", 100.0, 101.0), df, nq, AppSettings())
        assert any(e.dedup_key == "trade:7:stop_touched:101.0" for e in m.events)
        assert m.current_r == pytest.approx(-1.5)

    def test_stale_data_cannot_assess(self, nq):
        stale = FreshnessStatus("stale", "live", 3600, "Last bar 60 min ago")
        m = monitor_trade(view(), trend_bars(), nq, AppSettings(), freshness=stale)
        assert not m.data_ok and "cannot reliably assess" in m.data_message
        assert any(e.category == "trade_warning" and "stale" in e.dedup_key for e in m.events)

    def test_no_data(self, nq):
        m = monitor_trade(view(), pd.DataFrame(), nq, AppSettings())
        assert not m.data_ok and m.current_price is None

    def test_closed_trade_stops_evaluating(self, nq):
        exit_ts = T0 + pd.Timedelta(minutes=20)
        acts = [
            {
                "id": 1,
                "action_type": "full_exit",
                "action_ts": exit_ts.isoformat(),
                "price": 100.5,
                "quantity": 1,
                "is_void": 0,
                "new_stop": None,
            }
        ]
        m = monitor_trade(view(actions=acts), trend_bars(40), nq, AppSettings())
        assert m.bars_evaluated <= 5 and m.open_qty == 0

    def test_voided_action_ignored(self, nq):
        acts = [
            {
                "id": 1,
                "action_type": "stop_tightened",
                "action_ts": (T0 + pd.Timedelta(minutes=1)).isoformat(),
                "new_stop": 99.75,
                "is_void": 1,
                "quantity": None,
            }
        ]
        assert view(actions=acts).registered_stop == 99.0

    def test_invalidation_warning_from_linked_plan(self, nq):
        tv = view(stop=98.0)
        tv.setup_plan = {"zone_lower": 99.5, "zone_upper": 100.0, "structural_invalidation": 99.25}
        rows = [(100.0, 100.1, 99.0, 99.1)]
        m = monitor_trade(tv, make_bars(rows, start=T0), nq, AppSettings())
        keys = {e.dedup_key.split(":")[-1] for e in m.events}
        assert {"retest_failed", "invalidation"} <= keys
        assert all(
            "not" in e.message.lower() or "advisory" in e.message.lower()
            for e in m.events
            if e.category == "trade_warning"
        )

    def test_prompts_pause_after_reported_stop_traded(self, nq):
        # Long that immediately trades through its stop, then trends: no further prompts once stopped.
        rows = [(100.0, 100.1, 98.5, 98.8)] + [
            (98.8 + i * 0.5, 99.4 + i * 0.5, 98.7 + i * 0.5, 99.3 + i * 0.5) for i in range(30)
        ]
        m = monitor_trade(view(), make_bars(rows, start=T0), nq, AppSettings())
        cats = [e.category for e in m.events]
        assert cats == ["trade_warning"] and "stop_touched" in m.events[0].dedup_key
        assert any("paused" in w for w in m.warnings)
        # Recording a different (actual) stop resumes guidance.
        acts = [
            {
                "id": 1,
                "action_type": "stop_widened",
                "action_ts": (T0 + pd.Timedelta(minutes=1)).isoformat(),
                "new_stop": 97.0,
                "is_void": 0,
                "quantity": None,
            }
        ]
        m2 = monitor_trade(view(actions=acts), make_bars(rows, start=T0), nq, AppSettings())
        assert any(e.category == "breakeven_review" for e in m2.events)

    def test_structure_warning_once_per_leg(self, nq):
        s = AppSettings().updated(
            "swings",
            left_bars=2,
            right_bars=2,
            min_prominence_ticks=0,
            use_atr_prominence=False,
            duplicate_window_bars=0,
        )
        s = s.updated("management", momentum_enabled=False, time_stop_bars=0, trailing_enabled=False)
        # short with a far stop; price grinds higher (against) making successive broken swing highs
        rows = []
        p = 100.0
        for i in range(60):
            o = p
            c = p + (0.5 if i % 4 != 3 else -0.75)
            rows.append((o, max(o, c) + 0.1, min(o, c) - 0.1, c))
            p = c
        m = monitor_trade(view("short", 100.0, 200.0), make_bars(rows, start=T0), nq, s)
        structure = [e for e in m.events if "structure" in e.dedup_key]
        assert len(structure) <= 1
