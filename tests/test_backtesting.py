"""Backtester fill rules, costs, overlap policy, IS/OOS split and metrics."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from core.backtesting import (
    BacktestConfig,
    _Arrays,
    evaluate_signal_outcome,
    rescore,
    run_backtest,
    simulate_setups,
    simulate_trade,
    threshold_sensitivity,
)
from core.config import AppSettings
from core.performance import compute_metrics, equity_curve, score_bucket
from core.retest_engine import Setup, SetupState
from core.trade_planning import TradePlan
from tests.conftest import make_bars

T = pd.Timestamp("2026-03-10 14:00", tz="UTC")  # 10:00 ET


def plan(entry=100.0, stop=99.0, target=102.0, method="confirmation_close") -> TradePlan:
    return TradePlan(
        instrument="NQ",
        direction="long",
        zone_lower=99.0,
        zone_upper=99.75,
        breakout_price=100.5,
        retest_price=99.5,
        confirmation_price=entry,
        entry_method=method,
        entry_price=entry,
        entry_zone_low=99.75,
        entry_zone_high=entry,
        structural_invalidation=99.25,
        stop=stop,
        target1=target,
        target2=None,
        target1_basis="structure",
        target2_basis="none",
        risk_points=entry - stop,
        risk_ticks=(entry - stop) * 4,
        risk_per_contract=(entry - stop) * 20,
        reward_points=target - entry,
        reward_ticks=(target - entry) * 4,
        rr=(target - entry) / (entry - stop),
        rr_extended=None,
        opposing_zone=None,
        valid=True,
    )


def setup_at(confirm_idx: int, p: TradePlan, direction="long") -> Setup:
    s = Setup(
        setup_id=f"s{confirm_idx}",
        instrument="NQ",
        contract="NQM26",
        timeframe="5min",
        direction=direction,
        zone={"zone_id": 1, "lower": 99.0, "upper": 99.75},
        breakout_idx=max(confirm_idx - 2, 0),
        breakout_ts=T,
        breakout_bar={"o": 99.8, "h": 100.6, "l": 99.7, "c": 100.5},
        atr_at_breakout=1.0,
        trade_date=date(2026, 3, 10),
    )
    s.plan = p
    s.confirm_idx = confirm_idx
    s.state = SetupState.QUALIFIED
    return s


def run(rows, p=None, cfg=None, start=T):
    df = make_bars(rows, start=start)
    arr = _Arrays(df, "5min")
    from core.instruments import get_instrument

    return simulate_trade(
        setup_at(0, p or plan()),
        arr,
        get_instrument("NQ"),
        cfg or BacktestConfig(slippage_ticks_per_side=0, force_flat_time=None),
    )


CONF = (99.8, 100.1, 99.7, 100.0)  # confirmation bar closes at 100.0 (entry)


class TestFills:
    def test_target_hit(self):
        tr = run([CONF, (100.0, 100.5, 99.6, 100.4), (100.4, 102.2, 100.3, 102.0)])
        assert tr.exit_reason == "target" and tr.exit_price == 102.0 and not tr.ambiguous

    def test_same_bar_stop_and_target_is_a_loss(self):
        tr = run([CONF, (100.0, 102.5, 98.5, 101.0)])
        assert tr.exit_reason == "stop_ambiguous" and tr.ambiguous and tr.exit_price == 99.0

    def test_gap_through_stop_fills_at_open(self):
        tr = run([CONF, (98.0, 98.5, 97.5, 98.2)])
        assert tr.exit_reason == "stop_gap" and tr.exit_price == 98.0

    def test_gap_through_target_fills_at_target(self):
        tr = run([CONF, (103.0, 103.5, 102.8, 103.2)])
        assert tr.exit_reason == "target_gap" and tr.exit_price == 102.0

    def test_slippage_on_entry_and_stop(self):
        cfg = BacktestConfig(slippage_ticks_per_side=1, force_flat_time=None)
        tr = run([CONF, (100.0, 100.2, 98.9, 99.1)], cfg=cfg)
        assert tr.entry_price == 100.25 and tr.exit_price == 98.75

    def test_session_flat_exit(self):
        start = pd.Timestamp("2026-03-10 15:50", tz="America/New_York").tz_convert("UTC")
        cfg = BacktestConfig(slippage_ticks_per_side=0, force_flat_time="16:00")
        tr = run([CONF, (100.0, 100.4, 99.8, 100.2), (100.2, 100.5, 100.0, 100.3)], cfg=cfg, start=start)
        assert tr.exit_reason == "session_flat" and tr.exit_price == 100.2

    def test_breakeven_effective_next_bar_only(self):
        cfg = BacktestConfig(
            slippage_ticks_per_side=0, force_flat_time=None, simulate_breakeven=True, breakeven_r=1.0
        )
        # bar 1 reaches +1R; bar 2 trades back to entry -> breakeven stop
        tr = run([CONF, (100.0, 101.1, 99.9, 101.0), (101.0, 101.1, 99.9, 100.0)], cfg=cfg)
        assert tr.exit_reason == "breakeven_stop" and tr.exit_price == 100.0
        # same bar reaches +1R and the ORIGINAL stop: the move is not credited within the bar
        tr2 = run([CONF, (100.0, 101.1, 98.9, 99.0)], cfg=cfg)
        assert tr2.exit_reason == "stop" and tr2.exit_price == 99.0

    def test_zone_edge_limit(self):
        p = plan(entry=99.75, stop=99.0, target=101.25, method="zone_edge")
        tr = run([CONF, (100.0, 100.3, 99.9, 100.2), (100.2, 100.4, 100.0, 100.3)], p=p)
        assert tr.exit_reason == "not_filled"
        tr2 = run([CONF, (100.0, 101.5, 99.7, 101.4), (101.4, 101.5, 101.0, 101.2)], p=p)
        # filled during bar 1 but the target is not credited on the fill bar
        assert tr2.entry_price == 99.75 and tr2.exit_idx == 2 and tr2.exit_price == 101.25
        assert tr2.exit_reason == "target_gap"  # bar 2 opened above the target -> filled at the target

    def test_end_of_data(self):
        tr = run([CONF, (100.0, 100.3, 99.9, 100.2)])
        assert tr.exit_reason == "end_of_data"


class TestMetrics:
    def test_compute_metrics(self):
        t = pd.DataFrame({"pnl_net": [100.0, -50.0, -50.0, 200.0, -25.0], "r_multiple": [2, -1, -1, 4, -0.5]})
        m = compute_metrics(t)
        assert m["trades"] == 5 and m["wins"] == 2 and m["win_rate"] == pytest.approx(0.4)
        assert m["profit_factor"] == pytest.approx(300 / 125)
        assert m["net_pnl"] == 175 and m["expectancy"] == 35
        assert m["max_drawdown"] == -100 and m["max_losing_streak"] == 2
        assert m["expectancy_r"] == pytest.approx(0.7)
        assert compute_metrics(pd.DataFrame())["trades"] == 0

    def test_equity_and_buckets(self):
        t = pd.DataFrame({"pnl_net": [10.0, -5.0], "exit_ts": [T, T + pd.Timedelta(minutes=5)]})
        eq = equity_curve(t)
        assert list(eq["equity"]) == [10.0, 5.0] and list(eq["drawdown"]) == [0.0, -5.0]
        assert score_bucket(83) == "80-84" and score_bucket(None) == "n/a" and score_bucket(95) == "90+"


@pytest.fixture(scope="module")
def bt(synthetic_nq):
    from core.instruments import get_instrument

    s = AppSettings()
    cfg = BacktestConfig(commission_rt=4.0, slippage_ticks_per_side=1, oos_fraction=0.3)
    return run_backtest(synthetic_nq, get_instrument("NQ"), s, cfg, data_is_synthetic=True), s, cfg


class TestRunBacktest:
    def test_trades_from_qualified_setups_only(self, bt):
        res, s, _ = bt
        qualified = {x.setup_id for x in res.scan.setups if x.state == SetupState.QUALIFIED}
        assert set(res.trades["setup_id"]) <= qualified
        assert len(res.trades) >= 1

    def test_rescore_matches_scanner_qualification(self, bt):
        res, s, _ = bt
        for x in res.scan.setups:
            ok, _ = rescore(x, s, s.scoring.min_score, s.planning.min_rr)
            assert ok == (x.state == SetupState.QUALIFIED), x.setup_id

    def test_costs_applied(self, bt):
        res, _, _ = bt
        t = res.trades.iloc[0]
        assert t["costs"] == 4.0 and t["pnl_net"] == pytest.approx(t["pnl_gross"] - 4.0)

    def test_no_overlapping_positions(self, synthetic_nq, bt, nq):
        res, s, cfg = bt
        loose = simulate_setups(res.scan, synthetic_nq, nq, s, BacktestConfig(min_score=50, min_rr=1.0))
        ordered = loose.sort_values("entry_ts")
        assert (ordered["entry_ts"].iloc[1:].to_numpy() >= ordered["exit_ts"].iloc[:-1].to_numpy()).all()

    def test_is_oos_split_chronological(self, bt):
        res, _, _ = bt
        assert res.oos_cutoff is not None
        assert set(res.trades["sample"]) <= {"IS", "OOS"}
        assert ((res.trades["sample"] == "OOS") == (res.trades["entry_ts"] >= res.oos_cutoff)).all()

    def test_honesty_warnings(self, bt):
        res, _, _ = bt
        joined = " ".join(res.warnings)
        assert "SYNTHETIC" in joined and "Single historical period" in joined

    def test_threshold_grid_monotonic_candidates(self, bt, synthetic_nq, nq):
        res, s, cfg = bt
        grid = threshold_sensitivity(res.scan, synthetic_nq, nq, s, cfg, [60, 80], [1.5, 2.0])
        assert set(grid["sample"]) == {"IS", "OOS"} and len(grid) == 8
        n60 = sum(rescore(x, s, 60, 1.5)[0] for x in res.scan.setups)
        n80 = sum(rescore(x, s, 80, 2.0)[0] for x in res.scan.setups)
        assert n60 >= n80

    def test_hypothetical_outcome_labelled(self, bt, synthetic_nq, nq):
        res, s, _ = bt
        q = next(x for x in res.scan.setups if x.state == SetupState.QUALIFIED)
        out = evaluate_signal_outcome(q, synthetic_nq, nq, s)
        assert out is not None and "HYPOTHETICAL" in out["notes"]
        assert out["outcome"] in {
            "target",
            "target_gap",
            "stop",
            "stop_gap",
            "stop_ambiguous",
            "session_flat",
            "open",
            "breakeven_stop",
            "trailing_stop",
            "time",
        }
