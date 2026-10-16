"""Drawdown models, consistency, payout estimates and firm rule configuration integrity."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from core.config import AppSettings
from data import repositories as repo
from data.models import AccountInput, DailyPnlInput, PayoutInput
from firms.base import FirmRules, RuleStatus
from firms.drawdown import (
    DailyRecord,
    EodTrailingDrawdown,
    IntradayTrailingDrawdown,
    StaticDrawdown,
    build_drawdown_model,
)
from firms.evaluation import consistency, evaluation_progress, payout_checklist
from firms.registry import get_firm, load_firms
from services.account_service import build_dashboard, contract_limit_for
from services.bootstrap import bootstrap

D0 = date(2026, 10, 5)


def recs(pnls, min_eq=None, max_eq=None, payouts=None):
    out = []
    for i, p in enumerate(pnls):
        out.append(
            DailyRecord(
                D0 + timedelta(days=i),
                p,
                None if min_eq is None else min_eq[i],
                None if max_eq is None else max_eq[i],
                0.0 if payouts is None else payouts[i],
            )
        )
    return out


class TestDrawdownModels:
    def test_static(self):
        r = StaticDrawdown(2000).run(50000, recs([500, -800, 300]))
        assert r.current_floor == 48000 and r.current_balance == 50000 and r.remaining_allowance == 2000

    def test_eod_trailing_floor_rises_never_falls(self):
        r = EodTrailingDrawdown(2000).run(50000, recs([1000, -500, 700, -2000]))
        floors = [d.floor_after for d in r.days]
        assert floors == [49000, 49000, 49200, 49200]
        assert all(b >= a for a, b in zip(floors, floors[1:], strict=False))
        assert r.current_balance == 49200 and r.remaining_allowance == 0
        assert r.breached and r.breach_date == D0 + timedelta(days=3)  # EOD balance touched the floor

    def test_eod_floor_only_moves_at_close(self):
        # intraday high never matters for an EOD model; the floor in force during day 2 is day 1's
        r = EodTrailingDrawdown(2000).run(50000, recs([1000, 0], max_eq=[52500, 51000]))
        assert r.days[1].floor_during_day == 49000

    def test_lock_at_offset(self):
        # Tradeify-style: locks at start + 100 once EOD balance reaches start + max_loss + 100
        r = EodTrailingDrawdown(2000, lock_floor_offset=100).run(50000, recs([1500, 700, 1000]))
        assert r.locked and r.current_floor == 50100
        assert [d.floor_after for d in r.days] == [49500, 50100, 50100]

    def test_lock_on_payout(self):
        r = EodTrailingDrawdown(2000, lock_floor_offset=100, lock_on_payout=True).run(
            50000, recs([800, 400, 0], payouts=[0, 500, 0])
        )
        assert r.locked and r.current_floor == 50100
        assert r.current_balance == 50700

    def test_topstep_xfa_style_reset_on_payout(self):
        r = EodTrailingDrawdown(2000, lock_floor_offset=0, lock_on_payout=True).run(
            0, recs([600, 300], payouts=[0, 200])
        )
        assert r.current_floor == 0 and r.locked

    def test_intraday_breach_detection(self):
        r = EodTrailingDrawdown(2000).run(50000, recs([200, 100], min_eq=[48500, 47900]))
        assert r.breached and r.breach_date == D0 + timedelta(days=1)
        assert r.days[1].breach_basis == "reported intraday low equity"

    def test_unknown_intraday_reported_not_assumed(self):
        r = EodTrailingDrawdown(2000).run(50000, recs([200, -100]))
        assert not r.breached and r.breach_unknown_days == 2
        assert all(d.breached is None for d in r.days)

    def test_intraday_trailing_uses_highs_and_flags_estimate(self):
        r = IntradayTrailingDrawdown(2000).run(50000, recs([500, 0], max_eq=[51500, None]))
        assert r.days[0].floor_after == 49500  # 51,500 intraday high - 2,000
        assert r.estimate and any("OVERSTATED" in a for a in r.assumptions)
        full = IntradayTrailingDrawdown(2000).run(50000, recs([500], max_eq=[51500]))
        assert not full.estimate

    def test_intraday_vs_eod_differ(self):
        days = recs([300], max_eq=[51800])
        eod = EodTrailingDrawdown(2000).run(50000, days)
        intra = IntradayTrailingDrawdown(2000).run(50000, days)
        assert intra.current_floor > eod.current_floor

    def test_builder(self):
        assert build_drawdown_model(None, 2000)[0] is None
        assert build_drawdown_model({"type": "eod_trailing"}, None)[0] is None
        assert isinstance(build_drawdown_model({"type": "static"}, 1000)[0], StaticDrawdown)
        with pytest.raises(ValueError):
            StaticDrawdown(0)


class TestConsistency:
    def test_largest_day_over_total(self):
        c = consistency(recs([600, -250, 900, 400]), "largest_day_over_total_profit", 0.4)
        assert c.ratio == pytest.approx(900 / 1650) and c.passes is False
        assert c.required_total_profit == pytest.approx(2250)

    def test_passes_when_spread(self):
        c = consistency(recs([500, 400, 450, 350]), "largest_day_over_total_profit", 0.4)
        assert c.passes is True

    def test_undefined_when_not_profitable(self):
        c = consistency(recs([300, -500]), "largest_day_over_total_profit", 0.4)
        assert c.ratio is None and c.passes is None and "Undefined" in c.explanation
        assert consistency([], "largest_day_over_total_profit", 0.5).passes is None

    def test_alternative_formula_and_validation(self):
        c = consistency(recs([1200, 300]), "largest_day_over_profit_target", 0.55, profit_target=3000)
        assert c.ratio == pytest.approx(0.4) and c.passes
        with pytest.raises(ValueError):
            consistency(recs([1]), "nope", 0.5)
        with pytest.raises(ValueError):
            consistency(recs([1]), "largest_day_over_total_profit", 1.5)


@pytest.fixture(scope="module")
def firms():
    return load_firms()


class TestConfigs:
    def test_both_firms_load_with_account_types(self, firms):
        assert {"tradeify", "topstep"} <= set(firms)
        assert len(firms["topstep"].account_types()) >= 6
        assert len(firms["tradeify"].account_types()) >= 7

    def test_every_rule_has_status_and_official_source(self, firms):
        for f in firms.values():
            for row in f.config_rule_rows():
                assert row["status"] in {s.value for s in RuleStatus}
                assert row.get("source_url", "").startswith("https://"), row
                host = row["source_url"].split("/")[2]
                assert host.endswith(("topstep.com", "tradeify.co")), f"non-official source {host}"

    def test_nothing_shipped_as_verified(self, firms):
        """Rules could not be fetched from official pages during the build - none may claim VERIFIED."""
        for f in firms.values():
            assert all(r["status"] != "VERIFIED" for r in f.config_rule_rows())
            assert all(not r.get("verified_on") for r in f.config_rule_rows())

    def test_drawdown_models_differ_by_firm_and_stage(self, firms):
        tc = firms["topstep"].rules_for("topstep_tc_50k").value("drawdown_model")
        flex = firms["tradeify"].rules_for("tradeify_select_flex_50k").value("drawdown_model")
        assert tc["lock_floor_offset"] is None and flex["lock_floor_offset"] == 100
        assert (
            firms["tradeify"].rules_for("tradeify_select_eval_50k").value("consistency")["threshold"] == 0.4
        )
        assert firms["tradeify"].rules_for("tradeify_growth_eval_50k").value("consistency") is None

    def test_unknown_values_left_null_not_invented(self, firms):
        xfa = firms["topstep"].rules_for("topstep_xfa_50k")
        assert xfa.value("max_loss") is None and xfa.value("starting_balance") is None
        model, msg = build_drawdown_model(xfa.value("drawdown_model"), xfa.value("max_loss"))
        assert model is None and "not configured" in msg

    def test_user_override_takes_precedence(self, firms):
        f = firms["tradeify"]
        user = [
            {
                "origin": "user",
                "account_type_code": "tradeify_select_eval_50k",
                "rule_key": "profit_target",
                "value": 2500,
                "status": "USER_CONFIRMED",
                "source_url": "https://help.tradeify.co/x",
                "verified_on": "2026-10-16",
            }
        ]
        rs = f.rules_for("tradeify_select_eval_50k", user)
        r = rs.get("profit_target")
        assert r.value == 2500 and r.status == RuleStatus.USER_CONFIRMED and r.trusted and r.origin == "user"
        assert "profit_target" not in rs.untrusted_keys()
        assert rs.stale_keys(date(2027, 1, 1), 30) == ["profit_target"]

    def test_unknown_account_type(self, firms):
        with pytest.raises(KeyError):
            firms["topstep"].rules_for("nope")

    def test_new_firm_from_config_only(self, tmp_path, firms):
        cfg = json.loads(firms["topstep"].config_path.read_text())
        cfg["firm_code"], cfg["firm_name"] = "examplefirm", "Example Firm"
        (tmp_path / "examplefirm.json").write_text(json.dumps(cfg))
        loaded = load_firms(tmp_path)
        assert type(loaded["examplefirm"]) is FirmRules

    def test_contract_limits_in_instrument_units(self, firms, nq, mnq):
        rs = firms["topstep"].rules_for("topstep_tc_50k")
        assert contract_limit_for(rs, nq)[0] == 5
        assert contract_limit_for(rs, mnq)[0] == 50


class TestEvaluationAndPayout:
    def test_evaluation_progress_select(self, firms):
        rs = firms["tradeify"].rules_for("tradeify_select_eval_50k")
        p = evaluation_progress(rs, "evaluation", 50000, recs([600, -250, 900, 400]), D0 + timedelta(days=3))
        assert p.net_profit == 1650 and p.remaining_target == 1350 and not p.target_reached
        assert p.status == "In progress" and "consistency" in p.next_milestone
        assert p.today_pnl == 400 and p.trading_days == 4
        assert "profit_target" in p.untrusted_rules
        assert "ESTIMATE" in p.label

    def test_requirements_met_is_only_an_estimate(self, firms):
        rs = firms["tradeify"].rules_for("tradeify_select_eval_50k")
        p = evaluation_progress(rs, "evaluation", 50000, recs([800, 700, 900, 750]), D0 + timedelta(days=3))
        assert p.target_reached and p.status == "Requirements appear met (estimate)"
        assert "Confirm" in p.next_milestone

    def test_cycle_reset_excludes_old_days(self, firms):
        rs = firms["tradeify"].rules_for("tradeify_select_eval_50k")
        p = evaluation_progress(
            rs,
            "evaluation",
            50000,
            recs([-1500, 600]),
            D0 + timedelta(days=1),
            cycle_start=D0 + timedelta(days=1),
        )
        assert p.net_profit == 600

    def test_payout_paths_and_new_cycle(self, firms):
        rs = firms["topstep"].rules_for("topstep_xfa_50k")
        days = recs([200, 160, 300, 150, 175, -50, 400])
        paths, notes = payout_checklist(rs, 0.0, days, [])
        std = next(p for p in paths if p.path == "Standard")
        assert std.overall == "appears_eligible"
        assert std.estimated_max_payout == pytest.approx(min(0.5 * 1335, 5000))
        assert any("not a payout approval" in n for n in notes)
        # After a payout on day 5 the winning-day count restarts and the positive-cycle rule applies.
        paths2, _ = payout_checklist(rs, 0.0, days, [(D0 + timedelta(days=4), 300.0)])
        std2 = next(p for p in paths2 if p.path == "Standard")
        wd = next(i for i in std2.items if i.name.startswith("Winning days"))
        assert wd.status == "not_met" and wd.detail.startswith("1 of 5")
        assert std2.overall == "not_eligible"

    def test_payout_caps_by_basis(self, firms):
        flex = firms["tradeify"].rules_for("tradeify_select_flex_50k")
        paths, _ = payout_checklist(flex, 50000, recs([3000, 2000, 1500, 1000, 1200]), [])
        assert paths[0].estimated_max_payout == 2500  # 50% of 8,700 capped at 2,500
        daily = firms["tradeify"].rules_for("tradeify_select_daily_50k")
        p2, _ = payout_checklist(daily, 50000, recs([300]), [])
        assert p2[0].estimated_max_payout == 600  # 2x cycle profit, under the 1,000 cap

    def test_no_payout_rules(self, firms):
        paths, notes = payout_checklist(firms["topstep"].rules_for("topstep_tc_50k"), 50000, recs([100]), [])
        assert paths == [] and any("No payout rules" in n for n in notes)


class TestAccountDashboard:
    def test_dashboard_events_and_warnings(self, tmp_path):
        app = bootstrap(tmp_path / "acct.db", load_env=False)
        db = app.db
        aid = repo.create_account(
            db,
            AccountInput(
                firm_code="topstep",
                account_type_code="topstep_tc_50k",
                nickname="T1",
                stage="evaluation",
                account_size=50000,
                starting_balance=50000,
                personal_daily_loss_limit=500,
            ),
        )
        today = date(2026, 10, 15)
        repo.upsert_daily_pnl(
            db, DailyPnlInput(account_id=aid, trade_date=today - timedelta(days=1), net_pnl=-1000)
        )
        repo.upsert_daily_pnl(db, DailyPnlInput(account_id=aid, trade_date=today, net_pnl=-600))
        repo.add_payout(db, PayoutInput(account_id=aid, requested_on=today, amount=10, status="denied"))
        d = build_dashboard(db, repo.get_account(db, aid), AppSettings(), today)
        cats = {e.category for e in d.events}
        assert {"rules_outdated", "drawdown_warning", "daily_risk"} <= cats
        assert any("NOT confirmed" in w for w in d.warnings)
        assert d.progress.drawdown.remaining_allowance == pytest.approx(400)
        assert d.progress.current_balance == 48400  # denied payout ignored
        # stable dedup keys
        d2 = build_dashboard(db, repo.get_account(db, aid), AppSettings(), today)
        assert sorted(e.dedup_key for e in d.events) == sorted(e.dedup_key for e in d2.events)

    def test_firm_registry_accessors(self):
        assert get_firm("topstep").name == "Topstep"
        with pytest.raises(KeyError):
            get_firm("nobody")
