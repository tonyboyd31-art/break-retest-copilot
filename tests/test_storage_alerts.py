"""Database integrity rules, manual trades with audit trail, and alert de-duplication."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from core.config import AppSettings, NotificationSettings
from core.events import EngineEvent
from core.retest_engine import SetupState
from core.scanner import run_scan
from data import repositories as repo
from data.database import LATEST_VERSION, Database
from data.models import ManualTradeInput, TradeActionInput
from notifications.manager import AlertManager
from notifications.telegram import TelegramNotifier
from services.trade_service import TradeValidationError, record_action, register_trade

T = pd.Timestamp("2026-03-10 14:00", tz="UTC")


def trade_input(**kw):
    base = dict(
        instrument="NQ",
        contract="NQM26",
        direction="long",
        entry_ts=T,
        entry_price=20000.0,
        initial_stop=19990.0,
        quantity=3,
        planned_target=20030.0,
        commission_per_contract_rt=4.0,
    )
    base.update(kw)
    return ManualTradeInput(**base)


class TestSchema:
    def test_migrations_idempotent(self, tmp_path):
        d = Database(tmp_path / "a.db")
        assert d.migrate() == list(range(1, LATEST_VERSION + 1))
        assert d.migrate() == []
        assert d.current_version() == LATEST_VERSION

    def test_settings_round_trip(self, db):
        s = AppSettings().updated("scoring", min_score=85)
        repo.save_settings(db, s)
        loaded, warnings = repo.load_settings(db)
        assert loaded == s and not warnings

    def test_invalid_stored_settings_fall_back_with_warning(self, db):
        repo.save_settings(db, AppSettings())
        db.execute('UPDATE user_settings SET value_json = \'{"scoring": {"min_score": 500}}\'')
        loaded, warnings = repo.load_settings(db)
        assert loaded == AppSettings() and warnings

    def test_ephemeral_storage_detection(self, monkeypatch):
        import data.database as database

        monkeypatch.delenv("BRC_EPHEMERAL_STORAGE", raising=False)
        monkeypatch.setattr(database, "PROJECT_ROOT", Path("/home/me/break-retest-copilot"))
        assert not database.storage_is_ephemeral()
        monkeypatch.setattr(database, "PROJECT_ROOT", Path("/mount/src/break-retest-copilot"))
        assert database.storage_is_ephemeral()
        monkeypatch.setenv("BRC_EPHEMERAL_STORAGE", "0")
        assert not database.storage_is_ephemeral()
        monkeypatch.setenv("BRC_EPHEMERAL_STORAGE", "1")
        monkeypatch.setattr(database, "PROJECT_ROOT", Path("/home/me/break-retest-copilot"))
        assert database.storage_is_ephemeral()


class TestSetupPersistence:
    def test_terminal_setups_immutable_and_no_duplicates(self, db, synthetic_nq, nq):
        scan = run_scan(synthetic_nq.iloc[:3000], nq, AppSettings())
        res = repo.upsert_setups(db, scan.setups, "ctx", "synthetic", "simulated", scan.settings_hash)
        assert len(res["inserted"]) == len(scan.setups)
        again = repo.upsert_setups(db, scan.setups, "ctx", "synthetic", "simulated", scan.settings_hash)
        assert not again["inserted"] and not again["updated"]
        n = db.query_one("SELECT COUNT(*) AS n FROM setups")["n"]
        assert n == len(scan.setups)
        # Re-scoring a finished setup later (e.g. new settings) must not rewrite history.
        term = next(s for s in scan.setups if s.state == SetupState.REJECTED and s.score is not None)
        changed = copy.deepcopy(term)
        changed.score.total = 99.0
        repo.upsert_setups(db, [changed], "ctx", "synthetic", "simulated", "otherhash")
        assert repo.get_setup(db, term.setup_id)["score"] == term.score.total

    def test_pending_setup_moves_forward_only(self, db, synthetic_nq, nq):
        full = run_scan(synthetic_nq.iloc[:3000], nq, AppSettings())
        target = next(s for s in full.setups if s.retest_ts is not None and s.state != SetupState.REJECTED)
        early = run_scan(synthetic_nq.iloc[: target.breakout_idx + 1], nq, AppSettings())
        pending = early.setup(target.setup_id)
        assert pending is not None and not pending.is_terminal
        repo.upsert_setups(db, [pending], "c", "synthetic", "simulated", early.settings_hash)
        repo.upsert_setups(
            db, [full.setup(target.setup_id)], "c", "synthetic", "simulated", full.settings_hash
        )
        assert repo.get_setup(db, target.setup_id)["state"] == target.state.value
        # Rewinding the replay never moves a stored setup backwards.
        repo.upsert_setups(db, [pending], "c", "synthetic", "simulated", early.settings_hash)
        assert repo.get_setup(db, target.setup_id)["state"] == target.state.value


class TestManualTrades:
    def test_register_and_validation(self, db):
        tid = register_trade(db, trade_input())
        t = repo.get_trade(db, tid)
        assert t["status"] == "open" and t["quantity"] == 3
        with pytest.raises(TradeValidationError):
            register_trade(db, trade_input(entry_price=20000.1))  # off tick
        with pytest.raises(ValueError):
            trade_input(initial_stop=20010.0)  # stop on wrong side -> zero/negative risk

    def test_actions_partial_full_exit_pnl(self, db):
        tid = register_trade(db, trade_input())
        record_action(
            db,
            TradeActionInput(
                trade_id=tid,
                action_type="stop_to_breakeven",
                action_ts=T + pd.Timedelta(minutes=10),
                new_stop=20000.0,
            ),
        )
        record_action(
            db,
            TradeActionInput(
                trade_id=tid,
                action_type="partial_exit",
                action_ts=T + pd.Timedelta(minutes=20),
                price=20010.0,
                quantity=1,
            ),
        )
        assert repo.get_trade(db, tid)["status"] == "open"
        with pytest.raises(TradeValidationError):  # cannot partially exit the remaining 2 as "partial"
            record_action(
                db,
                TradeActionInput(
                    trade_id=tid,
                    action_type="partial_exit",
                    action_ts=T + pd.Timedelta(minutes=25),
                    price=20012.0,
                    quantity=2,
                ),
            )
        record_action(
            db,
            TradeActionInput(
                trade_id=tid, action_type="full_exit", action_ts=T + pd.Timedelta(minutes=30), price=20020.0
            ),
        )
        t = repo.get_trade(db, tid)
        # (10 pts x 1 + 20 pts x 2) x $20 - $4 x 3 commissions
        assert t["status"] == "closed" and t["realized_pnl"] == pytest.approx(988.0)
        assert t["pnl_source"] == "computed_from_user_exits"

    def test_action_before_entry_rejected(self, db):
        tid = register_trade(db, trade_input())
        with pytest.raises(TradeValidationError):
            record_action(
                db,
                TradeActionInput(
                    trade_id=tid,
                    action_type="stop_tightened",
                    action_ts=T - pd.Timedelta(minutes=1),
                    new_stop=19995.0,
                ),
            )

    def test_suggestion_never_inferred_as_action(self, db):
        tid = register_trade(db, trade_input())
        assert repo.list_actions(db, tid) == []

    def test_correction_preserves_audit_trail(self, db):
        tid = register_trade(db, trade_input())
        aid = record_action(
            db,
            TradeActionInput(
                trade_id=tid, action_type="full_exit", action_ts=T + pd.Timedelta(minutes=30), price=20200.0
            ),
        )
        assert repo.get_trade(db, tid)["status"] == "closed"
        repl = TradeActionInput(
            trade_id=tid, action_type="full_exit", action_ts=T + pd.Timedelta(minutes=30), price=20020.0
        )
        new_id = repo.correct_action(db, aid, repl, "typo in fill price")
        acts = repo.list_actions(db, tid)
        orig = next(a for a in acts if a["id"] == aid)
        assert orig["is_void"] == 1 and orig["void_reason"] == "typo in fill price"
        assert next(a for a in acts if a["id"] == new_id)["corrects_action_id"] == aid
        t = repo.get_trade(db, tid)
        assert t["realized_pnl"] == pytest.approx(20 * 20 * 3 - 12)
        events = repo.list_audit(db, "trade_action")
        assert {e["action"] for e in events} >= {"recorded", "voided"}
        with pytest.raises(ValueError):
            repo.correct_action(db, aid, None, "again")  # already void
        with pytest.raises(ValueError):
            repo.correct_action(db, new_id, None, "  ")  # reason required

    def test_journal_edit_audited(self, db):
        tid = register_trade(db, trade_input())
        repo.update_trade_fields(
            db, tid, reason="added notes", notes="chased entry", rule_adherence="broke rules"
        )
        ev = repo.list_audit(db, "trade", tid)
        assert ev[0]["action"] == "journal_edit"
        with pytest.raises(ValueError):
            repo.update_trade_fields(db, tid, entry_price=1.0)  # core fields are not editable


def ev(key="k1", cat="setup_qualified", sev=None):
    return EngineEvent(
        category=cat, ts=T, title="t", message="m", dedup_key=key, instrument="NQ", severity=sev
    )


class TestAlerts:
    def test_duplicate_suppression(self, db):
        am = AlertManager(db, NotificationSettings(), TelegramNotifier(token="", chat_id=""))
        first = am.publish(ev())
        second = am.publish(ev())
        assert first.created and not second.created and second.suppressed_reason.startswith("duplicate")
        assert len(repo.list_alerts(db)) == 1

    def test_disabled_category_not_stored(self, db):
        ns = NotificationSettings()
        ns.categories["setup_rejected"].enabled = False
        am = AlertManager(db, ns, TelegramNotifier(token="", chat_id=""))
        assert not am.publish(ev(cat="setup_rejected")).created
        assert repo.list_alerts(db) == []

    def test_telegram_not_configured_recorded_without_secret(self, db):
        ns = NotificationSettings(telegram_enabled=True)
        am = AlertManager(db, ns, TelegramNotifier(token="", chat_id=""))
        r = am.publish(ev())
        assert r.deliveries == {"in_app": "stored", "telegram": "not_configured"}
        d = repo.list_alerts(db)[0]["deliveries"]
        assert {x["channel"] for x in d} == {"in_app", "telegram"}

    def test_secret_scrubbed_from_errors(self):
        tg = TelegramNotifier(token="SECRET123", chat_id="42")
        assert "SECRET123" not in tg.describe()
        assert (
            tg._scrub("failed for https://api.telegram.org/botSECRET123/sendMessage")
            == "failed for https://api.telegram.org/bot***/sendMessage"
        )

    def test_acknowledgement_and_severity(self, db):
        am = AlertManager(db, NotificationSettings(), TelegramNotifier(token="", chat_id=""))
        aid = am.publish(ev(sev="critical")).alert_id
        repo.acknowledge_alert(db, aid, "reviewed", "seen")
        a = repo.list_alerts(db)[0]
        assert a["acknowledged"] == 1 and a["user_response"] == "reviewed" and a["severity"] == "critical"
        assert repo.count_unacknowledged(db) == {}

    def test_action_links_alert(self, db):
        am = AlertManager(db, NotificationSettings(), TelegramNotifier(token="", chat_id=""))
        tid = register_trade(db, trade_input())
        aid = am.publish(
            EngineEvent(
                category="breakeven_review",
                ts=T,
                title="BE",
                message="review",
                dedup_key=f"trade:{tid}:be:1.0",
                trade_id=tid,
            )
        ).alert_id
        act = record_action(
            db,
            TradeActionInput(
                trade_id=tid,
                action_type="stop_to_breakeven",
                action_ts=T + pd.Timedelta(minutes=5),
                new_stop=20000.0,
                alert_id=aid,
            ),
        )
        a = repo.list_alerts(db, trade_id=tid)[0]
        assert a["user_response"] == "action_recorded" and a["linked_action_id"] == act


def test_explainer_number_guard():
    from services.explainer import unsupported_numbers

    facts = "Entry 17,024.00, stop 17,043.50, T1 16,985.00, score 82/100, 78 ticks."
    assert (
        unsupported_numbers("Entry near 17,024 with a stop at 17,043.50 and target 16,985.", facts) == set()
    )
    assert unsupported_numbers("Expect a move to 16,900.25 soon.", facts) == {"16900.25"}
