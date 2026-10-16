"""Account dashboards: rules, drawdown, evaluation progress and payout estimates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from core.config import AppSettings
from core.events import EngineEvent
from core.instruments import InstrumentSpec
from core.risk import firm_limit_units, personal_limit_units
from core.timeutils import utcnow
from data import repositories as repo
from data.database import Database
from data.models import DailyPnlInput
from firms.base import FirmRules, RuleSet
from firms.drawdown import DailyRecord
from firms.evaluation import EvaluationProgress, PayoutPathResult, evaluation_progress, payout_checklist
from firms.registry import get_firm
from services.trade_service import journal_daily_pnl


@dataclass
class AccountDashboard:
    account: dict
    firm: FirmRules
    rules: RuleSet
    records: list[DailyRecord]
    progress: EvaluationProgress
    payout_paths: list[PayoutPathResult]
    payout_notes: list[str]
    interpretation: list[str]
    warnings: list[str] = field(default_factory=list)
    events: list[EngineEvent] = field(default_factory=list)


def account_rules(db: Database, account: dict) -> RuleSet:
    firm = get_firm(account["firm_code"])
    return firm.rules_for(
        account["account_type_code"], repo.list_rule_rows(db, firm.code, account["account_type_code"])
    )


def daily_records(db: Database, account_id: int) -> list[DailyRecord]:
    rows = repo.list_daily_pnl(db, account_id)
    payouts = [p for p in repo.list_payouts(db, account_id) if p["status"] in ("requested", "paid")]
    by_day: dict[date, DailyRecord] = {}
    for r in rows:
        d = date.fromisoformat(r["trade_date"])
        by_day[d] = DailyRecord(d, float(r["net_pnl"]), r["min_equity"], r["max_equity"])
    for p in payouts:
        d = date.fromisoformat(p["requested_on"])
        rec = by_day.setdefault(d, DailyRecord(d, 0.0))
        rec.payout += float(p["amount"])
    return [by_day[d] for d in sorted(by_day)]


def contract_limit_for(rules: RuleSet, spec: InstrumentSpec) -> tuple[int | None, str]:
    minis = rules.value("max_contracts_minis")
    micros = rules.value("max_contracts_micros")
    n = firm_limit_units(spec, minis, micros)
    status = [
        r.status.value
        for r in (rules.get("max_contracts_minis"), rules.get("max_contracts_micros"))
        if r is not None
    ]
    if n is None:
        return None, "No firm contract limit configured (scaling plan / limit not entered)."
    return n, f"{minis} minis / {micros} micros ({', '.join(sorted(set(status))) or 'n/a'})"


def build_dashboard(
    db: Database, account: dict, settings: AppSettings, today: date | None = None
) -> AccountDashboard:
    today = today or utcnow().tz_convert("America/New_York").date()
    firm = get_firm(account["firm_code"])
    rules = account_rules(db, account)
    recs = daily_records(db, account["id"])
    cycle = date.fromisoformat(account["cycle_start_date"]) if account.get("cycle_start_date") else None
    paths_rule = rules.value("payout_paths") or []
    win_thr = float(paths_rule[0].get("winning_day_threshold", 0)) if paths_rule else 0.0
    prog = evaluation_progress(
        rules, account["stage"], float(account["starting_balance"]), recs, today, cycle, win_thr
    )
    payouts = [
        (date.fromisoformat(p["requested_on"]), float(p["amount"]))
        for p in repo.list_payouts(db, account["id"])
        if p["status"] in ("requested", "paid")
    ]
    paths, pnotes = payout_checklist(
        rules, float(account["starting_balance"]), recs, payouts, cycle, prog.drawdown
    )
    warnings: list[str] = []
    events: list[EngineEvent] = []
    untrusted = rules.untrusted_keys()
    if untrusted:
        warnings.append(
            f"{len(untrusted)} rule(s) for this account are not confirmed: {', '.join(untrusted)}."
        )
    stale = rules.stale_keys(today, settings.notifications.rule_staleness_days)
    if stale:
        warnings.append(
            f"Rule verification older than {settings.notifications.rule_staleness_days} days: "
            f"{', '.join(stale)}."
        )
    if not account.get("drawdown_model_confirmed"):
        warnings.append(
            "Drawdown model applicability NOT confirmed for this account - drawdown figures are estimates."
        )
    if (
        account.get("current_balance") is not None
        and abs(float(account["current_balance"]) - prog.current_balance) > 0.5
    ):
        warnings.append(
            f"Entered current balance ${float(account['current_balance']):,.2f} differs from the balance "
            f"implied by your daily P&L records ${prog.current_balance:,.2f}."
        )
    if untrusted or stale:
        h = hashlib.sha1(",".join(sorted(untrusted + stale)).encode()).hexdigest()[:8]
        events.append(
            EngineEvent(
                category="rules_outdated",
                ts=utcnow(),
                title=f"Unconfirmed firm rules: {account['nickname']}",
                message=warnings[0] if warnings else "Rules need confirmation.",
                dedup_key=f"account:{account['id']}:rules:{h}",
                account_id=account["id"],
            )
        )
    dd = prog.drawdown
    max_loss = rules.value("max_loss")
    if dd is not None and max_loss:
        frac = dd.remaining_allowance / float(max_loss)
        if frac <= settings.risk.drawdown_warning_fraction:
            events.append(
                EngineEvent(
                    category="drawdown_warning",
                    ts=utcnow(),
                    title=f"Low drawdown allowance: {account['nickname']}",
                    message=(
                        f"Estimated remaining allowance ${dd.remaining_allowance:,.2f} ({frac:.0%} of "
                        f"${float(max_loss):,.2f}). Based on your records - check the official dashboard."
                    ),
                    dedup_key=f"account:{account['id']}:dd:{today.isoformat()}",
                    account_id=account["id"],
                )
            )
    limit = account.get("personal_daily_loss_limit") or settings.risk.personal_daily_loss_limit
    if limit and prog.today_pnl is not None and prog.today_pnl < 0:
        used = -prog.today_pnl / float(limit)
        if used >= settings.risk.daily_loss_alert_fraction:
            lvl = "reached" if used >= 1 else f"{used:.0%} used"
            events.append(
                EngineEvent(
                    category="daily_risk",
                    ts=utcnow(),
                    title=f"Daily loss limit {lvl}: {account['nickname']}",
                    message=f"Today's recorded P&L ${prog.today_pnl:,.2f} vs personal limit ${float(limit):,.2f}.",
                    dedup_key=f"account:{account['id']}:daily:{today.isoformat()}:{min(int(used * 100) // 20, 5)}",
                    account_id=account["id"],
                )
            )
    return AccountDashboard(
        account, firm, rules, recs, prog, paths, pnotes, firm.interpretation_notes(rules), warnings, events
    )


def import_journal_into_daily(db: Database, account_id: int, overwrite: bool = False) -> int:
    existing = {r["trade_date"]: r for r in repo.list_daily_pnl(db, account_id)}
    n = 0
    for d, pnl in sorted(journal_daily_pnl(db, account_id).items()):
        if d.isoformat() in existing and not overwrite:
            continue
        repo.upsert_daily_pnl(
            db,
            DailyPnlInput(
                account_id=account_id,
                trade_date=d,
                net_pnl=round(pnl, 2),
                source="journal",
                notes="Aggregated from journal trades",
            ),
        )
        n += 1
    return n


def drawdown_frame(progress: EvaluationProgress) -> pd.DataFrame:
    dd = progress.drawdown
    if dd is None or not dd.days:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "date": d.trade_date,
                "balance": d.eod_balance,
                "floor": d.floor_after,
                "allowance": d.allowance_after,
                "breach": d.breached,
                "basis": d.breach_basis,
                "locked": d.locked,
                "note": d.note,
            }
            for d in dd.days
        ]
    )


def account_contract_limits(
    db: Database, account: dict | None, spec: InstrumentSpec, settings: AppSettings
) -> tuple[int | None, int | None, str]:
    """(firm limit, personal limit) in contracts of ``spec`` for the risk calculator."""
    firm_n, detail = (None, "No account selected.")
    personal = personal_limit_units(spec, settings.risk.personal_max_contracts_minis)
    if account is not None:
        firm_n, detail = contract_limit_for(account_rules(db, account), spec)
        if account.get("personal_max_contracts"):
            personal = personal_limit_units(spec, float(account["personal_max_contracts"]))
    return firm_n, personal, detail
