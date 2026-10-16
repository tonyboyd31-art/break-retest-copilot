"""Manual trade registration, user-recorded actions and advisory monitoring."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from core.config import AppSettings
from core.instruments import default_registry, is_on_tick
from core.scanner import ScanResult
from core.timeutils import from_iso, trade_date
from core.trade_management import TradeMonitor, monitor_trade, trade_view_from_rows
from data import repositories as repo
from data.database import Database
from data.models import ManualTradeInput, TradeActionInput
from notifications.manager import AlertManager
from services.market_service import MarketData


class TradeValidationError(ValueError):
    pass


def register_trade(db: Database, inp: ManualTradeInput) -> int:
    spec = default_registry().get(inp.instrument)
    errors = inp.check_against(spec)
    if inp.account_id is not None:
        acct = repo.get_account(db, inp.account_id)
        if acct is None:
            errors.append("Selected account does not exist.")
        else:
            inp.firm_code = acct["firm_code"]
    if errors:
        raise TradeValidationError(" ".join(errors))
    return repo.create_trade(db, inp)


def record_action(db: Database, inp: TradeActionInput) -> int:
    trade = repo.get_trade(db, inp.trade_id)
    if trade is None:
        raise TradeValidationError("Trade not found.")
    spec = default_registry().get(trade["instrument"])
    for name in ("price", "new_stop"):
        v = getattr(inp, name)
        if v is not None and not is_on_tick(v, spec.tick_size):
            raise TradeValidationError(
                f"{name.replace('_', ' ')} {v} is not a multiple of the tick size {spec.tick_size}."
            )
    try:
        return repo.add_action(db, inp)
    except ValueError as exc:
        raise TradeValidationError(str(exc)) from exc


def monitor_open_trades(
    db: Database,
    settings: AppSettings,
    md: MarketData,
    scan: ScanResult | None,
    alerts: AlertManager | None = None,
) -> dict[int, TradeMonitor]:
    out: dict[int, TradeMonitor] = {}
    for t in repo.list_trades(db, status="open"):
        if t["instrument"] != md.spec.symbol:
            continue
        actions = repo.list_actions(db, t["id"])
        setup = repo.get_setup(db, t["setup_id"]) if t.get("setup_id") else None
        tv = trade_view_from_rows(t, actions, setup)
        vol_regime = (setup or {}).get("context", {}).get("vol_regime") if setup else None
        mon = monitor_trade(
            tv, md.visible, md.spec, settings, scan.zone_history if scan else None, md.freshness, vol_regime
        )
        if tv.contract and md.contract and tv.contract != md.contract:
            mon.data_ok = False
            mon.data_message = (
                f"Loaded data is {md.contract} but the trade is in {tv.contract}: cannot reliably "
                "assess this trade."
            )
        out[t["id"]] = mon
        if alerts:
            alerts.publish_many(mon.events, md.context_key)
    return out


def journal_daily_pnl(db: Database, account_id: int) -> dict[date, float]:
    """Realized P&L per CME trade date from the user's closed journal trades for an account."""
    by_day: dict[date, float] = defaultdict(float)
    for t in repo.list_trades(db, status="closed"):
        if t["account_id"] != account_id or t["realized_pnl"] is None:
            continue
        ts = from_iso(t["closed_ts"]) or from_iso(t["entry_ts"])
        by_day[trade_date(ts)] += float(t["realized_pnl"])
    return dict(by_day)
