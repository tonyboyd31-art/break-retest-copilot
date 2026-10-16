"""Optional: seed DEMO accounts, daily P&L and one example manual trade (simulated data).

Everything created is prefixed "DEMO" so it is never mistaken for real records.

    python scripts/seed_demo_data.py            # uses BRC_DB_PATH or var/copilot.db
    python scripts/seed_demo_data.py --db var/demo.db

Delete the accounts/trades from the UI (or remove the database file) to clear them.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.retest_engine import SetupState  # noqa: E402
from data import repositories as repo  # noqa: E402
from data.models import AccountInput, DailyPnlInput, ManualTradeInput  # noqa: E402
from notifications.manager import AlertManager  # noqa: E402
from services.bootstrap import bootstrap  # noqa: E402
from services.market_service import MarketSelection, load_market  # noqa: E402
from services.scan_service import persist_scan, scan_market  # noqa: E402
from services.trade_service import register_trade  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--instrument", default="MNQ")
    args = ap.parse_args()
    app = bootstrap(args.db)
    s = app.settings()
    db = app.db
    existing = {a["nickname"] for a in repo.list_accounts(db)}

    def account(nick: str, firm: str, at: str, stage: str, size: float, pnl: list[tuple[date, float]]) -> int:
        if nick in existing:
            return next(a["id"] for a in repo.list_accounts(db) if a["nickname"] == nick)
        aid = repo.create_account(
            db,
            AccountInput(
                firm_code=firm,
                account_type_code=at,
                nickname=nick,
                stage=stage,
                account_size=size,
                starting_balance=size,
                notes="DEMO account with made-up P&L",
            ),
        )
        for d, p in pnl:
            repo.upsert_daily_pnl(db, DailyPnlInput(account_id=aid, trade_date=d, net_pnl=p, notes="DEMO"))
        return aid

    tf_id = account(
        "DEMO Tradeify Select 50K",
        "tradeify",
        "tradeify_select_eval_50k",
        "evaluation",
        50000,
        [
            (date(2026, 10, 7), 420.0),
            (date(2026, 10, 8), -310.0),
            (date(2026, 10, 9), 980.0),
            (date(2026, 10, 12), 260.0),
            (date(2026, 10, 13), -140.0),
            (date(2026, 10, 14), 515.0),
        ],
    )
    account(
        "DEMO Topstep Combine 50K",
        "topstep",
        "topstep_tc_50k",
        "evaluation",
        50000,
        [
            (date(2026, 10, 12), -620.0),
            (date(2026, 10, 13), 1150.0),
            (date(2026, 10, 14), 340.0),
            (date(2026, 10, 15), -900.0),
        ],
    )

    md = load_market(db, s, MarketSelection(source="synthetic", instrument=args.instrument))
    if not md.ok:
        print("No market data:", md.error)
        return 1
    scan = scan_market(md, s)
    persist_scan(db, md, scan, s, AlertManager(db, s.notifications))
    q = sorted((x for x in scan.setups if x.state == SetupState.QUALIFIED), key=lambda x: x.confirm_ts)
    if not q:
        print("No qualified setup available to base a demo trade on.")
        return 0
    st = q[-1]
    if not any(t["setup_id"] == st.setup_id for t in repo.list_trades(db, include_void=True)):
        tid = register_trade(
            db,
            ManualTradeInput(
                account_id=tf_id,
                instrument=args.instrument,
                contract=md.contract,
                direction=st.direction,
                entry_ts=st.confirm_ts,
                entry_price=st.plan.entry_price,
                initial_stop=st.plan.stop,
                quantity=2,
                planned_target=st.plan.target1,
                setup_id=st.setup_id,
                commission_per_contract_rt=s.risk.commission_for(args.instrument),
                slippage_ticks=s.risk.slippage_ticks_per_side,
                notes="DEMO trade based on a simulated setup",
            ),
        )
        print(
            f"Registered DEMO trade #{tid} from setup {st.setup_id} ({st.direction} @ {st.plan.entry_price})."
        )
    print("Demo data ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
