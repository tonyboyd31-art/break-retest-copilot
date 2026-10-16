"""Headless monitor: scan, persist and alert without the dashboard open (analysis only).

Runs the same strategy scan, trade-management checks and account checks as
the dashboard, then delivers de-duplicated alerts (in-app store + Telegram if
configured). It NEVER places, modifies or cancels orders.

    python scripts/run_monitor.py --instrument MNQ --source synthetic --once
    python scripts/run_monitor.py --instrument MNQ --source databento --interval 60

With --source synthetic/csv the monitor replays to the latest bar once (useful
for testing alert delivery). With a live/delayed source it loops every
--interval seconds until interrupted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data import repositories as repo  # noqa: E402
from notifications.manager import AlertManager  # noqa: E402
from services.account_service import build_dashboard  # noqa: E402
from services.bootstrap import bootstrap  # noqa: E402
from services.market_service import MarketSelection, load_market  # noqa: E402
from services.scan_service import persist_scan, scan_market  # noqa: E402
from services.trade_service import monitor_open_trades  # noqa: E402


def cycle(app, sel: MarketSelection) -> None:
    s = app.settings()
    am = AlertManager(app.db, s.notifications)
    md = load_market(app.db, s, sel)
    stamp = time.strftime("%H:%M:%S")
    if not md.ok:
        print(f"[{stamp}] data unavailable: {md.error}")
        return
    scan = scan_market(md, s)
    summary = persist_scan(app.db, md, scan, s, am)
    mons = monitor_open_trades(app.db, s, md, scan, am)
    acct_alerts = 0
    for a in repo.list_accounts(app.db, include_closed=False):
        dash = build_dashboard(app.db, a, s)
        acct_alerts += sum(r.created for r in am.publish_many(dash.events, f"account:{a['id']}"))
    fresh = md.freshness.status if md.freshness else "?"
    print(
        f"[{stamp}] {md.spec.symbol} {md.contract} as of {md.as_of} ({fresh}) | setups +{len(summary.inserted)} "
        f"~{len(summary.updated)} | new alerts {summary.alerts_created + acct_alerts} | open trades {len(mons)}"
    )
    for t in summary.alert_titles:
        print("    ALERT:", t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    ap.add_argument("--instrument", default="MNQ")
    ap.add_argument("--source", default="synthetic", choices=["synthetic", "csv", "databento"])
    ap.add_argument("--dataset", type=int, default=None, help="CSV dataset id (for --source csv)")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    app = bootstrap(args.db)
    sel = MarketSelection(source=args.source, instrument=args.instrument, dataset_id=args.dataset)
    print("Break & Retest Copilot monitor - analysis only, no order execution. Ctrl+C to stop.")
    if args.once or args.source != "databento":
        cycle(app, sel)
        return 0
    try:
        while True:
            cycle(app, sel)
            time.sleep(max(args.interval, 15))
    except KeyboardInterrupt:
        print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
