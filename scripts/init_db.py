"""Initialise (or migrate) the SQLite database and sync instruments + firm rules.

    python scripts/init_db.py               # BRC_DB_PATH or var/copilot.db
    python scripts/init_db.py --db var/other.db

Safe to run repeatedly: migrations are versioned and only missing ones run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.database import LATEST_VERSION  # noqa: E402
from services.bootstrap import bootstrap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    app = bootstrap(args.db)
    print(f"Database: {app.db.path}")
    print(f"Schema version: {app.db.current_version()} (latest {LATEST_VERSION})")
    print(f"Instruments: {', '.join(app.registry.symbols())}")
    print(f"Firms: {', '.join(f.name for f in app.firms.values())}")
    for w in app.settings_warnings:
        print("WARNING:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
