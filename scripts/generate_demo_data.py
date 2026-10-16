"""Export deterministic SIMULATED candles to CSV (for testing the CSV importer).

    python scripts/generate_demo_data.py --symbol MNQ --days 10 --out sample_data/MNQ_SIMULATED_5min.csv

The output uses naive America/New_York timestamps (bar OPEN times), the most
common platform export format, plus a contract column. THE PRICES ARE NOT REAL.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.instruments import get_instrument  # noqa: E402
from data.synthetic_data import generate_synthetic_candles  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MNQ")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--end", default="2026-10-16")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = generate_synthetic_candles(
        get_instrument(args.symbol), end=date.fromisoformat(args.end), days=args.days, seed=args.seed
    )
    out = Path(args.out or f"sample_data/{args.symbol}_SIMULATED_5min.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    export = df.copy()
    export.index = export.index.tz_convert("America/New_York").tz_localize(None)
    export.index.name = "timestamp"
    export["volume"] = export["volume"].astype(int)
    export.to_csv(out, float_format="%.2f")
    print(f"Wrote {len(export)} SIMULATED bars to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
