"""Headless UI smoke test: open every page, report Streamlit exceptions, save screenshots.

Usage (with the app running):
    python scripts/ui_smoke.py --url http://localhost:8501 --out var/screenshots

Requires `pip install playwright` and a Chromium build (set PLAYWRIGHT_BROWSERS_PATH
or pass --chromium /path/to/chrome).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PAGES = [
    "",
    "live_signals",
    "levels",
    "trade_manager",
    "firm_accounts",
    "backtester",
    "analytics",
    "journal",
    "alerts",
    "settings",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", default="var/screenshots")
    ap.add_argument("--chromium", default=None)
    ap.add_argument("--pages", nargs="*", default=None)
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument(
        "--height",
        type=int,
        default=1000,
        help="Viewport height (Streamlit scrolls internally, so use a tall viewport to capture whole pages)",
    )
    ap.add_argument("--wait", type=float, default=6.0)
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    with sync_playwright() as p:
        launch = {"headless": True}
        if args.chromium:
            launch["executable_path"] = args.chromium
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        for name in args.pages or PAGES:
            url = f"{args.url}/{name}"
            page.goto(url)
            time.sleep(args.wait)
            # wait until Streamlit is idle (no running spinner)
            for _ in range(60):
                if page.locator("[data-testid='stStatusWidget']").count() == 0:
                    break
                time.sleep(1)
            errs = page.locator("[data-testid='stException']")
            n = errs.count()
            text = ""
            if n:
                failures += 1
                text = errs.first.inner_text()[:2000]
            shot = out / f"{name or 'overview'}.png"
            page.screenshot(path=str(shot), full_page=True)
            print(f"{'FAIL' if n else 'ok  '} {name or 'overview':15s} exceptions={n} -> {shot}")
            if text:
                print("    " + text.replace("\n", "\n    "))
        browser.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
