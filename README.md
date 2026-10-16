# Break & Retest Copilot

A personal futures-trading **analysis and decision-support** application for a discretionary
break-and-retest trader using Tradeify and Topstep accounts. It finds support/resistance, detects
break-and-retest setups on 5-minute candles, scores them transparently, proposes advisory trade plans,
watches the trades **you** register, and tracks firm-specific risk, evaluation and payout estimates.

> **It is not a trading bot.** The application never submits, modifies or cancels orders, never moves a
> stop, never takes a partial and never closes a position. It has no broker connection at all. Every
> level it shows is a suggestion for your manual review, and every trade decision is yours.

---

## Contents

1. [Quick start](#quick-start)
2. [What it does](#what-it-does)
3. [Safety model](#safety-model)
4. [Architecture](#architecture)
5. [Installation in detail](#installation-in-detail)
6. [Demo mode](#demo-mode)
7. [Using your own data (CSV)](#using-your-own-data-csv)
8. [Live / delayed data integration](#live--delayed-data-integration)
9. [Environment variables](#environment-variables)
10. [Strategy rules (what the engine actually does)](#strategy-rules)
11. [Prop-firm rules and the verification process](#prop-firm-rules-and-the-verification-process)
12. [Tests and quality checks](#tests-and-quality-checks)
13. [Troubleshooting](#troubleshooting)
14. [Known limitations](#known-limitations)
15. [Security considerations](#security-considerations)
16. [Project structure](#project-structure)

---

## Quick start

```bash
cd break-retest-copilot
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt   # exact tested versions (or: pip install -e ".[dev]")
python scripts/init_db.py          # optional - the app also migrates on first start
python -m streamlit run app.py
```

Open http://localhost:8501. Demo mode works immediately: no API keys, no accounts, no paid services.
Optionally add demo accounts and an example trade: `python scripts/seed_demo_data.py`.

Requirements: Python 3.11+ (tested on 3.13.16).

### Hosting on Streamlit Community Cloud (no local install)

The repository deploys as-is: Community Cloud installs `requirements.txt` (runtime packages only).

1. Push the project to a GitHub repository (private is fine).
2. At https://share.streamlit.io sign in with GitHub, choose **Create app -> Deploy a public app from
   GitHub**, select the repository, branch `main` and main file `app.py`.
3. Under **Advanced settings** pick Python **3.13**, then **Deploy**. Optional keys (Telegram, Claude) go in
   the app's **Secrets** as `KEY = "value"` lines; root-level secrets reach the app as environment variables.
4. Restrict who can open the app from its **Share** settings.

**Storage is temporary there.** The container's file system - including the SQLite database - is reset on
every reboot, sleep or redeploy, so trades, journal entries, accounts and settings can be lost. The app shows
a warning banner when it detects this. Use the hosted copy for demo mode and exploring; keep a real journal on
a local install.

---

## What it does

| Page | What you get |
|---|---|
| **Overview** | Feed status (live / delayed / historical / simulated, fresh / stale), pending and qualified setups, open trades with current R, account risk cards, recent alerts. |
| **Live Market Signals** | Candlestick chart with zones, confirmed swings, VWAP and breakout / retest / confirmation markers; qualified signals with the full advisory plan, score breakdown, breakout checks and state history; pending setups; every rejected / expired / invalidated setup with its reason. Replay auto-advance or 60-second refresh. Optional plain-English rephrasing by Claude. |
| **Support & Resistance** | Zones with reactions, tests, holds, failures, break / retest status, distance, relevance and an explainable strength breakdown; confirmed swing table with pivot vs confirmation times; provisional (unconfirmed) pivots shown separately. |
| **Trade Manager** | Register a trade you placed (optionally linked to a setup); R-multiple tracking; breakeven-review prompts; profit-protection prompts; trailing-stop suggestions from newly confirmed swings; thesis / invalidation warnings; record what you actually did (append-only audit trail with corrections); standalone risk / position-size calculator. |
| **Firm & Account Manager** | Multiple Tradeify / Topstep accounts; per-account drawdown model, evaluation progress, consistency ratio, payout checklist (all labelled *estimates*); daily P&L and payout logs; the full rule table with official source links, verification status and dates; confirm / override any rule. |
| **Backtesting Lab** | Chronological replay with conservative fills, commission and slippage, optional breakeven / trailing rules, in-sample / out-of-sample split, breakdowns by session / score bucket / regime / exit reason, threshold and structural sensitivity grids with overfitting warnings, saved runs with settings-mismatch warnings. |
| **Analytics** | Three separate views that are never mixed: backtests (hypothetical), signal outcomes (hypothetical), and your actual recorded trades. |
| **Trade Journal** | Signal journal (every detected setup, including rejected ones, with stored score components and hypothetical outcome) and the executed-trade journal (notes, tags, screenshot references, mistakes, rule adherence, emotions, broker-statement P&L); audit log. |
| **Alerts Center** | De-duplicated alerts with trigger time, issue time, severity, delivery status and your response (reviewed / dismissed / action recorded); channel and category configuration; Telegram test. |
| **Settings & Data Sources** | Every parameter, validated before saving (strategy changes are flagged against saved backtests); CSV import with a validation report; dataset management; adapter status; instrument registry with a calculated contract calendar. |

Instruments: **NQ, MNQ, ES, MES, GC, MGC** (configurable in `configurations/instruments.json`).

---

## Safety model

* **No order capability exists.** There is no broker adapter, no order model, no order endpoint. Market-data
  adapters are read-only by interface (`data/adapters/base.py`).
* **Distinct record types** - never conflated in storage or UI:
  1. detected setup, 2. qualified signal, 3. proposed trade plan, 4. trade you registered,
  5. management action you recorded, 6. verified trade data (only if a legitimate read-only source is added).
* **Suggestions are never inferred as actions.** A breakeven prompt does not change your stored stop; only an
  action you record does. The stop shown is always labelled *user-reported*.
* **Alert lifecycle is explicit:** price reached the trigger (trigger time) -> alert issued -> you reviewed
  it -> you recorded an action (linked to the alert).
* **Scores are rankings, not probabilities.** An 80/100 setup does not have an 80% chance of winning.
* **No LLM computes anything.** All prices, scores, risk, sizing, drawdown and consistency figures come from
  deterministic, tested code. The optional Claude feature only rephrases the engine's text, and its output
  is discarded if it contains any significant number the engine did not produce.
* **Firm compliance is never guaranteed**, payouts are never labelled approved, and every firm figure is an
  estimate from your own records.

---

## Architecture

```
                   +-------------------- app.py (Streamlit navigation) --------------------+
                   |  app_pages/*  (presentation only)        ui/*  (charts, theme, state)   |
                   +----------------------------------+-------------------------------------+
                                                      |
                                   services/*  (glue: load data, scan, persist, alert)
                     +------------------+-------------+--------------+------------------+
                     |                  |                            |                  |
               core/* (engine)    firms/* (rules)          notifications/*        data/*
         swings, zones, breakout,  Tradeify / Topstep      alert manager,       SQLite + migrations,
         retest state machine,     configs, drawdown       Telegram, browser    repositories, CSV import,
         scoring, plans, risk,     models, consistency,                         synthetic data, adapters
         trade management,         payout estimates
         backtesting, metrics
```

* `core/` is pure and deterministic. It knows nothing about firms, storage or Streamlit.
* The strategy is **firm-independent**: firm rules change risk limits and eligibility estimates, never
  signal generation.
* `services/` is used by the dashboard, the headless monitor (`scripts/run_monitor.py`) and the tests.
* SQLite schema is versioned (`schema_version` table, migrations in `data/database.py`).

**No look-ahead by construction.** The scanner replays bars in order: a swing is used only after its
confirmation bar closes; higher-timeframe bars only after they close; zone statistics only include
reactions seen so far; volume reliability is judged on trailing data only. Tests truncate and perturb future
data and assert that every finished historical setup is unchanged (`tests/test_scanner_lookahead.py`).

---

## Installation in detail

```bash
# 1. virtual environment
python3 -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. dependencies (pick one)
pip install -r requirements-lock.txt    # exact versions used for testing
pip install -e ".[dev]"                 # ranges from pyproject.toml + dev tools
pip install -e ".[dev,explain]"         # + optional Claude rephrasing
pip install -e ".[databento]"           # + optional (untested) Databento adapter

# 3. database (optional; also happens automatically on first launch)
python scripts/init_db.py               # creates var/copilot.db, syncs instruments and firm rules

# 4. run
python -m streamlit run app.py
```

Run commands from the project root. The database lives in `var/` (git-ignored); set `BRC_DB_PATH` to move it.

---

## Demo mode

The default data source is **Synthetic demo (simulated)**: deterministic simulated candles for every
instrument (seed 123, 60 trade days ending 2026-10-16 - change these under *Settings -> Data sources*).
**These are not real prices.** Every view labels them SIMULATED.

* **Replay clock (sidebar):** step the clock back and forward bar by bar (or auto-advance on the Live Signals
  page). Signals, zones and alerts are recomputed from only the bars up to the clock - exactly what was
  knowable then.
* **Simulate feed stall:** enter minutes in the sidebar to see the stale-data safeguards (banner, disabled
  qualification, data alert). A stall that lands outside Globex hours correctly reports *market closed*.
* **Contract roll:** the demo series crosses a quarterly roll. The two contracts are analysed separately
  (with a visible price gap) and never spliced.
* `python scripts/seed_demo_data.py` adds two **DEMO** accounts with made-up daily P&L and one example trade
  based on a simulated qualified setup.
* `python scripts/generate_demo_data.py --symbol MNQ --days 25` exports simulated candles to CSV to try the
  importer (an example is in `sample_data/MNQ_SIMULATED_5min.csv`).

---

## Using your own data (CSV)

*Settings & Data Sources -> Import CSV.* One row per 5-minute bar:

| Column | Required | Notes |
|---|---|---|
| `timestamp` (or `datetime`, or `date` + `time`) | yes | Bar **open** time (tick the box if your platform stamps bar close). |
| `open`, `high`, `low`, `close` | yes | Must be valid OHLC. |
| `volume` | no | Without trustworthy volume, VWAP and relative-volume factors are reported *unavailable* (never fabricated). |
| `contract` (or `symbol`) | recommended | e.g. `MNQZ26`. Must belong to the selected instrument (an MNQ file cannot be imported as NQ). |

Example:

```csv
timestamp,open,high,low,close,volume,contract
2026-09-14 09:30:00,20019.00,20023.25,20014.50,20018.50,1489,MNQZ26
```

Timestamps with an explicit UTC offset are converted; naive timestamps are interpreted in the timezone you
choose. DST-ambiguous or non-existent local times are rejected unless you choose a resolution. Unsorted rows
and duplicates are reported; sorting or dropping exact duplicates only happens when you tick the option.
Nothing is repaired silently. After saving, choose **Imported CSV** as the sidebar data source.

---

## Live / delayed data integration

No live feed is connected out of the box. One documented adapter is included but **untested** (no
credentials were available while building):

* **Databento** (`data/adapters/databento_adapter.py`): authorised CME Globex data (`GLBX.MDP3`), 1-minute
  OHLCV for one specific contract, aggregated to 5 minutes, labelled *delayed*.
  `pip install databento`, set `DATABENTO_API_KEY`, choose *Databento* in the sidebar. Verify its output
  against your platform before relying on it.

**Adding a provider:** subclass `MarketDataAdapter` (`status()` + `get_candles()`), return UTC bar-open
candles with a `contract` column, read credentials from environment variables, and register it in
`services/market_service.py`. Only use APIs whose terms permit this use, and never add order functionality.

**Headless monitoring** (alerts while the dashboard is closed):

```bash
python scripts/run_monitor.py --instrument MNQ --source databento --interval 60
python scripts/run_monitor.py --instrument MNQ --source synthetic --once      # test run
```

---

## Environment variables

Copy `.env.example` to `.env` (git-ignored). All are optional.

| Variable | Purpose |
|---|---|
| `BRC_DB_PATH` | SQLite path (default `var/copilot.db`). |
| `BRC_EPHEMERAL_STORAGE` | `1` shows / `0` hides the "temporary storage" warning; empty auto-detects Streamlit Community Cloud. |
| `DATABENTO_API_KEY` | Databento adapter (untested). |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram alerts (enable in *Alerts Center -> Channels*). |
| `ANTHROPIC_API_KEY`, `BRC_EXPLAINER_MODEL` | Optional Claude rephrasing of setup explanations (default model `claude-opus-5-5`, server-side refusal fallbacks enabled). |

Secrets are read only from the environment. They are never stored in SQLite, logged, or shown in the UI
(error messages are scrubbed).

---

## Strategy rules

All parameters below are defaults, editable and validated in *Settings*. They are engineering starting
points, not optimised values.

**Swings** - pivot high: strictly above the previous 3 highs and >= the next 3 highs (ties resolve to the
first bar). Known only when the 3rd right bar closes (`confirm_ts`). Minimum prominence max(8 ticks,
0.5 x ATR14 at confirmation). Same-type swings within 6 bars / 4 ticks are suppressed.

**Zones** - confirmed swings cluster within max(6 ticks, 0.25 ATR) while the zone stays narrower than 1 ATR.
Swings closer than 6 bars count as one reaction. A *test* is a run of bars touching the zone; a *hold* ends on
the side it came from; a *break* is a close beyond the far edge by max(2 ticks, 0.05 ATR); a *retest* is a
hold from the new side after a break; breaking back invalidates the zone; zones without a reaction for 1,500
bars expire. Strength = 0.35 reactions + 0.20 recency + 0.25 rejection + 0.20 hold ratio (each 0-1, formulas
shown in the UI).

**Breakout** (relevant zone: >= 2 reactions, recent, within 12 ATR) - completed close beyond the zone by
max(2 ticks, 0.05 ATR); body in the breakout direction >= 50% of range; range 0.5-3 ATR; extension <= 1.5 ATR;
>= 1.5 ATR of room to the next established opposing zone. HTF alignment, VWAP side and relative volume feed
the score and can be made mandatory.

**Retest & confirmation** - within 12 bars price returns within max(4 ticks, 0.15 ATR) of the zone; within 3
bars a candle closes back beyond the zone with a directional body, close in the favourable half of its range,
and a rejection wick >= 30% on a retest bar. **Invalidated** by a close back through the zone, running > 3 ATR
away first, or missing candles; **expired** if no retest / confirmation in time, after 24 bars, or at the
session end.

**Score (0-100)** - HTF alignment 20, zone quality 15, breakout candle 15, retest 20, VWAP 10, relative
volume 5, reward/risk 15. Missing factors are re-normalised over available weight (rejected below 75 weight or
if zone, breakout, retest or reward/risk is missing) - or scored as zero under the alternative policy. Minimum
80; presets 70-90. Scores and components are stored as computed and never rewritten.

**Plan** - entry at the confirmation close (or a limit at the zone edge); structural invalidation = beyond
the zone / retest extreme; stop = invalidation + max(2 ticks, 0.1 ATR), rounded away from entry; T1 just in
front of the next established opposing zone (a fixed-R projection is used and labelled only when no
structure exists); R:R >= 2 required; stops wider than 3 ATR rejected. Signal windows default to
09:30-15:45 ET, none within 15 minutes of the close.

**Trade management prompts** - breakeven review at +1R (configurable; optional high-volatility thresholds);
profit reviews at +2R / +3R, target approach (80%) / reached, approaching an opposing zone, momentum loss
(close through EMA9 after +1R); trailing suggestions = newly confirmed favourable swing -/+ max(2 ticks, 0.25
ATR14), only if tighter than your stop and on the correct side of price; warnings for retest-level failure,
structural invalidation, key-swing break (once per favourable leg), VWAP loss (optional), no progress after 36
bars, insufficient remaining reward, and price trading through your reported stop (further prompts pause until
you record the exit or your actual stop).

**Backtest fills** - entry at the confirmation close plus slippage; managed from the next bar; stop and target
in the same bar = loss; gaps through the stop fill at the open; targets fill at the target; optional rule
changes take effect from the next bar; one position at a time; flat at 16:00 ET by default.

---

## Prop-firm rules and the verification process

**Status of shipped rules: none is VERIFIED.** The official help centers (`help.topstep.com`,
`topstep.com`, `help.tradeify.co`) could not be fetched from the build environment (network policy). Values in
`firms/configurations/*.json` were taken from **web-search summaries of the official pages on 2026-10-16** and
are marked:

* `UNVERIFIED` - a starting point; confirm before use.
* `CONFLICTING` - the summaries disagreed (e.g. Tradeify 50K Select target $3,000 vs legacy $2,500; Select
  funded payout caps; Topstep Combine consistency wording and daily-loss-limit policy; Tradeify close-out
  4:45 vs 4:59 PM ET; Tradeify 25K Select contract limit).
* `null` - not found; enter it yourself (e.g. Topstep XFA maximum loss and starting balance, Topstep scaling
  plan tiers, Tradeify Growth contract limits, Tradeify Daily payout buffer).

Every rule records its official source URL. Shipped account types: Topstep Trading Combine 50K/100K/150K and
Express Funded 50K/100K/150K; Tradeify Select Evaluation 25K/50K/100K/150K, Growth Evaluation 50K, Select
Funded 50K (Flex and Daily payout paths).

**How to verify (recommended before each new account, and monthly):**

1. *Firm & Account Manager -> Firm rules*: open each source link (official help center only).
2. Check the value for **your exact account type and purchase date**.
3. Record it under *Confirm or override a rule* with status `USER_CONFIRMED`, the URL and today's date. Your
   confirmation overrides the shipped value, is kept in the audit log, and survives config updates.
4. In *Accounts -> Edit account*, confirm the drawdown model (type, amount, lock behaviour). Until then all
   drawdown figures are labelled estimates.
5. Confirmed rules older than the staleness window (default 30 days) raise an "outdated rules" alert.

**Drawdown models** (`firms/drawdown.py`, separately tested): static; end-of-day trailing (floor moves only at
the close, optional lock at start + offset and/or on payout); intraday trailing (needs intraday equity highs,
otherwise flagged as possibly overstating the allowance). Breaches are only *known* if you enter the day's
intraday low equity; otherwise the app says an intraday breach cannot be ruled out.

**Adding a firm:** copy a JSON file in `firms/configurations/`, edit account types and rules, restart. A Python
subclass (see `firms/topstep.py`) is only needed for custom interpretation notes.

---

## Tests and quality checks

```bash
python -m pytest                 # 201 tests (~1 min)
ruff check . && ruff format --check .
mypy                             # core, firms, data, notifications, services
```

Covered: swing confirmation timing and truncation invariance, zone clustering / holds / breaks / retests /
expiry, breakout checks, retest state transitions, scoring policies, plan levels and tick rounding (incl.
float-noise R:R), sizing limits, R-multiples, breakeven with costs, trailing suggestions, trade warnings and
pause-after-stop, data validation (duplicates, gaps vs halts/weekends, invalid OHLC, contract mixing, missing
volume), freshness (fresh / stale / market closed / historical / simulated stall), CSV import (timezones, DST
ambiguity, bar-close stamps, instrument mismatch), drawdown models, consistency formulas, payout cycles and
caps, firm config integrity (official sources, nothing shipped as VERIFIED), alert de-duplication and secret
scrubbing, setup immutability, audit-trailed corrections, backtest fills (same-bar ambiguity, gaps, session
flat, next-bar rule changes, limit fills), IS/OOS split, and end-to-end no-look-ahead (truncated and
perturbed futures).

**UI smoke test** (with the app running; needs `pip install playwright` and a Chromium):

```bash
python scripts/ui_smoke.py --url http://localhost:8501 --chromium /path/to/chrome --height 2400
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and run from the project root. |
| Port 8501 in use | `python -m streamlit run app.py --server.port 8502` |
| "No qualified setups" | Expected - the engine is selective. Check *Rejected* for reasons, lower the minimum score in *Settings -> Scoring*, or move the replay clock. |
| Stale-data banner | The feed's last bar is older than 2 x timeframe + 60 s while the market is open. For simulated data check *Simulate feed stall* is 0. |
| CSV import errors | Read the validation table: ordering, duplicates, DST, contract/instrument mismatch. Nothing is auto-fixed. |
| Settings reverted with a warning | Stored settings failed validation (e.g. after an upgrade); defaults were loaded. Re-apply your changes. |
| Reset everything | Stop the app and delete `var/copilot.db*`. |
| Telegram "not_configured" | Set both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`, restart, enable in Alerts Center. |

---

## Known limitations

* **No live data verified.** Synthetic data is simulated; the Databento adapter and Telegram channel are
  implemented but untested end-to-end; browser notifications are experimental (browsers may block them from
  embedded frames).
* **Firm rules are unverified** (see above). The app never guarantees compliance or payout eligibility.
* **Exchange holidays and early closes are not modelled** (session calendar, gap detection, contract calendar).
  Instrument specifications were entered from CME published specs at build time - confirm them (Settings ->
  Instruments).
* Alerts fire only while the dashboard or `scripts/run_monitor.py` is running.
* Breakeven / trailing logic in the Trade Manager uses completed 5-minute bars after the entry bar; intrabar
  events are not visible.
* Backtests on synthetic data say nothing about real profitability; small samples and a single period are
  flagged but not overcome. Sensitivity grids increase data-snooping risk - judge on out-of-sample results.
* Realized P&L is computed from the exits you record (or the broker figure you enter); the app cannot see fills.
* Single local user; SQLite is not designed for concurrent multi-user access.
* On Streamlit Community Cloud the database does not survive restarts (see Quick start); there is no
  built-in backup/restore yet.

---

## Security considerations

* No brokerage credentials are requested, stored or used. There is no order path to protect.
* API keys live only in environment variables / `.env` (git-ignored, excluded by `.gitignore` along with the
  database). The UI shows only whether a key is set.
* Telegram errors are scrubbed of the bot token; chat ids are masked.
* The optional Claude feature sends only the engine's textual setup explanation; no account data, balances or
  credentials are sent.
* The SQLite database contains your trading records - treat `var/` as private and back it up yourself.

---

## Project structure

```
break-retest-copilot/
  app.py                      Streamlit entry point (st.navigation)
  app_pages/                  10 pages (named app_pages/, not pages/, so Streamlit's legacy
                              auto-discovery cannot bypass the navigation on direct page URLs)
  core/                       deterministic engine: timeutils, instruments, config, indicators,
                              data_quality, resampling, swing_detection, support_resistance,
                              breakout_detection, retest_engine, scoring, trade_planning, scanner,
                              risk, trade_management, backtesting, performance, explain, events
  firms/                      base (rules interface), drawdown, evaluation, tradeify, topstep,
                              registry, configurations/{tradeify,topstep}.json
  data/                       database (migrations), repositories, models, csv_import,
                              synthetic_data, adapters/{base,synthetic,csv_dataset,databento_adapter}
  notifications/              manager (dedup + delivery), telegram
  services/                   bootstrap, market, scan, trade, account, backtest, explainer
  ui/                         theme, components, charts, state, safe_markdown, browser_notify
  configurations/             instruments.json
  scripts/                    init_db, seed_demo_data, generate_demo_data, run_monitor, ui_smoke
  tests/                      201 deterministic tests
  sample_data/                MNQ_SIMULATED_5min.csv (simulated, for the importer)
  .env.example  .gitignore  pyproject.toml  requirements-lock.txt  .streamlit/config.toml
```
