"""SQLite storage with a versioned migration mechanism.

* ``schema_version`` records every applied migration. ``Database.migrate()``
  applies any missing migrations in order inside a transaction.
* Timestamps are stored as ISO-8601 UTC strings (``core.timeutils.to_iso``).
* Brokerage credentials are never stored here; secrets live only in
  environment variables.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.timeutils import to_iso, utcnow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "var" / "copilot.db"

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE profiles (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL
    );
    CREATE TABLE user_settings (
        profile_id INTEGER NOT NULL REFERENCES profiles(id),
        key TEXT NOT NULL,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (profile_id, key)
    );
    CREATE TABLE instruments (
        symbol TEXT PRIMARY KEY,
        spec_json TEXT NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        metadata_status TEXT NOT NULL,
        verified_on TEXT,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE datasets (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        instrument TEXT NOT NULL,
        contract TEXT,
        timeframe TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        feed_type TEXT NOT NULL,
        source_tz TEXT,
        rows INTEGER NOT NULL,
        start_ts TEXT,
        end_ts TEXT,
        quality_json TEXT,
        notes TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE candles (
        dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL,
        contract TEXT,
        PRIMARY KEY (dataset_id, ts)
    );
    CREATE TABLE market_data_status (
        source_key TEXT PRIMARY KEY,
        instrument TEXT,
        contract TEXT,
        feed_type TEXT,
        data_source TEXT,
        last_bar_close TEXT,
        last_checked TEXT,
        status TEXT,
        message TEXT
    );
    CREATE TABLE swing_points (
        id TEXT PRIMARY KEY,
        context TEXT NOT NULL,
        instrument TEXT NOT NULL,
        contract TEXT,
        timeframe TEXT NOT NULL,
        swing_type TEXT NOT NULL,
        price REAL NOT NULL,
        pivot_ts TEXT NOT NULL,
        confirm_ts TEXT NOT NULL,
        prominence REAL,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_swings_ctx ON swing_points (context, instrument, confirm_ts);
    CREATE TABLE sr_zones (
        id INTEGER PRIMARY KEY,
        context TEXT NOT NULL,
        instrument TEXT NOT NULL,
        contract TEXT,
        timeframe TEXT NOT NULL,
        zone_ref INTEGER NOT NULL,
        lower REAL NOT NULL,
        upper REAL NOT NULL,
        kind TEXT,
        status TEXT,
        strength REAL,
        reactions INTEGER,
        holds INTEGER,
        failures INTEGER,
        created_ts TEXT,
        broken_ts TEXT,
        snapshot_ts TEXT NOT NULL,
        details_json TEXT
    );
    CREATE INDEX idx_zones_ctx ON sr_zones (context, instrument);
    CREATE TABLE setups (
        id TEXT PRIMARY KEY,
        context TEXT NOT NULL,
        instrument TEXT NOT NULL,
        contract TEXT,
        timeframe TEXT NOT NULL,
        direction TEXT NOT NULL,
        state TEXT NOT NULL,
        zone_lower REAL,
        zone_upper REAL,
        zone_json TEXT,
        breakout_ts TEXT NOT NULL,
        breakout_price REAL,
        retest_ts TEXT,
        retest_price REAL,
        confirm_ts TEXT,
        confirm_price REAL,
        score REAL,
        min_score REAL,
        qualified INTEGER NOT NULL DEFAULT 0,
        entry REAL,
        stop REAL,
        target1 REAL,
        target2 REAL,
        rr REAL,
        risk_ticks REAL,
        plan_json TEXT,
        breakout_eval_json TEXT,
        score_json TEXT,
        rejection_reasons_json TEXT,
        history_json TEXT,
        context_json TEXT,
        data_source TEXT,
        feed_type TEXT,
        settings_hash TEXT,
        last_event_ts TEXT,
        first_seen_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX idx_setups_inst_ts ON setups (instrument, breakout_ts);
    CREATE INDEX idx_setups_state ON setups (state);
    CREATE INDEX idx_setups_context ON setups (context);
    CREATE TABLE setup_score_components (
        setup_id TEXT NOT NULL REFERENCES setups(id) ON DELETE CASCADE,
        factor TEXT NOT NULL,
        weight REAL NOT NULL,
        raw REAL,
        points REAL NOT NULL,
        available INTEGER NOT NULL,
        explanation TEXT,
        PRIMARY KEY (setup_id, factor)
    );
    CREATE TABLE alerts (
        id INTEGER PRIMARY KEY,
        dedup_key TEXT NOT NULL UNIQUE,
        category TEXT NOT NULL,
        severity TEXT NOT NULL,
        trigger_ts TEXT NOT NULL,
        created_at TEXT NOT NULL,
        context TEXT,
        instrument TEXT,
        setup_id TEXT,
        trade_id INTEGER,
        account_id INTEGER,
        price REAL,
        title TEXT NOT NULL,
        message TEXT NOT NULL,
        details_json TEXT,
        acknowledged INTEGER NOT NULL DEFAULT 0,
        acknowledged_at TEXT,
        user_response TEXT,
        response_note TEXT,
        linked_action_id INTEGER
    );
    CREATE INDEX idx_alerts_created ON alerts (created_at);
    CREATE INDEX idx_alerts_ack ON alerts (acknowledged, category);
    CREATE TABLE alert_deliveries (
        id INTEGER PRIMARY KEY,
        alert_id INTEGER NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
        channel TEXT NOT NULL,
        status TEXT NOT NULL,
        detail TEXT,
        attempted_at TEXT NOT NULL
    );
    CREATE INDEX idx_deliveries_alert ON alert_deliveries (alert_id);
    CREATE TABLE firms (
        code TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        website TEXT,
        config_version TEXT,
        config_path TEXT,
        loaded_at TEXT NOT NULL
    );
    CREATE TABLE rule_configs (
        id INTEGER PRIMARY KEY,
        firm_code TEXT NOT NULL REFERENCES firms(code),
        account_type_code TEXT NOT NULL,
        rule_key TEXT NOT NULL,
        value_json TEXT,
        status TEXT NOT NULL,
        source_url TEXT,
        verified_on TEXT,
        effective_date TEXT,
        notes TEXT,
        origin TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (firm_code, account_type_code, rule_key, origin)
    );
    CREATE TABLE accounts (
        id INTEGER PRIMARY KEY,
        firm_code TEXT NOT NULL REFERENCES firms(code),
        account_type_code TEXT NOT NULL,
        nickname TEXT NOT NULL UNIQUE,
        stage TEXT NOT NULL,
        account_size REAL NOT NULL,
        starting_balance REAL NOT NULL,
        current_balance REAL,
        balance_source TEXT NOT NULL DEFAULT 'user_entered',
        current_equity REAL,
        equity_source TEXT,
        personal_daily_loss_limit REAL,
        personal_max_contracts REAL,
        drawdown_model_confirmed INTEGER NOT NULL DEFAULT 0,
        drawdown_confirmed_on TEXT,
        rules_verified_on TEXT,
        cycle_start_date TEXT,
        status TEXT NOT NULL DEFAULT 'active',
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE account_daily_pnl (
        id INTEGER PRIMARY KEY,
        account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
        trade_date TEXT NOT NULL,
        net_pnl REAL NOT NULL,
        min_equity REAL,
        max_equity REAL,
        source TEXT NOT NULL,
        notes TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (account_id, trade_date)
    );
    CREATE TABLE account_payouts (
        id INTEGER PRIMARY KEY,
        account_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
        requested_on TEXT NOT NULL,
        amount REAL NOT NULL,
        status TEXT NOT NULL,
        notes TEXT,
        created_at TEXT NOT NULL
    );
    CREATE TABLE manual_trades (
        id INTEGER PRIMARY KEY,
        account_id INTEGER REFERENCES accounts(id),
        firm_code TEXT,
        instrument TEXT NOT NULL,
        contract TEXT,
        direction TEXT NOT NULL,
        entry_ts TEXT NOT NULL,
        entry_price REAL NOT NULL,
        initial_stop REAL NOT NULL,
        quantity INTEGER NOT NULL,
        planned_target REAL,
        setup_id TEXT,
        commission_per_contract_rt REAL NOT NULL DEFAULT 0,
        slippage_ticks REAL NOT NULL DEFAULT 0,
        notes TEXT,
        tags_json TEXT,
        screenshot_refs_json TEXT,
        mistakes TEXT,
        rule_adherence TEXT,
        emotions TEXT,
        status TEXT NOT NULL DEFAULT 'open',
        realized_pnl REAL,
        pnl_source TEXT,
        closed_ts TEXT,
        is_void INTEGER NOT NULL DEFAULT 0,
        void_reason TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX idx_trades_status ON manual_trades (status);
    CREATE TABLE trade_actions (
        id INTEGER PRIMARY KEY,
        trade_id INTEGER NOT NULL REFERENCES manual_trades(id),
        action_type TEXT NOT NULL,
        action_ts TEXT NOT NULL,
        price REAL,
        quantity INTEGER,
        new_stop REAL,
        notes TEXT,
        alert_id INTEGER,
        corrects_action_id INTEGER REFERENCES trade_actions(id),
        is_void INTEGER NOT NULL DEFAULT 0,
        void_reason TEXT,
        recorded_at TEXT NOT NULL
    );
    CREATE INDEX idx_actions_trade ON trade_actions (trade_id);
    CREATE TABLE signal_outcomes (
        setup_id TEXT PRIMARY KEY REFERENCES setups(id) ON DELETE CASCADE,
        outcome TEXT NOT NULL,
        entry_ts TEXT,
        exit_ts TEXT,
        exit_price REAL,
        r_result REAL,
        mfe_r REAL,
        mae_r REAL,
        bars_held INTEGER,
        ambiguous INTEGER NOT NULL DEFAULT 0,
        evaluated_through_ts TEXT,
        evaluated_at TEXT NOT NULL,
        notes TEXT
    );
    CREATE TABLE backtest_runs (
        id INTEGER PRIMARY KEY,
        name TEXT,
        created_at TEXT NOT NULL,
        instrument TEXT NOT NULL,
        data_source TEXT NOT NULL,
        dataset_ref TEXT,
        contract TEXT,
        start_ts TEXT,
        end_ts TEXT,
        settings_json TEXT NOT NULL,
        settings_hash TEXT NOT NULL,
        params_json TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        breakdowns_json TEXT,
        notes TEXT
    );
    CREATE TABLE backtest_trades (
        id INTEGER PRIMARY KEY,
        run_id INTEGER NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
        setup_id TEXT,
        direction TEXT,
        entry_ts TEXT,
        entry_price REAL,
        exit_ts TEXT,
        exit_price REAL,
        stop REAL,
        target REAL,
        quantity INTEGER,
        pnl_points REAL,
        pnl_gross REAL,
        costs REAL,
        pnl_net REAL,
        r_multiple REAL,
        exit_reason TEXT,
        ambiguous INTEGER,
        session TEXT,
        score REAL,
        vol_regime TEXT,
        trend_regime TEXT,
        sample TEXT,
        bars_held INTEGER,
        details_json TEXT
    );
    CREATE INDEX idx_bt_trades_run ON backtest_trades (run_id);
    CREATE TABLE audit_events (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,
        entity TEXT NOT NULL,
        entity_id TEXT,
        action TEXT NOT NULL,
        details_json TEXT
    );
    CREATE INDEX idx_audit_entity ON audit_events (entity, entity_id);
    """,
}

LATEST_VERSION = max(MIGRATIONS)


def resolve_db_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("BRC_DB_PATH")
    if env:
        p = Path(env)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return DEFAULT_DB_PATH


def storage_is_ephemeral() -> bool:
    """Whether the database may be wiped when the app restarts (hosted deployments).

    ``BRC_EPHEMERAL_STORAGE`` (1/0) overrides the check. Otherwise Streamlit
    Community Cloud is detected by its ``/mount/src`` checkout location; its
    container file system is reset on every reboot, sleep or redeploy.
    """
    flag = os.environ.get("BRC_EPHEMERAL_STORAGE", "").strip().lower()
    if flag:
        return flag in ("1", "true", "yes")
    return PROJECT_ROOT.as_posix().startswith("/mount/src/")


class Database:
    """Thin SQLite wrapper. A new connection is opened per unit of work."""

    _lock = threading.Lock()

    def __init__(self, path: str | Path | None = None):
        self.path = resolve_db_path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        if str(self.path) == ":memory:":
            self._memory_conn = self._open()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if str(self.path) != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a transaction (commit on success, rollback on error)."""
        if self._memory_conn is not None:
            conn = self._memory_conn
            with self._lock:
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            return
        conn = self._open()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -- migrations ----------------------------------------------------------
    def current_version(self) -> int:
        with self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL, applied_at TEXT NOT NULL)"
            )
            row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
            return int(row["v"] or 0)

    def migrate(self) -> list[int]:
        applied: list[int] = []
        current = self.current_version()
        for version in sorted(MIGRATIONS):
            if version <= current:
                continue
            with self.connect() as conn:  # one transaction per migration
                for stmt in _split_sql(MIGRATIONS[version]):
                    conn.execute(stmt)
                conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, to_iso(utcnow())),
                )
            applied.append(version)
        with self.connect() as conn:
            if not conn.execute("SELECT 1 FROM profiles WHERE name = 'default'").fetchone():
                conn.execute(
                    "INSERT INTO profiles (name, created_at) VALUES ('default', ?)", (to_iso(utcnow()),)
                )
        return applied

    # -- convenience ---------------------------------------------------------
    def query(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def execute(self, sql: str, params: tuple | dict = ()) -> int:
        """Execute a write; returns lastrowid."""
        with self.connect() as conn:
            cur = conn.execute(sql, params)
            return int(cur.lastrowid or 0)


def _split_sql(script: str) -> list[str]:
    return [s.strip() for s in script.split(";") if s.strip()]


_DEFAULT_DB: Database | None = None


def get_database(path: str | Path | None = None) -> Database:
    """Process-wide default database (migrated on first use)."""
    global _DEFAULT_DB
    if path is not None:
        db = Database(path)
        db.migrate()
        return db
    if _DEFAULT_DB is None:
        _DEFAULT_DB = Database()
        _DEFAULT_DB.migrate()
    return _DEFAULT_DB
