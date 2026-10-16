"""Repository functions: every SQL statement used by the application.

Integrity rules enforced here:

* Setups are keyed by a stable ID. Once a setup reaches a terminal state in
  the database its score, plan and history are immutable - a later re-scan
  (even with different settings) can never rewrite what was known at the
  time. Non-terminal setups may only move forward in time.
* Alerts are de-duplicated by ``dedup_key`` (UNIQUE constraint).
* Trade-management actions are append-only. A correction voids the original
  row and inserts a new one referencing it, preserving the audit trail.
* Every user edit to trades, accounts and rules writes an audit event.
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from core.config import AppSettings
from core.timeutils import from_iso, to_iso, utcnow
from data.database import Database
from data.models import (
    AccountInput,
    DailyPnlInput,
    ManualTradeInput,
    PayoutInput,
    TradeActionInput,
)

TERMINAL = ("qualified", "rejected", "expired", "invalidated")


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def _default(o: Any) -> Any:
    if isinstance(o, pd.Timestamp):
        return to_iso(o) if o.tzinfo is not None else o.isoformat()
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if math.isnan(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def _clean(o: Any) -> Any:
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def dumps(obj: Any) -> str:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    return json.dumps(_clean(obj), default=_default)


def loads(raw: str | None, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    return json.loads(raw)


def now_iso() -> str:
    return to_iso(utcnow())  # type: ignore[return-value]


def audit(db: Database, entity: str, entity_id: Any, action: str, details: dict | None = None) -> None:
    db.execute(
        "INSERT INTO audit_events (ts, entity, entity_id, action, details_json) VALUES (?, ?, ?, ?, ?)",
        (now_iso(), entity, None if entity_id is None else str(entity_id), action, dumps(details or {})),
    )


def list_audit(
    db: Database, entity: str | None = None, entity_id: Any = None, limit: int = 200
) -> list[dict]:
    sql = "SELECT * FROM audit_events WHERE 1=1"
    params: list[Any] = []
    if entity:
        sql += " AND entity = ?"
        params.append(entity)
    if entity_id is not None:
        sql += " AND entity_id = ?"
        params.append(str(entity_id))
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    return db.query(sql, tuple(params))


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _profile_id(db: Database) -> int:
    row = db.query_one("SELECT id FROM profiles WHERE name = 'default'")
    if row is None:
        return db.execute("INSERT INTO profiles (name, created_at) VALUES ('default', ?)", (now_iso(),))
    return int(row["id"])


def load_settings(db: Database) -> tuple[AppSettings, list[str]]:
    """Return (settings, warnings). Invalid stored settings fall back to defaults with a warning."""
    row = db.query_one(
        "SELECT value_json FROM user_settings WHERE profile_id = ? AND key = 'app'", (_profile_id(db),)
    )
    if row is None:
        return AppSettings(), []
    try:
        return AppSettings.from_json(row["value_json"]), []
    except Exception as exc:  # noqa: BLE001
        return AppSettings(), [f"Stored settings were invalid and defaults were loaded: {exc}"]


def save_settings(db: Database, settings: AppSettings) -> None:
    AppSettings.model_validate(settings.model_dump())  # re-validate
    db.execute(
        "INSERT INTO user_settings (profile_id, key, value_json, updated_at) VALUES (?, 'app', ?, ?) "
        "ON CONFLICT(profile_id, key) DO UPDATE SET value_json = excluded.value_json, updated_at = excluded.updated_at",
        (_profile_id(db), settings.to_json(), now_iso()),
    )
    audit(db, "settings", "app", "saved", {"strategy_hash": settings.strategy_hash()})


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------


def sync_instruments(db: Database, specs: list) -> None:
    with db.connect() as conn:
        for spec in specs:
            existing = conn.execute(
                "SELECT verified_on FROM instruments WHERE symbol = ?", (spec.symbol,)
            ).fetchone()
            verified = (
                existing["verified_on"]
                if existing
                else (spec.metadata_verified_on.isoformat() if spec.metadata_verified_on else None)
            )
            conn.execute(
                "INSERT INTO instruments (symbol, spec_json, enabled, metadata_status, verified_on, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(symbol) DO UPDATE SET spec_json = excluded.spec_json, "
                "enabled = excluded.enabled, metadata_status = CASE WHEN instruments.verified_on IS NOT NULL "
                "THEN 'user_verified' ELSE excluded.metadata_status END, updated_at = excluded.updated_at",
                (
                    spec.symbol,
                    spec.model_dump_json(),
                    int(spec.enabled),
                    spec.metadata_status,
                    verified,
                    now_iso(),
                ),
            )


def mark_instrument_verified(db: Database, symbol: str, verified_on: date) -> None:
    db.execute(
        "UPDATE instruments SET verified_on = ?, metadata_status = 'user_verified', updated_at = ? "
        "WHERE symbol = ?",
        (verified_on.isoformat(), now_iso(), symbol),
    )
    audit(db, "instrument", symbol, "metadata_verified", {"verified_on": verified_on.isoformat()})


def list_instrument_rows(db: Database) -> list[dict]:
    return db.query(
        "SELECT symbol, metadata_status, verified_on, enabled, updated_at FROM instruments ORDER BY symbol"
    )


# ---------------------------------------------------------------------------
# Datasets / candles
# ---------------------------------------------------------------------------


def create_dataset(
    db: Database,
    name: str,
    instrument: str,
    timeframe: str,
    df: pd.DataFrame,
    source_kind: str,
    feed_type: str,
    source_tz: str | None,
    quality: dict,
    notes: str = "",
    contract: str | None = None,
) -> int:
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO datasets (name, instrument, contract, timeframe, source_kind, feed_type, source_tz, rows, "
            "start_ts, end_ts, quality_json, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                name,
                instrument,
                contract,
                timeframe,
                source_kind,
                feed_type,
                source_tz,
                len(df),
                to_iso(df.index.min()) if len(df) else None,
                to_iso(df.index.max()) if len(df) else None,
                dumps(quality),
                notes,
                now_iso(),
            ),
        )
        ds_id = int(cur.lastrowid or 0)
        vol = df["volume"] if "volume" in df else pd.Series(np.nan, index=df.index)
        con = df["contract"].astype(str) if "contract" in df else pd.Series(contract, index=df.index)
        rows = [
            (
                ds_id,
                to_iso(ts),
                float(o),
                float(h),
                float(lw),
                float(c),
                None if pd.isna(v) else float(v),
                None if k in (None, "None", "nan") else str(k),
            )
            for ts, o, h, lw, c, v, k in zip(
                df.index, df["open"], df["high"], df["low"], df["close"], vol, con, strict=True
            )
        ]
        conn.executemany(
            "INSERT INTO candles (dataset_id, ts, open, high, low, close, volume, contract) "
            "VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
    audit(db, "dataset", ds_id, "imported", {"name": name, "rows": len(df), "instrument": instrument})
    return ds_id


def list_datasets(db: Database) -> list[dict]:
    rows = db.query("SELECT * FROM datasets ORDER BY created_at DESC")
    for r in rows:
        r["quality"] = loads(r.pop("quality_json"), {})
    return rows


def get_dataset(db: Database, dataset_id: int) -> dict | None:
    r = db.query_one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if r:
        r["quality"] = loads(r.pop("quality_json"), {})
    return r


def load_dataset_candles(db: Database, dataset_id: int) -> pd.DataFrame:
    rows = db.query(
        "SELECT ts, open, high, low, close, volume, contract FROM candles WHERE dataset_id = ? ORDER BY ts",
        (dataset_id,),
    )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "contract"])
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"), utc=True), name="ts")
    if df["contract"].isna().all():
        df = df.drop(columns=["contract"])
    return df


def delete_dataset(db: Database, dataset_id: int) -> None:
    db.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    audit(db, "dataset", dataset_id, "deleted")


def upsert_market_status(
    db: Database,
    source_key: str,
    instrument: str,
    contract: str | None,
    feed_type: str,
    data_source: str,
    last_bar_close: pd.Timestamp | None,
    status: str,
    message: str,
) -> None:
    db.execute(
        "INSERT INTO market_data_status (source_key, instrument, contract, feed_type, data_source, last_bar_close, "
        "last_checked, status, message) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET "
        "instrument=excluded.instrument, contract=excluded.contract, feed_type=excluded.feed_type, "
        "data_source=excluded.data_source, last_bar_close=excluded.last_bar_close, last_checked=excluded.last_checked, "
        "status=excluded.status, message=excluded.message",
        (
            source_key,
            instrument,
            contract,
            feed_type,
            data_source,
            to_iso(last_bar_close),
            now_iso(),
            status,
            message,
        ),
    )


def list_market_status(db: Database) -> list[dict]:
    return db.query("SELECT * FROM market_data_status ORDER BY last_checked DESC")


# ---------------------------------------------------------------------------
# Swings and zones
# ---------------------------------------------------------------------------


def save_swings(
    db: Database, swings: pd.DataFrame, context: str, instrument: str, contract: str | None, timeframe: str
) -> int:
    if swings is None or swings.empty:
        return 0
    rows = [
        (
            f"{context}|{r['swing_id']}",
            context,
            instrument,
            contract,
            timeframe,
            r["swing_type"],
            float(r["price"]),
            to_iso(r["pivot_ts"]),
            to_iso(r["confirm_ts"]),
            float(r["prominence"]),
            r["status"],
            now_iso(),
        )
        for r in swings.to_dict("records")
    ]
    with db.connect() as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO swing_points (id, context, instrument, contract, timeframe, swing_type, price, "
            "pivot_ts, confirm_ts, prominence, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        return conn.total_changes - before


def replace_zones(
    db: Database,
    zones: list[dict],
    context: str,
    instrument: str,
    contract: str | None,
    timeframe: str,
    snapshot_ts: pd.Timestamp | None,
) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM sr_zones WHERE context = ? AND instrument = ?", (context, instrument))
        conn.executemany(
            "INSERT INTO sr_zones (context, instrument, contract, timeframe, zone_ref, lower, upper, kind, status, "
            "strength, reactions, holds, failures, created_ts, broken_ts, snapshot_ts, details_json) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    context,
                    instrument,
                    contract,
                    timeframe,
                    z["zone_id"],
                    z["lower"],
                    z["upper"],
                    z["kind"],
                    z["status"],
                    z["strength"],
                    z["reactions"],
                    z["holds"],
                    z["failures"],
                    to_iso(z["created_ts"]),
                    to_iso(z["broken_ts"]),
                    to_iso(snapshot_ts),
                    dumps(z),
                )
                for z in zones
            ],
        )


# ---------------------------------------------------------------------------
# Setups
# ---------------------------------------------------------------------------


def _setup_row(s: Any, context: str, data_source: str, feed_type: str, settings_hash: str) -> dict:
    plan = s.plan
    score = s.score
    return {
        "id": s.setup_id,
        "context": context,
        "instrument": s.instrument,
        "contract": s.contract,
        "timeframe": s.timeframe,
        "direction": s.direction,
        "state": s.state.value,
        "zone_lower": s.zone.get("lower"),
        "zone_upper": s.zone.get("upper"),
        "zone_json": dumps(s.zone),
        "breakout_ts": to_iso(s.breakout_ts),
        "breakout_price": s.breakout_bar.get("c"),
        "retest_ts": to_iso(s.retest_ts),
        "retest_price": s.retest_price,
        "confirm_ts": to_iso(s.confirm_ts),
        "confirm_price": s.confirm_price,
        "score": score.total if score else None,
        "min_score": score.min_score if score else None,
        "qualified": int(s.state.value == "qualified"),
        "entry": plan.entry_price if plan else None,
        "stop": plan.stop if plan else None,
        "target1": plan.target1 if plan else None,
        "target2": plan.target2 if plan else None,
        "rr": plan.rr if plan else None,
        "risk_ticks": plan.risk_ticks if plan else None,
        "plan_json": dumps(plan) if plan else None,
        "breakout_eval_json": dumps(s.breakout_eval) if s.breakout_eval else None,
        "score_json": dumps(score) if score else None,
        "rejection_reasons_json": dumps(s.rejection_reasons),
        "history_json": dumps(
            [{"state": t.state.value, "ts": t.ts, "bar_idx": t.bar_idx, "note": t.note} for t in s.history]
        ),
        "context_json": dumps(s.context),
        "data_source": data_source,
        "feed_type": feed_type,
        "settings_hash": settings_hash,
        "last_event_ts": to_iso(s.last_update_ts),
    }


def upsert_setups(
    db: Database, setups: list, context: str, data_source: str, feed_type: str, settings_hash: str
) -> dict[str, list[str]]:
    """Insert new setups and advance non-terminal ones. Terminal rows are immutable.

    Returns {"inserted": [...], "updated": [...], "unchanged": [...]}.
    """
    out: dict[str, list[str]] = {"inserted": [], "updated": [], "unchanged": []}
    if not setups:
        return out
    ts_now = now_iso()
    with db.connect() as conn:
        for s in setups:
            row = _setup_row(s, context, data_source, feed_type, settings_hash)
            existing = conn.execute(
                "SELECT state, last_event_ts FROM setups WHERE id = ?", (s.setup_id,)
            ).fetchone()
            if existing is None:
                row["first_seen_at"] = ts_now
                row["updated_at"] = ts_now
                cols = ", ".join(row)
                conn.execute(
                    f"INSERT INTO setups ({cols}) VALUES ({', '.join('?' * len(row))})", tuple(row.values())
                )
                _write_components(conn, s)
                out["inserted"].append(s.setup_id)
                continue
            if existing["state"] in TERMINAL:
                out["unchanged"].append(s.setup_id)
                continue
            prev_ts = from_iso(existing["last_event_ts"])
            if prev_ts is not None and s.last_update_ts < prev_ts:
                out["unchanged"].append(s.setup_id)  # replay rewound: never move backwards
                continue
            if existing["state"] == row["state"] and existing["last_event_ts"] == row["last_event_ts"]:
                out["unchanged"].append(s.setup_id)
                continue
            row["updated_at"] = ts_now
            sets = ", ".join(f"{k} = ?" for k in row if k != "id")
            conn.execute(
                f"UPDATE setups SET {sets} WHERE id = ?",
                tuple(v for k, v in row.items() if k != "id") + (s.setup_id,),
            )
            _write_components(conn, s)
            out["updated"].append(s.setup_id)
    return out


def _write_components(conn: Any, s: Any) -> None:
    if not s.score:
        return
    conn.execute("DELETE FROM setup_score_components WHERE setup_id = ?", (s.setup_id,))
    conn.executemany(
        "INSERT INTO setup_score_components (setup_id, factor, weight, raw, points, available, explanation) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            (s.setup_id, c.factor, c.weight, c.raw, c.points, int(c.available), c.explanation)
            for c in s.score.components
        ],
    )


def list_setups(
    db: Database,
    context: str | None = None,
    instrument: str | None = None,
    states: list[str] | None = None,
    min_score: float | None = None,
    limit: int = 2000,
) -> pd.DataFrame:
    sql = (
        "SELECT id, context, instrument, contract, direction, state, zone_lower, zone_upper, breakout_ts, "
        "retest_ts, confirm_ts, score, qualified, entry, stop, target1, target2, rr, risk_ticks, "
        "rejection_reasons_json, data_source, feed_type, settings_hash, first_seen_at FROM setups WHERE 1=1"
    )
    params: list[Any] = []
    if context:
        sql += " AND context = ?"
        params.append(context)
    if instrument:
        sql += " AND instrument = ?"
        params.append(instrument)
    if states:
        sql += f" AND state IN ({', '.join('?' * len(states))})"
        params += states
    if min_score is not None:
        sql += " AND score >= ?"
        params.append(min_score)
    sql += " ORDER BY breakout_ts DESC LIMIT ?"
    params.append(limit)
    rows = db.query(sql, tuple(params))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("breakout_ts", "retest_ts", "confirm_ts", "first_seen_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df["rejection_reasons"] = df.pop("rejection_reasons_json").map(lambda x: loads(x, []))
    return df


def get_setup(db: Database, setup_id: str) -> dict | None:
    r = db.query_one("SELECT * FROM setups WHERE id = ?", (setup_id,))
    if r is None:
        return None
    names = {
        "zone_json": "zone",
        "plan_json": "plan",
        "breakout_eval_json": "breakout_eval",
        "score_json": "score_detail",
        "rejection_reasons_json": "rejection_reasons",
        "history_json": "history",
        "context_json": "context",
    }
    for k, new in names.items():
        r[new] = loads(r.pop(k), None)
    r["components"] = db.query(
        "SELECT * FROM setup_score_components WHERE setup_id = ? ORDER BY weight DESC", (setup_id,)
    )
    r["outcome"] = db.query_one("SELECT * FROM signal_outcomes WHERE setup_id = ?", (setup_id,))
    return r


def list_setup_contexts(db: Database) -> list[str]:
    return [r["context"] for r in db.query("SELECT DISTINCT context FROM setups ORDER BY context")]


def upsert_signal_outcome(db: Database, setup_id: str, outcome: dict) -> None:
    db.execute(
        "INSERT INTO signal_outcomes (setup_id, outcome, entry_ts, exit_ts, exit_price, r_result, mfe_r, mae_r, "
        "bars_held, ambiguous, evaluated_through_ts, evaluated_at, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(setup_id) DO UPDATE SET outcome=excluded.outcome, entry_ts=excluded.entry_ts, "
        "exit_ts=excluded.exit_ts, exit_price=excluded.exit_price, r_result=excluded.r_result, mfe_r=excluded.mfe_r, "
        "mae_r=excluded.mae_r, bars_held=excluded.bars_held, ambiguous=excluded.ambiguous, "
        "evaluated_through_ts=excluded.evaluated_through_ts, evaluated_at=excluded.evaluated_at, notes=excluded.notes",
        (
            setup_id,
            outcome["outcome"],
            to_iso(outcome.get("entry_ts")),
            to_iso(outcome.get("exit_ts")),
            outcome.get("exit_price"),
            outcome.get("r_result"),
            outcome.get("mfe_r"),
            outcome.get("mae_r"),
            outcome.get("bars_held"),
            int(bool(outcome.get("ambiguous"))),
            to_iso(outcome.get("evaluated_through_ts")),
            now_iso(),
            outcome.get("notes"),
        ),
    )


def list_signal_outcomes(db: Database) -> pd.DataFrame:
    rows = db.query(
        "SELECT o.*, s.instrument, s.direction, s.score, s.context, s.confirm_ts FROM signal_outcomes o "
        "JOIN setups s ON s.id = o.setup_id ORDER BY s.confirm_ts DESC"
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


def insert_alert(
    db: Database,
    *,
    dedup_key: str,
    category: str,
    severity: str,
    trigger_ts: pd.Timestamp,
    title: str,
    message: str,
    context: str | None = None,
    instrument: str | None = None,
    setup_id: str | None = None,
    trade_id: int | None = None,
    account_id: int | None = None,
    price: float | None = None,
    details: dict | None = None,
) -> int | None:
    """Insert an alert unless its dedup key already exists. Returns the new id or None (duplicate)."""
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO alerts (dedup_key, category, severity, trigger_ts, created_at, context, instrument, "
            "setup_id, trade_id, account_id, price, title, message, details_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                dedup_key,
                category,
                severity,
                to_iso(trigger_ts),
                now_iso(),
                context,
                instrument,
                setup_id,
                trade_id,
                account_id,
                price,
                title,
                message,
                dumps(details or {}),
            ),
        )
        if cur.rowcount == 0:
            return None
        return int(cur.lastrowid or 0)


def add_delivery(db: Database, alert_id: int, channel: str, status: str, detail: str = "") -> None:
    db.execute(
        "INSERT INTO alert_deliveries (alert_id, channel, status, detail, attempted_at) VALUES (?,?,?,?,?)",
        (alert_id, channel, status, detail[:500], now_iso()),
    )


def list_alerts(
    db: Database,
    acknowledged: bool | None = None,
    categories: list[str] | None = None,
    severities: list[str] | None = None,
    trade_id: int | None = None,
    limit: int = 500,
) -> list[dict]:
    sql = "SELECT * FROM alerts WHERE 1=1"
    params: list[Any] = []
    if acknowledged is not None:
        sql += " AND acknowledged = ?"
        params.append(int(acknowledged))
    if categories:
        sql += f" AND category IN ({', '.join('?' * len(categories))})"
        params += categories
    if severities:
        sql += f" AND severity IN ({', '.join('?' * len(severities))})"
        params += severities
    if trade_id is not None:
        sql += " AND trade_id = ?"
        params.append(trade_id)
    sql += " ORDER BY trigger_ts DESC, id DESC LIMIT ?"
    params.append(limit)
    rows = db.query(sql, tuple(params))
    if rows:
        ids = [r["id"] for r in rows]
        deliveries = db.query(
            f"SELECT * FROM alert_deliveries WHERE alert_id IN ({', '.join('?' * len(ids))}) ORDER BY id",
            tuple(ids),
        )
        by_alert: dict[int, list[dict]] = {}
        for d in deliveries:
            by_alert.setdefault(d["alert_id"], []).append(d)
        for r in rows:
            r["details"] = loads(r.pop("details_json"), {})
            r["deliveries"] = by_alert.get(r["id"], [])
    return rows


def acknowledge_alert(
    db: Database,
    alert_id: int,
    response: str = "reviewed",
    note: str = "",
    linked_action_id: int | None = None,
) -> None:
    db.execute(
        "UPDATE alerts SET acknowledged = 1, acknowledged_at = COALESCE(acknowledged_at, ?), user_response = ?, "
        "response_note = ?, linked_action_id = COALESCE(?, linked_action_id) WHERE id = ?",
        (now_iso(), response, note, linked_action_id, alert_id),
    )
    audit(
        db,
        "alert",
        alert_id,
        f"acknowledged:{response}",
        {"note": note, "linked_action_id": linked_action_id},
    )


def count_unacknowledged(db: Database) -> dict[str, int]:
    rows = db.query("SELECT severity, COUNT(*) AS n FROM alerts WHERE acknowledged = 0 GROUP BY severity")
    return {r["severity"]: int(r["n"]) for r in rows}


# ---------------------------------------------------------------------------
# Firms and rules
# ---------------------------------------------------------------------------


def upsert_firm(
    db: Database, code: str, name: str, website: str | None, config_version: str, config_path: str
) -> None:
    db.execute(
        "INSERT INTO firms (code, name, website, config_version, config_path, loaded_at) VALUES (?,?,?,?,?,?) "
        "ON CONFLICT(code) DO UPDATE SET name=excluded.name, website=excluded.website, "
        "config_version=excluded.config_version, config_path=excluded.config_path, loaded_at=excluded.loaded_at",
        (code, name, website, config_version, config_path, now_iso()),
    )


def sync_config_rules(db: Database, firm_code: str, rows: list[dict]) -> None:
    """Replace 'config'-origin rule rows from the JSON configuration (user overrides untouched)."""
    with db.connect() as conn:
        conn.execute("DELETE FROM rule_configs WHERE firm_code = ? AND origin = 'config'", (firm_code,))
        conn.executemany(
            "INSERT INTO rule_configs (firm_code, account_type_code, rule_key, value_json, status, source_url, "
            "verified_on, effective_date, notes, origin, updated_at) VALUES (?,?,?,?,?,?,?,?,?,'config',?)",
            [
                (
                    firm_code,
                    r["account_type_code"],
                    r["rule_key"],
                    dumps(r.get("value")),
                    r["status"],
                    r.get("source_url"),
                    r.get("verified_on"),
                    r.get("effective_date"),
                    r.get("notes"),
                    now_iso(),
                )
                for r in rows
            ],
        )


def set_rule_override(
    db: Database,
    firm_code: str,
    account_type_code: str,
    rule_key: str,
    value: Any,
    status: str,
    source_url: str | None,
    verified_on: date | None,
    notes: str,
) -> None:
    db.execute(
        "INSERT INTO rule_configs (firm_code, account_type_code, rule_key, value_json, status, source_url, verified_on, "
        "effective_date, notes, origin, updated_at) VALUES (?,?,?,?,?,?,?,NULL,?,'user',?) "
        "ON CONFLICT(firm_code, account_type_code, rule_key, origin) DO UPDATE SET value_json=excluded.value_json, "
        "status=excluded.status, source_url=excluded.source_url, verified_on=excluded.verified_on, "
        "notes=excluded.notes, updated_at=excluded.updated_at",
        (
            firm_code,
            account_type_code,
            rule_key,
            dumps(value),
            status,
            source_url,
            verified_on.isoformat() if verified_on else None,
            notes,
            now_iso(),
        ),
    )
    audit(
        db,
        "rule",
        f"{firm_code}/{account_type_code}/{rule_key}",
        "user_override",
        {
            "value": value,
            "status": status,
            "source_url": source_url,
            "verified_on": verified_on.isoformat() if verified_on else None,
            "notes": notes,
        },
    )


def clear_rule_override(db: Database, firm_code: str, account_type_code: str, rule_key: str) -> None:
    db.execute(
        "DELETE FROM rule_configs WHERE firm_code=? AND account_type_code=? AND rule_key=? AND origin='user'",
        (firm_code, account_type_code, rule_key),
    )
    audit(db, "rule", f"{firm_code}/{account_type_code}/{rule_key}", "override_cleared")


def list_rule_rows(db: Database, firm_code: str, account_type_code: str | None = None) -> list[dict]:
    sql = "SELECT * FROM rule_configs WHERE firm_code = ?"
    params: list[Any] = [firm_code]
    if account_type_code is not None:
        sql += " AND account_type_code IN (?, '*')"
        params.append(account_type_code)
    rows = db.query(sql + " ORDER BY account_type_code, rule_key, origin", tuple(params))
    for r in rows:
        r["value"] = loads(r.pop("value_json"), None)
    return rows


# ---------------------------------------------------------------------------
# Accounts, daily P&L, payouts
# ---------------------------------------------------------------------------


def create_account(db: Database, a: AccountInput) -> int:
    ts = now_iso()
    aid = db.execute(
        "INSERT INTO accounts (firm_code, account_type_code, nickname, stage, account_size, starting_balance, "
        "current_balance, personal_daily_loss_limit, personal_max_contracts, cycle_start_date, notes, created_at, "
        "updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            a.firm_code,
            a.account_type_code,
            a.nickname,
            a.stage,
            a.account_size,
            a.starting_balance,
            a.current_balance,
            a.personal_daily_loss_limit,
            a.personal_max_contracts,
            a.cycle_start_date.isoformat() if a.cycle_start_date else None,
            a.notes,
            ts,
            ts,
        ),
    )
    audit(db, "account", aid, "created", a.model_dump(mode="json"))
    return aid


ACCOUNT_EDITABLE = {
    "nickname",
    "stage",
    "current_balance",
    "current_equity",
    "personal_daily_loss_limit",
    "personal_max_contracts",
    "drawdown_model_confirmed",
    "drawdown_confirmed_on",
    "rules_verified_on",
    "cycle_start_date",
    "status",
    "notes",
    "starting_balance",
}


def update_account(db: Database, account_id: int, **fields: Any) -> None:
    bad = set(fields) - ACCOUNT_EDITABLE
    if bad:
        raise ValueError(f"Fields not editable: {sorted(bad)}")
    if not fields:
        return
    vals = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in fields.items()}
    sets = ", ".join(f"{k} = ?" for k in vals)
    db.execute(
        f"UPDATE accounts SET {sets}, updated_at = ? WHERE id = ?", (*vals.values(), now_iso(), account_id)
    )
    audit(db, "account", account_id, "updated", vals)


def list_accounts(db: Database, include_closed: bool = True) -> list[dict]:
    sql = "SELECT * FROM accounts"
    if not include_closed:
        sql += " WHERE status NOT IN ('closed')"
    return db.query(sql + " ORDER BY firm_code, nickname")


def get_account(db: Database, account_id: int) -> dict | None:
    return db.query_one("SELECT * FROM accounts WHERE id = ?", (account_id,))


def delete_account(db: Database, account_id: int) -> None:
    with db.connect() as conn:
        conn.execute("UPDATE manual_trades SET account_id = NULL WHERE account_id = ?", (account_id,))
        conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
    audit(db, "account", account_id, "deleted")


def upsert_daily_pnl(db: Database, d: DailyPnlInput) -> None:
    ts = now_iso()
    db.execute(
        "INSERT INTO account_daily_pnl (account_id, trade_date, net_pnl, min_equity, max_equity, source, notes, "
        "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(account_id, trade_date) DO UPDATE SET "
        "net_pnl=excluded.net_pnl, min_equity=excluded.min_equity, max_equity=excluded.max_equity, "
        "source=excluded.source, notes=excluded.notes, updated_at=excluded.updated_at",
        (
            d.account_id,
            d.trade_date.isoformat(),
            d.net_pnl,
            d.min_equity,
            d.max_equity,
            d.source,
            d.notes,
            ts,
            ts,
        ),
    )
    audit(db, "account_daily_pnl", d.account_id, "upsert", d.model_dump(mode="json"))


def list_daily_pnl(db: Database, account_id: int) -> list[dict]:
    return db.query("SELECT * FROM account_daily_pnl WHERE account_id = ? ORDER BY trade_date", (account_id,))


def delete_daily_pnl(db: Database, account_id: int, trade_date: date) -> None:
    db.execute(
        "DELETE FROM account_daily_pnl WHERE account_id = ? AND trade_date = ?",
        (account_id, trade_date.isoformat()),
    )
    audit(db, "account_daily_pnl", account_id, "deleted", {"trade_date": trade_date.isoformat()})


def add_payout(db: Database, p: PayoutInput) -> int:
    pid = db.execute(
        "INSERT INTO account_payouts (account_id, requested_on, amount, status, notes, created_at) "
        "VALUES (?,?,?,?,?,?)",
        (p.account_id, p.requested_on.isoformat(), p.amount, p.status, p.notes, now_iso()),
    )
    audit(db, "account_payout", p.account_id, "added", p.model_dump(mode="json"))
    return pid


def list_payouts(db: Database, account_id: int) -> list[dict]:
    return db.query("SELECT * FROM account_payouts WHERE account_id = ? ORDER BY requested_on", (account_id,))


def update_payout_status(db: Database, payout_id: int, status: str) -> None:
    db.execute("UPDATE account_payouts SET status = ? WHERE id = ?", (status, payout_id))
    audit(db, "account_payout", payout_id, "status", {"status": status})


# ---------------------------------------------------------------------------
# Manual trades and management actions
# ---------------------------------------------------------------------------


def create_trade(db: Database, t: ManualTradeInput) -> int:
    ts = now_iso()
    tid = db.execute(
        "INSERT INTO manual_trades (account_id, firm_code, instrument, contract, direction, entry_ts, entry_price, "
        "initial_stop, quantity, planned_target, setup_id, commission_per_contract_rt, slippage_ticks, notes, "
        "tags_json, screenshot_refs_json, status, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'open', ?, ?)",
        (
            t.account_id,
            t.firm_code,
            t.instrument,
            t.contract,
            t.direction,
            to_iso(t.entry_ts),
            t.entry_price,
            t.initial_stop,
            t.quantity,
            t.planned_target,
            t.setup_id,
            t.commission_per_contract_rt,
            t.slippage_ticks,
            t.notes,
            dumps(t.tags),
            dumps([]),
            ts,
            ts,
        ),
    )
    audit(db, "trade", tid, "registered", t.model_dump(mode="json"))
    return tid


JOURNAL_FIELDS = {
    "notes",
    "tags_json",
    "screenshot_refs_json",
    "mistakes",
    "rule_adherence",
    "emotions",
    "planned_target",
    "realized_pnl",
    "pnl_source",
    "contract",
    "account_id",
}


def update_trade_fields(db: Database, trade_id: int, reason: str = "", **fields: Any) -> None:
    bad = set(fields) - JOURNAL_FIELDS
    if bad:
        raise ValueError(f"Fields not editable here: {sorted(bad)}")
    if not fields:
        return
    old = db.query_one("SELECT * FROM manual_trades WHERE id = ?", (trade_id,))
    sets = ", ".join(f"{k} = ?" for k in fields)
    db.execute(
        f"UPDATE manual_trades SET {sets}, updated_at = ? WHERE id = ?",
        (*fields.values(), now_iso(), trade_id),
    )
    audit(
        db,
        "trade",
        trade_id,
        "journal_edit",
        {"reason": reason, "before": {k: (old or {}).get(k) for k in fields}, "after": fields},
    )


def void_trade(db: Database, trade_id: int, reason: str) -> None:
    if not reason.strip():
        raise ValueError("A reason is required to void a trade record.")
    db.execute(
        "UPDATE manual_trades SET is_void = 1, void_reason = ?, updated_at = ? WHERE id = ?",
        (reason, now_iso(), trade_id),
    )
    audit(db, "trade", trade_id, "voided", {"reason": reason})


def list_trades(db: Database, status: str | None = None, include_void: bool = False) -> list[dict]:
    sql = "SELECT * FROM manual_trades WHERE 1=1"
    params: list[Any] = []
    if status:
        sql += " AND status = ?"
        params.append(status)
    if not include_void:
        sql += " AND is_void = 0"
    rows = db.query(sql + " ORDER BY entry_ts DESC", tuple(params))
    for r in rows:
        r["tags"] = loads(r.get("tags_json"), [])
        r["screenshot_refs"] = loads(r.get("screenshot_refs_json"), [])
    return rows


def get_trade(db: Database, trade_id: int) -> dict | None:
    r = db.query_one("SELECT * FROM manual_trades WHERE id = ?", (trade_id,))
    if r:
        r["tags"] = loads(r.get("tags_json"), [])
        r["screenshot_refs"] = loads(r.get("screenshot_refs_json"), [])
    return r


def list_actions(db: Database, trade_id: int, include_void: bool = True) -> list[dict]:
    sql = "SELECT * FROM trade_actions WHERE trade_id = ?"
    if not include_void:
        sql += " AND is_void = 0"
    return db.query(sql + " ORDER BY action_ts, id", (trade_id,))


def add_action(db: Database, a: TradeActionInput, corrects_action_id: int | None = None) -> int:
    """Append a user-recorded management action and recompute the trade's status/P&L."""
    trade = get_trade(db, a.trade_id)
    if trade is None:
        raise ValueError(f"Trade {a.trade_id} not found")
    if trade["is_void"]:
        raise ValueError("Cannot record actions on a voided trade.")
    entry_ts = from_iso(trade["entry_ts"])
    if entry_ts is not None and a.action_ts < entry_ts:
        raise ValueError("Action time is before the trade entry time.")
    actions = [x for x in list_actions(db, a.trade_id, include_void=False) if x["id"] != corrects_action_id]
    open_qty = trade["quantity"] - sum(
        int(x["quantity"] or 0) for x in actions if x["action_type"] in ("partial_exit", "full_exit")
    )
    if (
        trade["status"] == "closed"
        and corrects_action_id is None
        and a.action_type
        in ("partial_exit", "full_exit", "stop_to_breakeven", "stop_tightened", "stop_widened")
    ):
        raise ValueError("Trade is already closed.")
    qty = a.quantity
    if a.action_type == "partial_exit" and (qty is None or qty >= open_qty):
        raise ValueError(
            f"Partial exit quantity must be less than the open quantity ({open_qty}); "
            "use 'Position manually exited' to close the remainder."
        )
    if a.action_type == "full_exit":
        qty = open_qty
    if (
        a.action_type in ("stop_to_breakeven", "stop_tightened", "stop_widened")
        and a.new_stop is not None
        and a.price is not None
    ):
        long = trade["direction"] == "long"
        if (long and a.new_stop >= a.price) or (not long and a.new_stop <= a.price):
            raise ValueError(
                "A protective stop must be below the market price for a long (above for a short)."
            )
    aid = db.execute(
        "INSERT INTO trade_actions (trade_id, action_type, action_ts, price, quantity, new_stop, notes, alert_id, "
        "corrects_action_id, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            a.trade_id,
            a.action_type,
            to_iso(a.action_ts),
            a.price,
            qty,
            a.new_stop,
            a.notes,
            a.alert_id,
            corrects_action_id,
            now_iso(),
        ),
    )
    audit(db, "trade_action", aid, "recorded", {**a.model_dump(mode="json"), "corrects": corrects_action_id})
    if a.alert_id:
        acknowledge_alert(db, a.alert_id, "action_recorded", a.notes, aid)
    recompute_trade(db, a.trade_id)
    return aid


def correct_action(
    db: Database, action_id: int, replacement: TradeActionInput | None, reason: str
) -> int | None:
    """Void an erroneous action (keeping it for audit) and optionally record its replacement."""
    if not reason.strip():
        raise ValueError("A reason is required for a correction.")
    orig = db.query_one("SELECT * FROM trade_actions WHERE id = ?", (action_id,))
    if orig is None:
        raise ValueError(f"Action {action_id} not found")
    if orig["is_void"]:
        raise ValueError("Action already voided.")
    db.execute("UPDATE trade_actions SET is_void = 1, void_reason = ? WHERE id = ?", (reason, action_id))
    audit(db, "trade_action", action_id, "voided", {"reason": reason})
    new_id = None
    if replacement is not None:
        trade = get_trade(db, orig["trade_id"])
        if trade and trade["status"] == "closed":
            db.execute(
                "UPDATE manual_trades SET status = 'open', closed_ts = NULL WHERE id = ?", (orig["trade_id"],)
            )
        new_id = add_action(db, replacement, corrects_action_id=action_id)
    recompute_trade(db, orig["trade_id"])
    return new_id


def recompute_trade(db: Database, trade_id: int) -> None:
    """Derive status and realized P&L from the user's own recorded exits."""
    from core.instruments import get_instrument  # local import to avoid cycles

    trade = get_trade(db, trade_id)
    if trade is None:
        return
    actions = list_actions(db, trade_id, include_void=False)
    exits = [x for x in actions if x["action_type"] in ("partial_exit", "full_exit")]
    exited = sum(int(x["quantity"] or 0) for x in exits)
    status = "closed" if exited >= trade["quantity"] else "open"
    closed_ts = max((x["action_ts"] for x in exits), default=None) if status == "closed" else None
    realized = trade["realized_pnl"]
    pnl_source = trade["pnl_source"]
    if pnl_source not in ("user_entered", "verified"):
        try:
            spec = get_instrument(trade["instrument"])
            sign = 1 if trade["direction"] == "long" else -1
            gross = sum(
                (float(x["price"]) - trade["entry_price"]) * sign * spec.point_value * int(x["quantity"] or 0)
                for x in exits
            )
            comm = trade["commission_per_contract_rt"] * exited
            realized = round(gross - comm, 2) if exits else None
            pnl_source = "computed_from_user_exits" if exits else None
        except KeyError:
            pass
    db.execute(
        "UPDATE manual_trades SET status = ?, closed_ts = ?, realized_pnl = ?, pnl_source = ?, updated_at = ? "
        "WHERE id = ?",
        (status, closed_ts, realized, pnl_source, now_iso(), trade_id),
    )


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------


def save_backtest(
    db: Database,
    *,
    name: str,
    instrument: str,
    data_source: str,
    dataset_ref: str | None,
    contract: str | None,
    start_ts: Any,
    end_ts: Any,
    settings: AppSettings,
    params: dict,
    metrics: dict,
    breakdowns: dict,
    trades: pd.DataFrame,
    notes: str = "",
) -> int:
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO backtest_runs (name, created_at, instrument, data_source, dataset_ref, contract, start_ts, "
            "end_ts, settings_json, settings_hash, params_json, metrics_json, breakdowns_json, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                name,
                now_iso(),
                instrument,
                data_source,
                dataset_ref,
                contract,
                to_iso(start_ts),
                to_iso(end_ts),
                settings.to_json(),
                settings.strategy_hash(),
                dumps(params),
                dumps(metrics),
                dumps(breakdowns),
                notes,
            ),
        )
        run_id = int(cur.lastrowid or 0)
        if trades is not None and not trades.empty:
            cols = [
                "setup_id",
                "direction",
                "entry_ts",
                "entry_price",
                "exit_ts",
                "exit_price",
                "stop",
                "target",
                "quantity",
                "pnl_points",
                "pnl_gross",
                "costs",
                "pnl_net",
                "r_multiple",
                "exit_reason",
                "ambiguous",
                "session",
                "score",
                "vol_regime",
                "trend_regime",
                "sample",
                "bars_held",
            ]
            rows = []
            for r in trades.to_dict("records"):
                vals = [r.get(c) for c in cols]
                vals = [
                    to_iso(v)
                    if isinstance(v, pd.Timestamp)
                    else (
                        None
                        if isinstance(v, float) and math.isnan(v)
                        else (v.item() if hasattr(v, "item") else v)
                    )
                    for v in vals
                ]
                rows.append((run_id, *vals))
            conn.executemany(
                f"INSERT INTO backtest_trades (run_id, {', '.join(cols)}) "
                f"VALUES ({', '.join('?' * (len(cols) + 1))})",
                rows,
            )
    audit(db, "backtest", run_id, "saved", {"name": name, "trades": 0 if trades is None else len(trades)})
    return run_id


def list_backtests(db: Database) -> list[dict]:
    rows = db.query(
        "SELECT id, name, created_at, instrument, data_source, dataset_ref, contract, start_ts, end_ts, "
        "settings_hash, metrics_json, params_json FROM backtest_runs ORDER BY id DESC"
    )
    for r in rows:
        r["metrics"] = loads(r.pop("metrics_json"), {})
        r["params"] = loads(r.pop("params_json"), {})
    return rows


def get_backtest(db: Database, run_id: int) -> dict | None:
    r = db.query_one("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
    if r is None:
        return None
    r["metrics"] = loads(r.pop("metrics_json"), {})
    r["params"] = loads(r.pop("params_json"), {})
    r["breakdowns"] = loads(r.pop("breakdowns_json"), {})
    r["settings"] = AppSettings.from_json(r.pop("settings_json"))
    return r


def get_backtest_trades(db: Database, run_id: int) -> pd.DataFrame:
    df = pd.DataFrame(db.query("SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY entry_ts", (run_id,)))
    if not df.empty:
        for c in ("entry_ts", "exit_ts"):
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def delete_backtest(db: Database, run_id: int) -> None:
    db.execute("DELETE FROM backtest_runs WHERE id = ?", (run_id,))
    audit(db, "backtest", run_id, "deleted")
