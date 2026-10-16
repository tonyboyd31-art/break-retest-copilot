"""Running the strategy on loaded market data and persisting results + alerts."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.config import AppSettings
from core.events import EngineEvent
from core.scanner import ScanResult, run_scan
from core.timeutils import timeframe_delta
from data import repositories as repo
from data.database import Database
from notifications.manager import AlertManager
from services.market_service import MarketData


@dataclass
class PersistSummary:
    inserted: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    alerts_created: int = 0
    alert_titles: list[str] = field(default_factory=list)


def scan_market(md: MarketData, settings: AppSettings) -> ScanResult:
    block_from = block_reason = None
    if md.freshness is not None and md.freshness.is_stale:
        block_from = md.as_of
        block_reason = "Market data was stale when this setup completed - not qualified."
    return run_scan(md.visible, md.spec, settings, md.contract, block_from, block_reason)


def data_events(md: MarketData) -> list[EngineEvent]:
    """Alerts for stale or unreliable market data."""
    out: list[EngineEvent] = []
    if md.as_of is None:
        return out
    if md.freshness is not None and md.freshness.is_stale:
        out.append(
            EngineEvent(
                category="data_stale",
                ts=md.now or md.as_of,
                title=f"Market data stale: {md.spec.symbol}",
                message=md.freshness.message,
                dedup_key=f"data:stale:{md.context_key}:{md.as_of.isoformat()}",
                instrument=md.spec.symbol,
            )
        )
    if md.quality is not None and md.quality.errors:
        msg = "; ".join(e.message for e in md.quality.errors)
        out.append(
            EngineEvent(
                category="data_stale",
                ts=md.as_of,
                title=f"Market data unreliable: {md.spec.symbol}",
                message=f"Signal qualification disabled. {msg}",
                dedup_key=f"data:invalid:{md.context_key}:{md.as_of.isoformat()}",
                instrument=md.spec.symbol,
                severity="critical",
            )
        )
    if md.quality is not None:
        recent_gap = [p for p in md.quality.gap_positions if p >= len(md.visible) - 3]
        if recent_gap:
            out.append(
                EngineEvent(
                    category="data_stale",
                    ts=md.as_of,
                    title=f"Missing candles: {md.spec.symbol}",
                    message="Recent in-session candles are missing; setups spanning the gap were invalidated.",
                    dedup_key=f"data:gap:{md.context_key}:{md.visible.index[recent_gap[-1]].isoformat()}",
                    instrument=md.spec.symbol,
                )
            )
    return out


def persist_scan(
    db: Database,
    md: MarketData,
    scan: ScanResult,
    settings: AppSettings,
    alerts: AlertManager | None = None,
    alert_all: bool = False,
) -> PersistSummary:
    """Store setups/swings/zones and publish alerts for recent events only.

    ``alert_all`` publishes every historical event (used by tests); by default
    only events within ``alert_lookback_bars`` of the latest bar are alerted so
    loading a long history does not flood the Alerts Center.
    """
    summary = PersistSummary()
    if not scan.scanned:
        if alerts:
            for ev in data_events(md):
                r = alerts.publish(ev, md.context_key)
                summary.alerts_created += int(r.created)
        return summary
    res = repo.upsert_setups(
        db, scan.setups, md.context_key, md.selection.source, md.feed_type, scan.settings_hash
    )
    summary.inserted, summary.updated = res["inserted"], res["updated"]
    repo.save_swings(db, scan.swings, md.context_key, md.spec.symbol, md.contract, scan.timeframe)
    repo.replace_zones(
        db, scan.zones, md.context_key, md.spec.symbol, md.contract, scan.timeframe, scan.as_of
    )
    if md.freshness:
        repo.upsert_market_status(
            db,
            md.context_key,
            md.spec.symbol,
            md.contract,
            md.feed_type,
            md.selection.source,
            md.as_of,
            md.freshness.status,
            md.freshness.message,
        )
    if alerts:
        lookback = settings.data.alert_lookback_bars * timeframe_delta(scan.timeframe)
        cutoff = (scan.as_of - lookback) if scan.as_of is not None else None
        events = (
            list(scan.events) if alert_all else [e for e in scan.events if cutoff is None or e.ts > cutoff]
        )
        events += data_events(md)
        for ev in events:
            r = alerts.publish(ev, md.context_key)
            if r.created:
                summary.alerts_created += 1
                summary.alert_titles.append(ev.title)
    return summary


def latest_qualified(scan: ScanResult, n: int = 10) -> list:
    q = [s for s in scan.setups if s.state.value == "qualified"]
    return sorted(q, key=lambda s: s.confirm_ts or pd.Timestamp.min.tz_localize("UTC"), reverse=True)[:n]
