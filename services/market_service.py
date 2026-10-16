"""Loading market data for the selected source, instrument, contract and replay clock.

Contract handling: the full series is split into single-contract segments
and exactly one segment is analysed at a time (default: the most recent).
Different contracts are never spliced together.

Clock handling:
* live/delayed feeds use the wall clock;
* synthetic and historical data use a replay clock = close of the replay
  cursor bar (plus an optional simulated feed stall, to exercise the
  stale-data path in demo mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import pandas as pd

from core.config import AppSettings
from core.data_quality import DataQualityReport, FreshnessStatus, assess_freshness, validate_candles
from core.instruments import ContractInfo, InstrumentSpec, default_registry, front_contract
from core.scanner import split_contract_segments
from core.timeutils import timeframe_delta, utcnow
from data.adapters.base import AdapterStatus, AdapterUnavailable
from data.adapters.csv_dataset import CsvDatasetAdapter
from data.adapters.databento_adapter import DatabentoAdapter
from data.adapters.synthetic import SyntheticAdapter
from data.database import Database
from data.synthetic_data import default_end_date, generate_synthetic_candles

SOURCE_LABELS = {
    "synthetic": "Synthetic demo (simulated)",
    "csv": "Imported CSV (historical)",
    "databento": "Databento (delayed, untested)",
}


@dataclass
class MarketSelection:
    source: str = "synthetic"
    instrument: str = "MNQ"
    dataset_id: int | None = None
    contract: str | None = None  # None -> most recent segment
    replay_bar: int | None = None  # index within the selected segment; None -> latest
    stall_minutes: int = 0  # demo only: pretend the feed stopped this many minutes ago
    synthetic_end: date | None = None


@dataclass
class MarketData:
    selection: MarketSelection
    spec: InstrumentSpec
    feed_type: str
    source_label: str
    context_key: str
    segments: list[dict] = field(default_factory=list)
    contract: str | None = None
    segment: pd.DataFrame = field(default_factory=pd.DataFrame)  # full selected contract segment
    visible: pd.DataFrame = field(default_factory=pd.DataFrame)  # segment up to the replay cursor
    cursor: int | None = None
    now: pd.Timestamp | None = None
    as_of: pd.Timestamp | None = None
    freshness: FreshnessStatus | None = None
    quality: DataQualityReport | None = None
    adapter_status: AdapterStatus | None = None
    error: str | None = None
    front: ContractInfo | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and not self.visible.empty

    @property
    def is_replay(self) -> bool:
        return self.feed_type in ("simulated", "historical")


@lru_cache(maxsize=24)
def _synthetic(symbol: str, days: int, seed: int, end: date) -> pd.DataFrame:
    return generate_synthetic_candles(default_registry().get(symbol), end=end, days=days, seed=seed)


def load_market(db: Database, settings: AppSettings, sel: MarketSelection) -> MarketData:
    spec = default_registry().get(sel.instrument)
    tf = settings.timeframes.entry_timeframe
    md = MarketData(
        selection=sel,
        spec=spec,
        feed_type="simulated",
        source_label=SOURCE_LABELS.get(sel.source, sel.source),
        context_key="",
    )
    full: pd.DataFrame
    if sel.source == "synthetic":
        end = sel.synthetic_end or settings.data.synthetic_end_date or default_end_date()
        adapter = SyntheticAdapter(settings.data.synthetic_days, settings.data.synthetic_seed, end)
        md.adapter_status = adapter.status()
        md.feed_type = "simulated"
        full = _synthetic(spec.symbol, settings.data.synthetic_days, settings.data.synthetic_seed, end).copy()
    elif sel.source == "csv":
        if sel.dataset_id is None:
            md.error = "No CSV dataset selected. Import one in Settings & Data Sources."
            return md
        adapter_c = CsvDatasetAdapter(db, sel.dataset_id)
        md.adapter_status = adapter_c.status()
        md.feed_type = "historical"
        try:
            full = adapter_c.get_candles(spec.symbol)
        except AdapterUnavailable as exc:
            md.error = str(exc)
            return md
    elif sel.source == "databento":
        adapter_d = DatabentoAdapter()
        md.adapter_status = adapter_d.status()
        md.feed_type = "delayed"
        today = utcnow().date()
        contract = sel.contract or front_contract(spec, today).code
        end_ts = utcnow()
        start_ts = end_ts - pd.Timedelta(days=settings.data.live_history_days)
        try:
            full = adapter_d.get_candles(spec.symbol, start_ts, end_ts, tf, contract)
        except AdapterUnavailable as exc:
            md.error = f"Databento unavailable: {exc}"
            return md
    else:
        md.error = f"Unknown data source {sel.source!r}"
        return md

    if full.empty:
        md.error = "The data source returned no candles."
        return md

    segs = split_contract_segments(full)
    md.segments = [{"contract": c, "start": g.index[0], "end": g.index[-1], "bars": len(g)} for c, g in segs]
    chosen = segs[-1]
    if sel.contract:
        match = [s for s in segs if s[0] == sel.contract]
        if match:
            chosen = match[-1]
        else:
            md.notes.append(f"Contract {sel.contract} not in data; using {chosen[0]}.")
    md.contract = None if chosen[0] == "UNKNOWN" else chosen[0]
    seg = chosen[1]
    if len(segs) > 1:
        md.notes.append(
            f"Data contains {len(segs)} contract segments; analysing {chosen[0]} only "
            "(contracts are never spliced)."
        )
    md.segment = seg
    n = len(seg)
    cursor = n - 1 if sel.replay_bar is None else max(0, min(int(sel.replay_bar), n - 1))
    md.cursor = cursor
    md.visible = seg.iloc[: cursor + 1]
    md.as_of = md.visible.index[-1] + timeframe_delta(tf)
    src_key = f"csv{sel.dataset_id}" if sel.source == "csv" else sel.source
    md.context_key = f"{src_key}:{spec.symbol}:{md.contract or 'NA'}"

    if md.feed_type in ("simulated", "historical"):
        md.now = md.as_of + pd.Timedelta(minutes=sel.stall_minutes)
        replay = True
    else:
        md.now = utcnow()
        replay = False
    md.freshness = assess_freshness(
        md.as_of,
        md.now,
        tf,
        md.feed_type,  # type: ignore[arg-type]
        settings.data.stale_multiplier,
        settings.data.stale_grace_seconds,
        replay_clock=replay,
    )
    md.quality = validate_candles(md.visible, tf, spec.tick_size)
    try:
        md.front = front_contract(spec, md.visible.index[-1].date())
        if md.contract and md.front.code != md.contract:
            md.notes.append(
                f"By the {spec.symbol} roll rule the front contract on this date would be {md.front.code} "
                f"(calculated; holidays ignored)."
            )
    except Exception:  # noqa: BLE001
        md.front = None
    return md
