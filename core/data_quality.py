"""Market-data validation and feed-freshness assessment.

The signal engine refuses to qualify new setups unless the data passes
these checks. Problems are reported, never silently repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from core.indicators import volume_is_reliable
from core.instruments import is_on_tick
from core.timeutils import is_market_open, market_open_mask, timeframe_delta

REQUIRED_COLUMNS = ("open", "high", "low", "close")
FeedType = Literal["live", "delayed", "historical", "simulated"]


@dataclass
class DataIssue:
    code: str
    severity: Literal["error", "warning", "info"]
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    last_bar_close: pd.Timestamp | None
    contracts: list[str]
    volume_reliable: bool
    volume_note: str
    missing_bars: int
    gap_positions: list[int]  # row positions preceded by an unexpected in-session gap
    issues: list[DataIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[DataIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[DataIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok_for_signals(self) -> bool:
        """True when no blocking error exists."""
        return self.rows > 0 and not self.errors

    def summary(self) -> str:
        if self.rows == 0:
            return "No data"
        parts = [f"{self.rows} bars"]
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)


def _fmt(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")


def validate_candles(
    df: pd.DataFrame,
    timeframe: str = "5min",
    tick_size: float | None = None,
    allow_multiple_contracts: bool = False,
    max_examples: int = 5,
) -> DataQualityReport:
    """Validate an OHLCV frame indexed by tz-aware bar-open timestamps."""
    issues: list[DataIssue] = []
    n = len(df)
    if n == 0:
        return DataQualityReport(
            0,
            None,
            None,
            None,
            [],
            False,
            "No data",
            0,
            [],
            [DataIssue("no_data", "error", "No candles available.")],
        )

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(DataIssue("missing_columns", "error", f"Missing required columns: {missing_cols}"))
        return DataQualityReport(n, None, None, None, [], False, "n/a", 0, [], issues)

    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        issues.append(DataIssue("naive_timestamps", "error", "Index must be timezone-aware timestamps."))
        return DataQualityReport(n, None, None, None, [], False, "n/a", 0, [], issues)

    tf = timeframe_delta(timeframe)
    start, end = idx.min(), idx.max()

    if not idx.is_monotonic_increasing:
        bad = int((np.diff(idx.asi8) < 0).sum()) if n > 1 else 0
        issues.append(DataIssue("unsorted", "error", "Timestamps are not in ascending order.", bad))

    dup_mask = idx.duplicated(keep=False)
    if dup_mask.any():
        ex = [_fmt(t) for t in idx[dup_mask][:max_examples]]
        issues.append(
            DataIssue("duplicates", "error", "Duplicate bar timestamps.", int(idx.duplicated().sum()), ex)
        )

    o, h, low, c = (pd.to_numeric(df[k], errors="coerce") for k in REQUIRED_COLUMNS)
    nan_mask = o.isna() | h.isna() | low.isna() | c.isna()
    if nan_mask.any():
        issues.append(
            DataIssue(
                "nan_prices",
                "error",
                "Missing/non-numeric OHLC values.",
                int(nan_mask.sum()),
                [_fmt(t) for t in idx[nan_mask.to_numpy()][:max_examples]],
            )
        )
    bad_ohlc = (~nan_mask) & ((h < np.maximum(o, c)) | (low > np.minimum(o, c)) | (h < low) | (low <= 0))
    if bad_ohlc.any():
        issues.append(
            DataIssue(
                "invalid_ohlc",
                "error",
                "Invalid OHLC (high below open/close, low above open/close, or non-positive).",
                int(bad_ohlc.sum()),
                [_fmt(t) for t in idx[bad_ohlc.to_numpy()][:max_examples]],
            )
        )

    # Bar alignment to the timeframe grid
    secs = (idx - idx.normalize()).total_seconds()
    misaligned = (np.asarray(secs) % tf.total_seconds()) != 0
    if misaligned.any():
        issues.append(
            DataIssue(
                "misaligned",
                "warning",
                f"{int(misaligned.sum())} bars are not aligned to the {timeframe} grid.",
                int(misaligned.sum()),
            )
        )

    if tick_size:
        off = [not is_on_tick(float(p), tick_size) for p in c.dropna().to_numpy()[-2000:]]
        if any(off):
            issues.append(
                DataIssue(
                    "off_tick",
                    "warning",
                    "Some prices are not multiples of the tick size (back-adjusted series?).",
                    int(sum(off)),
                )
            )

    contracts: list[str] = []
    if "contract" in df.columns:
        contracts = [str(x) for x in pd.unique(df["contract"].astype(str))]
        if len(contracts) > 1 and not allow_multiple_contracts:
            issues.append(
                DataIssue(
                    "multiple_contracts",
                    "error",
                    f"Series mixes {len(contracts)} contracts {contracts[:4]}; analyse one contract segment at a time "
                    "(the engine never treats different contract prices as one continuous series).",
                    len(contracts),
                )
            )
    else:
        issues.append(
            DataIssue(
                "no_contract",
                "warning",
                "No contract column: contract month unknown. Roll boundaries cannot be detected.",
            )
        )

    # Unexpected in-session gaps
    gap_positions: list[int] = []
    missing_total = 0
    if n > 1 and idx.is_monotonic_increasing:
        diffs = idx[1:] - idx[:-1]
        cand = np.where(diffs > tf)[0]
        for pos in cand:
            prev_t, next_t = idx[pos], idx[pos + 1]
            steps = pd.date_range(prev_t + tf, next_t - tf, freq=tf)
            if len(steps) == 0:
                continue
            open_steps = int(market_open_mask(steps).sum())
            if open_steps > 0:
                missing_total += open_steps
                gap_positions.append(int(pos + 1))
        if missing_total:
            ex = [f"before {_fmt(idx[p])}" for p in gap_positions[:max_examples]]
            sev: Literal["error", "warning"] = "warning"
            issues.append(
                DataIssue(
                    "missing_bars",
                    sev,
                    f"{missing_total} expected in-session bars are missing across "
                    f"{len(gap_positions)} gap(s). Setups spanning a gap are invalidated.",
                    missing_total,
                    ex,
                )
            )

    # Bars during the maintenance halt / weekend are suspicious
    closed = ~market_open_mask(idx)
    if closed.any():
        issues.append(
            DataIssue(
                "bars_when_closed",
                "warning",
                f"{int(closed.sum())} bars fall outside normal Globex hours "
                "(holiday schedule or timezone error?).",
                int(closed.sum()),
                [_fmt(t) for t in idx[closed][:max_examples]],
            )
        )

    vol_ok, vol_note = volume_is_reliable(df)
    if not vol_ok:
        issues.append(DataIssue("volume_unreliable", "info", f"Volume filters and VWAP disabled: {vol_note}"))

    return DataQualityReport(
        rows=n,
        start=start,
        end=end,
        last_bar_close=end + tf,
        contracts=contracts,
        volume_reliable=vol_ok,
        volume_note=vol_note,
        missing_bars=missing_total,
        gap_positions=gap_positions,
        issues=issues,
    )


@dataclass
class FreshnessStatus:
    status: Literal["fresh", "stale", "market_closed", "historical", "simulated", "no_data"]
    feed_type: FeedType
    age_seconds: float | None
    message: str

    @property
    def allows_new_signals(self) -> bool:
        """Whether a *live* qualification is permitted under this freshness state."""
        return self.status in ("fresh", "simulated", "historical")

    @property
    def is_stale(self) -> bool:
        return self.status == "stale"


def assess_freshness(
    last_bar_close: pd.Timestamp | None,
    now: pd.Timestamp,
    timeframe: str,
    feed_type: FeedType,
    stale_multiplier: float = 2.0,
    grace_seconds: int = 60,
    replay_clock: bool = False,
) -> FreshnessStatus:
    """Assess whether the most recent completed bar is recent enough.

    For simulated or historical replay ``now`` is the replay clock. A feed
    is *stale* when the market is open and more than
    ``stale_multiplier x timeframe + grace`` has passed since the last bar
    closed. When the exchange is closed no new bars are expected.
    """
    if last_bar_close is None:
        return FreshnessStatus("no_data", feed_type, None, "No market data loaded.")
    age = (now - last_bar_close).total_seconds()
    limit = stale_multiplier * timeframe_delta(timeframe).total_seconds() + grace_seconds
    historical_msg = "Historical data: suitable for research/replay only, not live decisions."
    if feed_type == "historical" and not replay_clock:
        return FreshnessStatus("historical", feed_type, age, historical_msg)
    if age > limit:
        if not is_market_open(now) and not is_market_open(now - pd.Timedelta(seconds=limit)):
            return FreshnessStatus(
                "market_closed",
                feed_type,
                age,
                "Exchange closed (normal Globex hours); no new bars expected.",
            )
        return FreshnessStatus(
            "stale",
            feed_type,
            age,
            f"Last bar closed {age / 60:.1f} min ago (limit {limit / 60:.1f} min). "
            "New signal qualification is disabled.",
        )
    if feed_type == "simulated":
        return FreshnessStatus("simulated", feed_type, age, "Simulated demo data - not real market prices.")
    if feed_type == "historical":
        return FreshnessStatus("historical", feed_type, age, "Historical replay - " + historical_msg)
    return FreshnessStatus("fresh", feed_type, age, f"Last bar closed {max(age, 0) / 60:.1f} min ago.")
