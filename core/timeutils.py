"""Timezone, timeframe and futures-session helpers.

Conventions used throughout the application:

* All timestamps are timezone-aware. Internally they are stored in UTC.
* A candle's index timestamp is the bar **open** time. A bar becomes known
  (complete) at ``open + timeframe`` - its *close time*. Every signal is
  stamped with the close time of the bar that made it observable.
* CME Globex hours for equity-index and metals futures are modelled as
  Sunday 18:00 ET -> Friday 17:00 ET with a daily maintenance halt
  17:00-18:00 ET. The *trade date* of a bar is the calendar date on which
  its session closes (bars after 18:00 ET belong to the next trade date).
* Exchange holidays and early closes are NOT modelled; the data-quality
  layer reports unexpected gaps instead of guessing.
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

UTC = "UTC"
EXCHANGE_TZ_NAME = "America/New_York"
EXCHANGE_TZ = ZoneInfo(EXCHANGE_TZ_NAME)
DEFAULT_DISPLAY_TZ = "America/New_York"

SESSION_START = time(18, 0)  # Globex session open, ET
SESSION_END = time(17, 0)  # Globex session close / maintenance halt, ET

_TF_RE = re.compile(r"^(\d+)\s*(min|m|h|d)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Generic timestamp helpers
# ---------------------------------------------------------------------------


def utcnow() -> pd.Timestamp:
    return pd.Timestamp.now(tz=UTC)


def ensure_utc(ts: object) -> pd.Timestamp:
    """Return ``ts`` as a UTC ``Timestamp``. Naive values are rejected."""
    t = pd.Timestamp(ts)  # type: ignore[arg-type]
    if t.tzinfo is None:
        raise ValueError(f"Naive timestamp {ts!r}: timezone information is required")
    return t.tz_convert(UTC)


def to_iso(ts: object | None) -> str | None:
    """Serialise a timestamp as an ISO-8601 UTC string (for SQLite)."""
    if ts is None:
        return None
    if isinstance(ts, float) and np.isnan(ts):
        return None
    t = pd.Timestamp(ts)  # type: ignore[arg-type]
    if pd.isna(t):
        return None
    if t.tzinfo is None:
        raise ValueError(f"Refusing to serialise naive timestamp {ts!r}")
    return t.tz_convert(UTC).isoformat()


def from_iso(value: str | None) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        # Stored values are always UTC; treat a naive legacy value as UTC.
        t = t.tz_localize(UTC)
    return t.tz_convert(UTC)


def validate_timezone(name: str) -> str:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"Unknown timezone: {name!r}") from exc
    return name


def to_display(ts: object | None, tz: str = DEFAULT_DISPLAY_TZ, fmt: str = "%Y-%m-%d %H:%M %Z") -> str:
    if ts is None:
        return "-"
    t = pd.Timestamp(ts)  # type: ignore[arg-type]
    if pd.isna(t):
        return "-"
    if t.tzinfo is None:
        t = t.tz_localize(UTC)
    return t.tz_convert(tz).strftime(fmt)


# ---------------------------------------------------------------------------
# Timeframes
# ---------------------------------------------------------------------------


def timeframe_minutes(tf: str) -> int:
    """Parse '5min', '15min', '1h', '4h', '1d' into minutes."""
    m = _TF_RE.match(tf.strip())
    if not m:
        raise ValueError(f"Unsupported timeframe: {tf!r}")
    n, unit = int(m.group(1)), m.group(2).lower()
    if n <= 0:
        raise ValueError(f"Timeframe must be positive: {tf!r}")
    if unit in ("min", "m"):
        return n
    if unit == "h":
        return n * 60
    return n * 60 * 24


def timeframe_delta(tf: str) -> pd.Timedelta:
    return pd.Timedelta(minutes=timeframe_minutes(tf))


def bar_close_times(index: pd.DatetimeIndex, tf: str) -> pd.DatetimeIndex:
    """Close (availability) time for bars indexed by open time."""
    return index + timeframe_delta(tf)


# ---------------------------------------------------------------------------
# Futures session calendar (CME Globex, ET)
# ---------------------------------------------------------------------------


def _local(ts: pd.Timestamp, tz: str = EXCHANGE_TZ_NAME) -> pd.Timestamp:
    return ensure_utc(ts).tz_convert(tz)


def is_market_open(ts: object) -> bool:
    """Whether CME Globex (equity index / metals) is normally open at ``ts``.

    Holidays and early closes are not modelled.
    """
    t = _local(pd.Timestamp(ts))  # type: ignore[arg-type]
    wd = t.weekday()  # Mon=0 .. Sun=6
    tt = t.time()
    if wd == 5:  # Saturday
        return False
    if wd == 6:  # Sunday: opens 18:00
        return tt >= SESSION_START
    if wd == 4:  # Friday: closes 17:00
        return tt < SESSION_END
    # Mon-Thu: closed during 17:00-18:00 maintenance halt
    return not (SESSION_END <= tt < SESSION_START)


def market_open_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Vectorised :func:`is_market_open` for a tz-aware index."""
    local = index.tz_convert(EXCHANGE_TZ_NAME)
    wd = local.weekday.to_numpy()
    mins = (local.hour * 60 + local.minute).to_numpy()
    start = SESSION_START.hour * 60 + SESSION_START.minute
    end = SESSION_END.hour * 60 + SESSION_END.minute
    sat = wd == 5
    sun = wd == 6
    fri = wd == 4
    mask = np.ones(len(index), dtype=bool)
    mask[sat] = False
    mask[sun] = mins[sun] >= start
    mask[fri] = mins[fri] < end
    midweek = ~(sat | sun | fri)
    mask[midweek] = ~((mins[midweek] >= end) & (mins[midweek] < start))
    return mask


def trade_date(ts: object) -> date:
    """CME trade date: bars at/after 18:00 ET belong to the next calendar day."""
    t = _local(pd.Timestamp(ts))  # type: ignore[arg-type]
    d = t.date()
    if t.time() >= SESSION_START:
        d = d + timedelta(days=1)
    # Sunday-evening bars belong to Monday's trade date (already +1 above).
    return d


def trade_dates(index: pd.DatetimeIndex) -> np.ndarray:
    """Vectorised trade dates (numpy array of ``datetime.date``)."""
    local = index.tz_convert(EXCHANGE_TZ_NAME)
    naive = local.tz_localize(None)
    days = naive.normalize()
    after_open = (local.hour * 60 + local.minute) >= SESSION_START.hour * 60 + SESSION_START.minute
    out = days + pd.to_timedelta(np.where(after_open, 1, 0), unit="D")
    return np.array([d.date() for d in out])


def session_anchor_local(ts: object) -> datetime:
    """Most recent 18:00 ET session start (naive local datetime) at or before ``ts``."""
    t = _local(pd.Timestamp(ts)).tz_localize(None)  # type: ignore[arg-type]
    start_today = datetime.combine(t.date(), SESSION_START)
    if t.to_pydatetime() >= start_today:
        return start_today
    return start_today - timedelta(days=1)


def session_close_utc(td: date) -> pd.Timestamp:
    """UTC timestamp of the 17:00 ET close for a given trade date."""
    local = pd.Timestamp(datetime.combine(td, SESSION_END)).tz_localize(EXCHANGE_TZ_NAME)
    return local.tz_convert(UTC)


SESSION_LABELS = (
    # (label, start ET, end ET) - end exclusive; Asia wraps midnight.
    ("Asia", time(18, 0), time(3, 0)),
    ("London", time(3, 0), time(9, 30)),
    ("NY AM", time(9, 30), time(12, 0)),
    ("NY PM", time(12, 0), time(17, 0)),
)


def session_label(ts: object) -> str:
    t = _local(pd.Timestamp(ts)).time()  # type: ignore[arg-type]
    for label, start, end in SESSION_LABELS:
        if start <= end:
            if start <= t < end:
                return label
        elif t >= start or t < end:
            return label
    return "Closed"


def parse_hhmm(value: str) -> time:
    try:
        hh, mm = value.strip().split(":")
        return time(int(hh), int(mm))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid HH:MM time: {value!r}") from exc


def in_time_windows(ts: object, windows: list[tuple[str, str]], tz: str = EXCHANGE_TZ_NAME) -> bool:
    """True if the local time of ``ts`` falls in any [start, end) window.

    Windows may wrap midnight (e.g. ``("18:00", "03:00")``). An empty window
    list means "no restriction".
    """
    if not windows:
        return True
    t = _local(pd.Timestamp(ts), tz).time()  # type: ignore[arg-type]
    for start_s, end_s in windows:
        start, end = parse_hhmm(start_s), parse_hhmm(end_s)
        if start <= end:
            if start <= t < end:
                return True
        elif t >= start or t < end:
            return True
    return False


def minutes_until_session_close(ts: object) -> float:
    """Minutes from ``ts`` until that trade date's 17:00 ET close."""
    t = ensure_utc(pd.Timestamp(ts))  # type: ignore[arg-type]
    close = session_close_utc(trade_date(t))
    return (close - t).total_seconds() / 60.0
