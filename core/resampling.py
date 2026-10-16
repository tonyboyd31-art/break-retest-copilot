"""Session-aware candle resampling and higher-timeframe trend context.

Higher-timeframe (HTF) bars are aligned to the 18:00 ET Globex session
open, so a 4-hour bar is 18-22, 22-02, 02-06, 06-10, 10-14, 14-17 ET
regardless of daylight-saving time. The last bin of a session is truncated
at the 17:00 ET close.

Each HTF bar carries ``close_ts``: the moment the bar is complete. When the
entry timeframe asks for HTF context at time ``t`` it may only use HTF bars
with ``close_ts <= t`` (see :func:`htf_value_at`). This is the guard against
higher-timeframe look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import ema
from core.timeutils import EXCHANGE_TZ_NAME, timeframe_delta, timeframe_minutes

SESSION_START_MIN = 18 * 60
SESSION_LENGTH_MIN = 23 * 60  # 18:00 -> 17:00 next day


def resample_session_aware(df: pd.DataFrame, timeframe: str, base_timeframe: str = "5min") -> pd.DataFrame:
    """Resample OHLCV bars into session-aligned higher-timeframe bars.

    Bins never mix contracts. Output index = HTF bar open (UTC) with columns
    open, high, low, close, volume, contract, close_ts, n_bars, expected_bars.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "open",
                "high",
                "low",
                "close",
                "volume",
                "contract",
                "close_ts",
                "n_bars",
                "expected_bars",
            ]
        )
    minutes = timeframe_minutes(timeframe)
    base_min = timeframe_minutes(base_timeframe)
    if minutes % base_min:
        raise ValueError(f"{timeframe} is not a multiple of {base_timeframe}")

    local = df.index.tz_convert(EXCHANGE_TZ_NAME).tz_localize(None)
    mins_of_day = np.asarray(local.hour * 60 + local.minute)
    day = local.normalize()
    anchor = np.where(
        mins_of_day >= SESSION_START_MIN,
        day + pd.Timedelta(minutes=SESSION_START_MIN),
        day - pd.Timedelta(days=1) + pd.Timedelta(minutes=SESSION_START_MIN),
    )
    anchor = pd.DatetimeIndex(anchor)
    offset_min = np.asarray((local - anchor) / pd.Timedelta(minutes=1)).astype(np.int64)
    bin_idx = offset_min // minutes
    bin_start_local = anchor + pd.to_timedelta(bin_idx * minutes, unit="min")
    bin_end_offset = np.minimum((bin_idx + 1) * minutes, SESSION_LENGTH_MIN)
    bin_end_local = anchor + pd.to_timedelta(bin_end_offset, unit="min")

    # Sessions never straddle a DST change (changes occur Sunday 02:00 local,
    # while the market is closed) so localising bin edges is unambiguous.
    bin_start_utc = bin_start_local.tz_localize(EXCHANGE_TZ_NAME).tz_convert("UTC")
    bin_end_utc = bin_end_local.tz_localize(EXCHANGE_TZ_NAME).tz_convert("UTC")

    work = pd.DataFrame(
        {
            "open": df["open"].to_numpy(),
            "high": df["high"].to_numpy(),
            "low": df["low"].to_numpy(),
            "close": df["close"].to_numpy(),
            "volume": pd.to_numeric(df["volume"], errors="coerce").to_numpy() if "volume" in df else np.nan,
            "contract": df["contract"].astype(str).to_numpy() if "contract" in df else "UNKNOWN",
            "bin": bin_start_utc,
            "close_ts": bin_end_utc,
            "expected": (bin_end_offset - bin_idx * minutes) // base_min,
        }
    )
    g = work.groupby(["contract", "bin"], sort=False)
    out = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "volume": g["volume"].sum(min_count=1),
            "close_ts": g["close_ts"].first(),
            "n_bars": g["open"].size(),
            "expected_bars": g["expected"].first(),
        }
    ).reset_index()
    out = out.set_index("bin").sort_index()
    out.index.name = "ts"
    # A bin is only complete once its close time has passed relative to the
    # last base bar available.
    last_available_close = df.index[-1] + timeframe_delta(base_timeframe)
    out["complete"] = out["close_ts"] <= last_available_close
    return out[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "contract",
            "close_ts",
            "n_bars",
            "expected_bars",
            "complete",
        ]
    ]


def compute_trend(htf: pd.DataFrame, fast: int, slow: int, slope_bars: int = 3) -> pd.DataFrame:
    """Classify each completed HTF bar as up / down / neutral.

    up:      close > EMA(fast) > EMA(slow) and EMA(fast) rising over ``slope_bars``
    down:    close < EMA(fast) < EMA(slow) and EMA(fast) falling
    neutral: otherwise
    NaN (insufficient history) until EMA(slow) has enough bars.
    """
    if htf.empty:
        return pd.DataFrame(columns=["close_ts", "trend", "ema_fast", "ema_slow"])
    done = htf[htf["complete"]].copy()
    ef = ema(done["close"], fast)
    es = ema(done["close"], slow)
    slope = ef - ef.shift(slope_bars)
    c = done["close"]
    trend = np.where(
        es.isna() | slope.isna(),
        None,
        np.where(
            (c > ef) & (ef > es) & (slope > 0),
            "up",
            np.where((c < ef) & (ef < es) & (slope < 0), "down", "neutral"),
        ),
    )
    return pd.DataFrame(
        {
            "close_ts": done["close_ts"].to_numpy(),
            "trend": trend,
            "ema_fast": ef.to_numpy(),
            "ema_slow": es.to_numpy(),
        },
        index=done.index,
    )


def htf_value_at(
    trend_df: pd.DataFrame, query_ts: pd.DatetimeIndex | pd.Series, column: str = "trend"
) -> np.ndarray:
    """For each query time, return ``column`` of the latest HTF bar with close_ts <= t."""
    q = pd.DatetimeIndex(query_ts)
    if trend_df.empty:
        return np.array([None] * len(q), dtype=object)
    close_ts = pd.DatetimeIndex(trend_df["close_ts"]).tz_convert("UTC")
    order = np.argsort(close_ts.asi8, kind="stable")
    sorted_close = close_ts[order]
    pos = sorted_close.searchsorted(q.tz_convert("UTC"), side="right") - 1
    values = trend_df[column].to_numpy()[order]
    out = np.empty(len(q), dtype=object)
    for i, p in enumerate(pos):
        out[i] = values[p] if p >= 0 else None
    return out
