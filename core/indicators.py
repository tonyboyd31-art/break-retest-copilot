"""Deterministic technical indicators.

Every indicator at bar ``i`` uses only bars ``<= i`` so values are usable at
that bar's close. Indicators never span a futures contract boundary: the
``contract`` column (if present) resets true-range and VWAP calculations.

VWAP and relative volume require trustworthy volume. When volume is missing
or unreliable these functions return NaN - they never fabricate values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.timeutils import EXCHANGE_TZ_NAME


def _contract_break(df: pd.DataFrame) -> np.ndarray:
    """Boolean array: True where the contract differs from the previous bar."""
    if "contract" not in df.columns:
        out = np.zeros(len(df), dtype=bool)
        if len(df):
            out[0] = True
        return out
    c = df["contract"].astype(str).to_numpy()
    out = np.ones(len(df), dtype=bool)
    if len(df) > 1:
        out[1:] = c[1:] != c[:-1]
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    prev_close[_contract_break(df)] = np.nan
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR (RMA of true range). NaN until ``period`` bars exist."""
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().rename("atr")


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def volume_is_reliable(df: pd.DataFrame, max_bad_fraction: float = 0.05) -> tuple[bool, str]:
    """Decide whether the volume column can be trusted for filters."""
    if "volume" not in df.columns:
        return False, "No volume column in the data."
    v = pd.to_numeric(df["volume"], errors="coerce")
    if len(v) == 0:
        return False, "No bars."
    bad = v.isna() | (v <= 0)
    frac = float(bad.mean())
    if frac > max_bad_fraction:
        return False, f"{frac:.0%} of bars have missing or zero volume (limit {max_bad_fraction:.0%})."
    if v.nunique(dropna=True) <= 1:
        return False, "Volume is constant - likely a placeholder, not real volume."
    return True, "Volume present on all but a negligible share of bars."


def volume_reliability_mask(
    df: pd.DataFrame, window: int = 200, max_bad_fraction: float = 0.05
) -> np.ndarray:
    """Causal per-bar flag: volume over the trailing ``window`` bars is trustworthy.

    Uses only bars up to and including each bar, so whether VWAP / relative
    volume are available for a historical signal never depends on later data.
    """
    n = len(df)
    if "volume" not in df.columns or n == 0:
        return np.zeros(n, dtype=bool)
    v = pd.to_numeric(df["volume"], errors="coerce")
    bad = (v.isna() | (v <= 0)).astype(float)
    bad_frac = bad.rolling(window, min_periods=1).mean()
    varies = v.rolling(window, min_periods=2).std().fillna(0.0) > 0
    return ((bad_frac <= max_bad_fraction) & varies).to_numpy()


def vwap_anchor_keys(index: pd.DatetimeIndex, anchor: str = "globex") -> np.ndarray:
    """Group keys for anchored VWAP.

    ``globex``: resets at 18:00 ET (session open).
    ``rth``: resets at 09:30 ET and carries through the overnight session.
    """
    local = index.tz_convert(EXCHANGE_TZ_NAME).tz_localize(None)
    mins = local.hour * 60 + local.minute
    day = local.normalize()
    if anchor == "globex":
        key = np.where(mins >= 18 * 60, day + pd.Timedelta(days=1), day)
    elif anchor == "rth":
        key = np.where(mins >= 9 * 60 + 30, day, day - pd.Timedelta(days=1))
    else:
        raise ValueError(f"Unknown VWAP anchor {anchor!r}")
    return np.asarray(key)


def session_vwap(df: pd.DataFrame, anchor: str = "globex", volume_ok: bool | None = None) -> pd.Series:
    """Anchored VWAP using typical price. NaN when volume is unreliable."""
    if volume_ok is None:
        volume_ok, _ = volume_is_reliable(df)
    if not volume_ok or len(df) == 0:
        return pd.Series(np.nan, index=df.index, name="vwap")
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    keys = pd.Series(vwap_anchor_keys(df.index, anchor), index=df.index)
    if "contract" in df.columns:
        group = [keys, df["contract"].astype(str)]
    else:
        group = [keys]
    pv = (tp * vol).groupby(group).cumsum()
    cv = vol.groupby(group).cumsum()
    out = pv / cv.replace(0, np.nan)
    return out.rename("vwap")


def relative_volume(df: pd.DataFrame, lookback: int = 20, volume_ok: bool | None = None) -> pd.Series:
    """Volume divided by the mean of the *previous* ``lookback`` bars."""
    if volume_ok is None:
        volume_ok, _ = volume_is_reliable(df)
    if not volume_ok or len(df) == 0:
        return pd.Series(np.nan, index=df.index, name="rvol")
    v = pd.to_numeric(df["volume"], errors="coerce")
    base = v.shift(1).rolling(lookback, min_periods=lookback).mean()
    return (v / base.replace(0, np.nan)).rename("rvol")


def percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Trailing percentile rank (0..1) of each value within the last ``window`` values."""

    def _rank(x: np.ndarray) -> float:
        last = x[-1]
        if np.isnan(last):
            return np.nan
        valid = x[~np.isnan(x)]
        return float((valid <= last).mean())

    return series.rolling(window, min_periods=max(20, window // 5)).apply(_rank, raw=True)
