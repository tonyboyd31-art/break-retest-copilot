"""Deterministic swing-high / swing-low detection without look-ahead.

Definition (swing high at bar ``i`` with ``left``/``right`` confirmation
bars):

* ``high[i]`` is strictly greater than every high in the ``left`` bars
  before it, and
* ``high[i]`` is greater than or equal to every high in the ``right`` bars
  after it (ties resolve to the *first* bar, which suppresses equal-high
  duplicates).

A pivot needs ``right`` later bars, so it is **not known** until the close
of bar ``i + right``. That moment is stored as ``confirm_ts`` and every
consumer (zones, trailing stops, backtests) must only use swings with
``confirm_ts <= now``. The batch implementation below is equivalent to
running the rule bar-by-bar; ``tests/test_swings.py`` verifies that
truncating the data never changes previously confirmed swings.

Prominence of a swing high = ``high[i] - max(min(low[left window]),
min(low[right window]))`` - i.e. the shallower of the two pullbacks around
the pivot. Swings below the minimum prominence (ticks, optionally an ATR
multiple measured at confirmation time) are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from core.timeutils import timeframe_delta

SWING_COLUMNS = [
    "swing_id",
    "swing_type",
    "price",
    "pivot_idx",
    "pivot_ts",
    "confirm_idx",
    "confirm_ts",
    "prominence",
    "threshold",
    "status",
    "body_top",
    "body_bottom",
]


@dataclass(frozen=True)
class SwingParams:
    left: int = 3
    right: int = 3
    min_prominence_ticks: float = 8
    atr_mult: float | None = 0.5
    duplicate_window_bars: int = 6
    duplicate_tolerance_ticks: float = 4


def _window_extreme(values: np.ndarray, width: int, fn) -> np.ndarray:
    """fn over windows ending at each index (NaN where window incomplete)."""
    n = len(values)
    out = np.full(n, np.nan)
    if width <= 0 or n < width:
        return out
    win = sliding_window_view(values, width)
    out[width - 1 :] = fn(win, axis=1)
    return out


def detect_swings(
    df: pd.DataFrame,
    tick_size: float,
    params: SwingParams,
    atr: pd.Series | None = None,
    timeframe: str = "5min",
) -> pd.DataFrame:
    """Return confirmed (and suppressed) swing points for a single-contract frame."""
    n = len(df)
    left, right = params.left, params.right
    if n < left + right + 1:
        return pd.DataFrame(columns=SWING_COLUMNS)

    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    atr_v = atr.to_numpy(dtype=float) if atr is not None else np.full(n, np.nan)
    tf = timeframe_delta(timeframe)
    idx = df.index

    # Window stats: *_end[k] = stat over values[k-w+1 .. k]
    hmax_l = _window_extreme(h, left, np.max)
    lmin_l = _window_extreme(lo, left, np.min)
    hmax_r = _window_extreme(h, right, np.max)
    lmin_r = _window_extreme(lo, right, np.min)

    candidates = np.arange(left, n - right)
    rows: list[dict] = []

    # left window of i is [i-left, i-1] -> stat ending at i-1
    # right window of i is [i+1, i+right] -> stat ending at i+right
    lw_end = candidates - 1
    rw_end = candidates + right

    is_high = (h[candidates] > hmax_l[lw_end]) & (h[candidates] >= hmax_r[rw_end])
    is_low = (lo[candidates] < lmin_l[lw_end]) & (lo[candidates] <= lmin_r[rw_end])

    for kind, mask in (("high", is_high), ("low", is_low)):
        for i in candidates[mask]:
            conf = i + right
            if kind == "high":
                price = h[i]
                prominence = price - max(lmin_l[i - 1], lmin_r[conf])
            else:
                price = lo[i]
                prominence = min(hmax_l[i - 1], hmax_r[conf]) - price
            thr = params.min_prominence_ticks * tick_size
            if params.atr_mult is not None and params.atr_mult > 0 and not np.isnan(atr_v[conf]):
                thr = max(thr, params.atr_mult * atr_v[conf])
            rows.append(
                {
                    "swing_type": kind,
                    "price": float(price),
                    "pivot_idx": int(i),
                    "confirm_idx": int(conf),
                    "prominence": float(prominence),
                    "threshold": float(thr),
                    "body_top": float(max(o[i], c[i])),
                    "body_bottom": float(min(o[i], c[i])),
                }
            )

    if not rows:
        return pd.DataFrame(columns=SWING_COLUMNS)

    sw = pd.DataFrame(rows)
    sw = sw[sw["prominence"] >= sw["threshold"] - 1e-9]
    sw = sw.sort_values(["confirm_idx", "pivot_idx", "swing_type"]).reset_index(drop=True)

    # Sequential (causal) duplicate suppression: a new swing is suppressed if
    # an already-accepted swing of the same type is within the window and
    # price tolerance.
    status = []
    accepted: dict[str, list[tuple[int, float]]] = {"high": [], "low": []}
    tol = params.duplicate_tolerance_ticks * tick_size
    for row in sw.itertuples(index=False):
        dup = False
        if params.duplicate_window_bars > 0:
            for p_idx, p_price in reversed(accepted[row.swing_type]):
                if row.pivot_idx - p_idx > params.duplicate_window_bars:
                    break
                if abs(row.price - p_price) <= tol:
                    dup = True
                    break
        if dup:
            status.append("suppressed")
        else:
            status.append("confirmed")
            accepted[row.swing_type].append((row.pivot_idx, row.price))
    sw["status"] = status
    sw["pivot_ts"] = idx[sw["pivot_idx"].to_numpy()]
    sw["confirm_ts"] = idx[sw["confirm_idx"].to_numpy()] + tf
    sw["swing_id"] = [f"{t}:{ts.isoformat()}" for t, ts in zip(sw["swing_type"], sw["pivot_ts"], strict=True)]
    return sw[SWING_COLUMNS]


def swings_known_at(swings: pd.DataFrame, ts: pd.Timestamp, include_suppressed: bool = False) -> pd.DataFrame:
    """Swings whose confirmation time is at or before ``ts``."""
    if swings.empty:
        return swings
    mask = swings["confirm_ts"] <= ts
    if not include_suppressed:
        mask &= swings["status"] == "confirmed"
    return swings[mask]


def provisional_swings(df: pd.DataFrame, params: SwingParams) -> pd.DataFrame:
    """Unconfirmed pivot candidates at the right edge of the data.

    These satisfy the left condition and every right bar available so far,
    but fewer than ``right`` bars have completed. They are display-only and
    must never trigger a confirmed alert.
    """
    n = len(df)
    out = []
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    for i in range(max(params.left, n - params.right), n):
        lw = slice(i - params.left, i)
        rw = slice(i + 1, n)
        if h[i] > np.max(h[lw]) and (i + 1 >= n or h[i] >= np.max(h[rw])):
            out.append(
                {
                    "swing_type": "high",
                    "price": float(h[i]),
                    "pivot_idx": i,
                    "pivot_ts": df.index[i],
                    "bars_remaining": params.right - (n - 1 - i),
                    "status": "provisional",
                }
            )
        if lo[i] < np.min(lo[lw]) and (i + 1 >= n or lo[i] <= np.min(lo[rw])):
            out.append(
                {
                    "swing_type": "low",
                    "price": float(lo[i]),
                    "pivot_idx": i,
                    "pivot_ts": df.index[i],
                    "bars_remaining": params.right - (n - 1 - i),
                    "status": "provisional",
                }
            )
    return pd.DataFrame(out)


def params_from_settings(s) -> SwingParams:  # s: core.config.SwingSettings
    return SwingParams(
        left=s.left_bars,
        right=s.right_bars,
        min_prominence_ticks=s.min_prominence_ticks,
        atr_mult=s.atr_prominence_mult if s.use_atr_prominence else None,
        duplicate_window_bars=s.duplicate_window_bars,
        duplicate_tolerance_ticks=s.duplicate_tolerance_ticks,
    )
