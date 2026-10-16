"""Deterministic synthetic futures candles for demo mode and tests.

THIS IS SIMULATED DATA - NOT REAL MARKET PRICES. Base price levels are
arbitrary round numbers chosen only to give realistic tick/point scales;
they do not represent the actual market.

Model (per 5-minute bar, CME Globex hours only):

* intraday volatility profile (quiet overnight, active NY open / close);
* volatility clustering (AR(1) log-volatility);
* Markov regime switching between up-trend, down-trend and mean-reverting
  range, which naturally produces ranges, breakouts and pullbacks;
* fat-tailed (Student-t) returns, wicks drawn around the bar body;
* volume correlated with absolute return and time of day;
* contract months follow the instrument's roll rule; each earlier contract
  is offset by a fixed calendar spread so roll boundaries show a genuine
  price discontinuity (and are never spliced together by the engine).

Micro contracts share the path of their standard contract (same index).
Given the same arguments the output is byte-for-byte identical.
"""

from __future__ import annotations

import zlib
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from core.instruments import InstrumentSpec, front_contract, round_to_tick
from core.timeutils import EXCHANGE_TZ_NAME

ROOT_OF = {"MNQ": "NQ", "MES": "ES", "MGC": "GC"}
BASE_PRICE = {"NQ": 20000.0, "ES": 5600.0, "GC": 2400.0}  # arbitrary synthetic anchors
DAILY_VOL = {"NQ": 0.0115, "ES": 0.0085, "GC": 0.0090}
ROLL_SPREAD = {"NQ": 0.0035, "ES": 0.0030, "GC": 0.0075}  # fraction of price per contract step
VOLUME_BASE = {"NQ": 900.0, "MNQ": 4500.0, "ES": 2200.0, "MES": 6000.0, "GC": 450.0, "MGC": 900.0}

_PROFILE = [  # (start minute ET, end minute ET, volatility multiplier)
    (18 * 60, 20 * 60, 0.60),
    (20 * 60, 24 * 60, 0.45),
    (0, 3 * 60, 0.45),
    (3 * 60, 8 * 60, 0.75),
    (8 * 60, 9 * 60 + 30, 0.95),
    (9 * 60 + 30, 10 * 60 + 30, 2.00),
    (10 * 60 + 30, 12 * 60, 1.30),
    (12 * 60, 14 * 60, 0.90),
    (14 * 60, 15 * 60, 1.10),
    (15 * 60, 16 * 60, 1.40),
    (16 * 60, 17 * 60, 0.60),
]


def _profile_mult(minutes_of_day: np.ndarray) -> np.ndarray:
    out = np.ones(len(minutes_of_day))
    for start, end, m in _PROFILE:
        out[(minutes_of_day >= start) & (minutes_of_day < end)] = m
    return out


def trade_dates_back(end: date, days: int) -> list[date]:
    out: list[date] = []
    d = end
    while len(out) < days:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def session_bar_index(trade_dates: list[date], timeframe_minutes: int = 5) -> pd.DatetimeIndex:
    """UTC bar-open timestamps covering the Globex session of each trade date."""
    parts = []
    for td in trade_dates:
        start = pd.Timestamp(
            datetime.combine(td - timedelta(days=1), datetime.min.time()) + timedelta(hours=18)
        ).tz_localize(EXCHANGE_TZ_NAME)
        end = pd.Timestamp(datetime.combine(td, datetime.min.time()) + timedelta(hours=17)).tz_localize(
            EXCHANGE_TZ_NAME
        )
        rng = pd.date_range(start, end, freq=f"{timeframe_minutes}min", inclusive="left")
        parts.append(rng.tz_convert("UTC"))
    if not parts:
        return pd.DatetimeIndex([], tz="UTC")
    return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]


def default_end_date(today: date | None = None) -> date:
    d = today or date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def generate_synthetic_candles(
    spec: InstrumentSpec,
    end: date | None = None,
    days: int = 45,
    seed: int = 7,
    timeframe_minutes: int = 5,
) -> pd.DataFrame:
    """Generate simulated OHLCV candles indexed by UTC bar-open time."""
    root = ROOT_OF.get(spec.symbol, spec.symbol)
    end = end or default_end_date()
    tds = trade_dates_back(end, days)
    idx = session_bar_index(tds, timeframe_minutes)
    n = len(idx)
    rng = np.random.default_rng([seed, zlib.crc32(root.encode())])

    local = idx.tz_convert(EXCHANGE_TZ_NAME)
    mod = np.asarray(local.hour * 60 + local.minute)
    prof = _profile_mult(mod)
    prof = prof / np.sqrt(np.mean(prof**2))
    bars_per_day = 23 * 60 / timeframe_minutes
    base_sigma = DAILY_VOL.get(root, 0.01) / np.sqrt(bars_per_day)

    # log-volatility AR(1)
    phi = 0.985
    shocks = rng.standard_normal(n)
    lv = np.zeros(n)
    for i in range(1, n):
        lv[i] = phi * lv[i - 1] + 0.22 * np.sqrt(1 - phi**2) * shocks[i]
    vol_mult = np.exp(lv - 0.5 * np.var(lv))

    # regimes: 0 range, 1 up, -1 down
    regime = np.zeros(n, dtype=int)
    state = 0
    switch = rng.random(n)
    pick = rng.random(n)
    for i in range(n):
        if switch[i] < 1 / 240:
            state = 0 if pick[i] < 0.40 else (1 if pick[i] < 0.70 else -1)
        regime[i] = state

    t_shocks = rng.standard_t(5, n) / np.sqrt(5 / 3)
    wick_hi = np.abs(rng.standard_normal(n))
    wick_lo = np.abs(rng.standard_normal(n))
    vol_noise = rng.lognormal(0.0, 0.3, n)
    open_gap = rng.standard_normal(n)

    price = BASE_PRICE.get(root, 1000.0)
    center = price
    opens = np.empty(n)
    closes = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    rets = np.empty(n)
    prev_regime = regime[0]
    for i in range(n):
        sigma = base_sigma * prof[i] * vol_mult[i] * price
        if regime[i] != prev_regime or i == 0:
            center = price
        prev_regime = regime[i]
        if regime[i] == 0:
            drift = -0.03 * (price - center)
        else:
            drift = regime[i] * 0.09 * sigma
        o = price
        if mod[i] == 18 * 60:  # session open: small gap
            o = price + open_gap[i] * 1.5 * sigma
        r = drift + sigma * t_shocks[i]
        cl = o + r
        hi = max(o, cl) + wick_hi[i] * 0.55 * sigma
        lw = min(o, cl) - wick_lo[i] * 0.55 * sigma
        opens[i], closes[i], highs[i], lows[i], rets[i] = o, cl, hi, lw, abs(r) / max(sigma, 1e-9)
        price = cl

    tick = spec.tick_size
    dec = spec.price_decimals
    vround = np.vectorize(lambda x, m: round_to_tick(x, tick, m, dec), otypes=[float])

    # contract labels and calendar-spread offsets (latest contract = offset 0)
    tdate_of_bar = np.array([(t + timedelta(hours=6)).date() for t in local.tz_localize(None)])
    contracts = np.array([front_contract(spec, d).code for d in tdate_of_bar])
    uniq = list(dict.fromkeys(contracts))
    step = ROLL_SPREAD.get(root, 0.003) * BASE_PRICE.get(root, 1000.0)
    offset_by = {code: -(len(uniq) - 1 - k) * step for k, code in enumerate(uniq)}
    offs = np.array([offset_by[c] for c in contracts])

    o_r = vround(opens + offs, "nearest")
    c_r = vround(closes + offs, "nearest")
    h_r = np.maximum(vround(highs + offs, "up"), np.maximum(o_r, c_r))
    l_r = np.minimum(vround(lows + offs, "down"), np.minimum(o_r, c_r))

    vol_base = VOLUME_BASE.get(spec.symbol, 500.0)
    vprof = _profile_mult(mod)
    volume = np.maximum(1, np.round(vol_base * vprof * (0.6 + 0.8 * rets) * vol_noise)).astype(np.int64)

    df = pd.DataFrame(
        {
            "open": o_r,
            "high": h_r,
            "low": l_r,
            "close": c_r,
            "volume": volume.astype(float),
            "contract": contracts,
        },
        index=idx,
    )
    df.index.name = "ts"
    df.attrs.update(
        {
            "source": "synthetic",
            "feed_type": "simulated",
            "seed": seed,
            "symbol": spec.symbol,
            "note": "SIMULATED DATA - not real market prices",
        }
    )
    return df
