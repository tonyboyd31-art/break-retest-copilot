"""Shared deterministic fixtures."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from core.config import AppSettings
from core.instruments import get_instrument
from data.database import Database
from data.synthetic_data import generate_synthetic_candles

# Tuesday 2026-03-10 10:00 ET (14:00 UTC, EDT) - well inside the RTH session.
T0 = pd.Timestamp("2026-03-10 14:00", tz="UTC")


def make_bars(
    rows: list[tuple[float, float, float, float]],
    start: pd.Timestamp = T0,
    volume: float | None = 1000.0,
    contract: str | None = "NQM26",
    freq: str = "5min",
) -> pd.DataFrame:
    """Build a 5-minute OHLC frame from (open, high, low, close) tuples."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    if volume is not None:
        rng = np.random.default_rng(1)
        df["volume"] = volume + rng.integers(0, 200, len(rows))
    if contract is not None:
        df["contract"] = contract
    df.index.name = "ts"
    return df


def flat_bars(n: int, price: float = 100.0, width: float = 1.0, **kw) -> pd.DataFrame:
    rows = [(price, price + width / 2, price - width / 2, price) for _ in range(n)]
    return make_bars(rows, **kw)


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings()


@pytest.fixture
def nq():
    return get_instrument("NQ")


@pytest.fixture
def mnq():
    return get_instrument("MNQ")


@pytest.fixture
def gc():
    return get_instrument("GC")


@pytest.fixture(scope="session")
def synthetic_nq() -> pd.DataFrame:
    """Latest single-contract segment of the deterministic demo series."""
    df = generate_synthetic_candles(get_instrument("NQ"), end=date(2026, 10, 16), days=60, seed=123)
    last = df["contract"].iloc[-1]
    return df[df["contract"] == last]


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(tmp_path / "test.db")
    d.migrate()
    return d
