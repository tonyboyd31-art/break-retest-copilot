"""Synthetic (simulated) data adapter for demo mode. Requires no credentials."""

from __future__ import annotations

from datetime import date

import pandas as pd

from core.instruments import InstrumentRegistry, default_registry
from data.adapters.base import AdapterStatus, MarketDataAdapter
from data.synthetic_data import generate_synthetic_candles


class SyntheticAdapter(MarketDataAdapter):
    name = "synthetic"
    feed_type = "simulated"

    def __init__(
        self,
        days: int = 45,
        seed: int = 7,
        end: date | None = None,
        registry: InstrumentRegistry | None = None,
    ):
        self.days = days
        self.seed = seed
        self.end = end
        self.registry = registry or default_registry()

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            self.name,
            True,
            "simulated",
            "Deterministic simulated candles - NOT real market prices.",
            [],
            True,
        )

    def get_candles(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        timeframe: str = "5min",
        contract: str | None = None,
    ) -> pd.DataFrame:
        if timeframe != "5min":
            raise ValueError("Synthetic adapter produces 5-minute bars only")
        df = generate_synthetic_candles(
            self.registry.get(symbol), end=self.end, days=self.days, seed=self.seed
        )
        if contract:
            df = df[df["contract"] == contract]
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index < end]
        return df
