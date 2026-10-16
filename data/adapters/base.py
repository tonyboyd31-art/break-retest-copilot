"""Market-data adapter interface.

An adapter returns completed OHLCV candles indexed by UTC bar-open time with
columns ``open, high, low, close, volume, contract``. Adapters are strictly
read-only: the interface has no order, position or account methods, and no
adapter may be extended with them.

To add a provider, subclass :class:`MarketDataAdapter`, implement
``status()`` and ``get_candles()``, and register it in
``services.market_service.build_adapter``. If the provider needs
credentials, read them from environment variables and never log them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from core.data_quality import FeedType


class AdapterUnavailable(RuntimeError):
    """Raised when an adapter cannot serve data (missing credentials, package or network)."""


@dataclass
class AdapterStatus:
    name: str
    available: bool
    feed_type: FeedType
    message: str
    requires: list[str]
    tested: bool  # whether this adapter was exercised end-to-end during development


class MarketDataAdapter(ABC):
    name: str = "base"
    feed_type: FeedType = "historical"

    @abstractmethod
    def status(self) -> AdapterStatus: ...

    @abstractmethod
    def get_candles(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        timeframe: str = "5min",
        contract: str | None = None,
    ) -> pd.DataFrame: ...
