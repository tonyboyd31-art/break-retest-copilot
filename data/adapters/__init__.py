"""Market-data adapters (read-only). None of them can place orders."""

from data.adapters.base import AdapterStatus, AdapterUnavailable, MarketDataAdapter

__all__ = ["AdapterStatus", "AdapterUnavailable", "MarketDataAdapter"]
