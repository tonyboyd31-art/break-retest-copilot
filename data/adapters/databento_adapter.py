"""Databento historical OHLCV adapter - IMPLEMENTED BUT UNTESTED.

Databento is an authorised, documented vendor of CME Globex market data
(dataset ``GLBX.MDP3``). This adapter could not be exercised during
development because no credentials were available, so it is marked
``tested=False`` and the UI labels it accordingly.

Requirements:
    pip install databento
    export DATABENTO_API_KEY=...   (never commit it; see .env.example)

Behaviour:
* Requests 1-minute OHLCV bars for one *specific* contract (raw symbol such
  as ``NQZ6``) - never a continuous/spliced symbol - and aggregates them to
  5-minute bars aligned to the clock.
* Historical-API data may lag real time, so the feed is labelled
  ``delayed``. The freshness check in the UI decides whether it is usable.
* Read-only: this adapter has no order or account functionality.
"""

from __future__ import annotations

import os

import pandas as pd

from core.instruments import parse_contract_code
from data.adapters.base import AdapterStatus, AdapterUnavailable, MarketDataAdapter

DATASET = "GLBX.MDP3"


def to_databento_symbol(code: str) -> str:
    """'NQZ26' -> 'NQZ6' (CME raw symbols use a single-digit year)."""
    root, month, year = parse_contract_code(code)
    return f"{root}{month}{year % 10}"


class DatabentoAdapter(MarketDataAdapter):
    name = "databento"
    feed_type = "delayed"

    def __init__(self, api_key: str | None = None):
        self._key = api_key or os.environ.get("DATABENTO_API_KEY") or None

    def _client(self):  # pragma: no cover - requires network + credentials
        try:
            import databento as db  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AdapterUnavailable(
                "The 'databento' package is not installed (pip install databento)."
            ) from exc
        if not self._key:
            raise AdapterUnavailable("DATABENTO_API_KEY is not set.")
        return db.Historical(self._key)

    def status(self) -> AdapterStatus:
        try:
            import databento  # type: ignore[import-not-found]  # noqa: F401

            pkg = True
        except ImportError:
            pkg = False
        if not pkg:
            msg, ok = "Package 'databento' not installed.", False
        elif not self._key:
            msg, ok = "DATABENTO_API_KEY not set.", False
        else:
            msg, ok = (
                "Credentials present (key hidden). Adapter is UNTESTED - verify output before relying on it.",
                True,
            )
        return AdapterStatus(
            self.name, ok, "delayed", msg, ["DATABENTO_API_KEY", "pip install databento"], False
        )

    def get_candles(
        self,
        symbol: str,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        timeframe: str = "5min",
        contract: str | None = None,
    ) -> pd.DataFrame:  # pragma: no cover
        if contract is None:
            raise AdapterUnavailable(
                "A specific contract (e.g. NQZ26) is required; continuous symbols are not used."
            )
        if start is None or end is None:
            raise AdapterUnavailable("start and end are required for historical requests.")
        client = self._client()
        raw_symbol = to_databento_symbol(contract)
        try:
            store = client.timeseries.get_range(
                dataset=DATASET,
                schema="ohlcv-1m",
                symbols=[raw_symbol],
                stype_in="raw_symbol",
                start=start.tz_convert("UTC").isoformat(),
                end=end.tz_convert("UTC").isoformat(),
            )
            df1 = store.to_df()
        except Exception as exc:  # noqa: BLE001 - surface any provider error without leaking the key
            raise AdapterUnavailable(f"Databento request failed: {type(exc).__name__}: {exc}") from exc
        if df1.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "contract"])
        df1.index = pd.DatetimeIndex(df1.index).tz_convert("UTC")
        rule = timeframe
        agg = (
            df1.resample(rule, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["open"])
        )
        agg["contract"] = contract
        agg.index.name = "ts"
        return agg
