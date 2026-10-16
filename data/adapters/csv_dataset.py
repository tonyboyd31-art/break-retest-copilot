"""Adapter over a CSV dataset previously imported into SQLite."""

from __future__ import annotations

import pandas as pd

from data.adapters.base import AdapterStatus, AdapterUnavailable, MarketDataAdapter
from data.database import Database
from data.repositories import get_dataset, load_dataset_candles


class CsvDatasetAdapter(MarketDataAdapter):
    name = "csv"
    feed_type = "historical"

    def __init__(self, db: Database, dataset_id: int):
        self.db = db
        self.dataset_id = dataset_id
        self.meta = get_dataset(db, dataset_id)

    def status(self) -> AdapterStatus:
        if self.meta is None:
            return AdapterStatus(
                self.name, False, "historical", f"Dataset {self.dataset_id} not found.", [], True
            )
        return AdapterStatus(
            self.name,
            True,
            "historical",
            f"Imported CSV '{self.meta['name']}' ({self.meta['rows']} bars).",
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
        if self.meta is None:
            raise AdapterUnavailable(f"Dataset {self.dataset_id} not found")
        if self.meta["instrument"] != symbol:
            raise AdapterUnavailable(f"Dataset is for {self.meta['instrument']}, not {symbol}")
        df = load_dataset_candles(self.db, self.dataset_id)
        if contract and "contract" in df:
            df = df[df["contract"] == contract]
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index < end]
        return df
