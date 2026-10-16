"""Validated input models for user-entered records.

Everything a user types (manual trades, management actions, accounts,
daily P&L, payouts) passes through these models before it is stored.
The models never infer that a suggested action was performed: an action
exists only if the user records it.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.instruments import InstrumentSpec, is_on_tick

ACTION_TYPES = {
    "stop_to_breakeven": "Stop moved to breakeven",
    "stop_tightened": "Stop tightened",
    "stop_widened": "Stop widened (recorded for honesty)",
    "partial_exit": "Partial profit taken",
    "full_exit": "Position manually exited",
    "thesis_invalidated": "Trade thesis invalidated (noted)",
    "other": "Other management decision",
}
STOP_ACTIONS = {"stop_to_breakeven", "stop_tightened", "stop_widened"}


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


def _aware(ts: object) -> pd.Timestamp:
    t = pd.Timestamp(ts)  # type: ignore[arg-type]
    if t.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return t.tz_convert("UTC")


class ManualTradeInput(_M):
    account_id: int | None = None
    firm_code: str | None = None
    instrument: str
    contract: str | None = None
    direction: Literal["long", "short"]
    entry_ts: pd.Timestamp
    entry_price: float = Field(gt=0)
    initial_stop: float = Field(gt=0)
    quantity: int = Field(ge=1, le=1000)
    planned_target: float | None = Field(default=None, gt=0)
    setup_id: str | None = None
    commission_per_contract_rt: float = Field(default=0.0, ge=0)
    slippage_ticks: float = Field(default=0.0, ge=0)
    notes: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("entry_ts", mode="before")
    @classmethod
    def _ts(cls, v: object) -> pd.Timestamp:
        return _aware(v)

    @model_validator(mode="after")
    def _risk(self) -> ManualTradeInput:
        if self.direction == "long":
            if self.initial_stop >= self.entry_price:
                raise ValueError(
                    "For a long trade the initial stop must be below the entry (risk must be positive)."
                )
            if self.planned_target is not None and self.planned_target <= self.entry_price:
                raise ValueError("For a long trade the planned target must be above the entry.")
        else:
            if self.initial_stop <= self.entry_price:
                raise ValueError(
                    "For a short trade the initial stop must be above the entry (risk must be positive)."
                )
            if self.planned_target is not None and self.planned_target >= self.entry_price:
                raise ValueError("For a short trade the planned target must be below the entry.")
        return self

    def check_against(self, spec: InstrumentSpec) -> list[str]:
        """Instrument-specific validation errors (tick alignment, quantity increments)."""
        errors = []
        for name in ("entry_price", "initial_stop", "planned_target"):
            v = getattr(self, name)
            if v is not None and not is_on_tick(v, spec.tick_size):
                errors.append(
                    f"{name.replace('_', ' ')} {v} is not a multiple of the {spec.symbol} tick size "
                    f"{spec.tick_size}."
                )
        if self.quantity < spec.min_qty or self.quantity % spec.qty_increment:
            errors.append(f"Quantity must be a multiple of {spec.qty_increment} (min {spec.min_qty}).")
        return errors


class TradeActionInput(_M):
    trade_id: int
    action_type: Literal[
        "stop_to_breakeven",
        "stop_tightened",
        "stop_widened",
        "partial_exit",
        "full_exit",
        "thesis_invalidated",
        "other",
    ]
    action_ts: pd.Timestamp
    price: float | None = Field(default=None, gt=0)
    quantity: int | None = Field(default=None, ge=1)
    new_stop: float | None = Field(default=None, gt=0)
    notes: str = ""
    alert_id: int | None = None

    @field_validator("action_ts", mode="before")
    @classmethod
    def _ts(cls, v: object) -> pd.Timestamp:
        return _aware(v)

    @model_validator(mode="after")
    def _fields(self) -> TradeActionInput:
        if self.action_type in STOP_ACTIONS and self.new_stop is None:
            raise ValueError("A stop adjustment requires the new stop price you set at your broker.")
        if self.action_type == "partial_exit" and (self.price is None or self.quantity is None):
            raise ValueError("A partial exit requires the fill price and quantity.")
        if self.action_type == "full_exit" and self.price is None:
            raise ValueError("A full exit requires the exit fill price.")
        if self.action_type in ("thesis_invalidated", "other") and not self.notes.strip():
            raise ValueError("Please describe the decision in the notes.")
        return self


class AccountInput(_M):
    firm_code: str
    account_type_code: str
    nickname: str = Field(min_length=1, max_length=60)
    stage: Literal["evaluation", "funded", "live", "practice"]
    account_size: float = Field(gt=0)
    starting_balance: float = Field(ge=0)
    current_balance: float | None = None
    personal_daily_loss_limit: float | None = Field(default=None, gt=0)
    personal_max_contracts: float | None = Field(default=None, gt=0)
    cycle_start_date: date | None = None
    notes: str = ""


class DailyPnlInput(_M):
    account_id: int
    trade_date: date
    net_pnl: float
    min_equity: float | None = None
    max_equity: float | None = None
    source: Literal["user_entered", "journal", "verified"] = "user_entered"
    notes: str = ""

    @model_validator(mode="after")
    def _range(self) -> DailyPnlInput:
        if self.min_equity is not None and self.max_equity is not None and self.min_equity > self.max_equity:
            raise ValueError("Intraday minimum equity cannot exceed the intraday maximum.")
        return self


class PayoutInput(_M):
    account_id: int
    requested_on: date
    amount: float = Field(gt=0)
    status: Literal["requested", "paid", "denied", "cancelled"] = "requested"
    notes: str = ""
