"""Futures instrument metadata registry, tick math and contract calendar.

No calculation in the application may assume a tick value or multiplier;
everything is looked up here. The registry is loaded from
``configurations/instruments.json`` and every entry is validated
(``tick_value == tick_size * point_value`` etc.) before it can be used.

Contract-calendar functions (expiration, roll date, front month) implement
the published *rules* (third Friday; third-last business day) but do not
know exchange holidays. They are labelled as calculated estimates in the UI.
"""

from __future__ import annotations

import json
import math
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MONTH_CODES = "FGHJKMNQUVXZ"  # Jan..Dec
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configurations" / "instruments.json"

_CONTRACT_RE = re.compile(r"^(?P<root>[A-Z0-9]{1,4}?)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,4})$")


class InstrumentSpec(BaseModel):
    symbol: str
    name: str
    exchange: str
    asset_class: str
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    point_value: float = Field(gt=0, description="Contract multiplier: USD per 1.0 price point")
    currency: str = "USD"
    contract_months: str
    expiration_rule: Literal["third_friday", "third_last_business_day"]
    roll_rule: Literal["calendar_days_before_expiry", "business_days_before_first_notice"]
    roll_offset_days: int = Field(ge=0, le=40)
    session: str
    min_qty: int = Field(default=1, ge=1)
    qty_increment: int = Field(default=1, ge=1)
    mini_equivalent: float = Field(gt=0, le=1.0, description="1.0 for a standard contract, 0.1 for a micro")
    related_symbol: str | None = None
    price_decimals: int = Field(default=2, ge=0, le=6)
    source_url: str
    metadata_status: Literal["build_reference", "user_verified", "unverified"] = "unverified"
    metadata_verified_on: date | None = None
    enabled: bool = True

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{1,6}", v):
            raise ValueError(f"Invalid symbol {v!r}")
        return v

    @field_validator("contract_months")
    @classmethod
    def _months(cls, v: str) -> str:
        v = v.strip().upper()
        if not v or any(c not in MONTH_CODES for c in v):
            raise ValueError(f"Invalid contract month codes {v!r}")
        return "".join(sorted(set(v), key=MONTH_CODES.index))

    @model_validator(mode="after")
    def _consistent(self) -> InstrumentSpec:
        expected = self.tick_size * self.point_value
        if not math.isclose(expected, self.tick_value, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"{self.symbol}: tick_value {self.tick_value} != tick_size {self.tick_size} x "
                f"point_value {self.point_value} (= {expected})"
            )
        return self

    # -- tick math ---------------------------------------------------------
    @property
    def ticks_per_point(self) -> float:
        return 1.0 / self.tick_size

    def round_price(self, price: float, mode: Literal["nearest", "down", "up"] = "nearest") -> float:
        return round_to_tick(price, self.tick_size, mode, self.price_decimals)

    def ticks(self, price_distance: float) -> float:
        return price_to_ticks(price_distance, self.tick_size)

    def dollars(self, price_distance: float, qty: float = 1) -> float:
        return price_distance * self.point_value * qty

    def fmt(self, price: float | None) -> str:
        if price is None or (isinstance(price, float) and math.isnan(price)):
            return "-"
        return f"{price:,.{self.price_decimals}f}"


def round_to_tick(price: float, tick: float, mode: str = "nearest", decimals: int = 6) -> float:
    """Round ``price`` to a multiple of ``tick`` robustly against float noise."""
    if tick <= 0:
        raise ValueError("tick must be positive")
    n = price / tick
    eps = 1e-7
    if mode == "nearest":
        k = math.floor(n + 0.5)
    elif mode == "down":
        k = math.floor(n + eps)
    elif mode == "up":
        k = math.ceil(n - eps)
    else:
        raise ValueError(f"Unknown rounding mode {mode!r}")
    return round(k * tick, max(decimals, 0) + 2)


def price_to_ticks(distance: float, tick: float) -> float:
    """Convert a price distance to ticks (rounded to remove float noise)."""
    if tick <= 0:
        raise ValueError("tick must be positive")
    return round(distance / tick, 6)


def is_on_tick(price: float, tick: float, tol: float = 1e-6) -> bool:
    n = price / tick
    return abs(n - round(n)) <= tol


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class InstrumentRegistry:
    def __init__(self, specs: list[InstrumentSpec]):
        self._specs = {s.symbol: s for s in specs}

    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> InstrumentRegistry:
        raw = json.loads(Path(path).read_text())
        specs = [InstrumentSpec.model_validate(item) for item in raw["instruments"]]
        return cls(specs)

    def get(self, symbol: str) -> InstrumentSpec:
        sym = symbol.strip().upper()
        if sym not in self._specs:
            raise KeyError(f"Unknown instrument {symbol!r}; configure it in configurations/instruments.json")
        return self._specs[sym]

    def symbols(self, enabled_only: bool = True) -> list[str]:
        return [s for s, spec in self._specs.items() if spec.enabled or not enabled_only]

    def all(self) -> list[InstrumentSpec]:
        return list(self._specs.values())

    def with_override(self, spec: InstrumentSpec) -> InstrumentRegistry:
        specs = dict(self._specs)
        specs[spec.symbol] = spec
        return InstrumentRegistry(list(specs.values()))


_DEFAULT_REGISTRY: InstrumentRegistry | None = None


def default_registry() -> InstrumentRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = InstrumentRegistry.load()
    return _DEFAULT_REGISTRY


def get_instrument(symbol: str) -> InstrumentSpec:
    return default_registry().get(symbol)


# ---------------------------------------------------------------------------
# Contract calendar
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractInfo:
    root: str
    month_code: str
    year: int
    code: str  # e.g. "NQZ26"
    expiration: date  # calculated, holidays ignored
    roll_date: date  # calculated, holidays ignored
    calculation_note: str


def _business_days_back(d: date, n: int) -> date:
    """Step back ``n`` weekdays from ``d`` (exchange holidays ignored)."""
    out = d
    while n > 0:
        out -= timedelta(days=1)
        if out.weekday() < 5:
            n -= 1
    return out


def third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    offset = (4 - first.weekday()) % 7
    return first + timedelta(days=offset + 14)


def third_last_business_day(year: int, month: int) -> date:
    d = date(year, month, monthrange(year, month)[1])
    count = 0
    while True:
        if d.weekday() < 5:
            count += 1
            if count == 3:
                return d
        d -= timedelta(days=1)


def last_business_day(year: int, month: int) -> date:
    d = date(year, month, monthrange(year, month)[1])
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def contract_code(root: str, month_code: str, year: int) -> str:
    return f"{root}{month_code}{year % 100:02d}"


def parse_contract_code(code: str) -> tuple[str, str, int]:
    """Parse 'NQZ6', 'NQZ26' or 'NQZ2026' -> ('NQ', 'Z', 2026).

    Single-digit years are resolved to the nearest decade from 2020.
    """
    m = _CONTRACT_RE.match(code.strip().upper())
    if not m:
        raise ValueError(f"Unrecognised contract code {code!r}")
    root, month, ys = m.group("root"), m.group("month"), m.group("year")
    y = int(ys)
    if len(ys) == 1:
        y = 2020 + y if 2020 + y >= 2024 else 2030 + y
    elif len(ys) == 2:
        y = 2000 + y
    return root, month, y


def contract_info(spec: InstrumentSpec, month_code: str, year: int) -> ContractInfo:
    month = MONTH_CODES.index(month_code) + 1
    if spec.expiration_rule == "third_friday":
        exp = third_friday(year, month)
        exp_note = "third Friday of the contract month"
    else:
        exp = third_last_business_day(year, month)
        exp_note = "third-last business day of the contract month"
    if spec.roll_rule == "calendar_days_before_expiry":
        roll = exp - timedelta(days=spec.roll_offset_days)
        roll_note = f"{spec.roll_offset_days} calendar days before expiration"
    else:
        prev_year, prev_month = (year, month - 1) if month > 1 else (year - 1, 12)
        first_notice = last_business_day(prev_year, prev_month)
        roll = _business_days_back(first_notice, spec.roll_offset_days)
        roll_note = (
            f"{spec.roll_offset_days} business days before first notice day "
            f"(estimated as last business day of the prior month, {first_notice.isoformat()})"
        )
    note = (
        f"Expiration = {exp_note}; roll = {roll_note}. Calculated by rule; exchange holidays "
        f"are not modelled - confirm on the exchange calendar."
    )
    return ContractInfo(
        spec.symbol, month_code, year, contract_code(spec.symbol, month_code, year), exp, roll, note
    )


def listed_contracts(spec: InstrumentSpec, start: date, count: int = 6) -> list[ContractInfo]:
    """Upcoming contracts (by the instrument's active month cycle) from ``start``."""
    out: list[ContractInfo] = []
    year = start.year
    while len(out) < count:
        for code in spec.contract_months:
            month = MONTH_CODES.index(code) + 1
            if date(year, month, monthrange(year, month)[1]) < start:
                continue
            info = contract_info(spec, code, year)
            if info.expiration >= start:
                out.append(info)
            if len(out) >= count:
                break
        year += 1
    return out


def front_contract(spec: InstrumentSpec, as_of: date) -> ContractInfo:
    """Most liquid contract by the conventional roll rule at ``as_of``."""
    for info in listed_contracts(spec, as_of - timedelta(days=120), count=12):
        if as_of < info.roll_date:
            return info
    raise RuntimeError(f"No front contract found for {spec.symbol} at {as_of}")
