"""Drawdown models (firm-neutral, separately tested).

Each model replays daily records (user-entered or journal-derived) and
reports the floor, remaining allowance and any detectable breach.

* :class:`StaticDrawdown` - floor fixed at ``start - max_loss``.
* :class:`EodTrailingDrawdown` - floor = highest END-OF-DAY balance minus
  ``max_loss``; updated only at the close; optionally locks at
  ``start + lock_floor_offset`` and/or when a payout is recorded.
* :class:`IntradayTrailingDrawdown` - floor trails the highest INTRADAY
  equity (including unrealized P&L). Requires intraday high-equity data;
  without it the result is flagged as an estimate that may OVERSTATE the
  remaining allowance.

Breach detection: firms typically enforce the floor in real time. A breach
is only *known* if the user provided the day's intraday low equity (or the
end-of-day balance itself was at/below the floor). Otherwise the result says
an intraday breach cannot be ruled out.

These models do not know any firm's rules by themselves; a firm module
selects and parameterises one, and only after its applicability has been
confirmed should results be treated as more than an estimate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date


@dataclass
class DailyRecord:
    trade_date: date
    net_pnl: float
    min_equity: float | None = None  # lowest intraday equity (balance incl. open P&L), if known
    max_equity: float | None = None  # highest intraday equity, if known
    payout: float = 0.0  # withdrawal processed at the end of this day


@dataclass
class DrawdownDay:
    trade_date: date
    start_balance: float
    eod_balance: float  # after payouts
    floor_during_day: float
    floor_after: float
    allowance_after: float
    breached: bool | None  # None = unknown (no intraday data)
    breach_basis: str
    locked: bool
    note: str = ""


@dataclass
class DrawdownResult:
    model: str
    starting_balance: float
    max_loss: float
    days: list[DrawdownDay]
    current_balance: float
    current_floor: float
    remaining_allowance: float
    breached: bool
    breach_date: date | None
    breach_unknown_days: int
    locked: bool
    estimate: bool
    assumptions: list[str] = field(default_factory=list)


class DrawdownModel(ABC):
    name: str = "base"

    def __init__(self, max_loss: float):
        if max_loss is None or max_loss <= 0:
            raise ValueError("max_loss must be a positive amount")
        self.max_loss = float(max_loss)

    @abstractmethod
    def run(self, starting_balance: float, records: list[DailyRecord]) -> DrawdownResult: ...

    @staticmethod
    def _breach(
        floor: float, start_bal: float, eod_pre_payout: float, rec: DailyRecord
    ) -> tuple[bool | None, str]:
        if rec.min_equity is not None:
            return rec.min_equity <= floor, "reported intraday low equity"
        if eod_pre_payout <= floor or start_bal <= floor:
            return True, "end-of-day balance"
        return None, "intraday low not provided - an intraday breach cannot be ruled out"


class StaticDrawdown(DrawdownModel):
    name = "static"

    def run(self, starting_balance: float, records: list[DailyRecord]) -> DrawdownResult:
        floor = starting_balance - self.max_loss
        bal = starting_balance
        days, breach_date, unknown = [], None, 0
        for r in sorted(records, key=lambda x: x.trade_date):
            start = bal
            eod_pre = bal + r.net_pnl
            breached, basis = self._breach(floor, start, eod_pre, r)
            bal = eod_pre - r.payout
            unknown += breached is None
            if breached and breach_date is None:
                breach_date = r.trade_date
            days.append(
                DrawdownDay(r.trade_date, start, bal, floor, floor, bal - floor, breached, basis, True)
            )
        return DrawdownResult(
            self.name,
            starting_balance,
            self.max_loss,
            days,
            bal,
            floor,
            bal - floor,
            breach_date is not None,
            breach_date,
            unknown,
            True,
            False,
            ["Static floor: starting balance minus maximum loss; never moves."],
        )


class EodTrailingDrawdown(DrawdownModel):
    name = "eod_trailing"

    def __init__(self, max_loss: float, lock_floor_offset: float | None = None, lock_on_payout: bool = False):
        super().__init__(max_loss)
        self.lock_floor_offset = lock_floor_offset
        self.lock_on_payout = lock_on_payout

    def run(self, starting_balance: float, records: list[DailyRecord]) -> DrawdownResult:
        lock_level = None if self.lock_floor_offset is None else starting_balance + self.lock_floor_offset
        floor = starting_balance - self.max_loss
        peak = starting_balance
        bal = starting_balance
        locked = False
        days, breach_date, unknown = [], None, 0
        assumptions = [
            "Floor = highest end-of-day balance minus the maximum loss, updated only at the close."
        ]
        if lock_level is not None:
            assumptions.append(
                f"Trailing stops once the floor reaches {lock_level:,.2f} (start + "
                f"{self.lock_floor_offset:,.2f})."
            )
        else:
            assumptions.append("No lock level configured: the floor keeps trailing (conservative).")
        if self.lock_on_payout:
            assumptions.append("Recording a payout locks the floor at the lock level.")
        assumptions.append("Withdrawals reduce the balance; they do not lower the floor.")
        for r in sorted(records, key=lambda x: x.trade_date):
            start = bal
            floor_day = floor
            eod_pre = bal + r.net_pnl
            breached, basis = self._breach(floor_day, start, eod_pre, r)
            unknown += breached is None
            if breached and breach_date is None:
                breach_date = r.trade_date
            note = ""
            if not locked:
                peak = max(peak, eod_pre)
                new_floor = peak - self.max_loss
                if lock_level is not None and new_floor >= lock_level:
                    new_floor = lock_level
                    locked = True
                    note = "floor locked"
                floor = max(floor, new_floor)
            if r.payout > 0 and self.lock_on_payout and not locked:
                floor = max(floor, lock_level if lock_level is not None else starting_balance)
                locked = True
                note = "floor locked by payout"
            bal = eod_pre - r.payout
            days.append(
                DrawdownDay(
                    r.trade_date, start, bal, floor_day, floor, bal - floor, breached, basis, locked, note
                )
            )
        return DrawdownResult(
            self.name,
            starting_balance,
            self.max_loss,
            days,
            bal,
            floor,
            bal - floor,
            breach_date is not None,
            breach_date,
            unknown,
            locked,
            False,
            assumptions,
        )


class IntradayTrailingDrawdown(DrawdownModel):
    name = "intraday_trailing"

    def __init__(self, max_loss: float, lock_floor_offset: float | None = None, lock_on_payout: bool = False):
        super().__init__(max_loss)
        self.lock_floor_offset = lock_floor_offset
        self.lock_on_payout = lock_on_payout

    def run(self, starting_balance: float, records: list[DailyRecord]) -> DrawdownResult:
        lock_level = None if self.lock_floor_offset is None else starting_balance + self.lock_floor_offset
        floor = starting_balance - self.max_loss
        peak = starting_balance
        bal = starting_balance
        locked = False
        estimate = False
        days, breach_date, unknown = [], None, 0
        assumptions = ["Floor = highest intraday equity (incl. unrealized P&L) minus the maximum loss."]
        for r in sorted(records, key=lambda x: x.trade_date):
            start = bal
            eod_pre = bal + r.net_pnl
            hi = r.max_equity
            if hi is None:
                hi = max(start, eod_pre)
                estimate = True
            # Worst case within the day: the intraday high happened before the low.
            day_floor = floor
            if not locked:
                day_peak = max(peak, hi)
                day_floor = max(floor, day_peak - self.max_loss)
                if lock_level is not None and day_floor >= lock_level:
                    day_floor = lock_level
            breached, basis = self._breach(day_floor, start, eod_pre, r)
            unknown += breached is None
            if breached and breach_date is None:
                breach_date = r.trade_date
            note = ""
            if not locked:
                peak = max(peak, hi)
                floor = max(floor, peak - self.max_loss)
                if lock_level is not None and floor >= lock_level:
                    floor = lock_level
                    locked = True
                    note = "floor locked"
            if r.payout > 0 and self.lock_on_payout and not locked:
                floor = max(floor, lock_level if lock_level is not None else starting_balance)
                locked = True
                note = "floor locked by payout"
            bal = eod_pre - r.payout
            days.append(
                DrawdownDay(
                    r.trade_date, start, bal, day_floor, floor, bal - floor, breached, basis, locked, note
                )
            )
        if estimate:
            assumptions.append(
                "ESTIMATE: intraday equity highs were not provided for some days, so the floor may be "
                "UNDERSTATED and the remaining allowance OVERSTATED."
            )
        return DrawdownResult(
            self.name,
            starting_balance,
            self.max_loss,
            days,
            bal,
            floor,
            bal - floor,
            breach_date is not None,
            breach_date,
            unknown,
            locked,
            estimate,
            assumptions,
        )


def build_drawdown_model(spec: dict | None, max_loss: float | None) -> tuple[DrawdownModel | None, str]:
    """Create a model from a rule value like {"type": "eod_trailing", "lock_floor_offset": 100, ...}."""
    if not spec:
        return None, "No drawdown model configured."
    if max_loss is None:
        return None, "Maximum loss amount is not configured - enter it to enable drawdown tracking."
    t = spec.get("type")
    if t == "static":
        return StaticDrawdown(max_loss), "static"
    if t == "eod_trailing":
        return EodTrailingDrawdown(
            max_loss, spec.get("lock_floor_offset"), bool(spec.get("lock_on_payout"))
        ), t
    if t == "intraday_trailing":
        return IntradayTrailingDrawdown(
            max_loss, spec.get("lock_floor_offset"), bool(spec.get("lock_on_payout"))
        ), t
    return None, f"Unknown drawdown model type {t!r}."
