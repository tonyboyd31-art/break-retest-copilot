"""Deterministic risk, sizing and R-multiple calculations.

Position sizing never recommends more contracts than the most restrictive of:

* the per-trade dollar risk budget (stop distance + estimated costs);
* the available drawdown (a full stop-out must not breach it);
* the firm-imposed contract limit for this instrument;
* the user's personal contract limit.

It never increases size after losses or when approaching a target - it has
no memory of previous trades at all. The binding constraint is reported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.instruments import InstrumentSpec, is_on_tick


class RiskInputError(ValueError):
    pass


def initial_risk_points(direction: str, entry: float, initial_stop: float) -> float:
    """Signed-safe initial risk in price points; raises if not positive."""
    if entry <= 0 or initial_stop <= 0:
        raise RiskInputError("Prices must be positive.")
    risk = (entry - initial_stop) if direction == "long" else (initial_stop - entry)
    if risk <= 0:
        raise RiskInputError(
            "Initial risk must be positive: the stop must be below a long entry / above a short entry."
        )
    return risk


def r_multiple(direction: str, entry: float, initial_stop: float, price: float) -> float:
    """Long: (price - entry) / (entry - stop). Short: (entry - price) / (stop - entry)."""
    risk = initial_risk_points(direction, entry, initial_stop)
    if price <= 0:
        raise RiskInputError("Price must be positive.")
    move = (price - entry) if direction == "long" else (entry - price)
    return move / risk


def cost_per_contract(spec: InstrumentSpec, commission_rt: float, slippage_ticks_per_side: float) -> float:
    """Estimated round-trip cost per contract: commission + slippage on entry and exit."""
    if commission_rt < 0 or slippage_ticks_per_side < 0:
        raise RiskInputError("Costs cannot be negative.")
    return commission_rt + 2 * slippage_ticks_per_side * spec.tick_value


def breakeven_price(
    spec: InstrumentSpec,
    direction: str,
    entry: float,
    commission_rt: float,
    slippage_ticks_per_side: float,
    include_costs: bool = True,
) -> float:
    """Price at which closing the trade nets ~zero after estimated costs (rounded away from entry)."""
    if not include_costs:
        return entry
    cost_points = cost_per_contract(spec, commission_rt, slippage_ticks_per_side) / spec.point_value
    if direction == "long":
        return spec.round_price(entry + cost_points, "up")
    return spec.round_price(entry - cost_points, "down")


def firm_limit_units(spec: InstrumentSpec, max_minis: float | None, max_micros: float | None) -> int | None:
    """Convert a firm limit (expressed in minis and/or micros) into contracts of ``spec``."""
    if spec.mini_equivalent < 1.0:
        if max_micros is not None:
            return int(math.floor(max_micros + 1e-9))
        if max_minis is not None:
            return int(math.floor(max_minis / spec.mini_equivalent + 1e-9))
        return None
    if max_minis is not None:
        return int(math.floor(max_minis + 1e-9))
    if max_micros is not None:
        return int(math.floor(max_micros * 0.1 + 1e-9))
    return None


def personal_limit_units(spec: InstrumentSpec, max_mini_equivalents: float | None) -> int | None:
    if max_mini_equivalents is None:
        return None
    return int(math.floor(max_mini_equivalents / spec.mini_equivalent + 1e-9))


@dataclass
class LimitCheck:
    name: str
    contracts: int | None  # None = not applicable / not configured
    detail: str


@dataclass
class PositionSizeResult:
    instrument: str
    direction: str
    stop_ticks: float
    stop_points: float
    risk_per_contract: float
    cost_per_contract: float
    all_in_risk_per_contract: float
    limits: list[LimitCheck]
    final_contracts: int
    binding_constraints: list[str]
    total_all_in_risk: float
    remaining_drawdown_after_loss: float | None
    target_ticks: float | None = None
    potential_profit: float | None = None
    rr: float | None = None
    warnings: list[str] = field(default_factory=list)


def size_position(
    spec: InstrumentSpec,
    direction: str,
    entry: float,
    stop: float,
    max_dollar_risk: float,
    available_drawdown: float | None = None,
    firm_max_contracts: int | None = None,
    personal_max_contracts: int | None = None,
    commission_rt: float = 0.0,
    slippage_ticks_per_side: float = 0.0,
    target: float | None = None,
    drawdown_buffer: float = 0.0,
) -> PositionSizeResult:
    if direction not in ("long", "short"):
        raise RiskInputError("Direction must be 'long' or 'short'.")
    if max_dollar_risk <= 0:
        raise RiskInputError("Maximum dollar risk must be positive.")
    for name, v in (("entry", entry), ("stop", stop)):
        if not is_on_tick(v, spec.tick_size):
            raise RiskInputError(
                f"{name} {v} is not a multiple of the {spec.symbol} tick size {spec.tick_size}."
            )
    if target is not None and not is_on_tick(target, spec.tick_size):
        raise RiskInputError(f"target {target} is not a multiple of the tick size {spec.tick_size}.")
    risk_pts = initial_risk_points(direction, entry, stop)
    if available_drawdown is not None and available_drawdown < 0:
        raise RiskInputError("Available drawdown cannot be negative.")
    if drawdown_buffer < 0:
        raise RiskInputError("Drawdown buffer cannot be negative.")

    stop_ticks = spec.ticks(risk_pts)
    risk_pc = stop_ticks * spec.tick_value
    cost_pc = cost_per_contract(spec, commission_rt, slippage_ticks_per_side)
    all_in = risk_pc + cost_pc
    warnings: list[str] = []

    limits = [
        LimitCheck(
            "Risk budget",
            int(math.floor(max_dollar_risk / all_in + 1e-9)),
            f"${max_dollar_risk:,.2f} / ${all_in:,.2f} all-in risk per contract",
        )
    ]
    if available_drawdown is not None:
        usable = max(available_drawdown - drawdown_buffer, 0.0)
        n_dd = int(math.floor(usable / all_in + 1e-9))
        limits.append(
            LimitCheck(
                "Available drawdown",
                n_dd,
                f"(${available_drawdown:,.2f} - ${drawdown_buffer:,.2f} buffer) / ${all_in:,.2f}",
            )
        )
    else:
        limits.append(LimitCheck("Available drawdown", None, "Not provided - drawdown limit not applied."))
        warnings.append("No drawdown allowance provided; size is not checked against the account drawdown.")
    limits.append(
        LimitCheck(
            "Firm contract limit",
            firm_max_contracts,
            "From the selected account's rules"
            if firm_max_contracts is not None
            else "Not configured / unverified - not applied.",
        )
    )
    limits.append(
        LimitCheck(
            "Personal contract limit",
            personal_max_contracts,
            "From your settings" if personal_max_contracts is not None else "Not configured.",
        )
    )

    applicable = [lc for lc in limits if lc.contracts is not None]
    final = max(min(int(lc.contracts) for lc in applicable if lc.contracts is not None), 0)
    binding = [lc.name for lc in applicable if lc.contracts == final]
    if final == 0:
        warnings.append(
            "No contract fits within the most restrictive limit - the trade as specified is too large."
        )
    final -= final % spec.qty_increment

    remaining = None
    if available_drawdown is not None:
        remaining = available_drawdown - final * all_in

    tgt_ticks = profit = rr = None
    if target is not None:
        move = (target - entry) if direction == "long" else (entry - target)
        if move <= 0:
            warnings.append("Target is not on the profit side of the entry.")
        tgt_ticks = spec.ticks(move)
        profit = tgt_ticks * spec.tick_value * final - cost_pc * final
        rr = tgt_ticks / stop_ticks if stop_ticks > 0 else None

    return PositionSizeResult(
        instrument=spec.symbol,
        direction=direction,
        stop_ticks=stop_ticks,
        stop_points=risk_pts,
        risk_per_contract=round(risk_pc, 2),
        cost_per_contract=round(cost_pc, 2),
        all_in_risk_per_contract=round(all_in, 2),
        limits=limits,
        final_contracts=final,
        binding_constraints=binding,
        total_all_in_risk=round(final * all_in, 2),
        remaining_drawdown_after_loss=None if remaining is None else round(remaining, 2),
        target_ticks=tgt_ticks,
        potential_profit=None if profit is None else round(profit, 2),
        rr=None if rr is None else round(rr, 3),
        warnings=warnings,
    )
