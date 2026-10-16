"""Deterministic proposed trade plans for confirmed setups.

All levels are ADVISORY and require manual review. The plan distinguishes:

* **structural invalidation** - the price at which the break-and-retest idea
  is objectively wrong (beyond the retested zone / retest extreme);
* **proposed protective stop** - the invalidation level plus a buffer,
  rounded away from the entry to a valid tick;
* **first target** - in ``structure`` mode, just in front of the next
  opposing zone (never assumed beyond it); in ``fixed_r`` mode a fixed R
  multiple that is *rejected* if an opposing zone sits in the way;
* **extended target** - the next structural level beyond the first, or a
  fixed-R projection when no further structure is known (labelled as such).

A plan is marked invalid (and the setup cannot qualify) when the stop is
structurally invalid, the risk is outside the configured tick limits, the
target is blocked, or reward/risk is below the minimum. A 2:1 target is never
assumed achievable merely because it is arithmetically possible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from core.config import PlanningSettings
from core.instruments import InstrumentSpec


@dataclass
class TradePlan:
    instrument: str
    direction: str
    zone_lower: float
    zone_upper: float
    breakout_price: float
    retest_price: float
    confirmation_price: float
    entry_method: str
    entry_price: float
    entry_zone_low: float
    entry_zone_high: float
    structural_invalidation: float
    stop: float
    target1: float | None
    target2: float | None
    target1_basis: str
    target2_basis: str
    risk_points: float
    risk_ticks: float
    risk_per_contract: float
    reward_points: float | None
    reward_ticks: float | None
    rr: float | None
    rr_extended: float | None
    opposing_zone: dict | None
    valid: bool
    invalid_reasons: list[str] = field(default_factory=list)
    explanations: dict[str, str] = field(default_factory=dict)
    structural_valid: bool = True  # every check except the R:R minimum passed
    rr_ok: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def build_trade_plan(
    direction: str,
    spec: InstrumentSpec,
    zone_lower: float,
    zone_upper: float,
    breakout_price: float,
    retest_price: float,
    confirmation_close: float,
    retest_extreme: float,
    atr: float,
    opposing_zones: list[dict],
    cfg: PlanningSettings,
) -> TradePlan:
    """Build a plan. ``retest_extreme`` = lowest low (long) / highest high (short) from retest to confirmation.

    ``opposing_zones`` are zone dicts (lower/upper/strength/zone_id) that may
    act as obstacles; only those on the profit side of the entry are used.
    """
    long = direction == "long"
    tick = spec.tick_size
    reasons: list[str] = []
    expl: dict[str, str] = {}
    sign = 1 if long else -1

    # --- entry ---------------------------------------------------------------
    edge = zone_upper if long else zone_lower
    if cfg.entry_method == "confirmation_close":
        entry = spec.round_price(confirmation_close)
        expl["entry"] = (
            f"Reference entry at the confirmation candle close {spec.fmt(entry)}. Proposed entry zone "
            f"spans the zone edge {spec.fmt(edge)} to the confirmation close."
        )
    else:
        entry = spec.round_price(edge)
        expl["entry"] = (
            f"Reference entry as a limit at the retested zone edge {spec.fmt(entry)} (requires a "
            f"pullback; may not fill)."
        )
    ez_low, ez_high = sorted((edge, confirmation_close))

    # --- invalidation & stop ------------------------------------------------
    buffer = max(cfg.stop_buffer_ticks * tick, cfg.stop_buffer_atr_mult * atr if atr == atr else 0.0)
    if long:
        invalidation = min(zone_lower, retest_extreme)
        stop = spec.round_price(invalidation - buffer, "down")
        expl["invalidation"] = (
            f"Structural invalidation {spec.fmt(invalidation)}: the lower of the zone bottom "
            f"{spec.fmt(zone_lower)} and the retest low {spec.fmt(retest_extreme)}. A sustained "
            f"move below it means former resistance failed as support."
        )
    else:
        invalidation = max(zone_upper, retest_extreme)
        stop = spec.round_price(invalidation + buffer, "up")
        expl["invalidation"] = (
            f"Structural invalidation {spec.fmt(invalidation)}: the higher of the zone top "
            f"{spec.fmt(zone_upper)} and the retest high {spec.fmt(retest_extreme)}. A sustained "
            f"move above it means former support failed as resistance."
        )
    expl["stop"] = (
        f"Proposed protective stop {spec.fmt(stop)} = invalidation {'minus' if long else 'plus'} a "
        f"{buffer / tick:.0f}-tick buffer (max of {cfg.stop_buffer_ticks:g} ticks and "
        f"{cfg.stop_buffer_atr_mult:g} x ATR), rounded to a valid tick away from entry."
    )

    risk = (entry - stop) * sign
    risk_ticks = spec.ticks(risk)
    if risk <= 0:
        reasons.append("Stop is not on the protective side of the entry (zero or negative risk).")
    elif risk_ticks < cfg.min_stop_ticks:
        reasons.append(f"Stop distance {risk_ticks:.0f} ticks is below the minimum {cfg.min_stop_ticks:g}.")
    elif risk_ticks > cfg.max_stop_ticks:
        reasons.append(f"Stop distance {risk_ticks:.0f} ticks exceeds the maximum {cfg.max_stop_ticks:g}.")
    elif atr == atr and atr > 0 and risk / atr > cfg.max_stop_atr + 1e-9:
        reasons.append(f"Stop distance {risk / atr:.2f} ATR exceeds the maximum {cfg.max_stop_atr:g} ATR.")

    # --- opposing structure -------------------------------------------------
    if long:
        ahead = sorted((z for z in opposing_zones if z["upper"] > entry), key=lambda z: z["lower"])
    else:
        ahead = sorted((z for z in opposing_zones if z["lower"] < entry), key=lambda z: -z["upper"])
    inside = [z for z in ahead if z["lower"] <= entry <= z["upper"]]
    if inside:
        z = inside[0]
        reasons.append(f"Entry sits inside an opposing zone {spec.fmt(z['lower'])}-{spec.fmt(z['upper'])}.")
    ahead = [z for z in ahead if not (z["lower"] <= entry <= z["upper"])]
    first = ahead[0] if ahead else None
    front = cfg.target_front_run_ticks * tick

    def front_run(z: dict) -> float:
        return (
            spec.round_price(z["lower"] - front, "down")
            if long
            else spec.round_price(z["upper"] + front, "up")
        )

    t1: float | None = None
    t2: float | None = None
    t1_basis = t2_basis = "none"
    if risk > 0:
        if cfg.target_method == "structure":
            if first is not None:
                t1 = front_run(first)
                t1_basis = "structure"
                expl["target1"] = (
                    f"First target {spec.fmt(t1)}: {cfg.target_front_run_ticks:g} ticks in front of the "
                    f"next opposing zone {spec.fmt(first['lower'])}-{spec.fmt(first['upper'])} "
                    f"(strength {first.get('strength', 0):.2f})."
                )
            else:
                t1 = spec.round_price(entry + sign * cfg.target_r * risk, "down" if long else "up")
                t1_basis = "fixed_r_no_structure"
                expl["target1"] = (
                    f"First target {spec.fmt(t1)}: no opposing zone was identified, so this is a fixed "
                    f"{cfg.target_r:g}R projection - NOT a structural level."
                )
        else:
            t1 = spec.round_price(entry + sign * cfg.target_r * risk, "down" if long else "up")
            t1_basis = "fixed_r"
            expl["target1"] = f"First target {spec.fmt(t1)}: fixed {cfg.target_r:g}R from entry."
            if first is not None:
                block = front_run(first)
                if (long and block < t1) or (not long and block > t1):
                    reasons.append(
                        f"Fixed {cfg.target_r:g}R target {spec.fmt(t1)} is blocked by the opposing zone "
                        f"{spec.fmt(first['lower'])}-{spec.fmt(first['upper'])}."
                    )
        if t1 is not None and (t1 - entry) * sign <= 0:
            reasons.append("First target is not beyond the entry (opposing zone immediately ahead).")

        # Extended target
        beyond = None
        if first is not None:
            if long:
                beyond = next((z for z in ahead[1:] if z["lower"] > first["upper"]), None)
            else:
                beyond = next((z for z in ahead[1:] if z["upper"] < first["lower"]), None)
        fixed_ext = spec.round_price(entry + sign * cfg.extended_target_r * risk, "down" if long else "up")
        if cfg.target_method == "structure" and first is not None and beyond is not None:
            t2 = front_run(beyond)
            t2_basis = "structure"
            expl["target2"] = (
                f"Extended target {spec.fmt(t2)}: in front of the following zone "
                f"{spec.fmt(beyond['lower'])}-{spec.fmt(beyond['upper'])}."
            )
        else:
            t2 = fixed_ext
            t2_basis = "fixed_r"
            expl["target2"] = (
                f"Extended target {spec.fmt(t2)}: fixed {cfg.extended_target_r:g}R projection "
                f"(no further structure identified) - NOT a structural level."
            )
            if first is not None and cfg.target_method == "fixed_r":
                cap = front_run(beyond) if beyond is not None else None
                if cap is not None and ((long and cap < t2) or (not long and cap > t2)):
                    t2 = cap
                    t2_basis = "capped_by_structure"
                    expl["target2"] += f" Capped at {spec.fmt(cap)} in front of the next zone."
        if t1 is not None and t2 is not None and (t2 - t1) * sign <= 0:
            t2 = None
            t2_basis = "none"
            expl["target2"] = "No extended target beyond the first target."

    reward = (t1 - entry) * sign if t1 is not None else None
    # R:R is computed from whole ticks so float noise (e.g. 1.9999999) never
    # flips a pass/fail decision.
    reward_ticks = spec.ticks(reward) if reward is not None else None
    rr = (reward_ticks / risk_ticks) if (reward_ticks is not None and risk_ticks > 0) else None
    rr_ext = (spec.ticks((t2 - entry) * sign) / risk_ticks) if (t2 is not None and risk_ticks > 0) else None
    structural_valid = not reasons and rr is not None
    rr_ok = rr is not None and rr >= cfg.min_rr - 1e-9
    if rr is not None and not rr_ok:
        reasons.append(f"Planned reward/risk {rr:.2f} is below the minimum {cfg.min_rr:g}.")

    return TradePlan(
        instrument=spec.symbol,
        direction=direction,
        zone_lower=zone_lower,
        zone_upper=zone_upper,
        breakout_price=breakout_price,
        retest_price=retest_price,
        confirmation_price=confirmation_close,
        entry_method=cfg.entry_method,
        entry_price=entry,
        entry_zone_low=ez_low,
        entry_zone_high=ez_high,
        structural_invalidation=invalidation,
        stop=stop,
        target1=t1,
        target2=t2,
        target1_basis=t1_basis,
        target2_basis=t2_basis,
        risk_points=round(risk, 6),
        risk_ticks=risk_ticks,
        risk_per_contract=round(risk_ticks * spec.tick_value, 2),
        reward_points=round(reward, 6) if reward is not None else None,
        reward_ticks=reward_ticks,
        rr=round(rr, 3) if rr is not None else None,
        rr_extended=round(rr_ext, 3) if rr_ext is not None else None,
        opposing_zone={k: first[k] for k in ("zone_id", "lower", "upper", "strength") if k in first}
        if first
        else None,
        valid=not reasons,
        invalid_reasons=reasons,
        explanations=expl,
        structural_valid=structural_valid,
        rr_ok=rr_ok,
    )
