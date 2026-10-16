"""Advisory monitoring of a MANUALLY registered trade.

This module never moves a stop, takes a partial or closes a position. It
evaluates completed bars after the entry and produces review prompts:

* breakeven reviews at configured R thresholds;
* profit-protection reviews (R milestones, planned target, opposing zones,
  momentum deterioration);
* trailing-stop *suggestions* from newly confirmed swings (+ ATR buffer),
  only when tighter than the user-reported stop;
* thesis / invalidation warnings (retest level failed, structural
  invalidation crossed, key swing broken, VWAP lost, time stop, insufficient
  remaining reward, user-reported stop traded, stale data).

The "registered stop" is whatever the user last recorded. It is labelled
*user-reported* everywhere because the application cannot see the broker.

Lifecycle distinctions kept by the system:
1. price reaches the trigger  -> ``EngineEvent.ts`` (bar close time);
2. alert issued               -> alert row ``created_at``;
3. user reviews it            -> alert ``acknowledged_at``;
4. user records an action     -> ``trade_actions`` row linked to the alert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.config import AppSettings
from core.data_quality import FreshnessStatus
from core.events import EngineEvent
from core.indicators import atr as atr_fn
from core.indicators import ema, session_vwap, volume_is_reliable
from core.instruments import InstrumentSpec
from core.risk import breakeven_price, initial_risk_points
from core.swing_detection import detect_swings, params_from_settings, provisional_swings
from core.timeutils import from_iso, timeframe_delta

ADVISORY = "Advisory only - the application has not changed any order."


@dataclass
class TradeView:
    trade_id: int
    instrument: str
    contract: str | None
    direction: str
    entry_ts: pd.Timestamp
    entry_price: float
    initial_stop: float
    quantity: int
    planned_target: float | None
    commission_rt: float
    slippage_ticks: float
    setup_plan: dict | None = None
    actions: list[dict] = field(default_factory=list)

    @property
    def long(self) -> bool:
        return self.direction == "long"

    def stop_timeline(self) -> list[tuple[pd.Timestamp, float]]:
        """(time, user-reported stop) after each recorded stop change."""
        out = [(self.entry_ts, self.initial_stop)]
        for a in sorted(self.active_actions, key=lambda x: (x["action_ts"], x["id"])):
            if a["action_type"] in ("stop_to_breakeven", "stop_tightened", "stop_widened") and a.get(
                "new_stop"
            ):
                out.append((from_iso(a["action_ts"]), float(a["new_stop"])))  # type: ignore[arg-type]
        return out

    def registered_stop_at(self, ts: pd.Timestamp) -> float:
        stop = self.initial_stop
        for t, s in self.stop_timeline():
            if t <= ts:
                stop = s
        return stop

    @property
    def active_actions(self) -> list[dict]:
        return [a for a in self.actions if not a.get("is_void")]

    @property
    def registered_stop(self) -> float:
        return self.stop_timeline()[-1][1]

    @property
    def exited_qty(self) -> int:
        return sum(
            int(a.get("quantity") or 0)
            for a in self.active_actions
            if a["action_type"] in ("partial_exit", "full_exit")
        )

    @property
    def open_qty(self) -> int:
        return max(self.quantity - self.exited_qty, 0)

    @property
    def closed_ts(self) -> pd.Timestamp | None:
        if self.open_qty > 0:
            return None
        exits = [
            from_iso(a["action_ts"])
            for a in self.active_actions
            if a["action_type"] in ("partial_exit", "full_exit")
        ]
        return max(exits) if exits else None  # type: ignore[type-var]


def trade_view_from_rows(trade: dict, actions: list[dict], setup: dict | None = None) -> TradeView:
    return TradeView(
        trade_id=int(trade["id"]),
        instrument=trade["instrument"],
        contract=trade.get("contract"),
        direction=trade["direction"],
        entry_ts=from_iso(trade["entry_ts"]),  # type: ignore[arg-type]
        entry_price=float(trade["entry_price"]),
        initial_stop=float(trade["initial_stop"]),
        quantity=int(trade["quantity"]),
        planned_target=trade.get("planned_target"),
        commission_rt=float(trade.get("commission_per_contract_rt") or 0),
        slippage_ticks=float(trade.get("slippage_ticks") or 0),
        setup_plan=(setup or {}).get("plan") if setup else None,
        actions=actions,
    )


@dataclass
class TrailingSuggestion:
    swing_id: str
    swing_type: str
    swing_price: float
    pivot_ts: pd.Timestamp
    confirm_ts: pd.Timestamp
    buffer_points: float
    buffer_desc: str
    suggested_stop: float
    registered_stop: float
    reference_price: float
    distance_ticks: float
    distance_points: float
    locked_pnl: float  # +profit locked / -remaining risk if the stop were manually moved here
    tighter: bool
    valid: bool
    reason: str
    provisional: bool = False


@dataclass
class TradeMonitor:
    trade_id: int
    as_of: pd.Timestamp | None
    current_price: float | None
    current_r: float | None
    mfe_r: float
    mae_r: float
    initial_risk_points: float
    registered_stop: float
    registered_stop_r: float
    open_qty: int
    unrealized_pnl: float | None
    breakeven_price: float
    be_status: list[dict]
    events: list[EngineEvent]
    trailing: list[TrailingSuggestion]
    provisional_trailing: TrailingSuggestion | None
    warnings: list[str]
    data_ok: bool
    data_message: str
    bars_evaluated: int
    nearest_opposing_zone: dict | None
    target_distance_ticks: float | None

    @property
    def current_trailing(self) -> TrailingSuggestion | None:
        valid = [t for t in self.trailing if t.valid and t.tighter]
        return valid[-1] if valid else None


def _zones_alive_at(zone_history: list[dict], ts: pd.Timestamp) -> list[dict]:
    out = []
    for z in zone_history:
        created = z.get("created_ts")
        ended = z.get("ended_ts")
        if (
            created is not None
            and created <= ts
            and (ended is None or ended > ts)
            and z.get("reactions", 0) >= 1
        ):
            out.append(z)
    return out


def monitor_trade(
    tv: TradeView,
    df: pd.DataFrame,
    spec: InstrumentSpec,
    settings: AppSettings,
    zone_history: list[dict] | None = None,
    freshness: FreshnessStatus | None = None,
    vol_regime: str | None = None,
    timeframe: str = "5min",
) -> TradeMonitor:
    m = settings.management
    tick = spec.tick_size
    tf = timeframe_delta(timeframe)
    risk = initial_risk_points(tv.direction, tv.entry_price, tv.initial_stop)
    sign = 1 if tv.long else -1
    be_px = breakeven_price(
        spec, tv.direction, tv.entry_price, tv.commission_rt, tv.slippage_ticks, m.breakeven_include_costs
    )
    events: list[EngineEvent] = []
    warnings: list[str] = []
    suppressed = set(m.suppressed_categories)
    base_key = f"trade:{tv.trade_id}"

    def emit(
        category: str,
        ts: pd.Timestamp,
        key: str,
        title: str,
        message: str,
        price: float | None,
        details: dict[str, Any] | None = None,
        severity: str | None = None,
    ) -> None:
        if category in suppressed:
            return
        events.append(
            EngineEvent(
                category=category,
                ts=ts,
                title=title,
                message=message,
                dedup_key=f"{base_key}:{key}",
                instrument=tv.instrument,
                trade_id=tv.trade_id,
                price=price,
                details=details or {},
                severity=severity,
            )
        )

    thresholds = list(m.breakeven_thresholds_r)
    if vol_regime == "high volatility" and m.breakeven_thresholds_high_vol_r:
        thresholds = sorted(m.breakeven_thresholds_high_vol_r)
    be_status: list[dict[str, Any]] = [
        {"threshold_r": t, "reached": False, "trigger_ts": None, "price": None, "already_at_be": False}
        for t in thresholds
    ]

    data_ok = True
    data_msg = "Market data current."
    if freshness is not None and freshness.is_stale:
        data_ok = False
        data_msg = (
            "Market data is stale - the application cannot reliably assess this trade. " + freshness.message
        )
    if df is None or df.empty:
        data_ok = False
        data_msg = "No market data available - the application cannot assess this trade."

    # Only bars that OPEN after the entry are evaluated (the entry bar contains pre-entry prices).
    post = df[df.index >= tv.entry_ts] if df is not None and not df.empty else pd.DataFrame()
    if tv.contract and "contract" in post.columns:
        post = post[post["contract"] == tv.contract]
        if df is not None and "contract" in df.columns and df["contract"].iloc[-1] != tv.contract:
            warnings.append(
                f"Loaded data is for {df['contract'].iloc[-1]}, but this trade is in {tv.contract}."
            )
    closed_ts = tv.closed_ts
    if closed_ts is not None and not post.empty:
        post = post[post.index <= closed_ts]  # stop evaluating once the user recorded the final exit

    mfe = 0.0
    mae = 0.0
    current_price = None
    as_of = None
    trailing: list[TrailingSuggestion] = []
    provisional = None
    nearest = None
    target_dist = None

    if not post.empty:
        full = df[df.index <= post.index[-1]]
        if tv.contract and "contract" in full.columns:
            full = full[full["contract"] == tv.contract]
        atr_series = atr_fn(full, m.trailing_atr_period)
        swings = detect_swings(
            full,
            tick,
            params_from_settings(settings.swings),
            atr_fn(full, settings.swings.atr_period),
            timeframe,
        )
        vol_ok, _ = volume_is_reliable(full)
        vwap = session_vwap(full, settings.sessions.vwap_anchor, vol_ok)
        ema_m = ema(full["close"], m.momentum_ema_period)
        want = "low" if tv.long else "high"
        fav_swings = (
            swings[
                (swings["status"] == "confirmed")
                & (swings["swing_type"] == want)
                & (swings["pivot_ts"] >= tv.entry_ts)
            ]
            if not swings.empty
            else swings
        )
        last_momentum_k = -10_000
        entry_vwap_side = None
        trigger_flags: set[str] = set()
        last_structure_mfe: float | None = None
        paused_at_stop: float | None = None

        for k, (ts, row) in enumerate(post.iterrows()):
            close_t = ts + tf
            h, lo, c = float(row["high"]), float(row["low"]), float(row["close"])
            fav = ((h - tv.entry_price) if tv.long else (tv.entry_price - lo)) / risk
            adv = ((lo - tv.entry_price) if tv.long else (tv.entry_price - h)) / risk
            mfe = max(mfe, fav)
            mae = min(mae, adv)
            reg_stop = tv.registered_stop_at(close_t)
            stop_at_be = (reg_stop >= tv.entry_price) if tv.long else (reg_stop <= tv.entry_price)
            current_price, as_of = c, close_t

            # Once price trades through the user-reported stop the position may be closed: pause every other
            # prompt until the user records a different stop (or the exit).
            if paused_at_stop is not None:
                if reg_stop == paused_at_stop:
                    continue
                paused_at_stop = None
            stop_hit = (lo <= reg_stop) if tv.long else (h >= reg_stop)
            if stop_hit:
                paused_at_stop = reg_stop
                if f"stop_touched:{reg_stop}" not in trigger_flags:
                    trigger_flags.add(f"stop_touched:{reg_stop}")
                    emit(
                        "trade_warning",
                        close_t,
                        f"stop_touched:{reg_stop}",
                        f"Price traded at your reported stop: trade #{tv.trade_id}",
                        f"Price traded at/through your user-reported stop {spec.fmt(reg_stop)}. If your broker stop "
                        f"was there the position may already be closed - record the exit manually. Further prompts "
                        f"for this trade are paused until you record a different stop or the exit. The application "
                        f"cannot see your broker. {ADVISORY}",
                        reg_stop,
                        severity="critical",
                    )
                continue

            # breakeven review
            for st in be_status:
                if not st["reached"] and fav >= st["threshold_r"]:
                    px = tv.entry_price + sign * st["threshold_r"] * risk
                    st.update(reached=True, trigger_ts=close_t, price=px, already_at_be=stop_at_be)
                    if not stop_at_be:
                        emit(
                            "breakeven_review",
                            close_t,
                            f"be:{st['threshold_r']}",
                            f"Breakeven review: {tv.instrument} trade #{tv.trade_id} reached +{st['threshold_r']:g}R",
                            f"Price reached {spec.fmt(px)} (+{st['threshold_r']:g}R). Review whether moving your stop "
                            f"to breakeven ({spec.fmt(be_px)} incl. estimated costs) is appropriate. {ADVISORY}",
                            px,
                            {
                                "threshold_r": st["threshold_r"],
                                "breakeven_price": be_px,
                                "registered_stop": reg_stop,
                            },
                        )

            # R milestones
            for ms in m.profit_r_milestones:
                if fav >= ms and f"r:{ms}" not in trigger_flags:
                    trigger_flags.add(f"r:{ms}")
                    px = tv.entry_price + sign * ms * risk
                    emit(
                        "profit_protection",
                        close_t,
                        f"r:{ms}",
                        f"Profit review: trade #{tv.trade_id} reached +{ms:g}R",
                        f"Price reached {spec.fmt(px)} (+{ms:g}R). Consider whether to take a partial or secure "
                        f"profits. {ADVISORY}",
                        px,
                        {"milestone_r": ms},
                    )

            # planned target
            if tv.planned_target:
                dist_total = abs(tv.planned_target - tv.entry_price)
                progressed = (h - tv.entry_price) if tv.long else (tv.entry_price - lo)
                reached = (h >= tv.planned_target) if tv.long else (lo <= tv.planned_target)
                if reached and "target_reached" not in trigger_flags:
                    trigger_flags.update({"target_reached", "target_approach"})
                    emit(
                        "target_approaching",
                        close_t,
                        "target_reached",
                        f"Planned target reached: trade #{tv.trade_id}",
                        f"Price traded at your planned target {spec.fmt(tv.planned_target)}. Review your exit plan. "
                        f"{ADVISORY}",
                        tv.planned_target,
                    )
                elif (
                    dist_total > 0
                    and progressed >= m.target_approach_fraction * dist_total
                    and "target_approach" not in trigger_flags
                ):
                    trigger_flags.add("target_approach")
                    emit(
                        "target_approaching",
                        close_t,
                        "target_approach",
                        f"Target approaching: trade #{tv.trade_id}",
                        f"Price covered {progressed / dist_total:.0%} of the distance to the planned target "
                        f"{spec.fmt(tv.planned_target)}. {ADVISORY}",
                        c,
                    )

            atr_k = float(atr_series.loc[ts]) if ts in atr_series.index else float("nan")

            # opposing zones
            if zone_history:
                alive = _zones_alive_at(zone_history, close_t)
                ahead = [
                    z
                    for z in alive
                    if (z["lower"] > tv.entry_price if tv.long else z["upper"] < tv.entry_price)
                ]
                if ahead:
                    z = (
                        min(ahead, key=lambda z: z["lower"])
                        if tv.long
                        else max(ahead, key=lambda z: z["upper"])
                    )
                    edge = z["lower"] if tv.long else z["upper"]
                    approach = m.opposing_zone_approach_atr * (atr_k if atr_k == atr_k else 0.0)
                    near = (h >= edge - approach) if tv.long else (lo <= edge + approach)
                    if near and f"zone:{z['zone_id']}" not in trigger_flags and fav > 0:
                        trigger_flags.add(f"zone:{z['zone_id']}")
                        emit(
                            "profit_protection",
                            close_t,
                            f"zone:{z['zone_id']}",
                            f"Opposing zone ahead: trade #{tv.trade_id}",
                            f"Price is approaching the {'resistance' if tv.long else 'support'} zone "
                            f"{spec.fmt(z['lower'])}-{spec.fmt(z['upper'])} (strength {z.get('strength', 0):.2f}) at "
                            f"{fav:+.2f}R. Zones can cause reactions; consider whether to protect profit. Not every "
                            f"zone requires an exit. {ADVISORY}",
                            edge,
                            {"zone": {k: z[k] for k in ("zone_id", "lower", "upper")}},
                        )

            # trailing suggestions from swings confirmed at this bar
            if m.trailing_enabled and not fav_swings.empty:
                newly = fav_swings[(fav_swings["confirm_ts"] == close_t)]
                for sw in newly.to_dict("records"):
                    trailing.append(_trail(sw, tv, spec, settings, atr_series, reg_stop, c, risk))
                    t = trailing[-1]
                    if t.valid and t.tighter:
                        emit(
                            "trailing_stop",
                            close_t,
                            f"trail:{sw['swing_id']}",
                            f"Trailing-stop suggestion: trade #{tv.trade_id} -> {spec.fmt(t.suggested_stop)}",
                            f"{t.reason} Your user-reported stop is {spec.fmt(reg_stop)}. If you choose to adjust, do "
                            f"it at your broker and record it here. {ADVISORY}",
                            t.suggested_stop,
                            {
                                "suggested_stop": t.suggested_stop,
                                "swing_price": t.swing_price,
                                "buffer_points": t.buffer_points,
                            },
                        )

            # key swing broken (most recent favourable swing confirmed before this bar)
            if not fav_swings.empty:
                known = fav_swings[fav_swings["confirm_ts"] <= close_t - tf]
                if not known.empty:
                    key_sw = known.iloc[-1]
                    broken = (c < key_sw["price"]) if tv.long else (c > key_sw["price"])
                    # One warning per favourable leg: a further warning needs a new best excursion first.
                    new_leg = last_structure_mfe is None or mfe > last_structure_mfe + 1e-9
                    if broken and new_leg and f"structure:{key_sw['swing_id']}" not in trigger_flags:
                        trigger_flags.add(f"structure:{key_sw['swing_id']}")
                        last_structure_mfe = mfe
                        emit(
                            "trade_warning",
                            close_t,
                            f"structure:{key_sw['swing_id']}",
                            f"Market structure lost: trade #{tv.trade_id}",
                            f"Close {spec.fmt(c)} broke the most recent confirmed swing "
                            f"{'low' if tv.long else 'high'} {spec.fmt(key_sw['price'])}. This is a warning, not an "
                            f"instruction to exit. {ADVISORY}",
                            c,
                        )

            # momentum deterioration
            em = float(ema_m.loc[ts]) if ts in ema_m.index else float("nan")
            prev_ts = ts - tf
            prev_em = float(ema_m.loc[prev_ts]) if prev_ts in ema_m.index else float("nan")
            prev_c = float(full["close"].loc[prev_ts]) if prev_ts in full.index else float("nan")
            if (
                m.momentum_enabled
                and mfe >= m.momentum_min_r
                and em == em
                and prev_em == prev_em
                and prev_c == prev_c
            ):
                crossed = (prev_c >= prev_em and c < em) if tv.long else (prev_c <= prev_em and c > em)
                if crossed and k - last_momentum_k >= m.momentum_min_bars_between:
                    last_momentum_k = k
                    emit(
                        "profit_protection",
                        close_t,
                        f"momentum:{close_t.isoformat()}",
                        f"Momentum weakening: trade #{tv.trade_id}",
                        f"Close {spec.fmt(c)} crossed {'below' if tv.long else 'above'} the "
                        f"{m.momentum_ema_period}-EMA after reaching +{mfe:.2f}R. Consider securing profits. "
                        f"{ADVISORY}",
                        c,
                    )

            # thesis warnings from the linked setup plan
            plan = tv.setup_plan or {}
            if plan:
                zl, zu = plan.get("zone_lower"), plan.get("zone_upper")
                if zl is not None and zu is not None:
                    failed = (c < zl) if tv.long else (c > zu)
                    if failed and "retest_failed" not in trigger_flags:
                        trigger_flags.add("retest_failed")
                        emit(
                            "trade_warning",
                            close_t,
                            "retest_failed",
                            f"Retest level failed: trade #{tv.trade_id}",
                            f"Close {spec.fmt(c)} is back {'below' if tv.long else 'above'} the retested zone "
                            f"{spec.fmt(zl)}-{spec.fmt(zu)}. The break-and-retest thesis may be failing. {ADVISORY}",
                            c,
                        )
                inval = plan.get("structural_invalidation")
                if inval is not None:
                    crossed_inv = (c < inval) if tv.long else (c > inval)
                    if crossed_inv and "invalidation" not in trigger_flags:
                        trigger_flags.add("invalidation")
                        emit(
                            "trade_warning",
                            close_t,
                            "invalidation",
                            f"Structural invalidation crossed: trade #{tv.trade_id}",
                            f"Close {spec.fmt(c)} crossed the structural invalidation level {spec.fmt(inval)}. This is "
                            f"a warning, not an exit instruction. {ADVISORY}",
                            c,
                            severity="critical",
                        )

            # VWAP lost
            vw = float(vwap.loc[ts]) if ts in vwap.index else float("nan")
            if vw == vw:
                side = (c > vw) if tv.long else (c < vw)
                if entry_vwap_side is None:
                    entry_vwap_side = side
                if m.vwap_loss_warning and entry_vwap_side and not side and "vwap_lost" not in trigger_flags:
                    trigger_flags.add("vwap_lost")
                    emit(
                        "trade_warning",
                        close_t,
                        "vwap_lost",
                        f"VWAP lost: trade #{tv.trade_id}",
                        f"Close {spec.fmt(c)} moved to the wrong side of VWAP {spec.fmt(vw)}. {ADVISORY}",
                        c,
                    )

            # time stop / setup expiry
            if (
                m.time_stop_bars
                and k + 1 >= m.time_stop_bars
                and mfe < m.time_stop_min_r
                and "time_stop" not in trigger_flags
            ):
                trigger_flags.add("time_stop")
                emit(
                    "trade_warning",
                    close_t,
                    "time_stop",
                    f"Trade not progressing: trade #{tv.trade_id}",
                    f"After {k + 1} bars the trade has not reached +{m.time_stop_min_r:g}R (best {mfe:+.2f}R). The "
                    f"original setup may have expired. {ADVISORY}",
                    c,
                )

        if paused_at_stop is not None:
            warnings.append(
                f"Price traded through your user-reported stop {spec.fmt(paused_at_stop)}; management "
                "prompts are paused. Record the exit, or record your actual stop if it is elsewhere."
            )

        # remaining reward vs next opposing zone (at the latest bar)
        if zone_history and current_price is not None and as_of is not None and paused_at_stop is None:
            alive = _zones_alive_at(zone_history, as_of)
            ahead = [
                z for z in alive if (z["lower"] > current_price if tv.long else z["upper"] < current_price)
            ]
            if ahead:
                nearest = (
                    min(ahead, key=lambda z: z["lower"]) if tv.long else max(ahead, key=lambda z: z["upper"])
                )
                edge = nearest["lower"] if tv.long else nearest["upper"]
                room = abs(edge - current_price)
                if room < m.insufficient_reward_r * risk:
                    emit(
                        "trade_warning",
                        as_of,
                        f"insufficient_reward:{nearest['zone_id']}",
                        f"Limited remaining reward: trade #{tv.trade_id}",
                        f"The next opposing zone at {spec.fmt(edge)} is only {room / risk:.2f}R away. {ADVISORY}",
                        current_price,
                    )

        # provisional (unconfirmed) swing - display only, never alerted
        prov = provisional_swings(full, params_from_settings(settings.swings))
        if not prov.empty and m.trailing_enabled:
            p = prov[(prov["swing_type"] == want) & (prov["pivot_ts"] >= tv.entry_ts)]
            if not p.empty and current_price is not None:
                rec = p.iloc[-1].to_dict()
                rec.update(swing_id=f"provisional:{rec['pivot_ts'].isoformat()}", confirm_ts=as_of)
                provisional = _trail(
                    rec, tv, spec, settings, atr_series, tv.registered_stop, current_price, risk
                )
                provisional.provisional = True
                provisional.reason = (
                    "PROVISIONAL - swing not yet confirmed (needs "
                    f"{int(rec.get('bars_remaining', 0))} more bar(s)); excluded from alerts. "
                    + provisional.reason
                )
    else:
        if df is not None and not df.empty and df.index[-1] + tf >= tv.entry_ts:
            current_price = float(df["close"].iloc[-1])
            as_of = df.index[-1] + tf
        warnings.append(
            "No completed bars after the entry bar yet - management checks start after the next bar."
        )

    if not data_ok and as_of is not None:
        emit(
            "trade_warning",
            as_of,
            f"stale:{as_of.isoformat()}",
            f"Cannot assess trade #{tv.trade_id}",
            data_msg,
            current_price,
            severity="high",
        )

    if tv.planned_target and current_price is not None:
        target_dist = spec.ticks((tv.planned_target - current_price) * sign)

    reg = tv.registered_stop
    cur_r = None if current_price is None else ((current_price - tv.entry_price) * sign / risk)
    unreal = (
        None
        if current_price is None
        else (current_price - tv.entry_price) * sign * spec.point_value * tv.open_qty
    )
    return TradeMonitor(
        trade_id=tv.trade_id,
        as_of=as_of,
        current_price=current_price,
        current_r=cur_r,
        mfe_r=round(mfe, 4),
        mae_r=round(mae, 4),
        initial_risk_points=risk,
        registered_stop=reg,
        registered_stop_r=(reg - tv.entry_price) * sign / risk,
        open_qty=tv.open_qty,
        unrealized_pnl=None if unreal is None else round(unreal, 2),
        breakeven_price=be_px,
        be_status=be_status,
        events=events,
        trailing=trailing,
        provisional_trailing=provisional,
        warnings=warnings,
        data_ok=data_ok,
        data_message=data_msg,
        bars_evaluated=len(post),
        nearest_opposing_zone=nearest,
        target_distance_ticks=target_dist,
    )


def _trail(
    sw: dict,
    tv: TradeView,
    spec: InstrumentSpec,
    settings: AppSettings,
    atr_series: pd.Series,
    reg_stop: float,
    ref_price: float,
    risk: float,
) -> TrailingSuggestion:
    m = settings.management
    tick = spec.tick_size
    conf_ts = sw["confirm_ts"]
    atr_val = float("nan")
    if atr_series is not None and len(atr_series):
        upto = atr_series[atr_series.index + timeframe_delta("5min") <= conf_ts]
        if len(upto):
            atr_val = float(upto.iloc[-1])
    tick_buf = m.trailing_buffer_ticks * tick
    atr_buf = m.trailing_atr_mult * atr_val if not math.isnan(atr_val) else 0.0
    buf = max(tick_buf, atr_buf)
    desc = (
        f"max({m.trailing_buffer_ticks:g} ticks, {m.trailing_atr_mult:g} x ATR({m.trailing_atr_period})"
        f"{'=' + spec.fmt(atr_val) if not math.isnan(atr_val) else ' unavailable'}) = {spec.fmt(buf)} pts"
    )
    price = float(sw["price"])
    if tv.long:
        sugg = spec.round_price(price - buf, "down")
        tighter = sugg > reg_stop
        valid = sugg < ref_price
    else:
        sugg = spec.round_price(price + buf, "up")
        tighter = sugg < reg_stop
        valid = sugg > ref_price
    sign = 1 if tv.long else -1
    dist = abs(ref_price - sugg)
    locked = (sugg - tv.entry_price) * sign * spec.point_value * tv.open_qty
    if not valid:
        reason = (
            f"Swing {'low' if tv.long else 'high'} {spec.fmt(price)} minus buffer gives {spec.fmt(sugg)}, which is "
            f"beyond the current price - not usable."
        )
    elif not tighter:
        reason = (
            f"Swing-based level {spec.fmt(sugg)} would be looser than your user-reported stop "
            f"{spec.fmt(reg_stop)} - not suggested (a trailing stop never loosens)."
        )
    else:
        what = "lock in" if locked >= 0 else "reduce risk to"
        reason = (
            f"New confirmed swing {'low' if tv.long else 'high'} {spec.fmt(price)} (pivot "
            f"{sw['pivot_ts'].strftime('%m-%d %H:%M UTC')}); buffer {desc}. Suggested level {spec.fmt(sugg)} "
            f"would {what} ${abs(locked):,.2f} on {tv.open_qty} open contract(s) if you moved it manually."
        )
    return TrailingSuggestion(
        swing_id=str(sw["swing_id"]),
        swing_type=str(sw["swing_type"]),
        swing_price=price,
        pivot_ts=sw["pivot_ts"],
        confirm_ts=conf_ts,
        buffer_points=buf,
        buffer_desc=desc,
        suggested_stop=sugg,
        registered_stop=reg_stop,
        reference_price=ref_price,
        distance_ticks=spec.ticks(dist),
        distance_points=dist,
        locked_pnl=round(locked, 2),
        tighter=bool(tighter),
        valid=bool(valid),
        reason=reason,
    )


def r_progress_series(tv: TradeView, df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """Close-based R-multiple per completed bar after entry (for charts)."""
    risk = initial_risk_points(tv.direction, tv.entry_price, tv.initial_stop)
    post = df[df.index >= tv.entry_ts]
    sign = 1 if tv.long else -1
    r = (post["close"] - tv.entry_price) * sign / risk
    return pd.DataFrame({"close_ts": post.index + timeframe_delta(timeframe), "r": r.to_numpy(np.float64)})
