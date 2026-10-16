"""Bar-by-bar break-and-retest scanner (the strategy engine entry point).

``run_scan`` replays a single-contract 5-minute series in chronological
order. At each completed bar *i* it:

1. updates every live support/resistance zone with bar *i* (tests, holds,
   breaks) - first-time breaks are returned;
2. advances every pending setup through the retest state machine, and
   finalises (plans + scores) setups confirmed on bar *i* using only zones,
   indicators and higher-timeframe bars known at bar *i*'s close;
3. opens a new setup for each first-time break of a *relevant* zone and
   evaluates the breakout candle;
4. adds swings whose confirmation completes on bar *i* to the zone tracker.

The engine is independent of any prop firm: firm rules affect risk and
eligibility checks elsewhere, never signal generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.breakout_detection import evaluate_breakout
from core.config import AppSettings
from core.data_quality import DataQualityReport, validate_candles
from core.events import EngineEvent
from core.indicators import atr as atr_fn
from core.indicators import ema, relative_volume, session_vwap, volume_reliability_mask
from core.instruments import InstrumentSpec
from core.resampling import compute_trend, htf_value_at, resample_session_aware
from core.retest_engine import (
    BarCtx,
    Setup,
    SetupState,
    advance_setup,
    make_setup_id,
)
from core.scoring import (
    aggregate_score,
    factor_breakout,
    factor_htf,
    factor_retest,
    factor_rr,
    factor_rvol,
    factor_vwap,
    factor_zone,
)
from core.support_resistance import ZoneTracker, distance_to_zone, zone_to_dict
from core.swing_detection import detect_swings, params_from_settings
from core.timeutils import (
    in_time_windows,
    minutes_until_session_close,
    session_label,
    timeframe_delta,
    trade_dates,
)
from core.trade_planning import build_trade_plan


@dataclass
class ScanResult:
    instrument: str
    contract: str | None
    timeframe: str
    as_of: pd.Timestamp | None
    quality: DataQualityReport
    settings_hash: str
    setups: list[Setup] = field(default_factory=list)
    events: list[EngineEvent] = field(default_factory=list)
    swings: pd.DataFrame = field(default_factory=pd.DataFrame)
    zones: list[dict] = field(default_factory=list)  # alive zones at the last bar
    zone_history: list[dict] = field(default_factory=list)  # every zone incl. broken/invalidated
    indicators: pd.DataFrame = field(default_factory=pd.DataFrame)
    htf_trends: dict[str, pd.DataFrame] = field(default_factory=dict)
    scanned: bool = True
    message: str = ""

    def setup(self, setup_id: str) -> Setup | None:
        return next((s for s in self.setups if s.setup_id == setup_id), None)

    def by_state(self, *states: SetupState) -> list[Setup]:
        return [s for s in self.setups if s.state in states]

    @property
    def active_setups(self) -> list[Setup]:
        return [s for s in self.setups if not s.is_terminal]


def _vol_regime(atr_series: np.ndarray, i: int, window: int = 1000) -> str | None:
    lo = max(0, i - window + 1)
    hist = atr_series[lo : i + 1]
    hist = hist[~np.isnan(hist)]
    if len(hist) < 50 or math.isnan(atr_series[i]):
        return None
    pct = float((hist <= atr_series[i]).mean())
    if pct < 1 / 3:
        return "low volatility"
    if pct > 2 / 3:
        return "high volatility"
    return "normal volatility"


def run_scan(
    df: pd.DataFrame,
    spec: InstrumentSpec,
    settings: AppSettings,
    contract: str | None = None,
    block_qualification_from: pd.Timestamp | None = None,
    block_reason: str | None = None,
) -> ScanResult:
    """Run the full strategy over ``df`` (single contract, bar-open UTC index)."""
    tf = settings.timeframes.entry_timeframe
    tf_delta = timeframe_delta(tf)
    shash = settings.strategy_hash()
    quality = validate_candles(df, tf, spec.tick_size)
    if contract is None and "contract" in df.columns and len(df):
        contract = str(df["contract"].iloc[-1])
    result = ScanResult(spec.symbol, contract, tf, None, quality, shash)
    if quality.rows == 0 or quality.errors:
        result.scanned = False
        result.message = "Scan disabled - data failed validation: " + "; ".join(
            e.message for e in quality.errors
        )
        return result

    n = len(df)
    idx = df.index
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    close_ts = idx + tf_delta
    result.as_of = close_ts[-1]

    # --- indicators ------------------------------------------------------
    atr_s = atr_fn(df, settings.swings.atr_period)
    atr_v = atr_s.to_numpy(float)
    vwap_v = session_vwap(df, settings.sessions.vwap_anchor, volume_ok=True).to_numpy(float, copy=True)
    rvol_v = relative_volume(df, settings.breakout.rvol_lookback, volume_ok=True).to_numpy(float, copy=True)
    # Causal reliability: VWAP / relative volume exist at bar i only if volume was trustworthy up to bar i.
    vmask = volume_reliability_mask(df)
    vwap_v[~vmask] = np.nan
    rvol_v[~vmask] = np.nan
    ema_fast = ema(df["close"], settings.management.momentum_ema_period).to_numpy(float)
    result.indicators = pd.DataFrame(
        {"atr": atr_v, "vwap": vwap_v, "rvol": rvol_v, "ema_mom": ema_fast}, index=idx
    )

    # --- swings ------------------------------------------------------------
    swings = detect_swings(df, spec.tick_size, params_from_settings(settings.swings), atr_s, tf)
    result.swings = swings
    by_confirm: dict[int, list[dict]] = {}
    if not swings.empty:
        for rec in swings[swings["status"] == "confirmed"].to_dict("records"):
            by_confirm.setdefault(int(rec["confirm_idx"]), []).append(rec)

    # --- higher-timeframe trends (only bars with close_ts <= t are used) ----
    tcfg = settings.timeframes
    htf_at: dict[str, np.ndarray] = {}
    for htf in tcfg.enabled_htfs():
        bars = resample_session_aware(df, htf, tf)
        trend = compute_trend(bars, tcfg.trend_ema_fast, tcfg.trend_ema_slow, tcfg.trend_slope_bars)
        result.htf_trends[htf] = trend
        htf_at[htf] = htf_value_at(trend, close_ts)

    gap_before = np.zeros(n, dtype=bool)
    for p in quality.gap_positions:
        gap_before[p] = True
    tdates = trade_dates(idx)

    tracker = ZoneTracker(spec.tick_size, settings.zones, spec.price_decimals)
    setups: dict[str, Setup] = {}
    active: list[Setup] = []
    events: list[EngineEvent] = []
    tick = spec.tick_size

    def htf_trends_at(i: int) -> dict[str, str | None]:
        return {k: v[i] for k, v in htf_at.items()}

    def opposing(direction: str, exclude_id: int, ref_price: float, atr_i: float) -> list[dict]:
        out = []
        max_d = settings.zones.relevance_max_distance_atr * atr_i if atr_i == atr_i else float("inf")
        for z in tracker.alive_zones():
            if z.zone_id == exclude_id or z.reactions < settings.zones.opposing_min_reactions:
                continue
            if direction == "long" and z.upper <= ref_price:
                continue
            if direction == "short" and z.lower >= ref_price:
                continue
            if distance_to_zone(ref_price, z.lower, z.upper) > max_d:
                continue
            strength, _ = tracker.strength(z, i_cur[0])
            out.append({"zone_id": z.zone_id, "lower": z.lower, "upper": z.upper, "strength": strength})
        return out

    i_cur = [0]

    def emit(
        category: str,
        s: Setup,
        ts: pd.Timestamp,
        title: str,
        message: str,
        price: float | None,
        state: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        events.append(
            EngineEvent(
                category=category,
                ts=ts,
                title=title,
                message=message,
                dedup_key=f"setup:{s.setup_id}:{state}",
                instrument=spec.symbol,
                setup_id=s.setup_id,
                price=price,
                details=details or {},
            )
        )

    def finalize(s: Setup, b: BarCtx) -> None:
        long = s.is_long
        zl, zu = float(s.zone["lower"]), float(s.zone["upper"])
        atr_i = b.atr if b.atr == b.atr else s.atr_at_breakout
        assert s.retest_price is not None  # set when the retest was detected
        retest = s.retest_price
        plan = build_trade_plan(
            s.direction,
            spec,
            zl,
            zu,
            float(s.breakout_bar["c"]),
            float(retest),
            b.c,
            float(retest),
            atr_i,
            opposing(s.direction, int(s.zone["zone_id"]), b.c, atr_i),
            settings.planning,
        )
        s.plan = plan
        vw = vwap_v[b.idx]
        vdist = None if math.isnan(vw) else ((b.c - vw) if long else (vw - b.c)) / atr_i
        trends = htf_trends_at(b.idx)
        factors = {
            "htf_alignment": factor_htf(trends, s.direction)
            if trends
            else (None, "All HTF filters disabled."),
            "zone_quality": factor_zone(
                s.zone.get("strength"), s.zone.get("strength_parts"), s.zone.get("reactions")
            ),
            "breakout_quality": factor_breakout(
                s.breakout_eval.metrics if s.breakout_eval else {}, settings.breakout.max_extension_atr
            ),
            "retest_quality": factor_retest(
                s.best_wick_ratio, s.precision, (s.retest_idx or 0) - s.breakout_idx, s.confirm_close_pos
            ),
            "vwap_alignment": factor_vwap(vdist, settings.scoring.vwap_stretch_atr),
            "relative_volume": factor_rvol(s.context.get("rvol")),
            "reward_risk": factor_rr(plan.rr if plan.risk_points > 0 else None, settings.planning.min_rr),
        }
        score = aggregate_score(factors, settings.scoring)
        s.score = score
        s.context.update(
            {
                "htf_trends_at_signal": trends,
                "vwap_at_signal": None if math.isnan(vw) else float(vw),
                "session": session_label(b.ts - tf_delta),
                "vol_regime": _vol_regime(atr_v, b.idx),
                "trend_regime": next(
                    (t for t in (trends.get("1h"), trends.get("4h"), trends.get("15min")) if t), None
                ),
                "atr_at_signal": atr_i,
            }
        )
        reasons: list[str] = []
        if not plan.valid:
            reasons += plan.invalid_reasons
        if not score.passed:
            reasons += score.reasons
        sess = settings.sessions
        if sess.signal_windows_enabled and not in_time_windows(
            b.ts - tf_delta, sess.signal_windows, sess.windows_timezone
        ):
            reasons.append(
                "Signal bar is outside the configured session windows "
                f"{sess.signal_windows} ({sess.windows_timezone})."
            )
        if minutes_until_session_close(b.ts) < sess.no_new_setups_minutes_before_close:
            reasons.append(f"Within {sess.no_new_setups_minutes_before_close} minutes of the session close.")
        blocked = block_qualification_from is not None and b.ts >= block_qualification_from
        if blocked:
            reasons.append(block_reason or "New signal qualification is disabled (data not reliable).")
        in_window = not sess.signal_windows_enabled or in_time_windows(
            b.ts - tf_delta, sess.signal_windows, sess.windows_timezone
        )
        s.context["qual_checks"] = {
            "plan_structural_valid": plan.structural_valid,
            "session_window_ok": in_window,
            "before_close_ok": minutes_until_session_close(b.ts) >= sess.no_new_setups_minutes_before_close,
            "not_blocked": not blocked,
            "score_data_ok": score.available_weight
            >= (
                settings.scoring.min_available_weight
                if settings.scoring.missing_factor_policy == "renormalize"
                else 0
            )
            and not any(f in score.missing for f in settings.scoring.essential_factors),
        }
        label = f"{spec.symbol} {s.direction.upper()} {plan.zone_lower}-{plan.zone_upper}"
        if reasons:
            s.rejection_reasons += reasons
            s.transition(SetupState.REJECTED, b.ts, b.idx, reasons[0])
            emit(
                "setup_rejected",
                s,
                b.ts,
                f"Setup rejected: {label}",
                f"Score {score.total:.0f}/100. " + " ".join(reasons[:3]),
                b.c,
                "rejected",
                {"score": score.total, "reasons": reasons},
            )
        else:
            s.transition(
                SetupState.QUALIFIED,
                b.ts,
                b.idx,
                f"Qualified with score {score.total:.0f}/100 (minimum {score.min_score:g}).",
            )
            emit(
                "setup_qualified",
                s,
                b.ts,
                f"Qualified setup: {label} | score {score.total:.0f}",
                f"Proposed entry {spec.fmt(plan.entry_price)}, stop {spec.fmt(plan.stop)}, "
                f"T1 {spec.fmt(plan.target1)} (R:R {plan.rr:.2f}). Advisory only - review manually; "
                "no orders are placed.",
                plan.entry_price,
                "qualified",
                {
                    "score": score.total,
                    "entry": plan.entry_price,
                    "stop": plan.stop,
                    "target1": plan.target1,
                    "rr": plan.rr,
                },
            )

    for i in range(n):
        i_cur[0] = i
        b = BarCtx(i, close_ts[i], o[i], h[i], lo[i], c[i], atr_v[i], tdates[i], bool(gap_before[i]))

        # 1. zones see the bar
        breaks = tracker.update_bar(i, b.ts, b.o, b.h, b.low, b.c, b.atr)

        # 2. pending setups advance
        still: list[Setup] = []
        for s in active:
            res = advance_setup(s, b, settings.retest, tick)
            for st in res.changed:
                if st == SetupState.RETEST_DETECTED:
                    emit(
                        "retest_detected",
                        s,
                        b.ts,
                        f"Retest: {spec.symbol} {s.direction} at {s.zone['lower']}-{s.zone['upper']}",
                        f"Price revisited the broken zone (extreme {s.retest_price}). Waiting for confirmation.",
                        s.retest_price,
                        "retest",
                    )
                elif st in (SetupState.INVALIDATED, SetupState.EXPIRED):
                    emit(
                        "setup_invalidated",
                        s,
                        b.ts,
                        f"Setup {st.value}: {spec.symbol} {s.direction}",
                        s.rejection_reasons[-1] if s.rejection_reasons else st.value,
                        b.c,
                        st.value,
                    )
            if res.needs_finalize:
                finalize(s, b)
            if not s.is_terminal:
                still.append(s)
        active = still

        # 3. new setups from first-time breaks of relevant zones
        chosen: dict[str, Any] = {}
        for br in breaks:
            z = br.zone
            relevant, _why = tracker.relevance(z, i, b.c, b.atr)
            if not relevant:
                continue
            direction = "long" if br.direction == "up" else "short"
            prev = chosen.get(direction)
            edge = z.upper if direction == "long" else z.lower
            if (
                prev is None
                or (direction == "long" and edge > prev[1])
                or (direction == "short" and edge < prev[1])
            ):
                chosen[direction] = (z, edge)
        for direction, (z, _edge) in chosen.items():
            sid = make_setup_id(spec.symbol, contract, tf, direction, b.ts)
            if sid in setups:
                continue
            zd = zone_to_dict(z, tracker, i, b.c, b.atr)
            zone_frozen = {
                k: zd[k]
                for k in (
                    "zone_id",
                    "lower",
                    "upper",
                    "strength",
                    "strength_parts",
                    "reactions",
                    "tests",
                    "holds",
                    "failures",
                    "origin",
                )
            }
            zone_frozen["kind_before_break"] = "resistance" if direction == "long" else "support"
            s = Setup(
                setup_id=sid,
                instrument=spec.symbol,
                contract=contract,
                timeframe=tf,
                direction=direction,
                zone=zone_frozen,
                breakout_idx=i,
                breakout_ts=b.ts,
                breakout_bar={"o": b.o, "h": b.h, "l": b.low, "c": b.c},
                atr_at_breakout=b.atr,
                trade_date=b.trade_date,
            )
            s.transition(
                SetupState.LEVEL_IDENTIFIED,
                z.last_reaction_ts or z.created_ts,
                z.last_reaction_idx,
                f"Zone {z.lower}-{z.upper} with {z.reactions} reaction(s).",
            )
            s.transition(SetupState.BREAKOUT_DETECTED, b.ts, i, f"Close {b.c} beyond the zone.")
            opp = opposing(direction, z.zone_id, b.c, b.atr)
            if direction == "long":
                next_edge = min((x["lower"] for x in opp), default=None)
            else:
                next_edge = max((x["upper"] for x in opp), default=None)
            trends = htf_trends_at(i)
            htf_raw, _ = factor_htf(trends, direction) if trends else (None, "")
            rv = rvol_v[i]
            s.context["rvol"] = None if math.isnan(rv) else float(rv)
            s.context["htf_trends_at_breakout"] = trends
            ev = evaluate_breakout(
                direction,
                z.lower,
                z.upper,
                b.o,
                b.h,
                b.low,
                b.c,
                b.atr,
                tick,
                settings.breakout,
                vwap=None if math.isnan(vwap_v[i]) else float(vwap_v[i]),
                rvol=s.context["rvol"],
                htf_alignment=htf_raw,
                next_opposing_edge=next_edge,
            )
            s.breakout_eval = ev
            setups[sid] = s
            if ev.passed:
                s.transition(SetupState.AWAITING_RETEST, b.ts, i, "Breakout passed quality checks.")
                emit(
                    "breakout_detected",
                    s,
                    b.ts,
                    f"Breakout: {spec.symbol} {'above resistance' if direction == 'long' else 'below support'} "
                    f"{z.lower}-{z.upper}",
                    f"Completed candle closed at {b.c}. Waiting for a retest (max "
                    f"{settings.retest.max_bars_to_retest} bars).",
                    b.c,
                    "breakout",
                )
                active.append(s)
            else:
                reasons = [ch.message for ch in ev.failed_required] or ["Breakout failed quality checks."]
                s.rejection_reasons += reasons
                s.transition(SetupState.REJECTED, b.ts, i, "Breakout failed quality checks: " + reasons[0])
                emit(
                    "setup_rejected",
                    s,
                    b.ts,
                    f"Breakout rejected: {spec.symbol} {direction}",
                    " ".join(reasons[:3]),
                    b.c,
                    "rejected_breakout",
                )

        # 4. swings confirmed at this bar's close join the zone tracker
        for sw in by_confirm.get(i, []):
            tracker.add_swing(sw, i, b.ts, b.c, b.atr)

    last = n - 1
    result.setups = list(setups.values())
    result.events = events
    result.zones = tracker.snapshot(last, c[last], atr_v[last])
    result.zone_history = tracker.snapshot(last, c[last], atr_v[last], include_dead=True)
    return result


def split_contract_segments(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split a frame into consecutive single-contract segments (never merged)."""
    if "contract" not in df.columns or df.empty:
        return [("UNKNOWN", df)]
    c = df["contract"].astype(str)
    seg_id = (c != c.shift(1)).cumsum()
    return [(str(g["contract"].iloc[0]), g) for _, g in df.groupby(seg_id, sort=True)]
