"""Historical backtester with explicit look-ahead safeguards.

* Signals come from :func:`core.scanner.run_scan`, which is causal: swings
  are only used after confirmation, HTF bars only after they close, zones
  only with reactions seen so far.
* A hypothetical trade starts at the confirmation bar's CLOSE (or, for
  ``zone_edge`` entries, a later bar that trades through the limit) and is
  managed from the NEXT bar.
* Intrabar ambiguity is resolved conservatively: if a bar touches both the
  stop and the target, the stop is assumed to fill first. Gaps through the
  stop fill at the (worse) open; gaps through the target fill at the target.
* Simulated breakeven / trailing rules (optional, OFF by default) take
  effect from the bar AFTER the trigger, never within the same bar.
* Only one position at a time; signals during an open trade are skipped.
* Commission per contract round trip, and slippage ticks on entry and on
  stop/market exits.

These are hypothetical results. Synthetic data, a single historical period
or a small sample cannot establish that a strategy is profitable.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import AppSettings
from core.instruments import InstrumentSpec
from core.performance import breakdown, compute_metrics, equity_curve, score_bucket
from core.retest_engine import Setup
from core.scanner import ScanResult, run_scan
from core.scoring import aggregate_score, factor_rr
from core.swing_detection import detect_swings, params_from_settings
from core.timeutils import EXCHANGE_TZ_NAME, parse_hhmm, timeframe_delta, trade_dates


@dataclass
class BacktestConfig:
    quantity: int = 1
    commission_rt: float = 0.0
    slippage_ticks_per_side: float = 1.0
    force_flat_time: str | None = "16:00"  # ET
    max_holding_bars: int | None = None
    allow_overlap: bool = False
    simulate_breakeven: bool = False
    breakeven_r: float = 1.0
    simulate_trailing: bool = False
    entry_window_bars: int = 6
    oos_fraction: float = 0.3
    min_score: float | None = None  # None -> settings.scoring.min_score
    min_rr: float | None = None  # None -> settings.planning.min_rr


@dataclass
class SimTrade:
    setup_id: str
    direction: str
    entry_idx: int
    entry_ts: pd.Timestamp
    entry_price: float
    stop: float
    target: float
    quantity: int
    exit_idx: int | None = None
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str = "open"
    ambiguous: bool = False
    bars_held: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    be_moved: bool = False
    trail_moves: int = 0


@dataclass
class BacktestResult:
    scan: ScanResult
    trades: pd.DataFrame
    metrics: dict
    metrics_is: dict
    metrics_oos: dict
    oos_cutoff: pd.Timestamp | None
    breakdowns: dict[str, pd.DataFrame]
    equity: pd.DataFrame
    params: dict
    warnings: list[str] = field(default_factory=list)


class _Arrays:
    def __init__(self, df: pd.DataFrame, timeframe: str):
        self.idx = df.index
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.lo = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.tf = timeframe_delta(timeframe)
        self.close_ts = df.index + self.tf
        local = df.index.tz_convert(EXCHANGE_TZ_NAME)
        self.minute = np.asarray(local.hour * 60 + local.minute)
        self.tdate = trade_dates(df.index)


def rescore(
    setup: Setup, settings: AppSettings, min_score: float, min_rr: float
) -> tuple[bool, float | None]:
    """Re-apply alternative thresholds to a scanned setup without re-scanning.

    Returns (eligible, score). Only threshold-dependent parts change:
    the reward/risk factor (which depends on the minimum R:R) and the
    minimum score. Structural, session and data checks are reused as stored.
    """
    if setup.score is None or setup.plan is None:
        return False, None
    checks = setup.context.get("qual_checks") or {}
    if not all(checks.values()) or not setup.plan.structural_valid:
        return False, setup.score.total
    rr = setup.plan.rr
    if rr is None or rr < min_rr - 1e-9:
        return False, setup.score.total
    factors = {c.factor: (c.raw, c.explanation) for c in setup.score.components}
    factors["reward_risk"] = factor_rr(rr, min_rr)
    cfg = settings.scoring.model_copy(update={"min_score": min_score})
    res = aggregate_score(factors, cfg)
    return res.passed, res.total


def simulate_trade(
    setup: Setup,
    arr: _Arrays,
    spec: InstrumentSpec,
    cfg: BacktestConfig,
    swings_low_by_confirm: dict[int, list[float]] | None = None,
    swings_high_by_confirm: dict[int, list[float]] | None = None,
    atr: np.ndarray | None = None,
    end_idx: int | None = None,
) -> SimTrade | None:
    plan = setup.plan
    if plan is None or plan.target1 is None or setup.confirm_idx is None:
        return None
    long = setup.is_long
    sign = 1 if long else -1
    tick = spec.tick_size
    slip = cfg.slippage_ticks_per_side * tick
    n = len(arr.c) if end_idx is None else min(end_idx + 1, len(arr.c))
    ci = setup.confirm_idx
    flat_min = None
    if cfg.force_flat_time:
        t = parse_hhmm(cfg.force_flat_time)
        flat_min = t.hour * 60 + t.minute

    # ---- entry -------------------------------------------------------------
    if plan.entry_method == "confirmation_close":
        entry_idx = ci
        entry_px = arr.c[ci] + sign * slip
        entry_ts = arr.close_ts[ci]
        start = ci + 1
    else:
        limit = plan.entry_price
        entry_idx = None
        for j in range(ci + 1, min(ci + 1 + cfg.entry_window_bars, n)):
            touched = arr.lo[j] <= limit if long else arr.h[j] >= limit
            if touched:
                entry_idx = j
                break
        if entry_idx is None:
            tr = SimTrade(
                setup.setup_id,
                setup.direction,
                ci,
                arr.close_ts[ci],
                float("nan"),
                plan.stop,
                plan.target1,
                cfg.quantity,
                exit_reason="not_filled",
            )
            return tr
        entry_px = limit
        entry_ts = arr.idx[entry_idx]  # filled during this bar
        start = entry_idx
    stop = plan.stop
    target = plan.target1
    risk = (entry_px - stop) * sign
    if risk <= 0:
        return None
    assert entry_idx is not None  # filled above (or returned "not_filled")
    tr = SimTrade(
        setup.setup_id, setup.direction, entry_idx, entry_ts, float(entry_px), stop, target, cfg.quantity
    )
    entry_tdate = arr.tdate[entry_idx]
    pending_stop: float | None = None

    for k in range(start, n):
        if pending_stop is not None:
            stop = pending_stop
            pending_stop = None
        o, h, lo, c = arr.o[k], arr.h[k], arr.lo[k], arr.c[k]
        # session flat: exit at the open of the first bar at/after the flat time or in a new session
        if k > entry_idx and (
            arr.tdate[k] != entry_tdate or (flat_min is not None and flat_min <= arr.minute[k] < 18 * 60)
        ):
            tr.exit_idx, tr.exit_ts, tr.exit_price, tr.exit_reason = (
                k,
                arr.idx[k],
                o - sign * slip,
                "session_flat",
            )
            break
        fav = ((h - entry_px) if long else (entry_px - lo)) / risk
        adv = ((lo - entry_px) if long else (entry_px - h)) / risk
        same_bar_entry = k == entry_idx
        if long:
            gap_stop = o <= stop and not same_bar_entry
            gap_tgt = o >= target and not same_bar_entry
            hit_stop = lo <= stop
            hit_tgt = h >= target
        else:
            gap_stop = o >= stop and not same_bar_entry
            gap_tgt = o <= target and not same_bar_entry
            hit_stop = h >= stop
            hit_tgt = lo <= target
        if same_bar_entry:
            # Limit filled inside this bar: the target may have traded before the fill, so it is not
            # credited on the fill bar (the stop still counts - conservative).
            hit_tgt = False
        tr.mae_r = min(tr.mae_r, adv)
        if gap_stop:
            tr.exit_idx, tr.exit_ts, tr.exit_price, tr.exit_reason = (
                k,
                arr.idx[k],
                o - sign * slip,
                "stop_gap",
            )
            break
        if gap_tgt:
            tr.mfe_r = max(tr.mfe_r, fav)
            tr.exit_idx, tr.exit_ts, tr.exit_price, tr.exit_reason = k, arr.idx[k], target, "target_gap"
            break
        if hit_stop and hit_tgt:
            tr.exit_idx, tr.exit_ts, tr.exit_price = k, arr.close_ts[k], stop - sign * slip
            tr.exit_reason, tr.ambiguous = "stop_ambiguous", True
            break
        if hit_stop:
            tr.exit_idx, tr.exit_ts, tr.exit_price, tr.exit_reason = (
                k,
                arr.close_ts[k],
                stop - sign * slip,
                (
                    "breakeven_stop"
                    if tr.be_moved and tr.trail_moves == 0
                    else ("trailing_stop" if tr.trail_moves else "stop")
                ),
            )
            break
        if hit_tgt:
            tr.mfe_r = max(tr.mfe_r, fav)
            tr.exit_idx, tr.exit_ts, tr.exit_price, tr.exit_reason = k, arr.close_ts[k], target, "target"
            break
        tr.mfe_r = max(tr.mfe_r, fav)
        tr.bars_held += 1
        if cfg.max_holding_bars and tr.bars_held >= cfg.max_holding_bars:
            tr.exit_idx, tr.exit_ts, tr.exit_price, tr.exit_reason = (
                k,
                arr.close_ts[k],
                c - sign * slip,
                "time",
            )
            break
        # management rules (effective next bar)
        new_stop = stop
        if cfg.simulate_breakeven and not tr.be_moved and fav >= cfg.breakeven_r:
            be = entry_px
            if (long and be > new_stop) or (not long and be < new_stop):
                new_stop = be
                tr.be_moved = True
        if cfg.simulate_trailing and atr is not None:
            book = swings_low_by_confirm if long else swings_high_by_confirm
            for price in (book or {}).get(k, []):
                a = atr[k] if not math.isnan(atr[k]) else 0.0
                buf = max(2 * tick, 0.25 * a)
                cand = spec.round_price(price - buf, "down") if long else spec.round_price(price + buf, "up")
                if (long and new_stop < cand < c) or (not long and c < cand < new_stop):
                    new_stop = cand
                    tr.trail_moves += 1
        if new_stop != stop:
            pending_stop = new_stop
    else:
        k = n - 1
        tr.exit_idx, tr.exit_ts, tr.exit_price, tr.exit_reason = k, arr.close_ts[k], arr.c[k], "end_of_data"
    return tr


def _trade_row(
    tr: SimTrade, setup: Setup, spec: InstrumentSpec, cfg: BacktestConfig, score: float | None
) -> dict:
    sign = 1 if tr.direction == "long" else -1
    filled = tr.exit_reason != "not_filled"
    pts = (tr.exit_price - tr.entry_price) * sign if filled and tr.exit_price is not None else 0.0
    gross = pts * spec.point_value * tr.quantity
    costs = cfg.commission_rt * tr.quantity if filled else 0.0
    risk_pts = (tr.entry_price - tr.stop) * sign if filled else float("nan")
    r = pts / risk_pts if filled and risk_pts > 0 else float("nan")
    ctx = setup.context
    return {
        "setup_id": tr.setup_id,
        "direction": tr.direction,
        "entry_ts": tr.entry_ts,
        "entry_price": tr.entry_price,
        "exit_ts": tr.exit_ts,
        "exit_price": tr.exit_price,
        "stop": tr.stop,
        "target": tr.target,
        "quantity": tr.quantity,
        "pnl_points": round(pts, 6),
        "pnl_gross": round(gross, 2),
        "costs": round(costs, 2),
        "pnl_net": round(gross - costs, 2),
        "r_multiple": round(r, 4) if r == r else float("nan"),
        "exit_reason": tr.exit_reason,
        "ambiguous": tr.ambiguous,
        "bars_held": tr.bars_held,
        "mfe_r": round(tr.mfe_r, 3),
        "mae_r": round(tr.mae_r, 3),
        "session": ctx.get("session"),
        "score": score,
        "score_bucket": score_bucket(score),
        "vol_regime": ctx.get("vol_regime"),
        "trend_regime": ctx.get("trend_regime"),
        "instrument": setup.instrument,
        "be_moved": tr.be_moved,
        "trail_moves": tr.trail_moves,
    }


def simulate_setups(
    scan: ScanResult, df: pd.DataFrame, spec: InstrumentSpec, settings: AppSettings, cfg: BacktestConfig
) -> pd.DataFrame:
    """Simulate every setup eligible under cfg thresholds (one position at a time)."""
    min_score = cfg.min_score if cfg.min_score is not None else settings.scoring.min_score
    min_rr = cfg.min_rr if cfg.min_rr is not None else settings.planning.min_rr
    arr = _Arrays(df, settings.timeframes.entry_timeframe)
    lows: dict[int, list[float]] = {}
    highs: dict[int, list[float]] = {}
    atr = scan.indicators["atr"].to_numpy(float) if not scan.indicators.empty else None
    if cfg.simulate_trailing and not scan.swings.empty:
        for r in scan.swings[scan.swings["status"] == "confirmed"].itertuples():
            (lows if r.swing_type == "low" else highs).setdefault(int(r.confirm_idx), []).append(
                float(r.price)
            )
    cands = []
    for s in scan.setups:
        ok, sc = rescore(s, settings, min_score, min_rr)
        if ok:
            cands.append((s, sc))
    cands.sort(key=lambda x: x[0].confirm_idx or 0)
    rows = []
    busy_until: pd.Timestamp | None = None
    for s, sc in cands:
        if (
            not cfg.allow_overlap
            and busy_until is not None
            and s.confirm_ts is not None
            and s.confirm_ts < busy_until
        ):
            continue
        tr = simulate_trade(s, arr, spec, cfg, lows, highs, atr)
        if tr is None:
            continue
        if tr.exit_reason == "not_filled":
            continue
        rows.append(_trade_row(tr, s, spec, cfg, sc))
        busy_until = tr.exit_ts
    return pd.DataFrame(rows)


def run_backtest(
    df: pd.DataFrame,
    spec: InstrumentSpec,
    settings: AppSettings,
    cfg: BacktestConfig,
    scan: ScanResult | None = None,
    data_is_synthetic: bool = False,
) -> BacktestResult:
    scan = scan or run_scan(df, spec, settings)
    warnings: list[str] = []
    if not scan.scanned:
        warnings.append(scan.message)
        empty = pd.DataFrame()
        return BacktestResult(
            scan,
            empty,
            {"trades": 0},
            {"trades": 0},
            {"trades": 0},
            None,
            {},
            empty,
            _params(cfg, settings),
            warnings,
        )
    trades = simulate_setups(scan, df, spec, settings, cfg)
    cutoff = None
    if len(df) and 0 < cfg.oos_fraction < 1:
        start, end = df.index[0], df.index[-1]
        cutoff = start + (end - start) * (1 - cfg.oos_fraction)
    if not trades.empty:
        trades["sample"] = np.where(trades["entry_ts"] < cutoff, "IS", "OOS") if cutoff is not None else "IS"
    metrics = compute_metrics(trades)
    m_is = compute_metrics(trades[trades["sample"] == "IS"]) if not trades.empty else {"trades": 0}
    m_oos = compute_metrics(trades[trades["sample"] == "OOS"]) if not trades.empty else {"trades": 0}
    bds = {
        k: breakdown(trades, k)
        for k in (
            "session",
            "instrument",
            "score_bucket",
            "vol_regime",
            "trend_regime",
            "exit_reason",
            "direction",
        )
    }
    if data_is_synthetic:
        warnings.append(
            "SYNTHETIC DATA: these results describe simulated prices and say nothing about real-market "
            "profitability."
        )
    if metrics.get("trades", 0) < 30:
        warnings.append(f"Small sample ({metrics.get('trades', 0)} trades): statistics are highly uncertain.")
    warnings.append("Single historical period: results can differ materially in other regimes.")
    if metrics.get("ambiguous_bars"):
        warnings.append(
            f"{metrics['ambiguous_bars']} trade(s) touched stop and target in the same bar and were "
            "counted as losses (conservative assumption)."
        )
    return BacktestResult(
        scan,
        trades,
        metrics,
        m_is,
        m_oos,
        cutoff,
        bds,
        equity_curve(trades),
        _params(cfg, settings),
        warnings,
    )


def _params(cfg: BacktestConfig, settings: AppSettings) -> dict:
    return {
        **cfg.__dict__,
        "effective_min_score": cfg.min_score if cfg.min_score is not None else settings.scoring.min_score,
        "effective_min_rr": cfg.min_rr if cfg.min_rr is not None else settings.planning.min_rr,
    }


def threshold_sensitivity(
    scan: ScanResult,
    df: pd.DataFrame,
    spec: InstrumentSpec,
    settings: AppSettings,
    cfg: BacktestConfig,
    scores: list[float],
    rrs: list[float],
) -> pd.DataFrame:
    """Re-simulate a grid of (min_score, min_rr) using one scan; split IS/OOS."""
    rows = []
    cutoff = None
    if len(df) and 0 < cfg.oos_fraction < 1:
        cutoff = df.index[0] + (df.index[-1] - df.index[0]) * (1 - cfg.oos_fraction)
    for sc in scores:
        for rr in rrs:
            c2 = copy.copy(cfg)
            c2.min_score, c2.min_rr = sc, rr
            t = simulate_setups(scan, df, spec, settings, c2)
            for sample in ("IS", "OOS"):
                if t.empty:
                    sub = t
                elif cutoff is None:
                    sub = t if sample == "IS" else t.iloc[0:0]
                else:
                    sub = t[(t["entry_ts"] < cutoff) if sample == "IS" else (t["entry_ts"] >= cutoff)]
                m = compute_metrics(sub)
                rows.append(
                    {
                        "min_score": sc,
                        "min_rr": rr,
                        "sample": sample,
                        "trades": m.get("trades", 0),
                        "win_rate": m.get("win_rate"),
                        "expectancy_r": m.get("expectancy_r"),
                        "profit_factor": m.get("profit_factor"),
                        "net_pnl": m.get("net_pnl"),
                    }
                )
    return pd.DataFrame(rows)


def structural_sensitivity(
    df: pd.DataFrame,
    spec: InstrumentSpec,
    settings: AppSettings,
    cfg: BacktestConfig,
    right_bars: list[int],
    tolerance_atr: list[float],
    progress=None,
) -> pd.DataFrame:
    """Re-scan for each (swing right_bars, zone tolerance ATR) combination."""
    rows = []
    total = len(right_bars) * len(tolerance_atr)
    done = 0
    for rb in right_bars:
        for tol in tolerance_atr:
            s2 = settings.updated("swings", right_bars=rb).updated("zones", tolerance_atr_mult=tol)
            res = run_backtest(df, spec, s2, cfg)
            for sample, m in (("IS", res.metrics_is), ("OOS", res.metrics_oos)):
                rows.append(
                    {
                        "right_bars": rb,
                        "tolerance_atr": tol,
                        "sample": sample,
                        "trades": m.get("trades", 0),
                        "expectancy_r": m.get("expectancy_r"),
                        "win_rate": m.get("win_rate"),
                        "profit_factor": m.get("profit_factor"),
                    }
                )
            done += 1
            if progress:
                progress(done / total)
    return pd.DataFrame(rows)


def evaluate_signal_outcome(
    setup: Setup,
    df: pd.DataFrame,
    spec: InstrumentSpec,
    settings: AppSettings,
    cfg: BacktestConfig | None = None,
) -> dict | None:
    """Hypothetical outcome of a setup's proposed plan (labelled HYPOTHETICAL everywhere)."""
    if (
        setup.plan is None
        or setup.confirm_idx is None
        or setup.plan.target1 is None
        or not setup.plan.structural_valid
    ):
        return None
    cfg = cfg or BacktestConfig(force_flat_time=settings.backtest.force_flat_time)
    arr = _Arrays(df, settings.timeframes.entry_timeframe)
    if setup.confirm_idx >= len(arr.c):
        return None
    tr = simulate_trade(setup, arr, spec, cfg)
    if tr is None:
        return None
    sign = 1 if setup.is_long else -1
    if tr.exit_reason == "not_filled":
        return {
            "outcome": "not_filled",
            "evaluated_through_ts": arr.close_ts[-1],
            "notes": "HYPOTHETICAL - limit entry never traded.",
        }
    risk = (tr.entry_price - tr.stop) * sign
    r = (tr.exit_price - tr.entry_price) * sign / risk if tr.exit_price is not None and risk > 0 else None
    outcome = "open" if tr.exit_reason == "end_of_data" else tr.exit_reason
    return {
        "outcome": outcome,
        "entry_ts": tr.entry_ts,
        "exit_ts": tr.exit_ts if outcome != "open" else None,
        "exit_price": tr.exit_price if outcome != "open" else None,
        "r_result": round(r, 3) if r is not None else None,
        "mfe_r": round(tr.mfe_r, 3),
        "mae_r": round(tr.mae_r, 3),
        "bars_held": tr.bars_held,
        "ambiguous": tr.ambiguous,
        "evaluated_through_ts": arr.close_ts[-1],
        "notes": "HYPOTHETICAL outcome of the proposed plan - not an executed trade.",
    }


def swings_for_trailing(df: pd.DataFrame, spec: InstrumentSpec, settings: AppSettings) -> pd.DataFrame:
    return detect_swings(df, spec.tick_size, params_from_settings(settings.swings))
