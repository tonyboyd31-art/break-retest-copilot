"""Performance statistics for backtests, hypothetical signal outcomes and journaled trades."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SCORE_BUCKETS = [(0, 70, "<70"), (70, 80, "70-79"), (80, 85, "80-84"), (85, 90, "85-89"), (90, 101, "90+")]


def score_bucket(score: float | None) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "n/a"
    for lo, hi, label in SCORE_BUCKETS:
        if lo <= score < hi:
            return label
    return "n/a"


def _streak(signs: list[bool], value: bool) -> int:
    best = cur = 0
    for s in signs:
        cur = cur + 1 if s == value else 0
        best = max(best, cur)
    return best


def compute_metrics(trades: pd.DataFrame, pnl_col: str = "pnl_net", r_col: str = "r_multiple") -> dict:
    """Standard metrics. ``trades`` needs pnl_col; r_col, entry_ts, exit_ts optional."""
    n = 0 if trades is None else len(trades)
    if n == 0:
        return {"trades": 0}
    pnl = trades[pnl_col].astype(float).to_numpy()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(losses.sum())
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
    dd = equity - peak
    out = {
        "trades": int(n),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "scratch": int((pnl == 0).sum()),
        "win_rate": float((pnl > 0).mean()),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(float(pnl.sum()), 2),
        "profit_factor": round(gross_profit / abs(gross_loss), 3)
        if gross_loss < 0
        else (math.inf if gross_profit > 0 else None),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else None,
        "expectancy": round(float(pnl.mean()), 2),
        "max_drawdown": round(float(dd.min()), 2) if len(dd) else 0.0,
        "max_losing_streak": _streak(list(pnl < 0), True),
        "max_winning_streak": _streak(list(pnl > 0), True),
    }
    if r_col in trades:
        r = trades[r_col].astype(float).to_numpy()
        req = np.cumsum(r)
        rpeak = np.maximum.accumulate(np.concatenate([[0.0], req]))[1:]
        out["expectancy_r"] = round(float(np.nanmean(r)), 3)
        out["total_r"] = round(float(np.nansum(r)), 2)
        out["max_drawdown_r"] = round(float((req - rpeak).min()), 2)
    if "entry_ts" in trades and "exit_ts" in trades:
        hold = pd.to_datetime(trades["exit_ts"], utc=True) - pd.to_datetime(trades["entry_ts"], utc=True)
        out["avg_holding_minutes"] = round(float(hold.dt.total_seconds().mean() / 60.0), 1)
    if "costs" in trades:
        out["total_costs"] = round(float(trades["costs"].sum()), 2)
    if "ambiguous" in trades:
        out["ambiguous_bars"] = int(trades["ambiguous"].astype(bool).sum())
    return out


def breakdown(
    trades: pd.DataFrame, by: str, pnl_col: str = "pnl_net", r_col: str = "r_multiple"
) -> pd.DataFrame:
    if trades is None or trades.empty or by not in trades:
        return pd.DataFrame()
    rows = []
    for key, g in trades.groupby(trades[by].fillna("n/a"), sort=True):
        m = compute_metrics(g, pnl_col, r_col)
        rows.append(
            {
                by: key,
                "trades": m["trades"],
                "win_rate": m.get("win_rate"),
                "net_pnl": m.get("net_pnl"),
                "expectancy": m.get("expectancy"),
                "expectancy_r": m.get("expectancy_r"),
                "profit_factor": m.get("profit_factor"),
            }
        )
    return pd.DataFrame(rows)


def equity_curve(trades: pd.DataFrame, pnl_col: str = "pnl_net", ts_col: str = "exit_ts") -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame(columns=["ts", "equity", "drawdown"])
    t = trades.sort_values(ts_col)
    eq = t[pnl_col].astype(float).cumsum().to_numpy()
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return pd.DataFrame({"ts": t[ts_col].to_numpy(), "equity": eq, "drawdown": eq - peak})
