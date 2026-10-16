"""Running, persisting and evaluating backtests and hypothetical signal outcomes."""

from __future__ import annotations

import pandas as pd

from core.backtesting import BacktestConfig, BacktestResult, evaluate_signal_outcome, run_backtest
from core.config import AppSettings
from core.scanner import ScanResult
from data import repositories as repo
from data.database import Database
from services.market_service import MarketData


def config_from_settings(settings: AppSettings, spec_symbol: str, **overrides) -> BacktestConfig:
    b = settings.backtest
    cfg = BacktestConfig(
        quantity=b.quantity,
        commission_rt=settings.risk.commission_for(spec_symbol),
        slippage_ticks_per_side=settings.risk.slippage_ticks_per_side,
        force_flat_time=b.force_flat_time,
        max_holding_bars=b.max_holding_bars,
        allow_overlap=b.allow_overlap,
        simulate_breakeven=b.simulate_breakeven,
        breakeven_r=(settings.management.breakeven_thresholds_r or [1.0])[0],
        simulate_trailing=b.simulate_trailing,
        oos_fraction=b.oos_fraction,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def run_and_save(
    db: Database, md: MarketData, settings: AppSettings, cfg: BacktestConfig, name: str, save: bool = True
) -> tuple[BacktestResult, int | None]:
    res = run_backtest(md.segment, md.spec, settings, cfg, data_is_synthetic=md.feed_type == "simulated")
    run_id = None
    if save:
        bds = {k: v.to_dict("records") for k, v in res.breakdowns.items()}
        metrics = {
            "all": res.metrics,
            "in_sample": res.metrics_is,
            "out_of_sample": res.metrics_oos,
            "oos_cutoff": res.oos_cutoff.isoformat() if res.oos_cutoff is not None else None,
            "warnings": res.warnings,
        }
        run_id = repo.save_backtest(
            db,
            name=name,
            instrument=md.spec.symbol,
            data_source=md.selection.source,
            dataset_ref=str(md.selection.dataset_id) if md.selection.dataset_id else None,
            contract=md.contract,
            start_ts=md.segment.index[0],
            end_ts=md.segment.index[-1],
            settings=settings,
            params=res.params,
            metrics=metrics,
            breakdowns=bds,
            trades=res.trades,
        )
    return res, run_id


def evaluate_outcomes(db: Database, md: MarketData, scan: ScanResult, settings: AppSettings) -> int:
    """Store HYPOTHETICAL outcomes for qualified setups in the current scan."""
    n = 0
    for s in scan.setups:
        if s.state.value != "qualified":
            continue
        out = evaluate_signal_outcome(s, md.visible, md.spec, settings)
        if out is not None:
            repo.upsert_signal_outcome(db, s.setup_id, out)
            n += 1
    return n


def runs_frame(db: Database) -> pd.DataFrame:
    rows = repo.list_backtests(db)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "id": r["id"],
                "name": r["name"],
                "created": r["created_at"][:16],
                "instrument": r["instrument"],
                "source": r["data_source"],
                "contract": r["contract"],
                "settings_hash": r["settings_hash"],
                "trades": r["metrics"].get("all", {}).get("trades"),
                "expectancy_r": r["metrics"].get("all", {}).get("expectancy_r"),
                "oos_expectancy_r": r["metrics"].get("out_of_sample", {}).get("expectancy_r"),
            }
            for r in rows
        ]
    )
