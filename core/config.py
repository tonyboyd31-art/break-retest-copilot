"""Validated application settings.

All strategy, scoring, risk and notification parameters live here as
Pydantic models so invalid combinations are rejected before they reach the
engine. Settings are persisted as JSON in SQLite (see ``data.database``).

``AppSettings.strategy_hash()`` fingerprints every parameter that can change
which signals are generated; backtests store it so the UI can warn when the
current settings no longer match a saved backtest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.events import ALERT_CATEGORIES, SEVERITIES
from core.timeutils import parse_hhmm, validate_timezone

SCORE_FACTORS: dict[str, str] = {
    "htf_alignment": "Higher-timeframe trend alignment",
    "zone_quality": "Support/resistance quality",
    "breakout_quality": "Breakout candle quality",
    "retest_quality": "Retest quality",
    "vwap_alignment": "VWAP alignment",
    "relative_volume": "Relative volume",
    "reward_risk": "Available reward/risk",
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "htf_alignment": 20,
    "zone_quality": 15,
    "breakout_quality": 15,
    "retest_quality": 20,
    "vwap_alignment": 10,
    "relative_volume": 5,
    "reward_risk": 15,
}


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class GeneralSettings(_Base):
    default_instrument: str = "MNQ"
    display_timezone: str = "America/New_York"

    @field_validator("display_timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        return validate_timezone(v)


class TimeframeSettings(_Base):
    entry_timeframe: Literal["5min"] = "5min"
    use_15m: bool = True
    use_1h: bool = True
    use_4h: bool = True
    trend_ema_fast: int = Field(default=20, ge=2, le=200)
    trend_ema_slow: int = Field(default=50, ge=3, le=400)
    trend_slope_bars: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def _ema_order(self) -> TimeframeSettings:
        if self.trend_ema_fast >= self.trend_ema_slow:
            raise ValueError("trend_ema_fast must be smaller than trend_ema_slow")
        return self

    def enabled_htfs(self) -> list[str]:
        out = []
        if self.use_15m:
            out.append("15min")
        if self.use_1h:
            out.append("1h")
        if self.use_4h:
            out.append("4h")
        return out


class SwingSettings(_Base):
    left_bars: int = Field(default=3, ge=1, le=20)
    right_bars: int = Field(default=3, ge=1, le=20)
    min_prominence_ticks: float = Field(default=8, ge=0)
    use_atr_prominence: bool = True
    atr_prominence_mult: float = Field(default=0.5, ge=0, le=10)
    duplicate_window_bars: int = Field(default=6, ge=0, le=100)
    duplicate_tolerance_ticks: float = Field(default=4, ge=0)
    atr_period: int = Field(default=14, ge=2, le=200)


class ZoneSettings(_Base):
    tolerance_ticks: float = Field(default=6, ge=0)
    tolerance_atr_mult: float = Field(default=0.25, ge=0, le=5)
    max_zone_width_atr: float = Field(default=1.0, gt=0, le=10)
    min_zone_width_ticks: float = Field(default=2, ge=0)
    min_reaction_gap_bars: int = Field(default=6, ge=1, le=500)
    break_buffer_ticks: float = Field(default=2, ge=0)
    break_buffer_atr_mult: float = Field(default=0.05, ge=0, le=5)
    min_reactions: int = Field(default=2, ge=1, le=20)
    # Zones need at least this many reactions to count as an obstacle for
    # breakout room and targets (1 = every confirmed swing level blocks).
    opposing_min_reactions: int = Field(default=2, ge=1, le=20)
    max_age_bars: int = Field(default=1500, ge=50, le=50000)
    relevance_max_distance_atr: float = Field(default=12, gt=0, le=200)
    recency_half_life_bars: int = Field(default=576, ge=10, le=50000)
    reactions_saturation: int = Field(default=4, ge=1, le=20)
    rejection_saturation_atr: float = Field(default=2.0, gt=0, le=20)
    w_reactions: float = Field(default=0.35, ge=0)
    w_recency: float = Field(default=0.20, ge=0)
    w_rejection: float = Field(default=0.25, ge=0)
    w_holds: float = Field(default=0.20, ge=0)

    @model_validator(mode="after")
    def _weights(self) -> ZoneSettings:
        if self.w_reactions + self.w_recency + self.w_rejection + self.w_holds <= 0:
            raise ValueError("At least one zone-strength weight must be positive")
        return self


class BreakoutSettings(_Base):
    close_buffer_ticks: float = Field(default=2, ge=0)
    close_buffer_atr_mult: float = Field(default=0.05, ge=0, le=5)
    min_body_ratio: float = Field(default=0.5, ge=0, le=1)
    min_range_atr: float = Field(default=0.5, ge=0, le=20)
    max_range_atr: float = Field(default=3.0, gt=0, le=50)
    max_extension_atr: float = Field(default=1.5, gt=0, le=50)
    min_room_atr: float = Field(default=1.5, ge=0, le=100)
    require_htf_alignment: bool = False
    require_vwap_alignment: bool = False
    require_relative_volume: bool = False
    min_relative_volume: float = Field(default=1.0, ge=0, le=20)
    rvol_lookback: int = Field(default=20, ge=2, le=500)

    @model_validator(mode="after")
    def _range(self) -> BreakoutSettings:
        if self.min_range_atr >= self.max_range_atr:
            raise ValueError("min_range_atr must be smaller than max_range_atr")
        return self


class RetestSettings(_Base):
    max_bars_to_retest: int = Field(default=12, ge=1, le=500)
    retest_tolerance_ticks: float = Field(default=4, ge=0)
    retest_tolerance_atr_mult: float = Field(default=0.15, ge=0, le=5)
    max_run_atr: float = Field(default=3.0, gt=0, le=100)
    require_rejection_wick: bool = True
    min_wick_ratio: float = Field(default=0.3, ge=0, le=1)
    confirm_buffer_ticks: float = Field(default=1, ge=0)
    min_close_position: float = Field(default=0.5, ge=0, le=1)
    max_bars_to_confirm: int = Field(default=3, ge=0, le=100)
    invalidation_buffer_ticks: float = Field(default=2, ge=0)
    invalidation_buffer_atr_mult: float = Field(default=0.10, ge=0, le=5)
    setup_expiry_bars: int = Field(default=24, ge=2, le=2000)
    expire_at_session_end: bool = True


class ScoringSettings(_Base):
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    min_score: float = Field(default=80, ge=0, le=100)
    missing_factor_policy: Literal["renormalize", "zero"] = "renormalize"
    vwap_stretch_atr: float = Field(default=4.0, gt=0, le=50)
    min_available_weight: float = Field(default=75, ge=0, le=100)
    essential_factors: list[str] = Field(
        default_factory=lambda: ["zone_quality", "breakout_quality", "retest_quality", "reward_risk"]
    )

    @field_validator("weights")
    @classmethod
    def _weights(cls, v: dict[str, float]) -> dict[str, float]:
        unknown = set(v) - set(SCORE_FACTORS)
        missing = set(SCORE_FACTORS) - set(v)
        if unknown:
            raise ValueError(f"Unknown score factors: {sorted(unknown)}")
        if missing:
            raise ValueError(f"Missing score factors: {sorted(missing)}")
        if any(w < 0 for w in v.values()):
            raise ValueError("Score weights must be non-negative")
        total = sum(v.values())
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"Score weights must sum to 100 (currently {total:g})")
        return v

    @field_validator("essential_factors")
    @classmethod
    def _essential(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(SCORE_FACTORS)
        if unknown:
            raise ValueError(f"Unknown essential factors: {sorted(unknown)}")
        return v


class PlanningSettings(_Base):
    entry_method: Literal["confirmation_close", "zone_edge"] = "confirmation_close"
    stop_buffer_ticks: float = Field(default=2, ge=0)
    stop_buffer_atr_mult: float = Field(default=0.10, ge=0, le=5)
    min_stop_ticks: float = Field(default=4, ge=1)
    max_stop_ticks: float = Field(default=400, ge=1)  # absolute guard
    max_stop_atr: float = Field(default=3.0, gt=0, le=50)  # instrument-neutral width limit
    target_method: Literal["structure", "fixed_r"] = "structure"
    target_r: float = Field(default=2.0, gt=0, le=20)
    extended_target_r: float = Field(default=3.0, gt=0, le=40)
    target_front_run_ticks: float = Field(default=2, ge=0)
    min_rr: float = Field(default=2.0, gt=0, le=20)

    @model_validator(mode="after")
    def _check(self) -> PlanningSettings:
        if self.min_stop_ticks >= self.max_stop_ticks:
            raise ValueError("min_stop_ticks must be smaller than max_stop_ticks")
        if self.extended_target_r <= self.target_r:
            raise ValueError("extended_target_r must be larger than target_r")
        return self


class SessionSettings(_Base):
    signal_windows_enabled: bool = True
    # Local-time windows (in ``windows_timezone``) in which a setup may qualify.
    signal_windows: list[tuple[str, str]] = Field(default_factory=lambda: [("09:30", "15:45")])
    windows_timezone: str = "America/New_York"
    no_new_setups_minutes_before_close: int = Field(default=15, ge=0, le=600)
    vwap_anchor: Literal["globex", "rth"] = "globex"

    @field_validator("signal_windows")
    @classmethod
    def _windows(cls, v: list[tuple[str, str]]) -> list[tuple[str, str]]:
        for start, end in v:
            if parse_hhmm(start) == parse_hhmm(end):
                raise ValueError(f"Empty session window {start}-{end}")
        return v

    @field_validator("windows_timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        return validate_timezone(v)


class ManagementSettings(_Base):
    breakeven_thresholds_r: list[float] = Field(default_factory=lambda: [1.0])
    breakeven_include_costs: bool = True
    profit_r_milestones: list[float] = Field(default_factory=lambda: [2.0, 3.0])
    target_approach_fraction: float = Field(default=0.8, gt=0, lt=1)
    opposing_zone_approach_atr: float = Field(default=0.5, ge=0, le=10)
    momentum_enabled: bool = True
    momentum_ema_period: int = Field(default=9, ge=2, le=200)
    momentum_min_r: float = Field(default=1.0, ge=0, le=20)
    momentum_min_bars_between: int = Field(default=6, ge=1, le=500)
    trailing_enabled: bool = True
    trailing_atr_period: int = Field(default=14, ge=2, le=200)
    trailing_atr_mult: float = Field(default=0.25, ge=0, le=10)
    trailing_buffer_ticks: float = Field(default=2, ge=0)
    vwap_loss_warning: bool = False
    time_stop_bars: int = Field(default=36, ge=0, le=5000)
    time_stop_min_r: float = Field(default=0.5, ge=0, le=20)
    insufficient_reward_r: float = Field(default=0.5, ge=0, le=20)
    suppressed_categories: list[str] = Field(default_factory=list)
    # Volatility-regime-specific breakeven thresholds (optional overrides).
    breakeven_thresholds_high_vol_r: list[float] | None = None

    @field_validator("breakeven_thresholds_r", "profit_r_milestones")
    @classmethod
    def _positive(cls, v: list[float]) -> list[float]:
        if any(x <= 0 for x in v):
            raise ValueError("R thresholds must be positive")
        return sorted(set(v))

    @field_validator("suppressed_categories")
    @classmethod
    def _cats(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(ALERT_CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown alert categories: {sorted(unknown)}")
        return v


class RiskSettings(_Base):
    personal_max_dollar_risk_per_trade: float = Field(default=400, gt=0)
    personal_daily_loss_limit: float | None = Field(default=800, gt=0)
    personal_max_contracts_minis: float = Field(default=2, gt=0)
    commission_per_contract_rt: dict[str, float] = Field(
        default_factory=lambda: {"NQ": 4.0, "MNQ": 1.5, "ES": 4.0, "MES": 1.5, "GC": 4.5, "MGC": 1.8}
    )
    slippage_ticks_per_side: float = Field(default=1, ge=0)
    drawdown_safety_buffer: float = Field(default=0, ge=0)
    daily_loss_alert_fraction: float = Field(default=0.8, gt=0, le=1)
    drawdown_warning_fraction: float = Field(default=0.25, gt=0, le=1)

    def commission_for(self, symbol: str) -> float:
        return float(self.commission_per_contract_rt.get(symbol.upper(), 0.0))


class DataSettings(_Base):
    source: Literal["synthetic", "csv", "databento"] = "synthetic"
    stale_multiplier: float = Field(default=2.0, ge=1, le=100)
    stale_grace_seconds: int = Field(default=60, ge=0, le=3600)
    synthetic_days: int = Field(default=60, ge=5, le=400)
    synthetic_seed: int = 123
    # Fixed anchor so the demo is reproducible; change it to explore other simulated periods.
    synthetic_end_date: date = date(2026, 10, 16)
    live_history_days: int = Field(default=10, ge=2, le=120)
    alert_lookback_bars: int = Field(default=3, ge=1, le=100)
    chart_bars: int = Field(default=600, ge=100, le=10000)


class CategorySetting(_Base):
    enabled: bool = True
    severity: str = "info"
    telegram: bool = False

    @field_validator("severity")
    @classmethod
    def _sev(cls, v: str) -> str:
        if v not in SEVERITIES:
            raise ValueError(f"Unknown severity {v!r}")
        return v


def _default_categories() -> dict[str, CategorySetting]:
    return {
        key: CategorySetting(enabled=True, severity=sev, telegram=tg)
        for key, (_label, sev, tg, _d) in ALERT_CATEGORIES.items()
    }


class NotificationSettings(_Base):
    in_app: bool = True
    telegram_enabled: bool = False
    browser_enabled: bool = False
    categories: dict[str, CategorySetting] = Field(default_factory=_default_categories)
    rule_staleness_days: int = Field(default=30, ge=1, le=3650)

    @field_validator("categories")
    @classmethod
    def _complete(cls, v: dict[str, CategorySetting]) -> dict[str, CategorySetting]:
        unknown = set(v) - set(ALERT_CATEGORIES)
        if unknown:
            raise ValueError(f"Unknown alert categories: {sorted(unknown)}")
        merged = _default_categories()
        merged.update(v)
        return merged


class BacktestDefaults(_Base):
    force_flat_time: str | None = "16:00"  # ET; positions closed at first bar at/after this time
    max_holding_bars: int | None = None
    allow_overlap: bool = False
    simulate_breakeven: bool = False
    simulate_trailing: bool = False
    quantity: int = Field(default=1, ge=1)
    oos_fraction: float = Field(default=0.3, ge=0, lt=1)

    @field_validator("force_flat_time")
    @classmethod
    def _flat(cls, v: str | None) -> str | None:
        if v is not None:
            parse_hhmm(v)
        return v


STRATEGY_SECTIONS = ("timeframes", "swings", "zones", "breakout", "retest", "scoring", "planning", "sessions")


class AppSettings(_Base):
    schema_version: int = 1
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    timeframes: TimeframeSettings = Field(default_factory=TimeframeSettings)
    swings: SwingSettings = Field(default_factory=SwingSettings)
    zones: ZoneSettings = Field(default_factory=ZoneSettings)
    breakout: BreakoutSettings = Field(default_factory=BreakoutSettings)
    retest: RetestSettings = Field(default_factory=RetestSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    planning: PlanningSettings = Field(default_factory=PlanningSettings)
    sessions: SessionSettings = Field(default_factory=SessionSettings)
    management: ManagementSettings = Field(default_factory=ManagementSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    backtest: BacktestDefaults = Field(default_factory=BacktestDefaults)

    def strategy_dict(self) -> dict:
        full = self.model_dump(mode="json")
        return {k: full[k] for k in STRATEGY_SECTIONS}

    def strategy_hash(self) -> str:
        payload = json.dumps(self.strategy_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> AppSettings:
        return cls.model_validate_json(raw)

    def updated(self, section: str, **changes: object) -> AppSettings:
        """Return a validated copy with ``changes`` applied to ``section``."""
        data = self.model_dump(mode="json")
        if section not in data or not isinstance(data[section], dict):
            raise KeyError(section)
        data[section].update(changes)
        return AppSettings.model_validate(data)


def diff_strategy(a: AppSettings, b: AppSettings) -> list[str]:
    """Human-readable list of strategy parameters that differ between a and b."""
    da, db = a.strategy_dict(), b.strategy_dict()
    out: list[str] = []
    for section in STRATEGY_SECTIONS:
        sa, sb = da[section], db[section]
        for key in sorted(set(sa) | set(sb)):
            if sa.get(key) != sb.get(key):
                out.append(f"{section}.{key}: {sa.get(key)!r} -> {sb.get(key)!r}")
    return out
