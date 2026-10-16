"""Event and alert-category definitions shared by the engine and notifier.

Engine modules emit :class:`EngineEvent` objects (pure data). The
notification layer turns them into persisted, de-duplicated alerts.
Every event carries a ``dedup_key`` so the same market condition is never
alerted twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

SEVERITIES = ("info", "low", "medium", "high", "critical")

# category -> (label, default severity, default telegram, description)
ALERT_CATEGORIES: dict[str, tuple[str, str, bool, str]] = {
    "breakout_detected": (
        "Breakout detected",
        "info",
        False,
        "A completed candle closed beyond a relevant zone.",
    ),
    "retest_detected": ("Retest detected", "low", False, "Price revisited a broken zone after a breakout."),
    "setup_qualified": ("Setup qualified", "high", True, "A setup met every rule and the minimum score."),
    "setup_rejected": (
        "Setup rejected",
        "info",
        False,
        "A detected setup failed a rule or the score threshold.",
    ),
    "setup_invalidated": (
        "Setup invalidated / expired",
        "low",
        False,
        "A pending setup was invalidated or expired.",
    ),
    "breakeven_review": (
        "Breakeven review",
        "high",
        True,
        "A manual trade reached a configured R threshold.",
    ),
    "profit_protection": (
        "Profit-protection review",
        "medium",
        True,
        "Conditions that may justify securing profit.",
    ),
    "trailing_stop": (
        "Trailing-stop suggestion",
        "medium",
        True,
        "A newly confirmed swing gives a tighter stop level.",
    ),
    "target_approaching": (
        "Target approaching",
        "medium",
        True,
        "Price is approaching or reached the planned target.",
    ),
    "trade_warning": (
        "Trade invalidation / exit warning",
        "high",
        True,
        "The original trade thesis may be failing.",
    ),
    "daily_risk": (
        "Daily risk threshold",
        "high",
        True,
        "Today's realized loss reached a configured threshold.",
    ),
    "drawdown_warning": ("Drawdown warning", "critical", True, "Remaining drawdown allowance is low."),
    "data_stale": ("Market data stale / unreliable", "high", True, "Feed is stale, incomplete or malformed."),
    "rules_outdated": (
        "Account-rule configuration outdated",
        "medium",
        False,
        "Firm rules are unverified or old.",
    ),
}


@dataclass
class EngineEvent:
    category: str
    ts: pd.Timestamp  # when the condition became observable (bar close time)
    title: str
    message: str
    dedup_key: str
    instrument: str | None = None
    setup_id: str | None = None
    trade_id: int | None = None
    account_id: int | None = None
    price: float | None = None
    severity: str | None = None  # None -> category default
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in ALERT_CATEGORIES:
            raise ValueError(f"Unknown alert category {self.category!r}")
        if self.severity is not None and self.severity not in SEVERITIES:
            raise ValueError(f"Unknown severity {self.severity!r}")

    @property
    def effective_severity(self) -> str:
        return self.severity or ALERT_CATEGORIES[self.category][1]
