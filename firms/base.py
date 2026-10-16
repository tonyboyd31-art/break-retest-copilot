"""Firm-neutral rules interface.

A firm is described by a JSON configuration (``firms/configurations``)
holding firm-wide rules and per-account-type rules. Every rule carries its
value, official source URL, verification date and status:

* ``VERIFIED``       - checked against official documentation by the build (none shipped)
* ``USER_CONFIRMED`` - you confirmed it against the official page (stored in SQLite)
* ``UNVERIFIED``     - not confirmed; values are a starting point only
* ``CONFLICTING``    - sources disagreed; confirm before use

User confirmations/overrides take precedence over the shipped config and are
stored separately, so updating a config file never erases your confirmations
and changing rules never touches the strategy engine.

To add a firm: create ``firms/configurations/<code>.json`` following the
same schema. Subclass :class:`FirmRules` only if the firm needs custom
interpretation; otherwise the generic provider is used automatically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent / "configurations"

STANDARD_RULE_LABELS = {
    "starting_balance": "Starting balance",
    "profit_target": "Profit target",
    "max_loss": "Maximum loss / drawdown amount",
    "drawdown_model": "Drawdown model",
    "daily_loss_limit": "Daily loss limit",
    "max_contracts_minis": "Max contracts (minis)",
    "max_contracts_micros": "Max contracts (micros)",
    "scaling_plan": "Scaling plan",
    "consistency": "Consistency rule",
    "min_trading_days": "Minimum trading days",
    "payout_paths": "Payout rules",
    "trading_close_time": "Daily close-out time",
    "overnight_positions": "Overnight positions allowed",
    "automation_policy": "Automation / API policy",
    "prohibited_strategies": "Prohibited strategies",
    "news_trading": "News trading",
    "hedging": "Hedging",
    "inactivity": "Inactivity",
}


class RuleStatus(StrEnum):
    VERIFIED = "VERIFIED"
    USER_CONFIRMED = "USER_CONFIRMED"
    UNVERIFIED = "UNVERIFIED"
    CONFLICTING = "CONFLICTING"


TRUSTED = {RuleStatus.VERIFIED, RuleStatus.USER_CONFIRMED}


@dataclass
class RuleValue:
    key: str
    value: Any
    status: RuleStatus
    source_url: str | None = None
    verified_on: date | None = None
    effective_date: date | None = None
    notes: str = ""
    unit: str | None = None
    origin: str = "config"  # config | user

    @property
    def label(self) -> str:
        return STANDARD_RULE_LABELS.get(self.key, self.key.replace("_", " ").capitalize())

    @property
    def trusted(self) -> bool:
        return self.status in TRUSTED

    def age_days(self, today: date) -> int | None:
        return None if self.verified_on is None else (today - self.verified_on).days


@dataclass
class AccountTypeSpec:
    firm_code: str
    code: str
    label: str
    stage: str
    account_size: float


@dataclass
class RuleSet:
    firm_code: str
    account_type_code: str
    rules: dict[str, RuleValue] = field(default_factory=dict)

    def get(self, key: str) -> RuleValue | None:
        return self.rules.get(key)

    def value(self, key: str, default: Any = None) -> Any:
        r = self.rules.get(key)
        return default if r is None or r.value is None else r.value

    def status(self, key: str) -> RuleStatus | None:
        r = self.rules.get(key)
        return r.status if r else None

    def untrusted_keys(self, keys: list[str] | None = None) -> list[str]:
        keys = keys or list(self.rules)
        return [k for k in keys if k in self.rules and not self.rules[k].trusted]

    def stale_keys(self, today: date, max_age_days: int) -> list[str]:
        out = []
        for k, r in self.rules.items():
            age = r.age_days(today)
            if r.trusted and (age is None or age > max_age_days):
                out.append(k)
        return out


def _parse_date(v: Any) -> date | None:
    if v in (None, ""):
        return None
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


def rule_from_dict(key: str, d: dict, origin: str = "config") -> RuleValue:
    return RuleValue(
        key=key,
        value=d.get("value"),
        status=RuleStatus(d.get("status", "UNVERIFIED")),
        source_url=d.get("source_url"),
        verified_on=_parse_date(d.get("verified_on")),
        effective_date=_parse_date(d.get("effective_date")),
        notes=d.get("notes") or "",
        unit=d.get("unit"),
        origin=origin,
    )


class FirmRules:
    """Generic, configuration-driven firm provider."""

    def __init__(self, config: dict, config_path: Path | None = None):
        self.config = config
        self.config_path = config_path
        self.code: str = config["firm_code"]
        self.name: str = config["firm_name"]
        self.website: str | None = config.get("website")
        self.config_version: str = config.get("config_version", "")
        self.verification_summary: str = config.get("verification_summary", "")

    @classmethod
    def from_file(cls, path: Path) -> FirmRules:
        return cls(json.loads(path.read_text()), path)

    # -- account types ---------------------------------------------------------
    def account_types(self) -> list[AccountTypeSpec]:
        return [
            AccountTypeSpec(self.code, a["code"], a["label"], a["stage"], float(a["account_size"]))
            for a in self.config["account_types"]
        ]

    def account_type(self, code: str) -> AccountTypeSpec:
        for a in self.account_types():
            if a.code == code:
                return a
        raise KeyError(f"Unknown account type {code!r} for {self.name}")

    # -- rules -------------------------------------------------------------------
    def config_rule_rows(self) -> list[dict]:
        """Flatten config rules for database sync ('*' = firm-wide)."""
        rows = []
        for key, d in self.config.get("firm_rules", {}).items():
            rows.append({"account_type_code": "*", "rule_key": key, **d})
        for a in self.config["account_types"]:
            for key, d in a.get("rules", {}).items():
                rows.append({"account_type_code": a["code"], "rule_key": key, **d})
        return rows

    def rules_for(self, account_type_code: str, user_rows: list[dict] | None = None) -> RuleSet:
        rs = RuleSet(self.code, account_type_code)
        for key, d in self.config.get("firm_rules", {}).items():
            rs.rules[key] = rule_from_dict(key, d)
        at = next((a for a in self.config["account_types"] if a["code"] == account_type_code), None)
        if at is None:
            raise KeyError(f"Unknown account type {account_type_code!r}")
        for key, d in at.get("rules", {}).items():
            rs.rules[key] = rule_from_dict(key, d)
        for row in user_rows or []:
            if row.get("origin") != "user":
                continue
            if row["account_type_code"] not in (account_type_code, "*"):
                continue
            rs.rules[row["rule_key"]] = rule_from_dict(row["rule_key"], row, origin="user")
        return rs

    # -- interpretation hooks (override per firm) ---------------------------------
    def interpretation_notes(self, rules: RuleSet) -> list[str]:
        return []

    def payout_extra_checks(self, rules: RuleSet) -> list[str]:
        return []
