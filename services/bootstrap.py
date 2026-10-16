"""Application start-up: database migration, instrument and firm sync, settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from core.config import AppSettings
from core.instruments import InstrumentRegistry, default_registry
from data import repositories as repo
from data.database import PROJECT_ROOT, Database
from firms.base import FirmRules
from firms.registry import get_firms


@dataclass
class AppContext:
    db: Database
    registry: InstrumentRegistry
    firms: dict[str, FirmRules]
    settings_warnings: list[str] = field(default_factory=list)

    def settings(self) -> AppSettings:
        s, _ = repo.load_settings(self.db)
        return s


def bootstrap(db_path: str | Path | None = None, load_env: bool = True) -> AppContext:
    if load_env:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    db = Database(db_path)
    db.migrate()
    registry = default_registry()
    repo.sync_instruments(db, registry.all())
    firms = get_firms()
    for firm in firms.values():
        repo.upsert_firm(
            db,
            firm.code,
            firm.name,
            firm.website,
            firm.config_version,
            str(firm.config_path) if firm.config_path else "",
        )
        repo.sync_config_rules(db, firm.code, firm.config_rule_rows())
    _, warnings = repo.load_settings(db)
    return AppContext(db=db, registry=registry, firms=firms, settings_warnings=warnings)
