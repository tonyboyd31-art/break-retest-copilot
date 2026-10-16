"""Firm registry: discovers every JSON configuration in firms/configurations."""

from __future__ import annotations

from pathlib import Path

from firms.base import CONFIG_DIR, FirmRules
from firms.topstep import TopstepRules
from firms.tradeify import TradeifyRules

PROVIDERS: dict[str, type[FirmRules]] = {"topstep": TopstepRules, "tradeify": TradeifyRules}


def load_firms(config_dir: Path = CONFIG_DIR) -> dict[str, FirmRules]:
    firms: dict[str, FirmRules] = {}
    for path in sorted(config_dir.glob("*.json")):
        generic = FirmRules.from_file(path)
        cls = PROVIDERS.get(generic.code, FirmRules)
        firms[generic.code] = cls(generic.config, path)
    return firms


_FIRMS: dict[str, FirmRules] | None = None


def get_firms() -> dict[str, FirmRules]:
    global _FIRMS
    if _FIRMS is None:
        _FIRMS = load_firms()
    return _FIRMS


def get_firm(code: str) -> FirmRules:
    firms = get_firms()
    if code not in firms:
        raise KeyError(f"Unknown firm {code!r}")
    return firms[code]
