"""Topstep-specific interpretation on top of the generic configuration."""

from __future__ import annotations

from firms.base import FirmRules, RuleSet


class TopstepRules(FirmRules):
    def interpretation_notes(self, rules: RuleSet) -> list[str]:
        notes = [
            "Maximum Loss Limit is modelled as end-of-day trailing with real-time enforcement: the floor moves only "
            "at the close, but touching it intraday (including unrealized P&L) is a breach.",
            "Close-out: positions must be flat by the configured close time (reported as 3:10 PM CT).",
        ]
        dd = rules.value("drawdown_model") or {}
        if dd.get("lock_floor_offset") is None and rules.account_type_code.startswith("topstep_tc"):
            notes.append(
                "No MLL lock level is configured for the Combine, so the floor keeps trailing (more "
                "conservative). Set lock_floor_offset once you confirm the official behaviour."
            )
        if rules.account_type_code.startswith("topstep_xfa"):
            notes.append(
                "XFA: the retrieved summary says the MLL resets to $0 permanently after a payout - modelled "
                "as the floor locking at the starting balance when you record a payout."
            )
            notes.append(
                "The Standard payout path excludes the payout-request day from the next five winning days."
            )
        return notes
