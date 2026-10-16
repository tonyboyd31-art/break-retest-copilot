"""Tradeify-specific interpretation on top of the generic configuration."""

from __future__ import annotations

from firms.base import FirmRules, RuleSet


class TradeifyRules(FirmRules):
    def interpretation_notes(self, rules: RuleSet) -> list[str]:
        notes = [
            "Drawdown is modelled as end-of-day trailing, recalculated at the close but enforced in real time.",
            "No overnight positions on any Tradeify account type; flatten before the configured close time.",
        ]
        code = rules.account_type_code
        if "select_flex" in code or "select_daily" in code:
            notes.append(
                "Select funded: the trailing drawdown locks at starting balance + $100 once the EOD balance "
                "exceeds the drawdown by $100, or immediately on a payout request (whichever first)."
            )
            notes.append(
                "Select funded accounts must not hold minis and micros simultaneously (hedging rule)."
            )
        if "select_eval" in code:
            notes.append(
                "Select evaluation consistency: no single day may exceed 40% of total profit; evaluation "
                "consistency does not carry over to the funded stage."
            )
            notes.append(
                "Legacy accounts may keep older parameters (e.g. a $2,500 50K target) - confirm yours."
            )
        if "growth" in code:
            notes.append("Growth evaluation has a soft daily loss limit (pauses trading for the day).")
        return notes
