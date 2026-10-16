"""Evaluation progress, consistency and payout-eligibility ESTIMATES.

Everything here is computed from user-entered (or journal-derived) daily
P&L, never from official firm data. Results are labelled as estimates and
never as an approval or guarantee. Each figure records which rules it used
and whether those rules are trusted (VERIFIED / USER_CONFIRMED).

Consistency formulas (implemented only as named; the firm config chooses):

* ``largest_day_over_total_profit``: largest winning day / cumulative net
  profit in the cycle. Undefined when net profit <= 0.
* ``largest_day_over_profit_target``: largest winning day / profit target
  (an alternative reading some firms use).

Payout cycles start after the most recent recorded payout (the payout day
itself is excluded) or after the account's cycle start / reset date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from firms.base import RuleSet
from firms.drawdown import DailyRecord, DrawdownResult, build_drawdown_model

ESTIMATE_LABEL = (
    "ESTIMATE based on user-entered data - not official account data and not a payout approval or guarantee."
)


@dataclass
class ConsistencyResult:
    formula: str
    threshold: float
    ratio: float | None
    passes: bool | None
    largest_day: float
    total_profit: float
    required_total_profit: float | None
    explanation: str


def consistency(
    records: list[DailyRecord], formula: str, threshold: float, profit_target: float | None = None
) -> ConsistencyResult:
    if threshold <= 0 or threshold > 1:
        raise ValueError("Consistency threshold must be in (0, 1].")
    pnl = [r.net_pnl for r in records]
    total = float(sum(pnl))
    largest = max([p for p in pnl if p > 0], default=0.0)
    required = largest / threshold if largest > 0 else None
    if formula == "largest_day_over_total_profit":
        if total <= 0:
            return ConsistencyResult(
                formula,
                threshold,
                None,
                None,
                largest,
                total,
                required,
                f"Undefined while cumulative net profit is ${total:,.2f} (<= 0).",
            )
        ratio = largest / total
        ok = ratio <= threshold + 1e-12
        expl = (
            f"Largest winning day ${largest:,.2f} is {ratio:.1%} of net profit ${total:,.2f} (limit "
            f"{threshold:.0%})."
        )
        if not ok and required is not None:
            expl += (
                f" Net profit would need to reach about ${required:,.2f} for this day to be within the limit."
            )
        return ConsistencyResult(formula, threshold, ratio, ok, largest, total, required, expl)
    if formula == "largest_day_over_profit_target":
        if not profit_target:
            return ConsistencyResult(
                formula, threshold, None, None, largest, total, None, "Profit target not configured."
            )
        ratio = largest / profit_target
        ok = ratio <= threshold + 1e-12
        return ConsistencyResult(
            formula,
            threshold,
            ratio,
            ok,
            largest,
            total,
            None,
            f"Largest winning day ${largest:,.2f} is {ratio:.1%} of the ${profit_target:,.2f} "
            f"profit target (limit {threshold:.0%}).",
        )
    raise ValueError(f"Unknown consistency formula {formula!r}")


@dataclass
class EvaluationProgress:
    stage: str
    starting_balance: float
    net_profit: float
    current_balance: float
    profit_target: float | None
    remaining_target: float | None
    target_reached: bool | None
    today_pnl: float | None
    largest_winning_day: float
    winning_days: int
    trading_days: int
    min_trading_days: int | None
    consistency: ConsistencyResult | None
    drawdown: DrawdownResult | None
    drawdown_message: str
    daily_loss_limit: dict | None
    status: str
    next_milestone: str
    untrusted_rules: list[str]
    label: str = ESTIMATE_LABEL
    notes: list[str] = field(default_factory=list)


def cycle_records(
    records: list[DailyRecord], cycle_start: date | None = None, last_payout: date | None = None
) -> list[DailyRecord]:
    out = sorted(records, key=lambda r: r.trade_date)
    if cycle_start:
        out = [r for r in out if r.trade_date >= cycle_start]
    if last_payout:
        out = [r for r in out if r.trade_date > last_payout]
    return out


def evaluation_progress(
    rules: RuleSet,
    stage: str,
    starting_balance: float,
    records: list[DailyRecord],
    today: date,
    cycle_start: date | None = None,
    winning_day_threshold: float = 0.0,
) -> EvaluationProgress:
    recs = cycle_records(records, cycle_start)
    notes: list[str] = []
    net = float(sum(r.net_pnl for r in recs)) - 0.0
    payouts = float(sum(r.payout for r in recs))
    balance = starting_balance + net - payouts
    target = rules.value("profit_target")
    remaining = None if target is None else max(float(target) - net, 0.0)
    reached = None if target is None else net >= float(target)
    today_rec = next((r for r in recs if r.trade_date == today), None)
    largest = max([r.net_pnl for r in recs if r.net_pnl > 0], default=0.0)
    wins = sum(1 for r in recs if r.net_pnl > winning_day_threshold)
    traded = sum(1 for r in recs if r.net_pnl != 0)
    min_days = rules.value("min_trading_days")

    cons = None
    cons_rule = rules.value("consistency")
    if cons_rule:
        cons = consistency(
            recs, cons_rule["formula"], float(cons_rule["threshold"]), float(target) if target else None
        )

    model, msg = build_drawdown_model(rules.value("drawdown_model"), rules.value("max_loss"))
    dd = model.run(starting_balance, recs) if model else None

    used = [
        "profit_target",
        "max_loss",
        "drawdown_model",
        "consistency",
        "min_trading_days",
        "daily_loss_limit",
    ]
    untrusted = rules.untrusted_keys(used)

    dll = rules.value("daily_loss_limit")
    if dll and dll.get("amount") and today_rec is not None and today_rec.net_pnl <= -float(dll["amount"]):
        notes.append(
            f"Today's P&L ${today_rec.net_pnl:,.2f} reached the daily loss limit ${dll['amount']:,.2f} "
            f"({dll.get('type')})."
        )

    # status & next milestone
    if dd and dd.breached:
        status = "Drawdown breach detected (estimate)"
        nxt = "Per your records the drawdown floor was touched - check your official account status."
    elif stage == "evaluation":
        missing = []
        if reached is False:
            missing.append(f"${remaining:,.2f} more net profit to the ${float(target):,.2f} target")
        if min_days and traded < int(min_days):
            missing.append(f"{int(min_days) - traded} more trading day(s) (minimum {int(min_days)})")
        if cons is not None and cons.passes is False:
            missing.append(
                f"consistency: net profit near ${cons.required_total_profit:,.2f} for the largest day to "
                f"fit the {cons.threshold:.0%} limit"
            )
        if target is None:
            status = "Profit target not configured"
            nxt = "Enter the profit target for this account type in Firm Rules."
        elif not missing:
            status = "Requirements appear met (estimate)"
            nxt = (
                "Per your records the configured requirements appear met. Confirm in your official dashboard."
            )
        else:
            status = "In progress"
            nxt = "; ".join(missing)
    else:
        status = "Funded - see payout checklist"
        nxt = "See the payout checklist."
    if dd and dd.breach_unknown_days:
        notes.append(
            f"{dd.breach_unknown_days} day(s) lack intraday low equity; intraday breaches cannot be ruled "
            "out from your records."
        )
    if untrusted:
        notes.append("Uses unconfirmed rule(s): " + ", ".join(untrusted) + ".")
    return EvaluationProgress(
        stage=stage,
        starting_balance=starting_balance,
        net_profit=round(net, 2),
        current_balance=round(balance, 2),
        profit_target=None if target is None else float(target),
        remaining_target=remaining,
        target_reached=reached,
        today_pnl=None if today_rec is None else today_rec.net_pnl,
        largest_winning_day=largest,
        winning_days=wins,
        trading_days=traded,
        min_trading_days=None if min_days is None else int(min_days),
        consistency=cons,
        drawdown=dd,
        drawdown_message=msg,
        daily_loss_limit=dll,
        status=status,
        next_milestone=nxt,
        untrusted_rules=untrusted,
        notes=notes,
    )


@dataclass
class ChecklistItem:
    name: str
    status: str  # met | not_met | unknown | info
    detail: str


@dataclass
class PayoutPathResult:
    path: str
    items: list[ChecklistItem]
    overall: str  # appears_eligible | not_eligible | unknown
    estimated_max_payout: float | None
    cap_detail: str


def payout_checklist(
    rules: RuleSet,
    starting_balance: float,
    records: list[DailyRecord],
    payouts: list[tuple[date, float]],
    cycle_start: date | None = None,
    drawdown: DrawdownResult | None = None,
) -> tuple[list[PayoutPathResult], list[str]]:
    paths = rules.value("payout_paths") or []
    notes: list[str] = [ESTIMATE_LABEL]
    if not paths:
        return [], notes + ["No payout rules configured for this account type."]
    st = rules.status("payout_paths")
    if st is not None and st.value not in ("VERIFIED", "USER_CONFIRMED"):
        notes.append(f"Payout rules are {st.value} - confirm them on the official page.")
    last_payout = max((d for d, _ in payouts), default=None)
    first_payout = last_payout is None
    recs = cycle_records(records, cycle_start, last_payout)
    all_recs = cycle_records(records, cycle_start)
    balance = starting_balance + sum(r.net_pnl for r in all_recs) - sum(a for _, a in payouts)
    cycle_profit = float(sum(r.net_pnl for r in recs))
    total_profit = float(sum(r.net_pnl for r in all_recs))
    out: list[PayoutPathResult] = []
    for p in paths:
        items: list[ChecklistItem] = []
        if "winning_days_required" in p:
            thr = float(p.get("winning_day_threshold", 0))
            wins = sum(1 for r in recs if r.net_pnl >= thr and r.net_pnl > 0)
            need = int(p["winning_days_required"])
            items.append(
                ChecklistItem(
                    f"Winning days (>= ${thr:,.0f} net)",
                    "met" if wins >= need else "not_met",
                    f"{wins} of {need} in the current cycle.",
                )
            )
        if "min_trading_days" in p:
            need = int(p["min_trading_days"])
            traded = sum(1 for r in recs if r.net_pnl != 0)
            items.append(
                ChecklistItem(
                    "Trading days",
                    "met" if traded >= need else "not_met",
                    f"{traded} of {need} in the current cycle.",
                )
            )
        if "consistency_threshold" in p:
            c = consistency(recs, "largest_day_over_total_profit", float(p["consistency_threshold"]))
            items.append(
                ChecklistItem(
                    f"Consistency (largest day <= {c.threshold:.0%} of cycle profit)",
                    "unknown" if c.passes is None else ("met" if c.passes else "not_met"),
                    c.explanation,
                )
            )
        if p.get("require_positive_cycle_profit"):
            if first_payout:
                items.append(
                    ChecklistItem(
                        "Net profit since last payout",
                        "info",
                        "First payout - positive-cycle rule reported as exempt (confirm).",
                    )
                )
            else:
                items.append(
                    ChecklistItem(
                        "Net profit since last payout",
                        "met" if cycle_profit > 0 else "not_met",
                        f"${cycle_profit:,.2f} since {last_payout}.",
                    )
                )
        if drawdown is not None:
            items.append(
                ChecklistItem(
                    "No drawdown breach in your records",
                    "not_met"
                    if drawdown.breached
                    else ("unknown" if drawdown.breach_unknown_days else "met"),
                    f"Remaining allowance ${drawdown.remaining_allowance:,.2f}.",
                )
            )
        cap = p.get("cap") or {}
        est = None
        cap_detail = "No cap rule configured."
        basis = cap.get("basis")
        if basis == "profit":
            base = max(total_profit - sum(a for _, a in payouts), 0.0)
            est = base * float(cap.get("pct", 1.0))
            cap_detail = f"{float(cap.get('pct', 1.0)):.0%} of unwithdrawn profit ${base:,.2f}"
        elif basis == "balance_above_start":
            base = max(balance - starting_balance, 0.0)
            est = base * float(cap.get("pct", 1.0))
            cap_detail = f"{float(cap.get('pct', 1.0)):.0%} of balance above start ${base:,.2f}"
        elif basis == "cycle_profit_multiple":
            base = max(cycle_profit, 0.0)
            est = base * float(cap.get("multiple", 1.0))
            cap_detail = f"{float(cap.get('multiple', 1.0)):g}x cycle profit ${base:,.2f}"
        if est is not None and cap.get("max_amount") is not None:
            if est > float(cap["max_amount"]):
                cap_detail += f", capped at ${float(cap['max_amount']):,.2f}"
            est = min(est, float(cap["max_amount"]))
        statuses = [i.status for i in items if i.status != "info"]
        if any(s == "not_met" for s in statuses):
            overall = "not_eligible"
        elif any(s == "unknown" for s in statuses):
            overall = "unknown"
        else:
            overall = "appears_eligible"
        out.append(
            PayoutPathResult(
                p.get("name", "Payout"), items, overall, None if est is None else round(est, 2), cap_detail
            )
        )
    return out, notes
