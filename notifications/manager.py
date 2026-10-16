"""Alert manager: persistence, de-duplication, severity and delivery.

``AlertManager.publish(event)`` stores an engine event as an alert unless an
alert with the same ``dedup_key`` already exists (the database enforces
uniqueness), then delivers it to the enabled channels and records each
delivery attempt. One market condition therefore produces at most one alert,
however many times the page refreshes or the scanner re-runs.

Channels:
* ``in_app``  - always stored; shown in the Alerts Center and as toasts.
* ``telegram`` - optional, per-category, credentials from environment.
* ``browser`` - marked pending; the dashboard shows it via the browser
  Notification API when the user enabled it (best effort).
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import NotificationSettings
from core.events import ALERT_CATEGORIES, EngineEvent
from data import repositories as repo
from data.database import Database
from notifications.telegram import TelegramNotifier

SEVERITY_ICON = {"info": "i", "low": "-", "medium": "!", "high": "!!", "critical": "!!!"}


@dataclass
class PublishResult:
    alert_id: int | None
    created: bool
    deliveries: dict[str, str]
    suppressed_reason: str | None = None


class AlertManager:
    def __init__(
        self, db: Database, settings: NotificationSettings, telegram: TelegramNotifier | None = None
    ):
        self.db = db
        self.settings = settings
        self.telegram = telegram or TelegramNotifier()

    def publish(self, ev: EngineEvent, context: str | None = None) -> PublishResult:
        cat = self.settings.categories.get(ev.category)
        if cat is not None and not cat.enabled:
            return PublishResult(None, False, {}, "category disabled in settings")
        severity = ev.severity or (cat.severity if cat else ALERT_CATEGORIES[ev.category][1])
        alert_id = repo.insert_alert(
            self.db,
            dedup_key=ev.dedup_key,
            category=ev.category,
            severity=severity,
            trigger_ts=ev.ts,
            title=ev.title,
            message=ev.message,
            context=context,
            instrument=ev.instrument,
            setup_id=ev.setup_id,
            trade_id=ev.trade_id,
            account_id=ev.account_id,
            price=ev.price,
            details=ev.details,
        )
        if alert_id is None:
            return PublishResult(None, False, {}, "duplicate (already alerted)")
        deliveries: dict[str, str] = {}
        repo.add_delivery(self.db, alert_id, "in_app", "stored", "Visible in the Alerts Center.")
        deliveries["in_app"] = "stored"
        if self.settings.telegram_enabled and cat is not None and cat.telegram:
            res = self.telegram.send(self.format_text(ev, severity))
            repo.add_delivery(self.db, alert_id, "telegram", res.status, res.detail)
            deliveries["telegram"] = res.status
        if self.settings.browser_enabled:
            repo.add_delivery(self.db, alert_id, "browser", "pending", "Shown when the dashboard is open.")
            deliveries["browser"] = "pending"
        return PublishResult(alert_id, True, deliveries)

    def publish_many(self, events: list[EngineEvent], context: str | None = None) -> list[PublishResult]:
        return [self.publish(e, context) for e in events]

    @staticmethod
    def format_text(ev: EngineEvent, severity: str) -> str:
        label = ALERT_CATEGORIES[ev.category][0]
        parts = [f"[{severity.upper()}] {label}", ev.title, ev.message]
        if ev.price is not None:
            parts.append(f"Price: {ev.price}")
        parts.append(f"Time: {ev.ts.tz_convert('America/New_York'):%Y-%m-%d %H:%M %Z}")
        parts.append("Break & Retest Copilot - analysis only; no orders are placed.")
        return "\n".join(parts)
