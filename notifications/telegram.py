"""Optional Telegram delivery channel.

Credentials come only from environment variables (TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID). The token is never logged, stored in the database or
shown in the UI; error messages are scrubbed of it.

This channel could not be exercised end-to-end during development (no bot
credentials and api.telegram.org may be blocked by the build network), so it
is reported as untested in the UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

API = "https://api.telegram.org/bot{token}/sendMessage"


@dataclass
class DeliveryResult:
    status: str  # sent | failed | not_configured | disabled
    detail: str


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None, timeout: float = 8.0):
        self._token = token if token is not None else os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id if chat_id is not None else os.environ.get("TELEGRAM_CHAT_ID", "")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._token and self._chat_id)

    def describe(self) -> str:
        if not self.configured:
            missing = [
                n
                for n, v in (("TELEGRAM_BOT_TOKEN", self._token), ("TELEGRAM_CHAT_ID", self._chat_id))
                if not v
            ]
            return "Not configured (missing " + ", ".join(missing) + ")."
        masked = self._chat_id[:2] + "***" if len(self._chat_id) > 2 else "***"
        return f"Configured (token hidden, chat {masked})."

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "***") if self._token else text

    def send(self, text: str) -> DeliveryResult:
        if not self.configured:
            return DeliveryResult("not_configured", self.describe())
        try:
            resp = requests.post(
                API.format(token=self._token),
                json={"chat_id": self._chat_id, "text": text[:4000], "disable_web_page_preview": True},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            return DeliveryResult("failed", self._scrub(f"{type(exc).__name__}: {exc}"))
        if resp.status_code == 200 and resp.json().get("ok"):
            return DeliveryResult("sent", "Delivered to Telegram.")
        return DeliveryResult("failed", self._scrub(f"HTTP {resp.status_code}: {resp.text[:200]}"))
