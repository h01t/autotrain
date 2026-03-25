"""Notification dispatcher — route events to handlers."""

from __future__ import annotations

import structlog

from autotrain.notify.terminal import print_event
from autotrain.notify.webhook import send_webhook

log = structlog.get_logger()


class NotifyDispatcher:
    """Route notification events to configured handlers."""

    def __init__(
        self,
        webhook_url: str | None = None,
        webhook_events: list[str] | None = None,
        terminal: bool = True,
    ) -> None:
        self._webhook_url = webhook_url
        self._webhook_events = set(webhook_events or [])
        self._terminal = terminal

    def notify(self, event: str, **data) -> None:
        """Send a notification for an event."""
        if self._terminal:
            print_event(event, **data)

        if self._webhook_url and event in self._webhook_events:
            send_webhook(self._webhook_url, event, data)
