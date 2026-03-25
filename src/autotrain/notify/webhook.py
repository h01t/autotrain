"""Webhook notification sender."""

from __future__ import annotations

from typing import Any

import requests
import structlog

log = structlog.get_logger()


def send_webhook(url: str, event: str, data: dict[str, Any]) -> None:
    """Send a webhook notification. Fails silently (never blocks training)."""
    payload = {"event": event, **data}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code >= 400:
            log.warning(
                "webhook_failed",
                status=resp.status_code, url=url,
            )
    except requests.RequestException as e:
        log.warning("webhook_error", error=str(e), url=url)
