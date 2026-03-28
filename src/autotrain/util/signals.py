"""Graceful shutdown signal handling."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable

import structlog

log = structlog.get_logger()

# Global shutdown event — threads check this to know when to stop
_shutdown_event = threading.Event()

# Callbacks invoked immediately when a shutdown signal arrives
_shutdown_callbacks: list[Callable[[], None]] = []


def is_shutting_down() -> bool:
    """Check if a shutdown signal has been received."""
    return _shutdown_event.is_set()


def register_shutdown_callback(cb: Callable[[], None]) -> None:
    """Register a callback to be invoked on shutdown signal (e.g. executor.kill)."""
    _shutdown_callbacks.append(cb)


def install_signal_handlers() -> None:
    """Install SIGTERM and SIGINT handlers for graceful shutdown.

    Must be called from the main thread.
    """

    def _handler(signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        log.warning("shutdown_signal_received", signal=sig_name)
        _shutdown_event.set()
        for cb in _shutdown_callbacks:
            try:
                cb()
            except Exception:
                pass  # Best effort — don't crash in signal handler

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
