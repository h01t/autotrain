"""Graceful shutdown signal handling."""

from __future__ import annotations

import signal
import threading

import structlog

log = structlog.get_logger()

# Global shutdown event — threads check this to know when to stop
_shutdown_event = threading.Event()


def get_shutdown_event() -> threading.Event:
    """Get the global shutdown event."""
    return _shutdown_event


def is_shutting_down() -> bool:
    """Check if a shutdown signal has been received."""
    return _shutdown_event.is_set()


def request_shutdown() -> None:
    """Request a graceful shutdown."""
    _shutdown_event.set()


def install_signal_handlers() -> None:
    """Install SIGTERM and SIGINT handlers for graceful shutdown.

    Must be called from the main thread.
    """

    def _handler(signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        log.warning("shutdown_signal_received", signal=sig_name)
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
