"""Structured logging configuration via structlog."""

from __future__ import annotations

import sys
from pathlib import Path

import structlog


def configure_logging(log_file: Path | None = None, verbose: bool = False) -> None:
    """Configure structlog for JSON output.

    Args:
        log_file: Path to JSON lines log file. If None, logs go only to stderr.
        verbose: If True, also log to stderr in human-readable format.
    """
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Build the logger factory
    factories = []

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        factories.append(structlog.WriteLoggerFactory(file=open(log_file, "a")))  # noqa: SIM115

    if not factories:
        # Default: stderr only
        factories.append(structlog.WriteLoggerFactory(file=sys.stderr))

    # Use JSON for file output
    final_processors = [*processors, structlog.processors.JSONRenderer()]

    structlog.configure(
        processors=final_processors,
        logger_factory=factories[0] if len(factories) == 1 else factories[0],
        wrapper_class=structlog.make_filtering_bound_logger(0),
        cache_logger_on_first_use=True,
    )


def get_logger(**initial_values) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger with optional initial bound values."""
    return structlog.get_logger(**initial_values)
