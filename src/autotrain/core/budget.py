"""Budget enforcement — time, cost, iteration limits."""

from __future__ import annotations

import time

import structlog

from autotrain.config.schema import BudgetConfig

log = structlog.get_logger()


class BudgetExhausted(Exception):
    """Raised when any budget limit is exceeded."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class BudgetTracker:
    """Track resource usage against budget limits."""

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._start_time = time.monotonic()
        self._iterations = 0
        self._api_cost = 0.0

    def record_iteration(self) -> None:
        self._iterations += 1

    def record_api_cost(self, cost: float) -> None:
        self._api_cost += cost

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def api_cost(self) -> float:
        return self._api_cost

    def check(self) -> None:
        """Check all budget limits. Raises BudgetExhausted if any exceeded."""
        if (
            self._config.time_seconds
            and self.elapsed_seconds >= self._config.time_seconds
        ):
            raise BudgetExhausted(
                f"Time budget exhausted: {self.elapsed_seconds:.0f}s "
                f">= {self._config.time_seconds}s"
            )

        if (
            self._config.max_iterations
            and self._iterations >= self._config.max_iterations
        ):
            raise BudgetExhausted(
                f"Iteration budget exhausted: {self._iterations} "
                f">= {self._config.max_iterations}"
            )

        if (
            self._config.api_dollars
            and self._api_cost >= self._config.api_dollars
        ):
            raise BudgetExhausted(
                f"API cost budget exhausted: ${self._api_cost:.2f} "
                f">= ${self._config.api_dollars:.2f}"
            )

    def summary(self) -> dict:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "iterations": self._iterations,
            "api_cost": round(self._api_cost, 4),
        }
