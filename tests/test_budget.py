"""Tests for budget tracking."""

from __future__ import annotations

import pytest

from autotrain.config.schema import BudgetConfig
from autotrain.core.budget import BudgetExhausted, BudgetTracker


class TestBudgetTracker:
    def test_no_limits_passes(self):
        tracker = BudgetTracker(BudgetConfig())
        tracker.record_iteration()
        tracker.check()  # Should not raise

    def test_iteration_limit(self):
        tracker = BudgetTracker(BudgetConfig(max_iterations=3))
        for _ in range(3):
            tracker.record_iteration()
        with pytest.raises(BudgetExhausted, match="Iteration"):
            tracker.check()

    def test_api_cost_limit(self):
        tracker = BudgetTracker(BudgetConfig(api_dollars=1.0))
        tracker.record_api_cost(0.60)
        tracker.check()  # OK
        tracker.record_api_cost(0.50)
        with pytest.raises(BudgetExhausted, match="API cost"):
            tracker.check()

    def test_summary(self):
        tracker = BudgetTracker(BudgetConfig())
        tracker.record_iteration()
        tracker.record_api_cost(0.05)
        s = tracker.summary()
        assert s["iterations"] == 1
        assert s["api_cost"] == 0.05
        assert s["elapsed_seconds"] >= 0
