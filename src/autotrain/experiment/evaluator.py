"""Evaluator — target checking and improvement detection."""

from __future__ import annotations

from autotrain.config.schema import MetricConfig


def is_target_hit(metric_value: float, config: MetricConfig) -> bool:
    """Check if the metric has reached the target."""
    if config.direction == "maximize":
        return metric_value >= config.target
    else:
        return metric_value <= config.target


def is_improved(
    new_value: float,
    best_value: float | None,
    direction: str = "maximize",
) -> bool:
    """Check if new_value is an improvement over best_value."""
    if best_value is None:
        return True
    if direction == "maximize":
        return new_value > best_value
    else:
        return new_value < best_value
