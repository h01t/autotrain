"""Metric extraction from training output — JSON primary, regex fallback."""

from __future__ import annotations

import json
import re

import structlog

log = structlog.get_logger()


def extract_metrics_from_line(
    line: str,
    target_metric: str | None = None,
    regex_pattern: str | None = None,
) -> dict[str, float]:
    """Extract metrics from a single output line.

    Strategy:
    1. Try JSON parsing first (structured output)
    2. Fall back to regex pattern if provided
    3. Fall back to common key=value patterns

    Returns dict of metric_name -> value. Empty dict if nothing found.
    """
    line = line.strip()
    if not line:
        return {}

    # Strategy 1: JSON
    metrics = _try_json(line)
    if metrics:
        return metrics

    # Strategy 2: Explicit regex pattern
    if regex_pattern and target_metric:
        metrics = _try_regex(line, target_metric, regex_pattern)
        if metrics:
            return metrics

    # Strategy 3: Common patterns (key=value, key: value)
    return _try_common_patterns(line)


def extract_metric_from_output(
    output: str,
    metric_name: str,
    regex_pattern: str | None = None,
) -> float | None:
    """Extract the final value of a metric from full training output.

    Scans all lines and returns the last occurrence of the metric.
    """
    last_value = None
    for line in output.splitlines():
        metrics = extract_metrics_from_line(line, metric_name, regex_pattern)
        if metric_name in metrics:
            last_value = metrics[metric_name]
    return last_value


def extract_all_metrics_from_output(
    output: str,
    regex_pattern: str | None = None,
    target_metric: str | None = None,
) -> list[dict[str, float]]:
    """Extract all metric snapshots from training output.

    Returns a list of metric dicts, one per line that contained metrics.
    """
    snapshots = []
    for line in output.splitlines():
        metrics = extract_metrics_from_line(line, target_metric, regex_pattern)
        if metrics:
            snapshots.append(metrics)
    return snapshots


def _try_json(line: str) -> dict[str, float]:
    """Try to parse line as JSON and extract numeric values."""
    try:
        data = json.loads(line)
        if isinstance(data, dict):
            return {
                k: float(v)
                for k, v in data.items()
                if isinstance(v, int | float) and not isinstance(v, bool)
            }
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def _try_regex(line: str, metric_name: str, pattern: str) -> dict[str, float]:
    """Try to extract metric using a regex pattern."""
    try:
        match = re.search(pattern, line)
        if match:
            # Use first capture group if available, else full match
            value_str = match.group(1) if match.lastindex else match.group(0)
            return {metric_name: float(value_str)}
    except (ValueError, re.error) as e:
        log.debug("regex_extraction_failed", pattern=pattern, error=str(e))
    return {}


# Common metric output patterns
_COMMON_PATTERNS = [
    # key=value (e.g., val_auc=0.85, loss=0.15)
    re.compile(r"(\w+)\s*=\s*([0-9]+\.?[0-9]*)"),
    # key: value (e.g., val_auc: 0.85)
    re.compile(r"(\w+)\s*:\s+([0-9]+\.?[0-9]*)"),
]


def _try_common_patterns(line: str) -> dict[str, float]:
    """Try common key=value and key: value patterns."""
    metrics: dict[str, float] = {}
    for pattern in _COMMON_PATTERNS:
        for match in pattern.finditer(line):
            key = match.group(1)
            try:
                value = float(match.group(2))
                metrics[key] = value
            except ValueError:
                continue
    return metrics
