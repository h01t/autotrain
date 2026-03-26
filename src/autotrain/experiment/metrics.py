"""Metric extraction from training output — JSON primary, regex fallback."""

from __future__ import annotations

import json
import re

import structlog

log = structlog.get_logger()

# Patterns for detecting epoch-level output lines
_YOLO_EPOCH_RE = re.compile(
    r"^\s*(\d+)/(\d+)\s+"  # epoch/total (e.g. "3/50")
    r"[\d.]+G?\s+"          # GPU mem
    r"([\d.]+)\s+"          # box_loss
    r"([\d.]+)\s+"          # cls_loss
    r"([\d.]+)"             # dfl_loss
)

_YOLO_VAL_RE = re.compile(
    r"^\s*all\s+\d+\s+\d+\s+"   # "all  N  M"
    r"([\d.]+)\s+"               # precision
    r"([\d.]+)\s+"               # recall
    r"([\d.]+)\s+"               # mAP50
    r"([\d.]+)"                  # mAP50-95
)

# Keras-style: "Epoch 3/50 - loss: 0.15 - val_acc: 0.92"
_KERAS_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)"
)


def parse_epoch_line(line: str) -> tuple[int | None, dict[str, float]]:
    """Try to extract epoch number and metrics from a training output line.

    Returns (epoch_number, metrics_dict). If not an epoch line, returns (None, {}).
    """
    line = line.strip()
    if not line:
        return None, {}

    # YOLO training line: "3/50  3.18G  1.123  2.345  1.567  45  640"
    m = _YOLO_EPOCH_RE.match(line)
    if m:
        epoch = int(m.group(1))
        return epoch, {
            "box_loss": float(m.group(3)),
            "cls_loss": float(m.group(4)),
            "dfl_loss": float(m.group(5)),
        }

    # YOLO validation line: "all  N  M  0.835  0.760  0.838  0.598"
    m = _YOLO_VAL_RE.match(line)
    if m:
        return None, {
            "precision": float(m.group(1)),
            "recall": float(m.group(2)),
            "mAP": float(m.group(3)),
            "mAP50_95": float(m.group(4)),
        }

    # Keras-style: "Epoch 3/50 - loss: 0.15 - val_acc: 0.92"
    m = _KERAS_EPOCH_RE.search(line)
    if m:
        epoch = int(m.group(1))
        metrics = _try_common_patterns(line)
        if metrics:
            return epoch, metrics

    return None, {}


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
