"""Terminal notification output."""

from __future__ import annotations

_ICONS = {
    "target_hit": "[TARGET HIT]",
    "budget_exhausted": "[BUDGET]",
    "improved": "[IMPROVED]",
    "regressed": "[REGRESSED]",
    "crashed": "[CRASH]",
    "stuck": "[STUCK]",
    "failed": "[FAILED]",
    "error": "[ERROR]",
    "started": "[START]",
    "iteration": "[ITER]",
}


def print_event(event: str, **data) -> None:
    """Print a formatted event to the terminal."""
    icon = _ICONS.get(event, f"[{event.upper()}]")
    parts = [icon]

    if "message" in data:
        parts.append(str(data["message"]))
    elif "metric_value" in data and data["metric_value"] is not None:
        parts.append(f"metric={data['metric_value']:.4f}")
    elif "reason" in data:
        parts.append(str(data["reason"]))

    # Add iteration info if available
    if "iteration" in data:
        parts.append(f"(iter {data['iteration']})")

    print(" ".join(parts))
