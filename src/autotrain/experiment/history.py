"""Format experiment history for agent context."""

from __future__ import annotations

import json
import sqlite3

from autotrain.storage.queries import (
    get_best_iterations,
    get_epoch_metrics,
    get_recent_iterations,
)


def format_history_for_prompt(
    conn: sqlite3.Connection,
    run_id: str,
    direction: str = "maximize",
    recent_limit: int = 20,
    best_limit: int = 5,
) -> str:
    """Format experiment history as a text table for the agent prompt.

    Includes recent iterations, top-performing ones, and training curves
    for the most recent iteration.
    """
    recent = get_recent_iterations(conn, run_id, limit=recent_limit)
    best = get_best_iterations(conn, run_id, direction=direction, limit=best_limit)

    lines = []

    # Recent experiments
    lines.append(f"## Recent Experiments (last {len(recent)})")
    lines.append("")
    lines.append(
        "| # | Metric | Outcome | Duration | Changes | Error |"
    )
    lines.append(
        "|---|--------|---------|----------|---------|-------|"
    )

    for it in recent:
        metric_str = f"{it.metric_value:.4f}" if it.metric_value is not None else "N/A"
        outcome_str = it.outcome.value if it.outcome else "?"
        duration_str = f"{it.duration_seconds:.0f}s" if it.duration_seconds else "?"
        changes_str = (it.changes_summary or "")[:50]
        error_str = ""
        if it.error_message and it.outcome and it.outcome.value in (
            "crashed", "script_error", "timeout",
        ):
            # First line, truncated — enough for the agent to understand
            error_str = it.error_message.split("\n")[0][:80]
        lines.append(
            f"| {it.iteration_num} | {metric_str} | {outcome_str} "
            f"| {duration_str} | {changes_str} | {error_str} |"
        )

    # Best experiments
    if best:
        lines.append("")
        lines.append(f"## Best Experiments (top {len(best)})")
        lines.append("")
        for it in best:
            metric_str = f"{it.metric_value:.4f}" if it.metric_value is not None else "N/A"
            lines.append(
                f"- Iteration {it.iteration_num}: {metric_str} — {it.changes_summary or '?'}"
            )

    # Training curve for the most recent iteration
    if recent:
        last_iter = recent[-1]
        curve = format_training_curve(conn, run_id, last_iter.iteration_num)
        if curve:
            lines.append("")
            lines.append(curve)

    return "\n".join(lines)


def format_training_curve(
    conn: sqlite3.Connection,
    run_id: str,
    iteration_num: int,
    tail_epochs: int = 15,
) -> str | None:
    """Format per-epoch training curve as a compact markdown table.

    Returns None if no epoch data exists for this iteration.
    Shows the last `tail_epochs` epochs to keep context window manageable.
    """
    epochs = get_epoch_metrics(conn, run_id, iteration_num)
    if not epochs:
        return None

    # Only show tail
    epochs = epochs[-tail_epochs:]

    # Collect all metric keys across epochs
    all_keys: list[str] = []
    parsed: list[tuple[int, dict[str, float]]] = []
    for em in epochs:
        m = json.loads(em.metrics)
        parsed.append((em.epoch, m))
        for k in m:
            if k not in all_keys:
                all_keys.append(k)

    if not all_keys:
        return None

    # Build table
    lines = []
    total = get_epoch_metrics(conn, run_id, iteration_num)
    total_epochs = len(total)
    shown = len(parsed)
    header = f"## Iteration {iteration_num} Training Curve (last {shown} of {total_epochs} epochs)"
    lines.append(header)
    lines.append("")

    # Header row
    cols = ["Epoch"] + [k for k in all_keys]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")

    for epoch_num, metrics in parsed:
        vals = [str(epoch_num)]
        for k in all_keys:
            v = metrics.get(k)
            vals.append(f"{v:.4f}" if v is not None else "-")
        lines.append("| " + " | ".join(vals) + " |")

    # One-line summary: trend analysis
    if len(parsed) >= 3:
        summary = _summarize_curve(parsed, all_keys)
        if summary:
            lines.append("")
            lines.append(f"**Trend:** {summary}")

    return "\n".join(lines)


def _summarize_curve(
    parsed: list[tuple[int, dict[str, float]]],
    keys: list[str],
) -> str:
    """Generate a one-line trend summary from epoch data."""
    parts = []
    for key in keys:
        values = [m.get(key) for _, m in parsed if m.get(key) is not None]
        if len(values) < 3:
            continue
        # Compare first third vs last third
        n = len(values)
        first = sum(values[: n // 3]) / (n // 3)
        last = sum(values[-(n // 3) :]) / (n // 3)
        if "loss" in key.lower():
            if last < first * 0.95:
                parts.append(f"{key} decreasing")
            elif last > first * 1.05:
                parts.append(f"{key} increasing (possible issue)")
            else:
                parts.append(f"{key} plateaued")
        else:
            if last > first * 1.02:
                parts.append(f"{key} improving")
            elif last < first * 0.98:
                parts.append(f"{key} declining")
            else:
                parts.append(f"{key} stable")
    return ", ".join(parts) if parts else ""
