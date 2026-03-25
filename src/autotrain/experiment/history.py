"""Format experiment history for agent context."""

from __future__ import annotations

import sqlite3

from autotrain.storage.queries import get_best_iterations, get_recent_iterations


def format_history_for_prompt(
    conn: sqlite3.Connection,
    run_id: str,
    direction: str = "maximize",
    recent_limit: int = 20,
    best_limit: int = 5,
) -> str:
    """Format experiment history as a text table for the agent prompt.

    Includes both recent iterations and top-performing ones.
    """
    recent = get_recent_iterations(conn, run_id, limit=recent_limit)
    best = get_best_iterations(conn, run_id, direction=direction, limit=best_limit)

    lines = []

    # Recent experiments
    lines.append(f"## Recent Experiments (last {len(recent)})")
    lines.append("")
    lines.append(
        "| # | Metric | Outcome | Duration | Changes |"
    )
    lines.append(
        "|---|--------|---------|----------|---------|"
    )

    for it in recent:
        metric_str = f"{it.metric_value:.4f}" if it.metric_value is not None else "N/A"
        outcome_str = it.outcome.value if it.outcome else "?"
        duration_str = f"{it.duration_seconds:.0f}s" if it.duration_seconds else "?"
        changes_str = (it.changes_summary or "")[:50]
        lines.append(
            f"| {it.iteration_num} | {metric_str} | {outcome_str} "
            f"| {duration_str} | {changes_str} |"
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

    return "\n".join(lines)
