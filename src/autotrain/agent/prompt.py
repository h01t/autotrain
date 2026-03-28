"""System prompt builder — assembles context for the agent."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from autotrain.agent.framework_detector import detect_framework
from autotrain.config.schema import RunConfig
from autotrain.experiment.history import format_history_for_prompt


def build_system_prompt(
    config: RunConfig,
    template: str,
) -> str:
    """Build the system prompt from the template with config values filled in."""
    writable = ", ".join(f"`{f}`" for f in config.sandbox.writable_files)
    timeout_min = config.budget.experiment_timeout_seconds // 60

    prompt = template
    prompt = prompt.replace("{{ metric_name }}", config.metric.name)
    prompt = prompt.replace("{{ target }}", str(config.metric.target))
    prompt = prompt.replace(
        '{{ "≥" if direction == "maximize" else "≤" }}',
        "≥" if config.metric.direction == "maximize" else "≤",
    )
    prompt = prompt.replace("{{ writable_files }}", writable)
    prompt = prompt.replace("{{ experiment_timeout }}", f"{timeout_min} minutes")

    # Dynamic strategy section based on detected framework
    hint = detect_framework(config.repo_path, config.sandbox.writable_files)
    strategy = _load_strategy(hint.name)
    prompt = prompt.replace("{{ strategy_section }}", strategy)

    return prompt


def _load_strategy(framework: str) -> str:
    """Load the strategy file for a framework, falling back to generic."""
    strategies_dir = Path(__file__).parent / "templates" / "strategies"

    strategy_file = strategies_dir / f"{framework}.md"
    if strategy_file.exists():
        return strategy_file.read_text().strip()

    # Fall back to generic
    generic = strategies_dir / "generic.md"
    return generic.read_text().strip()


def build_user_message(
    config: RunConfig,
    conn: sqlite3.Connection,
    run_id: str,
    current_best: float | None,
    iteration_num: int,
    current_files: dict[str, str],
    last_error: str | None = None,
    stagnant: bool = False,
) -> str:
    """Build the user message with current state for the agent."""
    sections = []

    # Current status
    best_str = f"{current_best:.4f}" if current_best is not None else "no runs yet"
    sections.append("## Current Status")
    sections.append(f"- Iteration: {iteration_num}")
    sections.append(f"- Best {config.metric.name}: {best_str}")
    sections.append(f"- Target: {config.metric.target}")
    sections.append("")

    # Fill in the template placeholder
    if current_best is not None:
        gap = abs(config.metric.target - current_best)
        sections.append(f"- Gap to target: {gap:.4f}")
        sections.append("")

    # Last error (if any)
    if last_error:
        sections.append("## Last Experiment Error")
        sections.append(f"```\n{last_error[:500]}\n```")
        sections.append("")

    # Stagnation warning
    if stagnant:
        sections.append("## ⚠ STAGNATION DETECTED")
        sections.append(
            "The metric has not improved for several iterations. "
            "Try a fundamentally different approach — different architecture, "
            "different optimizer, different augmentation strategy."
        )
        sections.append("")

    # Experiment history
    history = format_history_for_prompt(
        conn, run_id, direction=config.metric.direction,
    )
    sections.append(history)
    sections.append("")

    # Current file contents
    sections.append("## Current Files")
    for filename, content in current_files.items():
        # Truncate large files
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated]"
        sections.append(f"### {filename}")
        sections.append(f"```python\n{content}\n```")
        sections.append("")

    return "\n".join(sections)


def load_template() -> str:
    """Load the program.md template."""
    template_path = Path(__file__).parent / "templates" / "program.md"
    return template_path.read_text()
