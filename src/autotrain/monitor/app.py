"""AutoTrain Monitor — Streamlit dashboard for tracking training runs."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Add src to path so we can import autotrain modules
_src = str(Path(__file__).resolve().parent.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

from autotrain.storage.models import IterationOutcome
from autotrain.storage.queries import (
    get_all_metric_snapshots,
    get_latest_run,
    get_recent_iterations,
)

# -- Config from CLI args --
_args = sys.argv[1:]
DB_PATH = Path(_args[_args.index("--db-path") + 1]) if "--db-path" in _args else None
REFRESH = int(_args[_args.index("--refresh") + 1]) if "--refresh" in _args else 10

OUTCOME_COLORS = {
    "improved": "#22c55e",
    "regressed": "#ef4444",
    "crashed": "#f97316",
    "sandbox_rejected": "#eab308",
    "no_change": "#94a3b8",
    "timeout": "#f97316",
}


def get_readonly_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def main():
    st.set_page_config(page_title="AutoTrain Monitor", layout="wide")
    st.title("AutoTrain Monitor")

    if DB_PATH is None or not DB_PATH.exists():
        st.error("No database found. Run `autotrain monitor --repo <path>` to start.")
        return

    conn = get_readonly_connection(DB_PATH)
    run = get_latest_run(conn)

    if run is None:
        st.info("No runs found in this database yet.")
        conn.close()
        return

    # -- Header metrics --
    status_icon = {
        "running": ":large_green_circle:",
        "completed": ":white_check_mark:",
        "budget_exhausted": ":warning:",
        "failed": ":red_circle:",
        "stopped": ":black_circle:",
    }.get(run.status.value, ":question:")

    direction_symbol = ">=" if run.metric_direction == "maximize" else "<="

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", f"{run.status.value.upper()}")
    col2.metric("Best", f"{run.best_metric_value:.4f}" if run.best_metric_value else "N/A")
    col3.metric("Iterations", run.total_iterations)
    col4.metric("API Cost", f"${run.total_api_cost:.2f}")

    st.caption(
        f"{status_icon} Run `{run.id}` | "
        f"Target: **{run.metric_name}** {direction_symbol} {run.metric_target} | "
        f"Branch: `{run.git_branch or 'N/A'}`"
    )

    st.divider()

    # -- Metric Progress Chart --
    st.subheader("Metric Progress")
    snapshots = get_all_metric_snapshots(conn, run.id)

    if snapshots:
        iterations_data = get_recent_iterations(conn, run.id, limit=500)
        outcome_map = {it.iteration_num: it.outcome for it in iterations_data}

        iters = [s.iteration_num for s in snapshots]
        values = [s.value for s in snapshots]
        colors = []
        for s in snapshots:
            oc = outcome_map.get(s.iteration_num)
            key = oc.value if isinstance(oc, IterationOutcome) else str(oc or "no_change")
            colors.append(OUTCOME_COLORS.get(key, "#94a3b8"))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iters, y=values, mode="lines+markers",
            marker=dict(color=colors, size=8),
            line=dict(color="#3b82f6", width=2),
            name=run.metric_name,
        ))
        fig.add_hline(
            y=run.metric_target, line_dash="dash", line_color="#ef4444",
            annotation_text=f"Target: {run.metric_target}",
        )
        if run.best_metric_value:
            fig.add_hline(
                y=run.best_metric_value, line_dash="dot", line_color="#22c55e",
                annotation_text=f"Best: {run.best_metric_value:.4f}",
            )
        fig.update_layout(
            xaxis_title="Iteration", yaxis_title=run.metric_name,
            height=350, margin=dict(l=40, r=40, t=20, b=40),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No metric data recorded yet.")

    # -- Iteration History Table --
    st.subheader("Iteration History")
    iterations = get_recent_iterations(conn, run.id, limit=100)

    if iterations:
        table_data = []
        for it in reversed(iterations):  # newest first
            table_data.append({
                "#": it.iteration_num,
                "Metric": f"{it.metric_value:.4f}" if it.metric_value is not None else "N/A",
                "Outcome": it.outcome.value if it.outcome else "?",
                "Hypothesis": (it.agent_hypothesis or "")[:80],
                "Duration": f"{it.duration_seconds:.0f}s" if it.duration_seconds else "-",
                "Commit": (it.commit_hash or "")[:7],
            })
        st.dataframe(table_data, width="stretch", hide_index=True)
    else:
        st.info("No iterations yet.")

    # -- Agent Reasoning Log --
    st.subheader("Agent Reasoning")

    if iterations:
        for it in reversed(iterations):  # newest first
            label = f"Iter {it.iteration_num}"
            if it.agent_hypothesis:
                label += f": {it.agent_hypothesis[:60]}"
            outcome_str = it.outcome.value if it.outcome else "?"
            label += f" [{outcome_str}]"

            with st.expander(label, expanded=(it == iterations[-1])):
                if it.agent_hypothesis:
                    st.markdown(f"**Hypothesis:** {it.agent_hypothesis}")
                if it.agent_reasoning:
                    st.markdown(f"**Reasoning:** {it.agent_reasoning}")
                if it.changes_summary:
                    st.markdown(f"**Changes:** {it.changes_summary}")
                if it.error_message:
                    st.error(f"Error: {it.error_message}")
                if it.metric_value is not None:
                    st.markdown(f"**Metric:** {it.metric_value:.4f}")
    else:
        st.info("No iterations yet.")

    # -- Cost & Budget Tracker --
    st.subheader("Cost & Budget")

    budget_config = _parse_budget_from_config(run.config_snapshot)
    elapsed = (datetime.now(UTC) - run.created_at).total_seconds()

    col1, col2 = st.columns(2)
    with col1:
        if budget_config.get("time_seconds"):
            budget_secs = budget_config["time_seconds"]
            time_pct = min(elapsed / budget_secs, 1.0)
            time_text = f"Time: {_fmt_duration(elapsed)} / {_fmt_duration(budget_secs)}"
            st.progress(time_pct, text=time_text)
        else:
            st.markdown(f"**Elapsed:** {_fmt_duration(elapsed)}")

        if budget_config.get("max_iterations"):
            max_iter = budget_config["max_iterations"]
            iter_pct = min(run.total_iterations / max_iter, 1.0)
            st.progress(iter_pct, text=f"Iterations: {run.total_iterations} / {max_iter}")

    with col2:
        if budget_config.get("max_api_cost"):
            max_cost = budget_config["max_api_cost"]
            cost_pct = min(run.total_api_cost / max_cost, 1.0)
            st.progress(cost_pct, text=f"API Cost: ${run.total_api_cost:.2f} / ${max_cost:.2f}")
        else:
            st.markdown(f"**API Cost:** ${run.total_api_cost:.2f}")

        if run.total_iterations > 0:
            rate_per_iter = run.total_api_cost / run.total_iterations
            time_per_iter = elapsed / run.total_iterations
            st.caption(f"Rate: ~${rate_per_iter:.3f}/iter, ~{_fmt_duration(time_per_iter)}/iter")

    conn.close()

    # -- Auto-refresh --
    if run.status.value == "running" and REFRESH > 0:
        time.sleep(REFRESH)
        st.rerun()


def _parse_budget_from_config(config_snapshot: str | None) -> dict:
    if not config_snapshot:
        return {}
    try:
        config = json.loads(config_snapshot)
        return config.get("budget", {})
    except (json.JSONDecodeError, TypeError):
        return {}


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{int(seconds % 60):02d}s"


if __name__ == "__main__":
    main()
