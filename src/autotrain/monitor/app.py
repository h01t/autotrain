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
    get_all_runs,
    get_epoch_metrics,
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

OUTCOME_ICONS = {
    "improved": "+",
    "regressed": "-",
    "crashed": "!",
    "sandbox_rejected": "x",
    "no_change": "=",
    "timeout": "T",
}


def get_readonly_connection(db_path: Path) -> sqlite3.Connection:
    # Run migrations first with a writable connection, then open read-only
    try:
        from autotrain.storage.db import init_db
        migrate_conn = init_db(db_path)
        migrate_conn.close()
    except Exception:
        pass  # Best-effort; read-only still works with graceful column handling

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def main():
    st.set_page_config(page_title="AutoTrain Monitor", layout="wide")

    if DB_PATH is None or not DB_PATH.exists():
        st.error("No database found. Run `autotrain monitor --repo <path>` to start.")
        return

    conn = get_readonly_connection(DB_PATH)

    # -- Sidebar: Run Selector --
    all_runs = get_all_runs(conn)
    if not all_runs:
        st.info("No runs found in this database yet.")
        conn.close()
        return

    with st.sidebar:
        st.title("AutoTrain")
        st.subheader("Runs")

        run_options = {}
        for r in all_runs:
            status_marker = {
                "running": ">>>",
                "completed": "[OK]",
                "budget_exhausted": "[$$]",
                "failed": "[!!]",
                "stopped": "[--]",
            }.get(r.status.value, "[??]")
            best_str = f"{r.best_metric_value:.4f}" if r.best_metric_value else "N/A"
            label = f"{status_marker} {r.id} | {r.metric_name}={best_str} | {r.total_iterations}it"
            run_options[label] = r.id

        selected_label = st.selectbox(
            "Select run",
            options=list(run_options.keys()),
            index=0,
        )
        selected_run_id = run_options[selected_label]
        run = next(r for r in all_runs if r.id == selected_run_id)

        st.divider()

        # Run summary in sidebar
        st.caption(f"Branch: `{run.git_branch or 'N/A'}`")
        st.caption(f"Created: {run.created_at.strftime('%Y-%m-%d %H:%M')}")
        st.caption(f"Cost: ${run.total_api_cost:.3f}")

        if len(all_runs) > 1:
            st.divider()
            st.subheader("All Runs")
            for r in all_runs:
                best = f"{r.best_metric_value:.4f}" if r.best_metric_value else "N/A"
                elapsed = (r.updated_at - r.created_at).total_seconds()
                st.caption(
                    f"`{r.id}` {r.status.value} | "
                    f"{r.metric_name}={best} | "
                    f"{r.total_iterations}it | "
                    f"{_fmt_duration(elapsed)} | "
                    f"${r.total_api_cost:.3f}"
                )

    # -- Main content --
    st.title("AutoTrain Monitor")

    # -- Header metrics --
    status_label = {
        "running": "RUNNING",
        "completed": "COMPLETED",
        "budget_exhausted": "BUDGET EXHAUSTED",
        "failed": "FAILED",
        "stopped": "STOPPED",
    }.get(run.status.value, run.status.value.upper())

    direction_symbol = ">=" if run.metric_direction == "maximize" else "<="

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", f"{run.status.value.upper()}")
    col2.metric("Best", f"{run.best_metric_value:.4f}" if run.best_metric_value else "N/A")
    col3.metric("Iterations", run.total_iterations)
    col4.metric("API Cost", f"${run.total_api_cost:.2f}")

    st.caption(
        f"{status_label} | Run `{run.id}` | "
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

    # -- Training Curves (per-epoch) --
    st.subheader("Training Curves")
    iterations_all = get_recent_iterations(conn, run.id, limit=100)
    iters_with_epochs = [
        it for it in iterations_all
        if get_epoch_metrics(conn, run.id, it.iteration_num)
    ]

    if iters_with_epochs:
        iter_options = {
            f"Iter {it.iteration_num}": it.iteration_num
            for it in reversed(iters_with_epochs)
        }
        default_selection = [list(iter_options.keys())[0]]
        selected = st.multiselect(
            "Select iterations to overlay",
            options=list(iter_options.keys()),
            default=default_selection,
        )
        selected_nums = [iter_options[s] for s in selected]

        if selected_nums:
            _render_training_curves(conn, run.id, selected_nums)
    else:
        st.info("No per-epoch training data yet.")

    # -- Iteration History Table --
    st.subheader("Iteration History")
    iterations = get_recent_iterations(conn, run.id, limit=100)

    if iterations:
        table_data = []
        for it in reversed(iterations):  # newest first
            outcome_str = it.outcome.value if it.outcome else "?"
            icon = OUTCOME_ICONS.get(outcome_str, "?")
            resumed = "R" if it.resumed_from_checkpoint else ""
            ckpt = "C" if it.checkpoint_path else ""
            flags = " ".join(f for f in [resumed, ckpt] if f)

            table_data.append({
                "#": it.iteration_num,
                "Metric": f"{it.metric_value:.4f}" if it.metric_value is not None else "N/A",
                "Outcome": f"[{icon}] {outcome_str}",
                "Hypothesis": (it.agent_hypothesis or "")[:80],
                "Duration": f"{it.duration_seconds:.0f}s" if it.duration_seconds else "-",
                "Commit": (it.commit_hash or "")[:7],
                "Flags": flags,
            })
        st.dataframe(table_data, width="stretch", hide_index=True)

        st.caption("Flags: R=resumed from checkpoint, C=checkpoint saved")
    else:
        st.info("No iterations yet.")

    # -- Iteration Comparison --
    if iterations and len(iterations) >= 2:
        st.subheader("Iteration Comparison")
        _render_comparison(conn, run, iterations)

    # -- Agent Reasoning Log --
    st.subheader("Agent Reasoning")
    _render_agent_reasoning(conn, run.id, iterations)

    # -- Cost & Budget Tracker --
    st.subheader("Cost & Budget")
    _render_budget(run)

    conn.close()

    # -- Auto-refresh --
    if run.status.value == "running" and REFRESH > 0:
        time.sleep(REFRESH)
        st.rerun()


# ---------------------------------------------------------------------------
# Training Curves
# ---------------------------------------------------------------------------

def _render_training_curves(
    conn: sqlite3.Connection,
    run_id: str,
    selected_nums: list[int],
) -> None:
    """Render per-epoch training curves with loss/score split."""
    all_epoch_data: dict[int, list[tuple[int, dict]]] = {}
    all_metric_keys: list[str] = []
    for inum in selected_nums:
        epoch_rows = get_epoch_metrics(conn, run_id, inum)
        parsed = []
        for em in epoch_rows:
            m = json.loads(em.metrics)
            parsed.append((em.epoch, m))
            for k in m:
                if k not in all_metric_keys:
                    all_metric_keys.append(k)
        all_epoch_data[inum] = parsed

    loss_keys = [k for k in all_metric_keys if "loss" in k.lower()]
    score_keys = [k for k in all_metric_keys if "loss" not in k.lower()]

    fig = go.Figure()
    palette = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6"]

    for idx, inum in enumerate(selected_nums):
        data = all_epoch_data[inum]
        prefix = f"Iter {inum}" if len(selected_nums) > 1 else ""
        epochs_x = [e for e, _ in data]

        for key in score_keys:
            vals = [m.get(key) for _, m in data]
            fig.add_trace(go.Scatter(
                x=epochs_x, y=vals, mode="lines+markers",
                name=f"{prefix} {key}".strip(),
                marker=dict(size=4),
                line=dict(width=2, color=palette[idx % len(palette)]),
            ))

        for key in loss_keys:
            vals = [m.get(key) for _, m in data]
            fig.add_trace(go.Scatter(
                x=epochs_x, y=vals, mode="lines",
                name=f"{prefix} {key}".strip(),
                yaxis="y2",
                line=dict(width=1, dash="dot", color=palette[idx % len(palette)]),
                opacity=0.6,
            ))

    layout = dict(
        xaxis_title="Epoch",
        yaxis_title="Score Metrics",
        height=400,
        margin=dict(l=40, r=60, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    if loss_keys:
        layout["yaxis2"] = dict(title="Loss", overlaying="y", side="right")
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Iteration Comparison
# ---------------------------------------------------------------------------

def _render_comparison(conn, run, iterations) -> None:
    """Side-by-side comparison of two selected iterations."""
    iter_labels = {
        f"Iter {it.iteration_num}": it for it in reversed(iterations)
        if it.outcome is not None
    }
    if len(iter_labels) < 2:
        st.info("Need at least 2 completed iterations to compare.")
        return

    labels = list(iter_labels.keys())
    col_a, col_b = st.columns(2)
    with col_a:
        pick_a = st.selectbox("Left", labels, index=min(1, len(labels) - 1), key="cmp_a")
    with col_b:
        pick_b = st.selectbox("Right", labels, index=0, key="cmp_b")

    it_a = iter_labels[pick_a]
    it_b = iter_labels[pick_b]

    # Metric comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{pick_a}**")
        _render_iter_card(it_a)
    with col2:
        st.markdown(f"**{pick_b}**")
        _render_iter_card(it_b)

    # Metric delta
    if it_a.metric_value is not None and it_b.metric_value is not None:
        delta = it_b.metric_value - it_a.metric_value
        direction = "+" if delta > 0 else ""
        better = (delta > 0) == (run.metric_direction == "maximize")
        color = "green" if better else "red"
        st.markdown(
            f"**{run.metric_name} delta:** "
            f":{color}[{direction}{delta:.4f}]"
        )

    # Training curve overlay if both have epoch data
    a_epochs = get_epoch_metrics(conn, run.id, it_a.iteration_num)
    b_epochs = get_epoch_metrics(conn, run.id, it_b.iteration_num)
    if a_epochs and b_epochs:
        st.markdown("**Training Curve Overlay**")
        _render_training_curves(
            conn, run.id,
            [it_a.iteration_num, it_b.iteration_num],
        )


def _render_iter_card(it) -> None:
    """Render a compact summary card for one iteration."""
    metric_str = f"{it.metric_value:.4f}" if it.metric_value is not None else "N/A"
    outcome_str = it.outcome.value if it.outcome else "?"
    color = OUTCOME_COLORS.get(outcome_str, "#94a3b8")
    duration = f"{it.duration_seconds:.0f}s" if it.duration_seconds else "-"

    st.markdown(
        f"- **Metric:** {metric_str}\n"
        f"- **Outcome:** :{_outcome_color_name(outcome_str)}[{outcome_str}]\n"
        f"- **Duration:** {duration}\n"
        f"- **Commit:** `{(it.commit_hash or '')[:7]}`"
    )
    if it.agent_hypothesis:
        st.markdown(f"- **Hypothesis:** {it.agent_hypothesis[:120]}")
    if it.resumed_from_checkpoint:
        st.markdown("- Resumed from checkpoint")
    if it.checkpoint_path:
        st.markdown(f"- Checkpoint: `{it.checkpoint_path.split('/')[-1]}`")


# ---------------------------------------------------------------------------
# Agent Reasoning
# ---------------------------------------------------------------------------

def _render_agent_reasoning(conn, run_id, iterations) -> None:
    """Render agent reasoning log with color-coded outcomes."""
    if not iterations:
        st.info("No iterations yet.")
        return

    for it in reversed(iterations):  # newest first
        outcome_str = it.outcome.value if it.outcome else "?"
        icon = OUTCOME_ICONS.get(outcome_str, "?")
        color_name = _outcome_color_name(outcome_str)

        label = f"[{icon}] Iter {it.iteration_num}"
        if it.agent_hypothesis:
            label += f": {it.agent_hypothesis[:55]}"
        label += f"  ({outcome_str})"

        is_latest = (it == iterations[-1])
        with st.expander(label, expanded=is_latest):
            # Outcome badge
            metric_str = f"{it.metric_value:.4f}" if it.metric_value is not None else "N/A"
            st.markdown(
                f"**Outcome:** :{color_name}[{outcome_str}] | "
                f"**Metric:** {metric_str}"
            )

            if it.resumed_from_checkpoint:
                st.info("Resumed from checkpoint")

            if it.agent_hypothesis:
                st.markdown(f"**Hypothesis:** {it.agent_hypothesis}")

            if it.agent_reasoning:
                st.markdown(f"**Reasoning:**")
                st.text(it.agent_reasoning[:500])

            if it.changes_summary:
                st.markdown(f"**Changes:** {it.changes_summary}")

            if it.error_message:
                st.error(f"Error: {it.error_message}")

            # Footer with metadata
            parts = []
            if it.duration_seconds:
                parts.append(f"{it.duration_seconds:.0f}s")
            if it.commit_hash:
                parts.append(f"`{it.commit_hash[:7]}`")
            if it.checkpoint_path:
                parts.append(f"ckpt: `{it.checkpoint_path.split('/')[-1]}`")
            if parts:
                st.caption(" | ".join(parts))


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

def _render_budget(run) -> None:
    """Render cost & budget tracker."""
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
        if budget_config.get("api_dollars"):
            max_cost = budget_config["api_dollars"]
            cost_pct = min(run.total_api_cost / max_cost, 1.0)
            st.progress(cost_pct, text=f"API Cost: ${run.total_api_cost:.2f} / ${max_cost:.2f}")
        else:
            st.markdown(f"**API Cost:** ${run.total_api_cost:.2f}")

        if run.total_iterations > 0:
            rate_per_iter = run.total_api_cost / run.total_iterations
            time_per_iter = elapsed / run.total_iterations
            st.caption(
                f"Rate: ~${rate_per_iter:.3f}/iter, ~{_fmt_duration(time_per_iter)}/iter"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outcome_color_name(outcome: str) -> str:
    """Map outcome to Streamlit color name for markdown."""
    return {
        "improved": "green",
        "regressed": "red",
        "crashed": "orange",
        "sandbox_rejected": "orange",
        "no_change": "gray",
        "timeout": "orange",
    }.get(outcome, "gray")


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
