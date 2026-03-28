"""CLI entry point for AutoTrain."""

from __future__ import annotations

from pathlib import Path

import click

from autotrain import __version__


@click.group()
@click.version_option(__version__)
def cli():
    """AutoTrain — Autonomous ML training platform."""
    pass


@cli.command()
@click.option("--repo", required=True, type=click.Path(exists=True), help="Path to ML project")
@click.option("--metric", default=None, help="Target metric name (e.g., val_auc)")
@click.option("--target", default=None, type=float, help="Target metric value")
@click.option("--budget", default="4h", help="Time budget (e.g., 4h, 30m)")
@click.option("--max-iterations", default=None, type=int, help="Max iterations")
@click.option("--gpu", default=None, help="GPU device (e.g., cuda:0)")
@click.option("--direction", default="maximize", type=click.Choice(["maximize", "minimize"]))
@click.option("--train-command", default="python train.py", help="Training command")
@click.option("--config", "config_file", default=None, type=click.Path(), help="Config YAML")
@click.option("--ssh-host", default=None, help="Remote GPU host for SSH execution")
@click.option("--ssh-remote-dir", default=None, help="Remote working directory")
@click.option("--webhook", default=None, help="Webhook URL for notifications")
@click.option(
    "--provider", default=None,
    type=click.Choice(["anthropic", "ollama", "deepseek"]), help="LLM provider",
)
@click.option("--model", "agent_model", default=None, help="LLM model name")
@click.option("--api-base", default=None, help="API base URL (for Ollama)")
@click.option("--dashboard-url", default=None, help="Dashboard URL for remote agent (e.g. ws://myhost:8000)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def run(
    repo, metric, target, budget, max_iterations, gpu, direction,
    train_command, config_file, ssh_host, ssh_remote_dir, webhook,
    provider, agent_model, api_base, dashboard_url, verbose,
):
    """Start an autonomous training run."""
    from autotrain.config.loader import load_config
    from autotrain.core.agent_loop import AgentLoop
    from autotrain.daemon import is_running, remove_pid, write_pid
    from autotrain.util.logging import configure_logging
    from autotrain.util.signals import install_signal_handlers

    repo_path = Path(repo).resolve()

    if is_running(repo_path):
        click.echo("Error: AutoTrain is already running for this repo.", err=True)
        raise SystemExit(1)

    # Build CLI overrides (only set fields explicitly provided)
    overrides = {}
    if metric or target:
        overrides["metric"] = {"direction": direction}
        if metric:
            overrides["metric"]["name"] = metric
        if target is not None:
            overrides["metric"]["target"] = target
    overrides["budget"] = {"time_seconds": budget}
    overrides["execution"] = {}
    if train_command != "python train.py":
        overrides["execution"]["train_command"] = train_command
    if max_iterations:
        overrides["budget"]["max_iterations"] = max_iterations
    if gpu:
        overrides["execution"]["gpu_device"] = gpu
    if ssh_host:
        overrides["execution"]["mode"] = "ssh"
        overrides["execution"]["ssh_host"] = ssh_host
    if ssh_remote_dir:
        overrides["execution"]["ssh_remote_dir"] = ssh_remote_dir
    if dashboard_url:
        overrides["execution"]["dashboard_url"] = dashboard_url
    if webhook:
        overrides["notify"] = {"webhook_url": webhook}
    if provider or agent_model or api_base:
        overrides["agent"] = {}
        if provider:
            overrides["agent"]["provider"] = provider
        if agent_model:
            overrides["agent"]["model"] = agent_model
        if api_base:
            overrides["agent"]["api_base"] = api_base

    config = load_config(
        repo_path,
        cli_overrides=overrides,
        config_file=Path(config_file) if config_file else None,
    )

    # Setup
    log_file = repo_path / ".autotrain" / "autotrain.log.jsonl"
    configure_logging(log_file=log_file, verbose=verbose)
    install_signal_handlers()
    write_pid(repo_path)

    click.echo(f"AutoTrain v{__version__} starting")
    click.echo(f"  Repo: {repo_path}")
    click.echo(f"  Target: {config.metric.name} {'>=' if config.metric.direction == 'maximize' else '<='} {config.metric.target}")
    click.echo(f"  Budget: {budget}")
    click.echo(f"  Logs: {log_file}")
    click.echo()

    try:
        loop = AgentLoop(config)
        status = loop.run()
        click.echo(f"\nTraining complete: {status.value}")
    finally:
        remove_pid(repo_path)


@cli.command()
@click.option("--repo", default=".", type=click.Path(exists=True), help="Path to ML project")
def status(repo):
    """Show status of current/recent runs."""
    from autotrain.daemon import is_running, read_pid
    from autotrain.storage.db import get_connection
    from autotrain.storage.queries import get_latest_run

    repo_path = Path(repo).resolve()
    db_path = repo_path / ".autotrain" / "state.db"

    if not db_path.exists():
        click.echo("No AutoTrain data found in this repo.")
        return

    running = is_running(repo_path)
    pid = read_pid(repo_path) if running else None
    click.echo(f"Running: {'yes (PID ' + str(pid) + ')' if running else 'no'}")

    conn = get_connection(db_path)
    run = get_latest_run(conn, str(repo_path))
    conn.close()

    if run is None:
        click.echo("No runs found.")
        return

    click.echo(f"Latest run: {run.id}")
    click.echo(f"  Status: {run.status.value}")
    click.echo(f"  Metric: {run.metric_name}")
    click.echo(f"  Target: {run.metric_target}")
    click.echo(f"  Best: {run.best_metric_value or 'N/A'}")
    click.echo(f"  Iterations: {run.total_iterations}")
    click.echo(f"  API cost: ${run.total_api_cost:.4f}")


@cli.command()
@click.option("--repo", default=".", type=click.Path(exists=True), help="Path to ML project")
@click.option("-n", default=20, help="Number of entries to show")
def history(repo, n):
    """Show experiment history."""
    from autotrain.storage.db import get_connection
    from autotrain.storage.queries import get_latest_run, get_recent_iterations

    repo_path = Path(repo).resolve()
    db_path = repo_path / ".autotrain" / "state.db"

    if not db_path.exists():
        click.echo("No AutoTrain data found.")
        return

    conn = get_connection(db_path)
    run = get_latest_run(conn, str(repo_path))
    if not run:
        click.echo("No runs found.")
        conn.close()
        return

    iterations = get_recent_iterations(conn, run.id, limit=n)
    conn.close()

    click.echo(f"Run {run.id} — {run.metric_name} target: {run.metric_target}")
    click.echo(f"{'#':>4} {'Metric':>10} {'Outcome':>12} {'Changes'}")
    click.echo("-" * 60)

    for it in iterations:
        metric = f"{it.metric_value:.4f}" if it.metric_value is not None else "N/A"
        outcome = it.outcome.value if it.outcome else "?"
        changes = (it.changes_summary or "")[:40]
        click.echo(f"{it.iteration_num:>4} {metric:>10} {outcome:>12} {changes}")


@cli.command()
@click.option("--repo", default=".", type=click.Path(exists=True), help="Path to ML project")
def stop(repo):
    """Stop a running AutoTrain process."""
    import signal

    from autotrain.daemon import is_running, read_pid

    repo_path = Path(repo).resolve()

    if not is_running(repo_path):
        click.echo("No AutoTrain process running for this repo.")
        return

    pid = read_pid(repo_path)
    try:
        import os
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Sent SIGTERM to PID {pid}.")
    except OSError as e:
        click.echo(f"Failed to stop: {e}", err=True)


@cli.command()
@click.option("--repo", default=".", type=click.Path(exists=True), help="Path to ML project")
@click.option("--port", default=8000, type=int, help="Dashboard server port")
@click.option("--host", default="127.0.0.1", help="Dashboard server host")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def dashboard(repo, port, host, no_browser):
    """Open the web monitoring dashboard (React + FastAPI)."""
    import uvicorn

    from autotrain.dashboard.server import create_app

    repo_path = Path(repo).resolve()
    db_path = repo_path / ".autotrain" / "state.db"

    if not db_path.exists():
        click.echo("No AutoTrain data found. Run a training first.", err=True)
        raise SystemExit(1)

    app = create_app(db_path)

    # Always open browser at localhost (0.0.0.0 is a bind address, not browsable)
    browse_host = "127.0.0.1" if host == "0.0.0.0" else host

    click.echo(f"Starting AutoTrain Dashboard on http://{browse_host}:{port}")
    click.echo(f"  Database: {db_path}")
    click.echo(f"  API docs: http://{browse_host}:{port}/docs")

    if not no_browser:
        import webbrowser
        import threading
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{browse_host}:{port}")).start()

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    except KeyboardInterrupt:
        click.echo("\nDashboard stopped.")


@cli.command()
@click.option("--repo", default=".", type=click.Path(exists=True), help="Path to ML project")
@click.option("--port", default=8501, type=int, help="Streamlit server port")
@click.option("--refresh", default=10, type=int, help="Auto-refresh interval in seconds")
def monitor(repo, port, refresh):
    """Open the Streamlit monitoring dashboard (legacy)."""
    import subprocess

    repo_path = Path(repo).resolve()
    db_path = repo_path / ".autotrain" / "state.db"

    if not db_path.exists():
        click.echo("No AutoTrain data found. Run a training first.", err=True)
        raise SystemExit(1)

    app_path = Path(__file__).parent / "monitor" / "app.py"
    click.echo(f"Starting AutoTrain Monitor on port {port}...")
    click.echo(f"  Database: {db_path}")

    try:
        subprocess.run(
            [
                "streamlit", "run", str(app_path),
                "--server.port", str(port),
                "--server.headless", "false",
                "--", "--db-path", str(db_path), "--refresh", str(refresh),
            ],
            check=True,
        )
    except KeyboardInterrupt:
        click.echo("\nMonitor stopped.")
    except FileNotFoundError:
        click.echo("Error: streamlit not found. Install with: pip install streamlit", err=True)
        raise SystemExit(1)
