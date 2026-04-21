"""Main agent loop — orchestrates one full training run."""

from __future__ import annotations

import uuid
from pathlib import Path

import structlog

from autotrain.agent.client import AgentClient
from autotrain.agent.parser import ParseError, parse_response
from autotrain.agent.prompt import build_system_prompt, build_user_message, load_template
from autotrain.config.defaults import MAX_CONSECUTIVE_CRASHES, MAX_SANDBOX_RETRIES
from autotrain.config.schema import RunConfig
from autotrain.core.budget import BudgetExhausted, BudgetTracker
from autotrain.core.state_machine import AgentState, StateMachine
from autotrain.execution.base import ExecutionResult
from autotrain.execution.local import LocalExecutor
from autotrain.execution.ssh import SSHExecutor
from autotrain.experiment.evaluator import is_improved, is_target_hit
from autotrain.experiment.git_ops import (
    commit,
    create_branch,
    has_uncommitted_changes,
    init_repo,
    revert_last_commit,
)
from autotrain.experiment.metrics import extract_metric_from_output, parse_epoch_line
from autotrain.experiment.sandbox import (
    FileChange,
    format_rejection_message,
    validate_changes,
)
from autotrain.notify.dispatcher import NotifyDispatcher
from autotrain.storage.db import init_db
from autotrain.storage.models import (
    Iteration,
    IterationOutcome,
    MetricSnapshot,
    Run,
    RunStatus,
)
from autotrain.storage.queries import (
    add_run_api_cost,
    create_iteration,
    create_run,
    increment_run_iterations,
    record_epoch_metric,
    record_metric,
    update_iteration,
    update_run_best,
    update_run_status,
)
from autotrain.util.signals import is_shutting_down, register_shutdown_callback
from autotrain.watchdog.gpu import query_gpu_local, query_gpu_ssh
from autotrain.watchdog.monitor import WatchdogMonitor

log = structlog.get_logger()


class AgentLoop:
    """The main orchestration loop for autonomous training."""

    def __init__(self, config: RunConfig) -> None:
        self._config = config
        self._run_id = str(uuid.uuid4())[:8]
        self._repo = config.repo_path

        # Initialize storage
        db_path = self._repo / ".autotrain" / "state.db"
        self._conn = init_db(db_path)

        # Initialize components
        self._agent = AgentClient(
            provider=config.agent.provider,
            api_base=config.agent.api_base,
            model=config.agent.model,
            max_retries=config.agent.max_retries,
            retry_base_seconds=config.agent.retry_base_seconds,
            hard_timeout_seconds=config.agent.hard_timeout_seconds,
            temperature=config.agent.temperature,
        )
        self._budget = BudgetTracker(config.budget)
        self._notifier = NotifyDispatcher(
            webhook_url=config.notify.webhook_url,
            webhook_events=config.notify.webhook_events,
            terminal=config.notify.terminal,
        )
        self._executor = self._create_executor()
        register_shutdown_callback(self._executor.kill)
        self._watchdog = WatchdogMonitor(
            config.watchdog, self._repo,
            on_alert=self._on_watchdog_alert,
            gpu_query_fn=self._build_gpu_query_fn(),
            db_path=db_path,
            run_id=self._run_id,
        )

        # State
        self._sm: StateMachine | None = None
        self._template = load_template()
        self._system_prompt = build_system_prompt(config, self._template)
        self._consecutive_crashes = 0
        self._resume_checkpoint: str | None = None  # Set after crash if checkpoint found
        self._last_error: str | None = None  # Error from last crashed iteration

    def run(self) -> RunStatus:
        """Execute the full agent loop. Returns final status."""
        self._notifier.notify("started", run_id=self._run_id, repo=str(self._repo))

        try:
            self._initialize()
            self._watchdog.start()

            while not is_shutting_down() and not self._sm.is_terminal:
                try:
                    self._budget.check()
                    self._run_iteration()
                except BudgetExhausted as e:
                    log.info("budget_exhausted", reason=str(e))
                    self._sm.transition(AgentState.BUDGET_EXHAUSTED)
                    self._notifier.notify(
                        "budget_exhausted", reason=str(e),
                        **self._budget.summary(),
                    )
                    break

            final_status = self._resolve_final_status()
            update_run_status(self._conn, self._run_id, final_status)
            return final_status

        except Exception as e:
            log.error("agent_loop_fatal", error=str(e), exc_info=True)
            self._notifier.notify("failed", error=str(e))
            update_run_status(self._conn, self._run_id, RunStatus.FAILED)
            return RunStatus.FAILED

        finally:
            self._watchdog.stop()
            self._executor.cleanup()
            # Kill remote agent if running
            if isinstance(self._executor, SSHExecutor):
                try:
                    self._executor._ssh_run(
                        "pkill -f 'standalone.py.*ws/agent' 2>/dev/null || true",
                        timeout=10, check=False,
                    )
                except Exception:
                    pass

    def _validate_config(self) -> None:
        """Validate configuration before starting. Fail fast with clear messages."""
        # 5C: Validate train_command
        cmd = self._config.execution.train_command
        if cmd == "python train.py" and not (self._repo / "train.py").exists():
            py_files = [f.name for f in self._repo.glob("*.py")]
            raise FileNotFoundError(
                f"Default train_command 'python train.py' but train.py not found. "
                f"Available .py files: {py_files or 'none'}. "
                f"Set execution.train_command in autotrain.yaml."
            )

        # 5D: Validate writable_files exist
        existing = [
            f for f in self._config.sandbox.writable_files
            if (self._repo / f).exists()
        ]
        if not existing:
            raise FileNotFoundError(
                f"No writable files found: {self._config.sandbox.writable_files}. "
                f"Check sandbox.writable_files in autotrain.yaml."
            )

    def _initialize(self) -> None:
        """Set up git, DB records, and state machine."""
        self._validate_config()
        init_repo(self._repo)
        branch = f"autotrain/{self._run_id}"
        create_branch(self._repo, branch)

        run = Run(
            id=self._run_id,
            repo_path=str(self._repo),
            metric_name=self._config.metric.name,
            metric_target=self._config.metric.target,
            metric_direction=self._config.metric.direction,
            status=RunStatus.RUNNING,
            git_branch=branch,
            config_snapshot=self._config.model_dump_json(),
        )
        create_run(self._conn, run)

        self._sm = StateMachine(self._conn, self._run_id)
        self._sm.transition(AgentState.READING_STATE)
        self._executor.setup(repo_path=self._repo)

        # Start remote metrics agent on the GPU machine if configured
        if (
            self._config.execution.mode == "ssh"
            and self._config.execution.dashboard_url
            and isinstance(self._executor, SSHExecutor)
        ):
            self._start_remote_agent()

    def _run_iteration(self) -> None:
        """Execute one complete iteration of the agent loop."""
        iteration_num = self._sm.advance_iteration()
        increment_run_iterations(self._conn, self._run_id)

        was_resumed = self._resume_checkpoint is not None
        iteration = Iteration(
            run_id=self._run_id,
            iteration_num=iteration_num,
            state=self._sm.state.value,
            resumed_from_checkpoint=was_resumed,
        )
        iteration_id = create_iteration(self._conn, iteration)

        log.info(
            "iteration_start", iteration=iteration_num,
            resuming=was_resumed,
        )

        try:
            # 1. Call agent
            self._sm.transition(AgentState.CALLING_AGENT)
            decision, rejection_detail = self._call_agent_with_validation(iteration_num)
            if decision is None:
                # Sandbox rejected too many times
                update_iteration(
                    self._conn, iteration_id,
                    outcome=IterationOutcome.SANDBOX_REJECTED,
                    error_message=rejection_detail,
                )
                self._last_error = rejection_detail
                self._sm.transition(AgentState.READING_STATE)
                return

            # 2. Apply changes
            self._sm.transition(AgentState.APPLYING)
            self._apply_changes(decision.changes)

            # Sanitize hypothesis for commit message (strip newlines, limit length)
            hyp = (decision.hypothesis or "no hypothesis").replace("\n", " ")[:80]
            commit_hash = commit(
                self._repo,
                f"Iter {iteration_num}: {hyp}",
                files=[c.file for c in decision.changes],
            )

            # 3. Execute training
            self._sm.transition(AgentState.EXECUTING)
            output = self._run_training(iteration_num)

            # 4. Extract metrics
            self._sm.transition(AgentState.EXTRACTING)
            metric_result = extract_metric_from_output(
                output,
                self._config.metric.name,
                self._config.metric.extraction_pattern,
            )
            metric_value = metric_result.value

            # 5. Evaluate
            self._sm.transition(AgentState.EVALUATING)
            exec_result = self._executor.get_result()

            # Build error message for failed iterations
            error_message = None
            if metric_value is None:
                error_parts = []
                if exec_result.exit_code != 0:
                    error_parts.append(f"Exit code: {exec_result.exit_code}")
                if metric_result.error:
                    error_parts.append(metric_result.error)
                # Include last 30 lines of output for context
                tail = "\n".join(output.splitlines()[-30:])
                if tail:
                    error_parts.append(f"--- Last output ---\n{tail}")
                error_message = "\n".join(error_parts) if error_parts else "Unknown error"
                self._last_error = error_message

            outcome = self._evaluate(metric_value, iteration_num, iteration_id, exec_result)

            if outcome == IterationOutcome.REGRESSED:
                revert_last_commit(self._repo)

            # Record metric
            if metric_value is not None:
                record_metric(self._conn, MetricSnapshot(
                    run_id=self._run_id,
                    iteration_num=iteration_num,
                    metric_name=self._config.metric.name,
                    value=metric_value,
                ))
                self._last_error = None  # Clear error on success

            # Detect checkpoint created by this training run
            checkpoint = self._executor.detect_checkpoint(
                self._config.execution.checkpoint_patterns,
            )

            # Update iteration
            update_iteration(
                self._conn, iteration_id,
                outcome=outcome,
                metric_value=metric_value,
                commit_hash=commit_hash,
                agent_reasoning=decision.reasoning,
                agent_hypothesis=decision.hypothesis,
                changes_summary=decision.hypothesis[:200],
                checkpoint_path=checkpoint,
                error_message=error_message,
            )

            if metric_value is not None:
                self._consecutive_crashes = 0

            # 6. Check target
            if metric_value and is_target_hit(metric_value, self._config.metric):
                self._sm.transition(AgentState.COMPLETED)
                self._notifier.notify(
                    "target_hit",
                    metric_value=metric_value,
                    iteration=iteration_num,
                )
                return

            # 7. Continue
            self._sm.transition(AgentState.DECIDING_NEXT)
            self._sm.transition(AgentState.READING_STATE)
            self._notifier.notify(
                "iteration",
                iteration=iteration_num,
                metric_value=metric_value,
                outcome=outcome.value if outcome else "unknown",
            )

        except BudgetExhausted:
            raise  # Let the main loop handle budget exhaustion

        except Exception as e:
            log.error("iteration_error", iteration=iteration_num, error=str(e))
            self._consecutive_crashes += 1

            # Check for checkpoint before reverting — we may be able to resume
            checkpoint = self._executor.detect_checkpoint(
                self._config.execution.checkpoint_patterns,
            )
            if checkpoint:
                self._resume_checkpoint = checkpoint
                log.info("checkpoint_found_after_crash", path=checkpoint)

            update_iteration(
                self._conn, iteration_id,
                outcome=IterationOutcome.CRASHED,
                error_message=str(e),
                checkpoint_path=checkpoint,
            )

            # Revert uncommitted changes
            if has_uncommitted_changes(self._repo):
                from autotrain.experiment.git_ops import _run_git
                _run_git(self._repo, "checkout", ".", check=False)

            if self._consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                self._sm.transition(AgentState.FAILED)
                self._notifier.notify(
                    "failed",
                    reason=f"{MAX_CONSECUTIVE_CRASHES} consecutive crashes",
                )
                return

            # Try to recover to reading state
            if self._sm.state not in (AgentState.READING_STATE,):
                try:
                    self._sm.transition(AgentState.READING_STATE)
                except ValueError:
                    self._sm.transition(AgentState.FAILED)

    def _call_agent_with_validation(self, iteration_num):
        """Call agent and validate response, retrying on sandbox rejection.

        Returns (decision, rejection_detail) tuple. decision is None if all retries exhausted.
        """
        from autotrain.storage.queries import get_run

        run = get_run(self._conn, self._run_id)
        current_files = self._read_writable_files()
        rejection_msg = None

        for attempt in range(MAX_SANDBOX_RETRIES):
            if attempt > 0 and self._sm.state != AgentState.CALLING_AGENT:
                self._sm.transition(AgentState.CALLING_AGENT)

            user_msg = build_user_message(
                self._config, self._conn, self._run_id,
                current_best=run.best_metric_value,
                iteration_num=iteration_num,
                current_files=current_files,
                last_error=rejection_msg or self._last_error,
                stagnant=(
                    self._consecutive_crashes
                    >= self._config.agent.stagnation_threshold
                ),
            )

            response = self._agent.call(self._system_prompt, user_msg)
            self._budget.record_api_cost(response.cost_estimate)
            add_run_api_cost(self._conn, self._run_id, response.cost_estimate)

            try:
                decision = parse_response(response.raw_text)
            except ParseError as e:
                log.warning("parse_error", error=str(e), attempt=attempt)
                rejection_msg = f"Parse error: {e}"
                continue

            # Validate
            self._sm.transition(AgentState.VALIDATING)
            result = validate_changes(
                decision.changes, self._config.sandbox, current_files,
            )
            if result.is_valid:
                return decision, None

            rejection_msg = format_rejection_message(result, current_files)
            log.warning("sandbox_rejected", errors=result.errors, attempt=attempt)

        return None, rejection_msg

    def _apply_changes(self, changes: list[FileChange]) -> None:
        """Apply file changes to the repo."""
        for change in changes:
            filepath = self._repo / change.file
            if change.action in ("create", "full_rewrite"):
                filepath.write_text(change.content or "")
            elif change.action == "replace" and filepath.exists():
                content = filepath.read_text()
                if change.search and change.search in content:
                    content = content.replace(change.search, change.replace or "", 1)
                    filepath.write_text(content)

    def _run_training(self, iteration_num: int) -> str:
        """Execute training and collect output, capturing per-epoch metrics."""
        self._executor.sync_files(self._repo)
        output_lines = []
        current_epoch = 0
        pending_val_metrics: dict[str, float] = {}

        env = {}
        if self._config.execution.gpu_device is not None:
            env["CUDA_VISIBLE_DEVICES"] = self._config.execution.gpu_device
        if self._resume_checkpoint:
            env["AUTOTRAIN_RESUME_FROM"] = self._resume_checkpoint
            log.info("resuming_from_checkpoint", path=self._resume_checkpoint)
            self._resume_checkpoint = None  # Consume it

        for line in self._executor.execute(
            self._config.execution.train_command,
            timeout_seconds=self._config.budget.experiment_timeout_seconds,
            env=env,
        ):
            output_lines.append(line)
            self._watchdog.report_stdout_activity()

            # Respond to shutdown signal immediately
            if is_shutting_down():
                log.info("shutdown_during_training")
                self._executor.kill()
                break

            # Check time budget mid-training — don't let a single run exceed total budget
            if (
                self._config.budget.time_seconds
                and self._budget.elapsed_seconds >= self._config.budget.time_seconds
            ):
                log.warning(
                    "time_budget_exceeded_mid_training",
                    elapsed=f"{self._budget.elapsed_seconds:.0f}s"
                )
                self._executor.kill()
                raise BudgetExhausted(
                    f"Time budget exhausted during training: {self._budget.elapsed_seconds:.0f}s "
                    f">= {self._config.budget.time_seconds}s"
                )

            # Try to extract per-epoch metrics
            epoch, metrics = parse_epoch_line(line)
            if epoch is not None:
                # Training line — update current epoch, merge any pending val metrics
                if pending_val_metrics and current_epoch > 0:
                    self._store_epoch_metrics(
                        iteration_num, current_epoch, pending_val_metrics,
                    )
                    pending_val_metrics = {}
                current_epoch = epoch
                pending_val_metrics.update(metrics)
            elif metrics and current_epoch > 0:
                # Validation line (no epoch num) — attach to current epoch
                pending_val_metrics.update(metrics)

        # Flush last epoch's metrics
        if pending_val_metrics and current_epoch > 0:
            self._store_epoch_metrics(
                iteration_num, current_epoch, pending_val_metrics,
            )

        return "\n".join(output_lines)

    def _store_epoch_metrics(
        self, iteration_num: int, epoch: int, metrics: dict[str, float],
    ) -> None:
        """Persist per-epoch metrics to the database."""
        import json
        record_epoch_metric(
            self._conn, self._run_id, iteration_num, epoch,
            json.dumps(metrics),
        )

    def _evaluate(
        self,
        metric_value: float | None,
        iteration_num: int,
        iteration_id: int,
        exec_result: ExecutionResult | None = None,
    ) -> IterationOutcome:
        """Evaluate the experiment result."""
        from autotrain.storage.queries import get_run

        if metric_value is None:
            self._consecutive_crashes += 1
            if exec_result and exec_result.exit_code != 0:
                return IterationOutcome.SCRIPT_ERROR
            return IterationOutcome.CRASHED

        run = get_run(self._conn, self._run_id)

        if is_improved(
            metric_value, run.best_metric_value, self._config.metric.direction,
        ):
            update_run_best(self._conn, self._run_id, metric_value, iteration_num)
            self._notifier.notify(
                "improved", metric_value=metric_value, iteration=iteration_num,
            )
            return IterationOutcome.IMPROVED

        return IterationOutcome.REGRESSED

    def _read_writable_files(self) -> dict[str, str]:
        """Read current contents of whitelisted files."""
        files = {}
        for filename in self._config.sandbox.writable_files:
            filepath = self._repo / filename
            if filepath.exists():
                files[filename] = filepath.read_text()
        return files

    def _remote_python(self) -> str:
        """Extract Python executable path for the remote machine.

        Infers from train_command if it contains a path (e.g. '.venv/bin/python'),
        otherwise falls back to 'python3'.
        """
        cmd = self._config.execution.train_command
        # If train_command starts with a python path, use it
        parts = cmd.split()
        if parts and ("python" in parts[0] or parts[0].endswith("/python3")):
            return parts[0]
        return "python3"

    def _start_remote_agent(self) -> None:
        """Deploy and start the remote metrics agent on the GPU machine."""
        cfg = self._config.execution
        base_url = cfg.dashboard_url.rstrip("/")
        agent_url = f"{base_url}/ws/agent/{self._run_id}"
        remote_dir = cfg.ssh_remote_dir
        log_path = f"{remote_dir}/.autotrain/train_output.log"

        # Kill any existing remote agent
        self._executor._ssh_run(
            "pkill -f 'standalone.py.*ws/agent' 2>/dev/null || true",
            timeout=10, check=False,
        )

        # Deploy standalone agent script (single file, no package dependency)
        agent_script = Path(__file__).parent.parent / "remote_agent" / "standalone.py"
        if not agent_script.exists():
            log.warning("remote_agent_script_not_found", path=str(agent_script))
            return

        import subprocess as _sp
        host = self._executor._host
        remote_script = f"{remote_dir}/.autotrain/standalone.py"

        try:
            # Copy script to remote
            _sp.run(
                ["scp", "-P", str(cfg.ssh_port), str(agent_script), f"{host}:{remote_script}"],
                check=True, capture_output=True, timeout=15,
            )

            # Ensure websockets is available for the remote agent
            python_cmd = self._remote_python()
            self._executor._ssh_run(
                f"cd {remote_dir} && {python_cmd} -c 'import websockets' 2>/dev/null "
                f"|| {python_cmd} -m pip install websockets -q 2>/dev/null || true",
                timeout=60, check=False,
            )

            # Start it in the background
            cmd = (
                f"cd {remote_dir} && "
                f"nohup {python_cmd} {remote_script} "
                f"--server '{agent_url}' "
                f"--log-path '{log_path}' "
                f"--interval 1.0 "
                f"> {remote_dir}/.autotrain/remote_agent.log 2>&1 </dev/null &"
            )
            self._executor._ssh_run(cmd, timeout=10, check=False)
            log.info("remote_agent_started", url=agent_url)
        except Exception as e:
            log.warning("remote_agent_start_failed", error=str(e))

    def _create_executor(self):
        """Create the appropriate executor based on config."""
        cfg = self._config.execution
        if cfg.mode == "ssh":
            return SSHExecutor(
                host=cfg.ssh_host,
                remote_dir=cfg.ssh_remote_dir,
                ssh_key=cfg.ssh_key,
                ssh_port=cfg.ssh_port,
                rsync_excludes=cfg.rsync_excludes,
                setup_command=cfg.ssh_setup_command,
            )
        return LocalExecutor(working_dir=self._repo)

    def _build_gpu_query_fn(self):
        """Build GPU query callable based on execution mode (local vs SSH)."""
        if self._config.execution.mode == "ssh":
            return lambda: query_gpu_ssh(self._executor._ssh_run)
        return query_gpu_local

    def _on_watchdog_alert(self, message: str) -> None:
        self._notifier.notify("error", message=message)

    def _resolve_final_status(self) -> RunStatus:
        if self._sm.state == AgentState.COMPLETED:
            return RunStatus.COMPLETED
        if self._sm.state == AgentState.BUDGET_EXHAUSTED:
            return RunStatus.BUDGET_EXHAUSTED
        if self._sm.state == AgentState.STOPPED:
            return RunStatus.STOPPED
        if self._sm.state == AgentState.FAILED:
            return RunStatus.FAILED
        if is_shutting_down():
            return RunStatus.STOPPED
        return RunStatus.FAILED
