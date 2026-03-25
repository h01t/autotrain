"""Main agent loop — orchestrates one full training run."""

from __future__ import annotations

import uuid

import structlog

from autotrain.agent.client import AgentClient
from autotrain.agent.parser import ParseError, parse_response
from autotrain.agent.prompt import build_system_prompt, build_user_message, load_template
from autotrain.config.defaults import MAX_CONSECUTIVE_CRASHES, MAX_SANDBOX_RETRIES
from autotrain.config.schema import RunConfig
from autotrain.core.budget import BudgetExhausted, BudgetTracker
from autotrain.core.state_machine import AgentState, StateMachine
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
from autotrain.experiment.metrics import extract_metric_from_output
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
    create_iteration,
    create_run,
    increment_run_iterations,
    record_metric,
    update_iteration,
    update_run_best,
    update_run_status,
)
from autotrain.util.signals import is_shutting_down
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
        self._watchdog = WatchdogMonitor(
            config.watchdog, self._repo, on_alert=self._on_watchdog_alert,
        )

        # State
        self._sm: StateMachine | None = None
        self._template = load_template()
        self._system_prompt = build_system_prompt(config, self._template)
        self._consecutive_crashes = 0

    def run(self) -> RunStatus:
        """Execute the full agent loop. Returns final status."""
        self._notifier.notify("started", run_id=self._run_id, repo=str(self._repo))

        try:
            self._initialize()
            self._watchdog.start()

            while not is_shutting_down() and not self._sm.is_terminal:
                try:
                    self._budget.check()
                except BudgetExhausted as e:
                    log.info("budget_exhausted", reason=str(e))
                    self._sm.transition(AgentState.BUDGET_EXHAUSTED)
                    self._notifier.notify(
                        "budget_exhausted", reason=str(e),
                        **self._budget.summary(),
                    )
                    break

                self._run_iteration()

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

    def _initialize(self) -> None:
        """Set up git, DB records, and state machine."""
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
        self._executor.setup()

    def _run_iteration(self) -> None:
        """Execute one complete iteration of the agent loop."""
        iteration_num = self._sm.advance_iteration()
        increment_run_iterations(self._conn, self._run_id)

        iteration = Iteration(
            run_id=self._run_id,
            iteration_num=iteration_num,
            state=self._sm.state.value,
        )
        iteration_id = create_iteration(self._conn, iteration)

        log.info("iteration_start", iteration=iteration_num)

        try:
            # 1. Call agent
            self._sm.transition(AgentState.CALLING_AGENT)
            decision = self._call_agent_with_validation(iteration_num)
            if decision is None:
                # Sandbox rejected too many times
                update_iteration(
                    self._conn, iteration_id,
                    outcome=IterationOutcome.SANDBOX_REJECTED,
                )
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
            output = self._run_training()

            # 4. Extract metrics
            self._sm.transition(AgentState.EXTRACTING)
            metric_value = extract_metric_from_output(
                output,
                self._config.metric.name,
                self._config.metric.extraction_pattern,
            )

            # 5. Evaluate
            self._sm.transition(AgentState.EVALUATING)
            outcome = self._evaluate(metric_value, iteration_num, iteration_id)

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

            # Update iteration
            update_iteration(
                self._conn, iteration_id,
                outcome=outcome,
                metric_value=metric_value,
                commit_hash=commit_hash,
                agent_reasoning=decision.reasoning,
                agent_hypothesis=decision.hypothesis,
                changes_summary=decision.hypothesis[:200],
            )

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

        except Exception as e:
            log.error("iteration_error", iteration=iteration_num, error=str(e))
            self._consecutive_crashes += 1
            update_iteration(
                self._conn, iteration_id,
                outcome=IterationOutcome.CRASHED,
                error_message=str(e),
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
        """Call agent and validate response, retrying on sandbox rejection."""
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
                last_error=rejection_msg,
                stagnant=(
                    self._consecutive_crashes
                    >= self._config.agent.stagnation_threshold
                ),
            )

            response = self._agent.call(self._system_prompt, user_msg)
            self._budget.record_api_cost(response.cost_estimate)

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
                return decision

            rejection_msg = format_rejection_message(result)
            log.warning("sandbox_rejected", errors=result.errors, attempt=attempt)

        return None

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

    def _run_training(self) -> str:
        """Execute training and collect output."""
        self._executor.sync_files(self._repo)
        output_lines = []
        for line in self._executor.execute(
            self._config.execution.train_command,
            timeout_seconds=self._config.budget.experiment_timeout_seconds,
            env={"CUDA_VISIBLE_DEVICES": self._config.execution.gpu_device or "0"},
        ):
            output_lines.append(line)
            self._watchdog.report_stdout_activity()

        return "\n".join(output_lines)

    def _evaluate(
        self, metric_value: float | None, iteration_num: int, iteration_id: int,
    ) -> IterationOutcome:
        """Evaluate the experiment result."""
        from autotrain.storage.queries import get_run

        if metric_value is None:
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
            )
        return LocalExecutor(working_dir=self._repo)

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
