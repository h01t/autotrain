"""Crash recovery — read journal and resume from correct state."""

from __future__ import annotations

import sqlite3

import structlog

from autotrain.config.defaults import CHECKPOINT_THRESHOLD_SECONDS
from autotrain.core.state_machine import AgentState, StateMachine
from autotrain.storage.queries import get_run

log = structlog.get_logger()


def attempt_recovery(
    conn: sqlite3.Connection,
    run_id: str,
    experiment_timeout: int,
) -> StateMachine | None:
    """Attempt to recover a crashed run.

    Returns a recovered StateMachine, or None if no recovery is possible.
    """
    run = get_run(conn, run_id)
    if run is None:
        log.warning("recovery_no_run", run_id=run_id)
        return None

    sm = StateMachine.recover(conn, run_id)
    if sm is None:
        log.warning("recovery_no_journal", run_id=run_id)
        return None

    action = sm.get_recovery_action()
    log.info(
        "recovery_attempt",
        state=sm.state.value,
        iteration=sm.iteration_num,
        action=action,
    )

    # Terminal states — nothing to recover
    if sm.is_terminal:
        log.info("recovery_already_terminal", state=sm.state.value)
        return sm

    # For EXECUTING state, apply adaptive recovery
    if sm.state == AgentState.EXECUTING:
        if experiment_timeout <= CHECKPOINT_THRESHOLD_SECONDS:
            # Short experiment — stateless, just skip to next iteration
            log.info("recovery_short_experiment_discard")
        else:
            # Long experiment — check for checkpoint
            log.info("recovery_long_experiment_check_checkpoint")
            # The agent loop will handle actual checkpoint detection

    return sm
