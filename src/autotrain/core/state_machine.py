"""Agent loop state machine with journaled transitions for crash recovery.

Every state transition is written to the journal *before* the side effect executes.
On crash, recovery reads the last journal entry and resumes from the correct state.
"""

from __future__ import annotations

import json
import sqlite3
from enum import StrEnum
from typing import Any

import structlog

from autotrain.storage.models import JournalEntry
from autotrain.storage.queries import get_latest_journal, write_journal

log = structlog.get_logger()


class AgentState(StrEnum):
    """States of the agent loop."""

    INITIALIZING = "initializing"
    READING_STATE = "reading_state"
    CALLING_AGENT = "calling_agent"
    VALIDATING = "validating"
    APPLYING = "applying"
    EXECUTING = "executing"
    EXTRACTING = "extracting"
    EVALUATING = "evaluating"
    DECIDING_NEXT = "deciding_next"

    # Terminal states
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"
    STOPPED = "stopped"


# Valid transitions: from_state -> set of valid to_states
TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.INITIALIZING: {
        AgentState.READING_STATE, AgentState.FAILED,
    },
    AgentState.READING_STATE: {
        AgentState.CALLING_AGENT, AgentState.COMPLETED,
        AgentState.BUDGET_EXHAUSTED, AgentState.STOPPED,
    },
    AgentState.CALLING_AGENT: {
        AgentState.VALIDATING, AgentState.FAILED, AgentState.STOPPED,
    },
    AgentState.VALIDATING: {
        AgentState.APPLYING, AgentState.CALLING_AGENT, AgentState.READING_STATE,
        AgentState.FAILED,
    },
    AgentState.APPLYING: {
        AgentState.EXECUTING, AgentState.FAILED,
    },
    AgentState.EXECUTING: {
        AgentState.EXTRACTING, AgentState.READING_STATE,
        AgentState.FAILED, AgentState.STOPPED,
    },
    AgentState.EXTRACTING: {
        AgentState.EVALUATING, AgentState.READING_STATE, AgentState.FAILED,
    },
    AgentState.EVALUATING: {
        AgentState.DECIDING_NEXT, AgentState.COMPLETED, AgentState.FAILED,
    },
    AgentState.DECIDING_NEXT: {
        AgentState.READING_STATE, AgentState.BUDGET_EXHAUSTED,
        AgentState.FAILED, AgentState.STOPPED,
    },
}

# Terminal states — no transitions out
TERMINAL_STATES = {
    AgentState.COMPLETED,
    AgentState.BUDGET_EXHAUSTED,
    AgentState.FAILED,
    AgentState.STOPPED,
}

# Recovery actions per state (what to do when we crash in this state)
RECOVERY_ACTIONS = {
    AgentState.INITIALIZING: "reinitialize",
    AgentState.READING_STATE: "reread",
    AgentState.CALLING_AGENT: "retry_api",
    AgentState.VALIDATING: "revalidate",
    AgentState.APPLYING: "check_commit_and_resume",
    AgentState.EXECUTING: "adaptive_recovery",
    AgentState.EXTRACTING: "reread_output",
    AgentState.EVALUATING: "reevaluate",
    AgentState.DECIDING_NEXT: "redecide",
}


class StateMachine:
    """Journaled state machine for the agent loop.

    Every transition is logged to SQLite before the action executes.
    This allows crash recovery to resume from the exact point of failure.
    """

    def __init__(self, conn: sqlite3.Connection, run_id: str) -> None:
        self._conn = conn
        self._run_id = run_id
        self._state = AgentState.INITIALIZING
        self._iteration_num = 0

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def iteration_num(self) -> int:
        return self._iteration_num

    @property
    def is_terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    def transition(self, new_state: AgentState, data: dict[str, Any] | None = None) -> None:
        """Transition to a new state, journaling the change.

        Raises ValueError if the transition is invalid.
        """
        if self._state in TERMINAL_STATES:
            raise ValueError(f"Cannot transition from terminal state {self._state}")

        valid_targets = TRANSITIONS.get(self._state, set())
        if new_state not in valid_targets:
            raise ValueError(
                f"Invalid transition: {self._state} -> {new_state}. "
                f"Valid targets: {valid_targets}"
            )

        # Journal the transition BEFORE executing the side effect
        entry = JournalEntry(
            run_id=self._run_id,
            iteration_num=self._iteration_num,
            state=new_state.value,
            data=json.dumps(data) if data else None,
        )
        write_journal(self._conn, entry)

        log.info(
            "state_transition",
            from_state=self._state.value,
            to_state=new_state.value,
            iteration=self._iteration_num,
            run_id=self._run_id,
        )

        self._state = new_state

    def advance_iteration(self) -> int:
        """Move to the next iteration. Returns the new iteration number."""
        self._iteration_num += 1
        return self._iteration_num

    def set_iteration(self, num: int) -> None:
        """Set the iteration number (used during recovery)."""
        self._iteration_num = num

    @classmethod
    def recover(cls, conn: sqlite3.Connection, run_id: str) -> StateMachine | None:
        """Attempt to recover a state machine from the journal.

        Returns None if no journal entries exist for this run.
        """
        entry = get_latest_journal(conn, run_id)
        if entry is None:
            return None

        sm = cls(conn, run_id)
        sm._state = AgentState(entry.state)
        sm._iteration_num = entry.iteration_num

        recovery_action = RECOVERY_ACTIONS.get(sm._state)
        log.info(
            "state_machine_recovered",
            state=sm._state.value,
            iteration=sm._iteration_num,
            recovery_action=recovery_action,
            run_id=run_id,
        )

        return sm

    def get_recovery_action(self) -> str | None:
        """Get the recovery action for the current state."""
        return RECOVERY_ACTIONS.get(self._state)

    def get_journal_data(self) -> dict[str, Any] | None:
        """Get the data payload from the most recent journal entry."""
        entry = get_latest_journal(self._conn, self._run_id)
        if entry and entry.data:
            return json.loads(entry.data)
        return None
