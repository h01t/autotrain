"""Tests for the state machine and journal."""

from __future__ import annotations

import pytest

from autotrain.core.state_machine import TERMINAL_STATES, AgentState, StateMachine
from autotrain.storage.queries import get_latest_journal


class TestAgentState:
    def test_terminal_states(self):
        assert AgentState.COMPLETED in TERMINAL_STATES
        assert AgentState.FAILED in TERMINAL_STATES
        assert AgentState.STOPPED in TERMINAL_STATES
        assert AgentState.BUDGET_EXHAUSTED in TERMINAL_STATES

    def test_non_terminal(self):
        assert AgentState.EXECUTING not in TERMINAL_STATES
        assert AgentState.READING_STATE not in TERMINAL_STATES


class TestStateMachine:
    def test_initial_state(self, db_conn):
        sm = StateMachine(db_conn, "run-1")
        assert sm.state == AgentState.INITIALIZING
        assert sm.iteration_num == 0
        assert not sm.is_terminal

    def test_valid_transition(self, db_conn_with_run):
        sm = StateMachine(db_conn_with_run, "run-1")
        sm.transition(AgentState.READING_STATE)
        assert sm.state == AgentState.READING_STATE

    def test_invalid_transition(self, db_conn):
        sm = StateMachine(db_conn, "run-1")
        with pytest.raises(ValueError, match="Invalid transition"):
            sm.transition(AgentState.EXECUTING)

    def test_terminal_state_blocks_transition(self, db_conn_with_run):
        sm = StateMachine(db_conn_with_run, "run-1")
        sm.transition(AgentState.FAILED)
        with pytest.raises(ValueError, match="Cannot transition from terminal"):
            sm.transition(AgentState.READING_STATE)

    def test_journal_written_on_transition(self, db_conn_with_run):
        sm = StateMachine(db_conn_with_run, "run-1")
        sm.transition(AgentState.READING_STATE, data={"info": "test"})

        entry = get_latest_journal(db_conn_with_run, "run-1")
        assert entry is not None
        assert entry.state == "reading_state"
        assert '"info"' in entry.data

    def test_full_happy_path(self, db_conn_with_run):
        """Walk through a complete successful iteration."""
        sm = StateMachine(db_conn_with_run, "run-1")
        sm.transition(AgentState.READING_STATE)
        sm.transition(AgentState.CALLING_AGENT)
        sm.transition(AgentState.VALIDATING)
        sm.transition(AgentState.APPLYING)
        sm.transition(AgentState.EXECUTING)
        sm.transition(AgentState.EXTRACTING)
        sm.transition(AgentState.EVALUATING)
        sm.transition(AgentState.DECIDING_NEXT)
        sm.transition(AgentState.READING_STATE)  # Loop back
        assert sm.state == AgentState.READING_STATE

    def test_advance_iteration(self, db_conn):
        sm = StateMachine(db_conn, "run-1")
        assert sm.iteration_num == 0
        n = sm.advance_iteration()
        assert n == 1
        assert sm.iteration_num == 1

    def test_target_hit_path(self, db_conn_with_run):
        sm = StateMachine(db_conn_with_run, "run-1")
        sm.transition(AgentState.READING_STATE)
        sm.transition(AgentState.CALLING_AGENT)
        sm.transition(AgentState.VALIDATING)
        sm.transition(AgentState.APPLYING)
        sm.transition(AgentState.EXECUTING)
        sm.transition(AgentState.EXTRACTING)
        sm.transition(AgentState.EVALUATING)
        sm.transition(AgentState.COMPLETED)
        assert sm.is_terminal

    def test_crash_recovery_from_executing(self, db_conn_with_run):
        """Simulate crash during EXECUTING and recovery."""
        sm = StateMachine(db_conn_with_run, "run-1")
        sm.transition(AgentState.READING_STATE)
        sm.transition(AgentState.CALLING_AGENT)
        sm.transition(AgentState.VALIDATING)
        sm.transition(AgentState.APPLYING)
        sm.transition(AgentState.EXECUTING, data={"train_pid": 12345})

        # "Crash" — create new state machine from journal
        recovered = StateMachine.recover(db_conn_with_run, "run-1")
        assert recovered is not None
        assert recovered.state == AgentState.EXECUTING
        assert recovered.get_recovery_action() == "adaptive_recovery"

        # Journal data preserved
        data = recovered.get_journal_data()
        assert data == {"train_pid": 12345}

    def test_recover_no_journal(self, db_conn):
        result = StateMachine.recover(db_conn, "nonexistent-run")
        assert result is None

    def test_validation_rejection_loops_back(self, db_conn_with_run):
        """Sandbox rejection should loop back to CALLING_AGENT."""
        sm = StateMachine(db_conn_with_run, "run-1")
        sm.transition(AgentState.READING_STATE)
        sm.transition(AgentState.CALLING_AGENT)
        sm.transition(AgentState.VALIDATING)
        # Validation fails → re-prompt agent
        sm.transition(AgentState.CALLING_AGENT)
        assert sm.state == AgentState.CALLING_AGENT

    def test_executing_crash_loops_to_reading(self, db_conn_with_run):
        """Training crash should be able to loop back to READING_STATE."""
        sm = StateMachine(db_conn_with_run, "run-1")
        sm.transition(AgentState.READING_STATE)
        sm.transition(AgentState.CALLING_AGENT)
        sm.transition(AgentState.VALIDATING)
        sm.transition(AgentState.APPLYING)
        sm.transition(AgentState.EXECUTING)
        # Training crashed — skip to next iteration
        sm.transition(AgentState.READING_STATE)
        assert sm.state == AgentState.READING_STATE
