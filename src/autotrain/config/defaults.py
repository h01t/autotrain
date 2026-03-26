"""Sensible default values referenced across the codebase."""

# Agent loop limits
MAX_SANDBOX_RETRIES = 3  # Re-prompt attempts when validation fails
MAX_CONSECUTIVE_CRASHES = 5  # Hard stop after N consecutive crashes

# State filenames
AUTOTRAIN_DIR = ".autotrain"
STATE_DB_NAME = "state.db"
PID_FILE = "daemon.pid"
