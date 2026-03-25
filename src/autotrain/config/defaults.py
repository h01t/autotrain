"""Sensible default values referenced across the codebase."""

# Checkpoint detection patterns
CHECKPOINT_PATTERNS = [
    "**/*.pt",
    "**/*.pth",
    "**/*.ckpt",
    "**/checkpoint-*",
    "**/last.pt",
    "**/best.pt",
]

# Short experiment threshold (below this: stateless recovery)
CHECKPOINT_THRESHOLD_SECONDS = 900  # 15 minutes

# Agent loop limits
MAX_SANDBOX_RETRIES = 3  # Re-prompt attempts when validation fails
MAX_CONSECUTIVE_CRASHES = 5  # Hard stop after N consecutive crashes

# Journal / state filenames
AUTOTRAIN_DIR = ".autotrain"
STATE_DB_NAME = "state.db"
JOURNAL_FILE = "journal.jsonl"
LOG_FILE = "autotrain.log.jsonl"
PID_FILE = "daemon.pid"
