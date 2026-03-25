"""Git operations via subprocess for reliability."""

from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

log = structlog.get_logger()


class GitError(Exception):
    """Raised when a git command fails."""


def _run_git(repo_path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command in the given repo."""
    cmd = ["git", "-C", str(repo_path), *args]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        log.error("git_command_failed", cmd=cmd, stderr=e.stderr.strip())
        raise GitError(f"git {' '.join(args)} failed: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        raise GitError(f"git {' '.join(args)} timed out") from e


def is_git_repo(repo_path: Path) -> bool:
    """Check if path is inside a git repository."""
    result = _run_git(repo_path, "rev-parse", "--is-inside-work-tree", check=False)
    return result.returncode == 0


def init_repo(repo_path: Path) -> None:
    """Initialize a git repo if not already one."""
    if not is_git_repo(repo_path):
        _run_git(repo_path, "init")
        log.info("git_repo_initialized", path=str(repo_path))


def create_branch(repo_path: Path, branch_name: str) -> None:
    """Create and checkout a new branch."""
    _run_git(repo_path, "checkout", "-b", branch_name)
    log.info("git_branch_created", branch=branch_name)


def current_branch(repo_path: Path) -> str:
    """Get the current branch name."""
    result = _run_git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    return result.stdout.strip()


def commit(repo_path: Path, message: str, files: list[str] | None = None) -> str | None:
    """Stage files and commit. Returns the commit hash, or None if nothing to commit.

    If files is None, stages all changes.
    """
    if files:
        for f in files:
            if (repo_path / f).exists():
                _run_git(repo_path, "add", f)
    else:
        _run_git(repo_path, "add", "-A")

    # Check if there's anything staged to commit
    status = _run_git(repo_path, "diff", "--cached", "--quiet", check=False)
    if status.returncode == 0:
        log.warning("git_nothing_to_commit")
        return None

    _run_git(
        repo_path, "commit", "-m", message,
        "--author", "AutoTrain Agent <agent@autotrain.local>",
    )

    result = _run_git(repo_path, "rev-parse", "--short", "HEAD")
    commit_hash = result.stdout.strip()
    log.info("git_committed", hash=commit_hash, message=message[:80])
    return commit_hash


def revert_last_commit(repo_path: Path) -> None:
    """Revert the most recent commit (keeps it in history as a revert)."""
    _run_git(repo_path, "revert", "--no-edit", "HEAD")
    log.info("git_reverted_last_commit")


def reset_to_commit(repo_path: Path, commit_hash: str) -> None:
    """Hard reset to a specific commit. Use with caution."""
    _run_git(repo_path, "reset", "--hard", commit_hash)
    log.warning("git_hard_reset", target=commit_hash)


def get_head_hash(repo_path: Path) -> str:
    """Get the current HEAD commit hash (short)."""
    result = _run_git(repo_path, "rev-parse", "--short", "HEAD")
    return result.stdout.strip()


def has_uncommitted_changes(repo_path: Path) -> bool:
    """Check if there are uncommitted changes."""
    result = _run_git(repo_path, "status", "--porcelain")
    return bool(result.stdout.strip())


def get_file_content_at_commit(
    repo_path: Path, commit_hash: str, file_path: str,
) -> str | None:
    """Get file content at a specific commit. Returns None if file doesn't exist."""
    result = _run_git(
        repo_path, "show", f"{commit_hash}:{file_path}", check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def get_log(repo_path: Path, limit: int = 20) -> list[dict[str, str]]:
    """Get recent git log entries as dicts."""
    fmt = "%H%n%h%n%s%n%ai%n---"
    result = _run_git(repo_path, "log", f"--format={fmt}", f"-{limit}")

    entries = []
    lines = result.stdout.strip().split("\n---\n")
    for block in lines:
        block = block.strip()
        if not block:
            continue
        parts = block.split("\n", 3)
        if len(parts) >= 4:
            entries.append({
                "hash": parts[0],
                "short_hash": parts[1],
                "message": parts[2],
                "date": parts[3],
            })
    return entries


def get_diff(repo_path: Path, ref1: str, ref2: str) -> str:
    """Get the diff between two refs."""
    result = _run_git(repo_path, "diff", ref1, ref2)
    return result.stdout
