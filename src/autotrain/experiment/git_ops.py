"""Git operations via subprocess for reliability.

Includes worktree isolation, precise staging, and commit-scope
verification for the atomic multi-file safety pipeline.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import structlog

from autotrain.experiment.patch_validation import AutoTrainEditError

log = structlog.get_logger()


class GitError(AutoTrainEditError):
    """Raised when a git command fails."""


# ── low-level helpers ────────────────────────────────────────────


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


# ── repo queries ──────────────────────────────────────────────────


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


def has_uncommitted_changes(repo_path: Path) -> bool:
    """Check if there are uncommitted changes."""
    result = _run_git(repo_path, "status", "--porcelain")
    return bool(result.stdout.strip())


def get_current_commit(repo_path: Path) -> str:
    """Return the short hash of HEAD."""
    result = _run_git(repo_path, "rev-parse", "--short", "HEAD")
    return result.stdout.strip()


# ── worktree isolation ────────────────────────────────────────────


def create_worktree(
    repo_path: Path, suffix: str = "autotrain",
) -> tuple[Path, str]:
    """Create a temporary git worktree on a new throwaway branch.

    The worktree is checked out from HEAD onto a uniquely-named
    temporary branch so that commits inside the worktree are
    reachable by the source repository for merging.

    Args:
        repo_path: Path to the main (bare or working) repo.
        suffix: Human-readable tag for the worktree directory name.

    Returns:
        Tuple of ``(worktree_path, branch_name)``.  The caller MUST
        call ``remove_worktree`` and delete the branch on cleanup.

    Raises:
        GitError: If the worktree cannot be created.
    """
    result = _run_git(repo_path, "rev-parse", "HEAD", check=False)
    if result.returncode != 0:
        _run_git(
            repo_path, "commit",
            "--allow-empty",
            "-m", "autotrain: initial empty commit for worktree",
            "--author", "AutoTrain Agent <agent@autotrain.local>",
        )
        log.info("git_empty_commit_created")

    branch_name = f"autotrain-wt-{uuid.uuid4().hex[:12]}"
    parent = Path(tempfile.mkdtemp(prefix="autotrain-wt-"))
    wt_path = parent / suffix

    try:
        _run_git(repo_path, "worktree", "add", "-b", branch_name, str(wt_path), "HEAD")
        log.info("git_worktree_created", path=str(wt_path), branch=branch_name)
        return wt_path, branch_name
    except GitError:
        shutil.rmtree(parent, ignore_errors=True)
        raise

def remove_worktree(repo_path: Path, wt_path: Path) -> None:
    """Remove a git worktree and its parent temp directory.

    Best-effort — logs but never raises on cleanup failure.
    """
    try:
        _run_git(repo_path, "worktree", "remove", "--force", str(wt_path), check=False)
    except GitError:
        log.warning("git_worktree_remove_git_failed", path=str(wt_path))

    # The worktree parent is a temp dir we own
    parent = wt_path.parent
    if parent.exists():
        shutil.rmtree(parent, ignore_errors=True)
        log.info("git_worktree_cleaned", path=str(parent))


def merge_worktree_branch(repo_path: Path, branch_name: str) -> str:
    """Merge a worktree's branch back into the current branch.

    Returns the merge commit short hash.
    Raises GitError if merge fails.
    """
    try:
        _run_git(repo_path, "merge", "--no-ff", branch_name, "-m",
                 "autotrain: merge atomic edit batch")
    except GitError:
        _run_git(repo_path, "merge", "--abort", check=False)
        _run_git(repo_path, "branch", "-D", branch_name, check=False)
        raise

    sha = get_current_commit(repo_path)
    log.info("git_worktree_merged", branch=branch_name, commit=sha)
    return sha


def delete_branch(repo_path: Path, branch_name: str) -> None:
    """Best-effort branch deletion — never raises."""
    _run_git(repo_path, "branch", "-D", branch_name, check=False)
    log.info("git_branch_deleted", branch=branch_name)


# ── precise staging & committing ──────────────────────────────────


def stage_exact_files(repo_path: Path, files: list[str]) -> None:
    """Stage exactly the listed files (repo-relative paths).

    This intentionally does **not** use ``git add -A`` — only the
    exact *files* are staged so that no incidental modifications
    can be swept into the commit.

    Handles both additions/modifications (``git add``) and
    deletions (``git rm``).
    """
    for f in files:
        target = repo_path / f
        if target.exists():
            _run_git(repo_path, "add", "--", f)
        else:
            # File was deleted — stage the removal
            _run_git(repo_path, "rm", "--", f, check=False)
            log.info("stage_removed_file", path=f)


def staged_files(repo_path: Path) -> set[str]:
    """Return the set of repo-relative paths currently staged.

    An empty set means nothing is staged.
    """
    result = _run_git(
        repo_path, "diff", "--cached", "--name-only", check=False,
    )
    if result.returncode != 0:
        return set()
    lines = result.stdout.strip().split("\n")
    return {line.strip() for line in lines if line.strip()}


def verify_staged_matches(
    repo_path: Path,
    expected: set[str],
) -> None:
    """Raise ``GitError`` if the currently staged files do not match
    *expected* exactly."""
    actual = staged_files(repo_path)
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise GitError(
            f"Staged files mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )


def commit_staged(repo_path: Path, message: str) -> str:
    """Commit currently staged changes and return the short hash.

    Raises GitError if nothing is staged or the commit fails.
    """
    # Guard: nothing to commit
    status = _run_git(repo_path, "diff", "--cached", "--quiet", check=False)
    if status.returncode == 0:
        raise GitError("Cannot commit: nothing staged")

    _run_git(
        repo_path, "commit", "-m", message,
        "--author", "AutoTrain Agent <agent@autotrain.local>",
    )

    result = _run_git(repo_path, "rev-parse", "--short", "HEAD")
    commit_hash = result.stdout.strip()
    log.info("git_committed", hash=commit_hash, message=message[:80])
    return commit_hash


# ── legacy helpers (kept for backward compat) ─────────────────────


def commit(repo_path: Path, message: str, files: list[str] | None = None) -> str | None:
    """Stage files and commit. Returns the commit hash, or None if nothing to commit.

    If files is None, stages all changes.
    """
    if files:
        stage_exact_files(repo_path, files)
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
