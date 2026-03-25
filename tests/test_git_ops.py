"""Tests for git operations."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from autotrain.experiment.git_ops import (
    commit,
    create_branch,
    current_branch,
    get_head_hash,
    get_log,
    has_uncommitted_changes,
    init_repo,
    is_git_repo,
    revert_last_commit,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create an initialized git repo with an initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test.com"],
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Test"],
        capture_output=True, check=True,
    )
    # Initial commit
    (repo / "README.md").write_text("# Test\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "Initial commit"],
        capture_output=True, check=True,
    )
    return repo


class TestGitOps:
    def test_is_git_repo(self, git_repo):
        assert is_git_repo(git_repo)

    def test_not_git_repo(self, tmp_path):
        assert not is_git_repo(tmp_path)

    def test_init_repo(self, tmp_path):
        repo = tmp_path / "new-repo"
        repo.mkdir()
        init_repo(repo)
        assert is_git_repo(repo)

    def test_create_branch(self, git_repo):
        create_branch(git_repo, "autotrain/test")
        assert current_branch(git_repo) == "autotrain/test"

    def test_commit(self, git_repo):
        (git_repo / "train.py").write_text("print('hello')\n")
        hash_ = commit(git_repo, "Add train.py", files=["train.py"])
        assert len(hash_) >= 7
        assert not has_uncommitted_changes(git_repo)

    def test_commit_all(self, git_repo):
        (git_repo / "a.txt").write_text("a")
        (git_repo / "b.txt").write_text("b")
        commit(git_repo, "Add files")
        assert not has_uncommitted_changes(git_repo)

    def test_has_uncommitted_changes(self, git_repo):
        assert not has_uncommitted_changes(git_repo)
        (git_repo / "new.txt").write_text("content")
        assert has_uncommitted_changes(git_repo)

    def test_revert_last_commit(self, git_repo):
        (git_repo / "train.py").write_text("v1\n")
        commit(git_repo, "v1")
        (git_repo / "train.py").write_text("v2\n")
        commit(git_repo, "v2", files=["train.py"])

        revert_last_commit(git_repo)
        assert (git_repo / "train.py").read_text() == "v1\n"

    def test_get_head_hash(self, git_repo):
        hash_ = get_head_hash(git_repo)
        assert len(hash_) >= 7

    def test_get_log(self, git_repo):
        (git_repo / "a.txt").write_text("a")
        commit(git_repo, "Second commit")

        log_entries = get_log(git_repo, limit=5)
        assert len(log_entries) >= 2
        assert log_entries[0]["message"] == "Second commit"
