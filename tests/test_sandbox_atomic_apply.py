"""Integration tests for atomic multi-file apply in sandbox.

These tests exercise the full pipeline: validate → isolate → apply →
verify → commit, with both success and rollback scenarios.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from autotrain.experiment.models import FileChange
from autotrain.experiment.sandbox import apply_patch_set_atomically


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """An initialized git repo with one committed file."""
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
    (repo / "train.py").write_text("lr = 0.001\nepochs = 10\n")
    (repo / "config.yaml").write_text("batch_size: 32\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "initial"],
        capture_output=True, check=True,
    )
    return repo


# ── success cases ────────────────────────────────────────────────


class TestAtomicApplySuccess:
    def test_single_file_create(self, git_repo: Path):
        changes = [
            FileChange(path="new_file.py", operation="create", content="# new file\n"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert result.success
        assert result.commit_sha is not None
        assert "new_file.py" in result.applied_paths
        # Verify file was created
        assert (git_repo / "new_file.py").exists()
        assert (git_repo / "new_file.py").read_text() == "# new file\n"

    def test_single_file_update(self, git_repo: Path):
        changes = [
            FileChange(path="train.py", operation="update", content="lr = 0.01\n"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert result.success
        assert (git_repo / "train.py").read_text() == "lr = 0.01\n"

    def test_single_file_delete(self, git_repo: Path):
        changes = [
            FileChange(path="config.yaml", operation="delete"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert result.success
        assert not (git_repo / "config.yaml").exists()

    def test_multi_file_create_and_update(self, git_repo: Path):
        changes = [
            FileChange(path="train.py", operation="update", content="lr = 0.01\n"),
            FileChange(path="utils.py", operation="create", content="def foo(): pass\n"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert result.success
        assert result.validated_count == 2
        assert len(result.applied_paths) == 2
        assert (git_repo / "train.py").read_text() == "lr = 0.01\n"
        assert (git_repo / "utils.py").exists()

    def test_multi_file_with_delete(self, git_repo: Path):
        changes = [
            FileChange(path="train.py", operation="update", content="x = 1\n"),
            FileChange(path="config.yaml", operation="delete"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert result.success
        assert (git_repo / "train.py").read_text() == "x = 1\n"
        assert not (git_repo / "config.yaml").exists()

    def test_nested_create(self, git_repo: Path):
        changes = [
            FileChange(path="src/models/__init__.py", operation="create", content=""),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert result.success
        assert (git_repo / "src" / "models" / "__init__.py").exists()


# ── failure / rollback cases ─────────────────────────────────────


class TestAtomicApplyRollback:
    def test_create_on_existing_fails(self, git_repo: Path):
        changes = [
            FileChange(path="train.py", operation="create", content="boom"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert not result.success
        assert "already exists" in (result.error or "")
        # The original file must be untouched
        assert (git_repo / "train.py").read_text() == "lr = 0.001\nepochs = 10\n"

    def test_delete_missing_fails(self, git_repo: Path):
        changes = [
            FileChange(path="nonexistent.py", operation="delete"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert not result.success
        assert "does not exist" in (result.error or "")

    def test_traversal_rejected_no_mutation(self, git_repo: Path):
        changes = [
            FileChange(path="../outside.txt", operation="create", content="evil"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert not result.success
        assert result.error_type == "UnsafePathError"

    def test_forbidden_path_rejected(self, git_repo: Path):
        changes = [
            FileChange(path=".git/config", operation="update", content="pwned"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert not result.success
        assert result.error_type == "UnsafePathError"

    def test_duplicate_paths_rejected(self, git_repo: Path):
        changes = [
            FileChange(path="train.py", operation="update", content="a"),
            FileChange(path="train.py", operation="update", content="b"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert not result.success
        assert "Duplicate" in (result.error or "")

    def test_partial_batch_rolled_back(self, git_repo: Path):
        """If one change in a batch fails precondition, nothing is committed."""
        original_train = (git_repo / "train.py").read_text()
        changes = [
            FileChange(path="train.py", operation="update", content="new content\n"),
            FileChange(path="ghost.py", operation="delete"),  # fails — missing
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert not result.success
        # train.py must be untouched — rollback succeeded
        assert (git_repo / "train.py").read_text() == original_train

    def test_batch_too_large_rejected(self, git_repo: Path):
        changes = [
            FileChange(path=f"f{i}.py", operation="create", content="x")
            for i in range(25)
        ]
        result = apply_patch_set_atomically(
            changes, git_repo, max_files=5, max_total_bytes=100_000,
        )
        assert not result.success
        # Nothing should have been created
        for i in range(25):
            assert not (git_repo / f"f{i}.py").exists()

    def test_empty_batch_rejected(self, git_repo: Path):
        """Empty batch should fail validation."""
        result = apply_patch_set_atomically([], git_repo)
        assert not result.success
        assert result.error is not None


# ── structured error metadata ────────────────────────────────────


class TestAtomicApplyErrorMetadata:
    def test_error_includes_type(self, git_repo: Path):
        changes = [
            FileChange(path="../escape", operation="create", content="x"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert not result.success
        assert result.error_type == "UnsafePathError"
        assert result.error is not None
        assert result.validated_count == 0

    def test_success_has_no_error(self, git_repo: Path):
        changes = [
            FileChange(path="ok.py", operation="create", content="x"),
        ]
        result = apply_patch_set_atomically(changes, git_repo)
        assert result.success
        assert result.error is None
        assert result.error_type is None
        assert result.commit_sha is not None
