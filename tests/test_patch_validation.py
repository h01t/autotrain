"""Unit tests for patch validation — path safety, policy, preconditions."""

from __future__ import annotations

from pathlib import Path

import pytest

from autotrain.experiment.models import FileChange, PatchRequest
from autotrain.experiment.patch_validation import (
    BatchValidationError,
    PatchPreconditionError,
    UnsafePathError,
    ValidatedFileChange,
    assert_path_allowed,
    check_batch_limits,
    detect_duplicate_canonical_paths,
    normalize_repo_relative_path,
    validate_change_preconditions,
    validate_patch_set,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A clean workspace with a predictable structure."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "existing.txt").write_text("hello")
    (ws / "subdir").mkdir()
    (ws / "subdir" / "nested.py").write_text("x = 1")
    # Create a symlink inside the workspace pointing outside
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    symlink = ws / "escape_link"
    symlink.symlink_to(outside)
    return ws


# ── path normalization ──────────────────────────────────────────


class TestNormalizeRepoRelativePath:
    def test_simple_relative(self, workspace):
        result = normalize_repo_relative_path("train.py", workspace)
        assert result == "train.py"

    def test_nested_path(self, workspace):
        result = normalize_repo_relative_path("subdir/nested.py", workspace)
        assert result == "subdir/nested.py"

    def test_dot_prefix(self, workspace):
        result = normalize_repo_relative_path("./train.py", workspace)
        assert result == "train.py"

    def test_backslash_conversion(self, workspace):
        result = normalize_repo_relative_path("subdir\\nested.py", workspace)
        assert result == "subdir/nested.py"

    def test_traversal_rejected(self, workspace):
        with pytest.raises(UnsafePathError, match="outside workspace"):
            normalize_repo_relative_path("../secrets.env", workspace)

    def test_deep_traversal_rejected(self, workspace):
        with pytest.raises(UnsafePathError, match="outside workspace"):
            normalize_repo_relative_path("../../etc/passwd", workspace)

    def test_absolute_path_rejected(self, workspace):
        with pytest.raises(UnsafePathError, match="outside workspace"):
            normalize_repo_relative_path("/tmp/evil.py", workspace)

    def test_symlink_escape_rejected(self, workspace):
        with pytest.raises(UnsafePathError, match="symlink"):
            normalize_repo_relative_path("escape_link", workspace)

    def test_strips_whitespace(self, workspace):
        result = normalize_repo_relative_path("  train.py  ", workspace)
        assert result == "train.py"


# ── denylist / forbidden paths ───────────────────────────────────


class TestAssertPathAllowed:
    def test_normal_file_allowed(self):
        assert_path_allowed("train.py")

    def test_nested_path_allowed(self):
        assert_path_allowed("src/utils/train.py")

    def test_dot_git_rejected(self):
        with pytest.raises(UnsafePathError, match="forbidden"):
            assert_path_allowed(".git/config")

    def test_dot_git_top_level_rejected(self):
        with pytest.raises(UnsafePathError, match="forbidden"):
            assert_path_allowed(".git")

    def test_dot_autotrain_rejected(self):
        with pytest.raises(UnsafePathError, match="forbidden"):
            assert_path_allowed(".autotrain/state.db")

    def test_pycache_rejected(self):
        with pytest.raises(UnsafePathError, match="forbidden"):
            assert_path_allowed("__pycache__/module.pyc")

    def test_venv_rejected(self):
        with pytest.raises(UnsafePathError, match="forbidden"):
            assert_path_allowed(".venv/bin/python")

    def test_node_modules_rejected(self):
        with pytest.raises(UnsafePathError, match="forbidden"):
            assert_path_allowed("node_modules/package/index.js")

    def test_dot_env_rejected(self):
        with pytest.raises(UnsafePathError, match="forbidden"):
            assert_path_allowed(".env")


# ── duplicate canonical path detection ───────────────────────────


class TestDetectDuplicates:
    def test_no_duplicates(self):
        vc = [
            ValidatedFileChange("a.py", "update", "c", None, None),
            ValidatedFileChange("b.py", "update", "c", None, None),
        ]
        detect_duplicate_canonical_paths(vc)  # should not raise

    def test_duplicate_rejected(self):
        vc = [
            ValidatedFileChange("a.py", "update", "c", None, None),
            ValidatedFileChange("a.py", "update", "c2", None, None),
        ]
        with pytest.raises(BatchValidationError, match="Duplicate"):
            detect_duplicate_canonical_paths(vc)


# ── batch limits ────────────────────────────────────────────────


class TestBatchLimits:
    def test_within_limits(self):
        check_batch_limits(5, 50_000, max_files=10, max_total_bytes=100_000)

    def test_too_many_files(self):
        with pytest.raises(BatchValidationError, match="file count"):
            check_batch_limits(15, 50_000, max_files=10, max_total_bytes=100_000)

    def test_too_many_bytes(self):
        with pytest.raises(BatchValidationError, match="payload"):
            check_batch_limits(5, 200_000, max_files=10, max_total_bytes=100_000)


# ── preconditions ───────────────────────────────────────────────


class TestPreconditions:
    def test_create_on_missing_ok(self, tmp_path: Path):
        vc = ValidatedFileChange("new.py", "create", "print('hi')", None, None)
        validate_change_preconditions(vc, tmp_path)

    def test_create_on_existing_fails(self, tmp_path: Path):
        (tmp_path / "exists.py").write_text("x")
        vc = ValidatedFileChange("exists.py", "create", "y", None, None)
        with pytest.raises(PatchPreconditionError, match="already exists"):
            validate_change_preconditions(vc, tmp_path)

    def test_update_on_existing_ok(self, tmp_path: Path):
        (tmp_path / "f.py").write_text("old")
        vc = ValidatedFileChange("f.py", "update", "new", None, None)
        validate_change_preconditions(vc, tmp_path)

    def test_update_on_missing_fails(self, tmp_path: Path):
        vc = ValidatedFileChange("ghost.py", "update", "x", None, None)
        with pytest.raises(PatchPreconditionError, match="does not exist"):
            validate_change_preconditions(vc, tmp_path)

    def test_delete_on_existing_ok(self, tmp_path: Path):
        (tmp_path / "rm_me.py").write_text("bye")
        vc = ValidatedFileChange("rm_me.py", "delete", None, None, None)
        validate_change_preconditions(vc, tmp_path)

    def test_delete_on_missing_fails(self, tmp_path: Path):
        vc = ValidatedFileChange("ghost.py", "delete", None, None, None)
        with pytest.raises(PatchPreconditionError, match="does not exist"):
            validate_change_preconditions(vc, tmp_path)


# ── top-level validate_patch_set ─────────────────────────────────


class TestValidatePatchSet:
    def test_simple_success(self, tmp_path: Path):
        changes = [
            FileChange(path="train.py", operation="update", content="x=1"),
        ]
        result = validate_patch_set(changes, tmp_path)
        assert result.total_file_count == 1
        assert result.canonical_paths == {"train.py"}
        assert result.total_payload_bytes == len(b"x=1")

    def test_duplicate_in_batch_rejected(self, tmp_path: Path):
        changes = [
            FileChange(path="a.py", operation="update", content="x"),
            FileChange(path="a.py", operation="update", content="y"),
        ]
        with pytest.raises(BatchValidationError, match="Duplicate"):
            validate_patch_set(changes, tmp_path)

    def test_traversal_rejected(self, tmp_path: Path):
        changes = [
            FileChange(path="../secrets", operation="create", content="pwned"),
        ]
        with pytest.raises(UnsafePathError):
            validate_patch_set(changes, tmp_path)

    def test_forbidden_path_rejected(self, tmp_path: Path):
        changes = [
            FileChange(path=".git/config", operation="update", content="x"),
        ]
        with pytest.raises(UnsafePathError):
            validate_patch_set(changes, tmp_path)

    def test_batch_too_large(self, tmp_path: Path):
        changes = [
            FileChange(path=f"f{i}.py", operation="create", content="x")
            for i in range(22)
        ]
        with pytest.raises(BatchValidationError):
            validate_patch_set(changes, tmp_path, max_files=5)


# ── Pydantic FileChange schema edge cases ────────────────────────


class TestFileChangeSchema:
    def test_create_requires_content(self):
        with pytest.raises(Exception):
            FileChange(path="new.py", operation="create")

    def test_create_rejects_patch(self):
        with pytest.raises(Exception):
            FileChange(path="new.py", operation="create", content="x", patch="y")

    def test_update_requires_content_or_patch(self):
        with pytest.raises(Exception):
            FileChange(path="x.py", operation="update")

    def test_update_accepts_content(self):
        fc = FileChange(path="x.py", operation="update", content="y")
        assert fc.content == "y"

    def test_update_accepts_patch(self):
        fc = FileChange(path="x.py", operation="update", patch="@@ -1 +1 @@")
        assert fc.patch is not None

    def test_update_rejects_both(self):
        with pytest.raises(Exception):
            FileChange(path="x.py", operation="update", content="x", patch="y")

    def test_delete_rejects_content(self):
        with pytest.raises(Exception):
            FileChange(path="x.py", operation="delete", content="boom")

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            FileChange(path="x.py", operation="update", content="y", evil="muahaha")

    def test_empty_path_rejected(self):
        with pytest.raises(Exception):
            FileChange(path="", operation="update", content="x")

    def test_whitespace_only_path_rejected(self):
        with pytest.raises(Exception):
            FileChange(path="   ", operation="update", content="x")


# ── PatchRequest batch model ─────────────────────────────────────


class TestPatchRequestModel:
    def test_minimal_valid(self):
        pr = PatchRequest(changes=[
            FileChange(path="t.py", operation="update", content="x"),
        ])
        assert len(pr.changes) == 1

    def test_empty_changes_rejected(self):
        with pytest.raises(Exception):
            PatchRequest(changes=[])

    def test_duplicate_paths_rejected(self):
        with pytest.raises(Exception):
            PatchRequest(changes=[
                FileChange(path="a.py", operation="update", content="x"),
                FileChange(path="a.py", operation="update", content="y"),
            ])
