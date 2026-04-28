"""Centralized patch-validation logic — path safety, policy, preconditions.

Every function in this module is designed to be independently unit-testable.
LLM-generated edits are treated as untrusted at every stage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import structlog

from autotrain.experiment.models import FileChange

log = structlog.get_logger()

# ── typed errors ──────────────────────────────────────────────────


class AutoTrainEditError(Exception):
    """Base class for all edit pipeline errors."""


class PatchValidationError(AutoTrainEditError):
    """Schema-level or structural validation failure."""


class UnsafePathError(PatchValidationError):
    """Path traversal, absolute, or symlink-escape attempt."""


class PatchPreconditionError(AutoTrainEditError):
    """Precondition check failed (e.g. create on existing file)."""


class BatchValidationError(AutoTrainEditError):
    """Batch-level error (duplicates, count, size, etc.)."""


class AtomicApplyError(AutoTrainEditError):
    """Failure during the actual apply phase."""


class PostApplyVerificationError(AutoTrainEditError):
    """Post-apply state did not match expected state."""


class GitCommitError(AutoTrainEditError):
    """Git staging or commit failed."""


# ── internal result types ─────────────────────────────────────────


@dataclass(frozen=True)
class ValidatedFileChange:
    """A single file change that has passed all validation gates."""

    canonical_path: str
    operation: Literal["create", "update", "delete"]
    content: str | None
    patch: str | None
    description: str | None

    @property
    def is_write(self) -> bool:
        return self.operation in ("create", "update")


@dataclass(frozen=True)
class ValidatedPatchSet:
    """A fully validated batch of changes ready for atomic apply."""

    changes: list[ValidatedFileChange]
    canonical_paths: set[str] = field(compare=False)
    total_file_count: int
    total_payload_bytes: int


# ── denylist ──────────────────────────────────────────────────────

_FORBIDDEN_PATH_PREFIXES: tuple[str, ...] = (
    ".git",
    ".autotrain",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".env",
    ".secrets",
)

_FORBIDDEN_PATH_COMPONENTS: tuple[str, ...] = (
    ".git",
)


# ── path normalisation & safety ───────────────────────────────────


def normalize_repo_relative_path(
    raw: str,
    workspace_root: Path,
) -> str:
    """Canonicalize a repo-relative path and reject unsafe values.

    Returns a normalized ``str`` (using ``/`` separators) relative to
    the workspace root.

    Raises:
        UnsafePathError: Path traversal, absolute external path, or
            symlink escape.
    """
    raw = raw.strip().replace("\\", "/")

    # Build the candidate absolute path
    candidate = (workspace_root / raw).resolve()

    # Reject if the resolved path escapes the workspace
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        raise UnsafePathError(
            f"Path {raw!r} resolves outside workspace: {candidate}"
        )

    # Resolve symlinks and re-check
    real = candidate
    try:
        real = candidate.resolve(strict=False)
    except (OSError, RuntimeError):
        raise UnsafePathError(
            f"Cannot resolve path {raw!r} (possible broken symlink)"
        )

    try:
        real.relative_to(workspace_root.resolve())
    except ValueError:
        raise UnsafePathError(
            f"Path {raw!r} resolves via symlink outside workspace: {real}"
        )

    # Build canonical relative form
    rel = str(real.relative_to(workspace_root.resolve()))
    return rel.replace(os.sep, "/")


def assert_path_allowed(canonical_path: str) -> None:
    """Raise ``UnsafePathError`` if *canonical_path* targets a
    forbidden directory or file."""
    parts = canonical_path.split("/")

    # Top-level forbidden prefixes
    for prefix in _FORBIDDEN_PATH_PREFIXES:
        if canonical_path == prefix or canonical_path.startswith(prefix + "/"):
            raise UnsafePathError(
                f"Path {canonical_path!r} is in a forbidden area: {prefix!r}"
            )

    # Forbidden path components anywhere
    for part in parts:
        if part in _FORBIDDEN_PATH_COMPONENTS:
            raise UnsafePathError(
                f"Path {canonical_path!r} contains forbidden component: {part!r}"
            )


# ── batch-level validation ────────────────────────────────────────


def detect_duplicate_canonical_paths(
    changes: list[ValidatedFileChange],
) -> None:
    """Raise ``BatchValidationError`` if any two changes target the
    same canonical path."""
    seen: dict[str, int] = {}
    for idx, vc in enumerate(changes):
        p = vc.canonical_path
        if p in seen:
            raise BatchValidationError(
                f"Duplicate canonical path {p!r} at positions "
                f"{seen[p]} and {idx}"
            )
        seen[p] = idx


def check_batch_limits(
    count: int,
    total_bytes: int,
    max_files: int,
    max_total_bytes: int,
) -> None:
    """Raise ``BatchValidationError`` if batch exceeds safety limits."""
    if count > max_files:
        raise BatchValidationError(
            f"Batch file count {count} exceeds limit {max_files}"
        )
    if total_bytes > max_total_bytes:
        raise BatchValidationError(
            f"Batch payload {total_bytes} bytes exceeds limit "
            f"{max_total_bytes}"
        )


# ── precondition checks ───────────────────────────────────────────


def validate_change_preconditions(
    vc: ValidatedFileChange,
    workspace_root: Path,
) -> None:
    """Check filesystem preconditions for a single validated change.

    Operates against *workspace_root* (which may be an isolated
    worktree, not the user's active branch).

    Raises:
        PatchPreconditionError: The precondition is not met.
    """
    target = workspace_root / vc.canonical_path
    exists = target.exists()

    if vc.operation == "create":
        if exists:
            raise PatchPreconditionError(
                f"create: {vc.canonical_path!r} already exists"
            )
    elif vc.operation == "update":
        if not exists:
            raise PatchPreconditionError(
                f"update: {vc.canonical_path!r} does not exist"
            )
    elif vc.operation == "delete":
        if not exists:
            raise PatchPreconditionError(
                f"delete: {vc.canonical_path!r} does not exist"
            )


# ── top-level validation entry point ──────────────────────────────


def validate_patch_set(
    changes: list[FileChange],
    workspace_root: Path,
    *,
    max_files: int = 20,
    max_total_bytes: int = 1_000_000,
) -> ValidatedPatchSet:
    """Validate an entire batch of *FileChange* objects.

    This is the single entry point that the sandbox / agent loop
    should call before any filesystem mutation.

    Returns a ``ValidatedPatchSet`` that is safe to pass to the
    atomic apply pipeline.
    """
    validated: list[ValidatedFileChange] = []
    total_bytes = 0

    for fc in changes:
        # 1. Normalize path
        canon = normalize_repo_relative_path(fc.path, workspace_root)

        # 2. Denylist check
        assert_path_allowed(canon)

        # 3. Build validated change (content/patch consistency
        #    already checked by Pydantic model validator)
        content = fc.content
        patch = fc.patch

        vc = ValidatedFileChange(
            canonical_path=canon,
            operation=fc.operation,
            content=content,
            patch=patch,
            description=fc.description,
        )
        validated.append(vc)

        if content:
            total_bytes += len(content.encode("utf-8"))
        if patch:
            total_bytes += len(patch.encode("utf-8"))

    # 4. Duplicate detection
    detect_duplicate_canonical_paths(validated)

    # 5. Batch limits
    check_batch_limits(
        len(validated), total_bytes, max_files, max_total_bytes,
    )

    # 6. Collect canonical paths
    paths = {vc.canonical_path for vc in validated}

    return ValidatedPatchSet(
        changes=validated,
        canonical_paths=paths,
        total_file_count=len(validated),
        total_payload_bytes=total_bytes,
    )
