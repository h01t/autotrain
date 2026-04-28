"""Code sandboxing — validate agent-generated changes before applying.

Provides both the legacy per-file validation path and the new
atomic multi-file safety pipeline (worktree isolation, batch
validate, apply, verify, commit).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from autotrain.config.schema import SandboxConfig
from autotrain.experiment.git_ops import (
    GitError,
    commit_staged,
    create_worktree,
    delete_branch,
    merge_worktree_branch,
    remove_worktree,
    stage_exact_files,
    verify_staged_matches,
)
from autotrain.experiment.models import FileChange as PydanticFileChange
from autotrain.experiment.patch_validation import (
    AutoTrainEditError,
    PatchPreconditionError,
    PostApplyVerificationError,
    ValidatedFileChange,
    ValidatedPatchSet,
    validate_change_preconditions,
    validate_patch_set,
)

log = structlog.get_logger()


# ── legacy dataclasses (kept for backward compat) ─────────────────


@dataclass
class FileChange:
    """A proposed change to a file.

    .. deprecated::
        Prefer ``autotrain.experiment.models.FileChange`` (Pydantic)
        for new code.  This dataclass remains for backward
        compatibility with existing callers.
    """

    file: str
    action: str  # "replace", "create", "full_rewrite"
    search: str | None = None  # For "replace": text to find
    replace: str | None = None  # For "replace": replacement text
    content: str | None = None  # For "create"/"full_rewrite": full content


@dataclass
class ValidationResult:
    """Result of sandbox validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


# ── new atomic apply result ──────────────────────────────────────


@dataclass
class AtomicApplyResult:
    """Result of an atomic multi-file apply operation.

    *error_type* is always a class name from the
    :class:`AutoTrainEditError` hierarchy (or ``None`` on success).
    """

    success: bool
    commit_sha: str | None = None
    applied_paths: list[str] = field(default_factory=list)
    validated_count: int = 0
    error: str | None = None
    error_type: str | None = None


# ── atomic pipeline ──────────────────────────────────────────────


def apply_patch_set_atomically(
    pydantic_changes: list[PydanticFileChange],
    repo_path: Path,
    *,
    max_files: int = 20,
    max_total_bytes: int = 1_000_000,
    commit_message: str = "autotrain: atomic multi-file edit",
) -> AtomicApplyResult:
    """Validate and apply a batch of file changes atomically.

    The pipeline:

    1. Validate the entire batch (paths, policy, preconditions).
    2. Create an isolated git worktree on a temporary branch.
    3. Validate preconditions inside the worktree.
    4. Apply every change inside the worktree.
    5. Post-apply verification.
    6. Stage & verify only the approved files.
    7. Commit in the worktree.
    8. Merge the worktree branch back into the source repo.
    9. Clean up (worktree + temp branch).

    If **any** step fails before the merge (step 8), the worktree
    and temp branch are discarded — the source repository is never
    touched.

    Returns:
        ``AtomicApplyResult`` with success flag, commit SHA, and/or
        structured error information.
    """
    # -- Step 1: batch validation (no filesystem access yet) --------
    try:
        validated = validate_patch_set(
            pydantic_changes,
            repo_path,
            max_files=max_files,
            max_total_bytes=max_total_bytes,
        )
    except AutoTrainEditError as exc:
        log.warning("atomic_validation_failed", error=str(exc))
        return AtomicApplyResult(
            success=False,
            validated_count=0,
            error=str(exc),
            error_type=type(exc).__name__,
        )

    # -- Step 2: create isolated worktree ---------------------------
    worktree: Path | None = None
    branch_name: str | None = None
    try:
        worktree, branch_name = create_worktree(repo_path)
    except GitError as exc:
        log.error("atomic_worktree_create_failed", error=str(exc))
        return AtomicApplyResult(
            success=False,
            validated_count=len(validated.changes),
            error=str(exc),
            error_type=type(exc).__name__,
        )

    try:
        # -- Step 3: precondition checks (in isolation) ------------
        for vc in validated.changes:
            try:
                validate_change_preconditions(vc, worktree)
            except PatchPreconditionError as exc:
                return AtomicApplyResult(
                    success=False,
                    validated_count=len(validated.changes),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

        # -- Step 4: apply all changes in worktree -----------------
        apply_errors = _apply_validated_changes(validated.changes, worktree)
        if apply_errors:
            return AtomicApplyResult(
                success=False,
                validated_count=len(validated.changes),
                error="; ".join(apply_errors),
                error_type="AtomicApplyError",
            )

        # -- Step 5: post-apply verification -----------------------
        try:
            _verify_applied_patch_set(validated, worktree)
        except PostApplyVerificationError as exc:
            return AtomicApplyResult(
                success=False,
                validated_count=len(validated.changes),
                error=str(exc),
                error_type=type(exc).__name__,
            )

        # -- Step 6: stage only approved files ---------------------
        approved_paths = sorted(validated.canonical_paths)
        stage_exact_files(worktree, approved_paths)

        # -- Step 7: verify staged set -----------------------------
        try:
            verify_staged_matches(worktree, validated.canonical_paths)
        except GitError as exc:
            return AtomicApplyResult(
                success=False,
                validated_count=len(validated.changes),
                error=str(exc),
                error_type=type(exc).__name__,
            )

        # -- Step 8: commit in worktree ----------------------------
        try:
            commit_staged(worktree, commit_message)
        except GitError as exc:
            return AtomicApplyResult(
                success=False,
                validated_count=len(validated.changes),
                error=str(exc),
                error_type=type(exc).__name__,
            )

        # -- Step 9: merge worktree branch back into source repo ---
        try:
            merge_sha = merge_worktree_branch(repo_path, branch_name)
        except GitError as exc:
            # Merge failed — source repo is untouched (merge aborted
            # inside merge_worktree_branch), but we still need to
            # clean up the worktree side.
            return AtomicApplyResult(
                success=False,
                validated_count=len(validated.changes),
                error=str(exc),
                error_type=type(exc).__name__,
            )

        # -- Success -----------------------------------------------
        log.info(
            "atomic_apply_success",
            commit=merge_sha,
            files=approved_paths,
        )
        return AtomicApplyResult(
            success=True,
            commit_sha=merge_sha,
            applied_paths=approved_paths,
            validated_count=len(validated.changes),
        )

    except Exception as exc:
        # Catch-all — source repo is never touched because we never
        # reached the merge step (or it was aborted).
        log.error("atomic_apply_unexpected_error", error=str(exc))
        return AtomicApplyResult(
            success=False,
            validated_count=len(validated.changes),
            error=str(exc),
            error_type=type(exc).__name__,
        )

    finally:
        # Always clean up: remove temp branch + worktree.
        # The source repo is untouched unless merge succeeded.
        if branch_name is not None:
            delete_branch(repo_path, branch_name)
        if worktree is not None:
            try:
                remove_worktree(repo_path, worktree)
            except Exception as exc:
                log.warning("atomic_worktree_cleanup_failed", error=str(exc))

# ── internal apply helpers ────────────────────────────────────────


def _apply_validated_changes(
    changes: list[ValidatedFileChange],
    worktree: Path,
) -> list[str]:
    """Apply every validated change inside *worktree*.

    Returns a (possibly empty) list of error strings.
    """
    errors: list[str] = []
    for vc in changes:
        target = worktree / vc.canonical_path

        try:
            if vc.operation == "delete":
                target.unlink(missing_ok=False)
            elif vc.operation == "create":
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(vc.content or "")
            elif vc.operation == "update":
                target.write_text(vc.content or "")
        except Exception as exc:
            errors.append(
                f"{vc.operation} {vc.canonical_path!r}: {exc}"
            )
    return errors


def _verify_applied_patch_set(
    validated: ValidatedPatchSet,
    worktree: Path,
) -> None:
    """Post-apply: assert filesystem matches the validated intent.

    Raises ``PostApplyVerificationError`` on any mismatch.
    """
    for vc in validated.changes:
        target = worktree / vc.canonical_path

        if vc.operation == "delete":
            if target.exists():
                raise PostApplyVerificationError(
                    f"Post-apply: {vc.canonical_path!r} should be "
                    f"deleted but still exists"
                )
        elif vc.is_write:
            if not target.exists():
                raise PostApplyVerificationError(
                    f"Post-apply: {vc.canonical_path!r} should exist "
                    f"but is missing"
                )
            if vc.content is not None:
                actual = target.read_text()
                if actual != vc.content:
                    raise PostApplyVerificationError(
                        f"Post-apply: {vc.canonical_path!r} content "
                        f"mismatch (expected {len(vc.content)} bytes, "
                        f"got {len(actual)} bytes)"
                    )

    # Extra-file scan deferred — not needed for this milestone


# ── legacy validation (unchanged for backward compat) ─────────────


def validate_changes(
    changes: list[FileChange],
    config: SandboxConfig,
    current_files: dict[str, str] | None = None,
) -> ValidationResult:
    """Validate proposed changes against sandbox rules.

    Args:
        changes: List of proposed file changes.
        config: Sandbox configuration.
        current_files: Dict of filename -> current content (for diff validation).

    Returns:
        ValidationResult with errors (blocking) and warnings (informational).
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check total change count
    if len(changes) > config.max_changes_per_iteration:
        errors.append(
            f"Too many changes: {len(changes)} > {config.max_changes_per_iteration}"
        )

    for change in changes:
        # Layer 1: File whitelist
        if change.file not in config.writable_files:
            errors.append(
                f"File '{change.file}' not in whitelist: {config.writable_files}"
            )
            continue

        # Determine the new content
        new_content = _resolve_content(change, current_files)
        if new_content is None:
            if change.action == "replace" and current_files:
                errors.append(
                    f"File '{change.file}': search text not found in current file"
                )
            continue

        # Layer 2: File size
        content_size = len(new_content.encode("utf-8"))
        if content_size > config.max_file_size_bytes:
            errors.append(
                f"File '{change.file}' would be {content_size} bytes "
                f"(max: {config.max_file_size_bytes})"
            )

        # Layer 3: Dangerous pattern scan on new content
        for pattern in config.forbidden_patterns:
            matches = re.findall(pattern, new_content)
            if matches:
                errors.append(
                    f"File '{change.file}': forbidden pattern '{pattern}' "
                    f"found: {matches[:3]}"
                )

        # Layer 4: Diff validation (only check added lines)
        if current_files and change.file in current_files:
            diff_errors = _validate_diff(
                current_files[change.file], new_content, config,
            )
            errors.extend(
                f"File '{change.file}': {e}" for e in diff_errors
            )

    is_valid = len(errors) == 0
    if not is_valid:
        log.warning("sandbox_validation_failed", errors=errors)
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def _resolve_content(
    change: FileChange,
    current_files: dict[str, str] | None,
) -> str | None:
    """Resolve the final file content from a change."""
    if change.action in ("create", "full_rewrite"):
        return change.content or ""

    if change.action == "replace":
        if not current_files or change.file not in current_files:
            return change.replace or ""
        current = current_files[change.file]
        if change.search and change.search in current:
            return current.replace(change.search, change.replace or "", 1)
        # Fallback: whitespace-normalized matching
        if change.search:
            result = _try_whitespace_match(current, change.search, change.replace or "")
            if result is not None:
                log.warning(
                    "sandbox_whitespace_match",
                    file=change.file,
                    msg="Exact match failed but whitespace-normalized match succeeded",
                )
                return result
        return None

    return None


def _validate_diff(
    original: str,
    modified: str,
    config: SandboxConfig,
) -> list[str]:
    """Validate only the added lines in a diff."""
    errors = []
    diff_lines = list(difflib.unified_diff(
        original.splitlines(), modified.splitlines(), lineterm="",
    ))

    added_lines = [
        line[1:] for line in diff_lines
        if line.startswith("+") and not line.startswith("+++")
    ]

    for line in added_lines:
        for pattern in config.forbidden_patterns:
            if re.search(pattern, line):
                errors.append(f"Dangerous code in added line: {line.strip()}")

    return errors


def format_rejection_message(
    result: ValidationResult,
    current_files: dict[str, str] | None = None,
) -> str:
    """Format a rejection message for the agent."""
    lines = ["Your proposed changes were REJECTED for safety reasons:"]
    for error in result.errors:
        lines.append(f"  - {error}")

    # If search text not found, show actual file excerpt so agent can correct
    if current_files:
        for error in result.errors:
            if "search text not found" in error:
                # Extract filename from error: "File 'train.py': search text not found..."
                for fname, content in current_files.items():
                    if fname in error:
                        excerpt = content[:500]
                        lines.append("")
                        lines.append(f"Current content of '{fname}' (first 500 chars):")
                        lines.append(f"```\n{excerpt}\n```")
                        break

    lines.append("")
    lines.append("Please propose different changes that comply with the rules.")
    return "\n".join(lines)


def _try_whitespace_match(
    content: str, search: str, replace: str,
) -> str | None:
    """Try whitespace-normalized matching as fallback.

    Strips trailing whitespace from each line, then attempts the match.
    Returns the modified content if match found, None otherwise.
    """
    # Normalize both content and search: strip trailing whitespace per line
    def normalize(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.split("\n"))

    norm_content = normalize(content)
    norm_search = normalize(search)

    if norm_search in norm_content:
        # Apply replacement on normalized content, then return
        return norm_content.replace(norm_search, replace, 1)

    return None
