"""Code sandboxing — validate agent-generated changes before applying."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

import structlog

from autotrain.config.schema import SandboxConfig

log = structlog.get_logger()


@dataclass
class FileChange:
    """A proposed change to a file."""

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


def format_rejection_message(result: ValidationResult) -> str:
    """Format a rejection message for the agent."""
    lines = ["Your proposed changes were REJECTED for safety reasons:"]
    for error in result.errors:
        lines.append(f"  - {error}")
    lines.append("")
    lines.append("Please propose different changes that comply with the rules.")
    return "\n".join(lines)
