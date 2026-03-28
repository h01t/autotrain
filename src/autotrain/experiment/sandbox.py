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
