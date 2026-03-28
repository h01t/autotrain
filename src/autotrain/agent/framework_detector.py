"""Detect ML framework from project source files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import structlog

log = structlog.get_logger()

# Import patterns → framework name
_FRAMEWORK_PATTERNS: list[tuple[str, str]] = [
    (r"(?:from|import)\s+ultralytics", "ultralytics"),
    (r"(?:from|import)\s+transformers", "huggingface"),
    (r"(?:from|import)\s+(?:pytorch_lightning|lightning)", "lightning"),
    (r"(?:from|import)\s+(?:keras|tensorflow)", "keras"),
    (r"(?:from|import)\s+sklearn", "sklearn"),
    (r"(?:from|import)\s+xgboost", "xgboost"),
    (r"(?:from|import)\s+torch", "pytorch"),
]


@dataclass
class FrameworkHint:
    """Detected ML framework info."""

    name: str  # e.g. "ultralytics", "huggingface", "generic"
    detected_from: str = ""  # which file/import triggered detection
    common_metrics: list[str] = field(default_factory=list)


def detect_framework(repo_path: Path, writable_files: list[str]) -> FrameworkHint:
    """Scan writable files to detect ML framework.

    Checks imports in writable files, falls back to scanning all .py files in repo root.
    """
    # First check writable files (most likely to contain training code)
    for filename in writable_files:
        filepath = repo_path / filename
        if filepath.exists() and filepath.suffix == ".py":
            hint = _scan_file(filepath)
            if hint:
                return hint

    # Fallback: scan .py files in repo root
    for filepath in repo_path.glob("*.py"):
        hint = _scan_file(filepath)
        if hint:
            return hint

    log.info("framework_detection", result="generic", reason="no framework imports found")
    return FrameworkHint(name="generic")


def _scan_file(filepath: Path) -> FrameworkHint | None:
    """Scan a single file for framework imports."""
    try:
        content = filepath.read_text(errors="replace")
    except OSError:
        return None

    for pattern, framework in _FRAMEWORK_PATTERNS:
        match = re.search(pattern, content)
        if match:
            hint = FrameworkHint(
                name=framework,
                detected_from=f"{filepath.name}: {match.group(0).strip()}",
            )
            log.info(
                "framework_detected",
                framework=framework,
                file=filepath.name,
                import_line=match.group(0).strip(),
            )
            return hint

    return None
